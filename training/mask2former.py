from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def amp_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16


@dataclass
class TrainConfig:
    dataset: str = "dataset"
    output: str = "runs/mask2former_baseline"
    checkpoint: str = "facebook/mask2former-swin-tiny-coco-instance"
    width: int = 512
    height: int = 384
    batch_size: int = 1
    accumulation_steps: int = 4
    max_steps: int = 1000
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    warmup_steps: int = 100
    train_limit: int | None = None
    val_limit: int = 200
    save_every: int = 250
    seed: int = 42


class CurveInstanceDataset(Dataset):
    def __init__(self, root: str | Path, split: str, width: int, height: int,
                 limit: int | None = None, augment: bool = False, seed: int = 42):
        self.root = Path(root)
        self.split = split
        self.width, self.height = width, height
        self.augment = augment
        self.seed = seed
        self.images = sorted((self.root / "images" / split).glob("*.jpg"))
        if limit is not None:
            self.images = self.images[:limit]
        if not self.images:
            raise FileNotFoundError(f"No images found for split {split!r} in {self.root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]
        mask_paths = sorted((self.root / "curve_masks" / self.split / image_path.stem).glob("curve_*.png"))
        if not mask_paths:
            raise RuntimeError(f"No curve masks for {image_path}")
        image = Image.open(image_path).convert("RGB").resize((self.width, self.height), Image.Resampling.BILINEAR)
        masks = [Image.open(path).convert("L").resize((self.width, self.height), Image.Resampling.NEAREST)
                 for path in mask_paths]
        rng = random.Random(self.seed + index)
        if self.augment and rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            masks = [mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for mask in masks]
        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        pixel_values = torch.from_numpy(image_array).permute(2, 0, 1)
        pixel_values = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD
        mask_array = np.stack([np.asarray(mask, dtype=np.uint8) > 0 for mask in masks])
        mask_labels = torch.from_numpy(mask_array.astype(np.float32))
        return {
            "pixel_values": pixel_values,
            "mask_labels": mask_labels,
            "class_labels": torch.zeros(len(masks), dtype=torch.long),
            "image_id": image_path.stem,
        }


def collate(items: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in items]),
        "mask_labels": [x["mask_labels"] for x in items],
        "class_labels": [x["class_labels"] for x in items],
        "image_ids": [x["image_id"] for x in items],
    }


def load_model(checkpoint: str, device: torch.device) -> Mask2FormerForUniversalSegmentation:
    config = Mask2FormerConfig.from_pretrained(checkpoint)
    config.num_labels = 1
    config.id2label = {0: "curve"}
    config.label2id = {"curve": 0}
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.config.use_auxiliary_loss = True
    return model.to(device)


def lr_factor(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return max(1e-3, (step + 1) / max(1, cfg.warmup_steps))
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def validation_loss(model, loader, device, max_batches: int = 25) -> float:
    model.eval()
    values = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            with torch.autocast(device_type=device.type, dtype=amp_dtype(device), enabled=device.type == "cuda"):
                output = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[x.to(device) for x in batch["mask_labels"]],
                    class_labels=[x.to(device) for x in batch["class_labels"]],
                )
            values.append(float(output.loss.detach().cpu()))
    model.train()
    return float(np.mean(values)) if values else float("nan")


def train(cfg: TrainConfig) -> Path:
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    output = Path(cfg.output); output.mkdir(parents=True, exist_ok=True)
    (output / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    train_data = CurveInstanceDataset(cfg.dataset, "train", cfg.width, cfg.height, cfg.train_limit, True, cfg.seed)
    val_data = CurveInstanceDataset(cfg.dataset, "val", cfg.width, cfg.height, cfg.val_limit, False, cfg.seed)
    loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
                        pin_memory=device.type == "cuda", collate_fn=collate, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1,
                            pin_memory=device.type == "cuda", collate_fn=collate)
    model = load_model(cfg.checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, cfg))
    model.train(); optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader); started = time.time(); history_path = output / "history.jsonl"
    for step in range(1, cfg.max_steps + 1):
        total_loss = 0.0
        for _ in range(cfg.accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); batch = next(iterator)
            with torch.autocast(device_type=device.type, dtype=amp_dtype(device), enabled=device.type == "cuda"):
                result = model(
                    pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                    mask_labels=[x.to(device, non_blocking=True) for x in batch["mask_labels"]],
                    class_labels=[x.to(device, non_blocking=True) for x in batch["class_labels"]],
                )
                loss = result.loss / cfg.accumulation_steps
            loss.backward(); total_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        record = {"step": step, "train_loss": total_loss, "lr": scheduler.get_last_lr()[0],
                  "elapsed_seconds": round(time.time() - started, 2)}
        if step == 1 or step % 25 == 0:
            print(json.dumps(record), flush=True)
        if step % cfg.save_every == 0 or step == cfg.max_steps:
            record["val_loss"] = validation_loss(model, val_loader, device)
            checkpoint_dir = output / f"checkpoint-{step:06d}"
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            (checkpoint_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
    final_dir = output / "final"
    model.save_pretrained(final_dir, safe_serialization=True)
    return final_dir


def mask_iou_matrix(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(target) == 0:
        return np.zeros((len(pred), len(target)), dtype=np.float32)
    pred_flat = pred.reshape(len(pred), -1).astype(np.float32)
    target_flat = target.reshape(len(target), -1).astype(np.float32)
    inter = pred_flat @ target_flat.T
    union = pred_flat.sum(1)[:, None] + target_flat.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1.0)


class MetricAccumulator:
    def __init__(self):
        self.images = self.tp = self.fp = self.fn = 0
        self.iou_sum = self.matches = self.count_error = self.union_iou = 0.0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        matrix = mask_iou_matrix(pred, target)
        rows, cols = linear_sum_assignment(-matrix) if matrix.size else (np.array([], int), np.array([], int))
        matched = matrix[rows, cols] if len(rows) else np.array([], dtype=np.float32)
        good = int((matched >= 0.5).sum())
        self.tp += good; self.fp += len(pred) - good; self.fn += len(target) - good
        self.iou_sum += float(matched.sum()); self.matches += len(matched)
        self.count_error += abs(len(pred) - len(target)); self.images += 1
        pu = pred.any(0) if len(pred) else np.zeros(target.shape[1:], bool)
        tu = target.any(0) if len(target) else np.zeros(pred.shape[1:], bool)
        inter = np.logical_and(pu, tu).sum(); union = np.logical_or(pu, tu).sum()
        self.union_iou += inter / max(1, union)

    def result(self) -> dict:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        return {"images": self.images, "tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision_iou50": precision, "recall_iou50": recall,
                "f1_iou50": 2*precision*recall/max(1e-9, precision+recall),
                "mean_matched_iou": self.iou_sum/max(1, self.matches),
                "count_mae": self.count_error/max(1, self.images),
                "semantic_union_iou": self.union_iou/max(1, self.images)}


@torch.inference_mode()
def evaluate(model_path: str, dataset_root: str, split: str, limit: int, width: int, height: int,
             thresholds: list[float], output_path: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = CurveInstanceDataset(dataset_root, split, width, height, limit, False)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device).eval()
    metrics = {threshold: MetricAccumulator() for threshold in thresholds}
    started = time.time()
    for index, batch in enumerate(loader, 1):
        with torch.autocast(device_type=device.type, dtype=amp_dtype(device), enabled=device.type == "cuda"):
            result = model(pixel_values=batch["pixel_values"].to(device))
        scores = result.class_queries_logits[0].softmax(-1)[:, 0]
        masks = F.interpolate(result.masks_queries_logits, size=(height, width), mode="bilinear", align_corners=False)[0].sigmoid()
        target = batch["mask_labels"][0].numpy() > 0.5
        for threshold in thresholds:
            keep = scores >= threshold
            pred = (masks[keep] >= 0.5).cpu().numpy()
            if len(pred): pred = pred[pred.reshape(len(pred), -1).sum(1) >= 8]
            metrics[threshold].update(pred, target)
        if index % 50 == 0:
            print(f"evaluated {index}/{len(data)}", flush=True)
    candidates = {str(k): v.result() for k, v in metrics.items()}
    best_threshold = max(thresholds, key=lambda t: candidates[str(t)]["f1_iou50"])
    report = {"model": model_path, "split": split, "width": width, "height": height,
              "elapsed_seconds": round(time.time()-started, 2), "best_threshold": best_threshold,
              "best": candidates[str(best_threshold)], "thresholds": candidates}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate Mask2Former on CurveForge")
    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train")
    train_p.add_argument("--dataset", default="dataset"); train_p.add_argument("--output", default="runs/mask2former_baseline")
    train_p.add_argument("--checkpoint", default="facebook/mask2former-swin-tiny-coco-instance")
    train_p.add_argument("--width", type=int, default=512); train_p.add_argument("--height", type=int, default=384)
    train_p.add_argument("--batch-size", type=int, default=1); train_p.add_argument("--accumulation-steps", type=int, default=4)
    train_p.add_argument("--max-steps", type=int, default=1000); train_p.add_argument("--learning-rate", type=float, default=5e-5)
    train_p.add_argument("--weight-decay", type=float, default=.05); train_p.add_argument("--warmup-steps", type=int, default=100)
    train_p.add_argument("--train-limit", type=int); train_p.add_argument("--val-limit", type=int, default=200)
    train_p.add_argument("--save-every", type=int, default=250); train_p.add_argument("--seed", type=int, default=42)
    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--model", required=True); eval_p.add_argument("--dataset", default="dataset")
    eval_p.add_argument("--split", choices=["train","val","test"], default="test")
    eval_p.add_argument("--limit", type=int, default=500); eval_p.add_argument("--width", type=int, default=512)
    eval_p.add_argument("--height", type=int, default=384); eval_p.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6")
    eval_p.add_argument("--output", default="runs/mask2former_baseline/test_metrics.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        cfg = TrainConfig(**{key: value for key, value in vars(args).items() if key != "command"})
        print(f"Saved final model to {train(cfg)}")
    else:
        evaluate(args.model, args.dataset, args.split, args.limit, args.width, args.height,
                 [float(x) for x in args.thresholds.split(",")], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
