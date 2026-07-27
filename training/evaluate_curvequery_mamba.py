from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.curvequery_mamba import (
    CurveManifestDataset,
    TrainConfig,
    collate,
    create_model,
    training_dtype,
)


def pairwise_iou(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not len(predicted) or not len(target):
        return np.zeros((len(predicted), len(target)), dtype=np.float32)
    pred = predicted.reshape(len(predicted), -1).astype(np.float32)
    true = target.reshape(len(target), -1).astype(np.float32)
    intersection = pred @ true.T
    union = pred.sum(1)[:, None] + true.sum(1)[None, :] - intersection
    return intersection / np.maximum(union, 1.0)


def pairwise_tolerant_f1(predicted: np.ndarray, target: np.ndarray, radius: int) -> np.ndarray:
    if not len(predicted) or not len(target):
        return np.zeros((len(predicted), len(target)), dtype=np.float32)
    structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
    pred_dilated = np.stack([binary_dilation(mask, structure=structure) for mask in predicted])
    target_dilated = np.stack([binary_dilation(mask, structure=structure) for mask in target])
    pred_flat = predicted.reshape(len(predicted), -1).astype(np.float32)
    target_flat = target.reshape(len(target), -1).astype(np.float32)
    pred_d = pred_dilated.reshape(len(predicted), -1).astype(np.float32)
    target_d = target_dilated.reshape(len(target), -1).astype(np.float32)
    precision = (pred_flat @ target_d.T) / np.maximum(pred_flat.sum(1)[:, None], 1.0)
    recall = (pred_d @ target_flat.T) / np.maximum(target_flat.sum(1)[None, :], 1.0)
    return 2 * precision * recall / np.maximum(precision + recall, 1e-8)


class Accumulator:
    def __init__(self):
        self.images = self.tp = self.fp = self.fn = 0
        self.exact_iou_sum = self.tolerant_f1_sum = 0.0
        self.matches = 0
        self.count_error = 0.0

    def update(self, predicted: np.ndarray, target: np.ndarray, tolerance: int) -> None:
        tolerant = pairwise_tolerant_f1(predicted, target, tolerance)
        if tolerant.size:
            rows, columns = linear_sum_assignment(-tolerant)
            matched_tolerant = tolerant[rows, columns]
            exact = pairwise_iou(predicted, target)[rows, columns]
        else:
            matched_tolerant = np.empty(0, dtype=np.float32)
            exact = np.empty(0, dtype=np.float32)
        good = matched_tolerant >= 0.5
        true_positives = int(good.sum())
        self.tp += true_positives
        self.fp += len(predicted) - true_positives
        self.fn += len(target) - true_positives
        self.tolerant_f1_sum += float(matched_tolerant[good].sum())
        self.exact_iou_sum += float(exact[good].sum())
        self.matches += true_positives
        self.count_error += abs(len(predicted) - len(target))
        self.images += 1

    def result(self) -> dict:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        return {
            "images": self.images,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision_tolerant50": precision,
            "recall_tolerant50": recall,
            "f1_tolerant50": 2 * precision * recall / max(precision + recall, 1e-9),
            "mean_tolerant_f1": self.tolerant_f1_sum / max(1, self.matches),
            "mean_exact_iou": self.exact_iou_sum / max(1, self.matches),
            "count_mae": self.count_error / max(1, self.images),
        }


def load_checkpoint(path: Path, device: torch.device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config_values = saved["config"]
    known = TrainConfig.__dataclass_fields__.keys()
    config = TrainConfig(**{key: value for key, value in config_values.items() if key in known})
    model = create_model(config, device)
    model.load_state_dict(saved["model"])
    return model.eval(), config


def save_overlay(
    dataset: CurveManifestDataset,
    index: int,
    predicted: np.ndarray,
    output: Path,
) -> None:
    record = dataset._record(index)
    source = Image.open(dataset.root / record["image"]).convert("RGB")
    source = source.resize((dataset.width, dataset.height), Image.Resampling.BILINEAR)
    base = np.asarray(source, dtype=np.float32).copy()
    colors = np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
        ],
        dtype=np.float32,
    )
    for index, mask in enumerate(predicted):
        color = colors[index % len(colors)]
        base[mask] = base[mask] * 0.35 + color * 0.65
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(output)


@torch.inference_mode()
def evaluate(
    checkpoint: Path,
    dataset_root: Path,
    split: str,
    thresholds: list[float],
    width: int,
    height: int,
    tolerance: int,
    limit: int | None,
    output: Path,
    visuals: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(checkpoint, device)
    dataset = CurveManifestDataset(dataset_root, split, width, height, limit, False, 42)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate)
    accumulators = {
        threshold: defaultdict(Accumulator)
        for threshold in thresholds
    }
    started = time.time()
    for index, batch in enumerate(loader):
        with torch.autocast(
            "cuda", dtype=training_dtype(device), enabled=device.type == "cuda"
        ):
            result = model(pixel_values=batch["pixel_values"].to(device, non_blocking=True))
        scores = result.class_queries_logits[0].softmax(-1)[:, 0]
        masks = F.interpolate(
            result.masks_queries_logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].sigmoid()
        target = batch["mask_labels"][0].numpy() > 0.5
        source = batch["dataset_sources"][0]
        for threshold in thresholds:
            keep = scores >= threshold
            predicted = (masks[keep] >= 0.5).cpu().numpy()
            if len(predicted):
                areas = predicted.reshape(len(predicted), -1).sum(1)
                predicted = predicted[areas >= 8]
            accumulators[threshold]["all"].update(predicted, target, tolerance)
            accumulators[threshold][source].update(predicted, target, tolerance)
            if threshold == thresholds[0] and index < visuals:
                save_overlay(
                    dataset,
                    index,
                    predicted,
                    output.parent / "visuals" / f"{split}_{index:05d}.png",
                )
        if (index + 1) % 100 == 0:
            print(f"{split}: {index + 1}/{len(dataset)}", flush=True)

    candidates = {}
    for threshold, groups in accumulators.items():
        candidates[str(threshold)] = {
            name: accumulator.result() for name, accumulator in groups.items()
        }
    best = max(thresholds, key=lambda value: candidates[str(value)]["all"]["f1_tolerant50"])
    report = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "split": split,
        "images": len(dataset),
        "resolution": [width, height],
        "tolerance_px": tolerance,
        "elapsed_seconds": round(time.time() - started, 2),
        "best_threshold": best,
        "best": candidates[str(best)],
        "thresholds": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visuals", type=int, default=20)
    args = parser.parse_args(argv)
    evaluate(
        Path(args.checkpoint).resolve(),
        Path(args.dataset).resolve(),
        args.split,
        [float(item) for item in args.thresholds.split(",")],
        args.width,
        args.height,
        args.tolerance,
        args.limit,
        Path(args.output).resolve(),
        args.visuals,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
