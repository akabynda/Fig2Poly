from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Iterator

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import (
    Mamba2Config,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
MODEL_SOURCE = "facebook/mask2former-swin-tiny-coco-instance"


def training_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@dataclass
class TrainConfig:
    dataset: str
    output: str = "runs/curvequery_mamba"
    split: str = "train"
    val_split: str = "val"
    model_source: str = MODEL_SOURCE
    decoder: str = "mamba"
    width: int = 512
    height: int = 384
    batch_size: int = 1
    accumulation_steps: int = 4
    epochs: int = 1
    learning_rate: float = 5e-5
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 0.05
    warmup_steps: int = 500
    save_every: int = 500
    keep_checkpoints: int = 2
    validate_every: int = 1000
    val_limit: int = 200
    train_limit: int | None = None
    num_workers: int = 2
    seed: int = 42
    resume: str | None = None
    init_checkpoint: str | None = None
    mamba_state_size: int = 32
    max_grad_norm: float = 1.0


class CurveManifestDataset(Dataset):
    """CurveForge v4 reader using JSONL instead of directory scans."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        width: int,
        height: int,
        limit: int | None = None,
        augment: bool = False,
        seed: int = 42,
    ):
        self.root = Path(root).resolve()
        self.split = split
        self.width = width
        self.height = height
        self.augment = augment
        self.seed = seed
        manifest = self.root / f"{split}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest}")
        self.offsets: list[int] = []
        with manifest.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
                    if limit is not None and len(self.offsets) >= limit:
                        break
        if not self.offsets:
            raise RuntimeError(f"Empty manifest: {manifest}")
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.offsets)

    def _record(self, index: int) -> dict:
        with self.manifest.open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())

    def __getitem__(self, index: int) -> dict:
        record = self._record(index)
        image_path = self.root / record["image"]
        image = Image.open(image_path).convert("RGB")
        masks = [
            Image.open(self.root / curve["mask"]).convert("L")
            for curve in record.get("curves", [])
            if curve.get("mask")
        ]

        rng = random.Random(self.seed + index)
        if self.augment and rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            masks = [mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for mask in masks]
        if self.augment and rng.random() < 0.15:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            masks = [mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for mask in masks]

        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        masks = [
            mask.resize((self.width, self.height), Image.Resampling.NEAREST)
            for mask in masks
        ]
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
        if masks:
            mask_array = np.stack([np.asarray(mask, dtype=np.uint8) > 0 for mask in masks])
            mask_labels = torch.from_numpy(mask_array.astype(np.float32))
        else:
            mask_labels = torch.empty((0, self.height, self.width), dtype=torch.float32)
        return {
            "pixel_values": pixels,
            "mask_labels": mask_labels,
            "class_labels": torch.zeros(len(masks), dtype=torch.long),
            "image_id": str(record.get("id", image_path.stem)),
            "dataset_source": str(record.get("dataset_source", "curveforge")),
        }


def collate(items: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in items]),
        "mask_labels": [item["mask_labels"] for item in items],
        "class_labels": [item["class_labels"] for item in items],
        "image_ids": [item["image_id"] for item in items],
        "dataset_sources": [item["dataset_source"] for item in items],
    }


class ResumePermutationSampler(Sampler[int]):
    def __init__(self, size: int, seed: int, epoch: int = 0, start: int = 0):
        self.size = size
        self.seed = seed
        self.epoch = epoch
        self.start = start

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        yield from order[self.start :]

    def __len__(self) -> int:
        return max(0, self.size - self.start)


class BidirectionalMambaFFN(nn.Module):
    """Drop-in FFN replacement operating along the mask-query sequence."""

    def __init__(self, hidden_size: int, state_size: int, layer_idx: int):
        super().__init__()
        inner_size = hidden_size * 2
        head_dim = 64 if inner_size % 64 == 0 else 32
        num_heads = inner_size // head_dim
        config = Mamba2Config(
            hidden_size=hidden_size,
            state_size=state_size,
            # The mixer indexes config.layer_types, so this must cover all
            # nine Mask2Former decoder layers even though each wrapper owns
            # only its own mixer parameters.
            num_hidden_layers=9,
            expand=2,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=min(8, num_heads),
            conv_kernel=4,
            chunk_size=128,
            use_cache=False,
            residual_in_fp32=False,
        )
        self.forward_mixer = Mamba2Mixer(config, layer_idx=layer_idx)
        self.backward_mixer = Mamba2Mixer(config, layer_idx=layer_idx)
        self.merge = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Mask2Former: [queries, batch, channels], Mamba2: [batch, sequence, channels].
        states = hidden_states.transpose(0, 1)
        forward = self.forward_mixer(states)
        backward = self.backward_mixer(torch.flip(states, dims=(1,)))
        backward = torch.flip(backward, dims=(1,))
        return self.merge(torch.cat((forward, backward), dim=-1)).transpose(0, 1)


def inject_mamba_decoder(model: Mask2FormerForUniversalSegmentation, state_size: int) -> None:
    layers = model.model.transformer_module.decoder.layers
    for index, layer in enumerate(layers):
        layer.fc1 = BidirectionalMambaFFN(layer.embed_dim, state_size, index)
        layer.activation_fn = nn.Identity()
        layer.fc2 = nn.Identity()
    model.config.curvequery_decoder = "bidirectional_mamba2"
    model.config.curvequery_mamba_state_size = state_size


def create_model(cfg: TrainConfig, device: torch.device) -> Mask2FormerForUniversalSegmentation:
    config = Mask2FormerConfig.from_pretrained(cfg.model_source)
    config.num_labels = 1
    config.id2label = {0: "curve"}
    config.label2id = {"curve": 0}
    config.num_queries = 100
    config.use_auxiliary_loss = True
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg.model_source,
        config=config,
        ignore_mismatched_sizes=True,
    )
    if cfg.decoder == "mamba":
        inject_mamba_decoder(model, cfg.mamba_state_size)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except ValueError:
            # Transformers 5 exposes the method on the common base class even
            # though Mask2Former does not advertise checkpointing support.
            pass
    return model.to(device)


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def checkpoint_payload(model, optimizer, scheduler, scaler, cfg, state) -> dict:
    return {
        "format_version": 1,
        "architecture": "CurveQuery-Mamba" if cfg.decoder == "mamba" else "Mask2Former-SwinT",
        "config": asdict(cfg),
        "state": state,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def rotate_checkpoints(output: Path, keep: int) -> None:
    checkpoints = sorted(output.glob("checkpoint-*.pt"))
    for path in checkpoints[: max(0, len(checkpoints) - keep)]:
        path.unlink()


def validate(model, loader, device, limit: int) -> float:
    model.eval()
    losses: list[float] = []
    dtype = training_dtype(device)
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                result = model(
                    pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                    mask_labels=[item.to(device, non_blocking=True) for item in batch["mask_labels"]],
                    class_labels=[item.to(device, non_blocking=True) for item in batch["class_labels"]],
                )
            losses.append(float(result.loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def cosine_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-3, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train(cfg: TrainConfig) -> Path:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    output = Path(cfg.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_data = CurveManifestDataset(
        cfg.dataset, cfg.split, cfg.width, cfg.height, cfg.train_limit, True, cfg.seed
    )
    val_data = CurveManifestDataset(
        cfg.dataset, cfg.val_split, cfg.width, cfg.height, cfg.val_limit, False, cfg.seed
    )
    model = create_model(cfg, device)
    if cfg.init_checkpoint:
        initialized = torch.load(
            Path(cfg.init_checkpoint).resolve(), map_location="cpu", weights_only=False
        )
        model.load_state_dict(initialized["model"])

    backbone_params, main_params = [], []
    for name, parameter in model.named_parameters():
        (backbone_params if "pixel_level_module.encoder" in name else main_params).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": cfg.learning_rate},
            {"params": backbone_params, "lr": cfg.backbone_learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_data) / (cfg.batch_size * cfg.accumulation_steps))
    total_steps = updates_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_factor(step, total_steps, cfg.warmup_steps)
    )
    amp_type = training_dtype(device)
    # BF16 has FP32-like exponent range and needs no loss scaling. On older
    # cards FP16 keeps dynamic scaling enabled.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and amp_type == torch.float16
    )
    state = {"epoch": 0, "sample_in_epoch": 0, "global_step": 0, "samples_seen": 0}

    resume_path = Path(cfg.resume).resolve() if cfg.resume else output / "last.pt"
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state.update(saved["state"])
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        if device.type == "cuda" and saved["rng"].get("cuda"):
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(1, cfg.num_workers)),
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    stop_requested = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    history_path = output / "history.jsonl"
    started = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(state["epoch"], cfg.epochs):
        start = state["sample_in_epoch"] if epoch == state["epoch"] else 0
        sampler = ResumePermutationSampler(len(train_data), cfg.seed, epoch, start)
        loader = DataLoader(
            train_data,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
            persistent_workers=cfg.num_workers > 0,
        )
        accumulated = 0
        accumulated_loss = 0.0
        for batch in loader:
            wants_stop = stop_requested or (output / "STOP").exists()
            # Finish the current gradient-accumulation group so a checkpoint
            # never claims samples whose gradients were silently discarded.
            if wants_stop and accumulated == 0:
                break
            with torch.autocast("cuda", dtype=amp_type, enabled=device.type == "cuda"):
                result = model(
                    pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                    mask_labels=[item.to(device, non_blocking=True) for item in batch["mask_labels"]],
                    class_labels=[item.to(device, non_blocking=True) for item in batch["class_labels"]],
                )
                loss = result.loss / cfg.accumulation_steps
            scaler.scale(loss).backward()
            accumulated += 1
            accumulated_loss += float(loss.detach().cpu())
            consumed = len(batch["image_ids"])
            state["sample_in_epoch"] += consumed
            state["samples_seen"] += consumed

            if accumulated < cfg.accumulation_steps:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulated = 0
            state["global_step"] += 1
            record = {
                **state,
                "train_loss": accumulated_loss,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": round(time.time() - started, 2),
            }
            accumulated_loss = 0.0
            if state["global_step"] == 1 or state["global_step"] % 25 == 0:
                print(json.dumps(record), flush=True)
            if cfg.validate_every > 0 and state["global_step"] % cfg.validate_every == 0:
                record["val_loss"] = validate(model, val_loader, device, cfg.val_limit)
            append_jsonl(history_path, record)
            if state["global_step"] % cfg.save_every == 0:
                payload = checkpoint_payload(model, optimizer, scheduler, scaler, cfg, state)
                atomic_torch_save(payload, output / "last.pt")
                atomic_torch_save(payload, output / f"checkpoint-{state['global_step']:07d}.pt")
                rotate_checkpoints(output, cfg.keep_checkpoints)
            if wants_stop:
                break

        if stop_requested or (output / "STOP").exists():
            atomic_torch_save(
                checkpoint_payload(model, optimizer, scheduler, scaler, cfg, state),
                output / "last.pt",
            )
            print(f"Stopped safely at step {state['global_step']}; checkpoint={output / 'last.pt'}")
            return output / "last.pt"
        state["epoch"] = epoch + 1
        state["sample_in_epoch"] = 0
        atomic_torch_save(
            checkpoint_payload(model, optimizer, scheduler, scaler, cfg, state),
            output / "last.pt",
        )

    final_path = output / "final.pt"
    atomic_torch_save(
        checkpoint_payload(model, optimizer, scheduler, scaler, cfg, state),
        final_path,
    )
    print(f"Training complete: {final_path}", flush=True)
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CurveQuery-Mamba on CurveForge manifests")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="runs/curvequery_mamba")
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--model-source", default=MODEL_SOURCE)
    parser.add_argument("--decoder", choices=("mamba", "transformer"), default="mamba")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    parser.add_argument(
        "--init-checkpoint",
        help="Load model weights only and start a new optimizer/schedule (for P2 fine-tuning).",
    )
    parser.add_argument("--mamba-state-size", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(TrainConfig(**vars(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
