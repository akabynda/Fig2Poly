"""Train the original LineFormer architecture with its published config schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.train_lineformer import main as train_lineformer


def training_arguments(args: argparse.Namespace) -> list[str]:
    # No --weights: the upstream config initializes only Swin-Tiny from ImageNet.
    command = [
        "--lineformer-root", str(args.lineformer_root),
        "--dataset", str(args.dataset),
        "--output", str(args.output),
        "--max-iters", str(args.max_iters),
        "--base-lr", "1e-4",
        "--samples-per-gpu", "4",
        "--workers-per-gpu", str(args.workers_per_gpu),
        "--num-gpus", str(args.num_gpus),
        "--eval-interval", "1" if args.smoke_test else "250",
        "--checkpoint-interval", "1" if args.smoke_test else "500",
        "--log-interval", "1" if args.smoke_test else "100",
        "--early-stopping-patience", "0",
        "--seed", str(args.seed),
    ]
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineformer-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-iters", type=int, default=100000,
                        help="Use the default for full training; override only for smoke tests")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Two iterations with validation/checkpoint/logging on each step")
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.max_iters = 2
    if args.num_gpus < 1 or args.workers_per_gpu < 1 or args.max_iters < 1:
        parser.error("GPU count, workers and iterations must be positive")
    mixture = args.dataset / "mixture_summary.json"
    if not mixture.is_file():
        parser.error(f"Prepared mixture is not complete: {mixture}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    request = {
        "model": "Mask2Former Swin-Tiny / original LineFormer architecture",
        "initialization": "ImageNet Swin-Tiny backbone; new segmentation head",
        "training_profile": "lineformer",
        "mixture_summary": str(mixture.resolve()),
        "max_iters": args.max_iters,
        "samples_per_gpu": 4,
        "num_gpus": args.num_gpus,
        "global_batch": 4 * args.num_gpus,
        "base_lr": 1e-4,
        "early_stopping": False,
        "seed": args.seed,
        "smoke_test": args.smoke_test,
        "augmentation": "original train_pipeline_LineEX for all mixture sources",
        "command_arguments": training_arguments(args),
    }
    (output / "lineformer_recipe_request.json").write_text(
        json.dumps(request, indent=2), encoding="utf-8"
    )
    return train_lineformer(training_arguments(args))


if __name__ == "__main__":
    raise SystemExit(main())
