from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


CONFIGS = {
    "r50": "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
    "swin_l": "configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
}


def image_count(dataset: Path, split: str) -> int:
    payload = json.loads(
        (dataset / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8")
    )
    return len(payload["images"])


def register_datasets(dataset: str) -> None:
    from detectron2.data.datasets import register_coco_instances

    root = Path(dataset)
    for split in ("train", "val", "test"):
        register_coco_instances(
            f"fig2poly_{split}",
            {},
            str(root / "annotations" / f"instances_{split}.json"),
            str(root / "images" / split),
        )


def worker(upstream_args, dataset: str, report: str | None):
    register_datasets(dataset)
    import train_net
    from detectron2.utils import comm

    result = train_net.main(upstream_args)
    if report and comm.is_main_process():
        Path(report).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train official MaskDINO on Fig2Poly COCO RLE")
    parser.add_argument("--maskdino-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=CONFIGS, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--global-batch", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--base-lr", type=float)
    parser.add_argument(
        "--max-iter",
        type=int,
        help="Override the epoch-derived iteration count (used by smoke tests)",
    )
    parser.add_argument(
        "--checkpoint-period",
        type=int,
        help="Override checkpoint interval in iterations",
    )
    parser.add_argument(
        "--eval-period",
        type=int,
        help="Override validation interval in iterations; 0 disables periodic evaluation",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    maskdino_root = args.maskdino_root.resolve()
    dataset = args.dataset.resolve()
    config = maskdino_root / CONFIGS[args.variant]
    if not config.is_file():
        parser.error(f"MaskDINO config not found: {config}")
    if not args.weights.is_file() and not args.resume:
        parser.error(f"pretrained checkpoint not found: {args.weights}")
    sys.path.insert(0, str(maskdino_root))
    import train_net
    from detectron2.engine import launch

    steps_per_epoch = math.ceil(image_count(dataset, "train") / args.global_batch)
    max_iter = args.max_iter or max(1, steps_per_epoch * args.epochs)
    checkpoint_period = args.checkpoint_period or max(100, steps_per_epoch // 4)
    eval_period = steps_per_epoch if args.eval_period is None else args.eval_period
    learning_rate = args.base_lr or 1e-4 * args.global_batch / 16
    opts = [
        "MODEL.WEIGHTS", str(args.weights.resolve()),
        "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",
        "DATASETS.TRAIN", '("fig2poly_train",)',
        "DATASETS.TEST", f'("fig2poly_{args.eval_split}",)',
        "DATALOADER.FILTER_EMPTY_ANNOTATIONS", "False",
        "DATALOADER.NUM_WORKERS", str(args.workers),
        "SOLVER.IMS_PER_BATCH", str(args.global_batch),
        "SOLVER.BASE_LR", str(learning_rate),
        "SOLVER.MAX_ITER", str(max_iter),
        "SOLVER.STEPS", f"({int(max_iter * .89)}, {int(max_iter * .96)})",
        "SOLVER.CHECKPOINT_PERIOD", str(checkpoint_period),
        "TEST.EVAL_PERIOD", str(eval_period),
        "INPUT.IMAGE_SIZE", str(args.image_size),
        "INPUT.MIN_SCALE", "0.5",
        "INPUT.MAX_SCALE", "1.5",
        "OUTPUT_DIR", str(args.output.resolve()),
    ]
    command = ["--config-file", str(config), "--num-gpus", str(args.num_gpus)]
    if args.resume:
        command.append("--resume")
    if args.eval_only:
        command.append("--eval-only")
    command.extend(opts)
    upstream_args = train_net.default_argument_parser().parse_args(command)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "train_images": image_count(dataset, "train"),
                "steps_per_epoch": steps_per_epoch,
                "max_iter": max_iter,
                "checkpoint_period": checkpoint_period,
                "eval_period": eval_period,
                "global_batch": args.global_batch,
                "base_lr": learning_rate,
                "resume": args.resume,
            },
            indent=2,
        ),
        flush=True,
    )
    launch(
        worker,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=upstream_args.dist_url,
        args=(upstream_args, str(dataset), str(args.report.resolve()) if args.report else None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
