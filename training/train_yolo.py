from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train YOLO26 instance segmentation on CurveForge")
    parser.add_argument("--model", default="yolo26x-seg.pt")
    parser.add_argument("--data", default="dataset/curve_yolo.yaml")
    parser.add_argument("--project", default="runs/yolo26")
    parser.add_argument("--name", default="yolo26x_seg")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--resume", type=Path,
                        help="Resume from a last.pt checkpoint, including optimizer and epoch state")
    args = parser.parse_args(argv)
    if args.resume:
        checkpoint = args.resume.resolve()
        if not checkpoint.is_file():
            parser.error(f"resume checkpoint does not exist: {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True, device=args.device, workers=args.workers)
        return 0
    model = YOLO(str(Path(args.model).resolve()) if Path(args.model).exists() else args.model)
    model.train(
        data=str(Path(args.data).resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device,
        project=str(Path(args.project).resolve()), name=args.name,
        plots=True, amp=True, mask_ratio=1, overlap_mask=False,
        cos_lr=True, mosaic=args.mosaic,
        close_mosaic=max(1, args.epochs // 5) if args.mosaic else 0,
        patience=args.patience, exist_ok=True,
        fraction=args.fraction,
        save=True, save_period=1,
        seed=args.seed, deterministic=True, max_det=args.max_det,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
