from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO and write machine-readable metrics")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    metrics = YOLO(args.weights).val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        mask_ratio=1,
        overlap_mask=False,
        max_det=100,
        plots=True,
        save_json=True,
    )
    payload = {
        "model": "yolo26x_seg",
        "split": args.split,
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
