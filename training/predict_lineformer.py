from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PALETTE = (
    (32, 80, 240), (50, 180, 50), (220, 90, 30), (180, 60, 180),
    (20, 180, 220), (200, 150, 30), (80, 200, 180), (160, 100, 240),
    (230, 180, 70), (90, 60, 210), (60, 210, 130), (200, 80, 120),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the official LineFormer checkpoint on arbitrary images"
    )
    parser.add_argument("--lineformer-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.48)
    args = parser.parse_args(argv)

    root = args.lineformer_root.resolve()
    sys.path.insert(0, str(root))
    from mmdet.apis import inference_detector, init_detector

    model = init_detector(
        str(root / "lineformer_swin_t_config.py"),
        str(args.weights.resolve()),
        device="cuda:0",
    )
    image_paths = sorted(
        path for path in args.input.resolve().iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read {image_path}")
        result = inference_detector(model, image)
        boxes = np.asarray(result[0][0])
        raw_masks = result[1][0]
        selected = [
            (float(box[4]), np.asarray(mask, dtype=bool), box[:4].tolist())
            for box, mask in zip(boxes, raw_masks)
            if float(box[4]) >= args.threshold
        ]
        selected.sort(key=lambda item: item[0], reverse=True)

        height, width = image.shape[:2]
        item_dir = output / image_path.stem
        item_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(item_dir / "original.png"), image)
        overlay = image.astype(np.float32)
        curves_only = np.full_like(image, 255)
        instance_map = np.zeros((height, width), dtype=np.uint16)
        records: list[dict] = []

        for index, (score, mask, bbox) in enumerate(selected, 1):
            color = np.asarray(PALETTE[(index - 1) % len(PALETTE)], dtype=np.float32)
            overlay[mask] = overlay[mask] * (1 - args.alpha) + color * args.alpha
            curves_only[mask] = color.astype(np.uint8)
            instance_map[mask] = index
            mask_file = f"mask_{index:03d}.png"
            cv2.imwrite(str(item_dir / mask_file), mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color.tolist(), 1, cv2.LINE_AA)
            x, y = int(bbox[0]), max(12, int(bbox[1]))
            cv2.putText(
                overlay, f"{index}: {score:.2f}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color.tolist(), 1, cv2.LINE_AA,
            )
            records.append(
                {"id": index, "score": score, "bbox_xyxy": bbox, "mask": mask_file}
            )

        cv2.imwrite(str(item_dir / "overlay.png"), np.clip(overlay, 0, 255).astype(np.uint8))
        cv2.imwrite(str(item_dir / "curves_only.png"), curves_only)
        cv2.imwrite(str(item_dir / "instance_ids.png"), instance_map)
        (item_dir / "predictions.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        summary.append(
            {"image": image_path.name, "instances": len(records), "predictions": records}
        )
        print(f"{image_path.name}: {len(records)} curves", flush=True)

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
