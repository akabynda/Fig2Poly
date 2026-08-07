from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


CONFIGS = {
    "r50": "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
}


def color_for(index: int) -> tuple[int, int, int]:
    # BGR colors chosen to remain distinct on both white and dark plots.
    palette = (
        (32, 80, 240), (50, 180, 50), (220, 90, 30), (180, 60, 180),
        (20, 180, 220), (200, 150, 30), (80, 200, 180), (160, 100, 240),
        (230, 180, 70), (90, 60, 210), (60, 210, 130), (200, 80, 120),
    )
    return palette[index % len(palette)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MaskDINO on arbitrary images")
    parser.add_argument("--maskdino-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=CONFIGS, default="r50")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--max-size", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=0.48)
    args = parser.parse_args(argv)

    maskdino_root = args.maskdino_root.resolve()
    config = maskdino_root / CONFIGS[args.variant]
    sys.path.insert(0, str(maskdino_root))

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import transforms as T
    from detectron2.projects.deeplab import add_deeplab_config
    from maskdino import add_maskdino_config
    import train_net

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(config))
    cfg.MODEL.WEIGHTS = str(args.weights.resolve())
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = args.threshold
    cfg.INPUT.MIN_SIZE_TEST = args.image_size
    cfg.INPUT.MAX_SIZE_TEST = args.max_size
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()

    model = train_net.Trainer.build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    transform = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST,
    )

    image_paths = sorted(
        path for path in args.input.resolve().iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for image_path in image_paths:
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"Unable to read {image_path}")
        height, width = original.shape[:2]
        resized = transform.get_transform(original).apply_image(original)
        tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
        with torch.inference_mode():
            prediction = model([{"image": tensor, "height": height, "width": width}])[0]
        instances = prediction["instances"].to("cpu")
        scores = instances.scores.numpy() if instances.has("scores") else np.ones(len(instances))
        keep = scores >= args.threshold
        scores = scores[keep]
        masks = instances.pred_masks.numpy()[keep].astype(bool)
        boxes = instances.pred_boxes.tensor.numpy()[keep] if instances.has("pred_boxes") else None
        order = np.argsort(-scores)
        scores, masks = scores[order], masks[order]
        if boxes is not None:
            boxes = boxes[order]

        item_dir = output / image_path.stem
        item_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(item_dir / "original.png"), original)
        overlay = original.astype(np.float32)
        curves_only = np.full_like(original, 255)
        instance_map = np.zeros((height, width), dtype=np.uint16)
        records = []

        for index, (score, mask) in enumerate(zip(scores, masks), 1):
            color = np.asarray(color_for(index - 1), dtype=np.float32)
            overlay[mask] = overlay[mask] * (1.0 - args.alpha) + color * args.alpha
            curves_only[mask] = color.astype(np.uint8)
            instance_map[mask] = index
            mask_file = f"mask_{index:03d}.png"
            cv2.imwrite(str(item_dir / mask_file), mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color.tolist(), 1, cv2.LINE_AA)
            bbox = boxes[index - 1].tolist() if boxes is not None else None
            if bbox:
                x, y = int(bbox[0]), max(12, int(bbox[1]))
                cv2.putText(
                    overlay, f"{index}: {score:.2f}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color.tolist(), 1, cv2.LINE_AA,
                )
            records.append({"id": index, "score": float(score), "bbox_xyxy": bbox, "mask": mask_file})

        cv2.imwrite(str(item_dir / "overlay.png"), np.clip(overlay, 0, 255).astype(np.uint8))
        cv2.imwrite(str(item_dir / "curves_only.png"), curves_only)
        cv2.imwrite(str(item_dir / "instance_ids.png"), instance_map)
        (item_dir / "predictions.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        summary.append({"image": image_path.name, "instances": len(records), "predictions": records})
        print(f"{image_path.name}: {len(records)} curves", flush=True)

    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
