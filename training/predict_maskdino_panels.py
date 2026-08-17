from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from training.predict_lineformer_panels import (
    PALETTE,
    clean_prediction_tracks,
    detect_plot_boxes,
    render_threshold,
    suppress_centerline_duplicates,
)
from training.predict_maskdino import CONFIGS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run MaskDINO per plot panel with centerline-aware postprocessing"
    )
    parser.add_argument("--maskdino-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=CONFIGS, default="r50")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.15, 0.25, 0.3, 0.5])
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--max-size", type=int, default=2048)
    args = parser.parse_args(argv)

    maskdino_root = args.maskdino_root.resolve()
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
    cfg.merge_from_file(str(maskdino_root / CONFIGS[args.variant]))
    cfg.MODEL.WEIGHTS = str(args.weights.resolve())
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = min(args.thresholds)
    cfg.INPUT.MIN_SIZE_TEST = args.image_size
    cfg.INPUT.MAX_SIZE_TEST = args.max_size
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()

    model = train_net.Trainer.build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    transform = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    summary: list[dict] = []
    image_paths = sorted(
        path for path in args.input.resolve().iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read {image_path}")
        height, width = image.shape[:2]
        plot_boxes = detect_plot_boxes(image)
        panel_view = image.copy()
        predictions: list[dict] = []
        for panel_index, (x1, y1, x2, y2) in enumerate(plot_boxes, 1):
            color = PALETTE[(panel_index - 1) % len(PALETTE)]
            cv2.rectangle(panel_view, (x1, y1), (x2 - 1, y2 - 1), color, 3)
            cv2.putText(panel_view, f"panel {panel_index}", (x1 + 5, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            crop = image[y1:y2, x1:x2]
            crop_height, crop_width = crop.shape[:2]
            resized = transform.get_transform(crop).apply_image(crop)
            tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
            with torch.inference_mode():
                output = model([{"image": tensor, "height": crop_height, "width": crop_width}])[0]
            instances = output["instances"].to("cpu")
            scores = instances.scores.numpy() if instances.has("scores") else np.ones(len(instances))
            masks = instances.pred_masks.numpy().astype(bool)
            boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
            for instance_index, (score, crop_mask) in enumerate(zip(scores, masks)):
                if score < min(args.thresholds):
                    continue
                mask = np.zeros((height, width), dtype=bool)
                mask[y1:y2, x1:x2] = crop_mask
                if boxes is not None:
                    box = boxes[instance_index]
                    bbox = [float(box[0] + x1), float(box[1] + y1),
                            float(box[2] + x1), float(box[3] + y1)]
                else:
                    ys, xs = np.nonzero(mask)
                    bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                predictions.append({
                    "score": float(score), "mask": mask, "panel": panel_index, "bbox": bbox
                })

        raw_count = len(predictions)
        predictions, reassigned = clean_prediction_tracks(predictions, height, image)
        predictions, suppressed = suppress_centerline_duplicates(predictions, height)
        image_dir = args.output.resolve() / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / "panels.png"), panel_view)
        (image_dir / "centerline_reassigned.json").write_text(
            json.dumps(reassigned, indent=2), encoding="utf-8"
        )
        (image_dir / "centerline_suppressed.json").write_text(
            json.dumps(suppressed, indent=2), encoding="utf-8"
        )
        counts: dict[str, int] = {}
        for threshold in args.thresholds:
            label = f"threshold_{threshold:.2f}"
            counts[label] = render_threshold(image, predictions, threshold, image_dir / label)
        summary.append({
            "image": image_path.name,
            "panels": len(plot_boxes),
            "boxes_xyxy": plot_boxes,
            "raw_predictions": raw_count,
            "centerline_reassigned": len(reassigned),
            "centerline_suppressed": len(suppressed),
            "counts": counts,
        })
        print(
            f"{image_path.name}: {len(plot_boxes)} panels; raw={raw_count}; "
            f"suppressed={len(suppressed)}; {counts}", flush=True
        )

    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output.resolve() / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
