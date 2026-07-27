from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def skeletonize(mask: np.ndarray) -> np.ndarray:
    image = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = eroded
    return skeleton > 0


def pair_metrics(pred: np.ndarray, target: np.ndarray, tolerance: float) -> tuple[float, float]:
    p, t = skeletonize(pred), skeletonize(target)
    if not p.any() or not t.any():
        return 0.0, float("inf")
    distance_to_target = distance_transform_edt(~t)
    distance_to_pred = distance_transform_edt(~p)
    precision = float((distance_to_target[p] <= tolerance).mean())
    recall = float((distance_to_pred[t] <= tolerance).mean())
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    chamfer = (float(distance_to_target[p].mean()) + float(distance_to_pred[t].mean())) / 2
    return f1, chamfer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Thickness-invariant YOLO curve evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--conf", type=float, default=.05)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("runs/yolo26/centerline_metrics.json"))
    args = parser.parse_args(argv)
    root = args.dataset.resolve()
    images = sorted((root / "images" / args.split).glob("*.jpg"))[:args.limit]
    model = YOLO(args.model)
    totals = {"images": 0, "tp": 0, "fp": 0, "fn": 0, "matched": 0,
              "f1_sum": 0.0, "chamfer_sum": 0.0, "count_error": 0.0}
    results = model.predict([str(path) for path in images], imgsz=args.imgsz, conf=args.conf,
                            retina_masks=True, stream=True, verbose=False)
    for image_path, result in zip(images, results):
        targets = [np.asarray(Image.open(path).convert("L")) > 0 for path in
                   sorted((root / "curve_masks" / args.split / image_path.stem).glob("curve_*.png"))]
        target = np.stack(targets) if targets else np.zeros((0, result.orig_shape[0], result.orig_shape[1]), bool)
        pred = (result.masks.data.cpu().numpy() > .5) if result.masks is not None else np.zeros((0, *result.orig_shape), bool)
        scores = np.zeros((len(pred), len(target)), np.float32)
        distances = np.full_like(scores, np.inf)
        for i in range(len(pred)):
            for j in range(len(target)):
                scores[i, j], distances[i, j] = pair_metrics(pred[i], target[j], args.tolerance)
        rows, cols = linear_sum_assignment(-scores) if scores.size else (np.array([], int), np.array([], int))
        matched_scores = scores[rows, cols] if len(rows) else np.array([])
        good = matched_scores >= .5
        totals["tp"] += int(good.sum()); totals["fp"] += len(pred) - int(good.sum())
        totals["fn"] += len(target) - int(good.sum()); totals["matched"] += len(rows)
        totals["f1_sum"] += float(matched_scores.sum())
        totals["chamfer_sum"] += float(distances[rows, cols][np.isfinite(distances[rows, cols])].sum()) if len(rows) else 0
        totals["count_error"] += abs(len(pred) - len(target)); totals["images"] += 1
        if totals["images"] % 50 == 0:
            print(f"evaluated {totals['images']}/{len(images)}", flush=True)
    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    report = {"model": args.model, "split": args.split, "tolerance_px": args.tolerance,
              "confidence": args.conf, "images": totals["images"],
              "instance_precision_centerline": precision, "instance_recall_centerline": recall,
              "instance_f1_centerline": 2*precision*recall/max(1e-9, precision+recall),
              "mean_matched_centerline_f1": totals["f1_sum"]/max(1,totals["matched"]),
              "mean_symmetric_chamfer_px": totals["chamfer_sum"]/max(1,totals["matched"]),
              "curve_count_mae": totals["count_error"]/max(1,totals["images"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
