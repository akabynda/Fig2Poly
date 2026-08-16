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


def intersection_over_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(1, union)


def overlap_over_smaller(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(1, min(area_a, area_b))


def detect_plot_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect open or closed plot areas from intersecting long horizontal/vertical axes."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 135)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(35, min(width, height) // 25),
        minLineLength=max(55, int(width * 0.11)), maxLineGap=max(8, int(width * 0.012)),
    )
    if lines is None:
        return [(0, 0, width, height)]

    horizontals: list[tuple[float, float, float, float]] = []
    verticals: list[tuple[float, float, float, float]] = []
    for raw in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = map(float, raw)
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx >= width * 0.11 and dx >= 7 * max(1.0, dy):
            horizontals.append((min(x1, x2), (y1 + y2) / 2, max(x1, x2), dx))
        elif dy >= height * 0.12 and dy >= 7 * max(1.0, dx):
            verticals.append(((x1 + x2) / 2, min(y1, y2), max(y1, y2), dy))

    # Hough often splits an axis at ticks or curve intersections. Join collinear
    # fragments before looking for corners.
    merged_h: list[tuple[float, float, float, float]] = []
    for x1, y, x2, _ in sorted(horizontals, key=lambda line: (line[1], line[0])):
        for idx, (old_x1, old_y, old_x2, _) in enumerate(merged_h):
            if abs(y - old_y) <= max(4, height * 0.004) and x1 <= old_x2 + width * 0.04 and x2 >= old_x1 - width * 0.04:
                new_x1, new_x2 = min(x1, old_x1), max(x2, old_x2)
                merged_h[idx] = (new_x1, (y + old_y) / 2, new_x2, new_x2 - new_x1)
                break
        else:
            merged_h.append((x1, y, x2, x2 - x1))
    horizontals = merged_h

    merged_v: list[tuple[float, float, float, float]] = []
    for x, y1, y2, _ in sorted(verticals, key=lambda line: (line[0], line[1])):
        for idx, (old_x, old_y1, old_y2, _) in enumerate(merged_v):
            if abs(x - old_x) <= max(4, width * 0.004) and y1 <= old_y2 + height * 0.04 and y2 >= old_y1 - height * 0.04:
                new_y1, new_y2 = min(y1, old_y1), max(y2, old_y2)
                merged_v[idx] = ((x + old_x) / 2, new_y1, new_y2, new_y2 - new_y1)
                break
        else:
            merged_v.append((x, y1, y2, y2 - y1))
    verticals = merged_v

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    join_x, join_y = width * 0.035, height * 0.045
    for hx1, hy, hx2, hlen in horizontals:
        for vx, vy1, vy2, vlen in verticals:
            if abs(vy2 - hy) > join_y or not (hx1 - join_x <= vx <= hx2 + join_x):
                continue
            # A real axis intersection is near an endpoint. Requiring this rejects
            # curve peaks and annotation strokes that happen to cross the x-axis.
            if min(abs(vx - hx1), abs(vx - hx2)) > max(join_x, hlen * 0.10):
                continue
            if abs(vx - hx1) <= abs(vx - hx2):
                px1, px2 = vx, hx2
            else:
                px1, px2 = hx1, vx
            py1, py2 = vy1, hy
            plot_w, plot_h = px2 - px1, py2 - py1
            if plot_w < width * 0.14 or plot_h < height * 0.12:
                continue
            if plot_w > width * 0.96 or plot_h > height * 0.92:
                continue
            border_supported = any(
                (abs(hy - py1) <= join_y or abs(hy - py2) <= join_y)
                and hx1 <= px1 + join_x and hx2 >= px2 - join_x
                for hx1, hy, hx2, _ in horizontals
            )
            if not border_supported:
                continue
            pad_x, pad_y = plot_w * 0.11, plot_h * 0.13
            box = (
                max(0, int(px1 - pad_x)), max(0, int(py1 - pad_y * 0.45)),
                min(width, int(px2 + pad_x * 0.45)), min(height, int(py2 + pad_y)),
            )
            candidates.append((hlen + vlen, box))

    # Closed plots are also recoverable from their two matching vertical sides.
    # This covers faint/broken horizontal axes that HoughLinesP may miss.
    for index, (vx1, vy11, vy12, vlen1) in enumerate(verticals):
        for vx2, vy21, vy22, vlen2 in verticals[index + 1:]:
            if abs(vy11 - vy21) > join_y or abs(vy12 - vy22) > join_y:
                continue
            px1, px2 = sorted((vx1, vx2))
            py1, py2 = min(vy11, vy21), max(vy12, vy22)
            plot_w, plot_h = px2 - px1, py2 - py1
            if plot_w < width * 0.14 or plot_h < height * 0.12:
                continue
            if plot_w > width * 0.96 or plot_h > height * 0.92:
                continue
            border_supported = any(
                (abs(hy - py1) <= join_y or abs(hy - py2) <= join_y)
                and hx1 <= px1 + join_x and hx2 >= px2 - join_x
                for hx1, hy, hx2, _ in horizontals
            )
            if not border_supported:
                continue
            pad_x, pad_y = plot_w * 0.11, plot_h * 0.13
            box = (max(0, int(px1 - pad_x)), max(0, int(py1 - pad_y * 0.45)),
                   min(width, int(px2 + pad_x * 0.45)), min(height, int(py2 + pad_y)))
            candidates.append((vlen1 + vlen2, box))

    selected: list[tuple[int, int, int, int]] = []
    for _, box in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(
            intersection_over_union(box, old) < 0.58
            and overlap_over_smaller(box, old) < 0.70
            for old in selected
        ):
            selected.append(box)
    selected.sort(key=lambda box: (box[1], box[0]))
    return selected if len(selected) >= 2 else [(0, 0, width, height)]


def render_threshold(
    image: np.ndarray,
    predictions: list[dict],
    threshold: float,
    output: Path,
) -> int:
    height, width = image.shape[:2]
    kept = [item for item in predictions if item["score"] >= threshold]
    kept.sort(key=lambda item: item["score"], reverse=True)
    overlay = image.astype(np.float32)
    curves_only = np.full_like(image, 255)
    instance_ids = np.zeros((height, width), dtype=np.uint16)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, item in enumerate(kept, 1):
        mask = item["mask"]
        color = np.asarray(PALETTE[(index - 1) % len(PALETTE)], dtype=np.float32)
        overlay[mask] = overlay[mask] * 0.52 + color * 0.48
        curves_only[mask] = color.astype(np.uint8)
        instance_ids[mask] = index
        name = f"mask_{index:03d}.png"
        cv2.imwrite(str(output / name), mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), 1, cv2.LINE_AA)
        x, y = item["bbox"][:2]
        cv2.putText(overlay, f"{index}: {item['score']:.2f}", (int(x), max(12, int(y))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color.tolist(), 1, cv2.LINE_AA)
        records.append({"id": index, "score": item["score"], "panel": item["panel"],
                        "bbox_xyxy": item["bbox"], "mask": name})
    cv2.imwrite(str(output / "overlay.png"), np.clip(overlay, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output / "curves_only.png"), curves_only)
    cv2.imwrite(str(output / "instance_ids.png"), instance_ids)
    (output / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.15, 0.3, 0.5, 0.7])
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from mmdet.apis import inference_detector, init_detector

    model = init_detector(str(root / "lineformer_swin_t_config.py"), str(args.weights), device="cuda:0")
    summary = []
    for image_path in sorted(args.input.resolve().iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        boxes = detect_plot_boxes(image)
        predictions: list[dict] = []
        panel_view = image.copy()
        for panel_index, (x1, y1, x2, y2) in enumerate(boxes, 1):
            cv2.rectangle(panel_view, (x1, y1), (x2 - 1, y2 - 1), PALETTE[(panel_index - 1) % len(PALETTE)], 3)
            cv2.putText(panel_view, f"panel {panel_index}", (x1 + 5, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, PALETTE[(panel_index - 1) % len(PALETTE)], 2)
            result = inference_detector(model, image[y1:y2, x1:x2])
            for box, crop_mask in zip(np.asarray(result[0][0]), result[1][0]):
                mask = np.zeros((height, width), dtype=bool)
                mask[y1:y2, x1:x2] = np.asarray(crop_mask, dtype=bool)
                predictions.append({"score": float(box[4]), "mask": mask, "panel": panel_index,
                                    "bbox": [float(box[0] + x1), float(box[1] + y1),
                                             float(box[2] + x1), float(box[3] + y1)]})
        image_dir = args.output.resolve() / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / "panels.png"), panel_view)
        counts = {}
        for threshold in args.thresholds:
            label = f"threshold_{threshold:.2f}"
            counts[label] = render_threshold(image, predictions, threshold, image_dir / label)
        summary.append({"image": image_path.name, "panels": len(boxes), "boxes_xyxy": boxes, "counts": counts})
        print(f"{image_path.name}: {len(boxes)} panels; {counts}", flush=True)
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output.resolve() / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
