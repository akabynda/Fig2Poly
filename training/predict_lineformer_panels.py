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
    return selected if selected else [(0, 0, width, height)]


def mask_centerline(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs: list[int] = []
    ys: list[float] = []
    for x in range(mask.shape[1]):
        column = np.flatnonzero(mask[:, x])
        if column.size:
            xs.append(x)
            ys.append(float(np.median(column)))
    return np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.float32)


def _track_baseline(mask: np.ndarray) -> float:
    _, ys = mask_centerline(mask)
    return float(np.median(ys)) if len(ys) else float("nan")


def _mask_color(lab_image: np.ndarray | None, bgr_image: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if lab_image is None or bgr_image is None or not np.any(mask):
        return None
    pixels = bgr_image[mask]
    # Ignore white antialiased background captured by a thick predicted mask.
    chroma = pixels.max(axis=1).astype(np.int16) - pixels.min(axis=1).astype(np.int16)
    useful = (chroma >= 12) | (pixels.max(axis=1) <= 190)
    values = lab_image[mask][useful]
    if len(values) < 5:
        values = lab_image[mask]
    return np.median(values.astype(np.float32), axis=0)


def clean_prediction_tracks(
    predictions: list[dict], image_height: int, image: np.ndarray | None = None
) -> tuple[list[dict], list[dict]]:
    """Keep one coherent curve track per query and reassign leaked fragments.

    LineFormer occasionally puts disconnected pieces of two neighbouring curves in
    one query. The largest connected component anchors the query. Components close
    to that anchor in image space or in their typical y-level stay with it; remaining
    components may be reassigned to a better query from the same plot panel.
    """
    tolerance = max(4.0, image_height * 0.008)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if image is not None else None
    tracks: list[dict] = []
    components: list[dict] = []
    diagnostics: list[dict] = []
    for source_index, item in enumerate(predictions):
        mask = item["mask"].astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        component_ids = sorted(
            range(1, count), key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]), reverse=True
        )
        if not component_ids:
            continue
        largest_area = int(stats[component_ids[0], cv2.CC_STAT_AREA])
        component_ids = [
            idx for idx in component_ids
            if int(stats[idx, cv2.CC_STAT_AREA]) >= max(4, int(largest_area * 0.002))
        ]
        primary = labels == component_ids[0]
        for component_id in component_ids[1:]:
            component = labels == component_id
            components.append({"mask": component, "source": source_index})
        cleaned = {**item, "mask": primary}
        cleaned["track_color"] = _mask_color(lab_image, image, primary)
        tracks.append(cleaned)

    # Assign every non-anchor connected component globally. At a T-junction the
    # nearest curve is often the wrong one; a strong colour match may therefore
    # override a small geometric disadvantage.
    for component_item in components:
        best_index = None
        best_cost = float("inf")
        component = component_item["mask"]
        component_baseline = _track_baseline(component)
        component_color = _mask_color(lab_image, image, component)
        for track_index, track in enumerate(tracks):
            if track["panel"] != predictions[component_item["source"]]["panel"]:
                continue
            distance = cv2.distanceTransform((~track["mask"]).astype(np.uint8), cv2.DIST_L2, 5)
            spatial_gap = float(distance[component].min())
            level_gap = abs(component_baseline - _track_baseline(track["mask"]))
            geometry_cost = min(spatial_gap / tolerance, level_gap / (tolerance * 2.5))
            color_distance = None
            if component_color is not None and track["track_color"] is not None:
                color_distance = float(np.linalg.norm(component_color - track["track_color"]))
            normally_reachable = geometry_cost <= 1.0
            color_bridge = color_distance is not None and color_distance <= 20.0 and spatial_gap <= tolerance * 3.0
            if not normally_reachable and not color_bridge:
                continue
            color_cost = min(4.0, color_distance / 30.0) if color_distance is not None else 0.0
            cost = geometry_cost + color_cost * 1.5
            if cost < best_cost:
                best_index, best_cost = track_index, cost
        if best_index is not None:
            tracks[best_index]["mask"] |= component
            diagnostics.append({
                "from_prediction": component_item["source"] + 1,
                "to_prediction": best_index + 1,
                "normalized_cost": best_cost,
            })

    for track in tracks:
        track.pop("track_color", None)
        ys, xs = np.nonzero(track["mask"])
        if len(xs):
            track["bbox"] = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    return tracks, diagnostics


def centerlines_are_duplicates(
    first: dict,
    second: dict,
    image_height: int,
    min_overlap: float = 0.65,
    distance_ratio: float = 0.008,
) -> tuple[bool, dict[str, float]]:
    if first["panel"] != second["panel"]:
        return False, {}
    x1, y1 = first["centerline"]
    x2, y2 = second["centerline"]
    common = np.intersect1d(x1, x2, assume_unique=True)
    overlap = len(common) / max(1, min(len(x1), len(x2)))
    if overlap < min_overlap or not len(common):
        return False, {"x_overlap": float(overlap)}
    distances = np.abs(np.interp(common, x1, y1) - np.interp(common, x2, y2))
    median = float(np.median(distances))
    p90 = float(np.percentile(distances, 90))
    tolerance = max(3.0, image_height * distance_ratio)
    duplicate = median <= tolerance and p90 <= tolerance * 2.0
    return duplicate, {
        "x_overlap": float(overlap), "median_distance": median,
        "p90_distance": p90, "tolerance": tolerance,
    }


def suppress_centerline_duplicates(
    predictions: list[dict], image_height: int
) -> tuple[list[dict], list[dict]]:
    for item in predictions:
        item["centerline"] = mask_centerline(item["mask"])
    kept: list[dict] = []
    suppressed: list[dict] = []
    for candidate in sorted(predictions, key=lambda item: item["score"], reverse=True):
        duplicate_of = None
        duplicate_metrics: dict[str, float] = {}
        for existing in kept:
            duplicate, metrics = centerlines_are_duplicates(candidate, existing, image_height)
            if duplicate:
                duplicate_of, duplicate_metrics = existing, metrics
                break
        if duplicate_of is None:
            kept.append(candidate)
        else:
            new_pixels = int(np.count_nonzero(candidate["mask"] & ~duplicate_of["mask"]))
            duplicate_of["mask"] |= candidate["mask"]
            duplicate_of["centerline"] = mask_centerline(duplicate_of["mask"])
            ys, xs = np.nonzero(duplicate_of["mask"])
            if len(xs):
                duplicate_of["bbox"] = [
                    float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
                ]
            suppressed.append({
                "panel": candidate["panel"],
                "score": candidate["score"],
                "kept_score": duplicate_of["score"],
                "merged_unique_pixels": new_pixels,
                **duplicate_metrics,
            })
    return kept, suppressed


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
    for stale_mask in output.glob("mask_*.png"):
        stale_mask.unlink()
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
        minimum_threshold = min(args.thresholds)
        predictions = [item for item in predictions if item["score"] >= minimum_threshold]
        raw_count = len(predictions)
        predictions, reassigned = clean_prediction_tracks(predictions, height, image)
        predictions, suppressed = suppress_centerline_duplicates(predictions, height)
        image_dir = args.output.resolve() / image_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / "panels.png"), panel_view)
        (image_dir / "centerline_suppressed.json").write_text(
            json.dumps(suppressed, indent=2), encoding="utf-8"
        )
        (image_dir / "centerline_reassigned.json").write_text(
            json.dumps(reassigned, indent=2), encoding="utf-8"
        )
        counts = {}
        for threshold in args.thresholds:
            label = f"threshold_{threshold:.2f}"
            counts[label] = render_threshold(image, predictions, threshold, image_dir / label)
        summary.append({"image": image_path.name, "panels": len(boxes), "boxes_xyxy": boxes,
                        "raw_predictions": raw_count, "centerline_reassigned": len(reassigned),
                        "centerline_suppressed": len(suppressed),
                        "counts": counts})
        print(f"{image_path.name}: {len(boxes)} panels; suppressed={len(suppressed)}; {counts}", flush=True)
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output.resolve() / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
