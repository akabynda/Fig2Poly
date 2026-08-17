from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import medfilt
from scipy.spatial import cKDTree

from training.benchmark_inference import parse_dataset
from training.benchmark_predictions import PredictionStore, iter_image_directory, iter_manifest, threshold_predictions
from training.fuse_curve_predictions import _prepare, add_complex_lineformer_rescues, fuse_panel
from training.line_metrics import continuous_line_score


VARIANTS = (
    "lineformer",
    "lineformer_post",
    "maskdino",
    "maskdino_post",
    "maskdino_lineformer",
)

PAPER_REFERENCE = (
    {"dataset": "adobe_synth19", "variant": "lineformer_paper_reported", "score_6a": 0.9751, "score_6b": 0.9702},
    {"dataset": "ub_pmc22", "variant": "lineformer_paper_reported", "score_6a": 0.9310, "score_6b": 0.8825},
    {"dataset": "lineex", "variant": "lineformer_paper_reported", "score_6a": 0.9920, "score_6b": 0.9757},
)


def mask_to_official_points(mask: np.ndarray, interval: int = 10) -> np.ndarray:
    """Reproduce LineFormer's mask -> center points -> linear interpolation."""
    binary = np.asarray(mask, dtype=bool)
    signal = medfilt(binary.sum(axis=0).astype(np.float32), kernel_size=5)
    nonzero = np.flatnonzero(signal)
    if not len(nonzero):
        return np.zeros((0, 2), dtype=np.float32)
    sampled: list[tuple[float, float]] = []
    for x in range(int(nonzero[0]), int(nonzero[-1]), interval):
        ys = np.flatnonzero(binary[:, x])
        if not len(ys):
            continue
        components = np.split(ys, np.flatnonzero(np.diff(ys) > 2) + 1)
        if len(components) == 1:
            sampled.append((float(x), float(round((int(ys[0]) + int(ys[-1])) / 2))))
    if len(sampled) < 2:
        return np.asarray(sampled, dtype=np.float32).reshape(-1, 2)
    points = np.asarray(sampled, dtype=np.float64)
    xs = np.arange(int(points[0, 0]), int(points[-1, 0]) + 1, dtype=np.float64)
    ys = np.interp(xs, points[:, 0], points[:, 1])
    return np.column_stack((xs, ys)).astype(np.float32)


def target_points(sample: dict) -> list[np.ndarray]:
    return [
        np.asarray([[float(point["x"]), float(point["y"])] for point in curve["source_points"]], dtype=np.float32)
        for curve in sample.get("curves", [])
    ]


def target_masks(sample: dict, shape: tuple[int, int]) -> list[np.ndarray]:
    root = Path(sample.get("manifest_root", "."))
    result = []
    for curve in sample.get("curves", []):
        path = root / curve["mask"]
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            result.append(np.zeros(shape, dtype=bool))
        else:
            result.append(mask > 0)
    return result


def matched_metrics(predictions: list[dict], sample: dict, height: int, width: int) -> dict[str, float | int | None]:
    predicted_points = [mask_to_official_points(item["mask"]) for item in predictions]
    gt_points = target_points(sample)
    scores = np.zeros((len(gt_points), len(predicted_points)), dtype=np.float64)
    for gt_index, gt in enumerate(gt_points):
        for pred_index, pred in enumerate(predicted_points):
            scores[gt_index, pred_index] = continuous_line_score(pred, gt)
    if scores.size:
        rows, columns = linear_sum_assignment(-scores)
        matched_scores = scores[rows, columns]
    else:
        rows = columns = np.asarray([], dtype=int)
        matched_scores = np.asarray([], dtype=float)
    matched_sum = float(matched_scores.sum())
    gt_count, pred_count = len(gt_points), len(predicted_points)
    denominator_6a = max(1, gt_count)
    denominator_6b = max(1, gt_count, pred_count)
    result: dict[str, float | int | None] = {
        "gt_count": gt_count,
        "pred_count": pred_count,
        "count_error": abs(pred_count - gt_count),
        "count_signed_error": pred_count - gt_count,
        "count_exact": int(pred_count == gt_count),
        "score_6a": matched_sum / denominator_6a,
        "score_6b": matched_sum / denominator_6b,
        "matched_instances": len(rows),
        "matched_similarity_mean": float(matched_scores.mean()) if len(matched_scores) else 0.0,
        "matched_similarity_median": float(np.median(matched_scores)) if len(matched_scores) else 0.0,
        "matched_similarity_min": float(matched_scores.min()) if len(matched_scores) else 0.0,
    }

    gt_masks = target_masks(sample, (height, width))
    ious, dices, chamfers, hausdorff95 = [], [], [], []
    tolerance_counts = {2: [0, 0, 0, 0], 5: [0, 0, 0, 0], 10: [0, 0, 0, 0]}
    for gt_index, pred_index in zip(rows, columns):
        gt_mask = gt_masks[int(gt_index)]
        pred_mask = predictions[int(pred_index)]["mask"]
        intersection = np.count_nonzero(gt_mask & pred_mask)
        union = np.count_nonzero(gt_mask | pred_mask)
        ious.append(intersection / max(1, union))
        dices.append(2 * intersection / max(1, np.count_nonzero(gt_mask) + np.count_nonzero(pred_mask)))
        gt = gt_points[int(gt_index)]
        pred = predicted_points[int(pred_index)]
        if len(gt) and len(pred):
            gt_dist = cKDTree(pred).query(gt, k=1)[0]
            pred_dist = cKDTree(gt).query(pred, k=1)[0]
            chamfers.append(float((gt_dist.mean() + pred_dist.mean()) / 2))
            hausdorff95.append(float(max(np.percentile(gt_dist, 95), np.percentile(pred_dist, 95))))
            for tolerance, counts in tolerance_counts.items():
                counts[0] += int(np.count_nonzero(pred_dist <= tolerance))
                counts[1] += len(pred_dist)
                counts[2] += int(np.count_nonzero(gt_dist <= tolerance))
                counts[3] += len(gt_dist)
    matched_gt, matched_pred = set(map(int, rows)), set(map(int, columns))
    for index, points in enumerate(predicted_points):
        if index not in matched_pred:
            for counts in tolerance_counts.values():
                counts[1] += len(points)
    for index, points in enumerate(gt_points):
        if index not in matched_gt:
            for counts in tolerance_counts.values():
                counts[3] += len(points)
    penalty_count = max(1, gt_count, pred_count)
    result.update({
        "mask_iou_matched_mean": float(np.mean(ious)) if ious else 0.0,
        "mask_iou_penalized": float(np.sum(ious) / penalty_count),
        "mask_dice_matched_mean": float(np.mean(dices)) if dices else 0.0,
        "mask_dice_penalized": float(np.sum(dices) / penalty_count),
        "chamfer_px_matched_mean": float(np.mean(chamfers)) if chamfers else None,
        "chamfer_norm_matched_mean": float(np.mean(chamfers) / math.hypot(width, height)) if chamfers else None,
        "hausdorff95_px_matched_mean": float(np.mean(hausdorff95)) if hausdorff95 else None,
        "hausdorff95_norm_matched_mean": float(np.mean(hausdorff95) / math.hypot(width, height)) if hausdorff95 else None,
    })
    for tolerance, (precision_hits, precision_total, recall_hits, recall_total) in tolerance_counts.items():
        precision = precision_hits / max(1, precision_total)
        recall = recall_hits / max(1, recall_total)
        result[f"centerline_precision_{tolerance}px"] = precision
        result[f"centerline_recall_{tolerance}px"] = recall
        result[f"centerline_f1_{tolerance}px"] = 2 * precision * recall / max(1e-12, precision + recall)
    return result


def no_mask_metrics(predictions: list[dict], expected_count: int | None, height: int, width: int) -> dict:
    coverages, roughness, components = [], [], []
    for item in predictions:
        points = mask_to_official_points(item["mask"], interval=4)
        coverages.append((float(np.ptp(points[:, 0])) + 1) / width if len(points) else 0.0)
        if len(points) >= 5:
            roughness.append(float(np.median(np.abs(np.diff(points[:, 1], n=2)))) / max(1, height))
        count, _ = cv2.connectedComponents(item["mask"].astype(np.uint8))
        components.append(max(0, count - 1))
    pred_count = len(predictions)
    return {
        "gt_count": expected_count,
        "pred_count": pred_count,
        "count_error": abs(pred_count - expected_count) if expected_count is not None else None,
        "count_signed_error": pred_count - expected_count if expected_count is not None else None,
        "count_exact": int(pred_count == expected_count) if expected_count is not None else None,
        "score_6a": None, "score_6b": None,
        "mean_horizontal_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_normalized_roughness": float(np.mean(roughness)) if roughness else 0.0,
        "mean_components_per_instance": float(np.mean(components)) if components else 0.0,
    }


def fused_predictions(maskdino: list[dict], lineformer: list[dict], lineformer_low: list[dict], height: int) -> list[dict]:
    maskdino = [_prepare(item) for item in maskdino]
    lineformer = [_prepare(item) for item in lineformer]
    lineformer_low = [_prepare(item) for item in lineformer_low]
    panels = sorted({item["panel"] for item in maskdino + lineformer + lineformer_low})
    fused: list[dict] = []
    for panel in panels:
        tracks, _ = fuse_panel(
            [item for item in maskdino if item["panel"] == panel],
            [item for item in lineformer if item["panel"] == panel], height,
        )
        fused.extend(tracks)
    fused, _ = add_complex_lineformer_rescues(fused, lineformer, lineformer_low, maskdino, height)
    return fused


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for variant in VARIANTS:
            group = [row for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            if not group:
                continue
            item = {"dataset": dataset, "variant": variant, "images": len(group),
                    "failures": sum(bool(row.get("error")) for row in group)}
            numeric_keys = sorted(set().union(*(row.keys() for row in group)) - {
                "dataset", "split", "image_id", "image", "variant", "error"
            })
            for key in numeric_keys:
                values = [float(row[key]) for row in group if row.get(key) not in (None, "") and not isinstance(row.get(key), str)]
                if not values:
                    continue
                item[f"{key}_mean"] = statistics.fmean(values)
                item[f"{key}_median"] = statistics.median(values)
                item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                item[f"{key}_p05"] = float(np.percentile(values, 5))
                item[f"{key}_p95"] = float(np.percentile(values, 95))
            summaries.append(item)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate five curve segmentation variants")
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--lineformer-db", type=Path, required=True)
    parser.add_argument("--maskdino-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--rescue-threshold", type=float, default=0.15)
    parser.add_argument("--real-test-counts", type=Path)
    args = parser.parse_args(argv)
    counts = json.loads(args.real_test_counts.read_text(encoding="utf-8")) if args.real_test_counts else {}
    lineformer_store = PredictionStore(args.lineformer_db, writable=False)
    maskdino_store = PredictionStore(args.maskdino_db, writable=False)
    rows: list[dict] = []
    dataset_counts: dict[str, int] = {}
    for dataset, path, kind in args.dataset:
        samples = iter_manifest(path) if kind == "manifest" else iter_image_directory(path, counts)
        for sample in samples:
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            sample_id = str(sample["id"])
            lf_raw, lf_meta = lineformer_store.get(dataset, sample_id, "raw", "lineformer")
            lf_post, _ = lineformer_store.get(dataset, sample_id, "processed", "lineformer")
            md_raw, md_meta = maskdino_store.get(dataset, sample_id, "raw", "maskdino")
            md_post, _ = maskdino_store.get(dataset, sample_id, "processed", "maskdino")
            height, width = int(lf_meta["height"]), int(lf_meta["width"])
            lf_raw_30 = threshold_predictions(lf_raw, args.threshold)
            lf_post_30 = threshold_predictions(lf_post, args.threshold)
            md_raw_30 = threshold_predictions(md_raw, args.threshold)
            md_post_30 = threshold_predictions(md_post, args.threshold)
            fusion_start = time.perf_counter()
            fusion = fused_predictions(md_post_30, lf_post_30,
                                       threshold_predictions(lf_post, args.rescue_threshold), height)
            fusion_seconds = time.perf_counter() - fusion_start
            variants = {
                "lineformer": (lf_raw_30, lf_meta["raw_inference_seconds"], 0.0, lf_meta["error"]),
                "lineformer_post": (lf_post_30, lf_meta["processed_inference_seconds"], lf_meta["postprocess_seconds"], lf_meta["error"]),
                "maskdino": (md_raw_30, md_meta["raw_inference_seconds"], 0.0, md_meta["error"]),
                "maskdino_post": (md_post_30, md_meta["processed_inference_seconds"], md_meta["postprocess_seconds"], md_meta["error"]),
                "maskdino_lineformer": (
                    fusion,
                    lf_meta["processed_inference_seconds"] + md_meta["processed_inference_seconds"],
                    lf_meta["postprocess_seconds"] + md_meta["postprocess_seconds"] + fusion_seconds,
                    lf_meta["error"] or md_meta["error"],
                ),
            }
            has_masks = bool(sample.get("curves"))
            for variant, (predictions, inference_seconds, postprocess_seconds, error) in variants.items():
                metrics = (matched_metrics(predictions, sample, height, width) if has_masks else
                           no_mask_metrics(predictions, sample.get("expected_count"), height, width))
                rows.append({
                    "dataset": dataset, "split": sample.get("official_source_split", kind),
                    "image_id": sample_id, "image": sample.get("image"), "variant": variant,
                    "width": width, "height": height, "panels": lf_meta["panels"],
                    "inference_seconds": inference_seconds,
                    "postprocess_seconds": postprocess_seconds,
                    "total_seconds": inference_seconds + postprocess_seconds,
                    "error": error, **metrics,
                })
        print(f"evaluated {dataset}", flush=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_image_metrics.csv", rows)
    write_csv(output / "summary_metrics.csv", summarize(rows))
    write_csv(output / "paper_reference_metrics.csv", [
        {
            **row,
            "split": "paper-reported split",
            "source": "Lal et al., LineFormer, ICDAR 2023",
            "doi": "10.1007/978-3-031-41734-4_24",
            "note": (
                "Values reported by the paper; not recomputed by this run. "
                "For UB-PMC22 the paper's 158-image filename manifest is not public."
            ),
        }
        for row in PAPER_REFERENCE
    ])
    (output / "protocol.json").write_text(json.dumps({
        "threshold": args.threshold, "rescue_threshold": args.rescue_threshold,
        "variants": VARIANTS,
        "dataset_image_counts": dataset_counts,
        "dataset_notes": {
            "lineex": "Official released validation split (10,000), used as test by the LineFormer config.",
            "adobe_synth19": "Official CHART-Info-19 Task-6 test release.",
            "ub_pmc22": (
                "Complete public ICPR2022 UB/UNITEC PMC TEST v2.1 line subset. "
                "It has 397 usable line charts; the paper reports a 158-image test "
                "but does not publish that exact filename manifest, so its reported "
                "93.10/88.25 values are reference-only rather than directly paired."
            ),
        },
        "score_6a_6b": "Official LineFormer/ChartInfo continuous-line formula; 6b pads count mismatches",
        "balanced_v5_test_note": (
            "Full 40,000-image held-out test split from combined_lineex_balanced_v5; "
            "instance masks and source points enable the complete metric suite"
        ),
    }, indent=2), encoding="utf-8")
    lineformer_store.close()
    maskdino_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
