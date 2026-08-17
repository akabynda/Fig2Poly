from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Callable, Iterable

import cv2
import numpy as np

from training.benchmark_predictions import PredictionStore, iter_image_directory, iter_manifest, prediction_bbox
from training.predict_lineformer_panels import clean_prediction_tracks, detect_plot_boxes, suppress_centerline_duplicates


InferenceFunction = Callable[[np.ndarray, float], list[dict]]


def parse_dataset(value: str) -> tuple[str, Path, str]:
    parts = value.split("=", 2)
    if len(parts) != 3 or parts[1] not in {"manifest", "images"}:
        raise argparse.ArgumentTypeError("dataset must be NAME=manifest=PATH or NAME=images=PATH")
    return parts[0], Path(parts[2]), parts[1]


def run_benchmark_inference(
    infer: InferenceFunction,
    datasets: Iterable[tuple[str, Path, str]],
    output: Path,
    model_name: str,
    min_threshold: float = 0.15,
    stop_on_error: bool = False,
) -> None:
    store = PredictionStore(output)
    for dataset, path, kind in datasets:
        samples = iter_manifest(path) if kind == "manifest" else iter_image_directory(path)
        completed = store.completed(dataset)
        processed_count = 0
        for sample in samples:
            sample_id = str(sample["id"])
            if sample_id in completed:
                continue
            image = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
            if image is None:
                message = f"Unable to read {sample['image_path']}"
                store.put(dataset, sample, 0, 0, None, None, 0, 0, 0, 0, message)
                if stop_on_error:
                    raise RuntimeError(message)
                continue
            height, width = image.shape[:2]
            try:
                start = time.perf_counter()
                raw = infer(image, min_threshold)
                raw_seconds = time.perf_counter() - start
                for item in raw:
                    item.setdefault("panel", 1)
                    item.setdefault("bbox", prediction_bbox(item["mask"]))

                boxes = detect_plot_boxes(image)
                panel_predictions: list[dict] = []
                panel_start = time.perf_counter()
                for panel_index, (x1, y1, x2, y2) in enumerate(boxes, 1):
                    crop = image[y1:y2, x1:x2]
                    for item in infer(crop, min_threshold):
                        crop_mask = np.asarray(item["mask"], dtype=bool)
                        mask = np.zeros((height, width), dtype=bool)
                        mask[y1:y2, x1:x2] = crop_mask
                        panel_predictions.append({
                            **item,
                            "mask": mask,
                            "panel": panel_index,
                            "bbox": prediction_bbox(mask),
                        })
                panel_seconds = time.perf_counter() - panel_start
                post_start = time.perf_counter()
                processed, _ = clean_prediction_tracks(panel_predictions, height, image)
                processed, _ = suppress_centerline_duplicates(processed, height)
                post_seconds = time.perf_counter() - post_start
                store.put(
                    dataset, sample, width, height, raw, processed,
                    raw_seconds, panel_seconds, post_seconds, len(boxes), None,
                )
                processed_count += 1
                if processed_count % 25 == 0:
                    print(f"{model_name}/{dataset}: {processed_count} new images; last={sample_id}", flush=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
                store.put(dataset, sample, width, height, None, None, 0, 0, 0, 0, error)
                print(f"ERROR {model_name}/{dataset}/{sample_id}: {exc}", flush=True)
                if stop_on_error:
                    store.close()
                    raise
        print(f"complete {model_name}/{dataset}: {processed_count} new images", flush=True)
    store.close()


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-threshold", type=float, default=0.15)
    parser.add_argument("--stop-on-error", action="store_true")
