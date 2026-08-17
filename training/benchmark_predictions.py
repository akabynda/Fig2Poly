from __future__ import annotations

import io
import json
from pathlib import Path
import sqlite3
import time
from typing import Iterable, Iterator

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_manifest(manifest: Path) -> Iterator[dict]:
    root = manifest.resolve().parent
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            record["image_path"] = str((root / record["image"]).resolve())
            record["manifest_root"] = str(root)
            yield record


def iter_image_directory(directory: Path, expected_counts: dict[str, int] | None = None) -> Iterator[dict]:
    expected_counts = expected_counts or {}
    for path in sorted(item for item in directory.resolve().iterdir() if item.suffix.lower() in IMAGE_SUFFIXES):
        yield {
            "id": path.stem,
            "image": path.name,
            "image_path": str(path),
            "dataset_source": "real_test",
            "official_source_split": "local",
            "expected_count": expected_counts.get(path.name),
            "curves": [],
        }


def encode_predictions(predictions: list[dict], height: int, width: int) -> bytes:
    masks = np.stack([item["mask"] for item in predictions]).astype(bool) if predictions else np.zeros((0, height, width), dtype=bool)
    scores = np.asarray([item.get("score", 1.0) for item in predictions], dtype=np.float32)
    panels = np.asarray([item.get("panel", 1) for item in predictions], dtype=np.int16)
    boxes = np.asarray([item.get("bbox", [0, 0, 0, 0]) for item in predictions], dtype=np.float32).reshape(-1, 4)
    flat = masks.reshape((len(masks), height * width))
    packed = np.packbits(flat, axis=1)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, masks=packed, scores=scores, panels=panels, boxes=boxes,
                        height=np.int32(height), width=np.int32(width))
    return buffer.getvalue()


def decode_predictions(payload: bytes, source: str) -> list[dict]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        height, width = int(data["height"]), int(data["width"])
        packed = data["masks"]
        masks = np.unpackbits(packed, axis=1, count=height * width).reshape(-1, height, width).astype(bool)
        return [
            {
                "id": index + 1,
                "score": float(score),
                "panel": int(panel),
                "bbox": box.astype(float).tolist(),
                "mask": mask,
                "source": source,
            }
            for index, (score, panel, box, mask) in enumerate(zip(data["scores"], data["panels"], data["boxes"], masks))
        ]


class PredictionStore:
    def __init__(self, path: Path, writable: bool = True):
        path.parent.mkdir(parents=True, exist_ok=True)
        uri = str(path.resolve()) if writable else f"file:{path.resolve().as_posix()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=not writable, timeout=60)
        if writable:
            self.connection.execute(
                """CREATE TABLE IF NOT EXISTS predictions (
                    dataset TEXT NOT NULL, sample_id TEXT NOT NULL,
                    image_path TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                    raw BLOB, processed BLOB, raw_inference_seconds REAL NOT NULL,
                    processed_inference_seconds REAL NOT NULL,
                    postprocess_seconds REAL NOT NULL, panels INTEGER NOT NULL,
                    error TEXT, completed_at REAL NOT NULL,
                    PRIMARY KEY(dataset, sample_id)
                )"""
            )
            self.connection.commit()

    def completed(self, dataset: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT sample_id FROM predictions WHERE dataset=?", (dataset,)
        )
        return {str(row[0]) for row in rows}

    def put(self, dataset: str, sample: dict, width: int, height: int,
            raw: list[dict] | None, processed: list[dict] | None,
            raw_inference_seconds: float, processed_inference_seconds: float,
            postprocess_seconds: float, panels: int,
            error: str | None = None) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (dataset, str(sample["id"]), str(sample["image_path"]), width, height,
             encode_predictions(raw or [], height, width) if raw is not None else None,
             encode_predictions(processed or [], height, width) if processed is not None else None,
             raw_inference_seconds, processed_inference_seconds,
             postprocess_seconds, panels, error, time.time()),
        )
        self.connection.commit()

    def get(self, dataset: str, sample_id: str, variant: str, source: str) -> tuple[list[dict], dict]:
        if variant not in {"raw", "processed"}:
            raise ValueError(variant)
        row = self.connection.execute(
            f"SELECT {variant}, width, height, raw_inference_seconds, "
            "processed_inference_seconds, postprocess_seconds, panels, error "
            "FROM predictions WHERE dataset=? AND sample_id=?", (dataset, sample_id),
        ).fetchone()
        if row is None:
            raise KeyError((dataset, sample_id))
        predictions = decode_predictions(row[0], source) if row[0] is not None else []
        return predictions, {
            "width": row[1], "height": row[2], "raw_inference_seconds": row[3],
            "processed_inference_seconds": row[4], "postprocess_seconds": row[5],
            "panels": row[6], "error": row[7],
        }

    def close(self) -> None:
        self.connection.close()


def prediction_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def threshold_predictions(predictions: Iterable[dict], threshold: float) -> list[dict]:
    return [item for item in predictions if float(item.get("score", 1.0)) >= threshold and np.any(item["mask"])]
