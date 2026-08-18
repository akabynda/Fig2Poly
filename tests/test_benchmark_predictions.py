from __future__ import annotations

import numpy as np

from training.benchmark_predictions import PredictionStore


def test_prediction_store_round_trip(tmp_path):
    mask = np.zeros((7, 11), dtype=bool)
    mask[2:5, 3:9] = True
    sample = {"id": "one", "image_path": "/data/one.png"}
    prediction = {"score": 0.75, "panel": 2, "bbox": [3, 2, 9, 5], "mask": mask}
    path = tmp_path / "predictions.sqlite"
    store = PredictionStore(path)
    store.put("set", sample, 11, 7, [prediction], [prediction], 1.0, 1.5, 0.2, 2)
    store.close()
    store = PredictionStore(path, writable=False)
    restored, metadata = store.get("set", "one", "processed", "lineformer")
    store.close()
    assert len(restored) == 1
    assert np.array_equal(restored[0]["mask"], mask)
    assert restored[0]["panel"] == 2
    assert metadata["processed_inference_seconds"] == 1.5


def test_prediction_store_handles_empty_predictions(tmp_path):
    path = tmp_path / "empty.sqlite"
    store = PredictionStore(path)
    sample = {"id": "empty", "image_path": "/data/empty.png"}
    store.put("set", sample, 5, 4, [], [], 0.1, 0.2, 0.0, 1)
    restored, _ = store.get("set", "empty", "raw", "maskdino")
    store.close()
    assert restored == []


def test_prediction_store_retries_errors(tmp_path):
    path = tmp_path / "retry.sqlite"
    store = PredictionStore(path)
    sample = {"id": "failed", "image_path": "/data/failed.png"}
    store.put("set", sample, 5, 4, None, None, 0, 0, 0, 0, "IndexError")

    assert "failed" not in store.completed("set")
    store.close()
