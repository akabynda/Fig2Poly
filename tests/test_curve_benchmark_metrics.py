from __future__ import annotations

import cv2
import numpy as np

from training.evaluate_curve_benchmark import mask_to_official_points, matched_metrics, target_points


def test_official_point_extraction_tracks_line_center():
    mask = np.zeros((40, 80), dtype=bool)
    for x in range(5, 75):
        mask[10 + x // 10:13 + x // 10, x] = True
    points = mask_to_official_points(mask)
    assert len(points) > 50
    assert points[0, 0] >= 5
    assert np.all(np.diff(points[:, 0]) == 1)


def test_perfect_prediction_has_perfect_metrics(tmp_path):
    mask = np.zeros((30, 50), dtype=np.uint8)
    points = [(x, 10 + x // 10) for x in range(3, 47)]
    cv2.polylines(mask, [np.asarray(points, dtype=np.int32)], False, 255, 3)
    mask_path = tmp_path / "curve.png"
    cv2.imwrite(str(mask_path), mask)
    sample = {
        "manifest_root": str(tmp_path),
        "curves": [{
            "mask": "curve.png",
            "source_points": [{"x": x, "y": y} for x, y in points],
        }],
    }
    metrics = matched_metrics([{"mask": mask > 0}], sample, 30, 50)
    assert metrics["pred_count"] == 1
    assert metrics["count_exact"] == 1
    # The official LineFormer extractor samples every 10 px and linearly
    # interpolates, so even a perfect raster mask is a slightly lossy proxy.
    assert metrics["score_6a"] > 0.95
    assert metrics["mask_iou_penalized"] == 1.0


def test_target_points_falls_back_to_instance_mask():
    mask = np.zeros((30, 50), dtype=bool)
    mask[12:15, 4:46] = True
    sample = {"curves": [{"mask": "curve.png"}]}

    points = target_points(sample, [mask])

    assert len(points) == 1
    assert len(points[0]) > 30
