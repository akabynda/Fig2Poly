import numpy as np

from training.line_metrics import (
    continuous_line_score,
    mask_to_centerline,
    match_line_instances,
)


def test_mask_to_centerline_ignores_thickness():
    mask = np.zeros((20, 24), dtype=bool)
    mask[7:12, 2:22] = True
    points = mask_to_centerline(mask, sample_interval=2)
    assert np.all(points[:, 1] == 9)


def test_continuous_score_rewards_same_curve():
    line = np.asarray([(x, 4 + x / 3) for x in range(20)], dtype=np.float32)
    shifted = line + np.asarray([0, 8])
    assert continuous_line_score(line, line) == 1
    assert continuous_line_score(line, shifted) < 0.5


def test_6b_penalizes_duplicate_predictions():
    line = np.asarray([(x, x) for x in range(10)], dtype=np.float32)
    result = match_line_instances([line, line], [line])
    assert result["score_6a"] == 1
    assert result["score_6b"] == 0.5
    assert result["count_error"] == 1
