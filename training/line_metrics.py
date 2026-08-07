from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def mask_to_centerline(mask: np.ndarray, sample_interval: int = 4) -> np.ndarray:
    """Convert one instance mask to LineFormer-style (x, centre-y) samples."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    points: list[tuple[float, float]] = []
    for x in range(0, binary.shape[1], sample_interval):
        ys = np.flatnonzero(binary[:, x])
        if ys.size:
            points.append((float(x), float(np.median(ys))))
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def continuous_line_score(prediction: np.ndarray, target: np.ndarray) -> float:
    """LineFormer/ChartInfo continuous-series score for one pair of lines."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    if not len(prediction) or not len(target):
        return float(not len(prediction) and not len(target))
    prediction = prediction[np.argsort(prediction[:, 0])]
    target = target[np.argsort(target[:, 0])]

    def weighted_recall(reference: np.ndarray, candidate: np.ndarray) -> float:
        if len(reference) == 1:
            intervals = np.ones(1, dtype=np.float64)
        else:
            intervals = np.empty(len(reference), dtype=np.float64)
            intervals[0] = (reference[1, 0] - reference[0, 0]) / 2
            intervals[-1] = (reference[-1, 0] - reference[-2, 0]) / 2
            if len(reference) > 2:
                intervals[1:-1] = (reference[2:, 0] - reference[:-2, 0]) / 2
            intervals = np.maximum(intervals, 0)
        epsilon = max(1e-6, float(np.ptp(reference[:, 1])) / 100)
        interpolated = np.interp(reference[:, 0], candidate[:, 0], candidate[:, 1])
        relative_error = np.minimum(
            1.0,
            np.abs(reference[:, 1] - interpolated)
            / (np.abs(reference[:, 1]) + epsilon),
        )
        denominator = float(intervals.sum())
        return float(((1 - relative_error) * intervals).sum() / max(denominator, 1e-9))

    recall = weighted_recall(target, prediction)
    precision = weighted_recall(prediction, target)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def match_line_instances(
    predictions: list[np.ndarray], targets: list[np.ndarray]
) -> dict[str, float]:
    """Hungarian matching with both ChartInfo 6a and count-penalized 6b."""
    if not predictions and not targets:
        return {"score_6a": 1.0, "score_6b": 1.0, "count_error": 0.0}
    scores = np.zeros((len(targets), len(predictions)), dtype=np.float64)
    for target_index, target in enumerate(targets):
        for prediction_index, prediction in enumerate(predictions):
            scores[target_index, prediction_index] = continuous_line_score(
                prediction, target
            )
    if scores.size:
        rows, columns = linear_sum_assignment(-scores)
        matched = float(scores[rows, columns].sum())
    else:
        matched = 0.0
    # Task 6a normalizes by GT instances; task 6b also penalizes extra curves.
    score_6a = matched / max(1, len(targets))
    score_6b = matched / max(1, len(targets), len(predictions))
    return {
        "score_6a": score_6a,
        "score_6b": score_6b,
        "count_error": float(abs(len(predictions) - len(targets))),
    }
