import cv2
import numpy as np

from training.predict_lineformer_panels import (
    centerlines_are_duplicates,
    clean_prediction_tracks,
    detect_plot_boxes,
    mask_centerline,
    suppress_centerline_duplicates,
)


def test_detects_open_and_closed_plot_panels() -> None:
    image = np.full((500, 900, 3), 255, dtype=np.uint8)
    # Closed panel.
    cv2.rectangle(image, (40, 40), (400, 420), (0, 0, 0), 3)
    cv2.line(image, (60, 250), (380, 180), (20, 80, 220), 3)
    # Open panel: left and bottom axes only.
    cv2.line(image, (500, 40), (500, 420), (0, 0, 0), 3)
    cv2.line(image, (500, 420), (860, 420), (0, 0, 0), 3)
    cv2.line(image, (520, 300), (840, 120), (20, 80, 220), 3)

    boxes = detect_plot_boxes(image)

    assert len(boxes) == 2
    assert boxes[0][0] < 100 and boxes[0][2] < 500
    assert boxes[1][0] > 400 and boxes[1][2] > 800


def test_falls_back_to_full_image_without_multiple_panels() -> None:
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.line(image, (30, 270), (470, 270), (0, 0, 0), 2)
    cv2.line(image, (30, 30), (30, 270), (0, 0, 0), 2)

    assert detect_plot_boxes(image) == [(0, 0, 500, 300)]


def prediction(y: int, score: float, panel: int = 1) -> dict:
    mask = np.zeros((120, 180), dtype=bool)
    cv2.line(mask, (10, y), (170, y + 5), True, 3)
    return {"mask": mask, "score": score, "panel": panel, "bbox": [10, y, 170, y + 5]}


def test_centerline_nms_removes_duplicate_but_keeps_stacked_curve() -> None:
    best = prediction(30, 0.9)
    duplicate = prediction(31, 0.6)
    stacked = prediction(70, 0.8)

    kept, suppressed = suppress_centerline_duplicates([duplicate, stacked, best], 120)

    assert [item["score"] for item in kept] == [0.9, 0.8]
    assert len(suppressed) == 1
    assert suppressed[0]["score"] == 0.6


def test_centerline_duplicate_requires_same_panel() -> None:
    first, second = prediction(30, 0.9, 1), prediction(30, 0.8, 2)
    first["centerline"] = mask_centerline(first["mask"])
    second["centerline"] = mask_centerline(second["mask"])

    duplicate, _ = centerlines_are_duplicates(first, second, 120)

    assert not duplicate


def test_centerline_cleanup_reassigns_leaked_neighbouring_curve_fragments() -> None:
    upper = prediction(30, 0.9)
    lower = prediction(75, 0.8)
    upper_leak = np.zeros_like(upper["mask"])
    lower_leak = np.zeros_like(lower["mask"])
    cv2.line(upper_leak, (120, 75), (170, 77), True, 3)
    cv2.line(lower_leak, (10, 30), (55, 32), True, 3)
    upper["mask"] |= upper_leak
    lower["mask"] |= lower_leak

    cleaned, reassigned = clean_prediction_tracks([upper, lower], 120)

    assert np.all(cleaned[0]["mask"][lower_leak])
    assert np.all(cleaned[0]["mask"][upper_leak] == 0)
    assert np.all(cleaned[1]["mask"][upper_leak])
    assert np.all(cleaned[1]["mask"][lower_leak] == 0)
    assert len(reassigned) == 2
