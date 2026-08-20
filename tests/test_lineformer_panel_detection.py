import cv2
import numpy as np

from training.predict_lineformer_panels import (
    centerlines_are_duplicates,
    clean_prediction_tracks,
    detect_plot_boxes,
    local_regression_error,
    mask_centerline,
    select_plot_boxes,
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


def test_detects_a_single_open_plot() -> None:
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.line(image, (30, 270), (470, 270), (0, 0, 0), 2)
    cv2.line(image, (30, 30), (30, 270), (0, 0, 0), 2)

    boxes = detect_plot_boxes(image)

    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 <= 30 <= x2 and y1 <= 30 < 270 <= y2
    assert boxes[0] != (0, 0, 500, 300)


def test_falls_back_to_full_image_without_plot_axes() -> None:
    image = np.full((300, 500, 3), 255, dtype=np.uint8)

    assert detect_plot_boxes(image) == [(0, 0, 500, 300)]


def test_nested_plot_candidates_keep_largest_outer_box() -> None:
    outer = (20, 20, 480, 280)
    inner = (60, 50, 450, 250)
    separate = (520, 20, 880, 280)
    candidates = [
        (900.0, inner),       # More Hough support must not make it win.
        (500.0, outer),
        (600.0, separate),
    ]

    assert select_plot_boxes(candidates) == [outer, separate]


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


def test_centerline_nms_fuses_unique_pixels_from_duplicate_query() -> None:
    best = prediction(30, 0.9)
    duplicate = prediction(31, 0.6)
    extension = np.zeros_like(best["mask"])
    cv2.line(extension, (165, 32), (175, 34), True, 3)
    duplicate["mask"] |= extension

    kept, suppressed = suppress_centerline_duplicates([best, duplicate], 120)

    assert len(kept) == 1
    assert np.all(kept[0]["mask"][extension])
    assert suppressed[0]["merged_unique_pixels"] > 0


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


def test_color_resolves_curve_ownership_at_nearby_junction() -> None:
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    upper = prediction(30, 0.9)
    middle = prediction(75, 0.8)
    branch = np.zeros_like(upper["mask"])
    cv2.line(branch, (82, 35), (82, 69), True, 2)
    middle["mask"] |= branch
    image[upper["mask"]] = (30, 30, 220)
    image[branch] = (30, 30, 220)
    image[middle["mask"] & ~branch] = (220, 70, 30)

    cleaned, _ = clean_prediction_tracks([upper, middle], 120, image)

    assert np.all(cleaned[0]["mask"][branch])
    assert np.all(cleaned[1]["mask"][branch] == 0)


def test_local_regression_resolves_same_color_curve_fragment() -> None:
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    upper = np.zeros((120, 180), dtype=bool)
    middle = np.zeros_like(upper)
    fragment = np.zeros_like(upper)
    upper_points = np.asarray([
        (x, 30 + int(0.004 * (x - 20) ** 2)) for x in range(10, 81)
    ], dtype=np.int32)
    fragment_points = np.asarray([
        (x, 30 + int(0.004 * (x - 20) ** 2)) for x in range(84, 111)
    ], dtype=np.int32)
    cv2.polylines(upper, [upper_points], False, True, 3)
    cv2.polylines(fragment, [fragment_points], False, True, 3)
    cv2.line(middle, (10, 75), (170, 75), True, 3)
    image[upper | middle | fragment] = 0
    predictions = [
        {"mask": upper, "score": 0.9, "panel": 1, "bbox": [0, 0, 180, 120]},
        {"mask": middle | fragment, "score": 0.8, "panel": 1, "bbox": [0, 0, 180, 120]},
    ]

    assert local_regression_error(upper, fragment) < local_regression_error(middle, fragment)
    cleaned, diagnostics = clean_prediction_tracks(predictions, 120, image)

    assert np.all(cleaned[0]["mask"][fragment])
    assert np.all(cleaned[1]["mask"][fragment] == 0)
    assert diagnostics[0]["selection_mode"] == "regression"


def test_clean_prediction_tracks_discards_subminimum_mask() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[4, 5] = True
    mask[12, 18] = True
    prediction_item = {
        "mask": mask, "score": 0.2, "panel": 1, "bbox": [5, 4, 19, 13]
    }

    cleaned, diagnostics = clean_prediction_tracks([prediction_item], 20)

    assert cleaned == []
    assert diagnostics == []
