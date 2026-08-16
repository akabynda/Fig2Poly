import cv2
import numpy as np

from training.predict_lineformer_panels import detect_plot_boxes


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
