import numpy as np

from training.fuse_curve_predictions import add_complex_lineformer_rescues, fuse_panel
from training.predict_lineformer_panels import mask_centerline


def track(identifier: int, y_values, source: str, panel: int = 1, score: float = 0.9) -> dict:
    mask = np.zeros((100, 120), dtype=bool)
    for x, y in enumerate(y_values, 10):
        mask[max(0, y - 1):min(mask.shape[0], y + 2), x] = True
    return {
        "id": identifier,
        "score": score,
        "panel": panel,
        "bbox": [10, min(y_values), 10 + len(y_values), max(y_values) + 1],
        "mask": mask,
        "source": source,
        "centerline": mask_centerline(mask),
    }


def test_crowded_panel_keeps_separate_maskdino_identities() -> None:
    maskdino = [
        track(1, [30] * 80, "maskdino"),
        track(2, [35] * 80, "maskdino"),
    ]
    lineformer = [track(1, [32] * 80, "lineformer")]

    fused, diagnostics = fuse_panel(maskdino, lineformer, image_height=100)

    assert diagnostics["preferred_expert"] == "maskdino"
    assert diagnostics["crowded"] is True
    assert len(fused) == 2
    assert all(item["source"] == "maskdino" for item in fused)


def test_non_crowded_panel_uses_lineformer_geometry() -> None:
    x = np.arange(80)
    maskdino = [track(1, [45] * 80, "maskdino")]
    lineformer_y = np.rint(45 + 8 * np.sin(x / 6)).astype(int).tolist()
    lineformer = [track(1, lineformer_y, "lineformer")]

    fused, diagnostics = fuse_panel(maskdino, lineformer, image_height=100)

    assert diagnostics["preferred_expert"] == "lineformer"
    assert len(fused) == 1
    assert fused[0]["source"].startswith("lineformer")
    assert np.all(fused[0]["mask"][lineformer[0]["mask"]])


def test_crowded_panel_adds_distant_confident_lineformer_miss() -> None:
    maskdino = [
        track(1, [20] * 80, "maskdino"),
        track(2, [25] * 80, "maskdino"),
    ]
    lineformer = [
        track(1, [21] * 80, "lineformer"),
        track(2, [75] * 80, "lineformer", score=0.8),
    ]

    fused, _ = fuse_panel(maskdino, lineformer, image_height=100)

    assert len(fused) == 3
    assert fused[-1]["reason"] == "confident_unmatched_lineformer"


def test_low_confidence_rescue_requires_peak_rich_curve() -> None:
    x = np.arange(80)
    peak_rich = track(
        1, np.rint(70 - 20 * np.exp(-((x - 40) / 8) ** 2)).astype(int).tolist(),
        "lineformer", score=0.2,
    )
    flat_clutter = track(2, [85] * 80, "lineformer", score=0.2)

    fused, diagnostics = add_complex_lineformer_rescues(
        [], [], [peak_rich, flat_clutter], image_height=100
    )

    assert len(fused) == 1
    assert fused[0]["id"] == 1
    assert [item["action"] for item in diagnostics] == ["rescue", "reject"]
