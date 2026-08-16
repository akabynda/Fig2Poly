from training.predict_lineformer import PALETTE


def test_lineformer_visualization_palette_is_nonempty() -> None:
    assert len(PALETTE) >= 8
    assert all(len(color) == 3 for color in PALETTE)
