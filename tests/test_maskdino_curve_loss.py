import pytest

torch = pytest.importorskip("torch")

from training.maskdino_curve_loss import curve_geometry_loss


def test_curve_geometry_loss_prefers_aligned_centerline():
    target = torch.zeros((1, 16, 20))
    target[:, 6:9, :] = 1
    aligned = torch.full_like(target, -6)
    aligned[:, 6:9, :] = 6
    shifted = torch.full_like(target, -6)
    shifted[:, 11:14, :] = 6
    assert curve_geometry_loss(aligned, target, 1) < curve_geometry_loss(shifted, target, 1)
