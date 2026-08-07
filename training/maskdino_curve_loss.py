from __future__ import annotations

from typing import Any


def curve_geometry_loss(
    logits,
    targets,
    num_masks: float,
    coordinate_weight: float = 1.0,
    tangent_weight: float = 0.25,
    temperature: float = 0.20,
):
    """Thickness-invariant centreline and local-tangent loss for x-y curves."""
    import torch
    import torch.nn.functional as functional

    if logits.shape[0] == 0:
        return logits.sum() * 0
    if targets.shape[-2:] != logits.shape[-2:]:
        targets = functional.interpolate(
            targets[:, None].float(), size=logits.shape[-2:], mode="nearest"
        ).squeeze(1)
    logits = logits.float()
    targets = targets.float()
    height = logits.shape[-2]
    y_coordinates = torch.linspace(0, 1, height, device=logits.device)[None, :, None]
    prediction_weights = functional.softmax(logits / temperature, dim=-2)
    prediction_y = (prediction_weights * y_coordinates).sum(dim=-2)
    target_mass = targets.sum(dim=-2)
    valid = target_mass > 0
    target_y = (targets * y_coordinates).sum(dim=-2) / target_mass.clamp_min(1e-6)

    coordinate_error = functional.smooth_l1_loss(
        prediction_y, target_y, reduction="none", beta=0.02
    )
    per_instance = (coordinate_error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    coordinate_loss = per_instance.sum() / max(float(num_masks), 1.0)

    adjacent = valid[:, 1:] & valid[:, :-1]
    prediction_dy = prediction_y[:, 1:] - prediction_y[:, :-1]
    target_dy = target_y[:, 1:] - target_y[:, :-1]
    cosine = (1 + prediction_dy * target_dy) / (
        torch.sqrt(1 + prediction_dy.square())
        * torch.sqrt(1 + target_dy.square())
    )
    tangent_error = (1 - cosine) * adjacent
    tangent_per_instance = tangent_error.sum(-1) / adjacent.sum(-1).clamp_min(1)
    tangent_loss = tangent_per_instance.sum() / max(float(num_masks), 1.0)
    return coordinate_weight * coordinate_loss + tangent_weight * tangent_loss


def install_curve_geometry_loss(weight: float) -> None:
    """Patch MaskDINO's criterion before model construction."""
    if weight <= 0:
        return
    import maskdino.modeling.criterion as criterion_module

    criterion_class = criterion_module.SetCriterion
    if getattr(criterion_class, "_fig2poly_curve_loss", False):
        return
    original_init = criterion_class.__init__
    original_loss_masks = criterion_class.loss_masks

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        additions = {}
        for key in list(self.weight_dict):
            if key.startswith("loss_mask"):
                additions["loss_line" + key[len("loss_mask"):]] = weight
        self.weight_dict.update(additions)

    def patched_loss_masks(self, outputs, targets, indices, num_masks):
        losses = original_loss_masks(self, outputs, targets, indices, num_masks)
        src_index = self._get_src_permutation_idx(indices)
        target_index = self._get_tgt_permutation_idx(indices)
        source_masks = outputs["pred_masks"][src_index]
        if source_masks.shape[0] == 0:
            losses["loss_line"] = source_masks.sum() * 0
            return losses
        masks = [target["masks"] for target in targets]
        target_masks, _ = criterion_module.nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)[target_index]
        losses["loss_line"] = curve_geometry_loss(
            source_masks, target_masks, num_masks
        )
        return losses

    criterion_class.__init__ = patched_init
    criterion_class.loss_masks = patched_loss_masks
    criterion_class._fig2poly_curve_loss = True
