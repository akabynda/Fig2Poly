from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class GeneratorConfig:
    width: int = 768
    height: int = 576
    supersample: int = 2
    min_curves: int = 1
    max_curves: int = 10
    min_degree: int = 1
    max_degree: int = 9
    max_points: int = 900
    crop_probability: float = 0.48
    rotation_probability: float = 0.40
    perspective_probability: float = 0.25
    degradation_probability: float = 0.72
    occlusion_probability: float = 0.20
    same_color_probability: float = 0.14
    grouped_color_probability: float = 0.24
    related_curves_probability: float = 0.36
    legend_probability: float = 0.58
    title_probability: float = 0.70
    labels_probability: float = 0.72
    markers_probability: float = 0.25
    annotations_probability: float = 0.20
    empty_plot_probability: float = 0.06
    non_polynomial_probability: float = 0.68
    page_layout_probability: float = 0.48
    watermark_probability: float = 0.22
    hard_negatives_probability: float = 0.45
    dense_text_probability: float = 0.30
    multi_panel_probability: float = 0.28
    seed: int = 20260722

    def validate(self) -> None:
        if self.width < 128 or self.height < 128:
            raise ValueError("width and height must be at least 128")
        if not 1 <= self.supersample <= 4:
            raise ValueError("supersample must be in [1, 4]")
        if not 1 <= self.min_curves <= self.max_curves <= 255:
            raise ValueError("curve count must satisfy 1 <= min <= max <= 255")
        if not 1 <= self.min_degree <= self.max_degree <= 20:
            raise ValueError("degree must satisfy 1 <= min <= max <= 20")
        probability_fields = (
            "crop_probability", "rotation_probability", "perspective_probability",
            "degradation_probability", "occlusion_probability", "same_color_probability",
            "grouped_color_probability", "related_curves_probability",
            "legend_probability", "title_probability",
            "labels_probability", "markers_probability", "annotations_probability",
            "empty_plot_probability", "non_polynomial_probability",
            "page_layout_probability", "watermark_probability",
            "hard_negatives_probability", "dense_text_probability",
            "multi_panel_probability",
        )
        for name in probability_fields:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "GeneratorConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        result = cls(**raw)
        result.validate()
        return result
