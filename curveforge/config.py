from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class GeneratorConfig:
    plot_domain: str = "general"
    dsc_probability: float = 0.50
    dsc_dense_probability: float = 0.18
    dsc_max_events: int = 18
    dsc_page_layout_probability: float = 0.35
    dsc_multipanel_page_probability: float = 0.55
    dsc_plot_min_fraction: float = 0.25
    dsc_surrounding_text_probability: float = 0.75
    dsc_caption_probability: float = 0.55
    dsc_foreign_graphics_probability: float = 0.30
    dsc_watermark_probability: float = 0.12
    dsc_tick_font_min: int = 8
    dsc_tick_font_max: int = 18
    dsc_axis_font_min: int = 11
    dsc_axis_font_max: int = 24
    dsc_curve_label_font_min: int = 8
    dsc_curve_label_font_max: int = 20
    dsc_annotation_font_min: int = 8
    dsc_annotation_font_max: int = 20
    dsc_direct_labels_probability: float = 0.90
    width: int = 768
    height: int = 576
    supersample: int = 2
    min_curves: int = 1
    max_curves: int = 10
    max_panels: int = 6
    max_curves_per_panel: int = 6
    min_degree: int = 1
    max_degree: int = 9
    max_points: int = 900
    curve_complexity: float = 1.0
    page_plot_min_fraction: float = 0.38
    crop_min_keep: float = 0.62
    rotation_max_degrees: float = 16.0
    perspective_max_strength: float = 0.075
    degradation_strength: float = 1.0
    min_jpeg_quality: int = 35
    max_jpeg_quality: int = 96
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
        if self.plot_domain not in {"general", "dsc", "mixed"}:
            raise ValueError("plot_domain must be one of: general, dsc, mixed")
        if not 7 <= self.dsc_max_events <= 64:
            raise ValueError("dsc_max_events must be in [7, 64]")
        if not 0.15 <= self.dsc_plot_min_fraction <= 0.75:
            raise ValueError("dsc_plot_min_fraction must be in [0.15, 0.75]")
        for prefix in ("dsc_tick_font", "dsc_axis_font", "dsc_curve_label_font",
                       "dsc_annotation_font"):
            minimum=getattr(self,f"{prefix}_min")
            maximum=getattr(self,f"{prefix}_max")
            if not 6 <= minimum <= maximum <= 48:
                raise ValueError(f"{prefix} must satisfy 6 <= min <= max <= 48")
        if self.width < 128 or self.height < 128:
            raise ValueError("width and height must be at least 128")
        if not 1 <= self.supersample <= 4:
            raise ValueError("supersample must be in [1, 4]")
        if not 1 <= self.min_curves <= self.max_curves <= 255:
            raise ValueError("curve count must satisfy 1 <= min <= max <= 255")
        if not 2 <= self.max_panels <= 6:
            raise ValueError("max_panels must be in [2, 6]")
        if not 1 <= self.max_curves_per_panel <= 255:
            raise ValueError("max_curves_per_panel must be in [1, 255]")
        if not 1 <= self.min_degree <= self.max_degree <= 20:
            raise ValueError("degree must satisfy 1 <= min <= max <= 20")
        if not 0 <= self.curve_complexity <= 1:
            raise ValueError("curve_complexity must be in [0, 1]")
        if not 0.35 <= self.page_plot_min_fraction <= 0.9:
            raise ValueError("page_plot_min_fraction must be in [0.35, 0.9]")
        if not 0.5 <= self.crop_min_keep <= 1:
            raise ValueError("crop_min_keep must be in [0.5, 1]")
        if not 0 <= self.rotation_max_degrees <= 45:
            raise ValueError("rotation_max_degrees must be in [0, 45]")
        if not 0 <= self.perspective_max_strength <= 0.2:
            raise ValueError("perspective_max_strength must be in [0, 0.2]")
        if not 0 <= self.degradation_strength <= 1:
            raise ValueError("degradation_strength must be in [0, 1]")
        if not 20 <= self.min_jpeg_quality <= self.max_jpeg_quality <= 100:
            raise ValueError("JPEG quality must satisfy 20 <= min <= max <= 100")
        probability_fields = (
            "dsc_probability",
            "dsc_dense_probability",
            "dsc_page_layout_probability", "dsc_surrounding_text_probability",
            "dsc_multipanel_page_probability",
            "dsc_caption_probability", "dsc_foreign_graphics_probability",
            "dsc_watermark_probability", "dsc_direct_labels_probability",
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
