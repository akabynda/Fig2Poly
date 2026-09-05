"""Route a COCO mixture through the vendored LineFormer training transforms.

The full pipelines are supplied by the upstream config, including loading and
formatting. Keeping them intact preserves its source-specific augmentation and
its native-image shift/crop order without introducing mask dilation.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose


SOURCE_ALIASES = {
    "pmc": "pmc",
    "adobe": "adobe",
    "adobesynth": "adobe",
    "lineex": "lineex",
    "dsc": "dsc",
}


@PIPELINES.register_module()
class SourceAwareLineFormerPipeline:
    """Use Shift/Crop for PMC/Adobe and the LineEX pipeline for LineEX/DSC."""

    def __init__(self, pmc_adobe_pipeline: list, lineex_dsc_pipeline: list) -> None:
        # Compose may instantiate transforms from mutable config values. Never
        # let either instance alter the lists owned by the generated config.
        self.pmc_adobe_pipeline = Compose(deepcopy(pmc_adobe_pipeline))
        self.lineex_dsc_pipeline = Compose(deepcopy(lineex_dsc_pipeline))
        self.min_shift_dimension = max(
            (int(transform.get("max_shift_px", 32))
             for transform in pmc_adobe_pipeline
             if transform.get("type") == "RandomShift"
             and transform.get("shift_ratio", 0.5) > 0),
            default=0,
        )

    def __call__(self, results: dict):
        image_info = results.get("img_info")
        provenance = image_info.get("mixture_provenance") if isinstance(image_info, Mapping) else None
        source = provenance.get("source") if isinstance(provenance, Mapping) else None
        if not isinstance(source, str) or source.strip().casefold() not in SOURCE_ALIASES:
            raise ValueError(
                "SourceAwareLineFormerPipeline requires a known "
                "img_info.mixture_provenance.source (pmc, adobe/AdobeSynth, lineex, dsc); "
                f"got {source!r}"
            )
        canonical_source = SOURCE_ALIASES[source.strip().casefold()]
        if canonical_source in ("pmc", "adobe"):
            dimensions = [image_info.get("height"), image_info.get("width")]
            if any(not isinstance(value, Integral) or isinstance(value, bool) or value <= 0
                   for value in dimensions):
                raise ValueError(f"{source}: native COCO image height/width must be positive integers")
            if min(dimensions) < self.min_shift_dimension:
                # The vendored shift slices with dimension - abs(shift). A
                # smaller image can produce incompatible negative-sized slices.
                # Fail deterministically before loading or drawing randomness.
                raise ValueError(
                    f"{source}: native image {image_info.get('file_name', image_info.get('filename', ''))!r} "
                    f"is {dimensions[1]}x{dimensions[0]}; original RandomShift requires "
                    f"both dimensions >= {self.min_shift_dimension}"
                )
            return self.pmc_adobe_pipeline(results)
        return self.lineex_dsc_pipeline(results)
