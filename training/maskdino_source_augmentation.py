"""LineFormer's source-specific image/mask geometry for the MaskDINO mapper."""

from __future__ import annotations

from pathlib import PurePosixPath

import numpy as np


FULL_AUGMENTATION_SOURCES = frozenset(("pmc", "adobe"))
LIGHT_AUGMENTATION_SOURCES = frozenset(("lineex", "dsc"))


def image_source(filename: str) -> str:
    # Prepared mixtures and smoke subsets retain images/train/{source}/{id}.png.
    source = PurePosixPath(str(filename).replace("\\", "/")).parent.name.casefold()
    if source not in FULL_AUGMENTATION_SOURCES | LIGHT_AUGMENTATION_SOURCES:
        raise ValueError(f"Unknown LineFormer mixture source in image path: {filename!r}")
    return source


def shift_pixels(array: np.ndarray, dx: int, dy: int, fill: int) -> np.ndarray:
    """Translate without interpolation or mask expansion, matching vendored slicing."""
    height, width = array.shape[:2]
    if abs(dx) > width or abs(dy) > height:
        raise ValueError(f"LineFormer shift ({dx}, {dy}) exceeds image dimensions {(height, width)}")
    output = np.full_like(array, fill)
    copy_height, copy_width = height - abs(dy), width - abs(dx)
    source_y, source_x = max(0, -dy), max(0, -dx)
    target_y, target_x = max(0, dy), max(0, dx)
    output[target_y:target_y + copy_height, target_x:target_x + copy_width] = array[
        source_y:source_y + copy_height, source_x:source_x + copy_width
    ]
    return output


class SourceAwareMaskDINOMapper:
    def __init__(self, full_mapper, light_mapper):
        self.full_mapper = full_mapper
        self.light_mapper = light_mapper

    def __call__(self, dataset_dict):
        source = image_source(dataset_dict["file_name"])
        if source in FULL_AUGMENTATION_SOURCES:
            height, width = int(dataset_dict["height"]), int(dataset_dict["width"])
            if min(height, width) < 51:
                raise ValueError(
                    f"Original LineFormer shift requires PMC/Adobe images at least 51px per side; "
                    f"got {width}x{height}: {dataset_dict['file_name']}"
                )
            return self.full_mapper(dataset_dict)
        return self.light_mapper(dataset_dict)


def build_source_aware_mapper(cfg):
    from detectron2.data import transforms as T
    from maskdino import COCOInstanceNewBaselineDatasetMapper

    class ExclusiveLineFlip(T.Augmentation):
        def get_transform(self, image):
            sample = self._rand_range()
            if sample < 0.3:
                return T.HFlipTransform(image.shape[1])
            if sample < 0.6:
                return T.VFlipTransform(image.shape[0])
            return T.NoOpTransform()

    class LineShiftTransform(T.Transform):
        def __init__(self, dx, dy):
            super().__init__()
            self.dx = int(dx)
            self.dy = int(dy)

        def apply_image(self, image):
            return shift_pixels(image, self.dx, self.dy, fill=255)

        def apply_segmentation(self, segmentation):
            return shift_pixels(segmentation, self.dx, self.dy, fill=0)

        def apply_coords(self, coords):
            shifted = coords.copy()
            shifted[:, 0] += self.dx
            shifted[:, 1] += self.dy
            return shifted

    class RandomLineShift(T.Augmentation):
        def get_transform(self, image):
            # LineFormer imports random from numpy: the upper bound is exclusive.
            return LineShiftTransform(np.random.randint(-51, 51), np.random.randint(-51, 51))

    def mapper(use_shift_and_crop):
        transforms = [ExclusiveLineFlip()]
        if use_shift_and_crop:
            transforms.extend([
                T.RandomApply(RandomLineShift(), prob=0.3),
                T.RandomApply(T.RandomCrop("absolute", (435, 435)), prob=0.3),
            ])
        size = cfg.INPUT.IMAGE_SIZE
        transforms.extend([
            T.ResizeScale(min_scale=1.0, max_scale=1.0, target_height=size, target_width=size),
            T.FixedSizeCrop(crop_size=(size, size), pad_value=255.0, seg_pad_value=0),
        ])
        # Upstream recomputes boxes from transformed RLE masks and drops empty
        # instances. MaskDINO's box losses require these valid boxes; LineFormer
        # has no box loss and retains even wholly cropped-away target masks.
        return COCOInstanceNewBaselineDatasetMapper(
            is_train=True, image_format=cfg.INPUT.FORMAT, tfm_gens=transforms,
        )

    return SourceAwareMaskDINOMapper(mapper(True), mapper(False))
