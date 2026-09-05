import ast
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from training.maskdino_source_augmentation import (
    SourceAwareMaskDINOMapper, build_source_aware_mapper, image_source, shift_pixels,
)


@pytest.mark.parametrize(("filename", "expected"), [
    ("/dataset/images/train/pmc/001.png", "full"),
    ("/dataset/images/train/adobe/002.png", "full"),
    ("/dataset/images/train/lineex/003.png", "light"),
    ("C:\\dataset\\images\\train\\DSC\\004.png", "light"),
])
def test_source_router_preserves_original_record_and_works_for_smoke_absolute_paths(filename, expected):
    record = {"file_name": filename, "width": 512, "height": 256, "annotations": []}
    calls = []
    mapper = SourceAwareMaskDINOMapper(
        lambda value: calls.append(("full", value)),
        lambda value: calls.append(("light", value)),
    )
    mapper(record)
    assert calls == [(expected, record)]
    assert calls[0][1] is record


def test_unknown_source_fails_instead_of_silently_omitting_augmentation():
    with pytest.raises(ValueError, match="Unknown LineFormer mixture source"):
        image_source("/dataset/images/train/unrecognized/001.png")


def test_tiny_original_images_fail_before_random_shift_broadcast_error():
    mapper = SourceAwareMaskDINOMapper(lambda value: value, lambda value: value)
    with pytest.raises(ValueError, match="at least 51px"):
        mapper({"file_name": "/train/pmc/001.png", "width": 50, "height": 512})
    record = {"file_name": "/train/dsc/001.png", "width": 32, "height": 512}
    assert mapper(record) is record


def original_shift_class(dx, dy):
    # Run the checked-in original implementation as the independent oracle;
    # avoid importing MMCV/CUDA merely to exercise its numpy mask geometry.
    source = Path(__file__).resolve().parents[1] / "third_party/LineFormer/mmdetection/mmdet/datasets/pipelines/transforms.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "RandomShift")
    node.decorator_list = []
    offsets = iter((dx, dy))

    class BitmapMasks:
        def __init__(self, masks, height, width):
            self.masks = masks

    namespace = {
        "np": np, "BitmapMasks": BitmapMasks,
        "random": SimpleNamespace(random=lambda: 0.0, randint=lambda lower, upper: next(offsets)),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["RandomShift"], BitmapMasks


@pytest.mark.parametrize(("dx", "dy"), [(-51, 50), (50, -51), (-4, 3), (0, 0)])
def test_shift_pixels_exactly_matches_original_image_and_mask_without_dilation(dx, dy):
    original_class, masks_class = original_shift_class(dx, dy)
    image = np.arange(68 * 72 * 3, dtype=np.uint32).reshape(68, 72, 3) % 256
    image = image.astype(np.uint8)
    mask = np.zeros((68, 72), dtype=np.uint8)
    mask[1:67, 36] = 1
    result = original_class(shift_ratio=0.3, max_shift_px=51)({
        "img": image.copy(), "mask_fields": ["gt_masks"],
        "gt_masks": masks_class(mask[None].copy(), 68, 72),
    })
    actual_image = shift_pixels(image, dx, dy, fill=255)
    actual_mask = shift_pixels(mask, dx, dy, fill=0)
    np.testing.assert_array_equal(actual_image, result["img"])
    np.testing.assert_array_equal(actual_mask, result["gt_masks"].masks[0])
    assert actual_mask.sum() <= mask.sum()
    assert set(np.unique(actual_mask)) <= {0, 1}


def test_shift_exceeding_dimensions_fails_instead_of_changing_canvas():
    with pytest.raises(ValueError, match="exceeds image dimensions"):
        shift_pixels(np.ones((20, 20), dtype=np.uint8), 51, 0, fill=0)


def test_mapper_augmentation_parameters_match_original_source_pipelines(monkeypatch):
    class Augmentation:
        pass

    class RandomApply:
        def __init__(self, aug, prob):
            self.aug, self.prob = aug, prob

    def crop(crop_type, crop_size):
        return SimpleNamespace(crop_type=crop_type, crop_size=crop_size)

    transforms = SimpleNamespace(
        Augmentation=Augmentation, Transform=object,
        RandomApply=RandomApply, RandomCrop=crop,
        ResizeScale=lambda **kwargs: SimpleNamespace(kind="resize", **kwargs),
        FixedSizeCrop=lambda **kwargs: SimpleNamespace(kind="pad", **kwargs),
    )
    monkeypatch.setitem(sys.modules, "detectron2.data", SimpleNamespace(transforms=transforms))
    monkeypatch.setitem(sys.modules, "maskdino", SimpleNamespace(
        COCOInstanceNewBaselineDatasetMapper=lambda **kwargs: SimpleNamespace(**kwargs),
    ))
    mapper = build_source_aware_mapper(SimpleNamespace(INPUT=SimpleNamespace(IMAGE_SIZE=512, FORMAT="RGB")))
    root = Path(__file__).resolve().parents[1]
    original = runpy.run_path(str(root / "third_party/LineFormer/lineformer_swin_t_config.py"))
    expected_shift = next(step for step in original["train_pipeline"] if step["type"] == "RandomShift")
    expected_crop = next(step for step in original["train_pipeline"] if step["type"] == "RandomCrop")
    full, light = mapper.full_mapper.tfm_gens, mapper.light_mapper.tfm_gens
    assert len(full) == 5
    assert len(light) == 3
    assert full[1].prob == expected_shift["shift_ratio"]
    assert full[2].prob == expected_crop["crop_ratio"]
    assert full[2].aug.crop_type == expected_crop["crop_type"]
    assert full[2].aug.crop_size == expected_crop["crop_size"] == (435, 435)
    assert full[-1].pad_value == light[-1].pad_value == 255
    assert full[-1].seg_pad_value == light[-1].seg_pad_value == 0

    bounds = []
    def fixed_offset(low, high):
        bounds.append((low, high))
        return -3 if len(bounds) == 1 else 4
    monkeypatch.setattr(np.random, "randint", fixed_offset)
    transform = full[1].aug.get_transform(np.zeros((70, 80, 3), dtype=np.uint8))
    assert bounds == [(-51, 51), (-51, 51)]
    np.testing.assert_array_equal(transform.apply_coords(np.array([[10.0, 20.0]])), [[7.0, 24.0]])
    mask = np.zeros((70, 80), dtype=np.uint8)
    mask[20, 10] = 1
    shifted = transform.apply_segmentation(mask)
    assert shifted.sum() == 1
    assert shifted[24, 7] == 1
