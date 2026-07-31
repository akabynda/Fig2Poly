from __future__ import annotations

import argparse
from pathlib import Path


OLD_INSTANCES = "instances = utils.annotations_to_instances(annos, image_shape)"
NEW_INSTANCES = (
    'instances = utils.annotations_to_instances('
    'annos, image_shape, mask_format="bitmask"'
    ')'
)
OLD_EMPTY = "instances.gt_masks = PolygonMasks([])"
NEW_EMPTY = (
    "instances.gt_masks = BitMasks(torch.zeros("
    "(0, image_shape[0], image_shape[1]), dtype=torch.bool))"
)
OLD_CONVERSION = """gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks"""
NEW_CONVERSION = """gt_masks = instances.gt_masks
                if isinstance(gt_masks, PolygonMasks):
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                elif isinstance(gt_masks, BitMasks):
                    gt_masks = gt_masks.tensor
                else:
                    raise TypeError(f\"Unsupported mask container: {type(gt_masks)!r}\")
                instances.gt_masks = gt_masks"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} pattern, found {count}")
    return text.replace(old, new, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Patch official MaskDINO mapper to preserve COCO RLE as BitMasks"
    )
    parser.add_argument("--maskdino-root", type=Path, required=True)
    args = parser.parse_args(argv)
    mapper = (
        args.maskdino_root.resolve()
        / "maskdino/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py"
    )
    text = mapper.read_text(encoding="utf-8")
    text = replace_once(text, OLD_INSTANCES, NEW_INSTANCES, "annotations conversion")
    text = replace_once(text, OLD_EMPTY, NEW_EMPTY, "empty BitMasks")
    text = replace_once(text, OLD_CONVERSION, NEW_CONVERSION, "mask tensor conversion")
    mapper.write_text(text, encoding="utf-8")
    print(f"MaskDINO RLE mapper ready: {mapper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
