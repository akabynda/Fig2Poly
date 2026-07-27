import json
from pathlib import Path

import numpy as np
from PIL import Image

from curveforge.config import GeneratorConfig
from curveforge.generator import DatasetGenerator
from training.convert_coco_instances import coco_counts, convert_split


def decode_counts(encoded: str) -> list[int]:
    counts = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        more = True
        char = 0
        while more:
            char = ord(encoded[position]) - 48
            position += 1
            value |= (char & 0x1F) << (5 * shift)
            more = bool(char & 0x20)
            shift += 1
        if char & 0x10:
            value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def decode_mask(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    values = []
    bit = 0
    for count in decode_counts(encoded):
        values.extend([bit] * count)
        bit = 1 - bit
    return np.asarray(values, dtype=bool).reshape(shape, order="F")


def test_coco_rle_roundtrip() -> None:
    mask = np.zeros((9, 13), dtype=bool)
    mask[1:8:2, 2:11] = True
    mask[4, 5:8] = True
    assert np.array_equal(decode_mask(coco_counts(mask), mask.shape), mask)


def test_convert_tiny_dataset_to_coco(tmp_path: Path) -> None:
    source = tmp_path / "source"
    cfg = GeneratorConfig(
        width=128, height=128, supersample=1, min_curves=1, max_curves=2, seed=71
    )
    DatasetGenerator(cfg).generate(source, 3, val_fraction=1 / 3, test_fraction=1 / 3)
    output = tmp_path / "coco"
    for split in ("train", "val", "test"):
        result = convert_split(source, output, split)
        assert result["images"] == 1
        payload = json.loads(
            (output / "annotations" / f"instances_{split}.json").read_text()
        )
        assert payload["categories"] == [
            {"id": 1, "name": "curve", "supercategory": "plot"}
        ]
        assert (output / "images" / split / payload["images"][0]["file_name"]).is_file()
        for annotation in payload["annotations"]:
            decoded = decode_mask(
                annotation["segmentation"]["counts"],
                tuple(annotation["segmentation"]["size"]),
            )
            assert int(decoded.sum()) == annotation["area"]
