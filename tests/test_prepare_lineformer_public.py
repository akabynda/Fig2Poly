import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from training.prepare_lineformer_public import (
    CocoWriter,
    lineex_groups,
    mask_annotation,
    prepare_chartinfo,
    prepare_lineex,
    recipe_source,
    verify_legacy_dsc_samples,
)
from training.convert_coco_instances import convert_split
from training.convert_public_instances import stable_validation


def write_fixture(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def chartinfo_annotation() -> dict:
    return {
        "task1": {"output": {"chart_type": "Line"}},
        "task6": {"output": {"visual elements": {
            "lines": [[{"x": 1, "y": 5}, {"x": 12, "y": 5}]],
        }}},
    }


def test_chartinfo_holdout_never_splits_official_test(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    annotations = raw / "annotations_JSON" / "line"
    raw.mkdir()
    # Ensure both holdout outcomes without relying on a random sample size.
    names = [next(str(i) for i in range(100) if stable_validation(str(i), 0.1) == value)
             for value in (False, True)]
    for name in names:
        write_fixture(annotations / f"{name}.json", chartinfo_annotation())
        Image.new("RGB", (16, 16), "white").save(raw / f"{name}.png")
    output = tmp_path / "coco"
    result = prepare_chartinfo(raw, None, output, "pmc", "train", 0.1, 1)
    assert result["train"]["images"] == result["val"]["images"] == 1
    train = json.loads((output / "annotations/instances_train.json").read_text())
    val = json.loads((output / "annotations/instances_val.json").read_text())
    assert train["images"][0]["source_id"] != val["images"][0]["source_id"]
    assert train["images"][0]["official_source_split"] == "train"
    assert val["images"][0]["official_source_split"] == "train"
    assert train["annotations"][0]["area"] == 12
    assert train["annotations"][0]["bbox"] == [1, 5, 12, 1]
    assert (raw / train["images"][0]["file_name"]).is_file()
    # Completed preparation reuses matching settings, refusing silent changes.
    assert prepare_chartinfo(raw, None, output, "pmc", "train", 0.1, 1) == result
    with pytest.raises(ValueError, match="settings changed"):
        prepare_chartinfo(raw, None, output, "pmc", "train", 0.2, 1)
    result = prepare_chartinfo(raw, None, tmp_path / "test", "pmc", "test", 0.9, 1)
    assert set(result) == {"test"}
    assert result["test"]["images"] == 2


def test_lineex_streams_instances_preserving_official_split_and_legend(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(raw / "images/chart.png")
    write_fixture(raw / "anno/line_anno.json", {
        "images": [{"id": 7, "file_name": "chart.png", "width": 16, "height": 16}],
        "annotations": [{"id": 1, "image_id": 7, "bbox": [1, 5, 12, 5]}],
    })
    write_fixture(raw / "anno/cls_anno.json", {
        "annotations": [{"category_id": 0, "image_id": 7, "bbox": [5, 0, 2, 10]}],
    })
    output = tmp_path / "coco"
    result = prepare_lineex(raw, output, "val", 1)
    assert result["images"] == result["instances"] == 1
    payload = json.loads((output / "annotations/instances_val.json").read_text())
    assert payload["images"][0]["source_id"] == "lineex__val__7"
    assert payload["images"][0]["file_name"] == "chart.png"
    annotation = payload["annotations"][0]
    assert annotation["area"] == 9  # 12 pixels minus the inclusive three-pixel legend.
    assert annotation["bbox"] == [1, 5, 12, 1]
    assert not list(output.rglob("*.png"))
    # Missing images must fail instead of silently reducing the training corpus.
    (raw / "images/chart.png").unlink()
    with pytest.raises(FileNotFoundError, match="Missing released LineEX"):
        prepare_lineex(raw, tmp_path / "new", "val", 1)


def test_lineex_requires_grouped_annotations(tmp_path: Path) -> None:
    path = tmp_path / "lines.json"
    write_fixture(path, {"annotations": [{"image_id": 2}, {"image_id": 1}]})
    with pytest.raises(ValueError, match="not grouped"):
        list(lineex_groups(path))


def test_coco_writer_does_not_publish_partial_results(tmp_path: Path) -> None:
    target = tmp_path / "annotations.json"
    target.write_text("previous-complete-result")
    with pytest.raises(RuntimeError):
        with CocoWriter(target, {}) as writer:
            writer.add({"file_name": "ignored.png"}, [])
            raise RuntimeError("interrupted")
    assert target.read_text() == "previous-complete-result"


def test_rle_roundtrip_and_nonempty_bounds() -> None:
    # This test verifies compatibility with the external COCO reader when installed.
    mask_utils = pytest.importorskip("pycocotools.mask")
    array = np.zeros((13, 17), dtype=np.uint8)
    array[2:6, 4:11] = 255
    annotation = mask_annotation(Image.fromarray(array))
    assert annotation["bbox"] == [4, 2, 7, 4]
    assert annotation["area"] == 28
    assert np.array_equal(mask_utils.decode(annotation["segmentation"]), array > 0)
    assert mask_annotation(Image.new("L", (17, 13), 0)) is None


def test_recipe_repeat_and_independent_lineex_image_roots(tmp_path: Path) -> None:
    roots = {split: tmp_path / split / "images" for split in ("train", "val", "test")}
    pmc = recipe_source("pmc", tmp_path / "pmc", roots)
    assert pmc["splits"]["train"]["repeat"] == 50
    assert pmc["splits"]["val"]["repeat"] == 1
    lineex = recipe_source("lineex", tmp_path / "lineex", roots)
    assert lineex["splits"]["train"]["image_root"] != lineex["splits"]["val"]["image_root"]
    assert all(split["repeat"] == 1 for split in lineex["splits"].values())


def test_legacy_dsc_verification_accepts_exact_and_rejects_dilated_masks(tmp_path: Path) -> None:
    source = tmp_path / "raw_dsc"
    source.mkdir()
    Image.new("RGB", (16, 16), "white").save(source / "image.png")
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5, 2:13] = 255
    Image.fromarray(mask).save(source / "mask.png")
    records = [{"id": f"sample{i}", "image": "image.png", "width": 16, "height": 16,
                "curves": [{"mask": "mask.png"}]} for i in range(8)]
    (source / "train.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8")
    coco = tmp_path / "coco"
    convert_split(source, coco, "train", category_name="line", mask_dilation=1)
    path = coco / "annotations/instances_train.json"
    payload = json.loads(path.read_text())
    del payload["info"]["mask_dilation"]
    path.write_text(json.dumps(payload))
    result = verify_legacy_dsc_samples(coco, source, "train")
    assert result["verified_images"] == result["verified_masks"] == 8
    assert result["exhaustive"] is False
    convert_split(source, coco, "train", category_name="line", mask_dilation=3)
    payload = json.loads(path.read_text())
    del payload["info"]["mask_dilation"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differ from exact raw masks"):
        verify_legacy_dsc_samples(coco, source, "train")
