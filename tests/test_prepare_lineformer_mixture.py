import json
import os
from pathlib import Path

import pytest

import training.prepare_lineformer_mixture as mixture


def write_source(root: Path, name: str, split: str, *, category: str = "line") -> dict:
    source_dir = root / name
    source_dir.mkdir(exist_ok=True)
    image_path = source_dir / f"{split}.png"
    image_path.write_bytes(b"fixture-image-" + split.encode())
    payload = {
        "info": {"mask_dilation": 3 if split == "train" else 1},
        "licenses": [{"id": 8, "name": "fixture license"}],
        "categories": [{"id": 7, "name": category}],
        "images": [{"id": 42, "file_name": image_path.name, "width": 8, "height": 8, "license": 8}],
        "annotations": [{"id": 96, "image_id": 42, "category_id": 7,
                         "segmentation": {"size": [8, 8], "counts": "1234"},
                         "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0}],
    }
    annotation_path = source_dir / f"{split}.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    return {"annotations": str(annotation_path.relative_to(root)), "image_root": name}


def write_recipe(root: Path) -> Path:
    pmc = {split: write_source(root, "pmc", split) for split in ("train", "val")}
    pmc["train"]["repeat"] = 50
    pmc["test"] = dict(pmc["val"])
    sources = [{"name": "pmc", "splits": pmc}]
    for name in ("adobe", "lineex", "dsc"):
        sources.append({"name": name, "splits": {
            "train": write_source(root, name, "train", category="curve" if name == "dsc" else "line"),
        }})
    recipe = root / "recipe.json"
    recipe.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    return recipe


def update_json(path: Path, change) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def load_split(root: Path, split: str) -> dict:
    return json.loads((root / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8"))


def test_four_source_mixture_repeat_masks_and_provenance(tmp_path: Path) -> None:
    recipe = write_recipe(tmp_path)
    output = tmp_path / "prepared"
    summary = mixture.prepare_mixture(recipe, output)
    train = load_split(output, "train")
    assert len(train["images"]) == len(train["annotations"]) == 53
    assert [image["id"] for image in train["images"]] == list(range(1, 54))
    assert [ann["id"] for ann in train["annotations"]] == list(range(1, 54))
    assert [ann["image_id"] for ann in train["annotations"]] == list(range(1, 54))
    assert train["categories"] == [{"id": 1, "name": "line", "supercategory": "plot"}]
    assert {ann["category_id"] for ann in train["annotations"]} == {1}
    assert len({image["file_name"] for image in train["images"][:50]}) == 1
    assert len(list((output / "images" / "train").rglob("*.png"))) == 4
    original = json.loads((tmp_path / "pmc" / "train.json").read_text())["annotations"][0]
    for ann in train["annotations"]:
        for key in ("segmentation", "bbox", "area", "iscrowd"):
            assert ann[key] == original[key]
    assert train["images"][49]["mixture_provenance"] == {"source": "pmc", "image_id": 42, "repetition": 49}
    assert train["annotations"][50]["mixture_provenance"]["source"] == "adobe"
    assert summary["splits"]["train"]["unique_image_files"] == 4
    assert summary["sources"][0]["original_info"] == {"mask_dilation": 3}
    assert summary["sources"][0]["annotations_sha256"] == mixture._sha256_file(tmp_path / "pmc" / "train.json")
    assert len(load_split(output, "val")["images"]) == len(load_split(output, "test")["images"]) == 1
    assert len({image["license"] for image in train["images"]}) == 4


def test_idempotency_and_repair_of_same_fingerprint(tmp_path: Path) -> None:
    recipe = write_recipe(tmp_path)
    output = tmp_path / "prepared"
    first = mixture.prepare_mixture(recipe, output)
    target = output / "annotations" / "instances_train.json"
    original_mtime = target.stat().st_mtime_ns
    assert mixture.prepare_mixture(recipe, output) == first
    assert target.stat().st_mtime_ns == original_mtime
    target.write_text("interrupted", encoding="utf-8")
    (output / mixture.SUMMARY_NAME).unlink()
    missing_image = output / "images" / "train" / "adobe" / "000000001.png"
    missing_image.unlink()
    repaired = mixture.prepare_mixture(recipe, output)
    assert repaired == first
    assert missing_image.is_file()
    assert len(load_split(output, "train")["images"]) == 53


@pytest.mark.parametrize("change_kind", ["recipe", "annotations", "image"])
def test_changed_source_or_recipe_cannot_overwrite_prepared_dataset(tmp_path: Path, change_kind: str) -> None:
    recipe = write_recipe(tmp_path)
    output = tmp_path / "prepared"
    first = mixture.prepare_mixture(recipe, output)
    if change_kind == "recipe":
        update_json(recipe, lambda data: data["sources"][0]["splits"]["train"].update(repeat=2))
    elif change_kind == "annotations":
        update_json(tmp_path / "pmc" / "train.json", lambda data: data["annotations"][0].update(area=13))
    else:
        image_path = tmp_path / "pmc" / "train.png"
        image_path.write_bytes(b"changed source image")
    with pytest.raises(ValueError, match="fingerprint changed"):
        mixture.prepare_mixture(recipe, output)
    assert json.loads((output / mixture.SUMMARY_NAME).read_text()) == first


@pytest.mark.parametrize("invalid", ["category", "unknown_category", "duplicate_image", "duplicate_annotation",
                                      "empty", "missing_file", "missing_test", "duplicate_name", "repeat_val"])
def test_invalid_inputs_fail_before_output_created(tmp_path: Path, invalid: str) -> None:
    recipe = write_recipe(tmp_path)
    annotation_path = tmp_path / "pmc" / "train.json"
    if invalid == "category":
        update_json(annotation_path, lambda data: data["categories"].append({"id": 8, "name": "axis"}))
    elif invalid == "unknown_category":
        update_json(annotation_path, lambda data: data["annotations"][0].update(category_id=999))
    elif invalid == "duplicate_image":
        update_json(annotation_path, lambda data: data["images"].append(data["images"][0]))
    elif invalid == "duplicate_annotation":
        update_json(annotation_path, lambda data: data["annotations"].append(data["annotations"][0]))
    elif invalid == "empty":
        update_json(annotation_path, lambda data: data.update(images=[], annotations=[]))
    elif invalid == "missing_file":
        (tmp_path / "pmc" / "train.png").unlink()
    elif invalid == "missing_test":
        update_json(recipe, lambda data: data["sources"][0]["splits"].pop("test"))
    elif invalid == "duplicate_name":
        update_json(recipe, lambda data: data["sources"][1].update(name="PMC"))
    elif invalid == "repeat_val":
        update_json(recipe, lambda data: data["sources"][0]["splits"]["val"].update(repeat=2))
    output = tmp_path / "prepared"
    with pytest.raises((ValueError, FileNotFoundError)):
        mixture.prepare_mixture(recipe, output)
    assert not output.exists()


def test_cross_source_resolved_train_validation_leakage(tmp_path: Path) -> None:
    recipe = write_recipe(tmp_path)
    # Separate annotation files still resolve to the same physical path.
    update_json(tmp_path / "pmc" / "val.json", lambda data: data["images"][0].update(file_name="../dsc/./train.png"))
    with pytest.raises(ValueError, match="Train/val image leakage"):
        mixture.prepare_mixture(recipe, tmp_path / "prepared")


def test_distinct_image_records_share_one_source_file(tmp_path: Path) -> None:
    recipe = write_recipe(tmp_path)
    update_json(tmp_path / "adobe" / "train.json", lambda data: data["images"].append({**data["images"][0], "id": 43}))
    output = tmp_path / "prepared"
    summary = mixture.prepare_mixture(recipe, output)
    train = load_split(output, "train")
    assert summary["splits"]["train"]["images"] == 54
    assert summary["splits"]["train"]["unique_image_files"] == 4
    assert [ann["image_id"] for ann in train["annotations"]][-3:] == [51, 53, 54]
    assert train["images"][50]["file_name"] == train["images"][51]["file_name"]


def test_link_fallback_and_failure_never_publish_ready_summary(tmp_path: Path, monkeypatch) -> None:
    recipe = write_recipe(tmp_path)
    output = tmp_path / "prepared"

    def no_links(*args, **kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", no_links)
    original_copy = mixture.shutil.copy2
    copies = []

    def fail_second_copy(source, destination):
        copies.append(source)
        if len(copies) == 2:
            raise OSError("disk full")
        return original_copy(source, destination)

    monkeypatch.setattr(mixture.shutil, "copy2", fail_second_copy)
    with pytest.raises(OSError, match="disk full"):
        mixture.prepare_mixture(recipe, output)
    assert (output / mixture.MARKER_NAME).is_file()
    assert not (output / mixture.SUMMARY_NAME).exists()
    monkeypatch.setattr(mixture.shutil, "copy2", original_copy)
    summary = mixture.prepare_mixture(recipe, output)
    assert summary["status"] == "ready"
    assert not list(output.rglob("*.tmp"))


def test_unrecognized_output_is_never_overwritten(tmp_path: Path) -> None:
    recipe = write_recipe(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "my_dataset.json"
    sentinel.write_text("existing dataset")
    with pytest.raises(ValueError, match="unrecognized nonempty"):
        mixture.prepare_mixture(recipe, output)
    assert sentinel.read_text() == "existing dataset"


def test_polygon_masks_are_preserved_without_rasterization(tmp_path: Path) -> None:
    recipe = write_recipe(tmp_path)
    polygon = [[1, 2, 4, 2, 4, 6, 1, 6]]
    update_json(tmp_path / "lineex" / "train.json", lambda data: data["annotations"][0].update(segmentation=polygon))
    output = tmp_path / "prepared"
    mixture.prepare_mixture(recipe, output)
    annotations = load_split(output, "train")["annotations"]
    assert annotations[51]["segmentation"] == polygon


def test_running_preparation_is_locked_and_lock_releases(tmp_path: Path) -> None:
    recipe = write_recipe(tmp_path)
    output = tmp_path / "prepared"
    with mixture._lock_output(output):
        with pytest.raises(ValueError, match="Another process"):
            mixture.prepare_mixture(recipe, output)
    assert mixture.prepare_mixture(recipe, output)["status"] == "ready"


def test_interrupted_mixture_rejects_changed_recipe(tmp_path: Path) -> None:
    recipe = write_recipe(tmp_path)
    output = tmp_path / "prepared"
    mixture.prepare_mixture(recipe, output)
    (output / mixture.SUMMARY_NAME).unlink()
    update_json(recipe, lambda data: data["sources"][0]["splits"]["train"].update(repeat=3))
    with pytest.raises(ValueError, match="fingerprint changed"):
        mixture.prepare_mixture(recipe, output)
    assert not (output / mixture.SUMMARY_NAME).exists()


@pytest.mark.parametrize("metadata", [None, [], "description", 1])
def test_recipe_metadata_requires_an_object(tmp_path: Path, metadata) -> None:
    recipe = write_recipe(tmp_path)
    update_json(recipe, lambda data: data.update(metadata=metadata))
    with pytest.raises(ValueError, match="metadata must be an object"):
        mixture.prepare_mixture(recipe, tmp_path / "prepared")


def test_public_recipe_with_actual_metadata_builds_mixture_then_smoke(tmp_path: Path, monkeypatch) -> None:
    import training.prepare_lineformer_public as public
    from training.prepare_lineformer_smoke import prepare_smoke
    from training.train_lineformer import validate_dataset

    prepared_public = tmp_path / "public"
    raw = tmp_path / "raw"
    dsc = tmp_path / "dsc"
    roots = {
        "pmc": {"train": raw / "pmc_train", "val": raw / "pmc_train", "test": raw / "pmc_test"},
        "adobe": {"train": raw / "adobe/images", "val": raw / "adobe/images",
                  "test": raw / "adobe/test_release/task6"},
        "lineex": {split: raw / "lineex" / split / "images" for split in mixture.SPLITS},
        "dsc": {split: dsc / "images" / split for split in mixture.SPLITS},
    }
    coco_roots = {name: dsc if name == "dsc" else prepared_public / name for name in roots}
    for source, image_roots in roots.items():
        for split, image_root in image_roots.items():
            image_root.mkdir(parents=True, exist_ok=True)
            (image_root / f"{split}.png").write_bytes(b"fixture-image")
            annotation_path = coco_roots[source] / "annotations" / f"instances_{split}.json"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(json.dumps({
                "info": {"mask_dilation": 1}, "licenses": [], "categories": public.CATEGORIES,
                "images": [{"id": 1, "file_name": f"{split}.png", "width": 8, "height": 8}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                 "segmentation": [[1, 2, 4, 2, 4, 6, 1, 6]],
                                 "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0}],
            }), encoding="utf-8")
    # Raster conversion has its own tests; exercise the real producer's recipe
    # and metadata construction using already prepared tiny COCO sources.
    monkeypatch.setattr(public, "validate_dsc", lambda *args: {})
    monkeypatch.setattr(public, "prepare_chartinfo", lambda *args: {})
    monkeypatch.setattr(public, "prepare_lineex", lambda *args: {})
    assert public.main([
        "--pmc-train-root", str(roots["pmc"]["train"]), "--pmc-test-root", str(roots["pmc"]["test"]),
        "--adobe-root", str(raw / "adobe"), "--lineex-root", str(raw / "lineex"),
        "--dsc-coco", str(dsc), "--output", str(prepared_public),
    ]) == 0
    recipe_path = prepared_public / "recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert recipe["sources"] == [public.recipe_source(name, coco_roots[name], roots[name]) for name in roots]
    assert recipe["metadata"]["line_width"] == 1
    assert recipe["metadata"]["validation_fraction"] == 0.1
    assert recipe["metadata"]["deviations_from_original"]

    output = tmp_path / "mixture"
    summary = mixture.prepare_mixture(recipe_path, output)
    assert summary["recipe"]["metadata"] == summary["source_recipe"]["metadata"] == recipe["metadata"]
    assert summary["splits"]["train"]["images"] == 53
    assert summary["status"] == "ready"
    smoke = tmp_path / "smoke"
    smoke_summary = prepare_smoke(output, smoke)
    validate_dataset(smoke)
    assert smoke_summary["source_fingerprint"] == summary["fingerprint"]
    for split in mixture.SPLITS:
        assert smoke_summary["splits"][split]["source_images"] == {name: 1 for name in roots}
    # Changes to provenance must not silently reuse a previous preparation.
    update_json(recipe_path, lambda data: data["metadata"].update(line_width=2))
    with pytest.raises(ValueError, match="fingerprint changed"):
        mixture.prepare_mixture(recipe_path, output)
