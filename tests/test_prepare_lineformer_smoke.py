import json
from collections import Counter
from pathlib import Path

import pytest

from training.prepare_lineformer_smoke import prepare_dsc_preflight, prepare_smoke
from training.train_lineformer import validate_dataset


def full_mixture(root: Path) -> Path:
    dataset = root / "full"
    (dataset / "annotations").mkdir(parents=True)
    counts = {}
    for split in ("train", "val", "test"):
        images, annotations = [], []
        unique_count = 10 if split == "train" else 3
        for source in ("pmc", "adobe", "lineex", "dsc_exact"):
            (dataset / "images" / split / source).mkdir(parents=True)
            for repetition in range(3 if source == "pmc" else 1):
                for index in range(unique_count):
                    filename = f"{source}/{index}.png"
                    (dataset / "images" / split / filename).write_bytes(b"image")
                    image_id = len(images) + 1
                    images.append({"id": image_id, "file_name": filename, "width": 10, "height": 10,
                                   "mixture_provenance": {"source": source, "repetition": repetition}})
                    if source == "dsc_exact" and index == unique_count - 1:
                        continue
                    annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                        "segmentation": [[0, 0, 3, 0, 3, 4]], "bbox": [0, 0, 3, 4],
                                        "area": 6, "iscrowd": 0})
        payload = {"info": {}, "licenses": [], "categories": [{"id": 1, "name": "line"}],
                   "images": images, "annotations": annotations}
        (dataset / "annotations" / f"instances_{split}.json").write_text(json.dumps(payload), encoding="utf-8")
        counts[split] = {"images": len(images), "annotations": len(annotations)}
    (dataset / "mixture_summary.json").write_text(json.dumps({"status": "ready", "fingerprint": "full-source-fingerprint",
                                                            "splits": counts}), encoding="utf-8")
    return dataset


def test_smoke_balances_sources_deduplicates_repeats_and_finds_dsc_negative(tmp_path: Path) -> None:
    dataset = full_mixture(tmp_path)
    output = tmp_path / "smoke"
    summary = prepare_smoke(dataset, output)
    validate_dataset(output)
    assert summary["status"] == "ready"
    assert summary["source_fingerprint"] == "full-source-fingerprint"
    for split, quota in (("train", 8), ("val", 2), ("test", 2)):
        payload = json.loads((output / "annotations" / f"instances_{split}.json").read_text())
        assert len(payload["images"]) == 4 * quota
        assert Counter(image["mixture_provenance"]["source"] for image in payload["images"]) == {
            "pmc": quota, "adobe": quota, "lineex": quota, "dsc_exact": quota,
        }
        assert len({image["file_name"] for image in payload["images"]}) == 4 * quota
        assert all(Path(image["file_name"]).is_absolute() and Path(image["file_name"]).is_file() for image in payload["images"])
        image_ids = {image["id"] for image in payload["images"]}
        assert {annotation["image_id"] for annotation in payload["annotations"]} <= image_ids
        negative_ids = set(summary["splits"][split]["empty_dsc_image_ids"])
        assert len(negative_ids) == 1
        assert negative_ids <= image_ids
        assert not negative_ids.intersection(annotation["image_id"] for annotation in payload["annotations"])
        original = json.loads((dataset / "annotations" / f"instances_{split}.json").read_text())
        original_annotations = {annotation["id"]: annotation for annotation in original["annotations"]}
        assert all(annotation == original_annotations[annotation["id"]] for annotation in payload["annotations"])
    second = prepare_smoke(dataset, tmp_path / "smoke_again")
    assert second["fingerprint"] == summary["fingerprint"]
    assert second["selected_image_ids"] == summary["selected_image_ids"]


def test_smoke_requires_ready_source_and_new_output(tmp_path: Path) -> None:
    dataset = full_mixture(tmp_path)
    output = tmp_path / "smoke"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="new or empty"):
        prepare_smoke(dataset, output)
    assert sentinel.read_text() == "keep"
    summary_path = dataset / "mixture_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "preparing"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="status=ready"):
        prepare_smoke(dataset, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_dsc_preflight_preserves_masks_and_routes_absolute_paths(tmp_path: Path) -> None:
    from training.maskdino_source_augmentation import image_source

    dataset = full_mixture(tmp_path)
    output = tmp_path / "preflight"
    summary = prepare_dsc_preflight(dataset, output)
    validate_dataset(output)
    assert summary["kind"] == "exact_dsc_runtime_preflight"
    for split in ("train", "val", "test"):
        before = json.loads((dataset / "annotations" / f"instances_{split}.json").read_text())
        after = json.loads((output / "annotations" / f"instances_{split}.json").read_text())
        selected = {item["id"] for item in after["images"]}
        assert after["annotations"] == [item for item in before["annotations"] if item["image_id"] in selected]
        assert all(image_source(item["file_name"]) == "dsc" for item in after["images"])
        assert all(Path(item["file_name"]).is_file() for item in after["images"])
