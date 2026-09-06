import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image
import pytest

from training import download_public_benchmarks as downloader
from training.adobe_release import ADOBE_V1_MISSING_LINE_IMAGES
from training.convert_public_instances import find_chartinfo_pairs
from training.prepare_lineformer_public import prepare_chartinfo


def annotation(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "task1": {"output": {"chart_type": "Line"}},
        "task6": {"output": {"visual elements": {
            "lines": [[{"x": 1, "y": 1}, {"x": 5, "y": 1}]],
        }}},
    }))


def fixture(root: Path) -> list:
    raw = root / "raw/adobe_synth19"
    images = raw / "images"
    images.mkdir(parents=True)
    shards = [a for a in downloader.ASSETS
              if a.dataset == "adobe_synth19" and a.filename.startswith("images_")]
    entries = []
    for i, asset in enumerate(shards):
        annotation(raw / "json_gt" / f"available{i}.json")
        Image.new("RGB", (8, 8)).save(images / f"available{i}.png")
        entries.append({**asdict(asset), "sha256": "a" * 64,
                        "policy": downloader.extraction_policy(asset, True),
                        "completed": 123, "extracted_files": 1, "extracted_to": str(raw)})
    for stem in ADOBE_V1_MISSING_LINE_IMAGES:
        annotation(raw / "json_gt" / f"{stem}.json")
    (root / "download_manifest.json").write_text(json.dumps({"assets": entries}))
    return shards


def test_recovery_reuses_all_shards_and_documents_only_missing_pairs(tmp_path):
    shards = fixture(tmp_path)
    downloader.recover_adobe_receipts(tmp_path, shards, tmp_path / "download.log")
    assert len(list((tmp_path / "_state/adobe_synth19").glob("*.json"))) == 10
    audit = json.loads((tmp_path / "adobe_line_image_audit.json").read_text())
    assert audit["available_images"] == 10
    assert set(audit["excluded_missing_image_stems"]) == ADOBE_V1_MISSING_LINE_IMAGES
    raw = tmp_path / "raw/adobe_synth19"
    with pytest.raises(RuntimeError, match="Missing 7"):
        find_chartinfo_pairs(raw / "images", annotation_root=raw / "json_gt")
    pairs = find_chartinfo_pairs(raw / "images", annotation_root=raw / "json_gt",
                                allowed_missing_stems=ADOBE_V1_MISSING_LINE_IMAGES)
    assert len(pairs) == 10
    output = tmp_path / "prepared"
    prepare_chartinfo(raw / "images", raw / "json_gt", output, "adobe", "train", 0.5, 1)
    receipt = json.loads((output / "prepare_train.json").read_text())
    assert set(receipt["settings"]["excluded_missing_image_stems"]) == ADOBE_V1_MISSING_LINE_IMAGES
    with pytest.raises(RuntimeError, match="Missing 7"):
        prepare_chartinfo(raw / "images", raw / "json_gt", tmp_path / "test",
                          "adobe", "test", 0.5, 1)
    # An allowlisted ID must still be included if its image is later recovered.
    stem = sorted(ADOBE_V1_MISSING_LINE_IMAGES)[0]
    Image.new("RGB", (8, 8)).save(raw / "images" / f"{stem}.png")
    pairs = find_chartinfo_pairs(raw / "images", annotation_root=raw / "json_gt",
                                allowed_missing_stems=ADOBE_V1_MISSING_LINE_IMAGES)
    assert len(pairs) == 11


def test_recovery_fails_before_receipts_if_another_image_is_missing(tmp_path):
    shards = fixture(tmp_path)
    (tmp_path / "raw/adobe_synth19/images/available0.png").unlink()
    with pytest.raises(RuntimeError, match="available0"):
        downloader.recover_adobe_receipts(tmp_path, shards, tmp_path / "download.log")
    assert not (tmp_path / "_state").exists()


def test_incomplete_shard_history_cannot_certify_existing_directory(tmp_path):
    shards = fixture(tmp_path)
    path = tmp_path / "download_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["assets"].pop()
    path.write_text(json.dumps(manifest))
    downloader.recover_adobe_receipts(tmp_path, shards, tmp_path / "download.log")
    assert not (tmp_path / "_state").exists()
