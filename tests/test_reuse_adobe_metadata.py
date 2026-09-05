import json
import os
from pathlib import Path

from training.reuse_adobe_metadata import MARKER, METADATA_ASSETS, reuse_metadata


def complete_source(root: Path) -> None:
    for asset in METADATA_ASSETS:
        receipt = root / "_state/adobe_synth19" / f"{asset.filename}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"completed": 123, "url": asset.url, "size": asset.size,
                                       "policy": "full-extract-v1"}))
    for name in ("json_gt", "test_release/task6/gt_json"):
        directory = root / "raw/adobe_synth19" / name
        directory.mkdir(parents=True)
        (directory / "123.json").write_text('{"original":true}')


def test_complete_metadata_is_hardlinked_and_reusable(tmp_path: Path) -> None:
    source, destination = tmp_path / "existing", tmp_path / "fresh"
    complete_source(source)
    assert reuse_metadata(source, destination)
    original = source / "raw/adobe_synth19/json_gt/123.json"
    staged = destination / "raw/adobe_synth19/json_gt/123.json"
    assert os.path.samefile(original, staged)
    assert original.read_text() == '{"original":true}'
    assert (destination / "raw/adobe_synth19" / MARKER).is_file()
    assert reuse_metadata(source, destination)
    assert not (destination / "raw/adobe_synth19/images").exists()


def test_missing_completed_receipt_does_not_stage(tmp_path: Path) -> None:
    source, destination = tmp_path / "existing", tmp_path / "fresh"
    complete_source(source)
    receipt = source / "_state/adobe_synth19/train_json_gt.tar.gz.json"
    payload = json.loads(receipt.read_text())
    payload["completed"] = None
    receipt.write_text(json.dumps(payload))
    assert not reuse_metadata(source, destination)
    assert not destination.exists()
