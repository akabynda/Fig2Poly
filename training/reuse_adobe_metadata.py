"""Stage verified Adobe metadata using hardlinks, keeping the originals unchanged."""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import uuid

from training.download_public_benchmarks import ASSETS


METADATA_ASSETS = [asset for asset in ASSETS
                   if asset.dataset == "adobe_synth19" and asset.split in {"train", "test"}]
MARKER = ".fig2poly_reused_metadata.json"


def verified_receipts(root: Path) -> list[dict] | None:
    receipts = []
    for asset in METADATA_ASSETS:
        path = root / "_state" / asset.dataset / f"{asset.filename}.json"
        if not path.is_file():
            return None
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if not (receipt.get("completed") and receipt.get("policy") == "full-extract-v1"
                and receipt.get("url") == asset.url and receipt.get("size") == asset.size):
            return None
        receipts.append(receipt)
    raw = root / "raw" / "adobe_synth19"
    if not all((raw / name).is_dir() for name in ("json_gt", "test_release")):
        return None
    return receipts


def link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)
    return destination


def reuse_metadata(source: Path, destination: Path) -> bool:
    """Return True when both metadata trees are ready; False permits a full download.

    Build both trees in a fresh staging directory and publish their common parent
    atomically. A marker prevents a later full extraction through these hardlinks.
    """
    source, destination = source.resolve(), destination.resolve()
    raw = destination / "raw" / "adobe_synth19"
    marker = raw / MARKER
    if marker.is_file():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("format") != "verified-adobe-metadata-hardlinks-v1":
            raise ValueError(f"Unrecognized metadata reuse marker: {marker}")
        if not all((raw / name).is_dir() for name in ("json_gt", "test_release")):
            raise ValueError(f"Incomplete previously reused metadata; repair without extracting over hardlinks: {raw}")
        return True
    if verified_receipts(destination) is not None:
        return True
    receipts = verified_receipts(source)
    if receipts is None or raw.exists():
        return False
    raw.parent.mkdir(parents=True, exist_ok=True)
    staging = raw.parent / f".adobe_metadata_staging_{uuid.uuid4().hex}"
    staging.mkdir()
    for name in ("json_gt", "test_release"):
        shutil.copytree(source / "raw" / "adobe_synth19" / name, staging / name,
                        copy_function=link_or_copy)
    (staging / MARKER).write_text(json.dumps({
        "format": "verified-adobe-metadata-hardlinks-v1",
        "source_root": str(source), "source_receipts": receipts,
        "download_constraint": "Fetch asset-splits=all (image shards) only; do not extract metadata over these hardlinks.",
    }, indent=2), encoding="utf-8")
    os.rename(staging, raw)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if reuse_metadata(args.source_root, args.destination_root):
        print("Adobe metadata available; download image shards only", flush=True)
        return 0
    print("No complete reusable Adobe metadata; download all released assets", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
