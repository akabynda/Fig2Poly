from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a one-empty-image COCO dataset for MaskDINO smoke tests"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()
    train = json.loads(
        (source / "annotations" / "instances_train.json").read_text(encoding="utf-8")
    )
    annotated_ids = {int(item["image_id"]) for item in train["annotations"]}
    empty = next(
        (item for item in train["images"] if int(item["id"]) not in annotated_ids),
        None,
    )
    if empty is None:
        parser.error("no empty training image found; empty-target smoke test is impossible")

    nonempty = next(
        (item for item in train["images"] if int(item["id"]) in annotated_ids),
        None,
    )
    if nonempty is None:
        parser.error("no annotated training image found; RLE smoke test is impossible")
    selected_ids = {int(empty["id"]), int(nonempty["id"])}
    selected_annotations = [
        item for item in train["annotations"] if int(item["image_id"]) in selected_ids
    ]
    payload = {
        "info": {"description": "MaskDINO empty-target and RLE smoke test"},
        "licenses": train.get("licenses", []),
        "categories": train["categories"],
        "images": [empty, nonempty],
        "annotations": selected_annotations,
    }
    annotations = output / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for image in (empty, nonempty):
            source_image = source / "images" / "train" / image["file_name"]
            split_image = output / "images" / split / image["file_name"]
            link_or_copy(source_image, split_image)
        (annotations / f"instances_{split}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "empty_image_id": empty["id"],
                "nonempty_image_id": nonempty["id"],
                "instances": len(selected_annotations),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
