"""Select a small, source-balanced COCO subset for GPU training/evaluation smoke tests."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import islice
import json
from pathlib import Path

import ijson

from training.prepare_lineformer_mixture import SPLITS, SUMMARY_NAME, _atomic_json, _json_bytes


def _items(path: Path, field: str, count: int):
    with path.open("rb") as stream:
        yield from islice(ijson.items(stream, f"{field}.item", use_float=True), count)


def _select_split(dataset: Path, split: str, counts: dict, quota: int) -> tuple[dict, dict]:
    annotation_path = dataset / "annotations" / f"instances_{split}.json"
    selected, selected_paths, dsc_candidates, dsc_paths = defaultdict(list), defaultdict(set), set(), set()
    for image in _items(annotation_path, "images", counts["images"]):
        source = image["mixture_provenance"]["source"]
        filename = image["file_name"]
        if len(selected[source]) < quota and filename not in selected_paths[source]:
            selected[source].append(image)
            selected_paths[source].add(filename)
        if "dsc" in source.lower() and (source, filename) not in dsc_paths:
            dsc_candidates.add(image["id"])
            dsc_paths.add((source, filename))
    selected_ids = {image["id"] for images in selected.values() for image in images}
    annotations, annotated_ids = [], set()
    for annotation in _items(annotation_path, "annotations", counts["annotations"]):
        image_id = annotation["image_id"]
        dsc_candidates.discard(image_id)
        if image_id in selected_ids:
            annotations.append(annotation)
            annotated_ids.add(image_id)
    # Include a hard negative even when the first DSC examples all have curves.
    dsc_selected = [image for source, images in selected.items() if "dsc" in source.lower() for image in images]
    if dsc_candidates and not any(image["id"] not in annotated_ids for image in dsc_selected):
        negative_id = min(dsc_candidates)
        for image in _items(annotation_path, "images", counts["images"]):
            if image["id"] == negative_id:
                replaced = selected[image["mixture_provenance"]["source"]][-1]
                selected[image["mixture_provenance"]["source"]][-1] = image
                annotations = [annotation for annotation in annotations if annotation["image_id"] != replaced["id"]]
                break
    images = [dict(image) for group in selected.values() for image in group]
    for image in images:
        image["file_name"] = str((dataset / "images" / split / image["file_name"]).resolve(strict=True))
    if not images or not annotations:
        raise ValueError(f"Smoke split {split} must contain images and annotations")
    with annotation_path.open("rb") as stream:
        categories = next(ijson.items(stream, "categories", use_float=True))
    with annotation_path.open("rb") as stream:
        licenses = next(ijson.items(stream, "licenses", use_float=True), [])
    payload = {"info": {"description": "LineFormer mixture GPU smoke subset"}, "licenses": licenses,
               "categories": categories, "images": images, "annotations": annotations}
    details = {"images": len(images), "annotations": len(annotations),
               "source_images": {source: len(group) for source, group in selected.items()},
               "selected_image_ids": [image["id"] for image in images],
               "empty_dsc_image_ids": [image["id"] for image in images
                                       if "dsc" in image["mixture_provenance"]["source"].lower()
                                       and image["id"] not in annotated_ids]}
    return payload, details


def prepare_smoke(dataset: Path, output: Path) -> dict:
    dataset, output = dataset.resolve(strict=True), output.resolve()
    full_summary = json.loads((dataset / SUMMARY_NAME).read_text(encoding="utf-8"))
    if full_summary.get("status") != "ready":
        raise ValueError("Source mixture must have status=ready")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Smoke output must be a new or empty directory")
    prepared = {split: _select_split(dataset, split, full_summary["splits"][split], 8 if split == "train" else 2)
                for split in SPLITS}
    details = {split: result[1] for split, result in prepared.items()}
    provenance = {"source_fingerprint": full_summary["fingerprint"],
                  "selected_image_ids": {split: detail["selected_image_ids"] for split, detail in details.items()}}
    summary = {"status": "ready", "kind": "gpu_smoke", "source_dataset": str(dataset), **provenance,
               "fingerprint": hashlib.sha256(_json_bytes(provenance)).hexdigest(), "splits": details}
    (output / "annotations").mkdir(parents=True, exist_ok=True)
    for split, (payload, _) in prepared.items():
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        _atomic_json(output / "annotations" / f"instances_{split}.json", payload)
    _atomic_json(output / SUMMARY_NAME, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_smoke(args.dataset, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
