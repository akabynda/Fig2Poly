"""Rebuild public LineFormer sources as COCO without materializing mask PNGs.

This reproduces the source families, not the unpublished author COCO exports.
Official PMC/Adobe train releases receive a deterministic 10% validation holdout;
their official test releases never participate in training or model selection.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import ijson
import numpy as np
from PIL import Image, ImageDraw

from training.convert_coco_instances import coco_counts
from training.convert_public_instances import (
    chartinfo_lines,
    chartinfo_occlusion_mask,
    find_chartinfo_pairs,
    stable_validation,
)


SPLITS = ("train", "val", "test")
CATEGORIES = [{"id": 1, "name": "line", "supercategory": "plot"}]
CONVERSION_VERSION = 1


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def mask_annotation(mask: Image.Image) -> dict | None:
    binary = np.asarray(mask, dtype=np.uint8) > 0
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return None
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        counts = coco_counts(binary)
    else:
        encoded = mask_utils.encode(np.asfortranarray(binary, dtype=np.uint8))
        counts = encoded["counts"].decode("ascii")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return {
        "category_id": 1,
        "segmentation": {"size": [mask.height, mask.width], "counts": counts},
        "area": int(binary.sum()),
        "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
        "iscrowd": 0,
    }


def rasterized_instances(
    lines: Iterable[list[tuple[float, float]]],
    size: tuple[int, int],
    line_width: int,
    occlusions: Image.Image,
) -> list[dict]:
    visible = Image.eval(occlusions, lambda value: 255 - value)
    empty = Image.new("L", size, 0)
    result = []
    for points in lines:
        if len(points) < 2:
            continue
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).line(points, fill=255, width=line_width, joint="curve")
        annotation = mask_annotation(Image.composite(mask, empty, visible))
        if annotation is not None:
            result.append(annotation)
    return result


class CocoWriter:
    """Stream the two COCO arrays to disk; hold only one image's masks in RAM."""

    def __init__(self, target: Path, metadata: dict):
        self.target = target
        self.metadata = metadata
        target.parent.mkdir(parents=True, exist_ok=True)
        self.image_path = target.with_suffix(".images.part")
        self.annotation_path = target.with_suffix(".annotations.part")
        self.images = self.image_path.open("w", encoding="utf-8")
        self.annotations = self.annotation_path.open("w", encoding="utf-8")
        self.image_count = self.annotation_count = self.skipped = 0

    def __enter__(self) -> CocoWriter:
        return self

    def add(self, image: dict, annotations: list[dict]) -> None:
        if not annotations:
            self.skipped += 1
            return
        self.image_count += 1
        if self.image_count > 1:
            self.images.write(",\n")
        self.images.write(json.dumps({**image, "id": self.image_count}))
        for annotation in annotations:
            self.annotation_count += 1
            if self.annotation_count > 1:
                self.annotations.write(",\n")
            self.annotations.write(json.dumps({
                **annotation, "id": self.annotation_count, "image_id": self.image_count,
            }))

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.images.close()
        self.annotations.close()
        if exception_type is not None:
            return
        temporary = self.target.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as result:
            result.write('{"info":' + json.dumps(self.metadata))
            result.write(',"licenses":[],"categories":' + json.dumps(CATEGORIES))
            for key, path in (("images", self.image_path), ("annotations", self.annotation_path)):
                result.write(',"' + key + '":[')
                with path.open("r", encoding="utf-8") as part:
                    shutil.copyfileobj(part, result, 1024 * 1024)
                result.write("]")
            result.write("}")
        os.replace(temporary, self.target)
        self.image_path.unlink()
        self.annotation_path.unlink()

    def statistics(self) -> dict:
        return {"images": self.image_count, "instances": self.annotation_count,
                "skipped_empty": self.skipped}


def file_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def cached_result(receipt_path: Path, settings: dict, targets: list[Path]) -> dict | None:
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("settings") != settings:
            raise ValueError(f"Preparation settings changed; use a new output directory: {receipt_path}")
        if all(path.is_file() and path.stat().st_size > 0 for path in targets):
            print(f"Reuse completed {receipt_path}", flush=True)
            return receipt["statistics"]
    return None


def prepare_chartinfo(
    raw_root: Path, annotation_root: Path | None, output: Path,
    source: str, official_split: str, validation_fraction: float, line_width: int,
) -> dict:
    pairs = find_chartinfo_pairs(raw_root, annotation_root=annotation_root)
    names = [path.stem for path, _ in pairs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate ChartInfo annotation stems in {raw_root}")
    splits = ("train", "val") if official_split == "train" else (official_split,)
    settings = {
        "conversion_version": CONVERSION_VERSION, "source": source,
        "official_split": official_split, "image_root": str(raw_root.resolve()),
        "annotation_fingerprint": file_fingerprint(path for path, _ in pairs),
        "chart_types": ["line", "scatter line", "unspecified-with-lines"],
        "validation_fraction": validation_fraction, "line_width": line_width,
        "mask_semantics": "polyline proxy; annotated text/legend occlusions removed",
    }
    targets = {split: output / "annotations" / f"instances_{split}.json" for split in splits}
    receipt_path = output / f"prepare_{official_split}.json"
    cached = cached_result(receipt_path, settings, list(targets.values()))
    if cached is not None:
        return cached
    with ExitStack() as stack:
        writers = {split: stack.enter_context(CocoWriter(path, settings))
                   for split, path in targets.items()}
        for index, (annotation_path, image_path) in enumerate(pairs, 1):
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            split = official_split
            if official_split == "train" and stable_validation(annotation_path.stem, validation_fraction):
                split = "val"
            with Image.open(image_path) as image:
                size = image.size
            instances = rasterized_instances(chartinfo_lines(annotation), size, line_width,
                                             chartinfo_occlusion_mask(annotation, size))
            writers[split].add({
                "file_name": image_path.relative_to(raw_root).as_posix(),
                "width": size[0], "height": size[1],
                "source_id": f"{source}__{official_split}__{annotation_path.stem}",
                "dataset_source": source, "official_source_split": official_split,
            }, instances)
            if index % 1000 == 0:
                print(f"{source}/{official_split}: {index}/{len(pairs)}", flush=True)
    statistics = {split: writer.statistics() for split, writer in writers.items()}
    if any(item["images"] == 0 for item in statistics.values()):
        raise ValueError(f"Empty prepared split: {source}/{official_split}: {statistics}")
    write_json(receipt_path, {"settings": settings, "statistics": statistics})
    return statistics


def lineex_groups(path: Path):
    current_id = None
    group = []
    with path.open("rb") as stream:
        for annotation in ijson.items(stream, "annotations.item"):
            image_id = int(annotation["image_id"])
            if current_id is not None and image_id < current_id:
                raise ValueError(f"LineEX annotations are not grouped by image_id: {path}")
            if current_id is not None and image_id != current_id:
                yield current_id, group
                group = []
            current_id = image_id
            group.append(annotation)
    if current_id is not None:
        yield current_id, group


def prepare_lineex(raw_root: Path, output: Path, split: str, line_width: int) -> dict:
    line_path = raw_root / "anno" / "line_anno.json"
    class_path = raw_root / "anno" / "cls_anno.json"
    target = output / "annotations" / f"instances_{split}.json"
    receipt_path = output / f"prepare_{split}.json"
    settings = {
        "conversion_version": CONVERSION_VERSION, "source": "lineex",
        "official_split": split, "image_root": str((raw_root / "images").resolve()),
        "annotation_fingerprint": file_fingerprint([line_path, class_path]),
        "line_width": line_width,
        "mask_semantics": "polyline proxy; released legend rectangle removed",
    }
    cached = cached_result(receipt_path, settings, [target])
    if cached is not None:
        return cached
    with line_path.open("rb") as stream:
        images = {int(item["id"]): item for item in ijson.items(stream, "images.item")}
    legends = {}
    with class_path.open("rb") as stream:
        for item in ijson.items(stream, "annotations.item"):
            if int(item["category_id"]) == 0:
                legends.setdefault(int(item["image_id"]), []).append(item["bbox"])
    with CocoWriter(target, settings) as writer:
        for index, (image_id, annotations) in enumerate(lineex_groups(line_path), 1):
            info = images[image_id]
            image_path = raw_root / "images" / info["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing released LineEX image: {image_path}")
            size = int(info["width"]), int(info["height"])
            occlusions = Image.new("L", size, 0)
            draw = ImageDraw.Draw(occlusions)
            for bbox in legends.get(image_id, []):
                x, y, width, height = map(float, bbox)
                draw.rectangle((x, y, x + width, y + height), fill=255)
            lines = []
            for annotation in sorted(annotations, key=lambda item: int(item["id"])):
                flat = annotation.get("bbox", [])
                lines.append([(float(flat[i]), float(flat[i + 1]))
                              for i in range(0, len(flat) - 1, 2)])
            writer.add({
                "file_name": info["file_name"], "width": size[0], "height": size[1],
                "source_id": f"lineex__{split}__{image_id}",
                "dataset_source": "lineex", "official_source_split": split,
            }, rasterized_instances(lines, size, line_width, occlusions))
            if index % 1000 == 0:
                print(f"lineex/{split}: {index}/{len(images)}", flush=True)
    statistics = writer.statistics()
    statistics["released_images"] = len(images)
    statistics["images_without_line_annotations"] = len(images) - writer.image_count - writer.skipped
    if writer.image_count == 0:
        raise ValueError(f"No converted LineEX images in {raw_root}")
    write_json(receipt_path, {"settings": settings, "statistics": statistics})
    return statistics


def verify_legacy_dsc_samples(
    coco_root: Path, source: Path, split: str, sample_count: int = 8,
) -> dict:
    """Verify legacy COCO against unchanged raw masks, without inferring dilation.

    Sample the first eight distinct annotated image IDs in annotation order and
    compare every instance for those images. This is a sampled provenance check,
    not an exhaustive assertion about the rest of the dataset.
    """
    path = coco_root / "annotations" / f"instances_{split}.json"
    sampled: dict[int, list[dict]] = {}
    with path.open("rb") as stream:
        for annotation in ijson.items(stream, "annotations.item"):
            image_id = int(annotation["image_id"])
            if image_id in sampled or len(sampled) < sample_count:
                sampled.setdefault(image_id, []).append(annotation)
    if len(sampled) != sample_count:
        raise ValueError(f"Legacy DSC verification needs {sample_count} annotated images: {path}")
    with path.open("rb") as stream:
        images = {int(item["id"]): item for item in ijson.items(stream, "images.item")
                  if int(item["id"]) in sampled}
    if len(images) != sample_count:
        raise ValueError(f"Legacy DSC annotation/image IDs mismatch: {path}")
    wanted = {str(item["source_id"]): image_id for image_id, item in images.items()}
    if len(wanted) != sample_count:
        raise ValueError(f"Legacy DSC has duplicate source IDs: {path}")
    manifest = source / f"{split}.jsonl"
    records = {}
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            source_id = str(record["id"])
            if source_id in wanted:
                if source_id in records:
                    raise ValueError(f"Duplicate DSC source ID {source_id} in {manifest}")
                records[source_id] = record
            if len(records) == sample_count:
                break
    if records.keys() != wanted.keys():
        raise ValueError(f"Legacy DSC sampled source IDs missing from {manifest}")
    verified_masks = 0
    for source_id, image_id in wanted.items():
        record, info = records[source_id], images[image_id]
        dimensions = int(record["width"]), int(record["height"])
        if dimensions != (int(info["width"]), int(info["height"])):
            raise ValueError(f"Legacy DSC dimensions mismatch: {split}/{source_id}")
        expected = []
        for curve in record.get("curves", []):
            with Image.open(source / curve["mask"]) as opened:
                mask = opened.convert("L")
            if mask.size != dimensions:
                raise ValueError(f"Raw DSC mask dimensions mismatch: {split}/{source_id}")
            annotation = mask_annotation(mask)
            if annotation is not None:
                expected.append(annotation)
        actual = sorted(sampled[image_id], key=lambda item: int(item["id"]))
        if len(expected) != len(actual):
            raise ValueError(f"Legacy DSC instance count mismatch: {split}/{source_id}")
        for left, right in zip(expected, actual):
            if any(left[key] != right.get(key) for key in (
                "segmentation", "area", "bbox", "category_id", "iscrowd",
            )):
                raise ValueError(f"Legacy DSC masks differ from exact raw masks: {split}/{source_id}")
        verified_masks += len(expected)
    return {
        "method": "sampled exact raw-mask compressed RLE, bbox and area comparison",
        "scope": "first eight distinct annotated image IDs; every instance in those images",
        "verified_images": sample_count, "verified_masks": verified_masks,
        "source_ids": list(wanted), "source_manifest": str(manifest.resolve()),
        "source_manifest_fingerprint": file_fingerprint([manifest]),
        "exhaustive": False,
    }


def validate_dsc(root: Path, source: Path | None = None) -> dict:
    statistics = {}
    for split, expected in zip(SPLITS, (80000, 10000, 10000)):
        path = root / "annotations" / f"instances_{split}.json"
        with path.open("rb") as stream:
            count = sum(1 for _ in ijson.items(stream, "images.item"))
        with path.open("rb") as stream:
            info = next(ijson.items(stream, "info"), {})
        with path.open("rb") as stream:
            categories = list(ijson.items(stream, "categories.item"))
        dilation = 1
        reported_dilation = info.get("mask_dilation")
        if count != expected or (reported_dilation is not None and reported_dilation != dilation):
            raise ValueError(f"Unexpected DSC {split}: {count} images, dilation={info.get('mask_dilation')}; "
                             f"expected {expected} and {dilation}; verify the earlier LineFormer DSC run")
        if categories != CATEGORIES:
            raise ValueError(f"Unexpected DSC category schema: {categories}")
        verification = None
        if reported_dilation is None:
            verification = verify_legacy_dsc_samples(root, source or root.parent / "dataset_dsc", split)
        statistics[split] = {"images": count, "mask_dilation": reported_dilation,
                             "legacy_exact_mask_verification": verification,
                             "annotation_fingerprint": file_fingerprint([path])}
    return statistics


def recipe_source(name: str, coco_root: Path, image_roots: dict[str, Path]) -> dict:
    return {"name": name, "splits": {
        split: {"annotations": str((coco_root / "annotations" / f"instances_{split}.json").resolve()),
                "image_root": str(image_roots[split].resolve()),
                "repeat": 50 if name == "pmc" and split == "train" else 1}
        for split in SPLITS
    }}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmc-train-root", type=Path, required=True)
    parser.add_argument("--pmc-test-root", type=Path, required=True)
    parser.add_argument("--adobe-root", type=Path, required=True)
    parser.add_argument("--lineex-root", type=Path, required=True)
    parser.add_argument("--dsc-coco", type=Path, required=True)
    parser.add_argument("--dsc-source", type=Path,
                        help="Raw DSC JSONL/masks for legacy COCO verification; default: DSC parent/dataset_dsc")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--line-width", type=int, default=1)
    args = parser.parse_args(argv)
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.line_width < 1:
        parser.error("--line-width must be positive")
    for name in ("pmc_train_root", "pmc_test_root", "adobe_root", "lineex_root", "dsc_coco", "output"):
        setattr(args, name, getattr(args, name).resolve())
    statistics = {"dsc": validate_dsc(args.dsc_coco, args.dsc_source)}
    statistics["pmc"] = {}
    statistics["adobe"] = {}
    adobe_train = args.adobe_root / "images"
    adobe_test = args.adobe_root / "test_release" / "task6"
    for name, split, root, annotations in (
        ("pmc", "train", args.pmc_train_root, None),
        ("pmc", "test", args.pmc_test_root, None),
        ("adobe", "train", adobe_train, args.adobe_root / "json_gt"),
        ("adobe", "test", adobe_test, adobe_test / "gt_json"),
    ):
        statistics[name].update(prepare_chartinfo(root, annotations, args.output / name,
                                                name, split, args.validation_fraction, args.line_width))
    statistics["lineex"] = {
        split: prepare_lineex(args.lineex_root / split, args.output / "lineex", split, args.line_width)
        for split in SPLITS
    }
    sources = [
        recipe_source("pmc", args.output / "pmc", {"train": args.pmc_train_root,
                      "val": args.pmc_train_root, "test": args.pmc_test_root}),
        recipe_source("adobe", args.output / "adobe", {"train": adobe_train,
                      "val": adobe_train, "test": adobe_test}),
        recipe_source("lineex", args.output / "lineex", {
            split: args.lineex_root / split / "images" for split in SPLITS}),
        recipe_source("dsc", args.dsc_coco, {split: args.dsc_coco / "images" / split for split in SPLITS}),
    ]
    metadata = {
        "description": "Public PMC + AdobeSynth + LineEX with the earlier LineFormer DSC COCO dataset",
        "conversion_version": CONVERSION_VERSION, "validation_fraction": args.validation_fraction,
        "validation_rule": "SHA256(original annotation stem), first 64 bits / 2**64 < fraction; TRAIN only",
        "line_width": args.line_width,
        "deviations_from_original": [
            "Author COCO exports and exact split IDs are unavailable; reconstructed from official raw releases.",
            "All eligible released training images used; original config comments suggest smaller subsets.",
            "PMC/Adobe validation is a train holdout; LineEX and DSC use their existing validation split.",
            "All four validation sources used for selection; official test is reserved for final evaluation.",
            "ChartInfo line/scatter-line (or unspecified type with lines); rasterized polyline proxy masks, not author masks.",
            "Public masks remove annotated occlusions; default rasterization is one pixel, with no mask dilation.",
            "Existing DSC exact masks reused: dilation metadata must be 1; legacy files require sampled raw-mask verification.",
        ],
    }
    write_json(args.output / "preparation_manifest.json", {"metadata": metadata, "statistics": statistics})
    write_json(args.output / "recipe.json", {"sources": sources, "metadata": metadata})
    print(json.dumps({"recipe": str(args.output / "recipe.json"), "statistics": statistics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
