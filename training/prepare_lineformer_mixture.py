"""Prepare one COCO mixture while preserving the source instance masks.

Example recipe (paths are relative to the recipe file):
    {"sources": [{"name": "pmc", "splits": {
        "train": {"annotations": "pmc/train.json", "image_root": "pmc/images", "repeat": 50},
        "val": {"annotations": "pmc/val.json", "image_root": "pmc/images"},
        "test": {"annotations": "pmc/val.json", "image_root": "pmc/images"}
    }}, {"name": "adobe", "splits": {"train": {...}}}, ...]}

Every source needs train; val/test can be omitted by individual sources. All
three aggregate splits must contain images and annotations. DSC must already be
COCO: this script never rasterizes, dilates, or otherwise changes masks.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterator
import uuid


SPLITS = ("train", "val", "test")
FORMAT_VERSION = 1
MARKER_NAME = ".mixture_build.json"
SUMMARY_NAME = "mixture_summary.json"
LOCK_NAME = ".mixture.lock"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value < 1):
        raise ValueError(f"{label} must be {'a positive ' if positive else 'an '}integer")
    return value


def _keys(value: dict, allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown fields in {label}: {sorted(unknown)}")


@dataclass
class ImageFile:
    path: Path
    file_name: str
    size: int
    mtime_ns: int


@dataclass
class SourceSplit:
    name: str
    split: str
    annotations: Path
    image_root: Path
    repeat: int
    metadata: dict
    files: dict[Path, ImageFile]


def _image_path(image_root: Path, image: dict, label: str) -> Path:
    filename = image.get("file_name")
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"Missing image file_name in {label}")
    path = (image_root / filename).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Image is not a file: {path}")
    return path


def _inspect_source(name: str, split: str, spec: dict, base: Path) -> SourceSplit:
    label = f"{name}/{split}"
    if not isinstance(spec, dict):
        raise ValueError(f"{label} must be an object")
    _keys(spec, {"annotations", "image_root", "repeat"}, label)
    for field in ("annotations", "image_root"):
        if not isinstance(spec.get(field), str) or not spec[field]:
            raise ValueError(f"{label}.{field} must be an explicit path")
    annotations = (base / spec["annotations"]).resolve(strict=True)
    image_root = (base / spec["image_root"]).resolve(strict=True)
    if not annotations.is_file() or not image_root.is_dir():
        raise ValueError(f"{label} requires an annotation file and an image directory")
    repeat = _integer(spec.get("repeat", 1), f"{label}.repeat", positive=True)
    if split != "train" and repeat != 1:
        raise ValueError(f"Only train can be repeated: {label}")
    payload, annotation_sha = _load_json(annotations)
    categories = payload.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"Missing COCO categories: {label}")
    category_ids: set[int] = set()
    for category in categories:
        if not isinstance(category, dict):
            raise ValueError(f"COCO categories must be objects: {label}")
        category_id = _integer(category.get("id"), f"{label} category id")
        category_name = category.get("name")
        if not isinstance(category_name, str) or category_name.strip().lower() not in {"line", "curve"}:
            raise ValueError(f"Unsupported category {category_name!r} in {label}; only line/curve are allowed")
        if category_id in category_ids:
            raise ValueError(f"Duplicate category id {category_id} in {label}")
        category_ids.add(category_id)

    images = payload.get("images")
    instances = payload.get("annotations")
    if not isinstance(images, list) or not images or not isinstance(instances, list) or not instances:
        raise ValueError(f"Empty or missing images/annotations in declared split {label}")
    image_ids: set[int] = set()
    files: dict[Path, ImageFile] = {}
    inventory_hash = hashlib.sha256()
    for image in images:
        if not isinstance(image, dict):
            raise ValueError(f"COCO images must be objects: {label}")
        image_id = _integer(image.get("id"), f"{label} image id")
        if image_id in image_ids:
            raise ValueError(f"Duplicate image id {image_id} in {label}")
        image_ids.add(image_id)
        for dimension in ("width", "height"):
            _integer(image.get(dimension), f"{label} image {image_id} {dimension}", positive=True)
        path = _image_path(image_root, image, label)
        if path not in files:
            stat = path.stat()
            if stat.st_size == 0:
                raise ValueError(f"Empty image file: {path}")
            suffix = path.suffix.lower()
            if not re.fullmatch(r"\.[a-z0-9]+", suffix):
                raise ValueError(f"Image file needs a conventional extension: {path}")
            file_name = f"{name}/{len(files) + 1:09d}{suffix}"
            files[path] = ImageFile(path, file_name, stat.st_size, stat.st_mtime_ns)
            inventory_hash.update(_json_bytes([str(path), stat.st_size, stat.st_mtime_ns]))
            inventory_hash.update(b"\n")
    annotation_ids: set[int] = set()
    for annotation in instances:
        if not isinstance(annotation, dict):
            raise ValueError(f"COCO annotations must be objects: {label}")
        annotation_id = _integer(annotation.get("id"), f"{label} annotation id")
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation id {annotation_id} in {label}")
        annotation_ids.add(annotation_id)
        referenced_image_id = _integer(annotation.get("image_id"), f"{label} annotation image_id")
        referenced_category_id = _integer(annotation.get("category_id"), f"{label} annotation category_id")
        if referenced_image_id not in image_ids:
            raise ValueError(f"Annotation {annotation_id} references an unknown image in {label}")
        if referenced_category_id not in category_ids:
            raise ValueError(f"Annotation {annotation_id} references an unknown category in {label}")
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, (dict, list)) or not segmentation:
            raise ValueError(f"Annotation {annotation_id} has no COCO instance mask in {label}")
        if "bbox" not in annotation or "area" not in annotation:
            raise ValueError(f"Annotation {annotation_id} lacks bbox/area in {label}")
    licenses = payload.get("licenses", [])
    if not isinstance(licenses, list) or any(not isinstance(record, dict) or "id" not in record for record in licenses):
        raise ValueError(f"Invalid COCO licenses in {label}")
    license_ids = [_integer(record["id"], f"{label} license id") for record in licenses]
    if len(set(license_ids)) != len(license_ids):
        raise ValueError(f"Duplicate license ids in {label}")
    metadata = {
        "source": name, "split": split, "annotations": str(annotations),
        "image_root": str(image_root), "repeat": repeat,
        "annotations_sha256": annotation_sha,
        "image_inventory_sha256": inventory_hash.hexdigest(),
        "image_inventory_fields": ["resolved_path", "size", "mtime_ns"],
        "input_images": len(images), "input_annotations": len(instances),
        "unique_image_files": len(files), "output_images": len(images) * repeat,
        "output_annotations": len(instances) * repeat,
        "original_categories": categories, "original_info": payload.get("info", {}),
        "original_licenses": licenses,
    }
    return SourceSplit(name, split, annotations, image_root, repeat, metadata, files)


def inspect_recipe(recipe_path: Path) -> tuple[dict, list[SourceSplit], str]:
    recipe_path = recipe_path.resolve(strict=True)
    recipe, _ = _load_json(recipe_path)
    _keys(recipe, {"schema_version", "description", "sources"}, "recipe")
    if recipe.get("schema_version", 1) != FORMAT_VERSION:
        raise ValueError("Unsupported recipe schema_version")
    sources = recipe.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Recipe must contain a nonempty sources list")
    names: set[str] = set()
    normalized_sources = []
    inspected: list[SourceSplit] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source must be an object")
        _keys(source, {"name", "splits", "provenance"}, "source")
        name = source.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("Source names must use letters, digits, underscores, or hyphens")
        if name.casefold() in names:
            raise ValueError(f"Duplicate source name: {name}")
        names.add(name.casefold())
        splits = source.get("splits")
        if not isinstance(splits, dict) or "train" not in splits:
            raise ValueError(f"Source {name} requires a train split")
        _keys(splits, set(SPLITS), f"{name}.splits")
        normalized_splits = {}
        for split in SPLITS:
            if split not in splits:
                continue
            item = _inspect_source(name, split, splits[split], recipe_path.parent)
            inspected.append(item)
            normalized_splits[split] = {
                "annotations": str(item.annotations), "image_root": str(item.image_root),
                "repeat": item.repeat,
            }
        normalized_sources.append({
            "name": name, "splits": normalized_splits, "provenance": source.get("provenance", {}),
        })
    for split in SPLITS:
        if not any(item.split == split for item in inspected):
            raise ValueError(f"Aggregate {split} split is empty")
    train_paths = set().union(*(set(item.files) for item in inspected if item.split == "train"))
    for item in inspected:
        if item.split != "train":
            overlap = train_paths.intersection(item.files)
            if overlap:
                example = str(next(iter(overlap)))
                raise ValueError(f"Train/{item.split} image leakage in {item.name}: {example}")
    normalized_recipe = {
        "schema_version": FORMAT_VERSION, "description": recipe.get("description", ""),
        "sources": normalized_sources,
    }
    fingerprint = hashlib.sha256(_json_bytes({
        "recipe": normalized_recipe, "sources": [item.metadata for item in inspected],
    })).hexdigest()
    return normalized_recipe, inspected, fingerprint


@contextmanager
def _lock_output(output: Path) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / LOCK_NAME).open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if not lock.tell():
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"Another process is preparing {output}") from error
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _check_output_identity(output: Path, fingerprint: str) -> None:
    guards = [path for name in (MARKER_NAME, SUMMARY_NAME) if (path := output / name).exists()]
    if not guards:
        if any(path.name != LOCK_NAME for path in output.iterdir()):
            raise ValueError(f"Refusing to overwrite an unrecognized nonempty output directory: {output}")
        return
    for guard in guards:
        payload, _ = _load_json(guard)
        if payload.get("fingerprint") != fingerprint:
            raise ValueError(f"Recipe/source fingerprint changed; use a new output directory: {output}")


def _valid_image(destination: Path, item: ImageFile) -> bool:
    if not destination.is_file():
        return False
    stat = destination.stat()
    return stat.st_size == item.size and stat.st_mtime_ns == item.mtime_ns


def _materialize_image(item: ImageFile, destination: Path) -> None:
    source_stat = item.path.stat()
    if source_stat.st_size != item.size or source_stat.st_mtime_ns != item.mtime_ns:
        raise ValueError(f"Source image changed during preparation: {item.path}")
    if _valid_image(destination, item):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(item.path, temporary)
        except OSError:
            shutil.copy2(item.path, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _unchanged_payload(source: SourceSplit) -> dict:
    payload, actual_sha = _load_json(source.annotations)
    if actual_sha != source.metadata["annotations_sha256"]:
        raise ValueError(f"Source annotations changed during preparation: {source.annotations}")
    return payload


def _ready_complete(output: Path, sources: list[SourceSplit], fingerprint: str) -> dict | None:
    if not (output / SUMMARY_NAME).is_file():
        return None
    summary, _ = _load_json(output / SUMMARY_NAME)
    if summary.get("status") != "ready" or summary.get("fingerprint") != fingerprint:
        return None
    for split in SPLITS:
        target = output / "annotations" / f"instances_{split}.json"
        expected_sha = summary.get("splits", {}).get(split, {}).get("annotations_sha256")
        if not target.is_file() or _sha256_file(target) != expected_sha:
            return None
    for source in sources:
        for item in source.files.values():
            if not _valid_image(output / "images" / source.split / item.file_name, item):
                return None
    return summary


def _write_split(output: Path, split: str, sources: list[SourceSplit], fingerprint: str) -> dict:
    selected = [source for source in sources if source.split == split]
    target = output / "annotations" / f"instances_{split}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    image_count = annotation_count = 0
    licenses = []
    license_maps = {}
    for source in selected:
        mapping = {}
        for license_record in source.metadata["original_licenses"]:
            converted = dict(license_record)
            new_id = len(licenses) + 1
            mapping[license_record["id"]] = new_id
            converted["id"] = new_id
            licenses.append(converted)
        license_maps[source.name] = mapping
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            header = {
                "info": {"description": "LineFormer original sources plus DSC", "mixture_fingerprint": fingerprint},
                "licenses": licenses, "categories": [{"id": 1, "name": "line", "supercategory": "plot"}],
            }
            stream.write(json.dumps(header, ensure_ascii=False, separators=(",", ":"))[:-1])
            stream.write(',"images":[')
            for source in selected:
                payload = _unchanged_payload(source)
                image_file_names = {
                    original["id"]: source.files[_image_path(source.image_root, original, source.name)].file_name
                    for original in payload["images"]
                }
                for item in source.files.values():
                    _materialize_image(item, output / "images" / split / item.file_name)
                for repetition in range(source.repeat):
                    for original in payload["images"]:
                        image_count += 1
                        converted = dict(original)
                        converted.update(
                            id=image_count,
                            file_name=image_file_names[original["id"]],
                            mixture_provenance={"source": source.name, "image_id": original["id"], "repetition": repetition},
                        )
                        if "license" in converted:
                            if converted["license"] in license_maps[source.name]:
                                converted["license"] = license_maps[source.name][converted["license"]]
                            else:
                                converted["mixture_provenance"]["original_license"] = converted.pop("license")
                        if image_count > 1:
                            stream.write(",")
                        stream.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")))
                del payload
                del image_file_names
            stream.write('],"annotations":[')
            image_offset = 0
            for source in selected:
                payload = _unchanged_payload(source)
                original_indices = {image["id"]: index for index, image in enumerate(payload["images"], 1)}
                for repetition in range(source.repeat):
                    for original in payload["annotations"]:
                        annotation_count += 1
                        converted = dict(original)
                        converted.update(
                            id=annotation_count, image_id=image_offset + original_indices[original["image_id"]],
                            category_id=1,
                            mixture_provenance={"source": source.name, "annotation_id": original["id"], "repetition": repetition},
                        )
                        if annotation_count > 1:
                            stream.write(",")
                        stream.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")))
                    image_offset += len(payload["images"])
                del payload
            stream.write("]}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "images": image_count, "annotations": annotation_count,
        "unique_image_files": sum(len(source.files) for source in selected),
        "annotation_file": target.relative_to(output).as_posix(),
        "image_root": f"images/{split}", "annotations_sha256": _sha256_file(target),
    }


def prepare_mixture(recipe_path: Path, output: Path) -> dict:
    recipe_path = recipe_path.resolve(strict=True)
    output = output.resolve()
    normalized_recipe, sources, fingerprint = inspect_recipe(recipe_path)
    # Output cannot own its own inputs, including on a partially prepared rerun.
    for source in sources:
        if output == source.image_root or output in source.image_root.parents:
            raise ValueError(f"Output must not contain a source image directory: {source.image_root}")
        if output in source.annotations.parents:
            raise ValueError(f"Output must not contain source annotations: {source.annotations}")
    with _lock_output(output):
        _check_output_identity(output, fingerprint)
        ready = _ready_complete(output, sources, fingerprint)
        if ready is not None:
            return ready
        _atomic_json(output / MARKER_NAME, {"format_version": FORMAT_VERSION, "fingerprint": fingerprint})
        (output / SUMMARY_NAME).unlink(missing_ok=True)
        results = {split: _write_split(output, split, sources, fingerprint) for split in SPLITS}
        # A ready summary is the commit marker. Recheck inputs so a concurrent
        # source edit cannot turn a partially inconsistent build into a ready one.
        for source in sources:
            if _sha256_file(source.annotations) != source.metadata["annotations_sha256"]:
                raise ValueError(f"Source annotations changed during preparation: {source.annotations}")
            for item in source.files.values():
                stat = item.path.stat()
                if stat.st_size != item.size or stat.st_mtime_ns != item.mtime_ns:
                    raise ValueError(f"Source image changed during preparation: {item.path}")
        summary = {
            "status": "ready", "format_version": FORMAT_VERSION, "fingerprint": fingerprint,
            "recipe_path": str(recipe_path), "recipe": normalized_recipe,
            "sources": [source.metadata for source in sources], "splits": results,
            "mask_processing": "none; source segmentation, bbox, area and iscrowd preserved",
            "category_mapping": {"line": 1, "curve": 1},
            "image_storage": "one file per resolved source image per split; hardlink with copy2 fallback",
            "leakage_check": "resolved input image paths; train disjoint from val/test; val/test overlap allowed",
        }
        _atomic_json(output / SUMMARY_NAME, summary)
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a fingerprinted COCO mixture from explicit source paths")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = prepare_mixture(args.recipe, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Mixture preparation failed: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), "fingerprint": summary["fingerprint"],
                      "splits": summary["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
