from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import ijson
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def stable_validation(name: str, fraction: float) -> bool:
    value = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
    return value / 2**64 < fraction


def normalized_chart_type(annotation: dict) -> str | None:
    chart_type = ((annotation.get("task1") or {}).get("output") or {}).get("chart_type")
    if chart_type is None:
        return None
    return str(chart_type).strip().lower().replace("_", " ").replace("-", " ")


def is_chartinfo_line(annotation: dict, chart_types: set[str] | None = None) -> bool:
    chart_type = normalized_chart_type(annotation)
    accepted = chart_types if chart_types is not None else {"line", "scatter line"}
    return (chart_type is None or chart_type in accepted) and bool(chartinfo_lines(annotation))


def eligible_chartinfo_stems(annotation_root: Path) -> set[str]:
    result = set()
    for path in annotation_root.rglob("*.json"):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        if is_chartinfo_line(annotation):
            result.add(path.stem)
    return result


def find_chartinfo_pairs(
    raw_root: Path,
    annotation_path_contains: str | None = None,
    annotation_root: Path | None = None,
    chart_types: set[str] | None = None,
) -> list[tuple[Path, Path]]:
    if annotation_root is not None:
        search_root = annotation_root
    else:
        search_root = raw_root / "final_full_GT" if (raw_root / "final_full_GT").is_dir() else raw_root
    if annotation_root is not None:
        all_annotations = list(search_root.rglob("*.json"))
    elif annotation_path_contains:
        token = annotation_path_contains.lower()
        all_annotations = [
            path for path in search_root.rglob("*.json") if token in str(path).lower()
        ]
    else:
        all_annotations = [
            path for path in search_root.rglob("*.json")
            if "annotations_JSON" in path.parts
        ]
    typed_line = [path for path in all_annotations if path.parent.name == "line"]
    annotations = sorted(typed_line or all_annotations)
    eligible_annotations = []
    for path in annotations:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        if is_chartinfo_line(annotation, chart_types):
            eligible_annotations.append(path)
    required = {path.stem for path in eligible_annotations}
    image_index: dict[str, Path] = {}
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem in required:
            image_index.setdefault(path.stem, path)
    missing = required - image_index.keys()
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        raise RuntimeError(
            f"Missing {len(missing)} eligible line-chart images in {raw_root}; examples: {examples}"
        )
    pairs = [(annotation, image_index[annotation.stem]) for annotation in eligible_annotations]
    if not pairs:
        raise RuntimeError(f"No line-chart image/JSON pairs found in {raw_root}")
    return pairs


def polygon_points(value: dict) -> list[tuple[float, float]]:
    if not isinstance(value, dict):
        return []
    points = []
    for index in range(16):
        x_key, y_key = f"x{index}", f"y{index}"
        if x_key not in value or y_key not in value:
            break
        points.append((float(value[x_key]), float(value[y_key])))
    return points


def chartinfo_lines(annotation: dict) -> list[list[tuple[float, float]]]:
    task6 = annotation.get("task6") or {}
    visual = (task6.get("output") or {}).get("visual elements", {})
    raw_lines = visual.get("lines", [])
    result = []
    for line in raw_lines:
        points = []
        for point in line:
            if isinstance(point, dict) and "x" in point and "y" in point:
                points.append((float(point["x"]), float(point["y"])))
        if len(points) >= 2:
            result.append(points)
    return result


def chartinfo_occlusion_mask(annotation: dict, size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    task2 = (annotation.get("task2") or {}).get("output") or {}
    text_blocks = task2.get("text_blocks", [])
    text_by_id = {item.get("id"): item for item in text_blocks}
    for block in text_blocks:
        points = polygon_points(block.get("polygon", {}))
        if len(points) >= 3:
            draw.polygon(points, fill=255)
        box = block.get("bb", {})
        if {"x0", "y0", "width", "height"} <= box.keys():
            draw.rectangle(
                (
                    float(box["x0"]),
                    float(box["y0"]),
                    float(box["x0"] + box["width"]),
                    float(box["y0"] + box["height"]),
                ),
                fill=255,
            )

    # Task 5 exposes marker boxes plus the linked text id. Their padded hull is
    # a conservative estimate of an opaque legend panel. It is only removed
    # when it overlaps the annotated plot area.
    legend_boxes = []
    task5_output = (annotation.get("task5") or {}).get("output") or {}
    for pair in task5_output.get("legend_pairs", []):
        box = pair.get("bb", {})
        if {"x0", "y0", "width", "height"} <= box.keys():
            legend_boxes.append(
                (
                    float(box["x0"]),
                    float(box["y0"]),
                    float(box["x0"] + box["width"]),
                    float(box["y0"] + box["height"]),
                )
            )
        linked = text_by_id.get(pair.get("id"))
        points = polygon_points(linked.get("polygon", {})) if linked else []
        if points:
            legend_boxes.append(
                (
                    min(point[0] for point in points),
                    min(point[1] for point in points),
                    max(point[0] for point in points),
                    max(point[1] for point in points),
                )
            )
        linked_box = linked.get("bb", {}) if linked else {}
        if {"x0", "y0", "width", "height"} <= linked_box.keys():
            legend_boxes.append(
                (
                    float(linked_box["x0"]),
                    float(linked_box["y0"]),
                    float(linked_box["x0"] + linked_box["width"]),
                    float(linked_box["y0"] + linked_box["height"]),
                )
            )
    task4_output = (annotation.get("task4") or {}).get("output") or {}
    plot = task4_output.get("_plot_bb", {})
    if legend_boxes and {"x0", "y0", "width", "height"} <= plot.keys():
        padding = 6
        hull = (
            min(box[0] for box in legend_boxes) - padding,
            min(box[1] for box in legend_boxes) - padding,
            max(box[2] for box in legend_boxes) + padding,
            max(box[3] for box in legend_boxes) + padding,
        )
        plot_box = (
            float(plot["x0"]),
            float(plot["y0"]),
            float(plot["x0"] + plot["width"]),
            float(plot["y0"] + plot["height"]),
        )
        intersects = not (
            hull[2] < plot_box[0] or hull[0] > plot_box[2]
            or hull[3] < plot_box[1] or hull[1] > plot_box[3]
        )
        if intersects:
            draw.rectangle(hull, fill=255)
    return mask


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def existing_sample_ids(output: Path) -> set[str]:
    result = set()
    for split in ("train", "val", "test"):
        manifest = output / f"{split}.jsonl"
        if not manifest.is_file():
            continue
        with manifest.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    result.add(str(json.loads(line)["id"]))
    return result


def existing_split_count(output: Path, split: str, dataset_source: str) -> int:
    manifest = output / f"{split}.jsonl"
    if not manifest.is_file():
        return 0
    count = 0
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("dataset_source") == dataset_source:
                count += 1
    return count


def convert_chartinfo(
    raw_root: Path,
    output: Path,
    dataset_name: str,
    official_split: str,
    validation_fraction: float,
    line_width: int,
    limit: int | None,
    annotation_path_contains: str | None = None,
    annotation_root: Path | None = None,
    chart_types: set[str] | None = None,
) -> dict:
    pairs = find_chartinfo_pairs(
        raw_root, annotation_path_contains, annotation_root, chart_types
    )
    if limit is not None:
        pairs = pairs[:limit]
    existing = existing_sample_ids(output)
    counts = {"train": 0, "val": 0, "test": 0, "curves": 0, "skipped": 0}
    counts[official_split] = existing_split_count(output, official_split, dataset_name)
    for annotation_path, image_path in pairs:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        lines = chartinfo_lines(annotation)
        if not lines:
            counts["skipped"] += 1
            continue
        split = official_split
        if official_split == "train" and validation_fraction > 0:
            split = "val" if stable_validation(annotation_path.stem, validation_fraction) else "train"
        sample_id = f"{dataset_name}__{annotation_path.stem}"
        if sample_id in existing:
            continue
        image_suffix = image_path.suffix.lower()
        relative_image = Path("images") / split / f"{sample_id}{image_suffix}"
        destination_image = output / relative_image
        link_or_copy(image_path, destination_image)
        with Image.open(image_path) as opened:
            size = opened.size
        occlusions = chartinfo_occlusion_mask(annotation, size)
        curve_records = []
        mask_dir = output / "curve_masks" / split / sample_id
        mask_dir.mkdir(parents=True, exist_ok=True)
        for index, points in enumerate(lines, 1):
            mask = Image.new("L", size, 0)
            ImageDraw.Draw(mask).line(points, fill=255, width=line_width, joint="curve")
            # Public annotations contain polylines, not true visible-pixel
            # masks. Removing annotated text/legend pixels keeps the same
            # visible-only convention as CurveForge v4.
            visible = Image.eval(occlusions, lambda value: 255 - value)
            mask = Image.composite(mask, Image.new("L", size, 0), visible)
            bbox = mask.getbbox()
            if bbox is None:
                continue
            mask_path = mask_dir / f"curve_{index:03d}.png"
            mask.save(mask_path, compress_level=1)
            curve_records.append(
                {
                    "id": index,
                    "mask": mask_path.relative_to(output).as_posix(),
                    "bbox": list(bbox),
                    "area": int(np_count(mask)),
                    "source_points": [{"x": x, "y": y} for x, y in points],
                }
            )
        if not curve_records:
            counts["skipped"] += 1
            continue
        record = {
            "id": sample_id,
            "split": split,
            "dataset_source": dataset_name,
            "official_source_split": official_split,
            "image": relative_image.as_posix(),
            "width": size[0],
            "height": size[1],
            "curve_count": len(curve_records),
            "mask_semantics": "3px polyline proxy, annotated text/legend occlusions removed",
            "annotation": str(annotation_path.resolve()),
            "curves": curve_records,
        }
        append_manifest(output / f"{split}.jsonl", record)
        existing.add(sample_id)
        counts[split] += 1
        counts["curves"] += len(curve_records)
    return counts


def convert_lineex(
    raw_split_root: Path,
    output: Path,
    official_split: str,
    line_width: int,
    limit: int | None,
) -> dict:
    line_path = raw_split_root / "anno" / "line_anno.json"
    class_path = raw_split_root / "anno" / "cls_anno.json"
    with line_path.open("rb") as stream:
        images = {
            int(item["id"]): {
                "file_name": item["file_name"],
                "width": int(item["width"]),
                "height": int(item["height"]),
            }
            for item in ijson.items(stream, "images.item")
        }
    # The released annotation file uses category_id=0 for the full legend
    # rectangle (the category names in its categories array are shifted).
    legends_by_image: dict[int, list[list[float]]] = {}
    with class_path.open("rb") as stream:
        for item in ijson.items(stream, "annotations.item"):
            if int(item["category_id"]) == 0:
                legends_by_image.setdefault(int(item["image_id"]), []).append(
                    [float(value) for value in item["bbox"]]
                )

    existing = existing_sample_ids(output)
    counts = {"train": 0, "val": 0, "test": 0, "curves": 0, "skipped": 0}
    counts[official_split] = existing_split_count(output, official_split, "lineex")

    def convert_group(image_id: int, annotations: list[dict]) -> None:
        if limit is not None and counts[official_split] + counts["skipped"] >= limit:
            return
        info = images[image_id]
        source_image = raw_split_root / "images" / info["file_name"]
        if not source_image.is_file():
            counts["skipped"] += 1
            return
        sample_id = f"lineex__{official_split}__{image_id}"
        if sample_id in existing:
            return
        relative_image = Path("images") / official_split / f"{sample_id}{source_image.suffix.lower()}"
        link_or_copy(source_image, output / relative_image)
        size = (int(info["width"]), int(info["height"]))
        occlusions = Image.new("L", size, 0)
        occ_draw = ImageDraw.Draw(occlusions)
        for x, y, width, height in legends_by_image.get(image_id, []):
            occ_draw.rectangle((x, y, x + width, y + height), fill=255)
        visible = Image.eval(occlusions, lambda value: 255 - value)

        curves = []
        mask_dir = output / "curve_masks" / official_split / sample_id
        mask_dir.mkdir(parents=True, exist_ok=True)
        annotations.sort(key=lambda item: int(item["id"]))
        for curve_index, annotation in enumerate(annotations, 1):
            flat = annotation.get("bbox", [])
            points = [(float(flat[index]), float(flat[index + 1])) for index in range(0, len(flat) - 1, 2)]
            if len(points) < 2:
                continue
            mask = Image.new("L", size, 0)
            ImageDraw.Draw(mask).line(points, fill=255, width=line_width, joint="curve")
            mask = Image.composite(mask, Image.new("L", size, 0), visible)
            bbox = mask.getbbox()
            if bbox is None:
                continue
            mask_path = mask_dir / f"curve_{curve_index:03d}.png"
            mask.save(mask_path, compress_level=1)
            curves.append(
                {
                    "id": curve_index,
                    "mask": mask_path.relative_to(output).as_posix(),
                    "bbox": list(bbox),
                    "area": int(np_count(mask)),
                    "source_points": [{"x": x, "y": y} for x, y in points],
                }
            )
        if not curves:
            counts["skipped"] += 1
            return
        record = {
            "id": sample_id,
            "split": official_split,
            "dataset_source": "lineex",
            "official_source_split": official_split,
            "image": relative_image.as_posix(),
            "width": size[0],
            "height": size[1],
            "curve_count": len(curves),
            "mask_semantics": "3px polyline proxy, released legend rectangle removed",
            "annotation": str((raw_split_root / "anno" / "line_anno.json").resolve()),
            "curves": curves,
        }
        append_manifest(output / f"{official_split}.jsonl", record)
        existing.add(sample_id)
        counts[official_split] += 1
        counts["curves"] += len(curves)

    current_id: int | None = None
    group: list[dict] = []
    previous_id = -1
    with line_path.open("rb") as stream:
        for annotation in ijson.items(stream, "annotations.item"):
            image_id = int(annotation["image_id"])
            if image_id < previous_id:
                raise RuntimeError("LineEX annotations are not grouped by image_id")
            previous_id = image_id
            if current_id is None:
                current_id = image_id
            if image_id != current_id:
                convert_group(current_id, group)
                if limit is not None and counts[official_split] + counts["skipped"] >= limit:
                    break
                current_id, group = image_id, []
            group.append(annotation)
        else:
            if current_id is not None:
                convert_group(current_id, group)
    return counts


def np_count(image: Image.Image) -> int:
    # Avoid importing NumPy in this conversion-only utility.
    histogram = image.histogram()
    return sum(histogram[1:])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert official chart benchmarks to CurveForge manifests")
    parser.add_argument("--format", choices=("chartinfo", "lineex"), default="chartinfo")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", default="datasets/public_instances")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--official-split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--annotation-path-contains")
    parser.add_argument("--annotation-root")
    parser.add_argument(
        "--chart-types",
        help="Comma-separated normalized ChartInfo types; default: line and scatter-line",
    )
    args = parser.parse_args(argv)
    if args.format == "lineex":
        result = convert_lineex(
            Path(args.raw_root).resolve(),
            Path(args.output).resolve(),
            args.official_split,
            args.line_width,
            args.limit,
        )
    else:
        result = convert_chartinfo(
            Path(args.raw_root).resolve(),
            Path(args.output).resolve(),
            args.dataset_name,
            args.official_split,
            args.validation_fraction,
            args.line_width,
            args.limit,
            args.annotation_path_contains,
            Path(args.annotation_root).resolve() if args.annotation_root else None,
            ({item.strip().lower().replace("-", " ") for item in args.chart_types.split(",")}
             if args.chart_types else None),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
