from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageFilter


def coco_counts(mask: np.ndarray) -> str:
    """Encode a binary mask as COCO's compressed column-major RLE string."""
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    transitions = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    counts = np.diff(np.concatenate(([0], transitions, [len(pixels)]))).tolist()
    if len(pixels) and pixels[0]:
        counts.insert(0, 0)
    encoded: list[str] = []
    for index, raw in enumerate(counts):
        value = int(raw)
        if index > 2:
            value -= counts[index - 2]
        more = True
        while more:
            char = value & 0x1F
            value >>= 5
            more = value != (-1 if char & 0x10 else 0)
            if more:
                char |= 0x20
            encoded.append(chr(char + 48))
    return "".join(encoded)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def convert_split(source: Path, output: Path, split: str,
                  category_name: str="curve",mask_dilation: int=1) -> dict:
    if not category_name.strip():
        raise ValueError("category_name must not be empty")
    if mask_dilation<1 or mask_dilation%2==0:
        raise ValueError("mask_dilation must be a positive odd integer")
    manifest = source / f"{split}.jsonl"
    images: list[dict] = []
    annotations: list[dict] = []
    image_dir = output / "images" / split
    annotation_id = 1
    with manifest.open("r", encoding="utf-8") as stream:
        for image_id, line in enumerate(stream, 1):
            record = json.loads(line)
            source_image = source / record["image"]
            suffix = source_image.suffix.lower()
            filename = f"{image_id:09d}{suffix}"
            link_or_copy(source_image, image_dir / filename)
            images.append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": int(record["width"]),
                    "height": int(record["height"]),
                    "source_id": str(record["id"]),
                    "dataset_source": record.get("dataset_source", "curveforge"),
                }
            )
            for curve in record.get("curves", []):
                mask_path = source / curve["mask"]
                mask_image=Image.open(mask_path).convert("L")
                if mask_dilation>1:
                    mask_image=mask_image.filter(ImageFilter.MaxFilter(mask_dilation))
                mask = np.asarray(mask_image) > 0
                ys, xs = np.nonzero(mask)
                if not len(xs):
                    continue
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": {
                            "size": [int(mask.shape[0]), int(mask.shape[1])],
                            "counts": coco_counts(mask),
                        },
                        "area": int(mask.sum()),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
    payload = {
        "info": {
            "description": "Fig2Poly visible curve instances",
            "mask_dilation": mask_dilation,
        },
        "licenses": [],
        "categories": [{"id": 1, "name": category_name, "supercategory": "plot"}],
        "images": images,
        "annotations": annotations,
    }
    annotations_dir = output / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    target = annotations_dir / f"instances_{split}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)
    return {"split": split, "images": len(images), "instances": len(annotations),
            "category_name":category_name,"mask_dilation":mask_dilation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Fig2Poly JSONL masks to COCO RLE")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category-name", default="curve")
    parser.add_argument(
        "--train-mask-dilation",type=int,default=1,
        help="Odd dilation kernel used for train masks",
    )
    parser.add_argument(
        "--val-mask-dilation",type=int,default=1,
        help="Odd dilation kernel used for val masks; test always stays exact",
    )
    args = parser.parse_args(argv)
    for name in ("train_mask_dilation", "val_mask_dilation"):
        value = getattr(args, name)
        if value < 1 or value % 2 == 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive odd integer")
    split_dilations = {
        "train": args.train_mask_dilation,
        "val": args.val_mask_dilation,
        "test": 1,
    }
    results = [
        convert_split(
            args.source.resolve(),args.output.resolve(),split,
            category_name=args.category_name,
            mask_dilation=split_dilations[split],
        )
        for split in ("train", "val", "test")
    ]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
