from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image
from ultralytics.data.converter import merge_multi_segment


def mask_to_polygon(path: Path, epsilon: float = 0.75, dilate: int = 0) -> np.ndarray | None:
    mask = np.asarray(Image.open(path).convert("L"))
    if dilate > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate((mask > 0).astype(np.uint8), kernel) * 255
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2) for c in contours if len(c) >= 3]
    contours = [c for c in contours if len(c) >= 3]
    if not contours:
        return None
    # Curves originate as y=f(x); even after the generator's modest rotation/perspective,
    # x-centroid order preserves the order of disconnected dash/marker components.
    contours.sort(key=lambda contour: float(contour[:, 0].mean()))
    if len(contours) == 1:
        polygon = contours[0]
    else:
        polygon = np.concatenate(merge_multi_segment([c.reshape(-1).tolist() for c in contours]), axis=0)
    # Remove consecutive duplicates; YOLO needs at least three vertices.
    keep = np.ones(len(polygon), dtype=bool)
    if len(polygon) > 1:
        keep[1:] = np.any(polygon[1:] != polygon[:-1], axis=1)
    polygon = polygon[keep]
    return polygon if len(polygon) >= 3 else None


def convert_one(job: tuple[str, str, str, float, int]) -> dict:
    root_raw, split, stem, epsilon, dilate = job
    root = Path(root_raw)
    mask_dir = root / "curve_masks" / split / stem
    image_path = root / "images" / split / f"{stem}.jpg"
    with Image.open(image_path) as image:
        width, height = image.size
    lines = []
    source_masks = sorted(mask_dir.glob("curve_*.png"))
    for mask_path in source_masks:
        polygon = mask_to_polygon(mask_path, epsilon, dilate)
        if polygon is None:
            continue
        normalized = polygon.astype(np.float64)
        normalized[:, 0] = np.clip(normalized[:, 0] / width, 0, 1)
        normalized[:, 1] = np.clip(normalized[:, 1] / height, 0, 1)
        coords = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
        lines.append(f"0 {coords}")
    label_path = root / "labels" / split / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {"stem": stem, "source_instances": len(source_masks), "labels": len(lines)}


def write_lists(root: Path, subset_train: int, subset_val: int) -> None:
    for split, limit in (("train", subset_train), ("val", subset_val), ("test", 0)):
        images = sorted((root / "images" / split).glob("*.jpg"))
        all_payload = "\n".join(path.resolve().as_posix() for path in images) + "\n"
        (root / f"yolo_{split}.txt").write_text(all_payload, encoding="utf-8")
        if limit:
            subset = images[:limit]
            payload = "\n".join(path.resolve().as_posix() for path in subset) + "\n"
            (root / f"yolo_{split}_{limit}.txt").write_text(payload, encoding="utf-8")


def write_yamls(root: Path, subset_train: int, subset_val: int) -> None:
    base = root.resolve().as_posix()
    full = (
        f"path: {base}\n"
        "train: yolo_train.txt\n"
        "val: yolo_val.txt\n"
        "test: yolo_test.txt\n"
        "names:\n"
        "  0: curve\n"
    )
    (root / "curve_yolo.yaml").write_text(full, encoding="utf-8")
    if subset_train and subset_val:
        subset = (
            f"path: {base}\n"
            f"train: yolo_train_{subset_train}.txt\n"
            f"val: yolo_val_{subset_val}.txt\n"
            "test: yolo_test.txt\n"
            "names:\n"
            "  0: curve\n"
        )
        (root / "curve_yolo_subset.yaml").write_text(subset, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert CurveForge masks to YOLO instance polygons")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=.75)
    parser.add_argument("--dilate", type=int, default=0,
                        help="Elliptical dilation kernel size; use 9 for centerline-oriented YOLO targets")
    parser.add_argument("--subset-train", type=int, default=10000)
    parser.add_argument("--subset-val", type=int, default=2000)
    args = parser.parse_args(argv)
    root = args.dataset.resolve()
    jobs = []
    for split in ("train", "val", "test"):
        for image in sorted((root / "images" / split).glob("*.jpg")):
            jobs.append((str(root), split, image.stem, args.epsilon, args.dilate))
    totals = {"images": 0, "source_instances": 0, "labels": 0, "dropped": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(convert_one, jobs, chunksize=max(1, len(jobs)//(args.workers*20))), 1):
            totals["images"] += 1
            totals["source_instances"] += result["source_instances"]
            totals["labels"] += result["labels"]
            totals["dropped"] += result["source_instances"] - result["labels"]
            if index % 5000 == 0:
                print(f"converted {index}/{len(jobs)}", flush=True)
    write_lists(root, args.subset_train, args.subset_val)
    write_yamls(root, args.subset_train, args.subset_val)
    totals["dilate_kernel"] = args.dilate
    (root / "yolo_conversion.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
