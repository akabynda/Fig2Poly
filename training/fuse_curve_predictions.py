from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from training.predict_lineformer_panels import PALETTE, mask_centerline


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def centerline_distance(first: dict, second: dict) -> dict[str, float]:
    """Compare two curves without making line thickness part of the decision."""
    x1, y1 = first["centerline"]
    x2, y2 = second["centerline"]
    common = np.intersect1d(x1, x2, assume_unique=True)
    overlap = len(common) / max(1, min(len(x1), len(x2)))
    if not len(common):
        return {"overlap": 0.0, "median": float("inf"), "p90": float("inf")}
    distances = np.abs(np.interp(common, x1, y1) - np.interp(common, x2, y2))
    return {
        "overlap": float(overlap),
        "median": float(np.median(distances)),
        "p90": float(np.percentile(distances, 90)),
    }


def curves_match(first: dict, second: dict, image_height: int) -> tuple[bool, dict[str, float]]:
    metrics = centerline_distance(first, second)
    median_limit = max(4.0, image_height * 0.012)
    p90_limit = max(9.0, image_height * 0.04)
    matched = (
        first["panel"] == second["panel"]
        and metrics["overlap"] >= 0.5
        and metrics["median"] <= median_limit
        and metrics["p90"] <= p90_limit
    )
    return matched, {**metrics, "median_limit": median_limit, "p90_limit": p90_limit}


def panel_is_crowded(tracks: list[dict], image_height: int) -> tuple[bool, dict | None]:
    """Detect parallel/nearby curves for which query separation matters most."""
    limit = max(7.0, image_height * 0.025)
    closest: dict | None = None
    for index, first in enumerate(tracks):
        for second in tracks[index + 1:]:
            metrics = centerline_distance(first, second)
            if metrics["overlap"] < 0.55:
                continue
            item = {"first": first["id"], "second": second["id"], **metrics, "limit": limit}
            if closest is None or item["median"] < closest["median"]:
                closest = item
    return bool(closest and closest["median"] <= limit), closest


def guided_union(primary: np.ndarray, secondary: np.ndarray, image_height: int) -> np.ndarray:
    """Add only secondary pixels close to the selected identity backbone."""
    radius = max(3.0, image_height * 0.006)
    distance = cv2.distanceTransform((~primary).astype(np.uint8), cv2.DIST_L2, 5)
    return primary | (secondary & (distance <= radius))


def _prepare(item: dict) -> dict:
    prepared = {**item}
    prepared["centerline"] = mask_centerline(prepared["mask"])
    return prepared


def fuse_panel(
    maskdino: list[dict], lineformer: list[dict], image_height: int
) -> tuple[list[dict], dict]:
    """Select identities from the suitable expert, then safely enrich their masks."""
    crowded, crowd_metrics = panel_is_crowded(maskdino, image_height)
    if crowded and maskdino:
        preferred_name, preferred, secondary = "maskdino", maskdino, lineformer
    elif lineformer:
        preferred_name, preferred, secondary = "lineformer", lineformer, maskdino
    else:
        preferred_name, preferred, secondary = "maskdino", maskdino, lineformer

    matches: dict[int, list[tuple[int, dict[str, float]]]] = defaultdict(list)
    reverse_matches: dict[int, list[int]] = defaultdict(list)
    for primary_index, primary in enumerate(preferred):
        for secondary_index, other in enumerate(secondary):
            matched, metrics = curves_match(primary, other, image_height)
            if matched:
                matches[primary_index].append((secondary_index, metrics))
                reverse_matches[secondary_index].append(primary_index)

    fused: list[dict] = []
    used_secondary: set[int] = set()
    decisions: list[dict] = []
    for primary_index, primary in enumerate(preferred):
        candidates = sorted(matches[primary_index], key=lambda item: item[1]["median"])
        unique = [item for item in candidates if len(reverse_matches[item[0]]) == 1]
        result = {**primary, "source": preferred_name, "reason": f"{preferred_name}_backbone"}
        if unique:
            secondary_index, metrics = unique[0]
            other = secondary[secondary_index]
            result["mask"] = guided_union(primary["mask"], other["mask"], image_height)
            result["centerline"] = mask_centerline(result["mask"])
            result["source"] = f"{preferred_name}+{other['source']}"
            result["reason"] = "unique_cross_model_match"
            used_secondary.add(secondary_index)
            decisions.append({
                "primary_id": primary["id"], "secondary_id": other["id"],
                "action": "guided_union", **metrics,
            })
        elif candidates:
            decisions.append({
                "primary_id": primary["id"], "action": "keep_identity_ambiguous_match",
                "candidate_ids": [secondary[index]["id"] for index, _ in candidates],
            })
        fused.append(result)

    # In crowded panels MaskDINO owns identities, but a confident LineFormer curve
    # that is far from every MaskDINO centerline can recover an outright miss.
    if preferred_name == "maskdino":
        for secondary_index, candidate in enumerate(secondary):
            if secondary_index in used_secondary or candidate["score"] < 0.5:
                continue
            x_values, _ = candidate["centerline"]
            horizontal_coverage = (
                (float(x_values[-1] - x_values[0] + 1) / candidate["mask"].shape[1])
                if len(x_values) else 0.0
            )
            if horizontal_coverage < 0.25:
                decisions.append({
                    "secondary_id": candidate["id"], "action": "reject_short_fragment",
                    "horizontal_coverage": horizontal_coverage,
                })
                continue
            relations = [centerline_distance(candidate, item) for item in preferred]
            near_existing = any(
                relation["overlap"] >= 0.35
                and relation["median"] <= max(8.0, image_height * 0.04)
                for relation in relations
            )
            if not near_existing:
                fused.append({
                    **candidate,
                    "source": "lineformer",
                    "reason": "confident_unmatched_lineformer",
                })
                decisions.append({
                    "secondary_id": candidate["id"], "action": "add_unmatched_lineformer"
                })

    return fused, {
        "preferred_expert": preferred_name,
        "crowded": crowded,
        "closest_maskdino_pair": crowd_metrics,
        "decisions": decisions,
    }


def add_complex_lineformer_rescues(
    fused: list[dict], high_confidence: list[dict], low_confidence: list[dict],
    maskdino: list[dict], image_height: int,
) -> tuple[list[dict], list[dict]]:
    """Recover low-score, peak-rich LineFormer curves without admitting flat clutter."""
    rescued: list[dict] = []
    diagnostics: list[dict] = []
    for candidate in low_confidence:
        known = [
            item for item in high_confidence
            if item["panel"] == candidate["panel"]
        ]
        already_high = any(
            centerline_distance(candidate, item)["overlap"] >= 0.5
            and centerline_distance(candidate, item)["median"] <= max(5.0, image_height * 0.02)
            for item in known
        )
        if already_high:
            continue
        x_values, y_values = candidate["centerline"]
        coverage = (
            float(x_values[-1] - x_values[0] + 1) / candidate["mask"].shape[1]
            if len(x_values) else 0.0
        )
        if len(y_values):
            smooth = cv2.GaussianBlur(y_values.astype(np.float32).reshape(1, -1), (0, 0), 3).ravel()
            vertical_excursion = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
        else:
            vertical_excursion = 0.0
        duplicate = any(
            item["panel"] == candidate["panel"]
            and centerline_distance(candidate, item)["overlap"] >= 0.35
            and centerline_distance(candidate, item)["median"] <= max(8.0, image_height * 0.04)
            for item in fused
        )
        accepted = (
            not duplicate
            and coverage >= 0.25
            and vertical_excursion >= max(6.0, image_height * 0.006)
        )
        maskdino_matches = []
        if not duplicate:
            for maskdino_candidate in maskdino:
                matched, metrics = curves_match(candidate, maskdino_candidate, image_height)
                maskdino_duplicate = any(
                    item["panel"] == maskdino_candidate["panel"]
                    and centerline_distance(maskdino_candidate, item)["overlap"] >= 0.5
                    and centerline_distance(maskdino_candidate, item)["median"]
                    <= max(5.0, image_height * 0.02)
                    for item in fused
                )
                if matched and not maskdino_duplicate:
                    maskdino_matches.append((maskdino_candidate, metrics))
        # Cross-model agreement is stronger evidence than morphology alone. Use
        # MaskDINO as the backbone here so a peak absent from LineFormer survives.
        cross_model_match = min(
            maskdino_matches, key=lambda item: item[1]["median"], default=None
        )
        accepted = accepted or cross_model_match is not None
        diagnostics.append({
            "candidate_id": candidate["id"], "panel": candidate["panel"],
            "score": candidate["score"], "horizontal_coverage": coverage,
            "vertical_excursion": vertical_excursion, "duplicate": duplicate,
            "action": (
                "cross_model_rescue" if cross_model_match is not None
                else "rescue" if accepted else "reject"
            ),
        })
        if accepted:
            if cross_model_match is not None:
                maskdino_candidate, _ = cross_model_match
                rescued.append({
                    **maskdino_candidate,
                    "mask": guided_union(
                        maskdino_candidate["mask"], candidate["mask"], image_height
                    ),
                    "source": "maskdino+lineformer",
                    "reason": "low_confidence_cross_model_rescue",
                })
            else:
                rescued.append({
                    **candidate,
                    "source": "lineformer",
                    "reason": "low_confidence_complex_lineformer_rescue",
                })
    return fused + rescued, diagnostics


def load_tracks(root: Path, stem: str, threshold_dir: str, source: str) -> list[dict]:
    directory = root / stem / threshold_dir
    records = json.loads((directory / "predictions.json").read_text(encoding="utf-8"))
    tracks = []
    for record in records:
        mask = cv2.imread(str(directory / record["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Unable to read {directory / record['mask']}")
        tracks.append(_prepare({
            "id": int(record["id"]),
            "score": float(record["score"]),
            "panel": int(record["panel"]),
            "bbox": record["bbox_xyxy"],
            "mask": mask > 0,
            "source": source,
        }))
    return tracks


def render(image: np.ndarray, tracks: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for stale_mask in output.glob("mask_*.png"):
        stale_mask.unlink()
    overlay = image.astype(np.float32)
    curves_only = np.full_like(image, 255)
    instance_ids = np.zeros(image.shape[:2], dtype=np.uint16)
    records = []
    annotations = []
    for index, track in enumerate(sorted(tracks, key=lambda item: (item["panel"], -item["score"])), 1):
        mask = track["mask"]
        color = np.asarray(PALETTE[(index - 1) % len(PALETTE)], dtype=np.float32)
        overlay[mask] = overlay[mask] * 0.52 + color * 0.48
        curves_only[mask] = color.astype(np.uint8)
        instance_ids[mask] = index
        mask_name = f"mask_{index:03d}.png"
        cv2.imwrite(str(output / mask_name), mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ys, xs = np.nonzero(mask)
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        annotations.append((index, track["source"], bbox, contours, color.astype(np.uint8).tolist()))
        records.append({
            "id": index, "score": track["score"], "panel": track["panel"],
            "source": track["source"], "reason": track["reason"],
            "bbox_xyxy": bbox, "mask": mask_name,
        })
    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
    for index, source, bbox, contours, color in annotations:
        cv2.drawContours(overlay_u8, contours, -1, color, 1, cv2.LINE_AA)
        cv2.putText(
            overlay_u8, f"{index}: {source}", (int(bbox[0]), max(12, int(bbox[1]))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(output / "overlay.png"), overlay_u8)
    cv2.imwrite(str(output / "curves_only.png"), curves_only)
    cv2.imwrite(str(output / "instance_ids.png"), instance_ids)
    (output / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuse MaskDINO and LineFormer curve instances")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--maskdino-results", type=Path, required=True)
    parser.add_argument("--lineformer-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maskdino-threshold", default="threshold_0.30")
    parser.add_argument("--lineformer-threshold", default="threshold_0.30")
    parser.add_argument("--lineformer-rescue-threshold", default="threshold_0.15")
    args = parser.parse_args(argv)

    summary = []
    for image_path in sorted(path for path in args.images.resolve().iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read {image_path}")
        maskdino = load_tracks(args.maskdino_results.resolve(), image_path.stem, args.maskdino_threshold, "maskdino")
        lineformer = load_tracks(args.lineformer_results.resolve(), image_path.stem, args.lineformer_threshold, "lineformer")
        lineformer_low = load_tracks(
            args.lineformer_results.resolve(), image_path.stem,
            args.lineformer_rescue_threshold, "lineformer",
        )
        panels = sorted({item["panel"] for item in maskdino + lineformer})
        fused: list[dict] = []
        diagnostics = []
        for panel in panels:
            panel_tracks, panel_diagnostics = fuse_panel(
                [item for item in maskdino if item["panel"] == panel],
                [item for item in lineformer if item["panel"] == panel],
                image.shape[0],
            )
            fused.extend(panel_tracks)
            diagnostics.append({"panel": panel, **panel_diagnostics})
        fused, rescue_diagnostics = add_complex_lineformer_rescues(
            fused, lineformer, lineformer_low, maskdino, image.shape[0]
        )
        image_output = args.output.resolve() / image_path.stem
        render(image, fused, image_output)
        (image_output / "fusion_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        summary.append({
            "image": image_path.name, "maskdino": len(maskdino),
            "lineformer": len(lineformer), "fused": len(fused), "panels": diagnostics,
            "lineformer_rescues": rescue_diagnostics,
        })
        print(f"{image_path.name}: MaskDINO={len(maskdino)}, LineFormer={len(lineformer)}, fused={len(fused)}")
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    (args.output.resolve() / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
