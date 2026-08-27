from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.benchmark_inference import parse_dataset
from training.benchmark_predictions import PredictionStore, iter_manifest, threshold_predictions
from training.evaluate_curve_benchmark import matched_metrics, summarize, write_csv


MODELS = ("classic", "finetuned")


def model_variants(models: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        variant
        for model in models
        for variant in (f"{model}_paper", f"{model}_panel_post")
    )


VARIANTS = model_variants(MODELS)


def article_summary(rows: list[dict], variants: tuple[str, ...] = VARIANTS) -> list[dict]:
    """Keep the two metrics reported as the main LineFormer paper results."""
    result = []
    for row in summarize(rows, variants=variants):
        if not row["variant"].endswith("_paper"):
            continue
        result.append({
            "dataset": row["dataset"],
            "model": row["variant"].removesuffix("_paper"),
            "images": row["images"],
            "failures": row["failures"],
            "task_6a_mean": row["score_6a_mean"],
            "task_6a_std": row["score_6a_std"],
            "task_6b_mean": row["score_6b_mean"],
            "task_6b_std": row["score_6b_std"],
            "score_threshold": row["score_threshold_mean"],
        })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare classic and fine-tuned LineFormer with paper Task 6a/6b metrics"
    )
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--classic-db", type=Path, required=True)
    parser.add_argument("--finetuned-db", type=Path, required=True)
    parser.add_argument("--classic-label", default="classic")
    parser.add_argument("--finetuned-label", default="finetuned")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args(argv)

    for _, _, kind in args.dataset:
        if kind != "manifest":
            parser.error("Task 6a/6b evaluation requires manifests with ground-truth curves")

    models = (args.classic_label.strip(), args.finetuned_label.strip())
    if not all(models) or len(set(models)) != 2:
        parser.error("model labels must be non-empty and distinct")
    variants = model_variants(models)

    stores = {
        models[0]: PredictionStore(args.classic_db, writable=False),
        models[1]: PredictionStore(args.finetuned_db, writable=False),
    }
    rows: list[dict] = []
    dataset_counts: dict[str, int] = {}
    try:
        for dataset, manifest, _ in args.dataset:
            for sample in iter_manifest(manifest):
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
                sample_id = str(sample["id"])
                for model_name, store in stores.items():
                    raw, raw_meta = store.get(dataset, sample_id, "raw", model_name)
                    processed, processed_meta = store.get(dataset, sample_id, "processed", model_name)
                    height = int(raw_meta["height"])
                    width = int(raw_meta["width"])
                    variants = {
                        # This is the paper-compatible path: whole-image inference
                        # and the score threshold used by LineFormer's infer.py.
                        f"{model_name}_paper": (
                            raw,
                            raw_meta["raw_inference_seconds"],
                            0.0,
                            raw_meta["error"],
                        ),
                        # Kept as a supplemental diagnostic, not a paper score.
                        f"{model_name}_panel_post": (
                            processed,
                            processed_meta["processed_inference_seconds"],
                            processed_meta["postprocess_seconds"],
                            processed_meta["error"],
                        ),
                    }
                    for variant, (predictions, inference_seconds, post_seconds, error) in variants.items():
                        predictions = threshold_predictions(predictions, args.threshold)
                        metrics = matched_metrics(predictions, sample, height, width)
                        rows.append({
                            "dataset": dataset,
                            "split": sample.get("official_source_split", "test"),
                            "image_id": sample_id,
                            "image": sample.get("image"),
                            "variant": variant,
                            "width": width,
                            "height": height,
                            "score_threshold": args.threshold,
                            "inference_seconds": inference_seconds,
                            "postprocess_seconds": post_seconds,
                            "total_seconds": inference_seconds + post_seconds,
                            "error": error,
                            **metrics,
                        })
            print(f"evaluated {dataset}: {dataset_counts[dataset]} images", flush=True)
    finally:
        for store in stores.values():
            store.close()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = summarize(rows, variants=variants)
    write_csv(output / "lineformer_article_metrics.csv", article_summary(rows, variants))
    write_csv(output / "summary_metrics.csv", summaries)
    write_csv(output / "per_image_metrics.csv", rows)
    (output / "protocol.json").write_text(json.dumps({
        "dataset_image_counts": dataset_counts,
        "score_threshold": args.threshold,
        "primary_variants": [f"{model}_paper" for model in models],
        "supplemental_variants": [f"{model}_panel_post" for model in models],
        "primary_metrics": {
            "task_6a": "LineFormer/ChartInfo continuous-line score normalized by ground-truth curve count",
            "task_6b": "The same continuous-line score with extra or missing curves penalized",
        },
        "source": "Lal et al., LineFormer, ICDAR 2023, DOI 10.1007/978-3-031-41734-4_24",
        "note": "The generated DSC test split is not one of the paper datasets; metric definitions are identical, scores are not directly comparable to the paper table.",
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
