from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from training.convert_public_curvequery import convert_chartinfo, convert_lineex
from training.curvequery_mamba import TrainConfig, train
from training.evaluate_curvequery_mamba import evaluate


def log(path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def download_complete(manifest: Path) -> bool:
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return "completed" in value


def manifest_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finish P2 conversion and fine-tuning after background downloads")
    parser.add_argument("--public-root", default="datasets/public")
    parser.add_argument("--normalized", default="datasets/public_curvequery")
    parser.add_argument("--synthetic-run", default="runs/curvequery_mamba_p2_synthetic")
    parser.add_argument("--output", default="runs/curvequery_mamba_p2_public")
    parser.add_argument(
        "--lineex-converter-marker",
        default="datasets/public_curvequery/lineex_train_converter.done",
    )
    args = parser.parse_args(argv)

    public_root = Path(args.public_root).resolve()
    normalized = Path(args.normalized).resolve()
    synthetic_run = Path(args.synthetic_run).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "pipeline.log"
    stop_path = output / "STOP_PIPELINE"
    lineex_converter_marker = Path(args.lineex_converter_marker).resolve()

    try:
        log(log_path, "waiting for verified public downloads")
        while not download_complete(public_root / "download_manifest.json"):
            if stop_path.exists():
                log(log_path, "pipeline stopped before conversion")
                return 0
            time.sleep(20)
        log(log_path, "public downloads complete; waiting for active LineEX converter")
        while not lineex_converter_marker.is_file():
            if stop_path.exists():
                log(log_path, "pipeline stopped while waiting for LineEX conversion")
                return 0
            time.sleep(20)
        log(log_path, "active LineEX converter ended; verifying/resuming idempotently")
        result = convert_lineex(
            public_root / "raw" / "lineex" / "train",
            normalized,
            "train",
            line_width=3,
            limit=None,
        )
        log(log_path, f"LineEX train converted: {json.dumps(result)}")

        adobe_root = public_root / "raw" / "adobe_synth19"
        log(log_path, "converting AdobeSynth19 official train with deterministic 10% validation")
        result = convert_chartinfo(
            adobe_root,
            normalized,
            "adobe_synth19",
            "train",
            validation_fraction=0.1,
            line_width=3,
            limit=None,
            annotation_root=adobe_root / "json_gt",
        )
        log(log_path, f"AdobeSynth19 train converted: {json.dumps(result)}")

        # The Adobe release stores train and test under one extraction root.
        # Idempotent ids prevent any train image from being appended to test.
        log(log_path, "converting AdobeSynth19 official test without training leakage")
        result = convert_chartinfo(
            adobe_root / "test_release" / "task6",
            normalized,
            "adobe_synth19_test",
            "test",
            validation_fraction=0.0,
            line_width=3,
            limit=None,
            annotation_root=adobe_root / "test_release" / "task6" / "gt_json",
        )
        log(log_path, f"AdobeSynth19 test converted: {json.dumps(result)}")

        summary = {
            split: manifest_count(normalized / f"{split}.jsonl")
            for split in ("train", "val", "test")
        }
        (normalized / "conversion_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        log(log_path, f"normalized public manifests: {json.dumps(summary)}")

        synthetic_final = synthetic_run / "final.pt"
        log(log_path, f"waiting for synthetic pretrain: {synthetic_final}")
        while not synthetic_final.is_file():
            if stop_path.exists():
                log(log_path, "pipeline stopped before public fine-tuning")
                return 0
            time.sleep(20)

        log(log_path, "starting P2 public fine-tuning from synthetic weights")
        final = train(
            TrainConfig(
                dataset=str(normalized),
                output=str(output),
                init_checkpoint=str(synthetic_final),
                decoder="mamba",
                width=512,
                height=384,
                batch_size=1,
                accumulation_steps=4,
                epochs=1,
                learning_rate=2e-5,
                backbone_learning_rate=5e-6,
                weight_decay=0.05,
                warmup_steps=500,
                save_every=500,
                keep_checkpoints=2,
                validate_every=5000,
                val_limit=500,
                num_workers=2,
                seed=42,
            )
        )
        log(log_path, f"P2 training complete: {final}")
        val_report = evaluate(
            Path(final),
            normalized,
            "val",
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            512,
            384,
            3,
            None,
            output / "val_metrics.json",
            20,
        )
        threshold = float(val_report["best_threshold"])
        log(log_path, f"validation threshold selected without test access: {threshold}")
        evaluate(
            Path(final),
            normalized,
            "test",
            [threshold],
            512,
            384,
            3,
            None,
            output / "test_metrics.json",
            40,
        )
        log(log_path, "P2 public test evaluation complete")
        return 0
    except Exception:
        log(log_path, traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
