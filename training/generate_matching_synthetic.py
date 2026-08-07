from __future__ import annotations

import argparse
import json
from pathlib import Path

from curveforge.config import GeneratorConfig
from curveforge.generator import DatasetGenerator


def manifest_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic splits matching a public instance dataset"
    )
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    counts = {
        split: manifest_count(args.public / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    total = sum(counts.values())
    if not total or not counts["train"]:
        parser.error(f"public manifests are empty: {counts}")
    config = GeneratorConfig.from_json(args.config)
    result = DatasetGenerator(config).generate(
        args.output,
        total,
        val_fraction=counts["val"] / total,
        test_fraction=counts["test"] / total,
        workers=args.workers,
        resume=args.resume,
    )
    if result["splits"] != counts:
        raise RuntimeError(
            f"generated splits do not match public splits: {result['splits']} != {counts}"
        )
    print(json.dumps({"public": counts, "synthetic": result["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
