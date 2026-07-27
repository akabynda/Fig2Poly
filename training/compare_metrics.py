from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten_metrics(path: Path) -> tuple[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model", path.parent.name)
    metrics = payload.get("metrics", payload)
    flattened = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            flattened[key] = float(value)
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, (int, float)):
                    flattened[f"{key}/{nested_key}"] = float(nested_value)
    return model, flattened


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine framework evaluation JSON files")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    models = dict(flatten_metrics(path) for path in args.reports if path.is_file())
    keys = sorted({key for metrics in models.values() for key in metrics})
    payload = {"models": models, "metric_keys": keys}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
