from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge instance datasets without duplicating files")
    parser.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    sources = []
    for value in args.source:
        name, separator, raw_path = value.partition("=")
        if not separator or not name:
            parser.error(f"invalid --source {value!r}; expected NAME=PATH")
        sources.append((name, Path(raw_path).resolve()))

    summary = {}
    for split in ("train", "val", "test"):
        records = []
        for source_name, root in sources:
            manifest = root / f"{split}.jsonl"
            if not manifest.is_file():
                continue
            with manifest.open("r", encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    sample_id = f"{source_name}__{record['id']}"
                    source_image = root / record["image"]
                    image_relative = Path("images") / split / f"{sample_id}{source_image.suffix.lower()}"
                    link_or_copy(source_image, output / image_relative)
                    curves = []
                    for index, curve in enumerate(record.get("curves", []), 1):
                        source_mask = root / curve["mask"]
                        mask_relative = (
                            Path("curve_masks") / split / sample_id / f"curve_{index:03d}.png"
                        )
                        link_or_copy(source_mask, output / mask_relative)
                        converted = dict(curve)
                        converted["mask"] = mask_relative.as_posix()
                        curves.append(converted)
                    converted_record = dict(record)
                    converted_record.update(
                        id=sample_id,
                        image=image_relative.as_posix(),
                        curves=curves,
                        curve_count=len(curves),
                        dataset_source=record.get("dataset_source", source_name),
                    )
                    records.append(converted_record)
        target = output / f"{split}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, target)
        summary[split] = len(records)
    (output / "merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
