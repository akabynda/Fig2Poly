from __future__ import annotations

import argparse
from pathlib import Path


OLD_CLASS_PROBABILITY = 'out_prob = outputs["pred_logits"][b].sigmoid()'
NEW_CLASS_PROBABILITY = """# Compute focal matching costs in FP32. With AMP, 1e-8 rounds to
            # zero in FP16 and saturated sigmoid values otherwise produce 0 * log(0) = NaN.
            out_prob = (
                outputs["pred_logits"][b]
                .float()
                .sigmoid()
                .clamp(min=1e-6, max=1.0 - 1e-6)
            )"""


def patch_matcher(maskdino_root: Path) -> Path:
    matcher = maskdino_root.resolve() / "maskdino/modeling/matcher.py"
    text = matcher.read_text(encoding="utf-8")
    if NEW_CLASS_PROBABILITY in text:
        return matcher
    count = text.count(OLD_CLASS_PROBABILITY)
    if count != 1:
        raise RuntimeError(
            f"Expected one MaskDINO class-probability pattern, found {count}"
        )
    matcher.write_text(
        text.replace(OLD_CLASS_PROBABILITY, NEW_CLASS_PROBABILITY, 1),
        encoding="utf-8",
    )
    return matcher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Make MaskDINO Hungarian focal costs safe under FP16 AMP"
    )
    parser.add_argument("--maskdino-root", type=Path, required=True)
    args = parser.parse_args(argv)
    matcher = patch_matcher(args.maskdino_root)
    print(f"MaskDINO FP16-safe matcher ready: {matcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
