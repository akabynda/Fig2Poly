from pathlib import Path

from training.patch_maskdino_numerics import (
    NEW_CLASS_PROBABILITY,
    OLD_CLASS_PROBABILITY,
    patch_matcher,
)


def test_patch_matcher_is_idempotent(tmp_path: Path) -> None:
    matcher = tmp_path / "maskdino/modeling/matcher.py"
    matcher.parent.mkdir(parents=True)
    matcher.write_text(f"before\n{OLD_CLASS_PROBABILITY}\nafter\n", encoding="utf-8")

    assert patch_matcher(tmp_path) == matcher
    first = matcher.read_text(encoding="utf-8")
    assert NEW_CLASS_PROBABILITY in first
    assert OLD_CLASS_PROBABILITY not in first

    patch_matcher(tmp_path)
    assert matcher.read_text(encoding="utf-8") == first
