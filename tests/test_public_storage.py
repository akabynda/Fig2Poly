import io
import json
from pathlib import Path
import tarfile

from training.convert_public_curvequery import eligible_chartinfo_stems
from training.download_public_benchmarks import safe_extract


def annotation(chart_type: str, with_curve: bool = True) -> dict:
    lines = [[{"x": 1, "y": 2}, {"x": 3, "y": 4}]] if with_curve else []
    return {
        "task1": {"output": {"chart_type": chart_type}},
        "task6": {"output": {"visual elements": {"lines": lines}}},
    }


def test_adobe_selection_and_filtered_extraction(tmp_path: Path) -> None:
    annotations = tmp_path / "json_gt"
    annotations.mkdir()
    (annotations / "10.json").write_text(
        json.dumps(annotation("Line")), encoding="utf-8"
    )
    (annotations / "11.json").write_text(
        json.dumps(annotation("Grouped bar")), encoding="utf-8"
    )
    (annotations / "12.json").write_text(
        json.dumps(annotation("Scatter-line")), encoding="utf-8"
    )
    (annotations / "13.json").write_text(
        json.dumps(annotation("Line", with_curve=False)), encoding="utf-8"
    )
    selected = eligible_chartinfo_stems(annotations)
    assert selected == {"10", "12"}

    archive = tmp_path / "images_a.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in ("images/10.png", "images/11.png", "images/12.png", "images/._10.png"):
            payload = name.encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))

    output = tmp_path / "raw"
    extracted, skipped = safe_extract(
        archive,
        output,
        lambda name: Path(name).stem in selected and not Path(name).name.startswith("._"),
    )
    assert extracted == 2
    assert skipped == 2
    assert (output / "images" / "10.png").is_file()
    assert (output / "images" / "12.png").is_file()
    assert not (output / "images" / "11.png").exists()
    assert not (output / "images" / "._10.png").exists()
