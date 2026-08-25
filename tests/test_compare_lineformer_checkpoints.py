import pytest

from training.compare_lineformer_checkpoints import article_summary


def test_article_summary_only_contains_paper_protocol_variants():
    rows = []
    for variant, score in (
        ("classic_paper", 0.4),
        ("classic_panel_post", 0.5),
        ("finetuned_paper", 0.8),
        ("finetuned_panel_post", 0.9),
    ):
        rows.append({
            "dataset": "dsc_test",
            "split": "test",
            "image_id": "1",
            "image": "1.png",
            "variant": variant,
            "error": None,
            "score_6a": score,
            "score_6b": score - 0.1,
            "score_threshold": 0.3,
        })

    summary = article_summary(rows)

    assert [item["model"] for item in summary] == ["classic", "finetuned"]
    assert summary[0]["task_6a_mean"] == 0.4
    assert summary[1]["task_6b_mean"] == pytest.approx(0.7)
    assert all(item["score_threshold"] == 0.3 for item in summary)
