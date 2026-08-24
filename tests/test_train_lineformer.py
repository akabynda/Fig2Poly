import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.train_lineformer import (
    configure_early_stopping,
    configure_logging,
    dataset_config,
    validate_dataset,
)


def make_dataset(root: Path,category: str="line") -> None:
    for split in ("train","val","test"):
        (root/"images"/split).mkdir(parents=True,exist_ok=True)
        annotations=root/"annotations"/f"instances_{split}.json"
        annotations.parent.mkdir(parents=True,exist_ok=True)
        annotations.write_text(json.dumps({"categories":[{"id":1,"name":category}]}))


def test_lineformer_dataset_config_keeps_empty_hard_negatives(tmp_path: Path) -> None:
    config=dataset_config(tmp_path/"instances.json",tmp_path/"images",[{"type":"LoadImageFromFile"}])
    assert config["classes"]==("line",)
    assert config["filter_empty_gt"] is False
    assert config["img_prefix"].endswith(("/","\\"))


def test_validate_lineformer_dataset_requires_line_category(tmp_path: Path) -> None:
    make_dataset(tmp_path,"line")
    validate_dataset(tmp_path)
    make_dataset(tmp_path,"curve")
    with pytest.raises(ValueError,match="categories"):
        validate_dataset(tmp_path)


def test_configure_logging_does_not_require_tensorboard() -> None:
    cfg=SimpleNamespace(log_config=SimpleNamespace(interval=50,hooks=[]))
    configure_logging(cfg,25)
    assert cfg.log_config=={"interval":25,"hooks":[{"type":"TextLoggerHook"}]}


class FakeConfig(SimpleNamespace):
    def get(self,name,default=None):
        return getattr(self,name,default)


def test_configure_lineformer_early_stopping_after_validation() -> None:
    cfg=FakeConfig(custom_hooks=[{"type":"NumClassCheckHook"}])
    args=SimpleNamespace(
        early_stopping_patience=2,
        early_stopping_min_delta=0.001,
        eval_interval=20000,
    )
    configure_early_stopping(cfg,args)
    assert cfg.custom_imports["imports"]==["training.lineformer_hooks"]
    assert cfg.custom_hooks[-1]=={
        "type":"ValidationEarlyStoppingHook",
        "metric":"segm_mAP",
        "interval":20000,
        "patience":2,
        "min_delta":0.001,
        "priority":80,
    }
