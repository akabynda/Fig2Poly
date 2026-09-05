"""CPU checks for source routing; the vendored augmentation itself is unchanged."""

import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from training.train_lineformer import configure_early_stopping, configure_training_pipeline
from training import train_mask2former_lineformer as wrapper
from training import train_lineformer as trainer


ROOT = Path(__file__).resolve().parents[1]


class FakeConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.fixture
def original_config():
    return FakeConfig(**runpy.run_path(str(ROOT / "third_party/LineFormer/lineformer_swin_t_config.py")))


@pytest.fixture
def router_type(monkeypatch):
    registered = {}

    class Registry:
        def register_module(self):
            def register(cls):
                registered[cls.__name__] = cls
                return cls
            return register

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
            self.calls = []
            self.skip = False

        def __call__(self, results):
            self.calls.append(results)
            return None if self.skip else results

    # Fake only the registration/composition boundary. This verifies routing
    # without importing CUDA/MMCV or reimplementing any upstream transforms.
    modules = {name: ModuleType(name) for name in (
        "mmdet", "mmdet.datasets", "mmdet.datasets.builder", "mmdet.datasets.pipelines",
    )}
    modules["mmdet.datasets.builder"].PIPELINES = Registry()
    modules["mmdet.datasets.pipelines"].Compose = Compose
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "source_pipeline_cpu_test", ROOT / "training/lineformer_source_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.SourceAwareLineFormerPipeline
    assert registered == {"SourceAwareLineFormerPipeline": cls}
    return cls


def make_router(router_type, original_config):
    return router_type(original_config.train_pipeline, original_config.train_pipeline_LineEX)


@pytest.mark.parametrize("source,heavy", [
    ("pmc", True), ("PMC", True), ("adobe", True), ("AdobeSynth", True),
    ("lineex", False), ("LineEX", False), ("dsc", False), ("DSC", False),
])
def test_routes_exact_original_pipelines_by_image_provenance(router_type, original_config, source, heavy):
    router = make_router(router_type, original_config)
    payload = {"img_info": {"height": 400, "width": 800, "mixture_provenance": {"source": source}},
               "ann_info": {"masks": [object()]}}
    assert router(payload) is payload
    assert router.pmc_adobe_pipeline.calls == ([payload] if heavy else [])
    assert router.lineex_dsc_pipeline.calls == ([] if heavy else [payload])
    assert router.pmc_adobe_pipeline.transforms == original_config.train_pipeline
    assert router.lineex_dsc_pipeline.transforms == original_config.train_pipeline_LineEX
    assert router.pmc_adobe_pipeline.transforms is not original_config.train_pipeline
    assert router.lineex_dsc_pipeline.transforms is not original_config.train_pipeline_LineEX


@pytest.mark.parametrize("image_info", [
    None, {}, {"mixture_provenance": None}, {"mixture_provenance": "pmc"},
    {"mixture_provenance": {}}, {"mixture_provenance": {"source": 1}},
    {"mixture_provenance": {"source": ""}}, {"mixture_provenance": {"source": "unknown"}},
])
def test_missing_or_unknown_provenance_fails_before_augmentation(router_type, original_config, image_info):
    router = make_router(router_type, original_config)
    with pytest.raises(ValueError, match="mixture_provenance.source"):
        router({"img_info": image_info})
    assert router.pmc_adobe_pipeline.calls == []
    assert router.lineex_dsc_pipeline.calls == []


@pytest.mark.parametrize("source", ["pmc", "adobe"])
def test_small_native_original_image_fails_before_random_shift(router_type, original_config, source):
    router = make_router(router_type, original_config)
    payload = {"img_info": {"height": 50, "width": 800, "file_name": "small.png",
                            "mixture_provenance": {"source": source}}}
    with pytest.raises(ValueError, match="original RandomShift requires both dimensions >= 51"):
        router(payload)
    assert router.pmc_adobe_pipeline.calls == []
    payload["img_info"]["height"] = 51
    assert router(payload) is payload


@pytest.mark.parametrize("source", ["lineex", "dsc"])
def test_no_shift_dimension_restriction_for_light_sources(router_type, original_config, source):
    router = make_router(router_type, original_config)
    payload = {"img_info": {"height": 32, "width": 64, "mixture_provenance": {"source": source}}}
    assert router(payload) is payload


def test_pipeline_skip_is_preserved(router_type, original_config):
    router = make_router(router_type, original_config)
    router.pmc_adobe_pipeline.skip = True
    assert router({"img_info": {"height": 512, "width": 512,
                                "mixture_provenance": {"source": "pmc"}}}) is None


def test_config_preserves_legacy_and_uses_unmodified_source_lists(original_config):
    cfg = original_config
    assert configure_training_pipeline(cfg, SimpleNamespace()) is cfg.train_pipeline_LineEX
    assert cfg.get("custom_imports") is None
    test_pipeline = cfg.test_pipeline
    result = configure_training_pipeline(cfg, SimpleNamespace(original_source_augmentation=True))
    assert result == [{"type": "SourceAwareLineFormerPipeline",
                       "pmc_adobe_pipeline": cfg.train_pipeline,
                       "lineex_dsc_pipeline": cfg.train_pipeline_LineEX}]
    heavy_types = [step["type"] for step in result[0]["pmc_adobe_pipeline"]]
    light_types = [step["type"] for step in result[0]["lineex_dsc_pipeline"]]
    assert heavy_types.index("RandomShift") < heavy_types.index("RandomCrop") < heavy_types.index("Resize")
    assert "RandomShift" not in light_types and "RandomCrop" not in light_types
    assert cfg.test_pipeline is test_pipeline


def test_router_and_early_stopping_keep_existing_imports(original_config):
    cfg = original_config
    cfg.custom_imports = {"imports": ["existing.module"], "allow_failed_imports": False}
    args = SimpleNamespace(original_source_augmentation=True, early_stopping_patience=2,
                           early_stopping_min_delta=0.0, eval_interval=250)
    configure_training_pipeline(cfg, args)
    configure_early_stopping(cfg, args)
    configure_training_pipeline(cfg, args)
    assert cfg.custom_imports == {
        "imports": ["existing.module", "training.lineformer_source_pipeline", "training.lineformer_hooks"],
        "allow_failed_imports": False,
    }
    assert cfg.custom_hooks[0] == {"type": "NumClassCheckHook"}


def test_recipe_enables_router_and_records_truthful_metadata(tmp_path, monkeypatch):
    dataset = tmp_path / "mixture"
    dataset.mkdir()
    (dataset / "mixture_summary.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    received = []
    monkeypatch.setattr(wrapper, "train_lineformer", lambda args: received.append(args) or 0)
    assert wrapper.main(["--lineformer-root", str(tmp_path / "upstream"), "--dataset", str(dataset),
                         "--output", str(output), "--dry-run"]) == 0
    request = json.loads((output / "lineformer_recipe_request.json").read_text(encoding="utf-8"))
    assert received[0].count("--original-source-augmentation") == 1
    assert request["command_arguments"] == received[0]
    assert request["augmentation"] == "PMC/Adobe: original train_pipeline; LineEX/DSC: original train_pipeline_LineEX"
    assert request["augmentation_source_field"] == "img_info.mixture_provenance.source"
    assert "no dilation" in request["mask_processing"]


def test_cli_flag_and_subprocess_can_import_project_pipeline(tmp_path, monkeypatch):
    generated = tmp_path / "generated.py"
    captured = {}

    def build_config(args):
        captured["args"] = args
        return generated

    def launch(command, *, env, check):
        captured["env"] = env
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(trainer, "build_config", build_config)
    monkeypatch.setattr(trainer.subprocess, "run", launch)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "existing"))
    assert trainer.main(["--lineformer-root", str(tmp_path / "upstream"),
                         "--dataset", str(tmp_path / "mixture"), "--output", str(tmp_path / "run"),
                         "--original-source-augmentation", "--num-gpus", "2"]) == 0
    assert captured["args"].original_source_augmentation is True
    paths = captured["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(ROOT) in paths
    assert str(tmp_path / "existing") in paths
