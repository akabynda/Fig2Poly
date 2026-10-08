import ast
from datetime import timedelta
from contextlib import contextmanager
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.train_maskdino import (
    install_training_profile, parse_args, profile_checkpoint_hooks,
    retain_periodic_checkpoints, training_options,
)


BASE_ARGS = [
    "--maskdino-root", "upstream", "--dataset", "dataset", "--output", "output",
    "--variant", "r50", "--weights", "imagenet_r50.pkl",
]


def resolve(*extra, images=12345):
    args = parse_args(BASE_ARGS + list(extra))
    opts, summary = training_options(args, images)
    return dict(zip(opts[::2], opts[1::2])), summary


def test_lineformer_budget_and_lr_do_not_scale_with_dataset_or_gpu_count():
    opts, summary = resolve("--training-profile", "lineformer", "--num-gpus", "2")
    larger_opts, larger_summary = resolve(
        "--training-profile", "lineformer", "--num-gpus", "4", images=500000
    )
    assert summary["global_batch"] == 8
    assert larger_summary["global_batch"] == 16
    assert summary["images_per_gpu"] == larger_summary["images_per_gpu"] == 4
    for key in ("SOLVER.BASE_LR", "SOLVER.MAX_ITER", "SOLVER.STEPS"):
        assert opts[key] == larger_opts[key]
    assert float(opts["SOLVER.BASE_LR"]) == 1e-4
    assert int(opts["SOLVER.MAX_ITER"]) == 100000
    milestones = ast.literal_eval(opts["SOLVER.STEPS"])
    assert milestones == tuple(range(5000, 100000, 5000))
    gamma = float(opts["SOLVER.GAMMA"])
    def lr_at(iteration):
        return float(opts["SOLVER.BASE_LR"]) * gamma ** sum(step <= iteration for step in milestones)
    assert lr_at(4999) == 1e-4
    assert lr_at(5000) == pytest.approx(7.5e-5)
    assert lr_at(10000) == pytest.approx(5.625e-5)
    assert opts["SOLVER.REFERENCE_WORLD_SIZE"] == "0"
    assert summary["log_interval"] == 100
    assert opts["SOLVER.CHECKPOINT_PERIOD"] == "500"
    assert opts["TEST.EVAL_PERIOD"] == "250"
    assert opts["SOLVER.CLIP_GRADIENTS.CLIP_TYPE"] == "full_model"
    assert opts["SOLVER.CLIP_GRADIENTS.CLIP_VALUE"] == "0.01"
    assert opts["INPUT.IMAGE_SIZE"] == opts["INPUT.MIN_SIZE_TEST"] == opts["INPUT.MAX_SIZE_TEST"] == "512"
    assert opts["MODEL.MaskDINO.NUM_OBJECT_QUERIES"] == "100"
    assert opts["SOLVER.AMP.ENABLED"] == "False"


def test_legacy_schedule_keeps_previous_defaults():
    opts, summary = resolve(images=1000)
    assert summary["global_batch"] == 2
    assert summary["max_iter"] == 25000
    assert summary["steps_per_epoch"] == 500
    assert summary["eval_period"] == 500
    assert summary["checkpoint_period"] == 125
    assert summary["base_lr"] == 1e-4 * 2 / 16
    assert summary["image_size"] == 1024
    assert ast.literal_eval(opts["SOLVER.STEPS"]) == (22250, 24000)
    assert "MODEL.MaskDINO.NUM_OBJECT_QUERIES" not in opts
    assert "SOLVER.GAMMA" not in opts
    assert summary["log_interval"] is None


def test_explicit_smoke_overrides_do_not_leave_out_of_budget_lr_steps():
    opts, summary = resolve(
        "--training-profile", "lineformer", "--num-gpus", "2", "--max-iter", "2",
        "--global-batch", "2", "--eval-period", "0", "--checkpoint-period", "1",
    )
    assert opts["SOLVER.STEPS"] == "()"
    assert summary["max_iter"] == 2
    assert summary["global_batch"] == 2
    assert summary["eval_period"] == 0
    assert summary["iteration_budget_source"] == "max_iter"


@pytest.mark.parametrize("extra", [
    ["--early-stopping-patience", "2"],
    ["--curve-loss-weight", "0.1"],
    ["--global-batch", "3", "--num-gpus", "2"],
    ["--num-gpus", "0"],
    ["--max-iter", "0"],
    ["--base-lr", "0"],
    ["--distributed-timeout-minutes", "0"],
    ["--distributed-timeout-minutes", "-1"],
])
def test_invalid_profile_combinations_fail_before_importing_cuda(extra):
    with pytest.raises(SystemExit):
        parse_args(BASE_ARGS + ["--training-profile", "lineformer"] + extra)


def test_timeout_reaches_distributed_launcher_without_changing_training_recipe(monkeypatch, tmp_path):
    from training import train_maskdino

    default_opts, default_summary = resolve("--training-profile", "lineformer", "--num-gpus", "2")
    changed_opts, changed_summary = resolve(
        "--training-profile", "lineformer", "--num-gpus", "2",
        "--distributed-timeout-minutes", "180",
    )
    assert changed_opts == default_opts
    assert changed_summary == default_summary

    upstream = tmp_path / "upstream"
    config = upstream / train_maskdino.CONFIGS["r50"]
    config.parent.mkdir(parents=True)
    config.write_text("fixture")
    dataset = tmp_path / "dataset"
    (dataset / "annotations").mkdir(parents=True)
    (dataset / "annotations/instances_train.json").write_text(json.dumps({"images": [{"id": 1}]}))
    output = tmp_path / "output"
    parsed_commands = []
    launches = []

    def parse_upstream(command):
        parsed_commands.append(command)
        return SimpleNamespace(dist_url="auto")

    monkeypatch.setattr("training.patch_maskdino_numerics.patch_matcher", lambda root: None)
    monkeypatch.setattr("training.patch_maskdino_rle.main", lambda args: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "train_net", SimpleNamespace(
        default_argument_parser=lambda: SimpleNamespace(parse_args=parse_upstream)))
    monkeypatch.setitem(sys.modules, "detectron2.engine", SimpleNamespace(
        launch=lambda *args, **kwargs: launches.append((args, kwargs))))

    assert train_maskdino.main([
        "--maskdino-root", str(upstream), "--dataset", str(dataset),
        "--output", str(output), "--variant", "r50", "--weights", str(tmp_path / "unused.pkl"),
        "--training-profile", "lineformer", "--num-gpus", "2", "--resume",
        "--distributed-timeout-minutes", "180",
    ]) == 0
    assert launches[0][1]["timeout"] == timedelta(hours=3)
    assert launches[0][0][1] == 2
    assert "--resume" in parsed_commands[0]
    recorded = json.loads((output / "training_profile.json").read_text())
    assert recorded["distributed_timeout_minutes"] == 180
    assert recorded["resume"] is True


def test_profile_runtime_preserves_hooks_and_exempts_standalone_level_embedding(monkeypatch):
    monkeypatch.setattr("training.train_maskdino.profile_checkpoint_hooks", lambda configured, trainer: configured)
    class PeriodicWriter:
        _period = 20

    writer = PeriodicWriter()
    unrelated_hook = object()
    level_embedding = object()
    convolution = object()
    optimizer = SimpleNamespace(param_groups=[
        {"params": [level_embedding], "weight_decay": 0.05},
        {"params": [convolution], "weight_decay": 0.05},
    ])

    class UpstreamTrainer:
        def build_hooks(self):
            return [unrelated_hook, writer]

        @classmethod
        def build_optimizer(cls, cfg, model):
            return optimizer

    monkeypatch.setitem(sys.modules, "detectron2.engine", SimpleNamespace(hooks=SimpleNamespace(PeriodicWriter=PeriodicWriter)))
    train_net = SimpleNamespace(Trainer=UpstreamTrainer)
    model = SimpleNamespace(named_parameters=lambda: [
        ("sem_seg_head.pixel_decoder.level_embed", level_embedding),
        ("backbone.layer1.conv1.weight", convolution),
    ])
    install_training_profile(train_net, "lineformer", 100)
    assert train_net.Trainer().build_hooks() == [unrelated_hook, writer]
    assert writer._period == 100
    result = train_net.Trainer.build_optimizer(None, model)
    assert result is optimizer
    assert result.param_groups[0]["weight_decay"] == 0
    assert result.param_groups[1]["weight_decay"] == 0.05


def test_legacy_does_not_replace_upstream_trainer():
    upstream = object()
    train_net = SimpleNamespace(Trainer=upstream)
    install_training_profile(train_net, "legacy", None)
    assert train_net.Trainer is upstream


def test_retention_reloads_previous_run_without_deleting_best_final_or_other_files(tmp_path):
    retained = ["model_0001499.pth", "model_0001999.pth", "model_0002499.pth"]
    protected = ["model_best.pth", "model_final.pth", "model_manual.pth", "initial_weights.pth"]
    removed = ["model_0000499.pth", "model_0000999.pth"]
    for name in removed + retained + protected:
        (tmp_path / name).write_bytes(b"checkpoint")
    assert [Path(path).name for path in retain_periodic_checkpoints(tmp_path)] == retained
    assert {path.name for path in tmp_path.iterdir()} == set(retained + protected)


def test_best_checkpoint_survives_resume_and_does_not_redirect_latest(monkeypatch, tmp_path):
    saves = []

    class Checkpointer:
        def __init__(self, model, output, **kwargs):
            self.output = Path(output)

        def tag_last_checkpoint(self, name):
            (self.output / "last_checkpoint").write_text(name)

        def save(self, name, **kwargs):
            saves.append((name, kwargs))
            (self.output / f"{name}.pth").write_bytes(b"checkpoint")
            self.tag_last_checkpoint(f"{name}.pth")

    class PeriodicCheckpointer:
        def __init__(self, checkpointer=None, period=500, max_to_keep=None):
            self.checkpointer = checkpointer
            self.period = period
            self.max_to_keep = max_to_keep

        def before_train(self):
            self.max_iter = self.trainer.max_iter

    class EvalHook:
        pass

    hook_types = SimpleNamespace(PeriodicCheckpointer=PeriodicCheckpointer, EvalHook=EvalHook, HookBase=object)
    monkeypatch.setitem(sys.modules, "detectron2.engine", SimpleNamespace(hooks=hook_types))
    monkeypatch.setitem(sys.modules, "detectron2.checkpoint", SimpleNamespace(DetectionCheckpointer=Checkpointer))
    monkeypatch.setitem(sys.modules, "detectron2.utils", SimpleNamespace(comm=SimpleNamespace(is_main_process=lambda: True)))
    metric = {"segm/AP": (42.0, 249)}
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(
            OUTPUT_DIR=str(tmp_path), SOLVER=SimpleNamespace(CHECKPOINT_PERIOD=500),
            TEST=SimpleNamespace(EVAL_PERIOD=250), DATASETS=SimpleNamespace(TEST=("val",)),
        ),
        iter=249, max_iter=100000, model=object(), checkpointer=object(),
        storage=SimpleNamespace(latest=lambda: metric),
    )
    (tmp_path / "last_checkpoint").write_text("model_0000499.pth")
    periodic, evaluation, best = profile_checkpoint_hooks([PeriodicCheckpointer(), EvalHook()], trainer)
    assert isinstance(evaluation, EvalHook)
    assert periodic.max_to_keep == 3
    best.trainer = trainer
    best.before_train()
    best.after_step()
    assert saves == [("model_best", {"iteration": 249, "best_metric": "segm/AP", "best_score": 42.0})]
    assert (tmp_path / "last_checkpoint").read_text() == "model_0000499.pth"
    metadata = json.loads((tmp_path / "best_checkpoint.json").read_text())
    assert metadata["completed_iterations"] == 250

    # Resume from a periodic checkpoint whose hook state predates the best save.
    _, restored = profile_checkpoint_hooks([EvalHook()], trainer)
    restored.trainer = trainer
    restored.before_train()
    assert restored.best_score == 42.0
    trainer.iter = 749
    metric["segm/AP"] = (41.0, 749)
    restored.after_step()
    assert len(saves) == 1
    trainer.iter = 999
    metric["segm/AP"] = (43.0, 999)
    restored.after_step()
    assert len(saves) == 2
    assert restored.state_dict() == {"best_score": 43.0, "best_iteration": 999}
    assert (tmp_path / "last_checkpoint").read_text() == "model_0000499.pth"


@pytest.mark.parametrize("main_rank", [True, False])
def test_interrupted_validation_is_recovered_before_training_and_recorded_at_checkpoint_iteration(
    monkeypatch, tmp_path, main_rank,
):
    from training.train_maskdino import recover_interrupted_validation

    events = []
    trainer = SimpleNamespace(
        start_iter=53500, max_iter=100000,
        cfg=SimpleNamespace(OUTPUT_DIR=str(tmp_path), TEST=SimpleNamespace(EVAL_PERIOD=250),
                            DATASETS=SimpleNamespace(TEST=("val",))),
    )

    @contextmanager
    def storage(iteration):
        values = {}
        yield SimpleNamespace(iter=iteration, latest=lambda: values)

    class Evaluation:
        def _do_eval(self):
            events.append("evaluation")
            if main_rank:
                trainer.storage.latest()["segm/AP"] = (3.1, trainer.storage.iter)

    class Best:
        def before_train(self):
            events.append("restore_best")

        def save_if_improved(self):
            events.append(("select_best", trainer.storage.latest()["segm/AP"]))

    trainer._hooks = [Evaluation(), Best()] if main_rank else [Evaluation()]
    monkeypatch.setitem(sys.modules, "detectron2.engine", SimpleNamespace(
        hooks=SimpleNamespace(EvalHook=Evaluation)))
    monkeypatch.setitem(sys.modules, "detectron2.utils", SimpleNamespace(comm=SimpleNamespace(
        is_main_process=lambda: main_rank, synchronize=lambda: events.append("barrier"))))
    monkeypatch.setitem(sys.modules, "detectron2.utils.events", SimpleNamespace(EventStorage=storage))
    assert recover_interrupted_validation(trainer) is True
    if main_rank:
        assert events == ["evaluation", "restore_best", ("select_best", (3.1, 53499)), "barrier"]
        row = json.loads((tmp_path / "metrics.json").read_text())
        assert row == {"segm/AP": 3.1, "iteration": 53499}
        assert json.loads((tmp_path / "resume_validation_53500.json").read_text()) == row
        assert recover_interrupted_validation(trainer) is False
    else:
        assert events == ["evaluation", "barrier"]
        assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize("start_iter", [0, 53501, 100000])
def test_resume_validation_skips_non_boundaries_or_finished_training(tmp_path, start_iter):
    from training.train_maskdino import recover_interrupted_validation

    trainer = SimpleNamespace(
        start_iter=start_iter, max_iter=100000,
        cfg=SimpleNamespace(OUTPUT_DIR=str(tmp_path), TEST=SimpleNamespace(EVAL_PERIOD=250),
                            DATASETS=SimpleNamespace(TEST=("val",))),
    )
    assert recover_interrupted_validation(trainer) is False
