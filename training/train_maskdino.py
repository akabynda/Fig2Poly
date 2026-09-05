from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys


CONFIGS = {
    "r50": "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
    "swin_l": "configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
}


def training_options(args, train_images: int) -> tuple[list[str], dict]:
    """Resolve the optional LineFormer schedule without importing CUDA libraries."""
    lineformer = args.training_profile == "lineformer"
    global_batch = args.global_batch
    if global_batch is None:
        global_batch = 4 * args.num_gpus if lineformer else 2
    image_size = args.image_size
    if image_size is None:
        image_size = 512 if lineformer else 1024
    steps_per_epoch = math.ceil(train_images / global_batch)
    max_iter = args.max_iter
    if max_iter is None:
        max_iter = 100000 if lineformer else max(1, steps_per_epoch * args.epochs)
    checkpoint_period = args.checkpoint_period
    if checkpoint_period is None:
        checkpoint_period = 500 if lineformer else max(100, steps_per_epoch // 4)
    eval_period = args.eval_period
    if eval_period is None:
        eval_period = 250 if lineformer else steps_per_epoch
    learning_rate = args.base_lr
    if learning_rate is None:
        learning_rate = 1e-4 if lineformer else 1e-4 * global_batch / 16
    log_interval = args.log_interval
    if log_interval is None and lineformer:
        log_interval = 100
    if lineformer:
        milestones = list(range(5000, max_iter, 5000))
    else:
        milestones = sorted({int(max_iter * .89), int(max_iter * .96)})
        milestones = [step for step in milestones if 0 < step < max_iter]
        if max_iter <= 2:
            milestones = []
    solver_steps = repr(tuple(milestones))
    opts = [
        "SOLVER.IMS_PER_BATCH", str(global_batch),
        "SOLVER.BASE_LR", str(learning_rate),
        "SOLVER.AMP.ENABLED", str(args.amp),
        "SOLVER.MAX_ITER", str(max_iter),
        "SOLVER.STEPS", solver_steps,
        "SOLVER.CHECKPOINT_PERIOD", str(checkpoint_period),
        "TEST.EVAL_PERIOD", str(eval_period),
        "INPUT.IMAGE_SIZE", str(image_size),
        "INPUT.MIN_SCALE", "1.0" if lineformer else "0.5",
        "INPUT.MAX_SCALE", "1.0" if lineformer else "1.5",
    ]
    if lineformer:
        opts.extend([
            "SEED", str(getattr(args, "seed", None) if getattr(args, "seed", None) is not None else 20260905),
            "MODEL.MaskDINO.NUM_OBJECT_QUERIES", "100",
            "SOLVER.OPTIMIZER", "ADAMW",
            "SOLVER.WEIGHT_DECAY", "0.05",
            "SOLVER.WEIGHT_DECAY_NORM", "0.0",
            "SOLVER.WEIGHT_DECAY_EMBED", "0.0",
            "SOLVER.BACKBONE_MULTIPLIER", "0.2",
            "SOLVER.REFERENCE_WORLD_SIZE", "0",
            "SOLVER.LR_SCHEDULER_NAME", "WarmupMultiStepLR",
            "SOLVER.GAMMA", "0.75",
            "SOLVER.WARMUP_FACTOR", "1.0",
            "SOLVER.WARMUP_ITERS", "10",
            "SOLVER.WARMUP_METHOD", "linear",
            "SOLVER.CLIP_GRADIENTS.ENABLED", "True",
            "SOLVER.CLIP_GRADIENTS.CLIP_TYPE", "full_model",
            "SOLVER.CLIP_GRADIENTS.CLIP_VALUE", "0.01",
            "SOLVER.CLIP_GRADIENTS.NORM_TYPE", "2.0",
            "INPUT.FORMAT", "RGB",
            "INPUT.MIN_SIZE_TEST", str(image_size),
            "INPUT.MAX_SIZE_TEST", str(image_size),
            "TEST.DETECTIONS_PER_IMAGE", "100",
        ])
    summary = {
        "training_profile": args.training_profile,
        "train_images": train_images,
        "epoch_based": False,
        "iteration_budget_source": "profile" if lineformer else "epochs",
        "steps_per_epoch": steps_per_epoch,
        "max_iter": max_iter,
        "checkpoint_period": checkpoint_period,
        "eval_period": eval_period,
        "log_interval": log_interval,
        "solver_steps": milestones,
        "global_batch": global_batch,
        "images_per_gpu": global_batch // args.num_gpus,
        "base_lr": learning_rate,
        "image_size": image_size,
    }
    if args.max_iter is not None:
        summary["iteration_budget_source"] = "max_iter"
    if lineformer:
        summary.update({
            "optimizer": "AdamW",
            "weight_decay": 0.05,
            "backbone_lr_multiplier": 0.2,
            "lr_gamma": 0.75,
            "lr_step": 5000,
            "gradient_clip_norm": 0.01,
            "num_queries": 100,
            "auto_scale_lr": False,
            "best_checkpoint_metric": "segm/AP",
            "max_keep_checkpoints": 3,
            "augmentation": f"LineFormer source-specific geometry: PMC/Adobe flip + shift(0.3, dx/dy -51..50) + absolute crop(0.3, 435x435); LineEX/DSC flip; all fit and white-pad {image_size}",
            "augmentation_source": "parent directory of prepared image path: pmc/adobe/lineex/dsc",
            "mask_dilation": 0,
            "architecture_differences": "MaskDINO R50 losses, denoising and box refinement retained; boxes recomputed and wholly clipped instances removed for box losses; LineFormer ignores boxes and retains empty masks",
        })
    return opts, summary


def retain_periodic_checkpoints(output: Path, keep: int = 3) -> list[str]:
    """Restore the retention queue after resume, excluding best/final artifacts."""
    root = output.resolve()
    candidates = []
    for path in root.glob("model_*.pth"):
        match = re.fullmatch(r"model_(\d+)\.pth", path.name)
        if match and path.is_file():
            if path.resolve().parent != root:
                raise ValueError(f"Checkpoint resolves outside output directory: {path}")
            candidates.append((int(match.group(1)), path))
    candidates.sort(key=lambda item: item[0])
    for _, path in candidates[:-keep]:
        path.unlink()
    return [str(path) for _, path in candidates[-keep:]]


def profile_checkpoint_hooks(configured, trainer):
    """Keep periodic checkpoints bounded and save the best model independently."""
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.engine import hooks
    from detectron2.utils import comm

    class RetainedPeriodicCheckpointer(hooks.PeriodicCheckpointer):
        def before_train(self):
            super().before_train()
            self.recent_checkpoints = retain_periodic_checkpoints(
                Path(self.trainer.cfg.OUTPUT_DIR), self.max_to_keep
            )

    class BestOnlyCheckpointer(DetectionCheckpointer):
        def tag_last_checkpoint(self, last_filename_basename):
            # Auto-resume must follow the latest periodic training state.
            pass

    class BestSegmentationCheckpoint(hooks.HookBase):
        def __init__(self):
            self.best_score = None
            self.best_iteration = None

        def state_dict(self):
            return {"best_score": self.best_score, "best_iteration": self.best_iteration}

        def load_state_dict(self, state_dict):
            self.best_score = state_dict.get("best_score")
            self.best_iteration = state_dict.get("best_iteration")

        def before_train(self):
            output = Path(self.trainer.cfg.OUTPUT_DIR)
            metadata = output / "best_checkpoint.json"
            if metadata.is_file() and (output / "model_best.pth").is_file():
                state = json.loads(metadata.read_text(encoding="utf-8"))
                # A best checkpoint can be newer than the last periodic one.
                if self.best_score is None or state["best_score"] > self.best_score:
                    self.load_state_dict(state)
            self.checkpointer = BestOnlyCheckpointer(
                self.trainer.model,
                self.trainer.cfg.OUTPUT_DIR,
                trainer=self.trainer,
            )

        def save_if_improved(self):
            latest = self.trainer.storage.latest().get("segm/AP")
            if latest is None:
                raise RuntimeError("Validation did not produce segm/AP for best checkpoint selection")
            score = float(latest[0])
            if not math.isfinite(score) or (self.best_score is not None and score <= self.best_score):
                return
            self.best_score = score
            # Detectron2 checkpoint iteration fields are zero-based.
            self.best_iteration = int(latest[1])
            self.checkpointer.save(
                "model_best", iteration=self.best_iteration,
                best_metric="segm/AP", best_score=score,
            )
            output = Path(self.trainer.cfg.OUTPUT_DIR)
            temporary = output / "best_checkpoint.json.tmp"
            temporary.write_text(json.dumps({
                **self.state_dict(), "metric": "segm/AP",
                "completed_iterations": self.best_iteration + 1,
            }, indent=2), encoding="utf-8")
            temporary.replace(output / "best_checkpoint.json")

        def after_step(self):
            period = self.trainer.cfg.TEST.EVAL_PERIOD
            next_iteration = self.trainer.iter + 1
            if period > 0 and next_iteration % period == 0 and next_iteration < self.trainer.max_iter:
                self.save_if_improved()

        def after_train(self):
            if self.trainer.iter + 1 >= self.trainer.max_iter:
                self.save_if_improved()

    result = []
    for hook in configured:
        if isinstance(hook, hooks.PeriodicCheckpointer):
            hook = RetainedPeriodicCheckpointer(
                trainer.checkpointer, trainer.cfg.SOLVER.CHECKPOINT_PERIOD,
                max_to_keep=3,
            )
        result.append(hook)
        if isinstance(hook, hooks.EvalHook) and trainer.cfg.DATASETS.TEST and comm.is_main_process():
            result.append(BestSegmentationCheckpoint())
    return result


def install_training_profile(train_net, profile: str, log_interval: int | None) -> None:
    """Install runtime-only hooks in each distributed worker, leaving upstream intact."""
    if profile != "lineformer" and log_interval is None:
        return
    from detectron2.engine import hooks

    upstream_trainer = train_net.Trainer

    class ProfileTrainer(upstream_trainer):
        def build_hooks(self):
            configured = super().build_hooks()
            if profile == "lineformer":
                configured = profile_checkpoint_hooks(configured, self)
            if log_interval is not None:
                for hook in configured:
                    if isinstance(hook, hooks.PeriodicWriter):
                        hook._period = log_interval
            return configured

        @classmethod
        def build_optimizer(cls, cfg, model):
            optimizer = super().build_optimizer(cfg, model)
            if profile == "lineformer":
                # Upstream already exempts nn.Embedding and normalization modules.
                # Its standalone level_embed parameter also needs zero decay.
                no_decay = {
                    id(value) for name, value in model.named_parameters()
                    if any(key in name for key in ("query_embed", "query_feat", "level_embed"))
                }
                for group in optimizer.param_groups:
                    exempt = [id(value) in no_decay for value in group["params"]]
                    if any(exempt):
                        if not all(exempt):
                            raise RuntimeError("Mixed MaskDINO parameter group cannot preserve embedding decay")
                        group["weight_decay"] = 0.0
            return optimizer

        @classmethod
        def build_train_loader(cls, cfg):
            if profile != "lineformer":
                return super().build_train_loader(cfg)
            from detectron2.data import build_detection_train_loader
            from training.maskdino_source_augmentation import build_source_aware_mapper

            mapper = build_source_aware_mapper(cfg)
            return build_detection_train_loader(cfg, mapper=mapper)

    train_net.Trainer = ProfileTrainer


def image_count(dataset: Path, split: str) -> int:
    payload = json.loads(
        (dataset / "annotations" / f"instances_{split}.json").read_text(encoding="utf-8")
    )
    return len(payload["images"])


def register_datasets(dataset: str) -> None:
    from detectron2.data.datasets import register_coco_instances

    root = Path(dataset)
    for split in ("train", "val", "test"):
        register_coco_instances(
            f"fig2poly_{split}",
            {},
            str(root / "annotations" / f"instances_{split}.json"),
            str(root / "images" / split),
        )


def install_early_stopping(
    train_net,
    patience: int,
    metric: str,
    min_delta: float,
) -> None:
    if patience <= 0:
        return
    from detectron2.engine import hooks
    from detectron2.engine.hooks import HookBase
    from detectron2.utils import comm

    class EarlyStoppingReached(BaseException):
        pass

    class ValidationEarlyStopping(HookBase):
        def __init__(self) -> None:
            self.best_score = float("-inf")
            self.best_iteration = -1
            self.bad_epochs = 0

        def state_dict(self) -> dict:
            return {
                "best_score": self.best_score,
                "best_iteration": self.best_iteration,
                "bad_epochs": self.bad_epochs,
            }

        def load_state_dict(self, state_dict: dict) -> None:
            self.best_score = float(state_dict.get("best_score", float("-inf")))
            self.best_iteration = int(state_dict.get("best_iteration", -1))
            self.bad_epochs = int(state_dict.get("bad_epochs", 0))

        def after_step(self) -> None:
            next_iteration = self.trainer.iter + 1
            period = int(self.trainer.cfg.TEST.EVAL_PERIOD)
            evaluated = next_iteration == self.trainer.max_iter or (
                period > 0 and next_iteration % period == 0
            )
            if not evaluated:
                return
            payload = None
            if comm.is_main_process():
                latest = self.trainer.storage.latest()
                if metric not in latest:
                    payload = {
                        "error": (
                            f"Early-stopping metric {metric!r} missing after "
                            f"validation; available keys: {sorted(latest)}"
                        )
                    }
                else:
                    score = float(latest[metric][0])
                    improved = score > self.best_score + min_delta
                    if improved:
                        self.best_score = score
                        self.best_iteration = next_iteration
                        self.bad_epochs = 0
                    else:
                        self.bad_epochs += 1
                    payload = {
                        "metric": metric,
                        "score": score,
                        "best_score": self.best_score,
                        "best_iteration": self.best_iteration,
                        "bad_epochs": self.bad_epochs,
                        "patience": patience,
                        "min_delta": min_delta,
                        "stopped": self.bad_epochs >= patience
                        and next_iteration < self.trainer.max_iter,
                    }
                    if improved:
                        self.trainer.checkpointer.save(
                            "model_best",
                            iteration=next_iteration,
                            early_stopping_metric=metric,
                            early_stopping_score=score,
                        )
                    target = Path(self.trainer.cfg.OUTPUT_DIR) / "early_stopping.json"
                    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    print(json.dumps({"early_stopping": payload}), flush=True)

            gathered = comm.all_gather(payload)
            payload = next(item for item in gathered if item is not None)
            if "error" in payload:
                raise RuntimeError(payload["error"])
            if not comm.is_main_process():
                self.best_score = float(payload["best_score"])
                self.best_iteration = int(payload["best_iteration"])
                self.bad_epochs = int(payload["bad_epochs"])
            if payload["stopped"]:
                raise EarlyStoppingReached(
                    f"No {metric} improvement for {patience} validation epochs"
                )

    upstream_trainer = train_net.Trainer

    class EarlyStoppingTrainer(upstream_trainer):
        def build_hooks(self):
            configured = super().build_hooks()
            insertion = next(
                (
                    index + 1
                    for index, hook in enumerate(configured)
                    if isinstance(hook, hooks.EvalHook)
                ),
                len(configured),
            )
            configured.insert(insertion, ValidationEarlyStopping())
            return configured

        def train(self):
            try:
                return super().train()
            except EarlyStoppingReached as error:
                print(f"Early stopping completed successfully: {error}", flush=True)
                return getattr(self, "_last_eval_results", {})

    train_net.Trainer = EarlyStoppingTrainer


def worker(
    upstream_args,
    dataset: str,
    report: str | None,
    early_stopping_patience: int,
    early_stopping_metric: str,
    early_stopping_min_delta: float,
    curve_loss_weight: float,
    curve_metrics: bool,
    curve_score_threshold: float,
    curve_sample_interval: int,
    training_profile: str = "legacy",
    log_interval: int | None = None,
):
    register_datasets(dataset)
    import train_net
    from detectron2.utils import comm

    install_training_profile(train_net, training_profile, log_interval)
    if curve_loss_weight > 0:
        from training.maskdino_curve_loss import install_curve_geometry_loss

        install_curve_geometry_loss(curve_loss_weight)
    if curve_metrics:
        from training.maskdino_line_evaluator import install_chart_line_evaluator

        install_chart_line_evaluator(
            train_net, curve_score_threshold, curve_sample_interval
        )

    install_early_stopping(
        train_net,
        early_stopping_patience,
        early_stopping_metric,
        early_stopping_min_delta,
    )
    result = train_net.main(upstream_args)
    if report and comm.is_main_process():
        Path(report).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train official MaskDINO on Fig2Poly COCO RLE")
    parser.add_argument("--maskdino-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=CONFIGS, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--training-profile", choices=("legacy", "lineformer"), default="legacy",
        help="lineformer: 100k iterations, 4 images/GPU, fixed LR 1e-4, 512px, 100 queries",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--global-batch", type=int, help="Default: legacy=2, lineformer=4*num-gpus")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, help="LineFormer profile defaults to 20260905")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, help="Default: legacy=1024, lineformer=512")
    parser.add_argument("--base-lr", type=float)
    parser.add_argument("--log-interval", type=int, help="Writer period; LineFormer profile defaults to 100")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable Detectron2 automatic mixed-precision training",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        help="Override the profile or epoch-derived iteration budget (also for smoke tests)",
    )
    parser.add_argument(
        "--checkpoint-period",
        type=int,
        help="Override checkpoint interval in iterations",
    )
    parser.add_argument(
        "--eval-period",
        type=int,
        help="Override validation interval in iterations; 0 disables periodic evaluation",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many validation epochs without improvement; 0 disables",
    )
    parser.add_argument("--early-stopping-metric", default="segm/AP")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.01)
    parser.add_argument(
        "--curve-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the thickness-invariant centreline/tangent auxiliary loss",
    )
    parser.add_argument(
        "--curve-metrics",
        action="store_true",
        help="Evaluate LineFormer/ChartInfo task 6a and 6b line scores",
    )
    parser.add_argument("--curve-score-threshold", type=float, default=0.25)
    parser.add_argument("--curve-sample-interval", type=int, default=4)
    parser.add_argument(
        "--disable-evaluation",
        action="store_true",
        help="Disable evaluation datasets entirely (benchmarking only)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.disable_evaluation and args.early_stopping_patience:
        parser.error("--disable-evaluation is incompatible with early stopping")
    for name in ("num_gpus", "global_batch", "image_size", "epochs", "max_iter", "checkpoint_period", "log_interval", "base_lr"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.eval_period is not None and args.eval_period < 0:
        parser.error("--eval-period must be nonnegative")
    if args.global_batch is not None and args.global_batch % args.num_gpus:
        parser.error("--global-batch must be divisible by --num-gpus")
    if args.training_profile == "lineformer":
        if args.variant != "r50":
            parser.error("the lineformer training profile currently supports --variant r50")
        if args.early_stopping_patience or args.curve_loss_weight:
            parser.error("the lineformer profile requires early stopping and auxiliary curve loss disabled")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    maskdino_root = args.maskdino_root.resolve()
    dataset = args.dataset.resolve()
    config = maskdino_root / CONFIGS[args.variant]
    if not config.is_file():
        raise FileNotFoundError(f"MaskDINO config not found: {config}")
    if not args.weights.is_file() and not args.resume:
        raise FileNotFoundError(f"pretrained checkpoint not found: {args.weights}")
    from training.patch_maskdino_numerics import patch_matcher

    patch_matcher(maskdino_root)
    if args.training_profile == "lineformer":
        from training.patch_maskdino_rle import main as patch_rle_mapper

        patch_rle_mapper(["--maskdino-root", str(maskdino_root)])
    sys.path.insert(0, str(maskdino_root))
    import train_net
    from detectron2.engine import launch

    schedule_opts, schedule_summary = training_options(args, image_count(dataset, "train"))
    opts = [
        "MODEL.WEIGHTS", str(args.weights.resolve()),
        "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",
        "DATASETS.TRAIN", '("fig2poly_train",)',
        "DATASETS.TEST",
        "()" if args.disable_evaluation else f'("fig2poly_{args.eval_split}",)',
        "DATALOADER.FILTER_EMPTY_ANNOTATIONS", "False",
        "DATALOADER.NUM_WORKERS", str(args.workers),
        "OUTPUT_DIR", str(args.output.resolve()),
    ]
    opts.extend(schedule_opts)
    command = ["--config-file", str(config), "--num-gpus", str(args.num_gpus)]
    if args.resume:
        command.append("--resume")
    if args.eval_only:
        command.append("--eval-only")
    command.extend(opts)
    upstream_args = train_net.default_argument_parser().parse_args(command)
    summary = {
        "variant": args.variant,
        **schedule_summary,
        "dataset": str(dataset),
        "initial_weights": str(args.weights.resolve()),
        "upstream_config": str(config),
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_metric": args.early_stopping_metric,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "evaluation_enabled": not args.disable_evaluation,
        "amp": args.amp,
        "resume": args.resume,
        "curve_loss_weight": args.curve_loss_weight,
        "curve_metrics": args.curve_metrics,
        "curve_score_threshold": args.curve_score_threshold,
        "curve_sample_interval": args.curve_sample_interval,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.training_profile == "lineformer":
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "training_profile.json").write_text(
            json.dumps({**summary, "config_overrides": dict(zip(opts[::2], opts[1::2]))}, indent=2),
            encoding="utf-8",
        )
    launch(
        worker,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url=upstream_args.dist_url,
        args=(
            upstream_args,
            str(dataset),
            str(args.report.resolve()) if args.report else None,
            args.early_stopping_patience,
            args.early_stopping_metric,
            args.early_stopping_min_delta,
            args.curve_loss_weight,
            args.curve_metrics,
            args.curve_score_threshold,
            args.curve_sample_interval,
            args.training_profile,
            schedule_summary["log_interval"],
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
