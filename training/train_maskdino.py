from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


CONFIGS = {
    "r50": "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
    "swin_l": "configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
}


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
            latest = self.trainer.storage.latest()
            if metric not in latest:
                raise RuntimeError(
                    f"Early-stopping metric {metric!r} missing after validation; "
                    f"available keys: {sorted(latest)}"
                )
            score = float(latest[metric][0])
            improved = score > self.best_score + min_delta
            if improved:
                self.best_score = score
                self.best_iteration = next_iteration
                self.bad_epochs = 0
                if comm.is_main_process():
                    self.trainer.checkpointer.save(
                        "model_best",
                        iteration=next_iteration,
                        early_stopping_metric=metric,
                        early_stopping_score=score,
                    )
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
            if comm.is_main_process():
                target = Path(self.trainer.cfg.OUTPUT_DIR) / "early_stopping.json"
                target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(json.dumps({"early_stopping": payload}), flush=True)
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
):
    register_datasets(dataset)
    import train_net
    from detectron2.utils import comm

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train official MaskDINO on Fig2Poly COCO RLE")
    parser.add_argument("--maskdino-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=CONFIGS, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--global-batch", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--base-lr", type=float)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable Detectron2 automatic mixed-precision training",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        help="Override the epoch-derived iteration count (used by smoke tests)",
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

    maskdino_root = args.maskdino_root.resolve()
    dataset = args.dataset.resolve()
    config = maskdino_root / CONFIGS[args.variant]
    if not config.is_file():
        parser.error(f"MaskDINO config not found: {config}")
    if not args.weights.is_file() and not args.resume:
        parser.error(f"pretrained checkpoint not found: {args.weights}")
    sys.path.insert(0, str(maskdino_root))
    import train_net
    from detectron2.engine import launch

    steps_per_epoch = math.ceil(image_count(dataset, "train") / args.global_batch)
    max_iter = args.max_iter or max(1, steps_per_epoch * args.epochs)
    checkpoint_period = args.checkpoint_period or max(100, steps_per_epoch // 4)
    eval_period = steps_per_epoch if args.eval_period is None else args.eval_period
    learning_rate = args.base_lr or 1e-4 * args.global_batch / 16
    if max_iter <= 2:
        solver_steps = "()"
    else:
        milestones = sorted({int(max_iter * .89), int(max_iter * .96)})
        milestones = [step for step in milestones if 0 < step < max_iter]
        solver_steps = f"({', '.join(str(step) for step in milestones)},)"
    opts = [
        "MODEL.WEIGHTS", str(args.weights.resolve()),
        "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",
        "DATASETS.TRAIN", '("fig2poly_train",)',
        "DATASETS.TEST",
        "()" if args.disable_evaluation else f'("fig2poly_{args.eval_split}",)',
        "DATALOADER.FILTER_EMPTY_ANNOTATIONS", "False",
        "DATALOADER.NUM_WORKERS", str(args.workers),
        "SOLVER.IMS_PER_BATCH", str(args.global_batch),
        "SOLVER.BASE_LR", str(learning_rate),
        "SOLVER.AMP.ENABLED", str(args.amp),
        "SOLVER.MAX_ITER", str(max_iter),
        "SOLVER.STEPS", solver_steps,
        "SOLVER.CHECKPOINT_PERIOD", str(checkpoint_period),
        "TEST.EVAL_PERIOD", str(eval_period),
        "INPUT.IMAGE_SIZE", str(args.image_size),
        "INPUT.MIN_SCALE", "0.5",
        "INPUT.MAX_SCALE", "1.5",
        "OUTPUT_DIR", str(args.output.resolve()),
    ]
    command = ["--config-file", str(config), "--num-gpus", str(args.num_gpus)]
    if args.resume:
        command.append("--resume")
    if args.eval_only:
        command.append("--eval-only")
    command.extend(opts)
    upstream_args = train_net.default_argument_parser().parse_args(command)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "train_images": image_count(dataset, "train"),
                "steps_per_epoch": steps_per_epoch,
                "max_iter": max_iter,
                "checkpoint_period": checkpoint_period,
                "eval_period": eval_period,
                "solver_steps": solver_steps,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_metric": args.early_stopping_metric,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "evaluation_enabled": not args.disable_evaluation,
                "global_batch": args.global_batch,
                "base_lr": learning_rate,
                "amp": args.amp,
                "resume": args.resume,
                "curve_loss_weight": args.curve_loss_weight,
                "curve_metrics": args.curve_metrics,
                "curve_score_threshold": args.curve_score_threshold,
                "curve_sample_interval": args.curve_sample_interval,
            },
            indent=2,
        ),
        flush=True,
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
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
