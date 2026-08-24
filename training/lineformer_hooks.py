from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ValidationEarlyStoppingHook(Hook):
    """Stop an iter-based run after repeated validation plateaus."""

    def __init__(
        self,
        metric: str,
        interval: int,
        patience: int,
        min_delta: float = 0.0,
    ) -> None:
        self.metric = metric
        self.interval = interval
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.best_iteration = -1
        self.bad_checks = 0

    def before_run(self, runner) -> None:
        state_path = Path(runner.work_dir) / "early_stopping.json"
        if not state_path.is_file():
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("metric") != self.metric:
            return
        self.best_score = float(state.get("best_score", float("-inf")))
        self.best_iteration = int(state.get("best_iteration", -1))
        self.bad_checks = int(state.get("bad_checks", 0))

    def after_train_iter(self, runner) -> None:
        iteration = runner.iter + 1
        if iteration % self.interval != 0 and iteration != runner.max_iters:
            return

        rank = getattr(runner, "rank", 0)
        stop = False
        missing_metric = False
        if rank == 0:
            value = runner.log_buffer.output.get(self.metric)
            if value is None:
                missing_metric = True
            else:
                score = float(value)
                if score > self.best_score + self.min_delta:
                    self.best_score = score
                    self.best_iteration = iteration
                    self.bad_checks = 0
                else:
                    self.bad_checks += 1
                stop = self.bad_checks >= self.patience and iteration < runner.max_iters
                state = {
                    "metric": self.metric,
                    "score": score,
                    "best_score": self.best_score,
                    "best_iteration": self.best_iteration,
                    "bad_checks": self.bad_checks,
                    "patience": self.patience,
                    "min_delta": self.min_delta,
                    "stopped": stop,
                }
                target = Path(runner.work_dir) / "early_stopping.json"
                target.write_text(json.dumps(state, indent=2), encoding="utf-8")
                runner.logger.info("LineFormer early stopping: %s", state)

        if dist.is_available() and dist.is_initialized():
            device = next(runner.model.parameters()).device
            status = torch.tensor(
                [int(stop), int(missing_metric)], dtype=torch.uint8, device=device
            )
            dist.broadcast(status, src=0)
            stop, missing_metric = (bool(item) for item in status.tolist())

        if missing_metric:
            raise RuntimeError(
                f"Early-stopping metric {self.metric!r} was not produced by validation"
            )
        if stop:
            runner.logger.info(
                "Stopping at iteration %d after %d checks without improvement",
                iteration,
                self.bad_checks,
            )
            runner._max_iters = iteration
