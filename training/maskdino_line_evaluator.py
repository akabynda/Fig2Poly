from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.line_metrics import mask_to_centerline, match_line_instances


class MaskDINOChartLineEvaluator:
    """Detectron2 evaluator for the LineFormer/ChartInfo 6a and 6b scores."""

    def __init__(
        self,
        dataset_name: str,
        output_dir: str | None,
        score_threshold: float = 0.25,
        sample_interval: int = 4,
    ) -> None:
        from detectron2.data import MetadataCatalog
        from pycocotools.coco import COCO

        metadata = MetadataCatalog.get(dataset_name)
        self.coco = COCO(metadata.json_file)
        self.output_dir = Path(output_dir) if output_dir else None
        self.score_threshold = score_threshold
        self.sample_interval = sample_interval
        self.records: list[dict[str, float]] = []

    def reset(self) -> None:
        self.records = []

    def process(self, inputs, outputs) -> None:
        for model_input, model_output in zip(inputs, outputs):
            image_id = int(model_input["image_id"])
            annotation_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
            targets = [
                mask_to_centerline(
                    self.coco.annToMask(annotation), self.sample_interval
                )
                for annotation in self.coco.loadAnns(annotation_ids)
            ]
            instances = model_output["instances"].to("cpu")
            predictions = [
                mask_to_centerline(mask.numpy(), self.sample_interval)
                for mask, score in zip(instances.pred_masks, instances.scores)
                if float(score) >= self.score_threshold
            ]
            self.records.append(match_line_instances(predictions, targets))

    def evaluate(self):
        from detectron2.utils import comm

        gathered = comm.gather(self.records, dst=0)
        if not comm.is_main_process():
            return {}
        records = [record for shard in gathered for record in shard]
        count = max(1, len(records))
        metrics = {
            "score_6a": 100 * sum(item["score_6a"] for item in records) / count,
            "score_6b": 100 * sum(item["score_6b"] for item in records) / count,
            "count_mae": sum(item["count_error"] for item in records) / count,
            "images": len(records),
        }
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "chart_line_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
        return {"curve": metrics}


def install_chart_line_evaluator(
    train_net,
    score_threshold: float = 0.25,
    sample_interval: int = 4,
) -> None:
    from detectron2.evaluation import DatasetEvaluators

    upstream_trainer = train_net.Trainer

    class ChartLineTrainer(upstream_trainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            base = super().build_evaluator(cfg, dataset_name, output_folder)
            curve = MaskDINOChartLineEvaluator(
                dataset_name,
                output_folder or str(Path(cfg.OUTPUT_DIR) / "inference"),
                score_threshold,
                sample_interval,
            )
            return DatasetEvaluators([base, curve])

    train_net.Trainer = ChartLineTrainer
