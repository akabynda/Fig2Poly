from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from training.benchmark_inference import add_common_arguments, run_benchmark_inference
from training.predict_maskdino import CONFIGS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable MaskDINO benchmark inference")
    add_common_arguments(parser)
    parser.add_argument("--maskdino-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--variant", choices=CONFIGS, default="r50")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--max-size", type=int, default=2048)
    args = parser.parse_args(argv)

    root = args.maskdino_root.resolve()
    sys.path.insert(0, str(root))
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import transforms as T
    from detectron2.projects.deeplab import add_deeplab_config
    from maskdino import add_maskdino_config
    import train_net

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(root / CONFIGS[args.variant]))
    cfg.MODEL.WEIGHTS = str(args.weights.resolve())
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = args.min_threshold
    cfg.INPUT.MIN_SIZE_TEST = args.image_size
    cfg.INPUT.MAX_SIZE_TEST = args.max_size
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    model = train_net.Trainer.build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    transform = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    def infer(image: np.ndarray, threshold: float) -> list[dict]:
        height, width = image.shape[:2]
        resized = transform.get_transform(image).apply_image(image)
        tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
        with torch.inference_mode():
            output = model([{"image": tensor, "height": height, "width": width}])[0]
        instances = output["instances"].to("cpu")
        scores = instances.scores.numpy() if instances.has("scores") else np.ones(len(instances))
        masks = instances.pred_masks.numpy().astype(bool)
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.zeros((len(instances), 4))
        return [
            {
                "score": float(score), "mask": mask,
                "bbox": [float(value) for value in box], "source": "maskdino",
            }
            for score, mask, box in zip(scores, masks, boxes)
            if float(score) >= threshold and np.any(mask)
        ]

    run_benchmark_inference(infer, args.dataset, args.output.resolve(), "maskdino",
                            args.min_threshold, args.stop_on_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
