from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from training.benchmark_inference import add_common_arguments, run_benchmark_inference
from training.benchmark_predictions import prediction_bbox


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable LineFormer benchmark inference")
    add_common_arguments(parser)
    parser.add_argument("--lineformer-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.lineformer_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "mmdetection"))
    from mmdet.apis import inference_detector, init_detector

    model = init_detector(str(root / "lineformer_swin_t_config.py"), str(args.weights.resolve()), device="cuda:0")

    def infer(image: np.ndarray, threshold: float) -> list[dict]:
        result = inference_detector(model, image)
        boxes = np.asarray(result[0][0])
        masks = result[1][0]
        return [
            {
                "score": float(box[4]),
                "mask": np.asarray(mask, dtype=bool),
                "bbox": [float(value) for value in box[:4]],
                "source": "lineformer",
            }
            for box, mask in zip(boxes, masks)
            if float(box[4]) >= threshold and np.any(mask)
        ]

    run_benchmark_inference(infer, args.dataset, args.output.resolve(), "lineformer",
                            args.min_threshold, args.stop_on_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
