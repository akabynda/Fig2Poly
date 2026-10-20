import argparse
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


@pytest.mark.parametrize("saved,overrides,expected_size,expected_format,expected_queries", [
    (True, [], (512, 512), "RGB", 100),
    (True, ["--image-size", "1024", "--max-size", "1536"], (1024, 1536), "RGB", 100),
    (False, [], (768, 2048), "BGR", 300),
])
def test_prediction_uses_saved_architecture_and_rgb_without_changing_exported_original(
    monkeypatch, tmp_path, saved, overrides, expected_size, expected_format, expected_queries,
):
    source = Path(__file__).resolve().parents[1] / "training/predict_maskdino.py"
    nodes = ast.parse(source.read_text(encoding="utf-8")).body
    functions = [node for node in nodes if isinstance(node, ast.FunctionDef)]
    captured = {}
    cfg = SimpleNamespace(
        MODEL=SimpleNamespace(SEM_SEG_HEAD=SimpleNamespace(NUM_CLASSES=80),
                              MaskDINO=SimpleNamespace(NUM_OBJECT_QUERIES=300,
                                                      TEST=SimpleNamespace(OBJECT_MASK_THRESHOLD=0))),
        INPUT=SimpleNamespace(MIN_SIZE_TEST=800, MAX_SIZE_TEST=1333, FORMAT="BGR"),
        freeze=lambda: None, dump=lambda: "resolved: true\n",
    )
    config = tmp_path / "config.yaml"
    config.write_text("fixture")

    def merge(filename):
        captured["config"] = filename
        if Path(filename) == config:
            cfg.INPUT.MIN_SIZE_TEST = cfg.INPUT.MAX_SIZE_TEST = 512
            cfg.INPUT.FORMAT = "RGB"
            cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 100

    cfg.merge_from_file = merge
    images = tmp_path / "images"
    images.mkdir()
    original = np.full((16, 20, 3), (10, 20, 30), dtype=np.uint8)
    cv2.imwrite(str(images / "example.png"), original)
    output = tmp_path / "output"
    instances = SimpleNamespace(
        has=lambda name: name == "scores",
        scores=SimpleNamespace(numpy=lambda: np.empty(0)),
        pred_masks=SimpleNamespace(numpy=lambda: np.empty((0, 16, 20), dtype=bool)),
    )
    instances.to = lambda device: instances

    class Model:
        def eval(self):
            pass

        def __call__(self, inputs):
            captured["input"] = inputs[0]
            return [{"instances": instances}]

    class Checkpointer:
        def __init__(self, model):
            pass

        def load(self, filename):
            captured["weights"] = filename

    class Resize:
        def __init__(self, sizes, max_size):
            captured["resize"] = (sizes, max_size)

        def get_transform(self, image):
            return SimpleNamespace(apply_image=lambda image: image)

    modules = {
        "detectron2.checkpoint": SimpleNamespace(DetectionCheckpointer=Checkpointer),
        "detectron2.config": SimpleNamespace(get_cfg=lambda: cfg),
        "detectron2.data": SimpleNamespace(transforms=SimpleNamespace(ResizeShortestEdge=Resize)),
        "detectron2.projects.deeplab": SimpleNamespace(add_deeplab_config=lambda cfg: None),
        "maskdino": SimpleNamespace(add_maskdino_config=lambda cfg: None),
        "train_net": SimpleNamespace(Trainer=SimpleNamespace(build_model=lambda cfg: Model())),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {"argparse": argparse, "Path": Path, "sys": sys, "cv2": cv2, "np": np,
                 "json": json, "CONFIGS": {"r50": "upstream.yaml"},
                 "torch": SimpleNamespace(as_tensor=lambda x: x, inference_mode=nullcontext)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    command = ["--maskdino-root", str(tmp_path), "--weights", str(tmp_path / "best.pth"),
               "--input", str(images), "--output", str(output)]
    if saved:
        command += ["--config", str(config)]
    assert namespace["main"](command + overrides) == 0
    assert cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES == expected_queries
    assert captured["resize"] == ([expected_size[0]] * 2, expected_size[1])
    expected_pixel = [30, 20, 10] if expected_format == "RGB" else [10, 20, 30]
    assert captured["input"]["image"][:, 0, 0].tolist() == expected_pixel
    assert np.array_equal(cv2.imread(str(output / "example/original.png")), original)
    request = json.loads((output / "inference_request.json").read_text())
    assert request["num_queries"] == expected_queries
    assert request["input_format"] == expected_format
    assert request["mask_dilation"] == 0
    assert len(json.loads((output / "summary.json").read_text())) == 1
