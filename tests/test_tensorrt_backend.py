from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import homography_web_app as vision


class _FakeModel:
    def __init__(self, *, fails: bool) -> None:
        self.fails = fails
        self.device = None

    def to(self, device: str) -> None:
        self.device = device

    def predict(self, *_args, **_kwargs):
        if self.fails:
            raise RuntimeError("incompatible TensorRT engine")
        return [SimpleNamespace(boxes=None)]


class TensorRTBackendTests(unittest.TestCase):
    def test_tensorrt_failure_retries_matching_pytorch_checkpoint(self) -> None:
        previous = (
            getattr(vision, "_args", None),
            vision._yolo_model,
            vision._yolo_model_path,
            vision._yolo_backend_error,
        )
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory) / "best.engine"
            checkpoint = engine.with_suffix(".pt")
            engine.write_bytes(b"engine")
            checkpoint.write_bytes(b"checkpoint")

            created: list[tuple[Path, _FakeModel]] = []

            def fake_yolo(path: str, **_kwargs):
                model = _FakeModel(fails=Path(path).suffix == ".engine")
                created.append((Path(path), model))
                return model

            vision._args = SimpleNamespace(model=engine, device="cpu")
            vision._yolo_model = None
            vision._yolo_model_path = None
            vision._yolo_backend_error = ""
            try:
                with (
                    patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=fake_yolo)}),
                    patch.object(
                        vision,
                        "yolo_device_info",
                        return_value={"device": "cpu"},
                    ),
                ):
                    boxes, diagnostics = vision.predict_yolo_boxes_with_rules(
                        np.zeros((32, 32, 3), dtype=np.uint8)
                    )

                backend = vision.yolo_backend_info()
                self.assertEqual(boxes, [])
                self.assertEqual(diagnostics["raw_count"], 0)
                self.assertEqual(backend["backend"], "pytorch")
                self.assertTrue(backend["fallback_active"])
                self.assertEqual(Path(backend["active_model"]), checkpoint.resolve())
                self.assertIn("incompatible TensorRT engine", backend["backend_error"])
                self.assertEqual([path.suffix for path, _model in created], [".engine", ".pt"])
                self.assertEqual(created[-1][1].device, "cpu")
            finally:
                vision._args, vision._yolo_model, vision._yolo_model_path, vision._yolo_backend_error = previous


if __name__ == "__main__":
    unittest.main()
