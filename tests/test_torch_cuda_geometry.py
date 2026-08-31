from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch_cuda_geometry import (
    OpenCvGeometryBackend,
    TorchCudaGeometryBackend,
    create_geometry_backend,
)
from yolo_roi_sobel_projection import edge_response_from_roi

try:
    import torch
except ImportError:
    torch = None


CUDA_AVAILABLE = bool(torch is not None and torch.cuda.is_available())


class GeometryBackendTests(unittest.TestCase):
    def test_cpu_backend_can_be_selected_explicitly(self) -> None:
        backend, error = create_geometry_backend("cpu", "cpu", False)

        self.assertIsInstance(backend, OpenCvGeometryBackend)
        self.assertEqual(backend.homography_name, "opencv_cpu")
        self.assertEqual(backend.sobel_name, "opencv_cpu")
        self.assertEqual(error, "")

    def test_cuda_request_without_cuda_reports_fallback(self) -> None:
        backend, error = create_geometry_backend("cuda", "cuda:0", False)

        self.assertIsInstance(backend, OpenCvGeometryBackend)
        self.assertIn("unavailable", error)

    def test_auto_mode_uses_faster_opencv_geometry(self) -> None:
        backend, error = create_geometry_backend("auto", "cuda:0", True)

        self.assertIsInstance(backend, OpenCvGeometryBackend)
        self.assertEqual(backend.homography_name, "opencv_cpu")
        self.assertEqual(backend.sobel_name, "opencv_cpu")
        self.assertEqual(error, "")

    @unittest.skipUnless(CUDA_AVAILABLE, "PyTorch CUDA is unavailable")
    def test_cuda_homography_matches_opencv(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (180, 240, 3), dtype=np.uint8)
        matrix = cv2.getPerspectiveTransform(
            np.float32(((5, 4), (230, 8), (8, 170), (235, 175))),
            np.float32(((0, 0), (199, 0), (0, 139), (199, 139))),
        ).astype(np.float64)
        backend = TorchCudaGeometryBackend("cuda:0")

        expected = cv2.warpPerspective(image, matrix, (200, 140))
        actual = backend.warp_perspective(image, matrix, (200, 140))
        difference = np.abs(expected.astype(np.int16) - actual.astype(np.int16))

        self.assertLessEqual(int(difference.max()), 1)
        self.assertLess(float(difference.mean()), 0.01)

    @unittest.skipUnless(CUDA_AVAILABLE, "PyTorch CUDA is unavailable")
    def test_cuda_sobel_remains_equivalent_to_opencv(self) -> None:
        rng = np.random.default_rng(73)
        roi = rng.integers(0, 256, (120, 200, 3), dtype=np.uint8)
        config = SimpleNamespace(
            clahe_clip=2.0,
            blur_ksize=(21, 1),
            sobel_ksize=3,
            edge_polarity="falling",
        )
        backend = TorchCudaGeometryBackend("cuda:0")

        expected_gray, expected_edge = edge_response_from_roi(roi, config)
        actual_gray, actual_edge = backend.edge_response_from_roi(roi, config)
        gray_difference = np.abs(
            expected_gray.astype(np.int16) - actual_gray.astype(np.int16)
        )
        difference = np.abs(expected_edge - actual_edge)

        self.assertLess(float(gray_difference.mean()), 0.5)
        self.assertLessEqual(int(gray_difference.max()), 2)
        self.assertLess(float(difference.mean()), 1.0)
        self.assertLess(float(difference.max()), 8.0)


if __name__ == "__main__":
    unittest.main()
