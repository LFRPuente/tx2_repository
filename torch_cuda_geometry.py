from __future__ import annotations

from collections import OrderedDict
from typing import Any

import cv2
import numpy as np


class OpenCvGeometryBackend:
    homography_name = "opencv_cpu"
    sobel_name = "opencv_cpu"

    def prepare_homography(
        self,
        input_size: tuple[int, int],
        matrix: np.ndarray,
        output_size: tuple[int, int],
    ) -> None:
        del input_size, matrix, output_size

    def warp_perspective(
        self,
        frame: np.ndarray,
        matrix: np.ndarray,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        return cv2.warpPerspective(frame, matrix, output_size)

    def edge_response_from_roi(self, roi: np.ndarray, config: Any):
        from yolo_roi_sobel_projection import edge_response_from_roi

        return edge_response_from_roi(roi, config)


class TorchCudaGeometryBackend:
    homography_name = "torch_cuda"

    def __init__(
        self,
        device: str = "cuda:0",
        grid_cache_size: int = 2,
        cuda_sobel: bool = True,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        self.torch = torch
        self.functional = torch.nn.functional
        self.device = torch.device(device if str(device).startswith("cuda") else "cuda:0")
        torch.empty(1, device=self.device)
        self.cuda_sobel = bool(cuda_sobel)
        self.sobel_name = "torch_cuda" if self.cuda_sobel else "opencv_cpu"
        self.grid_cache_size = max(1, int(grid_cache_size))
        self.grid_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

    @staticmethod
    def _grid_key(
        input_size: tuple[int, int],
        matrix: np.ndarray,
        output_size: tuple[int, int],
    ) -> tuple[Any, ...]:
        normalized_matrix = np.asarray(matrix, dtype=np.float64)
        return (
            int(input_size[0]),
            int(input_size[1]),
            int(output_size[0]),
            int(output_size[1]),
            normalized_matrix.tobytes(),
        )

    def _homography_grid(
        self,
        input_size: tuple[int, int],
        matrix: np.ndarray,
        output_size: tuple[int, int],
    ):
        key = self._grid_key(input_size, matrix, output_size)
        cached = self.grid_cache.get(key)
        if cached is not None:
            self.grid_cache.move_to_end(key)
            return cached

        input_height, input_width = input_size
        output_width, output_height = output_size
        torch = self.torch
        inverse = torch.as_tensor(
            np.linalg.inv(np.asarray(matrix, dtype=np.float64)),
            dtype=torch.float64,
            device=self.device,
        )
        ys, xs = torch.meshgrid(
            torch.arange(output_height, dtype=torch.float64, device=self.device),
            torch.arange(output_width, dtype=torch.float64, device=self.device),
            indexing="ij",
        )
        ones = torch.ones_like(xs)
        destination = torch.stack((xs, ys, ones), dim=0).reshape(3, -1)
        source = inverse @ destination
        denominator = source[2]
        epsilon = torch.full_like(denominator, 1e-12)
        denominator = torch.where(
            denominator.abs() < epsilon,
            torch.where(denominator < 0, -epsilon, epsilon),
            denominator,
        )
        source_x = source[0] / denominator
        source_y = source[1] / denominator
        if input_width > 1:
            source_x = source_x.mul(2.0 / float(input_width - 1)).sub(1.0)
        else:
            source_x.zero_()
        if input_height > 1:
            source_y = source_y.mul(2.0 / float(input_height - 1)).sub(1.0)
        else:
            source_y.zero_()
        grid = torch.stack((source_x, source_y), dim=1).reshape(
            1,
            output_height,
            output_width,
            2,
        ).to(dtype=torch.float32)
        self.grid_cache[key] = grid
        self.grid_cache.move_to_end(key)
        while len(self.grid_cache) > self.grid_cache_size:
            self.grid_cache.popitem(last=False)
        return grid

    def prepare_homography(
        self,
        input_size: tuple[int, int],
        matrix: np.ndarray,
        output_size: tuple[int, int],
    ) -> None:
        self._homography_grid(input_size, matrix, output_size)
        self.torch.cuda.synchronize(self.device)

    def warp_perspective(
        self,
        frame: np.ndarray,
        matrix: np.ndarray,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("CUDA homography requires a BGR image")
        torch = self.torch
        contiguous = np.ascontiguousarray(frame)
        grid = self._homography_grid(frame.shape[:2], matrix, output_size)
        with torch.inference_mode():
            source = torch.from_numpy(contiguous).to(
                device=self.device,
                dtype=torch.float32,
            ).permute(2, 0, 1).unsqueeze(0)
            warped = self.functional.grid_sample(
                source,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            output = warped.squeeze(0).permute(1, 2, 0).round().clamp_(0, 255)
            return output.to(dtype=torch.uint8).cpu().numpy()

    def edge_response_from_roi(self, roi: np.ndarray, config: Any):
        if not self.cuda_sobel:
            from yolo_roi_sobel_projection import edge_response_from_roi

            return edge_response_from_roi(roi, config)
        if roi.ndim != 3 or roi.shape[2] != 3:
            raise ValueError("CUDA Sobel requires a BGR ROI")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(
            clipLimit=float(config.clahe_clip),
            tileGridSize=(8, 8),
        ).apply(gray)

        torch = self.torch
        with torch.inference_mode():
            image = torch.from_numpy(np.ascontiguousarray(gray)).to(
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0)
            kernel_width, kernel_height = (
                int(config.blur_ksize[0]),
                int(config.blur_ksize[1]),
            )
            gaussian_x = cv2.getGaussianKernel(kernel_width, 0).reshape(1, -1)
            gaussian_y = cv2.getGaussianKernel(kernel_height, 0).reshape(-1, 1)
            gaussian = torch.as_tensor(
                gaussian_y @ gaussian_x,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).unsqueeze(0)
            pad_x = kernel_width // 2
            pad_y = kernel_height // 2
            if pad_x or pad_y:
                image = self.functional.pad(
                    image,
                    (pad_x, pad_x, pad_y, pad_y),
                    mode="reflect",
                )
            blurred = self.functional.conv2d(image, gaussian)
            blurred_uint8 = blurred.round().clamp_(0, 255).to(dtype=torch.uint8)
            blurred = blurred_uint8.to(dtype=torch.float32)

            if int(config.sobel_ksize) != 3:
                raise ValueError("CUDA Sobel currently supports ksize=3")
            sobel_y = torch.tensor(
                ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0)),
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 1, 3, 3)
            gradient = self.functional.conv2d(
                self.functional.pad(blurred, (1, 1, 1, 1), mode="reflect"),
                sobel_y,
            )
            if config.edge_polarity == "falling":
                edge = (-gradient).clamp_min_(0.0)
            elif config.edge_polarity == "rising":
                edge = gradient.clamp_min_(0.0)
            else:
                edge = gradient.abs_()
            filtered_gray = blurred_uint8.squeeze(0).squeeze(0).cpu().numpy()
            return filtered_gray, edge.squeeze(0).squeeze(0).cpu().numpy()


def create_geometry_backend(
    requested: str,
    inference_device: str,
    cuda_available: bool,
) -> tuple[OpenCvGeometryBackend | TorchCudaGeometryBackend, str]:
    mode = str(requested or "auto").lower()
    if mode in {"auto", "cpu"}:
        return OpenCvGeometryBackend(), ""
    if not cuda_available:
        detail = "CUDA geometry requested but PyTorch CUDA is unavailable" if mode == "cuda" else ""
        return OpenCvGeometryBackend(), detail
    try:
        return TorchCudaGeometryBackend(
            inference_device,
            cuda_sobel=mode == "cuda",
        ), ""
    except Exception as exc:
        return OpenCvGeometryBackend(), str(exc)
