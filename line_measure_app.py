from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np


DEFAULT_HOMOGRAPHY_JSON = Path(
    r"C:\Users\luis_\OneDrive\Desktop\tx2_cv\outputs\homography_selection.json"
)
DEFAULT_VIDEO_DIR = Path(r"C:\Users\luis_\OneDrive\Desktop\tx2_cv")


@dataclass
class DetectorConfig:
    process_width: int = 1366
    roi_norm_original: tuple[float, float, float, float] = (0.384, 0.013, 0.659, 0.677)
    roi_norm_rectified: tuple[float, float, float, float] = (0.00, 0.08, 1.00, 0.62)
    edge_mode: str = "bottom"
    search_band_original: tuple[float, float] | None = None
    search_band_rectified: tuple[float, float] = (0.45, 0.88)
    num_sample_points: int = 26
    sample_margin_x: int = 20
    column_half_width: int = 6
    tracking_radius_y: int = 24
    point_profile_smooth: int = 11
    score_keep_percentile: int = 45
    cluster_tol_y: int = 14
    line_inlier_tol: int = 10
    min_inliers: int = 8
    max_abs_slope: float = 0.20
    min_confidence: float = 0.30
    max_crm_px: float = 4.0
    clahe_clip: float = 2.0
    pre_sobel_blur_ksize: tuple[int, int] = (31, 1)
    smoothing_window: int = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selecciona un video y mide la linea en tiempo real."
    )
    parser.add_argument("--video", type=Path, help="Ruta del video a procesar.")
    parser.add_argument(
        "--homography-json",
        type=Path,
        default=DEFAULT_HOMOGRAPHY_JSON,
        help="JSON de homografia generado por el selector.",
    )
    parser.add_argument(
        "--no-homography",
        action="store_true",
        help="Ignora la homografia aunque el JSON exista.",
    )
    return parser.parse_args()


def pick_video_file() -> Path | None:
    root = Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askopenfilename(
        title="Selecciona un video",
        initialdir=str(DEFAULT_VIDEO_DIR),
        filetypes=[
            ("Videos", "*.mkv *.mp4 *.avi *.mov *.m4v"),
            ("Todos los archivos", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def resize_for_processing(frame: np.ndarray, process_width: int) -> tuple[np.ndarray, float]:
    scale = process_width / frame.shape[1]
    new_size = (process_width, int(round(frame.shape[0] * scale)))
    resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def norm_roi_to_px(
    frame_shape: tuple[int, int, int], roi_norm: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x0 = int(round(roi_norm[0] * width))
    y0 = int(round(roi_norm[1] * height))
    x1 = int(round(roi_norm[2] * width))
    y1 = int(round(roi_norm[3] * height))
    x0 = max(0, min(x0, width - 2))
    y0 = max(0, min(y0, height - 2))
    x1 = max(x0 + 1, min(x1, width - 1))
    y1 = max(y0 + 1, min(y1, height - 1))
    return x0, y0, x1, y1


def build_edge_response(gray_roi: np.ndarray) -> np.ndarray:
    grad_y = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(grad_y)


class LineDetector:
    def __init__(
        self,
        config: DetectorConfig,
        homography_matrix: np.ndarray | None = None,
        homography_output_size: tuple[int, int] | None = None,
    ) -> None:
        self.config = config
        self.homography_matrix = homography_matrix
        self.homography_output_size = homography_output_size
        self.previous_line: tuple[float, float] | None = None
        self.apply_homography = (
            homography_matrix is not None and homography_output_size is not None
        )
        self.roi_norm = (
            config.roi_norm_rectified if self.apply_homography else config.roi_norm_original
        )
        self.search_band = (
            config.search_band_rectified
            if self.apply_homography
            else config.search_band_original
        )

    def reset(self) -> None:
        self.previous_line = None

    def maybe_rectify(self, frame: np.ndarray) -> np.ndarray:
        if not self.apply_homography:
            return frame
        return cv2.warpPerspective(frame, self.homography_matrix, self.homography_output_size)

    def default_search_band(self) -> tuple[float, float]:
        if self.search_band is not None:
            return self.search_band
        if self.config.edge_mode == "top":
            return (0.02, 0.55)
        return (0.45, 0.98)

    def detect(self, frame: np.ndarray) -> dict[str, object]:
        rectified = self.maybe_rectify(frame)
        resized, scale = resize_for_processing(rectified, self.config.process_width)
        x0, y0, x1, y1 = norm_roi_to_px(resized.shape, self.roi_norm)
        roi = resized[y0:y1, x0:x1]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip, tileGridSize=(8, 8)
        ).apply(gray)
        gray = cv2.GaussianBlur(gray, self.config.pre_sobel_blur_ksize, 0)
        edge_response = build_edge_response(gray)

        height, width = edge_response.shape
        band_norm = self.default_search_band()
        band_start = int(round(band_norm[0] * height))
        band_end = int(round(band_norm[1] * height))
        band_start = max(0, min(band_start, height - 2))
        band_end = max(band_start + 1, min(band_end, height))

        sample_x = np.linspace(
            self.config.sample_margin_x,
            width - self.config.sample_margin_x - 1,
            self.config.num_sample_points,
        ).astype(int)
        sample_y = []
        sample_scores = []

        for x in sample_x:
            if self.previous_line is None:
                search_start, search_end = band_start, band_end
            else:
                pred_y = int(round(self.previous_line[0] * x + self.previous_line[1]))
                search_start = max(band_start, pred_y - self.config.tracking_radius_y)
                search_end = min(band_end, pred_y + self.config.tracking_radius_y + 1)
                if search_end <= search_start:
                    search_start, search_end = band_start, band_end

            stripe = edge_response[
                search_start:search_end,
                max(0, x - self.config.column_half_width) : min(
                    width, x + self.config.column_half_width + 1
                ),
            ]
            point_profile = smooth_1d(
                stripe.mean(axis=1), self.config.point_profile_smooth
            )
            rel_y = int(np.argmax(point_profile))
            y = search_start + rel_y
            score = float(point_profile[rel_y])
            sample_y.append(y)
            sample_scores.append(score)

        sample_y_arr = np.asarray(sample_y, dtype=np.float32)
        sample_scores_arr = np.asarray(sample_scores, dtype=np.float32)
        score_threshold = np.percentile(
            sample_scores_arr, self.config.score_keep_percentile
        )
        score_mask = sample_scores_arr >= score_threshold

        overlay = resized.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)

        result: dict[str, object] = {
            "overlay": overlay,
            "confidence": 0.0,
            "crm_px": float("inf"),
            "is_valid": False,
            "line_roi": None,
            "left_y_full": float("nan"),
            "right_y_full": float("nan"),
        }

        if int(score_mask.sum()) < self.config.min_inliers:
            self.previous_line = None
            return result

        median_y = float(np.median(sample_y_arr[score_mask]))
        cluster_mask = score_mask & (
            np.abs(sample_y_arr - median_y) <= self.config.cluster_tol_y
        )
        if int(cluster_mask.sum()) < self.config.min_inliers:
            self.previous_line = None
            return result

        slope, intercept = np.polyfit(sample_x[cluster_mask], sample_y_arr[cluster_mask], 1)
        pred = slope * sample_x + intercept
        residual = np.abs(sample_y_arr - pred)
        inlier_mask = cluster_mask & (residual <= self.config.line_inlier_tol)
        if int(inlier_mask.sum()) < self.config.min_inliers:
            self.previous_line = None
            return result

        slope, intercept = np.polyfit(sample_x[inlier_mask], sample_y_arr[inlier_mask], 1)
        pred_inliers = slope * sample_x[inlier_mask] + intercept
        crm_px = float(np.sqrt(np.mean((sample_y_arr[inlier_mask] - pred_inliers) ** 2)))
        confidence = float(inlier_mask.sum()) / float(len(sample_x))
        confidence *= float(np.median(sample_scores_arr[inlier_mask])) / (
            float(sample_scores_arr.mean()) + 1e-6
        )
        is_valid = (
            abs(float(slope)) <= self.config.max_abs_slope
            and confidence >= self.config.min_confidence
            and crm_px <= self.config.max_crm_px
        )

        line_color = (0, 0, 255) if is_valid else (0, 165, 255)
        for x, y, is_inlier in zip(sample_x, sample_y_arr.astype(int), inlier_mask):
            color = (0, 255, 0) if bool(is_inlier) else (0, 165, 255)
            cv2.circle(overlay, (x0 + int(x), y0 + int(y)), 3, color, -1)

        left_abs = y0 + (slope * 0.0 + intercept)
        right_abs = y0 + (slope * (width - 1) + intercept)
        cv2.line(
            overlay,
            (x0, int(round(left_abs))),
            (x1, int(round(right_abs))),
            line_color,
            3,
        )

        label = "linea | conf={:.2f} | crm={:.2f}px".format(confidence, crm_px)
        cv2.putText(
            overlay,
            label,
            (x0 + 6, max(24, y0 + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            line_color,
            2,
            cv2.LINE_AA,
        )

        if is_valid:
            self.previous_line = (float(slope), float(intercept))
        else:
            self.previous_line = None

        result.update(
            {
                "overlay": overlay,
                "confidence": float(confidence),
                "crm_px": float(crm_px),
                "is_valid": bool(is_valid),
                "line_roi": (float(slope), float(intercept)),
                "left_y_full": float(left_abs / scale) if is_valid else float("nan"),
                "right_y_full": float(right_abs / scale) if is_valid else float("nan"),
            }
        )
        return result


def load_homography(
    json_path: Path, ignore: bool = False
) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    if ignore or not json_path.exists():
        return None, None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["homography_matrix"], dtype=np.float32)
    output_size = tuple(int(v) for v in payload["output_size"])
    return matrix, output_size


def run_video_app(video_path: Path, detector: LineDetector) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    delay_ms = max(1, int(round(1000.0 / fps)))
    paused = False
    frame_index = 0
    window_name = "Line Measure App"

    print(f"Video: {video_path}")
    print("Controles: espacio pausa, n avanza un frame, r reinicia tracking, q salir")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            debug = detector.detect(frame)
            overlay = debug["overlay"]
            status = "OK" if debug["is_valid"] else "NO VALIDA"
            footer = "frame={} | {} | yL={:.1f} | yR={:.1f}".format(
                frame_index,
                status,
                debug["left_y_full"] if not np.isnan(debug["left_y_full"]) else -1.0,
                debug["right_y_full"] if not np.isnan(debug["right_y_full"]) else -1.0,
            )
            cv2.putText(
                overlay,
                footer,
                (15, overlay.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, overlay)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            detector.reset()
        elif key == ord("n") and paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            debug = detector.detect(frame)
            overlay = debug["overlay"]
            cv2.imshow(window_name, overlay)

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    video_path = args.video if args.video is not None else pick_video_file()
    if video_path is None:
        print("No se selecciono ningun video.")
        return
    if not video_path.exists():
        raise RuntimeError(f"No existe el video: {video_path}")

    homography_matrix, homography_output_size = load_homography(
        args.homography_json, ignore=args.no_homography
    )
    detector = LineDetector(
        config=DetectorConfig(),
        homography_matrix=homography_matrix,
        homography_output_size=homography_output_size,
    )
    run_video_app(video_path, detector)


if __name__ == "__main__":
    main()
