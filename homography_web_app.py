"""
Homography + YOLO annotation web app.

Usage:
    python homography_web_app.py --video "path/to/video.mkv" --second 155.0
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = Path(r"C:\Users\luis_\Downloads\20260724_10")
DEFAULT_VIDEO = DEFAULT_VIDEO_DIR / "20260724_100105_6439.mkv"
DEFAULT_SECOND = 30.0
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset_pieces"
DEFAULT_PIECE_MODEL = (
    PROJECT_ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v3" / "weights" / "best.pt"
)
DEFAULT_PIECE_ENGINE = DEFAULT_PIECE_MODEL.with_suffix(".engine")
PREVIOUS_PIECE_MODEL = (
    PROJECT_ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v2" / "weights" / "best.pt"
)
OLDER_PIECE_MODEL = (
    PROJECT_ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v1" / "weights" / "best.pt"
)
DEFAULT_LEGACY_MODEL = (
    PROJECT_ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_tubos_v1" / "weights" / "best.pt"
)
DEFAULT_MODEL = next(
    (
        path
        for path in (
            DEFAULT_PIECE_ENGINE,
            DEFAULT_PIECE_MODEL,
            PREVIOUS_PIECE_MODEL,
            OLDER_PIECE_MODEL,
            DEFAULT_LEGACY_MODEL,
        )
        if path.exists()
    ),
    DEFAULT_PIECE_MODEL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path)
    source.add_argument("--video-dir", type=Path)
    parser.add_argument(
        "--fallback-video-dir",
        type=Path,
        help="Use this directory when the primary directory has no complete videos.",
    )
    parser.add_argument("--second", type=float, default=DEFAULT_SECOND)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--legacy-candidates-dir",
        type=Path,
        help="Optional legacy dataset whose frames should appear as pending annotation candidates.",
    )
    parser.add_argument(
        "--candidate-samples-per-video",
        type=int,
        default=12,
        help="Evenly spaced pending annotation candidates generated for each playlist video.",
    )
    parser.add_argument(
        "--latest-video-format-only",
        action="store_true",
        help="Use only videos matching the resolution and FPS of the newest source video.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        default=os.environ.get("TX2_YOLO_DEVICE", "auto"),
        help="YOLO inference device: auto, cpu, cuda or cuda:N.",
    )
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    if args.video is None and args.video_dir is None:
        args.video_dir = DEFAULT_VIDEO_DIR
    return args


def open_video(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    return cap


@lru_cache(maxsize=512)
def video_meta(video_path: Path) -> dict:
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "fps": fps,
        "total_frames": total,
        "duration_sec": total / fps if fps else 0,
        "width": width,
        "height": height,
    }


def video_format_signature(meta: dict) -> tuple[int, int, int]:
    return (
        int(meta["width"]),
        int(meta["height"]),
        int(round(float(meta["fps"]))),
    )


def complete_video_meta(video_path: Path) -> dict:
    meta = video_meta(video_path)
    if (
        int(meta["total_frames"]) <= 0
        or int(meta["width"]) <= 0
        or int(meta["height"]) <= 0
        or float(meta["fps"]) <= 0
    ):
        raise RuntimeError(f"El video todavia no esta completo: {video_path}")
    return meta


def discover_video_paths(args: argparse.Namespace) -> list[Path]:
    video_dir = getattr(args, "video_dir", None)
    video = getattr(args, "video", None)
    if video_dir is not None:
        extensions = {".mkv", ".mp4", ".avi", ".mov", ".m4v"}
        paths = sorted(
            path
            for path in video_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        raw_paths = [path for path in paths if path.stem.lower().endswith("_raw")]
        if raw_paths:
            paths = raw_paths
        if not paths:
            raise RuntimeError(f"No se encontraron videos en: {video_dir}")
        return paths
    return [Path(video or DEFAULT_VIDEO)]


def latest_video_format_paths(video_paths: list[Path]) -> list[Path]:
    if not video_paths:
        return []
    readable = []
    for path in video_paths:
        try:
            readable.append((path, complete_video_meta(path)))
        except RuntimeError as exc:
            print(f"Advertencia: se omite video incompleto o ilegible: {exc}")
    if not readable:
        raise RuntimeError("No hay videos completos disponibles para la herramienta")
    latest_meta = readable[-1][1]
    latest_signature = video_format_signature(latest_meta)
    return [
        path
        for path, meta in readable
        if video_format_signature(meta) == latest_signature
    ]


def selected_video_paths(args: argparse.Namespace) -> list[Path]:
    video_dir = getattr(args, "video_dir", None)
    fallback_dir = getattr(args, "fallback_video_dir", None)
    latest_only = bool(getattr(args, "latest_video_format_only", False))
    if video_dir is None or fallback_dir is None:
        paths = discover_video_paths(args)
        return latest_video_format_paths(paths) if latest_only else paths

    directories = [Path(video_dir)]
    if Path(fallback_dir).resolve() != Path(video_dir).resolve():
        directories.append(Path(fallback_dir))

    errors = []
    for position, directory in enumerate(directories):
        try:
            paths = discover_video_paths(
                argparse.Namespace(video_dir=directory, video=None)
            )
            selected = latest_video_format_paths(paths) if latest_only else paths
            if position > 0:
                print(
                    "Advertencia: no hay clips Live completos; "
                    f"se usa el respaldo: {directory}"
                )
            return selected
        except RuntimeError as exc:
            errors.append(f"{directory}: {exc}")

    raise RuntimeError("; ".join(errors))


def build_video_playlist(video_paths: list[Path]) -> dict:
    if not video_paths:
        raise RuntimeError("La playlist no contiene videos")

    segments = []
    start_frame = 0
    expected = None
    for path in video_paths:
        try:
            meta = complete_video_meta(path)
        except RuntimeError as exc:
            print(f"Advertencia: se omite video incompleto o ilegible: {exc}")
            continue
        signature = video_format_signature(meta)
        if expected is None:
            expected = signature
        elif signature != expected:
            raise RuntimeError(
                f"El video {path.name} no coincide con la playlist "
                f"({signature[0]}x{signature[1]} @ {signature[2]} FPS nominales)"
            )
        total_frames = int(meta["total_frames"])
        segments.append(
            {
                "path": path,
                "name": path.name,
                "stem": path.stem,
                "start_frame": start_frame,
                "end_frame": start_frame + total_frames,
                "total_frames": total_frames,
                "duration_sec": float(meta["duration_sec"]),
            }
        )
        start_frame += total_frames

    if expected is None:
        raise RuntimeError("La playlist no contiene videos completos")
    fps = expected[2]
    return {
        "segments": segments,
        "fps": fps,
        "total_frames": start_frame,
        "duration_sec": start_frame / fps if fps else 0.0,
        "width": expected[0],
        "height": expected[1],
    }


def public_playlist_meta(playlist: dict) -> dict:
    return {
        "fps": float(playlist["fps"]),
        "total_frames": int(playlist["total_frames"]),
        "duration_sec": float(playlist["duration_sec"]),
        "width": int(playlist["width"]),
        "height": int(playlist["height"]),
        "video_count": len(playlist["segments"]),
        "videos": [
            {
                "name": segment["name"],
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
                "total_frames": int(segment["total_frames"]),
                "duration_sec": float(segment["duration_sec"]),
            }
            for segment in playlist["segments"]
        ],
    }


def playlist_segment_for_frame(playlist: dict, frame_idx: int) -> tuple[dict, int, int]:
    total = int(playlist["total_frames"])
    global_idx = max(0, min(int(frame_idx), max(0, total - 1)))
    for segment in playlist["segments"]:
        if global_idx < int(segment["end_frame"]):
            return segment, global_idx - int(segment["start_frame"]), global_idx
    segment = playlist["segments"][-1]
    return segment, int(segment["total_frames"]) - 1, global_idx


def playlist_source_for_frame(playlist: dict, frame_idx: int) -> dict:
    segment, source_frame_idx, _global_idx = playlist_segment_for_frame(playlist, frame_idx)
    fps = float(playlist["fps"])
    return {
        "video": str(segment["path"]),
        "video_name": str(segment["name"]),
        "video_stem": str(segment["stem"]),
        "source_frame_idx": int(source_frame_idx),
        "source_time_sec": source_frame_idx / fps,
    }


def read_playlist_frame_by_index(
    playlist: dict,
    frame_idx: int,
) -> tuple[np.ndarray, int, float, dict]:
    segment, source_frame_idx, global_idx = playlist_segment_for_frame(playlist, frame_idx)
    frame, source_frame_idx, source_time_sec = read_frame_by_index(
        Path(segment["path"]),
        source_frame_idx,
    )
    fps = float(playlist["fps"])
    source = playlist_source_for_frame(playlist, global_idx)
    source["source_frame_idx"] = int(source_frame_idx)
    source["source_time_sec"] = float(source_time_sec)
    return frame, global_idx, global_idx / fps, source


def read_active_frame_by_index(frame_idx: int) -> tuple[np.ndarray, int, float, dict]:
    if _video_playlist is None:
        raise RuntimeError("La playlist de videos no esta inicializada")
    return read_playlist_frame_by_index(_video_playlist, frame_idx)


def read_active_frame_by_second(second: float) -> tuple[np.ndarray, int, float, dict]:
    if _video_playlist is None:
        raise RuntimeError("La playlist de videos no esta inicializada")
    frame_idx = int(round(max(0.0, second) * float(_video_playlist["fps"])))
    return read_playlist_frame_by_index(_video_playlist, frame_idx)


def read_frame_by_second(video_path: Path, second: float) -> tuple[np.ndarray, int, float]:
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = max(0, min(int(round(second * fps)), max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer frame en {second:.3f}s")
    return frame, frame_idx, frame_idx / fps


def read_frame_by_index(video_path: Path, frame_idx: int) -> tuple[np.ndarray, int, float]:
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = max(0, min(int(frame_idx), max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer frame {frame_idx}")
    return frame, frame_idx, frame_idx / fps


def load_reference_image(args: argparse.Namespace) -> tuple[np.ndarray, str, int, float]:
    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise RuntimeError(f"No se pudo abrir: {args.image}")
        return image, str(args.image), 0, 0.0
    frame, frame_idx, time_sec, source = read_active_frame_by_second(args.second)
    label = (
        f"{source['video']} @ {source['source_time_sec']:.3f}s "
        f"(playlist {time_sec:.3f}s)"
    )
    return frame, label, frame_idx, time_sec


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def rect_size_from_ordered(ordered: np.ndarray) -> tuple[int, int]:
    tl, tr, br, bl = ordered
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    return max(width, 2), max(height, 2)


def normalize_roi_margins(
    roi_margins: dict | None,
    base_w: int,
    base_h: int,
    expand_pct: float = 0.0,
) -> dict:
    if roi_margins:
        margins = {
            "left": float(roi_margins.get("left", 0.0)),
            "right": float(roi_margins.get("right", 0.0)),
            "top": float(roi_margins.get("top", 0.0)),
            "bottom": float(roi_margins.get("bottom", 0.0)),
        }
    else:
        pct = max(0.0, min(float(expand_pct), 80.0)) / 100.0
        margins = {
            "left": base_w * pct,
            "right": base_w * pct,
            "top": base_h * pct,
            "bottom": base_h * pct,
        }
    return {key: max(0.0, value) for key, value in margins.items()}


def work_roi_from_margins(
    ordered: np.ndarray,
    roi_margins: dict,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict]:
    base_w, base_h = rect_size_from_ordered(ordered)
    base_dst = np.array(
        [[0, 0], [base_w - 1, 0], [base_w - 1, base_h - 1], [0, base_h - 1]],
        dtype=np.float32,
    )
    base_matrix = cv2.getPerspectiveTransform(ordered, base_dst)
    inv_matrix = np.linalg.inv(base_matrix)

    left = float(roi_margins["left"])
    right = float(roi_margins["right"])
    top = float(roi_margins["top"])
    bottom = float(roi_margins["bottom"])
    work_rect = {
        "x0": -left,
        "y0": -top,
        "x1": float(base_w - 1) + right,
        "y1": float(base_h - 1) + bottom,
    }
    rect_points = np.array(
        [
            [work_rect["x0"], work_rect["y0"]],
            [work_rect["x1"], work_rect["y0"]],
            [work_rect["x1"], work_rect["y1"]],
            [work_rect["x0"], work_rect["y1"]],
        ],
        dtype=np.float32,
    )
    warp_points = cv2.perspectiveTransform(rect_points.reshape(-1, 1, 2), inv_matrix).reshape(-1, 2)
    width = int(round(work_rect["x1"] - work_rect["x0"] + 1))
    height = int(round(work_rect["y1"] - work_rect["y0"] + 1))
    return warp_points.astype(np.float32), base_matrix, (max(width, 2), max(height, 2)), work_rect


def compute_warp(
    image: np.ndarray,
    pts: list,
    dest_w: int = 0,
    dest_h: int = 0,
    expand_pct: float = 0.0,
    warp_pts: list | None = None,
    roi_margins: dict | None = None,
    metric_scale_y: float = 1.0,
):
    ordered = order_points(np.asarray(pts, dtype=np.float32))
    base_w, base_h = rect_size_from_ordered(ordered)
    margins = normalize_roi_margins(roi_margins, base_w, base_h, expand_pct)
    warp_points, base_matrix, margin_size, work_rect = work_roi_from_margins(ordered, margins)
    width = margin_size[0] if dest_w <= 0 else dest_w
    metric_scale_y = normalize_metric_scale_y(metric_scale_y)
    height = int(round(margin_size[1] * metric_scale_y)) if dest_h <= 0 else dest_h
    width, height = max(width, 2), max(height, 2)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(warp_points, dst)
    warp = cv2.warpPerspective(image, matrix, (width, height))
    return warp, ordered, warp_points, dst, matrix, (width, height), base_matrix, (base_w, base_h), margins, work_rect


def img_to_b64(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen")
    return base64.b64encode(buf).decode("ascii")


def homography_json_path() -> Path:
    return _args.output_dir / "homography_selection.json"


def measurement_json_path() -> Path:
    return _args.output_dir / "table_measurement_calibration.json"


def player_capture_json_path() -> Path:
    return _args.output_dir / "player_measurement_captures.json"


def player_capture_dir() -> Path:
    return _args.output_dir / "player_measurement_captures"


MEASUREMENT_REFERENCE_OFFSET_IN = 475.0 + (1.0 / 16.0)
EXCLUSION_ZONE_MAX_BOX_OVERLAP = 0.20
LENS_CORRECTION_MODEL = "opencv_radial_k1"
LENS_K1_MIN = -0.35
LENS_K1_MAX = 0.35
LENS_METRIC_SCALE_MIN = 0.75
LENS_METRIC_SCALE_MAX = 1.25
LENS_AUTO_FIT_MIN_SEGMENTS = 3
SPATIAL_SCALE_MAP_VERSION = 1
SPATIAL_SCALE_AXIS_ALIGNMENT = 0.85
SPATIAL_SCALE_IDW_POWER = 2.0


def normalize_lens_correction(value: dict | None) -> dict:
    value = value if isinstance(value, dict) else {}
    k1 = float(value.get("k1", 0.0) or 0.0)
    k1 = max(LENS_K1_MIN, min(LENS_K1_MAX, k1))
    center_x = max(0.25, min(0.75, float(value.get("center_x", 0.5) or 0.5)))
    center_y = max(0.25, min(0.75, float(value.get("center_y", 0.5) or 0.5)))
    focal_ratio = max(0.25, min(1.5, float(value.get("focal_ratio", 0.5) or 0.5)))
    enabled = bool(value.get("enabled", abs(k1) > 1e-7)) and abs(k1) > 1e-7
    return {
        "enabled": enabled,
        "model": LENS_CORRECTION_MODEL,
        "k1": k1,
        "center_x": center_x,
        "center_y": center_y,
        "focal_ratio": focal_ratio,
    }


def normalize_metric_scale_y(value: float | int | None) -> float:
    scale = float(value or 1.0)
    return max(LENS_METRIC_SCALE_MIN, min(LENS_METRIC_SCALE_MAX, scale))


def lens_camera_matrix(image_shape: tuple[int, int], correction: dict | None) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    config = normalize_lens_correction(correction)
    focal = max(width, height) * float(config["focal_ratio"])
    return np.array(
        [
            [focal, 0.0, (width - 1) * float(config["center_x"])],
            [0.0, focal, (height - 1) * float(config["center_y"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def lens_distortion_coefficients(correction: dict | None) -> np.ndarray:
    config = normalize_lens_correction(correction)
    return np.array([float(config["k1"]), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def undistort_source_points(
    points: np.ndarray | list,
    image_shape: tuple[int, int],
    correction: dict | None,
) -> np.ndarray:
    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    config = normalize_lens_correction(correction)
    if not config["enabled"]:
        return source.reshape(-1, 2).copy()
    camera = lens_camera_matrix(image_shape, config)
    return cv2.undistortPoints(
        source,
        camera,
        lens_distortion_coefficients(config),
        P=camera,
    ).reshape(-1, 2)


def distort_source_points(
    points: np.ndarray | list,
    image_shape: tuple[int, int],
    correction: dict | None,
) -> np.ndarray:
    corrected = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    config = normalize_lens_correction(correction)
    if not config["enabled"]:
        return corrected.astype(np.float32)
    camera = lens_camera_matrix(image_shape, config)
    fx, fy = float(camera[0, 0]), float(camera[1, 1])
    cx, cy = float(camera[0, 2]), float(camera[1, 2])
    normalized = np.empty_like(corrected)
    normalized[:, 0] = (corrected[:, 0] - cx) / fx
    normalized[:, 1] = (corrected[:, 1] - cy) / fy
    radius_sq = np.sum(normalized * normalized, axis=1)
    factor = 1.0 + float(config["k1"]) * radius_sq
    distorted = np.empty_like(corrected)
    distorted[:, 0] = cx + normalized[:, 0] * factor * fx
    distorted[:, 1] = cy + normalized[:, 1] * factor * fy
    return distorted.astype(np.float32)


def apply_lens_correction(image: np.ndarray, correction: dict | None) -> np.ndarray:
    config = normalize_lens_correction(correction)
    if not config["enabled"]:
        return image
    height, width = image.shape[:2]
    cache_key = (
        height,
        width,
        round(float(config["k1"]), 8),
        round(float(config["center_x"]), 6),
        round(float(config["center_y"]), 6),
        round(float(config["focal_ratio"]), 6),
    )
    maps = _lens_map_cache.get(cache_key)
    if maps is None:
        camera = lens_camera_matrix((height, width), config)
        maps = cv2.initUndistortRectifyMap(
            camera,
            lens_distortion_coefficients(config),
            None,
            camera,
            (width, height),
            cv2.CV_32FC1,
        )
        _lens_map_cache.clear()
        _lens_map_cache[cache_key] = maps
    return cv2.remap(
        image,
        maps[0],
        maps[1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _scale_axis_knots(samples: list[dict], coordinate: str) -> list[dict]:
    ordered = sorted(samples, key=lambda sample: float(sample[coordinate]))
    groups: list[list[dict]] = []
    for sample in ordered:
        if not groups or abs(float(sample[coordinate]) - float(groups[-1][-1][coordinate])) > 1.0:
            groups.append([sample])
        else:
            groups[-1].append(sample)
    return [
        {
            "coordinate": float(np.mean([float(sample[coordinate]) for sample in group])),
            "inch_per_px": float(np.mean([float(sample["inch_per_px"]) for sample in group])),
            "sample_count": len(group),
        }
        for group in groups
    ]


def _build_scale_axis(
    axis: str,
    samples: list[dict],
    image_width: int,
    image_height: int,
) -> dict:
    if not samples:
        return {
            "axis": axis,
            "mode": "global",
            "samples": [],
            "knots": [],
            "hull": [],
            "sample_count": 0,
            "coverage": None,
            "minimum_inch_per_px": None,
            "maximum_inch_per_px": None,
        }

    xs = np.array([float(sample["x"]) for sample in samples], dtype=np.float64)
    ys = np.array([float(sample["y"]) for sample in samples], dtype=np.float64)
    values = np.array([float(sample["inch_per_px"]) for sample in samples], dtype=np.float64)
    x_span_ratio = float(np.ptp(xs) / max(1, image_width))
    y_span_ratio = float(np.ptp(ys) / max(1, image_height))
    if len(samples) >= 4 and x_span_ratio >= 0.15 and y_span_ratio >= 0.15:
        mode = "idw_2d"
        coordinate = None
        knots = []
        hull = cv2.convexHull(
            np.array([[sample["x"], sample["y"]] for sample in samples], dtype=np.float32)
        ).reshape(-1, 2).tolist()
    else:
        coordinate = "x" if x_span_ratio >= y_span_ratio else "y"
        mode = f"linear_{coordinate}"
        knots = _scale_axis_knots(samples, coordinate)
        hull = []

    confidence = "high" if len(samples) >= 8 else ("medium" if len(samples) >= 4 else "low")
    return {
        "axis": axis,
        "mode": mode,
        "coordinate": coordinate,
        "samples": samples,
        "knots": knots,
        "hull": hull,
        "sample_count": len(samples),
        "confidence": confidence,
        "coverage": {
            "x_min": float(np.min(xs)),
            "x_max": float(np.max(xs)),
            "y_min": float(np.min(ys)),
            "y_max": float(np.max(ys)),
            "x_span_ratio": x_span_ratio,
            "y_span_ratio": y_span_ratio,
        },
        "minimum_inch_per_px": float(np.min(values)),
        "maximum_inch_per_px": float(np.max(values)),
        "mean_inch_per_px": float(np.mean(values)),
    }


def build_spatial_scale_map(
    segments: list[dict],
    image_width: int,
    image_height: int,
) -> dict:
    axis_samples: dict[str, list[dict]] = {"x": [], "y": []}
    ignored = 0
    for index, segment in enumerate(valid_measurement_segments(segments)):
        points = segment["points"]
        vector = points[1] - points[0]
        length = float(np.linalg.norm(vector))
        if length <= 0.0:
            continue
        alignment_x = abs(float(vector[0])) / length
        alignment_y = abs(float(vector[1])) / length
        if alignment_x >= SPATIAL_SCALE_AXIS_ALIGNMENT:
            axis = "x"
            alignment = alignment_x
        elif alignment_y >= SPATIAL_SCALE_AXIS_ALIGNMENT:
            axis = "y"
            alignment = alignment_y
        else:
            ignored += 1
            continue
        axis_samples[axis].append(
            {
                "source_index": index,
                "x": float(np.mean(points[:, 0])),
                "y": float(np.mean(points[:, 1])),
                "inch_per_px": float(segment["inches"]) / length,
                "px_per_in": length / float(segment["inches"]),
                "alignment": alignment,
            }
        )

    return {
        "version": SPATIAL_SCALE_MAP_VERSION,
        "method": "axis_local_interpolation",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "idw_power": SPATIAL_SCALE_IDW_POWER,
        "axis_alignment_min": SPATIAL_SCALE_AXIS_ALIGNMENT,
        "axes": {
            axis: _build_scale_axis(axis, samples, image_width, image_height)
            for axis, samples in axis_samples.items()
        },
        "ignored_diagonal_segments": ignored,
    }


def _point_inside_sample_hull(samples: list[dict], x: float, y: float) -> bool:
    if len(samples) < 3:
        return False
    points = np.array([[sample["x"], sample["y"]] for sample in samples], dtype=np.float32)
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    return cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0


def spatial_scale_at(
    calibration: dict,
    x: float,
    y: float,
    axis: str = "y",
) -> dict:
    fallback = calibration.get("inch_per_px")
    fallback_value = float(fallback) if fallback is not None and float(fallback) > 0.0 else None
    scale_map = calibration.get("scale_map") if isinstance(calibration, dict) else None
    axis_data = (
        scale_map.get("axes", {}).get(axis)
        if isinstance(scale_map, dict)
        else None
    )
    samples = axis_data.get("samples", []) if isinstance(axis_data, dict) else []
    if not samples:
        return {
            "inch_per_px": fallback_value,
            "source": "global" if fallback_value is not None else "missing",
            "extrapolated": False,
            "sample_count": 0,
        }

    mode = str(axis_data.get("mode", "global"))
    if mode.startswith("linear_"):
        coordinate_name = str(axis_data.get("coordinate") or mode.replace("linear_", ""))
        coordinate = float(x if coordinate_name == "x" else y)
        knots = axis_data.get("knots") or _scale_axis_knots(samples, coordinate_name)
        knot_positions = np.array([float(knot["coordinate"]) for knot in knots], dtype=np.float64)
        knot_values = np.array([float(knot["inch_per_px"]) for knot in knots], dtype=np.float64)
        if len(knots) == 1:
            value = float(knot_values[0])
            extrapolated = True
        else:
            value = float(np.interp(coordinate, knot_positions, knot_values))
            extrapolated = coordinate < float(knot_positions[0]) or coordinate > float(knot_positions[-1])
        return {
            "inch_per_px": value,
            "source": mode,
            "extrapolated": extrapolated,
            "sample_count": len(samples),
        }

    image_width = max(1.0, float(scale_map.get("image_width", 1)))
    image_height = max(1.0, float(scale_map.get("image_height", 1)))
    distances = np.array(
        [
            np.hypot(
                (float(x) - float(sample["x"])) / image_width,
                (float(y) - float(sample["y"])) / image_height,
            )
            for sample in samples
        ],
        dtype=np.float64,
    )
    inside_hull = _point_inside_sample_hull(samples, x, y)
    if not inside_hull:
        nearest = int(np.argmin(distances))
        return {
            "inch_per_px": float(samples[nearest]["inch_per_px"]),
            "source": "nearest_2d",
            "extrapolated": True,
            "sample_count": len(samples),
        }
    exact = np.flatnonzero(distances < 1e-9)
    if exact.size:
        value = float(np.mean([float(samples[index]["inch_per_px"]) for index in exact]))
    else:
        power = float(scale_map.get("idw_power", SPATIAL_SCALE_IDW_POWER))
        weights = 1.0 / np.maximum(distances, 1e-9) ** power
        values = np.array([float(sample["inch_per_px"]) for sample in samples], dtype=np.float64)
        value = float(np.sum(weights * values) / np.sum(weights))
    return {
        "inch_per_px": value,
        "source": "idw_2d",
        "extrapolated": False,
        "sample_count": len(samples),
    }


def integrate_vertical_scale(
    calibration: dict,
    x: float,
    y_start: float,
    y_end: float,
) -> dict:
    delta_px = float(y_end) - float(y_start)
    if abs(delta_px) < 1e-9:
        scale = spatial_scale_at(calibration, x, y_start, "y")
        return {
            "distance_in": 0.0,
            "mean_inch_per_px": scale["inch_per_px"],
            "extrapolated": bool(scale["extrapolated"]),
            "coverage_ratio": 0.0 if scale["extrapolated"] else 1.0,
            "source": scale["source"],
            "sample_count": scale["sample_count"],
        }

    step_count = max(8, min(128, int(np.ceil(abs(delta_px) / 12.0))))
    edges = np.linspace(float(y_start), float(y_end), step_count + 1)
    total = 0.0
    covered = 0
    sources = set()
    sample_count = 0
    for index in range(step_count):
        midpoint = (edges[index] + edges[index + 1]) / 2.0
        scale = spatial_scale_at(calibration, x, midpoint, "y")
        value = scale["inch_per_px"]
        if value is None:
            return {
                "distance_in": None,
                "mean_inch_per_px": None,
                "extrapolated": False,
                "coverage_ratio": 0.0,
                "source": "missing",
                "sample_count": 0,
            }
        total += (edges[index + 1] - edges[index]) * float(value)
        covered += 0 if scale["extrapolated"] else 1
        sources.add(str(scale["source"]))
        sample_count = max(sample_count, int(scale["sample_count"]))
    return {
        "distance_in": total,
        "mean_inch_per_px": abs(total / delta_px),
        "extrapolated": covered < step_count,
        "coverage_ratio": covered / step_count,
        "source": "+".join(sorted(sources)),
        "sample_count": sample_count,
    }


def load_measurement_calibration() -> dict:
    path = measurement_json_path()
    if not path.exists():
        return {
            "segments": [],
            "inch_per_px": None,
            "reference_y": None,
            "reference_offset_in": MEASUREMENT_REFERENCE_OFFSET_IN,
            "exclusion_zones": [],
            "exclusion_max_box_overlap": EXCLUSION_ZONE_MAX_BOX_OVERLAP,
            "scale_map": build_spatial_scale_map([], 0, 0),
            "path": str(path),
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("reference_offset_in", MEASUREMENT_REFERENCE_OFFSET_IN)
    data.setdefault("exclusion_zones", [])
    data.setdefault("exclusion_max_box_overlap", EXCLUSION_ZONE_MAX_BOX_OVERLAP)
    if not isinstance(data.get("scale_map"), dict):
        data["scale_map"] = build_spatial_scale_map(
            data.get("segments") or [],
            int(data.get("img_w", 0) or 0),
            int(data.get("img_h", 0) or 0),
        )
    data["path"] = str(path)
    return data


def load_player_captures() -> list[dict]:
    path = player_capture_json_path()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    captures = data if isinstance(data, list) else data.get("captures", [])
    return sorted(captures, key=lambda item: int(item.get("frame_idx", 0)))


def save_player_captures(captures: list[dict]) -> None:
    path = player_capture_json_path()
    _args.output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"captures": captures}, indent=2), encoding="utf-8")


def measurement_from_sobel(sobel: dict, calibration: dict, width: int) -> dict | None:
    if not sobel or not sobel.get("line"):
        return None
    reference_y = calibration.get("reference_y")
    if reference_y is None:
        return None
    line = sobel["line"]
    x1, y1 = float(line["x1"]), float(line["y1"])
    x2, y2 = float(line["x2"]), float(line["y2"])
    x_mid = (x1 + x2) / 2.0
    if not np.isfinite(x_mid):
        x_mid = (width - 1) / 2.0
    if abs(x2 - x1) < 1e-6:
        y_mid = (y1 + y2) / 2.0
    else:
        t = (x_mid - x1) / (x2 - x1)
        y_mid = y1 + t * (y2 - y1)
    delta_px = y_mid - float(reference_y)
    spatial = integrate_vertical_scale(
        calibration,
        x_mid,
        float(reference_y),
        y_mid,
    )
    delta_in = spatial["distance_in"]
    if delta_in is None:
        return None
    offset_in = float(calibration.get("reference_offset_in", MEASUREMENT_REFERENCE_OFFSET_IN) or 0.0)
    measurement_in = delta_in + offset_in
    return {
        "x": x_mid,
        "line_y": y_mid,
        "reference_y": float(reference_y),
        "delta_px": delta_px,
        "delta_in": delta_in,
        "abs_delta_in": abs(delta_in),
        "reference_offset_in": offset_in,
        "measurement_in": measurement_in,
        "inch_per_px": spatial["mean_inch_per_px"],
        "scale_source": spatial["source"],
        "scale_sample_count": spatial["sample_count"],
        "scale_extrapolated": spatial["extrapolated"],
        "scale_coverage_ratio": spatial["coverage_ratio"],
    }


def rectified_line_to_original(
    line: dict | None,
    matrix: np.ndarray,
    lens_correction: dict | None = None,
    image_shape: tuple[int, int] | None = None,
) -> dict | None:
    if not line:
        return None
    try:
        points = np.array(
            [
                [[float(line["x1"]), float(line["y1"])]],
                [[float(line["x2"]), float(line["y2"])]],
            ],
            dtype=np.float32,
        )
        inverse = np.linalg.inv(matrix)
        mapped = cv2.perspectiveTransform(points, inverse).reshape(-1, 2)
        if image_shape is not None:
            mapped = distort_source_points(mapped, image_shape, lens_correction)
        return {
            "x1": float(mapped[0][0]),
            "y1": float(mapped[0][1]),
            "x2": float(mapped[1][0]),
            "y2": float(mapped[1][1]),
        }
    except Exception:
        return None


def mvp_original_overlay(
    sobel: dict,
    calibration: dict,
    matrix: np.ndarray,
    rect_width: int,
    lens_correction: dict | None = None,
    image_shape: tuple[int, int] | None = None,
) -> dict:
    reference_y = calibration.get("reference_y")
    reference_line = None
    if reference_y is not None:
        y = float(reference_y)
        reference_line = {"x1": 0.0, "y1": y, "x2": float(rect_width - 1), "y2": y}

    return {
        "front_line": rectified_line_to_original(
            (sobel or {}).get("line"),
            matrix,
            lens_correction,
            image_shape,
        ),
        "reference_line": rectified_line_to_original(
            reference_line,
            matrix,
            lens_correction,
            image_shape,
        ),
    }


def mvp_original_overlay_for_pieces(
    pieces: list[dict],
    calibration: dict,
    matrix: np.ndarray,
    rect_width: int,
    lens_correction: dict | None = None,
    image_shape: tuple[int, int] | None = None,
) -> dict:
    overlay = mvp_original_overlay(
        {},
        calibration,
        matrix,
        rect_width,
        lens_correction,
        image_shape,
    )
    piece_fronts = []
    for piece in pieces:
        sobel = piece.get("sobel") if isinstance(piece.get("sobel"), dict) else {}
        mapped_line = rectified_line_to_original(
            sobel.get("line"),
            matrix,
            lens_correction,
            image_shape,
        )
        if mapped_line is None:
            continue
        piece_fronts.append(
            {
                "piece_id": int(piece["piece_id"]),
                "line": mapped_line,
                "valid": bool(piece.get("valid")),
                "measurement": piece.get("measurement"),
            }
        )
    overlay["piece_fronts"] = piece_fronts
    overlay["front_line"] = next(
        (item["line"] for item in piece_fronts if item["valid"]),
        piece_fronts[0]["line"] if piece_fronts else None,
    )
    return overlay


def load_homography() -> tuple[np.ndarray, tuple[int, int], dict]:
    path = homography_json_path()
    if not path.exists():
        raise RuntimeError(f"No existe homografia guardada: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["lens_correction"] = normalize_lens_correction(data.get("lens_correction"))
    data["metric_scale_y"] = normalize_metric_scale_y(data.get("metric_scale_y", 1.0))
    matrix = np.array(data["homography_matrix"], dtype=np.float64)
    width, height = data["output_size"]
    return matrix, (int(width), int(height)), data


def apply_saved_homography(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    matrix, out_size, data = load_homography()
    corrected = apply_lens_correction(frame, data.get("lens_correction"))
    return cv2.warpPerspective(corrected, matrix, out_size), data


def valid_measurement_segments(segments: list[dict]) -> list[dict]:
    valid = []
    for segment in segments:
        try:
            inches = float(segment.get("inches", 0.0) or 0.0)
            points = np.array(
                [
                    [float(segment["x1"]), float(segment["y1"])],
                    [float(segment["x2"]), float(segment["y2"])],
                ],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if inches <= 0.0 or not np.all(np.isfinite(points)):
            continue
        if float(np.linalg.norm(points[1] - points[0])) < 2.0:
            continue
        valid.append({"points": points, "inches": inches, "source": segment})
    return valid


def candidate_homography_geometry(
    raw_selected_points: np.ndarray,
    image_shape: tuple[int, int],
    correction: dict,
    roi_margins: dict | None,
    expand_pct: float,
    metric_scale_y: float,
) -> dict:
    corrected_points = undistort_source_points(raw_selected_points, image_shape, correction)
    ordered = order_points(corrected_points)
    base_w, base_h = rect_size_from_ordered(ordered)
    margins = normalize_roi_margins(roi_margins, base_w, base_h, expand_pct)
    work_points, base_matrix, margin_size, work_rect = work_roi_from_margins(ordered, margins)
    width = max(2, int(margin_size[0]))
    height = max(2, int(round(margin_size[1] * normalize_metric_scale_y(metric_scale_y))))
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(work_points.astype(np.float32), destination)
    return {
        "selected_points": ordered,
        "work_points": work_points,
        "destination_points": destination,
        "matrix": matrix,
        "output_size": (width, height),
        "base_matrix": base_matrix,
        "base_size": (base_w, base_h),
        "roi_margins": margins,
        "work_rect": work_rect,
    }


def measurement_vectors_for_geometry(
    raw_segments: list[dict],
    image_shape: tuple[int, int],
    correction: dict,
    geometry: dict,
) -> np.ndarray:
    vectors = []
    matrix = np.asarray(geometry["matrix"], dtype=np.float64)
    for segment in raw_segments:
        corrected = undistort_source_points(segment["points"], image_shape, correction)
        rectified = cv2.perspectiveTransform(
            corrected.astype(np.float32).reshape(-1, 1, 2),
            matrix,
        ).reshape(-1, 2)
        vectors.append(rectified[1] - rectified[0])
    return np.asarray(vectors, dtype=np.float64)


def measurement_scale_report(
    ratios: np.ndarray,
    segments: list[dict],
) -> dict:
    ratios = np.asarray(ratios, dtype=np.float64)
    mean = float(np.mean(ratios)) if ratios.size else 0.0
    spread = float(np.std(ratios) / mean) if mean > 0.0 else 0.0
    return {
        "mean_px_per_in": mean,
        "mean_inch_per_px": (1.0 / mean) if mean > 0.0 else None,
        "spread_pct": spread * 100.0,
        "minimum_px_per_in": float(np.min(ratios)) if ratios.size else None,
        "maximum_px_per_in": float(np.max(ratios)) if ratios.size else None,
        "segments": [
            {
                "index": index,
                "inches": float(segment["inches"]),
                "px_per_in": float(ratios[index]),
                "inch_per_px": (1.0 / float(ratios[index])) if ratios[index] > 0.0 else None,
            }
            for index, segment in enumerate(segments)
        ],
    }


def auto_fit_lens_from_measurements(
    homography: dict,
    calibration_segments: list[dict],
    image_shape: tuple[int, int],
) -> dict:
    segments = valid_measurement_segments(calibration_segments)
    if len(segments) < LENS_AUTO_FIT_MIN_SEGMENTS:
        raise RuntimeError(
            f"Se necesitan al menos {LENS_AUTO_FIT_MIN_SEGMENTS} medidas conocidas "
            "en distintas zonas de la imagen."
        )

    old_matrix = np.asarray(homography["homography_matrix"], dtype=np.float64)
    if old_matrix.shape != (3, 3):
        raise RuntimeError("La matriz de homografia guardada no es valida.")
    old_inverse = np.linalg.inv(old_matrix)
    old_correction = normalize_lens_correction(homography.get("lens_correction"))
    selected = np.asarray(
        homography.get("selected_source_points") or homography.get("ordered_source_points"),
        dtype=np.float32,
    )
    if selected.shape != (4, 2):
        raise RuntimeError("La homografia guardada no contiene cuatro puntos base.")

    raw_selected = distort_source_points(selected, image_shape, old_correction)
    raw_segments = []
    radial_positions = []
    camera = lens_camera_matrix(image_shape, old_correction)
    center = np.array([camera[0, 2], camera[1, 2]], dtype=np.float64)
    focal = float(camera[0, 0])
    for segment in segments:
        corrected = cv2.perspectiveTransform(
            segment["points"].reshape(-1, 1, 2),
            old_inverse,
        ).reshape(-1, 2)
        raw_points = distort_source_points(corrected, image_shape, old_correction)
        raw_segments.append(
            {
                "points": raw_points,
                "inches": segment["inches"],
                "source": segment["source"],
            }
        )
        radial_positions.append(float(np.linalg.norm(np.mean(raw_points, axis=0) - center) / focal))

    radial_span = float(np.ptp(radial_positions)) if radial_positions else 0.0
    midpoint_x = np.array([np.mean(segment["points"][:, 0]) for segment in raw_segments])
    x_span_ratio = float(np.ptp(midpoint_x) / max(1, image_shape[1]))
    if radial_span < 0.04 and x_span_ratio < 0.20:
        raise RuntimeError(
            "Las medidas estan demasiado juntas. Coloca referencias cerca del lado "
            "izquierdo, centro y lado derecho."
        )

    old_ratios = np.array(
        [
            float(np.linalg.norm(segment["points"][1] - segment["points"][0]))
            / float(segment["inches"])
            for segment in segments
        ],
        dtype=np.float64,
    )
    before = measurement_scale_report(old_ratios, segments)
    roi_margins = homography.get("roi_margins") or None
    expand_pct = float(homography.get("expand_pct", 0.0) or 0.0)

    def evaluate(k1: float, metric_scale_y: float) -> tuple[float, np.ndarray, dict]:
        correction = normalize_lens_correction(
            {
                "enabled": abs(k1) > 1e-7,
                "k1": k1,
                "center_x": old_correction["center_x"],
                "center_y": old_correction["center_y"],
                "focal_ratio": old_correction["focal_ratio"],
            }
        )
        geometry = candidate_homography_geometry(
            raw_selected,
            image_shape,
            correction,
            roi_margins,
            expand_pct,
            metric_scale_y,
        )
        vectors = measurement_vectors_for_geometry(
            raw_segments,
            image_shape,
            correction,
            geometry,
        )
        lengths = np.linalg.norm(vectors, axis=1)
        ratios = lengths / np.array([segment["inches"] for segment in raw_segments])
        log_ratios = np.log(np.maximum(ratios, 1e-9))
        score = float(np.sqrt(np.mean((log_ratios - np.mean(log_ratios)) ** 2)))
        return score, ratios, geometry

    best: tuple[float, float, float, np.ndarray, dict] | None = None
    k_values = np.linspace(LENS_K1_MIN, LENS_K1_MAX, 141)
    scale_values = np.linspace(LENS_METRIC_SCALE_MIN, LENS_METRIC_SCALE_MAX, 101)
    for k1 in k_values:
        base_score, _base_ratios, base_geometry = evaluate(float(k1), 1.0)
        del base_score
        base_vectors = measurement_vectors_for_geometry(
            raw_segments,
            image_shape,
            normalize_lens_correction(
                {
                    "enabled": abs(float(k1)) > 1e-7,
                    "k1": float(k1),
                    "center_x": old_correction["center_x"],
                    "center_y": old_correction["center_y"],
                    "focal_ratio": old_correction["focal_ratio"],
                }
            ),
            base_geometry,
        )
        inches = np.array([segment["inches"] for segment in raw_segments], dtype=np.float64)
        for metric_scale_y in scale_values:
            scaled = base_vectors.copy()
            scaled[:, 1] *= float(metric_scale_y)
            ratios = np.linalg.norm(scaled, axis=1) / inches
            log_ratios = np.log(np.maximum(ratios, 1e-9))
            score = float(np.sqrt(np.mean((log_ratios - np.mean(log_ratios)) ** 2)))
            if best is None or score < best[0]:
                geometry = candidate_homography_geometry(
                    raw_selected,
                    image_shape,
                    normalize_lens_correction(
                        {
                            "enabled": abs(float(k1)) > 1e-7,
                            "k1": float(k1),
                            "center_x": old_correction["center_x"],
                            "center_y": old_correction["center_y"],
                            "focal_ratio": old_correction["focal_ratio"],
                        }
                    ),
                    roi_margins,
                    expand_pct,
                    float(metric_scale_y),
                )
                best = (
                    score,
                    float(k1),
                    float(metric_scale_y),
                    ratios,
                    geometry,
                )

    if best is None:
        raise RuntimeError("No se pudo ajustar la correccion radial.")

    _score, best_k1, best_scale_y, best_ratios, best_geometry = best
    correction = normalize_lens_correction(
        {
            "enabled": abs(best_k1) > 1e-7,
            "k1": best_k1,
            "center_x": old_correction["center_x"],
            "center_y": old_correction["center_y"],
            "focal_ratio": old_correction["focal_ratio"],
        }
    )
    after = measurement_scale_report(best_ratios, segments)
    improvement_pct = (
        max(0.0, (before["spread_pct"] - after["spread_pct"]) / before["spread_pct"] * 100.0)
        if before["spread_pct"] > 1e-9
        else 0.0
    )
    at_limit = (
        abs(best_k1 - LENS_K1_MIN) < 0.006
        or abs(best_k1 - LENS_K1_MAX) < 0.006
        or abs(best_scale_y - LENS_METRIC_SCALE_MIN) < 0.006
        or abs(best_scale_y - LENS_METRIC_SCALE_MAX) < 0.006
    )
    recommended = len(segments) >= 5 and improvement_pct >= 10.0 and not at_limit
    confidence = "alta" if len(segments) >= 8 else ("media" if len(segments) >= 5 else "baja")
    warning = None
    if at_limit:
        warning = "El ajuste llego al limite permitido; agrega mas referencias antes de aplicarlo."
    elif len(segments) < 5:
        warning = "Ajuste preliminar: agrega al menos cinco referencias para aplicarlo."
    elif not recommended:
        warning = "Las medidas no muestran una mejora suficiente para cambiar la geometria."

    return {
        "lens_correction": correction,
        "metric_scale_y": best_scale_y,
        "selected_source_points": best_geometry["selected_points"].tolist(),
        "work_roi_points": best_geometry["work_points"].tolist(),
        "output_size": list(best_geometry["output_size"]),
        "before": before,
        "after": after,
        "improvement_pct": improvement_pct,
        "radial_span": radial_span,
        "reference_count": len(segments),
        "confidence": confidence,
        "recommended": recommended,
        "warning": warning,
    }


def rectified_points_to_raw_source(
    points: np.ndarray | list,
    homography: dict,
    image_shape: tuple[int, int],
) -> np.ndarray:
    matrix = np.asarray(homography["homography_matrix"], dtype=np.float64)
    corrected = cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        np.linalg.inv(matrix),
    ).reshape(-1, 2)
    return distort_source_points(
        corrected,
        image_shape,
        homography.get("lens_correction"),
    )


def raw_source_points_to_rectified(
    points: np.ndarray | list,
    homography: dict,
    image_shape: tuple[int, int],
) -> np.ndarray:
    corrected = undistort_source_points(
        points,
        image_shape,
        homography.get("lens_correction"),
    )
    return cv2.perspectiveTransform(
        corrected.astype(np.float32).reshape(-1, 1, 2),
        np.asarray(homography["homography_matrix"], dtype=np.float64),
    ).reshape(-1, 2)


def migrate_measurement_calibration(
    calibration: dict,
    old_homography: dict,
    new_homography: dict,
    image_shape: tuple[int, int],
) -> dict:
    migrated = dict(calibration)
    new_segments = []
    for segment in valid_measurement_segments(calibration.get("segments") or []):
        raw = rectified_points_to_raw_source(segment["points"], old_homography, image_shape)
        mapped = raw_source_points_to_rectified(raw, new_homography, image_shape)
        px = float(np.linalg.norm(mapped[1] - mapped[0]))
        updated = dict(segment["source"])
        updated.update(
            {
                "x1": float(mapped[0, 0]),
                "y1": float(mapped[0, 1]),
                "x2": float(mapped[1, 0]),
                "y2": float(mapped[1, 1]),
                "px": px,
                "inch_per_px": float(segment["inches"]) / px if px > 0.0 else None,
            }
        )
        new_segments.append(updated)

    old_width = int(calibration.get("img_w", old_homography.get("output_size", [2, 2])[0]) or 2)
    reference_y = calibration.get("reference_y")
    if reference_y is not None:
        reference_points = np.array(
            [
                [0.0, float(reference_y)],
                [(old_width - 1) / 2.0, float(reference_y)],
                [float(old_width - 1), float(reference_y)],
            ],
            dtype=np.float32,
        )
        raw = rectified_points_to_raw_source(reference_points, old_homography, image_shape)
        mapped = raw_source_points_to_rectified(raw, new_homography, image_shape)
        migrated["reference_y"] = float(np.mean(mapped[:, 1]))

    new_zones = []
    for zone in calibration.get("exclusion_zones") or []:
        try:
            x0, y0 = float(zone["x"]), float(zone["y"])
            x1, y1 = x0 + float(zone["w"]), y0 + float(zone["h"])
        except (KeyError, TypeError, ValueError):
            continue
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        raw = rectified_points_to_raw_source(corners, old_homography, image_shape)
        mapped = raw_source_points_to_rectified(raw, new_homography, image_shape)
        minimum = np.min(mapped, axis=0)
        maximum = np.max(mapped, axis=0)
        new_zones.append(
            {
                "x": float(minimum[0]),
                "y": float(minimum[1]),
                "w": float(maximum[0] - minimum[0]),
                "h": float(maximum[1] - minimum[1]),
            }
        )

    total_in = sum(float(segment["inches"]) for segment in new_segments)
    total_px = sum(float(segment["px"]) for segment in new_segments)
    inch_per_px = total_in / total_px if total_px > 0.0 else None
    output_size = new_homography["output_size"]
    migrated.update(
        {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "img_w": int(output_size[0]),
            "img_h": int(output_size[1]),
            "segments": new_segments,
            "exclusion_zones": new_zones,
            "inch_per_px": inch_per_px,
            "px_per_in": (1.0 / inch_per_px) if inch_per_px else None,
            "scale_map": build_spatial_scale_map(
                new_segments,
                int(output_size[0]),
                int(output_size[1]),
            ),
            "homography_path": str(homography_json_path()),
            "migrated_after_homography": True,
        }
    )
    return migrated


def frame_stem(frame_idx: int) -> str:
    return f"frame_{frame_idx:06d}"


def saved_frame_metadata(frame_idx: int) -> dict | None:
    meta_path = _args.dataset_dir / "labels" / f"{frame_stem(frame_idx)}.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def dataset_history() -> list[dict]:
    images_dir = _args.dataset_dir / "images"
    labels_dir = _args.dataset_dir / "labels"

    items = []
    for image_path in sorted(images_dir.glob("frame_*.jpg")):
        try:
            frame_idx = int(image_path.stem.replace("frame_", ""))
        except ValueError:
            continue
        meta = saved_frame_metadata(frame_idx) or {}
        label_path = labels_dir / f"{image_path.stem}.txt"
        if "boxes" in meta:
            box_count = len(meta.get("boxes") or [])
        elif label_path.exists() and label_path.stat().st_size:
            box_count = len([line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        else:
            box_count = 0
        items.append(
            {
                "frame_idx": frame_idx,
                "time_sec": float(meta.get("time_sec", frame_idx / 30.0)),
                "box_count": box_count,
                "image": str(image_path),
                "label": str(label_path),
                "saved_at": image_path.stat().st_mtime,
                "candidate": False,
                "source": meta.get("source") if isinstance(meta.get("source"), dict) else None,
            }
        )

    existing_frames = {int(item["frame_idx"]) for item in items}
    sample_count = max(0, int(getattr(_args, "candidate_samples_per_video", 0)))
    if _video_playlist is not None and sample_count:
        fps = float(_video_playlist["fps"])
        for segment in _video_playlist["segments"]:
            segment_total = int(segment["total_frames"])
            for sample_index in range(sample_count):
                local_idx = int(round((sample_index + 1) * segment_total / (sample_count + 1)))
                local_idx = max(0, min(local_idx, max(0, segment_total - 1)))
                frame_idx = int(segment["start_frame"]) + local_idx
                if frame_idx in existing_frames:
                    continue
                items.append(
                    {
                        "frame_idx": frame_idx,
                        "time_sec": frame_idx / fps,
                        "box_count": 0,
                        "image": "",
                        "label": "",
                        "saved_at": 0.0,
                        "candidate": True,
                        "source": playlist_source_for_frame(_video_playlist, frame_idx),
                    }
                )
                existing_frames.add(frame_idx)

    legacy_dataset = _args.legacy_candidates_dir
    try:
        use_legacy_candidates = (
            legacy_dataset is not None
            and legacy_dataset.resolve() != _args.dataset_dir.resolve()
        )
    except OSError:
        use_legacy_candidates = False
    if legacy_dataset is not None and use_legacy_candidates and (legacy_dataset / "images").exists():
        for image_path in sorted((legacy_dataset / "images").glob("frame_*.jpg")):
            try:
                frame_idx = int(image_path.stem.replace("frame_", ""))
            except ValueError:
                continue
            if frame_idx in existing_frames:
                continue
            meta_path = legacy_dataset / "labels" / f"{image_path.stem}.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            except json.JSONDecodeError:
                meta = {}
            items.append(
                {
                    "frame_idx": frame_idx,
                    "time_sec": float(meta.get("time_sec", frame_idx / 30.0)),
                    "box_count": 0,
                    "image": str(image_path),
                    "label": "",
                    "saved_at": image_path.stat().st_mtime,
                    "candidate": True,
                }
            )
    return sorted(items, key=lambda item: item["frame_idx"])


def saved_piece_annotation_count(items: list[dict] | None = None) -> int:
    history = items if items is not None else dataset_history()
    return sum(1 for item in history if not item.get("candidate"))


def load_piece_box_geometry_profile(dataset_dir: Path | None = None) -> dict:
    global _piece_box_profile_cache
    if dataset_dir is None and _piece_box_profile_cache is not None:
        return dict(_piece_box_profile_cache)

    target_dir = dataset_dir
    if target_dir is None:
        args = globals().get("_args")
        target_dir = Path(args.dataset_dir) if args is not None else DEFAULT_DATASET_DIR

    width_ratios: list[float] = []
    height_ratios: list[float] = []
    pitch_ratios: list[float] = []
    frame_count = 0
    labels_dir = Path(target_dir) / "labels"
    for metadata_path in sorted(labels_dir.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            image_width = float(metadata.get("img_w", 0))
            image_height = float(metadata.get("img_h", 0))
            boxes = [
                box
                for box in (metadata.get("boxes") or [])
                if not box.get("inferred") and float(box.get("w", 0)) > 0 and float(box.get("h", 0)) > 0
            ]
            if image_width <= 0 or image_height <= 0 or not boxes:
                continue
            frame_count += 1
            widths = [float(box["w"]) for box in boxes]
            width_ratios.extend(width / image_width for width in widths)
            height_ratios.extend(float(box["h"]) / image_height for box in boxes)
            centers = sorted(float(box["x"]) + float(box["w"]) / 2.0 for box in boxes)
            frame_width = float(np.median(widths))
            for left, right in zip(centers, centers[1:]):
                gap = right - left
                if frame_width * 0.55 <= gap <= frame_width * 1.60:
                    pitch_ratios.append(gap / image_width)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue

    profile = {
        "source": "annotations" if len(width_ratios) >= 8 else "frame",
        "frame_count": frame_count,
        "sample_count": len(width_ratios),
        "median_width_ratio": float(np.median(width_ratios)) if width_ratios else None,
        "median_height_ratio": float(np.median(height_ratios)) if height_ratios else None,
        "median_pitch_ratio": float(np.median(pitch_ratios)) if pitch_ratios else None,
    }
    if dataset_dir is None:
        _piece_box_profile_cache = dict(profile)
    return profile


def piece_box_overlap_fraction(first: dict, second: dict) -> float:
    first_x1 = float(first["x"])
    first_y1 = float(first["y"])
    first_x2 = first_x1 + float(first["w"])
    first_y2 = first_y1 + float(first["h"])
    second_x1 = float(second["x"])
    second_y1 = float(second["y"])
    second_x2 = second_x1 + float(second["w"])
    second_y2 = second_y1 + float(second["h"])
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    smaller_area = min(
        max(0.0, float(first["w"]) * float(first["h"])),
        max(0.0, float(second["w"]) * float(second["h"])),
    )
    return intersection / smaller_area if smaller_area > 0 else 0.0


def sanitize_piece_boxes(boxes: list[dict], image_shape: tuple[int, ...]) -> list[dict]:
    image_height, image_width = image_shape[:2]
    sanitized = []
    for box in boxes:
        try:
            x0 = float(np.clip(float(box["x"]), 0.0, float(max(0, image_width - 2))))
            y0 = float(np.clip(float(box["y"]), 0.0, float(max(0, image_height - 2))))
            x1 = float(
                np.clip(float(box["x"]) + float(box["w"]), x0 + 2.0, float(max(2, image_width - 1)))
            )
            y1 = float(
                np.clip(float(box["y"]) + float(box["h"]), y0 + 2.0, float(max(2, image_height - 1)))
            )
        except (KeyError, TypeError, ValueError):
            continue
        if x1 - x0 < 4.0 or y1 - y0 < 4.0:
            continue
        sanitized.append(
            {
                **box,
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
                "conf": float(box.get("conf", 1.0)),
            }
        )
    return sanitized


def suppress_overlapping_piece_boxes(
    boxes: list[dict],
    overlap_threshold: float = 0.35,
) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    removed: list[dict] = []
    ordered = sorted(
        boxes,
        key=lambda box: (float(box.get("conf", 1.0)), float(box["w"]) * float(box["h"])),
        reverse=True,
    )
    for candidate in ordered:
        if any(piece_box_overlap_fraction(candidate, existing) >= overlap_threshold for existing in kept):
            removed.append(candidate)
        else:
            kept.append(candidate)
    return sorted(kept, key=lambda box: float(box["x"]) + float(box["w"]) / 2.0), removed


def resolve_piece_box_reference(
    boxes: list[dict],
    image_shape: tuple[int, ...],
    profile: dict | None = None,
) -> dict:
    image_height, image_width = image_shape[:2]
    active_profile = profile or load_piece_box_geometry_profile()
    widths = [float(box["w"]) for box in boxes]
    heights = [float(box["h"]) for box in boxes]
    frame_width = float(np.median(widths)) if widths else max(8.0, image_width * 0.08)
    frame_height = float(np.median(heights)) if heights else max(8.0, image_height * 0.13)
    width_inliers = [
        width
        for width in widths
        if frame_width * 0.80 <= width <= frame_width * 1.20
    ]
    required_width_inliers = max(3, int(np.ceil(len(widths) * 0.60)))
    frame_width_reliable = len(width_inliers) >= required_width_inliers
    if frame_width_reliable:
        frame_width = float(np.median(width_inliers))
    frame_width_cv = (
        float(np.std(width_inliers) / frame_width)
        if frame_width_reliable and frame_width > 0
        else None
    )

    profile_width = active_profile.get("median_width_ratio")
    profile_height = active_profile.get("median_height_ratio")
    profile_pitch = active_profile.get("median_pitch_ratio")
    typical_width = float(profile_width) * image_width if profile_width else frame_width
    typical_height = float(profile_height) * image_height if profile_height else frame_height

    if frame_width_reliable:
        typical_width = frame_width
    elif widths and 0.60 <= frame_width / max(typical_width, 1e-6) <= 1.60:
        typical_width = float(np.median([typical_width, frame_width]))
    if heights and 0.45 <= frame_height / max(typical_height, 1e-6) <= 2.00:
        typical_height = float(np.median([typical_height, frame_height]))

    pitch_boxes = [
        box
        for box in boxes
        if typical_width * 0.80 <= float(box["w"]) <= typical_width * 1.20
    ]
    centers = sorted(float(box["x"]) + float(box["w"]) / 2.0 for box in pitch_boxes)
    close_gaps = [
        right - left
        for left, right in zip(centers, centers[1:])
        if typical_width * 0.55 <= right - left <= typical_width * 1.60
    ]
    if len(close_gaps) >= 2:
        typical_pitch = float(np.median(close_gaps))
    elif profile_pitch:
        typical_pitch = float(profile_pitch) * image_width
    elif close_gaps:
        typical_pitch = float(np.median(close_gaps))
    else:
        typical_pitch = typical_width * 1.10
    if close_gaps and 0.70 <= float(np.median(close_gaps)) / max(typical_pitch, 1e-6) <= 1.35:
        typical_pitch = float(np.median([typical_pitch, float(np.median(close_gaps))]))

    return {
        "source": "frame" if frame_width_reliable else active_profile.get("source", "frame"),
        "sample_count": int(active_profile.get("sample_count", 0)),
        "typical_width": typical_width,
        "typical_height": typical_height,
        "typical_pitch": typical_pitch,
        "frame_width_reliable": frame_width_reliable,
        "frame_width_sample_count": len(width_inliers),
        "frame_width_cv": frame_width_cv,
    }


def normalize_piece_box_widths(
    boxes: list[dict],
    image_shape: tuple[int, ...],
    reference: dict,
) -> tuple[list[dict], list[dict]]:
    if not reference.get("frame_width_reliable"):
        return boxes, []

    image_width = float(image_shape[1])
    target_width = max(float(reference["typical_width"]), 1e-6)
    adjusted: list[dict] = []
    normalized: list[dict] = []
    for source_box in boxes:
        box = dict(source_box)
        width = float(box["w"])
        ratio = width / target_width
        touches_edge = float(box["x"]) <= 2.0 or float(box["x"]) + width >= image_width - 2.0
        if not touches_edge and 0.82 <= ratio <= 1.18 and abs(width - target_width) >= 1.0:
            center_x = float(box["x"]) + width / 2.0
            box["geometry_original_x"] = float(box["x"])
            box["geometry_original_w"] = width
            box["x"] = float(np.clip(center_x - target_width / 2.0, 0.0, image_width - target_width))
            box["w"] = target_width
            box["geometry_adjusted"] = True
            adjusted.append(box)
        normalized.append(box)
    return normalized, adjusted


def infer_missing_piece_boxes(
    boxes: list[dict],
    image_shape: tuple[int, ...],
    reference: dict,
    max_missing_per_gap: int = 3,
) -> list[dict]:
    if len(boxes) < 4:
        return []

    ordered = sorted(boxes, key=lambda box: float(box["x"]) + float(box["w"]) / 2.0)
    centers_x = [float(box["x"]) + float(box["w"]) / 2.0 for box in ordered]
    typical_width = float(reference["typical_width"])
    typical_pitch = float(reference["typical_pitch"])
    image_height, image_width = image_shape[:2]
    inferred: list[dict] = []

    for index in range(len(ordered) - 1):
        center_gap = centers_x[index + 1] - centers_x[index]
        if center_gap < typical_pitch * 1.75:
            continue
        left_supported = (
            index > 0
            and typical_pitch * 0.55 <= centers_x[index] - centers_x[index - 1] <= typical_pitch * 1.45
        )
        right_supported = (
            index + 2 < len(ordered)
            and typical_pitch * 0.55
            <= centers_x[index + 2] - centers_x[index + 1]
            <= typical_pitch * 1.45
        )
        if not (left_supported and right_supported):
            continue

        missing_count = min(max_missing_per_gap, max(0, int(round(center_gap / typical_pitch)) - 1))
        if missing_count == 0:
            continue

        left_box = ordered[index]
        right_box = ordered[index + 1]
        left_center_y = float(left_box["y"]) + float(left_box["h"]) / 2.0
        right_center_y = float(right_box["y"]) + float(right_box["h"]) / 2.0
        inferred_height = float(np.median([left_box["h"], right_box["h"], reference["typical_height"]]))
        inferred_width = float(np.median([left_box["w"], right_box["w"], typical_width]))

        for missing_index in range(1, missing_count + 1):
            ratio = missing_index / float(missing_count + 1)
            center_x = centers_x[index] + center_gap * ratio
            center_y = left_center_y + (right_center_y - left_center_y) * ratio
            candidate = {
                "x": float(np.clip(center_x - inferred_width / 2.0, 0.0, max(0.0, image_width - inferred_width))),
                "y": float(
                    np.clip(center_y - inferred_height / 2.0, 0.0, max(0.0, image_height - inferred_height))
                ),
                "w": inferred_width,
                "h": inferred_height,
                "conf": min(float(left_box.get("conf", 1.0)), float(right_box.get("conf", 1.0))) * 0.5,
                "inferred": True,
                "source": "geometry",
                "geometry_gap_px": center_gap,
                "geometry_pitch_px": typical_pitch,
            }
            if not any(
                piece_box_overlap_fraction(candidate, existing) >= 0.15
                for existing in [*ordered, *inferred]
            ):
                inferred.append(candidate)
    return inferred


def apply_piece_box_rules(
    rectified: np.ndarray,
    boxes: list[dict],
    *,
    verify_inferred: bool = True,
    profile: dict | None = None,
) -> tuple[list[dict], dict]:
    sanitized = sanitize_piece_boxes(boxes, rectified.shape)
    deduplicated, removed_overlaps = suppress_overlapping_piece_boxes(sanitized)
    reference = resolve_piece_box_reference(deduplicated, rectified.shape, profile=profile)
    typical_width = max(float(reference["typical_width"]), 1e-6)
    typical_height = max(float(reference["typical_height"]), 1e-6)

    width_limits = (0.80, 1.20) if reference.get("frame_width_reliable") else (0.70, 1.35)
    severe_width_limits = (0.70, 1.30) if reference.get("frame_width_reliable") else (0.65, 1.40)
    size_outliers = []
    for box in deduplicated:
        width_ratio = float(box["w"]) / typical_width
        height_ratio = float(box["h"]) / typical_height
        box["size_outlier"] = not (
            width_limits[0] <= width_ratio <= width_limits[1]
            and 0.55 <= height_ratio <= 1.65
        )
        if box["size_outlier"]:
            size_outliers.append(box)

    median_width = float(np.median([box["w"] for box in deduplicated])) if deduplicated else 0.0
    median_height = float(np.median([box["h"] for box in deduplicated])) if deduplicated else 0.0
    looks_like_individual_pieces = (
        len(deduplicated) >= 3
        and 0.55 <= median_width / typical_width <= 1.60
        and 0.40 <= median_height / typical_height <= 2.00
    )
    removed_size = []
    size_filtered = []
    for box in deduplicated:
        width_ratio = float(box["w"]) / typical_width
        height_ratio = float(box["h"]) / typical_height
        severe_size_outlier = not (
            severe_width_limits[0] <= width_ratio <= severe_width_limits[1]
            and 0.60 <= height_ratio <= 1.60
        )
        if looks_like_individual_pieces and severe_size_outlier:
            removed_size.append(box)
        else:
            size_filtered.append(box)

    size_filtered, adjusted_widths = normalize_piece_box_widths(
        size_filtered,
        rectified.shape,
        reference,
    )
    proposals = (
        infer_missing_piece_boxes(size_filtered, rectified.shape, reference)
        if looks_like_individual_pieces
        else []
    )

    accepted_inferred = []
    rejected_inferred = []
    for candidate in proposals:
        if verify_inferred:
            sobel = sobel_projection_for_piece(rectified, candidate)
            candidate["geometry_verified"] = bool(sobel.get("is_valid"))
            candidate["geometry_edge_confidence"] = float(sobel.get("edge_confidence", 0.0))
            candidate["geometry_crm_px"] = float(sobel.get("crm_px", 0.0))
        else:
            candidate["geometry_verified"] = None
        if not verify_inferred or candidate["geometry_verified"]:
            accepted_inferred.append(candidate)
        else:
            rejected_inferred.append(candidate)

    processed = sorted(
        [*size_filtered, *accepted_inferred],
        key=lambda box: float(box["x"]) + float(box["w"]) / 2.0,
    )
    diagnostics = {
        "raw_count": len(boxes),
        "kept_yolo_count": len(size_filtered),
        "removed_overlap_count": len(removed_overlaps),
        "removed_size_count": len(removed_size),
        "size_outlier_count": len(size_outliers),
        "adjusted_width_count": len(adjusted_widths),
        "missing_candidate_count": len(proposals),
        "inferred_count": len(accepted_inferred),
        "rejected_inferred_count": len(rejected_inferred),
        "profile_source": reference["source"],
        "profile_sample_count": reference["sample_count"],
        "typical_width_px": reference["typical_width"],
        "typical_height_px": reference["typical_height"],
        "typical_pitch_px": reference["typical_pitch"],
        "frame_width_reliable": reference["frame_width_reliable"],
        "frame_width_sample_count": reference["frame_width_sample_count"],
        "frame_width_cv": reference["frame_width_cv"],
    }
    return processed, diagnostics


def resolve_yolo_device(requested: str | None = None) -> dict:
    import torch

    configured = str(requested or "auto").strip().lower()
    if not configured:
        configured = "auto"
    if configured == "auto":
        selected = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif configured == "cuda":
        selected = "cuda:0"
    elif configured == "cpu" or configured.startswith("cuda:"):
        selected = configured
    else:
        raise RuntimeError(
            f"Dispositivo YOLO no valido: {configured}. Usa auto, cpu, cuda o cuda:N."
        )

    cuda_available = bool(torch.cuda.is_available())
    device_name = "CPU"
    if selected.startswith("cuda"):
        if not cuda_available:
            raise RuntimeError(
                f"Se solicito {selected}, pero PyTorch no tiene CUDA disponible."
            )
        device_index = torch.device(selected).index
        if device_index is None:
            device_index = 0
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Se solicito {selected}, pero solo hay "
                f"{torch.cuda.device_count()} GPU(s) CUDA."
            )
        device_name = str(torch.cuda.get_device_name(device_index))

    return {
        "requested": configured,
        "device": selected,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
    }


def yolo_device_info() -> dict:
    global _yolo_device_info
    if _yolo_device_info is None:
        args = globals().get("_args")
        _yolo_device_info = resolve_yolo_device(
            getattr(args, "device", "auto") if args is not None else "auto"
        )
    return _yolo_device_info.copy()


def load_yolo_model():
    global _yolo_backend_error, _yolo_model, _yolo_model_path
    if _yolo_model is None:
        from ultralytics import YOLO

        if not _args.model.exists():
            raise RuntimeError(f"No existe el modelo YOLO: {_args.model}")
        _yolo_model_path = _args.model.resolve()
        _yolo_model = YOLO(
            str(_yolo_model_path),
            task="detect" if _yolo_model_path.suffix.lower() == ".engine" else None,
        )
        if _yolo_model_path.suffix.lower() != ".engine":
            _yolo_model.to(yolo_device_info()["device"])
        _yolo_backend_error = ""
    return _yolo_model


def yolo_backend_info() -> dict:
    args = globals().get("_args")
    requested = Path(getattr(args, "model", DEFAULT_MODEL)).resolve()
    active = _yolo_model_path or requested
    return {
        "backend": "tensorrt" if active.suffix.lower() == ".engine" else "pytorch",
        "requested_model": str(requested),
        "active_model": str(active),
        "fallback_active": active != requested,
        "backend_error": _yolo_backend_error,
    }


def _pytorch_fallback_model(engine_path: Path) -> Path | None:
    matching_checkpoint = engine_path.with_suffix(".pt")
    if matching_checkpoint.exists():
        return matching_checkpoint.resolve()
    return next(
        (
            path.resolve()
            for path in (
                DEFAULT_PIECE_MODEL,
                PREVIOUS_PIECE_MODEL,
                OLDER_PIECE_MODEL,
                DEFAULT_LEGACY_MODEL,
            )
            if path.exists()
        ),
        None,
    )


def _predict_with_yolo_model(model, rectified: np.ndarray, conf: float, imgsz: int):
    return model.predict(
        rectified,
        conf=conf,
        imgsz=imgsz,
        device=yolo_device_info()["device"],
        verbose=False,
    )[0]


def _retry_yolo_with_pytorch(
    rectified: np.ndarray,
    conf: float,
    imgsz: int,
    engine_error: Exception,
):
    global _yolo_backend_error, _yolo_model, _yolo_model_path

    engine_path = _yolo_model_path or Path(_args.model).resolve()
    fallback_path = _pytorch_fallback_model(engine_path)
    if fallback_path is None:
        raise engine_error

    from ultralytics import YOLO

    _yolo_backend_error = f"{type(engine_error).__name__}: {engine_error}"
    _yolo_model_path = fallback_path
    _yolo_model = YOLO(str(fallback_path))
    _yolo_model.to(yolo_device_info()["device"])
    return _predict_with_yolo_model(_yolo_model, rectified, conf, imgsz)


def normalize_exclusion_zones(
    zones: list[dict] | None,
    image_shape: tuple[int, ...] | None = None,
) -> list[dict]:
    height = float(image_shape[0]) if image_shape and len(image_shape) >= 2 else None
    width = float(image_shape[1]) if image_shape and len(image_shape) >= 2 else None
    normalized = []
    for zone in zones or []:
        if not isinstance(zone, dict):
            continue
        try:
            x1 = float(zone.get("x", 0.0))
            y1 = float(zone.get("y", 0.0))
            x2 = x1 + float(zone.get("w", 0.0))
            y2 = y1 + float(zone.get("h", 0.0))
        except (TypeError, ValueError):
            continue
        x0, x1 = sorted((x1, x2))
        y0, y1 = sorted((y1, y2))
        if width is not None:
            x0 = float(np.clip(x0, 0.0, width))
            x1 = float(np.clip(x1, 0.0, width))
        if height is not None:
            y0 = float(np.clip(y0, 0.0, height))
            y1 = float(np.clip(y1, 0.0, height))
        if x1 - x0 < 2.0 or y1 - y0 < 2.0:
            continue
        normalized.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return normalized


def box_exclusion_overlap_ratio(box: dict, zone: dict) -> float:
    box_x0 = float(box.get("x", 0.0))
    box_y0 = float(box.get("y", 0.0))
    box_x1 = box_x0 + max(0.0, float(box.get("w", 0.0)))
    box_y1 = box_y0 + max(0.0, float(box.get("h", 0.0)))
    box_area = max(0.0, box_x1 - box_x0) * max(0.0, box_y1 - box_y0)
    if box_area <= 0.0:
        return 0.0
    zone_x0 = float(zone["x"])
    zone_y0 = float(zone["y"])
    zone_x1 = zone_x0 + float(zone["w"])
    zone_y1 = zone_y0 + float(zone["h"])
    intersection_w = max(0.0, min(box_x1, zone_x1) - max(box_x0, zone_x0))
    intersection_h = max(0.0, min(box_y1, zone_y1) - max(box_y0, zone_y0))
    return (intersection_w * intersection_h) / box_area


def filter_boxes_by_exclusion_zones(
    boxes: list[dict],
    zones: list[dict] | None,
    *,
    max_overlap: float = EXCLUSION_ZONE_MAX_BOX_OVERLAP,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[list[dict], list[dict]]:
    normalized_zones = normalize_exclusion_zones(zones, image_shape=image_shape)
    overlap_limit = float(np.clip(float(max_overlap), 0.0, 1.0))
    if not normalized_zones:
        return list(boxes), []

    kept = []
    removed = []
    for box in boxes:
        overlaps = [box_exclusion_overlap_ratio(box, zone) for zone in normalized_zones]
        largest_overlap = max(overlaps, default=0.0)
        if largest_overlap > overlap_limit:
            removed.append(
                {
                    "box": box,
                    "overlap_ratio": largest_overlap,
                    "zone_index": overlaps.index(largest_overlap),
                }
            )
        else:
            kept.append(box)
    return kept, removed


def predict_yolo_boxes_with_rules(
    rectified: np.ndarray,
    conf: float = 0.10,
    imgsz: int = 960,
    exclusion_zones: list[dict] | None = None,
    exclusion_max_overlap: float = EXCLUSION_ZONE_MAX_BOX_OVERLAP,
) -> tuple[list[dict], dict]:
    model = load_yolo_model()
    try:
        result = _predict_with_yolo_model(model, rectified, conf, imgsz)
    except Exception as exc:
        active_path = _yolo_model_path or Path(_args.model)
        if active_path.suffix.lower() != ".engine":
            raise
        result = _retry_yolo_with_pytorch(rectified, conf, imgsz, exc)
    normalized_zones = normalize_exclusion_zones(exclusion_zones, image_shape=rectified.shape)
    if result.boxes is None or len(result.boxes) == 0:
        return [], {
            "raw_count": 0,
            "kept_yolo_count": 0,
            "removed_overlap_count": 0,
            "removed_size_count": 0,
            "size_outlier_count": 0,
            "missing_candidate_count": 0,
            "inferred_count": 0,
            "rejected_inferred_count": 0,
            "exclusion_zone_count": len(normalized_zones),
            "removed_exclusion_count": 0,
            "exclusion_max_overlap": float(exclusion_max_overlap),
        }

    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    boxes = []
    for box, score in zip(xyxy, scores):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        boxes.append(
            {
                "x": x0,
                "y": y0,
                "w": max(0.0, x1 - x0),
                "h": max(0.0, y1 - y0),
                "conf": float(score),
            }
        )
    processed, diagnostics = apply_piece_box_rules(rectified, boxes)
    filtered, removed_exclusions = filter_boxes_by_exclusion_zones(
        processed,
        normalized_zones,
        max_overlap=exclusion_max_overlap,
        image_shape=rectified.shape,
    )
    diagnostics.update(
        exclusion_zone_count=len(normalized_zones),
        removed_exclusion_count=len(removed_exclusions),
        exclusion_max_overlap=float(exclusion_max_overlap),
        final_count=len(filtered),
    )
    return filtered, diagnostics


def predict_yolo_boxes(rectified: np.ndarray, conf: float = 0.10, imgsz: int = 960) -> list[dict]:
    boxes, _diagnostics = predict_yolo_boxes_with_rules(rectified, conf=conf, imgsz=imgsz)
    return boxes


def best_box_for_projection(rectified: np.ndarray, boxes: list[dict], conf: float) -> dict | None:
    candidates = boxes or predict_yolo_boxes(rectified, conf=conf)
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item["w"]) * float(item["h"]) * float(item.get("conf", 1.0)))


def sobel_projection_for_box(
    rectified: np.ndarray,
    box: dict,
    line_x_range: tuple[float, float] | None = None,
    config_overrides: dict | None = None,
    require_projection_valid: bool = True,
) -> dict:
    from yolo_roi_sobel_projection import ProjectionConfig, edge_response_from_roi, project_edge_line

    config = {
        "roi_pad_x": 0,
        "roi_pad_y": 0,
        "score_keep_percentile": 35.0,
        "min_points": 12,
        "line_inlier_tol": 8.0,
        "max_abs_slope": 0.35,
    }
    config.update(config_overrides or {})
    cfg = ProjectionConfig(
        **config,
    )
    height, width = rectified.shape[:2]
    x0 = int(np.floor(float(box["x"]) - cfg.roi_pad_x))
    y0 = int(np.floor(float(box["y"]) - cfg.roi_pad_y))
    x1 = int(np.ceil(float(box["x"]) + float(box["w"]) + cfg.roi_pad_x))
    y1 = int(np.ceil(float(box["y"]) + float(box["h"]) + cfg.roi_pad_y))
    x0 = max(0, min(x0, width - 2))
    y0 = max(0, min(y0, height - 2))
    x1 = max(x0 + 2, min(x1, width - 1))
    y1 = max(y0 + 2, min(y1, height - 1))

    roi = rectified[y0:y1, x0:x1]
    _gray, edge = edge_response_from_roi(roi, cfg)
    projection = project_edge_line(edge, cfg)

    points = []
    for point in projection["points"]:
        px, py, score, *rest = point
        points.append(
            {
                "x": float(x0 + px),
                "y": float(y0 + py),
                "score": float(score),
                "inlier": bool(rest[0]) if rest else False,
            }
        )

    def weighted_median(values: list[float], weights: list[float]) -> float:
        vals = np.asarray(values, dtype=np.float64)
        wts = np.asarray(weights, dtype=np.float64)
        if len(vals) == 0:
            return float("nan")
        order = np.argsort(vals)
        vals = vals[order]
        wts = np.maximum(wts[order], 1e-6)
        cutoff = float(wts.sum()) * 0.5
        return float(vals[np.searchsorted(np.cumsum(wts), cutoff, side="left")])

    line = None
    if projection["line"] is not None:
        slope, intercept = projection["line"]
        horizontal_candidates = [p for p in points if p.get("inlier")]
        if len(horizontal_candidates) < cfg.min_points:
            threshold = projection.get("threshold", float("nan"))
            if np.isfinite(float(threshold)):
                horizontal_candidates = [p for p in points if float(p.get("score", 0.0)) >= float(threshold)]
        if horizontal_candidates:
            y_roi_values = [float(p["y"]) - float(y0) for p in horizontal_candidates]
            score_values = [float(p.get("score", 1.0)) for p in horizontal_candidates]
            horizontal_y_roi = weighted_median(y_roi_values, score_values)
        else:
            horizontal_y_roi = float(slope) * ((float(x1 - x0) - 1.0) / 2.0) + float(intercept)
        horizontal_y_global = float(y0) + float(horizontal_y_roi)
        if line_x_range is None:
            global_left_x = 0.0
            global_right_x = float(width - 1)
        else:
            global_left_x = float(np.clip(min(line_x_range), 0.0, float(width - 1)))
            global_right_x = float(np.clip(max(line_x_range), 0.0, float(width - 1)))
        line = {
            "x1": global_left_x,
            "y1": horizontal_y_global,
            "x2": global_right_x,
            "y2": horizontal_y_global,
            "horizontal": True,
            "y": horizontal_y_global,
            "roi_line_y": float(horizontal_y_roi),
            "slope_roi": float(slope),
            "intercept_roi": float(intercept),
            "roi_x0": float(x0),
            "roi_y0": float(y0),
        }

    confidence = float(projection["confidence"])
    crm_px = float(projection["crm_px"])
    if line is not None:
        residual_points = [p for p in points if p.get("inlier")] or points
        residual = [float(p["y"]) - float(line["y"]) for p in residual_points]
        crm_px = float(np.sqrt(np.mean(np.square(residual)))) if residual else crm_px
    projection_valid = bool(projection["is_valid"]) if require_projection_valid else line is not None
    is_valid = projection_valid and confidence >= 0.30 and crm_px <= 8.0

    return {
        "has_roi": True,
        "is_valid": is_valid,
        "roi": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
        "roi_box": box,
        "line": line,
        "points": points,
        "edge_confidence": confidence,
        "crm_px": crm_px,
    }


def sobel_projection_for_piece(rectified: np.ndarray, box: dict) -> dict:
    height, width = rectified.shape[:2]
    piece_x0 = float(np.clip(float(box["x"]), 0.0, float(max(0, width - 2))))
    piece_y0 = float(np.clip(float(box["y"]), 0.0, float(max(0, height - 2))))
    piece_x1 = float(
        np.clip(float(box["x"]) + float(box["w"]), piece_x0 + 2.0, float(max(2, width - 1)))
    )
    piece_y1 = float(
        np.clip(float(box["y"]) + float(box["h"]), piece_y0 + 2.0, float(max(2, height - 1)))
    )
    piece_width = piece_x1 - piece_x0
    piece_height = piece_y1 - piece_y0

    band_height = min(piece_height, max(32.0, piece_height * 0.70))
    bottom_pad = min(12.0, max(5.0, piece_height * 0.05))
    side_inset = min(6.0, max(1.0, piece_width * 0.06))
    analysis_x0 = piece_x0 + side_inset
    analysis_x1 = piece_x1 - side_inset
    if analysis_x1 - analysis_x0 < 8.0:
        analysis_x0, analysis_x1 = piece_x0, piece_x1

    analysis_box = {
        "x": analysis_x0,
        "y": max(piece_y0, piece_y1 - band_height),
        "w": analysis_x1 - analysis_x0,
        "h": min(float(height - 1), piece_y1 + bottom_pad) - max(piece_y0, piece_y1 - band_height),
        "conf": float(box.get("conf", 1.0)),
    }
    analysis_width = max(8.0, float(analysis_box["w"]))
    analysis_height = max(2.0, float(analysis_box["h"]))
    edge_band_end = min(1.0, (band_height + min(2.0, bottom_pad)) / analysis_height)
    bin_width = 3 if analysis_width < 90.0 else 4
    estimated_samples = max(3, int(analysis_width // bin_width) - 1)
    min_points = max(4, min(8, int(np.floor(estimated_samples * 0.55))))
    blur_width = max(5, min(21, int(round(analysis_width * 0.35))))
    if blur_width % 2 == 0:
        blur_width += 1
    result = sobel_projection_for_box(
        rectified,
        analysis_box,
        line_x_range=(piece_x0, piece_x1),
        config_overrides={
            "bin_width": bin_width,
            "min_points": min_points,
            "blur_ksize": (blur_width, 1),
            "profile_smooth": 7,
            "line_inlier_tol": 4.0,
            "edge_band_start": 0.05,
            "edge_band_end": edge_band_end,
            "edge_polarity": "falling",
        },
        require_projection_valid=False,
    )
    result["roi_box"] = box
    result["analysis_box"] = analysis_box
    result["piece_box"] = {
        "x": piece_x0,
        "y": piece_y0,
        "w": piece_width,
        "h": piece_height,
        "conf": float(box.get("conf", 1.0)),
    }
    return result


def summarize_piece_measurements(pieces: list[dict]) -> dict:
    valid_pieces = [
        piece
        for piece in pieces
        if piece.get("valid") and isinstance(piece.get("measurement"), dict)
    ]
    measurements = [float(piece["measurement"]["measurement_in"]) for piece in valid_pieces]
    distances = [float(piece["measurement"]["delta_in"]) for piece in valid_pieces]
    return {
        "detected_count": len(pieces),
        "valid_count": len(valid_pieces),
        "invalid_count": len(pieces) - len(valid_pieces),
        "minimum_in": min(measurements) if measurements else None,
        "maximum_in": max(measurements) if measurements else None,
        "average_in": float(np.mean(measurements)) if measurements else None,
        "average_distance_to_reference_in": float(np.mean(distances)) if distances else None,
    }


def empty_sobel_result(
    frame_idx: int | None = None,
    time_sec: float | None = None,
) -> dict:
    return {
        "frame_idx": frame_idx,
        "time_sec": time_sec,
        "has_roi": False,
        "is_valid": False,
        "roi": None,
        "roi_box": None,
        "line": None,
        "points": [],
        "edge_confidence": 0.0,
        "crm_px": 0.0,
    }


def analyze_piece_boxes(
    rectified: np.ndarray,
    boxes: list[dict],
    calibration: dict,
    frame_idx: int | None = None,
    time_sec: float | None = None,
) -> tuple[list[dict], dict]:
    ordered_boxes = sorted(
        boxes,
        key=lambda box: (
            float(box["x"]) + float(box["w"]) / 2.0,
            float(box["y"]) + float(box["h"]) / 2.0,
        ),
    )
    pieces = []
    for piece_id, box in enumerate(ordered_boxes, start=1):
        sobel = sobel_projection_for_piece(rectified, box)
        sobel.update(frame_idx=frame_idx, time_sec=time_sec)
        measurement = measurement_from_sobel(sobel, calibration, rectified.shape[1])
        pieces.append(
            {
                "piece_id": piece_id,
                "box": box,
                "confidence": float(box.get("conf", 1.0)),
                "sobel": sobel,
                "measurement": measurement,
                "valid": bool(sobel.get("is_valid")) and measurement is not None,
            }
        )
    return pieces, summarize_piece_measurements(pieces)


def primary_piece_analysis(pieces: list[dict]) -> dict | None:
    return next(
        (piece for piece in pieces if piece.get("valid")),
        pieces[0] if pieces else None,
    )




app = Flask(__name__)
_args: argparse.Namespace
_video_playlist: dict | None = None
_image: np.ndarray | None = None
_source_label = ""
_yolo_model = None
_yolo_device_info: dict | None = None
_yolo_model_path: Path | None = None
_yolo_backend_error = ""
_piece_box_profile_cache: dict | None = None
_lens_map_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


@app.route("/")
def index():
    return render_template(
        "homography_tool.html",
        source_label=_source_label,
        second=_args.second,
    )


@app.route("/api/meta")
def api_meta():
    try:
        if _video_playlist is None:
            raise RuntimeError("La playlist de videos no esta inicializada")
        data = public_playlist_meta(_video_playlist)
        data["dataset_dir"] = str(_args.dataset_dir)
        data["video_dir"] = str(_args.video_dir) if _args.video_dir is not None else None
        data["homography_exists"] = homography_json_path().exists()
        data["measurement_exists"] = measurement_json_path().exists()
        data["piece_box_profile"] = load_piece_box_geometry_profile()
        return jsonify(data)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/frame", methods=["POST"])
def api_frame():
    global _image, _source_label
    data = request.get_json() or {}
    try:
        if "frame_idx" in data:
            _image, frame_idx, time_sec, source = read_active_frame_by_index(int(data["frame_idx"]))
            _source_label = (
                f"{source['video']} @ {source['source_time_sec']:.3f}s "
                f"(playlist {time_sec:.3f}s)"
            )
        else:
            second = float(data.get("second", _args.second))
            _args.second = second
            _image, _source_label, frame_idx, time_sec = load_reference_image(_args)
            source = playlist_source_for_frame(_video_playlist, frame_idx)
        _args.second = time_sec
        lens_correction = normalize_lens_correction(data.get("lens_correction"))
        display_image = apply_lens_correction(_image, lens_correction)
        hgt, wid = display_image.shape[:2]
        return jsonify(
            image=img_to_b64(display_image, quality=86),
            width=wid,
            height=hgt,
            frame_idx=frame_idx,
            time_sec=time_sec,
            label=_source_label,
            source=source,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/warp", methods=["POST"])
def api_warp():
    if _image is None:
        return jsonify(error="No hay imagen cargada"), 400
    data = request.get_json() or {}
    try:
        pts = [(p["x"], p["y"]) for p in data["points"]]
        expand_pct = float(data.get("expand_pct", 0.0))
        roi_margins = data.get("roi_margins") or None
        lens_correction = normalize_lens_correction(data.get("lens_correction"))
        metric_scale_y = normalize_metric_scale_y(data.get("metric_scale_y", 1.0))
        corrected_image = apply_lens_correction(_image, lens_correction)
        warp, ordered, warp_points, _dst, _matrix, size, base_matrix, base_size, margins, work_rect = compute_warp(
            corrected_image,
            pts,
            expand_pct=expand_pct,
            roi_margins=roi_margins,
            metric_scale_y=metric_scale_y,
        )
        return jsonify(
            image=img_to_b64(warp, quality=86),
            width=size[0],
            height=size[1],
            ordered_points=[{"x": float(x), "y": float(y)} for x, y in ordered.tolist()],
            warp_points=[{"x": float(x), "y": float(y)} for x, y in warp_points.tolist()],
            expand_pct=expand_pct,
            roi_margins=margins,
            roi_mode="side_margins" if roi_margins is not None else "auto",
            work_rect=work_rect,
            base_matrix=base_matrix.tolist(),
            base_size={"width": base_size[0], "height": base_size[1]},
            manual_roi=roi_margins is not None,
            lens_correction=lens_correction,
            metric_scale_y=metric_scale_y,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/homography/current")
def api_current_homography():
    try:
        path = homography_json_path()
        if not path.exists():
            return jsonify(error=f"No existe homografia guardada: {path}"), 404
        data = json.loads(path.read_text(encoding="utf-8"))
        selected = data.get("selected_source_points") or data.get("ordered_source_points") or []
        return jsonify(
            path=str(path),
            source=data.get("source", ""),
            saved_at=data.get("saved_at", ""),
            selected_source_points=selected,
            ordered_source_points=data.get("ordered_source_points", []),
            work_roi_points=data.get("work_roi_points") or data.get("ordered_source_points", []),
            roi_margins=data.get("roi_margins") or None,
            work_rect=data.get("work_rect") or None,
            base_homography_matrix=data.get("base_homography_matrix") or None,
            base_output_size=data.get("base_output_size") or [],
            expand_pct=float(data.get("expand_pct", 0.0)),
            roi_mode=data.get("roi_mode", "auto"),
            output_size=data.get("output_size", []),
            lens_correction=normalize_lens_correction(data.get("lens_correction")),
            metric_scale_y=normalize_metric_scale_y(data.get("metric_scale_y", 1.0)),
            lens_auto_fit=data.get("lens_auto_fit"),
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/lens/auto_fit", methods=["POST"])
def api_lens_auto_fit():
    data = request.get_json() or {}
    try:
        path = homography_json_path()
        if not path.exists():
            return jsonify(error="Guarda primero los cuatro puntos de homografia."), 400
        homography = json.loads(path.read_text(encoding="utf-8"))
        segments = data.get("segments")
        if not isinstance(segments, list) or not segments:
            segments = load_measurement_calibration().get("segments") or []
        if _image is not None:
            image_shape = _image.shape[:2]
        elif _video_playlist is not None:
            image_shape = (
                int(_video_playlist["height"]),
                int(_video_playlist["width"]),
            )
        else:
            raise RuntimeError("No hay una imagen fuente para ajustar la lente.")
        return jsonify(auto_fit_lens_from_measurements(homography, segments, image_shape))
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.route("/api/save", methods=["POST"])
@app.route("/api/save_homography", methods=["POST"])
def api_save_homography():
    if _image is None:
        return jsonify(error="No hay imagen cargada"), 400
    data = request.get_json() or {}
    try:
        pts = [(p["x"], p["y"]) for p in data["points"]]
        expand_pct = float(data.get("expand_pct", 0.0))
        roi_margins = data.get("roi_margins") or None
        lens_correction = normalize_lens_correction(data.get("lens_correction"))
        metric_scale_y = normalize_metric_scale_y(data.get("metric_scale_y", 1.0))
        corrected_image = apply_lens_correction(_image, lens_correction)
        warp, ordered, warp_points, dst, matrix, size, base_matrix, base_size, margins, work_rect = compute_warp(
            corrected_image,
            pts,
            expand_pct=expand_pct,
            roi_margins=roi_margins,
            metric_scale_y=metric_scale_y,
        )
        _args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = homography_json_path()
        src_path = _args.output_dir / "homography_selection_source.jpg"
        warp_path = _args.output_dir / "homography_selection_warp.jpg"
        backup_path = None
        old_homography = None
        if json_path.exists():
            old_homography = json.loads(json_path.read_text(encoding="utf-8"))
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = _args.output_dir / f"homography_selection_{stamp}.bak.json"
            backup_path.write_bytes(json_path.read_bytes())

        src_vis = corrected_image.copy()
        cv2.polylines(src_vis, [ordered.astype(np.int32).reshape(-1, 1, 2)], True, (83, 182, 137), 3, cv2.LINE_AA)
        if any(value > 0.0 for value in margins.values()):
            cv2.polylines(
                src_vis,
                [warp_points.astype(np.int32).reshape(-1, 1, 2)],
                True,
                (75, 155, 213),
                3,
                cv2.LINE_AA,
            )
        for idx, pt in enumerate(ordered.tolist()):
            cv2.circle(src_vis, (int(pt[0]), int(pt[1])), 14, (83, 182, 137), -1)
            cv2.putText(
                src_vis,
                str(idx + 1),
                (int(pt[0]) + 18, int(pt[1]) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(src_path), src_vis)
        cv2.imwrite(str(warp_path), warp)

        payload = {
            "source": _source_label,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "selected_source_points": ordered.tolist(),
            "ordered_source_points": warp_points.tolist(),
            "work_roi_points": warp_points.tolist(),
            "roi_mode": "side_margins" if roi_margins is not None else "auto",
            "roi_margins": margins,
            "work_rect": work_rect,
            "base_homography_matrix": base_matrix.tolist(),
            "base_output_size": list(base_size),
            "expand_pct": expand_pct,
            "lens_correction": lens_correction,
            "metric_scale_y": metric_scale_y,
            "lens_auto_fit": data.get("lens_auto_fit"),
            "destination_points": dst.tolist(),
            "output_size": list(size),
            "homography_matrix": matrix.tolist(),
            "source_preview": str(src_path),
            "warp_preview": str(warp_path),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        measurement_migrated = False
        measurement_backup = None
        calibration_path = measurement_json_path()
        if (
            bool(data.get("migrate_measurements", True))
            and old_homography is not None
            and calibration_path.exists()
        ):
            old_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            measurement_backup = _args.output_dir / f"table_measurement_calibration_{stamp}.bak.json"
            measurement_backup.write_bytes(calibration_path.read_bytes())
            migrated = migrate_measurement_calibration(
                old_calibration,
                old_homography,
                payload,
                _image.shape[:2],
            )
            calibration_path.write_text(json.dumps(migrated, indent=2), encoding="utf-8")
            measurement_migrated = True
        return jsonify(
            path=str(json_path),
            backup=str(backup_path) if backup_path else None,
            measurement_migrated=measurement_migrated,
            measurement_backup=str(measurement_backup) if measurement_backup else None,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/annotate/frame", methods=["POST"])
def api_annotate_frame():
    data = request.get_json() or {}
    try:
        frame_idx = int(data.get("frame_idx", 0))
        frame, frame_idx, time_sec, source = read_active_frame_by_index(frame_idx)
        frame, homography = apply_saved_homography(frame)
        saved_meta = saved_frame_metadata(frame_idx)
        hgt, wid = frame.shape[:2]
        return jsonify(
            image=img_to_b64(frame, quality=88),
            width=wid,
            height=hgt,
            frame_idx=frame_idx,
            time_sec=time_sec,
            boxes=(saved_meta or {}).get("boxes", []),
            is_saved=saved_meta is not None or (_args.dataset_dir / "images" / f"{frame_stem(frame_idx)}.jpg").exists(),
            homography_source=homography.get("source", ""),
            source=source,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/measure/frame", methods=["POST"])
def api_measure_frame():
    data = request.get_json() or {}
    try:
        frame_idx = int(data.get("frame_idx", 0))
        frame, frame_idx, time_sec, source = read_active_frame_by_index(frame_idx)
        frame, homography = apply_saved_homography(frame)
        hgt, wid = frame.shape[:2]
        return jsonify(
            image=img_to_b64(frame, quality=88),
            width=wid,
            height=hgt,
            frame_idx=frame_idx,
            time_sec=time_sec,
            calibration=load_measurement_calibration(),
            homography_source=homography.get("source", ""),
            source=source,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/measure/calibration")
def api_measure_calibration():
    try:
        return jsonify(load_measurement_calibration())
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/measure/save", methods=["POST"])
def api_measure_save():
    data = request.get_json() or {}
    try:
        segments = data.get("segments") or []
        image_shape = (
            max(0, int(data.get("img_h", 0))),
            max(0, int(data.get("img_w", 0))),
        )
        exclusion_zones = normalize_exclusion_zones(
            data.get("exclusion_zones") or [],
            image_shape=image_shape if all(image_shape) else None,
        )
        valid = [
            seg
            for seg in segments
            if float(seg.get("px", 0) or 0) > 0 and float(seg.get("inches", 0) or 0) > 0
        ]
        inch_per_px = data.get("inch_per_px")
        if inch_per_px is None and valid:
            total_in = sum(float(seg["inches"]) for seg in valid)
            total_px = sum(float(seg["px"]) for seg in valid)
            inch_per_px = total_in / total_px if total_px else None
        image_width = int(data.get("img_w", 0))
        image_height = int(data.get("img_h", 0))
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "frame_idx": int(data.get("frame_idx", 0)),
            "time_sec": float(data.get("time_sec", 0.0)),
            "img_w": image_width,
            "img_h": image_height,
            "segments": segments,
            "reference_y": data.get("reference_y"),
            "reference_offset_in": MEASUREMENT_REFERENCE_OFFSET_IN,
            "exclusion_zones": exclusion_zones,
            "exclusion_max_box_overlap": EXCLUSION_ZONE_MAX_BOX_OVERLAP,
            "inch_per_px": inch_per_px,
            "px_per_in": (1.0 / float(inch_per_px)) if inch_per_px else None,
            "scale_map": build_spatial_scale_map(
                segments,
                image_width,
                image_height,
            ),
            "homography_path": str(homography_json_path()),
        }
        _args.output_dir.mkdir(parents=True, exist_ok=True)
        path = measurement_json_path()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["path"] = str(path)
        return jsonify(payload)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/annotate/predict", methods=["POST"])
def api_annotate_predict():
    data = request.get_json() or {}
    try:
        frame_idx = int(data.get("frame_idx", 0))
        conf = float(data.get("conf", 0.10))
        frame, frame_idx, time_sec, source = read_active_frame_by_index(frame_idx)
        frame, _homography = apply_saved_homography(frame)
        calibration = load_measurement_calibration()
        boxes, box_rules = predict_yolo_boxes_with_rules(
            frame,
            conf=conf,
            exclusion_zones=calibration.get("exclusion_zones"),
        )
        return jsonify(
            frame_idx=frame_idx,
            time_sec=time_sec,
            boxes=boxes,
            count=len(boxes),
            box_rules=box_rules,
            model=str(_args.model),
            source=source,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/annotate/sobel_projection", methods=["POST"])
def api_annotate_sobel_projection():
    data = request.get_json() or {}
    try:
        frame_idx = int(data.get("frame_idx", 0))
        conf = float(data.get("conf", 0.10))
        boxes = data.get("boxes") or []
        frame, frame_idx, time_sec, source = read_active_frame_by_index(frame_idx)
        frame, _homography = apply_saved_homography(frame)
        calibration = load_measurement_calibration()
        if boxes:
            candidates = boxes
            box_rules = None
        else:
            candidates, box_rules = predict_yolo_boxes_with_rules(
                frame,
                conf=conf,
                exclusion_zones=calibration.get("exclusion_zones"),
            )
        pieces, measurement_summary = analyze_piece_boxes(
            frame,
            candidates,
            calibration,
            frame_idx=frame_idx,
            time_sec=time_sec,
        )
        primary = primary_piece_analysis(pieces)
        result = dict(primary["sobel"]) if primary else empty_sobel_result(frame_idx, time_sec)
        result.update(
            pieces=pieces,
            measurement_summary=measurement_summary,
            measurements=[piece["measurement"] for piece in pieces],
            box_rules=box_rules,
            source=source,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/player/frame", methods=["POST"])
def api_player_frame():
    data = request.get_json() or {}
    try:
        frame_idx = int(data.get("frame_idx", 0))
        conf = float(data.get("conf", 0.10))
        frame, frame_idx, time_sec, source = read_active_frame_by_index(frame_idx)
        frame, _homography = apply_saved_homography(frame)
        calibration = load_measurement_calibration()
        boxes, box_rules = predict_yolo_boxes_with_rules(
            frame,
            conf=conf,
            exclusion_zones=calibration.get("exclusion_zones"),
        )
        hgt, wid = frame.shape[:2]
        pieces, measurement_summary = analyze_piece_boxes(
            frame,
            boxes,
            calibration,
            frame_idx=frame_idx,
            time_sec=time_sec,
        )
        primary = primary_piece_analysis(pieces)
        sobel = primary["sobel"] if primary else empty_sobel_result(frame_idx, time_sec)
        measurement = primary["measurement"] if primary else None
        return jsonify(
            image=img_to_b64(frame, quality=88),
            width=wid,
            height=hgt,
            frame_idx=frame_idx,
            time_sec=time_sec,
            boxes=boxes,
            count=len(boxes),
            box_rules=box_rules,
            pieces=pieces,
            measurement_summary=measurement_summary,
            sobel=sobel,
            calibration=calibration,
            measurement=measurement,
            model=str(_args.model),
            source=source,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/mvp/frame", methods=["POST"])
def api_mvp_frame():
    data = request.get_json() or {}
    try:
        frame_idx = int(data.get("frame_idx", 0))
        conf = float(data.get("conf", 0.10))
        original, frame_idx, time_sec, source = read_active_frame_by_index(frame_idx)
        rectified, homography = apply_saved_homography(original)
        matrix = np.asarray(homography["homography_matrix"], dtype=np.float64)
        calibration = load_measurement_calibration()
        boxes, box_rules = predict_yolo_boxes_with_rules(
            rectified,
            conf=conf,
            exclusion_zones=calibration.get("exclusion_zones"),
        )

        src_h, src_w = original.shape[:2]
        rect_h, rect_w = rectified.shape[:2]
        pieces, measurement_summary = analyze_piece_boxes(
            rectified,
            boxes,
            calibration,
            frame_idx=frame_idx,
            time_sec=time_sec,
        )
        primary = primary_piece_analysis(pieces)
        sobel = primary["sobel"] if primary else empty_sobel_result(frame_idx, time_sec)
        measurement = primary["measurement"] if primary else None
        original_overlay = mvp_original_overlay_for_pieces(
            pieces,
            calibration,
            matrix,
            rect_w,
            homography.get("lens_correction"),
            original.shape[:2],
        )
        return jsonify(
            original_image=img_to_b64(original, quality=82),
            rectified_image=img_to_b64(rectified, quality=82),
            original_width=src_w,
            original_height=src_h,
            rectified_width=rect_w,
            rectified_height=rect_h,
            frame_idx=frame_idx,
            time_sec=time_sec,
            fps=float(_video_playlist["fps"]),
            total_frames=int(_video_playlist["total_frames"]),
            boxes=boxes,
            count=len(boxes),
            box_rules=box_rules,
            pieces=pieces,
            measurement_summary=measurement_summary,
            sobel=sobel,
            calibration=calibration,
            measurement=measurement,
            original_overlay=original_overlay,
            front_y_ratio=(float(sobel["line"]["y"]) / float(rect_h)) if sobel.get("line") else None,
            source=source,
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/player/captures")
def api_player_captures():
    try:
        captures = load_player_captures()
        return jsonify(captures=captures, count=len(captures), path=str(player_capture_json_path()))
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/player/captures", methods=["POST"])
def api_player_capture_save():
    data = request.get_json() or {}
    try:
        frame_idx = int(data["frame_idx"])
        frame, frame_idx, time_sec, source = read_active_frame_by_index(frame_idx)
        frame, _homography = apply_saved_homography(frame)
        hgt, wid = frame.shape[:2]

        capture_id = frame_stem(frame_idx)
        capture_folder = player_capture_dir()
        capture_folder.mkdir(parents=True, exist_ok=True)
        image_path = capture_folder / f"{capture_id}.jpg"
        cv2.imwrite(str(image_path), frame)

        measurement = data.get("measurement") if isinstance(data.get("measurement"), dict) else None
        measurement_in = None
        if measurement is not None:
            raw_value = measurement.get("measurement_in", measurement.get("delta_in"))
            if raw_value is not None:
                measurement_in = float(raw_value)

        sobel = data.get("sobel") if isinstance(data.get("sobel"), dict) else None
        boxes = data.get("boxes") if isinstance(data.get("boxes"), list) else []
        pieces = data.get("pieces") if isinstance(data.get("pieces"), list) else []
        measurement_summary = (
            data.get("measurement_summary")
            if isinstance(data.get("measurement_summary"), dict)
            else summarize_piece_measurements(pieces)
        )
        capture = {
            "id": capture_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "frame_idx": frame_idx,
            "time_sec": time_sec,
            "img_w": wid,
            "img_h": hgt,
            "measurement_in": measurement_in,
            "measurement": measurement,
            "sobel_valid": bool(sobel.get("is_valid")) if sobel else False,
            "sobel": sobel,
            "boxes_count": len(boxes),
            "pieces": pieces,
            "measurement_summary": measurement_summary,
            "measurement_count": int(measurement_summary.get("valid_count", 0)),
            "image": str(image_path),
            "source": source,
        }

        captures = [item for item in load_player_captures() if item.get("id") != capture_id]
        captures.append(capture)
        captures = sorted(captures, key=lambda item: int(item.get("frame_idx", 0)))
        save_player_captures(captures)
        return jsonify(capture=capture, captures=captures, count=len(captures), path=str(player_capture_json_path()))
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/player/captures/<capture_id>", methods=["DELETE"])
def api_player_capture_delete(capture_id: str):
    try:
        captures = load_player_captures()
        target = next((item for item in captures if str(item.get("id")) == capture_id), None)
        captures = [item for item in captures if str(item.get("id")) != capture_id]
        if target and target.get("image"):
            image_path = Path(str(target["image"]))
            if image_path.exists() and image_path.parent.resolve() == player_capture_dir().resolve():
                image_path.unlink()
        save_player_captures(captures)
        return jsonify(captures=captures, count=len(captures), deleted=target is not None)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/annotate/history")
def api_annotate_history():
    try:
        items = dataset_history()
        return jsonify(
            items=items,
            count=saved_piece_annotation_count(items),
            candidate_count=sum(1 for item in items if item.get("candidate")),
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/annotate/save", methods=["POST"])
def api_annotate_save():
    global _piece_box_profile_cache
    data = request.get_json() or {}
    try:
        frame_idx = int(data["frame_idx"])
        time_sec = float(data["time_sec"])
        img_w = int(data["img_w"])
        img_h = int(data["img_h"])
        boxes = data["boxes"]

        frame, _frame_idx, _time_sec, source = read_active_frame_by_index(frame_idx)
        frame, _homography = apply_saved_homography(frame)

        images_dir = _args.dataset_dir / "images"
        labels_dir = _args.dataset_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        stem = frame_stem(frame_idx)
        image_path = images_dir / f"{stem}.jpg"
        label_path = labels_dir / f"{stem}.txt"
        meta_path = labels_dir / f"{stem}.json"

        cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        lines = []
        for box in boxes:
            x = float(box["x"])
            y = float(box["y"])
            width = float(box["w"])
            height = float(box["h"])
            cx = (x + width / 2) / img_w
            cy = (y + height / 2) / img_h
            nw = width / img_w
            nh = height / img_h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        label_path.write_text("\n".join(lines), encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "frame_idx": frame_idx,
                    "time_sec": time_sec,
                    "img_w": img_w,
                    "img_h": img_h,
                    "boxes": boxes,
                    "class_id": 0,
                    "class_name": "piece",
                    "source": source,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _piece_box_profile_cache = None
        history = dataset_history()
        return jsonify(
            saved_count=saved_piece_annotation_count(history),
            history=history,
            image=str(image_path),
            label=str(label_path),
            metadata=str(meta_path),
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


def main() -> None:
    global _args, _video_playlist, _image, _source_label
    _args = parse_args()
    video_paths = selected_video_paths(_args)
    _video_playlist = build_video_playlist(video_paths)
    _args.video = Path(_video_playlist["segments"][0]["path"])
    _args.output_dir.mkdir(parents=True, exist_ok=True)
    _args.dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        _image, _source_label, _frame_idx, _time_sec = load_reference_image(_args)
    except Exception as exc:
        print(f"Advertencia al precargar frame: {exc}")
        _image = None
        _source_label = str(_args.video_dir or _args.video)
    print(
        f"  Playlist: {len(_video_playlist['segments'])} videos, "
        f"{_video_playlist['duration_sec'] / 60.0:.1f} min"
    )
    print(f"\n  TX2 Vision Tool en http://localhost:{_args.port}\n")
    app.run(host="0.0.0.0", port=_args.port, debug=False)


if __name__ == "__main__":
    main()
