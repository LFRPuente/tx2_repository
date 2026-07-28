"""
Homography + YOLO annotation web app.

Usage:
    python homography_web_app.py --video "path/to/video.mkv" --second 155.0
"""
from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = Path(r"C:\Users\luis_\Downloads\20260724_10")
DEFAULT_VIDEO = DEFAULT_VIDEO_DIR / "20260724_100105_6439.mkv"
DEFAULT_SECOND = 30.0
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset_pieces"
DEFAULT_PIECE_MODEL_V2 = (
    PROJECT_ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v2" / "weights" / "best.pt"
)
DEFAULT_PIECE_MODEL_V1 = (
    PROJECT_ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v1" / "weights" / "best.pt"
)
DEFAULT_LEGACY_MODEL = (
    PROJECT_ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_tubos_v1" / "weights" / "best.pt"
)
DEFAULT_MODEL = next(
    (
        path
        for path in (DEFAULT_PIECE_MODEL_V2, DEFAULT_PIECE_MODEL_V1, DEFAULT_LEGACY_MODEL)
        if path.exists()
    ),
    DEFAULT_LEGACY_MODEL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path)
    source.add_argument("--video-dir", type=Path)
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
):
    ordered = order_points(np.asarray(pts, dtype=np.float32))
    base_w, base_h = rect_size_from_ordered(ordered)
    margins = normalize_roi_margins(roi_margins, base_w, base_h, expand_pct)
    warp_points, base_matrix, margin_size, work_rect = work_roi_from_margins(ordered, margins)
    width = margin_size[0] if dest_w <= 0 else dest_w
    height = margin_size[1] if dest_h <= 0 else dest_h
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
            "path": str(path),
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("reference_offset_in", MEASUREMENT_REFERENCE_OFFSET_IN)
    data.setdefault("exclusion_zones", [])
    data.setdefault("exclusion_max_box_overlap", EXCLUSION_ZONE_MAX_BOX_OVERLAP)
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
    inch_per_px = calibration.get("inch_per_px")
    reference_y = calibration.get("reference_y")
    if inch_per_px is None or reference_y is None:
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
    delta_in = delta_px * float(inch_per_px)
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
        "inch_per_px": float(inch_per_px),
    }


def rectified_line_to_original(line: dict | None, matrix: np.ndarray) -> dict | None:
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
        return {
            "x1": float(mapped[0][0]),
            "y1": float(mapped[0][1]),
            "x2": float(mapped[1][0]),
            "y2": float(mapped[1][1]),
        }
    except Exception:
        return None


def mvp_original_overlay(sobel: dict, calibration: dict, matrix: np.ndarray, rect_width: int) -> dict:
    reference_y = calibration.get("reference_y")
    reference_line = None
    if reference_y is not None:
        y = float(reference_y)
        reference_line = {"x1": 0.0, "y1": y, "x2": float(rect_width - 1), "y2": y}

    return {
        "front_line": rectified_line_to_original((sobel or {}).get("line"), matrix),
        "reference_line": rectified_line_to_original(reference_line, matrix),
    }


def mvp_original_overlay_for_pieces(
    pieces: list[dict],
    calibration: dict,
    matrix: np.ndarray,
    rect_width: int,
) -> dict:
    overlay = mvp_original_overlay({}, calibration, matrix, rect_width)
    piece_fronts = []
    for piece in pieces:
        sobel = piece.get("sobel") if isinstance(piece.get("sobel"), dict) else {}
        mapped_line = rectified_line_to_original(sobel.get("line"), matrix)
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
    matrix = np.array(data["homography_matrix"], dtype=np.float64)
    width, height = data["output_size"]
    return matrix, (int(width), int(height)), data


def apply_saved_homography(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    matrix, out_size, data = load_homography()
    return cv2.warpPerspective(frame, matrix, out_size), data


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

    profile_width = active_profile.get("median_width_ratio")
    profile_height = active_profile.get("median_height_ratio")
    profile_pitch = active_profile.get("median_pitch_ratio")
    typical_width = float(profile_width) * image_width if profile_width else frame_width
    typical_height = float(profile_height) * image_height if profile_height else frame_height

    if widths and 0.60 <= frame_width / max(typical_width, 1e-6) <= 1.60:
        typical_width = float(np.median([typical_width, frame_width]))
    if heights and 0.45 <= frame_height / max(typical_height, 1e-6) <= 2.00:
        typical_height = float(np.median([typical_height, frame_height]))

    centers = sorted(float(box["x"]) + float(box["w"]) / 2.0 for box in boxes)
    close_gaps = [
        right - left
        for left, right in zip(centers, centers[1:])
        if typical_width * 0.55 <= right - left <= typical_width * 1.60
    ]
    if profile_pitch:
        typical_pitch = float(profile_pitch) * image_width
    elif close_gaps:
        typical_pitch = float(np.median(close_gaps))
    else:
        typical_pitch = typical_width * 1.10
    if close_gaps and 0.70 <= float(np.median(close_gaps)) / max(typical_pitch, 1e-6) <= 1.35:
        typical_pitch = float(np.median([typical_pitch, float(np.median(close_gaps))]))

    return {
        "source": active_profile.get("source", "frame"),
        "sample_count": int(active_profile.get("sample_count", 0)),
        "typical_width": typical_width,
        "typical_height": typical_height,
        "typical_pitch": typical_pitch,
    }


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

    size_outliers = []
    for box in deduplicated:
        width_ratio = float(box["w"]) / typical_width
        height_ratio = float(box["h"]) / typical_height
        box["size_outlier"] = not (0.70 <= width_ratio <= 1.35 and 0.55 <= height_ratio <= 1.65)
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
        severe_size_outlier = not (0.65 <= width_ratio <= 1.40 and 0.60 <= height_ratio <= 1.60)
        if looks_like_individual_pieces and severe_size_outlier:
            removed_size.append(box)
        else:
            size_filtered.append(box)

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
        "missing_candidate_count": len(proposals),
        "inferred_count": len(accepted_inferred),
        "rejected_inferred_count": len(rejected_inferred),
        "profile_source": reference["source"],
        "profile_sample_count": reference["sample_count"],
        "typical_width_px": reference["typical_width"],
        "typical_height_px": reference["typical_height"],
        "typical_pitch_px": reference["typical_pitch"],
    }
    return processed, diagnostics


def load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        if not _args.model.exists():
            raise RuntimeError(f"No existe el modelo YOLO: {_args.model}")
        _yolo_model = YOLO(str(_args.model))
    return _yolo_model


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
    result = model.predict(rectified, conf=conf, imgsz=imgsz, verbose=False)[0]
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


HTML = r"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TX2 Vision Tool</title>
<style>
:root {
  --bg: #111315;
  --panel: #191d20;
  --panel-2: #20262a;
  --line: #30383d;
  --text: #ece7dc;
  --muted: #a9b0ad;
  --accent: #53b689;
  --accent-2: #d6a34b;
  --blue: #5b9bd5;
  --danger: #d45b5b;
  --shadow: 0 16px 32px rgba(0, 0, 0, .28);
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  overflow: hidden;
}
button, input { font: inherit; }
button {
  border: 1px solid var(--line);
  background: #252b2f;
  color: var(--text);
  height: 32px;
  padding: 0 12px;
  border-radius: 6px;
  cursor: pointer;
  white-space: nowrap;
}
button:hover { border-color: #58646a; background: #2b3338; }
button:disabled { opacity: .45; cursor: default; }
.primary { background: var(--accent); color: #07120d; border-color: var(--accent); font-weight: 700; }
.warn { background: var(--accent-2); color: #171006; border-color: var(--accent-2); font-weight: 700; }
.danger { background: #442727; color: #ffd2d2; border-color: #6a3939; }
.ghost.active { background: #334039; border-color: var(--accent); color: #bff4d8; }
.zone-toggle { color: #ffb8b8; border-color: #733d3d; }
.zone-toggle.active { background: #562b2b; border-color: #ef6666; color: #ffffff; }
input[type=number] {
  width: 88px;
  height: 32px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #121517;
  color: var(--text);
  padding: 0 8px;
}
input[type=range] { accent-color: var(--accent); }
.app {
  display: grid;
  grid-template-rows: 52px 44px 1fr 30px;
  height: 100%;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 0 18px;
  border-bottom: 1px solid var(--line);
  background: #15181a;
}
.brand { display: flex; align-items: baseline; gap: 12px; min-width: 0; }
.brand h1 { margin: 0; font-size: 16px; letter-spacing: 0; }
.source { color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 58vw; }
.tabs { display: flex; gap: 8px; }
.toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 14px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-width: none;
}
.toolbar::-webkit-scrollbar { display: none; }
.toolbar label { color: var(--muted); font-size: 12px; }
.timeline-range { width: clamp(180px, 24vw, 360px); flex: 0 0 auto; }
.spacer { flex: 1; }
.pill {
  border: 1px solid var(--line);
  background: #121517;
  color: var(--muted);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  white-space: nowrap;
}
.pill.ok { color: #bff4d8; border-color: #366b54; background: #14241d; }
.workspace { min-height: 0; }
.view { height: 100%; display: none; }
.view.active { display: grid; }
#homography-view { grid-template-columns: minmax(0, 1fr) minmax(360px, .72fr); }
#annotate-view, #player-view, #measure-view { grid-template-columns: minmax(0, 1fr) 320px; }
.pane {
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--line);
}
.pane:last-child { border-right: 0; }
.pane-title {
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 12px;
  background: var(--panel-2);
  color: var(--muted);
  font-size: 12px;
  border-bottom: 1px solid var(--line);
}
.canvas-wrap {
  position: relative;
  flex: 1;
  min-height: 0;
  overflow: hidden;
  background: #050607;
  cursor: crosshair;
}
canvas { display: block; width: 100%; height: 100%; }
.hud {
  position: absolute;
  left: 10px;
  bottom: 10px;
  background: rgba(8, 10, 11, .82);
  border: 1px solid rgba(255,255,255,.08);
  color: #ccebdc;
  padding: 4px 8px;
  font: 12px Consolas, monospace;
  border-radius: 6px;
  pointer-events: none;
}
.points, .side {
  background: #15191b;
  border-top: 1px solid var(--line);
}
.points {
  min-height: 54px;
  padding: 8px 12px;
  color: var(--muted);
  font-size: 12px;
}
.side { border-top: 0; overflow: auto; }
.section { padding: 12px; border-bottom: 1px solid var(--line); }
.section h2 { margin: 0 0 8px; font-size: 13px; color: #d8d0c3; }
.kv { display: flex; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 12px; padding: 3px 0; }
.kv strong { color: #ccebdc; font-weight: 600; text-align: right; }
#p-info-measure { color: #ffffff; font-size: 16px; font-weight: 800; }
.box-list { display: flex; flex-direction: column; gap: 6px; }
.history-list { display: flex; flex-direction: column; gap: 6px; max-height: 180px; overflow: auto; }
.history-item {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
  align-items: center;
  border: 1px solid var(--line);
  background: #111517;
  border-radius: 6px;
  padding: 7px;
  color: var(--muted);
  font-size: 12px;
}
.history-item.current { border-color: var(--accent); background: #14241d; }
.history-item strong { color: #ccebdc; display: block; font-size: 12px; }
.history-item span { display: block; font-size: 11px; }
.history-item button { height: 26px; padding: 0 9px; }
.history-actions { display: flex; gap: 6px; }
.history-actions .danger { color: #f1c7c7; border-color: rgba(212,91,91,.45); }
.ruler-tool { position: relative; display: inline-flex; align-items: center; }
.icon-tool { display: inline-flex; align-items: center; gap: 6px; }
.ruler-icon {
  width: 18px;
  height: 8px;
  display: inline-block;
  position: relative;
  transform: rotate(-12deg);
  border: 1px solid currentColor;
  border-radius: 2px;
}
.ruler-icon::before {
  content: "";
  position: absolute;
  inset: 1px 2px;
  background: repeating-linear-gradient(90deg, currentColor 0 1px, transparent 1px 4px);
  opacity: .9;
}
.ruler-menu {
  position: absolute;
  right: 0;
  top: calc(100% + 6px);
  display: none;
  gap: 6px;
  padding: 7px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #111517;
  box-shadow: 0 12px 28px rgba(0,0,0,.28);
  z-index: 20;
}
.ruler-tool.active:hover .ruler-menu, .ruler-tool.active:focus-within .ruler-menu { display: flex; }
.ruler-tool.active > .icon-tool { background: #334039; border-color: var(--accent); color: #bff4d8; }
.ruler-menu button.active { background: #334039; border-color: var(--accent); color: #bff4d8; }
.box-item {
  display: grid;
  grid-template-columns: 12px 1fr auto;
  gap: 8px;
  align-items: center;
  border: 1px solid var(--line);
  background: #111517;
  border-radius: 6px;
  padding: 7px;
  color: var(--muted);
  font-size: 12px;
}
.swatch { width: 12px; height: 12px; border-radius: 3px; }
.box-item button { height: 24px; padding: 0 8px; }
.status {
  display: flex;
  align-items: center;
  padding: 0 14px;
  color: var(--muted);
  font-size: 12px;
  border-top: 1px solid var(--line);
  background: #15181a;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.status.ok { color: #bff4d8; }
.status.err { color: #ffb7b7; }
@media (max-width: 980px) {
  #homography-view, #annotate-view, #player-view, #measure-view { grid-template-columns: 1fr; grid-template-rows: 1fr 42%; }
  .source { max-width: 40vw; }
}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <h1>TX2 Vision Tool</h1>
      <div class="source" id="source-label">{{ source_label }}</div>
    </div>
    <nav class="tabs">
      <button class="ghost active" id="tab-homography" onclick="showView('homography')">Homografia</button>
      <button class="ghost" id="tab-measure" onclick="showView('measure')">Mediciones</button>
      <button class="ghost" id="tab-annotate" onclick="showView('annotate')">Anotar YOLO</button>
      <button class="ghost" id="tab-player" onclick="showView('player')">Reproductor</button>
    </nav>
  </header>

  <section class="toolbar" id="homography-toolbar">
    <label>Segundo</label>
    <input type="number" id="h-second" value="{{ second }}" step="0.5" min="0">
    <button onclick="loadHomographyFrame()">Cargar frame</button>
    <button onclick="loadSavedHomography()">Cargar guardada</button>
    <label>Zoom</label>
    <input type="range" id="h-zoom" min="1" max="20" step="0.1" value="1">
    <span class="pill" id="h-zoom-label">1.0x</span>
    <label>Expandir ROI</label>
    <input type="range" id="h-expand" min="0" max="40" step="1" value="0">
    <span class="pill" id="h-expand-label">0%</span>
    <button onclick="resetWorkRoi()">Reset ROI</button>
    <span class="spacer"></span>
    <span class="pill" id="points-badge">0 / 4 puntos</span>
    <button onclick="undoPoint()">Deshacer</button>
    <button class="danger" onclick="resetPoints()">Reiniciar</button>
    <button class="primary" id="save-homography" onclick="saveHomography()" disabled>Guardar homografia</button>
  </section>

  <section class="toolbar" id="annotate-toolbar" style="display:none">
    <label>Segundo</label>
    <input type="number" id="a-second" value="{{ second }}" step="1" min="0">
    <button onclick="loadAnnotateSecond()">Ir</button>
    <button onclick="stepAnnotate(-30)" title="Retroceder 1 segundo">-1s</button>
    <button onclick="stepAnnotate(-1)" title="Retroceder 1 frame">-1f</button>
    <button onclick="stepAnnotate(1)" title="Avanzar 1 frame">+1f</button>
    <button onclick="stepAnnotate(30)" title="Avanzar 1 segundo">+1s</button>
    <button onclick="stepAnnotateSeconds(-30)" title="Retroceder 30 segundos">-30s</button>
    <button onclick="stepAnnotateSeconds(30)" title="Avanzar 30 segundos">+30s</button>
    <label>Timeline</label>
    <input type="range" id="a-timeline" class="timeline-range" min="0" max="0" step="1" value="0"
      title="Arrastra o usa la rueda para navegar por todos los videos"
      oninput="previewAnnotateTimeline(this.value)"
      onchange="loadAnnotateTimeline(this.value)"
      onwheel="scrollAnnotateTimeline(event)">
    <span class="pill" id="a-timeline-label">00:00:00</span>
    <label>Zoom</label>
    <input type="range" id="a-zoom" min="1" max="20" step="0.1" value="1">
    <span class="pill" id="a-zoom-label">1.0x</span>
    <button id="a-run-model" onclick="runModel()">Run model</button>
    <span class="spacer"></span>
    <button onclick="undoBox()">Deshacer</button>
    <button class="danger" onclick="clearBoxes()">Limpiar</button>
    <button class="primary" id="save-frame" onclick="saveFrame()" disabled>Guardar frame</button>
  </section>

  <section class="toolbar" id="measure-toolbar" style="display:none">
    <label>Segundo</label>
    <input type="number" id="m-second" value="{{ second }}" step="1" min="0">
    <button onclick="loadMeasureSecond()">Ir</button>
    <button onclick="reloadMeasureRoi()" title="Recargar usando la homografia/ROI guardada mas reciente">Recargar ROI</button>
    <button onclick="stepMeasure(-30)" title="Retroceder 1 segundo">-1s</button>
    <button onclick="stepMeasure(30)" title="Avanzar 1 segundo">+1s</button>
    <button id="m-mode-segment" class="ghost active" onclick="setMeasureMode('segment')">Segmento</button>
    <button id="m-mode-ref" class="ghost" onclick="setMeasureMode('reference')">Linea Y</button>
    <button id="m-mode-exclusion" class="zone-toggle" onclick="setMeasureMode('exclusion')">Zona roja</button>
    <label>Zoom</label>
    <input type="range" id="m-zoom" min="1" max="20" step="0.1" value="1">
    <span class="pill" id="m-zoom-label">1.0x</span>
    <span class="spacer"></span>
    <button onclick="undoMeasureSegment()">Deshacer</button>
    <button class="danger" onclick="clearMeasureCalibration()">Limpiar</button>
    <button class="primary" onclick="saveMeasureCalibration()">Guardar mediciones</button>
  </section>

  <section class="toolbar" id="player-toolbar" style="display:none">
    <label>Segundo</label>
    <input type="number" id="p-second" value="{{ second }}" step="1" min="0">
    <button onclick="loadPlayerSecond()">Ir</button>
    <button id="p-play" onclick="togglePlayerPlay()">Play</button>
    <button id="p-speed" onclick="togglePlayerSpeed()">x1</button>
    <button onclick="stepPlayer(-30)" title="Retroceder 1 segundo">-1s</button>
    <button onclick="stepPlayer(-1)" title="Retroceder 1 frame">-1f</button>
    <button onclick="stepPlayer(1)" title="Avanzar 1 frame">+1f</button>
    <button onclick="stepPlayer(30)" title="Avanzar 1 segundo">+1s</button>
    <label>Timeline</label>
    <input type="range" id="p-timeline" class="timeline-range" min="0" max="0" step="1" value="0"
      title="Arrastra o usa la rueda para navegar por todos los videos"
      oninput="previewPlayerTimeline(this.value)"
      onchange="loadPlayerTimeline(this.value)"
      onwheel="scrollPlayerTimeline(event)">
    <span class="pill" id="p-timeline-label">00:00:00</span>
    <label>Zoom</label>
    <input type="range" id="p-zoom" min="1" max="20" step="0.1" value="1">
    <span class="pill" id="p-zoom-label">1.0x</span>
    <span class="spacer"></span>
    <div class="ruler-tool" id="p-ruler-tool">
      <button class="icon-tool" id="p-ruler-toggle" onclick="togglePlayerRulerTool()" title="Regla manual">
        <span class="ruler-icon"></span><span>Regla</span>
      </button>
      <div class="ruler-menu">
        <button id="p-ruler-x" onclick="setPlayerRulerMode('x')">X</button>
        <button id="p-ruler-y" onclick="setPlayerRulerMode('y')">Y</button>
        <button id="p-ruler-free" onclick="setPlayerRulerMode('free')">Libre</button>
        <button class="danger" onclick="clearPlayerRuler()">Limpiar</button>
      </div>
    </div>
    <button class="primary" onclick="savePlayerCapture()">Guardar medicion</button>
    <span class="pill" id="p-yolo-badge">YOLO -</span>
    <span class="pill" id="p-sobel-badge">Sobel -</span>
  </section>

  <main class="workspace">
    <section class="view active" id="homography-view">
      <div class="pane">
        <div class="pane-title"><span>Fuente</span><span>click: punto base | arrastra lados azules: ROI | rueda: zoom</span></div>
        <div class="canvas-wrap" id="h-wrap">
          <canvas id="h-canvas"></canvas>
          <div class="hud" id="h-hud">x: - y: -</div>
        </div>
        <div class="points" id="points-list">Sin puntos.</div>
      </div>
      <div class="pane">
        <div class="pane-title"><span>Warp preview</span><span id="warp-size">sin homografia</span></div>
        <div class="canvas-wrap" id="w-wrap" style="cursor:default">
          <canvas id="w-canvas"></canvas>
        </div>
      </div>
    </section>

    <section class="view" id="annotate-view">
      <div class="pane">
        <div class="pane-title"><span>Video rectificado</span><span>click x2: box | rueda: zoom | S: guardar</span></div>
        <div class="canvas-wrap" id="a-wrap">
          <canvas id="a-canvas"></canvas>
          <div class="hud" id="a-hud">x: - y: -</div>
        </div>
      </div>
      <aside class="side">
        <div class="section">
          <h2>Frame actual</h2>
          <div class="kv"><span>Video</span><strong id="info-video">-</strong></div>
          <div class="kv"><span>Frame origen</span><strong id="info-source-frame">-</strong></div>
          <div class="kv"><span>Frame</span><strong id="info-frame">-</strong></div>
          <div class="kv"><span>Tiempo</span><strong id="info-time">-</strong></div>
          <div class="kv"><span>Tamano</span><strong id="info-size">-</strong></div>
          <div class="kv"><span>Boxes</span><strong id="info-boxes">0</strong></div>
        </div>
        <div class="section">
          <h2>Dataset</h2>
          <div class="kv"><span>Frames guardados</span><strong id="info-saved">0</strong></div>
          <div class="kv"><span>Salida</span><strong id="info-dataset">dataset</strong></div>
        </div>
        <div class="section">
          <h2>Historial</h2>
          <div class="history-list" id="history-list"></div>
        </div>
        <div class="section">
          <h2>Run model</h2>
          <div class="kv"><span>Detecciones</span><strong id="info-model-boxes">-</strong></div>
          <div class="kv"><span>Coincidencias</span><strong id="info-model-matched">-</strong></div>
          <div class="kv"><span>Omitidas</span><strong id="info-model-missed">-</strong></div>
          <div class="box-list" id="model-box-list"></div>
        </div>
        <div class="section">
          <h2>Anotaciones</h2>
          <div class="box-list" id="box-list"></div>
        </div>
      </aside>
    </section>

    <section class="view" id="measure-view">
      <div class="pane">
        <div class="pane-title"><span>Mesa rectificada</span><span>Segmento/Zona roja: 2 clicks | Linea Y: click/arrastrar | rueda: zoom</span></div>
        <div class="canvas-wrap" id="m-wrap">
          <canvas id="m-canvas"></canvas>
          <div class="hud" id="m-hud">x: - y: -</div>
        </div>
      </div>
      <aside class="side">
        <div class="section">
          <h2>Frame</h2>
          <div class="kv"><span>Video</span><strong id="m-info-video">-</strong></div>
          <div class="kv"><span>Frame origen</span><strong id="m-info-source-frame">-</strong></div>
          <div class="kv"><span>Frame</span><strong id="m-info-frame">-</strong></div>
          <div class="kv"><span>Tiempo</span><strong id="m-info-time">-</strong></div>
          <div class="kv"><span>Tamano</span><strong id="m-info-size">-</strong></div>
        </div>
        <div class="section">
          <h2>Escala</h2>
          <div class="kv"><span>inch/px</span><strong id="m-info-inch-px">-</strong></div>
          <div class="kv"><span>px/in</span><strong id="m-info-px-inch">-</strong></div>
          <div class="kv"><span>Segmentos</span><strong id="m-info-segments">0</strong></div>
        </div>
        <div class="section">
          <h2>Referencia Y</h2>
          <div class="kv"><span>Linea</span><strong id="m-info-ref">-</strong></div>
        </div>
        <div class="section">
          <h2>Zonas sin boxes</h2>
          <div class="kv"><span>Zonas</span><strong id="m-info-zones">0</strong></div>
          <div class="kv"><span>Solapamiento permitido</span><strong>20%</strong></div>
          <div class="box-list" id="m-zone-list"></div>
        </div>
        <div class="section">
          <h2>Segmentos</h2>
          <div class="box-list" id="m-segment-list"></div>
        </div>
      </aside>
    </section>

    <section class="view" id="player-view">
      <div class="pane">
        <div class="pane-title"><span>Reproductor de piezas YOLO + Sobel</span><span>Sobel Y en el frente inferior de cada pieza</span></div>
        <div class="canvas-wrap" id="p-wrap" style="cursor:default">
          <canvas id="p-canvas"></canvas>
          <div class="hud" id="p-hud">x: - y: -</div>
        </div>
      </div>
      <aside class="side">
        <div class="section">
          <h2>Frame actual</h2>
          <div class="kv"><span>Video</span><strong id="p-info-video">-</strong></div>
          <div class="kv"><span>Frame origen</span><strong id="p-info-source-frame">-</strong></div>
          <div class="kv"><span>Frame</span><strong id="p-info-frame">-</strong></div>
          <div class="kv"><span>Tiempo</span><strong id="p-info-time">-</strong></div>
          <div class="kv"><span>Tamano</span><strong id="p-info-size">-</strong></div>
          <div class="kv"><span>Velocidad</span><strong id="p-info-speed">x1</strong></div>
        </div>
        <div class="section">
          <h2>YOLO</h2>
          <div class="kv"><span>Piezas</span><strong id="p-info-boxes">0</strong></div>
          <div class="kv"><span>Conf</span><strong id="p-info-yolo-conf">-</strong></div>
          <div class="kv"><span>Descartadas por zona</span><strong id="p-info-excluded">0</strong></div>
        </div>
        <div class="section">
          <h2>Mediciones por pieza</h2>
          <div class="kv"><span>Estado</span><strong id="p-info-sobel-state">sin correr</strong></div>
          <div class="kv"><span>Conf edge</span><strong id="p-info-sobel-conf">-</strong></div>
          <div class="kv"><span>CRM</span><strong id="p-info-sobel-crm">-</strong></div>
          <div class="kv"><span>Validas</span><strong id="p-info-measure">-</strong></div>
          <div class="box-list" id="p-piece-list"></div>
        </div>
        <div class="section">
          <h2>Regla manual</h2>
          <div class="kv"><span>Modo</span><strong id="p-ruler-info-mode">off</strong></div>
          <div class="kv"><span>Escala</span><strong id="p-ruler-info-scale">-</strong></div>
          <div class="kv"><span>Distancia</span><strong id="p-ruler-info-distance">-</strong></div>
          <div class="kv"><span>Delta</span><strong id="p-ruler-info-delta">-</strong></div>
        </div>
        <div class="section">
          <h2>Capturas</h2>
          <div class="history-list" id="p-capture-list"></div>
        </div>
      </aside>
    </section>
  </main>
  <footer class="status" id="status">Listo.</footer>
</div>

<script>
const COLORS = ['#53b689', '#d6a34b', '#5b9bd5', '#d45b5b', '#a78bd6', '#70c7c2', '#e18f62'];
const meta = { fps: 30, totalFrames: 0 };

const h = {
  canvas: document.getElementById('h-canvas'), wrap: document.getElementById('h-wrap'),
  points: [], img: null, imgW: 0, imgH: 0, zoom: 1, panX: 0, panY: 0,
  panning: false, panAnchor: null, panStart: null, didDrag: false, warpImg: null,
  expandedPoints: [], expandPct: 0, roiManual: false, draggingRoiSide: null,
  roiMargins: {left: 0, right: 0, top: 0, bottom: 0}, baseMatrix: null, baseSize: null
};
h.ctx = h.canvas.getContext('2d');
const w = { canvas: document.getElementById('w-canvas'), wrap: document.getElementById('w-wrap') };
w.ctx = w.canvas.getContext('2d');

const a = {
  canvas: document.getElementById('a-canvas'), wrap: document.getElementById('a-wrap'),
  img: null, imgW: 0, imgH: 0, zoom: 1, panX: 0, panY: 0,
  frameIdx: 0, timeSec: 0, source: null, boxes: [], cornerA: null, preview: null,
  panning: false, panAnchor: null, panStart: null, saved: 0, history: [], sobel: null, pieces: [],
  timelineTimer: null, modelBoxes: [], boxRules: null
};
a.ctx = a.canvas.getContext('2d');

const m = {
  canvas: document.getElementById('m-canvas'), wrap: document.getElementById('m-wrap'),
  img: null, imgW: 0, imgH: 0, zoom: 1, panX: 0, panY: 0,
  frameIdx: 0, timeSec: 0, source: null, segments: [], pending: null, preview: null,
  referenceY: null, inchPerPx: null, exclusionZones: [], mode: 'segment', draggingReference: false,
  selectedSegment: null, draggingSegment: null,
  panning: false, panAnchor: null, panStart: null, didDrag: false
};
m.ctx = m.canvas.getContext('2d');

const p = {
  canvas: document.getElementById('p-canvas'), wrap: document.getElementById('p-wrap'),
  img: null, imgW: 0, imgH: 0, zoom: 1, panX: 0, panY: 0,
  frameIdx: 0, timeSec: 0, source: null, boxes: [], pieces: [], measurementSummary: null,
  sobel: null, calibration: null, measurement: null, boxRules: null,
  captures: [], rulerActive: false, rulerMode: 'free', rulerStart: null, rulerEnd: null, rulerPreview: null,
  playing: false, speed: 1, playTask: null, panning: false, panAnchor: null, panStart: null,
  calibrationStale: false, timelineTimer: null
};
p.ctx = p.canvas.getContext('2d');

function status(msg, cls='') {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status ' + cls;
}

function sourceName(source) {
  return (source?.video_name || '-').replace(/^\d{8}_/, '').replace(/\.mkv$/i, '');
}

function showView(name) {
  stopPlayerPlayback();
  document.getElementById('homography-view').classList.toggle('active', name === 'homography');
  document.getElementById('measure-view').classList.toggle('active', name === 'measure');
  document.getElementById('annotate-view').classList.toggle('active', name === 'annotate');
  document.getElementById('player-view').classList.toggle('active', name === 'player');
  document.getElementById('homography-toolbar').style.display = name === 'homography' ? 'flex' : 'none';
  document.getElementById('measure-toolbar').style.display = name === 'measure' ? 'flex' : 'none';
  document.getElementById('annotate-toolbar').style.display = name === 'annotate' ? 'flex' : 'none';
  document.getElementById('player-toolbar').style.display = name === 'player' ? 'flex' : 'none';
  document.getElementById('tab-homography').classList.toggle('active', name === 'homography');
  document.getElementById('tab-measure').classList.toggle('active', name === 'measure');
  document.getElementById('tab-annotate').classList.toggle('active', name === 'annotate');
  document.getElementById('tab-player').classList.toggle('active', name === 'player');
  fitAll();
  if (name === 'measure' && !m.img) loadMeasureSecond();
  if (name === 'annotate' && !a.img) loadAnnotateSecond();
  if (name === 'player') {
    if (!p.img) {
      loadPlayerSecond();
    } else if (p.calibrationStale) {
      loadPlayerFrame(p.frameIdx, {resetView: false});
    }
  }
  drawAll();
}

function fitAll() {
  for (const s of [h, w, a, m, p]) {
    s.canvas.width = s.wrap.clientWidth;
    s.canvas.height = s.wrap.clientHeight;
  }
}

function clamp(state) {
  if (!state.imgW || !state.imgH) return;
  const vw = state.imgW / state.zoom;
  const vh = state.imgH / state.zoom;
  state.panX = Math.max(0, Math.min(state.panX, Math.max(0, state.imgW - vw)));
  state.panY = Math.max(0, Math.min(state.panY, Math.max(0, state.imgH - vh)));
}

function displayToImage(state, cx, cy) {
  const vw = state.imgW / state.zoom;
  const vh = state.imgH / state.zoom;
  return {
    x: Math.max(0, Math.min(state.panX + (cx / state.canvas.width) * vw, state.imgW - 1)),
    y: Math.max(0, Math.min(state.panY + (cy / state.canvas.height) * vh, state.imgH - 1)),
  };
}

function imageToDisplay(state, ix, iy) {
  const vw = state.imgW / state.zoom;
  const vh = state.imgH / state.zoom;
  return {
    x: ((ix - state.panX) / vw) * state.canvas.width,
    y: ((iy - state.panY) / vh) * state.canvas.height,
  };
}

function resetView(state) {
  state.zoom = 1; state.panX = 0; state.panY = 0;
}

function clearWarpDependentViews() {
  a.img = null; a.imgW = 0; a.imgH = 0; a.boxes = []; a.pieces = [];
  a.cornerA = null; a.preview = null; a.sobel = null;
  m.img = null; m.imgW = 0; m.imgH = 0; m.pending = null; m.preview = null;
  p.img = null; p.imgW = 0; p.imgH = 0; p.boxes = []; p.pieces = [];
  p.measurementSummary = null; p.sobel = null; p.measurement = null; p.boxRules = null;
  updateBoxes();
  updateMeasureInfo();
  updatePlayerInfo();
}

async function loadMeta() {
  const r = await fetch('/api/meta');
  const d = await r.json();
  if (!r.ok) throw new Error(d.error);
  meta.fps = d.fps;
  meta.totalFrames = d.total_frames;
  const annotateTimeline = document.getElementById('a-timeline');
  annotateTimeline.max = Math.max(0, meta.totalFrames - 1);
  document.getElementById('a-second').max = Math.max(0, (meta.totalFrames - 1) / meta.fps).toFixed(3);
  document.getElementById('a-timeline-label').textContent = formatClock(0);
  const playerTimeline = document.getElementById('p-timeline');
  playerTimeline.max = Math.max(0, meta.totalFrames - 1);
  document.getElementById('p-second').max = Math.max(0, (meta.totalFrames - 1) / meta.fps).toFixed(3);
  document.getElementById('p-timeline-label').textContent = formatClock(0);
  document.getElementById('info-dataset').textContent = d.dataset_dir;
  await refreshHistory();
  await refreshPlayerCaptures();
}

async function loadHomographyFrame() {
  const second = parseFloat(document.getElementById('h-second').value) || 0;
  status('Cargando frame...', '');
  const r = await fetch('/api/frame', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({second})
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  const img = new Image();
  img.onload = () => {
    h.img = img; h.imgW = d.width; h.imgH = d.height;
    h.points = []; h.expandedPoints = []; h.roiManual = false;
    h.roiMargins = {left: 0, right: 0, top: 0, bottom: 0};
    h.baseMatrix = null; h.baseSize = null; h.warpImg = null; resetView(h);
    document.getElementById('source-label').textContent = d.label;
    document.getElementById('h-zoom').value = 1;
    document.getElementById('h-zoom-label').textContent = '1.0x';
    updatePoints();
    fitAll(); drawAll();
    status(`Frame cargado: ${d.width}x${d.height}`, 'ok');
  };
  img.src = 'data:image/jpeg;base64,' + d.image;
}

function drawHomography() {
  const ctx = h.ctx, cw = h.canvas.width, ch = h.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!h.img) return;
  const vw = h.imgW / h.zoom, vh = h.imgH / h.zoom;
  ctx.drawImage(h.img, h.panX, h.panY, vw, vh, 0, 0, cw, ch);
  drawGrid(h, 200);
  if (h.expandedPoints.length === 4) {
    ctx.save();
    ctx.beginPath();
    h.expandedPoints.forEach((p, i) => {
      const q = imageToDisplay(h, p.x, p.y);
      if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
    });
    ctx.closePath();
    ctx.strokeStyle = '#5b9bd5';
    ctx.lineWidth = 2;
    ctx.setLineDash([9, 6]);
    ctx.stroke();
    ctx.restore();
  }
  if (h.points.length >= 2) {
    ctx.beginPath();
    h.points.forEach((p, i) => {
      const q = imageToDisplay(h, p.x, p.y);
      if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
    });
    if (h.points.length === 4) ctx.closePath();
    ctx.strokeStyle = '#53b689';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  h.points.forEach((p, i) => {
    const q = imageToDisplay(h, p.x, p.y);
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.beginPath(); ctx.arc(q.x, q.y, 8, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#fff'; ctx.font = '700 12px Arial';
    ctx.fillText(String(i + 1), q.x + 11, q.y - 8);
  });
  if (h.expandedPoints.length === 4) {
    const handles = roiSideMidpoints();
    Object.entries(handles).forEach(([side, q]) => {
      const label = {top: 'T', right: 'R', bottom: 'B', left: 'L'}[side];
      ctx.save();
      ctx.fillStyle = '#5b9bd5';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.fillRect(q.x - 7, q.y - 7, 14, 14);
      ctx.strokeRect(q.x - 7, q.y - 7, 14, 14);
      ctx.fillStyle = '#ffffff';
      ctx.font = '700 11px Arial';
      ctx.fillText(label, q.x + 10, q.y + 4);
      ctx.restore();
    });
  }
}

function drawWarp() {
  const ctx = w.ctx, cw = w.canvas.width, ch = w.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#050607'; ctx.fillRect(0, 0, cw, ch);
  if (!h.warpImg) {
    ctx.fillStyle = '#7d8580'; ctx.font = '14px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Marca 4 puntos para previsualizar', cw / 2, ch / 2);
    ctx.textAlign = 'left';
    return;
  }
  const scale = Math.min(cw / h.warpImg.naturalWidth, ch / h.warpImg.naturalHeight);
  const dw = h.warpImg.naturalWidth * scale, dh = h.warpImg.naturalHeight * scale;
  ctx.drawImage(h.warpImg, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
}

function drawGrid(state, step) {
  const ctx = state.ctx, cw = state.canvas.width, ch = state.canvas.height;
  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  ctx.lineWidth = 1; ctx.font = '10px Consolas';
  for (let gx = Math.ceil(state.panX / step) * step; gx < state.panX + vw; gx += step) {
    const px = ((gx - state.panX) / vw) * cw;
    ctx.strokeStyle = 'rgba(255,255,255,.12)';
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, ch); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,.45)'; ctx.fillText(String(gx), px + 3, 13);
  }
  for (let gy = Math.ceil(state.panY / step) * step; gy < state.panY + vh; gy += step) {
    const py = ((gy - state.panY) / vh) * ch;
    ctx.strokeStyle = 'rgba(255,255,255,.12)';
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(cw, py); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,.45)'; ctx.fillText(String(gy), 3, py + 12);
  }
}

function updatePoints() {
  document.getElementById('points-badge').textContent = `${h.points.length} / 4 puntos`;
  document.getElementById('points-badge').className = 'pill' + (h.points.length === 4 ? ' ok' : '');
  document.getElementById('save-homography').disabled = h.points.length !== 4;
  h.expandPct = parseFloat(document.getElementById('h-expand').value) || 0;
  document.getElementById('h-expand-label').textContent = Math.round(h.expandPct) + '%';
  const baseText = h.points.length
    ? h.points.map((p, i) => `<span style="color:${COLORS[i]}">P${i+1}</span> (${Math.round(p.x)}, ${Math.round(p.y)})`).join(' &nbsp; ')
    : 'Sin puntos.';
  const roiText = h.expandedPoints.length === 4
    ? ` &nbsp; <span style="color:#5b9bd5">ROI ${h.roiManual ? 'lados' : 'auto'}</span> L:${Math.round(h.roiMargins.left)} R:${Math.round(h.roiMargins.right)} T:${Math.round(h.roiMargins.top)} B:${Math.round(h.roiMargins.bottom)} px`
    : '';
  document.getElementById('points-list').innerHTML = baseText + roiText;
  requestWarp();
}

async function requestWarp() {
  if (h.points.length !== 4) {
    h.warpImg = null;
    h.expandedPoints = [];
    document.getElementById('warp-size').textContent = 'sin homografia';
    drawWarp();
    return;
  }
  h.expandPct = parseFloat(document.getElementById('h-expand').value) || 0;
  const payload = {points: h.points, expand_pct: h.expandPct};
  if (h.roiManual) payload.roi_margins = h.roiMargins;
  const r = await fetch('/api/warp', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  const img = new Image();
  img.onload = () => {
    h.warpImg = img;
    h.expandedPoints = d.warp_points || [];
    h.roiMargins = d.roi_margins || h.roiMargins;
    h.baseMatrix = d.base_matrix || h.baseMatrix;
    h.baseSize = d.base_size || h.baseSize;
    h.roiManual = d.roi_mode === 'side_margins';
    const mode = h.roiManual ? 'lados manuales' : `expansion ${Math.round(h.expandPct)}%`;
    document.getElementById('warp-size').textContent = `${d.width} x ${d.height} | ${mode}`;
    drawAll();
  };
  img.src = 'data:image/jpeg;base64,' + d.image;
}

async function saveHomography() {
  if (h.points.length !== 4) return;
  h.expandPct = parseFloat(document.getElementById('h-expand').value) || 0;
  const payload = {points: h.points, expand_pct: h.expandPct};
  if (h.roiManual) payload.roi_margins = h.roiMargins;
  const r = await fetch('/api/save_homography', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  const mode = h.roiManual ? 'lados manuales' : `expansion ${Math.round(h.expandPct)}%`;
  clearWarpDependentViews();
  status(`Homografia guardada con ${mode}: ${d.path}${d.backup ? ' | backup: ' + d.backup : ''}`, 'ok');
}

async function loadSavedHomography() {
  const r = await fetch('/api/homography/current');
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  h.points = (d.selected_source_points || d.ordered_source_points || []).map(p => ({x: Number(p[0] ?? p.x), y: Number(p[1] ?? p.y)}));
  h.expandedPoints = (d.work_roi_points || d.ordered_source_points || []).map(p => ({x: Number(p[0] ?? p.x), y: Number(p[1] ?? p.y)}));
  h.roiMargins = d.roi_margins || {left: 0, right: 0, top: 0, bottom: 0};
  h.baseMatrix = d.base_homography_matrix || null;
  h.baseSize = Array.isArray(d.base_output_size) && d.base_output_size.length === 2
    ? {width: Number(d.base_output_size[0]), height: Number(d.base_output_size[1])}
    : null;
  h.roiManual = d.roi_mode === 'side_margins';
  h.expandPct = Number(d.expand_pct || 0);
  document.getElementById('h-expand').value = h.expandPct;
  document.getElementById('h-expand-label').textContent = Math.round(h.expandPct) + '%';
  updatePoints();
  drawAll();
  status(`Homografia cargada: ${h.points.length} puntos | ${h.roiManual ? 'lados manuales' : 'expansion ' + Math.round(h.expandPct) + '%'}`, 'ok');
}

function undoPoint() { h.points.pop(); updatePoints(); drawAll(); }
function resetPoints() {
  h.points = []; h.expandedPoints = []; h.roiManual = false;
  h.roiMargins = {left: 0, right: 0, top: 0, bottom: 0};
  h.baseMatrix = null; h.baseSize = null; h.warpImg = null;
  updatePoints(); drawAll();
}
function resetWorkRoi() {
  h.roiManual = false;
  h.expandedPoints = [];
  h.roiMargins = {left: 0, right: 0, top: 0, bottom: 0};
  if (h.points.length === 4) requestWarp();
  drawAll();
}

async function loadAnnotateFrame(frameIdx, options = {}) {
  if (!options.quiet) status('Cargando frame rectificado...', '');
  const r = await fetch('/api/annotate/frame', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_idx: frameIdx})
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  const img = new Image();
  return await new Promise((resolve, reject) => {
    img.onload = async () => {
      a.img = img; a.imgW = d.width; a.imgH = d.height;
      a.frameIdx = d.frame_idx; a.timeSec = d.time_sec; a.source = d.source || null;
      a.boxes = (d.boxes || []).map(boxToCorners); a.modelBoxes = []; a.boxRules = null; a.pieces = [];
      a.cornerA = null; a.preview = null; a.sobel = null;
      if (options.resetView !== false) resetView(a);
      document.getElementById('a-second').value = d.time_sec.toFixed(2);
      document.getElementById('a-timeline').value = d.frame_idx;
      document.getElementById('a-timeline-label').textContent = formatClock(d.time_sec);
      document.getElementById('a-zoom').value = a.zoom;
      document.getElementById('a-zoom-label').textContent = a.zoom.toFixed(1) + 'x';
      document.getElementById('info-frame').textContent = d.frame_idx;
      document.getElementById('info-time').textContent = d.time_sec.toFixed(3) + 's';
      document.getElementById('info-size').textContent = `${d.width}x${d.height}`;
      document.getElementById('info-video').textContent = sourceName(d.source);
      document.getElementById('info-source-frame').textContent = d.source?.source_frame_idx ?? '-';
      updateBoxes();
      updateModelBoxes();
      updateSobelInfo();
      fitAll(); drawAll();
      updateHistoryUI();
      if (options.autoAnalyze) {
        await predictYoloBoxes({quiet: true, autoSobel: true});
      } else if (!options.quiet) {
        status(d.is_saved ? `Frame ${d.frame_idx} cargado desde historial` : `Frame ${d.frame_idx} listo para anotar`, 'ok');
      }
      resolve(d);
    };
    img.onerror = () => {
      const err = new Error('No se pudo cargar la imagen del frame');
      status(err.message, 'err');
      reject(err);
    };
    img.src = 'data:image/jpeg;base64,' + d.image;
  });
}

function loadAnnotateSecond() {
  const second = parseFloat(document.getElementById('a-second').value) || 0;
  loadAnnotateFrame(Math.round(second * meta.fps));
}
function stepAnnotate(delta) { loadAnnotateFrame(a.frameIdx + delta); }
function stepAnnotateSeconds(seconds) {
  loadAnnotateFrame(a.frameIdx + Math.round(seconds * meta.fps), {resetView: false});
}

function formatClock(totalSeconds) {
  const safeSeconds = Math.max(0, Number(totalSeconds) || 0);
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const seconds = Math.floor(safeSeconds % 60);
  return [hours, minutes, seconds].map(value => String(value).padStart(2, '0')).join(':');
}

function formatFeetInches(totalInches) {
  const value = Number(totalInches);
  if (!Number.isFinite(value)) return '-';
  const sign = value < 0 ? '-' : '';
  const totalSixteenths = Math.round(Math.abs(value) * 16);
  const feet = Math.floor(totalSixteenths / 192);
  const remainingSixteenths = totalSixteenths - feet * 192;
  const wholeInches = Math.floor(remainingSixteenths / 16);
  let numerator = remainingSixteenths % 16;
  let denominator = 16;
  while (numerator > 0 && numerator % 2 === 0 && denominator % 2 === 0) {
    numerator /= 2;
    denominator /= 2;
  }
  const fraction = numerator ? ` ${numerator}/${denominator}` : '';
  return `${sign}${feet}' ${wholeInches}${fraction}"`;
}

function previewAnnotateTimeline(frameValue) {
  const frameIdx = Math.max(0, Math.min(meta.totalFrames - 1, Number(frameValue) || 0));
  const second = frameIdx / meta.fps;
  document.getElementById('a-second').value = second.toFixed(2);
  document.getElementById('a-timeline-label').textContent = formatClock(second);
}

function loadAnnotateTimeline(frameValue) {
  clearTimeout(a.timelineTimer);
  a.timelineTimer = null;
  return loadAnnotateFrame(Number(frameValue) || 0, {resetView: false});
}

function scrollAnnotateTimeline(event) {
  event.preventDefault();
  const timeline = event.currentTarget;
  const direction = (event.deltaY || event.deltaX) > 0 ? 1 : -1;
  const jumpSeconds = event.shiftKey ? 5 : 30;
  const nextFrame = Math.max(
    0,
    Math.min(meta.totalFrames - 1, Number(timeline.value) + direction * Math.round(meta.fps * jumpSeconds))
  );
  timeline.value = nextFrame;
  previewAnnotateTimeline(nextFrame);
  clearTimeout(a.timelineTimer);
  a.timelineTimer = setTimeout(() => loadAnnotateTimeline(nextFrame), 180);
}

function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

async function loadMeasureFrame(frameIdx, options = {}) {
  if (!options.quiet) status('Cargando frame para mediciones...', '');
  const r = await fetch('/api/measure/frame', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_idx: frameIdx})
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return null; }
  const img = new Image();
  return await new Promise((resolve, reject) => {
    img.onload = () => {
      m.img = img; m.imgW = d.width; m.imgH = d.height;
      m.frameIdx = d.frame_idx; m.timeSec = d.time_sec; m.source = d.source || null;
      applyMeasureCalibration(d.calibration || {});
      if (options.resetView !== false) resetView(m);
      document.getElementById('m-second').value = d.time_sec.toFixed(2);
      document.getElementById('m-zoom').value = m.zoom;
      document.getElementById('m-zoom-label').textContent = m.zoom.toFixed(1) + 'x';
      updateMeasureInfo();
      fitAll(); drawAll();
      if (!options.quiet) status(`Frame ${d.frame_idx} listo para mediciones`, 'ok');
      resolve(d);
    };
    img.onerror = () => {
      const err = new Error('No se pudo cargar el frame de mediciones');
      status(err.message, 'err');
      reject(err);
    };
    img.src = 'data:image/jpeg;base64,' + d.image;
  });
}

function loadMeasureSecond() {
  const second = parseFloat(document.getElementById('m-second').value) || 0;
  return loadMeasureFrame(Math.round(second * meta.fps));
}

function reloadMeasureRoi() {
  const frameIdx = m.img ? m.frameIdx : Math.round((parseFloat(document.getElementById('m-second').value) || 0) * meta.fps);
  m.img = null; m.imgW = 0; m.imgH = 0; m.pending = null; m.preview = null;
  updateMeasureInfo(); drawAll();
  return loadMeasureFrame(frameIdx, {resetView: true});
}

function stepMeasure(delta) {
  return loadMeasureFrame(m.frameIdx + delta, {resetView: false});
}

function applyMeasureCalibration(calibration) {
  m.segments = (calibration.segments || []).map(s => ({...s}));
  m.referenceY = calibration.reference_y ?? null;
  m.exclusionZones = (calibration.exclusion_zones || []).map(zone => ({...zone}));
  m.inchPerPx = calibration.inch_per_px ?? computeInchPerPx();
  m.selectedSegment = null;
  m.draggingSegment = null;
}

function computeInchPerPx() {
  const valid = m.segments.filter(s => Number(s.px) > 0 && Number(s.inches) > 0);
  if (!valid.length) return null;
  const totalIn = valid.reduce((sum, s) => sum + Number(s.inches), 0);
  const totalPx = valid.reduce((sum, s) => sum + Number(s.px), 0);
  return totalPx > 0 ? totalIn / totalPx : null;
}

function updateMeasureInfo() {
  m.inchPerPx = computeInchPerPx();
  document.getElementById('m-info-video').textContent = sourceName(m.source);
  document.getElementById('m-info-source-frame').textContent = m.source?.source_frame_idx ?? '-';
  document.getElementById('m-info-frame').textContent = m.img ? m.frameIdx : '-';
  document.getElementById('m-info-time').textContent = m.img ? m.timeSec.toFixed(3) + 's' : '-';
  document.getElementById('m-info-size').textContent = m.img ? `${m.imgW}x${m.imgH}` : '-';
  document.getElementById('m-info-inch-px').textContent = m.inchPerPx ? m.inchPerPx.toFixed(5) : '-';
  document.getElementById('m-info-px-inch').textContent = m.inchPerPx ? (1 / m.inchPerPx).toFixed(2) : '-';
  document.getElementById('m-info-segments').textContent = m.segments.length;
  document.getElementById('m-info-ref').textContent = m.referenceY === null ? '-' : Math.round(m.referenceY) + ' px';
  document.getElementById('m-info-zones').textContent = m.exclusionZones.length;
  const zoneList = document.getElementById('m-zone-list');
  zoneList.innerHTML = m.exclusionZones.length
    ? m.exclusionZones.map((zone, index) => `<div class="box-item">
        <span class="swatch" style="background:#ef6666"></span>
        <span>#${index + 1} ${Math.round(zone.w)}x${Math.round(zone.h)} px</span>
        <button class="danger" onclick="deleteMeasureExclusionZone(${index})">Borrar</button>
      </div>`).join('')
    : '<div class="kv"><span>Sin zonas rojas.</span></div>';
  const list = document.getElementById('m-segment-list');
  if (!m.segments.length) {
    list.innerHTML = '<div class="kv"><span>Marca un segmento y escribe sus pulgadas.</span></div>';
    return;
  }
  list.innerHTML = m.segments.map((s, i) => {
    const inches = Number(s.inches);
    const px = Number(s.px);
    const inchPerPx = px > 0 ? inches / px : 0;
    const pxPerIn = inchPerPx > 0 ? 1 / inchPerPx : 0;
    return `<div class="box-item"><span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span><span>#${i+1} ${inches.toFixed(3)} in | ${px.toFixed(1)} px | ${inchPerPx.toFixed(5)} in/px | ${pxPerIn.toFixed(2)} px/in</span><button onclick="deleteMeasureSegment(${i})">Borrar</button></div>`;
  }).join('');
}

function setMeasureMode(mode) {
  m.mode = mode;
  m.pending = null; m.preview = null;
  m.draggingSegment = null;
  if (mode !== 'segment') m.selectedSegment = null;
  document.getElementById('m-mode-segment').classList.toggle('active', mode === 'segment');
  document.getElementById('m-mode-ref').classList.toggle('active', mode === 'reference');
  document.getElementById('m-mode-exclusion').classList.toggle('active', mode === 'exclusion');
  const message = mode === 'segment'
    ? 'Modo segmento: marca 2 puntos.'
    : (mode === 'reference'
      ? 'Modo Linea Y: click o arrastra la referencia horizontal.'
      : 'Modo Zona roja: marca dos esquinas del area sin boxes.');
  status(message, 'ok');
  drawAll();
}

function measureClick(point) {
  if (!m.img) return;
  if (m.mode === 'reference') {
    m.referenceY = point.y;
    updateMeasureInfo(); drawAll();
    status(`Linea Y en ${Math.round(point.y)} px`, 'ok');
    return;
  }
  if (m.mode === 'exclusion') {
    if (!m.pending) {
      m.pending = point;
      status('Primera esquina de la zona roja marcada. Marca la esquina opuesta.', 'ok');
      drawAll();
      return;
    }
    const x = Math.min(m.pending.x, point.x);
    const y = Math.min(m.pending.y, point.y);
    const w = Math.abs(point.x - m.pending.x);
    const h = Math.abs(point.y - m.pending.y);
    if (w >= 2 && h >= 2) {
      m.exclusionZones.push({x, y, w, h});
      status(`Zona roja guardada: ${Math.round(w)}x${Math.round(h)} px`, 'ok');
    } else {
      status('Zona cancelada: el area es demasiado pequena.', 'err');
    }
    m.pending = null; m.preview = null;
    updateMeasureInfo(); drawAll();
    return;
  }
  if (!m.pending) {
    m.pending = point;
    status('Primer punto de medicion marcado. Marca el segundo.', 'ok');
    drawAll();
    return;
  }
  const px = Math.hypot(point.x - m.pending.x, point.y - m.pending.y);
  const value = window.prompt('Longitud real del segmento en pulgadas:', '');
  const inches = Number(value);
  if (Number.isFinite(inches) && inches > 0 && px > 0) {
    m.segments.push({
      x1: m.pending.x, y1: m.pending.y, x2: point.x, y2: point.y,
      px, inches, inch_per_px: inches / px,
    });
    status(`Segmento guardado: ${inches.toFixed(3)} in | ${px.toFixed(1)} px`, 'ok');
  } else {
    status('Segmento cancelado: pulgadas invalidas.', 'err');
  }
  m.pending = null; m.preview = null;
  updateMeasureInfo(); drawAll();
}

function deleteMeasureSegment(i) {
  m.segments.splice(i, 1);
  if (m.selectedSegment === i) m.selectedSegment = null;
  else if (m.selectedSegment > i) m.selectedSegment -= 1;
  updateMeasureInfo(); drawAll();
}

function deleteMeasureExclusionZone(i) {
  m.exclusionZones.splice(i, 1);
  updateMeasureInfo(); drawAll();
}

function undoMeasureSegment() {
  if (m.pending) { m.pending = null; m.preview = null; }
  else if (m.mode === 'exclusion') m.exclusionZones.pop();
  else m.segments.pop();
  updateMeasureInfo(); drawAll();
}

function clearMeasureCalibration() {
  m.segments = []; m.pending = null; m.preview = null; m.referenceY = null; m.inchPerPx = null;
  m.exclusionZones = [];
  m.selectedSegment = null; m.draggingSegment = null;
  updateMeasureInfo(); drawAll();
}

async function saveMeasureCalibration() {
  updateMeasureInfo();
  const payload = {
    frame_idx: m.frameIdx,
    time_sec: m.timeSec,
    img_w: m.imgW,
    img_h: m.imgH,
    segments: m.segments,
    reference_y: m.referenceY,
    exclusion_zones: m.exclusionZones,
    inch_per_px: m.inchPerPx,
  };
  const r = await fetch('/api/measure/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  p.calibration = d;
  p.calibrationStale = true;
  updatePlayerRulerInfo();
  drawAll();
  status(`Mediciones guardadas. El reproductor recalculara el frame con la nueva escala.`, 'ok');
}

async function loadPlayerFrame(frameIdx, options = {}) {
  if (!options.quiet) status('Corriendo YOLO + Sobel...', '');
  const r = await fetch('/api/player/frame', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_idx: frameIdx, conf: 0.10})
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return null; }
  const img = new Image();
  return await new Promise((resolve, reject) => {
    img.onload = () => {
      p.img = img; p.imgW = d.width; p.imgH = d.height;
      p.frameIdx = d.frame_idx; p.timeSec = d.time_sec; p.source = d.source || null;
      p.boxes = d.boxes || [];
      p.pieces = d.pieces || [];
      p.measurementSummary = d.measurement_summary || null;
      p.sobel = d.sobel || null;
      p.calibration = d.calibration || null;
      p.measurement = d.measurement || null;
      p.boxRules = d.box_rules || null;
      p.calibrationStale = false;
      if (options.resetView !== false) resetView(p);
      document.getElementById('p-second').value = d.time_sec.toFixed(2);
      document.getElementById('p-timeline').value = d.frame_idx;
      document.getElementById('p-timeline-label').textContent = formatClock(d.time_sec);
      document.getElementById('p-zoom').value = p.zoom;
      document.getElementById('p-zoom-label').textContent = p.zoom.toFixed(1) + 'x';
      updatePlayerInfo();
      updatePlayerCapturesUI();
      fitAll(); drawAll();
      if (!options.quiet) {
        const cls = p.boxes.length ? 'ok' : 'err';
        const validCount = Number(p.measurementSummary?.valid_count || 0);
        status(`Frame ${d.frame_idx}: ${p.boxes.length} piezas, ${validCount} mediciones validas`, cls);
      }
      resolve(d);
    };
    img.onerror = () => {
      const err = new Error('No se pudo cargar la imagen del reproductor');
      status(err.message, 'err');
      reject(err);
    };
    img.src = 'data:image/jpeg;base64,' + d.image;
  });
}

function playerMeasureValue(measurement) {
  if (!measurement) return null;
  const value = Number(measurement.measurement_in ?? measurement.delta_in);
  return Number.isFinite(value) ? value : null;
}

function rulerModeLabel(mode) {
  if (mode === 'x') return 'X';
  if (mode === 'y') return 'Y';
  return 'Libre';
}

function constrainPlayerRulerPoint(start, point) {
  if (p.rulerMode === 'x') return {x: point.x, y: start.y};
  if (p.rulerMode === 'y') return {x: start.x, y: point.y};
  return {x: point.x, y: point.y};
}

function currentPlayerRulerEnd() {
  if (!p.rulerStart) return null;
  return p.rulerEnd || p.rulerPreview;
}

function playerRulerMeasurement() {
  const end = currentPlayerRulerEnd();
  if (!p.rulerStart || !end) return null;
  const dx = end.x - p.rulerStart.x;
  const dy = end.y - p.rulerStart.y;
  const px = Math.hypot(dx, dy);
  const inchPerPx = p.calibration && Number(p.calibration.inch_per_px);
  const inches = Number.isFinite(inchPerPx) && inchPerPx > 0 ? px * inchPerPx : null;
  return {dx, dy, px, inches};
}

function updatePlayerRulerInfo() {
  const tool = document.getElementById('p-ruler-tool');
  const toggle = document.getElementById('p-ruler-toggle');
  if (tool) tool.classList.toggle('active', p.rulerActive);
  if (toggle) toggle.classList.toggle('active', p.rulerActive);
  document.getElementById('p-ruler-x').classList.toggle('active', p.rulerActive && p.rulerMode === 'x');
  document.getElementById('p-ruler-y').classList.toggle('active', p.rulerActive && p.rulerMode === 'y');
  document.getElementById('p-ruler-free').classList.toggle('active', p.rulerActive && p.rulerMode === 'free');
  document.getElementById('p-ruler-info-mode').textContent = p.rulerActive ? rulerModeLabel(p.rulerMode) : 'off';
  const inchPerPx = p.calibration && Number(p.calibration.inch_per_px);
  document.getElementById('p-ruler-info-scale').textContent =
    Number.isFinite(inchPerPx) && inchPerPx > 0 ? `${inchPerPx.toFixed(6)} in/px` : '-';
  const measure = playerRulerMeasurement();
  if (!measure) {
    document.getElementById('p-ruler-info-distance').textContent = '-';
    document.getElementById('p-ruler-info-delta').textContent = '-';
    return;
  }
  const inches = measure.inches === null ? '' : ` | ${measure.inches.toFixed(3)} in`;
  document.getElementById('p-ruler-info-distance').textContent = `${measure.px.toFixed(1)} px${inches}`;
  document.getElementById('p-ruler-info-delta').textContent = `dx ${measure.dx.toFixed(1)} | dy ${measure.dy.toFixed(1)}`;
}

function togglePlayerRulerTool() {
  p.rulerActive = !p.rulerActive;
  if (p.rulerActive) {
    stopPlayerPlayback();
    if (!p.rulerMode) p.rulerMode = 'free';
    p.wrap.style.cursor = 'crosshair';
  } else {
    p.rulerStart = null; p.rulerEnd = null; p.rulerPreview = null;
    p.wrap.style.cursor = 'default';
  }
  updatePlayerRulerInfo();
  drawAll();
  status(p.rulerActive ? `Regla activa: ${rulerModeLabel(p.rulerMode)}` : 'Regla apagada.', 'ok');
}

function setPlayerRulerMode(mode) {
  p.rulerActive = true;
  p.rulerMode = mode;
  p.rulerStart = null; p.rulerEnd = null; p.rulerPreview = null;
  stopPlayerPlayback();
  p.wrap.style.cursor = 'crosshair';
  updatePlayerRulerInfo();
  drawAll();
  status(`Regla ${rulerModeLabel(mode)} activa: marca 2 puntos.`, 'ok');
}

function clearPlayerRuler() {
  p.rulerStart = null; p.rulerEnd = null; p.rulerPreview = null;
  updatePlayerRulerInfo();
  drawAll();
  status('Regla limpia.', 'ok');
}

function playerRulerClick(point) {
  if (!p.rulerActive || !p.img) return;
  stopPlayerPlayback();
  if (!p.rulerStart || p.rulerEnd) {
    p.rulerStart = {x: point.x, y: point.y};
    p.rulerEnd = null;
    p.rulerPreview = {x: point.x, y: point.y};
    status(`Regla ${rulerModeLabel(p.rulerMode)}: marca el segundo punto.`, 'ok');
  } else {
    p.rulerEnd = constrainPlayerRulerPoint(p.rulerStart, point);
    p.rulerPreview = null;
    const measure = playerRulerMeasurement();
    status(measure && measure.inches !== null
      ? `Regla: ${measure.px.toFixed(1)} px | ${measure.inches.toFixed(3)} in`
      : `Regla: ${measure ? measure.px.toFixed(1) : 0} px`, 'ok');
  }
  updatePlayerRulerInfo();
  drawAll();
}

function updatePlayerInfo() {
  const bestConf = p.boxes.reduce((m, b) => Math.max(m, Number(b.conf || 0)), 0);
  const validPieces = p.pieces.filter(piece => piece.valid && piece.measurement);
  const analyzedPieces = p.pieces.filter(piece => piece.sobel && piece.sobel.has_roi);
  const averageEdgeConf = analyzedPieces.length
    ? analyzedPieces.reduce((sum, piece) => sum + Number(piece.sobel.edge_confidence || 0), 0) / analyzedPieces.length
    : null;
  const averageCrm = analyzedPieces.length
    ? analyzedPieces.reduce((sum, piece) => sum + Number(piece.sobel.crm_px || 0), 0) / analyzedPieces.length
    : null;
  document.getElementById('p-info-video').textContent = sourceName(p.source);
  document.getElementById('p-info-source-frame').textContent = p.source?.source_frame_idx ?? '-';
  document.getElementById('p-info-frame').textContent = p.img ? p.frameIdx : '-';
  document.getElementById('p-info-time').textContent = p.img ? p.timeSec.toFixed(3) + 's' : '-';
  document.getElementById('p-info-size').textContent = p.img ? `${p.imgW}x${p.imgH}` : '-';
  document.getElementById('p-info-speed').textContent = `x${p.speed}`;
  document.getElementById('p-speed').textContent = `x${p.speed}`;
  document.getElementById('p-info-boxes').textContent = p.boxes.length;
  document.getElementById('p-info-yolo-conf').textContent = bestConf ? bestConf.toFixed(2) : '-';
  document.getElementById('p-info-excluded').textContent = Number(p.boxRules?.removed_exclusion_count || 0);
  const yoloBadge = document.getElementById('p-yolo-badge');
  yoloBadge.textContent = p.boxes.length ? `YOLO ${p.boxes.length}` : 'YOLO 0';
  yoloBadge.className = 'pill' + (p.boxes.length ? ' ok' : '');

  const state = document.getElementById('p-info-sobel-state');
  const conf = document.getElementById('p-info-sobel-conf');
  const crm = document.getElementById('p-info-sobel-crm');
  const measure = document.getElementById('p-info-measure');
  const sobelBadge = document.getElementById('p-sobel-badge');
  measure.textContent = `${validPieces.length} / ${p.pieces.length}`;
  updatePlayerRulerInfo();
  if (!p.pieces.length) {
    state.textContent = p.boxes.length ? 'sin ROI' : 'sin box YOLO';
    conf.textContent = '-';
    crm.textContent = '-';
    sobelBadge.textContent = 'Sobel -';
    sobelBadge.className = 'pill';
    updatePlayerPieceList();
    return;
  }
  state.textContent = validPieces.length === p.pieces.length ? 'todas validas' : 'revisar piezas';
  conf.textContent = averageEdgeConf === null ? '-' : averageEdgeConf.toFixed(2);
  crm.textContent = averageCrm === null || !Number.isFinite(averageCrm) ? '-' : averageCrm.toFixed(2) + ' px';
  sobelBadge.textContent = `Sobel ${validPieces.length}/${p.pieces.length}`;
  sobelBadge.className = 'pill' + (validPieces.length ? ' ok' : '');
  updatePlayerPieceList();
}

function updatePlayerPieceList() {
  const list = document.getElementById('p-piece-list');
  if (!list) return;
  if (!p.pieces.length) {
    list.innerHTML = '<div class="kv"><span>Sin piezas detectadas.</span></div>';
    return;
  }
  list.innerHTML = p.pieces.map((piece, index) => {
    const measurement = piece.measurement;
    const total = measurement ? formatFeetInches(measurement.measurement_in) : '-';
    const distance = measurement ? `${Number(measurement.delta_in).toFixed(3)} in ref` : 'sin borde';
    return `<div class="box-item">
      <span class="swatch" style="background:${COLORS[index % COLORS.length]}"></span>
      <span>P${Number(piece.piece_id)} | ${total}<br><small>${distance}</small></span>
    </div>`;
  }).join('');
}

async function refreshPlayerCaptures() {
  const r = await fetch('/api/player/captures');
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  p.captures = d.captures || [];
  updatePlayerCapturesUI();
}

function updatePlayerCapturesUI() {
  const list = document.getElementById('p-capture-list');
  if (!list) return;
  if (!p.captures.length) {
    list.innerHTML = '<div class="kv"><span>Sin capturas guardadas.</span></div>';
    return;
  }
  list.innerHTML = p.captures.map(item => {
    const cls = item.frame_idx === p.frameIdx ? 'history-item current' : 'history-item';
    const time = Number(item.time_sec || 0).toFixed(2);
    const measurementCount = Number(item.measurement_count || 0);
    const value = measurementCount
      ? `${measurementCount} mediciones`
      : (item.measurement_in === null || item.measurement_in === undefined
        ? 'sin medida'
        : formatFeetInches(item.measurement_in));
    return `<div class="${cls}" onclick="goToPlayerCapture(${Number(item.frame_idx)})" title="Ir a ${time}s">
      <div><strong>Frame ${Number(item.frame_idx)}</strong><span>${time}s | ${value}</span></div>
      <div class="history-actions">
        <button onclick="goToPlayerCapture(${Number(item.frame_idx)}); event.stopPropagation();">Ir</button>
        <button class="danger" onclick="deletePlayerCapture('${String(item.id || '').replace(/'/g, "\\'")}'); event.stopPropagation();">Borrar</button>
      </div>
    </div>`;
  }).join('');
}

function goToPlayerCapture(frameIdx) {
  stopPlayerPlayback();
  return loadPlayerFrame(Number(frameIdx), {resetView: false});
}

async function savePlayerCapture() {
  if (!p.img) {
    status('Carga un frame en el reproductor antes de guardar captura.', 'err');
    return;
  }
  const payload = {
    frame_idx: p.frameIdx,
    time_sec: p.timeSec,
    img_w: p.imgW,
    img_h: p.imgH,
    boxes: p.boxes,
    pieces: p.pieces,
    measurement_summary: p.measurementSummary,
    sobel: p.sobel,
    measurement: p.measurement,
  };
  const r = await fetch('/api/player/captures', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  p.captures = d.captures || [];
  updatePlayerCapturesUI();
  const value = playerMeasureValue(d.capture && d.capture.measurement);
  status(`Captura guardada: frame ${d.capture.frame_idx}${value === null ? '' : ' | ' + formatFeetInches(value)}`, 'ok');
}

async function deletePlayerCapture(captureId) {
  if (!captureId) return;
  const r = await fetch(`/api/player/captures/${encodeURIComponent(captureId)}`, {method: 'DELETE'});
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  p.captures = d.captures || [];
  updatePlayerCapturesUI();
  status(`Captura borrada: ${captureId}`, 'ok');
}

function loadPlayerSecond() {
  stopPlayerPlayback();
  const second = parseFloat(document.getElementById('p-second').value) || 0;
  return loadPlayerFrame(Math.round(second * meta.fps));
}

function previewPlayerTimeline(frameValue) {
  const frameIdx = Math.max(0, Math.min(meta.totalFrames - 1, Number(frameValue) || 0));
  const second = frameIdx / meta.fps;
  document.getElementById('p-second').value = second.toFixed(2);
  document.getElementById('p-timeline-label').textContent = formatClock(second);
}

function loadPlayerTimeline(frameValue) {
  stopPlayerPlayback();
  clearTimeout(p.timelineTimer);
  p.timelineTimer = null;
  return loadPlayerFrame(Number(frameValue) || 0, {resetView: false});
}

function scrollPlayerTimeline(event) {
  event.preventDefault();
  stopPlayerPlayback();
  const timeline = event.currentTarget;
  const direction = (event.deltaY || event.deltaX) > 0 ? 1 : -1;
  const jumpSeconds = event.shiftKey ? 5 : 30;
  const nextFrame = Math.max(
    0,
    Math.min(meta.totalFrames - 1, Number(timeline.value) + direction * Math.round(meta.fps * jumpSeconds))
  );
  timeline.value = nextFrame;
  previewPlayerTimeline(nextFrame);
  clearTimeout(p.timelineTimer);
  p.timelineTimer = setTimeout(() => loadPlayerTimeline(nextFrame), 180);
}

function stepPlayer(delta) {
  stopPlayerPlayback();
  return loadPlayerFrame(p.frameIdx + delta, {resetView: false});
}

function togglePlayerSpeed() {
  p.speed = p.speed === 1 ? 2 : 1;
  updatePlayerInfo();
  status(`Velocidad del reproductor: x${p.speed}`, 'ok');
}

function stopPlayerPlayback() {
  p.playing = false;
  const button = document.getElementById('p-play');
  if (button) button.textContent = 'Play';
}

async function togglePlayerPlay() {
  if (p.playing) {
    stopPlayerPlayback();
    return;
  }
  if (!p.img) {
    const loaded = await loadPlayerSecond();
    if (!loaded) return;
  }
  p.playing = true;
  document.getElementById('p-play').textContent = 'Pausa';
  if (!p.playTask) {
    p.playTask = playerLoop().finally(() => { p.playTask = null; });
  }
}

async function playerLoop() {
  while (p.playing) {
    const maxFrame = Math.max(0, meta.totalFrames - 1);
    const step = p.speed === 2 ? 2 : 1;
    const nextFrame = Math.min(p.frameIdx + step, maxFrame);
    if (nextFrame === p.frameIdx) {
      stopPlayerPlayback();
      break;
    }
    try {
      await loadPlayerFrame(nextFrame, {quiet: true, resetView: false});
    } catch (e) {
      stopPlayerPlayback();
      break;
    }
    await sleep(15);
  }
}

function drawAnnotate() {
  const ctx = a.ctx, cw = a.canvas.width, ch = a.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!a.img) return;
  const vw = a.imgW / a.zoom, vh = a.imgH / a.zoom;
  ctx.drawImage(a.img, a.panX, a.panY, vw, vh, 0, 0, cw, ch);
  drawGrid(a, 100);
  a.modelBoxes.forEach((b, i) => {
    const color = b.inferred ? '#d6a34b' : '#70c7c2';
    drawBox(b, color, true, `M${i + 1}`);
  });
  a.boxes.forEach((b, i) => drawBox(b, COLORS[i % COLORS.length], false, i + 1));
  if (a.preview) drawBox(a.preview, '#ffffff', true, '?');
  drawSobelProjection();
}

function drawExclusionZones(state, zones, showLabels = false) {
  const ctx = state.ctx;
  (zones || []).forEach((zone, index) => {
    const topLeft = imageToDisplay(state, zone.x, zone.y);
    const bottomRight = imageToDisplay(state, zone.x + zone.w, zone.y + zone.h);
    const width = bottomRight.x - topLeft.x;
    const height = bottomRight.y - topLeft.y;
    ctx.save();
    ctx.fillStyle = 'rgba(210, 48, 48, 0.18)';
    ctx.strokeStyle = '#ef6666';
    ctx.lineWidth = 2;
    ctx.fillRect(topLeft.x, topLeft.y, width, height);
    ctx.strokeRect(topLeft.x, topLeft.y, width, height);
    if (showLabels) {
      ctx.fillStyle = '#ffffff';
      ctx.font = '800 12px Arial';
      ctx.fillText(`NO BOX ${index + 1}`, topLeft.x + 7, topLeft.y + 17);
    }
    ctx.restore();
  });
}

function drawMeasure() {
  const ctx = m.ctx, cw = m.canvas.width, ch = m.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!m.img) return;
  const vw = m.imgW / m.zoom, vh = m.imgH / m.zoom;
  ctx.drawImage(m.img, m.panX, m.panY, vw, vh, 0, 0, cw, ch);
  drawGrid(m, 100);
  drawExclusionZones(m, m.exclusionZones, true);

  if (m.referenceY !== null) {
    const left = imageToDisplay(m, 0, m.referenceY);
    const right = imageToDisplay(m, m.imgW - 1, m.referenceY);
    ctx.save();
    ctx.strokeStyle = '#5b9bd5';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(left.x, left.y); ctx.lineTo(right.x, right.y); ctx.stroke();
    ctx.fillStyle = '#5b9bd5';
    ctx.font = '700 12px Arial';
    ctx.fillText(`Y ref ${Math.round(m.referenceY)} px`, 10, Math.max(16, left.y - 8));
    ctx.restore();
  }

  m.segments.forEach((s, i) => {
    const a1 = imageToDisplay(m, s.x1, s.y1);
    const a2 = imageToDisplay(m, s.x2, s.y2);
    const color = COLORS[i % COLORS.length];
    ctx.save();
    if (m.selectedSegment === i) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 7;
      ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y); ctx.stroke();
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y); ctx.stroke();
    ctx.fillStyle = color;
    const handleRadius = m.selectedSegment === i ? 7 : 5;
    ctx.beginPath(); ctx.arc(a1.x, a1.y, handleRadius, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(a2.x, a2.y, handleRadius, 0, Math.PI * 2); ctx.fill();
    ctx.font = '700 12px Arial';
    ctx.fillText(`${Number(s.inches).toFixed(2)} in`, (a1.x + a2.x) / 2 + 8, (a1.y + a2.y) / 2 - 8);
    ctx.restore();
  });

  if (m.pending && m.preview) {
    const p1 = imageToDisplay(m, m.pending.x, m.pending.y);
    const p2 = imageToDisplay(m, m.preview.x, m.preview.y);
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    if (m.mode === 'exclusion') {
      ctx.strokeStyle = '#ff7a7a';
      ctx.fillStyle = 'rgba(210, 48, 48, 0.22)';
      ctx.fillRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
    } else {
      ctx.strokeStyle = '#ffffff';
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    }
    ctx.restore();
  }
}

function drawPlayer() {
  const ctx = p.ctx, cw = p.canvas.width, ch = p.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!p.img) return;
  const vw = p.imgW / p.zoom, vh = p.imgH / p.zoom;
  ctx.drawImage(p.img, p.panX, p.panY, vw, vh, 0, 0, cw, ch);
  drawGrid(p, 100);
  drawExclusionZones(p, p.calibration?.exclusion_zones || [], false);
  drawPlayerMeasurement();
  p.boxes.forEach((b, i) => {
    const color = b.inferred ? '#d6a34b' : COLORS[i % COLORS.length];
    drawBoxOnState(p, b, color, Boolean(b.inferred), i + 1);
  });
  if (p.pieces.length) {
    p.pieces.forEach(piece => drawSobelOnState(p, piece.sobel));
  } else {
    drawSobelOnState(p, p.sobel);
  }
  drawPlayerRuler();
}

function normBox(b) {
  if ('x' in b && 'y' in b && 'w' in b && 'h' in b) {
    return {...b, x: b.x, y: b.y, w: b.w, h: b.h};
  }
  return {
    ...b,
    x: Math.min(b.x1, b.x2),
    y: Math.min(b.y1, b.y2),
    w: Math.abs(b.x2 - b.x1),
    h: Math.abs(b.y2 - b.y1)
  };
}

function boxToCorners(b) {
  if ('x1' in b && 'y1' in b && 'x2' in b && 'y2' in b) return b;
  return {...b, x1: b.x, y1: b.y, x2: b.x + b.w, y2: b.y + b.h};
}

function drawBox(b, color, dashed, label) {
  drawBoxOnState(a, b, color, dashed, label);
}

function drawBoxOnState(state, b, color, dashed, label) {
  const nb = normBox(b);
  const p1 = imageToDisplay(state, nb.x, nb.y);
  const p2 = imageToDisplay(state, nb.x + nb.w, nb.y + nb.h);
  const ctx = state.ctx;
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = dashed ? 1.5 : 2;
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
  ctx.globalAlpha = dashed ? .06 : .12;
  ctx.fillStyle = color; ctx.fillRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
  ctx.restore();
  if (!dashed || String(label).startsWith('M') || b.inferred) {
    ctx.fillStyle = color; ctx.font = '700 12px Arial';
    const prefix = String(label).startsWith('M') ? '' : '#';
    ctx.fillText(`${prefix}${label} ${Math.round(nb.w)}x${Math.round(nb.h)}`, p1.x + 4, p1.y + 14);
  }
}

function drawPlayerMeasurement() {
  if (!p.calibration || p.calibration.reference_y === null || p.calibration.reference_y === undefined) return;
  const ctx = p.ctx;
  const y = Number(p.calibration.reference_y);
  const left = imageToDisplay(p, 0, y);
  const right = imageToDisplay(p, p.imgW - 1, y);
  ctx.save();
  ctx.strokeStyle = '#5b9bd5';
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(left.x, left.y); ctx.lineTo(right.x, right.y); ctx.stroke();
  ctx.fillStyle = '#5b9bd5';
  ctx.font = '700 12px Arial';
  ctx.fillText('Y ref', 10, Math.max(16, left.y - 8));
  const measuredPieces = p.pieces.length
    ? p.pieces.filter(piece => piece.measurement)
    : (p.measurement ? [{piece_id: 1, measurement: p.measurement}] : []);
  measuredPieces.forEach((piece, index) => {
    const measurement = piece.measurement;
    const q1 = imageToDisplay(p, measurement.x, measurement.reference_y);
    const q2 = imageToDisplay(p, measurement.x, measurement.line_y);
    const color = COLORS[index % COLORS.length];
    const label = `P${Number(piece.piece_id)} ${formatFeetInches(measurement.measurement_in)}`;
    const labelY = Math.max(24, Math.min(p.canvas.height - 8, q2.y + 22 + (index % 2) * 20));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(q1.x, q1.y); ctx.lineTo(q2.x, q2.y); ctx.stroke();
    ctx.font = '800 13px Arial';
    const labelWidth = ctx.measureText(label).width;
    const labelX = Math.max(6, Math.min(p.canvas.width - labelWidth - 6, q2.x - 34));
    ctx.fillStyle = 'rgba(0, 0, 0, 0.62)';
    ctx.fillRect(labelX - 4, labelY - 15, labelWidth + 8, 20);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, labelX, labelY);
  });
  ctx.restore();
}

function drawPlayerRuler() {
  if (!p.rulerActive || !p.rulerStart) return;
  const end = currentPlayerRulerEnd();
  if (!end) return;
  const measure = playerRulerMeasurement();
  const a1 = imageToDisplay(p, p.rulerStart.x, p.rulerStart.y);
  const a2 = imageToDisplay(p, end.x, end.y);
  const ctx = p.ctx;
  const color = p.rulerMode === 'x' ? '#70c7c2' : (p.rulerMode === 'y' ? '#d6a34b' : '#ffffff');
  ctx.save();
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 7;
  ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y); ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y); ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(a1.x, a1.y, 5, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(a2.x, a2.y, 5, 0, Math.PI * 2); ctx.fill();
  if (measure) {
    const inches = measure.inches === null ? '' : ` | ${measure.inches.toFixed(3)} in`;
    const label = `${rulerModeLabel(p.rulerMode)} ${measure.px.toFixed(1)} px${inches}`;
    const mx = Math.max(8, Math.min(p.canvas.width - 250, (a1.x + a2.x) / 2 + 10));
    const my = Math.max(28, Math.min(p.canvas.height - 12, (a1.y + a2.y) / 2 - 10));
    ctx.font = '800 18px Arial';
    const metrics = ctx.measureText(label);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.62)';
    ctx.fillRect(mx - 7, my - 22, metrics.width + 14, 28);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, mx, my);
  }
  ctx.restore();
}

function drawSobelOnState(state, sobel) {
  if (!sobel) return;
  const ctx = state.ctx;
  if (sobel.roi) {
    const r1 = imageToDisplay(state, sobel.roi.x, sobel.roi.y);
    const r2 = imageToDisplay(state, sobel.roi.x + sobel.roi.w, sobel.roi.y + sobel.roi.h);
    ctx.save();
    ctx.strokeStyle = '#d6a34b';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 5]);
    ctx.strokeRect(r1.x, r1.y, r2.x - r1.x, r2.y - r1.y);
    ctx.restore();
  }
  if (sobel.points) {
    sobel.points.forEach(point => {
      const q = imageToDisplay(state, point.x, point.y);
      ctx.fillStyle = point.inlier ? '#53b689' : '#d6a34b';
      ctx.fillRect(q.x - 2, q.y - 2, 4, 4);
    });
  }
  if (sobel.line) {
    const p1 = imageToDisplay(state, sobel.line.x1, sobel.line.y1);
    const p2 = imageToDisplay(state, sobel.line.x2, sobel.line.y2);
    ctx.save();
    ctx.strokeStyle = sobel.is_valid ? '#53b689' : '#d6a34b';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    ctx.restore();
  }
}

function drawSobelProjection() {
  if (a.pieces.length) {
    a.pieces.forEach(piece => drawSobelOnState(a, piece.sobel));
    return;
  }
  if (!a.sobel) return;
  const ctx = a.ctx;
  if (a.sobel.roi) {
    const r1 = imageToDisplay(a, a.sobel.roi.x, a.sobel.roi.y);
    const r2 = imageToDisplay(a, a.sobel.roi.x + a.sobel.roi.w, a.sobel.roi.y + a.sobel.roi.h);
    ctx.save();
    ctx.strokeStyle = '#d6a34b';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 5]);
    ctx.strokeRect(r1.x, r1.y, r2.x - r1.x, r2.y - r1.y);
    ctx.restore();
  }
  if (a.sobel.points) {
    a.sobel.points.forEach(p => {
      const q = imageToDisplay(a, p.x, p.y);
      a.ctx.fillStyle = p.inlier ? '#53b689' : '#d6a34b';
      a.ctx.fillRect(q.x - 2, q.y - 2, 4, 4);
    });
  }
  if (a.sobel.line) {
    const p1 = imageToDisplay(a, a.sobel.line.x1, a.sobel.line.y1);
    const p2 = imageToDisplay(a, a.sobel.line.x2, a.sobel.line.y2);
    ctx.save();
    ctx.strokeStyle = a.sobel.is_valid ? '#53b689' : '#d6a34b';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    ctx.restore();
  }
}

function updateBoxes() {
  document.getElementById('info-boxes').textContent = a.boxes.length;
  document.getElementById('save-frame').disabled = !a.img;
  document.getElementById('save-frame').textContent = a.boxes.length ? 'Guardar frame' : 'Guardar negativo';
  const list = document.getElementById('box-list');
  if (!a.boxes.length) {
    list.innerHTML = '<div class="kv"><span>Sin boxes. Se guardara como negativo.</span></div>';
    updateModelBoxes();
    return;
  }
  list.innerHTML = a.boxes.map((b, i) => {
    const nb = normBox(b);
    return `<div class="box-item"><span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span><span>#${i+1} ${Math.round(nb.w)}x${Math.round(nb.h)} @ ${Math.round(nb.x)},${Math.round(nb.y)}</span><button onclick="deleteBox(${i})">Borrar</button></div>`;
  }).join('');
  updateModelBoxes();
}

function modelAnnotationComparison() {
  const annotationBoxes = a.boxes.map(normBox);
  const modelBoxes = a.modelBoxes.map(normBox);
  const matchedAnnotations = new Set();
  let matchedPredictions = 0;
  modelBoxes.forEach(modelBox => {
    let bestIndex = -1;
    let bestOverlap = 0;
    annotationBoxes.forEach((annotationBox, index) => {
      const overlap = boxOverlapFraction(modelBox, annotationBox);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestIndex = index;
      }
    });
    if (bestIndex >= 0 && bestOverlap >= 0.35) {
      matchedPredictions += 1;
      matchedAnnotations.add(bestIndex);
    }
  });
  return {
    matchedPredictions,
    missedAnnotations: Math.max(0, annotationBoxes.length - matchedAnnotations.size),
  };
}

function updateModelBoxes() {
  const comparison = modelAnnotationComparison();
  const inferredCount = a.modelBoxes.filter(box => box.inferred).length;
  document.getElementById('info-model-boxes').textContent = a.modelBoxes.length
    ? `${a.modelBoxes.length}${inferredCount ? ` (${inferredCount} por regla)` : ''}`
    : (a.boxRules ? '0' : '-');
  document.getElementById('info-model-matched').textContent = a.boxRules
    ? comparison.matchedPredictions
    : '-';
  document.getElementById('info-model-missed').textContent = a.boxRules
    ? comparison.missedAnnotations
    : '-';
  const list = document.getElementById('model-box-list');
  if (!a.boxRules) {
    list.innerHTML = '';
    return;
  }
  if (!a.modelBoxes.length) {
    list.innerHTML = '<div class="kv"><span>Sin detecciones.</span></div>';
    return;
  }
  list.innerHTML = a.modelBoxes.map((box, index) => {
    const nb = normBox(box);
    const color = box.inferred ? '#d6a34b' : '#70c7c2';
    const source = box.inferred ? 'regla + Sobel' : `conf ${Number(box.conf || 0).toFixed(2)}`;
    return `<div class="box-item"><span class="swatch" style="background:${color}"></span><span>M${index + 1} ${Math.round(nb.w)}x${Math.round(nb.h)} | ${source}</span><span></span></div>`;
  }).join('');
}

function boxOverlapFraction(first, second) {
  const left = Math.max(Number(first.x), Number(second.x));
  const top = Math.max(Number(first.y), Number(second.y));
  const right = Math.min(Number(first.x) + Number(first.w), Number(second.x) + Number(second.w));
  const bottom = Math.min(Number(first.y) + Number(first.h), Number(second.y) + Number(second.h));
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const smallerArea = Math.min(
    Math.max(0, Number(first.w) * Number(first.h)),
    Math.max(0, Number(second.w) * Number(second.h))
  );
  return smallerArea > 0 ? intersection / smallerArea : 0;
}

function updateSobelInfo() {
  const state = document.getElementById('info-sobel-state');
  const conf = document.getElementById('info-sobel-conf');
  const crm = document.getElementById('info-sobel-crm');
  if (!state || !conf || !crm) return;
  if (!a.sobel) {
    state.textContent = 'sin correr';
    conf.textContent = '-';
    crm.textContent = '-';
    return;
  }
  state.textContent = a.sobel.is_valid ? 'linea valida' : (a.sobel.has_roi ? 'linea debil' : 'sin ROI');
  conf.textContent = Number(a.sobel.edge_confidence || 0).toFixed(2);
  crm.textContent = Number(a.sobel.crm_px || 0).toFixed(2) + ' px';
}

async function predictYoloBoxes(options = {}) {
  if (!a.img) return;
  if (!options.quiet) status('Corriendo YOLO...', '');
  const r = await fetch('/api/annotate/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_idx: a.frameIdx, conf: 0.10})
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  a.modelBoxes = (d.boxes || []).map(boxToCorners);
  a.boxRules = d.box_rules || {};
  updateModelBoxes();
  drawAll();
  if (!options.quiet) {
    const comparison = modelAnnotationComparison();
    const inferred = Number(a.boxRules.inferred_count || 0);
    const excluded = Number(a.boxRules.removed_exclusion_count || 0);
    const message = a.modelBoxes.length
      ? `Modelo: ${a.modelBoxes.length} detecciones | ${comparison.matchedPredictions} coincidencias | ${comparison.missedAnnotations} omitidas${inferred ? ` | ${inferred} inferidas por regla` : ''}${excluded ? ` | ${excluded} descartadas por zona` : ''}`
      : `Modelo: sin detecciones | ${comparison.missedAnnotations} anotaciones omitidas${excluded ? ` | ${excluded} descartadas por zona` : ''}`;
    status(message, a.modelBoxes.length ? 'ok' : 'err');
  }
  return d;
}

async function runModel() {
  const button = document.getElementById('a-run-model');
  if (!a.img || button.disabled) return;
  button.disabled = true;
  button.textContent = 'Running...';
  try {
    await predictYoloBoxes();
  } finally {
    button.disabled = false;
    button.textContent = 'Run model';
  }
}

async function runSobelProjection(options = {}) {
  if (!a.img) return;
  if (!options.quiet) status('Calculando Sobel projection...', '');
  const r = await fetch('/api/annotate/sobel_projection', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_idx: a.frameIdx, boxes: a.boxes.map(normBox), conf: 0.10})
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  a.sobel = d;
  a.pieces = d.pieces || [];
  if (!a.boxes.length && d.roi_box) a.boxes = [boxToCorners(d.roi_box)];
  updateBoxes();
  updateSobelInfo();
  drawAll();
  const validCount = Number(d.measurement_summary?.valid_count || 0);
  const msg = `${validCount}/${a.pieces.length} piezas con medicion valida`;
  status(msg, validCount ? 'ok' : 'err');
  return d;
}

async function refreshHistory() {
  const r = await fetch('/api/annotate/history');
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  a.history = d.items || [];
  a.saved = d.count || 0;
  document.getElementById('info-saved').textContent = a.saved;
  updateHistoryUI();
}

function updateHistoryUI() {
  const list = document.getElementById('history-list');
  if (!a.history.length) {
    list.innerHTML = '<div class="kv"><span>Sin frames guardados.</span></div>';
    return;
  }
  list.innerHTML = a.history.map(item => {
    const cls = item.frame_idx === a.frameIdx ? 'history-item current' : 'history-item';
    const kind = item.candidate
      ? 'pendiente: anotar piezas'
      : (item.box_count === 0 ? 'negativo' : `${item.box_count} piezas`);
    const time = Number(item.time_sec || 0).toFixed(2);
    const source = sourceName(item.source);
    return `<div class="${cls}">
      <div><strong>${source} | Frame ${item.frame_idx}</strong><span>${time}s | ${kind}</span></div>
      <button onclick="loadAnnotateFrame(${item.frame_idx})">Ir</button>
    </div>`;
  }).join('');
}

function deleteBox(i) { a.boxes.splice(i, 1); a.sobel = null; a.pieces = []; updateSobelInfo(); updateBoxes(); drawAll(); }
function undoBox() {
  if (a.cornerA) { a.cornerA = null; a.preview = null; }
  else a.boxes.pop();
  a.sobel = null;
  a.pieces = [];
  updateSobelInfo();
  updateBoxes(); drawAll();
}
function clearBoxes() {
  a.boxes = []; a.cornerA = null; a.preview = null; a.sobel = null; a.pieces = [];
  updateSobelInfo(); updateBoxes(); drawAll();
}

async function saveFrame() {
  const payload = {
    frame_idx: a.frameIdx, time_sec: a.timeSec, img_w: a.imgW, img_h: a.imgH,
    boxes: a.boxes.map(normBox)
  };
  const r = await fetch('/api/annotate/save', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  a.saved = d.saved_count;
  document.getElementById('info-saved').textContent = d.saved_count;
  a.history = d.history || a.history;
  updateHistoryUI();
  const kind = a.boxes.length ? `${a.boxes.length} boxes` : 'negativo sin boxes';
  status(`Frame guardado (${kind}): ${d.image} | ${d.label}`, 'ok');
}

function drawAll() { drawHomography(); drawWarp(); drawAnnotate(); drawMeasure(); drawPlayer(); }

function roiSideMidpoints() {
  if (h.expandedPoints.length !== 4) return {};
  const pts = h.expandedPoints.map(p => imageToDisplay(h, p.x, p.y));
  return {
    top: {x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2},
    right: {x: (pts[1].x + pts[2].x) / 2, y: (pts[1].y + pts[2].y) / 2},
    bottom: {x: (pts[2].x + pts[3].x) / 2, y: (pts[2].y + pts[3].y) / 2},
    left: {x: (pts[3].x + pts[0].x) / 2, y: (pts[3].y + pts[0].y) / 2},
  };
}

function distToSegment(px, py, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  if (!len2) return Math.hypot(px - a.x, py - a.y);
  const t = Math.max(0, Math.min(1, ((px - a.x) * dx + (py - a.y) * dy) / len2));
  const x = a.x + t * dx, y = a.y + t * dy;
  return Math.hypot(px - x, py - y);
}

function nearestMeasureSegmentPart(cx, cy) {
  if (m.mode !== 'segment' || m.pending) return null;
  const candidates = m.segments.map((segment, index) => ({
    index,
    start: imageToDisplay(m, segment.x1, segment.y1),
    end: imageToDisplay(m, segment.x2, segment.y2),
  })).reverse();
  for (const candidate of candidates) {
    if (Math.hypot(cx - candidate.start.x, cy - candidate.start.y) <= 14) {
      return {index: candidate.index, part: 'start'};
    }
    if (Math.hypot(cx - candidate.end.x, cy - candidate.end.y) <= 14) {
      return {index: candidate.index, part: 'end'};
    }
  }
  for (const candidate of candidates) {
    if (distToSegment(cx, cy, candidate.start, candidate.end) <= 10) {
      return {index: candidate.index, part: 'line'};
    }
  }
  return null;
}

function beginMeasureSegmentDrag(hit, imagePoint) {
  const segment = m.segments[hit.index];
  if (!segment) return false;
  m.selectedSegment = hit.index;
  m.draggingSegment = {
    index: hit.index,
    part: hit.part,
    anchor: {...imagePoint},
    original: {
      x1: Number(segment.x1), y1: Number(segment.y1),
      x2: Number(segment.x2), y2: Number(segment.y2),
    },
  };
  return true;
}

function updateMeasureSegmentDrag(imagePoint) {
  const drag = m.draggingSegment;
  if (!drag) return;
  const segment = m.segments[drag.index];
  if (!segment) {
    m.draggingSegment = null;
    return;
  }
  const x = Math.max(0, Math.min(m.imgW - 1, imagePoint.x));
  const y = Math.max(0, Math.min(m.imgH - 1, imagePoint.y));
  if (drag.part === 'start') {
    segment.x1 = x; segment.y1 = y;
  } else if (drag.part === 'end') {
    segment.x2 = x; segment.y2 = y;
  } else {
    const original = drag.original;
    const minDx = -Math.min(original.x1, original.x2);
    const maxDx = (m.imgW - 1) - Math.max(original.x1, original.x2);
    const minDy = -Math.min(original.y1, original.y2);
    const maxDy = (m.imgH - 1) - Math.max(original.y1, original.y2);
    const dx = Math.max(minDx, Math.min(maxDx, imagePoint.x - drag.anchor.x));
    const dy = Math.max(minDy, Math.min(maxDy, imagePoint.y - drag.anchor.y));
    segment.x1 = original.x1 + dx; segment.y1 = original.y1 + dy;
    segment.x2 = original.x2 + dx; segment.y2 = original.y2 + dy;
  }
  segment.px = Math.hypot(segment.x2 - segment.x1, segment.y2 - segment.y1);
  segment.inch_per_px = segment.px > 0 ? Number(segment.inches) / segment.px : null;
}

function nearestWorkRoiSide(cx, cy) {
  if (h.expandedPoints.length !== 4) return null;
  const pts = h.expandedPoints.map(p => imageToDisplay(h, p.x, p.y));
  const sides = [
    ['top', pts[0], pts[1]],
    ['right', pts[1], pts[2]],
    ['bottom', pts[2], pts[3]],
    ['left', pts[3], pts[0]],
  ];
  let bestSide = null;
  let bestDist = 16;
  for (const [side, a, b] of sides) {
    const dist = distToSegment(cx, cy, a, b);
    if (dist < bestDist) {
      bestDist = dist;
      bestSide = side;
    }
  }
  return bestSide;
}

function applyHomographyPoint(matrix, point) {
  if (!matrix) return null;
  const den = matrix[2][0] * point.x + matrix[2][1] * point.y + matrix[2][2];
  if (!den) return null;
  return {
    x: (matrix[0][0] * point.x + matrix[0][1] * point.y + matrix[0][2]) / den,
    y: (matrix[1][0] * point.x + matrix[1][1] * point.y + matrix[1][2]) / den,
  };
}

function inverse3x3(m) {
  const a = m[0][0], b = m[0][1], c = m[0][2];
  const d = m[1][0], e = m[1][1], f = m[1][2];
  const g = m[2][0], h2 = m[2][1], i = m[2][2];
  const A = e * i - f * h2, B = c * h2 - b * i, C = b * f - c * e;
  const D = f * g - d * i, E = a * i - c * g, F = c * d - a * f;
  const G = d * h2 - e * g, H = b * g - a * h2, I = a * e - b * d;
  const det = a * A + b * D + c * G;
  if (Math.abs(det) < 1e-12) return null;
  return [[A / det, B / det, C / det], [D / det, E / det, F / det], [G / det, H / det, I / det]];
}

function sourcePointsFromMargins() {
  if (!h.baseMatrix || !h.baseSize) return h.expandedPoints;
  const inv = inverse3x3(h.baseMatrix);
  if (!inv) return h.expandedPoints;
  const m = h.roiMargins;
  const rect = [
    {x: -m.left, y: -m.top},
    {x: h.baseSize.width - 1 + m.right, y: -m.top},
    {x: h.baseSize.width - 1 + m.right, y: h.baseSize.height - 1 + m.bottom},
    {x: -m.left, y: h.baseSize.height - 1 + m.bottom},
  ];
  return rect.map(p => applyHomographyPoint(inv, p));
}

function updateRoiSideFromImagePoint(side, imagePoint) {
  if (!h.baseMatrix || !h.baseSize) return;
  const q = applyHomographyPoint(h.baseMatrix, imagePoint);
  if (!q) return;
  if (side === 'left') h.roiMargins.left = Math.max(0, -q.x);
  if (side === 'right') h.roiMargins.right = Math.max(0, q.x - (h.baseSize.width - 1));
  if (side === 'top') h.roiMargins.top = Math.max(0, -q.y);
  if (side === 'bottom') h.roiMargins.bottom = Math.max(0, q.y - (h.baseSize.height - 1));
  h.expandedPoints = sourcePointsFromMargins();
}

function installPanZoom(state, zoomInput, zoomLabel, hud, onClick) {
  state.wrap.addEventListener('click', ev => {
    if (!state.img || ev.button !== 0 || state.didDrag) return;
    const rect = state.canvas.getBoundingClientRect();
    const p = displayToImage(state, ev.clientX - rect.left, ev.clientY - rect.top);
    onClick(p);
  });
  state.wrap.addEventListener('mousedown', ev => {
    if (state === h && ev.button === 0 && h.expandedPoints.length === 4) {
      const rect = state.canvas.getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;
      const side = nearestWorkRoiSide(cx, cy);
      if (side) {
        ev.preventDefault();
        h.draggingRoiSide = side;
        h.roiManual = true;
        state.didDrag = false;
        state.wrap.style.cursor = 'grabbing';
        return;
      }
    }
    if (state === m && ev.button === 0 && m.mode === 'segment' && !m.pending) {
      const rect = state.canvas.getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;
      const hit = nearestMeasureSegmentPart(cx, cy);
      if (hit) {
        ev.preventDefault();
        const imagePoint = displayToImage(m, cx, cy);
        if (beginMeasureSegmentDrag(hit, imagePoint)) {
          state.didDrag = false;
          state.wrap.style.cursor = 'grabbing';
          drawAll();
          return;
        }
      }
    }
    if (state === m && ev.button === 0 && m.mode === 'reference' && m.referenceY !== null) {
      const rect = state.canvas.getBoundingClientRect();
      const cy = ev.clientY - rect.top;
      const ref = imageToDisplay(m, 0, m.referenceY);
      if (Math.abs(cy - ref.y) <= 16) {
        ev.preventDefault();
        m.draggingReference = true;
        state.didDrag = false;
        state.wrap.style.cursor = 'grabbing';
        return;
      }
    }
    if (ev.button === 1 || ev.button === 2) {
      ev.preventDefault();
      state.panning = true; state.didDrag = false;
      state.panAnchor = {x: ev.clientX, y: ev.clientY};
      state.panStart = {x: state.panX, y: state.panY};
      state.wrap.style.cursor = 'grabbing';
    }
  });
  window.addEventListener('mouseup', () => {
    if (state === h && h.draggingRoiSide) {
      h.draggingRoiSide = null;
      state.didDrag = true;
      state.wrap.style.cursor = 'crosshair';
      updatePoints();
      status('Lado del ROI de trabajo actualizado.', 'ok');
      setTimeout(() => { state.didDrag = false; }, 0);
      return;
    }
    if (state === m && m.draggingSegment) {
      const index = m.draggingSegment.index;
      m.draggingSegment = null;
      state.didDrag = true;
      state.wrap.style.cursor = 'crosshair';
      updateMeasureInfo();
      const segment = m.segments[index];
      status(`Medicion #${index + 1} actualizada: ${Number(segment?.px || 0).toFixed(1)} px.`, 'ok');
      setTimeout(() => { state.didDrag = false; }, 0);
      return;
    }
    if (state === m && m.draggingReference) {
      m.draggingReference = false;
      state.didDrag = true;
      state.wrap.style.cursor = 'crosshair';
      updateMeasureInfo();
      status('Linea Y actualizada.', 'ok');
      setTimeout(() => { state.didDrag = false; }, 0);
      return;
    }
    state.panning = false; state.panAnchor = null; state.wrap.style.cursor = 'crosshair';
    setTimeout(() => { state.didDrag = false; }, 0);
  });
  state.wrap.addEventListener('mousemove', ev => {
    if (!state.img) return;
    const rect = state.canvas.getBoundingClientRect();
    const cx = Math.max(0, Math.min(ev.clientX - rect.left, state.canvas.width - 1));
    const cy = Math.max(0, Math.min(ev.clientY - rect.top, state.canvas.height - 1));
    const imgPoint = displayToImage(state, cx, cy);
    hud.textContent = `x: ${Math.round(imgPoint.x)} y: ${Math.round(imgPoint.y)}`;
    if (state === h && h.draggingRoiSide) {
      updateRoiSideFromImagePoint(h.draggingRoiSide, imgPoint);
      h.roiManual = true;
      state.didDrag = true;
      drawAll();
      return;
    }
    if (state === m) {
      if (m.draggingSegment) {
        updateMeasureSegmentDrag(imgPoint);
        state.didDrag = true;
        updateMeasureInfo();
      } else if (m.draggingReference) {
        m.referenceY = imgPoint.y;
        state.didDrag = true;
        updateMeasureInfo();
      } else if (m.pending) {
        m.preview = imgPoint;
      } else if (m.mode === 'segment' && !state.panning) {
        state.wrap.style.cursor = nearestMeasureSegmentPart(cx, cy) ? 'grab' : 'crosshair';
      }
    }
    if (state === p && p.rulerActive && p.rulerStart && !p.rulerEnd && !state.panning) {
      p.rulerPreview = constrainPlayerRulerPoint(p.rulerStart, imgPoint);
      updatePlayerRulerInfo();
    }
    if (state === a && a.cornerA) a.preview = {x1: a.cornerA.x, y1: a.cornerA.y, x2: imgPoint.x, y2: imgPoint.y};
    if (state.panning && state.panAnchor) {
      const dx = ev.clientX - state.panAnchor.x, dy = ev.clientY - state.panAnchor.y;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) state.didDrag = true;
      const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
      state.panX = state.panStart.x - dx * vw / state.canvas.width;
      state.panY = state.panStart.y - dy * vh / state.canvas.height;
      clamp(state);
    }
    drawAll();
  });
  state.wrap.addEventListener('contextmenu', ev => ev.preventDefault());
  state.wrap.addEventListener('wheel', ev => {
    ev.preventDefault();
    if (!state.img) return;
    const rect = state.canvas.getBoundingClientRect();
    const cx = ev.clientX - rect.left, cy = ev.clientY - rect.top;
    const anchor = displayToImage(state, cx, cy);
    state.zoom = Math.max(1, Math.min(state.zoom * (ev.deltaY < 0 ? 1.15 : 1 / 1.15), 20));
    const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
    state.panX = anchor.x - (cx / state.canvas.width) * vw;
    state.panY = anchor.y - (cy / state.canvas.height) * vh;
    clamp(state);
    zoomInput.value = state.zoom; zoomLabel.textContent = state.zoom.toFixed(1) + 'x';
    drawAll();
  }, {passive: false});
  zoomInput.addEventListener('input', () => {
    const center = displayToImage(state, state.canvas.width / 2, state.canvas.height / 2);
    state.zoom = parseFloat(zoomInput.value);
    const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
    state.panX = center.x - vw / 2; state.panY = center.y - vh / 2;
    clamp(state);
    zoomLabel.textContent = state.zoom.toFixed(1) + 'x';
    drawAll();
  });
}

installPanZoom(h, document.getElementById('h-zoom'), document.getElementById('h-zoom-label'), document.getElementById('h-hud'), p => {
  if (h.points.length >= 4) return;
  h.points.push(p); updatePoints(); drawAll();
});
document.getElementById('h-expand').addEventListener('input', () => {
  h.expandPct = parseFloat(document.getElementById('h-expand').value) || 0;
  document.getElementById('h-expand-label').textContent = Math.round(h.expandPct) + '%';
  h.roiManual = false;
  h.expandedPoints = [];
  h.roiMargins = {left: 0, right: 0, top: 0, bottom: 0};
  if (h.points.length === 4) requestWarp();
  drawAll();
});
installPanZoom(a, document.getElementById('a-zoom'), document.getElementById('a-zoom-label'), document.getElementById('a-hud'), p => {
  if (!a.cornerA) { a.cornerA = p; a.preview = {x1: p.x, y1: p.y, x2: p.x, y2: p.y}; }
  else {
    const b = {x1: a.cornerA.x, y1: a.cornerA.y, x2: p.x, y2: p.y};
    const nb = normBox(b);
    a.cornerA = null; a.preview = null;
    if (nb.w >= 4 && nb.h >= 4) {
      const conflictIndex = a.boxes.findIndex(existing => boxOverlapFraction(nb, normBox(existing)) >= 0.35);
      if (conflictIndex >= 0) {
        status(`El nuevo box ocupa el mismo espacio que la anotacion #${conflictIndex + 1}.`, 'err');
      } else {
        a.boxes.push(b);
      }
    }
  }
  a.sobel = null;
  a.pieces = [];
  updateSobelInfo();
  updateBoxes(); drawAll();
});
installPanZoom(m, document.getElementById('m-zoom'), document.getElementById('m-zoom-label'), document.getElementById('m-hud'), p => {
  measureClick(p);
});
installPanZoom(p, document.getElementById('p-zoom'), document.getElementById('p-zoom-label'), document.getElementById('p-hud'), point => {
  playerRulerClick(point);
});

document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT') return;
  const annotate = document.getElementById('annotate-view').classList.contains('active');
  const measure = document.getElementById('measure-view').classList.contains('active');
  const player = document.getElementById('player-view').classList.contains('active');
  if (annotate) {
    if (ev.key === 's' || ev.key === 'S') saveFrame();
    if (ev.key === 'z' || ev.key === 'Z') undoBox();
    if (ev.key === 'r' || ev.key === 'R') clearBoxes();
    if (ev.key === 'ArrowLeft') stepAnnotate(-1);
    if (ev.key === 'ArrowRight') stepAnnotate(1);
    if (ev.key === 'ArrowUp') stepAnnotate(-30);
    if (ev.key === 'ArrowDown') stepAnnotate(30);
  } else if (measure) {
    if (ev.key === 's' || ev.key === 'S') saveMeasureCalibration();
    if (ev.key === 'z' || ev.key === 'Z') undoMeasureSegment();
    if (ev.key === 'r' || ev.key === 'R') setMeasureMode('reference');
    if (ev.key === 'm' || ev.key === 'M') setMeasureMode('segment');
    if (ev.key === 'e' || ev.key === 'E') setMeasureMode('exclusion');
    if (ev.key === 'ArrowLeft') stepMeasure(-1);
    if (ev.key === 'ArrowRight') stepMeasure(1);
    if (ev.key === 'ArrowUp') stepMeasure(-30);
    if (ev.key === 'ArrowDown') stepMeasure(30);
  } else if (player) {
    if (ev.key === ' ' || ev.key === 'p' || ev.key === 'P') { ev.preventDefault(); togglePlayerPlay(); return; }
    if (ev.key === 'x' || ev.key === 'X') togglePlayerSpeed();
    if (ev.key === 'ArrowLeft') stepPlayer(-1);
    if (ev.key === 'ArrowRight') stepPlayer(1);
    if (ev.key === 'ArrowUp') stepPlayer(-30);
    if (ev.key === 'ArrowDown') stepPlayer(30);
  } else {
    if (ev.key === 's' || ev.key === 'S') saveHomography();
    if (ev.key === 'z' || ev.key === 'Z') undoPoint();
    if (ev.key === 'r' || ev.key === 'R') resetPoints();
  }
});

window.addEventListener('resize', () => { fitAll(); drawAll(); });

(async function init() {
  fitAll();
  try { await loadMeta(); } catch (e) { status(String(e), 'err'); }
  await loadHomographyFrame();
})();
</script>
</body>
</html>
"""


app = Flask(__name__)
_args: argparse.Namespace
_video_playlist: dict | None = None
_image: np.ndarray | None = None
_source_label = ""
_yolo_model = None
_piece_box_profile_cache: dict | None = None


@app.route("/")
def index():
    return render_template_string(HTML, source_label=_source_label, second=_args.second)


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
    second = float(data.get("second", _args.second))
    try:
        _args.second = second
        _image, _source_label, frame_idx, time_sec = load_reference_image(_args)
        hgt, wid = _image.shape[:2]
        return jsonify(
            image=img_to_b64(_image, quality=86),
            width=wid,
            height=hgt,
            frame_idx=frame_idx,
            time_sec=time_sec,
            label=_source_label,
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
        warp, ordered, warp_points, _dst, _matrix, size, base_matrix, base_size, margins, work_rect = compute_warp(
            _image,
            pts,
            expand_pct=expand_pct,
            roi_margins=roi_margins,
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
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


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
        warp, ordered, warp_points, dst, matrix, size, base_matrix, base_size, margins, work_rect = compute_warp(
            _image,
            pts,
            expand_pct=expand_pct,
            roi_margins=roi_margins,
        )
        _args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = homography_json_path()
        src_path = _args.output_dir / "homography_selection_source.jpg"
        warp_path = _args.output_dir / "homography_selection_warp.jpg"
        backup_path = None
        if json_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = _args.output_dir / f"homography_selection_{stamp}.bak.json"
            backup_path.write_bytes(json_path.read_bytes())

        src_vis = _image.copy()
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
            "destination_points": dst.tolist(),
            "output_size": list(size),
            "homography_matrix": matrix.tolist(),
            "source_preview": str(src_path),
            "warp_preview": str(warp_path),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return jsonify(path=str(json_path), backup=str(backup_path) if backup_path else None)
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
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "frame_idx": int(data.get("frame_idx", 0)),
            "time_sec": float(data.get("time_sec", 0.0)),
            "img_w": int(data.get("img_w", 0)),
            "img_h": int(data.get("img_h", 0)),
            "segments": segments,
            "reference_y": data.get("reference_y"),
            "reference_offset_in": MEASUREMENT_REFERENCE_OFFSET_IN,
            "exclusion_zones": exclusion_zones,
            "exclusion_max_box_overlap": EXCLUSION_ZONE_MAX_BOX_OVERLAP,
            "inch_per_px": inch_per_px,
            "px_per_in": (1.0 / float(inch_per_px)) if inch_per_px else None,
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
        matrix, out_size, _homography = load_homography()
        rectified = cv2.warpPerspective(original, matrix, out_size)
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
        original_overlay = mvp_original_overlay_for_pieces(pieces, calibration, matrix, rect_w)
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
    video_paths = discover_video_paths(_args)
    if _args.latest_video_format_only:
        video_paths = latest_video_format_paths(video_paths)
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
