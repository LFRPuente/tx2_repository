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
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request


DEFAULT_VIDEO = Path(r"C:\Users\luis_\Downloads\20260508_000307_7F66.mkv")
DEFAULT_SECOND = 155.0
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\luis_\Desktop\tx2_cv_2026-05-11\outputs")
DEFAULT_DATASET_DIR = Path(r"C:\Users\luis_\Desktop\tx2_cv_2026-05-11\dataset")
DEFAULT_MODEL = Path(
    r"C:\Users\luis_\Desktop\tx2_cv_2026-05-11\runs\detect\runs_tx2\yolo11n_tubos_v1\weights\best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--second", type=float, default=DEFAULT_SECOND)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=5050)
    return parser.parse_args()


def open_video(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    return cap


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
    frame, frame_idx, time_sec = read_frame_by_second(args.video, args.second)
    return frame, f"{args.video} @ {time_sec:.3f}s", frame_idx, time_sec


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


def load_measurement_calibration() -> dict:
    path = measurement_json_path()
    if not path.exists():
        return {
            "segments": [],
            "inch_per_px": None,
            "reference_y": None,
            "reference_offset_in": MEASUREMENT_REFERENCE_OFFSET_IN,
            "path": str(path),
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("reference_offset_in", MEASUREMENT_REFERENCE_OFFSET_IN)
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
    if not images_dir.exists():
        return []

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
            }
        )
    return sorted(items, key=lambda item: item["frame_idx"])


def load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        if not _args.model.exists():
            raise RuntimeError(f"No existe el modelo YOLO: {_args.model}")
        _yolo_model = YOLO(str(_args.model))
    return _yolo_model


def predict_yolo_boxes(rectified: np.ndarray, conf: float = 0.10, imgsz: int = 960) -> list[dict]:
    model = load_yolo_model()
    result = model.predict(rectified, conf=conf, imgsz=imgsz, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

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
    return sorted(boxes, key=lambda item: item["w"] * item["h"] * item["conf"], reverse=True)


def best_box_for_projection(rectified: np.ndarray, boxes: list[dict], conf: float) -> dict | None:
    candidates = boxes or predict_yolo_boxes(rectified, conf=conf)
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item["w"]) * float(item["h"]) * float(item.get("conf", 1.0)))


def sobel_projection_for_box(rectified: np.ndarray, box: dict) -> dict:
    from yolo_roi_sobel_projection import ProjectionConfig, edge_response_from_roi, project_edge_line

    cfg = ProjectionConfig(
        roi_pad_x=0,
        roi_pad_y=0,
        score_keep_percentile=35.0,
        min_points=12,
        line_inlier_tol=8.0,
        max_abs_slope=0.35,
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
        global_left_x = 0.0
        global_right_x = float(width - 1)
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
    is_valid = bool(projection["is_valid"]) and confidence >= 0.30 and crm_px <= 8.0

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
}
.toolbar label { color: var(--muted); font-size: 12px; }
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
    <label>Zoom</label>
    <input type="range" id="a-zoom" min="1" max="20" step="0.1" value="1">
    <span class="pill" id="a-zoom-label">1.0x</span>
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
          <h2>Boxes</h2>
          <div class="box-list" id="box-list"></div>
        </div>
      </aside>
    </section>

    <section class="view" id="measure-view">
      <div class="pane">
        <div class="pane-title"><span>Mesa rectificada</span><span>Segmento: 2 clicks | Linea Y: click/arrastrar | rueda: zoom</span></div>
        <div class="canvas-wrap" id="m-wrap">
          <canvas id="m-canvas"></canvas>
          <div class="hud" id="m-hud">x: - y: -</div>
        </div>
      </div>
      <aside class="side">
        <div class="section">
          <h2>Frame</h2>
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
          <h2>Segmentos</h2>
          <div class="box-list" id="m-segment-list"></div>
        </div>
      </aside>
    </section>

    <section class="view" id="player-view">
      <div class="pane">
        <div class="pane-title"><span>Reproductor YOLO + Sobel</span><span>calculo en ROI, linea proyectada a todo el frame</span></div>
        <div class="canvas-wrap" id="p-wrap" style="cursor:default">
          <canvas id="p-canvas"></canvas>
          <div class="hud" id="p-hud">x: - y: -</div>
        </div>
      </div>
      <aside class="side">
        <div class="section">
          <h2>Frame actual</h2>
          <div class="kv"><span>Frame</span><strong id="p-info-frame">-</strong></div>
          <div class="kv"><span>Tiempo</span><strong id="p-info-time">-</strong></div>
          <div class="kv"><span>Tamano</span><strong id="p-info-size">-</strong></div>
          <div class="kv"><span>Velocidad</span><strong id="p-info-speed">x1</strong></div>
        </div>
        <div class="section">
          <h2>YOLO</h2>
          <div class="kv"><span>Boxes</span><strong id="p-info-boxes">0</strong></div>
          <div class="kv"><span>Conf</span><strong id="p-info-yolo-conf">-</strong></div>
        </div>
        <div class="section">
          <h2>Sobel projection</h2>
          <div class="kv"><span>Estado</span><strong id="p-info-sobel-state">sin correr</strong></div>
          <div class="kv"><span>Conf edge</span><strong id="p-info-sobel-conf">-</strong></div>
          <div class="kv"><span>CRM</span><strong id="p-info-sobel-crm">-</strong></div>
          <div class="kv"><span>Medida Y</span><strong id="p-info-measure">-</strong></div>
        </div>
        <div class="section">
          <h2>Regla manual</h2>
          <div class="kv"><span>Modo</span><strong id="p-ruler-info-mode">off</strong></div>
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
  frameIdx: 0, timeSec: 0, boxes: [], cornerA: null, preview: null,
  panning: false, panAnchor: null, panStart: null, saved: 0, history: [], sobel: null
};
a.ctx = a.canvas.getContext('2d');

const m = {
  canvas: document.getElementById('m-canvas'), wrap: document.getElementById('m-wrap'),
  img: null, imgW: 0, imgH: 0, zoom: 1, panX: 0, panY: 0,
  frameIdx: 0, timeSec: 0, segments: [], pending: null, preview: null,
  referenceY: null, inchPerPx: null, mode: 'segment', draggingReference: false,
  panning: false, panAnchor: null, panStart: null
};
m.ctx = m.canvas.getContext('2d');

const p = {
  canvas: document.getElementById('p-canvas'), wrap: document.getElementById('p-wrap'),
  img: null, imgW: 0, imgH: 0, zoom: 1, panX: 0, panY: 0,
  frameIdx: 0, timeSec: 0, boxes: [], sobel: null, calibration: null, measurement: null,
  captures: [], rulerActive: false, rulerMode: 'free', rulerStart: null, rulerEnd: null, rulerPreview: null,
  playing: false, speed: 1, playTask: null, panning: false, panAnchor: null, panStart: null
};
p.ctx = p.canvas.getContext('2d');

function status(msg, cls='') {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status ' + cls;
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
  if (name === 'player' && !p.img) loadPlayerSecond();
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
  a.img = null; a.imgW = 0; a.imgH = 0; a.boxes = []; a.cornerA = null; a.preview = null; a.sobel = null;
  m.img = null; m.imgW = 0; m.imgH = 0; m.pending = null; m.preview = null;
  p.img = null; p.imgW = 0; p.imgH = 0; p.boxes = []; p.sobel = null; p.measurement = null;
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
      a.frameIdx = d.frame_idx; a.timeSec = d.time_sec;
      a.boxes = (d.boxes || []).map(boxToCorners); a.cornerA = null; a.preview = null; a.sobel = null;
      if (options.resetView !== false) resetView(a);
      document.getElementById('a-second').value = d.time_sec.toFixed(2);
      document.getElementById('a-zoom').value = a.zoom;
      document.getElementById('a-zoom-label').textContent = a.zoom.toFixed(1) + 'x';
      document.getElementById('info-frame').textContent = d.frame_idx;
      document.getElementById('info-time').textContent = d.time_sec.toFixed(3) + 's';
      document.getElementById('info-size').textContent = `${d.width}x${d.height}`;
      updateBoxes();
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
      m.frameIdx = d.frame_idx; m.timeSec = d.time_sec;
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
  m.inchPerPx = calibration.inch_per_px ?? computeInchPerPx();
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
  document.getElementById('m-info-frame').textContent = m.img ? m.frameIdx : '-';
  document.getElementById('m-info-time').textContent = m.img ? m.timeSec.toFixed(3) + 's' : '-';
  document.getElementById('m-info-size').textContent = m.img ? `${m.imgW}x${m.imgH}` : '-';
  document.getElementById('m-info-inch-px').textContent = m.inchPerPx ? m.inchPerPx.toFixed(5) : '-';
  document.getElementById('m-info-px-inch').textContent = m.inchPerPx ? (1 / m.inchPerPx).toFixed(2) : '-';
  document.getElementById('m-info-segments').textContent = m.segments.length;
  document.getElementById('m-info-ref').textContent = m.referenceY === null ? '-' : Math.round(m.referenceY) + ' px';
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
  document.getElementById('m-mode-segment').classList.toggle('active', mode === 'segment');
  document.getElementById('m-mode-ref').classList.toggle('active', mode === 'reference');
  status(mode === 'segment' ? 'Modo segmento: marca 2 puntos.' : 'Modo Linea Y: click o arrastra la referencia horizontal.', 'ok');
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
  updateMeasureInfo(); drawAll();
}

function undoMeasureSegment() {
  if (m.pending) { m.pending = null; m.preview = null; }
  else m.segments.pop();
  updateMeasureInfo(); drawAll();
}

function clearMeasureCalibration() {
  m.segments = []; m.pending = null; m.preview = null; m.referenceY = null; m.inchPerPx = null;
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
    inch_per_px: m.inchPerPx,
  };
  const r = await fetch('/api/measure/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  status(`Mediciones guardadas: ${d.path}`, 'ok');
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
      p.frameIdx = d.frame_idx; p.timeSec = d.time_sec;
      p.boxes = d.boxes || [];
      p.sobel = d.sobel || null;
      p.calibration = d.calibration || null;
      p.measurement = d.measurement || null;
      if (options.resetView !== false) resetView(p);
      document.getElementById('p-second').value = d.time_sec.toFixed(2);
      document.getElementById('p-zoom').value = p.zoom;
      document.getElementById('p-zoom-label').textContent = p.zoom.toFixed(1) + 'x';
      updatePlayerInfo();
      updatePlayerCapturesUI();
      fitAll(); drawAll();
      if (!options.quiet) {
        const cls = p.boxes.length ? 'ok' : 'err';
        status(`Frame ${d.frame_idx}: YOLO ${p.boxes.length} box(es), Sobel ${p.sobel && p.sobel.line ? 'con linea' : 'sin linea'}`, cls);
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
  document.getElementById('p-info-frame').textContent = p.img ? p.frameIdx : '-';
  document.getElementById('p-info-time').textContent = p.img ? p.timeSec.toFixed(3) + 's' : '-';
  document.getElementById('p-info-size').textContent = p.img ? `${p.imgW}x${p.imgH}` : '-';
  document.getElementById('p-info-speed').textContent = `x${p.speed}`;
  document.getElementById('p-speed').textContent = `x${p.speed}`;
  document.getElementById('p-info-boxes').textContent = p.boxes.length;
  document.getElementById('p-info-yolo-conf').textContent = bestConf ? bestConf.toFixed(2) : '-';
  const yoloBadge = document.getElementById('p-yolo-badge');
  yoloBadge.textContent = p.boxes.length ? `YOLO ${p.boxes.length}` : 'YOLO 0';
  yoloBadge.className = 'pill' + (p.boxes.length ? ' ok' : '');

  const state = document.getElementById('p-info-sobel-state');
  const conf = document.getElementById('p-info-sobel-conf');
  const crm = document.getElementById('p-info-sobel-crm');
  const measure = document.getElementById('p-info-measure');
  const sobelBadge = document.getElementById('p-sobel-badge');
  const measureValue = playerMeasureValue(p.measurement);
  measure.textContent = measureValue === null ? '-' : measureValue.toFixed(3) + ' in';
  updatePlayerRulerInfo();
  if (!p.sobel || !p.sobel.has_roi) {
    state.textContent = p.boxes.length ? 'sin ROI' : 'sin box YOLO';
    conf.textContent = '-';
    crm.textContent = '-';
    sobelBadge.textContent = 'Sobel -';
    sobelBadge.className = 'pill';
    return;
  }
  state.textContent = p.sobel.is_valid ? 'linea valida' : 'linea debil';
  conf.textContent = Number(p.sobel.edge_confidence || 0).toFixed(2);
  crm.textContent = Number(p.sobel.crm_px || 0).toFixed(2) + ' px';
  sobelBadge.textContent = p.sobel.line ? 'Sobel linea' : 'Sobel sin linea';
  sobelBadge.className = 'pill' + (p.sobel.is_valid ? ' ok' : '');
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
    const value = item.measurement_in === null || item.measurement_in === undefined
      ? 'sin medida'
      : Number(item.measurement_in).toFixed(3) + ' in';
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
  status(`Captura guardada: frame ${d.capture.frame_idx}${value === null ? '' : ' | ' + value.toFixed(3) + ' in'}`, 'ok');
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
  a.boxes.forEach((b, i) => drawBox(b, COLORS[i % COLORS.length], false, i + 1));
  if (a.preview) drawBox(a.preview, '#ffffff', true, '?');
  drawSobelProjection();
}

function drawMeasure() {
  const ctx = m.ctx, cw = m.canvas.width, ch = m.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!m.img) return;
  const vw = m.imgW / m.zoom, vh = m.imgH / m.zoom;
  ctx.drawImage(m.img, m.panX, m.panY, vw, vh, 0, 0, cw, ch);
  drawGrid(m, 100);

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
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y); ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(a1.x, a1.y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(a2.x, a2.y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.font = '700 12px Arial';
    ctx.fillText(`${Number(s.inches).toFixed(2)} in`, (a1.x + a2.x) / 2 + 8, (a1.y + a2.y) / 2 - 8);
    ctx.restore();
  });

  if (m.pending && m.preview) {
    const p1 = imageToDisplay(m, m.pending.x, m.pending.y);
    const p2 = imageToDisplay(m, m.preview.x, m.preview.y);
    ctx.save();
    ctx.strokeStyle = '#ffffff';
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
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
  drawPlayerMeasurement();
  p.boxes.forEach((b, i) => drawBoxOnState(p, b, COLORS[i % COLORS.length], false, i + 1));
  drawSobelOnState(p, p.sobel);
  drawPlayerRuler();
}

function normBox(b) {
  if ('x' in b && 'y' in b && 'w' in b && 'h' in b) {
    return {x: b.x, y: b.y, w: b.w, h: b.h};
  }
  return {x: Math.min(b.x1, b.x2), y: Math.min(b.y1, b.y2), w: Math.abs(b.x2 - b.x1), h: Math.abs(b.y2 - b.y1)};
}

function boxToCorners(b) {
  if ('x1' in b && 'y1' in b && 'x2' in b && 'y2' in b) return b;
  return {x1: b.x, y1: b.y, x2: b.x + b.w, y2: b.y + b.h};
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
  if (!dashed) {
    ctx.fillStyle = color; ctx.font = '700 12px Arial';
    ctx.fillText(`#${label} ${Math.round(nb.w)}x${Math.round(nb.h)}`, p1.x + 4, p1.y + 14);
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
  if (p.measurement) {
    const q1 = imageToDisplay(p, p.measurement.x, p.measurement.reference_y);
    const q2 = imageToDisplay(p, p.measurement.x, p.measurement.line_y);
    const totalIn = Number(p.measurement.measurement_in ?? p.measurement.delta_in);
    const refDeltaIn = Number(p.measurement.delta_in);
    const labelTotal = `Total: ${totalIn.toFixed(3)} in`;
    const labelRef = `Ref: ${refDeltaIn >= 0 ? '+' : ''}${refDeltaIn.toFixed(3)} in`;
    const labelX = Math.min(p.canvas.width - 220, q2.x + 14);
    const labelY = Math.max(52, Math.min(p.canvas.height - 18, (q1.y + q2.y) / 2));
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(q1.x, q1.y); ctx.lineTo(q2.x, q2.y); ctx.stroke();
    ctx.font = '800 24px Arial';
    const widthTotal = ctx.measureText(labelTotal).width;
    ctx.font = '800 18px Arial';
    const widthRef = ctx.measureText(labelRef).width;
    const labelWidth = Math.max(widthTotal, widthRef);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.fillRect(labelX - 8, labelY - 46, labelWidth + 16, 54);
    ctx.fillStyle = '#ffffff';
    ctx.font = '800 24px Arial';
    ctx.fillText(labelTotal, labelX, labelY - 20);
    ctx.font = '800 18px Arial';
    ctx.fillText(labelRef, labelX, labelY + 2);
  }
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
    return;
  }
  list.innerHTML = a.boxes.map((b, i) => {
    const nb = normBox(b);
    return `<div class="box-item"><span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span><span>#${i+1} ${Math.round(nb.w)}x${Math.round(nb.h)} @ ${Math.round(nb.x)},${Math.round(nb.y)}</span><button onclick="deleteBox(${i})">Borrar</button></div>`;
  }).join('');
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
  a.boxes = (d.boxes || []).map(boxToCorners);
  a.sobel = null;
  updateBoxes();
  updateSobelInfo();
  drawAll();
  if (a.boxes.length && options.autoSobel !== false) {
    return await runSobelProjection({quiet: options.quiet});
  }
  if (!options.quiet) {
    status(a.boxes.length ? `YOLO detecto ${a.boxes.length} box(es)` : 'YOLO no detecto boxes en este frame', a.boxes.length ? 'ok' : 'err');
  }
  return d;
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
  if (!a.boxes.length && d.roi_box) a.boxes = [boxToCorners(d.roi_box)];
  updateBoxes();
  updateSobelInfo();
  drawAll();
  const msg = d.is_valid
    ? `Sobel projection OK | conf=${Number(d.edge_confidence).toFixed(2)} crm=${Number(d.crm_px).toFixed(2)}px`
    : `Sobel projection debil | conf=${Number(d.edge_confidence || 0).toFixed(2)} crm=${Number(d.crm_px || 0).toFixed(2)}px`;
  status(msg, d.is_valid ? 'ok' : 'err');
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
    const kind = item.box_count === 0 ? 'negativo' : `${item.box_count} boxes`;
    const time = Number(item.time_sec || 0).toFixed(2);
    return `<div class="${cls}">
      <div><strong>Frame ${item.frame_idx}</strong><span>${time}s · ${kind}</span></div>
      <button onclick="loadAnnotateFrame(${item.frame_idx})">Ir</button>
    </div>`;
  }).join('');
}

function deleteBox(i) { a.boxes.splice(i, 1); a.sobel = null; updateSobelInfo(); updateBoxes(); drawAll(); }
function undoBox() {
  if (a.cornerA) { a.cornerA = null; a.preview = null; }
  else a.boxes.pop();
  a.sobel = null;
  updateSobelInfo();
  updateBoxes(); drawAll();
}
function clearBoxes() { a.boxes = []; a.cornerA = null; a.preview = null; a.sobel = null; updateSobelInfo(); updateBoxes(); drawAll(); }

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
      if (m.draggingReference) {
        m.referenceY = imgPoint.y;
        state.didDrag = true;
        updateMeasureInfo();
      } else if (m.pending) {
        m.preview = imgPoint;
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
    if (nb.w >= 4 && nb.h >= 4) a.boxes.push(b);
  }
  a.sobel = null;
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
_image: np.ndarray | None = None
_source_label = ""
_yolo_model = None


@app.route("/")
def index():
    return render_template_string(HTML, source_label=_source_label, second=_args.second)


@app.route("/api/meta")
def api_meta():
    try:
        data = video_meta(_args.video)
        data["dataset_dir"] = str(_args.dataset_dir)
        data["homography_exists"] = homography_json_path().exists()
        data["measurement_exists"] = measurement_json_path().exists()
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
        frame, frame_idx, time_sec = read_frame_by_index(_args.video, frame_idx)
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
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/measure/frame", methods=["POST"])
def api_measure_frame():
    data = request.get_json() or {}
    try:
        frame_idx = int(data.get("frame_idx", 0))
        frame, frame_idx, time_sec = read_frame_by_index(_args.video, frame_idx)
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
        frame, frame_idx, time_sec = read_frame_by_index(_args.video, frame_idx)
        frame, _homography = apply_saved_homography(frame)
        boxes = predict_yolo_boxes(frame, conf=conf)
        return jsonify(
            frame_idx=frame_idx,
            time_sec=time_sec,
            boxes=boxes,
            count=len(boxes),
            model=str(_args.model),
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
        frame, frame_idx, time_sec = read_frame_by_index(_args.video, frame_idx)
        frame, _homography = apply_saved_homography(frame)
        box = best_box_for_projection(frame, boxes, conf=conf)
        if box is None:
            return jsonify(
                frame_idx=frame_idx,
                time_sec=time_sec,
                has_roi=False,
                is_valid=False,
                roi=None,
                roi_box=None,
                line=None,
                points=[],
                edge_confidence=0.0,
                crm_px=0.0,
            )
        result = sobel_projection_for_box(frame, box)
        result.update(frame_idx=frame_idx, time_sec=time_sec)
        return jsonify(result)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/player/frame", methods=["POST"])
def api_player_frame():
    data = request.get_json() or {}
    try:
        frame_idx = int(data.get("frame_idx", 0))
        conf = float(data.get("conf", 0.10))
        frame, frame_idx, time_sec = read_frame_by_index(_args.video, frame_idx)
        frame, _homography = apply_saved_homography(frame)
        boxes = predict_yolo_boxes(frame, conf=conf)
        box = max(
            boxes,
            key=lambda item: float(item["w"]) * float(item["h"]) * float(item.get("conf", 1.0)),
            default=None,
        )
        if box is None:
            sobel = {
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
        else:
            sobel = sobel_projection_for_box(frame, box)
            sobel.update(frame_idx=frame_idx, time_sec=time_sec)
        hgt, wid = frame.shape[:2]
        calibration = load_measurement_calibration()
        measurement = measurement_from_sobel(sobel, calibration, wid)
        return jsonify(
            image=img_to_b64(frame, quality=88),
            width=wid,
            height=hgt,
            frame_idx=frame_idx,
            time_sec=time_sec,
            boxes=boxes,
            count=len(boxes),
            sobel=sobel,
            calibration=calibration,
            measurement=measurement,
            model=str(_args.model),
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/mvp/frame", methods=["POST"])
def api_mvp_frame():
    data = request.get_json() or {}
    try:
        frame_idx = int(data.get("frame_idx", 0))
        conf = float(data.get("conf", 0.10))
        original, frame_idx, time_sec = read_frame_by_index(_args.video, frame_idx)
        matrix, out_size, _homography = load_homography()
        rectified = cv2.warpPerspective(original, matrix, out_size)
        boxes = predict_yolo_boxes(rectified, conf=conf)
        box = max(
            boxes,
            key=lambda item: float(item["w"]) * float(item["h"]) * float(item.get("conf", 1.0)),
            default=None,
        )
        if box is None:
            sobel = {
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
        else:
            sobel = sobel_projection_for_box(rectified, box)
            sobel.update(frame_idx=frame_idx, time_sec=time_sec)

        src_h, src_w = original.shape[:2]
        rect_h, rect_w = rectified.shape[:2]
        calibration = load_measurement_calibration()
        measurement = measurement_from_sobel(sobel, calibration, rect_w)
        original_overlay = mvp_original_overlay(sobel, calibration, matrix, rect_w)
        return jsonify(
            original_image=img_to_b64(original, quality=82),
            rectified_image=img_to_b64(rectified, quality=82),
            original_width=src_w,
            original_height=src_h,
            rectified_width=rect_w,
            rectified_height=rect_h,
            frame_idx=frame_idx,
            time_sec=time_sec,
            fps=video_meta(_args.video)["fps"],
            total_frames=video_meta(_args.video)["total_frames"],
            boxes=boxes,
            count=len(boxes),
            sobel=sobel,
            calibration=calibration,
            measurement=measurement,
            original_overlay=original_overlay,
            front_y_ratio=(float(sobel["line"]["y"]) / float(rect_h)) if sobel.get("line") else None,
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
        frame, frame_idx, time_sec = read_frame_by_index(_args.video, frame_idx)
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
            "image": str(image_path),
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
        return jsonify(items=items, count=len(items))
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/annotate/save", methods=["POST"])
def api_annotate_save():
    data = request.get_json() or {}
    try:
        frame_idx = int(data["frame_idx"])
        time_sec = float(data["time_sec"])
        img_w = int(data["img_w"])
        img_h = int(data["img_h"])
        boxes = data["boxes"]

        frame, _frame_idx, _time_sec = read_frame_by_index(_args.video, frame_idx)
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
                    "class_name": "tubo",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        history = dataset_history()
        return jsonify(
            saved_count=len(history),
            history=history,
            image=str(image_path),
            label=str(label_path),
            metadata=str(meta_path),
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 500


def main() -> None:
    global _args, _image, _source_label
    _args = parse_args()
    _args.output_dir.mkdir(parents=True, exist_ok=True)
    _args.dataset_dir.mkdir(parents=True, exist_ok=True)
    try:
        _image, _source_label, _frame_idx, _time_sec = load_reference_image(_args)
    except Exception as exc:
        print(f"Advertencia al precargar frame: {exc}")
        _image = None
        _source_label = str(_args.video)
    print(f"\n  TX2 Vision Tool en http://localhost:{_args.port}\n")
    app.run(host="0.0.0.0", port=_args.port, debug=False)


if __name__ == "__main__":
    main()
