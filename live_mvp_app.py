"""
Live MVP for TX2 vision measurement.

This app is intentionally separate from the existing React MVP. It runs a small
Flask UI plus a background camera reader, live processor, optional PLC monitor,
and PLC-triggered 8 second clip recorder.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template_string, request, send_file

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import homography_web_app as vision
from tx2_database import (
    DatabaseRepository,
    DatabaseUnavailable,
    RecordNotFound,
    RevisionConflict,
    build_event_key,
    build_vision_configuration,
    event_uuid,
    resolve_asset_path,
    select_canonical_snapshot,
    snapshot_pieces,
    snapshot_summary,
)
from tx2_sqlite_database import SQLiteDatabaseRepository

DEFAULT_VIDEO = Path(r"C:\Users\luis_\Downloads\20260724_10\20260724_100105_6439.mkv")
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_DATASET_DIR = ROOT / "dataset_pieces"
DEFAULT_PIECE_MODEL = ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v1" / "weights" / "best.pt"
DEFAULT_LEGACY_MODEL = ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_tubos_v1" / "weights" / "best.pt"
DEFAULT_MODEL = DEFAULT_PIECE_MODEL if DEFAULT_PIECE_MODEL.exists() else DEFAULT_LEGACY_MODEL
DEFAULT_ENDPOINT = "opc.tcp://10.14.6.48:49320"
DEFAULT_WATCHDOG_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD"
DEFAULT_EVENT_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength"
HISTORY_SNAPSHOT_LIMIT = 6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def file_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")


def clean_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    return str(value)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def representative_snapshots(snapshots: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not snapshots:
        return []
    if len(snapshots) <= limit:
        return snapshots.copy()
    if limit == 1:
        return [snapshots[0]]
    last_index = len(snapshots) - 1
    indices = [round(position * last_index / (limit - 1)) for position in range(limit)]
    return [snapshots[index] for index in indices]


def snapshots_contain_piece(snapshots: list[dict[str, Any]]) -> bool:
    for snapshot in snapshots:
        if snapshot_pieces(snapshot):
            return True
        try:
            if int(snapshot_summary(snapshot).get("detected_count") or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def img_to_b64(img: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Could not encode image")
    return base64.b64encode(buf).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live TX2 MVP with camera, PLC, YOLO and recording.")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--source", choices=("video", "rtsp", "auto"), default="rtsp")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--camera-ip", default="10.14.115.241")
    parser.add_argument("--rtsp-url", default=os.environ.get("AXIS_RTSP_URL", ""))
    parser.add_argument("--codec", choices=("jpeg", "h264"), default="h264")
    parser.add_argument("--camera-resolution", default="1920x1080")
    parser.add_argument("--camera-user", default=os.environ.get("AXIS_USER", ""))
    parser.add_argument("--camera-password", default=os.environ.get("AXIS_PASSWORD", ""))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--capture-fps", type=float, default=10.0)
    parser.add_argument("--process-fps", type=float, default=10.0)
    parser.add_argument("--buffer-seconds", type=float, default=2.0)
    parser.add_argument("--buffer-max-frames", type=int, default=60)
    parser.add_argument("--record-seconds", type=float, default=8.0)
    parser.add_argument("--record-fps", type=float, default=10.0)
    parser.add_argument("--max-clips", type=int, default=100)
    parser.add_argument("--plc-enabled", action="store_true")
    parser.add_argument("--plc-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--watchdog-node", default=DEFAULT_WATCHDOG_NODE)
    parser.add_argument("--event-node", default=DEFAULT_EVENT_NODE)
    parser.add_argument("--plc-poll-interval", type=float, default=0.01)
    parser.add_argument("--plc-timeout", type=float, default=8.0)
    parser.add_argument("--plc-edge", choices=("changed", "rising", "falling", "any"), default="rising")
    parser.add_argument("--postgres-dsn", default=os.environ.get("TX2_POSTGRES_DSN", ""))
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path(
            os.environ.get(
                "TX2_SQLITE_PATH",
                str(DEFAULT_OUTPUT_DIR / "tx2_live_mvp.sqlite3"),
            )
        ),
    )
    parser.add_argument("--db-disabled", action="store_true")
    parser.add_argument("--db-retry-seconds", type=float, default=15.0)
    return parser.parse_args()


def build_rtsp_url(args: argparse.Namespace) -> str:
    if args.rtsp_url:
        return args.rtsp_url
    auth = ""
    if args.camera_user and args.camera_password:
        auth = f"{args.camera_user}:{args.camera_password}@"
    fps = max(1, int(round(float(args.capture_fps))))
    return (
        f"rtsp://{auth}{args.camera_ip}/axis-media/media.amp"
        f"?videocodec={args.codec}&resolution={args.camera_resolution}&fps={fps}"
    )


def configure_vision_module(args: argparse.Namespace) -> None:
    vision._args = SimpleNamespace(
        video=args.video,
        second=0.0,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        model=args.model,
        port=args.port,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)


def event_edge(previous: Any, current: Any) -> str:
    if previous is None or previous == current:
        return ""
    if isinstance(previous, bool) and isinstance(current, bool):
        return "rising" if current else "falling"
    return "changed"


def edge_matches(configured: str, observed: str) -> bool:
    if not observed:
        return False
    if configured in ("any", "changed"):
        return True
    return configured == observed


def line_is_valid(line: dict | None) -> bool:
    return bool(line) and all(np.isfinite(float(line[key])) for key in ("x1", "y1", "x2", "y2"))


def format_inches_compact(value: float | int | Decimal | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "-"
    total_sixteenths = int(math.floor(abs(float(value)) * 16.0 + 0.5))
    sign = "-" if float(value) < 0 else ""
    feet, remainder = divmod(total_sixteenths, 12 * 16)
    inches, numerator = divmod(remainder, 16)
    if numerator:
        denominator = 16
        while numerator % 2 == 0:
            numerator //= 2
            denominator //= 2
        inch_text = f"{inches} {numerator}/{denominator}"
    else:
        inch_text = str(inches)
    return f"{sign}{feet}' {inch_text}\""


def draw_line(img: np.ndarray, line: dict | None, color: tuple[int, int, int], label: str) -> None:
    if not line_is_valid(line):
        return
    h, w = img.shape[:2]
    x1 = int(np.clip(float(line["x1"]), -w, w * 2))
    y1 = int(np.clip(float(line["y1"]), -h, h * 2))
    x2 = int(np.clip(float(line["x2"]), -w, w * 2))
    y2 = int(np.clip(float(line["y2"]), -h, h * 2))
    thickness = max(2, round(w / 520))
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness + 3, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    label_x = int(np.clip(min(x1, x2) + 12, 8, max(8, w - 180)))
    label_y = int(np.clip((y1 + y2) / 2 - 10, 24, max(24, h - 12)))
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_rectified_overlay(
    rectified: np.ndarray,
    pieces: list[dict],
    calibration: dict,
) -> np.ndarray:
    out = rectified.copy()
    exclusion_zones = vision.normalize_exclusion_zones(
        calibration.get("exclusion_zones"),
        image_shape=rectified.shape,
    )
    if exclusion_zones:
        zone_layer = out.copy()
        for zone in exclusion_zones:
            x0 = int(round(float(zone["x"])))
            y0 = int(round(float(zone["y"])))
            x1 = int(round(float(zone["x"]) + float(zone["w"])))
            y1 = int(round(float(zone["y"]) + float(zone["h"])))
            cv2.rectangle(zone_layer, (x0, y0), (x1, y1), (48, 62, 210), -1)
        out = cv2.addWeighted(zone_layer, 0.22, out, 0.78, 0.0)
        for zone in exclusion_zones:
            x0 = int(round(float(zone["x"])))
            y0 = int(round(float(zone["y"])))
            x1 = int(round(float(zone["x"]) + float(zone["w"])))
            y1 = int(round(float(zone["y"]) + float(zone["h"])))
            cv2.rectangle(out, (x0, y0), (x1, y1), (38, 48, 210), 2, cv2.LINE_AA)
    for index, piece in enumerate(pieces):
        box = piece["box"]
        piece_id = int(piece["piece_id"])
        color = (62, 214, 166) if piece.get("valid") else (48, 156, 220)
        x0 = int(float(box["x"]))
        y0 = int(float(box["y"]))
        x1 = int(float(box["x"]) + float(box["w"]))
        y1 = int(float(box["y"]) + float(box["h"]))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"P{piece_id} {float(box.get('conf', 0.0)):.2f}",
            (x0 + 3, max(18, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        draw_line(out, (piece.get("sobel") or {}).get("line"), color, f"P{piece_id}")
        measurement = piece.get("measurement")
        if isinstance(measurement, dict):
            label = format_inches_compact(measurement["measurement_in"])
            label_y = min(
                out.shape[0] - 8,
                max(24, int(float(measurement["line_y"])) + 24 + (index % 2) * 18),
            )
            cv2.putText(
                out,
                label,
                (x0 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                label,
                (x0 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    if calibration.get("reference_y") is not None:
        y = float(calibration["reference_y"])
        draw_line(out, {"x1": 0, "y1": y, "x2": out.shape[1] - 1, "y2": y}, (210, 130, 48), "REF")
    return out


def draw_original_overlay(original: np.ndarray, overlay: dict) -> np.ndarray:
    out = original.copy()
    draw_line(out, overlay.get("reference_line"), (210, 130, 48), "REF")
    piece_fronts = overlay.get("piece_fronts") or []
    if piece_fronts:
        for item in piece_fronts:
            color = (40, 210, 128) if item.get("valid") else (48, 156, 220)
            draw_line(out, item.get("line"), color, f"P{int(item['piece_id'])}")
    else:
        draw_line(out, overlay.get("front_line"), (40, 210, 128), "front")
    return out


class FrameBuffer:
    def __init__(self, maxlen: int) -> None:
        self._frames: deque[dict[str, Any]] = deque(maxlen=max(8, int(maxlen)))
        self._lock = threading.Lock()

    def append(self, item: dict[str, Any]) -> None:
        with self._lock:
            self._frames.append(item)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._frames:
                return None
            # Frames are immutable after append, so sharing the array avoids a Full HD copy.
            return self._frames[-1].copy()

    def frames_since(self, min_index: int) -> list[dict[str, Any]]:
        with self._lock:
            selected = [item for item in self._frames if int(item["index"]) > int(min_index)]
            return [item.copy() for item in selected]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "count": len(self._frames),
                "first_index": int(self._frames[0]["index"]) if self._frames else None,
                "last_index": int(self._frames[-1]["index"]) if self._frames else None,
            }


class CameraReader:
    def __init__(self, args: argparse.Namespace, buffer: FrameBuffer) -> None:
        self.args = args
        self.buffer = buffer
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="live-camera-reader", daemon=True)
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "connected": False,
            "source": args.source,
            "source_label": "",
            "error": "",
            "frames_read": 0,
            "fps": args.capture_fps,
            "width": None,
            "height": None,
            "last_frame_utc": None,
        }

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            data = self.state.copy()
        data["buffer"] = self.buffer.stats()
        return data

    def _set_state(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates)

    def _resolve_source(self) -> tuple[str, str]:
        if self.args.source == "rtsp":
            return build_rtsp_url(self.args), f"RTSP {self.args.camera_ip}"
        if self.args.source == "auto" and (self.args.rtsp_url or (self.args.camera_user and self.args.camera_password)):
            return build_rtsp_url(self.args), f"RTSP {self.args.camera_ip}"
        return str(self.args.video), f"Simulated video {self.args.video.name}"

    def _run(self) -> None:
        frame_index = 0
        while not self.stop_event.is_set():
            source, label = self._resolve_source()
            pending_error = "Connecting to video source..."
            if label.startswith("RTSP") and not (self.args.rtsp_url or (self.args.camera_user and self.args.camera_password)):
                pending_error = "Connecting to RTSP without credentials. If the camera requires auth, set AXIS_USER and AXIS_PASSWORD."
            self._set_state(connected=False, source_label=label, error=pending_error)

            cap = cv2.VideoCapture()
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            except Exception:
                pass
            cap.open(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self._set_state(connected=False, source_label=label, error=f"Could not open source: {label}")
                cap.release()
                time.sleep(2.0)
                continue

            fps = float(cap.get(cv2.CAP_PROP_FPS) or self.args.capture_fps or 15.0)
            if not np.isfinite(fps) or fps <= 0:
                fps = self.args.capture_fps or 15.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            simulated_video = label.startswith("Simulated video")
            self._set_state(connected=True, source_label=label, error="", fps=fps, width=width, height=height)

            last_push = time.perf_counter()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if simulated_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self._set_state(connected=False, error="Camera read failed; retrying")
                    break

                now_mono = time.perf_counter()
                item = {
                    "index": frame_index,
                    "utc": utc_now(),
                    "monotonic": now_mono,
                    "frame": frame,
                }
                self.buffer.append(item)
                frame_index += 1
                self._set_state(frames_read=frame_index, last_frame_utc=item["utc"], width=frame.shape[1], height=frame.shape[0])

                if simulated_video:
                    target_delay = 1.0 / max(1.0, float(self.args.capture_fps or fps))
                    elapsed = time.perf_counter() - last_push
                    if elapsed < target_delay:
                        time.sleep(target_delay - elapsed)
                    last_push = time.perf_counter()

            cap.release()
            time.sleep(0.4)


class LiveProcessor:
    def __init__(self, args: argparse.Namespace, buffer: FrameBuffer) -> None:
        self.args = args
        self.buffer = buffer
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="live-vision-processor", daemon=True)
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "ok": False,
            "processing": False,
            "error": "",
            "processed_count": 0,
            "last_frame_index": None,
            "last_processed_utc": None,
            "last_duration_ms": None,
            "result": None,
        }

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def snapshot(self, include_images: bool = False) -> dict[str, Any]:
        with self.lock:
            data = self.state.copy()
            result = self.state.get("result")
            if isinstance(result, dict):
                if include_images:
                    data["result"] = {
                        key: value
                        for key, value in result.items()
                        if key != "_recording_frame"
                    }
                else:
                    data["result"] = {
                        "frame_index": result.get("frame_index"),
                        "frame_utc": result.get("frame_utc"),
                        "processed_utc": result.get("processed_utc"),
                        "count": result.get("count"),
                        "measurement": result.get("measurement"),
                        "front_y_ratio": result.get("front_y_ratio"),
                        "conf": result.get("conf"),
                    }
        return data

    def recording_frame(self) -> dict[str, Any] | None:
        with self.lock:
            result = self.state.get("result")
            if not isinstance(result, dict):
                return None
            item = result.get("_recording_frame")
            return item.copy() if isinstance(item, dict) else None

    def _set_state(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates)

    def _run(self) -> None:
        last_processed_index = -1
        delay = 1.0 / max(0.2, float(self.args.process_fps))
        while not self.stop_event.is_set():
            started_wait = time.perf_counter()
            item = self.buffer.latest()
            if item is None or int(item["index"]) == last_processed_index:
                time.sleep(0.03)
                continue

            self._set_state(processing=True)
            started = time.perf_counter()
            try:
                result = self._process(item)
                last_processed_index = int(item["index"])
                duration_ms = round((time.perf_counter() - started) * 1000.0, 1)
                self._set_state(
                    ok=True,
                    processing=False,
                    error="",
                    processed_count=int(self.state.get("processed_count", 0)) + 1,
                    last_frame_index=last_processed_index,
                    last_processed_utc=utc_now(),
                    last_duration_ms=duration_ms,
                    result=result,
                )
            except Exception as exc:
                self._set_state(ok=False, processing=False, error=str(exc))
                time.sleep(0.5)

            elapsed = time.perf_counter() - started_wait
            if elapsed < delay:
                time.sleep(delay - elapsed)

    def _process(self, item: dict[str, Any]) -> dict[str, Any]:
        original = item["frame"]
        matrix, out_size, _homography = vision.load_homography()
        rectified = cv2.warpPerspective(original, matrix, out_size)
        calibration = vision.load_measurement_calibration()
        boxes, box_rules = vision.predict_yolo_boxes_with_rules(
            rectified,
            conf=float(self.args.conf),
            imgsz=int(self.args.imgsz),
            exclusion_zones=calibration.get("exclusion_zones"),
            exclusion_max_overlap=float(
                calibration.get(
                    "exclusion_max_box_overlap",
                    vision.EXCLUSION_ZONE_MAX_BOX_OVERLAP,
                )
            ),
        )

        rect_h, rect_w = rectified.shape[:2]
        src_h, src_w = original.shape[:2]
        pieces, measurement_summary = vision.analyze_piece_boxes(
            rectified,
            boxes,
            calibration,
            frame_idx=int(item["index"]),
            time_sec=None,
        )
        primary = vision.primary_piece_analysis(pieces)
        sobel = primary["sobel"] if primary else vision.empty_sobel_result(int(item["index"]), None)
        measurement = primary["measurement"] if primary else None
        original_overlay = vision.mvp_original_overlay_for_pieces(pieces, calibration, matrix, rect_w)
        original_viz = draw_original_overlay(original, original_overlay)
        rectified_viz = draw_rectified_overlay(rectified, pieces, calibration)

        return {
            "frame_index": int(item["index"]),
            "frame_utc": item["utc"],
            "frame_monotonic": float(item["monotonic"]),
            "processed_utc": utc_now(),
            "original_width": src_w,
            "original_height": src_h,
            "rectified_width": rect_w,
            "rectified_height": rect_h,
            "original_image": img_to_b64(original_viz, quality=80),
            "rectified_image": img_to_b64(rectified_viz, quality=82),
            "boxes": boxes,
            "box_rules": box_rules,
            "count": len(boxes),
            "pieces": pieces,
            "measurement_summary": measurement_summary,
            "sobel": sobel,
            "calibration": calibration,
            "measurement": measurement,
            "front_y_ratio": (float(sobel["line"]["y"]) / float(rect_h)) if sobel.get("line") else None,
            "piece_front_y_ratios": [
                float(piece["sobel"]["line"]["y"]) / float(rect_h)
                for piece in pieces
                if (piece.get("sobel") or {}).get("line")
            ],
            "model": str(self.args.model),
            "conf": float(self.args.conf),
            "_recording_frame": {
                "index": int(item["index"]),
                "utc": item["utc"],
                "monotonic": float(item["monotonic"]),
                "frame": original_viz,
            },
        }


class ClipRecorder:
    def __init__(
        self,
        args: argparse.Namespace,
        buffer: FrameBuffer,
        processor: LiveProcessor | None = None,
        database: DatabaseRepository | SQLiteDatabaseRepository | None = None,
    ) -> None:
        self.args = args
        self.buffer = buffer
        self.processor = processor
        self.database = database
        self.lock = threading.Lock()
        self.configuration_lock = threading.Lock()
        self.vision_configuration: dict[str, Any] | None = None
        self.active_recordings: set[int] = set()
        self.clip_index = 0
        self.last_clip: dict[str, Any] | None = None
        self.discarded_clip_count = 0
        self.last_discarded_clip: dict[str, Any] | None = None
        self.failed_recordings: deque[dict[str, Any]] = deque(maxlen=20)
        self.error = ""

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "recording": bool(self.active_recordings),
                "active_recordings": len(self.active_recordings),
                "clip_index": self.clip_index,
                "last_clip": self.last_clip,
                "discarded_clip_count": self.discarded_clip_count,
                "last_discarded_clip": self.last_discarded_clip,
                "failed_recording_count": len(self.failed_recordings),
                "failed_recordings": [failure.copy() for failure in self.failed_recordings],
                "error": self.error,
                "record_seconds": self.args.record_seconds,
                "database_enabled": self.database is not None,
            }

    def start_event_clip(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.clip_index += 1
            clip_index = self.clip_index
            self.active_recordings.add(clip_index)
        thread = threading.Thread(target=self._record_clip, args=(clip_index, event), daemon=True)
        thread.start()

    def _latest_recording_frame(self) -> dict[str, Any] | None:
        if self.processor is not None:
            return self.processor.recording_frame()
        return self.buffer.latest()

    def _configuration_snapshot(self) -> dict[str, Any] | None:
        with self.configuration_lock:
            if self.vision_configuration is not None:
                return self.vision_configuration
            try:
                self.vision_configuration = build_vision_configuration(self.args, ROOT)
            except (AttributeError, FileNotFoundError, ValueError):
                if self.database is not None:
                    raise
                return None
            return self.vision_configuration

    def validate_configuration(self) -> dict[str, Any]:
        configuration = self._configuration_snapshot()
        if configuration is None:
            raise RuntimeError("The vision configuration could not be snapshotted")
        return configuration

    def _camera_source(self) -> str:
        source = str(getattr(self.args, "source", "unknown"))
        if source == "rtsp":
            return f"rtsp://{getattr(self.args, 'camera_ip', 'camera')}"
        if source == "video":
            return str(getattr(self.args, "video", "video"))
        return source

    def _capture_processing_snapshot(
        self,
        analysis_dir: Path,
        snapshots: list[dict[str, Any]],
        seen_frame_indices: set[int],
        start_monotonic: float,
        end_monotonic: float,
    ) -> None:
        if self.processor is None:
            return
        processor_data = self.processor.snapshot(include_images=True)
        result = processor_data.get("result")
        if not isinstance(result, dict):
            return
        frame_index = result.get("frame_index")
        if frame_index is None:
            return
        frame_monotonic = float(result.get("frame_monotonic", 0.0))
        if frame_monotonic < start_monotonic or frame_monotonic >= end_monotonic:
            return
        frame_index = int(frame_index)
        if frame_index in seen_frame_indices:
            return
        seen_frame_indices.add(frame_index)

        snap_index = len(snapshots)
        snapshot: dict[str, Any] = {
            key: clean_value(value)
            for key, value in result.items()
            if key not in ("original_image", "rectified_image")
        }
        snapshot["snapshot_index"] = snap_index
        snapshot["processing_duration_ms"] = clean_value(processor_data.get("last_duration_ms"))

        for image_key, suffix in (("original_image", "original_overlay"), ("rectified_image", "rectified_overlay")):
            image_b64 = result.get(image_key)
            if not image_b64:
                continue
            image_name = f"analysis_{snap_index:03d}_{suffix}.jpg"
            image_path = analysis_dir / image_name
            image_path.write_bytes(base64.b64decode(str(image_b64)))
            snapshot[f"{suffix}_path"] = str(image_path)
            snapshot[f"{suffix}_file"] = image_name

        snapshots.append(snapshot)

    def _record_clip(self, clip_index: int, event: dict[str, Any]) -> None:
        processing_snapshots: list[dict[str, Any]] = []
        seen_processing_frames: set[int] = set()
        event_mono = float(event.get("event_read_monotonic") or time.perf_counter())
        event_key = build_event_key(
            event,
            str(getattr(self.args, "plc_endpoint", "")),
            str(getattr(self.args, "event_node", "")),
        )
        measurement_event_id = event_uuid(event_key)
        vision_configuration: dict[str, Any] | None = None
        db_sync_status = "disabled" if self.database is None else "pending"
        db_sync_error = ""
        try:
            vision_configuration = self._configuration_snapshot()
            if self.database is not None and vision_configuration is not None:
                measurement_event_id = self.database.create_measurement_event(
                    event_id=measurement_event_id,
                    event_key=event_key,
                    event=event,
                    configuration=vision_configuration,
                    plc_endpoint=str(getattr(self.args, "plc_endpoint", "")),
                    plc_event_node=str(getattr(self.args, "event_node", "")),
                    plc_watchdog_node=str(getattr(self.args, "watchdog_node", "")),
                    camera_source=self._camera_source(),
                )
                db_sync_status = "recording"
        except Exception as exc:
            db_sync_status = "pending"
            db_sync_error = str(exc)
        record_seconds = max(0.1, float(self.args.record_seconds))
        deadline = event_mono + record_seconds
        fps = max(1.0, min(float(self.args.record_fps or self.args.capture_fps or 10.0), 60.0))
        target_frame_count = max(1, int(round(record_seconds * fps)))
        frames_written = 0
        source_frames_seen = 0
        last_index = -1
        last_source: dict[str, Any] | None = None
        first_written_source: dict[str, Any] | None = None
        last_written_source: dict[str, Any] | None = None
        writer: cv2.VideoWriter | None = None
        video_path: Path | None = None
        analysis_dir: Path | None = None
        json_path: Path | None = None

        try:
            day_dir = self.args.output_dir / "live_plc_clips" / datetime.now().strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            edge = event.get("event_edge") or "event"
            base = f"live_{clip_index:04d}_{file_stamp()}_{edge}"
            analysis_dir = day_dir / f"{base}_analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            video_path = day_dir / f"{base}.mp4"
            json_path = day_dir / f"{base}.json"

            while time.perf_counter() < deadline:
                item = self._latest_recording_frame()
                if item is not None and int(item["index"]) > last_index:
                    last_index = int(item["index"])
                    item_mono = float(item["monotonic"])
                    if item_mono >= event_mono:
                        source_frames_seen += 1
                        if writer is None:
                            height, width = item["frame"].shape[:2]
                            writer_args = (
                                str(video_path),
                                cv2.VideoWriter_fourcc(*"avc1"),
                                fps,
                                (width, height),
                            )
                            if os.name == "nt":
                                writer = cv2.VideoWriter(
                                    writer_args[0],
                                    cv2.CAP_MSMF,
                                    *writer_args[1:],
                                )
                            else:
                                writer = cv2.VideoWriter(*writer_args)
                            if not writer.isOpened():
                                raise RuntimeError(f"Could not open VideoWriter: {video_path}")
                        if last_source is None:
                            last_source = item

                        while frames_written < target_frame_count:
                            sample_mono = event_mono + (frames_written / fps)
                            if sample_mono > item_mono:
                                break
                            writer.write(last_source["frame"])
                            first_written_source = first_written_source or last_source
                            last_written_source = last_source
                            frames_written += 1
                        last_source = item
                self._capture_processing_snapshot(
                    analysis_dir,
                    processing_snapshots,
                    seen_processing_frames,
                    event_mono,
                    deadline,
                )
                time.sleep(0.025)
            self._capture_processing_snapshot(
                analysis_dir,
                processing_snapshots,
                seen_processing_frames,
                event_mono,
                deadline,
            )

            if writer is None or last_source is None:
                raise RuntimeError("No frames were available to record the clip.")

            while frames_written < target_frame_count:
                writer.write(last_source["frame"])
                first_written_source = first_written_source or last_source
                last_written_source = last_source
                frames_written += 1
            writer.release()
            writer = None

            if first_written_source is None or last_written_source is None:
                raise RuntimeError("No frames were written to the clip.")

            if not snapshots_contain_piece(processing_snapshots):
                database_cleanup_error = ""
                if self.database is not None:
                    try:
                        self.database.delete_measurement_event(measurement_event_id)
                    except Exception as exc:
                        database_cleanup_error = str(exc)
                artifact_cleanup_errors = delete_clip_paths(
                    self.args.output_dir,
                    [video_path, analysis_dir, json_path],
                )
                discarded = {
                    "clip_index": clip_index,
                    "discarded_at": utc_now(),
                    "event_id": measurement_event_id,
                    "event": clean_value(event),
                    "reason": "no_piece_detected",
                    "processing_snapshot_count": len(processing_snapshots),
                    "database_cleanup_error": database_cleanup_error,
                    "artifact_cleanup_errors": artifact_cleanup_errors,
                }
                with self.lock:
                    self.discarded_clip_count += 1
                    self.last_discarded_clip = discarded
                    cleanup_errors = [
                        error
                        for error in [database_cleanup_error, *artifact_cleanup_errors]
                        if error
                    ]
                    if cleanup_errors:
                        self.error = (
                            "The empty clip was discarded with cleanup errors: "
                            + "; ".join(cleanup_errors)
                        )
                return

            canonical_snapshot = select_canonical_snapshot(
                processing_snapshots,
                event_monotonic=event_mono,
            )
            sidecar = {
                "clip_index": clip_index,
                "saved_at": utc_now(),
                "event_id": measurement_event_id,
                "event_key": event_key,
                "event": event,
                "plc_endpoint": str(getattr(self.args, "plc_endpoint", "")),
                "plc_event_node": str(getattr(self.args, "event_node", "")),
                "plc_watchdog_node": str(getattr(self.args, "watchdog_node", "")),
                "camera_source": self._camera_source(),
                "vision_configuration": vision_configuration,
                "record_seconds": record_seconds,
                "video_fps": fps,
                "video_codec": "h264",
                "video_content": "yolo_processed_overlay" if self.processor is not None else "raw_fallback",
                "video_duration_seconds": frames_written / fps,
                "frames_captured": source_frames_seen,
                "frames_written": frames_written,
                "source_frames_seen": source_frames_seen,
                "first_frame_utc": first_written_source["utc"],
                "last_frame_utc": last_written_source["utc"],
                "first_frame_index": int(first_written_source["index"]),
                "last_frame_index": int(last_written_source["index"]),
                "video_path": str(video_path),
                "analysis_dir": str(analysis_dir),
                "processing_snapshots": processing_snapshots,
                "processing_snapshot_count": len(processing_snapshots),
                "canonical_snapshot_frame_index": (
                    int(canonical_snapshot["frame_index"])
                    if canonical_snapshot is not None
                    else None
                ),
                "db_sync_status": db_sync_status,
                "db_sync_backend": None,
                "db_sync_error": db_sync_error,
                "db_sync_attempts": 0,
                "db_sync_last_attempt_at": None,
            }
            write_json_atomic(json_path, sidecar)
            if self.database is not None:
                sidecar["db_sync_attempts"] = 1
                sidecar["db_sync_last_attempt_at"] = utc_now()
                sidecar["db_sync_status"] = "pending"
                sidecar["db_sync_error"] = ""
                write_json_atomic(json_path, sidecar)
                try:
                    self.database.mark_measurement_event_processing(
                        measurement_event_id,
                        str(sidecar["last_frame_utc"]),
                    )
                    actual_event_id = self.database.sync_sidecar(json_path, self.args.output_dir)
                    if actual_event_id != sidecar["event_id"]:
                        sidecar["event_id"] = actual_event_id
                        write_json_atomic(json_path, sidecar)
                        self.database.sync_sidecar(json_path, self.args.output_dir)
                    sidecar["db_sync_status"] = "synced"
                    sidecar["db_sync_backend"] = self.database.backend_name
                    write_json_atomic(json_path, sidecar)
                except Exception as exc:
                    sidecar["db_sync_status"] = "pending"
                    sidecar["db_sync_error"] = str(exc)
                    write_json_atomic(json_path, sidecar)
            self._enforce_retention()
            with self.lock:
                self.last_clip = sidecar
        except Exception as exc:
            if writer is not None:
                writer.release()
                writer = None
            delete_clip_paths(
                self.args.output_dir,
                [video_path, analysis_dir, json_path],
            )
            failure = {
                "clip_index": clip_index,
                "failed_at": utc_now(),
                "event": clean_value(event),
                "error": str(exc),
            }
            if self.database is not None:
                try:
                    self.database.mark_measurement_event_failed(
                        measurement_event_id,
                        failure["error"],
                    )
                except Exception:
                    pass
            with self.lock:
                self.error = failure["error"]
                self.failed_recordings.append(failure)
        finally:
            if writer is not None:
                writer.release()
            with self.lock:
                self.active_recordings.discard(clip_index)

    def _enforce_retention(self) -> None:
        max_clips = max(1, int(self.args.max_clips or 100))
        sidecars = clip_sidecars(self.args.output_dir)
        for json_path in sidecars[max_clips:]:
            if self.database is not None:
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    event_id = data.get("event_id")
                    if event_id:
                        self.database.delete_measurement_event(str(event_id))
                except Exception as exc:
                    with self.lock:
                        self.error = f"Retention deferred for {json_path.name}: {exc}"
                    continue
            delete_clip_artifacts(self.args.output_dir, json_path)


class DatabaseReconciler:
    def __init__(
        self,
        args: argparse.Namespace,
        database: DatabaseRepository | SQLiteDatabaseRepository,
    ) -> None:
        self.args = args
        self.database = database
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="database-sidecar-reconciler",
            daemon=True,
        )
        self.lock = threading.Lock()
        self.vision_configuration = build_vision_configuration(args, ROOT)
        self.state: dict[str, Any] = {
            "running": False,
            "pending": 0,
            "synced": 0,
            "failed": 0,
            "pruned": 0,
            "last_run_utc": None,
            "last_error": "",
        }

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self.state.copy()

    def _set_state(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates)

    def _enrich_sidecar(self, data: dict[str, Any]) -> dict[str, Any]:
        event = data.get("event") if isinstance(data.get("event"), dict) else {}
        plc_endpoint = str(data.get("plc_endpoint") or self.args.plc_endpoint)
        plc_event_node = str(data.get("plc_event_node") or self.args.event_node)
        event_key = str(
            data.get("event_key")
            or build_event_key(event, plc_endpoint, plc_event_node)
        )
        data.setdefault("event_key", event_key)
        data.setdefault("event_id", event_uuid(event_key))
        data.setdefault("vision_configuration", self.vision_configuration)
        data.setdefault("plc_endpoint", plc_endpoint)
        data.setdefault("plc_event_node", plc_event_node)
        data.setdefault("plc_watchdog_node", str(self.args.watchdog_node))
        data.setdefault("camera_source", "legacy_live_mvp")
        return data

    def sync_once(self) -> None:
        pending_paths = []
        backend = self.database.backend_name
        for path in clip_sidecars(self.args.output_dir):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if data and (
                data.get("db_sync_status") == "pending"
                or data.get("db_sync_backend") != backend
            ):
                pending_paths.append(path)
        self._set_state(running=True, pending=len(pending_paths), last_run_utc=utc_now())
        synced = 0
        failed = 0
        last_error = ""
        for path in pending_paths:
            if self.stop_event.is_set():
                break
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = None
            if not data:
                continue
            data = self._enrich_sidecar(data)
            data["db_sync_attempts"] = int(data.get("db_sync_attempts") or 0) + 1
            data["db_sync_last_attempt_at"] = utc_now()
            data["db_sync_status"] = "pending"
            data["db_sync_error"] = ""
            write_json_atomic(path, data)
            try:
                actual_event_id = self.database.sync_sidecar(path, self.args.output_dir)
                if actual_event_id != data.get("event_id"):
                    data["event_id"] = actual_event_id
                    write_json_atomic(path, data)
                    self.database.sync_sidecar(path, self.args.output_dir)
                data["db_sync_status"] = "synced"
                data["db_sync_backend"] = backend
                data["db_sync_error"] = ""
                write_json_atomic(path, data)
                synced += 1
            except Exception as exc:
                data["db_sync_status"] = "pending"
                data["db_sync_error"] = str(exc)
                last_error = str(exc)
                failed += 1
                write_json_atomic(path, data)
        pruned = 0
        all_event_ids = getattr(
            self.database,
            "all_measurement_event_ids",
            None,
        )
        if callable(all_event_ids):
            retained_event_ids: set[str] = set()
            can_prune = True
            for path in clip_sidecars(self.args.output_dir):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    can_prune = False
                    break
                event_id = data.get("event_id")
                if event_id:
                    retained_event_ids.add(str(event_id))
            if can_prune:
                for stale_event_id in all_event_ids() - retained_event_ids:
                    self.database.delete_measurement_event(stale_event_id)
                    pruned += 1
        self._set_state(
            running=False,
            pending=max(0, len(pending_paths) - synced),
            synced=int(self.state.get("synced") or 0) + synced,
            failed=int(self.state.get("failed") or 0) + failed,
            pruned=int(self.state.get("pruned") or 0) + pruned,
            last_error=last_error,
        )

    def _run(self) -> None:
        delay = max(2.0, float(getattr(self.args, "db_retry_seconds", 15.0)))
        while not self.stop_event.is_set():
            try:
                self.sync_once()
            except Exception as exc:
                self._set_state(running=False, last_error=str(exc), last_run_utc=utc_now())
            self.stop_event.wait(delay)


class PLCMonitor:
    def __init__(self, args: argparse.Namespace, recorder: ClipRecorder) -> None:
        self.args = args
        self.recorder = recorder
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="live-plc-monitor", daemon=True)
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "enabled": bool(args.plc_enabled),
            "connected": False,
            "error": "",
            "endpoint": args.plc_endpoint,
            "watchdog_node": args.watchdog_node,
            "event_node": args.event_node,
            "watchdog_ticks": 0,
            "events_found": 0,
            "last_watchdog": None,
            "last_event": None,
            "last_trigger": None,
            "last_read_utc": None,
        }

    def start(self) -> None:
        if self.args.plc_enabled:
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self.state.copy()

    def _set_state(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                asyncio.run(self._monitor_once())
            except Exception as exc:
                self._set_state(connected=False, error=str(exc))
                time.sleep(2.0)

    async def _read_node(self, client: Any, node_id: str) -> dict[str, Any]:
        node = client.get_node(node_id)
        data_value = await node.read_data_value()
        return {
            "node_id": node_id,
            "status": str(data_value.StatusCode),
            "value": clean_value(data_value.Value.Value),
            "source_timestamp": clean_value(data_value.SourceTimestamp),
            "server_timestamp": clean_value(data_value.ServerTimestamp),
            "read_utc": utc_now(),
            "read_monotonic": time.perf_counter(),
        }

    async def _monitor_once(self) -> None:
        from asyncua import Client

        previous_watchdog: Any = None
        previous_event: Any = None
        async with Client(url=self.args.plc_endpoint, timeout=self.args.plc_timeout) as client:
            self._set_state(connected=True, error="")
            while not self.stop_event.is_set():
                watchdog = await self._read_node(client, self.args.watchdog_node)
                watchdog_value = watchdog.get("value")
                if watchdog_value != previous_watchdog:
                    event = await self._read_node(client, self.args.event_node)
                    edge = event_edge(previous_event, event.get("value"))
                    row = {
                        "read_utc": utc_now(),
                        "watchdog_value": watchdog_value,
                        "watchdog_source_timestamp": watchdog.get("source_timestamp"),
                        "watchdog_status": watchdog.get("status"),
                        "event_value": event.get("value"),
                        "previous_event_value": previous_event,
                        "event_source_timestamp": event.get("source_timestamp"),
                        "event_server_timestamp": event.get("server_timestamp"),
                        "event_read_monotonic": event.get("read_monotonic"),
                        "event_status": event.get("status"),
                        "event_edge": edge,
                    }
                    updates = {
                        "connected": True,
                        "error": "",
                        "watchdog_ticks": int(self.state.get("watchdog_ticks", 0)) + 1,
                        "last_watchdog": watchdog,
                        "last_event": event,
                        "last_read_utc": row["read_utc"],
                    }
                    if edge_matches(self.args.plc_edge, edge):
                        updates["events_found"] = int(self.state.get("events_found", 0)) + 1
                        updates["last_trigger"] = row
                        self.recorder.start_event_clip(row)
                    self._set_state(**updates)
                    previous_watchdog = watchdog_value
                    previous_event = event.get("value")
                await asyncio.sleep(float(self.args.plc_poll_interval))


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TX2 Live MVP</title>
<style>
:root {
  color: #172025;
  background: #f5f7f8;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; min-width: 320px; min-height: 100vh; background: #f5f7f8; }
.app { width: min(1760px, calc(100vw - 24px)); margin: 0 auto; padding: 14px 0 18px; }
.topbar { display: flex; align-items: center; justify-content: flex-end; gap: 14px; margin-bottom: 12px; }
h1, h2, p { margin: 0; letter-spacing: 0; }
h1 { font-size: 27px; line-height: 1.05; }
h2 { font-size: 16px; }
.eyebrow { color: #68787f; font-size: 12px; font-weight: 800; text-transform: uppercase; margin-bottom: 4px; }
.pill { border: 1px solid #cbd5da; border-radius: 999px; padding: 8px 12px; color: #485960; background: #ffffff; font-weight: 800; white-space: nowrap; }
.pill.ok { border-color: #58a680; color: #14784f; background: #ecfff5; }
.pill.warn { border-color: #d7a34d; color: #8a5a0a; background: #fff7e5; }
.pill.err { border-color: #d48282; color: #a23232; background: #fff0f0; }
.nav { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
.nav a { border: 1px solid #cbd5da; border-radius: 8px; padding: 9px 12px; color: #172025; background: #ffffff; text-decoration: none; font-weight: 900; }
.plc-signal { min-height: 36px; display: inline-flex; align-items: center; gap: 8px; border: 1px solid #cbd5da; border-radius: 8px; padding: 7px 10px; color: #586970; background: #ffffff; font-size: 13px; font-weight: 800; white-space: nowrap; }
.plc-signal .dot { width: 9px; height: 9px; border-radius: 50%; background: #9aa8ae; box-shadow: 0 0 0 3px rgba(154,168,174,.16); }
.plc-signal.seen { border-color: #84bda3; color: #14784f; background: #f4fff9; }
.plc-signal.seen .dot { background: #229966; box-shadow: 0 0 0 3px rgba(34,153,102,.16); }
.plc-signal.received { border-color: #229966; color: #0f6945; background: #e7fff2; }
.plc-signal.received .dot { background: #16a366; animation: plc-pulse .8s ease-out 2; }
.plc-signal.offline { border-color: #d7a34d; color: #8a5a0a; background: #fff7e5; }
.plc-signal.offline .dot { background: #d49a36; box-shadow: 0 0 0 3px rgba(212,154,54,.16); }
@keyframes plc-pulse { 0% { box-shadow: 0 0 0 0 rgba(22,163,102,.42); } 100% { box-shadow: 0 0 0 9px rgba(22,163,102,0); } }
.grid { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(360px, .75fr); gap: 12px; align-items: start; }
.panel { border: 1px solid #d8e0e4; border-radius: 8px; background: #ffffff; overflow: hidden; box-shadow: 0 14px 28px rgba(23,32,37,.08); }
.panel-head { min-height: 48px; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid #d8e0e4; }
.stage { display: grid; place-items: center; min-height: 300px; background: #eef2f4; }
.stage img { display: block; width: 100%; height: auto; max-height: calc(100vh - 260px); object-fit: contain; }
.side { display: grid; gap: 12px; }
.diagram { width: 100%; height: auto; display: block; background: #f7faf8; }
.log { padding: 12px; display: grid; gap: 8px; color: #506168; font-size: 13px; }
.row { display: flex; justify-content: space-between; gap: 14px; border-bottom: 1px solid rgba(23,32,37,.08); padding-bottom: 7px; }
.row:last-child { border-bottom: 0; padding-bottom: 0; }
.row strong { color: #172025; text-align: right; overflow-wrap: anywhere; }
.empty { color: #68787f; font-weight: 800; padding: 44px 12px; text-align: center; }
@media (max-width: 1150px) {
  .grid { grid-template-columns: 1fr; }
}
@media (max-width: 680px) {
  .topbar { align-items: start; flex-direction: column; }
}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="nav">
      <a href="/history">History</a>
      <div id="plc-signal" class="plc-signal offline" role="status" aria-live="polite">
        <span class="dot" aria-hidden="true"></span>
        <span id="plc-signal-text">Connecting to PLC...</span>
      </div>
      <div id="top-state" class="pill warn">connecting...</div>
    </div>
  </header>

  <main class="grid">
    <section class="panel">
      <div class="stage" id="original-stage"><div class="empty">Waiting for frame...</div></div>
    </section>

    <section class="panel">
      <svg class="diagram" viewBox="0 0 760 310" role="img" aria-label="Measurement diagram">
        <defs>
          <linearGradient id="live-steel" x1="0" x2="1">
            <stop offset="0%" stop-color="#69777c" />
            <stop offset="42%" stop-color="#d6dcde" />
            <stop offset="72%" stop-color="#929da1" />
            <stop offset="100%" stop-color="#5a666b" />
          </linearGradient>
        </defs>
        <rect x="38" y="34" width="684" height="232" rx="8" fill="#f7faf8" stroke="#cbd6cf" stroke-width="2" />
        <rect x="86" y="58" width="588" height="182" rx="6" fill="#edf2ef" stroke="#c0cbc6" />
        <line id="diagram-reference" x1="92" x2="668" y1="156" y2="156" stroke="#d28230" stroke-width="4" stroke-linecap="round" stroke-dasharray="10 8" />
        <text id="diagram-reference-label" x="100" y="180" fill="#a66324" font-size="16" font-weight="900">REFERENCE</text>
        <g id="diagram-pieces"></g>
        <text id="diagram-summary" x="380" y="286" text-anchor="middle" fill="#243c48" font-size="17" font-weight="900">Waiting for pieces</text>
      </svg>
    </section>
  </main>
</div>

<script>
const $ = (id) => document.getElementById(id);
let lastPlcEventCount = null;
let plcSignalHighlightUntil = 0;

function setImage(stage, b64, alt) {
  if (!b64) {
    stage.innerHTML = '<div class="empty">Waiting for frame...</div>';
    return;
  }
  let img = stage.querySelector('img');
  if (!img) {
    stage.innerHTML = '';
    img = document.createElement('img');
    img.alt = alt;
    stage.appendChild(img);
  }
  img.src = `data:image/jpeg;base64,${b64}`;
}

function pill(el, text, tone) {
  el.textContent = text;
  el.className = `pill ${tone || ''}`.trim();
}

function compactMeasurement(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '-';
  let total = Math.round(Math.abs(numeric) * 16);
  const sign = numeric < 0 ? '-' : '';
  const feet = Math.floor(total / 192);
  total -= feet * 192;
  const inches = Math.floor(total / 16);
  let numerator = total % 16;
  let denominator = 16;
  while (numerator && numerator % 2 === 0) {
    numerator /= 2;
    denominator /= 2;
  }
  const inchText = numerator ? `${inches} ${numerator}/${denominator}` : `${inches}`;
  return `${sign}${feet}' ${inchText}"`;
}

function updateDiagram(result) {
  const pieces = Array.isArray(result?.pieces) ? result.pieces : [];
  const rectWidth = Math.max(1, Number(result?.rectified_width || 1));
  const rectHeight = Math.max(1, Number(result?.rectified_height || 1));
  const mapY = (value) => Math.max(66, Math.min(232, 58 + (Number(value) / rectHeight) * 182));
  const referenceValue = result?.calibration?.reference_y;
  const referenceY = Number.isFinite(Number(referenceValue)) ? mapY(referenceValue) : 156;
  $('diagram-reference').setAttribute('y1', referenceY);
  $('diagram-reference').setAttribute('y2', referenceY);
  $('diagram-reference-label').setAttribute('y', Math.min(252, referenceY + 22));

  $('diagram-pieces').innerHTML = pieces.map((piece, index) => {
    const box = piece.box || {};
    const lineY = piece?.sobel?.line?.y;
    if (!Number.isFinite(Number(lineY))) return '';
    const centerRatio = (Number(box.x || 0) + Number(box.w || 0) / 2) / rectWidth;
    const x = 92 + Math.max(0, Math.min(1, centerRatio)) * 576;
    const width = Math.max(12, Math.min(34, (Number(box.w || 0) / rectWidth) * 576));
    const frontY = mapY(lineY);
    const valid = Boolean(piece.valid);
    const color = valid ? '#16845a' : '#c5782d';
    const measurement = piece.measurement?.measurement_in;
    const label = compactMeasurement(measurement);
    return `
      <g>
        <rect x="${x - width / 2}" y="68" width="${width}" height="${Math.max(8, frontY - 68)}" rx="3"
          fill="url(#live-steel)" stroke="#536066" stroke-width="1" />
        <line x1="${x - width / 2 - 2}" x2="${x + width / 2 + 2}" y1="${frontY}" y2="${frontY}"
          stroke="${color}" stroke-width="5" stroke-linecap="round" />
        <text x="${x}" y="${Math.min(254, frontY + 17 + (index % 2) * 14)}" text-anchor="middle"
          fill="${color}" font-size="11" font-weight="900">P${Number(piece.piece_id)} ${label}</text>
      </g>`;
  }).join('');

  const summary = result?.measurement_summary || {};
  $('diagram-summary').textContent = pieces.length
    ? `${Number(summary.valid_count || 0)} / ${pieces.length} valid piece measurements`
    : 'No pieces detected';
}

function signalTime(trigger) {
  const raw = trigger?.event_source_timestamp || trigger?.read_utc;
  if (!raw) return '';
  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime()) ? String(raw) : parsed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function updatePlcSignal(plc) {
  const signal = $('plc-signal');
  const text = $('plc-signal-text');
  const count = Number(plc.events_found || 0);
  const trigger = plc.last_trigger;

  if (!plc.connected) {
    plcSignalHighlightUntil = 0;
    signal.className = 'plc-signal offline';
    text.textContent = 'PLC disconnected';
    signal.title = '';
  } else if (!trigger) {
    plcSignalHighlightUntil = 0;
    signal.className = 'plc-signal';
    text.textContent = 'Waiting for PLC signal';
    signal.title = '';
  } else {
    const receivedNow = lastPlcEventCount !== null && count > lastPlcEventCount;
    if (receivedNow) plcSignalHighlightUntil = Date.now() + 4000;
    const highlighting = Date.now() < plcSignalHighlightUntil;
    const time = signalTime(trigger);
    signal.className = `plc-signal ${highlighting ? 'received' : 'seen'}`;
    text.textContent = `${highlighting ? 'PLC signal received' : 'Last PLC signal'}${time ? ` | ${time}` : ''}`;
    signal.title = trigger.event_source_timestamp || trigger.read_utc || '';
  }
  lastPlcEventCount = count;
}

async function refreshFrame() {
  try {
    const response = await fetch('/api/live/frame');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'frame error');
    const result = data.result;
    if (!result) return;
    setImage($('original-stage'), result.original_image, 'Live camera');
    updateDiagram(result);
  } catch (err) {
    pill($('top-state'), 'frame error', 'err');
  }
}

async function refreshStatus() {
  try {
    const response = await fetch('/api/live/status');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'status error');
    const camera = data.camera || {};
    const processor = data.processor || {};
    const database = data.database || {};
    updatePlcSignal(data.plc || {});
    const healthy = camera.connected && processor.ok && (!database.enabled || database.ok);
    const label = database.enabled ? (healthy ? 'live' : 'check status') : 'simulation';
    pill($('top-state'), label, healthy && database.enabled ? 'ok' : 'warn');
  } catch (err) {
    pill($('top-state'), 'error', 'err');
  }
}

setInterval(refreshFrame, 100);
setInterval(refreshStatus, 1000);
refreshFrame();
refreshStatus();
</script>
</body>
</html>
"""


HISTORY_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TX2 Measurement History</title>
<style>
:root { color: #172025; background: #f5f7f8; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; min-width: 320px; min-height: 100vh; background: #f5f7f8; }
button, input, textarea { font: inherit; }
.app { width: min(1580px, calc(100vw - 24px)); margin: 0 auto; padding: 14px 0 22px; }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 12px; }
h1, h2, h3, p { margin: 0; letter-spacing: 0; }
h1 { font-size: 25px; line-height: 1.1; }
h2 { font-size: 16px; }
.nav a, .btn { border: 1px solid #cbd5da; border-radius: 8px; padding: 9px 12px; color: #172025; background: #fff; text-decoration: none; font-weight: 800; cursor: pointer; }
.btn.primary { border-color: #247654; background: #247654; color: #fff; }
.btn.danger { border-color: #c77a7a; color: #9d2d2d; }
.mode { display: none; margin-bottom: 12px; border: 1px solid #d7a34d; border-radius: 8px; padding: 10px 12px; color: #7f550e; background: #fff8e8; font-weight: 750; overflow-wrap: anywhere; }
.grid { display: grid; grid-template-columns: 360px minmax(0, 1fr); gap: 12px; align-items: start; }
.panel { min-width: 0; border: 1px solid #d8e0e4; border-radius: 8px; background: #fff; overflow: hidden; box-shadow: 0 10px 24px rgba(23,32,37,.07); }
.panel-head { min-height: 48px; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid #d8e0e4; }
.muted { color: #68787f; font-size: 12px; font-weight: 750; }
.list { display: grid; max-height: calc(100vh - 120px); overflow: auto; }
.event { display: grid; gap: 5px; padding: 12px; color: #2c3a40; background: transparent; border: 0; border-bottom: 1px solid rgba(23,32,37,.08); text-align: left; cursor: pointer; }
.event:hover, .event.active { background: #edf5f8; }
.event strong { color: #172025; overflow-wrap: anywhere; }
.event span { color: #68787f; font-size: 12px; font-weight: 750; }
.viewer { padding: 12px; display: grid; gap: 14px; }
video, img { display: block; width: 100%; border-radius: 6px; background: #eef2f4; }
.meta { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid #d8e0e4; border-radius: 8px; overflow: hidden; }
.metric { padding: 10px; min-height: 62px; border-right: 1px solid #d8e0e4; }
.metric:last-child { border-right: 0; }
.metric span { display: block; color: #68787f; font-size: 12px; font-weight: 750; }
.metric strong { display: block; margin-top: 5px; color: #172025; font-size: 17px; overflow-wrap: anywhere; }
.section-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.pieces { width: 100%; border-collapse: collapse; border: 1px solid #d8e0e4; }
.pieces th, .pieces td { padding: 9px 10px; border-bottom: 1px solid #e2e8eb; text-align: left; vertical-align: middle; }
.pieces th { color: #5b6b72; background: #f6f8f9; font-size: 12px; }
.pieces td { font-size: 13px; }
.pieces tr:last-child td { border-bottom: 0; }
.state { display: inline-flex; border-radius: 999px; padding: 4px 8px; background: #edf2f4; color: #4f6067; font-size: 11px; font-weight: 850; }
.state.ok { background: #e9f8f0; color: #14784f; }
.state.review { background: #fff4df; color: #8a5a0a; }
.revision-log { border: 1px solid #d8e0e4; border-top: 0; }
.revision { display: grid; grid-template-columns: 90px 150px minmax(0, 1fr); gap: 10px; padding: 8px 10px; border-bottom: 1px solid #e2e8eb; color: #4d5e65; font-size: 12px; }
.revision:last-child { border-bottom: 0; }
.snapshots { display: grid; gap: 10px; }
.snapshot { border: 1px solid #d8e0e4; border-radius: 8px; overflow: hidden; background: #fff; }
.snapshot-head { display: flex; justify-content: space-between; gap: 10px; padding: 9px 10px; border-bottom: 1px solid #e2e8eb; color: #3c4d54; font-size: 12px; font-weight: 800; }
.snapshot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding: 8px; }
.empty { color: #68787f; font-weight: 750; padding: 44px 12px; text-align: center; }
dialog { width: min(520px, calc(100vw - 24px)); border: 1px solid #cbd5da; border-radius: 8px; padding: 0; box-shadow: 0 24px 60px rgba(23,32,37,.22); }
dialog::backdrop { background: rgba(23,32,37,.35); }
.dialog-body { display: grid; gap: 12px; padding: 16px; }
.dialog-actions { display: flex; justify-content: flex-end; gap: 8px; padding: 12px 16px; border-top: 1px solid #d8e0e4; }
.measure-inputs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
label { display: grid; gap: 5px; color: #4d5e65; font-size: 12px; font-weight: 800; }
input, textarea { width: 100%; border: 1px solid #bac7cc; border-radius: 6px; padding: 9px; color: #172025; background: #fff; }
textarea { min-height: 78px; resize: vertical; }
@media (max-width: 1050px) { .grid { grid-template-columns: 1fr; } .list { max-height: 340px; } }
@media (max-width: 760px) { .meta { grid-template-columns: repeat(2, 1fr); } .metric:nth-child(2) { border-right: 0; } .pieces { display: block; overflow-x: auto; } .snapshot-grid { grid-template-columns: 1fr; } }
@media (max-width: 520px) { .topbar h1 { font-size: 21px; } .meta { grid-template-columns: 1fr; } .metric { border-right: 0; border-bottom: 1px solid #d8e0e4; } .metric:last-child { border-bottom: 0; } .revision { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <h1>Measurement history</h1>
    <nav class="nav"><a href="/">Live</a></nav>
  </header>
  <div id="mode" class="mode"></div>
  <main class="grid">
    <section class="panel">
      <div class="panel-head">
        <h2 id="event-count">Loading...</h2>
      </div>
      <div class="list" id="event-list"><div class="empty">Loading events...</div></div>
    </section>
    <section class="panel">
      <div class="panel-head">
        <h2 id="detail-title">Select an event</h2>
        <span class="muted" id="detail-state"></span>
      </div>
      <div class="viewer" id="viewer"><div class="empty">Select a saved PLC event.</div></div>
    </section>
  </main>
</div>

<dialog id="correction-dialog">
  <form id="correction-form">
    <div class="dialog-body">
      <h2 id="correction-title">Operator measurement</h2>
      <div class="measure-inputs">
        <label>Feet<input id="feet" type="number" min="0" step="1" required></label>
        <label>Inches<input id="inches" type="number" min="0" max="11" step="1" required></label>
        <label>Sixteenths<input id="sixteenths" type="number" min="0" max="15" step="1" required></label>
      </div>
      <label>Operator ID<input id="operator-id" autocomplete="username" required></label>
      <label>Display name<input id="operator-name"></label>
      <label>Reason<textarea id="reason" required></textarea></label>
      <div id="form-error" class="muted"></div>
    </div>
    <div class="dialog-actions">
      <button type="button" class="btn" id="cancel-correction">Cancel</button>
      <button type="submit" class="btn primary">Save correction</button>
    </div>
  </form>
</dialog>

<script>
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
const number = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '-';
const dateTime = (value) => value ? new Date(value).toLocaleString() : '-';
let activeEventId = '';
let activeEvent = null;
let editingPiece = null;
const initialEventId = location.pathname.startsWith('/history/') ? decodeURIComponent(location.pathname.split('/').pop() || '') : '';

function measurement(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '-';
  let total = Math.round(Math.abs(numeric) * 16);
  const sign = numeric < 0 ? '-' : '';
  const feet = Math.floor(total / 192);
  total -= feet * 192;
  const inches = Math.floor(total / 16);
  let numerator = total % 16;
  let denominator = 16;
  while (numerator && numerator % 2 === 0) {
    numerator /= 2;
    denominator /= 2;
  }
  const inchText = numerator ? `${inches} ${numerator}/${denominator}` : `${inches}`;
  return `${sign}${feet}' ${inchText}"`;
}

function metric(label, value) {
  return `<div class="metric"><span>${esc(label)}</span><strong>${esc(value ?? '-')}</strong></div>`;
}

function renderPieces(pieces, databaseMode) {
  if (!pieces?.length) return '<div class="empty">No pieces were stored for the canonical frame.</div>';
  const table = `<table class="pieces">
    <thead><tr><th>Piece</th><th>Automatic</th><th>Operator</th><th>Effective</th><th>Difference</th><th>Confidence</th><th>Status</th><th></th></tr></thead>
    <tbody>${pieces.map((piece) => {
      const automatic = Number(piece.automatic_measurement_in);
      const operator = Number(piece.operator_measurement_in);
      const hasOperator = Number.isFinite(operator);
      const difference = hasOperator && Number.isFinite(automatic) ? measurement(operator - automatic) : '-';
      const state = piece.review_required || !piece.is_valid ? 'review' : 'ok';
      return `<tr>
        <td><strong>${esc(piece.piece_number)}</strong></td>
        <td>${esc(measurement(piece.automatic_measurement_in))}</td>
        <td>${esc(hasOperator ? measurement(piece.operator_measurement_in) : '-')}</td>
        <td><strong>${esc(measurement(piece.effective_measurement_in))}</strong></td>
        <td>${esc(difference)}</td>
        <td>YOLO ${number(piece.yolo_confidence, 2)} / Edge ${number(piece.sobel_confidence, 2)}</td>
        <td><span class="state ${state}">${state === 'ok' ? 'Valid' : 'Review'}</span><br><span class="muted">rev ${esc(piece.operator_revision ?? 0)}</span></td>
        <td>${databaseMode !== 'simulation' ? `<button class="btn edit-piece" data-piece-id="${esc(piece.id)}">Edit</button>${hasOperator ? ` <button class="btn danger clear-piece" data-piece-id="${esc(piece.id)}">Clear</button>` : ''}` : ''}</td>
      </tr>`;
    }).join('')}</tbody>
  </table>`;
  const revisions = pieces.flatMap((piece) => (piece.revisions || []).map((revision) => ({piece, revision})));
  if (!revisions.length) return table;
  return `${table}<div class="revision-log">${revisions.map(({piece, revision}) => `
    <div class="revision">
      <strong>Piece ${esc(piece.piece_number)} / rev ${esc(revision.revision)}</strong>
      <span>${esc(revision.action)} | ${esc(dateTime(revision.changed_at))}</span>
      <span>${esc(revision.operator_display_name || revision.operator_id)}: ${esc(revision.reason)} (${esc(measurement(revision.previous_operator_measurement_in))} to ${esc(measurement(revision.new_operator_measurement_in))})</span>
    </div>`).join('')}</div>`;
}

function renderSnapshots(snapshots) {
  if (!snapshots?.length) return '<div class="empty">No processing evidence was stored.</div>';
  return `<div class="snapshots">${snapshots.map((snapshot) => {
    const summary = snapshot.measurement_summary || {};
    return `<article class="snapshot">
      <div class="snapshot-head">
        <span>${esc(dateTime(snapshot.processed_utc || snapshot.frame_utc))}</span>
        <span>${esc(summary.valid_count ?? snapshot.valid_piece_count ?? 0)} valid of ${esc(summary.detected_count ?? snapshot.detected_piece_count ?? 0)}</span>
      </div>
      <div class="snapshot-grid">
        ${snapshot.original_overlay_url ? `<img src="${esc(snapshot.original_overlay_url)}" alt="Processed camera view">` : '<div class="empty">No camera evidence</div>'}
        ${snapshot.rectified_overlay_url ? `<img src="${esc(snapshot.rectified_overlay_url)}" alt="Processed diagram view">` : '<div class="empty">No diagram evidence</div>'}
      </div>
    </article>`;
  }).join('')}</div>`;
}

async function loadEvent(eventId) {
  activeEventId = eventId;
  document.querySelectorAll('.event').forEach((element) => element.classList.toggle('active', element.dataset.eventId === eventId));
  const response = await fetch(`/api/history/events/${encodeURIComponent(eventId)}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load measurement event');
  activeEvent = data;
  $('detail-title').textContent = dateTime(data.plc_source_timestamp || data.app_received_at || data.created_at);
  $('detail-state').textContent = data.status || '';
  $('viewer').innerHTML = `
    ${data.video_url ? `<video controls preload="metadata" src="${esc(data.video_url)}"></video>` : '<div class="empty">No video asset is available.</div>'}
    <div class="meta">
      ${metric('PLC signal', dateTime(data.plc_source_timestamp || data.app_received_at))}
      ${metric('Recording', `${dateTime(data.recording_started_at)} - ${dateTime(data.recording_ended_at)}`)}
      ${metric('Pieces', `${data.valid_piece_count ?? 0} valid / ${data.detected_piece_count ?? 0} detected`)}
      ${metric('Status', data.status || '-')}
    </div>
    <div class="section-head"><h3>Piece measurements</h3><span class="muted">Canonical processed frame</span></div>
    ${renderPieces(data.pieces, data.database_mode)}
    <div class="section-head"><h3>Processing evidence</h3><span class="muted">Up to 6 representative captures</span></div>
    ${renderSnapshots(data.snapshots)}
  `;
  document.querySelectorAll('.edit-piece').forEach((button) => button.addEventListener('click', () => openCorrection(button.dataset.pieceId)));
  document.querySelectorAll('.clear-piece').forEach((button) => button.addEventListener('click', () => clearCorrection(button.dataset.pieceId)));
}

async function loadHistory() {
  const response = await fetch('/api/history/events?limit=100');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load history');
  if (data.database_mode !== 'postgresql') {
    $('mode').style.display = 'block';
    $('mode').textContent = data.warning || 'The database is disabled.';
  }
  $('event-count').textContent = `${data.count} saved events`;
  if (!data.events.length) {
    $('event-list').innerHTML = '<div class="empty">No PLC events have been saved.</div>';
    return;
  }
  $('event-list').innerHTML = data.events.map((event) => `
    <button class="event" data-event-id="${esc(event.id)}">
      <strong>${esc(dateTime(event.plc_source_timestamp || event.app_received_at || event.created_at))}</strong>
      <span>${esc(event.valid_piece_count ?? 0)} valid / ${esc(event.detected_piece_count ?? 0)} detected | ${esc(event.status || '-')}</span>
    </button>
  `).join('');
  document.querySelectorAll('.event').forEach((button) => button.addEventListener('click', () => loadEvent(button.dataset.eventId)));
  const initial = data.events.find((event) => String(event.id) === initialEventId) || data.events[0];
  await loadEvent(String(initial.id));
}

function openCorrection(pieceId) {
  editingPiece = activeEvent?.pieces?.find((piece) => String(piece.id) === String(pieceId));
  if (!editingPiece) return;
  const total = Number(editingPiece.operator_measurement_in ?? editingPiece.automatic_measurement_in ?? 0);
  let units = Math.max(0, Math.round(total * 16));
  $('feet').value = Math.floor(units / 192);
  units %= 192;
  $('inches').value = Math.floor(units / 16);
  $('sixteenths').value = units % 16;
  $('operator-id').value = localStorage.getItem('tx2OperatorId') || '';
  $('operator-name').value = localStorage.getItem('tx2OperatorName') || '';
  $('reason').value = '';
  $('form-error').textContent = '';
  $('correction-title').textContent = `Piece ${editingPiece.piece_number} operator measurement`;
  $('correction-dialog').showModal();
}

async function clearCorrection(pieceId) {
  const piece = activeEvent?.pieces?.find((item) => String(item.id) === String(pieceId));
  if (!piece || !confirm(`Clear the operator correction for piece ${piece.piece_number}?`)) return;
  const operatorId = localStorage.getItem('tx2OperatorId') || '';
  const reason = prompt('Reason for clearing this correction:') || '';
  if (!operatorId || !reason) {
    alert('Operator ID and reason are required. Open Edit once to store the operator ID.');
    return;
  }
  await patchCorrection(piece, {clear: true, reason, operator_id: operatorId, operator_display_name: localStorage.getItem('tx2OperatorName') || null});
}

async function patchCorrection(piece, payload) {
  const response = await fetch(`/api/history/events/${encodeURIComponent(activeEventId)}/pieces/${encodeURIComponent(piece.id)}/operator-measurement`, {
    method: 'PATCH',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({...payload, expected_revision: piece.operator_revision ?? 0}),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not save correction');
  await loadEvent(activeEventId);
}

$('correction-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const operatorId = $('operator-id').value.trim();
  const operatorName = $('operator-name').value.trim();
  localStorage.setItem('tx2OperatorId', operatorId);
  localStorage.setItem('tx2OperatorName', operatorName);
  try {
    await patchCorrection(editingPiece, {
      feet: Number($('feet').value),
      inches: Number($('inches').value),
      sixteenths: Number($('sixteenths').value),
      operator_id: operatorId,
      operator_display_name: operatorName || null,
      reason: $('reason').value.trim(),
    });
    $('correction-dialog').close();
  } catch (error) {
    $('form-error').textContent = error.message || error;
  }
});
$('cancel-correction').addEventListener('click', () => $('correction-dialog').close());

loadHistory().catch((error) => {
  $('event-list').innerHTML = `<div class="empty">${esc(error.message || error)}</div>`;
});
</script>
</body>
</html>
"""


app = Flask(__name__)
_args: argparse.Namespace
_buffer: FrameBuffer
_camera: CameraReader
_processor: LiveProcessor
_recorder: ClipRecorder
_plc: PLCMonitor
_database: DatabaseRepository | SQLiteDatabaseRepository | None = None
_reconciler: DatabaseReconciler | None = None


def database_mode() -> str:
    if _database is None:
        return "simulation"
    return str(getattr(_database, "backend_name", "postgresql"))


def clips_root(output_dir: Path) -> Path:
    return output_dir / "live_plc_clips"


def clip_sidecars(output_dir: Path) -> list[Path]:
    root = clips_root(output_dir)
    if not root.exists():
        return []
    existing = []
    for path in root.rglob("*.json"):
        try:
            existing.append((path.stat().st_mtime, path))
        except FileNotFoundError:
            continue
    return [path for _mtime, path in sorted(existing, reverse=True)]


def path_is_inside(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        resolved_root = root.resolve()
    except Exception:
        return False
    return resolved == resolved_root or resolved_root in resolved.parents


def delete_clip_paths(output_dir: Path, candidates: list[Path | None]) -> list[str]:
    root = clips_root(output_dir)
    errors = []
    for candidate in candidates:
        if candidate is None or not path_is_inside(candidate, root):
            continue
        try:
            if candidate.is_dir():
                for child in sorted(candidate.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                candidate.rmdir()
            else:
                candidate.unlink(missing_ok=True)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    return errors


def read_clip_sidecar(json_path: Path, snapshot_limit: int | None = None) -> dict[str, Any] | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    snapshots = data.get("processing_snapshots", []) or []
    data["processing_snapshot_count"] = int(data.get("processing_snapshot_count", len(snapshots)))
    if snapshot_limit is not None:
        snapshots = representative_snapshots(snapshots, snapshot_limit)
        data["processing_snapshots"] = snapshots
    data["processing_snapshots_shown"] = len(snapshots)
    data["clip_id"] = json_path.stem
    data["json_path"] = str(json_path)
    data["video_url"] = f"/api/live/clips/{json_path.stem}/video"
    data["detail_url"] = f"/history/{json_path.stem}"
    for snapshot in snapshots:
        if snapshot.get("original_overlay_file"):
            snapshot["original_overlay_url"] = f"/api/live/clips/{json_path.stem}/asset/{snapshot['original_overlay_file']}"
        if snapshot.get("rectified_overlay_file"):
            snapshot["rectified_overlay_url"] = f"/api/live/clips/{json_path.stem}/asset/{snapshot['rectified_overlay_file']}"
    return data


def find_clip_json(clip_id: str) -> Path | None:
    for json_path in clip_sidecars(_args.output_dir):
        if json_path.stem == clip_id:
            return json_path
    return None


def legacy_history_event(json_path: Path, *, detail: bool) -> dict[str, Any] | None:
    data = read_clip_sidecar(
        json_path,
        snapshot_limit=HISTORY_SNAPSHOT_LIMIT if detail else 0,
    )
    if data is None:
        return None
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        all_snapshots = raw.get("processing_snapshots", [])
    except Exception:
        all_snapshots = data.get("processing_snapshots", [])
    canonical = select_canonical_snapshot(
        all_snapshots,
        event_monotonic=(data.get("event") or {}).get("event_read_monotonic"),
    )
    summary = snapshot_summary(canonical) if canonical else {}
    event_id = str(data.get("event_id") or data["clip_id"])
    result: dict[str, Any] = {
        "id": event_id,
        "clip_id": data["clip_id"],
        "status": "simulation",
        "created_at": data.get("saved_at"),
        "app_received_at": (data.get("event") or {}).get("read_utc"),
        "plc_source_timestamp": (data.get("event") or {}).get("event_source_timestamp"),
        "plc_edge": (data.get("event") or {}).get("event_edge"),
        "detected_piece_count": int(summary.get("detected_count") or 0),
        "valid_piece_count": int(summary.get("valid_count") or 0),
        "video_url": data.get("video_url"),
        "detail_url": f"/history/{event_id}",
        "database_mode": "simulation",
    }
    if not detail:
        return result
    canonical_pieces = snapshot_pieces(canonical) if canonical else []
    result.update(
        recording_started_at=data.get("first_frame_utc"),
        recording_ended_at=data.get("last_frame_utc"),
        snapshots=data.get("processing_snapshots", []),
        pieces=[
            {
                "id": f"legacy-{index}",
                "piece_number": int(piece.get("piece_id") or index),
                "is_valid": bool(piece.get("valid")),
                "review_required": not bool(piece.get("valid")),
                "yolo_confidence": piece.get("confidence"),
                "sobel_confidence": (piece.get("sobel") or {}).get("edge_confidence"),
                "distance_to_reference_in": (piece.get("measurement") or {}).get("delta_in"),
                "automatic_measurement_in": (piece.get("measurement") or {}).get("measurement_in"),
                "operator_measurement_in": None,
                "effective_measurement_in": (piece.get("measurement") or {}).get("measurement_in"),
                "operator_revision": 0,
                "revisions": [],
            }
            for index, piece in enumerate(canonical_pieces, start=1)
        ],
        assets=[],
    )
    return result


def database_event_detail(event_id: str) -> dict[str, Any] | None:
    if _database is None:
        json_path = find_clip_json(event_id)
        if json_path is None:
            for candidate in clip_sidecars(_args.output_dir):
                try:
                    raw = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(raw.get("event_id")) == event_id:
                    json_path = candidate
                    break
        return legacy_history_event(json_path, detail=True) if json_path else None

    event = _database.get_measurement_event(event_id)
    if event is None:
        return None
    assets = [clean_value(asset) for asset in event.get("assets", [])]
    asset_by_path = {asset["relative_path"]: asset for asset in assets}
    snapshots = representative_snapshots(event.get("snapshots", []), HISTORY_SNAPSHOT_LIMIT)
    for snapshot in snapshots:
        for path_key, url_key in (
            ("original_overlay_path", "original_overlay_url"),
            ("rectified_overlay_path", "rectified_overlay_url"),
        ):
            asset = asset_by_path.get(snapshot.get(path_key))
            if asset:
                snapshot[url_key] = f"/api/history/events/{event_id}/assets/{asset['id']}"
    video = next((asset for asset in assets if asset.get("asset_type") == "video"), None)
    event["snapshots"] = snapshots
    event["assets"] = assets
    event["video_url"] = (
        f"/api/history/events/{event_id}/assets/{video['id']}" if video else None
    )
    event["detail_url"] = f"/history/{event_id}"
    event["database_mode"] = database_mode()
    return clean_value(event)


def delete_clip_artifacts(output_dir: Path, json_path: Path) -> None:
    data = read_clip_sidecar(json_path, snapshot_limit=0) or {}
    candidates: list[Path] = [json_path]
    for key in ("video_path", "analysis_dir"):
        value = data.get(key)
        if value:
            candidates.append(Path(value))
    delete_clip_paths(output_dir, candidates)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/history")
def history():
    return render_template_string(HISTORY_HTML)


@app.route("/history/<clip_id>")
def history_clip(clip_id: str):
    return render_template_string(HISTORY_HTML)


@app.route("/api/live/status")
def api_live_status():
    database_status = (
        _database.health().as_dict()
        if _database is not None
        else {
            "enabled": False,
            "ok": False,
            "error": "Database disabled explicitly for simulation",
            "mode": "simulation",
        }
    )
    return jsonify(
        camera=_camera.snapshot(),
        processor=_processor.snapshot(include_images=False),
        plc=_plc.snapshot(),
        recorder=_recorder.snapshot(),
        database=database_status,
        reconciler=_reconciler.snapshot() if _reconciler is not None else None,
    )


@app.route("/api/live/frame")
def api_live_frame():
    data = _processor.snapshot(include_images=True)
    if data.get("result") is None:
        return jsonify(error=data.get("error") or "No processed frame is available yet", processor=data), 503
    return jsonify(data)


@app.route("/api/history/events")
def api_history_events():
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify(error="limit must be an integer"), 400
    limit = max(1, min(limit, 100))
    if _database is None:
        events = []
        for json_path in clip_sidecars(_args.output_dir)[:limit]:
            event = legacy_history_event(json_path, detail=False)
            if event is not None:
                events.append(event)
        return jsonify(
            events=events,
            count=len(events),
            database_mode="simulation",
            warning="The database is disabled explicitly; operator corrections are unavailable.",
        )
    mode = database_mode()
    try:
        events = _database.list_measurement_events(
            limit=limit,
            before=request.args.get("before"),
        )
    except DatabaseUnavailable as exc:
        return jsonify(error=str(exc), database_mode=mode), 503
    for event in events:
        event["id"] = str(event["id"])
        event["detail_url"] = f"/history/{event['id']}"
        event["video_url"] = f"/api/history/events/{event['id']}/video"
    warning = (
        "SQLite temporal is active. History and corrections will be migrated "
        "to PostgreSQL when the server is available."
        if mode == "sqlite"
        else ""
    )
    return jsonify(
        events=clean_value(events),
        count=len(events),
        database_mode=mode,
        warning=warning,
    )


@app.route("/api/history/events/<event_id>")
def api_history_event(event_id: str):
    try:
        event = database_event_detail(event_id)
    except DatabaseUnavailable as exc:
        return jsonify(error=str(exc)), 503
    if event is None:
        return jsonify(error="Measurement event not found"), 404
    return jsonify(event)


@app.route("/api/history/events/<event_id>/video")
def api_history_event_video(event_id: str):
    try:
        event = database_event_detail(event_id)
    except DatabaseUnavailable:
        abort(503)
    if event is None:
        abort(404)
    if _database is None:
        clip_id = event.get("clip_id")
        json_path = find_clip_json(str(clip_id)) if clip_id else None
        if json_path is None:
            abort(404)
        data = read_clip_sidecar(json_path, snapshot_limit=0) or {}
        video_path = Path(str(data.get("video_path", "")))
    else:
        video_asset = next(
            (asset for asset in event.get("assets", []) if asset.get("asset_type") == "video"),
            None,
        )
        if video_asset is None:
            abort(404)
        try:
            video_path = resolve_asset_path(video_asset["relative_path"], _args.output_dir)
        except ValueError:
            abort(404)
    if not video_path.is_file() or not path_is_inside(video_path, _args.output_dir):
        abort(404)
    return send_file(video_path, mimetype="video/mp4", conditional=True)


@app.route("/api/history/events/<event_id>/assets/<int:asset_id>")
def api_history_event_asset(event_id: str, asset_id: int):
    if _database is None:
        abort(404)
    try:
        event = database_event_detail(event_id)
    except DatabaseUnavailable:
        abort(503)
    if event is None:
        abort(404)
    asset = next(
        (item for item in event.get("assets", []) if int(item["id"]) == asset_id),
        None,
    )
    if asset is None:
        abort(404)
    try:
        asset_path = resolve_asset_path(asset["relative_path"], _args.output_dir)
    except ValueError:
        abort(404)
    if not asset_path.is_file():
        abort(404)
    return send_file(
        asset_path,
        mimetype=asset.get("mime_type") or "application/octet-stream",
        conditional=True,
    )


def operator_measurement_inches(payload: dict[str, Any]) -> Decimal:
    if payload.get("measurement_in") is not None:
        return Decimal(str(payload["measurement_in"]))
    try:
        feet = int(payload.get("feet", 0))
        inches = int(payload.get("inches", 0))
        sixteenths = int(payload.get("sixteenths", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Feet, inches and sixteenths must be integers") from exc
    if feet < 0 or not 0 <= inches <= 11 or not 0 <= sixteenths <= 15:
        raise ValueError("Use non-negative feet, 0-11 inches and 0-15 sixteenths")
    return Decimal(feet * 12 + inches) + (Decimal(sixteenths) / Decimal(16))


@app.route(
    "/api/history/events/<event_id>/pieces/<piece_id>/operator-measurement",
    methods=["PATCH"],
)
def api_history_operator_measurement(event_id: str, piece_id: str):
    if _database is None:
        return jsonify(error="Operator corrections require a database"), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="A JSON request body is required"), 400
    try:
        expected_revision = int(payload["expected_revision"])
        common = {
            "event_id": event_id,
            "piece_id": piece_id,
            "reason": str(payload.get("reason") or ""),
            "operator_id": str(payload.get("operator_id") or ""),
            "operator_display_name": (
                str(payload["operator_display_name"])
                if payload.get("operator_display_name")
                else None
            ),
            "expected_revision": expected_revision,
            "source_ip": request.remote_addr,
        }
        if payload.get("clear") is True:
            _database.clear_operator_measurement(**common)
        else:
            _database.set_operator_measurement(
                **common,
                measurement_in=operator_measurement_inches(payload),
            )
        event = database_event_detail(event_id)
        return jsonify(event)
    except KeyError:
        return jsonify(error="expected_revision is required"), 400
    except RevisionConflict as exc:
        return jsonify(error=str(exc), conflict=True), 409
    except RecordNotFound as exc:
        return jsonify(error=str(exc)), 404
    except (ValueError, ArithmeticError) as exc:
        return jsonify(error=str(exc)), 400
    except DatabaseUnavailable as exc:
        return jsonify(error=str(exc)), 503


@app.route("/api/live/clips")
def api_live_clips():
    clips = []
    for json_path in clip_sidecars(_args.output_dir):
        data = read_clip_sidecar(json_path, snapshot_limit=0)
        if data is not None:
            clips.append(data)
    return jsonify(clips=clips[:100], count=len(clips))


@app.route("/api/live/clips/<clip_id>")
def api_live_clip(clip_id: str):
    json_path = find_clip_json(clip_id)
    if json_path is None:
        return jsonify(error="Clip not found"), 404
    data = read_clip_sidecar(json_path, snapshot_limit=HISTORY_SNAPSHOT_LIMIT)
    if data is None:
        return jsonify(error="Clip metadata could not be read"), 500
    return jsonify(data)


@app.route("/api/live/clips/<clip_id>/video")
def api_live_clip_video(clip_id: str):
    json_path = find_clip_json(clip_id)
    if json_path is None:
        abort(404)
    data = read_clip_sidecar(json_path, snapshot_limit=0) or {}
    video_path = Path(str(data.get("video_path", "")))
    if not video_path.exists() or not path_is_inside(video_path, clips_root(_args.output_dir)):
        abort(404)
    return send_file(video_path, mimetype="video/mp4", conditional=True)


@app.route("/api/live/clips/<clip_id>/asset/<path:asset_name>")
def api_live_clip_asset(clip_id: str, asset_name: str):
    json_path = find_clip_json(clip_id)
    if json_path is None:
        abort(404)
    data = read_clip_sidecar(json_path, snapshot_limit=0) or {}
    analysis_dir = Path(str(data.get("analysis_dir", "")))
    asset_path = analysis_dir / asset_name
    if not asset_path.exists() or not path_is_inside(asset_path, analysis_dir) or not path_is_inside(asset_path, clips_root(_args.output_dir)):
        abort(404)
    return send_file(asset_path, mimetype="image/jpeg", conditional=True)


def main() -> int:
    global _args, _buffer, _camera, _processor, _recorder, _plc, _database, _reconciler
    _args = parse_args()
    configure_vision_module(_args)
    if not _args.db_disabled:
        try:
            if _args.postgres_dsn.strip():
                _database = DatabaseRepository(_args.postgres_dsn)
            else:
                _database = SQLiteDatabaseRepository(_args.sqlite_path)
            _database.open(timeout=float(_args.plc_timeout))
            _database.validate_schema()
            _database.recover_stale_measurement_events(
                older_than_seconds=max(60, int(float(_args.record_seconds) * 3))
            )
        except Exception as exc:
            if _database is not None:
                _database.close()
                _database = None
            print(f"ERROR: Database startup validation failed: {exc}", file=sys.stderr)
            return 2

    requested_buffer_frames = int(max(8, float(_args.buffer_seconds) * max(1.0, float(_args.capture_fps))))
    buffer_len = min(requested_buffer_frames, max(8, int(_args.buffer_max_frames)))
    _buffer = FrameBuffer(maxlen=buffer_len)
    _camera = CameraReader(_args, _buffer)
    _processor = LiveProcessor(_args, _buffer)
    _recorder = ClipRecorder(_args, _buffer, _processor, _database)
    try:
        _recorder.validate_configuration()
    except Exception as exc:
        if _database is not None:
            _database.close()
            _database = None
        print(f"ERROR: Vision configuration validation failed: {exc}", file=sys.stderr)
        return 2
    _plc = PLCMonitor(_args, _recorder)
    _reconciler = DatabaseReconciler(_args, _database) if _database is not None else None

    _camera.start()
    _processor.start()
    _plc.start()
    if _reconciler is not None:
        _reconciler.start()

    print(f"\n  TX2 Live MVP at http://127.0.0.1:{_args.port}\n")
    print(f"  Source: {_args.source}")
    print(f"  PLC: {'enabled' if _args.plc_enabled else 'disabled'}")
    print(
        "  Database: "
        + ("disabled (simulation)" if _args.db_disabled else database_mode())
    )
    try:
        app.run(host="127.0.0.1", port=_args.port, debug=False, threaded=True)
    finally:
        if _reconciler is not None:
            _reconciler.stop()
        _plc.stop()
        _processor.stop()
        _camera.stop()
        if _database is not None:
            _database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
