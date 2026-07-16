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
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template_string, send_file

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import homography_web_app as vision

DEFAULT_VIDEO = Path(r"C:\Users\luis_\Downloads\20260508_000307_7F66.mkv")
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_DATASET_DIR = ROOT / "dataset"
DEFAULT_MODEL = ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_tubos_v1" / "weights" / "best.pt"
DEFAULT_ENDPOINT = "opc.tcp://10.14.6.48:49320"
DEFAULT_WATCHDOG_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD"
DEFAULT_EVENT_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength"


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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    return str(value)


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
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--capture-fps", type=float, default=10.0)
    parser.add_argument("--process-fps", type=float, default=10.0)
    parser.add_argument("--buffer-seconds", type=float, default=45.0)
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


def draw_rectified_overlay(rectified: np.ndarray, boxes: list[dict], sobel: dict, calibration: dict) -> np.ndarray:
    out = rectified.copy()
    for box in boxes:
        x0 = int(float(box["x"]))
        y0 = int(float(box["y"]))
        x1 = int(float(box["x"]) + float(box["w"]))
        y1 = int(float(box["y"]) + float(box["h"]))
        cv2.rectangle(out, (x0, y0), (x1, y1), (62, 214, 166), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"{float(box.get('conf', 0.0)):.2f}",
            (x0 + 3, max(18, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (62, 214, 166),
            2,
            cv2.LINE_AA,
        )
    if calibration.get("reference_y") is not None:
        y = float(calibration["reference_y"])
        draw_line(out, {"x1": 0, "y1": y, "x2": out.shape[1] - 1, "y2": y}, (210, 130, 48), "REF")
    draw_line(out, (sobel or {}).get("line"), (40, 210, 128), "front")
    return out


def draw_original_overlay(original: np.ndarray, overlay: dict) -> np.ndarray:
    out = original.copy()
    draw_line(out, overlay.get("reference_line"), (210, 130, 48), "REF")
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
            item = self._frames[-1].copy()
            item["frame"] = self._frames[-1]["frame"].copy()
            return item

    def frames_since(self, min_index: int) -> list[dict[str, Any]]:
        with self._lock:
            selected = [item for item in self._frames if int(item["index"]) > int(min_index)]
            copies = []
            for item in selected:
                copy = item.copy()
                copy["frame"] = item["frame"].copy()
                copies.append(copy)
            return copies

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
                    data["result"] = result.copy()
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
        boxes = vision.predict_yolo_boxes(rectified, conf=float(self.args.conf), imgsz=int(self.args.imgsz))
        box = max(
            boxes,
            key=lambda candidate: float(candidate["w"]) * float(candidate["h"]) * float(candidate.get("conf", 1.0)),
            default=None,
        )

        if box is None:
            sobel = {
                "frame_idx": int(item["index"]),
                "time_sec": None,
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
            sobel = vision.sobel_projection_for_box(rectified, box)
            sobel.update(frame_idx=int(item["index"]), time_sec=None)

        rect_h, rect_w = rectified.shape[:2]
        src_h, src_w = original.shape[:2]
        calibration = vision.load_measurement_calibration()
        measurement = vision.measurement_from_sobel(sobel, calibration, rect_w)
        original_overlay = vision.mvp_original_overlay(sobel, calibration, matrix, rect_w)
        original_viz = draw_original_overlay(original, original_overlay)
        rectified_viz = draw_rectified_overlay(rectified, boxes, sobel, calibration)

        return {
            "frame_index": int(item["index"]),
            "frame_utc": item["utc"],
            "processed_utc": utc_now(),
            "original_width": src_w,
            "original_height": src_h,
            "rectified_width": rect_w,
            "rectified_height": rect_h,
            "original_image": img_to_b64(original_viz, quality=80),
            "rectified_image": img_to_b64(rectified_viz, quality=82),
            "boxes": boxes,
            "count": len(boxes),
            "sobel": sobel,
            "calibration": calibration,
            "measurement": measurement,
            "front_y_ratio": (float(sobel["line"]["y"]) / float(rect_h)) if sobel.get("line") else None,
            "model": str(self.args.model),
            "conf": float(self.args.conf),
        }


class ClipRecorder:
    def __init__(self, args: argparse.Namespace, buffer: FrameBuffer, processor: LiveProcessor | None = None) -> None:
        self.args = args
        self.buffer = buffer
        self.processor = processor
        self.lock = threading.Lock()
        self.recording = False
        self.clip_index = 0
        self.last_clip: dict[str, Any] | None = None
        self.error = ""

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "recording": self.recording,
                "clip_index": self.clip_index,
                "last_clip": self.last_clip,
                "error": self.error,
                "record_seconds": self.args.record_seconds,
            }

    def start_event_clip(self, event: dict[str, Any]) -> None:
        with self.lock:
            if self.recording:
                self.error = "Ya habia una grabacion activa; evento ignorado para clip."
                return
            self.recording = True
            self.clip_index += 1
            clip_index = self.clip_index
            self.error = ""
        thread = threading.Thread(target=self._record_clip, args=(clip_index, event), daemon=True)
        thread.start()

    def _capture_processing_snapshot(
        self,
        analysis_dir: Path,
        snapshots: list[dict[str, Any]],
        seen_frame_indices: set[int],
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

    @staticmethod
    def _sample_frames(frames: list[dict[str, Any]], fps: float, duration: float) -> list[dict[str, Any]]:
        target_count = max(1, int(round(duration * fps)))
        start_mono = float(frames[0]["monotonic"])
        selected: list[dict[str, Any]] = []
        frame_index = 0
        for sample_index in range(target_count):
            target_mono = start_mono + sample_index / fps
            while frame_index + 1 < len(frames):
                current_delta = abs(float(frames[frame_index]["monotonic"]) - target_mono)
                next_delta = abs(float(frames[frame_index + 1]["monotonic"]) - target_mono)
                if next_delta > current_delta:
                    break
                frame_index += 1
            selected.append(frames[frame_index])
        return selected

    def _record_clip(self, clip_index: int, event: dict[str, Any]) -> None:
        frames: list[dict[str, Any]] = []
        processing_snapshots: list[dict[str, Any]] = []
        seen_processing_frames: set[int] = set()
        start_mono = time.perf_counter()
        last_index = -1
        latest = self.buffer.latest()
        if latest:
            last_index = int(latest["index"]) - 1

        try:
            day_dir = self.args.output_dir / "live_plc_clips" / datetime.now().strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            edge = event.get("event_edge") or "event"
            base = f"live_{clip_index:04d}_{file_stamp()}_{edge}"
            analysis_dir = day_dir / f"{base}_analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)

            while time.perf_counter() - start_mono < float(self.args.record_seconds):
                new_frames = self.buffer.frames_since(last_index)
                if new_frames:
                    frames.extend(new_frames)
                    last_index = int(new_frames[-1]["index"])
                self._capture_processing_snapshot(analysis_dir, processing_snapshots, seen_processing_frames)
                time.sleep(0.025)
            self._capture_processing_snapshot(analysis_dir, processing_snapshots, seen_processing_frames)

            if not frames:
                raise RuntimeError("No frames were available to record the clip.")

            video_path = day_dir / f"{base}.mp4"
            json_path = day_dir / f"{base}.json"

            first_frame = frames[0]["frame"]
            height, width = first_frame.shape[:2]
            fps = float(self.args.record_fps or self.args.capture_fps or 10.0)
            output_frames = self._sample_frames(frames, fps, float(self.args.record_seconds))
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter: {video_path}")
            for item in output_frames:
                frame = item["frame"]
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
            writer.release()

            sidecar = {
                "clip_index": clip_index,
                "saved_at": utc_now(),
                "event": event,
                "record_seconds": float(self.args.record_seconds),
                "video_fps": fps,
                "frames_captured": len(frames),
                "frames_written": len(output_frames),
                "first_frame_utc": output_frames[0]["utc"],
                "last_frame_utc": output_frames[-1]["utc"],
                "first_frame_index": int(output_frames[0]["index"]),
                "last_frame_index": int(output_frames[-1]["index"]),
                "video_path": str(video_path),
                "analysis_dir": str(analysis_dir),
                "processing_snapshots": processing_snapshots,
                "processing_snapshot_count": len(processing_snapshots),
            }
            json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
            self._enforce_retention()
            with self.lock:
                self.last_clip = sidecar
                self.error = ""
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            with self.lock:
                self.recording = False

    def _enforce_retention(self) -> None:
        max_clips = max(1, int(self.args.max_clips or 100))
        sidecars = clip_sidecars(self.args.output_dir)
        for json_path in sidecars[max_clips:]:
            delete_clip_artifacts(self.args.output_dir, json_path)


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
                        self.recorder.start_event_clip(row)
                    self._set_state(**updates)
                    previous_watchdog = watchdog_value
                    previous_event = event.get("value")
                await asyncio.sleep(float(self.args.plc_poll_interval))


HTML = r"""
<!doctype html>
<html lang="es">
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
.topbar { display: flex; align-items: end; justify-content: space-between; gap: 14px; margin-bottom: 12px; }
h1, h2, p { margin: 0; letter-spacing: 0; }
h1 { font-size: 27px; line-height: 1.05; }
h2 { font-size: 16px; }
.eyebrow { color: #68787f; font-size: 12px; font-weight: 800; text-transform: uppercase; margin-bottom: 4px; }
.pill { border: 1px solid #cbd5da; border-radius: 999px; padding: 8px 12px; color: #485960; background: #ffffff; font-weight: 800; white-space: nowrap; }
.pill.ok { border-color: #58a680; color: #14784f; background: #ecfff5; }
.pill.warn { border-color: #d7a34d; color: #8a5a0a; background: #fff7e5; }
.pill.err { border-color: #d48282; color: #a23232; background: #fff0f0; }
.nav { display: flex; align-items: center; gap: 10px; }
.nav a { border: 1px solid #cbd5da; border-radius: 8px; padding: 9px 12px; color: #172025; background: #ffffff; text-decoration: none; font-weight: 900; }
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
    <div>
      <p class="eyebrow">TX2 Vision</p>
      <h1>Live MVP</h1>
    </div>
    <div class="nav">
      <a href="/history">History</a>
      <div id="top-state" class="pill warn">connecting...</div>
    </div>
  </header>

  <main class="grid">
    <section class="panel">
      <div class="stage" id="original-stage"><div class="empty">Waiting for frame...</div></div>
    </section>

    <section class="panel">
      <svg class="diagram" viewBox="0 0 760 310" role="img" aria-label="Measurement diagram">
        <rect x="38" y="34" width="684" height="232" rx="8" fill="#f7faf8" stroke="#cbd6cf" stroke-width="2" />
        <rect x="86" y="82" width="588" height="132" rx="6" fill="#e5ece8" stroke="#c0cbc6" />
        <line x1="92" x2="668" y1="156" y2="156" stroke="#d28230" stroke-width="5" stroke-linecap="round" stroke-dasharray="12 9" />
        <text x="104" y="184" fill="#a66324" font-size="20" font-weight="900">REF</text>
        <line id="diagram-front" x1="92" x2="668" y1="205" y2="205" stroke="#28a96e" stroke-width="7" stroke-linecap="round" />
        <text id="diagram-label" x="104" y="235" fill="#14784f" font-size="20" font-weight="900">front</text>
        <line id="diagram-measure" x1="700" x2="700" y1="156" y2="205" stroke="#243c48" stroke-width="3" stroke-dasharray="8 7" />
      </svg>
    </section>
  </main>
</div>

<script>
const $ = (id) => document.getElementById(id);

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

function updateDiagram(ratio) {
  const has = Number.isFinite(Number(ratio));
  const y = has ? Math.max(112, Math.min(238, 92 + Number(ratio) * 160)) : 205;
  $('diagram-front').setAttribute('y1', y);
  $('diagram-front').setAttribute('y2', y);
  $('diagram-label').setAttribute('y', Math.max(118, Math.min(252, y + 30)));
  $('diagram-measure').setAttribute('y2', y);
}

async function refreshFrame() {
  try {
    const response = await fetch('/api/live/frame');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'frame error');
    const result = data.result;
    if (!result) return;
    setImage($('original-stage'), result.original_image, 'Live camera');
    updateDiagram(result.front_y_ratio);
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
    const healthy = camera.connected && processor.ok;
    pill($('top-state'), healthy ? 'live' : 'check status', healthy ? 'ok' : 'warn');
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
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TX2 Clip History</title>
<style>
:root { color: #172025; background: #f5f7f8; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; min-width: 320px; min-height: 100vh; background: #f5f7f8; }
.app { width: min(1500px, calc(100vw - 24px)); margin: 0 auto; padding: 14px 0 22px; }
.topbar { display: flex; align-items: end; justify-content: space-between; gap: 14px; margin-bottom: 12px; }
h1, h2, p { margin: 0; letter-spacing: 0; }
h1 { font-size: 27px; line-height: 1.05; }
h2 { font-size: 16px; }
.eyebrow { color: #68787f; font-size: 12px; font-weight: 800; text-transform: uppercase; margin-bottom: 4px; }
.nav a, .btn { border: 1px solid #cbd5da; border-radius: 8px; padding: 9px 12px; color: #172025; background: #ffffff; text-decoration: none; font-weight: 900; cursor: pointer; }
.grid { display: grid; grid-template-columns: 380px minmax(0, 1fr); gap: 12px; align-items: start; }
.panel { border: 1px solid #d8e0e4; border-radius: 8px; background: #ffffff; overflow: hidden; box-shadow: 0 14px 28px rgba(23,32,37,.08); }
.panel-head { min-height: 48px; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid #d8e0e4; }
.list { display: grid; max-height: calc(100vh - 120px); overflow: auto; }
.clip { display: grid; gap: 5px; padding: 11px 12px; color: #2c3a40; background: transparent; border: 0; border-bottom: 1px solid rgba(23,32,37,.08); text-align: left; cursor: pointer; }
.clip:hover, .clip.active { background: #edf5f8; }
.clip strong { color: #172025; overflow-wrap: anywhere; }
.clip span { color: #68787f; font-size: 12px; font-weight: 800; }
.viewer { padding: 12px; display: grid; gap: 12px; }
video, img { display: block; width: 100%; border-radius: 8px; background: #eef2f4; }
.meta { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.metric { border: 1px solid #d8e0e4; border-radius: 8px; background: #ffffff; padding: 10px; min-height: 62px; }
.metric span { display: block; color: #68787f; font-size: 12px; font-weight: 800; }
.metric strong { display: block; margin-top: 5px; color: #172025; font-size: 18px; overflow-wrap: anywhere; }
.snapshots { display: grid; gap: 10px; }
.snapshot { border: 1px solid #d8e0e4; border-radius: 8px; overflow: hidden; background: #ffffff; }
.snapshot-head { display: flex; justify-content: space-between; gap: 10px; padding: 9px 10px; border-bottom: 1px solid rgba(23,32,37,.08); color: #3c4d54; font-size: 13px; font-weight: 900; }
.snapshot-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding: 8px; }
.empty { color: #68787f; font-weight: 800; padding: 44px 12px; text-align: center; }
@media (max-width: 980px) { .grid { grid-template-columns: 1fr; } .meta { grid-template-columns: repeat(2, minmax(0, 1fr)); } .snapshot-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div>
      <p class="eyebrow">TX2 Vision</p>
      <h1>Video history</h1>
    </div>
    <div class="nav"><a href="/">Live</a></div>
  </header>
  <main class="grid">
    <section class="panel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">PLC clips</p>
          <h2 id="clip-count">Loading...</h2>
        </div>
      </div>
      <div class="list" id="clip-list"><div class="empty">Loading clips...</div></div>
    </section>
    <section class="panel">
      <div class="panel-head">
        <div>
          <p class="eyebrow">Replay</p>
          <h2 id="detail-title">Select a clip</h2>
        </div>
      </div>
      <div class="viewer" id="viewer"><div class="empty">Select a saved PLC clip.</div></div>
    </section>
  </main>
</div>
<script>
const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 3) => {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : '-';
};
let activeClip = '';
const initialClipId = location.pathname.startsWith('/history/') ? decodeURIComponent(location.pathname.split('/').pop() || '') : '';

function clipLabel(clip) {
  return clip.saved_at || clip.first_frame_utc || clip.clip_id;
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value ?? '-'}</strong></div>`;
}

function renderSnapshots(snapshots) {
  if (!snapshots || !snapshots.length) return '<div class="empty">No processing snapshots saved for this clip.</div>';
  return `<div class="snapshots">${snapshots.map((snap) => {
    const measurement = snap.measurement || {};
    const total = measurement.measurement_in != null ? `${fmt(measurement.measurement_in)} in` : '-';
    const delta = measurement.delta_in != null ? `${fmt(measurement.delta_in)} in` : '-';
    return `<article class="snapshot">
      <div class="snapshot-head">
        <span>${snap.processed_utc || snap.frame_utc || '-'}</span>
        <span>YOLO ${snap.count ?? 0} | Total ${total} | REF ${delta}</span>
      </div>
      <div class="snapshot-grid">
        ${snap.original_overlay_url ? `<img src="${snap.original_overlay_url}" alt="Original overlay">` : '<div class="empty">No original overlay</div>'}
        ${snap.rectified_overlay_url ? `<img src="${snap.rectified_overlay_url}" alt="Rectified overlay">` : '<div class="empty">No rectified overlay</div>'}
      </div>
    </article>`;
  }).join('')}</div>`;
}

async function loadClip(clipId) {
  activeClip = clipId;
  document.querySelectorAll('.clip').forEach((el) => el.classList.toggle('active', el.dataset.clipId === clipId));
  const response = await fetch(`/api/live/clips/${clipId}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load clip');
  $('detail-title').textContent = data.clip_id;
  $('viewer').innerHTML = `
    <video controls src="${data.video_url}"></video>
    <div class="meta">
      ${metric('Saved', data.saved_at || '-')}
      ${metric('Duration', `${data.record_seconds ?? '-'} s`)}
      ${metric('PLC edge', data.event?.event_edge || '-')}
      ${metric('Snapshots', data.processing_snapshot_count ?? 0)}
    </div>
    ${renderSnapshots(data.processing_snapshots)}
  `;
}

async function loadHistory() {
  const response = await fetch('/api/live/clips');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load history');
  $('clip-count').textContent = `${data.count} saved clips`;
  if (!data.clips.length) {
    $('clip-list').innerHTML = '<div class="empty">No PLC videos saved yet.</div>';
    return;
  }
  $('clip-list').innerHTML = data.clips.map((clip) => `
    <button class="clip" data-clip-id="${clip.clip_id}">
      <strong>${clipLabel(clip)}</strong>
      <span>${clip.record_seconds ?? '-'} s | ${clip.processing_snapshot_count ?? 0} snapshots | ${clip.event?.event_edge || 'event'}</span>
    </button>
  `).join('');
  document.querySelectorAll('.clip').forEach((button) => button.addEventListener('click', () => loadClip(button.dataset.clipId)));
  const initial = data.clips.find((clip) => clip.clip_id === initialClipId) || data.clips[0];
  loadClip(initial.clip_id);
}

loadHistory().catch((err) => {
  $('clip-list').innerHTML = `<div class="empty">${err.message || err}</div>`;
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


def clips_root(output_dir: Path) -> Path:
    return output_dir / "live_plc_clips"


def clip_sidecars(output_dir: Path) -> list[Path]:
    root = clips_root(output_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def path_is_inside(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        resolved_root = root.resolve()
    except Exception:
        return False
    return resolved == resolved_root or resolved_root in resolved.parents


def read_clip_sidecar(json_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    data["clip_id"] = json_path.stem
    data["json_path"] = str(json_path)
    data["video_url"] = f"/api/live/clips/{json_path.stem}/video"
    data["detail_url"] = f"/history/{json_path.stem}"
    for snapshot in data.get("processing_snapshots", []) or []:
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


def delete_clip_artifacts(output_dir: Path, json_path: Path) -> None:
    root = clips_root(output_dir)
    data = read_clip_sidecar(json_path) or {}
    candidates: list[Path] = [json_path]
    for key in ("video_path", "analysis_dir"):
        value = data.get(key)
        if value:
            candidates.append(Path(value))
    for candidate in candidates:
        if not path_is_inside(candidate, root):
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
        except Exception:
            pass


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
    return jsonify(
        camera=_camera.snapshot(),
        processor=_processor.snapshot(include_images=False),
        plc=_plc.snapshot(),
        recorder=_recorder.snapshot(),
    )


@app.route("/api/live/frame")
def api_live_frame():
    data = _processor.snapshot(include_images=True)
    if data.get("result") is None:
        return jsonify(error=data.get("error") or "No processed frame is available yet", processor=data), 503
    return jsonify(data)


@app.route("/api/live/clips")
def api_live_clips():
    clips = []
    for json_path in clip_sidecars(_args.output_dir):
        data = read_clip_sidecar(json_path)
        if data is not None:
            clips.append(data)
    return jsonify(clips=clips[:100], count=len(clips))


@app.route("/api/live/clips/<clip_id>")
def api_live_clip(clip_id: str):
    json_path = find_clip_json(clip_id)
    if json_path is None:
        return jsonify(error="Clip not found"), 404
    data = read_clip_sidecar(json_path)
    if data is None:
        return jsonify(error="Clip metadata could not be read"), 500
    return jsonify(data)


@app.route("/api/live/clips/<clip_id>/video")
def api_live_clip_video(clip_id: str):
    json_path = find_clip_json(clip_id)
    if json_path is None:
        abort(404)
    data = read_clip_sidecar(json_path) or {}
    video_path = Path(str(data.get("video_path", "")))
    if not video_path.exists() or not path_is_inside(video_path, clips_root(_args.output_dir)):
        abort(404)
    return send_file(video_path, mimetype="video/mp4", conditional=True)


@app.route("/api/live/clips/<clip_id>/asset/<path:asset_name>")
def api_live_clip_asset(clip_id: str, asset_name: str):
    json_path = find_clip_json(clip_id)
    if json_path is None:
        abort(404)
    data = read_clip_sidecar(json_path) or {}
    analysis_dir = Path(str(data.get("analysis_dir", "")))
    asset_path = analysis_dir / asset_name
    if not asset_path.exists() or not path_is_inside(asset_path, analysis_dir) or not path_is_inside(asset_path, clips_root(_args.output_dir)):
        abort(404)
    return send_file(asset_path, mimetype="image/jpeg", conditional=True)


def main() -> int:
    global _args, _buffer, _camera, _processor, _recorder, _plc
    _args = parse_args()
    configure_vision_module(_args)

    buffer_len = int(max(8, float(_args.buffer_seconds) * max(1.0, float(_args.capture_fps))))
    _buffer = FrameBuffer(maxlen=buffer_len)
    _camera = CameraReader(_args, _buffer)
    _processor = LiveProcessor(_args, _buffer)
    _recorder = ClipRecorder(_args, _buffer, _processor)
    _plc = PLCMonitor(_args, _recorder)

    _camera.start()
    _processor.start()
    _plc.start()

    print(f"\n  TX2 Live MVP at http://127.0.0.1:{_args.port}\n")
    print(f"  Source: {_args.source}")
    print(f"  PLC: {'enabled' if _args.plc_enabled else 'disabled'}")
    app.run(host="127.0.0.1", port=_args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
