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
import queue
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

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
DEFAULT_PIECE_MODEL = ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v3" / "weights" / "best.pt"
PREVIOUS_PIECE_MODEL = ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v2" / "weights" / "best.pt"
OLDER_PIECE_MODEL = ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v1" / "weights" / "best.pt"
DEFAULT_LEGACY_MODEL = ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_tubos_v1" / "weights" / "best.pt"
DEFAULT_MODEL = next(
    (
        path
        for path in (
            DEFAULT_PIECE_MODEL,
            PREVIOUS_PIECE_MODEL,
            OLDER_PIECE_MODEL,
            DEFAULT_LEGACY_MODEL,
        )
        if path.exists()
    ),
    DEFAULT_PIECE_MODEL,
)
DEFAULT_ENDPOINT = "opc.tcp://10.14.6.48:49320"
DEFAULT_WATCHDOG_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD"
DEFAULT_EVENT_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength"
DEFAULT_PRE_TRIGGER_SECONDS = 2.0
DEFAULT_MEASUREMENT_DELAY_SECONDS = 0.0
MEASUREMENT_MARKER_DURATION_SECONDS = 0.8


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


def measurement_evidence_snapshots(
    snapshots: list[dict[str, Any]],
    *,
    event_monotonic: float | None = None,
    measurement_delay_seconds: float | None = None,
) -> list[dict[str, Any]]:
    if not snapshots:
        return []
    canonical = next(
        (
            snapshot
            for snapshot in snapshots
            if snapshot.get("is_canonical") is True
            or snapshot.get("is_canonical") == 1
        ),
        None,
    )
    if canonical is None:
        canonical = select_canonical_snapshot(
            snapshots,
            event_monotonic=event_monotonic,
            target_offset_seconds=measurement_delay_seconds,
        )
    return [canonical] if canonical is not None else []


def snapshots_contain_piece(snapshots: list[dict[str, Any]]) -> bool:
    for snapshot in snapshots:
        if snapshot_pieces(snapshot):
            return True
        detected_count = snapshot_summary(snapshot).get("detected_count")
        if isinstance(detected_count, (int, float, np.number)) and detected_count > 0:
            return True
    return False


def measurement_marker_active(
    sample_monotonic: float,
    event_monotonic: float,
    delay_seconds: float = DEFAULT_MEASUREMENT_DELAY_SECONDS,
    duration_seconds: float = MEASUREMENT_MARKER_DURATION_SECONDS,
) -> bool:
    marker_start = float(event_monotonic) + max(0.0, float(delay_seconds))
    marker_end = marker_start + max(0.0, float(duration_seconds))
    return marker_start <= float(sample_monotonic) < marker_end


def draw_measurement_perimeter(frame: np.ndarray) -> np.ndarray:
    marked = frame.copy()
    height, width = marked.shape[:2]
    thickness = max(4, int(round(min(height, width) * 0.012)))
    inset = max(1, thickness // 2)
    cv2.rectangle(
        marked,
        (inset, inset),
        (max(inset, width - inset - 1), max(inset, height - inset - 1)),
        (46, 204, 113),
        thickness,
        cv2.LINE_AA,
    )
    return marked


def mark_measurement_evidence_snapshot(snapshot: dict[str, Any] | None) -> bool:
    if snapshot is None:
        return False
    value = snapshot.get("original_overlay_path")
    if not value:
        return False
    image_path = Path(str(value))
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    if not cv2.imwrite(str(image_path), draw_measurement_perimeter(image)):
        raise RuntimeError(f"Could not mark measurement evidence: {image_path}")
    snapshot["measurement_evidence"] = True
    return True


def img_to_jpeg(img: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Could not encode image")
    return buf.tobytes()


def img_to_b64(img: np.ndarray, quality: int = 82) -> str:
    return base64.b64encode(img_to_jpeg(img, quality=quality)).decode("ascii")


def compact_live_result(result: dict[str, Any]) -> dict[str, Any]:
    pieces: list[dict[str, Any]] = []
    for piece in result.get("pieces") or []:
        box = piece.get("box") or {}
        line = (piece.get("sobel") or {}).get("line") or {}
        measurement = piece.get("measurement") or {}
        pieces.append(
            {
                "piece_id": piece.get("piece_id"),
                "box": {
                    key: clean_value(box.get(key))
                    for key in ("x", "y", "w", "h", "conf")
                    if box.get(key) is not None
                },
                "sobel": {
                    "line": {"y": clean_value(line.get("y"))}
                    if line.get("y") is not None
                    else None
                },
                "measurement": {
                    "measurement_in": clean_value(measurement.get("measurement_in"))
                }
                if measurement.get("measurement_in") is not None
                else None,
                "valid": bool(piece.get("valid")),
            }
        )

    calibration = result.get("calibration") or {}
    keys = (
        "frame_index",
        "frame_utc",
        "processed_utc",
        "original_width",
        "original_height",
        "rectified_width",
        "rectified_height",
        "count",
        "measurement_summary",
        "measurement",
        "front_y_ratio",
        "conf",
    )
    payload = {
        key: clean_value(result.get(key))
        for key in keys
        if result.get(key) is not None
    }
    payload["pieces"] = pieces
    payload["calibration"] = {
        "reference_y": clean_value(calibration.get("reference_y"))
    }
    return payload


def compact_recorder_status(status: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "recording",
        "active_recordings",
        "clip_index",
        "discarded_clip_count",
        "failed_recording_count",
        "error",
        "record_seconds",
        "pre_trigger_seconds",
        "measurement_delay_seconds",
        "measurement_marker_active",
        "save_raw_clips",
        "raw_camera_resolution",
        "raw_record_fps",
        "video_encoder_requested",
        "video_encoder",
        "video_encoder_error",
        "database_enabled",
    )
    return {
        key: clean_value(status.get(key))
        for key in keys
        if key in status
    }


def compact_processor_status(status: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "ok",
        "error",
        "processing",
        "processing_mode",
        "processed_count",
        "last_frame_index",
        "last_processed_utc",
        "last_duration_ms",
        "inference_device",
        "inference_device_name",
        "cuda_available",
        "homography_backend",
    )
    return {
        key: clean_value(status.get(key))
        for key in keys
        if key in status
    }


def plc_status_with_signal_state(status: dict[str, Any]) -> dict[str, Any]:
    payload = status.copy()
    trigger = payload.get("last_trigger")
    event_monotonic = (
        trigger.get("event_read_monotonic")
        if isinstance(trigger, dict)
        else None
    )
    try:
        trigger_age_seconds = max(
            0.0,
            time.perf_counter() - float(event_monotonic),
        )
    except (TypeError, ValueError):
        trigger_age_seconds = None
    payload["last_trigger_age_seconds"] = clean_value(trigger_age_seconds)
    payload["signal_recent"] = bool(
        trigger_age_seconds is not None and trigger_age_seconds < 4.0
    )
    return payload


def resolution_dimensions(value: str) -> tuple[int, int]:
    parts = str(value).lower().split("x", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid camera resolution: {value}")
    width, height = (int(part.strip()) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid camera resolution: {value}")
    return width, height


def bundled_ffmpeg_executable() -> Path | None:
    try:
        import imageio_ffmpeg

        path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, OSError, RuntimeError):
        return None
    return path if path.exists() else None


def build_nvdec_camera_command(
    ffmpeg_executable: Path,
    source: str,
    output_fps: float,
) -> list[str]:
    fps_filter = f"fps={max(1.0, float(output_fps)):g}"
    return [
        str(ffmpeg_executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-c:v",
        "h264_cuvid",
        "-i",
        source,
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"{fps_filter},hwdownload,format=nv12,format=bgr24",
        "-pix_fmt",
        "bgr24",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def build_live_stream_command(
    ffmpeg_executable: Path,
    source: str,
    *,
    rtsp_source: bool,
) -> list[str]:
    command = [
        str(ffmpeg_executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if rtsp_source:
        command.extend(
            (
                "-rtsp_transport",
                "tcp",
                "-analyzeduration",
                "0",
                "-probesize",
                "32768",
            )
        )
    else:
        command.extend(("-stream_loop", "-1", "-re"))
    command.extend(
        (
            "-i",
            source,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "copy",
            "-movflags",
            "frag_every_frame+empty_moov+default_base_moof",
            "-avoid_negative_ts",
            "make_zero",
            "-flush_packets",
            "1",
            "-f",
            "mp4",
            "pipe:1",
        )
    )
    return command


def build_nvenc_writer_command(
    ffmpeg_executable: Path,
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
    bitrate_mbps: float,
) -> list[str]:
    width, height = frame_size
    bitrate = max(1.0, float(bitrate_mbps))
    return [
        str(ffmpeg_executable),
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{int(width)}x{int(height)}",
        "-framerate",
        f"{float(fps):g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "19",
        "-b:v",
        f"{bitrate:g}M",
        "-maxrate",
        f"{bitrate * 2.0:g}M",
        "-bufsize",
        f"{bitrate * 2.0:g}M",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def validate_nvenc(ffmpeg_executable: Path) -> tuple[bool, str]:
    command = [
        str(ffmpeg_executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=256x256:rate=1",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15.0,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return completed.returncode == 0, completed.stderr.strip()


class NvencVideoWriter:
    def __init__(
        self,
        ffmpeg_executable: Path,
        output_path: Path,
        fps: float,
        frame_size: tuple[int, int],
        bitrate_mbps: float,
    ) -> None:
        self.output_path = output_path
        self.frame_size = tuple(int(value) for value in frame_size)
        self.process = subprocess.Popen(
            build_nvenc_writer_command(
                ffmpeg_executable,
                output_path,
                fps,
                self.frame_size,
                bitrate_mbps,
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
        self.closed = False
        self.frame_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=4)
        self.worker_error = ""
        self.worker_frame_count = 0
        self.worker_write_duration_ms = 0.0
        self.max_queue_depth = 0
        self.worker = threading.Thread(
            target=self._write_frames,
            name=f"nvenc-writer-{output_path.stem}",
            daemon=True,
        )
        self.worker.start()

    def isOpened(self) -> bool:
        return (
            not self.closed
            and not self.worker_error
            and self.process.poll() is None
            and self.process.stdin is not None
        )

    def _write_frames(self) -> None:
        try:
            while True:
                frame = self.frame_queue.get()
                try:
                    if frame is None:
                        return
                    if self.process.stdin is None:
                        raise RuntimeError("NVENC stdin is unavailable")
                    started = time.perf_counter()
                    self.process.stdin.write(memoryview(frame).cast("B"))
                    self.worker_write_duration_ms += (
                        time.perf_counter() - started
                    ) * 1000.0
                    self.worker_frame_count += 1
                finally:
                    self.frame_queue.task_done()
        except (BrokenPipeError, OSError, RuntimeError) as exc:
            self.worker_error = str(exc) or self._failure_message()

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            raise RuntimeError(f"NVENC writer is not open: {self.output_path}")
        height, width = frame.shape[:2]
        if (width, height) != self.frame_size:
            raise RuntimeError(
                "NVENC frame size changed from "
                f"{self.frame_size[0]}x{self.frame_size[1]} to {width}x{height}"
            )
        contiguous = np.ascontiguousarray(frame)
        while True:
            if self.worker_error:
                raise RuntimeError(self.worker_error)
            if self.process.poll() is not None:
                raise RuntimeError(self._failure_message())
            try:
                self.frame_queue.put(contiguous, timeout=0.5)
                self.max_queue_depth = max(
                    self.max_queue_depth,
                    self.frame_queue.qsize(),
                )
                return
            except queue.Full:
                continue

    def _failure_message(self) -> str:
        error = b""
        if self.process.stderr is not None and self.process.poll() is not None:
            error = self.process.stderr.read()
        detail = error.decode("utf-8", errors="replace").strip()
        return detail or f"NVENC failed while writing {self.output_path}"

    def release(self) -> None:
        if self.closed:
            return
        self.closed = True
        while self.worker.is_alive():
            try:
                self.frame_queue.put(None, timeout=0.5)
                break
            except queue.Full:
                continue
        self.worker.join(timeout=30.0)
        if self.worker.is_alive():
            self.process.kill()
            self.worker.join(timeout=5.0)
            raise RuntimeError(f"NVENC writer queue did not finish {self.output_path}")
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            return_code = self.process.wait(timeout=30.0)
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            self.process.wait(timeout=5.0)
            raise RuntimeError(f"NVENC did not finish {self.output_path}") from exc
        error = self.process.stderr.read() if self.process.stderr is not None else b""
        if self.worker_error:
            detail = error.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or self.worker_error)
        if return_code != 0:
            detail = error.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or f"NVENC failed with exit code {return_code}")

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)
        while True:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except queue.Empty:
                break
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=3.0)
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live TX2 MVP with camera, PLC, YOLO and recording.")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--source", choices=("video", "rtsp", "auto"), default="rtsp")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--camera-ip", default="10.14.115.241")
    parser.add_argument("--rtsp-url", default=os.environ.get("AXIS_RTSP_URL", ""))
    parser.add_argument("--codec", choices=("jpeg", "h264"), default="h264")
    parser.add_argument("--camera-resolution", default="2880x2160")
    parser.add_argument(
        "--camera-decoder",
        choices=("auto", "nvdec", "opencv"),
        default="auto",
    )
    parser.add_argument("--camera-user", default=os.environ.get("AXIS_USER", ""))
    parser.add_argument("--camera-password", default=os.environ.get("AXIS_PASSWORD", ""))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        default=os.environ.get("TX2_YOLO_DEVICE", "auto"),
        help="YOLO inference device: auto, cpu, cuda or cuda:N.",
    )
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--capture-fps", type=float, default=10.0)
    parser.add_argument("--live-stream-fps", type=float, default=10.0)
    parser.add_argument(
        "--live-rtsp-url",
        default=os.environ.get("AXIS_LIVE_RTSP_URL", ""),
        help="Optional RTSP URL used only by the browser live stream.",
    )
    parser.add_argument("--buffer-seconds", type=float, default=3.0)
    parser.add_argument("--buffer-max-frames", type=int, default=60)
    parser.add_argument("--record-seconds", type=float, default=8.0)
    parser.add_argument("--record-fps", type=float, default=10.0)
    parser.add_argument(
        "--video-encoder",
        choices=("auto", "nvenc", "opencv"),
        default="auto",
    )
    parser.add_argument("--video-bitrate-mbps", type=float, default=16.0)
    parser.add_argument("--processing-cache-frames", type=int, default=160)
    parser.add_argument(
        "--pre-trigger-seconds",
        type=float,
        default=DEFAULT_PRE_TRIGGER_SECONDS,
        help="Seconds kept before the PLC event inside each fixed-duration clip.",
    )
    parser.add_argument(
        "--save-raw-clips",
        action="store_true",
        help="Temporarily save a camera-only MP4 beside each processed clip.",
    )
    parser.add_argument(
        "--raw-rtsp-url",
        default=os.environ.get("AXIS_RAW_RTSP_URL", ""),
        help="Optional RTSP URL used only for temporary raw recordings.",
    )
    parser.add_argument("--raw-camera-resolution", default="2880x2160")
    parser.add_argument("--raw-record-fps", type=float, default=10.0)
    parser.add_argument(
        "--measurement-delay-seconds",
        type=float,
        default=DEFAULT_MEASUREMENT_DELAY_SECONDS,
    )
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
    args = parser.parse_args(argv)
    if args.pre_trigger_seconds < 0:
        parser.error("--pre-trigger-seconds must be zero or greater")
    if args.pre_trigger_seconds >= args.record_seconds:
        parser.error("--pre-trigger-seconds must be shorter than --record-seconds")
    if args.buffer_seconds < args.pre_trigger_seconds:
        parser.error("--buffer-seconds must be at least --pre-trigger-seconds")
    if args.measurement_delay_seconds < 0:
        parser.error("--measurement-delay-seconds must be zero or greater")
    if args.measurement_delay_seconds >= (
        args.record_seconds - args.pre_trigger_seconds
    ):
        parser.error(
            "--measurement-delay-seconds must fit after the PLC event inside the clip"
        )
    if args.raw_record_fps <= 0 or args.raw_record_fps > 60:
        parser.error("--raw-record-fps must be greater than zero and no more than 60")
    if args.video_bitrate_mbps <= 0:
        parser.error("--video-bitrate-mbps must be greater than zero")
    if args.processing_cache_frames < 1:
        parser.error("--processing-cache-frames must be at least one")
    for option, value in (
        ("--capture-fps", args.capture_fps),
        ("--live-stream-fps", args.live_stream_fps),
        ("--record-fps", args.record_fps),
    ):
        if value <= 0 or value > 60:
            parser.error(f"{option} must be greater than zero and no more than 60")
    return args


def _axis_rtsp_url(
    args: argparse.Namespace,
    resolution: str,
    fps: float,
    explicit_url: str = "",
) -> str:
    if explicit_url:
        return explicit_url
    auth = ""
    if args.camera_user and args.camera_password:
        user = quote(str(args.camera_user), safe="")
        password = quote(str(args.camera_password), safe="")
        auth = f"{user}:{password}@"
    requested_fps = max(1, int(round(float(fps))))
    return (
        f"rtsp://{auth}{args.camera_ip}/axis-media/media.amp"
        f"?videocodec={args.codec}&resolution={resolution}&fps={requested_fps}"
    )


def build_rtsp_url(args: argparse.Namespace) -> str:
    return _axis_rtsp_url(
        args,
        str(args.camera_resolution),
        float(args.capture_fps),
        str(args.rtsp_url),
    )


def build_live_rtsp_url(args: argparse.Namespace) -> str:
    url = _axis_rtsp_url(
        args,
        str(args.camera_resolution),
        float(args.live_stream_fps),
        str(args.live_rtsp_url),
    )
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}videozfpsmode=fixed"


def resolve_live_stream_source(args: argparse.Namespace) -> tuple[str, bool]:
    if args.source == "rtsp":
        return build_live_rtsp_url(args), True
    if args.source == "auto" and (
        args.live_rtsp_url
        or args.rtsp_url
        or (args.camera_user and args.camera_password)
    ):
        return build_live_rtsp_url(args), True
    return str(args.video), False


def build_raw_rtsp_url(args: argparse.Namespace) -> str:
    url = _axis_rtsp_url(
        args,
        str(getattr(args, "raw_camera_resolution", "2880x2160")),
        float(getattr(args, "raw_record_fps", 30.0)),
        str(getattr(args, "raw_rtsp_url", "")),
    )
    separator = "&" if "?" in url else "?"
    return (
        f"{url}{separator}videozfpsmode=fixed"
        "&videokeyframeinterval=30"
    )


def direct_raw_capture_enabled(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "save_raw_clips", False)
        and str(getattr(args, "source", "")) in ("rtsp", "auto")
    )


def start_direct_raw_capture(
    args: argparse.Namespace,
    output_path: Path,
    duration_seconds: float,
) -> subprocess.Popen:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as exc:
        raise RuntimeError(
            "Temporary raw recording requires imageio-ffmpeg. "
            "Install the repository requirements."
        ) from exc

    command = [
        get_ffmpeg_exe(),
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        build_raw_rtsp_url(args),
        "-t",
        f"{float(duration_seconds):.3f}",
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
    )


def finish_direct_raw_capture(
    process: subprocess.Popen,
    output_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        process.communicate(timeout=max(5.0, timeout_seconds))
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.communicate(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        return {"ok": False, "error": "Raw RTSP copy timed out"}

    if process.returncode != 0 or not output_path.is_file() or output_path.stat().st_size <= 0:
        return {
            "ok": False,
            "error": f"Raw RTSP copy failed with exit code {process.returncode}",
        }

    cap = cv2.VideoCapture(str(output_path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    duration = frame_count / fps if frame_count > 0 and fps > 0 else None
    return {
        "ok": True,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "size_bytes": output_path.stat().st_size,
    }


def stop_direct_raw_capture(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.communicate(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()


def configure_vision_module(args: argparse.Namespace) -> None:
    vision._args = SimpleNamespace(
        video=args.video,
        second=0.0,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        model=args.model,
        device=args.device,
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

    def frames_between(
        self,
        start_monotonic: float,
        end_monotonic: float,
    ) -> list[dict[str, Any]]:
        with self._lock:
            selected = [
                item
                for item in self._frames
                if float(start_monotonic)
                <= float(item["monotonic"])
                < float(end_monotonic)
            ]
            return [item.copy() for item in selected]

    def latest_at_or_before(self, target_monotonic: float) -> dict[str, Any] | None:
        with self._lock:
            candidates = [
                item
                for item in self._frames
                if float(item["monotonic"]) <= float(target_monotonic)
            ]
            if not candidates:
                return None
            return max(
                candidates,
                key=lambda item: float(item["monotonic"]),
            ).copy()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "count": len(self._frames),
                "first_index": int(self._frames[0]["index"]) if self._frames else None,
                "last_index": int(self._frames[-1]["index"]) if self._frames else None,
            }


class CameraReader:
    def __init__(
        self,
        args: argparse.Namespace,
        buffer: FrameBuffer,
    ) -> None:
        self.args = args
        self.buffer = buffer
        self.next_buffer_monotonic: float | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="live-camera-reader", daemon=True)
        self.lock = threading.Lock()
        self.capture_process_lock = threading.Lock()
        self.capture_process: subprocess.Popen[bytes] | None = None
        self.state: dict[str, Any] = {
            "connected": False,
            "source": args.source,
            "source_label": "",
            "decoder": "pending",
            "error": "",
            "frames_read": 0,
            "fps": args.record_fps,
            "width": None,
            "height": None,
            "last_frame_utc": None,
        }

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.capture_process_lock:
            capture_process = self.capture_process
        if capture_process is not None and capture_process.poll() is None:
            capture_process.terminate()
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            data = self.state.copy()
        data["buffer"] = self.buffer.stats()
        return data

    def _set_state(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates)

    def _publish_frame(self, item: dict[str, Any]) -> None:
        frame_monotonic = float(item["monotonic"])
        interval = 1.0 / max(1.0, float(self.args.record_fps))
        if self.next_buffer_monotonic is None:
            self.buffer.append(item)
            self.next_buffer_monotonic = frame_monotonic + interval
            return
        if frame_monotonic < self.next_buffer_monotonic:
            return

        self.buffer.append(item)
        elapsed_intervals = int(
            (frame_monotonic - self.next_buffer_monotonic) // interval
        )
        self.next_buffer_monotonic += (elapsed_intervals + 1) * interval

    def _resolve_source(self) -> tuple[str, str]:
        if self.args.source == "rtsp":
            return build_rtsp_url(self.args), f"RTSP {self.args.camera_ip}"
        if self.args.source == "auto" and (self.args.rtsp_url or (self.args.camera_user and self.args.camera_password)):
            return build_rtsp_url(self.args), f"RTSP {self.args.camera_ip}"
        return str(self.args.video), f"Simulated video {self.args.video.name}"

    def _read_nvdec(
        self,
        source: str,
        label: str,
        frame_index: int,
    ) -> tuple[int, bool]:
        decoder_mode = str(getattr(self.args, "camera_decoder", "auto"))
        if (
            decoder_mode == "opencv"
            or not label.startswith("RTSP")
            or str(getattr(self.args, "codec", "h264")).lower() != "h264"
        ):
            return frame_index, False
        ffmpeg_executable = bundled_ffmpeg_executable()
        if ffmpeg_executable is None:
            return frame_index, False
        try:
            width, height = resolution_dimensions(self.args.camera_resolution)
        except (TypeError, ValueError):
            return frame_index, False

        frame_bytes = width * height * 3
        process = subprocess.Popen(
            build_nvdec_camera_command(
                ffmpeg_executable,
                source,
                float(self.args.record_fps),
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=frame_bytes * 2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with self.capture_process_lock:
            self.capture_process = process

        frames_received = 0
        try:
            if process.stdout is None:
                return frame_index, False
            while not self.stop_event.is_set():
                data = bytearray()
                while len(data) < frame_bytes and not self.stop_event.is_set():
                    chunk = process.stdout.read(frame_bytes - len(data))
                    if not chunk:
                        break
                    data.extend(chunk)
                if len(data) != frame_bytes:
                    break

                frame = np.frombuffer(data, dtype=np.uint8).reshape(
                    (height, width, 3)
                )
                item = {
                    "index": frame_index,
                    "utc": utc_now(),
                    "monotonic": time.perf_counter(),
                    "frame": frame,
                }
                self.buffer.append(item)
                frame_index += 1
                frames_received += 1
                self._set_state(
                    connected=True,
                    source_label=label,
                    decoder="nvidia_nvdec",
                    error="",
                    frames_read=frame_index,
                    fps=float(self.args.record_fps),
                    width=width,
                    height=height,
                    last_frame_utc=item["utc"],
                )
        finally:
            with self.capture_process_lock:
                if self.capture_process is process:
                    self.capture_process = None
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)

        if frames_received:
            self._set_state(
                connected=False,
                error="NVDEC camera read stopped; retrying",
            )
        return frame_index, frames_received > 0

    def _run(self) -> None:
        frame_index = 0
        while not self.stop_event.is_set():
            source, label = self._resolve_source()
            pending_error = "Connecting to video source..."
            if label.startswith("RTSP") and not (self.args.rtsp_url or (self.args.camera_user and self.args.camera_password)):
                pending_error = "Connecting to RTSP without credentials. If the camera requires auth, set AXIS_USER and AXIS_PASSWORD."
            self._set_state(connected=False, source_label=label, error=pending_error)

            frame_index, used_nvdec = self._read_nvdec(
                source,
                label,
                frame_index,
            )
            if used_nvdec:
                time.sleep(0.4)
                continue

            cap = cv2.VideoCapture()
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
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
            self._set_state(
                connected=True,
                source_label=label,
                decoder="opencv_ffmpeg",
                error="",
                fps=float(self.args.record_fps),
                width=width,
                height=height,
            )

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
                self._publish_frame(item)
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
        device_info = vision.resolve_yolo_device(getattr(args, "device", "auto"))
        self.inference_lock = threading.Lock()
        self.lock = threading.Lock()
        self.result_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self.result_cache_limit = max(
            1,
            int(getattr(args, "processing_cache_frames", 160)),
        )
        self.state: dict[str, Any] = {
            "ok": True,
            "processing": False,
            "processing_mode": "plc_triggered_clip",
            "error": "",
            "processed_count": 0,
            "last_frame_index": None,
            "last_processed_utc": None,
            "last_duration_ms": None,
            "inference_device": device_info["device"],
            "inference_device_name": device_info["device_name"],
            "cuda_available": device_info["cuda_available"],
            "torch_version": device_info["torch_version"],
            "torch_cuda_version": device_info["torch_cuda_version"],
            "homography_backend": "opencv_cpu",
            "processing_cache_hits": 0,
            "processing_cache_misses": 0,
            "last_stage_durations_ms": None,
            "result": None,
        }

    def snapshot(self, include_images: bool = False) -> dict[str, Any]:
        with self.lock:
            data = self.state.copy()
            result = self.state.get("result")
            if isinstance(result, dict):
                if include_images:
                    data["result"] = {
                        key: value
                        for key, value in result.items()
                        if key
                        not in (
                            "_recording_frame",
                            "_original_jpeg",
                            "_original_overlay",
                        )
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

    def process_clip_frame(
        self,
        item: dict[str, Any],
        *,
        include_evidence_images: bool = False,
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        self._set_state(processing=True)
        try:
            with self.inference_lock:
                cache_key = self._cache_key(item)
                cached = self.result_cache.get(cache_key)
                if cached is None:
                    result = self._process(
                        item,
                        include_evidence_images=include_evidence_images,
                    )
                    self._cache_result(cache_key, result)
                    cache_hit = False
                else:
                    self.result_cache.move_to_end(cache_key)
                    result = self._materialize_cached_result(
                        cached,
                        item,
                        include_evidence_images=include_evidence_images,
                    )
                    cache_hit = True
                result["processing_cache_hit"] = cache_hit
            duration_ms = round((time.perf_counter() - started) * 1000.0, 1)
            result_frame_index = int(result["frame_index"])
            with self.lock:
                updates = dict(
                    ok=True,
                    processing=False,
                    error="",
                    processed_count=int(self.state.get("processed_count", 0)) + 1,
                    last_processed_utc=utc_now(),
                    last_duration_ms=duration_ms,
                    homography_backend="opencv_cpu",
                    last_stage_durations_ms=result.get("stage_durations_ms"),
                )
                cache_state_key = (
                    "processing_cache_hits"
                    if cache_hit
                    else "processing_cache_misses"
                )
                updates[cache_state_key] = int(self.state.get(cache_state_key, 0)) + 1
                last_frame_index = self.state.get("last_frame_index")
                if last_frame_index is None or result_frame_index >= int(last_frame_index):
                    updates.update(
                        last_frame_index=result_frame_index,
                        result=result,
                    )
                self.state.update(updates)
            return result, duration_ms
        except Exception as exc:
            self._set_state(ok=False, processing=False, error=str(exc))
            raise

    def _set_state(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates)

    def _cache_key(self, item: dict[str, Any]) -> tuple[Any, ...]:
        paths = []
        output_dir = getattr(self.args, "output_dir", None)
        if output_dir is not None:
            paths.extend(
                (
                    Path(output_dir) / "homography_selection.json",
                    Path(output_dir) / "table_measurement_calibration.json",
                )
            )
        model = getattr(self.args, "model", None)
        if model is not None:
            paths.append(Path(model))
        fingerprint: list[tuple[str, int | None, int | None]] = []
        for path in paths:
            try:
                stat = path.stat()
                fingerprint.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                fingerprint.append((str(path), None, None))
        return int(item["index"]), tuple(fingerprint)

    def _cache_result(
        self,
        key: tuple[Any, ...],
        result: dict[str, Any],
    ) -> None:
        self.result_cache[key] = {
            name: value
            for name, value in result.items()
            if name
            not in (
                "_recording_frame",
                "original_image",
                "rectified_image",
                "processing_cache_hit",
            )
        }
        self.result_cache.move_to_end(key)
        while len(self.result_cache) > self.result_cache_limit:
            self.result_cache.popitem(last=False)

    def _materialize_cached_result(
        self,
        cached: dict[str, Any],
        item: dict[str, Any],
        *,
        include_evidence_images: bool,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result = cached.copy()
        original = item["frame"]
        original_viz = draw_original_overlay(
            original,
            result.get("_original_overlay") or {},
        )
        result["_recording_frame"] = {
            "index": int(item["index"]),
            "utc": item["utc"],
            "monotonic": float(item["monotonic"]),
            "frame": original_viz,
            "raw_frame": original,
        }
        stage_durations = {
            "cache_materialize": round(
                (time.perf_counter() - started) * 1000.0,
                2,
            )
        }
        if include_evidence_images:
            evidence_started = time.perf_counter()
            matrix, out_size, _homography = vision.load_homography()
            rectified = cv2.warpPerspective(original, matrix, out_size)
            rectified_viz = draw_rectified_overlay(
                rectified,
                result.get("pieces") or [],
                result.get("calibration") or {},
            )
            result["original_image"] = base64.b64encode(
                img_to_jpeg(original_viz, quality=80)
            ).decode("ascii")
            result["rectified_image"] = img_to_b64(rectified_viz, quality=82)
            stage_durations["evidence"] = round(
                (time.perf_counter() - evidence_started) * 1000.0,
                2,
            )
        result["stage_durations_ms"] = stage_durations
        return result

    def _process(
        self,
        item: dict[str, Any],
        *,
        include_evidence_images: bool = False,
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        stage_started = total_started
        original = item["frame"]
        matrix, out_size, _homography = vision.load_homography()
        calibration = vision.load_measurement_calibration()
        stage_durations = {
            "configuration": round(
                (time.perf_counter() - stage_started) * 1000.0,
                2,
            )
        }
        stage_started = time.perf_counter()
        rectified = cv2.warpPerspective(original, matrix, out_size)
        stage_durations["homography"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            2,
        )
        stage_started = time.perf_counter()
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
        stage_durations["yolo_and_rules"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            2,
        )

        rect_h, rect_w = rectified.shape[:2]
        src_h, src_w = original.shape[:2]
        stage_started = time.perf_counter()
        pieces, measurement_summary = vision.analyze_piece_boxes(
            rectified,
            boxes,
            calibration,
            frame_idx=int(item["index"]),
            time_sec=None,
        )
        stage_durations["sobel_and_measurement"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            2,
        )
        primary = vision.primary_piece_analysis(pieces)
        sobel = primary["sobel"] if primary else vision.empty_sobel_result(int(item["index"]), None)
        measurement = primary["measurement"] if primary else None
        stage_started = time.perf_counter()
        original_overlay = vision.mvp_original_overlay_for_pieces(pieces, calibration, matrix, rect_w)
        original_viz = draw_original_overlay(original, original_overlay)
        stage_durations["overlay"] = round(
            (time.perf_counter() - stage_started) * 1000.0,
            2,
        )

        result = {
            "frame_index": int(item["index"]),
            "frame_utc": item["utc"],
            "frame_monotonic": float(item["monotonic"]),
            "processed_utc": utc_now(),
            "original_width": src_w,
            "original_height": src_h,
            "rectified_width": rect_w,
            "rectified_height": rect_h,
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
            "homography_backend": "opencv_cpu",
            "_original_overlay": original_overlay,
            "_recording_frame": {
                "index": int(item["index"]),
                "utc": item["utc"],
                "monotonic": float(item["monotonic"]),
                "frame": original_viz,
                "raw_frame": original,
            },
        }
        if include_evidence_images:
            stage_started = time.perf_counter()
            rectified_viz = draw_rectified_overlay(rectified, pieces, calibration)
            result["original_image"] = base64.b64encode(
                img_to_jpeg(original_viz, quality=80)
            ).decode("ascii")
            result["rectified_image"] = img_to_b64(rectified_viz, quality=82)
            stage_durations["evidence"] = round(
                (time.perf_counter() - stage_started) * 1000.0,
                2,
            )
        stage_durations["total"] = round(
            (time.perf_counter() - total_started) * 1000.0,
            2,
        )
        result["stage_durations_ms"] = stage_durations
        return result


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
        self.configuration_fingerprint: tuple[tuple[str, int | None, int | None], ...] | None = None
        self.active_recordings: dict[int, float] = {}
        self.clip_index = 0
        self.last_clip: dict[str, Any] | None = None
        self.discarded_clip_count = 0
        self.last_discarded_clip: dict[str, Any] | None = None
        self.failed_recordings: deque[dict[str, Any]] = deque(maxlen=20)
        self.error = ""
        self.video_encoder_requested = str(
            getattr(args, "video_encoder", "opencv")
        )
        self.video_encoder = "opencv"
        self.video_encoder_error = ""
        self.video_ffmpeg_executable: Path | None = None
        self.video_encoder_configured = False
        self.stop_event = threading.Event()
        self.recording_threads: set[threading.Thread] = set()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.perf_counter()
            measurement_delay_seconds = self._measurement_delay_seconds()
            pre_trigger_seconds = self._pre_trigger_seconds()
            measurement_marker_active_now = any(
                measurement_marker_active(
                    now,
                    event_monotonic,
                    measurement_delay_seconds,
                )
                for event_monotonic in self.active_recordings.values()
            )
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
                "pre_trigger_seconds": pre_trigger_seconds,
                "save_raw_clips": bool(
                    getattr(self.args, "save_raw_clips", False)
                ),
                "raw_camera_resolution": str(
                    getattr(self.args, "raw_camera_resolution", "2880x2160")
                ),
                "raw_record_fps": float(
                    getattr(self.args, "raw_record_fps", 30.0)
                ),
                "video_encoder_requested": self.video_encoder_requested,
                "video_encoder": self.video_encoder,
                "video_encoder_error": self.video_encoder_error,
                "measurement_delay_seconds": measurement_delay_seconds,
                "measurement_marker_active": measurement_marker_active_now,
                "database_enabled": self.database is not None,
                "shutting_down": self.stop_event.is_set(),
            }

    def start_event_clip(self, event: dict[str, Any]) -> None:
        event = dict(event)
        event_monotonic = float(event.get("event_read_monotonic") or time.perf_counter())
        event["event_read_monotonic"] = event_monotonic
        with self.lock:
            if self.stop_event.is_set():
                return
            self.clip_index += 1
            clip_index = self.clip_index
            self.active_recordings[clip_index] = event_monotonic
            thread = threading.Thread(
                target=self._run_recording_thread,
                args=(clip_index, event),
                name=f"plc-clip-recorder-{clip_index}",
                daemon=True,
            )
            self.recording_threads.add(thread)
        try:
            thread.start()
        except Exception:
            with self.lock:
                self.recording_threads.discard(thread)
                self.active_recordings.pop(clip_index, None)
            raise

    def _run_recording_thread(self, clip_index: int, event: dict[str, Any]) -> None:
        try:
            self._record_clip(clip_index, event)
        finally:
            with self.lock:
                self.recording_threads.discard(threading.current_thread())

    def stop(self, timeout_seconds: float | None = None) -> None:
        self.stop_event.set()
        timeout = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else max(15.0, float(self.args.record_seconds) + 10.0)
        )
        deadline = time.perf_counter() + timeout
        while True:
            with self.lock:
                threads = [
                    thread
                    for thread in self.recording_threads
                    if thread.is_alive()
                ]
            if not threads:
                return
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise RuntimeError(
                    f"Timed out waiting for {len(threads)} active recording(s)"
                )
            for thread in threads:
                thread.join(timeout=min(remaining, 0.5))

    def _measurement_delay_seconds(self) -> float:
        return max(
            0.0,
            float(
                getattr(
                    self.args,
                    "measurement_delay_seconds",
                    DEFAULT_MEASUREMENT_DELAY_SECONDS,
                )
            ),
        )

    def _pre_trigger_seconds(self) -> float:
        return max(
            0.0,
            float(getattr(self.args, "pre_trigger_seconds", 0.0)),
        )

    def _configuration_files_fingerprint(
        self,
    ) -> tuple[tuple[str, int | None, int | None], ...]:
        output_dir = Path(self.args.output_dir).resolve()
        paths = (
            output_dir / "homography_selection.json",
            output_dir / "table_measurement_calibration.json",
            Path(self.args.model).resolve(),
        )
        fingerprint = []
        for path in paths:
            try:
                stat = path.stat()
                fingerprint.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                fingerprint.append((str(path), None, None))
        return tuple(fingerprint)

    def _configuration_snapshot(self) -> dict[str, Any] | None:
        with self.configuration_lock:
            if self.vision_configuration is not None:
                if self.configuration_fingerprint is None:
                    return self.vision_configuration
                fingerprint = self._configuration_files_fingerprint()
                if fingerprint == self.configuration_fingerprint:
                    return self.vision_configuration
            else:
                fingerprint = self._configuration_files_fingerprint()
            try:
                self.vision_configuration = build_vision_configuration(self.args, ROOT)
                self.configuration_fingerprint = fingerprint
            except (AttributeError, FileNotFoundError, ValueError):
                if self.database is not None:
                    raise
                return None
            return self.vision_configuration

    def validate_configuration(self) -> dict[str, Any]:
        configuration = self._configuration_snapshot()
        if configuration is None:
            raise RuntimeError("The vision configuration could not be snapshotted")
        self._configure_video_encoder()
        return configuration

    def _configure_video_encoder(self) -> None:
        if self.video_encoder_configured:
            return
        self.video_encoder_configured = True
        if self.video_encoder_requested == "opencv":
            return
        ffmpeg_executable = bundled_ffmpeg_executable()
        if ffmpeg_executable is None:
            self.video_encoder_error = "FFmpeg is unavailable"
        else:
            available, error = validate_nvenc(ffmpeg_executable)
            if available:
                self.video_ffmpeg_executable = ffmpeg_executable
                self.video_encoder = "nvidia_nvenc"
                self.video_encoder_error = ""
                return
            self.video_encoder_error = error or "NVIDIA NVENC is unavailable"
        if self.video_encoder_requested == "nvenc":
            raise RuntimeError(self.video_encoder_error)

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
        *,
        result_override: dict[str, Any] | None = None,
        processing_duration_ms: float | None = None,
    ) -> None:
        if self.processor is None:
            return
        processor_data: dict[str, Any] = {}
        if result_override is None:
            processor_data = self.processor.snapshot(include_images=True)
            result = processor_data.get("result")
        else:
            result = result_override
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
            if key
            not in (
                "original_image",
                "rectified_image",
                "_recording_frame",
                "_original_jpeg",
                "_original_overlay",
            )
        }
        snapshot["snapshot_index"] = snap_index
        snapshot["processing_duration_ms"] = clean_value(
            processing_duration_ms
            if processing_duration_ms is not None
            else processor_data.get("last_duration_ms")
        )

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
        event_source_frame = self.buffer.latest_at_or_before(event_mono)
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
        pre_trigger_seconds = min(
            self._pre_trigger_seconds(),
            max(0.0, record_seconds - 0.1),
        )
        clip_start_mono = event_mono - pre_trigger_seconds
        deadline = clip_start_mono + record_seconds
        post_trigger_seconds = max(0.0, deadline - event_mono)
        measurement_delay_seconds = self._measurement_delay_seconds()
        measurement_target_mono = event_mono + measurement_delay_seconds
        save_raw_clips = bool(getattr(self.args, "save_raw_clips", False))
        direct_raw_capture = direct_raw_capture_enabled(self.args)
        buffer_raw_capture = save_raw_clips and not direct_raw_capture
        fps = max(1.0, min(float(self.args.record_fps or self.args.capture_fps or 10.0), 60.0))
        target_frame_count = max(1, int(round(record_seconds * fps)))
        frames_written = 0
        source_frames_seen = 0
        last_index = -1
        last_source: dict[str, Any] | None = None
        first_written_source: dict[str, Any] | None = None
        last_written_source: dict[str, Any] | None = None
        writer: cv2.VideoWriter | NvencVideoWriter | None = None
        raw_writer: cv2.VideoWriter | None = None
        video_path: Path | None = None
        raw_video_path: Path | None = None
        raw_capture_process: subprocess.Popen | None = None
        raw_capture_result: dict[str, Any] | None = None
        raw_capture_started_at: str | None = None
        raw_capture_error = ""
        analysis_dir: Path | None = None
        json_path: Path | None = None
        measurement_marker_frames_written = 0
        measurement_marker_first_video_frame_index: int | None = None
        measurement_source_frame_offset_seconds: float | None = None
        event_frame_processing_error = ""
        event_recording_frame: dict[str, Any] | None = None
        processed_source_frame_count = 0
        clip_detected_piece = False
        video_write_duration_ms = 0.0
        video_queue_max_depth = 0
        historical_frames = (
            self.buffer.frames_between(clip_start_mono, event_mono)
            if pre_trigger_seconds > 0
            else []
        )
        post_frame_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        collector_stop = threading.Event()
        collector_done = threading.Event()
        collector_thread: threading.Thread | None = None
        initial_collected_index = max(
            [int(frame["index"]) for frame in historical_frames]
            + (
                [int(event_source_frame["index"])]
                if event_source_frame is not None
                else []
            )
            + [-1]
        )

        def collect_post_trigger_frames() -> None:
            last_collected_index = initial_collected_index
            try:
                while not collector_stop.is_set():
                    for camera_frame in self.buffer.frames_since(
                        last_collected_index
                    ):
                        last_collected_index = max(
                            last_collected_index,
                            int(camera_frame["index"]),
                        )
                        frame_monotonic = float(camera_frame["monotonic"])
                        if event_mono <= frame_monotonic < deadline:
                            post_frame_queue.put(camera_frame)

                    now = time.perf_counter()
                    latest = self.buffer.latest()
                    latest_monotonic = (
                        float(latest["monotonic"])
                        if latest is not None
                        else None
                    )
                    if now >= deadline and (
                        latest_monotonic is None
                        or latest_monotonic >= deadline
                        or now >= deadline + 1.0
                    ):
                        break
                    time.sleep(0.01)
            finally:
                collector_done.set()

        def write_sample(source: dict[str, Any], sample_monotonic: float) -> None:
            nonlocal frames_written
            nonlocal first_written_source
            nonlocal last_written_source
            nonlocal measurement_marker_frames_written
            nonlocal measurement_marker_first_video_frame_index
            nonlocal video_write_duration_ms
            if writer is None or (buffer_raw_capture and raw_writer is None):
                raise RuntimeError("Required VideoWriters are not available")
            frame = source["frame"]
            raw_frame = source.get("raw_frame", frame)
            if measurement_marker_active(
                sample_monotonic,
                event_mono,
                measurement_delay_seconds,
            ):
                if measurement_marker_first_video_frame_index is None:
                    measurement_marker_first_video_frame_index = frames_written
                measurement_marker_frames_written += 1
                frame = draw_measurement_perimeter(frame)
            write_started = time.perf_counter()
            writer.write(frame)
            video_write_duration_ms += (
                time.perf_counter() - write_started
            ) * 1000.0
            if raw_writer is not None:
                raw_writer.write(raw_frame)
            first_written_source = first_written_source or source
            last_written_source = source
            frames_written += 1

        def ensure_writers(source: dict[str, Any]) -> None:
            nonlocal writer
            nonlocal raw_writer
            if writer is not None:
                return
            if video_path is None:
                raise RuntimeError("Video output path is not available")
            height, width = source["frame"].shape[:2]
            raw_height, raw_width = source.get("raw_frame", source["frame"]).shape[:2]
            if buffer_raw_capture and (raw_width, raw_height) != (width, height):
                raise RuntimeError(
                    "Processed and raw recording frames must have the same dimensions"
                )
            writer_args = (
                cv2.VideoWriter_fourcc(*"avc1"),
                fps,
                (width, height),
            )
            if (
                self.video_encoder == "nvidia_nvenc"
                and self.video_ffmpeg_executable is not None
            ):
                writer = NvencVideoWriter(
                    self.video_ffmpeg_executable,
                    video_path,
                    fps,
                    (width, height),
                    float(getattr(self.args, "video_bitrate_mbps", 16.0)),
                )
            elif os.name == "nt":
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.CAP_MSMF,
                    *writer_args,
                )
            else:
                writer = cv2.VideoWriter(str(video_path), *writer_args)
            if buffer_raw_capture and raw_video_path is not None:
                if os.name == "nt":
                    raw_writer = cv2.VideoWriter(
                        str(raw_video_path),
                        cv2.CAP_MSMF,
                        *writer_args,
                    )
                else:
                    raw_writer = cv2.VideoWriter(
                        str(raw_video_path),
                        *writer_args,
                    )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter: {video_path}")
            if buffer_raw_capture and raw_video_path is not None and (
                raw_writer is None or not raw_writer.isOpened()
            ):
                raise RuntimeError(f"Could not open raw VideoWriter: {raw_video_path}")

        def process_camera_source(
            source: dict[str, Any],
            *,
            include_evidence_images: bool = False,
        ) -> tuple[dict[str, Any], dict[str, Any] | None, float | None]:
            nonlocal processed_source_frame_count
            nonlocal clip_detected_piece
            frame_processor = getattr(
                self.processor,
                "process_clip_frame",
                None,
            )
            if not callable(frame_processor):
                raise RuntimeError(
                    "Vision processor does not support PLC-triggered clip processing"
                )
            result, duration_ms = frame_processor(
                source,
                include_evidence_images=include_evidence_images,
            )
            recording_frame = result.get("_recording_frame")
            if not isinstance(recording_frame, dict):
                raise RuntimeError(
                    "Vision processing did not return a recording frame"
                )
            processed_source_frame_count += 1
            if snapshots_contain_piece([result]):
                clip_detected_piece = True
            return recording_frame.copy(), result, duration_ms

        def consume_source(source: dict[str, Any]) -> None:
            nonlocal last_index
            nonlocal last_source
            nonlocal source_frames_seen
            source_index = int(source["index"])
            if source_index <= last_index:
                return
            last_index = source_index
            source_monotonic = float(source["monotonic"])
            if source_monotonic < clip_start_mono or source_monotonic >= deadline:
                return
            source_frames_seen += 1
            ensure_writers(source)
            if last_source is None:
                last_source = source
            next_sample_mono = clip_start_mono + (frames_written / fps)
            if (
                float(last_source["monotonic"]) < event_mono <= source_monotonic
                and next_sample_mono >= event_mono
            ):
                last_source = source
            while frames_written < target_frame_count:
                sample_mono = clip_start_mono + (frames_written / fps)
                if sample_mono > source_monotonic:
                    break
                write_sample(last_source, sample_mono)
            last_source = source

        try:
            day_dir = self.args.output_dir / "live_plc_clips" / datetime.now().strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            edge = event.get("event_edge") or "event"
            base = f"live_{clip_index:04d}_{file_stamp()}_{edge}"
            analysis_dir = day_dir / f"{base}_analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            video_path = day_dir / f"{base}.mp4"
            if save_raw_clips:
                raw_video_path = day_dir / f"{base}_raw.mp4"
            json_path = day_dir / f"{base}.json"
            collector_thread = threading.Thread(
                target=collect_post_trigger_frames,
                name=f"plc-frame-collector-{clip_index}",
                daemon=True,
            )
            collector_thread.start()
            if direct_raw_capture and raw_video_path is not None:
                raw_capture_started_at = utc_now()
                raw_capture_process = start_direct_raw_capture(
                    self.args,
                    raw_video_path,
                    record_seconds,
                )

            if event_source_frame is not None:
                try:
                    (
                        processed_event_frame,
                        event_result,
                        event_processing_duration_ms,
                    ) = process_camera_source(
                        event_source_frame,
                        include_evidence_images=True,
                    )
                    if event_result is None:
                        raise RuntimeError(
                            "Vision processor is required for PLC-triggered clips"
                        )
                    source_frame_monotonic = float(
                        event_result.get(
                            "frame_monotonic",
                            event_source_frame["monotonic"],
                        )
                    )
                    measurement_source_frame_offset_seconds = (
                        source_frame_monotonic - event_mono
                    )
                    event_result = event_result.copy()
                    event_result["measurement_event_frame"] = True
                    event_result["measurement_source_frame_monotonic"] = (
                        source_frame_monotonic
                    )
                    event_result["measurement_source_frame_offset_seconds"] = (
                        measurement_source_frame_offset_seconds
                    )
                    event_result["frame_monotonic"] = event_mono
                    event_recording_frame = processed_event_frame
                    event_recording_frame["source_monotonic"] = (
                        source_frame_monotonic
                    )
                    event_recording_frame["monotonic"] = event_mono
                    self._capture_processing_snapshot(
                        analysis_dir,
                        processing_snapshots,
                        seen_processing_frames,
                        event_mono,
                        deadline,
                        result_override=event_result,
                        processing_duration_ms=event_processing_duration_ms,
                    )
                except Exception as exc:
                    event_frame_processing_error = str(exc)

            for historical_frame in historical_frames:
                if (
                    event_recording_frame is not None
                    and int(historical_frame["index"])
                    == int(event_source_frame["index"])
                ):
                    continue
                processed_frame, _result, _duration_ms = process_camera_source(
                    historical_frame
                )
                consume_source(processed_frame)
            if event_recording_frame is not None:
                consume_source(event_recording_frame)

            while not collector_done.is_set() or not post_frame_queue.empty():
                try:
                    camera_frame = post_frame_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                processed_frame, _result, _duration_ms = process_camera_source(
                    camera_frame
                )
                consume_source(processed_frame)

            if writer is None or last_source is None:
                raise RuntimeError("No frames were available to record the clip.")

            while frames_written < target_frame_count:
                sample_mono = clip_start_mono + (frames_written / fps)
                write_sample(last_source, sample_mono)
            completed_writer = writer
            completed_writer.release()
            if isinstance(completed_writer, NvencVideoWriter):
                video_write_duration_ms = (
                    completed_writer.worker_write_duration_ms
                )
                video_queue_max_depth = completed_writer.max_queue_depth
            writer = None
            if raw_writer is not None:
                raw_writer.release()
                raw_writer = None
            if direct_raw_capture and raw_capture_process is not None and raw_video_path is not None:
                raw_capture_result = finish_direct_raw_capture(
                    raw_capture_process,
                    raw_video_path,
                    record_seconds + 8.0,
                )
                raw_capture_process = None
                if not raw_capture_result.get("ok"):
                    raw_capture_error = str(raw_capture_result.get("error") or "Raw recording failed")
                    delete_clip_paths(self.args.output_dir, [raw_video_path])
                    raw_video_path = None

            if first_written_source is None or last_written_source is None:
                raise RuntimeError("No frames were written to the clip.")

            if not clip_detected_piece:
                database_cleanup_error = ""
                if self.database is not None:
                    try:
                        self.database.delete_measurement_event(measurement_event_id)
                    except Exception as exc:
                        database_cleanup_error = str(exc)
                artifact_cleanup_errors = delete_clip_paths(
                    self.args.output_dir,
                    [video_path, raw_video_path, analysis_dir, json_path],
                )
                discarded = {
                    "clip_index": clip_index,
                    "discarded_at": utc_now(),
                    "event_id": measurement_event_id,
                    "event": clean_value(event),
                    "reason": "no_piece_detected",
                    "processing_mode": "plc_triggered_clip",
                    "processed_source_frame_count": processed_source_frame_count,
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
                target_offset_seconds=measurement_delay_seconds,
            )
            measurement_evidence_marked = mark_measurement_evidence_snapshot(
                canonical_snapshot
            )
            measurement_actual_offset_seconds = (
                float(canonical_snapshot["frame_monotonic"]) - event_mono
                if canonical_snapshot is not None
                and canonical_snapshot.get("frame_monotonic") is not None
                else None
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
                "pre_trigger_seconds": pre_trigger_seconds,
                "post_trigger_seconds": post_trigger_seconds,
                "plc_event_video_offset_seconds": pre_trigger_seconds,
                "clip_start_monotonic": clip_start_mono,
                "clip_end_monotonic": deadline,
                "video_fps": fps,
                "video_codec": "h264",
                "video_encoder": self.video_encoder,
                "video_write_duration_ms": round(video_write_duration_ms, 2),
                "video_write_average_ms": round(
                    video_write_duration_ms / max(1, frames_written),
                    2,
                ),
                "video_queue_max_depth": video_queue_max_depth,
                "video_content": "yolo_processed_overlay" if self.processor is not None else "raw_fallback",
                "processing_mode": "plc_triggered_clip",
                "processed_source_frame_count": processed_source_frame_count,
                "measurement_delay_seconds": measurement_delay_seconds,
                "measurement_target_monotonic": measurement_target_mono,
                "measurement_actual_offset_seconds": measurement_actual_offset_seconds,
                "measurement_source_frame_offset_seconds": (
                    measurement_source_frame_offset_seconds
                ),
                "measurement_event_frame_processed": (
                    measurement_source_frame_offset_seconds is not None
                ),
                "event_frame_processing_error": event_frame_processing_error,
                "measurement_snapshot_utc": (
                    canonical_snapshot.get("frame_utc")
                    if canonical_snapshot is not None
                    else None
                ),
                "measurement_snapshot_frame_index": (
                    int(canonical_snapshot["frame_index"])
                    if canonical_snapshot is not None
                    else None
                ),
                "measurement_evidence_marked": measurement_evidence_marked,
                "measurement_marker_duration_seconds": MEASUREMENT_MARKER_DURATION_SECONDS,
                "measurement_marker_frames_written": measurement_marker_frames_written,
                "measurement_marker_first_video_frame_index": (
                    measurement_marker_first_video_frame_index
                ),
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
            if raw_video_path is not None:
                if buffer_raw_capture:
                    raw_height, raw_width = last_written_source.get(
                        "raw_frame",
                        last_written_source["frame"],
                    ).shape[:2]
                    raw_capture_result = {
                        "ok": True,
                        "width": raw_width,
                        "height": raw_height,
                        "fps": fps,
                        "frame_count": frames_written,
                        "duration_seconds": frames_written / fps,
                        "size_bytes": raw_video_path.stat().st_size,
                    }
                sidecar.update(
                    raw_video_path=str(raw_video_path),
                    raw_video_codec="h264",
                    raw_video_content="axis_camera_raw",
                    raw_video_capture_mode=(
                        "direct_rtsp_copy" if direct_raw_capture else "processed_buffer"
                    ),
                    raw_video_pre_trigger_seconds=(
                        0.0 if direct_raw_capture else pre_trigger_seconds
                    ),
                    raw_video_plc_event_offset_seconds=(
                        0.0 if direct_raw_capture else pre_trigger_seconds
                    ),
                    raw_video_capture_started_at=raw_capture_started_at,
                    raw_video_requested_resolution=str(
                        getattr(
                            self.args,
                            "raw_camera_resolution",
                            getattr(self.args, "camera_resolution", "unknown"),
                        )
                    ),
                    raw_video_requested_fps=float(
                        getattr(self.args, "raw_record_fps", fps)
                    ),
                    raw_video_width=raw_capture_result.get("width") if raw_capture_result else None,
                    raw_video_height=raw_capture_result.get("height") if raw_capture_result else None,
                    raw_video_fps=raw_capture_result.get("fps") if raw_capture_result else None,
                    raw_video_frames=(
                        raw_capture_result.get("frame_count") if raw_capture_result else None
                    ),
                    raw_video_duration_seconds=(
                        raw_capture_result.get("duration_seconds")
                        if raw_capture_result
                        else None
                    ),
                )
            elif save_raw_clips and raw_capture_error:
                sidecar["raw_video_error"] = raw_capture_error
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
            stop_direct_raw_capture(raw_capture_process)
            raw_capture_process = None
            if writer is not None:
                if isinstance(writer, NvencVideoWriter):
                    writer.abort()
                else:
                    writer.release()
                writer = None
            if raw_writer is not None:
                raw_writer.release()
                raw_writer = None
            delete_clip_paths(
                self.args.output_dir,
                [video_path, raw_video_path, analysis_dir, json_path],
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
                except Exception as database_exc:
                    failure["database_error"] = str(database_exc)
            with self.lock:
                self.error = failure["error"]
                self.failed_recordings.append(failure)
        finally:
            collector_stop.set()
            if collector_thread is not None:
                collector_thread.join(timeout=2.0)
            stop_direct_raw_capture(raw_capture_process)
            if writer is not None:
                writer.release()
            if raw_writer is not None:
                raw_writer.release()
            with self.lock:
                self.active_recordings.pop(clip_index, None)

    def _enforce_retention(self) -> None:
        max_clips = max(1, int(self.args.max_clips or 100))
        valid_sidecars: list[tuple[Path, dict[str, Any]]] = []
        for json_path in clip_sidecars(self.args.output_dir):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                delete_clip_artifacts(self.args.output_dir, json_path)
                continue
            valid_sidecars.append((json_path, data))

        for json_path, data in valid_sidecars[max_clips:]:
            if self.database is not None:
                try:
                    event_id = data.get("event_id")
                    if event_id:
                        self.database.delete_measurement_event(str(event_id))
                except Exception as exc:
                    with self.lock:
                        self.error = (
                            f"Could not remove retained database event "
                            f"{data.get('event_id')}: {exc}"
                        )
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
        all_event_ids = getattr(
            self.database,
            "all_measurement_event_ids",
            None,
        )
        existing_event_ids = all_event_ids() if callable(all_event_ids) else None
        for path in clip_sidecars(self.args.output_dir):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                data = None
            if data and (
                data.get("db_sync_status") == "pending"
                or data.get("db_sync_backend") != backend
                or (
                    existing_event_ids is not None
                    and data.get("event_id")
                    and str(data["event_id"]) not in existing_event_ids
                )
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
            except (OSError, UnicodeError, json.JSONDecodeError):
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
        if callable(all_event_ids):
            retained_event_ids: set[str] = set()
            can_prune = True
            for path in clip_sidecars(self.args.output_dir):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
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


class LiveMvpRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.buffer: FrameBuffer | None = None
        self.camera: CameraReader | None = None
        self.processor: LiveProcessor | None = None
        self.recorder: ClipRecorder | None = None
        self.plc: PLCMonitor | None = None
        self.database: DatabaseRepository | SQLiteDatabaseRepository | None = None
        self.reconciler: DatabaseReconciler | None = None
        self.started = False
        self._lifecycle_lock = threading.Lock()

    def _initialize(self) -> None:
        configure_vision_module(self.args)
        if not self.args.db_disabled:
            self.database = (
                DatabaseRepository(self.args.postgres_dsn)
                if self.args.postgres_dsn.strip()
                else SQLiteDatabaseRepository(self.args.sqlite_path)
            )
            try:
                self.database.open(timeout=float(self.args.plc_timeout))
                self.database.validate_schema()
                self.database.recover_stale_measurement_events(
                    older_than_seconds=max(
                        60,
                        int(float(self.args.record_seconds) * 3),
                    )
                )
            except Exception:
                self.database.close()
                self.database = None
                raise

        buffer_limit = max(8, int(self.args.buffer_max_frames))
        buffer_len = min(
            int(
                max(
                    8,
                    float(self.args.buffer_seconds)
                    * max(1.0, float(self.args.record_fps)),
                )
            ),
            buffer_limit,
        )
        self.buffer = FrameBuffer(maxlen=buffer_len)
        self.camera = CameraReader(self.args, self.buffer)
        self.processor = LiveProcessor(self.args, self.buffer)
        self.recorder = ClipRecorder(
            self.args,
            self.buffer,
            self.processor,
            self.database,
        )
        self.recorder.validate_configuration()
        self.plc = PLCMonitor(self.args, self.recorder)
        self.reconciler = (
            DatabaseReconciler(self.args, self.database)
            if self.database is not None
            else None
        )

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.started:
                return
            if self.camera is not None:
                raise RuntimeError("A stopped Live MVP runtime cannot be restarted")
            try:
                self._initialize()
                self.camera.start()
                self.plc.start()
                if self.reconciler is not None:
                    self.reconciler.start()
                self.started = True
            except Exception:
                self._stop_components()
                raise

    def _stop_components(self) -> list[str]:
        errors = []
        components = (
            ("reconciler", self.reconciler),
            ("PLC", self.plc),
            ("recorder", self.recorder),
            ("camera", self.camera),
        )
        for name, component in components:
            if component is None:
                continue
            try:
                component.stop()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if self.database is not None:
            try:
                self.database.close()
            except Exception as exc:
                errors.append(f"database: {exc}")
        self.started = False
        return errors

    def stop(self) -> None:
        with self._lifecycle_lock:
            errors = self._stop_components()
        if errors:
            raise RuntimeError("; ".join(errors))


app = Flask(__name__)
_args: argparse.Namespace
_buffer: FrameBuffer
_camera: CameraReader
_processor: LiveProcessor
_recorder: ClipRecorder
_plc: PLCMonitor
_database: DatabaseRepository | SQLiteDatabaseRepository | None = None
_reconciler: DatabaseReconciler | None = None
_runtime: LiveMvpRuntime | None = None
_runtime_lock = threading.Lock()


def _publish_runtime(runtime: LiveMvpRuntime) -> None:
    global _args, _buffer
    global _camera, _processor, _recorder, _plc, _database, _reconciler
    if (
        runtime.buffer is None
        or runtime.camera is None
        or runtime.processor is None
        or runtime.recorder is None
        or runtime.plc is None
    ):
        raise RuntimeError("The Live MVP runtime is not initialized")
    _args = runtime.args
    _buffer = runtime.buffer
    _camera = runtime.camera
    _processor = runtime.processor
    _recorder = runtime.recorder
    _plc = runtime.plc
    _database = runtime.database
    _reconciler = runtime.reconciler


def start_runtime(args: argparse.Namespace) -> LiveMvpRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is not None and _runtime.started:
            raise RuntimeError("The Live MVP runtime is already running")
        runtime = LiveMvpRuntime(args)
        runtime.start()
        _publish_runtime(runtime)
        _runtime = runtime
        return runtime


def stop_runtime() -> None:
    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
    if runtime is not None:
        runtime.stop()


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
    resolved = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
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
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    return errors


def read_clip_sidecar(
    json_path: Path,
    *,
    include_snapshots: bool = True,
    measurement_evidence_only: bool = False,
) -> dict[str, Any] | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    snapshots = data.get("processing_snapshots", []) or []
    data["processing_snapshot_count"] = int(data.get("processing_snapshot_count", len(snapshots)))
    if measurement_evidence_only:
        snapshots = measurement_evidence_snapshots(
            snapshots,
            event_monotonic=(data.get("event") or {}).get("event_read_monotonic"),
            measurement_delay_seconds=data.get("measurement_delay_seconds"),
        )
        data["processing_snapshots"] = snapshots
    elif not include_snapshots:
        snapshots = []
        data["processing_snapshots"] = snapshots
    data["processing_snapshots_shown"] = len(snapshots)
    data["clip_id"] = json_path.stem
    data["json_path"] = str(json_path)
    data["video_url"] = f"/api/live/clips/{json_path.stem}/video"
    if data.get("raw_video_path"):
        data["raw_video_url"] = f"/api/live/clips/{json_path.stem}/raw-video"
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
        include_snapshots=detail,
        measurement_evidence_only=detail,
    )
    if data is None:
        return None
    evidence_snapshots = data.get("processing_snapshots", [])
    canonical = evidence_snapshots[0] if evidence_snapshots else None
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
        "raw_video_url": data.get("raw_video_url"),
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
                except (OSError, UnicodeError, json.JSONDecodeError):
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
    snapshots = measurement_evidence_snapshots(event.get("snapshots", []))
    for snapshot in snapshots:
        for path_key, url_key in (
            ("original_overlay_path", "original_overlay_url"),
            ("rectified_overlay_path", "rectified_overlay_url"),
        ):
            asset = asset_by_path.get(snapshot.get(path_key))
            if asset:
                snapshot[url_key] = f"/api/history/events/{event_id}/assets/{asset['id']}"
    video = next((asset for asset in assets if asset.get("asset_type") == "video"), None)
    raw_video = next(
        (asset for asset in assets if asset.get("asset_type") == "raw_video"),
        None,
    )
    event["snapshots"] = snapshots
    event["assets"] = assets
    event["video_url"] = (
        f"/api/history/events/{event_id}/assets/{video['id']}" if video else None
    )
    event["raw_video_url"] = (
        f"/api/history/events/{event_id}/assets/{raw_video['id']}"
        if raw_video
        else None
    )
    event["detail_url"] = f"/history/{event_id}"
    event["database_mode"] = database_mode()
    return clean_value(event)


def delete_clip_artifacts(output_dir: Path, json_path: Path) -> None:
    data = read_clip_sidecar(json_path, include_snapshots=False) or {}
    candidates: list[Path] = [json_path]
    for key in ("video_path", "raw_video_path", "analysis_dir"):
        value = data.get(key)
        if value:
            candidates.append(Path(value))
    if not data:
        base = json_path.parent / json_path.stem
        candidates.extend(
            [
                base.with_suffix(".mp4"),
                json_path.parent / f"{json_path.stem}_raw.mp4",
                json_path.parent / f"{json_path.stem}_analysis",
            ]
        )
    delete_clip_paths(output_dir, candidates)


@app.route("/")
def index():
    return render_template("live.html")


@app.route("/history")
def history():
    return render_template("history.html")


@app.route("/history/<clip_id>")
def history_clip(clip_id: str):
    return render_template("history.html")


@app.route("/api/live/status")
def api_live_status():
    summary = request.args.get("summary", "").strip().lower() in ("1", "true", "yes")
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
    recorder_status = _recorder.snapshot()
    processor_status = _processor.snapshot(include_images=False)
    return jsonify(
        camera=_camera.snapshot(),
        processor=(
            compact_processor_status(processor_status)
            if summary
            else processor_status
        ),
        live_stream={
            "codec": "h264",
            "transport": "fragmented_mp4",
            "encoding": "copy",
            "fps": float(_args.live_stream_fps),
            "resolution": str(_args.camera_resolution),
        },
        plc=plc_status_with_signal_state(_plc.snapshot()),
        recorder=(
            compact_recorder_status(recorder_status)
            if summary
            else recorder_status
        ),
        database=database_status,
        reconciler=_reconciler.snapshot() if _reconciler is not None else None,
    )


@app.route("/api/health")
def api_health():
    runtime = _runtime
    if runtime is None or not runtime.started:
        return jsonify(status="unavailable", checks={"runtime": False}), 503

    camera = _camera.snapshot()
    processor = _processor.snapshot(include_images=False)
    plc = _plc.snapshot()
    recorder = _recorder.snapshot()
    database = _database.health().as_dict() if _database is not None else None
    checks = {
        "runtime": True,
        "camera": bool(camera.get("connected")),
        "processor": bool(processor.get("ok")) and not processor.get("error"),
        "plc": not bool(plc.get("enabled")) or bool(plc.get("connected")),
        "recorder": not recorder.get("error")
        and not recorder.get("video_encoder_error"),
        "database": bool(database and database.get("ok")),
    }
    healthy = all(checks.values())
    return jsonify(
        status="ok" if healthy else "degraded",
        checks=checks,
        checked_at=utc_now(),
    ), (200 if healthy else 503)


@app.route("/api/live/stream.mp4")
def api_live_stream():
    if str(_args.codec).lower() != "h264":
        abort(503, description="The browser live stream requires H.264")
    ffmpeg_executable = bundled_ffmpeg_executable()
    if ffmpeg_executable is None:
        abort(503, description="FFmpeg is unavailable")

    source, rtsp_source = resolve_live_stream_source(_args)
    if not rtsp_source and not Path(source).is_file():
        abort(503, description="The simulated live video is unavailable")
    process = subprocess.Popen(
        build_live_stream_command(
            ffmpeg_executable,
            source,
            rtsp_source=rtsp_source,
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )

    def stream():
        try:
            if process.stdout is None:
                return
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)

    response = Response(stream(), mimetype="video/mp4", direct_passthrough=True)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Disposition"] = "inline"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["X-Live-Stream-Fps"] = str(int(round(_args.live_stream_fps)))
    return response


@app.route("/api/live/frame")
def api_live_frame():
    metadata_only = request.args.get("metadata", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    data = _processor.snapshot(include_images=not metadata_only)
    latest_item = _buffer.latest()
    if metadata_only:
        result = data.get("result")
        data["result"] = (
            compact_live_result(result)
            if isinstance(result, dict)
            else None
        )
        data["preview_frame_index"] = (
            int(latest_item["index"]) if latest_item is not None else None
        )
        data["recorder"] = compact_recorder_status(_recorder.snapshot())
        data["plc"] = plc_status_with_signal_state(_plc.snapshot())
    else:
        if latest_item is None:
            return jsonify(
                error=data.get("error") or "No camera frame is available yet",
                processor=data,
            ), 503
        jpeg = img_to_jpeg(latest_item["frame"], quality=80)
        result = data.get("result")
        if not isinstance(result, dict):
            result = {
                "frame_index": int(latest_item["index"]),
                "pieces": [],
                "count": 0,
            }
            data["result"] = result
        if "original_image" not in result:
            result["original_image"] = base64.b64encode(jpeg).decode(
                "ascii"
            )
        data["recorder"] = _recorder.snapshot()
    return jsonify(data)


@app.route("/api/live/image.jpg")
def api_live_image():
    latest_item = _buffer.latest()
    if latest_item is None:
        abort(503, description="No camera preview is available yet")
    frame_index = int(latest_item["index"])
    jpeg = img_to_jpeg(latest_item["frame"], quality=80)
    response = Response(jpeg, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["X-Frame-Index"] = str(frame_index)
    return response


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
    return jsonify(
        events=clean_value(events),
        count=len(events),
        database_mode=mode,
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
        data = read_clip_sidecar(json_path, include_snapshots=False) or {}
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


@app.route("/api/history/events/<event_id>/raw-video")
def api_history_event_raw_video(event_id: str):
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
        data = read_clip_sidecar(json_path, include_snapshots=False) or {}
        raw_video_path = Path(str(data.get("raw_video_path", "")))
    else:
        raw_video_asset = next(
            (
                asset
                for asset in event.get("assets", [])
                if asset.get("asset_type") == "raw_video"
            ),
            None,
        )
        if raw_video_asset is None:
            abort(404)
        try:
            raw_video_path = resolve_asset_path(
                raw_video_asset["relative_path"],
                _args.output_dir,
            )
        except ValueError:
            abort(404)
    if not raw_video_path.is_file() or not path_is_inside(
        raw_video_path,
        _args.output_dir,
    ):
        abort(404)
    return send_file(raw_video_path, mimetype="video/mp4", conditional=True)


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
        data = read_clip_sidecar(json_path, include_snapshots=False)
        if data is not None:
            clips.append(data)
    return jsonify(clips=clips[:100], count=len(clips))


@app.route("/api/live/clips/<clip_id>")
def api_live_clip(clip_id: str):
    json_path = find_clip_json(clip_id)
    if json_path is None:
        return jsonify(error="Clip not found"), 404
    data = read_clip_sidecar(json_path, measurement_evidence_only=True)
    if data is None:
        return jsonify(error="Clip metadata could not be read"), 500
    return jsonify(data)


@app.route("/api/live/clips/<clip_id>/video")
def api_live_clip_video(clip_id: str):
    json_path = find_clip_json(clip_id)
    if json_path is None:
        abort(404)
    data = read_clip_sidecar(json_path, include_snapshots=False) or {}
    video_path = Path(str(data.get("video_path", "")))
    if not video_path.exists() or not path_is_inside(video_path, clips_root(_args.output_dir)):
        abort(404)
    return send_file(video_path, mimetype="video/mp4", conditional=True)


@app.route("/api/live/clips/<clip_id>/raw-video")
def api_live_clip_raw_video(clip_id: str):
    json_path = find_clip_json(clip_id)
    if json_path is None:
        abort(404)
    data = read_clip_sidecar(json_path, include_snapshots=False) or {}
    raw_video_path = Path(str(data.get("raw_video_path", "")))
    if not raw_video_path.exists() or not path_is_inside(
        raw_video_path,
        clips_root(_args.output_dir),
    ):
        abort(404)
    return send_file(raw_video_path, mimetype="video/mp4", conditional=True)


@app.route("/api/live/clips/<clip_id>/asset/<path:asset_name>")
def api_live_clip_asset(clip_id: str, asset_name: str):
    json_path = find_clip_json(clip_id)
    if json_path is None:
        abort(404)
    data = read_clip_sidecar(json_path, include_snapshots=False) or {}
    analysis_dir = Path(str(data.get("analysis_dir", "")))
    asset_path = analysis_dir / asset_name
    if not asset_path.exists() or not path_is_inside(asset_path, analysis_dir) or not path_is_inside(asset_path, clips_root(_args.output_dir)):
        abort(404)
    return send_file(asset_path, mimetype="image/jpeg", conditional=True)


def main() -> int:
    try:
        runtime = start_runtime(parse_args())
    except Exception as exc:
        print(f"ERROR: Live MVP startup failed: {exc}", file=sys.stderr)
        return 2

    print(f"\n  TX2 Live MVP at http://127.0.0.1:{runtime.args.port}\n")
    print(f"  Source: {runtime.args.source}")
    print(f"  PLC: {'enabled' if runtime.args.plc_enabled else 'disabled'}")
    print(
        "  Database: "
        + (
            "disabled (simulation)"
            if runtime.args.db_disabled
            else database_mode()
        )
    )
    exit_code = 0
    try:
        app.run(
            host="127.0.0.1",
            port=runtime.args.port,
            debug=False,
            threaded=True,
        )
    finally:
        try:
            stop_runtime()
        except Exception as exc:
            print(f"ERROR: Live MVP shutdown failed: {exc}", file=sys.stderr)
            exit_code = 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
