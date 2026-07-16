#!/usr/bin/env python3
"""Record AXIS camera clips when a PLC OPC UA event changes.

The recorder keeps a small in-memory pre-roll from the RTSP stream. When the
configured PLC tag changes, it writes a clip containing frames before and after
the event, plus a JSON sidecar with the PLC timestamps.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2


DEFAULT_ENDPOINT = "opc.tcp://10.14.6.48:49320"
DEFAULT_WATCHDOG_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD"
DEFAULT_EVENT_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength"
DEFAULT_CAMERA_IP = "10.14.115.241"


@dataclass
class FrameSample:
    monotonic: float
    utc: str
    frame: Any


@dataclass
class PlcEvent:
    index: int
    edge: str
    value: Any
    previous_value: Any
    read_monotonic: float
    read_utc: str
    source_timestamp: Any
    server_timestamp: Any
    watchdog_value: Any
    watchdog_source_timestamp: Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def file_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")


def clean(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return str(value)


def rtsp_url(ip: str, username: str, password: str, codec: str) -> str:
    user = quote(username, safe="")
    pwd = quote(password, safe="")
    return f"rtsp://{user}:{pwd}@{ip}/axis-media/media.amp?videocodec={codec}"


def event_edge(previous: Any, current: Any) -> str:
    if previous is None or previous == current:
        return ""
    if isinstance(previous, bool) and isinstance(current, bool):
        return "rising" if current else "falling"
    return "changed"


def edge_matches(edge: str, mode: str) -> bool:
    if mode in ("any", "changed"):
        return bool(edge)
    return edge == mode


def read_frame_loop(
    url: str,
    pre_roll: deque[FrameSample],
    pre_roll_lock: threading.Lock,
    stop_event: threading.Event,
    fps_holder: dict[str, float],
) -> None:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        logging.error("Could not open RTSP stream.")
        stop_event.set()
        return

    reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if reported_fps > 0.0:
        fps_holder["fps"] = reported_fps

    last_tick = time.monotonic()
    fps = reported_fps if reported_fps > 0.0 else 0.0
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            logging.warning("Frame read failed; retrying.")
            time.sleep(0.1)
            continue

        now = time.monotonic()
        sample = FrameSample(monotonic=now, utc=utc_now(), frame=frame)
        with pre_roll_lock:
            pre_roll.append(sample)

        dt = max(1e-6, now - last_tick)
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
        fps_holder["fps"] = fps
        last_tick = now

    cap.release()


def write_clip(path: Path, frames: list[FrameSample], fps: float) -> None:
    if not frames:
        raise ValueError("No frames available for clip.")
    height, width = frames[0].frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for sample in frames:
            writer.write(sample.frame)
    finally:
        writer.release()


def post_roll_frames(
    event_time: float,
    seconds: float,
    pre_roll: deque[FrameSample],
    pre_roll_lock: threading.Lock,
    stop_event: threading.Event,
) -> list[FrameSample]:
    deadline = event_time + seconds
    while time.monotonic() < deadline and not stop_event.is_set():
        time.sleep(0.02)
    with pre_roll_lock:
        return [sample for sample in pre_roll if sample.monotonic >= event_time and sample.monotonic <= deadline]


def save_event_clip(
    event: PlcEvent,
    args: argparse.Namespace,
    pre_roll: deque[FrameSample],
    pre_roll_lock: threading.Lock,
    stop_event: threading.Event,
    fps_holder: dict[str, float],
) -> None:
    with pre_roll_lock:
        before = [
            sample
            for sample in pre_roll
            if event.read_monotonic - args.pre_seconds <= sample.monotonic < event.read_monotonic
        ]

    after = post_roll_frames(event.read_monotonic, args.post_seconds, pre_roll, pre_roll_lock, stop_event)
    frames = before + after
    fps = max(1.0, min(float(args.output_fps or fps_holder.get("fps") or 15.0), 60.0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cut_{event.index:04d}_{file_stamp()}_{event.edge}"
    video_path = args.output_dir / f"{stem}.mp4"
    json_path = args.output_dir / f"{stem}.json"

    write_clip(video_path, frames, fps)
    sidecar = {
        "event_index": event.index,
        "event_edge": event.edge,
        "event_value": clean(event.value),
        "previous_event_value": clean(event.previous_value),
        "event_read_utc": event.read_utc,
        "event_source_timestamp": clean(event.source_timestamp),
        "event_server_timestamp": clean(event.server_timestamp),
        "event_read_monotonic": event.read_monotonic,
        "watchdog_value": clean(event.watchdog_value),
        "watchdog_source_timestamp": clean(event.watchdog_source_timestamp),
        "camera_ip": args.ip,
        "pre_seconds": args.pre_seconds,
        "post_seconds": args.post_seconds,
        "frames_written": len(frames),
        "video_fps": fps,
        "first_frame_utc": frames[0].utc if frames else None,
        "last_frame_utc": frames[-1].utc if frames else None,
        "video_path": str(video_path),
    }
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.warning("Saved cut clip: %s | frames=%s | sidecar=%s", video_path, len(frames), json_path)


async def read_node(client: Any, node_id: str) -> dict[str, Any]:
    node = client.get_node(node_id)
    data_value = await node.read_data_value()
    return {
        "value": data_value.Value.Value,
        "source_timestamp": data_value.SourceTimestamp,
        "server_timestamp": data_value.ServerTimestamp,
        "status": str(data_value.StatusCode),
        "read_utc": utc_now(),
        "read_monotonic": time.monotonic(),
    }


async def monitor_plc(
    args: argparse.Namespace,
    event_queue: queue.Queue[PlcEvent],
    stop_event: threading.Event,
) -> None:
    from asyncua import Client

    previous_watchdog: Any = None
    previous_value: Any = None
    event_index = 0
    logging.info("Connecting to PLC OPC UA: %s", args.endpoint)
    async with Client(url=args.endpoint, timeout=args.timeout) as client:
        logging.info("Connected to PLC. Watching %s", args.event_node)
        while not stop_event.is_set():
            watchdog = await read_node(client, args.watchdog_node)
            watchdog_value = watchdog["value"]
            if watchdog_value != previous_watchdog:
                event_read = await read_node(client, args.event_node)
                edge = event_edge(previous_value, event_read["value"])
                if edge and edge_matches(edge, args.edge):
                    event_index += 1
                    event = PlcEvent(
                        index=event_index,
                        edge=edge,
                        value=event_read["value"],
                        previous_value=previous_value,
                        read_monotonic=event_read["read_monotonic"],
                        read_utc=event_read["read_utc"],
                        source_timestamp=event_read["source_timestamp"],
                        server_timestamp=event_read["server_timestamp"],
                        watchdog_value=watchdog_value,
                        watchdog_source_timestamp=watchdog["source_timestamp"],
                    )
                    event_queue.put(event)
                    logging.warning(
                        "PLC event %s | value=%r | source=%s",
                        edge,
                        event_read["value"],
                        clean(event_read["source_timestamp"]),
                    )
                previous_watchdog = watchdog_value
                previous_value = event_read["value"]

            await asyncio.sleep(args.poll_interval)


def clip_writer_loop(
    args: argparse.Namespace,
    event_queue: queue.Queue[PlcEvent],
    pre_roll: deque[FrameSample],
    pre_roll_lock: threading.Lock,
    stop_event: threading.Event,
    fps_holder: dict[str, float],
) -> None:
    saved_count = len(list(args.output_dir.glob("*.mp4")))
    if args.max_clips is not None and saved_count >= args.max_clips:
        logging.warning(
            "Clip limit already reached: %s/%s in %s. Stopping recorder.",
            saved_count,
            args.max_clips,
            args.output_dir,
        )
        stop_event.set()
        return

    while not stop_event.is_set():
        try:
            event = event_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            save_event_clip(event, args, pre_roll, pre_roll_lock, stop_event, fps_holder)
            saved_count = len(list(args.output_dir.glob("*.mp4")))
            if args.max_clips is not None and saved_count >= args.max_clips:
                logging.warning(
                    "Clip limit reached: %s/%s in %s. Stopping recorder.",
                    saved_count,
                    args.max_clips,
                    args.output_dir,
                )
                stop_event.set()
        except Exception as exc:
            logging.exception("Failed to save clip for event %s: %s", event.index, exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record AXIS clips when a PLC tag changes.")
    parser.add_argument("--ip", default=DEFAULT_CAMERA_IP, help="AXIS camera IP.")
    parser.add_argument("--user", default=os.getenv("AXIS_USER"), help="AXIS username.")
    parser.add_argument("--password", default=os.getenv("AXIS_PASSWORD"), help="AXIS password.")
    parser.add_argument("--codec", choices=("h264", "jpeg"), default="h264")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--watchdog-node", default=DEFAULT_WATCHDOG_NODE)
    parser.add_argument("--event-node", default=DEFAULT_EVENT_NODE)
    parser.add_argument("--edge", choices=("any", "rising", "falling", "changed"), default="rising")
    parser.add_argument("--pre-seconds", type=float, default=3.0, help="Seconds to keep before PLC event.")
    parser.add_argument("--post-seconds", type=float, default=3.0, help="Seconds to record after PLC event.")
    parser.add_argument("--poll-interval", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--output-fps", type=float, help="Force output FPS. Defaults to camera/observed FPS.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "plc_cut_clips")
    parser.add_argument("--max-clips", type=int, help="Stop once this many MP4 clips exist in the output folder.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    configure_logging()
    args = parse_args()
    if args.pre_seconds < 0 or args.post_seconds <= 0:
        logging.error("--pre-seconds must be >= 0 and --post-seconds must be > 0")
        return 2
    if args.max_clips is not None and args.max_clips <= 0:
        logging.error("--max-clips must be > 0")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_clips = len(list(args.output_dir.glob("*.mp4")))
    if args.max_clips is not None and existing_clips >= args.max_clips:
        logging.warning(
            "Clip limit already reached: %s/%s in %s. Nothing to record.",
            existing_clips,
            args.max_clips,
            args.output_dir,
        )
        return 0

    username = args.user or input("AXIS user: ").strip()
    password = args.password if args.password is not None else getpass.getpass("AXIS password: ")

    max_frames = max(30, int((args.pre_seconds + args.post_seconds + 2.0) * 60.0))
    pre_roll: deque[FrameSample] = deque(maxlen=max_frames)
    pre_roll_lock = threading.Lock()
    event_queue: queue.Queue[PlcEvent] = queue.Queue()
    stop_event = threading.Event()
    fps_holder: dict[str, float] = {}

    camera_thread = threading.Thread(
        target=read_frame_loop,
        args=(rtsp_url(args.ip, username, password, args.codec), pre_roll, pre_roll_lock, stop_event, fps_holder),
        daemon=True,
    )
    writer_thread = threading.Thread(
        target=clip_writer_loop,
        args=(args, event_queue, pre_roll, pre_roll_lock, stop_event, fps_holder),
        daemon=True,
    )
    camera_thread.start()
    writer_thread.start()

    logging.info(
        "Recording armed | camera=%s | edge=%s | pre=%.2fs | post=%.2fs | output=%s",
        args.ip,
        args.edge,
        args.pre_seconds,
        args.post_seconds,
        args.output_dir,
    )
    try:
        asyncio.run(monitor_plc(args, event_queue, stop_event))
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
    except Exception as exc:
        logging.exception("Recorder failed: %s", exc)
        stop_event.set()
        return 1
    finally:
        stop_event.set()
        camera_thread.join(timeout=2.0)
        writer_thread.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
