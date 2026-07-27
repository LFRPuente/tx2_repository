"""Live AXIS camera processing with the existing TX2 app pipeline.

This opens an AXIS RTSP stream, rectifies each frame with the saved homography,
runs YOLO + Sobel front detection, estimates the calibrated measurement, and
shows a live OpenCV overlay.

Example:
    python tools/axis_live_processor.py --ip 10.14.115.74 --output-dir outputs
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import homography_web_app as tx2_app  # noqa: E402


DEFAULT_PIECE_MODEL = REPO_ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_pieces_v1" / "weights" / "best.pt"
DEFAULT_LEGACY_MODEL = REPO_ROOT / "runs" / "detect" / "runs_tx2" / "yolo11n_tubos_v1" / "weights" / "best.pt"
DEFAULT_MODEL = DEFAULT_PIECE_MODEL if DEFAULT_PIECE_MODEL.exists() else DEFAULT_LEGACY_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process an AXIS camera live with TX2 vision logic.")
    parser.add_argument("--ip", required=True, help="Camera IP, for example 10.14.115.74")
    parser.add_argument("--user", default=os.getenv("AXIS_USER"), help="AXIS username")
    parser.add_argument("--password", default=os.getenv("AXIS_PASSWORD"), help="AXIS password")
    parser.add_argument("--codec", choices=("h264", "jpeg"), default="h264")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "dataset_pieces")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.10, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    parser.add_argument("--process-every", type=int, default=1, help="Run YOLO/Sobel every N frames")
    parser.add_argument(
        "--view",
        choices=("original", "rectified"),
        default="original",
        help="Initial live display view. Press v while running to toggle.",
    )
    parser.add_argument("--max-width", type=int, default=1280, help="Maximum display window width")
    parser.add_argument("--max-height", type=int, default=720, help="Maximum display window height")
    parser.add_argument("--window", default="TX2 live AXIS processor")
    parser.add_argument("--snapshot-dir", type=Path, help="Optional folder for snapshots with key s")
    return parser.parse_args()


def rtsp_url(ip: str, username: str, password: str, codec: str) -> str:
    user = quote(username, safe="")
    pwd = quote(password, safe="")
    return f"rtsp://{user}:{pwd}@{ip}/axis-media/media.amp?videocodec={codec}"


def setup_tx2_app(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tx2_app._args = SimpleNamespace(
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        model=args.model,
        video=Path(""),
        second=0.0,
    )


def empty_sobel(frame_idx: int, time_sec: float) -> dict:
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


def process_frame(frame: np.ndarray, frame_idx: int, started_at: float, conf: float, imgsz: int) -> dict:
    matrix, out_size, _homography = tx2_app.load_homography()
    rectified = cv2.warpPerspective(frame, matrix, out_size)
    boxes = tx2_app.predict_yolo_boxes(rectified, conf=conf, imgsz=imgsz)
    time_sec = time.time() - started_at
    calibration = tx2_app.load_measurement_calibration()
    pieces, measurement_summary = tx2_app.analyze_piece_boxes(
        rectified,
        boxes,
        calibration,
        frame_idx=frame_idx,
        time_sec=time_sec,
    )
    primary = tx2_app.primary_piece_analysis(pieces)
    sobel = primary["sobel"] if primary else empty_sobel(frame_idx, time_sec)
    measurement = primary["measurement"] if primary else None
    original_overlay = tx2_app.mvp_original_overlay_for_pieces(
        pieces,
        calibration,
        matrix,
        rectified.shape[1],
    )
    return {
        "rectified": rectified,
        "boxes": boxes,
        "pieces": pieces,
        "measurement_summary": measurement_summary,
        "sobel": sobel,
        "measurement": measurement,
        "original_overlay": original_overlay,
        "time_sec": time_sec,
    }


def draw_overlay(rectified: np.ndarray, result: dict | None, fps: float) -> np.ndarray:
    vis = rectified.copy()
    if result:
        for piece in result.get("pieces") or []:
            box = piece["box"]
            piece_id = int(piece["piece_id"])
            color = (83, 182, 137) if piece.get("valid") else (72, 180, 255)
            x, y, w, h = (int(round(float(box[key]))) for key in ("x", "y", "w", "h"))
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                vis,
                f"P{piece_id} {float(box.get('conf', 0.0)):.2f}",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            line = (piece.get("sobel") or {}).get("line")
            if line:
                p1 = (int(round(line["x1"])), int(round(line["y1"])))
                p2 = (int(round(line["x2"])), int(round(line["y2"])))
                cv2.line(vis, p1, p2, color, 3, cv2.LINE_AA)

        summary = result.get("measurement_summary") or {}
        text = f"{int(summary.get('valid_count', 0))}/{len(result.get('pieces') or [])} valid piece measurements"
    else:
        text = "waiting for first processing pass"

    cv2.rectangle(vis, (12, 12), (430, 72), (17, 19, 21), -1)
    cv2.putText(vis, text, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (236, 231, 220), 2, cv2.LINE_AA)
    cv2.putText(vis, f"{fps:.1f} FPS", (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (169, 176, 173), 1, cv2.LINE_AA)
    return vis


def draw_original_overlay(original: np.ndarray, result: dict | None, fps: float) -> np.ndarray:
    vis = original.copy()
    text = "waiting for first processing pass"
    if result:
        overlay = result.get("original_overlay") or {}
        reference_line = overlay.get("reference_line")
        piece_fronts = overlay.get("piece_fronts") or []
        if reference_line:
            p1 = (int(round(reference_line["x1"])), int(round(reference_line["y1"])))
            p2 = (int(round(reference_line["x2"])), int(round(reference_line["y2"])))
            cv2.line(vis, p1, p2, (83, 182, 137), 2, cv2.LINE_AA)
        for item in piece_fronts:
            front_line = item.get("line")
            if not front_line:
                continue
            color = (83, 182, 137) if item.get("valid") else (72, 180, 255)
            p1 = (int(round(front_line["x1"])), int(round(front_line["y1"])))
            p2 = (int(round(front_line["x2"])), int(round(front_line["y2"])))
            cv2.line(vis, p1, p2, color, 3, cv2.LINE_AA)

        summary = result.get("measurement_summary") or {}
        text = f"{int(summary.get('valid_count', 0))}/{len(result.get('pieces') or [])} valid piece measurements"

    cv2.rectangle(vis, (12, 12), (500, 78), (17, 19, 21), -1)
    cv2.putText(vis, "full original view", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (169, 176, 173), 1, cv2.LINE_AA)
    cv2.putText(vis, text, (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (236, 231, 220), 2, cv2.LINE_AA)
    cv2.putText(vis, f"{fps:.1f} FPS | press v to toggle view", (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (169, 176, 173), 1, cv2.LINE_AA)
    return vis


def fit_for_display(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(float(max_width) / float(width), float(max_height) / float(height), 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)


def main() -> int:
    args = parse_args()
    username = args.user or input("AXIS user: ").strip()
    password = args.password if args.password is not None else getpass.getpass("AXIS password: ")

    setup_tx2_app(args)
    url = rtsp_url(args.ip, username, password, args.codec)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Could not open RTSP for {args.ip}. Check credentials/network.")
        return 2

    if args.snapshot_dir:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)

    print("Live processing is open. Keys: q/Esc exit, s snapshot, v toggle view.")
    started_at = time.time()
    frame_idx = 0
    last_result = None
    fps = 0.0
    last_tick = time.time()
    view = args.view

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read a frame from the stream.")
            break

        frame_idx += 1
        if frame_idx % max(1, args.process_every) == 0:
            try:
                last_result = process_frame(frame, frame_idx, started_at, args.conf, args.imgsz)
            except Exception as exc:
                print(f"Error processing frame {frame_idx}: {exc}")
                last_result = {
                    "rectified": frame,
                    "boxes": [],
                    "pieces": [],
                    "measurement_summary": {},
                    "sobel": empty_sobel(frame_idx, time.time() - started_at),
                    "measurement": None,
                }

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_tick)) if fps else 1.0 / max(1e-6, now - last_tick)
        last_tick = now

        if view == "rectified":
            display_frame = last_result["rectified"] if last_result and "rectified" in last_result else frame
            vis = draw_overlay(display_frame, last_result, fps)
        else:
            vis = draw_original_overlay(frame, last_result, fps)
        vis = fit_for_display(vis, args.max_width, args.max_height)
        cv2.imshow(args.window, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("v"):
            view = "rectified" if view == "original" else "original"
        if key == ord("s") and args.snapshot_dir:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = args.snapshot_dir / f"axis_live_{args.ip.replace('.', '_')}_{stamp}.jpg"
            cv2.imwrite(str(path), vis)
            print(f"Snapshot: {path}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
