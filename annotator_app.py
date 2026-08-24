"""
YOLO Annotator — Flask web app.

Carga frames del video con homografia aplicada.
Permite anotar una bounding box por pieza para entrenar YOLOv11.
Guarda anotaciones en formato YOLO (labels/frame_XXXXX.txt + images/frame_XXXXX.jpg).

Usage:
    python annotator_app.py
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

DEFAULT_VIDEO          = Path(r"C:\Users\luis_\Downloads\20260508_000307_7F66.mkv")
DEFAULT_HOMOGRAPHY     = Path(r"C:\Users\luis_\Desktop\tx2_cv_2026-05-11\outputs\homography_selection.json")
DEFAULT_OUTPUT_DIR     = Path(r"C:\Users\luis_\Desktop\tx2_cv_2026-05-11\dataset_pieces")
DEFAULT_PORT           = 5052

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video",      type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--homography", type=Path, default=DEFAULT_HOMOGRAPHY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--port",       type=int,  default=DEFAULT_PORT)
    return p.parse_args()

def img_to_b64(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(buf).decode()

def apply_homography(frame: np.ndarray, H: np.ndarray, out_size: tuple) -> np.ndarray:
    return cv2.warpPerspective(frame, H, out_size)


# ── Flask ─────────────────────────────────────────────────────────────────────

app    = Flask(__name__)
_args  : argparse.Namespace = None
_cap   : cv2.VideoCapture   = None
_H     : np.ndarray         = None
_out_size: tuple            = None
_saved_count: int           = 0

def load_homography(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    H    = np.array(data["homography_matrix"], dtype=np.float64)
    w, h = data["output_size"]
    return H, (w, h)

def get_frame(frame_idx: int) -> np.ndarray:
    _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = _cap.read()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {frame_idx}")
    return frame

@app.route("/")
def index():
    return render_template("annotator.html")

@app.route("/api/frame", methods=["POST"])
def api_frame():
    data      = request.get_json()
    frame_idx = int(data.get("frame_idx", 0))
    total     = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, total - 1))
    fps       = _cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        frame = get_frame(frame_idx)
        if _H is not None:
            frame = apply_homography(frame, _H, _out_size)
        b64 = img_to_b64(frame, quality=88)
        h, w = frame.shape[:2]
        return jsonify(image=b64, width=w, height=h,
                       frame_idx=frame_idx, time_sec=round(frame_idx / fps, 3))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/save", methods=["POST"])
def api_save():
    global _saved_count
    data      = request.get_json()
    frame_idx = int(data["frame_idx"])
    time_sec  = float(data["time_sec"])
    img_w     = int(data["img_w"])
    img_h     = int(data["img_h"])
    boxes     = data["boxes"]  # [{x, y, w, h}] in image pixels

    images_dir = _args.output_dir / "images"
    labels_dir = _args.output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stem = f"frame_{frame_idx:06d}"

    # Save image
    frame = get_frame(frame_idx)
    if _H is not None:
        frame = apply_homography(frame, _H, _out_size)
    img_path = images_dir / f"{stem}.jpg"
    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save YOLO label (class 0 = piece)
    label_path = labels_dir / f"{stem}.txt"
    lines = []
    for b in boxes:
        cx = (b["x"] + b["w"] / 2) / img_w
        cy = (b["y"] + b["h"] / 2) / img_h
        nw = b["w"] / img_w
        nh = b["h"] / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")

    # Save metadata JSON alongside
    meta_path = labels_dir / f"{stem}.json"
    meta_path.write_text(json.dumps({
        "frame_idx": frame_idx, "time_sec": time_sec,
        "img_w": img_w, "img_h": img_h, "boxes": boxes,
    }, indent=2), encoding="utf-8")

    _saved_count += 1
    return jsonify(saved_count=_saved_count, image=str(img_path), label=str(label_path))

def main():
    global _args, _cap, _H, _out_size
    _args = parse_args()
    _args.output_dir.mkdir(parents=True, exist_ok=True)

    _cap = cv2.VideoCapture(str(_args.video))
    if not _cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {_args.video}")

    if _args.homography.exists():
        _H, _out_size = load_homography(_args.homography)
        print(f"  Homografía cargada — warp: {_out_size[0]}×{_out_size[1]}")
    else:
        print("  Sin homografía — usando frame crudo")

    fps   = _cap.get(cv2.CAP_PROP_FPS)
    total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {total} frames @ {fps:.1f}fps ({total/fps:.1f}s)")
    print(f"  Dataset: {_args.output_dir}")
    print(f"\n  Annotator en  http://localhost:{_args.port}\n")
    app.run(host="0.0.0.0", port=_args.port, debug=False)

if __name__ == "__main__":
    main()
