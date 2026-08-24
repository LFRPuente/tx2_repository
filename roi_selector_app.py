"""
ROI Selector — Flask web app.

Carga el warp rectificado de la homografia y permite seleccionar
un rectangulo de interes (ROI) con drag, zoom y pan.
Guarda el ROI como coordenadas absolutas y normalizadas en JSON.

Usage:
    python roi_selector_app.py
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

DEFAULT_HOMOGRAPHY_JSON = Path(r"C:\Users\luis_\Desktop\tx2_cv_2026-05-11\outputs\homography_selection.json")
DEFAULT_OUTPUT_DIR      = Path(r"C:\Users\luis_\Desktop\tx2_cv_2026-05-11\outputs")
DEFAULT_PORT            = 5051

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--homography", type=Path, default=DEFAULT_HOMOGRAPHY_JSON)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    return p.parse_args()

def img_to_b64(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("imencode failed")
    return base64.b64encode(buf).decode()

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

app   = Flask(__name__)
_args: argparse.Namespace = None
_warp_image: np.ndarray  = None
_warp_label: str         = ""

def load_warp(homography_json: Path) -> tuple[np.ndarray, str]:
    data = json.loads(homography_json.read_text(encoding="utf-8"))
    warp_preview = Path(data["warp_preview"])
    if not warp_preview.exists():
        raise RuntimeError(
            f"No se encontró el warp preview: {warp_preview}\n"
            "Re-ejecuta homography_web_app.py y guarda primero."
        )
    img = cv2.imread(str(warp_preview))
    if img is None:
        raise RuntimeError(f"No se pudo leer: {warp_preview}")
    label = data.get("source", str(warp_preview))
    return img, label

@app.route("/")
def index():
    return render_template("roi_selector.html")

@app.route("/api/image")
def api_image():
    if _warp_image is None:
        return jsonify(error="No hay imagen cargada"), 500
    b64 = img_to_b64(_warp_image, quality=90)
    h, w = _warp_image.shape[:2]
    return jsonify(image=b64, width=w, height=h, label=_warp_label)

@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json()
    x, y = float(data["x"]), float(data["y"])
    w, h = float(data["w"]), float(data["h"])
    iw, ih = float(data["img_w"]), float(data["img_h"])

    out_dir = _args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "roi_selection.json"

    # Draw ROI on warp image and save preview
    vis = _warp_image.copy()
    cv2.rectangle(vis, (int(x), int(y)), (int(x+w), int(y+h)), (0, 80, 255), 3)
    cv2.putText(vis, f"ROI  {int(w)}x{int(h)}",
                (int(x)+6, int(y)+26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2, cv2.LINE_AA)
    preview_path = out_dir / "roi_selection_preview.jpg"
    cv2.imwrite(str(preview_path), vis)

    payload = {
        "source_homography": str(_args.homography),
        "image_size": [int(iw), int(ih)],
        "roi_abs": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "roi_norm": {
            "x": round(x / iw, 6), "y": round(y / ih, 6),
            "w": round(w / iw, 6), "h": round(h / ih, 6),
        },
        "preview": str(preview_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return jsonify(path=str(json_path))


def main():
    global _args, _warp_image, _warp_label
    _args = parse_args()
    try:
        _warp_image, _warp_label = load_warp(_args.homography)
        h, w = _warp_image.shape[:2]
        print(f"  Warp cargado: {w}×{h}")
    except Exception as e:
        print(f"ERROR cargando warp: {e}")
        raise
    print(f"\n  ROI Selector en  http://localhost:{_args.port}\n")
    app.run(host="0.0.0.0", port=_args.port, debug=False)


if __name__ == "__main__":
    main()
