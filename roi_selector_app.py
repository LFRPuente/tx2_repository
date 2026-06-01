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
from flask import Flask, jsonify, render_template_string, request

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

HTML = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>ROI Selector</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif;
       height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

header {
  background: #16213e; padding: 10px 20px;
  display: flex; align-items: center; gap: 14px;
  border-bottom: 1px solid #0f3460; flex-shrink: 0;
}
header h1 { font-size: 1rem; color: #e94560; font-weight: 600; }
.badge { background: #0f3460; color: #eee; padding: 3px 10px;
         border-radius: 20px; font-size: 0.78rem; }
.badge.green  { background: #1a6b3a; color: #7dff9a; }
.badge.yellow { background: #5c4a00; color: #ffd700; }

.toolbar {
  background: #16213e; padding: 8px 16px;
  display: flex; align-items: center; gap: 10px;
  border-bottom: 1px solid #0f3460; flex-shrink: 0; flex-wrap: wrap;
}
.toolbar label { font-size: 0.8rem; color: #aaa; }
.toolbar input[type=range] { width: 130px; accent-color: #e94560; }
button { padding: 5px 14px; border: none; border-radius: 5px;
         cursor: pointer; font-size: 0.82rem; font-weight: 600; transition: opacity .15s; }
button:hover { opacity: .85; }
button:disabled { opacity: .35; cursor: default; }
.btn-danger  { background: #c0392b; color: #fff; }
.btn-success { background: #1e8449; color: #fff; }
.btn-info    { background: #1a6b9a; color: #fff; }
.sep { width: 1px; height: 24px; background: #0f3460; }

.main { flex: 1; display: flex; overflow: hidden; }

.left-panel {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
  border-right: 1px solid #0f3460;
}
.panel-title { background: #0f3460; padding: 4px 12px;
               font-size: 0.78rem; color: #aad4ff; flex-shrink: 0; }

.canvas-wrap { flex: 1; overflow: hidden; position: relative;
               cursor: crosshair; background: #000; }
canvas { display: block; }

.coords-overlay {
  position: absolute; bottom: 8px; left: 8px;
  background: rgba(0,0,0,.65); color: #7dff9a;
  font-size: 0.72rem; padding: 3px 8px; border-radius: 4px;
  pointer-events: none;
}

.right-panel {
  width: 280px; flex-shrink: 0; background: #0d1b2a;
  display: flex; flex-direction: column; overflow-y: auto;
}
.info-block { padding: 14px 16px; border-bottom: 1px solid #0f3460; }
.info-block h3 { font-size: 0.82rem; color: #aad4ff; margin-bottom: 8px; }
.info-row { display: flex; justify-content: space-between;
            font-size: 0.78rem; padding: 3px 0; }
.info-row .lbl { color: #888; }
.info-row .val { color: #7dff9a; font-family: monospace; }
.info-row .val.empty { color: #555; }

#status-bar { background: #0d1b2a; padding: 5px 16px;
              font-size: 0.76rem; color: #aaa;
              border-top: 1px solid #0f3460; flex-shrink: 0; }
#status-bar.ok  { color: #7dff9a; }
#status-bar.err { color: #ff6b6b; }
</style>
</head>
<body>

<header>
  <h1>ROI Selector</h1>
  <span class="badge" id="src-badge">warp rectificado</span>
  <span class="badge" id="roi-badge">Sin ROI</span>
</header>

<div class="toolbar">
  <label>Zoom:</label>
  <input type="range" id="zoom-range" min="1" max="16" step="0.1" value="1">
  <span id="zoom-label" style="font-size:.8rem;min-width:36px;">1.0x</span>
  <div class="sep"></div>
  <button class="btn-danger" onclick="resetRoi()">Reiniciar ROI</button>
  <div class="sep"></div>
  <button class="btn-success" id="btn-save" onclick="saveRoi()" disabled>&#128190; Guardar ROI</button>
</div>

<div class="main">
  <div class="left-panel">
    <div class="panel-title">
      1er click → esquina A &nbsp;|&nbsp; 2do click → esquina B (confirma ROI) &nbsp;|&nbsp; Rueda: zoom &nbsp;|&nbsp; Botón derecho + arrastrar: pan
    </div>
    <div class="canvas-wrap" id="wrap">
      <canvas id="canvas"></canvas>
      <div class="coords-overlay" id="coords">x: — &nbsp; y: —</div>
    </div>
  </div>

  <div class="right-panel">
    <div class="info-block">
      <h3>Imagen</h3>
      <div class="info-row"><span class="lbl">Tamaño</span>
        <span class="val" id="img-size">—</span></div>
    </div>
    <div class="info-block">
      <h3>ROI — píxeles absolutos</h3>
      <div class="info-row"><span class="lbl">x</span>     <span class="val empty" id="r-x">—</span></div>
      <div class="info-row"><span class="lbl">y</span>     <span class="val empty" id="r-y">—</span></div>
      <div class="info-row"><span class="lbl">ancho</span> <span class="val empty" id="r-w">—</span></div>
      <div class="info-row"><span class="lbl">alto</span>  <span class="val empty" id="r-h">—</span></div>
    </div>
    <div class="info-block">
      <h3>ROI — normalizado [0–1]</h3>
      <div class="info-row"><span class="lbl">x_norm</span> <span class="val empty" id="r-xn">—</span></div>
      <div class="info-row"><span class="lbl">y_norm</span> <span class="val empty" id="r-yn">—</span></div>
      <div class="info-row"><span class="lbl">w_norm</span> <span class="val empty" id="r-wn">—</span></div>
      <div class="info-row"><span class="lbl">h_norm</span> <span class="val empty" id="r-hn">—</span></div>
    </div>
    <div class="info-block">
      <h3>Atajos de teclado</h3>
      <div class="info-row"><span class="lbl">Click ×2</span>    <span class="val">Definir ROI</span></div>
      <div class="info-row"><span class="lbl">R</span>            <span class="val">Reiniciar ROI</span></div>
      <div class="info-row"><span class="lbl">S</span>            <span class="val">Guardar</span></div>
      <div class="info-row"><span class="lbl">+  /  -</span>      <span class="val">Zoom</span></div>
      <div class="info-row"><span class="lbl">WASD / ↑↓←→</span> <span class="val">Pan</span></div>
    </div>
  </div>
</div>

<div id="status-bar">Cargando imagen…</div>

<script>
const state = {
  imgW: 0, imgH: 0,
  zoom: 1, panX: 0, panY: 0,
  roi: null,          // {x1,y1,x2,y2} confirmed ROI (never overwritten by preview)
  roiPreview: null,   // live preview while waiting for second click
  cornerA: null,      // {x,y} — first click, waiting for second
  panning: false,
  panAnchor: null,
  panPanAnchor: null,
  img: null,
};

const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');
const wrap   = document.getElementById('wrap');

// ── helpers ─────────────────────────────────────────────────────────────────
function setStatus(msg, cls='') {
  const el = document.getElementById('status-bar');
  el.textContent = msg; el.className = cls;
}

function fitCanvas() {
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
}

function clampPan() {
  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  state.panX = Math.max(0, Math.min(state.panX, Math.max(0, state.imgW - vw)));
  state.panY = Math.max(0, Math.min(state.panY, Math.max(0, state.imgH - vh)));
}

function dispToImg(cx, cy) {
  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  return {
    x: Math.max(0, Math.min(state.panX + (cx / canvas.width)  * vw, state.imgW - 1)),
    y: Math.max(0, Math.min(state.panY + (cy / canvas.height) * vh, state.imgH - 1)),
  };
}

function imgToDisp(ix, iy) {
  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  return {
    x: ((ix - state.panX) / vw) * canvas.width,
    y: ((iy - state.panY) / vh) * canvas.height,
  };
}

// ── render ───────────────────────────────────────────────────────────────────
function draw() {
  const cw = canvas.width, ch = canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!state.img) return;

  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  ctx.drawImage(state.img, state.panX, state.panY, vw, vh, 0, 0, cw, ch);

  // Grid
  const step = 100;
  ctx.lineWidth = 1;
  ctx.font = '11px monospace';
  const sx = state.panX, sy = state.panY;
  for (let gx = Math.ceil(sx / step) * step; gx < sx + vw; gx += step) {
    const px = ((gx - sx) / vw) * cw;
    ctx.strokeStyle = 'rgba(0,255,255,0.2)'; ctx.beginPath();
    ctx.moveTo(px, 0); ctx.lineTo(px, ch); ctx.stroke();
    ctx.fillStyle = 'rgba(0,255,255,0.6)'; ctx.fillText(gx, px + 3, 13);
  }
  for (let gy = Math.ceil(sy / step) * step; gy < sy + vh; gy += step) {
    const py = ((gy - sy) / vh) * ch;
    ctx.strokeStyle = 'rgba(255,255,0,0.2)'; ctx.beginPath();
    ctx.moveTo(0, py); ctx.lineTo(cw, py); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,0,0.6)'; ctx.fillText(gy, 3, py + 12);
  }

  // ROI rect — confirmed takes priority, otherwise show live preview
  const roiToDraw = state.roi || state.roiPreview;
  if (roiToDraw) {
    const {x1,y1,x2,y2} = roiToDraw;
    const rx = Math.min(x1,x2), ry = Math.min(y1,y2);
    const rw = Math.abs(x2-x1),  rh = Math.abs(y2-y1);
    const d1 = imgToDisp(rx,    ry);
    const d2 = imgToDisp(rx+rw, ry+rh);
    const dw = d2.x - d1.x, dh = d2.y - d1.y;

    // Dimmed outside
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, cw, d1.y);                         // top
    ctx.fillRect(0, d1.y, d1.x, dh);                      // left
    ctx.fillRect(d2.x, d1.y, cw - d2.x, dh);              // right
    ctx.fillRect(0, d2.y, cw, ch - d2.y);                  // bottom

    // Border
    ctx.strokeStyle = '#e94560';
    ctx.lineWidth = 2.5;
    ctx.setLineDash([]);
    ctx.strokeRect(d1.x, d1.y, dw, dh);

    // Corner handles
    const handleSize = 7;
    ctx.fillStyle = '#e94560';
    [[d1.x,d1.y],[d2.x,d1.y],[d2.x,d2.y],[d1.x,d2.y]].forEach(([hx,hy]) => {
      ctx.fillRect(hx - handleSize/2, hy - handleSize/2, handleSize, handleSize);
    });

    // Label
    ctx.fillStyle = 'rgba(233,69,96,0.9)';
    ctx.font = 'bold 12px monospace';
    ctx.fillText(`${Math.round(rw)} × ${Math.round(rh)} px`, d1.x + 6, d1.y + 18);
  }
}

// ── info panel ───────────────────────────────────────────────────────────────
function updateInfo() {
  const badge = document.getElementById('roi-badge');
  const btn   = document.getElementById('btn-save');

  if (!state.roi) {
    badge.textContent = 'Sin ROI'; badge.className = 'badge';
    btn.disabled = true;
    ['r-x','r-y','r-w','r-h','r-xn','r-yn','r-wn','r-hn'].forEach(id => {
      const el = document.getElementById(id);
      el.textContent = '—'; el.className = 'val empty';
    });
    return;
  }

  const {x1,y1,x2,y2} = state.roi;
  const rx = Math.round(Math.min(x1,x2)), ry = Math.round(Math.min(y1,y2));
  const rw = Math.round(Math.abs(x2-x1)),  rh = Math.round(Math.abs(y2-y1));

  badge.textContent = `ROI: ${rw}×${rh}`; badge.className = 'badge green';
  btn.disabled = (rw < 4 || rh < 4);

  const set = (id, v) => { const el = document.getElementById(id); el.textContent = v; el.className = 'val'; };
  set('r-x', rx); set('r-y', ry); set('r-w', rw); set('r-h', rh);
  set('r-xn', (rx / state.imgW).toFixed(4));
  set('r-yn', (ry / state.imgH).toFixed(4));
  set('r-wn', (rw / state.imgW).toFixed(4));
  set('r-hn', (rh / state.imgH).toFixed(4));
}

// ── load image ───────────────────────────────────────────────────────────────
async function loadImage() {
  setStatus('Cargando imagen rectificada…');
  const r = await fetch('/api/image');
  const d = await r.json();
  if (!r.ok) { setStatus(d.error, 'err'); return; }
  const img = new Image();
  img.onload = () => {
    state.img  = img;
    state.imgW = d.width;
    state.imgH = d.height;
    document.getElementById('img-size').textContent = `${d.width} × ${d.height}`;
    document.getElementById('src-badge').textContent = d.label;
    fitCanvas(); draw();
    setStatus(`Imagen cargada: ${d.width}×${d.height}`, 'ok');
  };
  img.src = 'data:image/jpeg;base64,' + d.image;
}

// ── save ─────────────────────────────────────────────────────────────────────
async function saveRoi() {
  if (!state.roi) return;
  const {x1,y1,x2,y2} = state.roi;
  const rx = Math.min(x1,x2), ry = Math.min(y1,y2);
  const rw = Math.abs(x2-x1),  rh = Math.abs(y2-y1);
  setStatus('Guardando…');
  const r = await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({x: rx, y: ry, w: rw, h: rh,
                          img_w: state.imgW, img_h: state.imgH})
  });
  const d = await r.json();
  if (!r.ok) { setStatus(d.error, 'err'); return; }
  setStatus('✔ ROI guardado en: ' + d.path, 'ok');
}

function resetRoi() {
  state.roi = null;
  state.roiPreview = null;
  state.cornerA = null;
  updateInfo(); draw();
  setStatus('ROI reiniciado. Haz click en la esquina A.');
}

// ── mouse ────────────────────────────────────────────────────────────────────
// Two-click ROI mode: first click = corner A, second click = corner B
// Right-click drag = pan (never interferes with ROI clicks)

wrap.addEventListener('click', e => {
  if (e.button !== 0) return;
  if (state.panning) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  const {x, y} = dispToImg(cx, cy);

  if (!state.cornerA) {
    // First click — record corner A, clear any previous confirmed ROI
    state.cornerA = {x, y};
    state.roi = null;
    state.roiPreview = {x1: x, y1: y, x2: x, y2: y};
    setStatus(`Esquina A: (${Math.round(x)}, ${Math.round(y)}) — ahora haz click en la esquina opuesta`);
    updateInfo(); draw();
  } else {
    // Second click — confirm ROI, stop live preview
    const confirmed = {x1: state.cornerA.x, y1: state.cornerA.y, x2: x, y2: y};
    state.cornerA = null;
    state.roiPreview = null;
    const rw = Math.abs(confirmed.x2 - confirmed.x1);
    const rh = Math.abs(confirmed.y2 - confirmed.y1);
    if (rw < 5 || rh < 5) {
      state.roi = null;
      setStatus('ROI demasiado pequeño, inténtalo de nuevo.');
    } else {
      state.roi = confirmed;
      setStatus(`ROI confirmado: ${Math.round(rw)} × ${Math.round(rh)} px — presiona S para guardar`, 'ok');
    }
    updateInfo(); draw();
  }
});

// Live preview while waiting for second click
document.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const cx = Math.max(0, Math.min(e.clientX - rect.left, canvas.width  - 1));
  const cy = Math.max(0, Math.min(e.clientY - rect.top,  canvas.height - 1));
  const {x, y} = dispToImg(cx, cy);
  document.getElementById('coords').textContent = `x: ${Math.round(x)}  y: ${Math.round(y)}`;

  if (state.cornerA) {
    // Update live preview only — never touch confirmed roi
    state.roiPreview = {x1: state.cornerA.x, y1: state.cornerA.y, x2: x, y2: y};
    updateInfo(); draw();
  } else if (state.panning && state.panAnchor) {
    const dx = e.clientX - state.panAnchor.x;
    const dy = e.clientY - state.panAnchor.y;
    const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
    state.panX = state.panPanAnchor.x - dx * vw / canvas.width;
    state.panY = state.panPanAnchor.y - dy * vh / canvas.height;
    clampPan(); draw();
  }
});

wrap.addEventListener('mousedown', e => {
  if (e.button === 2 || e.button === 1) {
    e.preventDefault();
    state.panning = true;
    state.panAnchor = {x: e.clientX, y: e.clientY};
    state.panPanAnchor = {x: state.panX, y: state.panY};
    wrap.style.cursor = 'grabbing';
  }
});

document.addEventListener('mouseup', e => {
  if (state.panning) {
    state.panning = false;
    state.panAnchor = null;
    wrap.style.cursor = 'crosshair';
  }
});

wrap.addEventListener('contextmenu', e => e.preventDefault());

wrap.addEventListener('wheel', e => {
  e.preventDefault();
  if (!state.img) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  const {x: ax, y: ay} = dispToImg(cx, cy);
  const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
  state.zoom = Math.max(1, Math.min(state.zoom * factor, 20));
  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  state.panX = ax - (cx / canvas.width)  * vw;
  state.panY = ay - (cy / canvas.height) * vh;
  clampPan();
  document.getElementById('zoom-range').value = state.zoom;
  document.getElementById('zoom-label').textContent = state.zoom.toFixed(1) + 'x';
  draw();
}, {passive: false});

document.getElementById('zoom-range').addEventListener('input', function() {
  const cx = canvas.width/2, cy = canvas.height/2;
  const {x: ax, y: ay} = dispToImg(cx, cy);
  state.zoom = parseFloat(this.value);
  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  state.panX = ax - 0.5 * vw; state.panY = ay - 0.5 * vh;
  clampPan();
  document.getElementById('zoom-label').textContent = state.zoom.toFixed(1) + 'x';
  draw();
});

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'r' || e.key === 'R') { resetRoi(); return; }
  if (e.key === 's' || e.key === 'S') { saveRoi(); return; }
  const step = 40 / state.zoom;
  if (e.key === 'ArrowLeft'  || e.key === 'a') { state.panX -= step; clampPan(); draw(); }
  if (e.key === 'ArrowRight' || e.key === 'd') { state.panX += step; clampPan(); draw(); }
  if (e.key === 'ArrowUp'    || e.key === 'w') { state.panY -= step; clampPan(); draw(); }
  if (e.key === 'ArrowDown')                   { state.panY += step; clampPan(); draw(); }
  if (e.key === '+' || e.key === '=') { state.zoom = Math.min(state.zoom*1.2,20); clampPan(); draw(); }
  if (e.key === '-' || e.key === '_') { state.zoom = Math.max(state.zoom/1.2,1);  clampPan(); draw(); }
});

window.addEventListener('resize', () => { fitCanvas(); draw(); });

fitCanvas();
loadImage();
</script>
</body>
</html>
"""

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
    return render_template_string(HTML)

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
