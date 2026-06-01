"""
YOLO Annotator — Flask web app.

Carga frames del video con homografia aplicada.
Permite anotar bounding boxes de tubos para entrenar YOLOv11.
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
from flask import Flask, jsonify, render_template_string, request

DEFAULT_VIDEO          = Path(r"C:\Users\luis_\Downloads\20260508_000307_7F66.mkv")
DEFAULT_HOMOGRAPHY     = Path(r"C:\Users\luis_\Desktop\tx2_cv_2026-05-11\outputs\homography_selection.json")
DEFAULT_OUTPUT_DIR     = Path(r"C:\Users\luis_\Desktop\tx2_cv_2026-05-11\dataset")
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

HTML = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>YOLO Annotator</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif;
       height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

header { background: #16213e; padding: 8px 16px; display: flex; align-items: center;
         gap: 12px; border-bottom: 1px solid #0f3460; flex-shrink: 0; }
header h1 { font-size: .95rem; color: #e94560; font-weight: 700; }
.badge { background: #0f3460; color: #eee; padding: 2px 9px; border-radius: 20px; font-size: .75rem; }
.badge.green  { background: #1a6b3a; color: #7dff9a; }
.badge.yellow { background: #5c4a00; color: #ffd700; }
.badge.red    { background: #6b1a1a; color: #ff9a9a; }

.toolbar { background: #16213e; padding: 6px 14px; display: flex; align-items: center;
           gap: 8px; border-bottom: 1px solid #0f3460; flex-shrink: 0; flex-wrap: wrap; }
.toolbar label { font-size: .78rem; color: #aaa; }
input[type=range]  { accent-color: #e94560; }
input[type=number] { width: 72px; background: #0f3460; border: 1px solid #1e5fa0;
                     color: #eee; padding: 2px 5px; border-radius: 4px; font-size: .78rem; }
button { padding: 4px 12px; border: none; border-radius: 5px; cursor: pointer;
         font-size: .78rem; font-weight: 600; transition: opacity .15s; }
button:hover { opacity: .82; }
button:disabled { opacity: .3; cursor: default; }
.btn-primary { background: #1a6b9a; color: #fff; }
.btn-success { background: #1e8449; color: #fff; }
.btn-danger  { background: #c0392b; color: #fff; }
.btn-warn    { background: #b7770d; color: #fff; }
.sep { width: 1px; height: 22px; background: #0f3460; }

.main { flex: 1; display: flex; overflow: hidden; }

.canvas-col { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.panel-title { background: #0f3460; padding: 3px 12px; font-size: .75rem; color: #aad4ff; flex-shrink: 0; }
.canvas-wrap { flex: 1; position: relative; overflow: hidden; cursor: crosshair; background: #000; }
canvas { display: block; }
.coords-hud { position: absolute; bottom: 8px; left: 8px; background: rgba(0,0,0,.7);
              color: #7dff9a; font-size: .7rem; padding: 2px 8px; border-radius: 4px; pointer-events: none; }

.side-panel { width: 260px; flex-shrink: 0; background: #0d1b2a; display: flex;
              flex-direction: column; overflow: hidden; border-left: 1px solid #0f3460; }
.side-section { padding: 10px 12px; border-bottom: 1px solid #0f3460; }
.side-section h3 { font-size: .78rem; color: #aad4ff; margin-bottom: 6px; }
.kv { display: flex; justify-content: space-between; font-size: .74rem; padding: 2px 0; }
.kv .k { color: #777; } .kv .v { color: #7dff9a; font-family: monospace; }

.box-list { flex: 1; overflow-y: auto; padding: 6px 0; }
.box-item { display: flex; align-items: center; gap: 6px; padding: 4px 10px;
            border-bottom: 1px solid #0f3460; font-size: .72rem; cursor: pointer; }
.box-item:hover { background: #1a2a3a; }
.box-item.selected { background: #0f3460; }
.box-color { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
.box-del { margin-left: auto; color: #e94560; font-weight: 700; cursor: pointer;
           padding: 0 4px; font-size: .9rem; }
.box-del:hover { color: #ff6b6b; }

#status-bar { background: #0d1b2a; padding: 4px 14px; font-size: .72rem; color: #aaa;
              border-top: 1px solid #0f3460; flex-shrink: 0; }
#status-bar.ok  { color: #7dff9a; }
#status-bar.err { color: #ff6b6b; }
#status-bar.info { color: #88ccff; }
</style>
</head>
<body>

<header>
  <h1>YOLO Annotator — Tubos</h1>
  <span class="badge" id="badge-frame">frame —</span>
  <span class="badge" id="badge-boxes">0 boxes</span>
  <span class="badge" id="badge-saved">0 frames anotados</span>
  <span class="badge yellow" id="badge-mode">modo: tubos</span>
</header>

<div class="toolbar">
  <!-- Frame navigation -->
  <label>Segundo:</label>
  <input type="number" id="inp-second" value="10" step="1" min="0" max="301" style="width:64px">
  <button class="btn-primary" onclick="loadSecond()">Ir</button>
  <button class="btn-primary" onclick="stepFrame(-30)" title="−1s">◀◀</button>
  <button class="btn-primary" onclick="stepFrame(-1)"  title="−1 frame">◀</button>
  <button class="btn-primary" onclick="stepFrame(1)"   title="+1 frame">▶</button>
  <button class="btn-primary" onclick="stepFrame(30)"  title="+1s">▶▶</button>
  <div class="sep"></div>
  <!-- Zoom -->
  <label>Zoom:</label>
  <input type="range" id="zoom-range" min="1" max="16" step="0.1" value="1" style="width:100px">
  <span id="zoom-lbl" style="font-size:.75rem;min-width:32px">1.0x</span>
  <div class="sep"></div>
  <button class="btn-warn"    onclick="clearBoxes()">Limpiar boxes</button>
  <button class="btn-success" id="btn-save" onclick="saveFrame()" disabled>💾 Guardar frame</button>
  <div class="sep"></div>
  <label style="color:#88ccff;font-size:.75rem">Z=deshacer · R=limpiar · S=guardar · flechas=navegar</label>
</div>

<div class="main">
  <div class="canvas-col">
    <div class="panel-title" id="canvas-hint">
      1er click → esquina A de la box &nbsp;|&nbsp; 2do click → esquina B &nbsp;|&nbsp; Rueda: zoom &nbsp;|&nbsp; Botón derecho + arrastrar: pan
    </div>
    <div class="canvas-wrap" id="wrap">
      <canvas id="canvas"></canvas>
      <div class="coords-hud" id="coords-hud">x: — &nbsp; y: —</div>
    </div>
  </div>

  <div class="side-panel">
    <div class="side-section">
      <h3>Frame actual</h3>
      <div class="kv"><span class="k">Frame</span>   <span class="v" id="inf-frame">—</span></div>
      <div class="kv"><span class="k">Tiempo</span>  <span class="v" id="inf-time">—</span></div>
      <div class="kv"><span class="k">Tamaño</span>  <span class="v" id="inf-size">—</span></div>
    </div>
    <div class="side-section">
      <h3>Boxes en este frame</h3>
      <div class="box-list" id="box-list"><span style="color:#555;font-size:.72rem;padding:6px 10px;display:block">Sin boxes.</span></div>
    </div>
    <div class="side-section">
      <h3>Atajos</h3>
      <div class="kv"><span class="k">Click ×2</span>   <span class="v">Dibujar box</span></div>
      <div class="kv"><span class="k">Z</span>           <span class="v">Deshacer última</span></div>
      <div class="kv"><span class="k">R</span>           <span class="v">Limpiar todas</span></div>
      <div class="kv"><span class="k">S</span>           <span class="v">Guardar frame</span></div>
      <div class="kv"><span class="k">← →</span>         <span class="v">±1 frame</span></div>
      <div class="kv"><span class="k">A / D</span>       <span class="v">±1 s</span></div>
    </div>
    <div class="side-section" style="margin-top:auto">
      <h3>Dataset</h3>
      <div class="kv"><span class="k">Frames guardados</span> <span class="v" id="inf-saved">0</span></div>
      <div class="kv"><span class="k">Total boxes</span>      <span class="v" id="inf-total-boxes">0</span></div>
    </div>
  </div>
</div>

<div id="status-bar" class="info">Cargando frame…</div>

<script>
// ── palette ──────────────────────────────────────────────────────────────────
const COLORS = [
  '#e94560','#44ff88','#4488ff','#ffcc00','#ff88cc',
  '#00ffff','#ff8800','#aa88ff','#88ff00','#ff4488',
];

// ── state ────────────────────────────────────────────────────────────────────
const state = {
  imgW: 0, imgH: 0,
  frameIdx: 0, timeSec: 0,
  zoom: 1, panX: 0, panY: 0,
  img: null,
  boxes: [],          // [{x1,y1,x2,y2}] in image coords — confirmed
  boxPreview: null,   // live preview {x1,y1,x2,y2}
  cornerA: null,      // {x,y} — waiting for second click
  panning: false, panAnchor: null, panPanAnchor: null,
  savedCount: 0,
  totalBoxes: 0,
  savedFrames: new Set(),
};

const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');
const wrap   = document.getElementById('wrap');

// ── helpers ───────────────────────────────────────────────────────────────────
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
function normalizeBox(b) {
  return {
    x: Math.min(b.x1, b.x2), y: Math.min(b.y1, b.y2),
    w: Math.abs(b.x2 - b.x1), h: Math.abs(b.y2 - b.y1),
  };
}

// ── render ────────────────────────────────────────────────────────────────────
function draw() {
  const cw = canvas.width, ch = canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!state.img) return;

  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  ctx.drawImage(state.img, state.panX, state.panY, vw, vh, 0, 0, cw, ch);

  // Light grid
  ctx.font = '10px monospace'; ctx.lineWidth = 1;
  const step = 100, sx = state.panX, sy = state.panY;
  for (let gx = Math.ceil(sx/step)*step; gx < sx+vw; gx += step) {
    const px = ((gx-sx)/vw)*cw;
    ctx.strokeStyle='rgba(0,255,255,0.15)'; ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,ch); ctx.stroke();
    ctx.fillStyle='rgba(0,255,255,0.5)'; ctx.fillText(gx, px+2, 12);
  }
  for (let gy = Math.ceil(sy/step)*step; gy < sy+vh; gy += step) {
    const py = ((gy-sy)/vh)*ch;
    ctx.strokeStyle='rgba(255,255,0,0.15)'; ctx.beginPath(); ctx.moveTo(0,py); ctx.lineTo(cw,py); ctx.stroke();
    ctx.fillStyle='rgba(255,255,0,0.5)'; ctx.fillText(gy, 2, py+12);
  }

  // Confirmed boxes
  state.boxes.forEach((b, i) => drawBox(b, COLORS[i % COLORS.length], i+1, false));

  // Live preview
  if (state.boxPreview) drawBox(state.boxPreview, '#ffffff', '?', true);
}

function drawBox(b, color, label, dashed) {
  const nb = normalizeBox(b);
  const d1 = imgToDisp(nb.x,        nb.y);
  const d2 = imgToDisp(nb.x + nb.w, nb.y + nb.h);
  const dw = d2.x - d1.x, dh = d2.y - d1.y;

  ctx.strokeStyle = color;
  ctx.lineWidth   = dashed ? 1.5 : 2;
  ctx.setLineDash(dashed ? [6,4] : []);
  ctx.strokeRect(d1.x, d1.y, dw, dh);
  ctx.setLineDash([]);

  // Fill semi-transparent
  ctx.fillStyle = color.replace(')', ',0.08)').replace('rgb', 'rgba').replace('#', 'rgba(').replace('rgba(', 'rgba(') ;
  // simpler:
  ctx.save();
  ctx.globalAlpha = 0.08;
  ctx.fillStyle = color;
  ctx.fillRect(d1.x, d1.y, dw, dh);
  ctx.restore();

  // Label
  if (!dashed) {
    ctx.fillStyle = color;
    ctx.font = 'bold 11px monospace';
    ctx.fillText(`#${label}  ${Math.round(nb.w)}×${Math.round(nb.h)}`, d1.x + 4, d1.y + 14);
  }

  // Corners
  if (!dashed) {
    const hs = 5;
    ctx.fillStyle = color;
    [[d1.x,d1.y],[d2.x,d1.y],[d2.x,d2.y],[d1.x,d2.y]].forEach(([hx,hy]) => {
      ctx.fillRect(hx-hs/2, hy-hs/2, hs, hs);
    });
  }
}

// ── box list sidebar ──────────────────────────────────────────────────────────
function updateBoxList() {
  const list = document.getElementById('box-list');
  document.getElementById('badge-boxes').textContent = `${state.boxes.length} boxes`;
  document.getElementById('badge-boxes').className = 'badge' + (state.boxes.length > 0 ? ' green' : '');
  document.getElementById('btn-save').disabled = state.boxes.length === 0;

  if (state.boxes.length === 0) {
    list.innerHTML = '<span style="color:#555;font-size:.72rem;padding:6px 10px;display:block">Sin boxes.</span>';
    return;
  }
  list.innerHTML = state.boxes.map((b, i) => {
    const nb = normalizeBox(b);
    return `<div class="box-item" onclick="selectBox(${i})">
      <div class="box-color" style="background:${COLORS[i%COLORS.length]}"></div>
      <span>#${i+1} &nbsp; ${Math.round(nb.w)}×${Math.round(nb.h)} @ (${Math.round(nb.x)},${Math.round(nb.y)})</span>
      <span class="box-del" onclick="deleteBox(event,${i})">✕</span>
    </div>`;
  }).join('');
}

function selectBox(i) { /* visual highlight future */ }
function deleteBox(e, i) {
  e.stopPropagation();
  state.boxes.splice(i, 1);
  updateBoxList(); draw();
}

// ── frame loading ─────────────────────────────────────────────────────────────
async function loadFrame(frameIdx) {
  setStatus('Cargando frame…', 'info');
  try {
    const r = await fetch('/api/frame', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({frame_idx: frameIdx})
    });
    const d = await r.json();
    if (!r.ok) { setStatus(d.error, 'err'); return; }

    const img = new Image();
    img.onload = () => {
      state.img      = img;
      state.imgW     = d.width;
      state.imgH     = d.height;
      state.frameIdx = d.frame_idx;
      state.timeSec  = d.time_sec;
      state.boxes    = [];
      state.boxPreview = null;
      state.cornerA    = null;

      document.getElementById('inp-second').value = d.time_sec.toFixed(1);
      document.getElementById('badge-frame').textContent  = `frame ${d.frame_idx}`;
      document.getElementById('inf-frame').textContent    = d.frame_idx;
      document.getElementById('inf-time').textContent     = d.time_sec.toFixed(2) + 's';
      document.getElementById('inf-size').textContent     = `${d.width}×${d.height}`;
      const alreadySaved = state.savedFrames.has(d.frame_idx);
      document.getElementById('badge-frame').className = 'badge' + (alreadySaved ? ' green' : '');
      fitCanvas(); updateBoxList(); draw();
      setStatus(`Frame ${d.frame_idx} @ ${d.time_sec.toFixed(2)}s${alreadySaved ? ' — ya anotado' : ''}`, alreadySaved ? 'ok' : 'info');
    };
    img.src = 'data:image/jpeg;base64,' + d.image;
  } catch(e) { setStatus('Error: ' + e, 'err'); }
}

function loadSecond() {
  const s = parseFloat(document.getElementById('inp-second').value) || 0;
  const fi = Math.round(s * 30);
  loadFrame(fi);
}

function stepFrame(delta) {
  loadFrame(Math.max(0, state.frameIdx + delta));
}

// ── save frame ────────────────────────────────────────────────────────────────
async function saveFrame() {
  if (state.boxes.length === 0) return;
  setStatus('Guardando…', 'info');
  const payload = {
    frame_idx: state.frameIdx,
    time_sec:  state.timeSec,
    img_w:     state.imgW,
    img_h:     state.imgH,
    boxes:     state.boxes.map(b => normalizeBox(b)),
  };
  try {
    const r = await fetch('/api/save', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (!r.ok) { setStatus(d.error, 'err'); return; }
    state.savedFrames.add(state.frameIdx);
    state.savedCount   = d.saved_count;
    state.totalBoxes  += state.boxes.length;
    document.getElementById('badge-saved').textContent   = `${d.saved_count} frames anotados`;
    document.getElementById('badge-saved').className     = 'badge green';
    document.getElementById('badge-frame').className     = 'badge green';
    document.getElementById('inf-saved').textContent     = d.saved_count;
    document.getElementById('inf-total-boxes').textContent = state.totalBoxes;
    setStatus(`✔ Frame ${state.frameIdx} guardado — ${state.boxes.length} boxes`, 'ok');
  } catch(e) { setStatus('Error: ' + e, 'err'); }
}

// ── clear ────────────────────────────────────────────────────────────────────
function clearBoxes() {
  state.boxes = []; state.boxPreview = null; state.cornerA = null;
  updateBoxList(); draw();
  setStatus('Boxes eliminadas.', 'info');
}

// ── mouse: two-click box drawing ──────────────────────────────────────────────
wrap.addEventListener('click', e => {
  if (e.button !== 0 || state.panning) return;
  const rect = canvas.getBoundingClientRect();
  const {x, y} = dispToImg(e.clientX - rect.left, e.clientY - rect.top);

  if (!state.cornerA) {
    state.cornerA    = {x, y};
    state.boxPreview = {x1: x, y1: y, x2: x, y2: y};
    setStatus(`Esquina A (${Math.round(x)}, ${Math.round(y)}) — ahora click en esquina opuesta`, 'info');
    draw();
  } else {
    const confirmed = {x1: state.cornerA.x, y1: state.cornerA.y, x2: x, y2: y};
    state.cornerA    = null;
    state.boxPreview = null;
    const nb = normalizeBox(confirmed);
    if (nb.w < 4 || nb.h < 4) {
      setStatus('Box demasiado pequeña, inténtalo de nuevo.', 'err');
    } else {
      state.boxes.push(confirmed);
      setStatus(`Box #${state.boxes.length} añadida (${Math.round(nb.w)}×${Math.round(nb.h)}) — S para guardar`, 'ok');
    }
    updateBoxList(); draw();
  }
});

document.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const cx = Math.max(0, Math.min(e.clientX - rect.left, canvas.width  - 1));
  const cy = Math.max(0, Math.min(e.clientY - rect.top,  canvas.height - 1));
  const {x, y} = dispToImg(cx, cy);
  document.getElementById('coords-hud').textContent = `x: ${Math.round(x)}  y: ${Math.round(y)}`;

  if (state.cornerA) {
    state.boxPreview = {x1: state.cornerA.x, y1: state.cornerA.y, x2: x, y2: y};
    draw();
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
    state.panAnchor     = {x: e.clientX, y: e.clientY};
    state.panPanAnchor  = {x: state.panX, y: state.panY};
    wrap.style.cursor   = 'grabbing';
  }
});
document.addEventListener('mouseup', e => {
  if (state.panning) {
    state.panning = false; state.panAnchor = null;
    wrap.style.cursor = 'crosshair';
  }
});
wrap.addEventListener('contextmenu', e => e.preventDefault());

// ── zoom ──────────────────────────────────────────────────────────────────────
wrap.addEventListener('wheel', e => {
  e.preventDefault();
  if (!state.img) return;
  const rect = canvas.getBoundingClientRect();
  const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
  const {x: ax, y: ay} = dispToImg(cx, cy);
  const f = e.deltaY < 0 ? 1.15 : 1/1.15;
  state.zoom = Math.max(1, Math.min(state.zoom * f, 20));
  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  state.panX = ax - (cx / canvas.width)  * vw;
  state.panY = ay - (cy / canvas.height) * vh;
  clampPan();
  document.getElementById('zoom-range').value = state.zoom;
  document.getElementById('zoom-lbl').textContent = state.zoom.toFixed(1) + 'x';
  draw();
}, {passive: false});

document.getElementById('zoom-range').addEventListener('input', function() {
  const cx = canvas.width/2, cy = canvas.height/2;
  const {x: ax, y: ay} = dispToImg(cx, cy);
  state.zoom = parseFloat(this.value);
  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  state.panX = ax - .5*vw; state.panY = ay - .5*vh;
  clampPan();
  document.getElementById('zoom-lbl').textContent = state.zoom.toFixed(1) + 'x';
  draw();
});

// ── keyboard ──────────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'z' || e.key === 'Z') {
    if (state.cornerA) { state.cornerA = null; state.boxPreview = null; setStatus('Cancelado.', 'info'); draw(); }
    else if (state.boxes.length) { state.boxes.pop(); updateBoxList(); draw(); setStatus('Última box eliminada.', 'info'); }
    return;
  }
  if (e.key === 'r' || e.key === 'R') { clearBoxes(); return; }
  if ((e.key === 's' || e.key === 'S') && !e.ctrlKey) { saveFrame(); return; }
  if (e.key === 'ArrowLeft')  { stepFrame(-1);  return; }
  if (e.key === 'ArrowRight') { stepFrame(1);   return; }
  if (e.key === 'ArrowUp')    { stepFrame(-30); return; }
  if (e.key === 'ArrowDown')  { stepFrame(30);  return; }
  if (e.key === 'a' || e.key === 'A') { stepFrame(-30); return; }
  if (e.key === 'd' || e.key === 'D') { stepFrame(30);  return; }
  if (e.key === '+' || e.key === '=') { state.zoom = Math.min(state.zoom*1.2,20); clampPan(); draw(); }
  if (e.key === '-' || e.key === '_') { state.zoom = Math.max(state.zoom/1.2,1);  clampPan(); draw(); }
});

window.addEventListener('resize', () => { fitCanvas(); draw(); });

// ── init ──────────────────────────────────────────────────────────────────────
fitCanvas();
loadFrame(300); // start at frame 300 (~10s)
</script>
</body>
</html>
"""

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
    return render_template_string(HTML)

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

    # Save YOLO label (class 0 = tubo)
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
