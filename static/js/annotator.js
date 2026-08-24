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
