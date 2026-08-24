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
