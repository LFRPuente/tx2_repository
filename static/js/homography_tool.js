const COLORS = ['#53b689', '#d6a34b', '#5b9bd5', '#d45b5b', '#a78bd6', '#70c7c2', '#e18f62'];
const meta = { fps: 30, totalFrames: 0, videos: [] };

const h = {
  canvas: document.getElementById('h-canvas'), wrap: document.getElementById('h-wrap'),
  points: [], img: null, imgW: 0, imgH: 0, zoom: 1, panX: 0, panY: 0,
  frameIdx: 0, timeSec: 0, source: null, timelineTimer: null, loadToken: 0,
  panning: false, panAnchor: null, panStart: null, didDrag: false, warpImg: null,
  expandedPoints: [], expandPct: 0, roiManual: false, draggingRoiSide: null,
  draggingPointIndex: null, dragPointOrigin: null, warpRequestToken: 0,
  roiMargins: {left: 0, right: 0, top: 0, bottom: 0}, baseMatrix: null, baseSize: null,
  lensCorrection: {enabled: false, model: 'opencv_radial_k1', k1: 0, center_x: 0.5, center_y: 0.5, focal_ratio: 0.5},
  metricScaleY: 1, lensTimer: null, lensFit: null
};
h.ctx = h.canvas.getContext('2d');
const w = { canvas: document.getElementById('w-canvas'), wrap: document.getElementById('w-wrap') };
w.ctx = w.canvas.getContext('2d');

const a = {
  canvas: document.getElementById('a-canvas'), wrap: document.getElementById('a-wrap'),
  img: null, imgW: 0, imgH: 0, zoom: 1, panX: 0, panY: 0,
  frameIdx: 0, timeSec: 0, source: null, boxes: [], cornerA: null, preview: null,
  panning: false, panAnchor: null, panStart: null, saved: 0, history: [], sobel: null, pieces: [],
  timelineTimer: null, modelBoxes: [], boxRules: null, selectedBoxIndex: -1,
  dragMode: null, dragStart: null, dragOrigin: null, dragHandle: null,
  dragAdopted: false, dragModelBox: null, interactionChanged: false, editVersion: 0, dirty: false,
  pendingSaves: 0, saveChain: Promise.resolve(), loadToken: 0
};
a.ctx = a.canvas.getContext('2d');

const m = {
  canvas: document.getElementById('m-canvas'), wrap: document.getElementById('m-wrap'),
  img: null, imgW: 0, imgH: 0, zoom: 1, panX: 0, panY: 0,
  frameIdx: 0, timeSec: 0, source: null, segments: [], pending: null, preview: null,
  referenceY: null, inchPerPx: null, exclusionZones: [], mode: 'segment', draggingReference: false,
  selectedSegment: null, draggingSegment: null,
  scaleMap: null, showScaleMap: true,
  panning: false, panAnchor: null, panStart: null, didDrag: false
};
m.ctx = m.canvas.getContext('2d');

const p = {
  canvas: document.getElementById('p-canvas'), wrap: document.getElementById('p-wrap'),
  img: null, imgW: 0, imgH: 0, zoom: 1, panX: 0, panY: 0,
  frameIdx: 0, timeSec: 0, source: null, boxes: [], pieces: [], measurementSummary: null,
  sobel: null, calibration: null, measurement: null, boxRules: null,
  captures: [], rulerActive: false, rulerMode: 'free', rulerStart: null, rulerEnd: null, rulerPreview: null,
  playing: false, speed: 1, playTask: null, panning: false, panAnchor: null, panStart: null,
  calibrationStale: false, timelineTimer: null
};
p.ctx = p.canvas.getContext('2d');

function status(msg, cls='') {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status ' + cls;
}

function sourceName(source) {
  return (source?.video_name || '-').replace(/^\d{8}_/, '').replace(/\.mkv$/i, '');
}

function showView(name) {
  stopPlayerPlayback();
  document.getElementById('homography-view').classList.toggle('active', name === 'homography');
  document.getElementById('measure-view').classList.toggle('active', name === 'measure');
  document.getElementById('annotate-view').classList.toggle('active', name === 'annotate');
  document.getElementById('player-view').classList.toggle('active', name === 'player');
  document.getElementById('homography-toolbar').style.display = name === 'homography' ? 'flex' : 'none';
  document.getElementById('measure-toolbar').style.display = name === 'measure' ? 'flex' : 'none';
  document.getElementById('annotate-toolbar').style.display = name === 'annotate' ? 'flex' : 'none';
  document.getElementById('player-toolbar').style.display = name === 'player' ? 'flex' : 'none';
  document.getElementById('tab-homography').classList.toggle('active', name === 'homography');
  document.getElementById('tab-measure').classList.toggle('active', name === 'measure');
  document.getElementById('tab-annotate').classList.toggle('active', name === 'annotate');
  document.getElementById('tab-player').classList.toggle('active', name === 'player');
  fitAll();
  if (name === 'measure' && !m.img) loadMeasureSecond();
  if (name === 'annotate' && !a.img) loadAnnotateSecond();
  if (name === 'player') {
    if (!p.img) {
      loadPlayerSecond();
    } else if (p.calibrationStale) {
      loadPlayerFrame(p.frameIdx, {resetView: false});
    }
  }
  drawAll();
}

function fitAll() {
  for (const s of [h, w, a, m, p]) {
    s.canvas.width = s.wrap.clientWidth;
    s.canvas.height = s.wrap.clientHeight;
  }
}

function clamp(state) {
  if (!state.imgW || !state.imgH) return;
  const vw = state.imgW / state.zoom;
  const vh = state.imgH / state.zoom;
  state.panX = Math.max(0, Math.min(state.panX, Math.max(0, state.imgW - vw)));
  state.panY = Math.max(0, Math.min(state.panY, Math.max(0, state.imgH - vh)));
}

function canvasImageRect(state) {
  const cw = state.canvas.width;
  const ch = state.canvas.height;
  if (state !== h || !state.imgW || !state.imgH) {
    return {x: 0, y: 0, width: cw, height: ch};
  }
  const vw = state.imgW / state.zoom;
  const vh = state.imgH / state.zoom;
  const scale = Math.min(cw / vw, ch / vh);
  const width = vw * scale;
  const height = vh * scale;
  return {
    x: (cw - width) / 2,
    y: (ch - height) / 2,
    width,
    height,
  };
}

function isInsideCanvasImage(state, cx, cy) {
  const rect = canvasImageRect(state);
  return cx >= rect.x && cx <= rect.x + rect.width && cy >= rect.y && cy <= rect.y + rect.height;
}

function displayToImage(state, cx, cy) {
  const vw = state.imgW / state.zoom;
  const vh = state.imgH / state.zoom;
  const rect = canvasImageRect(state);
  return {
    x: Math.max(0, Math.min(state.panX + ((cx - rect.x) / rect.width) * vw, state.imgW - 1)),
    y: Math.max(0, Math.min(state.panY + ((cy - rect.y) / rect.height) * vh, state.imgH - 1)),
  };
}

function imageToDisplay(state, ix, iy) {
  const vw = state.imgW / state.zoom;
  const vh = state.imgH / state.zoom;
  const rect = canvasImageRect(state);
  return {
    x: rect.x + ((ix - state.panX) / vw) * rect.width,
    y: rect.y + ((iy - state.panY) / vh) * rect.height,
  };
}

function resetView(state) {
  state.zoom = 1; state.panX = 0; state.panY = 0;
}

function clearWarpDependentViews() {
  a.img = null; a.imgW = 0; a.imgH = 0; a.boxes = []; a.pieces = [];
  a.cornerA = null; a.preview = null; a.sobel = null; a.selectedBoxIndex = -1; a.dirty = false;
  resetAnnotationInteraction();
  m.img = null; m.imgW = 0; m.imgH = 0; m.pending = null; m.preview = null;
  p.img = null; p.imgW = 0; p.imgH = 0; p.boxes = []; p.pieces = [];
  p.measurementSummary = null; p.sobel = null; p.measurement = null; p.boxRules = null;
  updateBoxes();
  updateMeasureInfo();
  updatePlayerInfo();
}

async function loadMeta() {
  const r = await fetch('/api/meta');
  const d = await r.json();
  if (!r.ok) throw new Error(d.error);
  meta.fps = d.fps;
  meta.totalFrames = d.total_frames;
  meta.videos = d.videos || [];
  const homographyTimeline = document.getElementById('h-timeline');
  homographyTimeline.max = Math.max(0, meta.totalFrames - 1);
  document.getElementById('h-second').max = Math.max(0, (meta.totalFrames - 1) / meta.fps).toFixed(3);
  const homographySecond = Math.max(
    0,
    Math.min((meta.totalFrames - 1) / meta.fps, parseFloat(document.getElementById('h-second').value) || 0)
  );
  const homographyFrame = Math.round(homographySecond * meta.fps);
  homographyTimeline.value = homographyFrame;
  document.getElementById('h-timeline-label').textContent = homographyTimelineText(homographyFrame);
  const annotateTimeline = document.getElementById('a-timeline');
  annotateTimeline.max = Math.max(0, meta.totalFrames - 1);
  document.getElementById('a-second').max = Math.max(0, (meta.totalFrames - 1) / meta.fps).toFixed(3);
  document.getElementById('a-timeline-label').textContent = formatClock(0);
  const playerTimeline = document.getElementById('p-timeline');
  playerTimeline.max = Math.max(0, meta.totalFrames - 1);
  document.getElementById('p-second').max = Math.max(0, (meta.totalFrames - 1) / meta.fps).toFixed(3);
  document.getElementById('p-timeline-label').textContent = formatClock(0);
  document.getElementById('info-dataset').textContent = d.dataset_dir;
  await refreshHistory();
  await refreshPlayerCaptures();
  return d;
}

function normalizedLensCorrection(value) {
  const source = value || {};
  const k1 = Math.max(-0.35, Math.min(0.35, Number(source.k1) || 0));
  return {
    enabled: Boolean(source.enabled) && Math.abs(k1) > 1e-7,
    model: 'opencv_radial_k1',
    k1,
    center_x: Math.max(0.25, Math.min(0.75, Number(source.center_x) || 0.5)),
    center_y: Math.max(0.25, Math.min(0.75, Number(source.center_y) || 0.5)),
    focal_ratio: Math.max(0.25, Math.min(1.5, Number(source.focal_ratio) || 0.5)),
  };
}

function correctedPointToRaw(point, correction) {
  const config = normalizedLensCorrection(correction);
  if (!config.enabled || !h.imgW || !h.imgH) return {...point};
  const focal = Math.max(h.imgW, h.imgH) * config.focal_ratio;
  const cx = (h.imgW - 1) * config.center_x;
  const cy = (h.imgH - 1) * config.center_y;
  const nx = (point.x - cx) / focal;
  const ny = (point.y - cy) / focal;
  const factor = 1 + config.k1 * (nx * nx + ny * ny);
  return {x: cx + nx * factor * focal, y: cy + ny * factor * focal};
}

function rawPointToCorrected(point, correction) {
  const config = normalizedLensCorrection(correction);
  if (!config.enabled || !h.imgW || !h.imgH) return {...point};
  const focal = Math.max(h.imgW, h.imgH) * config.focal_ratio;
  const cx = (h.imgW - 1) * config.center_x;
  const cy = (h.imgH - 1) * config.center_y;
  const dx = (point.x - cx) / focal;
  const dy = (point.y - cy) / focal;
  const distortedRadius = Math.hypot(dx, dy);
  if (distortedRadius < 1e-12) return {x: cx, y: cy};
  let radius = distortedRadius;
  for (let i = 0; i < 16; i += 1) {
    const radiusSq = radius * radius;
    const residual = radius * (1 + config.k1 * radiusSq) - distortedRadius;
    const derivative = 1 + 3 * config.k1 * radiusSq;
    if (Math.abs(derivative) < 1e-8) break;
    radius -= residual / derivative;
  }
  const scale = radius / distortedRadius;
  return {x: cx + dx * scale * focal, y: cy + dy * scale * focal};
}

function remapHomographyPoints(oldCorrection, newCorrection) {
  if (!h.imgW || !h.imgH || !h.points.length) return;
  h.points = h.points.map(point => {
    const raw = correctedPointToRaw(point, oldCorrection);
    return rawPointToCorrected(raw, newCorrection);
  });
}

function updateLensControls() {
  const enabledControl = document.getElementById('h-lens-enabled');
  if (!enabledControl) return;
  const config = normalizedLensCorrection(h.lensCorrection);
  enabledControl.checked = config.enabled;
  document.getElementById('h-lens-k1').value = config.k1;
  document.getElementById('h-lens-k1').disabled = !config.enabled;
  const label = config.enabled ? `k1 ${config.k1.toFixed(3)} | Y ${h.metricScaleY.toFixed(3)}x` : `off | Y ${h.metricScaleY.toFixed(3)}x`;
  document.getElementById('h-lens-label').textContent = label;
  document.getElementById('h-lens-label').className = 'pill' + (config.enabled || Math.abs(h.metricScaleY - 1) > 0.001 ? ' ok' : '');
}

function applyLensCorrectionDraft(nextCorrection, options = {}) {
  const oldCorrection = normalizedLensCorrection(h.lensCorrection);
  const next = normalizedLensCorrection(nextCorrection);
  if (options.remapPoints !== false) remapHomographyPoints(oldCorrection, next);
  h.lensCorrection = next;
  h.expandedPoints = [];
  h.warpImg = null;
  h.lensFit = options.fit || null;
  if (Number.isFinite(Number(options.metricScaleY))) {
    h.metricScaleY = Math.max(0.75, Math.min(1.25, Number(options.metricScaleY)));
  }
  updateLensControls();
  updatePoints();
  loadHomographyFrame({frameIdx: h.frameIdx, resetView: false});
}

function toggleLensCorrection() {
  const enabled = document.getElementById('h-lens-enabled').checked;
  const slider = document.getElementById('h-lens-k1');
  let k1 = Number(slider.value) || 0;
  if (enabled && Math.abs(k1) < 1e-7) {
    k1 = 0.05;
    slider.value = k1;
  }
  applyLensCorrectionDraft({...h.lensCorrection, enabled, k1});
}

function previewLensCorrection(value) {
  clearTimeout(h.lensTimer);
  const k1 = Number(value) || 0;
  document.getElementById('h-lens-enabled').checked = Math.abs(k1) > 1e-7;
  document.getElementById('h-lens-label').textContent = `k1 ${k1.toFixed(3)} | Y ${h.metricScaleY.toFixed(3)}x`;
  h.lensTimer = setTimeout(() => {
    applyLensCorrectionDraft({...h.lensCorrection, enabled: Math.abs(k1) > 1e-7, k1});
  }, 160);
}

async function autoFitLensCorrection() {
  status('Ajustando distorsion radial con las medidas conocidas...', '');
  const payload = m.segments.length >= 3 ? {segments: m.segments} : {};
  const r = await fetch('/api/lens/auto_fit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  const before = Number(d.before?.spread_pct || 0);
  const after = Number(d.after?.spread_pct || 0);
  document.getElementById('h-lens-fit').textContent = `${before.toFixed(2)}% -> ${after.toFixed(2)}%`;
  document.getElementById('h-lens-fit').className = 'pill' + (d.recommended ? ' ok' : '');
  if (!d.recommended) {
    status(d.warning || 'No hay evidencia suficiente para cambiar la correccion.', 'err');
    return;
  }
  h.points = (d.selected_source_points || []).map(point => ({
    x: Number(point[0] ?? point.x),
    y: Number(point[1] ?? point.y),
  }));
  applyLensCorrectionDraft(d.lens_correction, {
    remapPoints: false,
    metricScaleY: d.metric_scale_y,
    fit: d,
  });
  showView('homography');
  status(
    `Ajuste listo (${d.confidence}): dispersion ${before.toFixed(2)}% -> ${after.toFixed(2)}%. Revisa y guarda la homografia.`,
    'ok'
  );
}

async function loadHomographyFrame(options = {}) {
  const requestedFrame = Number.isFinite(Number(options.frameIdx))
    ? Math.max(0, Math.min(meta.totalFrames - 1, Math.round(Number(options.frameIdx))))
    : null;
  const second = requestedFrame === null
    ? (parseFloat(document.getElementById('h-second').value) || 0)
    : requestedFrame / meta.fps;
  const loadToken = ++h.loadToken;
  status('Cargando frame...', '');
  const r = await fetch('/api/frame', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      ...(requestedFrame === null ? {second} : {frame_idx: requestedFrame}),
      lens_correction: h.lensCorrection,
    })
  });
  const d = await r.json();
  if (loadToken !== h.loadToken) return null;
  if (!r.ok) { status(d.error, 'err'); return; }
  const img = new Image();
  img.onload = () => {
    if (loadToken !== h.loadToken) return;
    h.img = img; h.imgW = d.width; h.imgH = d.height;
    h.frameIdx = d.frame_idx; h.timeSec = d.time_sec; h.source = d.source || null;
    h.draggingPointIndex = null; h.dragPointOrigin = null;
    h.draggingRoiSide = null; h.warpImg = null;
    if (options.resetView !== false) resetView(h);
    document.getElementById('source-label').textContent = d.label;
    document.getElementById('h-second').value = d.time_sec.toFixed(2);
    document.getElementById('h-timeline').value = d.frame_idx;
    document.getElementById('h-timeline-label').textContent = homographyTimelineText(d.frame_idx);
    document.getElementById('h-zoom').value = h.zoom;
    document.getElementById('h-zoom-label').textContent = h.zoom.toFixed(1) + 'x';
    updateLensControls();
    updatePoints();
    fitAll(); drawAll();
    const video = sourceName(d.source);
    const sourceFrame = d.source?.source_frame_idx ?? d.frame_idx;
    status(`Frame cargado: ${video} | frame ${sourceFrame} | ${d.width}x${d.height}`, 'ok');
  };
  img.src = 'data:image/jpeg;base64,' + d.image;
  return d;
}

function playlistVideoForFrame(frameValue) {
  const frameIdx = Math.max(0, Math.min(meta.totalFrames - 1, Number(frameValue) || 0));
  return meta.videos.find(video => frameIdx >= video.start_frame && frameIdx < video.end_frame) || null;
}

function homographyTimelineText(frameValue) {
  const frameIdx = Math.max(0, Math.min(meta.totalFrames - 1, Number(frameValue) || 0));
  const video = playlistVideoForFrame(frameIdx);
  const videoName = video ? sourceName({video_name: video.name}) : '-';
  return `${formatClock(frameIdx / meta.fps)} | ${videoName}`;
}

function previewHomographyTimeline(frameValue) {
  const frameIdx = Math.max(0, Math.min(meta.totalFrames - 1, Number(frameValue) || 0));
  document.getElementById('h-second').value = (frameIdx / meta.fps).toFixed(2);
  document.getElementById('h-timeline-label').textContent = homographyTimelineText(frameIdx);
}

function loadHomographyTimeline(frameValue) {
  clearTimeout(h.timelineTimer);
  h.timelineTimer = null;
  return loadHomographyFrame({frameIdx: Number(frameValue) || 0, resetView: false});
}

function scrollHomographyTimeline(event) {
  event.preventDefault();
  const timeline = event.currentTarget;
  const direction = (event.deltaY || event.deltaX) > 0 ? 1 : -1;
  const jumpSeconds = event.shiftKey ? 5 : 30;
  const nextFrame = Math.max(
    0,
    Math.min(meta.totalFrames - 1, Number(timeline.value) + direction * Math.round(meta.fps * jumpSeconds))
  );
  timeline.value = nextFrame;
  previewHomographyTimeline(nextFrame);
  clearTimeout(h.timelineTimer);
  h.timelineTimer = setTimeout(() => loadHomographyTimeline(nextFrame), 180);
}

function stepHomography(delta) {
  const currentFrame = h.img ? h.frameIdx : Number(document.getElementById('h-timeline').value);
  return loadHomographyFrame({frameIdx: currentFrame + delta, resetView: false});
}

function drawHomography() {
  const ctx = h.ctx, cw = h.canvas.width, ch = h.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!h.img) return;
  const vw = h.imgW / h.zoom, vh = h.imgH / h.zoom;
  const rect = canvasImageRect(h);
  ctx.drawImage(h.img, h.panX, h.panY, vw, vh, rect.x, rect.y, rect.width, rect.height);
  drawGrid(h, 200);
  if (h.expandedPoints.length === 4) {
    ctx.save();
    ctx.beginPath();
    h.expandedPoints.forEach((p, i) => {
      const q = imageToDisplay(h, p.x, p.y);
      if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
    });
    ctx.closePath();
    ctx.strokeStyle = '#5b9bd5';
    ctx.lineWidth = 2;
    ctx.setLineDash([9, 6]);
    ctx.stroke();
    ctx.restore();
  }
  if (h.points.length >= 2) {
    ctx.beginPath();
    h.points.forEach((p, i) => {
      const q = imageToDisplay(h, p.x, p.y);
      if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
    });
    if (h.points.length === 4) ctx.closePath();
    ctx.strokeStyle = '#53b689';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  h.points.forEach((p, i) => {
    const q = imageToDisplay(h, p.x, p.y);
    ctx.fillStyle = COLORS[i % COLORS.length];
    const active = h.draggingPointIndex === i;
    ctx.beginPath(); ctx.arc(q.x, q.y, active ? 11 : 8, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    if (active) {
      ctx.beginPath(); ctx.arc(q.x, q.y, 15, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = '#fff'; ctx.font = '700 12px Arial';
    ctx.fillText(String(i + 1), q.x + 11, q.y - 8);
  });
  if (h.expandedPoints.length === 4) {
    const handles = roiSideMidpoints();
    Object.entries(handles).forEach(([side, q]) => {
      const label = {top: 'T', right: 'R', bottom: 'B', left: 'L'}[side];
      ctx.save();
      ctx.fillStyle = '#5b9bd5';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.fillRect(q.x - 7, q.y - 7, 14, 14);
      ctx.strokeRect(q.x - 7, q.y - 7, 14, 14);
      ctx.fillStyle = '#ffffff';
      ctx.font = '700 11px Arial';
      ctx.fillText(label, q.x + 10, q.y + 4);
      ctx.restore();
    });
  }
}

function drawWarp() {
  const ctx = w.ctx, cw = w.canvas.width, ch = w.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#050607'; ctx.fillRect(0, 0, cw, ch);
  if (!h.warpImg) {
    ctx.fillStyle = '#7d8580'; ctx.font = '14px Arial'; ctx.textAlign = 'center';
    ctx.fillText('Marca 4 puntos para previsualizar', cw / 2, ch / 2);
    ctx.textAlign = 'left';
    return;
  }
  const scale = Math.min(cw / h.warpImg.naturalWidth, ch / h.warpImg.naturalHeight);
  const dw = h.warpImg.naturalWidth * scale, dh = h.warpImg.naturalHeight * scale;
  ctx.drawImage(h.warpImg, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
}

function drawGrid(state, step) {
  const ctx = state.ctx;
  const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
  const rect = canvasImageRect(state);
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.width, rect.height);
  ctx.clip();
  ctx.lineWidth = 1; ctx.font = '10px Consolas';
  for (let gx = Math.ceil(state.panX / step) * step; gx < state.panX + vw; gx += step) {
    const px = imageToDisplay(state, gx, state.panY).x;
    ctx.strokeStyle = 'rgba(255,255,255,.12)';
    ctx.beginPath(); ctx.moveTo(px, rect.y); ctx.lineTo(px, rect.y + rect.height); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,.45)'; ctx.fillText(String(gx), px + 3, rect.y + 13);
  }
  for (let gy = Math.ceil(state.panY / step) * step; gy < state.panY + vh; gy += step) {
    const py = imageToDisplay(state, state.panX, gy).y;
    ctx.strokeStyle = 'rgba(255,255,255,.12)';
    ctx.beginPath(); ctx.moveTo(rect.x, py); ctx.lineTo(rect.x + rect.width, py); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,.45)'; ctx.fillText(String(gy), rect.x + 3, py + 12);
  }
  ctx.restore();
}

function renderHomographyPointInfo() {
  document.getElementById('points-badge').textContent = `${h.points.length} / 4 puntos`;
  document.getElementById('points-badge').className = 'pill' + (h.points.length === 4 ? ' ok' : '');
  document.getElementById('save-homography').disabled = h.points.length !== 4;
  h.expandPct = parseFloat(document.getElementById('h-expand').value) || 0;
  document.getElementById('h-expand-label').textContent = Math.round(h.expandPct) + '%';
  const baseText = h.points.length
    ? h.points.map((p, i) => `<span style="color:${COLORS[i]}">P${i+1}</span> (${Math.round(p.x)}, ${Math.round(p.y)})`).join(' &nbsp; ')
    : 'Sin puntos.';
  const roiText = h.expandedPoints.length === 4
    ? ` &nbsp; <span style="color:#5b9bd5">ROI ${h.roiManual ? 'lados' : 'auto'}</span> L:${Math.round(h.roiMargins.left)} R:${Math.round(h.roiMargins.right)} T:${Math.round(h.roiMargins.top)} B:${Math.round(h.roiMargins.bottom)} px`
    : '';
  const lens = normalizedLensCorrection(h.lensCorrection);
  const lensText = lens.enabled || Math.abs(h.metricScaleY - 1) > 0.001
    ? ` &nbsp; <span style="color:#d6a34b">Lente</span> k1:${lens.k1.toFixed(3)} Y:${h.metricScaleY.toFixed(3)}x`
    : '';
  document.getElementById('points-list').innerHTML = baseText + roiText + lensText;
}

function updatePoints() {
  renderHomographyPointInfo();
  requestWarp();
}

async function requestWarp() {
  const requestToken = ++h.warpRequestToken;
  if (h.points.length !== 4) {
    h.warpImg = null;
    h.expandedPoints = [];
    document.getElementById('warp-size').textContent = 'sin homografia';
    drawWarp();
    return;
  }
  h.expandPct = parseFloat(document.getElementById('h-expand').value) || 0;
  const payload = {
    points: h.points,
    expand_pct: h.expandPct,
    lens_correction: h.lensCorrection,
    metric_scale_y: h.metricScaleY,
  };
  if (h.roiManual) payload.roi_margins = h.roiMargins;
  const r = await fetch('/api/warp', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (requestToken !== h.warpRequestToken) return;
  if (!r.ok) { status(d.error, 'err'); return; }
  const img = new Image();
  img.onload = () => {
    if (requestToken !== h.warpRequestToken) return;
    h.warpImg = img;
    h.expandedPoints = d.warp_points || [];
    h.roiMargins = d.roi_margins || h.roiMargins;
    h.baseMatrix = d.base_matrix || h.baseMatrix;
    h.baseSize = d.base_size || h.baseSize;
    h.roiManual = d.roi_mode === 'side_margins';
    const mode = h.roiManual ? 'lados manuales' : `expansion ${Math.round(h.expandPct)}%`;
    document.getElementById('warp-size').textContent = `${d.width} x ${d.height} | ${mode}`;
    renderHomographyPointInfo();
    drawAll();
  };
  img.src = 'data:image/jpeg;base64,' + d.image;
}

async function saveHomography() {
  if (h.points.length !== 4) return;
  h.expandPct = parseFloat(document.getElementById('h-expand').value) || 0;
  const payload = {
    points: h.points,
    expand_pct: h.expandPct,
    lens_correction: h.lensCorrection,
    metric_scale_y: h.metricScaleY,
    lens_auto_fit: h.lensFit,
    migrate_measurements: true,
  };
  if (h.roiManual) payload.roi_margins = h.roiMargins;
  const r = await fetch('/api/save_homography', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  const mode = h.roiManual ? 'lados manuales' : `expansion ${Math.round(h.expandPct)}%`;
  clearWarpDependentViews();
  const migration = d.measurement_migrated ? ' | mediciones remapeadas' : '';
  status(`Homografia guardada con ${mode}${migration}: ${d.path}${d.backup ? ' | backup: ' + d.backup : ''}`, 'ok');
}

async function loadSavedHomography() {
  const r = await fetch('/api/homography/current');
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  h.draggingPointIndex = null;
  h.dragPointOrigin = null;
  h.lensCorrection = normalizedLensCorrection(d.lens_correction);
  h.metricScaleY = Math.max(0.75, Math.min(1.25, Number(d.metric_scale_y) || 1));
  h.lensFit = d.lens_auto_fit || null;
  h.points = (d.selected_source_points || d.ordered_source_points || []).map(p => ({x: Number(p[0] ?? p.x), y: Number(p[1] ?? p.y)}));
  h.expandedPoints = (d.work_roi_points || d.ordered_source_points || []).map(p => ({x: Number(p[0] ?? p.x), y: Number(p[1] ?? p.y)}));
  h.roiMargins = d.roi_margins || {left: 0, right: 0, top: 0, bottom: 0};
  h.baseMatrix = d.base_homography_matrix || null;
  h.baseSize = Array.isArray(d.base_output_size) && d.base_output_size.length === 2
    ? {width: Number(d.base_output_size[0]), height: Number(d.base_output_size[1])}
    : null;
  h.roiManual = d.roi_mode === 'side_margins';
  h.expandPct = Number(d.expand_pct || 0);
  document.getElementById('h-expand').value = h.expandPct;
  document.getElementById('h-expand-label').textContent = Math.round(h.expandPct) + '%';
  updateLensControls();
  const lensFitBadge = document.getElementById('h-lens-fit');
  if (lensFitBadge && h.lensFit?.before && h.lensFit?.after) {
    lensFitBadge.textContent =
      `${Number(h.lensFit.before.spread_pct).toFixed(2)}% -> ${Number(h.lensFit.after.spread_pct).toFixed(2)}%`;
    lensFitBadge.className = 'pill ok';
  }
  await loadHomographyFrame({frameIdx: h.frameIdx, resetView: false});
  updatePoints();
  drawAll();
  status(`Homografia cargada: ${h.points.length} puntos | ${h.roiManual ? 'lados manuales' : 'expansion ' + Math.round(h.expandPct) + '%'}`, 'ok');
}

function undoPoint() { h.points.pop(); updatePoints(); drawAll(); }
function resetPoints() {
  h.points = []; h.expandedPoints = []; h.roiManual = false;
  h.draggingPointIndex = null; h.dragPointOrigin = null;
  h.roiMargins = {left: 0, right: 0, top: 0, bottom: 0};
  h.baseMatrix = null; h.baseSize = null; h.warpImg = null;
  updatePoints(); drawAll();
}
function resetWorkRoi() {
  h.roiManual = false;
  h.expandedPoints = [];
  h.roiMargins = {left: 0, right: 0, top: 0, bottom: 0};
  if (h.points.length === 4) requestWarp();
  drawAll();
}

async function loadAnnotateFrame(frameIdx, options = {}) {
  if (!options.skipAutosave) {
    const ready = await flushAnnotationAutosave();
    if (!ready) return null;
  }
  const loadToken = ++a.loadToken;
  if (!options.quiet) status('Cargando frame rectificado...', '');
  const r = await fetch('/api/annotate/frame', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_idx: frameIdx})
  });
  const d = await r.json();
  if (loadToken !== a.loadToken) return null;
  if (!r.ok) { status(d.error, 'err'); return; }
  const img = new Image();
  return await new Promise((resolve, reject) => {
    img.onload = async () => {
      if (loadToken !== a.loadToken) {
        resolve(null);
        return;
      }
      a.img = img; a.imgW = d.width; a.imgH = d.height;
      a.frameIdx = d.frame_idx; a.timeSec = d.time_sec; a.source = d.source || null;
      a.boxes = (d.boxes || []).map(boxToCorners); a.modelBoxes = []; a.boxRules = null; a.pieces = [];
      resetAnnotationInteraction();
      a.selectedBoxIndex = -1;
      a.dirty = false;
      a.editVersion += 1;
      a.cornerA = null; a.preview = null; a.sobel = null;
      if (options.resetView !== false) resetView(a);
      document.getElementById('a-second').value = d.time_sec.toFixed(2);
      document.getElementById('a-timeline').value = d.frame_idx;
      document.getElementById('a-timeline-label').textContent = formatClock(d.time_sec);
      document.getElementById('a-zoom').value = a.zoom;
      document.getElementById('a-zoom-label').textContent = a.zoom.toFixed(1) + 'x';
      document.getElementById('info-frame').textContent = d.frame_idx;
      document.getElementById('info-time').textContent = d.time_sec.toFixed(3) + 's';
      document.getElementById('info-size').textContent = `${d.width}x${d.height}`;
      document.getElementById('info-video').textContent = sourceName(d.source);
      document.getElementById('info-source-frame').textContent = d.source?.source_frame_idx ?? '-';
      updateBoxes();
      updateModelBoxes();
      updateSobelInfo();
      fitAll(); drawAll();
      updateHistoryUI();
      if (options.autoAnalyze) {
        await predictYoloBoxes({quiet: true, autoSobel: true});
      } else if (!options.quiet) {
        status(d.is_saved ? `Frame ${d.frame_idx} cargado desde historial` : `Frame ${d.frame_idx} listo para anotar`, 'ok');
      }
      resolve(d);
    };
    img.onerror = () => {
      const err = new Error('No se pudo cargar la imagen del frame');
      status(err.message, 'err');
      reject(err);
    };
    img.src = 'data:image/jpeg;base64,' + d.image;
  });
}

function loadAnnotateSecond() {
  const second = parseFloat(document.getElementById('a-second').value) || 0;
  loadAnnotateFrame(Math.round(second * meta.fps));
}
function stepAnnotate(delta) { loadAnnotateFrame(a.frameIdx + delta); }
function stepAnnotateSeconds(seconds) {
  loadAnnotateFrame(a.frameIdx + Math.round(seconds * meta.fps), {resetView: false});
}

function formatClock(totalSeconds) {
  const safeSeconds = Math.max(0, Number(totalSeconds) || 0);
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const seconds = Math.floor(safeSeconds % 60);
  return [hours, minutes, seconds].map(value => String(value).padStart(2, '0')).join(':');
}

function formatFeetInches(totalInches) {
  const value = Number(totalInches);
  if (!Number.isFinite(value)) return '-';
  const sign = value < 0 ? '-' : '';
  const totalSixteenths = Math.round(Math.abs(value) * 16);
  const feet = Math.floor(totalSixteenths / 192);
  const remainingSixteenths = totalSixteenths - feet * 192;
  const wholeInches = Math.floor(remainingSixteenths / 16);
  let numerator = remainingSixteenths % 16;
  let denominator = 16;
  while (numerator > 0 && numerator % 2 === 0 && denominator % 2 === 0) {
    numerator /= 2;
    denominator /= 2;
  }
  const fraction = numerator ? ` ${numerator}/${denominator}` : '';
  return `${sign}${feet}' ${wholeInches}${fraction}"`;
}

function previewAnnotateTimeline(frameValue) {
  const frameIdx = Math.max(0, Math.min(meta.totalFrames - 1, Number(frameValue) || 0));
  const second = frameIdx / meta.fps;
  document.getElementById('a-second').value = second.toFixed(2);
  document.getElementById('a-timeline-label').textContent = formatClock(second);
}

function loadAnnotateTimeline(frameValue) {
  clearTimeout(a.timelineTimer);
  a.timelineTimer = null;
  return loadAnnotateFrame(Number(frameValue) || 0, {resetView: false});
}

function scrollAnnotateTimeline(event) {
  event.preventDefault();
  const timeline = event.currentTarget;
  const direction = (event.deltaY || event.deltaX) > 0 ? 1 : -1;
  const jumpSeconds = event.shiftKey ? 5 : 30;
  const nextFrame = Math.max(
    0,
    Math.min(meta.totalFrames - 1, Number(timeline.value) + direction * Math.round(meta.fps * jumpSeconds))
  );
  timeline.value = nextFrame;
  previewAnnotateTimeline(nextFrame);
  clearTimeout(a.timelineTimer);
  a.timelineTimer = setTimeout(() => loadAnnotateTimeline(nextFrame), 180);
}

function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

async function loadMeasureFrame(frameIdx, options = {}) {
  if (!options.quiet) status('Cargando frame para mediciones...', '');
  const r = await fetch('/api/measure/frame', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_idx: frameIdx})
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return null; }
  const img = new Image();
  return await new Promise((resolve, reject) => {
    img.onload = () => {
      m.img = img; m.imgW = d.width; m.imgH = d.height;
      m.frameIdx = d.frame_idx; m.timeSec = d.time_sec; m.source = d.source || null;
      applyMeasureCalibration(d.calibration || {});
      if (options.resetView !== false) resetView(m);
      document.getElementById('m-second').value = d.time_sec.toFixed(2);
      document.getElementById('m-zoom').value = m.zoom;
      document.getElementById('m-zoom-label').textContent = m.zoom.toFixed(1) + 'x';
      updateMeasureInfo();
      fitAll(); drawAll();
      if (!options.quiet) status(`Frame ${d.frame_idx} listo para mediciones`, 'ok');
      resolve(d);
    };
    img.onerror = () => {
      const err = new Error('No se pudo cargar el frame de mediciones');
      status(err.message, 'err');
      reject(err);
    };
    img.src = 'data:image/jpeg;base64,' + d.image;
  });
}

function loadMeasureSecond() {
  const second = parseFloat(document.getElementById('m-second').value) || 0;
  return loadMeasureFrame(Math.round(second * meta.fps));
}

function reloadMeasureRoi() {
  const frameIdx = m.img ? m.frameIdx : Math.round((parseFloat(document.getElementById('m-second').value) || 0) * meta.fps);
  m.img = null; m.imgW = 0; m.imgH = 0; m.pending = null; m.preview = null;
  updateMeasureInfo(); drawAll();
  return loadMeasureFrame(frameIdx, {resetView: true});
}

function stepMeasure(delta) {
  return loadMeasureFrame(m.frameIdx + delta, {resetView: false});
}

function scaleSegmentSample(segment, sourceIndex) {
  const dx = Number(segment.x2) - Number(segment.x1);
  const dy = Number(segment.y2) - Number(segment.y1);
  const px = Math.hypot(dx, dy);
  const inches = Number(segment.inches);
  if (!(px > 0) || !(inches > 0)) return null;
  const alignmentX = Math.abs(dx) / px;
  const alignmentY = Math.abs(dy) / px;
  let axis = null;
  let alignment = 0;
  if (alignmentX >= 0.85) { axis = 'x'; alignment = alignmentX; }
  else if (alignmentY >= 0.85) { axis = 'y'; alignment = alignmentY; }
  if (!axis) return {axis: null};
  return {
    axis,
    source_index: sourceIndex,
    x: (Number(segment.x1) + Number(segment.x2)) / 2,
    y: (Number(segment.y1) + Number(segment.y2)) / 2,
    inch_per_px: inches / px,
    px_per_in: px / inches,
    alignment,
  };
}

function scaleMapKnots(samples, coordinate) {
  const ordered = [...samples].sort((left, right) => left[coordinate] - right[coordinate]);
  const groups = [];
  ordered.forEach(sample => {
    const group = groups[groups.length - 1];
    if (!group || Math.abs(sample[coordinate] - group[group.length - 1][coordinate]) > 1) {
      groups.push([sample]);
    } else {
      group.push(sample);
    }
  });
  return groups.map(group => ({
    coordinate: group.reduce((sum, sample) => sum + sample[coordinate], 0) / group.length,
    inch_per_px: group.reduce((sum, sample) => sum + sample.inch_per_px, 0) / group.length,
    sample_count: group.length,
  }));
}

function scaleMapConvexHull(samples) {
  const points = samples
    .map(sample => ({x: Number(sample.x), y: Number(sample.y)}))
    .sort((left, right) => left.x - right.x || left.y - right.y);
  if (points.length <= 2) return points;
  const cross = (origin, a, b) =>
    (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
  const lower = [];
  points.forEach(point => {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
    lower.push(point);
  });
  const upper = [];
  [...points].reverse().forEach(point => {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
    upper.push(point);
  });
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function pointInsideScaleHull(x, y, hull) {
  if (!hull || hull.length < 3) return false;
  let inside = false;
  for (let index = 0, previous = hull.length - 1; index < hull.length; previous = index++) {
    const currentPoint = hull[index], previousPoint = hull[previous];
    const intersects = ((currentPoint.y > y) !== (previousPoint.y > y)) &&
      (x < (previousPoint.x - currentPoint.x) * (y - currentPoint.y) /
        Math.max(1e-12, previousPoint.y - currentPoint.y) + currentPoint.x);
    if (intersects) inside = !inside;
  }
  return inside;
}

function buildClientScaleAxis(axis, samples, width, height) {
  if (!samples.length) {
    return {axis, mode: 'global', samples: [], knots: [], hull: [], sample_count: 0, coverage: null};
  }
  const xs = samples.map(sample => sample.x);
  const ys = samples.map(sample => sample.y);
  const values = samples.map(sample => sample.inch_per_px);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xSpanRatio = (xMax - xMin) / Math.max(1, width);
  const ySpanRatio = (yMax - yMin) / Math.max(1, height);
  const use2d = samples.length >= 4 && xSpanRatio >= 0.15 && ySpanRatio >= 0.15;
  const coordinate = use2d ? null : (xSpanRatio >= ySpanRatio ? 'x' : 'y');
  return {
    axis,
    mode: use2d ? 'idw_2d' : `linear_${coordinate}`,
    coordinate,
    samples,
    knots: use2d ? [] : scaleMapKnots(samples, coordinate),
    hull: use2d ? scaleMapConvexHull(samples) : [],
    sample_count: samples.length,
    confidence: samples.length >= 8 ? 'high' : (samples.length >= 4 ? 'medium' : 'low'),
    coverage: {x_min: xMin, x_max: xMax, y_min: yMin, y_max: yMax, x_span_ratio: xSpanRatio, y_span_ratio: ySpanRatio},
    minimum_inch_per_px: Math.min(...values),
    maximum_inch_per_px: Math.max(...values),
    mean_inch_per_px: values.reduce((sum, value) => sum + value, 0) / values.length,
  };
}

function buildClientScaleMap(segments, width, height) {
  const axes = {x: [], y: []};
  let ignored = 0;
  (segments || []).forEach((segment, index) => {
    const sample = scaleSegmentSample(segment, index);
    if (!sample || !sample.axis) {
      if (sample) ignored += 1;
      return;
    }
    axes[sample.axis].push(sample);
  });
  return {
    version: 1,
    method: 'axis_local_interpolation',
    image_width: width,
    image_height: height,
    idw_power: 2,
    axes: {
      x: buildClientScaleAxis('x', axes.x, width, height),
      y: buildClientScaleAxis('y', axes.y, width, height),
    },
    ignored_diagonal_segments: ignored,
  };
}

function spatialScaleAtClient(calibration, x, y, axis = 'y') {
  const fallback = Number(calibration?.inch_per_px);
  const fallbackValue = Number.isFinite(fallback) && fallback > 0 ? fallback : null;
  const scaleMap = calibration?.scale_map;
  const axisData = scaleMap?.axes?.[axis];
  const samples = axisData?.samples || [];
  if (!samples.length) {
    return {inchPerPx: fallbackValue, source: fallbackValue ? 'global' : 'missing', extrapolated: false, sampleCount: 0};
  }
  const mode = String(axisData.mode || 'global');
  if (mode.startsWith('linear_')) {
    const coordinateName = axisData.coordinate || mode.replace('linear_', '');
    const coordinate = coordinateName === 'x' ? x : y;
    const knots = axisData.knots || scaleMapKnots(samples, coordinateName);
    if (knots.length === 1) {
      return {inchPerPx: Number(knots[0].inch_per_px), source: mode, extrapolated: true, sampleCount: samples.length};
    }
    let value = Number(knots[0].inch_per_px);
    let extrapolated = coordinate < knots[0].coordinate || coordinate > knots[knots.length - 1].coordinate;
    if (coordinate >= knots[knots.length - 1].coordinate) {
      value = Number(knots[knots.length - 1].inch_per_px);
    } else if (coordinate > knots[0].coordinate) {
      for (let index = 0; index < knots.length - 1; index += 1) {
        const left = knots[index], right = knots[index + 1];
        if (coordinate >= left.coordinate && coordinate <= right.coordinate) {
          const ratio = (coordinate - left.coordinate) / Math.max(1e-9, right.coordinate - left.coordinate);
          value = Number(left.inch_per_px) + ratio * (Number(right.inch_per_px) - Number(left.inch_per_px));
          break;
        }
      }
    }
    return {inchPerPx: value, source: mode, extrapolated, sampleCount: samples.length};
  }
  const width = Math.max(1, Number(scaleMap.image_width) || 1);
  const height = Math.max(1, Number(scaleMap.image_height) || 1);
  let weighted = 0, totalWeight = 0, exact = null;
  let nearestDistance = Number.POSITIVE_INFINITY, nearestValue = null;
  samples.forEach(sample => {
    const distance = Math.hypot((x - sample.x) / width, (y - sample.y) / height);
    if (distance < 1e-9) exact = Number(sample.inch_per_px);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestValue = Number(sample.inch_per_px);
    }
    const weight = 1 / Math.max(1e-9, distance) ** Number(scaleMap.idw_power || 2);
    weighted += weight * Number(sample.inch_per_px);
    totalWeight += weight;
  });
  if (exact !== null) {
    return {inchPerPx: exact, source: 'idw_2d', extrapolated: false, sampleCount: samples.length};
  }
  const insideHull = axisData.hull?.length >= 3 && pointInsideScaleHull(x, y, axisData.hull);
  if (!insideHull) {
    return {inchPerPx: nearestValue, source: 'nearest_2d', extrapolated: true, sampleCount: samples.length};
  }
  return {
    inchPerPx: exact ?? (weighted / totalWeight),
    source: 'idw_2d',
    extrapolated: false,
    sampleCount: samples.length,
  };
}

function currentMeasureCalibration() {
  return {inch_per_px: m.inchPerPx, scale_map: m.scaleMap};
}

function toggleScaleMap() {
  m.showScaleMap = !m.showScaleMap;
  document.getElementById('m-map-toggle').classList.toggle('active', m.showScaleMap);
  drawAll();
}

function applyMeasureCalibration(calibration) {
  m.segments = (calibration.segments || []).map(s => ({...s}));
  m.referenceY = calibration.reference_y ?? null;
  m.exclusionZones = (calibration.exclusion_zones || []).map(zone => ({...zone}));
  m.inchPerPx = calibration.inch_per_px ?? computeInchPerPx();
  m.selectedSegment = null;
  m.draggingSegment = null;
  m.scaleMap = calibration.scale_map || buildClientScaleMap(m.segments, m.imgW, m.imgH);
}

function computeInchPerPx() {
  const valid = m.segments.filter(s => Number(s.px) > 0 && Number(s.inches) > 0);
  if (!valid.length) return null;
  const totalIn = valid.reduce((sum, s) => sum + Number(s.inches), 0);
  const totalPx = valid.reduce((sum, s) => sum + Number(s.px), 0);
  return totalPx > 0 ? totalIn / totalPx : null;
}

function updateMeasureInfo() {
  m.inchPerPx = computeInchPerPx();
  m.scaleMap = buildClientScaleMap(m.segments, m.imgW, m.imgH);
  document.getElementById('m-info-video').textContent = sourceName(m.source);
  document.getElementById('m-info-source-frame').textContent = m.source?.source_frame_idx ?? '-';
  document.getElementById('m-info-frame').textContent = m.img ? m.frameIdx : '-';
  document.getElementById('m-info-time').textContent = m.img ? m.timeSec.toFixed(3) + 's' : '-';
  document.getElementById('m-info-size').textContent = m.img ? `${m.imgW}x${m.imgH}` : '-';
  document.getElementById('m-info-inch-px').textContent = m.inchPerPx ? m.inchPerPx.toFixed(5) : '-';
  document.getElementById('m-info-px-inch').textContent = m.inchPerPx ? (1 / m.inchPerPx).toFixed(2) : '-';
  document.getElementById('m-info-segments').textContent = m.segments.length;
  const mapX = m.scaleMap.axes.x;
  const mapY = m.scaleMap.axes.y;
  document.getElementById('m-map-y-count').textContent = mapY.sample_count;
  document.getElementById('m-map-x-count').textContent = mapX.sample_count;
  document.getElementById('m-map-y-mode').textContent =
    mapY.mode === 'idw_2d' ? 'IDW 2D' : (mapY.mode === 'global' ? 'global' : `lineal ${mapY.coordinate.toUpperCase()}`);
  document.getElementById('m-map-y-range').textContent = mapY.sample_count
    ? `${Number(mapY.minimum_inch_per_px).toFixed(5)} - ${Number(mapY.maximum_inch_per_px).toFixed(5)}`
    : '-';
  const coverage = mapY.coverage;
  document.getElementById('m-map-coverage').textContent = !coverage
    ? '-'
    : `${Math.round(coverage.x_span_ratio * 100)}% X | ${Math.round(coverage.y_span_ratio * 100)}% Y`;
  document.getElementById('m-map-ignored').textContent = m.scaleMap.ignored_diagonal_segments || 0;
  document.getElementById('m-info-ref').textContent = m.referenceY === null ? '-' : Math.round(m.referenceY) + ' px';
  document.getElementById('m-info-zones').textContent = m.exclusionZones.length;
  const zoneList = document.getElementById('m-zone-list');
  zoneList.innerHTML = m.exclusionZones.length
    ? m.exclusionZones.map((zone, index) => `<div class="box-item">
        <span class="swatch" style="background:#ef6666"></span>
        <span>#${index + 1} ${Math.round(zone.w)}x${Math.round(zone.h)} px</span>
        <button class="danger" onclick="deleteMeasureExclusionZone(${index})">Borrar</button>
      </div>`).join('')
    : '<div class="kv"><span>Sin zonas rojas.</span></div>';
  const list = document.getElementById('m-segment-list');
  if (!m.segments.length) {
    list.innerHTML = '<div class="kv"><span>Marca un segmento y escribe sus pulgadas.</span></div>';
    return;
  }
  list.innerHTML = m.segments.map((s, i) => {
    const inches = Number(s.inches);
    const px = Number(s.px);
    const inchPerPx = px > 0 ? inches / px : 0;
    const pxPerIn = inchPerPx > 0 ? 1 / inchPerPx : 0;
    const sample = scaleSegmentSample(s, i);
    const axis = sample?.axis ? sample.axis.toUpperCase() : 'diag';
    return `<div class="box-item"><span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span><span>#${i+1} [${axis}] ${inches.toFixed(3)} in | ${px.toFixed(1)} px | ${inchPerPx.toFixed(5)} in/px | ${pxPerIn.toFixed(2)} px/in</span><button onclick="deleteMeasureSegment(${i})">Borrar</button></div>`;
  }).join('');
}

function setMeasureMode(mode) {
  m.mode = mode;
  m.pending = null; m.preview = null;
  m.draggingSegment = null;
  if (mode !== 'segment') m.selectedSegment = null;
  document.getElementById('m-mode-segment').classList.toggle('active', mode === 'segment');
  document.getElementById('m-mode-ref').classList.toggle('active', mode === 'reference');
  document.getElementById('m-mode-exclusion').classList.toggle('active', mode === 'exclusion');
  const message = mode === 'segment'
    ? 'Modo segmento: marca 2 puntos.'
    : (mode === 'reference'
      ? 'Modo Linea Y: click o arrastra la referencia horizontal.'
      : 'Modo Zona roja: marca dos esquinas del area sin boxes.');
  status(message, 'ok');
  drawAll();
}

function measureClick(point) {
  if (!m.img) return;
  if (m.mode === 'reference') {
    m.referenceY = point.y;
    updateMeasureInfo(); drawAll();
    status(`Linea Y en ${Math.round(point.y)} px`, 'ok');
    return;
  }
  if (m.mode === 'exclusion') {
    if (!m.pending) {
      m.pending = point;
      status('Primera esquina de la zona roja marcada. Marca la esquina opuesta.', 'ok');
      drawAll();
      return;
    }
    const x = Math.min(m.pending.x, point.x);
    const y = Math.min(m.pending.y, point.y);
    const w = Math.abs(point.x - m.pending.x);
    const h = Math.abs(point.y - m.pending.y);
    if (w >= 2 && h >= 2) {
      m.exclusionZones.push({x, y, w, h});
      status(`Zona roja guardada: ${Math.round(w)}x${Math.round(h)} px`, 'ok');
    } else {
      status('Zona cancelada: el area es demasiado pequena.', 'err');
    }
    m.pending = null; m.preview = null;
    updateMeasureInfo(); drawAll();
    return;
  }
  if (!m.pending) {
    m.pending = point;
    status('Primer punto de medicion marcado. Marca el segundo.', 'ok');
    drawAll();
    return;
  }
  const px = Math.hypot(point.x - m.pending.x, point.y - m.pending.y);
  const value = window.prompt('Longitud real del segmento en pulgadas:', '');
  const inches = Number(value);
  if (Number.isFinite(inches) && inches > 0 && px > 0) {
    m.segments.push({
      x1: m.pending.x, y1: m.pending.y, x2: point.x, y2: point.y,
      px, inches, inch_per_px: inches / px,
    });
    status(`Segmento guardado: ${inches.toFixed(3)} in | ${px.toFixed(1)} px`, 'ok');
  } else {
    status('Segmento cancelado: pulgadas invalidas.', 'err');
  }
  m.pending = null; m.preview = null;
  updateMeasureInfo(); drawAll();
}

function deleteMeasureSegment(i) {
  m.segments.splice(i, 1);
  if (m.selectedSegment === i) m.selectedSegment = null;
  else if (m.selectedSegment > i) m.selectedSegment -= 1;
  updateMeasureInfo(); drawAll();
}

function deleteMeasureExclusionZone(i) {
  m.exclusionZones.splice(i, 1);
  updateMeasureInfo(); drawAll();
}

function undoMeasureSegment() {
  if (m.pending) { m.pending = null; m.preview = null; }
  else if (m.mode === 'exclusion') m.exclusionZones.pop();
  else m.segments.pop();
  updateMeasureInfo(); drawAll();
}

function clearMeasureCalibration() {
  m.segments = []; m.pending = null; m.preview = null; m.referenceY = null; m.inchPerPx = null;
  m.exclusionZones = [];
  m.selectedSegment = null; m.draggingSegment = null;
  updateMeasureInfo(); drawAll();
}

async function saveMeasureCalibration() {
  updateMeasureInfo();
  const payload = {
    frame_idx: m.frameIdx,
    time_sec: m.timeSec,
    img_w: m.imgW,
    img_h: m.imgH,
    segments: m.segments,
    reference_y: m.referenceY,
    exclusion_zones: m.exclusionZones,
    inch_per_px: m.inchPerPx,
  };
  const r = await fetch('/api/measure/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return null; }
  m.scaleMap = d.scale_map || m.scaleMap;
  p.calibration = d;
  p.calibrationStale = true;
  updatePlayerRulerInfo();
  updateMeasureInfo();
  drawAll();
  const yReferences = Number(d.scale_map?.axes?.y?.sample_count || 0);
  const xReferences = Number(d.scale_map?.axes?.x?.sample_count || 0);
  status(`Mapa guardado: ${yReferences} referencias Y, ${xReferences} referencias X.`, 'ok');
  return d;
}

async function loadPlayerFrame(frameIdx, options = {}) {
  if (!options.quiet) status('Corriendo YOLO + Sobel...', '');
  const r = await fetch('/api/player/frame', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_idx: frameIdx, conf: 0.10})
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return null; }
  const img = new Image();
  return await new Promise((resolve, reject) => {
    img.onload = () => {
      p.img = img; p.imgW = d.width; p.imgH = d.height;
      p.frameIdx = d.frame_idx; p.timeSec = d.time_sec; p.source = d.source || null;
      p.boxes = d.boxes || [];
      p.pieces = d.pieces || [];
      p.measurementSummary = d.measurement_summary || null;
      p.sobel = d.sobel || null;
      p.calibration = d.calibration || null;
      p.measurement = d.measurement || null;
      p.boxRules = d.box_rules || null;
      p.calibrationStale = false;
      if (options.resetView !== false) resetView(p);
      document.getElementById('p-second').value = d.time_sec.toFixed(2);
      document.getElementById('p-timeline').value = d.frame_idx;
      document.getElementById('p-timeline-label').textContent = formatClock(d.time_sec);
      document.getElementById('p-zoom').value = p.zoom;
      document.getElementById('p-zoom-label').textContent = p.zoom.toFixed(1) + 'x';
      updatePlayerInfo();
      updatePlayerCapturesUI();
      fitAll(); drawAll();
      if (!options.quiet) {
        const cls = p.boxes.length ? 'ok' : 'err';
        const validCount = Number(p.measurementSummary?.valid_count || 0);
        status(`Frame ${d.frame_idx}: ${p.boxes.length} piezas, ${validCount} mediciones validas`, cls);
      }
      resolve(d);
    };
    img.onerror = () => {
      const err = new Error('No se pudo cargar la imagen del reproductor');
      status(err.message, 'err');
      reject(err);
    };
    img.src = 'data:image/jpeg;base64,' + d.image;
  });
}

function playerMeasureValue(measurement) {
  if (!measurement) return null;
  const value = Number(measurement.measurement_in ?? measurement.delta_in);
  return Number.isFinite(value) ? value : null;
}

function rulerModeLabel(mode) {
  if (mode === 'x') return 'X';
  if (mode === 'y') return 'Y';
  return 'Libre';
}

function constrainPlayerRulerPoint(start, point) {
  if (p.rulerMode === 'x') return {x: point.x, y: start.y};
  if (p.rulerMode === 'y') return {x: start.x, y: point.y};
  return {x: point.x, y: point.y};
}

function currentPlayerRulerEnd() {
  if (!p.rulerStart) return null;
  return p.rulerEnd || p.rulerPreview;
}

function integrateClientRulerScale(start, end, calibration) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const px = Math.hypot(dx, dy);
  if (!(px > 0)) return {inches: 0, meanInchPerPx: null, extrapolated: false, coverageRatio: 1};
  const steps = Math.max(8, Math.min(128, Math.ceil(px / 12)));
  const stepX = dx / steps;
  const stepY = dy / steps;
  let inches = 0;
  let covered = 0;
  const sources = new Set();
  for (let index = 0; index < steps; index += 1) {
    const x = start.x + (index + 0.5) * stepX;
    const y = start.y + (index + 0.5) * stepY;
    const scaleX = spatialScaleAtClient(calibration, x, y, 'x');
    const scaleY = spatialScaleAtClient(calibration, x, y, 'y');
    if (!(scaleX.inchPerPx > 0) || !(scaleY.inchPerPx > 0)) {
      return {inches: null, meanInchPerPx: null, extrapolated: false, coverageRatio: 0, source: 'missing'};
    }
    inches += Math.hypot(stepX * scaleX.inchPerPx, stepY * scaleY.inchPerPx);
    const relevantOutside =
      (Math.abs(stepX) > 1e-9 && scaleX.extrapolated) ||
      (Math.abs(stepY) > 1e-9 && scaleY.extrapolated);
    if (!relevantOutside) covered += 1;
    if (Math.abs(stepX) > 1e-9) sources.add(scaleX.source);
    if (Math.abs(stepY) > 1e-9) sources.add(scaleY.source);
  }
  return {
    inches,
    meanInchPerPx: inches / px,
    extrapolated: covered < steps,
    coverageRatio: covered / steps,
    source: [...sources].sort().join('+'),
  };
}

function playerRulerMeasurement() {
  const end = currentPlayerRulerEnd();
  if (!p.rulerStart || !end) return null;
  const dx = end.x - p.rulerStart.x;
  const dy = end.y - p.rulerStart.y;
  const px = Math.hypot(dx, dy);
  const spatial = integrateClientRulerScale(p.rulerStart, end, p.calibration || {});
  return {
    dx,
    dy,
    px,
    inches: spatial.inches,
    localInchPerPx: spatial.meanInchPerPx,
    extrapolated: spatial.extrapolated,
    coverageRatio: spatial.coverageRatio,
    scaleSource: spatial.source,
  };
}

function updatePlayerRulerInfo() {
  const tool = document.getElementById('p-ruler-tool');
  const toggle = document.getElementById('p-ruler-toggle');
  if (tool) tool.classList.toggle('active', p.rulerActive);
  if (toggle) toggle.classList.toggle('active', p.rulerActive);
  document.getElementById('p-ruler-x').classList.toggle('active', p.rulerActive && p.rulerMode === 'x');
  document.getElementById('p-ruler-y').classList.toggle('active', p.rulerActive && p.rulerMode === 'y');
  document.getElementById('p-ruler-free').classList.toggle('active', p.rulerActive && p.rulerMode === 'free');
  document.getElementById('p-ruler-info-mode').textContent = p.rulerActive ? rulerModeLabel(p.rulerMode) : 'off';
  const mapXCount = Number(p.calibration?.scale_map?.axes?.x?.sample_count || 0);
  const mapYCount = Number(p.calibration?.scale_map?.axes?.y?.sample_count || 0);
  const inchPerPx = p.calibration && Number(p.calibration.inch_per_px);
  document.getElementById('p-ruler-info-scale').textContent = mapXCount || mapYCount
    ? `mapa X:${mapXCount} Y:${mapYCount}`
    : (Number.isFinite(inchPerPx) && inchPerPx > 0 ? `${inchPerPx.toFixed(6)} in/px` : '-');
  const measure = playerRulerMeasurement();
  if (!measure) {
    document.getElementById('p-ruler-info-distance').textContent = '-';
    document.getElementById('p-ruler-info-delta').textContent = '-';
    return;
  }
  const inches = measure.inches === null ? '' : ` | ${measure.inches.toFixed(3)} in`;
  document.getElementById('p-ruler-info-distance').textContent = `${measure.px.toFixed(1)} px${inches}`;
  const coverage = measure.extrapolated ? ` | fuera mapa ${Math.round(measure.coverageRatio * 100)}%` : '';
  document.getElementById('p-ruler-info-delta').textContent =
    `dx ${measure.dx.toFixed(1)} | dy ${measure.dy.toFixed(1)}${coverage}`;
}

function togglePlayerRulerTool() {
  p.rulerActive = !p.rulerActive;
  if (p.rulerActive) {
    stopPlayerPlayback();
    if (!p.rulerMode) p.rulerMode = 'free';
    p.wrap.style.cursor = 'crosshair';
  } else {
    p.rulerStart = null; p.rulerEnd = null; p.rulerPreview = null;
    p.wrap.style.cursor = 'default';
  }
  updatePlayerRulerInfo();
  drawAll();
  status(p.rulerActive ? `Regla activa: ${rulerModeLabel(p.rulerMode)}` : 'Regla apagada.', 'ok');
}

function setPlayerRulerMode(mode) {
  p.rulerActive = true;
  p.rulerMode = mode;
  p.rulerStart = null; p.rulerEnd = null; p.rulerPreview = null;
  stopPlayerPlayback();
  p.wrap.style.cursor = 'crosshair';
  updatePlayerRulerInfo();
  drawAll();
  status(`Regla ${rulerModeLabel(mode)} activa: marca 2 puntos.`, 'ok');
}

function clearPlayerRuler() {
  p.rulerStart = null; p.rulerEnd = null; p.rulerPreview = null;
  updatePlayerRulerInfo();
  drawAll();
  status('Regla limpia.', 'ok');
}

function playerRulerClick(point) {
  if (!p.rulerActive || !p.img) return;
  stopPlayerPlayback();
  if (!p.rulerStart || p.rulerEnd) {
    p.rulerStart = {x: point.x, y: point.y};
    p.rulerEnd = null;
    p.rulerPreview = {x: point.x, y: point.y};
    status(`Regla ${rulerModeLabel(p.rulerMode)}: marca el segundo punto.`, 'ok');
  } else {
    p.rulerEnd = constrainPlayerRulerPoint(p.rulerStart, point);
    p.rulerPreview = null;
    const measure = playerRulerMeasurement();
    status(measure && measure.inches !== null
      ? `Regla: ${measure.px.toFixed(1)} px | ${measure.inches.toFixed(3)} in`
      : `Regla: ${measure ? measure.px.toFixed(1) : 0} px`, 'ok');
  }
  updatePlayerRulerInfo();
  drawAll();
}

function updatePlayerInfo() {
  const bestConf = p.boxes.reduce((m, b) => Math.max(m, Number(b.conf || 0)), 0);
  const validPieces = p.pieces.filter(piece => piece.valid && piece.measurement);
  const analyzedPieces = p.pieces.filter(piece => piece.sobel && piece.sobel.has_roi);
  const averageEdgeConf = analyzedPieces.length
    ? analyzedPieces.reduce((sum, piece) => sum + Number(piece.sobel.edge_confidence || 0), 0) / analyzedPieces.length
    : null;
  const averageCrm = analyzedPieces.length
    ? analyzedPieces.reduce((sum, piece) => sum + Number(piece.sobel.crm_px || 0), 0) / analyzedPieces.length
    : null;
  document.getElementById('p-info-video').textContent = sourceName(p.source);
  document.getElementById('p-info-source-frame').textContent = p.source?.source_frame_idx ?? '-';
  document.getElementById('p-info-frame').textContent = p.img ? p.frameIdx : '-';
  document.getElementById('p-info-time').textContent = p.img ? p.timeSec.toFixed(3) + 's' : '-';
  document.getElementById('p-info-size').textContent = p.img ? `${p.imgW}x${p.imgH}` : '-';
  document.getElementById('p-info-speed').textContent = `x${p.speed}`;
  document.getElementById('p-speed').textContent = `x${p.speed}`;
  document.getElementById('p-info-boxes').textContent = p.boxes.length;
  document.getElementById('p-info-yolo-conf').textContent = bestConf ? bestConf.toFixed(2) : '-';
  document.getElementById('p-info-excluded').textContent = Number(p.boxRules?.removed_exclusion_count || 0);
  const yoloBadge = document.getElementById('p-yolo-badge');
  yoloBadge.textContent = p.boxes.length ? `YOLO ${p.boxes.length}` : 'YOLO 0';
  yoloBadge.className = 'pill' + (p.boxes.length ? ' ok' : '');

  const state = document.getElementById('p-info-sobel-state');
  const conf = document.getElementById('p-info-sobel-conf');
  const crm = document.getElementById('p-info-sobel-crm');
  const measure = document.getElementById('p-info-measure');
  const sobelBadge = document.getElementById('p-sobel-badge');
  measure.textContent = `${validPieces.length} / ${p.pieces.length}`;
  updatePlayerRulerInfo();
  if (!p.pieces.length) {
    state.textContent = p.boxes.length ? 'sin ROI' : 'sin box YOLO';
    conf.textContent = '-';
    crm.textContent = '-';
    sobelBadge.textContent = 'Sobel -';
    sobelBadge.className = 'pill';
    updatePlayerPieceList();
    return;
  }
  state.textContent = validPieces.length === p.pieces.length ? 'todas validas' : 'revisar piezas';
  conf.textContent = averageEdgeConf === null ? '-' : averageEdgeConf.toFixed(2);
  crm.textContent = averageCrm === null || !Number.isFinite(averageCrm) ? '-' : averageCrm.toFixed(2) + ' px';
  sobelBadge.textContent = `Sobel ${validPieces.length}/${p.pieces.length}`;
  sobelBadge.className = 'pill' + (validPieces.length ? ' ok' : '');
  updatePlayerPieceList();
}

function updatePlayerPieceList() {
  const list = document.getElementById('p-piece-list');
  if (!list) return;
  if (!p.pieces.length) {
    list.innerHTML = '<div class="kv"><span>Sin piezas detectadas.</span></div>';
    return;
  }
  list.innerHTML = p.pieces.map((piece, index) => {
    const measurement = piece.measurement;
    const total = measurement ? formatFeetInches(measurement.measurement_in) : '-';
    const distance = measurement ? `${Number(measurement.delta_in).toFixed(3)} in ref` : 'sin borde';
    const coverage = measurement?.scale_extrapolated
      ? ` | mapa ${Math.round(Number(measurement.scale_coverage_ratio || 0) * 100)}%`
      : '';
    return `<div class="box-item">
      <span class="swatch" style="background:${COLORS[index % COLORS.length]}"></span>
      <span>P${Number(piece.piece_id)} | ${total}<br><small>${distance}${coverage}</small></span>
    </div>`;
  }).join('');
}

async function refreshPlayerCaptures() {
  const r = await fetch('/api/player/captures');
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  p.captures = d.captures || [];
  updatePlayerCapturesUI();
}

function updatePlayerCapturesUI() {
  const list = document.getElementById('p-capture-list');
  if (!list) return;
  if (!p.captures.length) {
    list.innerHTML = '<div class="kv"><span>Sin capturas guardadas.</span></div>';
    return;
  }
  list.innerHTML = p.captures.map(item => {
    const cls = item.frame_idx === p.frameIdx ? 'history-item current' : 'history-item';
    const time = Number(item.time_sec || 0).toFixed(2);
    const measurementCount = Number(item.measurement_count || 0);
    const value = measurementCount
      ? `${measurementCount} mediciones`
      : (item.measurement_in === null || item.measurement_in === undefined
        ? 'sin medida'
        : formatFeetInches(item.measurement_in));
    return `<div class="${cls}" onclick="goToPlayerCapture(${Number(item.frame_idx)})" title="Ir a ${time}s">
      <div><strong>Frame ${Number(item.frame_idx)}</strong><span>${time}s | ${value}</span></div>
      <div class="history-actions">
        <button onclick="goToPlayerCapture(${Number(item.frame_idx)}); event.stopPropagation();">Ir</button>
        <button class="danger" onclick="deletePlayerCapture('${String(item.id || '').replace(/'/g, "\\'")}'); event.stopPropagation();">Borrar</button>
      </div>
    </div>`;
  }).join('');
}

function goToPlayerCapture(frameIdx) {
  stopPlayerPlayback();
  return loadPlayerFrame(Number(frameIdx), {resetView: false});
}

async function savePlayerCapture() {
  if (!p.img) {
    status('Carga un frame en el reproductor antes de guardar captura.', 'err');
    return;
  }
  const payload = {
    frame_idx: p.frameIdx,
    time_sec: p.timeSec,
    img_w: p.imgW,
    img_h: p.imgH,
    boxes: p.boxes,
    pieces: p.pieces,
    measurement_summary: p.measurementSummary,
    sobel: p.sobel,
    measurement: p.measurement,
  };
  const r = await fetch('/api/player/captures', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  p.captures = d.captures || [];
  updatePlayerCapturesUI();
  const value = playerMeasureValue(d.capture && d.capture.measurement);
  status(`Captura guardada: frame ${d.capture.frame_idx}${value === null ? '' : ' | ' + formatFeetInches(value)}`, 'ok');
}

async function deletePlayerCapture(captureId) {
  if (!captureId) return;
  const r = await fetch(`/api/player/captures/${encodeURIComponent(captureId)}`, {method: 'DELETE'});
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  p.captures = d.captures || [];
  updatePlayerCapturesUI();
  status(`Captura borrada: ${captureId}`, 'ok');
}

function loadPlayerSecond() {
  stopPlayerPlayback();
  const second = parseFloat(document.getElementById('p-second').value) || 0;
  return loadPlayerFrame(Math.round(second * meta.fps));
}

function previewPlayerTimeline(frameValue) {
  const frameIdx = Math.max(0, Math.min(meta.totalFrames - 1, Number(frameValue) || 0));
  const second = frameIdx / meta.fps;
  document.getElementById('p-second').value = second.toFixed(2);
  document.getElementById('p-timeline-label').textContent = formatClock(second);
}

function loadPlayerTimeline(frameValue) {
  stopPlayerPlayback();
  clearTimeout(p.timelineTimer);
  p.timelineTimer = null;
  return loadPlayerFrame(Number(frameValue) || 0, {resetView: false});
}

function scrollPlayerTimeline(event) {
  event.preventDefault();
  stopPlayerPlayback();
  const timeline = event.currentTarget;
  const direction = (event.deltaY || event.deltaX) > 0 ? 1 : -1;
  const jumpSeconds = event.shiftKey ? 5 : 30;
  const nextFrame = Math.max(
    0,
    Math.min(meta.totalFrames - 1, Number(timeline.value) + direction * Math.round(meta.fps * jumpSeconds))
  );
  timeline.value = nextFrame;
  previewPlayerTimeline(nextFrame);
  clearTimeout(p.timelineTimer);
  p.timelineTimer = setTimeout(() => loadPlayerTimeline(nextFrame), 180);
}

function stepPlayer(delta) {
  stopPlayerPlayback();
  return loadPlayerFrame(p.frameIdx + delta, {resetView: false});
}

function togglePlayerSpeed() {
  p.speed = p.speed === 1 ? 2 : 1;
  updatePlayerInfo();
  status(`Velocidad del reproductor: x${p.speed}`, 'ok');
}

function stopPlayerPlayback() {
  p.playing = false;
  const button = document.getElementById('p-play');
  if (button) button.textContent = 'Play';
}

async function togglePlayerPlay() {
  if (p.playing) {
    stopPlayerPlayback();
    return;
  }
  if (!p.img) {
    const loaded = await loadPlayerSecond();
    if (!loaded) return;
  }
  p.playing = true;
  document.getElementById('p-play').textContent = 'Pausa';
  if (!p.playTask) {
    p.playTask = playerLoop().finally(() => { p.playTask = null; });
  }
}

async function playerLoop() {
  while (p.playing) {
    const maxFrame = Math.max(0, meta.totalFrames - 1);
    const step = p.speed === 2 ? 2 : 1;
    const nextFrame = Math.min(p.frameIdx + step, maxFrame);
    if (nextFrame === p.frameIdx) {
      stopPlayerPlayback();
      break;
    }
    try {
      await loadPlayerFrame(nextFrame, {quiet: true, resetView: false});
    } catch (e) {
      stopPlayerPlayback();
      break;
    }
    await sleep(15);
  }
}

function drawAnnotate() {
  const ctx = a.ctx, cw = a.canvas.width, ch = a.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!a.img) return;
  const vw = a.imgW / a.zoom, vh = a.imgH / a.zoom;
  ctx.drawImage(a.img, a.panX, a.panY, vw, vh, 0, 0, cw, ch);
  drawGrid(a, 100);
  a.modelBoxes.forEach((b, i) => {
    const color = b.inferred ? '#d6a34b' : '#70c7c2';
    drawBox(b, color, true, `M${i + 1}`);
  });
  a.boxes.forEach((b, i) => drawBox(b, COLORS[i % COLORS.length], false, i + 1));
  drawAnnotationSelection();
  if (a.preview) drawBox(a.preview, '#ffffff', true, '?');
  drawSobelProjection();
}

function drawExclusionZones(state, zones, showLabels = false) {
  const ctx = state.ctx;
  (zones || []).forEach((zone, index) => {
    const topLeft = imageToDisplay(state, zone.x, zone.y);
    const bottomRight = imageToDisplay(state, zone.x + zone.w, zone.y + zone.h);
    const width = bottomRight.x - topLeft.x;
    const height = bottomRight.y - topLeft.y;
    ctx.save();
    ctx.fillStyle = 'rgba(210, 48, 48, 0.18)';
    ctx.strokeStyle = '#ef6666';
    ctx.lineWidth = 2;
    ctx.fillRect(topLeft.x, topLeft.y, width, height);
    ctx.strokeRect(topLeft.x, topLeft.y, width, height);
    if (showLabels) {
      ctx.fillStyle = '#ffffff';
      ctx.font = '800 12px Arial';
      ctx.fillText(`NO BOX ${index + 1}`, topLeft.x + 7, topLeft.y + 17);
    }
    ctx.restore();
  });
}

function scaleMapColor(value, minimum, maximum, alpha) {
  const span = Math.max(1e-9, maximum - minimum);
  const ratio = Math.max(0, Math.min(1, (value - minimum) / span));
  const stops = [
    [72, 126, 176],
    [70, 158, 122],
    [214, 163, 75],
    [204, 83, 83],
  ];
  const scaled = ratio * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const local = scaled - index;
  const color = stops[index].map((channel, channelIndex) =>
    Math.round(channel + (stops[index + 1][channelIndex] - channel) * local)
  );
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
}

function drawScaleMapOverlay() {
  const axis = m.scaleMap?.axes?.y;
  if (!m.showScaleMap || !axis || !axis.sample_count) return;
  const ctx = m.ctx;
  const columns = 24;
  const rows = axis.mode === 'idw_2d' ? 12 : 1;
  const minimum = Number(axis.minimum_inch_per_px);
  const maximum = Number(axis.maximum_inch_per_px);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const x0 = column * m.imgW / columns;
      const x1 = (column + 1) * m.imgW / columns;
      const y0 = row * m.imgH / rows;
      const y1 = (row + 1) * m.imgH / rows;
      const scale = spatialScaleAtClient(
        currentMeasureCalibration(),
        (x0 + x1) / 2,
        (y0 + y1) / 2,
        'y'
      );
      if (!(scale.inchPerPx > 0)) continue;
      const topLeft = imageToDisplay(m, x0, y0);
      const bottomRight = imageToDisplay(m, x1, y1);
      ctx.fillStyle = scaleMapColor(
        scale.inchPerPx,
        minimum,
        maximum,
        scale.extrapolated ? 0.08 : 0.18
      );
      ctx.fillRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    }
  }
  const coverage = axis.coverage;
  if (coverage) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,.72)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([7, 5]);
    if (axis.mode === 'idw_2d' && axis.hull?.length >= 3) {
      ctx.beginPath();
      axis.hull.forEach((point, index) => {
        const display = imageToDisplay(m, point.x, point.y);
        if (index === 0) ctx.moveTo(display.x, display.y);
        else ctx.lineTo(display.x, display.y);
      });
      ctx.closePath();
      ctx.stroke();
    } else {
      const coverageX0 = axis.mode === 'linear_y' ? 0 : coverage.x_min;
      const coverageX1 = axis.mode === 'linear_y' ? m.imgW : coverage.x_max;
      const coverageY0 = axis.mode === 'linear_x' ? 0 : coverage.y_min;
      const coverageY1 = axis.mode === 'linear_x' ? m.imgH : coverage.y_max;
      const topLeft = imageToDisplay(m, coverageX0, coverageY0);
      const bottomRight = imageToDisplay(m, coverageX1, coverageY1);
      ctx.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    }
    ctx.restore();
  }
}

function drawMeasure() {
  const ctx = m.ctx, cw = m.canvas.width, ch = m.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!m.img) return;
  const vw = m.imgW / m.zoom, vh = m.imgH / m.zoom;
  ctx.drawImage(m.img, m.panX, m.panY, vw, vh, 0, 0, cw, ch);
  drawScaleMapOverlay();
  drawGrid(m, 100);
  drawExclusionZones(m, m.exclusionZones, true);

  if (m.referenceY !== null) {
    const left = imageToDisplay(m, 0, m.referenceY);
    const right = imageToDisplay(m, m.imgW - 1, m.referenceY);
    ctx.save();
    ctx.strokeStyle = '#5b9bd5';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(left.x, left.y); ctx.lineTo(right.x, right.y); ctx.stroke();
    ctx.fillStyle = '#5b9bd5';
    ctx.font = '700 12px Arial';
    ctx.fillText(`Y ref ${Math.round(m.referenceY)} px`, 10, Math.max(16, left.y - 8));
    ctx.restore();
  }

  m.segments.forEach((s, i) => {
    const a1 = imageToDisplay(m, s.x1, s.y1);
    const a2 = imageToDisplay(m, s.x2, s.y2);
    const color = COLORS[i % COLORS.length];
    const activeStart = m.draggingSegment?.index === i && m.draggingSegment?.part === 'start';
    const activeEnd = m.draggingSegment?.index === i && m.draggingSegment?.part === 'end';
    ctx.save();
    if (m.selectedSegment === i) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 7;
      ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y); ctx.stroke();
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y); ctx.stroke();
    ctx.fillStyle = color;
    const handleRadius = m.selectedSegment === i ? 7 : 6;
    ctx.beginPath(); ctx.arc(a1.x, a1.y, activeStart ? 9 : handleRadius, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(a2.x, a2.y, activeEnd ? 9 : handleRadius, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(a1.x, a1.y, activeStart ? 12 : handleRadius + 3, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(a2.x, a2.y, activeEnd ? 12 : handleRadius + 3, 0, Math.PI * 2); ctx.stroke();
    ctx.font = '700 12px Arial';
    ctx.fillStyle = color;
    ctx.fillText(`${Number(s.inches).toFixed(2)} in`, (a1.x + a2.x) / 2 + 8, (a1.y + a2.y) / 2 - 8);
    ctx.restore();
  });

  if (m.pending && m.preview) {
    const p1 = imageToDisplay(m, m.pending.x, m.pending.y);
    const p2 = imageToDisplay(m, m.preview.x, m.preview.y);
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    if (m.mode === 'exclusion') {
      ctx.strokeStyle = '#ff7a7a';
      ctx.fillStyle = 'rgba(210, 48, 48, 0.22)';
      ctx.fillRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
    } else {
      ctx.strokeStyle = '#ffffff';
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    }
    ctx.restore();
  }
}

function drawPlayer() {
  const ctx = p.ctx, cw = p.canvas.width, ch = p.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!p.img) return;
  const vw = p.imgW / p.zoom, vh = p.imgH / p.zoom;
  ctx.drawImage(p.img, p.panX, p.panY, vw, vh, 0, 0, cw, ch);
  drawGrid(p, 100);
  drawExclusionZones(p, p.calibration?.exclusion_zones || [], false);
  drawPlayerMeasurement();
  p.boxes.forEach((b, i) => {
    const color = b.inferred ? '#d6a34b' : COLORS[i % COLORS.length];
    drawBoxOnState(p, b, color, Boolean(b.inferred), i + 1);
  });
  if (p.pieces.length) {
    p.pieces.forEach(piece => drawSobelOnState(p, piece.sobel));
  } else {
    drawSobelOnState(p, p.sobel);
  }
  drawPlayerRuler();
}

function normBox(b) {
  if ('x' in b && 'y' in b && 'w' in b && 'h' in b) {
    return {...b, x: b.x, y: b.y, w: b.w, h: b.h};
  }
  return {
    ...b,
    x: Math.min(b.x1, b.x2),
    y: Math.min(b.y1, b.y2),
    w: Math.abs(b.x2 - b.x1),
    h: Math.abs(b.y2 - b.y1)
  };
}

function boxToCorners(b) {
  if ('x1' in b && 'y1' in b && 'x2' in b && 'y2' in b) return b;
  return {...b, x1: b.x, y1: b.y, x2: b.x + b.w, y2: b.y + b.h};
}

function drawBox(b, color, dashed, label) {
  drawBoxOnState(a, b, color, dashed, label);
}

function drawBoxOnState(state, b, color, dashed, label) {
  const nb = normBox(b);
  const p1 = imageToDisplay(state, nb.x, nb.y);
  const p2 = imageToDisplay(state, nb.x + nb.w, nb.y + nb.h);
  const ctx = state.ctx;
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = dashed ? 1.5 : 2;
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
  ctx.globalAlpha = dashed ? .06 : .12;
  ctx.fillStyle = color; ctx.fillRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
  ctx.restore();
  if (!dashed || String(label).startsWith('M') || b.inferred) {
    ctx.fillStyle = color; ctx.font = '700 12px Arial';
    const prefix = String(label).startsWith('M') ? '' : '#';
    ctx.fillText(`${prefix}${label} ${Math.round(nb.w)}x${Math.round(nb.h)}`, p1.x + 4, p1.y + 14);
  }
}

function drawAnnotationSelection() {
  if (a.selectedBoxIndex < 0 || a.selectedBoxIndex >= a.boxes.length) return;
  const box = normBox(a.boxes[a.selectedBoxIndex]);
  const topLeft = imageToDisplay(a, box.x, box.y);
  const bottomRight = imageToDisplay(a, box.x + box.w, box.y + box.h);
  const handles = [
    {x: topLeft.x, y: topLeft.y},
    {x: bottomRight.x, y: topLeft.y},
    {x: bottomRight.x, y: bottomRight.y},
    {x: topLeft.x, y: bottomRight.y},
  ];
  const ctx = a.ctx;
  ctx.save();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(topLeft.x - 2, topLeft.y - 2, bottomRight.x - topLeft.x + 4, bottomRight.y - topLeft.y + 4);
  ctx.setLineDash([]);
  handles.forEach(handle => {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(handle.x - 5, handle.y - 5, 10, 10);
    ctx.strokeStyle = '#172027';
    ctx.strokeRect(handle.x - 5, handle.y - 5, 10, 10);
  });
  ctx.restore();
}

function drawPlayerMeasurement() {
  if (!p.calibration || p.calibration.reference_y === null || p.calibration.reference_y === undefined) return;
  const ctx = p.ctx;
  const y = Number(p.calibration.reference_y);
  const left = imageToDisplay(p, 0, y);
  const right = imageToDisplay(p, p.imgW - 1, y);
  ctx.save();
  ctx.strokeStyle = '#5b9bd5';
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(left.x, left.y); ctx.lineTo(right.x, right.y); ctx.stroke();
  ctx.fillStyle = '#5b9bd5';
  ctx.font = '700 12px Arial';
  ctx.fillText('Y ref', 10, Math.max(16, left.y - 8));
  const measuredPieces = p.pieces.length
    ? p.pieces.filter(piece => piece.measurement)
    : (p.measurement ? [{piece_id: 1, measurement: p.measurement}] : []);
  measuredPieces.forEach((piece, index) => {
    const measurement = piece.measurement;
    const q1 = imageToDisplay(p, measurement.x, measurement.reference_y);
    const q2 = imageToDisplay(p, measurement.x, measurement.line_y);
    const color = COLORS[index % COLORS.length];
    const label = `P${Number(piece.piece_id)} ${formatFeetInches(measurement.measurement_in)}`;
    const labelY = Math.max(24, Math.min(p.canvas.height - 8, q2.y + 22 + (index % 2) * 20));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(q1.x, q1.y); ctx.lineTo(q2.x, q2.y); ctx.stroke();
    ctx.font = '800 13px Arial';
    const labelWidth = ctx.measureText(label).width;
    const labelX = Math.max(6, Math.min(p.canvas.width - labelWidth - 6, q2.x - 34));
    ctx.fillStyle = 'rgba(0, 0, 0, 0.62)';
    ctx.fillRect(labelX - 4, labelY - 15, labelWidth + 8, 20);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, labelX, labelY);
  });
  ctx.restore();
}

function drawPlayerRuler() {
  if (!p.rulerActive || !p.rulerStart) return;
  const end = currentPlayerRulerEnd();
  if (!end) return;
  const measure = playerRulerMeasurement();
  const a1 = imageToDisplay(p, p.rulerStart.x, p.rulerStart.y);
  const a2 = imageToDisplay(p, end.x, end.y);
  const ctx = p.ctx;
  const color = p.rulerMode === 'x' ? '#70c7c2' : (p.rulerMode === 'y' ? '#d6a34b' : '#ffffff');
  ctx.save();
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 7;
  ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y); ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a2.x, a2.y); ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(a1.x, a1.y, 5, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(a2.x, a2.y, 5, 0, Math.PI * 2); ctx.fill();
  if (measure) {
    const inches = measure.inches === null ? '' : ` | ${measure.inches.toFixed(3)} in`;
    const label = `${rulerModeLabel(p.rulerMode)} ${measure.px.toFixed(1)} px${inches}`;
    const mx = Math.max(8, Math.min(p.canvas.width - 250, (a1.x + a2.x) / 2 + 10));
    const my = Math.max(28, Math.min(p.canvas.height - 12, (a1.y + a2.y) / 2 - 10));
    ctx.font = '800 18px Arial';
    const metrics = ctx.measureText(label);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.62)';
    ctx.fillRect(mx - 7, my - 22, metrics.width + 14, 28);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, mx, my);
  }
  ctx.restore();
}

function drawSobelOnState(state, sobel) {
  if (!sobel) return;
  const ctx = state.ctx;
  if (sobel.roi) {
    const r1 = imageToDisplay(state, sobel.roi.x, sobel.roi.y);
    const r2 = imageToDisplay(state, sobel.roi.x + sobel.roi.w, sobel.roi.y + sobel.roi.h);
    ctx.save();
    ctx.strokeStyle = '#d6a34b';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 5]);
    ctx.strokeRect(r1.x, r1.y, r2.x - r1.x, r2.y - r1.y);
    ctx.restore();
  }
  if (sobel.points) {
    sobel.points.forEach(point => {
      const q = imageToDisplay(state, point.x, point.y);
      ctx.fillStyle = point.inlier ? '#53b689' : '#d6a34b';
      ctx.fillRect(q.x - 2, q.y - 2, 4, 4);
    });
  }
  if (sobel.line) {
    const p1 = imageToDisplay(state, sobel.line.x1, sobel.line.y1);
    const p2 = imageToDisplay(state, sobel.line.x2, sobel.line.y2);
    ctx.save();
    ctx.strokeStyle = sobel.is_valid ? '#53b689' : '#d6a34b';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    ctx.restore();
  }
}

function drawSobelProjection() {
  if (a.pieces.length) {
    a.pieces.forEach(piece => drawSobelOnState(a, piece.sobel));
    return;
  }
  if (!a.sobel) return;
  const ctx = a.ctx;
  if (a.sobel.roi) {
    const r1 = imageToDisplay(a, a.sobel.roi.x, a.sobel.roi.y);
    const r2 = imageToDisplay(a, a.sobel.roi.x + a.sobel.roi.w, a.sobel.roi.y + a.sobel.roi.h);
    ctx.save();
    ctx.strokeStyle = '#d6a34b';
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 5]);
    ctx.strokeRect(r1.x, r1.y, r2.x - r1.x, r2.y - r1.y);
    ctx.restore();
  }
  if (a.sobel.points) {
    a.sobel.points.forEach(p => {
      const q = imageToDisplay(a, p.x, p.y);
      a.ctx.fillStyle = p.inlier ? '#53b689' : '#d6a34b';
      a.ctx.fillRect(q.x - 2, q.y - 2, 4, 4);
    });
  }
  if (a.sobel.line) {
    const p1 = imageToDisplay(a, a.sobel.line.x1, a.sobel.line.y1);
    const p2 = imageToDisplay(a, a.sobel.line.x2, a.sobel.line.y2);
    ctx.save();
    ctx.strokeStyle = a.sobel.is_valid ? '#53b689' : '#d6a34b';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    ctx.restore();
  }
}

function updateBoxes() {
  document.getElementById('info-boxes').textContent = a.boxes.length;
  const saveButton = document.getElementById('save-frame');
  saveButton.disabled = !a.img || a.pendingSaves > 0;
  saveButton.textContent = a.pendingSaves > 0
    ? 'Guardando...'
    : (a.boxes.length ? 'Guardar ahora' : 'Guardar negativo');
  const list = document.getElementById('box-list');
  if (!a.boxes.length) {
    list.innerHTML = '<div class="kv"><span>Sin boxes. Se guardara como negativo.</span></div>';
    updateModelBoxes();
    return;
  }
  list.innerHTML = a.boxes.map((b, i) => {
    const nb = normBox(b);
    return `<div class="box-item"><span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span><span>#${i+1} ${Math.round(nb.w)}x${Math.round(nb.h)} @ ${Math.round(nb.x)},${Math.round(nb.y)}</span><button onclick="deleteBox(${i})">Borrar</button></div>`;
  }).join('');
  updateModelBoxes();
}

function modelAnnotationComparison() {
  const annotationBoxes = a.boxes.map(normBox);
  const modelBoxes = a.modelBoxes.map(normBox);
  const matchedAnnotations = new Set();
  let matchedPredictions = 0;
  modelBoxes.forEach(modelBox => {
    let bestIndex = -1;
    let bestOverlap = 0;
    annotationBoxes.forEach((annotationBox, index) => {
      const overlap = boxOverlapFraction(modelBox, annotationBox);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestIndex = index;
      }
    });
    if (bestIndex >= 0 && bestOverlap >= 0.35) {
      matchedPredictions += 1;
      matchedAnnotations.add(bestIndex);
    }
  });
  return {
    matchedPredictions,
    missedAnnotations: Math.max(0, annotationBoxes.length - matchedAnnotations.size),
  };
}

function updateModelBoxes() {
  const comparison = modelAnnotationComparison();
  const inferredCount = a.modelBoxes.filter(box => box.inferred).length;
  document.getElementById('info-model-boxes').textContent = a.modelBoxes.length
    ? `${a.modelBoxes.length}${inferredCount ? ` (${inferredCount} por regla)` : ''}`
    : (a.boxRules ? '0' : '-');
  document.getElementById('info-model-matched').textContent = a.boxRules
    ? comparison.matchedPredictions
    : '-';
  document.getElementById('info-model-missed').textContent = a.boxRules
    ? comparison.missedAnnotations
    : '-';
  const list = document.getElementById('model-box-list');
  if (!a.boxRules) {
    list.innerHTML = '';
    return;
  }
  if (!a.modelBoxes.length) {
    list.innerHTML = '<div class="kv"><span>Sin detecciones.</span></div>';
    return;
  }
  list.innerHTML = a.modelBoxes.map((box, index) => {
    const nb = normBox(box);
    const color = box.inferred ? '#d6a34b' : '#70c7c2';
    const source = box.inferred ? 'regla + Sobel' : `conf ${Number(box.conf || 0).toFixed(2)}`;
    return `<div class="box-item"><span class="swatch" style="background:${color}"></span><span>M${index + 1} ${Math.round(nb.w)}x${Math.round(nb.h)} | ${source}</span><span></span></div>`;
  }).join('');
}

function boxOverlapFraction(first, second) {
  const left = Math.max(Number(first.x), Number(second.x));
  const top = Math.max(Number(first.y), Number(second.y));
  const right = Math.min(Number(first.x) + Number(first.w), Number(second.x) + Number(second.w));
  const bottom = Math.min(Number(first.y) + Number(first.h), Number(second.y) + Number(second.h));
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const smallerArea = Math.min(
    Math.max(0, Number(first.w) * Number(first.h)),
    Math.max(0, Number(second.w) * Number(second.h))
  );
  return smallerArea > 0 ? intersection / smallerArea : 0;
}

function updateSobelInfo() {
  const state = document.getElementById('info-sobel-state');
  const conf = document.getElementById('info-sobel-conf');
  const crm = document.getElementById('info-sobel-crm');
  if (!state || !conf || !crm) return;
  if (!a.sobel) {
    state.textContent = 'sin correr';
    conf.textContent = '-';
    crm.textContent = '-';
    return;
  }
  state.textContent = a.sobel.is_valid ? 'linea valida' : (a.sobel.has_roi ? 'linea debil' : 'sin ROI');
  conf.textContent = Number(a.sobel.edge_confidence || 0).toFixed(2);
  crm.textContent = Number(a.sobel.crm_px || 0).toFixed(2) + ' px';
}

async function predictYoloBoxes(options = {}) {
  if (!a.img) return;
  if (!options.quiet) status('Corriendo YOLO...', '');
  const r = await fetch('/api/annotate/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_idx: a.frameIdx, conf: 0.10})
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  a.modelBoxes = (d.boxes || []).map(boxToCorners);
  a.boxRules = d.box_rules || {};
  updateModelBoxes();
  drawAll();
  if (!options.quiet) {
    const comparison = modelAnnotationComparison();
    const inferred = Number(a.boxRules.inferred_count || 0);
    const excluded = Number(a.boxRules.removed_exclusion_count || 0);
    const message = a.modelBoxes.length
      ? `Modelo: ${a.modelBoxes.length} detecciones | ${comparison.matchedPredictions} coincidencias | ${comparison.missedAnnotations} omitidas${inferred ? ` | ${inferred} inferidas por regla` : ''}${excluded ? ` | ${excluded} descartadas por zona` : ''}`
      : `Modelo: sin detecciones | ${comparison.missedAnnotations} anotaciones omitidas${excluded ? ` | ${excluded} descartadas por zona` : ''}`;
    status(message, a.modelBoxes.length ? 'ok' : 'err');
  }
  return d;
}

async function runModel() {
  const button = document.getElementById('a-run-model');
  if (!a.img || button.disabled) return;
  button.disabled = true;
  button.textContent = 'Running...';
  try {
    await predictYoloBoxes();
  } finally {
    button.disabled = false;
    button.textContent = 'Run model';
  }
}

async function runSobelProjection(options = {}) {
  if (!a.img) return;
  if (!options.quiet) status('Calculando Sobel projection...', '');
  const r = await fetch('/api/annotate/sobel_projection', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_idx: a.frameIdx, boxes: a.boxes.map(normBox), conf: 0.10})
  });
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  a.sobel = d;
  a.pieces = d.pieces || [];
  if (!a.boxes.length && d.roi_box) a.boxes = [boxToCorners(d.roi_box)];
  updateBoxes();
  updateSobelInfo();
  drawAll();
  const validCount = Number(d.measurement_summary?.valid_count || 0);
  const msg = `${validCount}/${a.pieces.length} piezas con medicion valida`;
  status(msg, validCount ? 'ok' : 'err');
  return d;
}

async function refreshHistory() {
  const r = await fetch('/api/annotate/history');
  const d = await r.json();
  if (!r.ok) { status(d.error, 'err'); return; }
  a.history = d.items || [];
  a.saved = d.count || 0;
  document.getElementById('info-saved').textContent = a.saved;
  updateHistoryUI();
}

function updateHistoryUI() {
  const list = document.getElementById('history-list');
  if (!a.history.length) {
    list.innerHTML = '<div class="kv"><span>Sin frames guardados.</span></div>';
    return;
  }
  list.innerHTML = a.history.map(item => {
    const cls = item.frame_idx === a.frameIdx ? 'history-item current' : 'history-item';
    const kind = item.candidate
      ? 'pendiente: anotar piezas'
      : (item.box_count === 0 ? 'negativo' : `${item.box_count} piezas`);
    const time = Number(item.time_sec || 0).toFixed(2);
    const source = sourceName(item.source);
    return `<div class="${cls}">
      <div><strong>${source} | Frame ${item.frame_idx}</strong><span>${time}s | ${kind}</span></div>
      <button onclick="loadAnnotateFrame(${item.frame_idx})">Ir</button>
    </div>`;
  }).join('');
}

function annotationPayloadBoxes() {
  return a.boxes.map(box => {
    const normalized = normBox(box);
    return {x: normalized.x, y: normalized.y, w: normalized.w, h: normalized.h};
  });
}

function resetAnnotationInteraction() {
  a.dragMode = null;
  a.dragStart = null;
  a.dragOrigin = null;
  a.dragHandle = null;
  a.dragAdopted = false;
  a.dragModelBox = null;
  a.interactionChanged = false;
  a.preview = null;
}

function clearAnnotationAnalysis() {
  a.sobel = null;
  a.pieces = [];
  updateSobelInfo();
}

function annotationContentChanged(message = 'Anotacion actualizada.') {
  a.editVersion += 1;
  a.dirty = true;
  clearAnnotationAnalysis();
  updateBoxes();
  updateModelBoxes();
  drawAll();
  status(`${message} Guardando automaticamente...`, '');
  void saveFrame({automatic: true});
}

async function flushAnnotationAutosave() {
  await a.saveChain;
  if (!a.dirty || !a.img) return true;
  const result = await saveFrame({automatic: true, quiet: true});
  return Boolean(result) && !a.dirty;
}

function deleteBox(i) {
  if (i < 0 || i >= a.boxes.length) return;
  a.boxes.splice(i, 1);
  if (a.selectedBoxIndex === i) {
    a.selectedBoxIndex = Math.min(i, a.boxes.length - 1);
  } else if (a.selectedBoxIndex > i) {
    a.selectedBoxIndex -= 1;
  }
  resetAnnotationInteraction();
  annotationContentChanged('Box eliminado.');
}

function undoBox() {
  if (a.dragMode) {
    cancelAnnotationInteraction();
    return;
  }
  if (!a.boxes.length) return;
  a.boxes.pop();
  a.selectedBoxIndex = Math.min(a.selectedBoxIndex, a.boxes.length - 1);
  annotationContentChanged('Ultimo box eliminado.');
}

function clearBoxes() {
  if (!a.boxes.length) return;
  a.boxes = [];
  a.selectedBoxIndex = -1;
  resetAnnotationInteraction();
  annotationContentChanged('Boxes eliminados; el frame quedara como negativo.');
}

async function saveFrame(options = {}) {
  if (!a.img) return null;
  const automatic = Boolean(options.automatic);
  if (automatic && !a.dirty) return null;
  const frameIdx = a.frameIdx;
  const version = a.editVersion;
  const payload = {
    frame_idx: frameIdx, time_sec: a.timeSec, img_w: a.imgW, img_h: a.imgH,
    boxes: annotationPayloadBoxes()
  };
  a.pendingSaves += 1;
  updateBoxes();

  const task = a.saveChain.catch(() => null).then(async () => {
    const r = await fetch('/api/annotate/save', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'No se pudo guardar el frame');
    return d;
  });
  a.saveChain = task.catch(() => null);

  try {
    const d = await task;
    a.saved = d.saved_count;
    document.getElementById('info-saved').textContent = d.saved_count;
    a.history = d.history || a.history;
    updateHistoryUI();
    if (a.frameIdx === frameIdx && a.editVersion === version) {
      a.dirty = false;
      const kind = payload.boxes.length ? `${payload.boxes.length} boxes` : 'negativo sin boxes';
      if (!options.quiet) {
        const saveMode = automatic ? 'guardado automaticamente' : 'guardado';
        status(`Frame ${frameIdx} ${saveMode} (${kind}).`, 'ok');
      }
    }
    return d;
  } catch (error) {
    if (a.frameIdx === frameIdx) {
      a.dirty = true;
      status(`No se pudo guardar automaticamente: ${error.message || error}`, 'err');
    }
    return null;
  } finally {
    a.pendingSaves = Math.max(0, a.pendingSaves - 1);
    updateBoxes();
  }
}

function setAnnotationBoxFromNorm(index, box) {
  if (index < 0 || index >= a.boxes.length) return;
  a.boxes[index] = {
    x1: Number(box.x),
    y1: Number(box.y),
    x2: Number(box.x) + Number(box.w),
    y2: Number(box.y) + Number(box.h),
  };
}

function annotationHitTolerance() {
  if (!a.imgW || !a.imgH || !a.canvas.width || !a.canvas.height) return 6;
  const unitsPerDisplayPixel = Math.max(
    (a.imgW / a.zoom) / a.canvas.width,
    (a.imgH / a.zoom) / a.canvas.height,
  );
  return Math.max(4, unitsPerDisplayPixel * 10);
}

function pointInsideAnnotationBox(point, box, padding = 0) {
  const normalized = normBox(box);
  return point.x >= normalized.x - padding
    && point.x <= normalized.x + normalized.w + padding
    && point.y >= normalized.y - padding
    && point.y <= normalized.y + normalized.h + padding;
}

function annotationResizeHandleAtPoint(point) {
  if (a.selectedBoxIndex < 0 || a.selectedBoxIndex >= a.boxes.length) return null;
  const box = normBox(a.boxes[a.selectedBoxIndex]);
  const tolerance = annotationHitTolerance();
  const handles = {
    nw: {x: box.x, y: box.y},
    ne: {x: box.x + box.w, y: box.y},
    se: {x: box.x + box.w, y: box.y + box.h},
    sw: {x: box.x, y: box.y + box.h},
  };
  for (const [name, handle] of Object.entries(handles)) {
    if (Math.hypot(point.x - handle.x, point.y - handle.y) <= tolerance) return name;
  }
  return null;
}

function annotationBoxAtPoint(point) {
  for (let index = a.boxes.length - 1; index >= 0; index -= 1) {
    if (pointInsideAnnotationBox(point, a.boxes[index])) return index;
  }
  return -1;
}

function modelBoxAtPoint(point) {
  const candidates = [];
  a.modelBoxes.forEach((box, index) => {
    if (pointInsideAnnotationBox(point, box)) {
      const normalized = normBox(box);
      candidates.push({index, area: normalized.w * normalized.h});
    }
  });
  candidates.sort((first, second) => first.area - second.area);
  return candidates.length ? candidates[0].index : -1;
}

function annotationConflictIndex(box, ignoreIndex = -1) {
  const normalized = normBox(box);
  return a.boxes.findIndex((existing, index) => (
    index !== ignoreIndex
    && boxOverlapFraction(normalized, normBox(existing)) >= 0.35
  ));
}

function beginAnnotationPointer(point) {
  if (!a.img) return false;
  a.dragStart = {...point};
  a.dragOrigin = null;
  a.dragHandle = null;
  a.dragAdopted = false;
  a.dragModelBox = null;
  a.interactionChanged = false;
  a.preview = null;

  const handle = annotationResizeHandleAtPoint(point);
  if (handle) {
    a.dragMode = 'resize';
    a.dragHandle = handle;
    a.dragOrigin = normBox(a.boxes[a.selectedBoxIndex]);
    return true;
  }

  const annotationIndex = annotationBoxAtPoint(point);
  if (annotationIndex >= 0) {
    a.selectedBoxIndex = annotationIndex;
    a.dragMode = 'move';
    a.dragOrigin = normBox(a.boxes[annotationIndex]);
    drawAll();
    return true;
  }

  const modelIndex = modelBoxAtPoint(point);
  if (modelIndex >= 0) {
    const modelBox = a.modelBoxes[modelIndex];
    const normalized = normBox(modelBox);
    const conflictIndex = annotationConflictIndex(normalized);
    if (conflictIndex >= 0) {
      a.selectedBoxIndex = conflictIndex;
      a.dragMode = 'move';
      a.dragOrigin = normBox(a.boxes[conflictIndex]);
      drawAll();
      return true;
    }
    a.dragModelBox = modelBox;
    a.modelBoxes.splice(modelIndex, 1);
    a.boxes.push({
      x1: normalized.x,
      y1: normalized.y,
      x2: normalized.x + normalized.w,
      y2: normalized.y + normalized.h,
    });
    a.selectedBoxIndex = a.boxes.length - 1;
    a.dragMode = 'move';
    a.dragOrigin = normalized;
    a.dragAdopted = true;
    a.interactionChanged = true;
    updateBoxes();
    updateModelBoxes();
    drawAll();
    return true;
  }

  a.selectedBoxIndex = -1;
  a.dragMode = 'draw';
  a.preview = {x1: point.x, y1: point.y, x2: point.x, y2: point.y};
  drawAll();
  return true;
}

function updateAnnotationPointer(point) {
  if (!a.dragMode || !a.dragStart) return;
  if (a.dragMode === 'draw') {
    a.preview = {x1: a.dragStart.x, y1: a.dragStart.y, x2: point.x, y2: point.y};
    a.interactionChanged = (
      Math.abs(point.x - a.dragStart.x) >= 1
      || Math.abs(point.y - a.dragStart.y) >= 1
    );
    drawAll();
    return;
  }
  if (a.selectedBoxIndex < 0 || a.selectedBoxIndex >= a.boxes.length || !a.dragOrigin) return;

  const origin = a.dragOrigin;
  if (a.dragMode === 'move') {
    const x = Math.max(0, Math.min(origin.x + point.x - a.dragStart.x, a.imgW - origin.w));
    const y = Math.max(0, Math.min(origin.y + point.y - a.dragStart.y, a.imgH - origin.h));
    setAnnotationBoxFromNorm(a.selectedBoxIndex, {x, y, w: origin.w, h: origin.h});
    a.interactionChanged = a.dragAdopted
      || Math.abs(x - origin.x) >= 0.5
      || Math.abs(y - origin.y) >= 0.5;
  } else if (a.dragMode === 'resize') {
    const minimum = 4;
    let left = origin.x;
    let top = origin.y;
    let right = origin.x + origin.w;
    let bottom = origin.y + origin.h;
    if (a.dragHandle.includes('w')) left = Math.max(0, Math.min(point.x, right - minimum));
    if (a.dragHandle.includes('e')) right = Math.min(a.imgW - 1, Math.max(point.x, left + minimum));
    if (a.dragHandle.includes('n')) top = Math.max(0, Math.min(point.y, bottom - minimum));
    if (a.dragHandle.includes('s')) bottom = Math.min(a.imgH - 1, Math.max(point.y, top + minimum));
    setAnnotationBoxFromNorm(a.selectedBoxIndex, {
      x: left,
      y: top,
      w: right - left,
      h: bottom - top,
    });
    a.interactionChanged = (
      Math.abs(left - origin.x) >= 0.5
      || Math.abs(top - origin.y) >= 0.5
      || Math.abs(right - (origin.x + origin.w)) >= 0.5
      || Math.abs(bottom - (origin.y + origin.h)) >= 0.5
    );
  }
  drawAll();
}

function cancelAnnotationInteraction() {
  if (!a.dragMode) return;
  if (a.dragAdopted && a.selectedBoxIndex >= 0 && a.selectedBoxIndex < a.boxes.length) {
    a.boxes.splice(a.selectedBoxIndex, 1);
    if (a.dragModelBox) a.modelBoxes.push(a.dragModelBox);
    a.selectedBoxIndex = -1;
  } else if (a.dragOrigin && a.selectedBoxIndex >= 0 && a.selectedBoxIndex < a.boxes.length) {
    setAnnotationBoxFromNorm(a.selectedBoxIndex, a.dragOrigin);
  }
  resetAnnotationInteraction();
  updateBoxes();
  updateModelBoxes();
  drawAll();
}

function finishAnnotationPointer(point) {
  if (!a.dragMode) return;
  const mode = a.dragMode;
  let changed = a.interactionChanged;
  let message = 'Box actualizado.';

  if (mode === 'draw') {
    const candidate = {x1: a.dragStart.x, y1: a.dragStart.y, x2: point.x, y2: point.y};
    const normalized = normBox(candidate);
    changed = false;
    if (normalized.w >= 4 && normalized.h >= 4) {
      const conflictIndex = annotationConflictIndex(normalized);
      if (conflictIndex >= 0) {
        a.selectedBoxIndex = conflictIndex;
        status(`El nuevo box ocupa el mismo espacio que la anotacion #${conflictIndex + 1}.`, 'err');
      } else {
        a.boxes.push(candidate);
        a.selectedBoxIndex = a.boxes.length - 1;
        changed = true;
        message = 'Box agregado.';
      }
    }
  } else {
    updateAnnotationPointer(point);
    changed = a.interactionChanged;
    if (a.selectedBoxIndex >= 0 && a.selectedBoxIndex < a.boxes.length) {
      const conflictIndex = annotationConflictIndex(a.boxes[a.selectedBoxIndex], a.selectedBoxIndex);
      if (conflictIndex >= 0) {
        setAnnotationBoxFromNorm(a.selectedBoxIndex, a.dragOrigin);
        changed = a.dragAdopted;
        status(`El box no puede ocupar el espacio de la anotacion #${conflictIndex + 1}.`, 'err');
      } else if (a.dragAdopted) {
        changed = true;
        message = 'Deteccion del modelo agregada como anotacion.';
      } else if (mode === 'move') {
        message = 'Box movido.';
      } else {
        message = 'Box redimensionado.';
      }
    }
  }

  resetAnnotationInteraction();
  if (changed) {
    annotationContentChanged(message);
  } else {
    updateBoxes();
    updateModelBoxes();
    drawAll();
  }
}

function annotationCursor(point) {
  const handle = annotationResizeHandleAtPoint(point);
  if (handle === 'nw' || handle === 'se') return 'nwse-resize';
  if (handle === 'ne' || handle === 'sw') return 'nesw-resize';
  if (annotationBoxAtPoint(point) >= 0) return 'move';
  if (modelBoxAtPoint(point) >= 0) return 'copy';
  return 'crosshair';
}

function drawAll() { drawHomography(); drawWarp(); drawAnnotate(); drawMeasure(); drawPlayer(); }

function roiSideMidpoints() {
  if (h.expandedPoints.length !== 4) return {};
  const pts = h.expandedPoints.map(p => imageToDisplay(h, p.x, p.y));
  return {
    top: {x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2},
    right: {x: (pts[1].x + pts[2].x) / 2, y: (pts[1].y + pts[2].y) / 2},
    bottom: {x: (pts[2].x + pts[3].x) / 2, y: (pts[2].y + pts[3].y) / 2},
    left: {x: (pts[3].x + pts[0].x) / 2, y: (pts[3].y + pts[0].y) / 2},
  };
}

function distToSegment(px, py, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  if (!len2) return Math.hypot(px - a.x, py - a.y);
  const t = Math.max(0, Math.min(1, ((px - a.x) * dx + (py - a.y) * dy) / len2));
  const x = a.x + t * dx, y = a.y + t * dy;
  return Math.hypot(px - x, py - y);
}

function nearestMeasureSegmentPart(cx, cy) {
  if (m.mode !== 'segment' || m.pending) return null;
  const candidates = m.segments.map((segment, index) => ({
    index,
    start: imageToDisplay(m, segment.x1, segment.y1),
    end: imageToDisplay(m, segment.x2, segment.y2),
  })).reverse();
  for (const candidate of candidates) {
    if (Math.hypot(cx - candidate.start.x, cy - candidate.start.y) <= 14) {
      return {index: candidate.index, part: 'start'};
    }
    if (Math.hypot(cx - candidate.end.x, cy - candidate.end.y) <= 14) {
      return {index: candidate.index, part: 'end'};
    }
  }
  for (const candidate of candidates) {
    if (distToSegment(cx, cy, candidate.start, candidate.end) <= 10) {
      return {index: candidate.index, part: 'line'};
    }
  }
  return null;
}

function beginMeasureSegmentDrag(hit, imagePoint) {
  const segment = m.segments[hit.index];
  if (!segment) return false;
  m.selectedSegment = hit.index;
  m.draggingSegment = {
    index: hit.index,
    part: hit.part,
    anchor: {...imagePoint},
    original: {
      x1: Number(segment.x1), y1: Number(segment.y1),
      x2: Number(segment.x2), y2: Number(segment.y2),
    },
  };
  return true;
}

function updateMeasureSegmentDrag(imagePoint) {
  const drag = m.draggingSegment;
  if (!drag) return;
  const segment = m.segments[drag.index];
  if (!segment) {
    m.draggingSegment = null;
    return;
  }
  const x = Math.max(0, Math.min(m.imgW - 1, imagePoint.x));
  const y = Math.max(0, Math.min(m.imgH - 1, imagePoint.y));
  if (drag.part === 'start') {
    segment.x1 = x; segment.y1 = y;
  } else if (drag.part === 'end') {
    segment.x2 = x; segment.y2 = y;
  } else {
    const original = drag.original;
    const minDx = -Math.min(original.x1, original.x2);
    const maxDx = (m.imgW - 1) - Math.max(original.x1, original.x2);
    const minDy = -Math.min(original.y1, original.y2);
    const maxDy = (m.imgH - 1) - Math.max(original.y1, original.y2);
    const dx = Math.max(minDx, Math.min(maxDx, imagePoint.x - drag.anchor.x));
    const dy = Math.max(minDy, Math.min(maxDy, imagePoint.y - drag.anchor.y));
    segment.x1 = original.x1 + dx; segment.y1 = original.y1 + dy;
    segment.x2 = original.x2 + dx; segment.y2 = original.y2 + dy;
  }
  segment.px = Math.hypot(segment.x2 - segment.x1, segment.y2 - segment.y1);
  segment.inch_per_px = segment.px > 0 ? Number(segment.inches) / segment.px : null;
}

function nearestWorkRoiSide(cx, cy) {
  if (h.expandedPoints.length !== 4) return null;
  const pts = h.expandedPoints.map(p => imageToDisplay(h, p.x, p.y));
  const sides = [
    ['top', pts[0], pts[1]],
    ['right', pts[1], pts[2]],
    ['bottom', pts[2], pts[3]],
    ['left', pts[3], pts[0]],
  ];
  let bestSide = null;
  let bestDist = 16;
  for (const [side, a, b] of sides) {
    const dist = distToSegment(cx, cy, a, b);
    if (dist < bestDist) {
      bestDist = dist;
      bestSide = side;
    }
  }
  return bestSide;
}

function nearestHomographyPoint(cx, cy) {
  let bestIndex = null;
  let bestDistance = 18;
  h.points.forEach((point, index) => {
    const displayPoint = imageToDisplay(h, point.x, point.y);
    const distance = Math.hypot(cx - displayPoint.x, cy - displayPoint.y);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function applyHomographyPoint(matrix, point) {
  if (!matrix) return null;
  const den = matrix[2][0] * point.x + matrix[2][1] * point.y + matrix[2][2];
  if (!den) return null;
  return {
    x: (matrix[0][0] * point.x + matrix[0][1] * point.y + matrix[0][2]) / den,
    y: (matrix[1][0] * point.x + matrix[1][1] * point.y + matrix[1][2]) / den,
  };
}

function inverse3x3(m) {
  const a = m[0][0], b = m[0][1], c = m[0][2];
  const d = m[1][0], e = m[1][1], f = m[1][2];
  const g = m[2][0], h2 = m[2][1], i = m[2][2];
  const A = e * i - f * h2, B = c * h2 - b * i, C = b * f - c * e;
  const D = f * g - d * i, E = a * i - c * g, F = c * d - a * f;
  const G = d * h2 - e * g, H = b * g - a * h2, I = a * e - b * d;
  const det = a * A + b * D + c * G;
  if (Math.abs(det) < 1e-12) return null;
  return [[A / det, B / det, C / det], [D / det, E / det, F / det], [G / det, H / det, I / det]];
}

function sourcePointsFromMargins() {
  if (!h.baseMatrix || !h.baseSize) return h.expandedPoints;
  const inv = inverse3x3(h.baseMatrix);
  if (!inv) return h.expandedPoints;
  const m = h.roiMargins;
  const rect = [
    {x: -m.left, y: -m.top},
    {x: h.baseSize.width - 1 + m.right, y: -m.top},
    {x: h.baseSize.width - 1 + m.right, y: h.baseSize.height - 1 + m.bottom},
    {x: -m.left, y: h.baseSize.height - 1 + m.bottom},
  ];
  return rect.map(p => applyHomographyPoint(inv, p));
}

function updateRoiSideFromImagePoint(side, imagePoint) {
  if (!h.baseMatrix || !h.baseSize) return;
  const q = applyHomographyPoint(h.baseMatrix, imagePoint);
  if (!q) return;
  if (side === 'left') h.roiMargins.left = Math.max(0, -q.x);
  if (side === 'right') h.roiMargins.right = Math.max(0, q.x - (h.baseSize.width - 1));
  if (side === 'top') h.roiMargins.top = Math.max(0, -q.y);
  if (side === 'bottom') h.roiMargins.bottom = Math.max(0, q.y - (h.baseSize.height - 1));
  h.expandedPoints = sourcePointsFromMargins();
}

function installPanZoom(state, zoomInput, zoomLabel, hud, onClick) {
  state.wrap.addEventListener('click', ev => {
    if (!state.img || state === a || ev.button !== 0 || state.didDrag) return;
    const rect = state.canvas.getBoundingClientRect();
    const cx = ev.clientX - rect.left;
    const cy = ev.clientY - rect.top;
    if (!isInsideCanvasImage(state, cx, cy)) return;
    const p = displayToImage(state, cx, cy);
    onClick(p);
  });
  state.wrap.addEventListener('mousedown', ev => {
    if (state === a && ev.button === 0) {
      const rect = state.canvas.getBoundingClientRect();
      const cx = Math.max(0, Math.min(ev.clientX - rect.left, state.canvas.width - 1));
      const cy = Math.max(0, Math.min(ev.clientY - rect.top, state.canvas.height - 1));
      const point = displayToImage(state, cx, cy);
      if (beginAnnotationPointer(point)) {
        ev.preventDefault();
        state.didDrag = true;
        state.wrap.style.cursor = a.dragMode === 'move' ? 'move' : 'crosshair';
        return;
      }
    }
    if (state === h && ev.button === 0 && h.points.length) {
      const rect = state.canvas.getBoundingClientRect();
      const cx = Math.max(0, Math.min(ev.clientX - rect.left, state.canvas.width - 1));
      const cy = Math.max(0, Math.min(ev.clientY - rect.top, state.canvas.height - 1));
      const pointIndex = nearestHomographyPoint(cx, cy);
      if (pointIndex !== null) {
        ev.preventDefault();
        h.draggingPointIndex = pointIndex;
        h.dragPointOrigin = {...h.points[pointIndex]};
        state.didDrag = false;
        state.wrap.style.cursor = 'grabbing';
        drawAll();
        return;
      }
    }
    if (state === h && ev.button === 0 && h.expandedPoints.length === 4) {
      const rect = state.canvas.getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;
      const side = nearestWorkRoiSide(cx, cy);
      if (side) {
        ev.preventDefault();
        h.draggingRoiSide = side;
        h.roiManual = true;
        state.didDrag = false;
        state.wrap.style.cursor = 'grabbing';
        return;
      }
    }
    if (state === m && ev.button === 0 && m.mode === 'segment' && !m.pending) {
      const rect = state.canvas.getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;
      const hit = nearestMeasureSegmentPart(cx, cy);
      if (hit) {
        ev.preventDefault();
        const imagePoint = displayToImage(m, cx, cy);
        if (beginMeasureSegmentDrag(hit, imagePoint)) {
          state.didDrag = false;
          state.wrap.style.cursor = 'grabbing';
          drawAll();
          return;
        }
      }
    }
    if (state === m && ev.button === 0 && m.mode === 'reference' && m.referenceY !== null) {
      const rect = state.canvas.getBoundingClientRect();
      const cy = ev.clientY - rect.top;
      const ref = imageToDisplay(m, 0, m.referenceY);
      if (Math.abs(cy - ref.y) <= 16) {
        ev.preventDefault();
        m.draggingReference = true;
        state.didDrag = false;
        state.wrap.style.cursor = 'grabbing';
        return;
      }
    }
    if (ev.button === 1 || ev.button === 2) {
      ev.preventDefault();
      state.panning = true; state.didDrag = false;
      state.panAnchor = {x: ev.clientX, y: ev.clientY};
      state.panStart = {x: state.panX, y: state.panY};
      state.wrap.style.cursor = 'grabbing';
    }
  });
  window.addEventListener('mouseup', ev => {
    if (state === a && a.dragMode) {
      const rect = state.canvas.getBoundingClientRect();
      const cx = Math.max(0, Math.min(ev.clientX - rect.left, state.canvas.width - 1));
      const cy = Math.max(0, Math.min(ev.clientY - rect.top, state.canvas.height - 1));
      const point = displayToImage(state, cx, cy);
      finishAnnotationPointer(point);
      state.didDrag = true;
      state.wrap.style.cursor = annotationCursor(point);
      setTimeout(() => { state.didDrag = false; }, 0);
      return;
    }
    if (state === h && h.draggingPointIndex !== null) {
      const pointIndex = h.draggingPointIndex;
      h.draggingPointIndex = null;
      h.dragPointOrigin = null;
      state.didDrag = true;
      state.wrap.style.cursor = 'crosshair';
      updatePoints();
      status(`Punto ${pointIndex + 1} actualizado; recalculando homografia.`, 'ok');
      setTimeout(() => { state.didDrag = false; }, 0);
      return;
    }
    if (state === h && h.draggingRoiSide) {
      h.draggingRoiSide = null;
      state.didDrag = true;
      state.wrap.style.cursor = 'crosshair';
      updatePoints();
      status('Lado del ROI de trabajo actualizado.', 'ok');
      setTimeout(() => { state.didDrag = false; }, 0);
      return;
    }
    if (state === m && m.draggingSegment) {
      const index = m.draggingSegment.index;
      m.draggingSegment = null;
      state.didDrag = true;
      state.wrap.style.cursor = 'crosshair';
      updateMeasureInfo();
      const segment = m.segments[index];
      status(`Medicion #${index + 1} actualizada: ${Number(segment?.px || 0).toFixed(1)} px.`, 'ok');
      setTimeout(() => { state.didDrag = false; }, 0);
      return;
    }
    if (state === m && m.draggingReference) {
      m.draggingReference = false;
      state.didDrag = true;
      state.wrap.style.cursor = 'crosshair';
      updateMeasureInfo();
      status('Linea Y actualizada.', 'ok');
      setTimeout(() => { state.didDrag = false; }, 0);
      return;
    }
    state.panning = false; state.panAnchor = null; state.wrap.style.cursor = 'crosshair';
    setTimeout(() => { state.didDrag = false; }, 0);
  });
  state.wrap.addEventListener('mousemove', ev => {
    if (!state.img) return;
    const rect = state.canvas.getBoundingClientRect();
    const cx = Math.max(0, Math.min(ev.clientX - rect.left, state.canvas.width - 1));
    const cy = Math.max(0, Math.min(ev.clientY - rect.top, state.canvas.height - 1));
    const imgPoint = displayToImage(state, cx, cy);
    hud.textContent = `x: ${Math.round(imgPoint.x)} y: ${Math.round(imgPoint.y)}`;
    if (state === a) {
      if (a.dragMode) {
        updateAnnotationPointer(imgPoint);
        state.didDrag = true;
        if (a.dragMode === 'move') state.wrap.style.cursor = 'move';
        if (a.dragMode === 'resize') state.wrap.style.cursor = annotationCursor(imgPoint);
        return;
      }
      if (!state.panning) state.wrap.style.cursor = annotationCursor(imgPoint);
    }
    if (state === h && h.draggingPointIndex !== null) {
      h.points[h.draggingPointIndex] = imgPoint;
      h.expandedPoints = [];
      h.warpImg = null;
      state.didDrag = true;
      drawAll();
      return;
    }
    if (state === h && h.draggingRoiSide) {
      updateRoiSideFromImagePoint(h.draggingRoiSide, imgPoint);
      h.roiManual = true;
      state.didDrag = true;
      drawAll();
      return;
    }
    if (state === m) {
      if (m.draggingSegment) {
        updateMeasureSegmentDrag(imgPoint);
        state.didDrag = true;
        updateMeasureInfo();
      } else if (m.draggingReference) {
        m.referenceY = imgPoint.y;
        state.didDrag = true;
        updateMeasureInfo();
      } else if (m.pending) {
        m.preview = imgPoint;
      } else if (m.mode === 'segment' && !state.panning) {
        state.wrap.style.cursor = nearestMeasureSegmentPart(cx, cy) ? 'grab' : 'crosshair';
      }
    }
    if (!state.panning && state === h) {
      if (nearestHomographyPoint(cx, cy) !== null) {
        state.wrap.style.cursor = 'grab';
      } else if (nearestWorkRoiSide(cx, cy)) {
        state.wrap.style.cursor = 'grab';
      } else {
        state.wrap.style.cursor = 'crosshair';
      }
    }
    if (!state.panning && state === m) {
      if (m.mode === 'segment' && nearestMeasureSegmentPart(cx, cy)) {
        state.wrap.style.cursor = 'grab';
      } else {
        state.wrap.style.cursor = 'crosshair';
      }
    }
    if (state === p && p.rulerActive && p.rulerStart && !p.rulerEnd && !state.panning) {
      p.rulerPreview = constrainPlayerRulerPoint(p.rulerStart, imgPoint);
      updatePlayerRulerInfo();
    }
    if (state.panning && state.panAnchor) {
      const dx = ev.clientX - state.panAnchor.x, dy = ev.clientY - state.panAnchor.y;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) state.didDrag = true;
      const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
      const imageRect = canvasImageRect(state);
      state.panX = state.panStart.x - dx * vw / imageRect.width;
      state.panY = state.panStart.y - dy * vh / imageRect.height;
      clamp(state);
    }
    drawAll();
  });
  state.wrap.addEventListener('contextmenu', ev => ev.preventDefault());
  state.wrap.addEventListener('wheel', ev => {
    ev.preventDefault();
    if (!state.img) return;
    const rect = state.canvas.getBoundingClientRect();
    const cx = ev.clientX - rect.left, cy = ev.clientY - rect.top;
    const anchor = displayToImage(state, cx, cy);
    state.zoom = Math.max(1, Math.min(state.zoom * (ev.deltaY < 0 ? 1.15 : 1 / 1.15), 20));
    const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
    const imageRect = canvasImageRect(state);
    state.panX = anchor.x - ((cx - imageRect.x) / imageRect.width) * vw;
    state.panY = anchor.y - ((cy - imageRect.y) / imageRect.height) * vh;
    clamp(state);
    zoomInput.value = state.zoom; zoomLabel.textContent = state.zoom.toFixed(1) + 'x';
    drawAll();
  }, {passive: false});
  zoomInput.addEventListener('input', () => {
    const center = displayToImage(state, state.canvas.width / 2, state.canvas.height / 2);
    state.zoom = parseFloat(zoomInput.value);
    const vw = state.imgW / state.zoom, vh = state.imgH / state.zoom;
    state.panX = center.x - vw / 2; state.panY = center.y - vh / 2;
    clamp(state);
    zoomLabel.textContent = state.zoom.toFixed(1) + 'x';
    drawAll();
  });
}

installPanZoom(h, document.getElementById('h-zoom'), document.getElementById('h-zoom-label'), document.getElementById('h-hud'), p => {
  if (h.points.length >= 4) return;
  h.points.push(p); updatePoints(); drawAll();
});
document.getElementById('h-expand').addEventListener('input', () => {
  h.expandPct = parseFloat(document.getElementById('h-expand').value) || 0;
  document.getElementById('h-expand-label').textContent = Math.round(h.expandPct) + '%';
  h.roiManual = false;
  h.expandedPoints = [];
  h.roiMargins = {left: 0, right: 0, top: 0, bottom: 0};
  if (h.points.length === 4) requestWarp();
  drawAll();
});
installPanZoom(a, document.getElementById('a-zoom'), document.getElementById('a-zoom-label'), document.getElementById('a-hud'), () => {});
installPanZoom(m, document.getElementById('m-zoom'), document.getElementById('m-zoom-label'), document.getElementById('m-hud'), p => {
  measureClick(p);
});
installPanZoom(p, document.getElementById('p-zoom'), document.getElementById('p-zoom-label'), document.getElementById('p-hud'), point => {
  playerRulerClick(point);
});

document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT') return;
  const annotate = document.getElementById('annotate-view').classList.contains('active');
  const measure = document.getElementById('measure-view').classList.contains('active');
  const player = document.getElementById('player-view').classList.contains('active');
  if (annotate) {
    if (ev.key === 'Escape') { cancelAnnotationInteraction(); return; }
    if ((ev.key === 'Delete' || ev.key === 'Backspace') && a.selectedBoxIndex >= 0) {
      ev.preventDefault();
      deleteBox(a.selectedBoxIndex);
      return;
    }
    if (ev.key === 's' || ev.key === 'S') saveFrame();
    if (ev.key === 'z' || ev.key === 'Z') undoBox();
    if (ev.key === 'r' || ev.key === 'R') clearBoxes();
    if (ev.key === 'ArrowLeft') stepAnnotate(-1);
    if (ev.key === 'ArrowRight') stepAnnotate(1);
    if (ev.key === 'ArrowUp') stepAnnotate(-30);
    if (ev.key === 'ArrowDown') stepAnnotate(30);
  } else if (measure) {
    if (ev.key === 's' || ev.key === 'S') saveMeasureCalibration();
    if (ev.key === 'z' || ev.key === 'Z') undoMeasureSegment();
    if (ev.key === 'r' || ev.key === 'R') setMeasureMode('reference');
    if (ev.key === 'm' || ev.key === 'M') setMeasureMode('segment');
    if (ev.key === 'e' || ev.key === 'E') setMeasureMode('exclusion');
    if (ev.key === 'ArrowLeft') stepMeasure(-1);
    if (ev.key === 'ArrowRight') stepMeasure(1);
    if (ev.key === 'ArrowUp') stepMeasure(-30);
    if (ev.key === 'ArrowDown') stepMeasure(30);
  } else if (player) {
    if (ev.key === ' ' || ev.key === 'p' || ev.key === 'P') { ev.preventDefault(); togglePlayerPlay(); return; }
    if (ev.key === 'x' || ev.key === 'X') togglePlayerSpeed();
    if (ev.key === 'ArrowLeft') stepPlayer(-1);
    if (ev.key === 'ArrowRight') stepPlayer(1);
    if (ev.key === 'ArrowUp') stepPlayer(-30);
    if (ev.key === 'ArrowDown') stepPlayer(30);
  } else {
    if (ev.key === 's' || ev.key === 'S') saveHomography();
    if (ev.key === 'z' || ev.key === 'Z') undoPoint();
    if (ev.key === 'r' || ev.key === 'R') resetPoints();
    if (ev.key === 'ArrowLeft') stepHomography(-1);
    if (ev.key === 'ArrowRight') stepHomography(1);
    if (ev.key === 'ArrowUp') stepHomography(-30);
    if (ev.key === 'ArrowDown') stepHomography(30);
  }
});

window.addEventListener('resize', () => { fitAll(); drawAll(); });

(async function init() {
  fitAll();
  let projectMeta = null;
  try { projectMeta = await loadMeta(); } catch (e) { status(String(e), 'err'); }
  await loadHomographyFrame();
  if (projectMeta?.homography_exists) await loadSavedHomography();
})();
