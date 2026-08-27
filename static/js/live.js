const byId = (id) => document.getElementById(id);

let lastPlcTriggerKey = null;
let plcSignalHighlightUntil = 0;
let measurementHighlightUntil = 0;
let pendingPlcPresentation = null;
let pendingDiagramResult = null;
let exactMeasurementLoader = null;
let exactMeasurementFrameKey = "";
let exactMeasurementVisibleUntil = 0;
let analysisRequestInFlight = false;
let statusRequestInFlight = false;
let lastDiagramFrameIndex = null;
let streamRetryTimer = null;
let streamRetryDelayMs = 1000;
let consecutiveStatusFailures = 0;
let lastLiveVideoProgressAt = Date.now();
let lastLiveVideoCurrentTime = -1;

const STATUS_FAILURE_THRESHOLD = 3;
const PLC_SIGNAL_HIGHLIGHT_MS = 4000;
const MEASUREMENT_HIGHLIGHT_MS = 900;
const LIVE_TARGET_LATENCY_SECONDS = 0.18;
const LIVE_SOFT_LATENCY_SECONDS = 0.35;
const LIVE_HARD_LATENCY_SECONDS = 0.9;
const LIVE_STALL_TIMEOUT_MS = 3500;

function setPill(element, text, tone) {
  element.textContent = text;
  element.className = `pill ${tone || ""}`.trim();
}

function compactMeasurement(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";

  let total = Math.round(Math.abs(numeric) * 16);
  const sign = numeric < 0 ? "-" : "";
  const feet = Math.floor(total / 192);
  total -= feet * 192;
  const inches = Math.floor(total / 16);
  let numerator = total % 16;
  let denominator = 16;
  while (numerator && numerator % 2 === 0) {
    numerator /= 2;
    denominator /= 2;
  }
  const inchText = numerator
    ? `${inches} ${numerator}/${denominator}`
    : `${inches}`;
  return `${sign}${feet}' ${inchText}"`;
}

function updateDiagram(result) {
  const pieces = Array.isArray(result?.pieces) ? result.pieces : [];
  const rectWidth = Math.max(1, Number(result?.rectified_width || 1));
  const rectHeight = Math.max(1, Number(result?.rectified_height || 1));
  const mapY = (value) => (
    Math.max(66, Math.min(232, 58 + (Number(value) / rectHeight) * 182))
  );
  const referenceValue = result?.calibration?.reference_y;
  const referenceY = Number.isFinite(Number(referenceValue))
    ? mapY(referenceValue)
    : 156;

  byId("diagram-reference").setAttribute("y1", referenceY);
  byId("diagram-reference").setAttribute("y2", referenceY);
  byId("diagram-reference-label").setAttribute(
    "y",
    Math.min(252, referenceY + 22),
  );

  byId("diagram-pieces").innerHTML = pieces.map((piece, index) => {
    const box = piece.box || {};
    const lineY = piece?.sobel?.line?.y;
    if (!Number.isFinite(Number(lineY))) return "";

    const centerRatio = (
      Number(box.x || 0) + Number(box.w || 0) / 2
    ) / rectWidth;
    const x = 92 + Math.max(0, Math.min(1, centerRatio)) * 576;
    const width = Math.max(
      12,
      Math.min(34, (Number(box.w || 0) / rectWidth) * 576),
    );
    const frontY = mapY(lineY);
    const valid = Boolean(piece.valid);
    const color = valid ? "#16845a" : "#c5782d";
    const label = compactMeasurement(piece.measurement?.measurement_in);

    return `
      <g>
        <rect x="${x - width / 2}" y="68" width="${width}" height="${Math.max(8, frontY - 68)}" rx="3"
          fill="url(#live-steel)" stroke="#536066" stroke-width="1"></rect>
        <line x1="${x - width / 2 - 2}" x2="${x + width / 2 + 2}" y1="${frontY}" y2="${frontY}"
          stroke="${color}" stroke-width="5" stroke-linecap="round"></line>
        <text x="${x}" y="${Math.min(254, frontY + 17 + (index % 2) * 14)}" text-anchor="middle"
          fill="${color}" font-size="11" font-weight="900">P${Number(piece.piece_id)} ${label}</text>
      </g>`;
  }).join("");

  const summary = result?.measurement_summary || {};
  byId("diagram-summary").textContent = pieces.length
    ? `${Number(summary.valid_count || 0)} / ${pieces.length} valid piece measurements`
    : "No pieces detected";
}

function signalTime(trigger) {
  const raw = trigger?.event_source_timestamp || trigger?.read_utc;
  if (!raw) return "";

  const parsed = new Date(raw);
  return Number.isNaN(parsed.getTime())
    ? String(raw)
    : parsed.toLocaleTimeString(
      [],
      { hour: "2-digit", minute: "2-digit", second: "2-digit" },
    );
}

function plcTriggerKey(plc, trigger) {
  if (!trigger) return "";
  if (trigger.event_key) return String(trigger.event_key);
  return [
    trigger.event_source_timestamp || "",
    trigger.read_utc || "",
    Number(plc.events_found || 0),
  ].join("|");
}

function requestExactMeasurementFrame(eventKey) {
  const pending = pendingPlcPresentation;
  if (!pending || pending.key !== eventKey || pending.imageRequested) return;

  pending.imageRequested = true;
  const loader = new Image();
  exactMeasurementLoader = loader;
  loader.addEventListener("load", () => {
    if (
      exactMeasurementLoader !== loader
      || !pendingPlcPresentation
      || pendingPlcPresentation.key !== eventKey
    ) return;

    const frame = byId("plc-measurement-frame");
    frame.src = loader.src;
    frame.dataset.eventKey = eventKey;
    pendingPlcPresentation.exactFrameReady = true;
    presentPendingPlcEvent();
  });
  loader.addEventListener("error", () => {
    if (pendingPlcPresentation?.key === eventKey) {
      pendingPlcPresentation.imageRequested = false;
    }
  });
  loader.src = `/api/live/measurement.jpg?event_key=${
    encodeURIComponent(eventKey)
  }&t=${Date.now()}`;
}

function presentPendingPlcEvent() {
  if (
    pendingPlcPresentation
    && pendingPlcPresentation.resultReady
    && pendingPlcPresentation.exactFrameReady
  ) {
    const now = Date.now();
    plcSignalHighlightUntil = now + PLC_SIGNAL_HIGHLIGHT_MS;
    measurementHighlightUntil = now + MEASUREMENT_HIGHLIGHT_MS;
    exactMeasurementFrameKey = pendingPlcPresentation.key;
    exactMeasurementVisibleUntil = measurementHighlightUntil;
    pendingPlcPresentation = null;
  }

  if (
    pendingDiagramResult
    && !pendingPlcPresentation
  ) {
    lastDiagramFrameIndex = pendingDiagramResult.frame_index;
    updateDiagram(pendingDiagramResult);
    pendingDiagramResult = null;
  }

  const now = Date.now();
  const measurementFrame = byId("plc-measurement-frame");
  const exactFrameVisible = (
    now < exactMeasurementVisibleUntil
    && measurementFrame.dataset.eventKey === exactMeasurementFrameKey
  );
  measurementFrame.classList.toggle("visible", exactFrameVisible);
  byId("original-stage").classList.toggle(
    "measurement-taken",
    now < measurementHighlightUntil && exactFrameVisible,
  );
}

function updatePlcSignal(plc) {
  const signal = byId("plc-signal");
  const text = byId("plc-signal-text");
  const trigger = plc.last_trigger;

  if (!plc.connected) {
    plcSignalHighlightUntil = 0;
    measurementHighlightUntil = 0;
    exactMeasurementVisibleUntil = 0;
    pendingPlcPresentation = null;
    signal.className = "plc-signal offline";
    text.textContent = "PLC disconnected";
    signal.title = "";
  } else if (!trigger) {
    plcSignalHighlightUntil = 0;
    measurementHighlightUntil = 0;
    exactMeasurementVisibleUntil = 0;
    pendingPlcPresentation = null;
    signal.className = "plc-signal";
    text.textContent = "Waiting for PLC signal";
    signal.title = "";
  } else {
    const triggerKey = plcTriggerKey(plc, trigger);
    const receivedNow = (
      triggerKey !== lastPlcTriggerKey
      && (lastPlcTriggerKey !== null || Boolean(plc.signal_recent))
    );
    if (receivedNow) {
      pendingPlcPresentation = {
        key: triggerKey,
        resultReady: false,
        exactFrameReady: false,
        imageRequested: false,
      };
      pendingDiagramResult = null;
    }

    presentPendingPlcEvent();

    const highlighting = Date.now() < plcSignalHighlightUntil;
    const time = signalTime(trigger);
    signal.className = `plc-signal ${highlighting ? "received" : "seen"}`;
    text.textContent = `${
      highlighting ? "PLC signal received" : "Last PLC Cut Signal"
    }${time ? ` | ${time}` : ""}`;
    signal.title = (
      trigger.event_source_timestamp || trigger.read_utc || ""
    );
    lastPlcTriggerKey = triggerKey;
  }
}

async function refreshAnalysis() {
  if (analysisRequestInFlight) return;
  analysisRequestInFlight = true;

  try {
    const response = await fetch(
      "/api/live/frame?metadata=1",
      { cache: "no-store" },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "frame error");

    const result = data.result;
    updatePlcSignal(data.plc || {});
    if (
      result
      && pendingPlcPresentation
      && result.plc_event_key === pendingPlcPresentation.key
    ) {
      pendingPlcPresentation.resultReady = true;
      pendingDiagramResult = result;
      requestExactMeasurementFrame(pendingPlcPresentation.key);
    } else if (result && result.frame_index !== lastDiagramFrameIndex) {
      lastDiagramFrameIndex = result.frame_index;
      updateDiagram(result);
    }
    presentPendingPlcEvent();
  } catch {
    // Preserve the last valid UI state during a transient metadata failure.
  } finally {
    analysisRequestInFlight = false;
  }
}

async function refreshStatus() {
  if (statusRequestInFlight) return;
  statusRequestInFlight = true;

  try {
    const response = await fetch(
      "/api/live/status?summary=1",
      { cache: "no-store" },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "status error");

    const camera = data.camera || {};
    const processor = data.processor || {};
    const database = data.database || {};
    consecutiveStatusFailures = 0;
    updatePlcSignal(data.plc || {});
    const healthy = (
      camera.connected
      && processor.ok
      && (!database.enabled || database.ok)
    );
    const label = database.enabled
      ? (healthy ? "live" : "check status")
      : "simulation";
    setPill(
      byId("top-state"),
      label,
      healthy && database.enabled ? "ok" : "warn",
    );
  } catch {
    consecutiveStatusFailures += 1;
    if (consecutiveStatusFailures >= STATUS_FAILURE_THRESHOLD) {
      setPill(byId("top-state"), "disconnected", "err");
    }
  } finally {
    statusRequestInFlight = false;
  }
}

const liveVideo = byId("live-video");
const liveStage = byId("original-stage");
const liveVideoState = byId("live-video-state");

function keepLiveVideoNearEdge() {
  if (!liveVideo.buffered.length || liveVideo.readyState < 2) return;
  const liveEdge = liveVideo.buffered.end(liveVideo.buffered.length - 1);
  const latency = liveEdge - liveVideo.currentTime;

  if (latency > LIVE_HARD_LATENCY_SECONDS) {
    liveVideo.currentTime = Math.max(
      0,
      liveEdge - LIVE_TARGET_LATENCY_SECONDS,
    );
    liveVideo.playbackRate = 1;
  } else if (latency > LIVE_SOFT_LATENCY_SECONDS) {
    liveVideo.playbackRate = 1.12;
  } else if (liveVideo.playbackRate !== 1) {
    liveVideo.playbackRate = 1;
  }
}

function reconnectLiveVideo() {
  if (streamRetryTimer !== null) return;
  streamRetryTimer = setTimeout(() => {
    streamRetryTimer = null;
    liveStage.classList.remove("stream-ready");
    liveVideoState.textContent = "Reconnecting...";
    lastLiveVideoProgressAt = Date.now();
    lastLiveVideoCurrentTime = -1;
    liveVideo.src = `/api/live/stream.mp4?retry=${Date.now()}`;
    liveVideo.load();
    liveVideo.play().catch(() => {});
    streamRetryDelayMs = Math.min(streamRetryDelayMs * 2, 8000);
  }, streamRetryDelayMs);
}

function noteLiveVideoProgress() {
  const currentTime = Number(liveVideo.currentTime || 0);
  if (
    currentTime > lastLiveVideoCurrentTime + 0.01
    || currentTime < lastLiveVideoCurrentTime
  ) {
    lastLiveVideoCurrentTime = currentTime;
    lastLiveVideoProgressAt = Date.now();
  }
}

function monitorLiveVideo() {
  noteLiveVideoProgress();
  if (
    !document.hidden
    && liveStage.classList.contains("stream-ready")
    && Date.now() - lastLiveVideoProgressAt > LIVE_STALL_TIMEOUT_MS
  ) reconnectLiveVideo();
}

liveVideo.addEventListener("playing", () => {
  clearTimeout(streamRetryTimer);
  streamRetryTimer = null;
  streamRetryDelayMs = 1000;
  liveStage.classList.add("stream-ready");
  noteLiveVideoProgress();
  keepLiveVideoNearEdge();
});
liveVideo.addEventListener("timeupdate", noteLiveVideoProgress);
liveVideo.addEventListener("stalled", reconnectLiveVideo);
liveVideo.addEventListener("error", reconnectLiveVideo);

setInterval(keepLiveVideoNearEdge, 500);
setInterval(monitorLiveVideo, 1000);
setInterval(refreshAnalysis, 100);
setInterval(refreshStatus, 1000);
refreshAnalysis();
refreshStatus();
