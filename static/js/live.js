const byId = (id) => document.getElementById(id);

let lastPlcTriggerKey = null;
let plcSignalHighlightUntil = 0;
let measurementHighlightUntil = 0;
let pendingPlcPresentation = null;
let pendingDiagramResult = null;
let analysisRequestInFlight = false;
let statusRequestInFlight = false;
let lastDiagramFrameIndex = null;
let streamRetryTimer = null;
let streamRetryDelayMs = 1000;
let consecutiveStatusFailures = 0;
let lastLiveVideoProgressAt = Date.now();
let lastLiveVideoCurrentTime = -1;
let liveStreamGeneration = 0;
let liveStreamAbortController = null;
let liveMediaSource = null;
let liveObjectUrl = "";
const liveClientId = window.crypto?.randomUUID?.()
  || `${Date.now()}-${Math.random().toString(16).slice(2)}`;

const STATUS_FAILURE_THRESHOLD = 3;
const PLC_SIGNAL_HIGHLIGHT_MS = 4000;
const MEASUREMENT_HIGHLIGHT_MS = 900;
const LIVE_TARGET_LATENCY_SECONDS = 0.35;
const LIVE_SOFT_LATENCY_SECONDS = 0.7;
const LIVE_HARD_LATENCY_SECONDS = 1.5;
const LIVE_STALL_TIMEOUT_MS = 8000;
const LIVE_STARTUP_TIMEOUT_MS = 15000;
const LIVE_APPEND_BYTES = 512 * 1024;
const LIVE_MIME_TYPE = 'video/mp4; codecs="avc1.640033"';

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

function livePresentationDelayMs(trigger) {
  let liveLatencyMs = 0;
  if (liveVideo.buffered.length && liveVideo.readyState >= 1) {
    const liveEdge = liveVideo.buffered.end(liveVideo.buffered.length - 1);
    liveLatencyMs = Math.max(0, (liveEdge - liveVideo.currentTime) * 1000);
  }
  const timestamp = trigger.event_source_timestamp || trigger.read_utc;
  const parsed = timestamp ? new Date(timestamp).getTime() : Number.NaN;
  const eventAgeMs = Number.isFinite(parsed) ? Math.max(0, Date.now() - parsed) : 0;
  return Math.min(2000, Math.max(0, liveLatencyMs - eventAgeMs));
}

function presentPendingPlcEvent() {
  const now = Date.now();
  if (
    pendingPlcPresentation
    && !pendingPlcPresentation.presentationStarted
    && now >= pendingPlcPresentation.presentAt
  ) {
    plcSignalHighlightUntil = now + PLC_SIGNAL_HIGHLIGHT_MS;
    measurementHighlightUntil = now + MEASUREMENT_HIGHLIGHT_MS;
    pendingPlcPresentation.presentationStarted = true;
  }
  if (
    pendingPlcPresentation?.resultReady
    && pendingPlcPresentation.presentationStarted
  ) {
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

  byId("original-stage").classList.toggle(
    "measurement-taken",
    Date.now() < measurementHighlightUntil,
  );
}

function updatePlcSignal(plc) {
  const signal = byId("plc-signal");
  const text = byId("plc-signal-text");
  const trigger = plc.last_trigger;

  if (!plc.connected) {
    plcSignalHighlightUntil = 0;
    measurementHighlightUntil = 0;
    pendingPlcPresentation = null;
    signal.className = "plc-signal offline";
    text.textContent = "PLC disconnected";
    signal.title = "";
  } else if (!trigger) {
    plcSignalHighlightUntil = 0;
    measurementHighlightUntil = 0;
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
      const now = Date.now();
      const presentationDelayMs = livePresentationDelayMs(trigger);
      byId("original-stage").dataset.plcPresentationDelayMs = (
        presentationDelayMs.toFixed(0)
      );
      pendingPlcPresentation = {
        key: triggerKey,
        resultReady: false,
        presentAt: now + presentationDelayMs,
        presentationStarted: false,
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
    liveVideo.playbackRate = 1.25;
  } else if (latency > LIVE_SOFT_LATENCY_SECONDS) {
    liveVideo.playbackRate = 1.1;
  } else if (liveVideo.playbackRate !== 1) {
    liveVideo.playbackRate = 1;
  }
}

function stopLiveVideoTransport() {
  if (liveStreamAbortController) liveStreamAbortController.abort();
  liveStreamAbortController = null;
  liveMediaSource = null;
  if (liveObjectUrl) URL.revokeObjectURL(liveObjectUrl);
  liveObjectUrl = "";
}

function appendLiveChunk(sourceBuffer, chunk, generation) {
  return new Promise((resolve, reject) => {
    if (generation !== liveStreamGeneration) {
      resolve();
      return;
    }

    const cleanup = () => {
      sourceBuffer.removeEventListener("updateend", onUpdateEnd);
      sourceBuffer.removeEventListener("error", onError);
    };
    const onUpdateEnd = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("The browser rejected a live video fragment."));
    };

    sourceBuffer.addEventListener("updateend", onUpdateEnd, { once: true });
    sourceBuffer.addEventListener("error", onError, { once: true });
    try {
      sourceBuffer.appendBuffer(chunk);
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}

async function trimLiveBuffer(sourceBuffer, generation) {
  if (
    generation !== liveStreamGeneration
    || liveVideo.currentTime < 30
    || !sourceBuffer.buffered.length
  ) return;
  const removeEnd = liveVideo.currentTime - 15;
  if (sourceBuffer.buffered.start(0) >= removeEnd) return;
  await new Promise((resolve, reject) => {
    const cleanup = () => {
      sourceBuffer.removeEventListener("updateend", onUpdateEnd);
      sourceBuffer.removeEventListener("error", onError);
    };
    const onUpdateEnd = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Could not trim the live video buffer."));
    };
    sourceBuffer.addEventListener("updateend", onUpdateEnd, { once: true });
    sourceBuffer.addEventListener("error", onError, { once: true });
    try {
      sourceBuffer.remove(0, removeEnd);
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}

async function consumeLiveStream(mediaSource, generation) {
  const sourceBuffer = mediaSource.addSourceBuffer(LIVE_MIME_TYPE);
  const controller = new AbortController();
  liveStreamAbortController = controller;
  const response = await fetch(
    `/api/live/stream.mp4?generation=${generation}&client_id=${
      encodeURIComponent(liveClientId)
    }`,
    { cache: "no-store", signal: controller.signal },
  );
  if (!response.ok || !response.body) {
    throw new Error(`Live stream returned HTTP ${response.status}.`);
  }

  const reader = response.body.getReader();
  let pendingChunks = [];
  let pendingBytes = 0;
  while (generation === liveStreamGeneration) {
    const { done, value } = await reader.read();
    if (done) throw new Error("The live stream ended.");
    if (!value?.byteLength) continue;
    pendingChunks.push(value);
    pendingBytes += value.byteLength;
    if (pendingBytes < LIVE_APPEND_BYTES) continue;

    const appendData = new Uint8Array(pendingBytes);
    let offset = 0;
    pendingChunks.forEach((part) => {
      appendData.set(part, offset);
      offset += part.byteLength;
    });
    pendingChunks = [];
    pendingBytes = 0;
    await appendLiveChunk(sourceBuffer, appendData, generation);
    await trimLiveBuffer(sourceBuffer, generation);
    if (generation !== liveStreamGeneration) return;
    keepLiveVideoNearEdge();
    liveVideo.play().catch(() => {});
  }
}

function startLiveVideo() {
  const generation = liveStreamGeneration + 1;
  liveStreamGeneration = generation;
  stopLiveVideoTransport();
  liveVideo.dataset.streamError = "";
  liveStage.classList.remove("stream-ready");
  liveVideoState.textContent = "Connecting to camera...";
  lastLiveVideoProgressAt = Date.now();
  lastLiveVideoCurrentTime = -1;

  if (!window.MediaSource || !MediaSource.isTypeSupported(LIVE_MIME_TYPE)) {
    liveVideo.src = `/api/live/stream.mp4?fallback=${Date.now()}&client_id=${
      encodeURIComponent(liveClientId)
    }`;
    liveVideo.load();
    liveVideo.play().catch(() => {});
    return;
  }

  const mediaSource = new MediaSource();
  liveMediaSource = mediaSource;
  liveObjectUrl = URL.createObjectURL(mediaSource);
  liveVideo.src = liveObjectUrl;
  mediaSource.addEventListener("sourceopen", () => {
    if (generation !== liveStreamGeneration) return;
    consumeLiveStream(mediaSource, generation).catch((error) => {
      if (error?.name !== "AbortError" && generation === liveStreamGeneration) {
        liveVideo.dataset.streamError = String(error?.message || error);
        reconnectLiveVideo();
      }
    });
  }, { once: true });
  liveVideo.load();
}

function reconnectLiveVideo() {
  if (streamRetryTimer !== null) return;
  streamRetryTimer = setTimeout(() => {
    streamRetryTimer = null;
    liveVideoState.textContent = "Reconnecting...";
    startLiveVideo();
    streamRetryDelayMs = Math.min(streamRetryDelayMs * 2, 8000);
  }, streamRetryDelayMs);
}

function noteLiveVideoProgress() {
  const currentTime = Number(liveVideo.currentTime || 0);
  liveVideo.dataset.currentTime = currentTime.toFixed(3);
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
  const progressAge = Date.now() - lastLiveVideoProgressAt;
  liveVideo.dataset.lastProgressAgeMs = String(progressAge);
  const timeout = liveStage.classList.contains("stream-ready")
    ? LIVE_STALL_TIMEOUT_MS
    : LIVE_STARTUP_TIMEOUT_MS;
  if (
    !document.hidden
    && progressAge > timeout
  ) reconnectLiveVideo();
}

liveVideo.addEventListener("playing", () => {
  clearTimeout(streamRetryTimer);
  streamRetryTimer = null;
  streamRetryDelayMs = 1000;
  liveVideo.dataset.streamError = "";
  liveStage.classList.add("stream-ready");
  noteLiveVideoProgress();
  keepLiveVideoNearEdge();
});
liveVideo.addEventListener("timeupdate", noteLiveVideoProgress);
liveVideo.addEventListener("error", reconnectLiveVideo);
document.addEventListener("visibilitychange", () => {
  if (
    !document.hidden
    && Date.now() - lastLiveVideoProgressAt > LIVE_STALL_TIMEOUT_MS
  ) reconnectLiveVideo();
});
window.addEventListener("pagehide", () => {
  stopLiveVideoTransport();
  navigator.sendBeacon(
    `/api/live/disconnect?client_id=${encodeURIComponent(liveClientId)}`,
  );
});

function sendLiveHeartbeat() {
  fetch(
    `/api/live/heartbeat?client_id=${encodeURIComponent(liveClientId)}`,
    { method: "POST", cache: "no-store", keepalive: true },
  ).catch(() => {});
}

setInterval(keepLiveVideoNearEdge, 500);
setInterval(monitorLiveVideo, 1000);
setInterval(sendLiveHeartbeat, 2000);
setInterval(refreshAnalysis, 100);
setInterval(refreshStatus, 1000);
startLiveVideo();
sendLiveHeartbeat();
refreshAnalysis();
refreshStatus();
