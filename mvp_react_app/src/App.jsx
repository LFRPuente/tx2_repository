import { useEffect, useState } from 'react';

const DEFAULT_SECOND = 115;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function fmt(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '-';
}

function imageSrc(base64) {
  return base64 ? `data:image/jpeg;base64,${base64}` : '';
}

function validLine(line) {
  return line && ['x1', 'y1', 'x2', 'y2'].every((key) => Number.isFinite(Number(line[key])));
}

function PieceDiagram({ frame }) {
  const ratio = frame?.front_y_ratio;
  const hasLine = Number.isFinite(Number(ratio));
  const frontY = hasLine ? clamp(Number(ratio) * 300 + 55, 150, 345) : 230;
  const pieceCols = Array.from({ length: 12 }, (_, index) => index);
  const referenceY = 213;
  const packTop = 86;
  const packBottom = Math.max(packTop + 24, frontY - 8);
  const pieceHeight = Math.max(18, packBottom - packTop);

  return (
    <section className="panel diagram-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Diagram</p>
          <h2>Piece bundle</h2>
        </div>
      </div>

      <svg className="piece-diagram" viewBox="0 0 760 430" role="img" aria-label="Piece front measurement diagram">
        <rect x="44" y="44" width="672" height="342" rx="8" fill="#f7faf8" stroke="#cbd6cf" strokeWidth="2" />
        <rect x="72" y="82" width="616" height="262" rx="6" fill="#ffffff" stroke="#d9e0dc" />
        <path d="M380 372 V96" stroke="#aeb8b2" strokeWidth="3" strokeDasharray="10 9" />
        <path d="M380 96 l-10 18 h20 z" fill="#aeb8b2" />

        <line
          x1="86"
          x2="674"
          y1={referenceY}
          y2={referenceY}
          stroke="#2f5f9d"
          strokeWidth="4"
          strokeDasharray="10 8"
          strokeLinecap="round"
        />
        <text x="96" y={referenceY + 24} fill="#2f5f9d" fontSize="18" fontWeight="800">
          Reference
        </text>

        <rect
          x="112"
          y={packTop}
          width="524"
          height={Math.max(16, packBottom - packTop)}
          rx="7"
          fill="#edf5f1"
          opacity=".78"
        />

        {pieceCols.map((col) => {
          const x = 132 + col * 40;
          const shade = col % 2 ? '#d7e5df' : '#c2d7ce';
          return (
            <g key={col}>
              <rect x={x} y={packTop} width="24" height={pieceHeight} rx="4" fill={shade} stroke="#8ba99b" />
              <line x1={x + 6} y1={packTop + 6} x2={x + 6} y2={packTop + pieceHeight - 6} stroke="#f6fbf8" strokeWidth="2" opacity=".75" />
            </g>
          );
        })}

        <line
          x1="86"
          x2="674"
          y1={frontY}
          y2={frontY}
          stroke={hasLine ? '#15845c' : '#b78c38'}
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={hasLine ? '0' : '12 10'}
        />
        <text x="96" y={frontY - 14} fill={hasLine ? '#116b4b' : '#8a6627'} fontSize="20" fontWeight="800">
          Piece front
        </text>

        {frame?.measurement && (
          <g>
            <line
              x1="690"
              x2="690"
              y1={frontY}
              y2={referenceY}
              stroke="#2f5f9d"
              strokeWidth="3"
              strokeDasharray="7 6"
            />
            <circle cx="690" cy={frontY} r="6" fill="#15845c" />
            <circle cx="690" cy={referenceY} r="6" fill="#2f5f9d" />
            <text x="340" y="405" fill="#1e2f3a" fontSize="18" fontWeight="800">
              Total {fmt(frame.measurement.measurement_in)} in | Distance {fmt(frame.measurement.delta_in)} in
            </text>
          </g>
        )}
      </svg>
    </section>
  );
}

function ImageLine({ line, color, label, dashed = false, width, height, labelOffset = -14 }) {
  if (!validLine(line)) return null;
  const x1 = Number(line.x1);
  const y1 = Number(line.y1);
  const x2 = Number(line.x2);
  const y2 = Number(line.y2);
  const labelX = clamp(Math.min(x1, x2) + 18, 12, Math.max(12, width - 190));
  const labelY = clamp((y1 + y2) / 2 + labelOffset, 26, Math.max(26, height - 18));
  const strokeWidth = Math.max(5, width / 360);

  return (
    <g>
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="rgba(255,255,255,.86)"
        strokeWidth={strokeWidth + 4}
        strokeLinecap="round"
      />
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={dashed ? `${strokeWidth * 2.2} ${strokeWidth * 1.8}` : undefined}
        strokeLinecap="round"
      />
      <text
        x={labelX}
        y={labelY}
        fill={color}
        fontSize={Math.max(20, width / 70)}
        fontWeight="900"
        paintOrder="stroke"
        stroke="rgba(255,255,255,.92)"
        strokeWidth={Math.max(4, width / 420)}
      >
        {label}
      </text>
    </g>
  );
}

function OriginalOverlay({ frame }) {
  const width = Number(frame?.original_width);
  const height = Number(frame?.original_height);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return null;

  const overlay = frame?.original_overlay || {};
  if (!validLine(overlay.reference_line) && !validLine(overlay.front_line)) return null;

  return (
    <svg
      className="video-overlay"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      aria-hidden="true"
    >
      <ImageLine line={overlay.reference_line} color="#2f5f9d" label="Reference" dashed width={width} height={height} labelOffset={34} />
      <ImageLine line={overlay.front_line} color="#15845c" label="Piece front" width={width} height={height} labelOffset={-18} />
    </svg>
  );
}

function OriginalImage({ frame }) {
  return (
    <section className="panel image-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Live video</p>
          <h2>Original image</h2>
        </div>
        <span className="state-pill neutral">Frame {frame?.frame_idx ?? '-'}</span>
      </div>
      <div className="image-stage">
        {frame?.original_image ? (
          <>
            <img src={imageSrc(frame.original_image)} alt="Original TX2 video frame" />
            <OriginalOverlay frame={frame} />
          </>
        ) : (
          <div className="empty">Loading image...</div>
        )}
      </div>
    </section>
  );
}

function Metric({ label, value, tone = '' }) {
  return (
    <div className={`metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function App() {
  const [meta, setMeta] = useState(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [draftFrame, setDraftFrame] = useState('0');
  const [frame, setFrame] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [playing, setPlaying] = useState(false);
  const [playStep, setPlayStep] = useState(1);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/meta')
      .then((response) => response.json())
      .then((data) => {
        if (cancelled) return;
        setMeta(data);
        const start = Math.round(DEFAULT_SECOND * Number(data.fps || 30));
        const safeStart = clamp(start, 0, Math.max(0, Number(data.total_frames || 1) - 1));
        setFrameIdx(safeStart);
        setDraftFrame(String(safeStart));
      })
      .catch((err) => setError(err.message || 'Could not load metadata'));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!meta) return;
    const controller = new AbortController();
    setLoading(true);
    setError('');
    fetch('/api/mvp/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame_idx: frameIdx, conf: 0.1 }),
      signal: controller.signal,
    })
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Error loading frame');
        return data;
      })
      .then((data) => {
        setFrame(data);
        setDraftFrame(String(data.frame_idx));
      })
      .catch((err) => {
        if (err.name !== 'AbortError') setError(err.message || 'Error loading frame');
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [frameIdx, meta]);

  const maxFrame = Math.max(0, Number(meta?.total_frames || frame?.total_frames || 1) - 1);
  const fps = Number(meta?.fps || frame?.fps || 30);
  const timeSec = frame ? frame.time_sec : frameIdx / fps;
  const measured = frame?.measurement;

  useEffect(() => {
    if (!playing || loading || !meta) return undefined;
    const delayMs = playStep > 1 ? 260 : 420;
    const timer = window.setTimeout(() => {
      setFrameIdx((current) => {
        const next = clamp(current + playStep, 0, maxFrame);
        if (next === current) {
          setPlaying(false);
          return current;
        }
        setDraftFrame(String(next));
        return next;
      });
    }, delayMs);
    return () => window.clearTimeout(timer);
  }, [playing, loading, meta, maxFrame, playStep]);

  function goTo(nextFrame, options = {}) {
    if (!options.keepPlaying) setPlaying(false);
    const safe = clamp(Math.round(nextFrame), 0, maxFrame);
    setFrameIdx(safe);
    setDraftFrame(String(safe));
  }

  function goDraft() {
    goTo(Number(draftFrame || 0));
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">TX2 Vision MVP</p>
          <h1>Piece front measurement</h1>
        </div>
      </header>

      <main>
        {error && <div className="error-banner">{error}</div>}

        <section className="controls panel">
          <div className="transport">
            <button className={playing ? 'primary' : 'play-button'} onClick={() => setPlaying((value) => !value)}>
              {playing ? 'Pause' : 'Play'}
            </button>
            <button onClick={() => setPlayStep((value) => (value === 1 ? 2 : 1))}>
              x{playStep}
            </button>
            <button onClick={() => goTo(frameIdx - Math.round(fps * 5))} disabled={loading}>-5s</button>
            <button onClick={() => goTo(frameIdx - Math.round(fps))} disabled={loading}>-1s</button>
            <button onClick={() => goTo(frameIdx - 1)} disabled={loading}>-1f</button>
            <button className="primary" onClick={() => goTo(frameIdx + 1)} disabled={loading}>+1f</button>
            <button onClick={() => goTo(frameIdx + Math.round(fps))} disabled={loading}>+1s</button>
            <button onClick={() => goTo(frameIdx + Math.round(fps * 5))} disabled={loading}>+5s</button>
          </div>

          <div className="timeline">
            <input
              type="range"
              min="0"
              max={maxFrame}
              value={Number(draftFrame || 0)}
              onChange={(event) => {
                setPlaying(false);
                setDraftFrame(event.target.value);
              }}
              onPointerUp={goDraft}
              onKeyUp={(event) => {
                if (event.key === 'Enter') goDraft();
              }}
            />
            <div className="goto">
              <label>Frame</label>
              <input value={draftFrame} onChange={(event) => {
                setPlaying(false);
                setDraftFrame(event.target.value);
              }} onKeyDown={(event) => {
                if (event.key === 'Enter') goDraft();
              }} />
              <button onClick={goDraft} disabled={loading}>Go</button>
            </div>
          </div>
        </section>

        <section className="metrics">
          <Metric label="Time" value={`${fmt(timeSec, 2)} s`} />
          <Metric label="Frame" value={frame?.frame_idx ?? frameIdx} />
          <Metric label="Total measurement" value={measured ? `${fmt(measured.measurement_in)} in` : '-'} tone="total" />
          <Metric label="Distance to ref" value={measured ? `${fmt(measured.delta_in)} in` : '-'} tone="reference" />
        </section>

        <section className="main-grid">
          <PieceDiagram frame={frame} />
          <OriginalImage frame={frame} />
        </section>
      </main>
    </div>
  );
}
