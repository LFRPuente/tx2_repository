CREATE TABLE IF NOT EXISTS vision_configuration (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    homography_sha256 TEXT NOT NULL,
    calibration_sha256 TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    homography_json TEXT NOT NULL,
    calibration_json TEXT NOT NULL,
    model_path TEXT NOT NULL,
    model_name TEXT NOT NULL,
    confidence_threshold REAL NOT NULL,
    inference_size INTEGER NOT NULL,
    app_commit_sha TEXT,
    UNIQUE (
        homography_sha256,
        calibration_sha256,
        model_sha256,
        confidence_threshold,
        inference_size
    )
);

CREATE TABLE IF NOT EXISTS measurement_event (
    id TEXT PRIMARY KEY,
    event_key TEXT NOT NULL UNIQUE,
    configuration_id TEXT NOT NULL REFERENCES vision_configuration(id),
    status TEXT NOT NULL CHECK (status IN (
        'recording',
        'processing',
        'complete',
        'needs_review',
        'failed'
    )),
    plc_endpoint TEXT NOT NULL,
    plc_event_node TEXT NOT NULL,
    plc_watchdog_node TEXT NOT NULL,
    plc_edge TEXT NOT NULL,
    plc_event_value TEXT,
    plc_previous_event_value TEXT,
    plc_watchdog_value TEXT,
    plc_source_timestamp TEXT,
    plc_server_timestamp TEXT,
    app_received_at TEXT NOT NULL,
    recording_started_at TEXT,
    recording_ended_at TEXT,
    stop_reason TEXT,
    camera_source TEXT,
    camera_width INTEGER,
    camera_height INTEGER,
    first_frame_index INTEGER,
    last_frame_index INTEGER,
    first_frame_utc TEXT,
    last_frame_utc TEXT,
    detected_piece_count INTEGER NOT NULL DEFAULT 0,
    valid_piece_count INTEGER NOT NULL DEFAULT 0,
    canonical_snapshot_id INTEGER,
    error_text TEXT,
    legacy_sidecar_path TEXT UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS measurement_event_source_time_idx
    ON measurement_event (plc_source_timestamp DESC);

CREATE INDEX IF NOT EXISTS measurement_event_created_idx
    ON measurement_event (created_at DESC);

CREATE INDEX IF NOT EXISTS measurement_event_status_idx
    ON measurement_event (status, created_at DESC);

CREATE TABLE IF NOT EXISTS analysis_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL REFERENCES measurement_event(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    frame_utc TEXT,
    processed_utc TEXT NOT NULL,
    processing_duration_ms REAL,
    is_canonical INTEGER NOT NULL DEFAULT 0,
    detected_piece_count INTEGER NOT NULL DEFAULT 0,
    valid_piece_count INTEGER NOT NULL DEFAULT 0,
    measurement_summary TEXT NOT NULL,
    box_rules TEXT NOT NULL,
    original_overlay_path TEXT,
    rectified_overlay_path TEXT,
    result_json TEXT NOT NULL,
    UNIQUE (event_id, frame_index)
);

CREATE INDEX IF NOT EXISTS analysis_snapshot_event_idx
    ON analysis_snapshot (event_id, frame_index);

CREATE UNIQUE INDEX IF NOT EXISTS analysis_snapshot_one_canonical_idx
    ON analysis_snapshot (event_id)
    WHERE is_canonical = 1;

CREATE TABLE IF NOT EXISTS piece_measurement (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL REFERENCES measurement_event(id) ON DELETE CASCADE,
    snapshot_id INTEGER NOT NULL REFERENCES analysis_snapshot(id) ON DELETE CASCADE,
    piece_number INTEGER NOT NULL CHECK (piece_number > 0),
    is_valid INTEGER NOT NULL,
    review_required INTEGER NOT NULL DEFAULT 0,
    yolo_confidence REAL,
    sobel_confidence REAL,
    crm_px REAL,
    delta_px REAL,
    distance_to_reference_in REAL,
    automatic_measurement_in REAL,
    operator_measurement_in REAL,
    operator_revision INTEGER NOT NULL DEFAULT 0,
    box_json TEXT NOT NULL,
    sobel_json TEXT NOT NULL,
    automatic_result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
        automatic_measurement_in IS NULL
        OR automatic_measurement_in BETWEEN 0 AND 2400
    ),
    CHECK (
        operator_measurement_in IS NULL
        OR operator_measurement_in BETWEEN 0 AND 2400
    ),
    UNIQUE (event_id, piece_number)
);

CREATE INDEX IF NOT EXISTS piece_measurement_event_idx
    ON piece_measurement (event_id, piece_number);

CREATE TABLE IF NOT EXISTS piece_measurement_revision (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    piece_measurement_id TEXT NOT NULL
        REFERENCES piece_measurement(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL CHECK (revision > 0),
    action TEXT NOT NULL CHECK (action IN ('set', 'change', 'clear')),
    previous_operator_measurement_in REAL,
    new_operator_measurement_in REAL,
    automatic_measurement_in REAL,
    operator_id TEXT NOT NULL,
    operator_display_name TEXT,
    reason TEXT NOT NULL,
    source_ip TEXT,
    changed_at TEXT NOT NULL,
    UNIQUE (piece_measurement_id, revision)
);

CREATE INDEX IF NOT EXISTS piece_revision_piece_idx
    ON piece_measurement_revision (piece_measurement_id, revision DESC);

CREATE TABLE IF NOT EXISTS event_asset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL REFERENCES measurement_event(id) ON DELETE CASCADE,
    asset_type TEXT NOT NULL CHECK (asset_type IN (
        'video',
        'raw_video',
        'sidecar',
        'original_overlay',
        'rectified_overlay'
    )),
    relative_path TEXT NOT NULL,
    mime_type TEXT,
    size_bytes INTEGER,
    sha256 TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (event_id, asset_type, relative_path)
);

CREATE VIEW IF NOT EXISTS piece_measurement_effective AS
SELECT
    pm.*,
    COALESCE(
        pm.operator_measurement_in,
        pm.automatic_measurement_in
    ) AS effective_measurement_in,
    pm.operator_measurement_in IS NOT NULL AS has_operator_override
FROM piece_measurement pm;
