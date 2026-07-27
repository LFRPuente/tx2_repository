BEGIN;

CREATE TABLE vision_configuration (
    id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    homography_sha256 text NOT NULL,
    calibration_sha256 text NOT NULL,
    model_sha256 text NOT NULL,
    homography_json jsonb NOT NULL,
    calibration_json jsonb NOT NULL,
    model_path text NOT NULL,
    model_name text NOT NULL,
    confidence_threshold double precision NOT NULL,
    inference_size integer NOT NULL,
    app_commit_sha text,
    UNIQUE (
        homography_sha256,
        calibration_sha256,
        model_sha256,
        confidence_threshold,
        inference_size
    )
);

CREATE TABLE measurement_event (
    id uuid PRIMARY KEY,
    event_key text NOT NULL UNIQUE,
    configuration_id uuid NOT NULL
        REFERENCES vision_configuration(id),
    status text NOT NULL
        CHECK (status IN (
            'recording',
            'processing',
            'complete',
            'needs_review',
            'failed'
        )),
    plc_endpoint text NOT NULL,
    plc_event_node text NOT NULL,
    plc_watchdog_node text NOT NULL,
    plc_edge text NOT NULL,
    plc_event_value jsonb,
    plc_previous_event_value jsonb,
    plc_watchdog_value jsonb,
    plc_source_timestamp timestamptz,
    plc_server_timestamp timestamptz,
    app_received_at timestamptz NOT NULL,
    recording_started_at timestamptz,
    recording_ended_at timestamptz,
    stop_reason text,
    camera_source text,
    camera_width integer,
    camera_height integer,
    first_frame_index bigint,
    last_frame_index bigint,
    first_frame_utc timestamptz,
    last_frame_utc timestamptz,
    detected_piece_count integer NOT NULL DEFAULT 0,
    valid_piece_count integer NOT NULL DEFAULT 0,
    canonical_snapshot_id bigint,
    error_text text,
    legacy_sidecar_path text UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX measurement_event_source_time_idx
    ON measurement_event (plc_source_timestamp DESC);

CREATE INDEX measurement_event_created_idx
    ON measurement_event (created_at DESC);

CREATE INDEX measurement_event_status_idx
    ON measurement_event (status, created_at DESC);

CREATE TABLE analysis_snapshot (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id uuid NOT NULL
        REFERENCES measurement_event(id) ON DELETE CASCADE,
    frame_index bigint NOT NULL,
    frame_utc timestamptz,
    processed_utc timestamptz NOT NULL,
    processing_duration_ms double precision,
    is_canonical boolean NOT NULL DEFAULT false,
    detected_piece_count integer NOT NULL DEFAULT 0,
    valid_piece_count integer NOT NULL DEFAULT 0,
    measurement_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    box_rules jsonb NOT NULL DEFAULT '{}'::jsonb,
    original_overlay_path text,
    rectified_overlay_path text,
    result_json jsonb NOT NULL,
    UNIQUE (event_id, frame_index)
);

CREATE INDEX analysis_snapshot_event_idx
    ON analysis_snapshot (event_id, frame_index);

CREATE UNIQUE INDEX analysis_snapshot_one_canonical_idx
    ON analysis_snapshot (event_id)
    WHERE is_canonical;

CREATE TABLE piece_measurement (
    id uuid PRIMARY KEY,
    event_id uuid NOT NULL
        REFERENCES measurement_event(id) ON DELETE CASCADE,
    snapshot_id bigint NOT NULL
        REFERENCES analysis_snapshot(id) ON DELETE CASCADE,
    piece_number integer NOT NULL CHECK (piece_number > 0),
    is_valid boolean NOT NULL,
    review_required boolean NOT NULL DEFAULT false,
    yolo_confidence double precision,
    sobel_confidence double precision,
    crm_px double precision,
    delta_px double precision,
    distance_to_reference_in numeric(12, 6),
    automatic_measurement_in numeric(12, 6),
    operator_measurement_in numeric(12, 6),
    operator_revision integer NOT NULL DEFAULT 0,
    box_json jsonb NOT NULL,
    sobel_json jsonb NOT NULL,
    automatic_result_json jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
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

CREATE INDEX piece_measurement_event_idx
    ON piece_measurement (event_id, piece_number);

CREATE TABLE piece_measurement_revision (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    piece_measurement_id uuid NOT NULL
        REFERENCES piece_measurement(id) ON DELETE CASCADE,
    revision integer NOT NULL CHECK (revision > 0),
    action text NOT NULL
        CHECK (action IN ('set', 'change', 'clear')),
    previous_operator_measurement_in numeric(12, 6),
    new_operator_measurement_in numeric(12, 6),
    automatic_measurement_in numeric(12, 6),
    operator_id text NOT NULL,
    operator_display_name text,
    reason text NOT NULL,
    source_ip inet,
    changed_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (piece_measurement_id, revision)
);

CREATE INDEX piece_revision_piece_idx
    ON piece_measurement_revision (
        piece_measurement_id,
        revision DESC
    );

CREATE TABLE event_asset (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id uuid NOT NULL
        REFERENCES measurement_event(id) ON DELETE CASCADE,
    asset_type text NOT NULL
        CHECK (asset_type IN (
            'video',
            'sidecar',
            'original_overlay',
            'rectified_overlay'
        )),
    relative_path text NOT NULL,
    mime_type text,
    size_bytes bigint,
    sha256 text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (event_id, asset_type, relative_path)
);

CREATE VIEW piece_measurement_effective AS
SELECT
    pm.*,
    COALESCE(
        pm.operator_measurement_in,
        pm.automatic_measurement_in
    ) AS effective_measurement_in,
    pm.operator_measurement_in IS NOT NULL AS has_operator_override
FROM piece_measurement pm;

ALTER TABLE measurement_event
    ADD CONSTRAINT measurement_event_canonical_snapshot_fk
    FOREIGN KEY (canonical_snapshot_id)
    REFERENCES analysis_snapshot(id);

COMMIT;
