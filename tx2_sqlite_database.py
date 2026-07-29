"""Temporary SQLite persistence for the TX2 Live MVP."""
from __future__ import annotations

import getpass
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from tx2_database import (
    DatabaseHealth,
    DatabaseUnavailable,
    RecordNotFound,
    RevisionConflict,
    _optional_relative_path,
    _sidecar_assets,
    build_event_key,
    event_uuid,
    load_json_file,
    relative_asset_path,
    select_canonical_snapshot,
    snapshot_pieces,
    snapshot_summary,
    utc_now,
)


SQLITE_JSON_COLUMNS = {
    "homography_json",
    "calibration_json",
    "plc_event_value",
    "plc_previous_event_value",
    "plc_watchdog_value",
    "measurement_summary",
    "box_rules",
    "result_json",
    "box_json",
    "sobel_json",
    "automatic_result_json",
}
SQLITE_BOOLEAN_COLUMNS = {
    "is_canonical",
    "is_valid",
    "review_required",
    "has_operator_override",
}


def _ensure_raw_video_asset_type(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = 'event_asset'
        """
    ).fetchone()
    schema = str(row["sql"] if row is not None else "")
    if "'raw_video'" in schema:
        return

    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            ALTER TABLE event_asset RENAME TO event_asset_legacy;
            CREATE TABLE event_asset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL
                    REFERENCES measurement_event(id) ON DELETE CASCADE,
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
            INSERT INTO event_asset (
                id, event_id, asset_type, relative_path, mime_type,
                size_bytes, sha256, created_at
            )
            SELECT
                id, event_id, asset_type, relative_path, mime_type,
                size_bytes, sha256, created_at
            FROM event_asset_legacy;
            DROP TABLE event_asset_legacy;
            COMMIT;
            """
        )
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")


def _json_text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


def _decode_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for key in SQLITE_JSON_COLUMNS:
        value = result.get(key)
        if isinstance(value, str):
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    for key in SQLITE_BOOLEAN_COLUMNS:
        if key in result and result[key] is not None:
            result[key] = bool(result[key])
    return result


class SQLiteDatabaseRepository:
    """SQLite implementation of the Live MVP database repository contract."""

    backend_name = "sqlite"

    def __init__(self, path: str | Path) -> None:
        value = str(path).strip()
        if not value:
            raise ValueError("A SQLite database path is required")
        self.path = Path(value).expanduser().resolve()
        self._opened = False

    def _connect(self, timeout: float = 8.0) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=max(1.0, float(timeout)),
            isolation_level="DEFERRED",
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 8000")
        return connection

    def open(self, timeout: float = 8.0) -> None:
        del timeout
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            migration_path = (
                Path(__file__).resolve().parent
                / "db"
                / "migrations"
                / "001_sqlite_initial.sql"
            )
            migration = migration_path.read_text(encoding="utf-8")
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")
                connection.executescript(migration)
                _ensure_raw_video_asset_type(connection)
                connection.commit()
            finally:
                connection.close()
            self._opened = True
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not initialize SQLite: {exc}") from exc

    def close(self) -> None:
        self._opened = False

    def health(self) -> DatabaseHealth:
        try:
            connection = self._connect(timeout=5.0)
            try:
                connection.execute("SELECT 1").fetchone()
                journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
            finally:
                connection.close()
            return DatabaseHealth(
                enabled=True,
                ok=True,
                database=str(self.path),
                user=getpass.getuser(),
                timezone=f"UTC / WAL={journal_mode}",
                backend=self.backend_name,
            )
        except Exception as exc:
            return DatabaseHealth(
                enabled=True,
                ok=False,
                error=str(exc),
                database=str(self.path),
                backend=self.backend_name,
            )

    def validate_schema(self) -> None:
        required = {
            "vision_configuration",
            "measurement_event",
            "analysis_snapshot",
            "piece_measurement",
            "piece_measurement_revision",
            "event_asset",
        }
        try:
            connection = self._connect(timeout=5.0)
            try:
                rows = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    """
                ).fetchall()
            finally:
                connection.close()
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not inspect SQLite schema: {exc}") from exc
        present = {str(row["name"]) for row in rows}
        missing = sorted(required - present)
        if missing:
            raise DatabaseUnavailable(
                "SQLite schema is incomplete; missing: " + ", ".join(missing)
            )

    def recover_stale_measurement_events(self, older_than_seconds: int = 60) -> int:
        threshold = max(30, int(older_than_seconds))
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=threshold)
        ).isoformat(timespec="milliseconds")
        now = utc_now()
        try:
            connection = self._connect()
            try:
                with connection:
                    cursor = connection.execute(
                        """
                        UPDATE measurement_event
                        SET status = 'failed',
                            error_text = COALESCE(
                                error_text,
                                'Application restarted before the event was finalized'
                            ),
                            recording_ended_at = COALESCE(recording_ended_at, ?),
                            updated_at = ?
                        WHERE status IN ('recording', 'processing')
                          AND updated_at < ?
                        """,
                        (now, now, cutoff),
                    )
                    return int(cursor.rowcount)
            finally:
                connection.close()
        except Exception as exc:
            raise DatabaseUnavailable(
                f"Could not recover stale SQLite measurement events: {exc}"
            ) from exc

    @staticmethod
    def _configuration_id(
        connection: sqlite3.Connection,
        configuration: dict[str, Any],
    ) -> str:
        key = (
            configuration["homography_sha256"],
            configuration["calibration_sha256"],
            configuration["model_sha256"],
            float(configuration["confidence_threshold"]),
            int(configuration["inference_size"]),
        )
        row = connection.execute(
            """
            SELECT id
            FROM vision_configuration
            WHERE homography_sha256 = ?
              AND calibration_sha256 = ?
              AND model_sha256 = ?
              AND confidence_threshold = ?
              AND inference_size = ?
            """,
            key,
        ).fetchone()
        if row is not None:
            configuration_id = str(row["id"])
            connection.execute(
                """
                UPDATE vision_configuration
                SET model_path = ?, model_name = ?, app_commit_sha = ?
                WHERE id = ?
                """,
                (
                    configuration["model_path"],
                    configuration["model_name"],
                    configuration.get("app_commit_sha"),
                    configuration_id,
                ),
            )
            return configuration_id

        configuration_id = str(uuid.uuid4())
        connection.execute(
            """
            INSERT INTO vision_configuration (
                id, created_at, homography_sha256, calibration_sha256, model_sha256,
                homography_json, calibration_json, model_path, model_name,
                confidence_threshold, inference_size, app_commit_sha
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                configuration_id,
                utc_now(),
                configuration["homography_sha256"],
                configuration["calibration_sha256"],
                configuration["model_sha256"],
                _json_text(configuration["homography_json"]),
                _json_text(configuration["calibration_json"]),
                configuration["model_path"],
                configuration["model_name"],
                float(configuration["confidence_threshold"]),
                int(configuration["inference_size"]),
                configuration.get("app_commit_sha"),
            ),
        )
        return configuration_id

    def create_measurement_event(
        self,
        *,
        event_id: str,
        event_key: str,
        event: dict[str, Any],
        configuration: dict[str, Any],
        plc_endpoint: str,
        plc_event_node: str,
        plc_watchdog_node: str,
        camera_source: str,
    ) -> str:
        try:
            connection = self._connect()
            try:
                with connection:
                    configuration_id = self._configuration_id(connection, configuration)
                    existing = connection.execute(
                        "SELECT id FROM measurement_event WHERE event_key = ?",
                        (event_key,),
                    ).fetchone()
                    if existing is not None:
                        connection.execute(
                            "UPDATE measurement_event SET updated_at = ? WHERE event_key = ?",
                            (utc_now(), event_key),
                        )
                        return str(existing["id"])
                    received_at = event.get("read_utc") or utc_now()
                    connection.execute(
                        """
                        INSERT INTO measurement_event (
                            id, event_key, configuration_id, status,
                            plc_endpoint, plc_event_node, plc_watchdog_node, plc_edge,
                            plc_event_value, plc_previous_event_value, plc_watchdog_value,
                            plc_source_timestamp, plc_server_timestamp, app_received_at,
                            recording_started_at, camera_source, created_at, updated_at
                        )
                        VALUES (?, ?, ?, 'recording', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event_id,
                            event_key,
                            configuration_id,
                            plc_endpoint,
                            plc_event_node,
                            plc_watchdog_node,
                            event.get("event_edge") or "event",
                            _json_text(event.get("event_value")),
                            _json_text(event.get("previous_event_value")),
                            _json_text(event.get("watchdog_value")),
                            event.get("event_source_timestamp"),
                            event.get("event_server_timestamp"),
                            received_at,
                            received_at,
                            camera_source,
                            utc_now(),
                            utc_now(),
                        ),
                    )
                    return event_id
            finally:
                connection.close()
        except Exception as exc:
            raise DatabaseUnavailable(
                f"Could not create SQLite measurement event: {exc}"
            ) from exc

    def mark_measurement_event_failed(self, event_id: str, error_text: str) -> None:
        self._update_event_status(
            event_id,
            status="failed",
            recording_ended_at=utc_now(),
            error_text=error_text[:4000],
        )

    def mark_measurement_event_processing(
        self,
        event_id: str,
        recording_ended_at: str,
    ) -> None:
        self._update_event_status(
            event_id,
            status="processing",
            recording_ended_at=recording_ended_at,
        )

    def _update_event_status(
        self,
        event_id: str,
        *,
        status: str,
        recording_ended_at: str,
        error_text: str | None = None,
    ) -> None:
        try:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        """
                        UPDATE measurement_event
                        SET status = ?, recording_ended_at = ?,
                            error_text = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (status, recording_ended_at, error_text, utc_now(), event_id),
                    )
            finally:
                connection.close()
        except Exception as exc:
            raise DatabaseUnavailable(
                f"Could not update SQLite measurement event: {exc}"
            ) from exc

    def delete_measurement_event(self, event_id: str) -> None:
        try:
            connection = self._connect()
            try:
                with connection:
                    connection.execute(
                        "DELETE FROM measurement_event WHERE id = ?",
                        (event_id,),
                    )
            finally:
                connection.close()
        except Exception as exc:
            raise DatabaseUnavailable(
                f"Could not delete retained SQLite measurement event: {exc}"
            ) from exc

    def sync_sidecar(self, sidecar_path: Path, output_dir: Path) -> str:
        sidecar_path = sidecar_path.resolve()
        output_dir = output_dir.resolve()
        data = load_json_file(sidecar_path)
        configuration = data.get("vision_configuration")
        if not isinstance(configuration, dict):
            raise ValueError(
                f"{sidecar_path} has no vision_configuration snapshot; "
                "provide one during legacy migration"
            )
        event = data.get("event") if isinstance(data.get("event"), dict) else {}
        event_key = str(
            data.get("event_key")
            or build_event_key(
                event,
                str(data.get("plc_endpoint") or ""),
                str(data.get("plc_event_node") or ""),
            )
        )
        requested_event_id = str(data.get("event_id") or event_uuid(event_key))
        snapshots = [
            item
            for item in data.get("processing_snapshots", [])
            if isinstance(item, dict)
        ]
        canonical = select_canonical_snapshot(
            snapshots,
            event_monotonic=event.get("event_read_monotonic"),
            target_offset_seconds=data.get("measurement_delay_seconds"),
        )
        canonical_frame_index = int(canonical["frame_index"]) if canonical else None
        canonical_summary = (
            snapshot_summary(canonical)
            if canonical
            else {"detected_count": 0, "valid_count": 0, "invalid_count": 0}
        )
        valid_count = int(canonical_summary.get("valid_count") or 0)
        detected_count = int(canonical_summary.get("detected_count") or 0)
        invalid_count = int(
            canonical_summary.get("invalid_count")
            or max(0, detected_count - valid_count)
        )
        status = "complete" if valid_count > 0 and invalid_count == 0 else "needs_review"
        legacy_sidecar_path = relative_asset_path(sidecar_path, output_dir)

        try:
            connection = self._connect()
            try:
                with connection:
                    configuration_id = self._configuration_id(connection, configuration)
                    existing = connection.execute(
                        "SELECT id FROM measurement_event WHERE event_key = ?",
                        (event_key,),
                    ).fetchone()
                    actual_event_id = (
                        str(existing["id"]) if existing is not None else requested_event_id
                    )
                    values = (
                        configuration_id,
                        status,
                        data.get("plc_endpoint") or "",
                        data.get("plc_event_node") or "",
                        data.get("plc_watchdog_node") or "",
                        event.get("event_edge") or "event",
                        _json_text(event.get("event_value")),
                        _json_text(event.get("previous_event_value")),
                        _json_text(event.get("watchdog_value")),
                        event.get("event_source_timestamp"),
                        event.get("event_server_timestamp"),
                        event.get("read_utc") or data.get("saved_at") or utc_now(),
                        data.get("first_frame_utc") or event.get("read_utc"),
                        data.get("last_frame_utc") or data.get("saved_at"),
                        data.get("stop_reason") or "fixed_8_second_window",
                        data.get("camera_source") or "",
                        canonical.get("original_width") if canonical else None,
                        canonical.get("original_height") if canonical else None,
                        data.get("first_frame_index"),
                        data.get("last_frame_index"),
                        data.get("first_frame_utc"),
                        data.get("last_frame_utc"),
                        detected_count,
                        valid_count,
                        legacy_sidecar_path,
                        utc_now(),
                    )
                    if existing is None:
                        connection.execute(
                            """
                            INSERT INTO measurement_event (
                                id, event_key, configuration_id, status,
                                plc_endpoint, plc_event_node, plc_watchdog_node, plc_edge,
                                plc_event_value, plc_previous_event_value, plc_watchdog_value,
                                plc_source_timestamp, plc_server_timestamp, app_received_at,
                                recording_started_at, recording_ended_at, stop_reason,
                                camera_source, camera_width, camera_height,
                                first_frame_index, last_frame_index,
                                first_frame_utc, last_frame_utc,
                                detected_piece_count, valid_piece_count,
                                legacy_sidecar_path, created_at, updated_at
                            )
                            VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                            )
                            """,
                            (
                                actual_event_id,
                                event_key,
                                *values[:-1],
                                values[-1],
                                values[-1],
                            ),
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE measurement_event
                            SET configuration_id = ?, status = ?,
                                plc_endpoint = ?, plc_event_node = ?,
                                plc_watchdog_node = ?, plc_edge = ?,
                                plc_event_value = ?, plc_previous_event_value = ?,
                                plc_watchdog_value = ?, plc_source_timestamp = ?,
                                plc_server_timestamp = ?, app_received_at = ?,
                                recording_started_at = ?, recording_ended_at = ?,
                                stop_reason = ?, camera_source = ?,
                                camera_width = ?, camera_height = ?,
                                first_frame_index = ?, last_frame_index = ?,
                                first_frame_utc = ?, last_frame_utc = ?,
                                detected_piece_count = ?, valid_piece_count = ?,
                                legacy_sidecar_path = ?, error_text = NULL,
                                updated_at = ?
                            WHERE id = ?
                            """,
                            (*values, actual_event_id),
                        )

                    connection.execute(
                        "UPDATE analysis_snapshot SET is_canonical = 0 WHERE event_id = ?",
                        (actual_event_id,),
                    )
                    snapshot_ids: dict[int, int] = {}
                    for snapshot in snapshots:
                        frame_index = int(snapshot["frame_index"])
                        summary = snapshot_summary(snapshot)
                        snapshot_values = (
                            snapshot.get("frame_utc"),
                            snapshot.get("processed_utc")
                            or data.get("saved_at")
                            or utc_now(),
                            snapshot.get("processing_duration_ms"),
                            int(frame_index == canonical_frame_index),
                            int(summary.get("detected_count") or 0),
                            int(summary.get("valid_count") or 0),
                            _json_text(summary),
                            _json_text(snapshot.get("box_rules") or {}),
                            _optional_relative_path(
                                snapshot.get("original_overlay_path"),
                                output_dir,
                            ),
                            _optional_relative_path(
                                snapshot.get("rectified_overlay_path"),
                                output_dir,
                            ),
                            _json_text(snapshot),
                        )
                        row = connection.execute(
                            """
                            SELECT id FROM analysis_snapshot
                            WHERE event_id = ? AND frame_index = ?
                            """,
                            (actual_event_id, frame_index),
                        ).fetchone()
                        if row is None:
                            cursor = connection.execute(
                                """
                                INSERT INTO analysis_snapshot (
                                    event_id, frame_index, frame_utc, processed_utc,
                                    processing_duration_ms, is_canonical,
                                    detected_piece_count, valid_piece_count,
                                    measurement_summary, box_rules,
                                    original_overlay_path, rectified_overlay_path,
                                    result_json
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (actual_event_id, frame_index, *snapshot_values),
                            )
                            snapshot_id = int(cursor.lastrowid)
                        else:
                            snapshot_id = int(row["id"])
                            connection.execute(
                                """
                                UPDATE analysis_snapshot
                                SET frame_utc = ?, processed_utc = ?,
                                    processing_duration_ms = ?, is_canonical = ?,
                                    detected_piece_count = ?, valid_piece_count = ?,
                                    measurement_summary = ?, box_rules = ?,
                                    original_overlay_path = ?,
                                    rectified_overlay_path = ?, result_json = ?
                                WHERE id = ?
                                """,
                                (*snapshot_values, snapshot_id),
                            )
                        snapshot_ids[frame_index] = snapshot_id

                    canonical_snapshot_id = (
                        snapshot_ids.get(canonical_frame_index)
                        if canonical_frame_index is not None
                        else None
                    )
                    if canonical is not None and canonical_snapshot_id is not None:
                        for position, piece in enumerate(
                            snapshot_pieces(canonical),
                            start=1,
                        ):
                            piece_number = int(piece.get("piece_id") or position)
                            measurement = (
                                piece.get("measurement")
                                if isinstance(piece.get("measurement"), dict)
                                else {}
                            )
                            sobel = (
                                piece.get("sobel")
                                if isinstance(piece.get("sobel"), dict)
                                else {}
                            )
                            box = (
                                piece.get("box")
                                if isinstance(piece.get("box"), dict)
                                else {}
                            )
                            piece_id = str(
                                uuid.uuid5(
                                    uuid.UUID(actual_event_id),
                                    f"piece:{piece_number}",
                                )
                            )
                            piece_values = (
                                canonical_snapshot_id,
                                int(bool(piece.get("valid"))),
                                int(not bool(piece.get("valid"))),
                                piece.get("confidence", box.get("conf")),
                                sobel.get("edge_confidence"),
                                sobel.get("crm_px"),
                                measurement.get("delta_px"),
                                measurement.get("delta_in"),
                                measurement.get("measurement_in"),
                                _json_text(box),
                                _json_text(sobel),
                                _json_text(piece),
                                utc_now(),
                            )
                            existing_piece = connection.execute(
                                """
                                SELECT id FROM piece_measurement
                                WHERE event_id = ? AND piece_number = ?
                                """,
                                (actual_event_id, piece_number),
                            ).fetchone()
                            if existing_piece is None:
                                connection.execute(
                                    """
                                    INSERT INTO piece_measurement (
                                        id, event_id, snapshot_id, piece_number,
                                        is_valid, review_required,
                                        yolo_confidence, sobel_confidence,
                                        crm_px, delta_px, distance_to_reference_in,
                                        automatic_measurement_in,
                                        box_json, sobel_json, automatic_result_json,
                                        created_at, updated_at
                                    )
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        piece_id,
                                        actual_event_id,
                                        piece_values[0],
                                        piece_number,
                                        *piece_values[1:-1],
                                        piece_values[-1],
                                        piece_values[-1],
                                    ),
                                )
                            else:
                                connection.execute(
                                    """
                                    UPDATE piece_measurement
                                    SET snapshot_id = ?, is_valid = ?,
                                        review_required = ?, yolo_confidence = ?,
                                        sobel_confidence = ?, crm_px = ?, delta_px = ?,
                                        distance_to_reference_in = ?,
                                        automatic_measurement_in = ?, box_json = ?,
                                        sobel_json = ?, automatic_result_json = ?,
                                        updated_at = ?
                                    WHERE id = ?
                                    """,
                                    (*piece_values, str(existing_piece["id"])),
                                )

                    for asset in _sidecar_assets(
                        actual_event_id,
                        sidecar_path,
                        data,
                        snapshots,
                        output_dir,
                    ):
                        existing_asset = connection.execute(
                            """
                            SELECT id FROM event_asset
                            WHERE event_id = ? AND asset_type = ? AND relative_path = ?
                            """,
                            (
                                asset["event_id"],
                                asset["asset_type"],
                                asset["relative_path"],
                            ),
                        ).fetchone()
                        if existing_asset is None:
                            connection.execute(
                                """
                                INSERT INTO event_asset (
                                    event_id, asset_type, relative_path, mime_type,
                                    size_bytes, sha256, created_at
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    asset["event_id"],
                                    asset["asset_type"],
                                    asset["relative_path"],
                                    asset["mime_type"],
                                    asset["size_bytes"],
                                    asset["sha256"],
                                    utc_now(),
                                ),
                            )
                        else:
                            connection.execute(
                                """
                                UPDATE event_asset
                                SET mime_type = ?, size_bytes = ?, sha256 = ?
                                WHERE id = ?
                                """,
                                (
                                    asset["mime_type"],
                                    asset["size_bytes"],
                                    asset["sha256"],
                                    int(existing_asset["id"]),
                                ),
                            )
                    connection.execute(
                        """
                        UPDATE measurement_event
                        SET canonical_snapshot_id = ?, status = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (canonical_snapshot_id, status, utc_now(), actual_event_id),
                    )
                    return actual_event_id
            finally:
                connection.close()
        except (ValueError, FileNotFoundError):
            raise
        except Exception as exc:
            raise DatabaseUnavailable(
                f"Could not synchronize sidecar with SQLite: {exc}"
            ) from exc

    def list_measurement_events(
        self,
        *,
        limit: int = 50,
        before: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        try:
            connection = self._connect(timeout=5.0)
            try:
                rows = connection.execute(
                    """
                    SELECT
                        me.id, me.event_key, me.status, me.plc_edge,
                        me.plc_source_timestamp, me.app_received_at,
                        me.recording_started_at, me.recording_ended_at,
                        me.detected_piece_count, me.valid_piece_count,
                        me.error_text, me.created_at,
                        (
                            SELECT relative_path
                            FROM event_asset
                            WHERE event_id = me.id AND asset_type = 'video'
                            ORDER BY id
                            LIMIT 1
                        ) AS video_path,
                        (
                            SELECT count(*)
                            FROM piece_measurement pm
                            WHERE pm.event_id = me.id
                              AND pm.operator_measurement_in IS NOT NULL
                        ) AS operator_override_count
                    FROM measurement_event me
                    WHERE (? IS NULL OR me.created_at < ?)
                    ORDER BY me.created_at DESC
                    LIMIT ?
                    """,
                    (before, before, limit),
                ).fetchall()
                return [_decode_row(row) or {} for row in rows]
            finally:
                connection.close()
        except Exception as exc:
            raise DatabaseUnavailable(
                f"Could not list SQLite measurement events: {exc}"
            ) from exc

    def all_measurement_event_ids(self) -> set[str]:
        connection = self._connect(timeout=5.0)
        try:
            return {
                str(row["id"])
                for row in connection.execute(
                    "SELECT id FROM measurement_event"
                ).fetchall()
            }
        finally:
            connection.close()

    def get_measurement_event(self, event_id: str) -> dict[str, Any] | None:
        try:
            connection = self._connect(timeout=5.0)
            try:
                event = _decode_row(
                    connection.execute(
                        "SELECT * FROM measurement_event WHERE id = ?",
                        (event_id,),
                    ).fetchone()
                )
                if event is None:
                    return None
                configuration = _decode_row(
                    connection.execute(
                        "SELECT * FROM vision_configuration WHERE id = ?",
                        (event["configuration_id"],),
                    ).fetchone()
                )
                event["configuration"] = configuration
                event["snapshots"] = [
                    _decode_row(row) or {}
                    for row in connection.execute(
                        """
                        SELECT * FROM analysis_snapshot
                        WHERE event_id = ?
                        ORDER BY frame_index
                        """,
                        (event_id,),
                    ).fetchall()
                ]
                pieces = [
                    _decode_row(row) or {}
                    for row in connection.execute(
                        """
                        SELECT * FROM piece_measurement_effective
                        WHERE event_id = ?
                        ORDER BY piece_number
                        """,
                        (event_id,),
                    ).fetchall()
                ]
                for piece in pieces:
                    piece["revisions"] = [
                        _decode_row(row) or {}
                        for row in connection.execute(
                            """
                            SELECT * FROM piece_measurement_revision
                            WHERE piece_measurement_id = ?
                            ORDER BY revision DESC
                            """,
                            (piece["id"],),
                        ).fetchall()
                    ]
                event["pieces"] = pieces
                event["assets"] = [
                    _decode_row(row) or {}
                    for row in connection.execute(
                        """
                        SELECT * FROM event_asset
                        WHERE event_id = ?
                        ORDER BY id
                        """,
                        (event_id,),
                    ).fetchall()
                ]
                return event
            finally:
                connection.close()
        except Exception as exc:
            raise DatabaseUnavailable(
                f"Could not load SQLite measurement event: {exc}"
            ) from exc

    def set_operator_measurement(
        self,
        *,
        event_id: str,
        piece_id: str,
        measurement_in: Decimal,
        reason: str,
        operator_id: str,
        operator_display_name: str | None,
        expected_revision: int,
        source_ip: str | None,
    ) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("A reason is required")
        if not operator_id.strip():
            raise ValueError("An operator identity is required")
        if measurement_in < Decimal("0") or measurement_in > Decimal("2400"):
            raise ValueError("Measurement must be between 0 and 2400 inches")
        return self._change_operator_measurement(
            event_id=event_id,
            piece_id=piece_id,
            measurement_in=float(measurement_in),
            reason=reason,
            operator_id=operator_id,
            operator_display_name=operator_display_name,
            expected_revision=expected_revision,
            source_ip=source_ip,
            clear=False,
        )

    def clear_operator_measurement(
        self,
        *,
        event_id: str,
        piece_id: str,
        reason: str,
        operator_id: str,
        operator_display_name: str | None,
        expected_revision: int,
        source_ip: str | None,
    ) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("A reason is required")
        if not operator_id.strip():
            raise ValueError("An operator identity is required")
        return self._change_operator_measurement(
            event_id=event_id,
            piece_id=piece_id,
            measurement_in=None,
            reason=reason,
            operator_id=operator_id,
            operator_display_name=operator_display_name,
            expected_revision=expected_revision,
            source_ip=source_ip,
            clear=True,
        )

    def _change_operator_measurement(
        self,
        *,
        event_id: str,
        piece_id: str,
        measurement_in: float | None,
        reason: str,
        operator_id: str,
        operator_display_name: str | None,
        expected_revision: int,
        source_ip: str | None,
        clear: bool,
    ) -> dict[str, Any]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            piece = _decode_row(
                connection.execute(
                    """
                    SELECT * FROM piece_measurement
                    WHERE id = ? AND event_id = ?
                    """,
                    (piece_id, event_id),
                ).fetchone()
            )
            if piece is None:
                raise RecordNotFound("Piece measurement was not found")
            current_revision = int(piece["operator_revision"])
            if current_revision != int(expected_revision):
                raise RevisionConflict(
                    f"Expected revision {expected_revision}, "
                    f"current revision is {current_revision}"
                )
            previous = piece["operator_measurement_in"]
            if clear and previous is None:
                raise ValueError("This piece has no operator correction")
            new_revision = current_revision + 1
            action = "clear" if clear else ("set" if previous is None else "change")
            connection.execute(
                """
                INSERT INTO piece_measurement_revision (
                    piece_measurement_id, revision, action,
                    previous_operator_measurement_in,
                    new_operator_measurement_in,
                    automatic_measurement_in,
                    operator_id, operator_display_name,
                    reason, source_ip, changed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    piece["id"],
                    new_revision,
                    action,
                    previous,
                    measurement_in,
                    piece["automatic_measurement_in"],
                    operator_id.strip(),
                    operator_display_name,
                    reason.strip(),
                    source_ip,
                    utc_now(),
                ),
            )
            connection.execute(
                """
                UPDATE piece_measurement
                SET operator_measurement_in = ?, operator_revision = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (measurement_in, new_revision, utc_now(), piece["id"]),
            )
            updated = _decode_row(
                connection.execute(
                    "SELECT * FROM piece_measurement WHERE id = ?",
                    (piece["id"],),
                ).fetchone()
            )
            connection.commit()
            return updated or {}
        except (RevisionConflict, RecordNotFound, ValueError):
            connection.rollback()
            raise
        except Exception as exc:
            connection.rollback()
            raise DatabaseUnavailable(
                f"Could not save SQLite operator measurement: {exc}"
            ) from exc
        finally:
            connection.close()

    def operator_histories(self) -> list[dict[str, Any]]:
        """Return correction histories for a later PostgreSQL migration."""
        connection = self._connect(timeout=5.0)
        try:
            pieces = connection.execute(
                """
                SELECT id, event_id, piece_number,
                       operator_measurement_in, operator_revision
                FROM piece_measurement
                WHERE operator_revision > 0
                ORDER BY event_id, piece_number
                """
            ).fetchall()
            result = []
            for piece in pieces:
                revisions = [
                    _decode_row(row) or {}
                    for row in connection.execute(
                        """
                        SELECT * FROM piece_measurement_revision
                        WHERE piece_measurement_id = ?
                        ORDER BY revision
                        """,
                        (piece["id"],),
                    ).fetchall()
                ]
                result.append(
                    {
                        "piece_id": str(piece["id"]),
                        "event_id": str(piece["event_id"]),
                        "piece_number": int(piece["piece_number"]),
                        "operator_measurement_in": piece["operator_measurement_in"],
                        "operator_revision": int(piece["operator_revision"]),
                        "revisions": revisions,
                    }
                )
            return result
        finally:
            connection.close()
