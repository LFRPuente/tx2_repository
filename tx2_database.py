"""PostgreSQL persistence and sidecar reconciliation for the TX2 Live MVP."""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


CANONICAL_WINDOW_SECONDS = 1.5
EVENT_NAMESPACE = uuid.UUID("a89ad35f-5031-4a80-8d8f-0d30248eb401")


class DatabaseUnavailable(RuntimeError):
    """Raised when PostgreSQL cannot be reached."""


class RevisionConflict(RuntimeError):
    """Raised when an operator edit uses a stale revision."""


class RecordNotFound(RuntimeError):
    """Raised when an event or piece does not exist."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def app_commit_sha(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def build_vision_configuration(args: Any, root: Path) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    homography_path = output_dir / "homography_selection.json"
    calibration_path = output_dir / "table_measurement_calibration.json"
    model_path = Path(args.model).resolve()
    for path in (homography_path, calibration_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required vision artifact is missing: {path}")
    return {
        "homography_sha256": file_sha256(homography_path),
        "calibration_sha256": file_sha256(calibration_path),
        "model_sha256": file_sha256(model_path),
        "homography_json": load_json_file(homography_path),
        "calibration_json": load_json_file(calibration_path),
        "model_path": str(model_path),
        "model_name": model_path.name,
        "confidence_threshold": float(args.conf),
        "inference_size": int(args.imgsz),
        "app_commit_sha": app_commit_sha(root),
    }


def build_event_key(event: dict[str, Any], endpoint: str, event_node: str) -> str:
    identity = {
        "endpoint": endpoint,
        "event_node": event_node,
        "edge": event.get("event_edge"),
        "source_timestamp": event.get("event_source_timestamp"),
        "server_timestamp": event.get("event_server_timestamp"),
        "read_utc": event.get("read_utc"),
        "event_value": event.get("event_value"),
        "previous_event_value": event.get("previous_event_value"),
        "watchdog_value": event.get("watchdog_value"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def event_uuid(event_key: str) -> str:
    return str(uuid.uuid5(EVENT_NAMESPACE, event_key))


def snapshot_pieces(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    pieces = snapshot.get("pieces")
    if isinstance(pieces, list):
        return [piece for piece in pieces if isinstance(piece, dict)]
    measurement = snapshot.get("measurement")
    if not isinstance(measurement, dict):
        return []
    sobel = snapshot.get("sobel") if isinstance(snapshot.get("sobel"), dict) else {}
    boxes = snapshot.get("boxes") if isinstance(snapshot.get("boxes"), list) else []
    box = boxes[0] if boxes and isinstance(boxes[0], dict) else {}
    return [
        {
            "piece_id": 1,
            "box": box,
            "confidence": box.get("conf"),
            "sobel": sobel,
            "measurement": measurement,
            "valid": bool(sobel.get("is_valid", measurement is not None)),
        }
    ]


def snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    summary = snapshot.get("measurement_summary")
    if isinstance(summary, dict):
        return summary
    pieces = snapshot_pieces(snapshot)
    valid_count = sum(
        1
        for piece in pieces
        if piece.get("valid") and isinstance(piece.get("measurement"), dict)
    )
    return {
        "detected_count": len(pieces),
        "valid_count": valid_count,
        "invalid_count": len(pieces) - valid_count,
    }


def _snapshot_edge_confidence(snapshot: dict[str, Any]) -> float:
    values = []
    for piece in snapshot_pieces(snapshot):
        sobel = piece.get("sobel")
        if not isinstance(sobel, dict):
            continue
        value = sobel.get("edge_confidence")
        if value is not None:
            values.append(float(value))
    if not values and isinstance(snapshot.get("sobel"), dict):
        value = snapshot["sobel"].get("edge_confidence")
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else 0.0


def select_canonical_snapshot(
    snapshots: list[dict[str, Any]],
    event_monotonic: float | None = None,
    window_seconds: float = CANONICAL_WINDOW_SECONDS,
) -> dict[str, Any] | None:
    if not snapshots:
        return None

    candidates = []
    if event_monotonic is not None:
        for snapshot in snapshots:
            value = snapshot.get("frame_monotonic")
            if value is None:
                continue
            delta = float(value) - float(event_monotonic)
            if 0.0 <= delta <= float(window_seconds):
                candidates.append(snapshot)
    if not candidates:
        candidates = snapshots

    def score(snapshot: dict[str, Any]) -> tuple[float, float, float, float]:
        summary = snapshot_summary(snapshot)
        valid_count = int(summary.get("valid_count") or 0)
        detected_count = int(summary.get("detected_count") or len(snapshot_pieces(snapshot)))
        invalid_count = int(summary.get("invalid_count") or max(0, detected_count - valid_count))
        edge_confidence = _snapshot_edge_confidence(snapshot)
        frame_monotonic = snapshot.get("frame_monotonic")
        proximity = (
            abs(float(frame_monotonic) - float(event_monotonic))
            if frame_monotonic is not None and event_monotonic is not None
            else float(snapshot.get("snapshot_index") or 0)
        )
        return (-valid_count, invalid_count, -edge_confidence, proximity)

    return min(candidates, key=score)


def relative_asset_path(path: str | Path, output_dir: Path) -> str:
    root = output_dir.resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Asset path is outside the configured output directory: {resolved}") from exc
    return relative.as_posix()


def resolve_asset_path(relative_path: str, output_dir: Path) -> Path:
    root = output_dir.resolve()
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Asset path escapes the configured output directory") from exc
    return candidate


def _asset_metadata(
    event_id: str,
    asset_type: str,
    path: Path,
    output_dir: Path,
    mime_type: str,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "asset_type": asset_type,
        "relative_path": relative_asset_path(path, output_dir),
        "mime_type": mime_type,
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": file_sha256(path) if path.is_file() else None,
    }


def _json_value(value: Any) -> Any:
    try:
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise DatabaseUnavailable(
            "PostgreSQL support is not installed. Install requirements.txt."
        ) from exc
    return Jsonb(value)


def _row_dict(row: Any) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


@dataclass
class DatabaseHealth:
    enabled: bool
    ok: bool
    error: str = ""
    database: str | None = None
    user: str | None = None
    timezone: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ok": self.ok,
            "error": self.error,
            "database": self.database,
            "user": self.user,
            "timezone": self.timezone,
        }


class DatabaseRepository:
    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 4) -> None:
        if not dsn.strip():
            raise ValueError("TX2_POSTGRES_DSN is required unless --db-disabled is used")
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise DatabaseUnavailable(
                "PostgreSQL support is not installed. Install requirements.txt."
            ) from exc
        self.dsn = dsn
        self.pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row, "autocommit": False},
            open=False,
            name="tx2-live-mvp",
        )

    def open(self, timeout: float = 8.0) -> None:
        try:
            self.pool.open(wait=True, timeout=timeout)
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not connect to PostgreSQL: {exc}") from exc

    def close(self) -> None:
        self.pool.close()

    def health(self) -> DatabaseHealth:
        try:
            with self.pool.connection(timeout=5.0) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT current_database() AS database, current_user AS user, "
                        "current_setting('TimeZone') AS timezone"
                    )
                    row = cursor.fetchone()
            return DatabaseHealth(
                enabled=True,
                ok=True,
                database=row["database"],
                user=row["user"],
                timezone=row["timezone"],
            )
        except Exception as exc:
            return DatabaseHealth(enabled=True, ok=False, error=str(exc))

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
            with self.pool.connection(timeout=5.0) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = current_schema()
                          AND table_name = ANY(%s)
                        """,
                        (list(required),),
                    )
                    present = {row["table_name"] for row in cursor.fetchall()}
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not inspect PostgreSQL schema: {exc}") from exc
        missing = sorted(required - present)
        if missing:
            raise DatabaseUnavailable(
                "PostgreSQL migrations have not been applied; missing: " + ", ".join(missing)
            )

    def recover_stale_measurement_events(self, older_than_seconds: int = 60) -> int:
        threshold = max(30, int(older_than_seconds))
        try:
            with self.pool.connection(timeout=5.0) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE measurement_event
                        SET status = 'failed',
                            error_text = COALESCE(
                                error_text,
                                'Application restarted before the event was finalized'
                            ),
                            recording_ended_at = COALESCE(recording_ended_at, now()),
                            updated_at = now()
                        WHERE status IN ('recording', 'processing')
                          AND updated_at < now() - (%s * interval '1 second')
                        """,
                        (threshold,),
                    )
                    recovered = cursor.rowcount
                connection.commit()
            return int(recovered)
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not recover stale measurement events: {exc}") from exc

    def _configuration_id(self, cursor: Any, configuration: dict[str, Any]) -> str:
        configuration_id = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO vision_configuration (
                id, homography_sha256, calibration_sha256, model_sha256,
                homography_json, calibration_json, model_path, model_name,
                confidence_threshold, inference_size, app_commit_sha
            )
            VALUES (
                %(id)s, %(homography_sha256)s, %(calibration_sha256)s, %(model_sha256)s,
                %(homography_json)s, %(calibration_json)s, %(model_path)s, %(model_name)s,
                %(confidence_threshold)s, %(inference_size)s, %(app_commit_sha)s
            )
            ON CONFLICT (
                homography_sha256, calibration_sha256, model_sha256,
                confidence_threshold, inference_size
            )
            DO UPDATE SET
                model_path = EXCLUDED.model_path,
                model_name = EXCLUDED.model_name,
                app_commit_sha = EXCLUDED.app_commit_sha
            RETURNING id
            """,
            {
                **configuration,
                "id": configuration_id,
                "homography_json": _json_value(configuration["homography_json"]),
                "calibration_json": _json_value(configuration["calibration_json"]),
            },
        )
        return str(cursor.fetchone()["id"])

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
            with self.pool.connection(timeout=5.0) as connection:
                with connection.transaction():
                    with connection.cursor() as cursor:
                        configuration_id = self._configuration_id(cursor, configuration)
                        cursor.execute(
                            """
                            INSERT INTO measurement_event (
                                id, event_key, configuration_id, status,
                                plc_endpoint, plc_event_node, plc_watchdog_node, plc_edge,
                                plc_event_value, plc_previous_event_value, plc_watchdog_value,
                                plc_source_timestamp, plc_server_timestamp, app_received_at,
                                recording_started_at, camera_source
                            )
                            VALUES (
                                %(id)s, %(event_key)s, %(configuration_id)s, 'recording',
                                %(plc_endpoint)s, %(plc_event_node)s, %(plc_watchdog_node)s, %(plc_edge)s,
                                %(plc_event_value)s, %(plc_previous_event_value)s, %(plc_watchdog_value)s,
                                %(plc_source_timestamp)s, %(plc_server_timestamp)s, %(app_received_at)s,
                                %(recording_started_at)s, %(camera_source)s
                            )
                            ON CONFLICT (event_key)
                            DO UPDATE SET updated_at = now()
                            RETURNING id
                            """,
                            {
                                "id": event_id,
                                "event_key": event_key,
                                "configuration_id": configuration_id,
                                "plc_endpoint": plc_endpoint,
                                "plc_event_node": plc_event_node,
                                "plc_watchdog_node": plc_watchdog_node,
                                "plc_edge": event.get("event_edge") or "event",
                                "plc_event_value": _json_value(event.get("event_value")),
                                "plc_previous_event_value": _json_value(event.get("previous_event_value")),
                                "plc_watchdog_value": _json_value(event.get("watchdog_value")),
                                "plc_source_timestamp": event.get("event_source_timestamp"),
                                "plc_server_timestamp": event.get("event_server_timestamp"),
                                "app_received_at": event.get("read_utc") or utc_now(),
                                "recording_started_at": event.get("read_utc") or utc_now(),
                                "camera_source": camera_source,
                            },
                        )
                        return str(cursor.fetchone()["id"])
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not create measurement event: {exc}") from exc

    def mark_measurement_event_failed(self, event_id: str, error_text: str) -> None:
        try:
            with self.pool.connection(timeout=5.0) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE measurement_event
                        SET status = 'failed',
                            error_text = %s,
                            recording_ended_at = now(),
                            updated_at = now()
                        WHERE id = %s
                        """,
                        (error_text[:4000], event_id),
                    )
                connection.commit()
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not mark measurement event failed: {exc}") from exc

    def mark_measurement_event_processing(
        self,
        event_id: str,
        recording_ended_at: str,
    ) -> None:
        try:
            with self.pool.connection(timeout=5.0) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE measurement_event
                        SET status = 'processing',
                            recording_ended_at = %s,
                            updated_at = now()
                        WHERE id = %s
                        """,
                        (recording_ended_at, event_id),
                    )
                connection.commit()
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not mark measurement event processing: {exc}") from exc

    def delete_measurement_event(self, event_id: str) -> None:
        try:
            with self.pool.connection(timeout=5.0) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("DELETE FROM measurement_event WHERE id = %s", (event_id,))
                connection.commit()
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not delete retained measurement event: {exc}") from exc

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
        event_key = str(data.get("event_key") or build_event_key(
            event,
            str(data.get("plc_endpoint") or ""),
            str(data.get("plc_event_node") or ""),
        ))
        requested_event_id = str(data.get("event_id") or event_uuid(event_key))
        snapshots = [
            snapshot
            for snapshot in data.get("processing_snapshots", [])
            if isinstance(snapshot, dict)
        ]
        canonical = select_canonical_snapshot(
            snapshots,
            event_monotonic=event.get("event_read_monotonic"),
        )
        canonical_frame_index = int(canonical["frame_index"]) if canonical else None
        canonical_summary = snapshot_summary(canonical) if canonical else {
            "detected_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
        }
        valid_count = int(canonical_summary.get("valid_count") or 0)
        detected_count = int(canonical_summary.get("detected_count") or 0)
        invalid_count = int(canonical_summary.get("invalid_count") or max(0, detected_count - valid_count))
        status = "complete" if valid_count > 0 and invalid_count == 0 else "needs_review"
        legacy_sidecar_path = relative_asset_path(sidecar_path, output_dir)

        try:
            with self.pool.connection(timeout=8.0) as connection:
                with connection.transaction():
                    with connection.cursor() as cursor:
                        configuration_id = self._configuration_id(cursor, configuration)
                        cursor.execute(
                            """
                            INSERT INTO measurement_event (
                                id, event_key, configuration_id, status,
                                plc_endpoint, plc_event_node, plc_watchdog_node, plc_edge,
                                plc_event_value, plc_previous_event_value, plc_watchdog_value,
                                plc_source_timestamp, plc_server_timestamp, app_received_at,
                                recording_started_at, recording_ended_at, stop_reason,
                                camera_source, camera_width, camera_height,
                                first_frame_index, last_frame_index, first_frame_utc, last_frame_utc,
                                detected_piece_count, valid_piece_count, legacy_sidecar_path
                            )
                            VALUES (
                                %(id)s, %(event_key)s, %(configuration_id)s, %(status)s,
                                %(plc_endpoint)s, %(plc_event_node)s, %(plc_watchdog_node)s, %(plc_edge)s,
                                %(plc_event_value)s, %(plc_previous_event_value)s, %(plc_watchdog_value)s,
                                %(plc_source_timestamp)s, %(plc_server_timestamp)s, %(app_received_at)s,
                                %(recording_started_at)s, %(recording_ended_at)s, %(stop_reason)s,
                                %(camera_source)s, %(camera_width)s, %(camera_height)s,
                                %(first_frame_index)s, %(last_frame_index)s, %(first_frame_utc)s, %(last_frame_utc)s,
                                %(detected_piece_count)s, %(valid_piece_count)s, %(legacy_sidecar_path)s
                            )
                            ON CONFLICT (event_key)
                            DO UPDATE SET
                                configuration_id = EXCLUDED.configuration_id,
                                status = EXCLUDED.status,
                                recording_ended_at = EXCLUDED.recording_ended_at,
                                stop_reason = EXCLUDED.stop_reason,
                                camera_width = EXCLUDED.camera_width,
                                camera_height = EXCLUDED.camera_height,
                                first_frame_index = EXCLUDED.first_frame_index,
                                last_frame_index = EXCLUDED.last_frame_index,
                                first_frame_utc = EXCLUDED.first_frame_utc,
                                last_frame_utc = EXCLUDED.last_frame_utc,
                                detected_piece_count = EXCLUDED.detected_piece_count,
                                valid_piece_count = EXCLUDED.valid_piece_count,
                                legacy_sidecar_path = EXCLUDED.legacy_sidecar_path,
                                error_text = NULL,
                                updated_at = now()
                            RETURNING id
                            """,
                            {
                                "id": requested_event_id,
                                "event_key": event_key,
                                "configuration_id": configuration_id,
                                "status": status,
                                "plc_endpoint": data.get("plc_endpoint") or "",
                                "plc_event_node": data.get("plc_event_node") or "",
                                "plc_watchdog_node": data.get("plc_watchdog_node") or "",
                                "plc_edge": event.get("event_edge") or "event",
                                "plc_event_value": _json_value(event.get("event_value")),
                                "plc_previous_event_value": _json_value(event.get("previous_event_value")),
                                "plc_watchdog_value": _json_value(event.get("watchdog_value")),
                                "plc_source_timestamp": event.get("event_source_timestamp"),
                                "plc_server_timestamp": event.get("event_server_timestamp"),
                                "app_received_at": event.get("read_utc") or data.get("saved_at") or utc_now(),
                                "recording_started_at": data.get("first_frame_utc") or event.get("read_utc"),
                                "recording_ended_at": data.get("last_frame_utc") or data.get("saved_at"),
                                "stop_reason": data.get("stop_reason") or "fixed_8_second_window",
                                "camera_source": data.get("camera_source") or "",
                                "camera_width": canonical.get("original_width") if canonical else None,
                                "camera_height": canonical.get("original_height") if canonical else None,
                                "first_frame_index": data.get("first_frame_index"),
                                "last_frame_index": data.get("last_frame_index"),
                                "first_frame_utc": data.get("first_frame_utc"),
                                "last_frame_utc": data.get("last_frame_utc"),
                                "detected_piece_count": detected_count,
                                "valid_piece_count": valid_count,
                                "legacy_sidecar_path": legacy_sidecar_path,
                            },
                        )
                        actual_event_id = str(cursor.fetchone()["id"])
                        cursor.execute(
                            "UPDATE analysis_snapshot SET is_canonical = false WHERE event_id = %s",
                            (actual_event_id,),
                        )
                        snapshot_ids: dict[int, int] = {}
                        for snapshot in snapshots:
                            frame_index = int(snapshot["frame_index"])
                            summary = snapshot_summary(snapshot)
                            cursor.execute(
                                """
                                INSERT INTO analysis_snapshot (
                                    event_id, frame_index, frame_utc, processed_utc,
                                    processing_duration_ms, is_canonical,
                                    detected_piece_count, valid_piece_count,
                                    measurement_summary, box_rules,
                                    original_overlay_path, rectified_overlay_path, result_json
                                )
                                VALUES (
                                    %(event_id)s, %(frame_index)s, %(frame_utc)s, %(processed_utc)s,
                                    %(processing_duration_ms)s, %(is_canonical)s,
                                    %(detected_piece_count)s, %(valid_piece_count)s,
                                    %(measurement_summary)s, %(box_rules)s,
                                    %(original_overlay_path)s, %(rectified_overlay_path)s, %(result_json)s
                                )
                                ON CONFLICT (event_id, frame_index)
                                DO UPDATE SET
                                    frame_utc = EXCLUDED.frame_utc,
                                    processed_utc = EXCLUDED.processed_utc,
                                    processing_duration_ms = EXCLUDED.processing_duration_ms,
                                    is_canonical = EXCLUDED.is_canonical,
                                    detected_piece_count = EXCLUDED.detected_piece_count,
                                    valid_piece_count = EXCLUDED.valid_piece_count,
                                    measurement_summary = EXCLUDED.measurement_summary,
                                    box_rules = EXCLUDED.box_rules,
                                    original_overlay_path = EXCLUDED.original_overlay_path,
                                    rectified_overlay_path = EXCLUDED.rectified_overlay_path,
                                    result_json = EXCLUDED.result_json
                                RETURNING id
                                """,
                                {
                                    "event_id": actual_event_id,
                                    "frame_index": frame_index,
                                    "frame_utc": snapshot.get("frame_utc"),
                                    "processed_utc": snapshot.get("processed_utc") or data.get("saved_at") or utc_now(),
                                    "processing_duration_ms": snapshot.get("processing_duration_ms"),
                                    "is_canonical": frame_index == canonical_frame_index,
                                    "detected_piece_count": int(summary.get("detected_count") or 0),
                                    "valid_piece_count": int(summary.get("valid_count") or 0),
                                    "measurement_summary": _json_value(summary),
                                    "box_rules": _json_value(snapshot.get("box_rules") or {}),
                                    "original_overlay_path": _optional_relative_path(
                                        snapshot.get("original_overlay_path"), output_dir
                                    ),
                                    "rectified_overlay_path": _optional_relative_path(
                                        snapshot.get("rectified_overlay_path"), output_dir
                                    ),
                                    "result_json": _json_value(snapshot),
                                },
                            )
                            snapshot_ids[frame_index] = int(cursor.fetchone()["id"])

                        canonical_snapshot_id = (
                            snapshot_ids.get(canonical_frame_index)
                            if canonical_frame_index is not None
                            else None
                        )
                        if canonical is not None and canonical_snapshot_id is not None:
                            for position, piece in enumerate(snapshot_pieces(canonical), start=1):
                                piece_number = int(piece.get("piece_id") or position)
                                measurement = piece.get("measurement")
                                measurement = measurement if isinstance(measurement, dict) else {}
                                sobel = piece.get("sobel")
                                sobel = sobel if isinstance(sobel, dict) else {}
                                box = piece.get("box")
                                box = box if isinstance(box, dict) else {}
                                automatic_measurement = measurement.get("measurement_in")
                                cursor.execute(
                                    """
                                    INSERT INTO piece_measurement (
                                        id, event_id, snapshot_id, piece_number,
                                        is_valid, review_required, yolo_confidence, sobel_confidence,
                                        crm_px, delta_px, distance_to_reference_in,
                                        automatic_measurement_in, box_json, sobel_json,
                                        automatic_result_json
                                    )
                                    VALUES (
                                        %(id)s, %(event_id)s, %(snapshot_id)s, %(piece_number)s,
                                        %(is_valid)s, %(review_required)s, %(yolo_confidence)s, %(sobel_confidence)s,
                                        %(crm_px)s, %(delta_px)s, %(distance_to_reference_in)s,
                                        %(automatic_measurement_in)s, %(box_json)s, %(sobel_json)s,
                                        %(automatic_result_json)s
                                    )
                                    ON CONFLICT (event_id, piece_number)
                                    DO UPDATE SET
                                        snapshot_id = EXCLUDED.snapshot_id,
                                        is_valid = EXCLUDED.is_valid,
                                        review_required = EXCLUDED.review_required,
                                        yolo_confidence = EXCLUDED.yolo_confidence,
                                        sobel_confidence = EXCLUDED.sobel_confidence,
                                        crm_px = EXCLUDED.crm_px,
                                        delta_px = EXCLUDED.delta_px,
                                        distance_to_reference_in = EXCLUDED.distance_to_reference_in,
                                        automatic_measurement_in = EXCLUDED.automatic_measurement_in,
                                        box_json = EXCLUDED.box_json,
                                        sobel_json = EXCLUDED.sobel_json,
                                        automatic_result_json = EXCLUDED.automatic_result_json,
                                        updated_at = now()
                                    """,
                                    {
                                        "id": str(uuid.uuid5(uuid.UUID(actual_event_id), f"piece:{piece_number}")),
                                        "event_id": actual_event_id,
                                        "snapshot_id": canonical_snapshot_id,
                                        "piece_number": piece_number,
                                        "is_valid": bool(piece.get("valid")),
                                        "review_required": not bool(piece.get("valid")),
                                        "yolo_confidence": piece.get("confidence", box.get("conf")),
                                        "sobel_confidence": sobel.get("edge_confidence"),
                                        "crm_px": sobel.get("crm_px"),
                                        "delta_px": measurement.get("delta_px"),
                                        "distance_to_reference_in": measurement.get("delta_in"),
                                        "automatic_measurement_in": automatic_measurement,
                                        "box_json": _json_value(box),
                                        "sobel_json": _json_value(sobel),
                                        "automatic_result_json": _json_value(piece),
                                    },
                                )

                        assets = list(_sidecar_assets(
                            actual_event_id,
                            sidecar_path,
                            data,
                            snapshots,
                            output_dir,
                        ))
                        for asset in assets:
                            cursor.execute(
                                """
                                INSERT INTO event_asset (
                                    event_id, asset_type, relative_path, mime_type,
                                    size_bytes, sha256
                                )
                                VALUES (
                                    %(event_id)s, %(asset_type)s, %(relative_path)s, %(mime_type)s,
                                    %(size_bytes)s, %(sha256)s
                                )
                                ON CONFLICT (event_id, asset_type, relative_path)
                                DO UPDATE SET
                                    mime_type = EXCLUDED.mime_type,
                                    size_bytes = EXCLUDED.size_bytes,
                                    sha256 = EXCLUDED.sha256
                                """,
                                asset,
                            )
                        cursor.execute(
                            """
                            UPDATE measurement_event
                            SET canonical_snapshot_id = %(canonical_snapshot_id)s,
                                status = %(status)s,
                                updated_at = now()
                            WHERE id = %(event_id)s
                            """,
                            {
                                "canonical_snapshot_id": canonical_snapshot_id,
                                "status": status,
                                "event_id": actual_event_id,
                            },
                        )
                        return actual_event_id
        except (ValueError, FileNotFoundError):
            raise
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not synchronize sidecar: {exc}") from exc

    def list_measurement_events(
        self,
        *,
        limit: int = 50,
        before: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        query = """
            SELECT
                me.id, me.event_key, me.status, me.plc_edge,
                me.plc_source_timestamp, me.app_received_at,
                me.recording_started_at, me.recording_ended_at,
                me.detected_piece_count, me.valid_piece_count,
                me.error_text, me.created_at,
                video.relative_path AS video_path,
                (
                    SELECT count(*)
                    FROM piece_measurement pm
                    WHERE pm.event_id = me.id
                      AND pm.operator_measurement_in IS NOT NULL
                ) AS operator_override_count
            FROM measurement_event me
            LEFT JOIN LATERAL (
                SELECT relative_path
                FROM event_asset
                WHERE event_id = me.id AND asset_type = 'video'
                ORDER BY id
                LIMIT 1
            ) video ON true
            WHERE (%(before)s::timestamptz IS NULL OR me.created_at < %(before)s::timestamptz)
            ORDER BY me.created_at DESC
            LIMIT %(limit)s
        """
        try:
            with self.pool.connection(timeout=5.0) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, {"limit": limit, "before": before})
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not list measurement events: {exc}") from exc

    def get_measurement_event(self, event_id: str) -> dict[str, Any] | None:
        try:
            with self.pool.connection(timeout=5.0) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT me.*, to_jsonb(vc) AS configuration
                        FROM measurement_event me
                        JOIN vision_configuration vc ON vc.id = me.configuration_id
                        WHERE me.id = %s
                        """,
                        (event_id,),
                    )
                    event = _row_dict(cursor.fetchone())
                    if event is None:
                        return None
                    cursor.execute(
                        """
                        SELECT *
                        FROM analysis_snapshot
                        WHERE event_id = %s
                        ORDER BY frame_index
                        """,
                        (event_id,),
                    )
                    event["snapshots"] = [dict(row) for row in cursor.fetchall()]
                    cursor.execute(
                        """
                        SELECT pme.*
                        FROM piece_measurement_effective pme
                        WHERE pme.event_id = %s
                        ORDER BY pme.piece_number
                        """,
                        (event_id,),
                    )
                    pieces = [dict(row) for row in cursor.fetchall()]
                    for piece in pieces:
                        cursor.execute(
                            """
                            SELECT *
                            FROM piece_measurement_revision
                            WHERE piece_measurement_id = %s
                            ORDER BY revision DESC
                            """,
                            (piece["id"],),
                        )
                        piece["revisions"] = [dict(row) for row in cursor.fetchall()]
                    event["pieces"] = pieces
                    cursor.execute(
                        "SELECT * FROM event_asset WHERE event_id = %s ORDER BY id",
                        (event_id,),
                    )
                    event["assets"] = [dict(row) for row in cursor.fetchall()]
                    return event
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not load measurement event: {exc}") from exc

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
        try:
            with self.pool.connection(timeout=5.0) as connection:
                with connection.transaction():
                    with connection.cursor() as cursor:
                        piece = self._lock_piece(cursor, event_id, piece_id)
                        current_revision = int(piece["operator_revision"])
                        if current_revision != int(expected_revision):
                            raise RevisionConflict(
                                f"Expected revision {expected_revision}, current revision is {current_revision}"
                            )
                        previous = piece["operator_measurement_in"]
                        new_revision = current_revision + 1
                        action = "set" if previous is None else "change"
                        cursor.execute(
                            """
                            INSERT INTO piece_measurement_revision (
                                piece_measurement_id, revision, action,
                                previous_operator_measurement_in, new_operator_measurement_in,
                                automatic_measurement_in, operator_id, operator_display_name,
                                reason, source_ip
                            )
                            VALUES (
                                %(piece_id)s, %(revision)s, %(action)s,
                                %(previous)s, %(new)s,
                                %(automatic)s, %(operator_id)s, %(operator_display_name)s,
                                %(reason)s, %(source_ip)s
                            )
                            """,
                            {
                                "piece_id": piece["id"],
                                "revision": new_revision,
                                "action": action,
                                "previous": previous,
                                "new": measurement_in,
                                "automatic": piece["automatic_measurement_in"],
                                "operator_id": operator_id.strip(),
                                "operator_display_name": operator_display_name,
                                "reason": reason.strip(),
                                "source_ip": source_ip,
                            },
                        )
                        cursor.execute(
                            """
                            UPDATE piece_measurement
                            SET operator_measurement_in = %s,
                                operator_revision = %s,
                                updated_at = now()
                            WHERE id = %s
                            RETURNING *
                            """,
                            (measurement_in, new_revision, piece["id"]),
                        )
                        return dict(cursor.fetchone())
        except (RevisionConflict, RecordNotFound, ValueError):
            raise
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not save operator measurement: {exc}") from exc

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
        try:
            with self.pool.connection(timeout=5.0) as connection:
                with connection.transaction():
                    with connection.cursor() as cursor:
                        piece = self._lock_piece(cursor, event_id, piece_id)
                        current_revision = int(piece["operator_revision"])
                        if current_revision != int(expected_revision):
                            raise RevisionConflict(
                                f"Expected revision {expected_revision}, current revision is {current_revision}"
                            )
                        if piece["operator_measurement_in"] is None:
                            raise ValueError("This piece has no operator correction")
                        new_revision = current_revision + 1
                        cursor.execute(
                            """
                            INSERT INTO piece_measurement_revision (
                                piece_measurement_id, revision, action,
                                previous_operator_measurement_in, new_operator_measurement_in,
                                automatic_measurement_in, operator_id, operator_display_name,
                                reason, source_ip
                            )
                            VALUES (
                                %(piece_id)s, %(revision)s, 'clear',
                                %(previous)s, NULL,
                                %(automatic)s, %(operator_id)s, %(operator_display_name)s,
                                %(reason)s, %(source_ip)s
                            )
                            """,
                            {
                                "piece_id": piece["id"],
                                "revision": new_revision,
                                "previous": piece["operator_measurement_in"],
                                "automatic": piece["automatic_measurement_in"],
                                "operator_id": operator_id.strip(),
                                "operator_display_name": operator_display_name,
                                "reason": reason.strip(),
                                "source_ip": source_ip,
                            },
                        )
                        cursor.execute(
                            """
                            UPDATE piece_measurement
                            SET operator_measurement_in = NULL,
                                operator_revision = %s,
                                updated_at = now()
                            WHERE id = %s
                            RETURNING *
                            """,
                            (new_revision, piece["id"]),
                        )
                        return dict(cursor.fetchone())
        except (RevisionConflict, RecordNotFound, ValueError):
            raise
        except Exception as exc:
            raise DatabaseUnavailable(f"Could not clear operator measurement: {exc}") from exc

    @staticmethod
    def _lock_piece(cursor: Any, event_id: str, piece_id: str) -> dict[str, Any]:
        cursor.execute(
            """
            SELECT *
            FROM piece_measurement
            WHERE id = %s AND event_id = %s
            FOR UPDATE
            """,
            (piece_id, event_id),
        )
        piece = cursor.fetchone()
        if piece is None:
            raise RecordNotFound("Piece measurement was not found")
        return dict(piece)


def _optional_relative_path(value: Any, output_dir: Path) -> str | None:
    if not value:
        return None
    return relative_asset_path(str(value), output_dir)


def _sidecar_assets(
    event_id: str,
    sidecar_path: Path,
    sidecar: dict[str, Any],
    snapshots: list[dict[str, Any]],
    output_dir: Path,
) -> Iterable[dict[str, Any]]:
    yield _asset_metadata(event_id, "sidecar", sidecar_path, output_dir, "application/json")
    video_path = sidecar.get("video_path")
    if video_path:
        path = Path(str(video_path))
        if path.is_file():
            yield _asset_metadata(event_id, "video", path, output_dir, "video/mp4")
    for snapshot in snapshots:
        for key, asset_type in (
            ("original_overlay_path", "original_overlay"),
            ("rectified_overlay_path", "rectified_overlay"),
        ):
            value = snapshot.get(key)
            if not value:
                continue
            path = Path(str(value))
            if path.is_file():
                yield _asset_metadata(event_id, asset_type, path, output_dir, "image/jpeg")
