from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tx2_database import DatabaseRepository, resolve_asset_path  # noqa: E402
from tx2_sqlite_database import SQLiteDatabaseRepository  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently migrate the temporary Live MVP SQLite history "
            "and operator audit revisions to PostgreSQL."
        )
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path(
            os.environ.get(
                "TX2_SQLITE_PATH",
                str(ROOT / "outputs" / "tx2_live_mvp.sqlite3"),
            )
        ),
    )
    parser.add_argument("--dsn", default=os.environ.get("TX2_POSTGRES_DSN", ""))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def update_sidecar_backend(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["db_sync_status"] = "synced"
    data["db_sync_backend"] = "postgresql"
    data["db_sync_error"] = ""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    sqlite_path = args.sqlite_path.resolve()
    output_dir = args.output_dir.resolve()
    if not sqlite_path.is_file():
        print(f"ERROR: SQLite database does not exist: {sqlite_path}", file=sys.stderr)
        return 2

    source = SQLiteDatabaseRepository(sqlite_path)
    try:
        source.open()
        source.validate_schema()
        events = source.list_measurement_events(limit=100)
        histories = source.operator_histories()
        sidecars: dict[str, Path] = {}
        missing_sidecars = []
        for event in events:
            detail = source.get_measurement_event(str(event["id"])) or {}
            asset = next(
                (
                    item
                    for item in detail.get("assets", [])
                    if item.get("asset_type") == "sidecar"
                ),
                None,
            )
            if asset is None:
                missing_sidecars.append(str(event["id"]))
                continue
            sidecar = resolve_asset_path(str(asset["relative_path"]), output_dir)
            if not sidecar.is_file():
                missing_sidecars.append(str(event["id"]))
                continue
            sidecars[str(event["id"])] = sidecar

        if missing_sidecars:
            print(
                "ERROR: Events without a readable sidecar: "
                + ", ".join(missing_sidecars),
                file=sys.stderr,
            )
            return 1
        if args.dry_run:
            print(
                f"DRY RUN: {len(events)} events, {len(sidecars)} sidecars and "
                f"{sum(len(item['revisions']) for item in histories)} operator "
                "revisions are ready for PostgreSQL."
            )
            return 0
        if not args.dsn.strip():
            print("ERROR: TX2_POSTGRES_DSN is required.", file=sys.stderr)
            return 2

        target = DatabaseRepository(args.dsn)
        target.open()
        try:
            target.validate_schema()
            event_id_map: dict[str, str] = {}
            imported_sidecars: list[Path] = []
            for source_event_id, sidecar in sidecars.items():
                target_event_id = target.sync_sidecar(sidecar, output_dir)
                event_id_map[source_event_id] = target_event_id
                imported_sidecars.append(sidecar)
                print(f"IMPORTED {sidecar.relative_to(output_dir)}")

            revision_count = 0
            for history in histories:
                source_event_id = str(history["event_id"])
                target_event_id = event_id_map[source_event_id]
                target_piece_id = str(
                    uuid.uuid5(
                        uuid.UUID(target_event_id),
                        f"piece:{int(history['piece_number'])}",
                    )
                )
                target.import_operator_history(
                    {
                        **history,
                        "event_id": target_event_id,
                        "piece_id": target_piece_id,
                    }
                )
                revision_count += len(history["revisions"])

            for sidecar in imported_sidecars:
                update_sidecar_backend(sidecar)
            print(
                f"Migrated {len(imported_sidecars)} events and "
                f"{revision_count} operator revisions to PostgreSQL."
            )
            return 0
        finally:
            target.close()
    except Exception as exc:
        print(f"ERROR: SQLite migration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
