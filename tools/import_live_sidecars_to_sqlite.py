from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.migrate_live_sidecars_to_postgres import (  # noqa: E402
    enrich_legacy_sidecar,
    sidecars,
)
from tx2_database import build_vision_configuration  # noqa: E402
from tx2_sqlite_database import SQLiteDatabaseRepository  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotently import Live MVP sidecars into temporary SQLite."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=960)
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def write_sidecar(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    candidates = sidecars(output_dir)
    if args.limit is not None:
        candidates = candidates[: max(0, args.limit)]
    vision_args = SimpleNamespace(
        output_dir=output_dir,
        model=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
    )
    try:
        configuration = build_vision_configuration(vision_args, ROOT)
    except Exception as exc:
        print(f"ERROR: Could not snapshot vision configuration: {exc}", file=sys.stderr)
        return 2

    prepared: list[tuple[Path, dict]] = []
    for path in candidates:
        try:
            data = enrich_legacy_sidecar(
                path,
                output_dir,
                configuration,
                write=False,
            )
            prepared.append((path, data))
        except Exception as exc:
            print(f"INVALID {path}: {exc}", file=sys.stderr)
    if args.dry_run:
        print(
            f"DRY RUN: {len(prepared)}/{len(candidates)} sidecars are ready "
            "for SQLite."
        )
        return 0 if len(prepared) == len(candidates) else 1

    repository = SQLiteDatabaseRepository(args.sqlite_path)
    try:
        repository.open()
        repository.validate_schema()
        imported = 0
        failed = 0
        for path, data in prepared:
            data["db_sync_status"] = "pending"
            data["db_sync_backend"] = None
            data["db_sync_error"] = ""
            write_sidecar(path, data)
            try:
                actual_event_id = repository.sync_sidecar(path, output_dir)
                data["event_id"] = actual_event_id
                data["db_sync_status"] = "synced"
                data["db_sync_backend"] = "sqlite"
                write_sidecar(path, data)
                imported += 1
                print(f"IMPORTED {path.relative_to(output_dir)}")
            except Exception as exc:
                data["db_sync_status"] = "pending"
                data["db_sync_error"] = str(exc)
                write_sidecar(path, data)
                failed += 1
                print(
                    f"PENDING {path.relative_to(output_dir)}: {exc}",
                    file=sys.stderr,
                )
        print(
            f"Imported {imported}/{len(candidates)} sidecars into SQLite; "
            f"{failed} pending."
        )
        return 0 if failed == 0 and imported == len(candidates) else 1
    except Exception as exc:
        print(f"ERROR: SQLite import failed: {exc}", file=sys.stderr)
        return 1
    finally:
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
