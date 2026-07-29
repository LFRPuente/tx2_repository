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

from tx2_database import (  # noqa: E402
    DatabaseRepository,
    build_event_key,
    build_vision_configuration,
    event_uuid,
    load_json_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Idempotently import existing Live MVP sidecars into PostgreSQL."
    )
    parser.add_argument("--dsn", default=os.environ.get("TX2_POSTGRES_DSN", ""))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sidecars(output_dir: Path) -> list[Path]:
    root = output_dir / "live_plc_clips"
    return sorted(root.glob("*/*.json"))


def enrich_legacy_sidecar(
    path: Path,
    output_dir: Path,
    configuration: dict,
    *,
    write: bool,
) -> dict:
    data = load_json_file(path)
    event = data.get("event") if isinstance(data.get("event"), dict) else {}
    plc_endpoint = str(data.get("plc_endpoint") or "opc.tcp://10.14.6.48:49320")
    plc_event_node = str(
        data.get("plc_event_node")
        or "ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength"
    )
    event_key = str(data.get("event_key") or build_event_key(event, plc_endpoint, plc_event_node))
    data.setdefault("event_key", event_key)
    data.setdefault("event_id", event_uuid(event_key))
    data.setdefault("vision_configuration", configuration)
    data.setdefault("plc_endpoint", plc_endpoint)
    data.setdefault("plc_event_node", plc_event_node)
    data.setdefault(
        "plc_watchdog_node",
        "ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD",
    )
    data.setdefault("camera_source", "legacy_live_mvp")
    if write:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


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

    if args.dry_run:
        valid = 0
        for path in candidates:
            try:
                enrich_legacy_sidecar(path, output_dir, configuration, write=False)
                valid += 1
            except Exception as exc:
                print(f"INVALID {path}: {exc}", file=sys.stderr)
        print(f"DRY RUN: {valid}/{len(candidates)} sidecars are ready for import.")
        return 0 if valid == len(candidates) else 1

    if not args.dsn.strip():
        print("ERROR: TX2_POSTGRES_DSN is required unless --dry-run is used.", file=sys.stderr)
        return 2

    repository = DatabaseRepository(args.dsn)
    try:
        repository.open()
        imported = 0
        for path in candidates:
            data = enrich_legacy_sidecar(path, output_dir, configuration, write=False)
            data["db_sync_status"] = "synced"
            data["db_sync_error"] = ""
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            event_id = repository.sync_sidecar(path, output_dir)
            if event_id != data["event_id"]:
                data["event_id"] = event_id
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                repository.sync_sidecar(path, output_dir)
            imported += 1
            print(f"IMPORTED {path.relative_to(output_dir)}")
        print(f"Imported {imported}/{len(candidates)} sidecars.")
        return 0
    except Exception as exc:
        print(f"ERROR: Sidecar migration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
