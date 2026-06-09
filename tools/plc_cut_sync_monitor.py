#!/usr/bin/env python3
"""Synchronize PLC reads to VisionWD changes and watch for cut/measure events.

This monitor uses VisionWD as a heartbeat. Every time VisionWD changes, it reads
MeasureLength and records both OPC UA source timestamps. If MeasureLength is a
boolean trigger, rising/falling edges are logged as candidate event moments.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "opc.tcp://10.14.6.48:49320"
DEFAULT_WATCHDOG_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD"
DEFAULT_EVENT_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def clean(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return str(value)


async def read_node(client: Any, node_id: str) -> dict[str, Any]:
    node = client.get_node(node_id)
    data_value = await node.read_data_value()
    return {
        "node_id": node_id,
        "status": str(data_value.StatusCode),
        "value": clean(data_value.Value.Value),
        "source_timestamp": clean(data_value.SourceTimestamp),
        "server_timestamp": clean(data_value.ServerTimestamp),
        "read_utc": utc_now(),
    }


def event_edge(previous: Any, current: Any) -> str:
    if previous is None or previous == current:
        return ""
    if isinstance(previous, bool) and isinstance(current, bool):
        return "rising" if current else "falling"
    return "changed"


async def monitor(args: argparse.Namespace) -> dict[str, Any]:
    from asyncua import Client

    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    previous_watchdog: Any = None
    previous_event: Any = None
    deadline = asyncio.get_running_loop().time() + args.duration

    logging.info("Connecting to %s", args.endpoint)
    async with Client(url=args.endpoint, timeout=args.timeout) as client:
        logging.info("Connected")

        while asyncio.get_running_loop().time() < deadline:
            watchdog = await read_node(client, args.watchdog_node)
            watchdog_value = watchdog.get("value")

            if watchdog_value != previous_watchdog:
                event = await read_node(client, args.event_node)
                edge = event_edge(previous_event, event.get("value"))

                row = {
                    "index": len(rows),
                    "read_utc": utc_now(),
                    "watchdog_value": watchdog_value,
                    "watchdog_source_timestamp": watchdog.get("source_timestamp"),
                    "watchdog_status": watchdog.get("status"),
                    "event_value": event.get("value"),
                    "event_source_timestamp": event.get("source_timestamp"),
                    "event_status": event.get("status"),
                    "event_edge": edge,
                }
                rows.append(row)

                if edge:
                    events.append(row)
                    logging.warning(
                        "EVENT %s | %s=%r | source=%s | wd=%r | wd_source=%s",
                        edge,
                        args.event_label,
                        event.get("value"),
                        event.get("source_timestamp"),
                        watchdog_value,
                        watchdog.get("source_timestamp"),
                    )
                elif args.log_each_tick:
                    logging.info(
                        "WD tick | wd=%r | %s=%r | wd_source=%s | event_source=%s",
                        watchdog_value,
                        args.event_label,
                        event.get("value"),
                        watchdog.get("source_timestamp"),
                        event.get("source_timestamp"),
                    )

                previous_watchdog = watchdog_value
                previous_event = event.get("value")

            await asyncio.sleep(args.poll_interval)

    result = "EVENTS_FOUND" if events else "NO_EVENT"
    return {
        "run_utc": utc_now(),
        "endpoint": args.endpoint,
        "watchdog_node": args.watchdog_node,
        "event_node": args.event_node,
        "event_label": args.event_label,
        "duration_seconds": args.duration,
        "poll_interval_seconds": args.poll_interval,
        "watchdog_ticks": len(rows),
        "events_found": len(events),
        "result": result,
        "events": events,
        "rows": rows,
    }


def write_outputs(report: dict[str, Any], output_base: Path) -> tuple[Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = report["rows"]
    fieldnames = [
        "index",
        "read_utc",
        "watchdog_value",
        "watchdog_source_timestamp",
        "watchdog_status",
        "event_value",
        "event_source_timestamp",
        "event_status",
        "event_edge",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize PLC event reads to VisionWD changes.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--watchdog-node", default=DEFAULT_WATCHDOG_NODE)
    parser.add_argument("--event-node", default=DEFAULT_EVENT_NODE)
    parser.add_argument("--event-label", default="MeasureLength")
    parser.add_argument("--duration", type=float, default=60.0, help="Monitor duration in seconds.")
    parser.add_argument("--poll-interval", type=float, default=0.01, help="Watchdog polling interval.")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--output", type=Path, help="Output base path without extension.")
    parser.add_argument("--log-each-tick", action="store_true", help="Log every VisionWD tick.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    configure_logging()
    args = parse_args()

    if args.duration <= 0:
        logging.error("--duration must be > 0")
        return 2
    if args.poll_interval < 0:
        logging.error("--poll-interval must be >= 0")
        return 2

    try:
        report = asyncio.run(monitor(args))
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        return 130
    except Exception as exc:
        logging.exception("Monitor failed: %s", exc)
        return 1

    if args.output:
        output_base = args.output
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_base = Path("outputs") / f"plc_cut_sync_monitor_{stamp}"

    json_path, csv_path = write_outputs(report, output_base)
    logging.info("Watchdog ticks captured: %s", report["watchdog_ticks"])
    logging.info("Events found: %s", report["events_found"])
    logging.info("JSON saved: %s", json_path)
    logging.info("CSV saved: %s", csv_path)

    if report["events_found"]:
        logging.warning("RESULT: EVENTS_FOUND")
    else:
        logging.info("RESULT: NO_EVENT | No %s edge/change observed.", args.event_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
