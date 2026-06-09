#!/usr/bin/env python3
"""Monitor AS20 VisionSystem using VisionWD as the timing tick.

This is read-only. It polls VisionWD, records only value changes, and reads
MeasureLength on the same observed watchdog tick.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "opc.tcp://10.14.6.48:49320"
DEFAULT_MEASURE_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength"
DEFAULT_WATCHDOG_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def clean(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


async def read_value(client: Any, node_id: str) -> dict[str, Any]:
    node = client.get_node(node_id)
    data_value = await node.read_data_value()
    return {
        "node_id": node_id,
        "status": str(data_value.StatusCode),
        "value": clean(data_value.Value.Value),
        "source_timestamp": clean(data_value.SourceTimestamp),
        "server_timestamp": clean(data_value.ServerTimestamp),
        "read_utc": now_utc(),
    }


def append_csv(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


async def monitor(args: argparse.Namespace) -> list[dict[str, Any]]:
    from asyncua import Client

    events: list[dict[str, Any]] = []
    csv_path = Path(args.csv) if args.csv else None
    last_watchdog_value: Any = None
    start = datetime.now(timezone.utc)

    async with Client(url=args.endpoint, timeout=args.timeout) as client:
        while True:
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            if args.seconds > 0 and elapsed >= args.seconds:
                break

            watchdog = await read_value(client, args.watchdog_node)
            watchdog_value = watchdog["value"]

            if last_watchdog_value is None or watchdog_value != last_watchdog_value:
                measure = await read_value(client, args.measure_node)
                event = {
                    "event_utc": now_utc(),
                    "watchdog_value": watchdog_value,
                    "watchdog_status": watchdog["status"],
                    "watchdog_source_timestamp": watchdog["source_timestamp"],
                    "watchdog_read_utc": watchdog["read_utc"],
                    "measure_value": measure["value"],
                    "measure_status": measure["status"],
                    "measure_source_timestamp": measure["source_timestamp"],
                    "measure_read_utc": measure["read_utc"],
                }
                events.append(event)
                print(json.dumps(event, ensure_ascii=False))
                if csv_path:
                    append_csv(csv_path, event)
                last_watchdog_value = watchdog_value

            await asyncio.sleep(args.poll_seconds)

    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize AS20 reads to VisionWD changes.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--watchdog-node", default=DEFAULT_WATCHDOG_NODE)
    parser.add_argument("--measure-node", default=DEFAULT_MEASURE_NODE)
    parser.add_argument("--poll-seconds", type=float, default=0.069, help="Default: 0.069 seconds.")
    parser.add_argument("--seconds", type=float, default=60.0, help="Run duration. Use 0 to run forever.")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--csv", default="watchdog_sync_log.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asyncio.run(monitor(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
