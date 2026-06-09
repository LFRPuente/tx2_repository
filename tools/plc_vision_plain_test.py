#!/usr/bin/env python3
"""Plain Python OPC UA test for TX2 AS20 VisionSystem.

This is a notebook-free version of tools/plc_timestamp_probe.ipynb. It is
read-only: it connects to Kepware OPC UA, reads MeasureLength and VisionWD, then
samples VisionWD to confirm the watchdog is changing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "opc.tcp://10.14.6.48:49320"
DEFAULT_MEASURE_LENGTH_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.MeasureLength"
DEFAULT_VISION_WD_NODE = "ns=2;s=ControlLogix.AS20.VisionSystem.VisionWD"


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

    try:
        display_name = (await node.read_display_name()).Text
    except Exception:
        display_name = ""

    try:
        browse_name = (await node.read_browse_name()).Name
    except Exception:
        browse_name = ""

    return {
        "node_id": node_id,
        "browse_name": browse_name,
        "display_name": display_name,
        "status": str(data_value.StatusCode),
        "value": clean(data_value.Value.Value),
        "source_timestamp": clean(data_value.SourceTimestamp),
        "server_timestamp": clean(data_value.ServerTimestamp),
        "read_utc": utc_now(),
    }


def log_read(label: str, item: dict[str, Any]) -> None:
    if "error" in item:
        logging.error("%s read failed | node=%s | error=%s", label, item.get("node_id"), item["error"])
        return

    logging.info(
        "%s | value=%r | status=%s | source=%s | server=%s | node=%s",
        label,
        item.get("value"),
        item.get("status"),
        item.get("source_timestamp"),
        item.get("server_timestamp"),
        item.get("node_id"),
    )


async def safe_read(client: Any, label: str, node_id: str) -> dict[str, Any]:
    try:
        item = await read_node(client, node_id)
    except Exception as exc:
        item = {"node_id": node_id, "label": label, "error": str(exc), "read_utc": utc_now()}
    log_read(label, item)
    return item


async def sample_watchdog(client: Any, node_id: str, samples: int, interval: float) -> dict[str, Any]:
    logging.info("Sampling VisionWD | samples=%s | interval=%.3fs", samples, interval)

    rows: list[dict[str, Any]] = []
    previous_value: Any = None
    changes = 0

    for index in range(samples):
        item = await safe_read(client, f"VisionWD sample {index + 1:03d}", node_id)
        item["sample_index"] = index
        rows.append(item)

        value = item.get("value")
        if index > 0 and "error" not in item and value != previous_value:
            changes += 1
        if "error" not in item:
            previous_value = value

        if index + 1 < samples:
            await asyncio.sleep(interval)

    total_transitions = max(samples - 1, 0)
    changed = changes > 0
    logging.info("VisionWD changes detected: %s / %s", changes, total_transitions)

    return {
        "node_id": node_id,
        "samples": rows,
        "changes": changes,
        "total_transitions": total_transitions,
        "changed": changed,
    }


async def run_test(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from asyncua import Client
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: asyncua. Install it with: python -m pip install asyncua"
        ) from exc

    report: dict[str, Any] = {
        "run_utc": utc_now(),
        "endpoint": args.endpoint,
        "measure_length_node": args.measure_length_node,
        "vision_wd_node": args.vision_wd_node,
        "timeout_seconds": args.timeout,
        "watch_samples": args.samples,
        "watch_interval_seconds": args.interval,
        "result": "FAIL",
        "reads": {},
        "watchdog": {},
        "errors": [],
    }

    logging.info("Connecting to OPC UA endpoint: %s", args.endpoint)
    async with Client(url=args.endpoint, timeout=args.timeout) as client:
        logging.info("Connected")

        measure = await safe_read(client, "MeasureLength", args.measure_length_node)
        watchdog_once = await safe_read(client, "VisionWD", args.vision_wd_node)
        watchdog_samples = await sample_watchdog(
            client=client,
            node_id=args.vision_wd_node,
            samples=args.samples,
            interval=args.interval,
        )

    report["reads"] = {
        "MeasureLength": measure,
        "VisionWD": watchdog_once,
    }
    report["watchdog"] = watchdog_samples

    read_errors = [
        label for label, item in report["reads"].items() if isinstance(item, dict) and item.get("error")
    ]
    if read_errors:
        report["errors"].append(f"Failed reads: {', '.join(read_errors)}")

    if read_errors:
        report["result"] = "FAIL"
    elif watchdog_samples["changed"]:
        report["result"] = "PASS"
    else:
        report["result"] = "WARN"
        report["errors"].append("VisionWD was readable but did not change during the sample window.")

    return report


def write_report(report: dict[str, Any], output_path: Path | None) -> None:
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        output_path = Path("outputs") / f"plc_vision_plain_test_{stamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("JSON report saved: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain Python PLC/OPC UA test for TX2 AS20 VisionSystem.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"Default: {DEFAULT_ENDPOINT}")
    parser.add_argument("--measure-length-node", default=DEFAULT_MEASURE_LENGTH_NODE)
    parser.add_argument("--vision-wd-node", default=DEFAULT_VISION_WD_NODE)
    parser.add_argument("--timeout", type=float, default=8.0, help="OPC UA timeout in seconds.")
    parser.add_argument("--samples", type=int, default=30, help="VisionWD samples to read.")
    parser.add_argument("--interval", type=float, default=0.07, help="Delay between VisionWD samples.")
    parser.add_argument("--output", type=Path, help="JSON report path. Defaults to outputs/plc_vision_plain_test_<utc>.json")
    parser.add_argument("--no-output", action="store_true", help="Do not write a JSON report.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    if args.samples < 1:
        logging.error("--samples must be >= 1")
        return 2
    if args.interval < 0:
        logging.error("--interval must be >= 0")
        return 2

    try:
        report = asyncio.run(run_test(args))
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        return 130
    except Exception as exc:
        logging.exception("PLC test failed: %s", exc)
        return 1

    if not args.no_output:
        write_report(report, args.output)

    result = report["result"]
    if result == "PASS":
        logging.info("RESULT: PASS | OPC UA reads worked and VisionWD changed.")
        return 0
    if result == "WARN":
        logging.warning("RESULT: WARN | Reads worked, but VisionWD did not change.")
        return 0

    logging.error("RESULT: FAIL | %s", "; ".join(report.get("errors", [])))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
