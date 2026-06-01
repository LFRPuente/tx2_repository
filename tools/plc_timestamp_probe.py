#!/usr/bin/env python3
"""Probe OPC UA tags that may contain the PLC execution timestamp.

Usage examples:

  python tools/plc_timestamp_probe.py --endpoint opc.tcp://10.14.6.48:49320
  python tools/plc_timestamp_probe.py --node-id "ns=2;s=Channel.Device.Tag"
  python tools/plc_timestamp_probe.py --keyword timestamp --keyword execution --watch 1

The script is intentionally read-only. It only browses nodes and reads values.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import deque
from datetime import datetime, timezone
from typing import Any


DEFAULT_ENDPOINT_TX2 = "opc.tcp://10.14.6.48:49320"
DEFAULT_KEYWORDS = ("timestamp", "time", "date", "execution", "exec", "scan", "trigger")


def now_utc() -> str:
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


def matches_keywords(record: dict[str, Any], keywords: list[str]) -> bool:
    haystack = " ".join(
        str(record.get(key, "")) for key in ("node_id", "browse_name", "display_name", "path")
    ).lower()
    return any(keyword.lower() in haystack for keyword in keywords)


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
        "read_utc": now_utc(),
    }


async def browse_candidates(client: Any, keywords: list[str], max_depth: int, max_nodes: int) -> list[dict[str, Any]]:
    from asyncua import ua

    objects = client.get_objects_node()
    queue = deque([(objects, "Objects", 0)])
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    visited = 0

    while queue and visited < max_nodes:
        node, path, depth = queue.popleft()
        node_key = node.nodeid.to_string()
        if node_key in seen:
            continue
        seen.add(node_key)
        visited += 1

        try:
            children = await node.get_children()
        except Exception:
            continue

        for child in children:
            child_id = child.nodeid.to_string()
            try:
                browse_name = (await child.read_browse_name()).Name
            except Exception:
                browse_name = ""
            try:
                display_name = (await child.read_display_name()).Text
            except Exception:
                display_name = ""
            try:
                node_class = await child.read_node_class()
            except Exception:
                node_class = None

            child_path = f"{path}/{display_name or browse_name or child_id}"
            record = {
                "node_id": child_id,
                "browse_name": browse_name,
                "display_name": display_name,
                "path": child_path,
                "node_class": str(node_class),
            }

            if node_class == ua.NodeClass.Variable and matches_keywords(record, keywords):
                try:
                    record.update(await read_node(client, child_id))
                except Exception as exc:
                    record["read_error"] = str(exc)
                    record["read_utc"] = now_utc()
                candidates.append(record)

            if depth + 1 <= max_depth and child_id not in seen:
                queue.append((child, child_path, depth + 1))

    candidates.sort(key=lambda item: item.get("path", ""))
    return candidates


async def run_once(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from asyncua import Client
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: asyncua\n"
            "Install it on the VPN machine with:\n"
            "  py -m pip install asyncua\n"
            "or:\n"
            "  python -m pip install asyncua"
        ) from exc

    async with Client(url=args.endpoint, timeout=args.timeout) as client:
        direct_reads = []
        for node_id in args.node_id:
            try:
                direct_reads.append(await read_node(client, node_id))
            except Exception as exc:
                direct_reads.append({"node_id": node_id, "error": str(exc), "read_utc": now_utc()})

        candidates = []
        if args.browse or not args.node_id:
            candidates = await browse_candidates(
                client=client,
                keywords=args.keyword,
                max_depth=args.max_depth,
                max_nodes=args.max_nodes,
            )

    return {
        "endpoint": args.endpoint,
        "run_utc": now_utc(),
        "keywords": args.keyword,
        "direct_reads": direct_reads,
        "candidates": candidates,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Endpoint: {report['endpoint']}")
    print(f"UTC read:  {report['run_utc']}")

    if report["direct_reads"]:
        print("\nDirect reads:")
        for item in report["direct_reads"]:
            print(json.dumps(item, indent=2, ensure_ascii=False))

    if report["candidates"]:
        print("\nCandidate timestamp-like tags:")
        for item in report["candidates"]:
            print("-" * 80)
            print(f"path:   {item.get('path')}")
            print(f"node:   {item.get('node_id')}")
            print(f"name:   {item.get('display_name') or item.get('browse_name')}")
            print(f"value:  {item.get('value')!r}")
            print(f"source: {item.get('source_timestamp')}")
            print(f"server: {item.get('server_timestamp')}")
            if item.get("read_error"):
                print(f"error:  {item['read_error']}")
    elif not report["direct_reads"]:
        print("\nNo matching tags found. Try broader keywords or a larger --max-depth.")


async def main_async(args: argparse.Namespace) -> int:
    if args.watch <= 0:
        report = await run_once(args)
        print_report(report)
        if args.json:
            print("\nJSON:")
            print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    while True:
        report = await run_once(args)
        print("=" * 88)
        print_report(report)
        await asyncio.sleep(args.watch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read/browse PLC timestamp tags through OPC UA.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT_TX2, help=f"Default TX2: {DEFAULT_ENDPOINT_TX2}")
    parser.add_argument("--node-id", action="append", default=[], help='Exact OPC UA NodeId, e.g. "ns=2;s=Channel.Device.Tag"')
    parser.add_argument("--keyword", action="append", default=list(DEFAULT_KEYWORDS), help="Keyword used while browsing. Can be repeated.")
    parser.add_argument("--browse", action="store_true", help="Browse even when --node-id is provided.")
    parser.add_argument("--max-depth", type=int, default=7, help="Browse depth from Objects.")
    parser.add_argument("--max-nodes", type=int, default=5000, help="Safety limit for browsing.")
    parser.add_argument("--timeout", type=float, default=8.0, help="OPC UA connection timeout seconds.")
    parser.add_argument("--watch", type=float, default=0.0, help="Repeat every N seconds.")
    parser.add_argument("--json", action="store_true", help="Also print full JSON output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
