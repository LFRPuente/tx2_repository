#!/usr/bin/env python3
"""Probe BilletScanning APIs from inside the plant/VPN network.

The script intentionally uses only Python's standard library so it can run on a
plain Windows workstation. It checks TCP reachability, selected HTTP endpoints,
and performs a basic WebSocket upgrade handshake. For JSON responses it reports
top-level keys and any fields that look timestamp-related.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://10.14.2.250"
DEFAULT_MILL = "TX2"
TIMESTAMP_WORDS = (
    "timestamp",
    "time",
    "date",
    "created",
    "updated",
    "scan",
    "event",
)


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 10:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def normalize_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme:
        base_url = f"https://{base_url}"
        parsed = urllib.parse.urlparse(base_url)
    if not parsed.netloc:
        raise ValueError(f"Base URL invalida: {base_url}")
    return base_url.rstrip("/")


def endpoint_paths(mill: str) -> list[str]:
    mill = mill.upper()
    return [
        "/api/health/",
        "/api/scanning_db_health_check/",
        "/api/scanning_prod_db_health_check/",
        "/api/oracle_db_health_check/",
        f"/api/dataman/{mill}/",
        f"/api/active_mill_order/{mill}/",
        f"/api/mill_orders/{mill}/",
        f"/api/stocking_table_billets/{mill}/",
        "/api/logs/services/",
        f"/opc-{mill.lower()}/",
        f"/opc-{mill.lower()}/health",
        f"/opc-{mill.lower()}/docs",
        f"/opc-{mill.lower()}/openapi.json",
    ]


def tcp_probe(host: str, port: int, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            return {"host": host, "port": port, "ok": True, "elapsed_ms": elapsed_ms}
    except OSError as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "host": host,
            "port": port,
            "ok": False,
            "elapsed_ms": elapsed_ms,
            "error": str(exc),
        }


def make_opener(insecure: bool) -> urllib.request.OpenerDirector:
    handlers: list[Any] = [NoRedirectHandler()]
    if insecure:
        context = ssl._create_unverified_context()
        handlers.append(urllib.request.HTTPSHandler(context=context))
    return urllib.request.build_opener(*handlers)


def body_preview(body: bytes, limit: int = 1200) -> str:
    text = body.decode("utf-8", errors="replace")
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def flatten_time_fields(value: Any, prefix: str = "$", limit: int = 40) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if len(found) >= limit:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}"
                if any(word in key_text.lower() for word in TIMESTAMP_WORDS):
                    found.append({"path": child_path, "value": child})
                    if len(found) >= limit:
                        return
                walk(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node[:8]):
                walk(child, f"{path}[{index}]")
                if len(found) >= limit:
                    return

    walk(value, prefix)
    return found


def json_summary(body: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(body.decode("utf-8"))
    except Exception:
        return {"is_json": False}

    summary: dict[str, Any] = {"is_json": True}
    if isinstance(decoded, dict):
        summary["top_level_type"] = "object"
        summary["top_level_keys"] = list(decoded.keys())[:50]
    elif isinstance(decoded, list):
        summary["top_level_type"] = "array"
        summary["array_length"] = len(decoded)
        if decoded and isinstance(decoded[0], dict):
            summary["first_item_keys"] = list(decoded[0].keys())[:50]
    else:
        summary["top_level_type"] = type(decoded).__name__

    summary["timestamp_like_fields"] = flatten_time_fields(decoded)
    return summary


def http_probe(
    opener: urllib.request.OpenerDirector,
    url: str,
    timeout: float,
    headers: dict[str, str],
    max_body_bytes: int,
    save_bodies: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    req = urllib.request.Request(url, headers=headers, method="GET")
    result: dict[str, Any] = {"url": url, "method": "GET"}

    try:
        with opener.open(req, timeout=timeout) as response:
            body = response.read(max_body_bytes)
            status = response.status
            response_headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        body = exc.read(max_body_bytes)
        status = exc.code
        response_headers = dict(exc.headers.items())
        result["http_error"] = exc.reason
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        result.update({"ok": False, "elapsed_ms": elapsed_ms, "error": str(exc)})
        return result

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    content_type = response_headers.get("Content-Type", "")
    result.update(
        {
            "ok": 200 <= status < 400,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "content_type": content_type,
            "content_length_read": len(body),
            "location": response_headers.get("Location"),
            "www_authenticate": response_headers.get("WWW-Authenticate"),
            "preview": body_preview(body),
        }
    )
    result.update(json_summary(body))
    if save_bodies:
        result["body"] = body.decode("utf-8", errors="replace")
    return result


def websocket_probe(
    base_url: str,
    path: str,
    timeout: float,
    insecure: bool,
    headers: dict[str, str],
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(base_url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    if not host:
        return {"ok": False, "error": f"No host in {base_url}"}

    port = parsed.port
    use_tls = scheme == "https"
    if port is None:
        port = 443 if use_tls else 80

    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request_headers = {
        "Host": parsed.netloc,
        "Upgrade": "websocket",
        "Connection": "Upgrade",
        "Sec-WebSocket-Key": key,
        "Sec-WebSocket-Version": "13",
        "User-Agent": "tx2-cv-api-probe/1.0",
        **headers,
    }
    request = f"GET {path} HTTP/1.1\r\n"
    request += "".join(f"{name}: {value}\r\n" for name, value in request_headers.items())
    request += "\r\n"

    started = time.perf_counter()
    sock: socket.socket | ssl.SSLSocket | None = None
    try:
        raw_sock = socket.create_connection((host, port), timeout=timeout)
        raw_sock.settimeout(timeout)
        if use_tls:
            context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
            sock = context.wrap_socket(raw_sock, server_hostname=host)
        else:
            sock = raw_sock
        sock.sendall(request.encode("ascii"))
        response = sock.recv(4096).decode("iso-8859-1", errors="replace")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return {"path": path, "ok": False, "elapsed_ms": elapsed_ms, "error": str(exc)}
    finally:
        if sock is not None:
            sock.close()

    status_line = response.splitlines()[0] if response else ""
    status_code = None
    parts = status_line.split()
    if len(parts) >= 2 and parts[1].isdigit():
        status_code = int(parts[1])

    return {
        "path": path,
        "ok": status_code == 101,
        "elapsed_ms": elapsed_ms,
        "status": status_code,
        "status_line": status_line,
        "preview": " ".join(response.split())[:1200],
    }


def print_http_result(path: str, result: dict[str, Any]) -> None:
    if "status" in result:
        marker = "OK" if result.get("ok") else "WARN"
        print(f"[{marker}] GET {path} -> {result['status']} ({result['elapsed_ms']} ms)")
        if result.get("location"):
            print(f"      redirect: {result['location']}")
        if result.get("www_authenticate"):
            print(f"      auth: {result['www_authenticate']}")
        if result.get("is_json"):
            keys = result.get("top_level_keys") or result.get("first_item_keys") or []
            if keys:
                print(f"      json keys: {', '.join(map(str, keys[:12]))}")
            fields = result.get("timestamp_like_fields") or []
            for field in fields[:8]:
                print(f"      time? {field['path']} = {field['value']!r}")
        elif result.get("preview"):
            print(f"      preview: {result['preview'][:220]}")
    else:
        print(f"[FAIL] GET {path} -> {result.get('error')} ({result.get('elapsed_ms')} ms)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test BilletScanning API reachability and timestamp fields from VPN."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--mill", default=DEFAULT_MILL, help=f"Default: {DEFAULT_MILL}")
    parser.add_argument("--timeout", type=float, default=8.0, help="Timeout per request in seconds.")
    parser.add_argument("--insecure", action="store_true", help="Ignore HTTPS certificate validation.")
    parser.add_argument("--bearer-token", help="Bearer token to send as Authorization header.")
    parser.add_argument("--bearer-token-env", help="Environment variable containing a bearer token.")
    parser.add_argument("--cookie", help="Cookie header, for authenticated browser sessions.")
    parser.add_argument("--extra-path", action="append", default=[], help="Additional path to test.")
    parser.add_argument("--save-bodies", action="store_true", help="Store full response bodies in JSON output.")
    parser.add_argument("--max-body-bytes", type=int, default=256_000)
    parser.add_argument("--output", help="Output JSON path. Defaults to outputs/billet_api_probe_<utc>.json")
    parser.add_argument("--no-output", action="store_true", help="Do not write a JSON output file.")
    args = parser.parse_args()

    try:
        base_url = normalize_base_url(args.base_url)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname
    if not host:
        print(f"No se pudo obtener host desde {base_url}", file=sys.stderr)
        return 2

    token = args.bearer_token
    if args.bearer_token_env:
        token = os.environ.get(args.bearer_token_env)
        if not token:
            print(f"WARNING: env var {args.bearer_token_env} no existe o esta vacia.")

    headers = {"Accept": "application/json, text/plain, */*", "User-Agent": "tx2-cv-api-probe/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if args.cookie:
        headers["Cookie"] = args.cookie

    paths = endpoint_paths(args.mill) + args.extra_path
    opener = make_opener(args.insecure)

    print("=== BilletScanning API probe ===")
    print(f"Base URL: {base_url}")
    print(f"Mill: {args.mill.upper()}")
    print(f"UTC: {iso_now()}")
    print(f"TLS verify: {'off' if args.insecure else 'on'}")
    print()

    tcp_ports = sorted({parsed.port or (443 if parsed.scheme == "https" else 80), 8000, 8001, 8002})
    tcp_results = [tcp_probe(host, port, args.timeout) for port in tcp_ports]
    for result in tcp_results:
        marker = "OK" if result["ok"] else "FAIL"
        detail = f"{result['elapsed_ms']} ms" if result["ok"] else result.get("error", "")
        print(f"[{marker}] TCP {result['host']}:{result['port']} {detail}")
    print()

    http_results: list[dict[str, Any]] = []
    for path in paths:
        url = urllib.parse.urljoin(base_url + "/", path.lstrip("/"))
        result = http_probe(opener, url, args.timeout, headers, args.max_body_bytes, args.save_bodies)
        result["path"] = path
        http_results.append(result)
        print_http_result(path, result)

    ws_path = f"/ws/stocking_table_changes/{args.mill.upper()}/"
    ws_result = websocket_probe(base_url, ws_path, args.timeout, args.insecure, headers)
    print()
    marker = "OK" if ws_result.get("ok") else "WARN"
    print(f"[{marker}] WS {ws_path} -> {ws_result.get('status_line') or ws_result.get('error')}")

    report = {
        "run_utc": iso_now(),
        "base_url": base_url,
        "mill": args.mill.upper(),
        "timeout_seconds": args.timeout,
        "insecure_tls": args.insecure,
        "auth": {
            "bearer_token": mask_secret(token),
            "cookie_present": bool(args.cookie),
        },
        "tcp": tcp_results,
        "http": http_results,
        "websocket": ws_result,
    }

    if not args.no_output:
        output = Path(args.output) if args.output else Path("outputs") / f"billet_api_probe_{utc_stamp()}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print()
        print(f"Reporte guardado: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
