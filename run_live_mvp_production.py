"""Production entry point for the single-process TX2 Live MVP runtime."""
from __future__ import annotations

import argparse
import signal
import sys

from waitress import create_server

import live_mvp_app as live


def parse_production_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--waitress-host", default="127.0.0.1")
    parser.add_argument("--waitress-threads", type=int, default=8)
    parser.add_argument("--waitress-channel-timeout", type=int, default=120)
    production, runtime_argv = parser.parse_known_args(argv)
    if production.waitress_threads < 4:
        parser.error("--waitress-threads must be at least 4")
    if production.waitress_channel_timeout < 30:
        parser.error("--waitress-channel-timeout must be at least 30 seconds")
    production.waitress_host = production.waitress_host.strip()
    if not production.waitress_host:
        parser.error("--waitress-host must not be empty")
    return production, live.parse_args(runtime_argv)


def main() -> int:
    production, args = parse_production_args()
    try:
        runtime = live.start_runtime(args)
    except Exception as exc:
        print(f"ERROR: Live MVP startup failed: {exc}", file=sys.stderr)
        return 2

    server = create_server(
        live.app,
        host=production.waitress_host,
        port=int(runtime.args.port),
        threads=int(production.waitress_threads),
        channel_timeout=int(production.waitress_channel_timeout),
        ident="TX2-Measurement",
    )

    def request_shutdown(_signum: int, _frame: object) -> None:
        server.close()

    for signal_name in ("SIGINT", "SIGTERM"):
        shutdown_signal = getattr(signal, signal_name, None)
        if shutdown_signal is not None:
            signal.signal(shutdown_signal, request_shutdown)

    print(
        f"\n  TX2 Live MVP production at "
        f"http://{production.waitress_host}:{runtime.args.port}\n"
    )
    print("  Server: Waitress")
    print(f"  Threads: {production.waitress_threads}")
    exit_code = 0
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        try:
            live.stop_runtime()
        except Exception as exc:
            print(f"ERROR: Live MVP shutdown failed: {exc}", file=sys.stderr)
            exit_code = 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
