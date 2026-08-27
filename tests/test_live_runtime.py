from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import live_mvp_app as live
import run_live_mvp_production as production


class FakeComponent:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def start(self) -> None:
        self.events.append(f"{self.name}.start")

    def stop(self) -> None:
        self.events.append(f"{self.name}.stop")


class FakeProcessor(FakeComponent):
    def warm_up(self) -> None:
        self.events.append("processor.warm_up")


class FakeRecorder(FakeComponent):
    def validate_configuration(self) -> None:
        self.events.append("recorder.validate")


class FakeSnapshot:
    def __init__(self, value: dict) -> None:
        self.value = value

    def snapshot(self, **_kwargs) -> dict:
        return self.value.copy()


class FakeHealth:
    def as_dict(self) -> dict:
        return {"ok": True}


class FakeDatabase:
    def health(self) -> FakeHealth:
        return FakeHealth()


class LiveRuntimeTests(unittest.TestCase):
    def test_production_parser_separates_waitress_and_runtime_arguments(self) -> None:
        server, runtime = production.parse_production_args(
            [
                "--waitress-threads",
                "12",
                "--waitress-host",
                "10.14.6.84",
                "--port",
                "9001",
                "--live-stream-fps",
                "10",
            ]
        )

        self.assertEqual(server.waitress_threads, 12)
        self.assertEqual(server.waitress_host, "10.14.6.84")
        self.assertEqual(runtime.port, 9001)
        self.assertEqual(runtime.live_stream_fps, 10.0)

    def test_runtime_starts_once_and_stops_in_dependency_order(self) -> None:
        events: list[str] = []
        camera = FakeComponent("camera", events)
        processor = FakeProcessor("processor", events)
        recorder = FakeRecorder("recorder", events)
        plc = FakeComponent("plc", events)
        args = SimpleNamespace(
            db_disabled=True,
            postgres_dsn="",
            sqlite_path=Path("unused.sqlite3"),
            plc_timeout=8.0,
            record_seconds=8.0,
            buffer_max_frames=60,
            buffer_seconds=3.0,
            record_fps=10.0,
        )

        with (
            patch.object(live, "configure_vision_module"),
            patch.object(live, "CameraReader", return_value=camera),
            patch.object(live, "LiveProcessor", return_value=processor),
            patch.object(live, "ClipRecorder", return_value=recorder),
            patch.object(live, "PLCMonitor", return_value=plc),
        ):
            runtime = live.LiveMvpRuntime(args)
            runtime.start()
            runtime.start()
            runtime.stop()

        self.assertEqual(
            events,
            [
                "recorder.validate",
                "processor.warm_up",
                "camera.start",
                "plc.start",
                "plc.stop",
                "recorder.stop",
                "camera.stop",
            ],
        )
        self.assertFalse(runtime.started)

    def test_health_endpoint_reports_only_component_checks(self) -> None:
        names = (
            "_runtime",
            "_camera",
            "_processor",
            "_plc",
            "_recorder",
            "_database",
        )
        previous = {name: getattr(live, name, None) for name in names}
        try:
            live._runtime = SimpleNamespace(started=True)
            live._camera = FakeSnapshot({"connected": True})
            live._processor = FakeSnapshot({"ok": True, "error": ""})
            live._plc = FakeSnapshot({"enabled": True, "connected": True})
            live._recorder = FakeSnapshot(
                {"error": "", "video_encoder_error": ""}
            )
            live._database = FakeDatabase()

            response = live.app.test_client().get("/api/health")
        finally:
            for name, value in previous.items():
                setattr(live, name, value)

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(all(payload["checks"].values()))
        self.assertNotIn("endpoint", response.get_data(as_text=True))
        self.assertNotIn("password", response.get_data(as_text=True).lower())

    def test_argument_parser_accepts_an_explicit_argument_list(self) -> None:
        args = live.parse_args(["--source", "video", "--db-disabled"])

        self.assertEqual(args.source, "video")
        self.assertTrue(args.db_disabled)
