from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import live_mvp_app as live


class LiveVisionPipelineTests(unittest.TestCase):
    def test_nvdec_camera_command_uses_nvidia_decoder_and_bgr_output(self) -> None:
        command = live.build_nvdec_camera_command(
            Path("ffmpeg.exe"),
            "rtsp://camera.example/stream",
            10.0,
        )

        self.assertIn("h264_cuvid", command)
        self.assertIn("fps=10,hwdownload,format=nv12,format=bgr24", command)
        self.assertIn("bgr24", command)
        self.assertEqual(command[command.index("-fflags") + 1], "nobuffer")
        self.assertEqual(command[command.index("-flags") + 1], "low_delay")
        self.assertEqual(command[command.index("-analyzeduration") + 1], "0")
        self.assertEqual(command[command.index("-reorder_queue_size") + 1], "0")
        self.assertEqual(command[-1], "pipe:1")

    def test_camera_snapshot_rejects_a_stale_connected_state(self) -> None:
        reader = live.CameraReader(
            SimpleNamespace(source="rtsp", record_fps=10.0),
            live.FrameBuffer(maxlen=8),
        )
        reader._set_state(
            connected=True,
            last_frame_monotonic=100.0,
            last_frame_utc="frame-time",
        )

        with patch("live_mvp_app.time.perf_counter", return_value=104.5):
            status = reader.snapshot()

        self.assertFalse(status["connected"])
        self.assertEqual(status["last_frame_age_seconds"], 4.5)
        self.assertIn("stale", status["error"].lower())

    def test_resolution_dimensions_parses_native_axis_resolution(self) -> None:
        self.assertEqual(live.resolution_dimensions("2880x2160"), (2880, 2160))

    def test_live_stream_command_copies_h264_into_fragmented_mp4(self) -> None:
        command = live.build_live_stream_command(
            Path("ffmpeg.exe"),
            "rtsp://camera.example/stream",
            rtsp_source=True,
        )

        self.assertIn("-rtsp_transport", command)
        self.assertNotIn("nobuffer", command)
        self.assertNotIn("low_delay", command)
        self.assertEqual(command[command.index("-analyzeduration") + 1], "0")
        self.assertEqual(command[command.index("-probesize") + 1], "32768")
        self.assertEqual(command[command.index("-flush_packets") + 1], "1")
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertIn(
            "frag_every_frame+empty_moov+default_base_moof",
            command,
        )
        self.assertEqual(command[-2:], ["mp4", "pipe:1"])
        self.assertNotIn("h264_nvenc", command)

    def test_clip_writer_command_uses_nvidia_encoder(self) -> None:
        command = live.build_nvenc_writer_command(
            Path("ffmpeg.exe"),
            Path("clip.mp4"),
            10.0,
            (2880, 2160),
            16.0,
        )

        self.assertEqual(command[command.index("-c:v") + 1], "h264_nvenc")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "bgr24")
        self.assertEqual(command[command.index("-preset") + 1], "p1")
        self.assertEqual(command[command.index("-tune") + 1], "ll")
        self.assertEqual(command[command.index("-rc") + 1], "cbr")
        self.assertIn("2880x2160", command)
        self.assertNotIn("libx264", command)

    def test_nvenc_probe_uses_a_supported_frame_size(self) -> None:
        completed = SimpleNamespace(returncode=0, stderr="")
        with patch.object(live.subprocess, "run", return_value=completed) as run:
            available, error = live.validate_nvenc(Path("ffmpeg.exe"))

        command = run.call_args.args[0]
        self.assertTrue(available)
        self.assertEqual(error, "")
        self.assertIn("color=size=256x256:rate=1", command)

    def test_live_rtsp_url_requests_deployed_camera_maximum_fps(self) -> None:
        url = live.build_live_rtsp_url(
            SimpleNamespace(
                camera_user="",
                camera_password="",
                camera_ip="camera.example",
                codec="h264",
                camera_resolution="2880x2160",
                live_stream_fps=30.0,
                live_rtsp_url="",
            )
        )

        self.assertIn("resolution=2880x2160", url)
        self.assertIn("fps=30", url)
        self.assertIn("videozfpsmode=fixed", url)
        self.assertNotIn("videokeyframeinterval", url)

    def test_processing_rtsp_url_shares_the_fixed_30_fps_camera_profile(self) -> None:
        url = live.build_rtsp_url(
            SimpleNamespace(
                camera_user="",
                camera_password="",
                camera_ip="camera.example",
                codec="h264",
                camera_resolution="2880x2160",
                capture_fps=30.0,
                rtsp_url="",
            )
        )

        self.assertIn("resolution=2880x2160", url)
        self.assertIn("fps=30", url)
        self.assertIn("videozfpsmode=fixed", url)

    def test_live_processor_reports_resolved_inference_device(self) -> None:
        device_info = {
            "requested": "auto",
            "device": "cuda:0",
            "device_name": "NVIDIA L40S",
            "cuda_available": True,
            "cuda_device_count": 1,
            "torch_version": "2.12.1+cu130",
            "torch_cuda_version": "13.0",
        }
        with patch.object(
            live.vision,
            "resolve_yolo_device",
            return_value=device_info,
        ):
            processor = live.LiveProcessor(
                SimpleNamespace(process_fps=10.0, device="auto"),
                live.FrameBuffer(maxlen=8),
            )

        status = processor.snapshot()
        self.assertEqual(status["inference_device"], "cuda:0")
        self.assertEqual(status["inference_device_name"], "NVIDIA L40S")
        self.assertTrue(status["cuda_available"])
        self.assertEqual(status["homography_backend"], "opencv_cpu")

    def test_idle_live_processor_does_not_process_or_encode_frames(self) -> None:
        buffer = live.FrameBuffer(maxlen=8)
        buffer.append(
            {
                "index": 3,
                "utc": "frame-3",
                "monotonic": 10.0,
                "frame": np.zeros((24, 32, 3), dtype=np.uint8),
            }
        )
        processor = live.LiveProcessor(
            SimpleNamespace(device="cpu"),
            buffer,
        )

        with patch.object(processor, "_process") as process:
            status = processor.snapshot()

        process.assert_not_called()
        self.assertTrue(status["ok"])
        self.assertEqual(status["processing_mode"], "plc_triggered_clip")
        self.assertEqual(status["processed_count"], 0)

    def test_processor_warmup_uses_rectified_shape_without_publishing_a_result(self) -> None:
        processor = live.LiveProcessor(
            SimpleNamespace(device="cpu", conf=0.1, imgsz=960),
            live.FrameBuffer(maxlen=8),
        )
        calibration = {
            "exclusion_zones": [],
            "exclusion_max_box_overlap": 0.2,
        }
        with (
            patch.object(
                live.vision,
                "load_homography",
                return_value=(np.eye(3), (32, 24), {}),
            ),
            patch.object(
                live.vision,
                "load_measurement_calibration",
                return_value=calibration,
            ),
            patch.object(
                live.vision,
                "predict_yolo_boxes_with_rules",
                return_value=([], {}),
            ) as predict,
        ):
            processor.warm_up()

        self.assertEqual(predict.call_args.args[0].shape, (24, 32, 3))
        status = processor.snapshot()
        self.assertTrue(status["warmup_complete"])
        self.assertIsNotNone(status["warmup_duration_ms"])
        self.assertIsNone(status["result"])

    def test_live_processor_applies_calibrated_zones_and_keeps_box_rules(self) -> None:
        processor = live.LiveProcessor(
            SimpleNamespace(
                process_fps=10.0,
                conf=0.10,
                imgsz=960,
                model=Path("individual-pieces.pt"),
            ),
            live.FrameBuffer(maxlen=8),
        )
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        item = {
            "index": 17,
            "utc": "frame-17",
            "monotonic": 101.25,
            "frame": frame,
        }
        zones = [{"x": 0.0, "y": 0.0, "w": 5.0, "h": 24.0}]
        calibration = {
            "reference_y": None,
            "exclusion_zones": zones,
            "exclusion_max_box_overlap": 0.20,
        }
        boxes = [{"x": 8.0, "y": 3.0, "w": 8.0, "h": 16.0, "conf": 0.91}]
        diagnostics = {
            "raw_count": 2,
            "removed_exclusion_count": 1,
            "final_count": 1,
            "exclusion_max_overlap": 0.20,
        }
        pieces = [
            {
                "piece_id": 1,
                "box": boxes[0],
                "confidence": 0.91,
                "sobel": {"line": None, "is_valid": False},
                "measurement": None,
                "valid": False,
            }
        ]
        summary = {"detected_count": 1, "valid_count": 0, "invalid_count": 1}

        with (
            patch.object(
                live.vision,
                "load_homography",
                return_value=(np.eye(3), (32, 24), {}),
            ),
            patch.object(
                live.vision,
                "load_measurement_calibration",
                return_value=calibration,
            ),
            patch.object(
                live.vision,
                "predict_yolo_boxes_with_rules",
                return_value=(boxes, diagnostics),
            ) as predict,
            patch.object(
                live.vision,
                "analyze_piece_boxes",
                return_value=(pieces, summary),
            ) as analyze,
            patch.object(
                live.vision,
                "mvp_original_overlay_for_pieces",
                return_value={"reference_line": None, "piece_fronts": []},
            ),
        ):
            result = processor._process(item)

        predict.assert_called_once()
        predict_kwargs = predict.call_args.kwargs
        self.assertEqual(predict_kwargs["exclusion_zones"], zones)
        self.assertEqual(predict_kwargs["exclusion_max_overlap"], 0.20)
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.args[1], boxes)
        self.assertEqual(result["box_rules"], diagnostics)
        self.assertEqual(result["pieces"], pieces)
        self.assertEqual(result["measurement_summary"], summary)
        self.assertNotIn("_original_jpeg", result)
        self.assertIn("_recording_frame", result)
        self.assertIn("stage_durations_ms", result)
        self.assertNotIn("original_image", result)
        self.assertNotIn("rectified_image", result)

        processor._set_state(result=result)
        snapshot = processor.snapshot(include_images=True)
        self.assertNotIn("_original_jpeg", snapshot["result"])
        self.assertNotIn("_original_overlay", snapshot["result"])

    def test_overlapping_clips_reuse_frame_analysis(self) -> None:
        processor = live.LiveProcessor(
            SimpleNamespace(device="cpu", processing_cache_frames=4),
            live.FrameBuffer(maxlen=8),
        )
        item = {
            "index": 23,
            "utc": "frame-23",
            "monotonic": 23.0,
            "frame": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        analyzed = {
            "frame_index": 23,
            "frame_utc": "frame-23",
            "frame_monotonic": 23.0,
            "pieces": [],
            "calibration": {},
            "_original_overlay": {},
            "_recording_frame": {
                "index": 23,
                "utc": "frame-23",
                "monotonic": 23.0,
                "frame": item["frame"],
                "raw_frame": item["frame"],
            },
        }

        with patch.object(processor, "_process", return_value=analyzed) as process:
            first, _ = processor.process_clip_frame(item)
            second, _ = processor.process_clip_frame(item)

        process.assert_called_once()
        self.assertFalse(first["processing_cache_hit"])
        self.assertTrue(second["processing_cache_hit"])
        self.assertEqual(processor.snapshot()["processing_cache_hits"], 1)

    def test_cached_overlay_is_materialized_outside_the_inference_lock(self) -> None:
        processor = live.LiveProcessor(
            SimpleNamespace(device="cpu", processing_cache_frames=4),
            live.FrameBuffer(maxlen=8),
        )
        item = {
            "index": 24,
            "utc": "frame-24",
            "monotonic": 24.0,
            "frame": np.zeros((8, 8, 3), dtype=np.uint8),
        }
        analyzed = {
            "frame_index": 24,
            "frame_utc": "frame-24",
            "frame_monotonic": 24.0,
            "pieces": [],
            "calibration": {},
            "_original_overlay": {},
            "_recording_frame": {
                "index": 24,
                "utc": "frame-24",
                "monotonic": 24.0,
                "frame": item["frame"],
                "raw_frame": item["frame"],
            },
        }

        with patch.object(processor, "_process", return_value=analyzed):
            processor.process_clip_frame(item)
        original_materialize = processor._materialize_cached_result

        def materialize(*args, **kwargs):
            self.assertFalse(processor.inference_lock.locked())
            return original_materialize(*args, **kwargs)

        with patch.object(
            processor,
            "_materialize_cached_result",
            side_effect=materialize,
        ):
            processor.process_clip_frame(item)

    def test_measurement_frame_runs_before_waiting_clip_work(self) -> None:
        processor = live.LiveProcessor(
            SimpleNamespace(device="cpu", processing_cache_frames=8),
            live.FrameBuffer(maxlen=8),
        )
        first_started = threading.Event()
        release_first = threading.Event()
        processed_order: list[int] = []

        def analyze(item, *, include_evidence_images=False):
            index = int(item["index"])
            processed_order.append(index)
            if index == 1:
                first_started.set()
                release_first.wait(timeout=1.0)
            return {
                "frame_index": index,
                "frame_utc": item["utc"],
                "frame_monotonic": item["monotonic"],
                "pieces": [],
                "calibration": {},
                "_original_overlay": {},
                "_recording_frame": {
                    **item,
                    "raw_frame": item["frame"],
                },
            }

        def item(index: int) -> dict:
            return {
                "index": index,
                "utc": f"frame-{index}",
                "monotonic": float(index),
                "frame": np.zeros((8, 8, 3), dtype=np.uint8),
            }

        with patch.object(processor, "_process", side_effect=analyze):
            first = threading.Thread(
                target=processor.process_clip_frame,
                args=(item(1),),
            )
            first.start()
            self.assertTrue(first_started.wait(timeout=1.0))

            measurement = threading.Thread(
                target=processor.process_measurement_frame,
                args=(item(2),),
            )
            measurement.start()
            deadline = time.perf_counter() + 1.0
            while (
                not processor.measurement_priority.is_set()
                and time.perf_counter() < deadline
            ):
                time.sleep(0.005)

            trailing = threading.Thread(
                target=processor.process_clip_frame,
                args=(item(3),),
            )
            trailing.start()
            release_first.set()
            for thread in (first, measurement, trailing):
                thread.join(timeout=2.0)
                self.assertFalse(thread.is_alive())

        self.assertEqual(processed_order, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
