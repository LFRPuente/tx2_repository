from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import homography_web_app as vision


class VideoPlaylistTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paths = [Path("video_a.mkv"), Path("video_b.mkv")]

    def test_playlist_maps_global_frames_across_video_boundaries(self) -> None:
        metas = {
            self.paths[0]: {
                "fps": 30.0,
                "total_frames": 100,
                "duration_sec": 100 / 30.0,
                "width": 1920,
                "height": 1080,
            },
            self.paths[1]: {
                "fps": 30.0,
                "total_frames": 80,
                "duration_sec": 80 / 30.0,
                "width": 1920,
                "height": 1080,
            },
        }
        with patch.object(vision, "video_meta", side_effect=lambda path: metas[path]):
            playlist = vision.build_video_playlist(self.paths)

        first, first_local, first_global = vision.playlist_segment_for_frame(playlist, 99)
        second, second_local, second_global = vision.playlist_segment_for_frame(playlist, 100)

        self.assertEqual(first["name"], "video_a.mkv")
        self.assertEqual((first_local, first_global), (99, 99))
        self.assertEqual(second["name"], "video_b.mkv")
        self.assertEqual((second_local, second_global), (0, 100))
        self.assertEqual(playlist["total_frames"], 180)
        self.assertAlmostEqual(playlist["duration_sec"], 6.0)

    def test_discovery_recurses_and_prefers_raw_live_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            day_dir = root / "2026-07-28"
            day_dir.mkdir()
            processed = day_dir / "live_0001.mp4"
            raw = day_dir / "live_0001_raw.mp4"
            legacy = root / "legacy.mkv"
            processed.touch()
            raw.touch()
            legacy.touch()

            paths = vision.discover_video_paths(
                SimpleNamespace(video_dir=root, video=None)
            )

        self.assertEqual(paths, [raw])

    def test_latest_format_filter_keeps_only_matching_raw_clips(self) -> None:
        paths = [
            Path("live_0001_raw.mp4"),
            Path("live_0002_raw.mp4"),
            Path("live_0003_raw.mp4"),
        ]
        metas = {
            paths[0]: {
                "width": 1920, "height": 1080, "fps": 10.0,
                "total_frames": 80, "duration_sec": 8.0,
            },
            paths[1]: {
                "width": 2560, "height": 1440, "fps": 10.0,
                "total_frames": 80, "duration_sec": 8.0,
            },
            paths[2]: {
                "width": 2560, "height": 1440, "fps": 10.0,
                "total_frames": 80, "duration_sec": 8.0,
            },
        }
        with patch.object(vision, "video_meta", side_effect=lambda path: metas[path]):
            matching = vision.latest_video_format_paths(paths)

        self.assertEqual(matching, paths[1:])

    def test_latest_format_filter_groups_small_fps_variations(self) -> None:
        paths = [
            Path("live_0001_raw.mp4"),
            Path("live_0002_raw.mp4"),
            Path("live_0003_raw.mp4"),
        ]
        metas = {
            paths[0]: {
                "width": 2880, "height": 2160, "fps": 25.07,
                "total_frames": 200, "duration_sec": 8.0,
            },
            paths[1]: {
                "width": 2880, "height": 2160, "fps": 24.96,
                "total_frames": 200, "duration_sec": 8.0,
            },
            paths[2]: {
                "width": 2880, "height": 2160, "fps": 25.01,
                "total_frames": 200, "duration_sec": 8.0,
            },
        }
        with patch.object(vision, "video_meta", side_effect=lambda path: metas[path]):
            matching = vision.latest_video_format_paths(paths)

        self.assertEqual(matching, paths)

    def test_latest_format_filter_skips_an_in_progress_newest_clip(self) -> None:
        complete = Path("live_0001_raw.mp4")
        in_progress = Path("live_0002_raw.mp4")
        meta = {
            "width": 2880,
            "height": 2160,
            "fps": 29.0,
            "total_frames": 232,
            "duration_sec": 8.0,
        }

        def read_meta(path: Path) -> dict:
            if path == in_progress:
                raise RuntimeError("moov atom not found")
            return meta

        with patch.object(vision, "video_meta", side_effect=read_meta):
            matching = vision.latest_video_format_paths([complete, in_progress])

        self.assertEqual(matching, [complete])

    def test_playlist_skips_a_clip_that_becomes_unreadable(self) -> None:
        complete = Path("live_0001_raw.mp4")
        unavailable = Path("live_0002_raw.mp4")
        meta = {
            "width": 2880,
            "height": 2160,
            "fps": 29.0,
            "total_frames": 232,
            "duration_sec": 8.0,
        }

        def read_meta(path: Path) -> dict:
            if path == unavailable:
                raise RuntimeError("retention race")
            return meta

        with patch.object(vision, "video_meta", side_effect=read_meta):
            playlist = vision.build_video_playlist([complete, unavailable])

        self.assertEqual([segment["path"] for segment in playlist["segments"]], [complete])
        self.assertEqual(playlist["total_frames"], 232)

    def test_latest_format_filter_rejects_a_folder_without_complete_clips(self) -> None:
        with patch.object(
            vision,
            "video_meta",
            side_effect=RuntimeError("moov atom not found"),
        ):
            with self.assertRaisesRegex(RuntimeError, "No hay videos completos"):
                vision.latest_video_format_paths([Path("live_0001_raw.mp4")])

    def test_frame_read_retains_source_video_traceability(self) -> None:
        playlist = {
            "fps": 30.0,
            "total_frames": 180,
            "duration_sec": 6.0,
            "width": 1920,
            "height": 1080,
            "segments": [
                {
                    "path": self.paths[0],
                    "name": self.paths[0].name,
                    "stem": self.paths[0].stem,
                    "start_frame": 0,
                    "end_frame": 100,
                    "total_frames": 100,
                    "duration_sec": 100 / 30.0,
                },
                {
                    "path": self.paths[1],
                    "name": self.paths[1].name,
                    "stem": self.paths[1].stem,
                    "start_frame": 100,
                    "end_frame": 180,
                    "total_frames": 80,
                    "duration_sec": 80 / 30.0,
                },
            ],
        }
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        with patch.object(
            vision,
            "read_frame_by_index",
            return_value=(image, 7, 7 / 30.0),
        ) as read_frame:
            frame, global_idx, global_time, source = vision.read_playlist_frame_by_index(
                playlist,
                107,
            )

        read_frame.assert_called_once_with(self.paths[1], 7)
        self.assertIs(frame, image)
        self.assertEqual(global_idx, 107)
        self.assertAlmostEqual(global_time, 107 / 30.0)
        self.assertEqual(source["video_name"], "video_b.mkv")
        self.assertEqual(source["source_frame_idx"], 7)
        self.assertAlmostEqual(source["source_time_sec"], 7 / 30.0)

    def test_generated_candidates_are_not_counted_as_saved_annotations(self) -> None:
        playlist = {
            "fps": 30.0,
            "total_frames": 180,
            "duration_sec": 6.0,
            "width": 1920,
            "height": 1080,
            "segments": [
                {
                    "path": self.paths[0],
                    "name": self.paths[0].name,
                    "stem": self.paths[0].stem,
                    "start_frame": 0,
                    "end_frame": 100,
                    "total_frames": 100,
                    "duration_sec": 100 / 30.0,
                },
                {
                    "path": self.paths[1],
                    "name": self.paths[1].name,
                    "stem": self.paths[1].stem,
                    "start_frame": 100,
                    "end_frame": 180,
                    "total_frames": 80,
                    "duration_sec": 80 / 30.0,
                },
            ],
        }
        old_args = getattr(vision, "_args", None)
        old_playlist = vision._video_playlist
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                dataset_dir = Path(temp_dir)
                (dataset_dir / "images").mkdir()
                (dataset_dir / "labels").mkdir()
                vision._args = SimpleNamespace(
                    dataset_dir=dataset_dir,
                    candidate_samples_per_video=2,
                    legacy_candidates_dir=None,
                )
                vision._video_playlist = playlist

                history = vision.dataset_history()

            self.assertEqual(len(history), 4)
            self.assertTrue(all(item["candidate"] for item in history))
            self.assertEqual(vision.saved_piece_annotation_count(history), 0)
            self.assertEqual(
                {item["source"]["video_name"] for item in history},
                {"video_a.mkv", "video_b.mkv"},
            )
        finally:
            vision._args = old_args
            vision._video_playlist = old_playlist


if __name__ == "__main__":
    unittest.main()
