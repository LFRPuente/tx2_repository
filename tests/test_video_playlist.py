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
