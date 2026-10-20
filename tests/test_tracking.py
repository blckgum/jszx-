"""跟踪模块的契约与真实 ByteTrack 机制测试，不代表校园视频精度评估。"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from detection.core import prepare_runtime
from tracking.core import DEFAULT_TRACKER, TrackingSession, VideoClock, check_display, read_tracker_config
from tracking.records import FrameRecord, TargetRecord, extract_tracks
from tracking.visualization import PlaybackController, TrailHistory

prepare_runtime()
from ultralytics.engine.results import Boxes, Results
from ultralytics.trackers.byte_tracker import BYTETracker


class TrackingContractTests(unittest.TestCase):
    def test_config_rejects_invalid_thresholds_and_tracker(self):
        original = read_tracker_config(DEFAULT_TRACKER)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tracker.yaml"
            for change in ({"track_low_thresh": 0.8}, {"tracker_type": "botsort"}, {"track_buffer": -1}):
                with self.subTest(change=change):
                    path.write_text(yaml.safe_dump({**original, **change}))
                    with self.assertRaises(ValueError):
                        read_tracker_config(path)

    def test_pending_and_confirmed_ids_have_correct_box_semantics(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        pending = Results(image, "test.jpg", {0: "person"}, boxes=np.array([[10, 20, 30, 80, 0.9, 0]]))
        confirmed = Results(image, "test.jpg", {0: "person"}, boxes=np.array([[10, 20, 30, 80, 7, 0.9, 0]]))
        self.assertIsNone(extract_tracks(pending)[0].track_id)
        target = extract_tracks(confirmed)[0]
        self.assertEqual(target.track_id, 7)
        self.assertEqual(target.anchor_xy, (20, 80))
        self.assertEqual(target.category, "pedestrian")
        self.assertEqual(target.observation, "detected")

    def test_conf_does_not_discard_low_score_association_candidates(self):
        with patch("tracking.core.load_model") as load:
            with self.assertRaisesRegex(ValueError, "不高于"):
                TrackingSession("unused.mp4", "unused.pt", DEFAULT_TRACKER, conf=0.25)
            load.assert_not_called()

    def test_video_clock_marks_estimated_time_and_rejects_regression(self):
        clock = VideoClock(10)
        self.assertEqual(clock.timestamp(0, 0), (0.0, "video_pts"))
        self.assertEqual(clock.timestamp(1, 0), (0.1, "fps_estimate"))
        self.assertEqual(clock.timestamp(2, float("nan")), (0.2, "fps_estimate"))
        override = VideoClock(5, force_fps=True)
        self.assertEqual(override.timestamp(0, 100), (0.0, "fps_estimate"))
        clock.previous = 3.0
        with self.assertRaisesRegex(ValueError, "不递增"):
            clock.timestamp(3, 0)

    def test_trail_does_not_bridge_missing_frames_and_expires(self):
        history = TrailHistory(length=3, max_idle_frames=2)
        target = TargetRecord(1, 0, "person", "pedestrian", 0.9, (10, 10, 20, 30), (15, 30))
        for index in (0, 1, 3):
            history.update(FrameRecord("session", "video", index, index / 10, "video_pts", 100, 100, [target]))
        self.assertEqual([len(segment) for segment in history.segments(1)], [1, 1])
        history.update(FrameRecord("session", "video", 7, 0.7, "video_pts", 100, 100))
        self.assertNotIn(1, history.points)

    def test_playback_keys(self):
        playback = PlaybackController()
        playback.handle_key(32)
        self.assertTrue(playback.paused)
        playback.handle_key(ord("p"))
        self.assertFalse(playback.paused)
        playback.handle_key(27)
        self.assertTrue(playback.quit_requested)

    def test_headless_show_fails_before_opening_a_window(self):
        with patch("tracking.core.sys.platform", "linux"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "没有桌面"):
                check_display()


class ByteTrackMechanismTests(unittest.TestCase):
    @staticmethod
    def detections(rows):
        # 测试使用人为给定的框，隔离检测误差，验证真正的 ByteTrack 状态转换。
        # 六列为 x1,y1,x2,y2,score,class；不是手写的假跟踪器。
        values = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        return Boxes(values, (200, 300)).cpu().numpy()

    def make_tracker(self):
        return BYTETracker(SimpleNamespace(**read_tracker_config(DEFAULT_TRACKER)))

    def test_two_objects_keep_ids_with_low_score_and_short_miss(self):
        tracker = self.make_tracker()
        initial = tracker.update(self.detections([[10, 10, 40, 90, 0.9, 0], [140, 40, 240, 120, 0.9, 2]]))
        initial_ids = {int(row[6]): int(row[4]) for row in initial}
        low_score = tracker.update(self.detections([[12, 10, 42, 90, 0.15, 0], [142, 40, 242, 120, 0.9, 2]]))
        self.assertEqual({int(row[6]): int(row[4]) for row in low_score}, initial_ids)
        self.assertEqual(len(tracker.update(self.detections([]))), 0)
        recovered = tracker.update(self.detections([[16, 10, 46, 90, 0.9, 0], [146, 40, 246, 120, 0.9, 2]]))
        self.assertEqual({int(row[6]): int(row[4]) for row in recovered}, initial_ids)
        self.assertEqual(tracker.frame_id, 4)  # 无检测帧同样推进状态。

    def test_reset_clears_previous_video_state_and_id_counter(self):
        tracker = self.make_tracker()
        boxes = self.detections([[10, 10, 40, 90, 0.9, 0], [140, 40, 240, 120, 0.9, 2]])
        tracker.update(boxes)
        tracker.update(self.detections([]))
        tracker.reset()
        self.assertEqual(tracker.frame_id, 0)
        self.assertFalse(tracker.tracked_stracks or tracker.lost_stracks or tracker.removed_stracks)
        restarted = tracker.update(boxes)
        self.assertEqual(sorted(int(row[4]) for row in restarted), [1, 2])


if __name__ == "__main__":
    unittest.main()
