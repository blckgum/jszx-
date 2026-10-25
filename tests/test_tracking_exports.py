"""受控框驱动真实 ByteTrack，核对生命周期和流式导出，不报告真实场景精度。"""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from detection.core import prepare_runtime, write_json
from tracking.batch import track_directory
from tracking.core import DEFAULT_TRACKER, TrackingSession, VideoClock, read_tracker_config, track_video
from tracking.export import RunExports
from tracking.records import FrameRecord, TargetRecord, extract_tracks
from tracking.__main__ import main

prepare_runtime()
from ultralytics.engine.results import Boxes, Results
from ultralytics.trackers.byte_tracker import BYTETracker


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TrackingExportTests(unittest.TestCase):
    def test_removal_boundary_recovery_does_not_split_one_id_into_two_tracklets(self):
        tracker = BYTETracker(SimpleNamespace(**{**read_tracker_config(DEFAULT_TRACKER), "track_buffer": 2}))
        session = SimpleNamespace(model=SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker])))
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        row = [10, 10, 40, 90, 0.9, 0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = RunExports(root)
            try:
                for index, rows in enumerate(([row], [], [], [], [row])):
                    boxes = Boxes(np.asarray(rows, dtype=np.float32).reshape(-1, 6), image.shape[:2]).numpy()
                    tracks = tracker.update(boxes).reshape(-1, 8)
                    result = Results(image, "synthetic", {0: "person"}, boxes=tracks[:, :7])
                    export.write(FrameRecord("s", "v", index, index / 10, "fps_estimate", 300, 200,
                                             extract_tracks(result)), TrackingSession.retained_ids(session))
                export.finish("eof")
            finally:
                export.close()
            tracklet, = read_lines(root / "tracklets.jsonl")
            self.assertEqual(tracklet["track_id"], 1)
            self.assertEqual(tracklet["observations"], 2)
            self.assertEqual(tracklet["recoveries"], 1)
            self.assertEqual(tracklet["max_recovered_gap_frames"], 3)
            self.assertEqual(tracklet["close_reason"], "session_end:eof")

    def test_real_tracker_long_loss_closes_old_id_and_confirms_new_id(self):
        config = {**read_tracker_config(DEFAULT_TRACKER), "track_buffer": 2}
        tracker = BYTETracker(SimpleNamespace(**config))
        session = SimpleNamespace(model=SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker])))
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        row = [10, 10, 40, 90, 0.9, 0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = RunExports(root)
            try:
                # 四个空帧保证旧 ID 已从关联池清除；再出现需重新建立并确认轨迹。
                for index, rows in enumerate(([row], [], [], [], [], [row], [row])):
                    boxes = Boxes(np.asarray(rows, dtype=np.float32).reshape(-1, 6), image.shape[:2]).cpu().numpy()
                    tracks = tracker.update(boxes).reshape(-1, 8)
                    result = Results(image, "synthetic", {0: "person"}, boxes=tracks[:, :7])
                    record = FrameRecord("session", "synthetic", index, index / 10, "fps_estimate",
                                         300, 200, extract_tracks(result))
                    export.write(record, TrackingSession.retained_ids(session))
                export.finish("eof")
                self.assertFalse(export.active)
            finally:
                export.close()
            frames = read_lines(root / "tracks.jsonl")
            self.assertEqual([t["track_id"] for t in frames[0]["tracks"]], [1])
            self.assertEqual([t["track_id"] for t in frames[6]["tracks"]], [2])
            self.assertTrue(all(not frames[i]["tracks"] for i in (1, 2, 3, 4, 5)))
            tracklets = {t["track_id"]: t for t in read_lines(root / "tracklets.jsonl")}
            self.assertEqual(tracklets[1]["close_reason"], "tracker_removed")
            self.assertEqual(tracklets[1]["close_frame_index"], 4)
            self.assertEqual(tracklets[2]["close_reason"], "session_end:eof")
            self.assertEqual([e["event"] for e in read_lines(root / "events.jsonl")],
                             ["started", "lost", "closed", "started", "closed"])

    def test_recovery_class_change_pending_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = RunExports(root)
            person = TargetRecord(7, 0, "person", "pedestrian", 0.9, (10, 20, 30, 80), (20, 80))
            changed = TargetRecord(7, 2, "car", "vehicle", 0.7, (12, 20, 32, 80), (22, 80))
            pending = TargetRecord(None, 0, "person", "pedestrian", 0.3, (40, 20, 60, 80), (50, 80))
            try:
                for index, targets in enumerate(([person, pending], [], [], [changed])):
                    record = FrameRecord("session", "video,中文.mp4", index, index / 10, "video_pts", 200, 100, targets)
                    export.write(record, {7})
                export.finish("max_frames")
            finally:
                export.close()
            events = read_lines(root / "events.jsonl")
            self.assertEqual([e["event"] for e in events], ["started", "lost", "recovered", "class_changed", "closed"])
            self.assertEqual(events[2]["gap_frames"], 2)
            tracklet, = read_lines(root / "tracklets.jsonl")
            self.assertEqual(tracklet["observations"], 2)
            self.assertEqual(tracklet["unobserved_frames_within_span"], 2)
            self.assertEqual(tracklet["max_recovered_gap_frames"], 2)
            self.assertEqual(tracklet["class_changes"], 1)
            self.assertEqual(tracklet["class_counts"], {"person": 1, "car": 1})
            self.assertAlmostEqual(tracklet["mean_confidence"], 0.8)
            with (root / "observations.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["track_id"], "")
            self.assertEqual(rows[0]["source"], "video,中文.mp4")
            self.assertEqual(float(rows[0]["anchor_y"]), 80)

    def test_closed_track_statistics_do_not_accumulate_in_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            export = RunExports(Path(temporary))
            try:
                for index in range(100):
                    target = TargetRecord(index + 1, 0, "person", "pedestrian", 0.9, (1, 2, 3, 4), (2, 4))
                    export.write(FrameRecord("s", "v", index, index / 10, "video_pts", 100, 100, [target]), {index + 1})
                    self.assertEqual(len(export.active), 1)
                export.finish("eof")
                self.assertEqual(export.closed_tracks, 100)
            finally:
                export.close()


class TrackingInputTests(unittest.TestCase):
    def test_early_input_failure_writes_summary(self):
        import cv2

        cases = ("closed", "missing_fps")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "video.mp4"
                source.touch()
                capture = Mock()
                capture.isOpened.return_value = case != "closed"
                if case == "missing_fps":
                    capture.get.side_effect = lambda prop: {
                        cv2.CAP_PROP_FPS: 0.0,
                    }.get(prop, 0.0)
                with patch("cv2.VideoCapture", return_value=capture):
                    with self.assertRaises(ValueError):
                        track_video(source, output=root / "output")
                capture.release.assert_called_once()
                summary = json.loads(
                    (root / "output" / "summary.json").read_text(encoding="utf-8")
                )
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["stop_reason"], "error")
                self.assertEqual(summary["frames_processed"], 0)

    def test_model_initialization_error_is_recorded_and_capture_is_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "video.mp4"
            source.touch()
            capture = Mock()
            capture.get.side_effect = [10.0, 4.0]
            with patch("cv2.VideoCapture", return_value=capture), \
                    patch("tracking.core.load_model", side_effect=RuntimeError("model unavailable")):
                with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                    track_video(source, output=root / "output")
            capture.release.assert_called_once()
            summary = json.loads((root / "output" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["frames_processed"], 0)
            self.assertIsNone(summary["session_id"])
            self.assertEqual(summary["error"]["type"], "RuntimeError")

    def test_invalid_frame_and_time_do_not_advance_model(self):
        with patch("tracking.core.load_model", return_value=Mock(names={0: "person"})):
            session = TrackingSession("test.mp4", "unused.pt", DEFAULT_TRACKER)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        invalid = ((image.astype(np.float32), 0, "video_pts"), (image[:, :, 0], 0, "video_pts"),
                   (image, float("nan"), "video_pts"), (image, -1, "video_pts"), (image, 0, "unknown"))
        for frame, timestamp, kind in invalid:
            with self.subTest(timestamp=timestamp, kind=kind), self.assertRaises(ValueError):
                session.process(frame, timestamp, kind)
        session.previous_time = 1
        with self.assertRaises(ValueError):
            session.process(image, 1, "video_pts")
        session.frame_shape = (50, 100, 3)
        with self.assertRaisesRegex(ValueError, "分辨率"):
            session.process(image, 2, "video_pts")
        session.model.track.assert_not_called()
        self.assertEqual(session.frame_index, 0)

    def test_invalid_fps_is_rejected(self):
        for fps in (0, -1, float("inf"), float("nan")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                VideoClock(fps)


class TrackingBatchTests(unittest.TestCase):
    def test_batch_cli_reports_failure_and_interrupt_with_nonzero_exit_codes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for status, code in (("completed_with_errors", 1), ("interrupted", 130)):
                write_json(root / "batch_summary.json", {"status": status, "results": [], "videos_discovered": 1})
                with self.subTest(status=status), patch("tracking.__main__.track_directory", return_value=root), \
                        patch("sys.stdout", new_callable=io.StringIO), self.assertRaises(SystemExit) as exit_context:
                    main(["--source", str(root)])
                self.assertEqual(exit_context.exception.code, code)

    def test_bad_video_does_not_block_next_video_and_manifest_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.mp4").touch()
            (root / "b.MP4").touch()
            (root / "ignore.txt").touch()

            def run(source, *, output, **options):
                output.mkdir()
                if source.name == "a.mp4":
                    raise ValueError("bad video")
                write_json(output / "summary.json", {"status": "completed", "stop_reason": "eof",
                                                     "frames_processed": 2, "session_id": "second"})

            with patch("tracking.batch.track_video", side_effect=run) as process:
                output = track_directory(root, output=root / "output", device="cpu")
            summary = json.loads((output / "batch_summary.json").read_text())
            self.assertEqual(process.call_count, 2)
            self.assertEqual(summary["status"], "completed_with_errors")
            self.assertEqual([r["status"] for r in summary["results"]], ["failed", "completed"])
            self.assertEqual(summary["videos_not_processed"], 0)

    def test_interrupt_stops_batch_before_next_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a.mp4", "b.mp4"):
                (root / name).touch()
            with patch("tracking.batch.track_video", side_effect=KeyboardInterrupt) as process:
                output = track_directory(root, output=root / "output")
            summary = json.loads((output / "batch_summary.json").read_text())
            self.assertEqual(process.call_count, 1)
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(summary["videos_not_processed"], 1)

    def test_empty_directory_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "没有支持"):
                track_directory(root, output=root / "output")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
