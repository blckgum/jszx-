"""真实 YOLOv8 + ByteTrack 端到端验证。素材是构造视频，不是校园实拍基准。"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2

from detection.core import DEFAULT_WEIGHTS
from tracking.batch import track_directory
from tracking.core import TrackingSession, track_video

from tracking.demo import BUS_IMAGE, make_sample_video


@unittest.skipUnless(DEFAULT_WEIGHTS.is_file() and BUS_IMAGE.is_file(), "需要本地 YOLOv8n 权重和官方 bus 样例")
class TrackingIntegrationTests(unittest.TestCase):
    def test_two_videos_have_persistent_ids_and_independent_sessions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a, source_b = root / "video_a.mp4", root / "video_b.mp4"
            make_sample_video(source_a)
            shutil.copyfile(source_a, source_b)
            batch = track_directory(root, output=root / "batch", device="cpu")
            manifest = json.loads((batch / "batch_summary.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["videos_not_processed"], 0)
            records_by_video = []
            for item in manifest["results"]:
                output = Path(item["output"])
                records = [json.loads(line) for line in (output / "tracks.jsonl").read_text().splitlines()]
                records_by_video.append(records)
                self.assertEqual([r["frame_index"] for r in records], list(range(18)))
                self.assertEqual(records[8]["tracks"], [])
                self.assertEqual(records[9]["tracks"], [])
                first_vehicles = {t["track_id"] for t in records[0]["tracks"] if t["category"] == "vehicle"}
                first_people = {t["track_id"] for t in records[0]["tracks"] if t["category"] == "pedestrian"}
                self.assertGreaterEqual(len(first_vehicles), 2)
                self.assertGreaterEqual(len(first_people), 2)
                final_ids = {t["track_id"] for t in records[-1]["tracks"]}
                self.assertTrue(first_vehicles <= final_ids, "构造视频的公交车应在短暂空帧后恢复原 ID")
                for frame_index, record in enumerate(records):
                    self.assertAlmostEqual(record["timestamp_s"], frame_index / 10, places=5)
                    for target in record["tracks"]:
                        x1, y1, x2, y2 = target["bbox_xyxy"]
                        self.assertTrue(0 <= x1 < x2 <= record["width"])
                        self.assertTrue(0 <= y1 < y2 <= record["height"])
                capture = cv2.VideoCapture(str(output / "annotated.mp4"))
                decoded = 0
                while capture.read()[0]:
                    decoded += 1
                capture.release()
                self.assertEqual(decoded, 18)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["stop_reason"], "eof")
                self.assertGreater(summary["processing_fps"], 0)
                self.assertEqual(summary["status"], "completed")
                events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
                recovered = {e["track_id"] for e in events if e["event"] == "recovered"}
                self.assertTrue(first_vehicles <= recovered)
                tracklets = [json.loads(line) for line in (output / "tracklets.jsonl").read_text().splitlines()]
                self.assertEqual({t["track_id"] for t in tracklets}, set(summary["confirmed_track_ids"]))
                self.assertEqual(sum(t["observations"] for t in tracklets),
                                 sum(t["track_id"] is not None for r in records for t in r["tracks"]))
            self.assertNotEqual(records_by_video[0][0]["session_id"], records_by_video[1][0]["session_id"])
            # 同一 Python 进程先后处理两个视频，第二段仍从自己的新 ID 空间开始。
            ids = lambda record: {t["track_id"] for t in record["tracks"] if t["track_id"] is not None}
            self.assertEqual(min(ids(records_by_video[0][0])), 1)
            self.assertEqual(ids(records_by_video[0][0]), ids(records_by_video[1][0]))

    def test_simulated_pause_does_not_duplicate_frames_and_quit_saves_partial_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "keys.mp4"
            make_sample_video(source, frames=4)
            # 只模拟窗口事件，网络和跟踪器真实运行。实际桌面窗口仍需本地人工验证。
            with patch("tracking.core.check_display"), patch("cv2.namedWindow"), patch("cv2.imshow"), \
                    patch("cv2.destroyWindow"), patch("cv2.getWindowProperty", return_value=1), \
                    patch("cv2.waitKey", side_effect=[32, -1, ord("p"), ord("q")]):
                output = track_video(source, output=root / "output", device="cpu", show=True)
            records = [json.loads(line) for line in (output / "tracks.jsonl").read_text().splitlines()]
            self.assertEqual([record["frame_index"] for record in records], [0, 1])
            self.assertEqual(json.loads((output / "summary.json").read_text())["stop_reason"], "user_quit")
            capture = cv2.VideoCapture(str(output / "annotated.mp4"))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, 2)

    def test_inference_failure_and_interrupt_preserve_completed_results(self):
        original = TrackingSession.process
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "partial.mp4"
            make_sample_video(source, frames=4)
            for name, exception in (("failed", RuntimeError("injected inference failure")),
                                    ("interrupted", KeyboardInterrupt())):
                with self.subTest(case=name):
                    destination = root / name

                    def process(session, frame, timestamp, timestamp_source):
                        if session.frame_index == 2:
                            raise exception
                        return original(session, frame, timestamp, timestamp_source)

                    with patch.object(TrackingSession, "process", process):
                        if name == "failed":
                            with self.assertRaisesRegex(RuntimeError, "injected"):
                                track_video(source, output=destination, device="cpu")
                        else:
                            track_video(source, output=destination, device="cpu")
                    summary = json.loads((destination / "summary.json").read_text())
                    self.assertEqual(summary["status"], name)
                    self.assertEqual(summary["frames_processed"], 2)
                    records = [json.loads(line) for line in (destination / "tracks.jsonl").read_text().splitlines()]
                    self.assertEqual(len(records), 2)
                    capture = cv2.VideoCapture(str(destination / "annotated.mp4"))
                    decoded = 0
                    while capture.read()[0]:
                        decoded += 1
                    capture.release()
                    self.assertEqual(decoded, 2)
                    tracklets = [json.loads(line) for line in (destination / "tracklets.jsonl").read_text().splitlines()]
                    reason = "session_end:error" if name == "failed" else "session_end:interrupted"
                    self.assertTrue(all(t["close_reason"] == reason for t in tracklets))


if __name__ == "__main__":
    unittest.main()
