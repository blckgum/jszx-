"""真实 YOLOv8 + ByteTrack 端到端验证。素材是构造视频，不是校园实拍基准。"""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from detection.core import DEFAULT_WEIGHTS
from tracking.core import track_video

ULTRALYTICS_ROOT = Path(importlib.util.find_spec("ultralytics").origin).parent
BUS_IMAGE = ULTRALYTICS_ROOT / "assets" / "bus.jpg"


def make_sample_video(destination, frames=18):
    # 两幅并排的公交车样例构成多车、多人画面；平移模拟小幅运动，8/9 帧模拟漏检。
    # 静态样例构造仅用于功能验证，不作为真实道路跟踪效果或速度真值。
    image = cv2.imread(str(BUS_IMAGE))
    image = cv2.resize(image, (405, 540))
    scene = np.concatenate((image, cv2.flip(image, 1)), axis=1)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (810, 540))
    if not writer.isOpened():
        raise RuntimeError("无法创建集成测试视频")
    try:
        for index in range(frames):
            matrix = np.float32([[1, 0, index], [0, 1, 0]])
            frame = cv2.warpAffine(scene, matrix, (810, 540))
            if index in (8, 9):
                frame[:] = 0
            writer.write(frame)
    finally:
        writer.release()


@unittest.skipUnless(DEFAULT_WEIGHTS.is_file() and BUS_IMAGE.is_file(), "需要本地 YOLOv8n 权重和官方 bus 样例")
class TrackingIntegrationTests(unittest.TestCase):
    def test_two_videos_have_persistent_ids_and_independent_sessions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a, source_b = root / "video_a.mp4", root / "video_b.mp4"
            make_sample_video(source_a)
            shutil.copyfile(source_a, source_b)
            records_by_video = []
            for index, source in enumerate((source_a, source_b)):
                output = track_video(source, output=root / f"output_{index}", device="cpu")
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


if __name__ == "__main__":
    unittest.main()
