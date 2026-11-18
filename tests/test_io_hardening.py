import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from analysis.__main__ import main as analysis_main
from analysis.calibration import RoadCalibrator
from analysis.evaluate import evaluate_run
from analysis.pipeline import (
    ConflictPerceptionPipeline,
    LiveStreamClock,
    is_live_source,
    parse_video_source,
)
from analysis.schema import (
    SchemaValidationError,
    validate_batch_summary,
    validate_run,
    validate_speeds_csv,
)
from analysis.speed import SpeedEstimator
from detection.core import prepare_runtime
from tracking.core import (
    DEFAULT_TRACKER,
    TrackingSession,
    normalize_session_source,
    read_tracker_config,
    redact_stream_url,
)
from tracking.records import FrameRecord, extract_tracks

prepare_runtime()
from ultralytics.engine.results import Boxes, Results
from ultralytics.trackers.byte_tracker import BYTETracker


def _sample_calibrator() -> RoadCalibrator:
    return RoadCalibrator(
        [(10.0, 10.0), (290.0, 10.0), (290.0, 190.0), (10.0, 190.0)],
        width_m=10.0,
        length_m=20.0,
    )


def _write_dummy_video(path: Path, n_frames: int = 5, fps: float = 10.0, size=(64, 64)):
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 20 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_tracks_jsonl(path: Path, frames: list[dict]):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in frames) + "\n",
        encoding="utf-8",
    )


def _make_track_row(
    frame_index: int,
    timestamp_s: float,
    tracks: list[dict],
    *,
    session_id: str = "sess-1",
    width: int = 300,
    height: int = 200,
) -> dict:
    return {
        "schema_version": 1,
        "session_id": session_id,
        "source": "synthetic.mp4",
        "frame_index": frame_index,
        "timestamp_s": timestamp_s,
        "timestamp_source": "video_pts",
        "width": width,
        "height": height,
        "tracks": tracks,
    }


def _make_target_dict(track_id: int, y_bottom: float, cls_name: str = "car", cat: str = "vehicle") -> dict:
    return {
        "track_id": track_id,
        "class_id": 2 if cat == "vehicle" else 0,
        "class_name": cls_name,
        "category": cat,
        "confidence": 0.9,
        "bbox_xyxy": [100.0, y_bottom - 30.0, 140.0, y_bottom],
        "anchor_xy": [120.0, y_bottom],
        "anchor_type": "bbox_bottom_center",
        "observation": "detected",
    }


class SchemaCliWarningTests(unittest.TestCase):
    def test_python_m_analysis_schema_emits_no_runtime_warning(self):
        proc = subprocess.run(
            [sys.executable, "-W", "error::RuntimeWarning", "-m", "analysis.schema", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertNotIn("RuntimeWarning", proc.stderr)


class OfflineReplayParityTests(unittest.TestCase):
    def test_process_tracks_track_buffer_plus_one_matches_online_retained_ids(self):
        tracker_cfg = {**read_tracker_config(DEFAULT_TRACKER), "track_buffer": 2}
        tracker = BYTETracker(SimpleNamespace(**tracker_cfg))
        fake_session = SimpleNamespace(
            model=SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker]))
        )
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        # 5 帧序列: 0 帧出现, 1/2/3 帧连续漏检 (第 3 帧 gap == track_buffer + 1), 第 4 帧恢复同一 ID
        det_sequence = (
            [[80, 120, 120, 160, 0.9, 2]],
            [],
            [],
            [],
            [[80, 132, 120, 172, 0.9, 2]],
        )

        calib = _sample_calibrator()
        online_estimator = SpeedEstimator(stationary_min_frames=2, reliable_min_span_s=0.05)
        online_frames = []
        online_retained_per_frame = []
        online_readings = []

        for idx, rows in enumerate(det_sequence):
            boxes = Boxes(np.asarray(rows, dtype=np.float32).reshape(-1, 6), image.shape[:2]).numpy()
            tracks = tracker.update(boxes).reshape(-1, 8)
            result = Results(image, "synthetic", {2: "car"}, boxes=tracks[:, :7])
            extracted = extract_tracks(result)
            ts = idx * 0.1
            record = FrameRecord("s1", "v.mp4", idx, ts, "fps_estimate", 300, 200, extracted)
            retained = TrackingSession.retained_ids(fake_session)
            online_retained_per_frame.append(retained)
            for t in extracted:
                if t.track_id is None:
                    continue
                x_m, y_m = calib.pixel_to_ground(*t.anchor_xy)
                reading = online_estimator.update(
                    t.track_id, t.category, ts, x_m, y_m, object_kind=t.class_name
                )
                online_readings.append((idx, reading.track_id, round(reading.speed_kmh, 1)))
            online_estimator.prune_stale_tracks(retained)
            online_frames.append({
                "schema_version": 1,
                "session_id": record.session_id,
                "source": record.source,
                "frame_index": record.frame_index,
                "timestamp_s": record.timestamp_s,
                "timestamp_source": record.timestamp_source,
                "width": record.width,
                "height": record.height,
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "class_id": t.class_id,
                        "class_name": t.class_name,
                        "category": t.category,
                        "confidence": t.confidence,
                        "bbox_xyxy": list(t.bbox_xyxy),
                        "anchor_xy": list(t.anchor_xy),
                        "anchor_type": "bbox_bottom_center",
                        "observation": "detected",
                    }
                    for t in extracted
                ],
            })

        # 在线模式第 3 帧 (gap == track_buffer + 1 == 3) 仍保留 track 1，第 4 帧恢复同 ID
        self.assertIn(1, online_retained_per_frame[3])
        self.assertEqual(online_readings[-1][1], 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path = root / "bytetrack.yaml"
            cfg_path.write_text(
                "tracker_type: bytetrack\n"
                "track_high_thresh: 0.5\n"
                "track_low_thresh: 0.1\n"
                "new_track_thresh: 0.6\n"
                "track_buffer: 2\n"
                "match_thresh: 0.8\n"
                "fuse_score: true\n",
                encoding="utf-8",
            )
            tracks_path = root / "tracks.jsonl"
            _write_tracks_jsonl(tracks_path, online_frames)

            pipe = ConflictPerceptionPipeline(
                calibrator=calib,
                tracker_config=str(cfg_path),
                stationary_min_frames=2,
                reliable_min_span_s=0.05,
                fps=10.0,
            )
            out_dir = root / "out"
            pipe.process_tracks(tracks_path, out_dir)
            with (out_dir / "speeds.csv").open(encoding="utf-8", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            offline_readings = [
                (int(r["frame_idx"]), int(r["track_id"]), float(r["speed_kmh"]))
                for r in csv_rows
            ]
            self.assertEqual(offline_readings, online_readings)

            # 第 3 帧恰好处于 gap == track_buffer + 1 边界: 必须仍然保留在速度池中
            gap_boundary_frames = [
                _make_track_row(0, 0.0, [_make_target_dict(1, 160.0)]),
                _make_track_row(1, 0.1, []),
                _make_track_row(2, 0.2, []),
                _make_track_row(3, 0.3, []),
            ]
            tracks_boundary = root / "tracks_boundary.jsonl"
            _write_tracks_jsonl(tracks_boundary, gap_boundary_frames)
            pipe_boundary = ConflictPerceptionPipeline(
                calibrator=calib,
                tracker_config=str(cfg_path),
                fps=10.0,
            )
            pipe_boundary.process_tracks(tracks_boundary, root / "out_boundary")
            self.assertIn(1, pipe_boundary.speed_estimator.tracks)

            # 第 4 帧超过 track_buffer + 1 (4 帧全空): 离线在第 4 帧剪枝应与在线一致
            long_loss_frames = [
                *gap_boundary_frames,
                _make_track_row(4, 0.4, []),
            ]
            tracks_long = root / "tracks_long.jsonl"
            _write_tracks_jsonl(tracks_long, long_loss_frames)
            pipe_long = ConflictPerceptionPipeline(
                calibrator=calib,
                tracker_config=str(cfg_path),
                fps=10.0,
            )
            pipe_long.process_tracks(tracks_long, root / "out_long")
            self.assertEqual(pipe_long.speed_estimator.tracks, {})

    def test_process_tracks_rejects_resolution_change_across_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks_path = root / "tracks.jsonl"
            _write_tracks_jsonl(tracks_path, [
                _make_track_row(0, 0.0, [], width=300, height=200),
                _make_track_row(1, 0.1, [], width=320, height=240),
            ])
            pipe = ConflictPerceptionPipeline(calibrator=_sample_calibrator(), fps=10.0)
            out_dir = root / "out"
            with self.assertRaisesRegex(ValueError, "分辨率"):
                pipe.process_tracks(tracks_path, out_dir)
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["stop_reason"], "initialization")

    def test_process_tracks_frame_jump_uses_cap_grab_to_align_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks_path = root / "tracks.jsonl"
            _write_tracks_jsonl(tracks_path, [
                _make_track_row(0, 0.0, [], width=64, height=64),
                _make_track_row(3, 0.3, [], width=64, height=64),
                _make_track_row(5, 0.5, [], width=64, height=64),
            ])

            frames_bank = [np.full((64, 64, 3), i, dtype=np.uint8) for i in range(6)]
            cursor = {"idx": 0}
            read_indices = []
            grabbed_indices = []

            fake_cap = mock.Mock()
            fake_cap.isOpened.return_value = True
            fake_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 10.0,
                cv2.CAP_PROP_FRAME_WIDTH: 64.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 64.0,
            }.get(prop, 0.0)

            def fake_grab():
                if cursor["idx"] >= len(frames_bank):
                    return False
                grabbed_indices.append(cursor["idx"])
                cursor["idx"] += 1
                return True

            def fake_read():
                if cursor["idx"] >= len(frames_bank):
                    return False, None
                read_indices.append(cursor["idx"])
                frame = frames_bank[cursor["idx"]]
                cursor["idx"] += 1
                return True, frame

            fake_cap.grab.side_effect = fake_grab
            fake_cap.read.side_effect = fake_read

            fake_writer = mock.Mock()
            fake_writer.isOpened.return_value = True

            pipe = ConflictPerceptionPipeline(calibrator=_sample_calibrator())
            with mock.patch("analysis.pipeline.cv2.VideoCapture", return_value=fake_cap), \
                    mock.patch("analysis.pipeline.cv2.VideoWriter", return_value=fake_writer):
                summary = pipe.process_tracks(tracks_path, root / "out", video_path=root / "v.mp4")

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["frames_processed"], 3)
            self.assertEqual(grabbed_indices, [1, 2, 4])
            self.assertEqual(read_indices, [0, 3, 5])

            # 跳帧超过视频总帧数时应报错而非错位渲染
            _write_tracks_jsonl(tracks_path, [
                _make_track_row(0, 0.0, [], width=64, height=64),
                _make_track_row(8, 0.8, [], width=64, height=64),
            ])
            cursor["idx"] = 0
            with mock.patch("analysis.pipeline.cv2.VideoCapture", return_value=fake_cap), \
                    mock.patch("analysis.pipeline.cv2.VideoWriter", return_value=fake_writer):
                with self.assertRaises(RuntimeError):
                    pipe.process_tracks(tracks_path, root / "out_short", video_path=root / "v.mp4")

    def test_process_tracks_inferred_fps_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipe = ConflictPerceptionPipeline(calibrator=_sample_calibrator())

            # 极小 FPS (median_dt = 2_000_000 -> round 舍入为 0.0) 应被拒绝
            too_slow = root / "slow.jsonl"
            _write_tracks_jsonl(too_slow, [
                _make_track_row(0, 0.0, []),
                _make_track_row(1, 2_000_000.0, []),
            ])
            with self.assertRaises(ValueError):
                pipe.process_tracks(too_slow, root / "out_slow")

            # 极大/溢出 FPS (median_dt = 1e-320 -> inf / >1000) 应抛 ValueError 而非 OverflowError
            too_fast = root / "fast.jsonl"
            _write_tracks_jsonl(too_fast, [
                _make_track_row(0, 0.0, []),
                _make_track_row(1, 1e-320, []),
            ])
            with self.assertRaises(ValueError):
                pipe.process_tracks(too_fast, root / "out_fast")

            # 边界内合法 FPS (1e-3 及 1000.0) 可正常运行
            ok_low = root / "ok_low.jsonl"
            _write_tracks_jsonl(ok_low, [
                _make_track_row(0, 0.0, []),
                _make_track_row(1, 1000.0, []),
            ])
            s_low = pipe.process_tracks(ok_low, root / "out_ok_low")
            self.assertEqual(s_low["configuration"]["fps_used"], 0.001)

            ok_high = root / "ok_high.jsonl"
            _write_tracks_jsonl(ok_high, [
                _make_track_row(0, 0.0, []),
                _make_track_row(1, 0.001, []),
            ])
            s_high = pipe.process_tracks(ok_high, root / "out_ok_high")
            self.assertEqual(s_high["configuration"]["fps_used"], 1000.0)


class LiveClockAndRedactionTests(unittest.TestCase):
    def test_live_stream_clock_nonzero_initial_pts_and_huge_pts_monotonicity(self):
        times = iter([100.00, 100.04, 100.08, 100.10, 100.15])
        with mock.patch("analysis.pipeline.time.perf_counter", side_effect=lambda: next(times)):
            clock = LiveStreamClock()
            t0, k0 = clock.timestamp(0, 60000.0)
            t1, k1 = clock.timestamp(1, float("nan"))
            t2, k2 = clock.timestamp(2, 60000.0)  # 重复 PTS -> 回退墙钟
            t3, k3 = clock.timestamp(3, 60200.0)  # 有效递增 PTS -> 重新锚定
            t4, k4 = clock.timestamp(4, -1.0)     # 负 PTS -> 基于新锚点推进墙钟

        self.assertEqual((t0, k0), (60.0, "video_pts"))
        self.assertEqual(k1, "wall_clock")
        self.assertAlmostEqual(t1 - t0, 0.04, places=6)
        self.assertEqual(k2, "wall_clock")
        self.assertAlmostEqual(t2 - t1, 0.04, places=6)
        self.assertEqual((t3, k3), (60.2, "video_pts"))
        self.assertEqual(k4, "wall_clock")
        self.assertAlmostEqual(t4 - t3, 0.05, places=6)

        # 超大异常 PTS (1e13 秒, float64 下 1e13 + 1e-4 == 1e13) 仍能严格递增
        with mock.patch("analysis.pipeline.time.perf_counter", return_value=200.0):
            huge_clock = LiveStreamClock()
            h0, hk0 = huge_clock.timestamp(0, 1e16)
            h1, hk1 = huge_clock.timestamp(1, float("nan"))
        self.assertEqual((h0, hk0), (1e13, "video_pts"))
        self.assertEqual(hk1, "wall_clock")
        self.assertGreater(h1, h0)

    def test_redact_stream_url_handles_unencoded_and_standard_urls(self):
        cases = [
            ("rtsp://admin:p[ss@192.168.1.20/stream", "rtsp://***@192.168.1.20/stream"),
            ("rtsp://admin:p]ss@[2001:db8::1]:554/stream", "rtsp://***@[2001:db8::1]:554/stream"),
            ("rtsp://admin:pass?123@192.168.1.20:554/stream?ch=1", "rtsp://***@192.168.1.20:554/stream?ch=1"),
            ("rtsp://admin:pass#123@192.168.1.20/stream", "rtsp://***@192.168.1.20/stream"),
            ("rtsp://admin:pass/123@192.168.1.20:554/stream", "rtsp://***@192.168.1.20:554/stream"),
            ("rtsp://admin:1234/abc@192.168.1.20/stream", "rtsp://***@192.168.1.20/stream"),
            ("rtsp://admin:p[1]?2#3/4@localhost:8554/live?k=v", "rtsp://***@localhost:8554/live?k=v"),
            (
                "rtsp://admin:p?a@b/c@cam-01.campus.edu:554/live?email=u@d.com",
                "rtsp://***@cam-01.campus.edu:554/live?email=u@d.com",
            ),
            ("rtsp://admin@192.168.1.20/stream", "rtsp://***@192.168.1.20/stream"),
            ("rtsp://admin:@192.168.1.20/stream", "rtsp://***@192.168.1.20/stream"),
            ("rtsp://admin:p%23ss@192.168.1.20:554/stream?a=1#f", "rtsp://***@192.168.1.20:554/stream?a=1#f"),
            ("rtsp://admin:p@ss@192.168.1.20:554/stream", "rtsp://***@192.168.1.20:554/stream"),
            ("rtsp://192.168.1.20:554/stream?ch=1", "rtsp://192.168.1.20:554/stream?ch=1"),
            ("rtsp://192.168.1.20:554/stream?email=a@b.com", "rtsp://192.168.1.20:554/stream?email=a@b.com"),
            ("rtsp://[bad-ipv6]/stream", "rtsp://[bad-ipv6]/stream"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(redact_stream_url(raw), expected)

    def test_ascii_digits_and_camera_prefix_consistency(self):
        for bad in ("²", "camera:abc", "camera:", "camera:²", "camera:-1"):
            with self.subTest(bad=bad):
                self.assertFalse(is_live_source(bad))
                if bad != "²":
                    with self.assertRaises(ValueError):
                        parse_video_source(bad)
                    with self.assertRaises(ValueError):
                        normalize_session_source(bad)
                else:
                    # 非 ASCII 上标数字不被当作摄像头索引
                    self.assertFalse(parse_video_source(bad)[2])
                    self.assertFalse(normalize_session_source(bad).startswith("camera:"))

        self.assertTrue(is_live_source(" 02 "))
        self.assertTrue(is_live_source("CAMERA: 02 "))
        self.assertEqual(parse_video_source("CAMERA: 02 "), (2, "camera:2", True))
        self.assertEqual(normalize_session_source("CAMERA: 02 "), "camera:2")


class PipelineHardeningTests(unittest.TestCase):
    def test_pipeline_init_rejects_bool_for_all_numeric_params(self):
        calib = _sample_calibrator()
        bool_kwargs = [
            {"iou": True},
            {"fps": True},
            {"zone_margin_m": False},
            {"window_seconds": True},
            {"stationary_seconds": True},
            {"alert_confirm_seconds": True},
            {"alert_cooldown_seconds": False},
            {"ema_alpha": True},
            {"speed_limit_kmh": True},
            {"ttc_critical_s": True},
            {"ttc_caution_s": True},
            {"horizon_s": True},
            {"stationary_speed_ms": True},
            {"reliable_min_span_s": False},
            {"parking_zone_max_speed_kmh": False},
            {"imgsz": True},
            {"window_size": True},
            {"stationary_min_frames": True},
            {"category_confirm_frames": True},
            {"alert_confirm_frames": True},
            {"alert_cooldown_frames": False},
        ]
        for kwargs in bool_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ConflictPerceptionPipeline(calibrator=calib, **kwargs)

    def test_process_video_frame30_zero_elapsed_and_monotonic_perf_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vpath = root / "v31.mp4"
            _write_dummy_video(vpath, n_frames=31)
            fake_session = mock.Mock()
            fake_rec = mock.Mock()
            fake_rec.tracks = []
            fake_session.process.return_value = fake_rec
            fake_session.retained_ids.return_value = set()

            pipe = ConflictPerceptionPipeline(calibrator=_sample_calibrator())
            out_dir = root / "out"
            # 模拟 perf_counter 恒定不变 (elapsed_now == 0.0) 与 time.time 回拨
            with mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session), \
                    mock.patch("analysis.pipeline.time.perf_counter", return_value=10.0), \
                    mock.patch("analysis.pipeline.time.time", side_effect=[1000.0, 500.0, 100.0]):
                summary = pipe.process_video(vpath, out_dir)

            self.assertEqual(summary["frames_processed"], 31)
            self.assertGreaterEqual(summary["elapsed_seconds"], 0.0)
            self.assertTrue(validate_run(out_dir)["valid"])

    def test_finally_close_failure_still_writes_summary_and_destroys_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vpath = root / "v.mp4"
            _write_dummy_video(vpath, n_frames=3)
            pipe = ConflictPerceptionPipeline(calibrator=_sample_calibrator())

            # 1) process_video: 主循环无异常但 resources.close() 失败
            fake_session = mock.Mock()
            fake_rec = mock.Mock()
            fake_rec.tracks = []
            fake_session.process.return_value = fake_rec
            fake_session.retained_ids.return_value = set()

            fake_writer = mock.Mock()
            fake_writer.isOpened.return_value = True
            fake_writer.release.side_effect = OSError("writer close failed")

            out1 = root / "pv_close_fail"
            with mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session), \
                    mock.patch("analysis.pipeline.cv2.VideoWriter", return_value=fake_writer), \
                    mock.patch("analysis.pipeline.cv2.imshow"), \
                    mock.patch("analysis.pipeline.cv2.waitKey", return_value=0), \
                    mock.patch("analysis.pipeline.cv2.destroyAllWindows") as m_destroy:
                with self.assertRaisesRegex(OSError, "writer close failed"):
                    pipe.process_video(vpath, out1, show_window=True)

            m_destroy.assert_called_once()
            s1 = json.loads((out1 / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(s1["status"], "failed")
            self.assertEqual(s1["stop_reason"], "exception")
            self.assertIn("OSError: writer close failed", s1["error"])
            for field in ("source_video", "annotated_output", "conflicts_jsonl", "speeds_csv"):
                self.assertIn(field, s1)

            # 2) process_video: 主循环异常且 close 也失败 -> 保留主循环异常且仍落盘 summary 和清理窗口
            fake_session.process.side_effect = RuntimeError("infer boom")
            out2 = root / "pv_both_fail"
            with mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session), \
                    mock.patch("analysis.pipeline.cv2.VideoWriter", return_value=fake_writer), \
                    mock.patch("analysis.pipeline.cv2.destroyAllWindows") as m_destroy2:
                with self.assertRaisesRegex(RuntimeError, "infer boom"):
                    pipe.process_video(vpath, out2, show_window=True)

            m_destroy2.assert_called_once()
            s2 = json.loads((out2 / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(s2["status"], "failed")
            self.assertEqual(s2["stop_reason"], "exception")
            self.assertIn("RuntimeError: infer boom", s2["error"])

            # 3) process_tracks: 主循环无异常但 close 失败 -> 落盘含 input_mode 与对称字段的失败 summary 并抛异常
            tracks_path = root / "tracks.jsonl"
            _write_tracks_jsonl(tracks_path, [
                _make_track_row(0, 0.0, [], width=64, height=64),
                _make_track_row(1, 0.1, [], width=64, height=64),
            ])
            out3 = root / "pt_close_fail"
            with mock.patch("analysis.pipeline.cv2.VideoWriter", return_value=fake_writer), \
                    mock.patch("analysis.pipeline.cv2.imshow"), \
                    mock.patch("analysis.pipeline.cv2.waitKey", return_value=0), \
                    mock.patch("analysis.pipeline.cv2.destroyAllWindows") as m_destroy3:
                with self.assertRaisesRegex(OSError, "writer close failed"):
                    pipe.process_tracks(tracks_path, out3, video_path=vpath, show_window=True)

            m_destroy3.assert_called_once()
            s3 = json.loads((out3 / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(s3["input_mode"], "tracks_jsonl")
            self.assertEqual(s3["status"], "failed")
            self.assertEqual(s3["stop_reason"], "exception")
            for field in ("source_video", "annotated_output", "conflicts_jsonl", "speeds_csv"):
                self.assertIn(field, s3)

    def test_init_keyboard_interrupt_writes_interrupted_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vpath = root / "v.mp4"
            _write_dummy_video(vpath, n_frames=3)
            pipe = ConflictPerceptionPipeline(calibrator=_sample_calibrator())

            out_v = root / "init_intr_video"
            with mock.patch("analysis.pipeline.TrackingSession", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    pipe.process_video(vpath, out_v)
            sv = json.loads((out_v / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(sv["status"], "interrupted")
            self.assertEqual(sv["stop_reason"], "interrupt")

            out_t = root / "init_intr_tracks"
            tracks_path = root / "tracks.jsonl"
            tracks_path.write_text("", encoding="utf-8")
            with mock.patch.object(pipe, "_load_validated_tracks", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    pipe.process_tracks(tracks_path, out_t)
            st = json.loads((out_t / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(st["status"], "interrupted")
            self.assertEqual(st["stop_reason"], "interrupt")
            self.assertEqual(st["input_mode"], "tracks_jsonl")


class CliSchemaEvaluateHardeningTests(unittest.TestCase):
    def test_cli_expands_tilde_and_defers_synthetic_calib_until_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_home = root / "home"
            fake_home.mkdir()
            empty_dir = root / "empty_videos"
            empty_dir.mkdir()
            vpath = root / "clip.mp4"
            _write_dummy_video(vpath, n_frames=3)

            prev_cwd = Path.cwd()
            try:
                os.chdir(root)
                with mock.patch.dict(os.environ, {"HOME": str(fake_home), "USERPROFILE": str(fake_home)}):
                    # 1) 空批处理目录: 报错且不残留 ~ 目录或 calibration.synthetic.yaml
                    with self.assertRaisesRegex(ValueError, "没有支持"):
                        analysis_main(["--source", str(empty_dir), "--demo", "--output", "~/batch_empty"])
                    self.assertFalse((root / "~").exists())
                    self.assertFalse((fake_home / "batch_empty").exists())

                    # 2) 非法设备号初始化失败: 报错且不残留 ~ 目录或 calibration.synthetic.yaml
                    with self.assertRaisesRegex(ValueError, "无效的摄像头标识"):
                        analysis_main(["--source", "camera:abc", "--demo", "--output", "~/bad_cam"])
                    self.assertFalse((root / "~").exists())
                    self.assertFalse((fake_home / "bad_cam").exists())

                    # 3) 正常执行: calibration.synthetic.yaml 落在展开后的 home/ok_run 下
                    fake_session = mock.Mock()
                    fake_rec = mock.Mock()
                    fake_rec.tracks = []
                    fake_session.process.return_value = fake_rec
                    fake_session.retained_ids.return_value = set()
                    with mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session):
                        rc = analysis_main(["--source", str(vpath), "--demo", "--output", "~/ok_run"])
                    self.assertEqual(rc, 0)
                    self.assertFalse((root / "~").exists())
                    self.assertTrue((fake_home / "ok_run" / "calibration.synthetic.yaml").is_file())
                    self.assertTrue(validate_run(fake_home / "ok_run")["valid"])
            finally:
                os.chdir(prev_cwd)

    def test_schema_speeds_csv_and_batch_summary_hardening(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            header = [
                "schema_version", "frame_idx", "timestamp_s", "track_id", "category",
                "speed_kmh", "risk_level", "event_id", "is_reliable", "is_stationary",
            ]
            # 多余未命名列 (trailing extra value -> None in row)
            extra_csv = root / "extra.csv"
            with extra_csv.open("w", newline="", encoding="utf-8") as stream:
                w = csv.writer(stream)
                w.writerow(header)
                w.writerow([1, 0, 0.0, 1, "vehicle", 12.0, "LEVEL_0_SAFE", "", 1, 0, "EXTRA"])
            with self.assertRaisesRegex(SchemaValidationError, "多余"):
                validate_speeds_csv(extra_csv)

            # 空 category / 空白 risk_level
            for bad_cat, bad_risk in (("", "LEVEL_0_SAFE"), ("vehicle", "   ")):
                bad_csv = root / "bad_text.csv"
                with bad_csv.open("w", newline="", encoding="utf-8") as stream:
                    w = csv.writer(stream)
                    w.writerow(header)
                    w.writerow([1, 0, 0.0, 1, bad_cat, 12.0, bad_risk, "", 1, 0])
                with self.assertRaises(SchemaValidationError):
                    validate_speeds_csv(bad_csv)

            # validate_batch_summary 拒绝 videos_not_processed > videos_discovered
            bad_batch = {
                "schema_version": 1,
                "kind": "analysis_batch",
                "source_dir": str(root),
                "output_dir": str(root),
                "status": "completed",
                "videos_discovered": 1,
                "videos_not_processed": 2,
                "total_alert_events": 0,
                "total_speeding_alerts": 0,
                "total_caution_events": 0,
                "total_critical_conflicts": 0,
                "results": [],
            }
            with self.assertRaisesRegex(SchemaValidationError, "videos_not_processed"):
                validate_batch_summary(bad_batch)

    def test_evaluate_run_rejects_nan_infinity_even_without_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text('{"frames_processed": NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_run(root, validate=False)

            (root / "summary.json").write_text('{"frames_processed": 1}', encoding="utf-8")
            (root / "conflicts.jsonl").write_text('{"frame_index": 0, "x": Infinity}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_run(root, validate=False)

            (root / "conflicts.jsonl").write_text("", encoding="utf-8")
            (root / "events.jsonl").write_text('{"event_id": 1, "duration_s": -Infinity}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_run(root, validate=False)


if __name__ == "__main__":
    unittest.main()
