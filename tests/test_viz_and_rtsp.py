# -*- coding: utf-8 -*-
"""OBB/SSM/二阶轨迹可视化、RTSP 断流重连与 VRU/重连 CLI 开关测试。"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from analysis.__main__ import main as analysis_main
from analysis.calibration import RoadCalibrator
from analysis.conflict import ConflictAssessment, RiskLevel
from analysis.pipeline import ConflictPerceptionPipeline
from analysis.schema import load_json_strict, validate_run
from analysis.speed import SpeedReading
from analysis.visualization import EnhancedConflictVisualizer
from tracking.records import FrameRecord, TargetRecord


def _sample_calibrator() -> RoadCalibrator:
    return RoadCalibrator(
        image_points=[(200.0, 120.0), (600.0, 120.0), (760.0, 520.0), (40.0, 520.0)],
        width_m=8.0,
        length_m=24.0,
        name="viz_rtsp_test_road",
    )


def _write_dummy_video(path: Path, n_frames: int = 3, fps: float = 10.0, size=(64, 64)):
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), (i * 25) % 255, dtype=np.uint8))
    writer.release()


def _make_fake_session_with_tracks(tracks_factory):
    sess = mock.Mock()
    sess.process.side_effect = lambda frame, ts, kind: SimpleNamespace(tracks=tracks_factory())
    sess.retained_ids.side_effect = lambda: {
        t.track_id for t in tracks_factory() if t.track_id is not None
    }
    return sess


class TestEnhancedVisualizationUpgrades(unittest.TestCase):
    def setUp(self):
        self.calib = _sample_calibrator()
        self.viz = EnhancedConflictVisualizer(self.calib, speed_limit_kmh=30.0)

    def test_draw_future_trajectory_second_order_and_nan_fallback(self):
        # 1) ax=ay=0 时与纯一阶等价；给定有限加速度时应沿二阶抛物线偏折
        base = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=4.0,
            vx_ms=0.0,
            vy_ms=3.0,
            speed_ms=3.0,
            speed_kmh=10.8,
            is_speeding=False,
            heading_deg=90.0,
            is_reliable=True,
            ax_ms2=0.0,
            ay_ms2=0.0,
        )
        curved = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=4.0,
            vx_ms=0.0,
            vy_ms=3.0,
            speed_ms=3.0,
            speed_kmh=10.8,
            is_speeding=False,
            heading_deg=90.0,
            is_reliable=True,
            ax_ms2=1.2,
            ay_ms2=1.0,
        )
        f_base = np.zeros((600, 800, 3), dtype=np.uint8)
        f_curved = np.zeros((600, 800, 3), dtype=np.uint8)
        self.viz.draw_future_trajectory(f_base, base, horizon_s=2.0, steps=4)
        self.viz.draw_future_trajectory(f_curved, curved, horizon_s=2.0, steps=4)
        self.assertGreater(int(f_base.sum()), 0)
        self.assertFalse(np.array_equal(f_base, f_curved))

        # 验证末点精确命中二阶外推坐标 (x0 + vx*t + 0.5*ax*t^2, y0 + vy*t + 0.5*ay*t^2)
        expected_end = self.calib.ground_to_pixel_checked(
            3.0 + 0.5 * 1.2 * (2.0 ** 2),
            4.0 + 3.0 * 2.0 + 0.5 * 1.0 * (2.0 ** 2),
        )
        self.assertIsNotNone(expected_end)
        eu, ev = expected_end
        self.assertGreater(int(f_curved[ev, eu, 1]), 0)

        # 2) NaN/Inf/None/bool 加速度安全退化为 0.0，与一阶完全一致且无 RuntimeWarning
        for bad_ax, bad_ay in (
            (float("nan"), float("inf")),
            (None, True),
            (np.float64("nan"), np.bool_(False)),
        ):
            bad_reading = SimpleNamespace(
                track_id=1,
                category="vehicle",
                x_m=3.0,
                y_m=4.0,
                vx_ms=0.0,
                vy_ms=3.0,
                speed_ms=3.0,
                speed_kmh=10.8,
                is_speeding=False,
                heading_deg=90.0,
                is_reliable=True,
                ax_ms2=bad_ax,
                ay_ms2=bad_ay,
            )
            f_bad = np.zeros((600, 800, 3), dtype=np.uint8)
            self.viz.draw_future_trajectory(f_bad, bad_reading, horizon_s=2.0, steps=4)
            self.assertTrue(np.array_equal(f_bad, f_base))

    def test_draw_vehicle_obb_normal_fallbacks_and_behind_camera_skip(self):
        # 1) 正常朝向绘制 OBB 四边形与箭头；heading_deg 为 None/NaN 时回退 atan2(vy, vx)
        r_heading = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=4.0,
            y_m=10.0,
            vx_ms=0.0,
            vy_ms=4.0,
            speed_ms=4.0,
            speed_kmh=14.4,
            is_speeding=False,
            heading_deg=90.0,
            is_reliable=True,
        )
        r_fallback_vel = SimpleNamespace(
            track_id=1,
            category="vehicle",
            x_m=4.0,
            y_m=10.0,
            vx_ms=0.0,
            vy_ms=4.0,
            speed_ms=4.0,
            heading_deg=None,
        )
        f1 = np.zeros((600, 800, 3), dtype=np.uint8)
        f2 = np.zeros((600, 800, 3), dtype=np.uint8)
        self.viz.draw_vehicle_obb(f1, r_heading, length_m=4.5, width_m=1.9)
        self.viz.draw_vehicle_obb(f2, r_fallback_vel, length_m=4.5, width_m=1.9)
        self.assertGreater(int(f1.sum()), 0)
        self.assertTrue(np.array_equal(f1, f2))

        # 2) 静止且 heading_deg 非有限时安全回退 0 度并完成绘制
        r_stationary = SimpleNamespace(
            track_id=2,
            category="vehicle",
            x_m=4.0,
            y_m=10.0,
            vx_ms=0.0,
            vy_ms=0.0,
            heading_deg=float("nan"),
        )
        f_stat = np.zeros((600, 800, 3), dtype=np.uint8)
        self.viz.draw_vehicle_obb(f_stat, r_stationary, length_m=4.5, width_m=1.9)
        self.assertGreater(int(f_stat.sum()), 0)

        # 3) 相机后方或严重越界点（ground_to_pixel_checked 返回 None 或超大像素）必须整框跳过不画
        behind_calib = SimpleNamespace(
            image_points=self.calib.image_points,
            ground_to_pixel_checked=lambda x, y: None if x > 4.5 else (400, 300),
        )
        viz_behind = EnhancedConflictVisualizer(behind_calib)
        f_behind = np.zeros((600, 800, 3), dtype=np.uint8)
        viz_behind.draw_vehicle_obb(f_behind, r_heading, length_m=4.5, width_m=1.9)
        self.assertEqual(int(f_behind.sum()), 0)

        far_calib = SimpleNamespace(
            image_points=self.calib.image_points,
            ground_to_pixel_checked=lambda x, y: (50000, 50000),
        )
        viz_far = EnhancedConflictVisualizer(far_calib)
        f_far = np.zeros((600, 800, 3), dtype=np.uint8)
        viz_far.draw_vehicle_obb(f_far, r_heading, length_m=4.5, width_m=1.9)
        self.assertEqual(int(f_far.sum()), 0)

    def test_hud_banner_and_speed_badge_ssm_and_vru_display(self):
        reading = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=4.0,
            y_m=8.0,
            vx_ms=0.0,
            vy_ms=6.0,
            speed_ms=6.0,
            speed_kmh=21.6,
            is_speeding=False,
            heading_deg=90.0,
            is_reliable=True,
        )
        # 1) 未启用 SSM（全 None）且目标为行人：保持原有 48px 单行布局与文案
        c_cvm = ConflictAssessment(
            veh_track_id=1,
            ped_track_id=2,
            risk_level=RiskLevel.LEVEL_3_CRITICAL,
            risk_desc="cvm",
            veh_speed_kmh=21.6,
            ped_speed_ms=1.2,
            current_distance_m=3.0,
            min_distance_m=0.5,
            ttc_seconds=1.2,
            conflict_ground_xy=(4.0, 12.0),
        )
        texts_cvm = []
        orig_put_text = cv2.putText

        def _spy_put_text(img, text, org, *args, **kwargs):
            texts_cvm.append((text, org))
            return orig_put_text(img, text, org, *args, **kwargs)

        frame_cvm = np.zeros((240, 960, 3), dtype=np.uint8)
        with mock.patch("analysis.visualization.cv2.putText", side_effect=_spy_put_text):
            self.viz.draw_hud_banner(frame_cvm, [c_cvm])
            self.viz.draw_speed_badge(frame_cvm, (50, 80, 150, 180), reading, c_cvm)
        self.assertEqual(len(texts_cvm), 2)
        self.assertIn("Ped #2", texts_cvm[0][0])
        self.assertNotIn("[VRU]", texts_cvm[0][0])
        self.assertNotIn("MTTC:", texts_cvm[0][0])
        self.assertNotIn("[VRU]", texts_cvm[1][0])

        # 2) 启用 SSM 指标且目标为骑行者 VRU：HUD 多行展示紧凑 SSM 汇总并标注 [VRU]
        c_ssm_vru = ConflictAssessment(
            veh_track_id=1,
            ped_track_id=9,
            risk_level=RiskLevel.LEVEL_3_CRITICAL,
            risk_desc="obb",
            veh_speed_kmh=21.6,
            ped_speed_ms=3.5,
            current_distance_m=4.0,
            min_distance_m=0.0,
            ttc_seconds=1.4,
            conflict_ground_xy=(4.0, 12.0),
            mttc_seconds=1.4,
            drac_ms2=3.2,
            pet_seconds=0.4,
            collision_prob=0.82,
            conflict_model="obb_ssm",
            target_kind="bicycle",
        )
        texts_ssm = []

        def _spy_put_text_ssm(img, text, org, *args, **kwargs):
            texts_ssm.append((text, org))
            return orig_put_text(img, text, org, *args, **kwargs)

        frame_ssm = np.zeros((240, 960, 3), dtype=np.uint8)
        with mock.patch("analysis.visualization.cv2.putText", side_effect=_spy_put_text_ssm):
            self.viz.draw_hud_banner(frame_ssm, [c_ssm_vru])
            self.viz.draw_speed_badge(frame_ssm, (50, 80, 150, 180), reading, c_ssm_vru)

        # HUD 应调用 2 次 putText（主横幅 + SSM 紧凑行），徽章调用 1 次 putText
        self.assertEqual(len(texts_ssm), 3)
        self.assertIn("[VRU]", texts_ssm[0][0])
        self.assertIn("MTTC:1.4s", texts_ssm[1][0])
        self.assertIn("DRAC:3.2", texts_ssm[1][0])
        self.assertIn("PET:0.4s", texts_ssm[1][0])
        self.assertIn("P:82%", texts_ssm[1][0])
        self.assertIn("[VRU]", texts_ssm[2][0])


class TestPipelineObbAndRtspReconnect(unittest.TestCase):
    def test_pipeline_param_validation_and_configuration_snapshot(self):
        calib = _sample_calibrator()
        for bad_kw in (
            {"include_cyclist_vru": 1},
            {"include_cyclist_vru": "true"},
            {"live_reconnect_attempts": -1},
            {"live_reconnect_attempts": True},
            {"live_reconnect_attempts": 1.5},
            {"live_reconnect_delay_s": -0.5},
            {"live_reconnect_delay_s": False},
            {"live_reconnect_delay_s": float("nan")},
            {"live_reconnect_delay_s": float("inf")},
        ):
            with self.subTest(bad_kw=bad_kw), self.assertRaises(ValueError):
                ConflictPerceptionPipeline(calibrator=calib, **bad_kw)

        pipe = ConflictPerceptionPipeline(
            calibrator=calib,
            include_cyclist_vru=True,
            live_reconnect_attempts=5,
            live_reconnect_delay_s=0.25,
        )
        self.assertTrue(pipe.conflict_analyzer.include_cyclist_vru)
        snap = pipe._configuration_snapshot(25.0)
        self.assertIs(snap["include_cyclist_vru"], True)
        self.assertEqual(snap["live_reconnect_attempts"], 5)
        self.assertEqual(snap["live_reconnect_delay_s"], 0.25)

    def test_obb_ssm_pipeline_draws_vehicle_obb(self):
        calib = _sample_calibrator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vpath = root / "clip.mp4"
            _write_dummy_video(vpath, n_frames=2, fps=10.0, size=(800, 600))
            tracks_path = root / "tracks.jsonl"
            pt = calib.ground_to_pixel_checked(4.0, 10.0)
            assert pt is not None
            u, v = pt
            rows = [
                {
                    "schema_version": 1,
                    "session_id": "s-obb",
                    "source": str(vpath),
                    "frame_index": idx,
                    "timestamp_s": idx * 0.1,
                    "timestamp_source": "video_pts",
                    "width": 800,
                    "height": 600,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "category": "vehicle",
                            "confidence": 0.9,
                            "bbox_xyxy": [u - 20.0, v - 40.0, u + 20.0, float(v)],
                            "anchor_xy": [float(u), float(v)],
                            "anchor_type": "bbox_bottom_center",
                            "observation": "detected",
                        }
                    ],
                }
                for idx in range(2)
            ]
            tracks_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows) + "\n",
                encoding="utf-8",
            )

            pipe_obb = ConflictPerceptionPipeline(calibrator=calib, conflict_model="obb_ssm")
            with mock.patch.object(
                pipe_obb.visualizer, "draw_vehicle_obb", wraps=pipe_obb.visualizer.draw_vehicle_obb
            ) as spy_obb:
                pipe_obb.process_tracks(tracks_path, root / "out_obb", video_path=vpath)
            self.assertEqual(spy_obb.call_count, 2)
            self.assertAlmostEqual(spy_obb.call_args.kwargs["length_m"], 4.5)
            self.assertAlmostEqual(spy_obb.call_args.kwargs["width_m"], 1.9)

            pipe_cvm = ConflictPerceptionPipeline(calibrator=calib, conflict_model="cvm_circle")
            with mock.patch.object(
                pipe_cvm.visualizer, "draw_vehicle_obb", wraps=pipe_cvm.visualizer.draw_vehicle_obb
            ) as spy_cvm:
                pipe_cvm.process_tracks(tracks_path, root / "out_cvm", video_path=vpath)
            spy_cvm.assert_not_called()

    def test_process_video_live_stream_reconnect_success_and_exhausted(self):
        calib = _sample_calibrator()
        raw_rtsp = "rtsp://admin:secret_pw@192.168.1.50:554/live"
        good_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        bad_size_frame = np.zeros((32, 32, 3), dtype=np.uint8)

        def _make_cap(reads, *, opened=True):
            cap = mock.Mock()
            cap.isOpened.return_value = opened
            cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FRAME_WIDTH: 64.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 64.0,
                cv2.CAP_PROP_POS_MSEC: 0.0,
            }.get(prop, 0.0)
            cap.read.side_effect = list(reads)
            return cap

        # 场景 1：首次断流在第 1 次尝试打开失败，第 2 次重连成功（计数+1并重置连续计数器）；
        # 随后再次断流并在第 1 次重连成功（累计重连次数 = 2），共完整解码 4 帧后结束。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cap1 = _make_cap([(True, good_frame), (False, None)])
            cap2_unopened = _make_cap([], opened=False)
            cap3_reconnected = _make_cap([(True, good_frame), (True, good_frame), (False, None)])
            cap4_reconnected = _make_cap([(True, good_frame), (False, None)])
            cap5_fail = _make_cap([], opened=False)
            cap6_fail = _make_cap([], opened=False)

            fake_session = _make_fake_session_with_tracks(lambda: [])
            pipe = ConflictPerceptionPipeline(
                calibrator=calib,
                live_reconnect_attempts=2,
                live_reconnect_delay_s=0.15,
            )
            out1 = root / "live_ok"
            with mock.patch(
                "analysis.pipeline.cv2.VideoCapture",
                side_effect=[cap1, cap2_unopened, cap3_reconnected, cap4_reconnected, cap5_fail, cap6_fail],
            ), mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session), \
                    mock.patch("analysis.pipeline.time.sleep") as m_sleep:
                summary = pipe.process_video(raw_rtsp, out1)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["frames_processed"], 4)
            self.assertEqual(summary["live_reconnect_count"], 2)
            self.assertNotIn("secret_pw", summary["source_video"])
            self.assertIn("***@", summary["source_video"])
            self.assertEqual(m_sleep.call_count, 5)
            m_sleep.assert_called_with(0.15)
            cap1.release.assert_called_once()
            cap2_unopened.release.assert_called_once()
            cap3_reconnected.release.assert_called_once()
            cap4_reconnected.release.assert_called_once()
            self.assertTrue(validate_run(out1)["valid"])

            # 场景 2：重连后分辨率不匹配或持续无法解码，耗尽尝试次数后安全退出且重连计数为 0
            cap_init = _make_cap([(True, good_frame), (False, None)])
            cap_mismatch = _make_cap([(True, bad_size_frame)])
            cap_empty = _make_cap([(False, None)])
            out2 = root / "live_exhausted"
            with mock.patch(
                "analysis.pipeline.cv2.VideoCapture",
                side_effect=[cap_init, cap_mismatch, cap_empty],
            ), mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session), \
                    mock.patch("analysis.pipeline.time.sleep") as m_sleep2:
                summary2 = pipe.process_video(raw_rtsp, out2)

            self.assertEqual(summary2["status"], "ok")
            self.assertEqual(summary2["frames_processed"], 1)
            self.assertEqual(summary2["live_reconnect_count"], 0)
            self.assertEqual(m_sleep2.call_count, 2)
            self.assertTrue(validate_run(out2)["valid"])

    def test_process_video_local_file_no_reconnect_on_eof(self):
        calib = _sample_calibrator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vpath = root / "local.mp4"
            _write_dummy_video(vpath, n_frames=2, fps=10.0, size=(64, 64))
            fake_session = _make_fake_session_with_tracks(lambda: [])
            pipe = ConflictPerceptionPipeline(
                calibrator=calib,
                live_reconnect_attempts=3,
                live_reconnect_delay_s=0.5,
            )
            out_dir = root / "local_out"
            with mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session), \
                    mock.patch("analysis.pipeline.time.sleep") as m_sleep:
                summary = pipe.process_video(vpath, out_dir)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["frames_processed"], 2)
            self.assertEqual(summary["live_reconnect_count"], 0)
            m_sleep.assert_not_called()
            loaded = load_json_strict(out_dir / "summary.json")
            self.assertEqual(loaded["live_reconnect_count"], 0)

    def test_cli_parses_and_validates_vru_and_reconnect_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vpath = root / "clip.mp4"
            _write_dummy_video(vpath, n_frames=2, fps=10.0, size=(64, 64))

            for bad_cli in (
                ["--source", str(vpath), "--demo", "--live-reconnect-attempts", "-1"],
                ["--source", str(vpath), "--demo", "--live-reconnect-delay", "-0.1"],
                ["--source", str(vpath), "--demo", "--live-reconnect-delay", "nan"],
                ["--source", str(vpath), "--demo", "--live-reconnect-delay", "inf"],
            ):
                with self.subTest(bad_cli=bad_cli), \
                        contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    analysis_main(bad_cli)

            fake_pipe = mock.Mock()
            with mock.patch("analysis.__main__.ConflictPerceptionPipeline", return_value=fake_pipe) as m_pipe:
                rc = analysis_main([
                    "--source", str(vpath),
                    "--demo",
                    "--output", str(root / "cli_out"),
                    "--include-cyclist-vru",
                    "--live-reconnect-attempts", "5",
                    "--live-reconnect-delay", "0.5",
                ])
            self.assertEqual(rc, 0)
            kwargs = m_pipe.call_args.kwargs
            self.assertIs(kwargs["include_cyclist_vru"], True)
            self.assertEqual(kwargs["live_reconnect_attempts"], 5)
            self.assertEqual(kwargs["live_reconnect_delay_s"], 0.5)

    def test_pipeline_obb_ssm_with_vru_renders_hud_and_passes_schema(self):
        calib = _sample_calibrator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vpath = root / "vru.mp4"
            _write_dummy_video(vpath, n_frames=12, fps=10.0, size=(800, 600))
            tracks_path = root / "tracks_vru.jsonl"
            rows = []
            # 车辆沿 y 轴快速向横穿自行车逼近，触发 car vs bicycle (VRU) 冲突与 HUD SSM/VRU 渲染
            for idx in range(12):
                ts = round(idx * 0.1, 3)
                car_pt = calib.ground_to_pixel_checked(4.0, 2.0 + idx * 1.0)
                bike_pt = calib.ground_to_pixel_checked(3.6 + idx * 0.05, 14.0)
                assert car_pt is not None and bike_pt is not None
                cu, cv_p = car_pt
                bu, bv_p = bike_pt
                rows.append({
                    "schema_version": 1,
                    "session_id": "s-vru-hud",
                    "source": str(vpath),
                    "frame_index": idx,
                    "timestamp_s": ts,
                    "timestamp_source": "video_pts",
                    "width": 800,
                    "height": 600,
                    "tracks": [
                        {
                            "track_id": 1, "class_id": 2, "class_name": "car",
                            "category": "vehicle", "confidence": 0.92,
                            "bbox_xyxy": [cu - 20.0, cv_p - 40.0, cu + 20.0, float(cv_p)],
                            "anchor_xy": [float(cu), float(cv_p)],
                            "anchor_type": "bbox_bottom_center", "observation": "detected",
                        },
                        {
                            "track_id": 2, "class_id": 1, "class_name": "bicycle",
                            "category": "vehicle", "confidence": 0.90,
                            "bbox_xyxy": [bu - 12.0, bv_p - 28.0, bu + 12.0, float(bv_p)],
                            "anchor_xy": [float(bu), float(bv_p)],
                            "anchor_type": "bbox_bottom_center", "observation": "detected",
                        },
                    ],
                })
            tracks_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows) + "\n",
                encoding="utf-8",
            )

            rendered_texts = []
            orig_put_text = cv2.putText

            def _spy_text(img, text, *args, **kwargs):
                rendered_texts.append(text)
                return orig_put_text(img, text, *args, **kwargs)

            out_dir = root / "out_vru_obb"
            pipe = ConflictPerceptionPipeline(
                calibrator=calib,
                speed_model="kalman",
                conflict_model="obb_ssm",
                include_cyclist_vru=True,
                alert_confirm_frames=2,
            )
            with mock.patch("analysis.visualization.cv2.putText", side_effect=_spy_text):
                summary = pipe.process_tracks(tracks_path, out_dir, video_path=vpath)

            self.assertEqual(summary["status"], "ok")
            self.assertTrue(validate_run(out_dir)["valid"])
            self.assertTrue(any("[VRU]" in txt for txt in rendered_texts))
            self.assertTrue(any("DRAC:" in txt for txt in rendered_texts))

    def test_speed_badge_critical_contrast_edge_clamp_and_collision_avoidance(self):
        frame = np.zeros((320, 480, 3), dtype=np.uint8)
        r1 = SpeedReading(
            track_id=1, category="vehicle", x_m=3.0, y_m=6.0,
            vx_ms=0.0, vy_ms=4.0, speed_ms=4.0, speed_kmh=14.4,
            is_speeding=False, heading_deg=90.0, is_reliable=True,
        )
        r2 = SpeedReading(
            track_id=2, category="pedestrian", x_m=3.1, y_m=6.2,
            vx_ms=0.5, vy_ms=0.0, speed_ms=0.5, speed_kmh=1.8,
            is_speeding=False, heading_deg=0.0, is_reliable=True,
        )
        c_crit = ConflictAssessment(
            veh_track_id=1, ped_track_id=2, risk_level=RiskLevel.LEVEL_3_CRITICAL,
            risk_desc="crit", veh_speed_kmh=14.4, ped_speed_ms=0.5,
            current_distance_m=1.2, min_distance_m=0.2, ttc_seconds=1.1,
            conflict_ground_xy=(3.0, 6.2),
        )
        calls = []
        orig_put = cv2.putText
        orig_rect = cv2.rectangle

        def _spy_put(img, text, org, font, scale, color, *args, **kwargs):
            calls.append((text, org, scale, color))
            return orig_put(img, text, org, font, scale, color, *args, **kwargs)

        rects = []

        def _spy_rect(img, pt1, pt2, color, thickness=1, *args, **kwargs):
            if thickness == -1:
                rects.append((pt1[0], pt1[1], pt2[0], pt2[1]))
            return orig_rect(img, pt1, pt2, color, thickness, *args, **kwargs)

        t1 = SimpleNamespace(track_id=1, bbox_xyxy=(410, 120, 475, 220))
        t2 = SimpleNamespace(track_id=2, bbox_xyxy=(415, 120, 460, 210))
        viz = EnhancedConflictVisualizer(_sample_calibrator(), speed_limit_kmh=30.0)
        with (
            mock.patch("analysis.visualization.cv2.putText", side_effect=_spy_put),
            mock.patch("analysis.visualization.cv2.rectangle", side_effect=_spy_rect),
        ):
            viz.draw_frame(frame, [t1, t2], [r1, r2], [c_crit])

        # L3 badge text uses high-contrast white (255, 255, 255) and stays inside narrow frame width
        badge_calls = [c for c in calls if c[0].startswith("ID:")]
        self.assertEqual(len(badge_calls), 2)
        self.assertEqual(badge_calls[0][3], (255, 255, 255))
        badge_rects = [r for r in rects if r[0] >= 2]
        self.assertEqual(len(badge_rects), 2)
        for rx1, ry1, rx2, ry2 in badge_rects:
            self.assertGreaterEqual(rx1, 2)
            self.assertLessEqual(rx2, 478)
        # Adjacent badges must not overlap vertically
        a, b = badge_rects
        overlap = not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])
        self.assertFalse(overlap)

        # Narrow frame HUD banner scales font down so long text fits inside frame width
        hud_calls = [c for c in calls if "CRITICAL CONFLICT ALERT" in c[0]]
        self.assertEqual(len(hud_calls), 1)
        self.assertLess(hud_calls[0][2], 0.65)

    def test_pipeline_renders_non_overlapping_badges_and_reserves_hud_zone(self):
        calib = _sample_calibrator()
        pipe = ConflictPerceptionPipeline(calibrator=calib, speed_model="ema")

        raw_cap = mock.Mock()
        raw_cap.isOpened.return_value = True
        raw_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_WIDTH: 800.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 600.0,
            cv2.CAP_PROP_POS_MSEC: 0.0,
        }.get(prop, 0.0)
        raw_cap.read.return_value = (True, np.zeros((600, 800, 3), dtype=np.uint8))

        t1 = TargetRecord(
            track_id=1,
            class_id=2,
            class_name="car",
            category="vehicle",
            confidence=0.92,
            bbox_xyxy=(280.0, 220.0, 360.0, 320.0),
            anchor_xy=(320.0, 320.0),
        )
        t2 = TargetRecord(
            track_id=2,
            class_id=0,
            class_name="person",
            category="pedestrian",
            confidence=0.89,
            bbox_xyxy=(284.0, 220.0, 340.0, 315.0),
            anchor_xy=(312.0, 315.0),
        )
        t_top = TargetRecord(
            track_id=3,
            class_id=0,
            class_name="person",
            category="pedestrian",
            confidence=0.88,
            bbox_xyxy=(380.0, 18.0, 420.0, 160.0),
            anchor_xy=(400.0, 160.0),
        )
        fake_sess = mock.Mock()
        fake_sess.process.return_value = FrameRecord(
            session_id="viz-pipe-test",
            source="dummy.mp4",
            frame_index=0,
            timestamp_s=0.0,
            timestamp_source="fps_estimate",
            width=800,
            height=600,
            tracks=[t1, t2, t_top],
        )
        fake_sess.retained_ids.return_value = {1, 2, 3}

        drawn_rects = []
        orig_badge = pipe.visualizer.draw_speed_badge

        def _spy_badge(frame, box_xyxy, reading, conflict=None, event_id=None):
            active = pipe.visualizer._frame_badge_rects
            self.assertIsNotNone(active)
            before = len(active)
            orig_badge(frame, box_xyxy, reading, conflict, event_id=event_id)
            self.assertEqual(len(active), before + 1)
            drawn_rects.append(active[-1])

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "pipe_badge_out"
            with (
                mock.patch("analysis.pipeline.cv2.VideoCapture", return_value=raw_cap),
                mock.patch("analysis.pipeline.TrackingSession", return_value=fake_sess),
                mock.patch.object(pipe.visualizer, "draw_speed_badge", side_effect=_spy_badge),
            ):
                summary = pipe.process_video("dummy.mp4", out_dir, max_frames=1)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(len(drawn_rects), 3)
            for idx_a in range(len(drawn_rects)):
                ra = drawn_rects[idx_a]
                # All badges stay clear of the top 50px HUD banner
                self.assertGreaterEqual(ra[1], 50)
                for idx_b in range(idx_a + 1, len(drawn_rects)):
                    rb = drawn_rects[idx_b]
                    overlaps = not (ra[2] <= rb[0] or ra[0] >= rb[2] or ra[3] <= rb[1] or ra[1] >= rb[3])
                    self.assertFalse(overlaps)


if __name__ == "__main__":
    unittest.main()
