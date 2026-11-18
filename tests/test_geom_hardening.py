# -*- coding: utf-8 -*-
"""几何标定、测速、策略、TTC 冲突、事件状态机与可视化强化回归测试。"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from analysis.calibration import RoadCalibrator
from analysis.conflict import (
    ConflictAnalyzer,
    ConflictAssessment,
    RiskLevel,
    build_conflict_maps,
)
from analysis.events import ConflictEventTracker
from analysis.interactive_calibrator import main as calib_main, run_interactive_calibration
from analysis.policy import read_object_policy_config
from analysis.speed import SpeedEstimator, SpeedReading, TrackMotionTracker
from analysis.visualization import EnhancedConflictVisualizer


class TestCalibrationHardening(unittest.TestCase):
    def setUp(self):
        self.trapezoid = [
            (400.0, 300.0),
            (880.0, 300.0),
            (1150.0, 680.0),
            (130.0, 680.0),
        ]
        self.calibrator = RoadCalibrator(
            image_points=self.trapezoid,
            width_m=6.0,
            length_m=15.0,
            name="campus_main",
        )

    def test_invalid_name_rejected_in_init_and_from_yaml(self):
        for bad_name in (None, True, False, 123, 1.5, float("nan"), "", "   ", "\t\n"):
            with self.subTest(bad_name=bad_name):
                with self.assertRaises(ValueError):
                    RoadCalibrator(self.trapezoid, 6.0, 15.0, name=bad_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for idx, name_line in enumerate((
                "name: null",
                "name: true",
                "name: 123",
                "name: .nan",
                'name: ""',
                'name: "   "',
            )):
                p = root / f"bad_name_{idx}.yaml"
                p.write_text(
                    f"{name_line}\n"
                    "synthetic: false\n"
                    "width_m: 6.0\n"
                    "length_m: 15.0\n"
                    "image_points: [[400,300],[880,300],[1150,680],[130,680]]\n",
                    encoding="utf-8",
                )
                with self.subTest(yaml_case=name_line):
                    with self.assertRaises(ValueError):
                        RoadCalibrator.from_yaml(p)

    def test_degenerate_collinear_concave_and_bowtie_quads_rejected(self):
        bad_quads = {
            "all_same": [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0), (5.0, 5.0)],
            "four_collinear": [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
            "three_collinear": [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (50.0, 100.0)],
            "duplicate_corner": [(0.0, 0.0), (100.0, 0.0), (100.0, 0.0), (0.0, 100.0)],
            "concave_dart": [(0.0, 0.0), (100.0, 0.0), (20.0, 20.0), (0.0, 100.0)],
            "bowtie_crossing": [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)],
            "tiny_area": [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)],
        }
        for label, pts in bad_quads.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    RoadCalibrator(pts, 6.0, 15.0)

        # 逆时针有效凸四边形仍应正常通过
        ccw = RoadCalibrator(list(reversed(self.trapezoid)), 6.0, 15.0, name="ccw_road")
        self.assertIn(ccw._front_w_sign, (-1, 1))

        # 仓库默认标定与策略配置文件应可通过强化校验
        repo_root = Path(__file__).resolve().parents[1]
        default_calib = RoadCalibrator.from_yaml(repo_root / "configs" / "calibration.yaml")
        self.assertTrue(default_calib.synthetic)
        default_policies = read_object_policy_config(repo_root / "configs" / "object_policy.yaml")
        self.assertIn("car", default_policies)

        # 极端病态单应性（cond >= 1e12）应被拦截
        with self.assertRaises(ValueError):
            RoadCalibrator(
                [(0.0, 0.0), (1e9, 0.0), (1e9, 1e-3), (0.0, 1e-3)],
                width_m=1e-6,
                length_m=1e6,
            )

    def test_ground_to_pixel_checked_blocks_nan_horizon_overflow_and_sky_wraparound(self):
        # 正常路面点与前向微出左/上画幅坐标
        self.assertIsNotNone(self.calibrator.ground_to_pixel_checked(3.0, 7.5))
        left_pt = self.calibrator.ground_to_pixel_checked(-30.0, 15.0)
        self.assertIsNotNone(left_pt)
        self.assertLess(left_pt[0], 0)
        # ground_to_pixel 保持负坐标返回 (-1, -1) 的既有兼容契约
        self.assertEqual(self.calibrator.ground_to_pixel(-30.0, 15.0), (-1, -1))

        # 非有限输入与 bool 不崩溃且返回 None / (-1, -1)
        for bad_xy in (
            (float("nan"), 5.0),
            (3.0, float("inf")),
            (float("-inf"), 5.0),
            (True, 5.0),
        ):
            self.assertIsNone(self.calibrator.ground_to_pixel_checked(*bad_xy))
            self.assertEqual(self.calibrator.ground_to_pixel(*bad_xy), (-1, -1))

        # 近地平线奇异点（scale 极小导致 int32 溢出）
        self.assertIsNone(self.calibrator.ground_to_pixel_checked(3.0, 28.33333))
        self.assertEqual(self.calibrator.ground_to_pixel(3.0, 28.33333), (-1, -1))

        # 跨越相机主平面（位于相机后方）的点：未校验符号时会反演折返到天空 (640, ~168)
        for behind_y in (40.0, 100.0):
            self.assertIsNone(self.calibrator.ground_to_pixel_checked(3.0, behind_y))
            self.assertEqual(self.calibrator.ground_to_pixel(3.0, behind_y), (-1, -1))


class TestVisualizationHardening(unittest.TestCase):
    def setUp(self):
        self.trapezoid = [
            (400.0, 300.0),
            (880.0, 300.0),
            (1150.0, 680.0),
            (130.0, 680.0),
        ]
        self.calibrator = RoadCalibrator(
            image_points=self.trapezoid,
            width_m=6.0,
            length_m=15.0,
            name="viz_road",
        )
        self.viz = EnhancedConflictVisualizer(self.calibrator, speed_limit_kmh=30.0)

    def test_parking_zone_partially_off_left_edge_still_draws_and_horizon_skips(self):
        # 左侧微出画幅 (u < 0, 但仍位于相机前方且未溢出) 的停车区不应整块跳过
        self.calibrator.parking_zones = [[(-5.0, 10.0), (1.0, 10.0), (1.0, 14.0), (-5.0, 14.0)]]
        left_pt = self.calibrator.ground_to_pixel_checked(-5.0, 10.0)
        self.assertIsNotNone(left_pt)
        self.assertLess(left_pt[0], 0)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.viz.draw_calibrated_ground_zone(frame)
        # 停车区轮廓色 (180, 180, 80) 必须在画面左侧区域出现
        self.assertTrue(np.any(frame[:, :400, 0] == 180))

        # 含近地平线溢出顶点（ground_to_pixel_checked 返回 None）时安全跳过不抛 OverflowError
        self.calibrator.parking_zones = [[(1.0, 10.0), (3.0, 28.33333), (4.0, 10.0)]]
        frame2 = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.viz.draw_calibrated_ground_zone(frame2)

    def test_trajectory_and_conflict_link_skip_horizon_and_sky_wraparound(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        veh = SpeedReading(
            track_id=1, category="vehicle",
            x_m=3.0, y_m=28.33333, vx_ms=0.0, vy_ms=30.0,
            speed_ms=30.0, speed_kmh=108.0, is_speeding=True,
            heading_deg=90.0, is_reliable=True,
        )
        ped = SpeedReading(
            track_id=2, category="pedestrian",
            x_m=3.0, y_m=100.0, vx_ms=0.0, vy_ms=0.0,
            speed_ms=0.0, speed_kmh=0.0, is_speeding=False,
            heading_deg=0.0, is_reliable=True,
        )
        # 不会触发 OverflowError / cv2.error / 天空折返绘制
        self.viz.draw_future_trajectory(frame, veh, horizon_s=3.0, steps=6)
        self.assertEqual(int(frame[:300, :, :].sum()), 0)

        conflict = ConflictAssessment(
            veh_track_id=1, ped_track_id=2,
            risk_level=RiskLevel.LEVEL_3_CRITICAL, risk_desc="x",
            veh_speed_kmh=40.0, ped_speed_ms=0.0,
            current_distance_m=5.0, min_distance_m=0.5,
            ttc_seconds=0.8, conflict_ground_xy=(3.0, 100.0),
        )
        self.viz.draw_conflict_link(frame, conflict, {1: veh, 2: ped})
        self.assertEqual(int(frame.sum()), 0)

        # veh_pt 为 None 但 conflict_ground_xy 在画内时：跳过连线，仍绘制靶标圈
        conflict_in = ConflictAssessment(
            veh_track_id=1, ped_track_id=2,
            risk_level=RiskLevel.LEVEL_3_CRITICAL, risk_desc="x",
            veh_speed_kmh=40.0, ped_speed_ms=0.0,
            current_distance_m=5.0, min_distance_m=0.5,
            ttc_seconds=0.8, conflict_ground_xy=(3.0, 7.5),
        )
        frame_target_only = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.viz.draw_conflict_link(frame_target_only, conflict_in, {1: veh, 2: ped})
        cu, cv_pt = self.calibrator.ground_to_pixel_checked(3.0, 7.5)
        self.assertEqual(tuple(int(c) for c in frame_target_only[cv_pt, cu]), self.viz.COLOR_CRITICAL)

        # 回退至仅实现 ground_to_pixel 的旧接口时也不得将 (-1, -1) 或超大值传给 OpenCV
        legacy_calib = SimpleNamespace(
            image_points=self.trapezoid,
            parking_zones=[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]],
            ground_to_pixel=lambda x, y: (-1, -1) if y > 20 else (2_000_000, 2_000_000),
        )
        legacy_viz = EnhancedConflictVisualizer(legacy_calib)
        legacy_viz.draw_calibrated_ground_zone(np.zeros((720, 1280, 3), dtype=np.uint8))
        legacy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        legacy_viz.draw_future_trajectory(legacy_frame, veh)
        legacy_viz.draw_conflict_link(legacy_frame, conflict, {1: veh, 2: ped})
        self.assertEqual(int(legacy_frame.sum()), 0)

    def test_speed_badge_respects_conflict_level1_speeding(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        reading = SpeedReading(
            track_id=1, category="vehicle",
            x_m=3.0, y_m=5.0, vx_ms=0.0, vy_ms=7.0,
            speed_ms=7.0, speed_kmh=25.2, is_speeding=False,
            heading_deg=90.0, is_reliable=True,
        )
        conflict = ConflictAssessment(
            veh_track_id=1, ped_track_id=None,
            risk_level=RiskLevel.LEVEL_1_SPEEDING, risk_desc="speeding",
            veh_speed_kmh=25.2, ped_speed_ms=0.0,
            current_distance_m=0.0, min_distance_m=0.0,
            ttc_seconds=None, conflict_ground_xy=None,
        )
        self.viz.draw_speed_badge(frame, (20, 40, 120, 140), reading, conflict)
        # 框边界像素应使用 COLOR_SPEEDING=(0, 215, 255) 而非 COLOR_SAFE=(50, 205, 50)
        self.assertEqual(tuple(int(c) for c in frame[140, 60]), self.viz.COLOR_SPEEDING)

    def test_hud_banner_prefers_confirmed_and_smallest_ttc(self):
        def mk(veh_id, ped_id, level, ttc):
            return ConflictAssessment(
                veh_track_id=veh_id, ped_track_id=ped_id,
                risk_level=level, risk_desc="x",
                veh_speed_kmh=30.0, ped_speed_ms=1.0,
                current_distance_m=4.0, min_distance_m=0.5,
                ttc_seconds=ttc, conflict_ground_xy=(3.0, 5.0),
            )

        # 未确认且 TTC 更小的 L3 不得掩盖同级已确认 L3
        c_unconf = mk(1, 10, RiskLevel.LEVEL_3_CRITICAL, 0.7)
        c_conf = mk(2, 20, RiskLevel.LEVEL_3_CRITICAL, 1.4)
        frame1 = np.zeros((100, 800, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 800, 3), dtype=np.uint8)
        self.viz.draw_hud_banner(frame1, [c_unconf, c_conf], event_ids={(2, 20): 7})
        self.viz.draw_hud_banner(frame2, [c_conf], event_ids={(2, 20): 7})
        self.assertTrue(np.array_equal(frame1, frame2))

        # 同为未确认 L3 时选择最小 TTC
        c_far = mk(3, 30, RiskLevel.LEVEL_3_CRITICAL, 1.7)
        c_near = mk(4, 40, RiskLevel.LEVEL_3_CRITICAL, 0.5)
        f_pair = np.zeros((100, 800, 3), dtype=np.uint8)
        f_near = np.zeros((100, 800, 3), dtype=np.uint8)
        self.viz.draw_hud_banner(f_pair, [c_far, c_near])
        self.viz.draw_hud_banner(f_near, [c_near])
        self.assertTrue(np.array_equal(f_pair, f_near))


class TestSpeedAndPolicyHardening(unittest.TestCase):
    def test_category_hysteresis_blocks_single_frame_person_object_kind_leak(self):
        est = SpeedEstimator(category_confirm_frames=3)
        analyzer = ConflictAnalyzer()
        ped = SpeedReading(
            track_id=99, category="pedestrian", object_kind="pedestrian",
            x_m=5.5, y_m=6.0, vx_ms=0.0, vy_ms=0.0,
            speed_ms=0.0, speed_kmh=0.0, is_speeding=False,
            heading_deg=0.0, is_reliable=True,
        )

        # 4 帧 car -> 第 5 帧误检为 ("pedestrian", "person")
        for i in range(4):
            r = est.update(1, "vehicle", i * 0.1, 3.0, i * 0.6, object_kind="car")
            self.assertEqual(r.object_kind, "car")

        flicker = est.update(1, "pedestrian", 0.4, 3.0, 2.4, object_kind="person")
        self.assertEqual(flicker.category, "vehicle")
        self.assertEqual(flicker.object_kind, "car")

        # 车辆安全半径必须仍按 car (2.2m + 0.8m = 3.0m) 评估，而不是被缩成 0.8m + 0.8m = 1.6m
        veh_eval = SpeedReading(
            track_id=1, category=flicker.category, object_kind=flicker.object_kind,
            x_m=3.0, y_m=0.0, vx_ms=0.0, vy_ms=6.0,
            speed_ms=6.0, speed_kmh=21.6, is_speeding=False,
            heading_deg=90.0, is_reliable=True,
        )
        res = analyzer.evaluate_pair(veh_eval, ped)
        self.assertIsNotNone(res)
        self.assertEqual(res.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertEqual(res.vehicle_kind, "car")

        # 后续恢复 vehicle 且不传 object_kind 时仍保持 car
        rec = est.update(1, "vehicle", 0.5, 3.0, 3.0)
        self.assertEqual(rec.object_kind, "car")

        # 即便观测 category=="vehicle" 但错传 object_kind="person"，也不得缩成 pedestrian
        weird = est.update(1, "vehicle", 0.55, 3.0, 3.3, object_kind="person")
        self.assertEqual(weird.object_kind, "car")

        # 若连续 3 帧切为 pedestrian，单帧 vehicle/car 闪烁亦不得泄露给 pedestrian
        for i in range(3):
            r_p = est.update(1, "pedestrian", (6 + i) * 0.1, 3.0, 3.6 + i * 0.1, object_kind="person")
        self.assertEqual((r_p.category, r_p.object_kind), ("pedestrian", "pedestrian"))
        ped_flicker = est.update(1, "vehicle", 0.85, 3.0, 3.85, object_kind="car")
        self.assertEqual((ped_flicker.category, ped_flicker.object_kind), ("pedestrian", "pedestrian"))
        for i in range(3):
            r_v = est.update(1, "vehicle", (9 + i) * 0.1, 3.0, 4.0 + i * 0.6, object_kind=None)
        self.assertEqual((r_v.category, r_v.object_kind), ("vehicle", "motor_vehicle"))

    def test_speed_numeric_params_reject_bool_and_first_frame_nan_returns_zero(self):
        for kwargs in (
            {"ema_alpha": True},
            {"speed_limit_kmh": True},
            {"stationary_speed_ms": True},
            {"reliable_min_span_s": False},
            {"window_size": True},
            {"stationary_min_frames": True},
            {"category_confirm_frames": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SpeedEstimator(**kwargs)

        for kwargs in (
            {"ema_alpha": True},
            {"speed_limit_kmh": False},
            {"stationary_speed_ms": True},
            {"reliable_min_span_s": True},
            {"window_size": True},
            {"stationary_min_frames": True},
        ):
            with self.subTest(tracker_kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TrackMotionTracker(1, "vehicle", **kwargs)

        tracker = TrackMotionTracker(1, "vehicle")
        r = tracker.update(0.0, float("nan"), float("inf"))
        self.assertEqual((r.x_m, r.y_m), (0.0, 0.0))
        self.assertTrue(math.isfinite(r.x_m) and math.isfinite(r.y_m))
        self.assertFalse(r.is_reliable)

    def test_policy_rejects_bool_float_schema_version_and_blank_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for idx, text in enumerate((
                "schema_version: true\npolicies:\n  person: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                "  pedestrian: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                "  vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n"
                "  motor_vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n",
                "schema_version: 1.0\npolicies:\n  person: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                "  pedestrian: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                "  vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n"
                "  motor_vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n",
                'schema_version: 1\npolicies:\n  "   ": {kind: car, safe_radius_veh_m: 2.2, label: c}\n'
                "  person: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                "  pedestrian: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                "  vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n"
                "  motor_vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n",
            )):
                p = root / f"bad_policy_{idx}.yaml"
                p.write_text(text, encoding="utf-8")
                with self.subTest(case=idx):
                    with self.assertRaises(ValueError):
                        read_object_policy_config(p)


class TestConflictEventsAndCalibratorCliHardening(unittest.TestCase):
    def test_conflict_rejects_bool_enforces_caution_ttc_and_handles_zero_ttc(self):
        for kwargs in (
            {"speed_limit_kmh": True},
            {"safe_radius_veh_m": True},
            {"safe_radius_ped_m": False},
            {"ttc_critical_s": True, "ttc_caution_s": 3.5, "horizon_s": 3.5},
            {"ttc_critical_s": 1.0, "ttc_caution_s": True, "horizon_s": 3.5},
            {"ttc_critical_s": 0.5, "ttc_caution_s": 0.8, "horizon_s": True},
            {"parking_zone_max_speed_kmh": False},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ConflictAnalyzer(**kwargs)

        # ttc_caution_s=3.5 < horizon_s=5.0 且 min_dist <= critical_dist_m 时，
        # t_cpa=4.0 > ttc_caution_s 不得再误判为 LEVEL_2_CAUTION
        analyzer = ConflictAnalyzer(ttc_critical_s=1.8, ttc_caution_s=3.5, horizon_s=5.0)
        veh = SpeedReading(
            track_id=1, category="vehicle", object_kind="car",
            x_m=3.0, y_m=0.0, vx_ms=0.0, vy_ms=5.0,
            speed_ms=5.0, speed_kmh=18.0, is_speeding=False,
            heading_deg=90.0, is_reliable=True,
        )
        ped = SpeedReading(
            track_id=2, category="pedestrian", object_kind="pedestrian",
            x_m=3.0, y_m=20.0, vx_ms=0.0, vy_ms=0.0,
            speed_ms=0.0, speed_kmh=0.0, is_speeding=False,
            heading_deg=0.0, is_reliable=True,
        )
        self.assertIsNone(analyzer.evaluate_pair(veh, ped))

        # ttc_seconds == 0.0 在排序键中必须比 0.5s 更紧急，而非回退到 999.0
        c_zero = ConflictAssessment(
            veh_track_id=1, ped_track_id=9, risk_level=RiskLevel.LEVEL_3_CRITICAL,
            risk_desc="now", veh_speed_kmh=20.0, ped_speed_ms=1.0,
            current_distance_m=0.1, min_distance_m=0.0, ttc_seconds=0.0,
            conflict_ground_xy=(3.0, 5.0),
        )
        c_later = ConflictAssessment(
            veh_track_id=2, ped_track_id=9, risk_level=RiskLevel.LEVEL_3_CRITICAL,
            risk_desc="later", veh_speed_kmh=20.0, ped_speed_ms=1.0,
            current_distance_m=2.0, min_distance_m=0.0, ttc_seconds=0.5,
            conflict_ground_xy=(3.0, 5.0),
        )
        _, ped_map = build_conflict_maps([c_later, c_zero])
        self.assertIs(ped_map[9], c_zero)

    def test_events_update_type_checks_and_flush_clears_pending_states(self):
        tr = ConflictEventTracker(confirm_frames=3, cooldown_frames=2)
        for bad_frame, bad_ts in (
            (True, 0.0),
            (0.0, 0.0),
            (0, True),
            (0, -0.1),
        ):
            with self.subTest(bad_frame=bad_frame, bad_ts=bad_ts):
                with self.assertRaises(ValueError):
                    tr.update([], bad_frame, bad_ts)

        c = ConflictAssessment(
            veh_track_id=1, ped_track_id=2, risk_level=RiskLevel.LEVEL_3_CRITICAL,
            risk_desc="x", veh_speed_kmh=20.0, ped_speed_ms=1.0,
            current_distance_m=3.0, min_distance_m=0.5, ttc_seconds=1.0,
            conflict_ground_xy=(3.0, 5.0),
        )
        # 仅挂起 2 帧（未确认）后 flush() 必须彻底清空 _states
        tr.update([c], 0, 0.0)
        tr.update([c], 1, 0.1)
        tr.flush()
        self.assertEqual(tr.pop_finalized(), [])
        self.assertEqual(len(tr._states), 0)

        # 复用实例：后续第 1 帧告警不得沿用 flush 前残留的 streak=2 而立即确认
        ann = tr.update([c], 2, 0.2)
        self.assertEqual(ann, {})

    def test_interactive_calibrator_does_not_overwrite_output_when_check_points_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_path = root / "road.png"
            ok, encoded = cv2.imencode(".png", np.zeros((720, 1280, 3), dtype=np.uint8))
            self.assertTrue(ok)
            encoded.tofile(str(img_path))

            out_yaml = root / "calib.yaml"
            out_yaml.write_text("sentinel: keep_me\n", encoding="utf-8")

            # save_output=False 时不写文件
            cal = run_interactive_calibration(
                img_path, 6.0, 15.0, output_path=out_yaml,
                predefined_points=[(400.0, 300.0), (880.0, 300.0), (1150.0, 680.0), (130.0, 680.0)],
                save_output=False,
            )
            self.assertEqual(cal.width_m, 6.0)
            self.assertEqual(out_yaml.read_text(encoding="utf-8"), "sentinel: keep_me\n")

            # 检查点与标定角点重合报错时不得覆写既有输出文件
            val_json = root / "val.json"
            with self.assertRaises(ValueError):
                calib_main([
                    "--source", str(img_path),
                    "--width", "6.0",
                    "--length", "15.0",
                    "--points", "400,300,880,300,1150,680,130,680",
                    "--check-image-points", "400,300",
                    "--check-ground-points", "0,0",
                    "--validation-output", str(val_json),
                    "--output", str(out_yaml),
                ])
            self.assertEqual(out_yaml.read_text(encoding="utf-8"), "sentinel: keep_me\n")
            self.assertFalse(val_json.exists())

            # 检查点合法时正常验证并落盘
            self.assertEqual(
                calib_main([
                    "--source", str(img_path),
                    "--width", "6.0",
                    "--length", "15.0",
                    "--points", "400,300,880,300,1150,680,130,680",
                    "--check-image-points", "640,490",
                    "--check-ground-points", "3.0,7.5",
                    "--validation-output", str(val_json),
                    "--output", str(out_yaml),
                ]),
                0,
            )
            loaded = RoadCalibrator.from_yaml(out_yaml)
            self.assertFalse(loaded.synthetic)
            report = json.loads(val_json.read_text(encoding="utf-8"))
            self.assertEqual(report["count"], 1)


if __name__ == "__main__":
    unittest.main()
