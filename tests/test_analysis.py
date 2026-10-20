# -*- coding: utf-8 -*-
"""分析模块 (标定、测速、冲突预测与 TTC 评估) 单元测试。"""

import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.calibration import RoadCalibrator
from analysis.conflict import ConflictAnalyzer, ConflictAssessment, RiskLevel
from analysis.events import ConflictEventTracker
from analysis.speed import SpeedEstimator, SpeedReading, TrackMotionTracker
from analysis.visualization import EnhancedConflictVisualizer


class TestRoadCalibrator(unittest.TestCase):
    def setUp(self):
        # 模拟一个 6m 宽、20m 长的校园道路在相机视场中的透视梯形
        self.image_points = [
            (200.0, 100.0),  # 左上
            (600.0, 100.0),  # 右上
            (750.0, 500.0),  # 右下
            (50.0, 500.0),   # 左下
        ]
        self.width_m = 6.0
        self.length_m = 20.0
        self.calibrator = RoadCalibrator(
            image_points=self.image_points,
            width_m=self.width_m,
            length_m=self.length_m,
            name="test_road",
        )

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            RoadCalibrator(self.image_points[:3], 6.0, 20.0)
        with self.assertRaises(ValueError):
            RoadCalibrator(self.image_points, -1.0, 20.0)
        with self.assertRaises(ValueError):
            RoadCalibrator(self.image_points, 6.0, 0.0)

    def test_corner_projection_accuracy(self):
        # 4 个角点经变换后应严格对应 (0,0), (6,0), (6,20), (0,20)
        expected_ground = [
            (0.0, 0.0),
            (6.0, 0.0),
            (6.0, 20.0),
            (0.0, 20.0),
        ]
        for img_pt, exp_gd in zip(self.image_points, expected_ground):
            gx, gy = self.calibrator.pixel_to_ground(img_pt[0], img_pt[1])
            self.assertAlmostEqual(gx, exp_gd[0], delta=1e-3)
            self.assertAlmostEqual(gy, exp_gd[1], delta=1e-3)

    def test_round_trip_invertibility(self):
        # 任意图像点转地面后再转回图像，误差极小
        test_u, test_v = 400.0, 300.0
        gx, gy = self.calibrator.pixel_to_ground(test_u, test_v)
        u_rec, v_rec = self.calibrator.ground_to_pixel(gx, gy)
        self.assertAlmostEqual(test_u, u_rec, delta=1e-3)
        self.assertAlmostEqual(test_v, v_rec, delta=1e-3)

    def test_yaml_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir).resolve() / "test_calib.yaml"
            self.calibrator.to_yaml(yaml_path)
            self.assertTrue(yaml_path.exists())

            loaded = RoadCalibrator.from_yaml(yaml_path)
            self.assertEqual(loaded.name, "test_road")
            self.assertEqual(loaded.width_m, 6.0)
            self.assertEqual(loaded.length_m, 20.0)
            self.assertEqual(len(loaded.image_points), 4)

            # 验证复原的矩阵映射一致性
            gx1, gy1 = self.calibrator.pixel_to_ground(300, 200)
            gx2, gy2 = loaded.pixel_to_ground(300, 200)
            self.assertAlmostEqual(gx1, gx2, places=4)
            self.assertAlmostEqual(gy1, gy2, places=4)

    def test_synthetic_field_must_be_real_bool(self):
        # 字符串 "false"/数值 0 不得经 bool() 被静默吞并：非真布尔直接拒绝
        for bad_value in ('"false"', "0", '"yes"'):
            with self.subTest(bad_value=bad_value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    yaml_path = Path(tmpdir) / "calib.yaml"
                    yaml_path.write_text(
                        "name: t\n"
                        f"synthetic: {bad_value}\n"
                        "width_m: 6.0\nlength_m: 15.0\n"
                        "image_points: [[0,0],[1,0],[1,1],[0,1]]\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        RoadCalibrator.from_yaml(yaml_path)

    def test_missing_synthetic_field_warns_and_loads_as_real(self):
        # 旧版文件无溯源字段：无法区分实测/占位，按实测加载但警告重新标定
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "calib.yaml"
            yaml_path.write_text(
                "name: old\n"
                "width_m: 6.0\nlength_m: 15.0\n"
                "image_points: [[0,0],[200,0],[200,100],[0,100]]\n",
                encoding="utf-8",
            )
            with self.assertWarns(UserWarning):
                loaded = RoadCalibrator.from_yaml(yaml_path)
            self.assertFalse(loaded.synthetic)


class TestSpeedEstimator(unittest.TestCase):
    def test_warmup_requires_sufficient_points(self):
        tracker = TrackMotionTracker(track_id=1, category="vehicle")
        r1 = tracker.update(timestamp_s=0.0, x_m=3.0, y_m=0.0)
        self.assertFalse(r1.is_reliable)
        self.assertEqual(r1.speed_kmh, 0.0)

        r2 = tracker.update(timestamp_s=0.1, x_m=3.0, y_m=0.5)
        self.assertFalse(r2.is_reliable)

    def test_constant_speed_measurement(self):
        estimator = SpeedEstimator(speed_limit_kmh=30.0, window_size=5)
        # 车辆以 5 m/s (18 km/h) 沿 Y 轴前进，dt = 0.1s, dy = 0.5m
        readings = []
        for i in range(15):
            t = i * 0.1
            y = i * 0.5
            r = estimator.update(track_id=1, category="vehicle", timestamp_s=t, x_m=3.0, y_m=y)
            readings.append(r)

        final = readings[-1]
        self.assertTrue(final.is_reliable)
        # EMA 滤波后车速应稳定收敛在 ~18 km/h (5 m/s)
        self.assertAlmostEqual(final.speed_ms, 5.0, delta=0.5)
        self.assertAlmostEqual(final.speed_kmh, 18.0, delta=1.8)
        self.assertFalse(final.is_speeding)

    def test_speeding_detection_over_30kmh(self):
        estimator = SpeedEstimator(speed_limit_kmh=30.0)
        # 车辆以 10 m/s (36 km/h) 高速行驶，超越 30 km/h 限速
        final = None
        for i in range(12):
            t = i * 0.1
            y = i * 1.0  # 1.0m 每 0.1s = 10 m/s
            final = estimator.update(track_id=2, category="vehicle", timestamp_s=t, x_m=3.0, y_m=y)

        self.assertIsNotNone(final)
        self.assertTrue(final.is_reliable)
        self.assertTrue(final.is_speeding)
        self.assertGreater(final.speed_kmh, 30.0)

    def test_pedestrian_jitter_suppression(self):
        estimator = SpeedEstimator()
        # 行人原地微动 (0.1 m/s = 0.36 km/h < 0.8 km/h 截断)
        final = None
        for i in range(10):
            t = i * 0.1
            y = (i % 2) * 0.01
            final = estimator.update(track_id=3, category="pedestrian", timestamp_s=t, x_m=1.0, y_m=y)

        self.assertIsNotNone(final)
        self.assertEqual(final.speed_kmh, 0.0)

    def test_prune_stale_tracks(self):
        estimator = SpeedEstimator()
        estimator.update(1, "vehicle", 0.0, 0.0, 0.0)
        estimator.update(2, "pedestrian", 0.0, 1.0, 1.0)
        self.assertEqual(len(estimator.tracks), 2)

        estimator.prune_stale_tracks({2})
        self.assertEqual(set(estimator.tracks.keys()), {2})

    def test_nan_coordinate_does_not_poison_filter(self):
        # 地平线退化投影返回 NaN；该观测不得进入历史污染差分与 EMA
        estimator = SpeedEstimator()
        for i in range(4):
            estimator.update(1, "vehicle", i * 0.1, 3.0, i * 0.5)

        r_nan = estimator.update(1, "vehicle", 0.4, float("nan"), float("nan"))
        self.assertFalse(r_nan.is_reliable)
        # 无效观测回填最后有效位置而非暴露 NaN
        self.assertEqual((r_nan.x_m, r_nan.y_m), (3.0, 1.5))

        final = None
        for i in range(5, 14):
            final = estimator.update(1, "vehicle", i * 0.1, 3.0, i * 0.5)

        self.assertTrue(math.isfinite(final.speed_kmh))
        self.assertAlmostEqual(final.speed_ms, 5.0, delta=0.5)

    def test_nan_timestamp_rejected_without_corruption(self):
        estimator = SpeedEstimator()
        estimator.update(1, "vehicle", 0.0, 3.0, 0.0)
        estimator.update(1, "vehicle", 0.1, 3.0, 0.5)
        r = estimator.update(1, "vehicle", float("nan"), 3.0, 1.0)
        self.assertFalse(r.is_reliable)
        self.assertEqual(len(estimator.tracks[1].history), 2)

    def test_stationary_streak_requires_sustained_low_speed(self):
        # 真实滤波器路径：静止目标需连续 parked-frames 帧后才标记 is_stationary
        estimator = SpeedEstimator(stationary_min_frames=5)
        readings = [
            estimator.update(1, "vehicle", i * 0.1, 3.0, 5.0) for i in range(10)
        ]
        self.assertFalse(readings[5].is_stationary)   # 尚未连续 5 帧
        self.assertTrue(readings[-1].is_stationary)

    def test_slow_crawling_vehicle_never_marked_stationary(self):
        # 0.4 m/s 蠕行车辆：速度超过静止阈值，永不被豁免
        estimator = SpeedEstimator(stationary_min_frames=5)
        final = None
        for i in range(12):
            final = estimator.update(1, "vehicle", i * 0.1, 3.0, i * 0.04)
        self.assertTrue(final.is_reliable)
        self.assertFalse(final.is_stationary)
        self.assertGreater(final.speed_ms, 0.1)

    def test_crawl_below_micro_clamp_not_stationary(self):
        # 0.2 m/s (0.72 km/h) 与 0.27 m/s (0.97 km/h)：标量被微动截断为 0，
        # 但未截断平滑幅值仍超静止阈值，绝不能标记为"泊车"
        for ms_per_frame, label in ((0.02, "0.2 m/s"), (0.027, "0.27 m/s")):
            with self.subTest(label=label):
                est = SpeedEstimator(stationary_min_frames=5)
                final = None
                for i in range(15):
                    final = est.update(1, "vehicle", i * 0.1, 3.0, i * ms_per_frame)
                self.assertTrue(final.is_reliable)
                self.assertFalse(final.is_stationary)

    def test_sub_threshold_drift_becomes_stationary(self):
        # 0.05 m/s 平滑幅值低于 0.1 m/s 阈值，连续帧后判定静止
        est = SpeedEstimator(stationary_min_frames=5)
        final = None
        for i in range(12):
            final = est.update(1, "vehicle", i * 0.1, 3.0, i * 0.005)
        self.assertTrue(final.is_stationary)

    def test_nan_estimator_params_rejected(self):
        with self.assertRaises(ValueError):
            SpeedEstimator(speed_limit_kmh=float("nan"))
        with self.assertRaises(ValueError):
            SpeedEstimator(ema_alpha=float("nan"))
        with self.assertRaises(ValueError):
            SpeedEstimator(stationary_speed_ms=float("inf"))
        with self.assertRaises(ValueError):
            TrackMotionTracker(1, "vehicle", speed_limit_kmh=float("nan"))

    def test_category_hysteresis_suppresses_flicker(self):
        est = SpeedEstimator(category_confirm_frames=3)
        est.update(1, "pedestrian", 0.0, 0.0, 0.0)
        est.update(1, "vehicle", 0.1, 0.0, 0.1)
        est.update(1, "pedestrian", 0.2, 0.0, 0.2)   # 抵消 pending
        est.update(1, "vehicle", 0.3, 0.0, 0.3)      # pending 1
        self.assertEqual(est.tracks[1].category, "pedestrian")  # 未满 3 帧不切换

        est.update(1, "vehicle", 0.4, 0.0, 0.4)      # pending 2
        r = est.update(1, "vehicle", 0.5, 0.0, 0.5)  # pending 3 -> 确认
        self.assertEqual(r.category, "vehicle")

    def test_invalid_estimator_params_rejected(self):
        with self.assertRaises(ValueError):
            SpeedEstimator(ema_alpha=0.0)
        with self.assertRaises(ValueError):
            SpeedEstimator(ema_alpha=1.5)
        with self.assertRaises(ValueError):
            SpeedEstimator(window_size=2)
        with self.assertRaises(ValueError):
            SpeedEstimator(speed_limit_kmh=-1.0)
        with self.assertRaises(ValueError):
            SpeedEstimator(category_confirm_frames=0)

    def test_reliable_requires_min_time_span(self):
        # 时间门：40fps 下 4 样本仅覆盖 0.12s，不足 0.2s 跨度仍不可信；
        # 6 样本覆盖 0.2s 才收敛，避免高帧率下过早置位
        est = SpeedEstimator(reliable_min_span_s=0.2)
        readings = [
            est.update(1, "vehicle", i * 0.04, 3.0, i * 0.2) for i in range(6)
        ]
        self.assertFalse(readings[3].is_reliable)   # span=0.12 < 0.2
        self.assertTrue(readings[5].is_reliable)    # span=0.20 >= 0.2

    def test_zero_min_span_disables_time_gate(self):
        est = SpeedEstimator(reliable_min_span_s=0.0)
        final = None
        for i in range(4):
            final = est.update(1, "vehicle", i * 0.04, 3.0, i * 0.2)
        self.assertTrue(final.is_reliable)

    def test_invalid_reliable_span_rejected(self):
        with self.assertRaises(ValueError):
            SpeedEstimator(reliable_min_span_s=-0.1)
        with self.assertRaises(ValueError):
            SpeedEstimator(reliable_min_span_s=float("nan"))
        with self.assertRaises(ValueError):
            TrackMotionTracker(1, "vehicle", reliable_min_span_s=float("inf"))

    def test_vehicle_restarting_after_parking_recovers(self):
        # 泊车确认后重新起步：静止标记必须即时解除，
        # 恢复参与碰撞研判（停车 -> 起步 -> 与行人冲突 的常见序列）
        est = SpeedEstimator(stationary_min_frames=5)
        parked = [
            est.update(1, "vehicle", i * 0.1, 3.0, 5.0) for i in range(10)
        ]
        self.assertTrue(parked[-1].is_stationary)
        self.assertTrue(parked[-1].is_reliable)

        moving = [
            est.update(1, "vehicle", (10 + i) * 0.1, 3.0, 5.0 + (i + 1) * 0.5)
            for i in range(6)
        ]
        # 起步首帧平滑速度即越过静止阈值，标记立刻解除
        self.assertFalse(moving[0].is_stationary)
        self.assertGreater(moving[-1].speed_ms, 1.0)

    def test_reclassified_vehicle_can_be_flagged_speeding(self):
        # ByteTrack ID 首次观测被误检为 person，纠正为 vehicle 后应能判超速
        estimator = SpeedEstimator()
        for i in range(6):
            estimator.update(1, "pedestrian", i * 0.1, 3.0, i * 0.5)

        final = None
        for i in range(6, 16):
            final = estimator.update(1, "vehicle", i * 0.1, 3.0, i * 1.0)

        self.assertTrue(final.is_reliable)
        self.assertTrue(final.is_speeding)


class TestConflictAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = ConflictAnalyzer(
            speed_limit_kmh=30.0,
            safe_radius_veh_m=2.2,
            safe_radius_ped_m=0.8,  # critical_dist = 3.0m
            ttc_critical_s=1.8,
            ttc_caution_s=3.5,
            horizon_s=3.5,
        )

    def test_converging_trajectory_critical_collision(self):
        # 场景：机动车向北行驶 (vx=0, vy=10 m/s 即 36 km/h) 当前位置 (3, 0)
        # 行人从东向西横穿 (vx=-2 m/s, vy=0) 当前位置 (5, 10)
        # 在 t = 1.0s 时：
        # 机动车到达 (3, 10)
        # 行人到达 (5 - 2*1, 10) = (3, 10)
        # 此时恰好相撞！D_min = 0, TTC = 1.0s <= 1.8s -> LEVEL_3_CRITICAL
        veh = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=0.0,
            vx_ms=0.0,
            vy_ms=10.0,
            speed_ms=10.0,
            speed_kmh=36.0,
            is_speeding=True,
            heading_deg=90.0,
            is_reliable=True,
        )
        ped = SpeedReading(
            track_id=2,
            category="pedestrian",
            x_m=5.0,
            y_m=10.0,
            vx_ms=-2.0,
            vy_ms=0.0,
            speed_ms=2.0,
            speed_kmh=7.2,
            is_speeding=False,
            heading_deg=180.0,
            is_reliable=True,
        )

        assessment = self.analyzer.evaluate_pair(veh, ped)
        self.assertIsNotNone(assessment)
        self.assertEqual(assessment.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertAlmostEqual(assessment.ttc_seconds, 1.0, delta=0.05)
        self.assertAlmostEqual(assessment.min_distance_m, 0.0, delta=0.1)
        self.assertIsNotNone(assessment.conflict_ground_xy)
        self.assertAlmostEqual(assessment.conflict_ground_xy[0], 3.0, delta=0.1)
        self.assertAlmostEqual(assessment.conflict_ground_xy[1], 10.0, delta=0.1)

    def test_caution_warning_when_ttc_moderate(self):
        # 场景：机动车 (3, 0), vy=6 m/s (21.6 km/h)
        # 行人 (5, 15), vx=-1 m/s, vy=0
        # t = 2.0s: 车到达 (3, 12), 人到达 (3, 15), 间距 3m
        # t_cpa ~ 2.4s (介于 1.8s ~ 3.5s) -> LEVEL_2_CAUTION
        veh = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=0.0,
            vx_ms=0.0,
            vy_ms=6.0,
            speed_ms=6.0,
            speed_kmh=21.6,
            is_speeding=False,
            heading_deg=90.0,
            is_reliable=True,
        )
        ped = SpeedReading(
            track_id=2,
            category="pedestrian",
            x_m=5.0,
            y_m=14.0,
            vx_ms=-1.0,
            vy_ms=0.0,
            speed_ms=1.0,
            speed_kmh=3.6,
            is_speeding=False,
            heading_deg=180.0,
            is_reliable=True,
        )

        assessment = self.analyzer.evaluate_pair(veh, ped)
        self.assertIsNotNone(assessment)
        self.assertEqual(assessment.risk_level, RiskLevel.LEVEL_2_CAUTION)
        self.assertGreater(assessment.ttc_seconds, 1.8)
        self.assertLessEqual(assessment.ttc_seconds, 3.5)

    def test_diverging_trajectories_return_none(self):
        # 场景：机动车向北开，行人已经在车后方或远离运动
        veh = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=10.0,
            vx_ms=0.0,
            vy_ms=8.0,
            speed_ms=8.0,
            speed_kmh=28.8,
            is_speeding=False,
            heading_deg=90.0,
            is_reliable=True,
        )
        ped = SpeedReading(
            track_id=2,
            category="pedestrian",
            x_m=3.0,
            y_m=2.0,
            vx_ms=0.0,
            vy_ms=-1.0,
            speed_ms=1.0,
            speed_kmh=3.6,
            is_speeding=False,
            heading_deg=270.0,
            is_reliable=True,
        )

        assessment = self.analyzer.evaluate_pair(veh, ped)
        self.assertIsNone(assessment)

    def test_analyze_frame_isolated_speeding(self):
        # 仅有车辆超速，无行人
        veh = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=10.0,
            vx_ms=0.0,
            vy_ms=10.0,
            speed_ms=10.0,
            speed_kmh=36.0,
            is_speeding=True,
            heading_deg=90.0,
            is_reliable=True,
        )
        assessments = self.analyzer.analyze_frame([veh])
        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].risk_level, RiskLevel.LEVEL_1_SPEEDING)

    def test_stationary_vehicle_exempt_from_conflict(self):
        # 持续静止的泊车/停靠车辆（is_stationary=True）：行人走近不构成交通冲突
        parked = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=5.0,
            vx_ms=0.0,
            vy_ms=0.0,
            speed_ms=0.0,
            speed_kmh=0.0,
            is_speeding=False,
            heading_deg=0.0,
            is_reliable=True,
            is_stationary=True,
        )
        ped = SpeedReading(
            track_id=2,
            category="pedestrian",
            x_m=7.0,
            y_m=5.0,
            vx_ms=-2.0,
            vy_ms=0.0,
            speed_ms=2.0,
            speed_kmh=7.2,
            is_speeding=False,
            heading_deg=180.0,
            is_reliable=True,
        )
        self.assertIsNone(self.analyzer.evaluate_pair(parked, ped))

    def test_slow_crawling_vehicle_not_exempt(self):
        # 0.4 m/s 蠕行车辆（is_stationary=False）必须参与碰撞研判，不得漏报
        crawler = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=0.0,
            vx_ms=0.4,
            vy_ms=0.0,
            speed_ms=0.4,
            speed_kmh=1.44,
            is_speeding=False,
            heading_deg=0.0,
            is_reliable=True,
            is_stationary=False,
        )
        ped = SpeedReading(
            track_id=2,
            category="pedestrian",
            x_m=6.0,
            y_m=0.0,
            vx_ms=-2.0,
            vy_ms=0.0,
            speed_ms=2.0,
            speed_kmh=7.2,
            is_speeding=False,
            heading_deg=180.0,
            is_reliable=True,
        )
        assessment = self.analyzer.evaluate_pair(crawler, ped)
        self.assertIsNotNone(assessment)
        self.assertEqual(assessment.risk_level, RiskLevel.LEVEL_3_CRITICAL)

    def test_invalid_analyzer_params_rejected(self):
        with self.assertRaises(ValueError):
            ConflictAnalyzer(ttc_critical_s=3.5, ttc_caution_s=1.8)
        with self.assertRaises(ValueError):
            ConflictAnalyzer(ttc_caution_s=4.0, horizon_s=3.0)
        with self.assertRaises(ValueError):
            ConflictAnalyzer(speed_limit_kmh=0.0)
        with self.assertRaises(ValueError):
            ConflictAnalyzer(horizon_s=float("nan"))
        with self.assertRaises(ValueError):
            ConflictAnalyzer(safe_radius_veh_m=float("nan"))

    def test_ped_conflict_map_picks_highest_risk(self):
        # 同一行人被两车关联：行人侧映射保留最高风险项，同级取最小 TTC
        from analysis.conflict import build_conflict_maps

        def mk(veh_id, ped_id, level, ttc):
            return ConflictAssessment(
                veh_track_id=veh_id,
                ped_track_id=ped_id,
                risk_level=level,
                risk_desc="x",
                veh_speed_kmh=20.0,
                ped_speed_ms=1.0,
                current_distance_m=5.0,
                min_distance_m=1.0,
                ttc_seconds=ttc,
                conflict_ground_xy=(1.0, 2.0),
            )

        c_caution = mk(1, 9, RiskLevel.LEVEL_2_CAUTION, 2.5)
        c_critical = mk(2, 9, RiskLevel.LEVEL_3_CRITICAL, 1.0)
        veh_map, ped_map = build_conflict_maps([c_caution, c_critical])
        self.assertIs(ped_map[9], c_critical)
        self.assertIs(veh_map[1], c_caution)
        self.assertIs(veh_map[2], c_critical)

        # 同级冲突取更小 TTC
        c_far = mk(1, 9, RiskLevel.LEVEL_2_CAUTION, 3.0)
        c_near = mk(2, 9, RiskLevel.LEVEL_2_CAUTION, 2.0)
        _, ped_map2 = build_conflict_maps([c_far, c_near])
        self.assertIs(ped_map2[9], c_near)

    def test_unreliable_stationary_vehicle_still_evaluated(self):
        # 速度未收敛的车辆不做静止豁免，保留保守预警
        warming = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=5.0,
            vx_ms=0.0,
            vy_ms=0.0,
            speed_ms=0.0,
            speed_kmh=0.0,
            is_speeding=False,
            heading_deg=0.0,
            is_reliable=False,
        )
        ped = SpeedReading(
            track_id=2,
            category="pedestrian",
            x_m=7.0,
            y_m=5.0,
            vx_ms=-2.0,
            vy_ms=0.0,
            speed_ms=2.0,
            speed_kmh=7.2,
            is_speeding=False,
            heading_deg=180.0,
            is_reliable=True,
        )
        self.assertIsNotNone(self.analyzer.evaluate_pair(warming, ped))

    def test_stationary_but_unreliable_not_exempt(self):
        # --parked-frames 1 或低 FPS 下，静止标记可能在收敛前置位：
        # is_stationary=True 且 is_reliable=False 的车辆不得豁免，
        # 仍按收敛前保守策略参与碰撞研判
        early_parked = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=5.0,
            vx_ms=0.0,
            vy_ms=0.0,
            speed_ms=0.0,
            speed_kmh=0.0,
            is_speeding=False,
            heading_deg=0.0,
            is_reliable=False,
            is_stationary=True,
        )
        ped = SpeedReading(
            track_id=2,
            category="pedestrian",
            x_m=7.0,
            y_m=5.0,
            vx_ms=-2.0,
            vy_ms=0.0,
            speed_ms=2.0,
            speed_kmh=7.2,
            is_speeding=False,
            heading_deg=180.0,
            is_reliable=True,
        )
        self.assertIsNotNone(self.analyzer.evaluate_pair(early_parked, ped))

    def test_analyze_frame_isolated_safe(self):
        # 仅有车辆正常行驶，无行人
        veh = SpeedReading(
            track_id=1,
            category="vehicle",
            x_m=3.0,
            y_m=10.0,
            vx_ms=0.0,
            vy_ms=5.0,
            speed_ms=5.0,
            speed_kmh=18.0,
            is_speeding=False,
            heading_deg=90.0,
            is_reliable=True,
        )
        assessments = self.analyzer.analyze_frame([veh])
        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].risk_level, RiskLevel.LEVEL_0_SAFE)


class TestConflictEventTracker(unittest.TestCase):
    """告警事件状态机：连续确认、等级滞回、冷却合并"""

    def _mk(self, veh, ped, level, ttc=None, speed=20.0):
        return ConflictAssessment(
            veh_track_id=veh,
            ped_track_id=ped,
            risk_level=level,
            risk_desc="x",
            veh_speed_kmh=speed,
            ped_speed_ms=1.0 if ped is not None else 0.0,
            current_distance_m=5.0,
            min_distance_m=1.0,
            ttc_seconds=ttc,
            conflict_ground_xy=(1.0, 2.0) if ped is not None else None,
        )

    def test_single_frame_flicker_produces_no_event(self):
        tr = ConflictEventTracker(confirm_frames=3, cooldown_frames=5)
        ann = tr.update([self._mk(1, 2, RiskLevel.LEVEL_3_CRITICAL, 1.0)], 0, 0.0)
        self.assertEqual(ann, {})
        for i in range(1, 8):
            tr.update([], i, i * 0.1)
        tr.flush()
        self.assertEqual(tr.pop_finalized(), [])

    def test_sustained_alert_becomes_single_event(self):
        tr = ConflictEventTracker(confirm_frames=3, cooldown_frames=5)
        anns = [
            tr.update(
                [self._mk(1, 2, RiskLevel.LEVEL_3_CRITICAL, ttc=1.0, speed=36.0)],
                i, i * 0.04,
            )
            for i in range(10)
        ]
        # 前两帧挂起不注解，第 3 帧确认后全程同一 event_id
        self.assertEqual(anns[0], {})
        self.assertEqual(anns[1], {})
        self.assertTrue(all(a == {(1, 2): 1} for a in anns[2:]))

        tr.flush()
        (ev,) = tr.pop_finalized()
        self.assertEqual(ev.event_id, 1)
        self.assertEqual(ev.start_frame, 0)   # 起始帧回溯包含确认前挂起帧
        self.assertEqual(ev.end_frame, 9)
        self.assertEqual(ev.alarm_frames, 10)
        self.assertEqual(ev.gap_frames, 0)
        self.assertEqual(ev.peak_risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertEqual(ev.min_ttc_seconds, 1.0)
        self.assertEqual(ev.max_veh_speed_kmh, 36.0)
        self.assertEqual(ev.kind, "conflict")
        self.assertEqual(ev.end_reason, "eof")

    def test_gap_within_cooldown_merges_same_event(self):
        tr = ConflictEventTracker(confirm_frames=2, cooldown_frames=4)
        for i in range(5):  # f0-4 告警
            ann = tr.update(
                [self._mk(1, 2, RiskLevel.LEVEL_2_CAUTION, 2.5)], i, i * 0.1
            )
        self.assertEqual(ann, {(1, 2): 1})
        for i in range(5, 8):  # f5-7 静默 3 帧 <= cooldown=4，不归档
            tr.update([], i, i * 0.1)
        self.assertEqual(tr.pop_finalized(), [])

        ann = tr.update([self._mk(1, 2, RiskLevel.LEVEL_3_CRITICAL, 1.0)], 8, 0.8)
        self.assertEqual(ann, {(1, 2): 1})  # 同一事件延续且峰值升级为 L3
        tr.flush()
        (ev,) = tr.pop_finalized()
        self.assertEqual(ev.end_frame, 8)
        self.assertEqual(ev.gap_frames, 3)
        self.assertEqual(ev.alarm_frames, 6)
        self.assertEqual(ev.peak_risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertEqual(ev.min_ttc_seconds, 1.0)

    def test_gap_beyond_cooldown_splits_into_two_events(self):
        tr = ConflictEventTracker(confirm_frames=2, cooldown_frames=2)
        for i in range(3):
            tr.update([self._mk(1, 2, RiskLevel.LEVEL_2_CAUTION, 2.0)], i, i * 0.1)
        for i in range(3, 7):  # 静默 4 帧 > cooldown=2 -> 归档
            tr.update([], i, i * 0.1)
        (first,) = tr.pop_finalized()
        self.assertEqual(first.end_reason, "cooldown")
        self.assertEqual(first.end_frame, 2)

        anns = [
            tr.update([self._mk(1, 2, RiskLevel.LEVEL_2_CAUTION, 2.0)], i, i * 0.1)
            for i in range(7, 11)
        ]
        self.assertEqual(anns[-1], {(1, 2): 2})
        tr.flush()
        (second,) = tr.pop_finalized()
        self.assertEqual(second.event_id, 2)
        self.assertEqual(second.start_frame, 7)

    def test_level_hysteresis_peak_never_downgrades(self):
        tr = ConflictEventTracker(confirm_frames=2, cooldown_frames=3)
        tr.update([self._mk(1, 2, RiskLevel.LEVEL_3_CRITICAL, 1.0)], 0, 0.0)
        tr.update([self._mk(1, 2, RiskLevel.LEVEL_3_CRITICAL, 1.0)], 1, 0.1)
        tr.update([self._mk(1, 2, RiskLevel.LEVEL_2_CAUTION, 2.5)], 2, 0.2)
        tr.flush()
        (ev,) = tr.pop_finalized()
        self.assertEqual(ev.peak_risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertEqual(ev.alarm_frames, 3)

    def test_speeding_and_conflict_are_independent_events(self):
        tr = ConflictEventTracker(confirm_frames=2, cooldown_frames=2)
        for i in range(4):
            ann = tr.update(
                [
                    self._mk(1, None, RiskLevel.LEVEL_1_SPEEDING, speed=40.0),
                    self._mk(1, 5, RiskLevel.LEVEL_2_CAUTION, ttc=2.0, speed=40.0),
                ],
                i, i * 0.1,
            )
        self.assertEqual(ann[(1, None)], 1)
        self.assertEqual(ann[(1, 5)], 2)
        tr.flush()
        evts = sorted(tr.pop_finalized(), key=lambda e: e.event_id)
        self.assertEqual(len(evts), 2)
        self.assertEqual(evts[0].kind, "speeding")
        self.assertIsNone(evts[0].ped_track_id)
        self.assertEqual(evts[1].kind, "conflict")

    def test_pending_streak_interrupted_resets(self):
        tr = ConflictEventTracker(confirm_frames=3, cooldown_frames=2)
        tr.update([self._mk(1, 2, RiskLevel.LEVEL_2_CAUTION, 2.0)], 0, 0.0)
        tr.update([self._mk(1, 2, RiskLevel.LEVEL_2_CAUTION, 2.0)], 1, 0.1)
        tr.update([], 2, 0.2)  # 中断挂起
        tr.update([self._mk(1, 2, RiskLevel.LEVEL_2_CAUTION, 2.0)], 3, 0.3)
        ann = tr.update([self._mk(1, 2, RiskLevel.LEVEL_2_CAUTION, 2.0)], 4, 0.4)
        self.assertEqual(ann, {})  # 重新累计仅 2 帧，未确认
        tr.flush()
        self.assertEqual(tr.pop_finalized(), [])

    def test_safe_assessment_never_alarms(self):
        tr = ConflictEventTracker()
        ann = tr.update([self._mk(1, None, RiskLevel.LEVEL_0_SAFE)], 0, 0.0)
        self.assertEqual(ann, {})
        tr.flush()
        self.assertEqual(tr.pop_finalized(), [])

    def test_min_ttc_tracks_worst_case(self):
        tr = ConflictEventTracker(confirm_frames=2, cooldown_frames=2)
        tr.update([self._mk(1, 2, RiskLevel.LEVEL_2_CAUTION, 2.8)], 0, 0.0)
        tr.update([self._mk(1, 2, RiskLevel.LEVEL_3_CRITICAL, 1.2)], 1, 0.1)
        tr.update([self._mk(1, 2, RiskLevel.LEVEL_3_CRITICAL, 0.9)], 2, 0.2)
        tr.flush()
        (ev,) = tr.pop_finalized()
        self.assertAlmostEqual(ev.min_ttc_seconds, 0.9)

    def test_invalid_params_rejected(self):
        with self.assertRaises(ValueError):
            ConflictEventTracker(confirm_frames=0)
        with self.assertRaises(ValueError):
            ConflictEventTracker(cooldown_frames=-1)

    def test_update_rejects_invalid_frame_args(self):
        tr = ConflictEventTracker()
        with self.assertRaises(ValueError):
            tr.update([], -1, 0.0)
        with self.assertRaises(ValueError):
            tr.update([], 0, float("nan"))

    def test_event_to_dict_is_json_serializable(self):
        import json

        tr = ConflictEventTracker(confirm_frames=1, cooldown_frames=1)
        tr.update([self._mk(1, 2, RiskLevel.LEVEL_3_CRITICAL, 1.0)], 0, 0.0)
        tr.flush()
        (ev,) = tr.pop_finalized()
        d = ev.to_dict()
        json.dumps(d)
        self.assertEqual(d["peak_risk_name"], "LEVEL_3_CRITICAL")
        self.assertEqual(d["kind"], "conflict")
        self.assertEqual(d["event_id"], 1)


class TestVisualization(unittest.TestCase):
    def test_visualizer_render_frame_without_error(self):
        image_points = [(100.0, 50.0), (300.0, 50.0), (350.0, 250.0), (50.0, 250.0)]
        calibrator = RoadCalibrator(image_points, 6.0, 20.0)
        visualizer = EnhancedConflictVisualizer(calibrator)

        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        readings = {
            1: SpeedReading(
                track_id=1,
                category="vehicle",
                x_m=3.0,
                y_m=5.0,
                vx_ms=0.0,
                vy_ms=5.0,
                speed_ms=5.0,
                speed_kmh=18.0,
                is_speeding=False,
                heading_deg=90.0,
                is_reliable=True,
            )
        }
        assessments = [
            ConflictAssessment(
                veh_track_id=1,
                ped_track_id=None,
                risk_level=RiskLevel.LEVEL_0_SAFE,
                risk_desc="正常行驶 (18.0 km/h)",
                veh_speed_kmh=18.0,
                ped_speed_ms=0.0,
                current_distance_m=0.0,
                min_distance_m=0.0,
                ttc_seconds=None,
                conflict_ground_xy=None,
            )
        ]

        # 传入虚拟目标渲染
        annotated = visualizer.draw_frame(
            frame=frame,
            tracks=[],
            speed_readings=readings,
            conflicts=assessments,
        )
        self.assertEqual(annotated.shape, frame.shape)

        # 测试独立渲染方法
        visualizer.draw_calibrated_ground_zone(annotated)
        visualizer.draw_future_trajectory(annotated, readings[1])
        visualizer.draw_speed_badge(annotated, (10, 10, 50, 50), readings[1], assessments[0])
        visualizer.draw_hud_banner(annotated, assessments)


class TestPipelineIO(unittest.TestCase):
    def test_video_writer_failure_raises(self):
        import cv2
        import numpy as np
        from unittest import mock

        from analysis.pipeline import ConflictPerceptionPipeline

        with tempfile.TemporaryDirectory() as d:
            vpath = Path(d) / "v.mp4"
            w = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64))
            self.assertTrue(w.isOpened())
            for _ in range(3):
                w.write(np.zeros((64, 64, 3), np.uint8))
            w.release()

            calib = RoadCalibrator(
                [(10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)], 4.0, 4.0
            )
            pipeline = ConflictPerceptionPipeline(calibrator=calib)
            with mock.patch("analysis.pipeline.cv2.VideoWriter") as m_writer:
                m_writer.return_value.isOpened.return_value = False
                with self.assertRaises(RuntimeError):
                    pipeline.process_video(vpath, Path(d) / "out")

    def test_cli_rejects_invalid_numeric_args(self):
        from analysis.__main__ import main

        for bad_args in (
            ["--ema-alpha", "0"],
            ["--ema-alpha", "1.5"],
            ["--ema-alpha", "nan"],
            ["--window", "1"],
            ["--window-s", "0"],
            ["--window-s", "nan"],
            ["--ttc-critical", "3.5", "--ttc-caution", "1.8"],
            ["--parked-speed", "0"],
            ["--parked-speed", "nan"],
            ["--parked-frames", "0"],
            ["--parked-seconds", "0"],
            ["--alert-confirm", "0"],
            ["--alert-cooldown", "-1"],
            ["--alert-cooldown-s", "-0.5"],
            ["--alert-cooldown-s", "nan"],
            ["--fps", "0"],
            ["--fps", "-25"],
            ["--fps", "nan"],
            ["--reliable-span-s", "-1"],
            ["--reliable-span-s", "nan"],
            ["--speed-limit", "nan"],
        ):
            with self.assertRaises(SystemExit, msg=bad_args):
                main(["--source", "x.mp4", *bad_args])

    def test_synthetic_calibration_rejected_without_demo(self):
        import yaml

        from analysis.__main__ import main

        with tempfile.TemporaryDirectory() as d:
            calib = Path(d) / "calib.yaml"
            calib.write_text(yaml.safe_dump({
                "name": "fake", "synthetic": True,
                "width_m": 6.0, "length_m": 15.0,
                "image_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "ground_points": [[0, 0], [6, 0], [6, 15], [0, 15]],
            }), encoding="utf-8")
            with self.assertRaises(SystemExit):
                main(["--source", "x.mp4", "--calib", str(calib)])

    def test_estimator_rebuilt_per_video_and_fps_scaled(self):
        from analysis.pipeline import ConflictPerceptionPipeline

        calib = RoadCalibrator(
            [(10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)], 4.0, 4.0
        )
        pipe = ConflictPerceptionPipeline(
            calibrator=calib, window_seconds=0.4, stationary_seconds=0.5
        )
        pipe.speed_estimator.update(1, "vehicle", 0.0, 0.0, 0.0)
        self.assertEqual(len(pipe.speed_estimator.tracks), 1)

        # 25 fps 下 0.4s 窗口 = 10 帧；0.5s 静止确认 = 12 帧；且旧轨迹状态清空
        pipe._configure_speed_estimator(25.0)
        self.assertEqual(pipe.speed_estimator.window_size, 10)
        self.assertEqual(pipe.speed_estimator.stationary_min_frames, 12)
        self.assertEqual(len(pipe.speed_estimator.tracks), 0)

    def test_event_tracker_rebuilt_per_video_and_fps_scaled(self):
        from analysis.conflict import RiskLevel
        from analysis.pipeline import ConflictPerceptionPipeline

        calib = RoadCalibrator(
            [(10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)], 4.0, 4.0
        )
        pipe = ConflictPerceptionPipeline(
            calibrator=calib, alert_confirm_frames=2, alert_cooldown_seconds=0.4
        )
        # 旧视频中遗留一个存续事件：重建后不得串到下一视频
        pipe.event_tracker.update(
            [
                ConflictAssessment(
                    veh_track_id=1, ped_track_id=None,
                    risk_level=RiskLevel.LEVEL_1_SPEEDING, risk_desc="x",
                    veh_speed_kmh=40.0, ped_speed_ms=0.0, current_distance_m=0.0,
                    min_distance_m=0.0, ttc_seconds=None, conflict_ground_xy=None,
                )
            ],
            0, 0.0,
        )
        pipe._configure_event_tracker(25.0)
        # 0.4s 冷却 @25fps = 10 帧；内部状态清空
        self.assertEqual(pipe.event_tracker.cooldown_frames, 10)
        self.assertEqual(len(pipe.event_tracker._states), 0)

    def test_retained_ids_failure_fails_loud(self):
        import cv2
        import numpy as np
        from unittest import mock

        from analysis.pipeline import ConflictPerceptionPipeline

        with tempfile.TemporaryDirectory() as d:
            vpath = Path(d) / "v.mp4"
            w = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64))
            for _ in range(3):
                w.write(np.zeros((64, 64, 3), np.uint8))
            w.release()

            calib = RoadCalibrator(
                [(10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)], 4.0, 4.0
            )
            pipe = ConflictPerceptionPipeline(calibrator=calib)
            fake_session = mock.Mock()
            fake_session.retained_ids.side_effect = AttributeError("no trackers")
            frame_rec = mock.Mock()
            frame_rec.tracks = []
            fake_session.process.return_value = frame_rec
            with mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session):
                with self.assertRaises(RuntimeError):
                    pipe.process_video(vpath, Path(d) / "out")

    def test_pipeline_writes_debounced_alert_events(self):
        # 端到端：单目标持续超速 -> 去抖后 events.jsonl 恰有 1 条事件，
        # conflicts.jsonl 已确认帧回填 event_id，summary 按事件计数
        import json as jsonlib

        import cv2
        import numpy as np
        from unittest import mock

        from analysis.pipeline import ConflictPerceptionPipeline

        with tempfile.TemporaryDirectory() as d:
            vpath = Path(d) / "v.mp4"
            w = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64))
            self.assertTrue(w.isOpened())
            for _ in range(15):
                w.write(np.zeros((64, 64, 3), np.uint8))
            w.release()

            # 40m x 40m 标定区：锚点每帧下移 0.7px ≈ 地面 0.56m/帧 ≈ 50 km/h
            calib = RoadCalibrator(
                [(10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)], 40.0, 40.0
            )
            pipe = ConflictPerceptionPipeline(
                calibrator=calib, alert_confirm_frames=2, alert_cooldown_frames=2
            )

            frame_i = {"n": 0}

            def make_frame_rec(frame, timestamp_s, time_kind):
                t = mock.Mock()
                t.track_id = 1
                t.category = "vehicle"
                t.anchor_xy = (30.0, 10.0 + frame_i["n"] * 0.7)
                t.bbox_xyxy = (20.0, 10.0, 40.0, 30.0)
                frame_i["n"] += 1
                rec = mock.Mock()
                rec.tracks = [t]
                return rec

            fake_session = mock.Mock()
            fake_session.process.side_effect = make_frame_rec
            fake_session.retained_ids.return_value = {1}

            out_dir = Path(d) / "out"
            with mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session):
                summary = pipe.process_video(vpath, out_dir)

            events_path = out_dir / "events.jsonl"
            events = [
                jsonlib.loads(l)
                for l in events_path.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            self.assertEqual(len(events), 1)
            ev = events[0]
            self.assertEqual(ev["event_id"], 1)
            self.assertEqual(ev["veh_track_id"], 1)
            self.assertIsNone(ev["ped_track_id"])
            self.assertEqual(ev["kind"], "speeding")
            self.assertEqual(ev["peak_risk_name"], "LEVEL_1_SPEEDING")
            self.assertEqual(ev["end_reason"], "eof")
            self.assertGreaterEqual(ev["alarm_frames"], 2)

            self.assertEqual(summary["total_alert_events"], 1)
            self.assertEqual(summary["total_speeding_alerts"], 1)
            self.assertEqual(summary["total_critical_conflicts"], 0)

            lines = [
                jsonlib.loads(l)
                for l in (out_dir / "conflicts.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            confirmed = [l for l in lines if l.get("event_id") == 1]
            self.assertGreaterEqual(len(confirmed), 1)
            # 每帧一行原始评估仍在，确认前帧 event_id 为空
            self.assertTrue(any(l.get("event_id") is None for l in lines))

    def test_missing_fps_rejected_without_override(self):
        # 与 tracking 模块同策略：源 FPS 缺失且未显式 --fps 时拒绝运行，
        # 避免把测速/TTC 建立在错误的 25fps 兜底时间基准上
        import cv2
        from unittest import mock

        from analysis.pipeline import ConflictPerceptionPipeline

        with tempfile.TemporaryDirectory() as d:
            calib = RoadCalibrator(
                [(10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)], 4.0, 4.0
            )
            fake_cap = mock.Mock()
            fake_cap.isOpened.return_value = True
            fake_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 0.0,
                cv2.CAP_PROP_FRAME_WIDTH: 64.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 64.0,
            }.get(prop, 0.0)
            fake_cap.read.return_value = (False, None)

            with mock.patch("analysis.pipeline.cv2.VideoCapture", return_value=fake_cap):
                with mock.patch("analysis.pipeline.TrackingSession"):
                    with self.assertRaises(RuntimeError):
                        ConflictPerceptionPipeline(calibrator=calib).process_video(
                            "v.mp4", Path(d) / "out1"
                        )

                    # 显式指定 fps 后允许运行（0 帧也产出 summary）
                    pipe = ConflictPerceptionPipeline(calibrator=calib, fps=25.0)
                    summary = pipe.process_video("v.mp4", Path(d) / "out2")
                    self.assertEqual(summary["frames_processed"], 0)

    def test_invalid_fps_param_rejected(self):
        from analysis.pipeline import ConflictPerceptionPipeline

        calib = RoadCalibrator(
            [(10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)], 4.0, 4.0
        )
        with self.assertRaises(ValueError):
            ConflictPerceptionPipeline(calibrator=calib, fps=-1.0)
        with self.assertRaises(ValueError):
            ConflictPerceptionPipeline(calibrator=calib, fps=float("nan"))


if __name__ == "__main__":
    unittest.main()
