# -*- coding: utf-8 -*-
"""分析模块 (标定、测速、冲突预测与 TTC 评估) 单元测试。"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.calibration import RoadCalibrator
from analysis.conflict import ConflictAnalyzer, ConflictAssessment, RiskLevel
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



if __name__ == "__main__":
    unittest.main()
