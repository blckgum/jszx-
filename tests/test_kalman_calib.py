# -*- coding: utf-8 -*-
"""标定雅可比、多点/RANSAC/去畸变标定与 2D 匀加速卡尔曼测速单元测试。"""

from dataclasses import asdict
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

from analysis.calibration import RoadCalibrator
from analysis.kalman import ConstantAccelerationKalman2D
from analysis.speed import SpeedEstimator, SpeedReading, TrackMotionTracker


class TestCalibrationJacobianAndCovariance(unittest.TestCase):
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
            name="calib_jac_test",
        )
        self.camera_matrix = [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ]
        self.dist_coeffs = [-0.25, 0.08, 0.001, -0.001, 0.0]

    def _central_fd_jacobian(self, cal: RoadCalibrator, u: float, v: float, h: float = 1e-4) -> np.ndarray:
        x_up, y_up = cal.pixel_to_ground(u + h, v)
        x_um, y_um = cal.pixel_to_ground(u - h, v)
        x_vp, y_vp = cal.pixel_to_ground(u, v + h)
        x_vm, y_vm = cal.pixel_to_ground(u, v - h)
        return np.array(
            [
                [(x_up - x_um) / (2.0 * h), (x_vp - x_vm) / (2.0 * h)],
                [(y_up - y_um) / (2.0 * h), (y_vp - y_vm) / (2.0 * h)],
            ],
            dtype=np.float64,
        )

    def test_jacobian_matches_central_finite_differences_without_distortion(self):
        for u, v in ((640.0, 490.0), (420.0, 320.0), (950.0, 620.0), (260.0, 640.0)):
            with self.subTest(u=u, v=v):
                jac = self.calibrator.pixel_to_ground_jacobian(u, v)
                self.assertIsNotNone(jac)
                self.assertEqual(jac.shape, (2, 2))
                fd = self._central_fd_jacobian(self.calibrator, u, v)
                max_err = float(np.max(np.abs(jac - fd)))
                self.assertLess(max_err, 1e-6)

    def test_jacobian_matches_central_finite_differences_with_radial_distortion(self):
        cal_dist = RoadCalibrator(
            image_points=self.trapezoid,
            width_m=6.0,
            length_m=15.0,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
        )
        for u, v in ((640.0, 490.0), (430.0, 330.0), (980.0, 610.0), (220.0, 630.0)):
            with self.subTest(u=u, v=v):
                jac = cal_dist.pixel_to_ground_jacobian(u, v)
                self.assertIsNotNone(jac)
                self.assertEqual(jac.shape, (2, 2))
                fd = self._central_fd_jacobian(cal_dist, u, v)
                max_err = float(np.max(np.abs(jac - fd)))
                self.assertLess(max_err, 1e-6)

    def test_jacobian_and_covariance_reject_invalid_and_horizon_inputs(self):
        for bad_u, bad_v in (
            (True, 400.0),
            (640.0, False),
            (float("nan"), 400.0),
            (640.0, float("inf")),
            ("640", 400.0),
        ):
            self.assertIsNone(self.calibrator.pixel_to_ground_jacobian(bad_u, bad_v))
            self.assertIsNone(self.calibrator.ground_covariance_from_pixel(bad_u, bad_v))

        # 近地平线奇异点 |w| < 1e-8 返回 None
        w_row = self.calibrator.H[2, :]
        # 构造 H[2,0]*u + H[2,1]*v + H[2,2] == 0
        u_test = 640.0
        v_horizon = -(w_row[0] * u_test + w_row[2]) / w_row[1]
        self.assertIsNone(self.calibrator.pixel_to_ground_jacobian(u_test, v_horizon))
        self.assertIsNone(self.calibrator.ground_covariance_from_pixel(u_test, v_horizon))

        for kwargs in (
            {"sigma_u_px": True},
            {"sigma_v_px": -1.0},
            {"min_var_m2": 0.0},
            {"min_var_m2": float("nan")},
        ):
            with self.assertRaises(ValueError):
                self.calibrator.ground_covariance_from_pixel(640.0, 490.0, **kwargs)

    def test_far_field_covariance_exceeds_near_field_covariance(self):
        # 透视梯形顶部 v=310 为远端（像素密度低、单像素对应地面跨度大），底部 v=660 为近端
        r_far = self.calibrator.ground_covariance_from_pixel(640.0, 310.0)
        r_near = self.calibrator.ground_covariance_from_pixel(640.0, 660.0)
        self.assertIsNotNone(r_far)
        self.assertIsNotNone(r_near)

        for cov in (r_far, r_near):
            self.assertEqual(cov.shape, (2, 2))
            self.assertTrue(np.allclose(cov, cov.T))
            eigvals = np.linalg.eigvalsh(cov)
            self.assertTrue(np.all(eigvals > 0.0))

        self.assertGreater(float(np.trace(r_far)), float(np.trace(r_near)) * 2.0)
        self.assertGreater(float(np.linalg.det(r_far)), float(np.linalg.det(r_near)) * 2.0)


class TestMultiPointAndDistortionCalibration(unittest.TestCase):
    def setUp(self):
        self.base_img = [
            (400.0, 300.0),
            (880.0, 300.0),
            (1150.0, 680.0),
            (130.0, 680.0),
        ]
        self.base_cal = RoadCalibrator(self.base_img, 6.0, 15.0)
        self.camera_matrix = [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ]
        self.dist_coeffs = [-0.25, 0.08, 0.001, -0.001, 0.0]

    def _project_ground_points(self, gd_pts):
        img_pts = []
        for gx, gy in gd_pts:
            p = self.base_cal.H_inv @ np.array([gx, gy, 1.0], dtype=np.float64)
            img_pts.append((float(p[0] / p[2]), float(p[1] / p[2])))
        return img_pts

    def test_from_correspondences_least_squares_and_shifted_bounds(self):
        # 地面坐标不以 (0,0) 为原点（x in [-2, 4], y in [2, 17]），检验跨度与区域边界
        gd_pts = [
            (-2.0, 2.0), (1.0, 2.0), (4.0, 2.0),
            (-2.0, 8.0), (1.0, 8.0), (4.0, 8.0),
            (-2.0, 17.0), (1.0, 17.0), (4.0, 17.0),
            (0.0, 5.0), (2.5, 12.0), (3.5, 14.0),
        ]
        # 通过已知单应性生成对应图像点并加微小亚像素扰动
        rng = np.random.default_rng(123)
        ref_cal = RoadCalibrator(self.base_img, 6.0, 15.0)
        # 平移映射：让 [-2,4]x[2,17] 对应标准域附近
        img_pts = []
        for gx, gy in gd_pts:
            sx, sy = gx + 2.0, gy - 2.0
            p = ref_cal.H_inv @ np.array([sx, sy, 1.0], dtype=np.float64)
            u = float(p[0] / p[2]) + float(rng.normal(0.0, 0.25))
            v = float(p[1] / p[2]) + float(rng.normal(0.0, 0.25))
            img_pts.append((u, v))

        cal = RoadCalibrator.from_correspondences(
            img_pts, gd_pts, name="multi_ls", use_ransac=False
        )
        self.assertAlmostEqual(cal.width_m, 6.0, places=6)
        self.assertAlmostEqual(cal.length_m, 15.0, places=6)
        self.assertLess(cal.reprojection_rmse_m, 0.05)
        self.assertEqual(cal.inlier_count, len(gd_pts))
        self.assertTrue(cal.is_in_zone(-1.5, 10.0, margin_m=0.5))
        self.assertFalse(cal.is_in_zone(0.0, -5.0, margin_m=1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            ypath = Path(tmpdir) / "multi_ls.yaml"
            cal.to_yaml(ypath)
            raw = yaml.safe_load(ypath.read_text(encoding="utf-8"))
            self.assertNotIn("camera_matrix", raw)
            self.assertNotIn("dist_coeffs", raw)
            loaded = RoadCalibrator.from_yaml(ypath)
            self.assertAlmostEqual(loaded.width_m, 6.0, places=6)
            self.assertAlmostEqual(loaded.length_m, 15.0, places=6)
            self.assertTrue(np.allclose(loaded.H, cal.H, atol=1e-6))

    def test_from_correspondences_ransac_rejects_gross_outliers(self):
        gd_pts = [
            (float(x), float(y))
            for x in (0.5, 2.0, 4.0, 5.5)
            for y in (1.0, 6.0, 12.0)
        ]
        img_pts = self._project_ground_points(gd_pts)

        # 注入 2 个剧烈外点 (5m 级地面标注偏差)
        gd_corrupted = list(gd_pts)
        gd_corrupted[1] = (gd_corrupted[1][0] + 5.0, gd_corrupted[1][1] - 4.0)
        gd_corrupted[8] = (gd_corrupted[8][0] - 4.5, gd_corrupted[8][1] + 5.5)

        cal_ls = RoadCalibrator.from_correspondences(img_pts, gd_corrupted, use_ransac=False)
        cal_ransac = RoadCalibrator.from_correspondences(
            img_pts, gd_corrupted, use_ransac=True, ransac_reproj_threshold_m=0.5
        )
        self.assertEqual(cal_ransac.inlier_count, len(gd_pts) - 2)
        self.assertLess(cal_ransac.reprojection_rmse_m, 1e-3)
        self.assertGreater(cal_ls.reprojection_rmse_m, 1.0)

        # 验证真实中间点投影精度
        test_u, test_v = self._project_ground_points([(3.0, 7.5)])[0]
        gx, gy = cal_ransac.pixel_to_ground(test_u, test_v)
        self.assertAlmostEqual(gx, 3.0, delta=1e-3)
        self.assertAlmostEqual(gy, 7.5, delta=1e-3)

        # RANSAC 持久化与重载应保持一致
        with tempfile.TemporaryDirectory() as tmpdir:
            ypath = Path(tmpdir) / "ransac.yaml"
            cal_ransac.to_yaml(ypath)
            loaded = RoadCalibrator.from_yaml(ypath)
            gx2, gy2 = loaded.pixel_to_ground(test_u, test_v)
            self.assertAlmostEqual(gx2, 3.0, delta=1e-3)
            self.assertAlmostEqual(gy2, 7.5, delta=1e-3)

    def test_from_correspondences_ransac_bounds_exclude_outliers_and_check_inlier_span(self):
        # 12 组 6m x 15m 对应点中注入 2 个远距地面粗差外点，确认外点不撑大区域边界与快照宽高
        gd_pts = [
            (float(x), float(y))
            for x in (0.0, 2.0, 4.0, 6.0)
            for y in (0.0, 7.5, 15.0)
        ]
        img_pts = self._project_ground_points(gd_pts)
        gd_corrupt = list(gd_pts)
        gd_corrupt[1] = (100.0, 200.0)
        gd_corrupt[10] = (-50.0, -80.0)

        cal = RoadCalibrator.from_correspondences(
            img_pts, gd_corrupt, use_ransac=True, ransac_reproj_threshold_m=0.5
        )
        self.assertEqual(cal.inlier_count, 10)
        self.assertAlmostEqual(cal.width_m, 6.0, places=6)
        self.assertAlmostEqual(cal.length_m, 15.0, places=6)
        self.assertEqual(cal._zone_bounds, (0.0, 6.0, 0.0, 15.0))
        self.assertFalse(cal.is_in_zone(90.0, 190.0))
        self.assertFalse(cal.is_in_zone(-40.0, -70.0))
        self.assertTrue(cal.is_in_zone(3.0, 7.5))

        with tempfile.TemporaryDirectory() as tmpdir:
            ypath = Path(tmpdir) / "ransac_bounds.yaml"
            cal.to_yaml(ypath)
            loaded = RoadCalibrator.from_yaml(ypath)
            self.assertAlmostEqual(loaded.width_m, 6.0, places=6)
            self.assertAlmostEqual(loaded.length_m, 15.0, places=6)
            self.assertFalse(loaded.is_in_zone(90.0, 190.0))

        # 若内点本身全共线（仅靠外点撑开跨度），RANSAC 后也应被内点跨度或单应性检验拒绝
        collinear_inliers = [(3.0, float(i)) for i in range(10)] + [(10.0, 0.0), (0.0, 10.0)]
        with self.assertRaises(ValueError):
            RoadCalibrator.from_correspondences(
                img_pts, collinear_inliers, use_ransac=True, ransac_reproj_threshold_m=0.5
            )

    def test_from_yaml_strict_ground_points_width_and_numpy_scalar_zone_checks(self):
        # 1. 显式提供 4 个损坏 ground_points 时不再静默吞错
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_gd_yaml = root / "bad_gd4.yaml"
            bad_gd_yaml.write_text(
                "name: bad_gd4\n"
                "synthetic: false\n"
                "width_m: 6.0\n"
                "length_m: 15.0\n"
                "image_points: [[400,300],[880,300],[1150,680],[130,680]]\n"
                'ground_points: [[0,0],["bad",0],[6,15],[0,15]]\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                RoadCalibrator.from_yaml(bad_gd_yaml)

            # 2. >4 点对应点分支同样拦截 bool/非正/非有限 width_m 与 length_m
            gd6 = [(0.0, 0.0), (6.0, 0.0), (6.0, 15.0), (0.0, 15.0), (3.0, 7.5), (1.5, 10.0)]
            img6 = self._project_ground_points(gd6)
            for bad_key, bad_val in (
                ("width_m", "true"),
                ("width_m", "-1.0"),
                ("length_m", "0.0"),
                ("length_m", ".nan"),
            ):
                p = root / f"bad_{bad_key}_{bad_val}.yaml"
                payload = {
                    "name": "bad_corresp_dims",
                    "synthetic": False,
                    "width_m": 6.0,
                    "length_m": 15.0,
                    "image_points": [list(pt) for pt in img6],
                    "ground_points": [list(pt) for pt in gd6],
                }
                text = yaml.safe_dump(payload, sort_keys=False).replace(
                    f"{bad_key}: {payload[bad_key]}",
                    f"{bad_key}: {bad_val}",
                )
                p.write_text(text, encoding="utf-8")
                with self.subTest(bad_key=bad_key, bad_val=bad_val):
                    with self.assertRaises(ValueError):
                        RoadCalibrator.from_yaml(p)

        # 3. is_in_zone / is_in_parking_zone 及投影接口统一支持 NumPy 标量并拒绝 bool/np.bool_
        cal = RoadCalibrator(
            self.base_img,
            6.0,
            15.0,
            parking_zones=[[(0.0, 0.0), (2.0, 0.0), (2.0, 4.0), (0.0, 4.0)]],
        )
        self.assertTrue(cal.is_in_zone(np.float32(3.0), np.int64(5)))
        self.assertFalse(cal.is_in_zone(np.bool_(True), np.float32(5.0)))
        self.assertFalse(cal.is_in_zone(np.float32(3.0), np.float32(5.0), margin_m=True))
        self.assertTrue(cal.is_in_parking_zone(np.float32(1.0), np.int64(2)))
        self.assertFalse(cal.is_in_parking_zone(np.bool_(True), np.float32(2.0)))
        gx, gy = cal.pixel_to_ground(np.float32(640.0), np.int64(490))
        self.assertTrue(math.isfinite(gx) and math.isfinite(gy))
        self.assertTrue(math.isnan(cal.pixel_to_ground(np.bool_(True), 490.0)[0]))
        self.assertIsNotNone(cal.pixel_to_ground_jacobian(np.float32(640.0), np.int64(490)))
        self.assertIsNone(cal.pixel_to_ground_jacobian(np.bool_(False), 490.0))
        self.assertIsNotNone(cal.ground_to_pixel_checked(np.float32(3.0), np.int64(7)))
        self.assertIsNone(cal.ground_to_pixel_checked(np.bool_(True), 7.0))

    def test_from_correspondences_rejects_degenerate_and_invalid_inputs(self):
        gd4 = [(0.0, 0.0), (6.0, 0.0), (6.0, 15.0), (0.0, 15.0)]
        with self.assertRaises(ValueError):
            RoadCalibrator.from_correspondences(self.base_img[:3], gd4[:3])
        with self.assertRaises(ValueError):
            RoadCalibrator.from_correspondences(self.base_img, gd4[:3])
        with self.assertRaises(ValueError):
            RoadCalibrator.from_correspondences(self.base_img, gd4, use_ransac=1)
        with self.assertRaises(ValueError):
            RoadCalibrator.from_correspondences(self.base_img, gd4, ransac_reproj_threshold_m=0.0)
        # 共线地面点
        with self.assertRaises(ValueError):
            RoadCalibrator.from_correspondences(
                self.base_img, [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)]
            )
        # 共线图像点
        with self.assertRaises(ValueError):
            RoadCalibrator.from_correspondences(
                [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (150.0, 0.0)], gd4
            )

    def test_radial_distortion_round_trip_and_yaml_persistence(self):
        K_np = np.array(self.camera_matrix, dtype=np.float64)
        for d_len, coeffs in (
            (4, [-0.25, 0.08, 0.001, -0.001]),
            (5, self.dist_coeffs),
            (8, [-0.2, 0.05, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0]),
        ):
            with self.subTest(dist_len=d_len):
                dist_np = np.array(coeffs, dtype=np.float64)
                # 先将理想针孔四角点加畸变，模拟实际镜头拍到的畸变角点
                K_inv = np.linalg.inv(K_np)
                distorted_corners = []
                for u, v in self.base_img:
                    ray = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
                    obj = np.array([[[ray[0] / ray[2], ray[1] / ray[2], 1.0]]], dtype=np.float64)
                    dp, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K_np, dist_np)
                    distorted_corners.append((float(dp[0, 0, 0]), float(dp[0, 0, 1])))

                cal = RoadCalibrator(
                    image_points=distorted_corners,
                    width_m=6.0,
                    length_m=15.0,
                    camera_matrix=self.camera_matrix,
                    dist_coeffs=coeffs,
                )
                # 地面中心点 (3.0, 7.5) 往返投影应在 1 像素舍入误差内闭环
                px = cal.ground_to_pixel_checked(3.0, 7.5)
                self.assertIsNotNone(px)
                gx, gy = cal.pixel_to_ground(float(px[0]), float(px[1]))
                self.assertAlmostEqual(gx, 3.0, delta=0.05)
                self.assertAlmostEqual(gy, 7.5, delta=0.1)

                # 角点往返
                for (du, dv), (gx_exp, gy_exp) in zip(distorted_corners, [(0.0, 0.0), (6.0, 0.0), (6.0, 15.0), (0.0, 15.0)]):
                    px_g = cal.pixel_to_ground(du, dv)
                    self.assertAlmostEqual(px_g[0], gx_exp, delta=1e-3)
                    self.assertAlmostEqual(px_g[1], gy_exp, delta=1e-3)
                    uv_rec = cal.ground_to_pixel_checked(gx_exp, gy_exp)
                    self.assertIsNotNone(uv_rec)
                    self.assertLessEqual(abs(uv_rec[0] - du), 1.0)
                    self.assertLessEqual(abs(uv_rec[1] - dv), 1.0)

                # YAML 持久化应完整还原 camera_matrix 与 dist_coeffs
                with tempfile.TemporaryDirectory() as tmpdir:
                    ypath = Path(tmpdir) / f"dist_{d_len}.yaml"
                    cal.to_yaml(ypath)
                    loaded = RoadCalibrator.from_yaml(ypath)
                    self.assertEqual(loaded.camera_matrix, self.camera_matrix)
                    self.assertEqual(loaded.dist_coeffs, coeffs)
                    gx2, gy2 = loaded.pixel_to_ground(float(px[0]), float(px[1]))
                    self.assertAlmostEqual(gx, gx2, places=6)
                    self.assertAlmostEqual(gy, gy2, places=6)

    def test_intrinsics_validation_rejects_bool_nan_and_bad_lengths(self):
        # dist_coeffs 不能脱离 camera_matrix 单独传入
        with self.assertRaises(ValueError):
            RoadCalibrator(self.base_img, 6.0, 15.0, dist_coeffs=[0.0, 0.0, 0.0, 0.0])

        bad_K_cases = [
            [[800.0, 0.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
            [[True, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
            [[float("nan"), 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
            "not_a_matrix",
        ]
        for bad_k in bad_K_cases:
            with self.subTest(bad_k=bad_k):
                with self.assertRaises(ValueError):
                    RoadCalibrator(self.base_img, 6.0, 15.0, camera_matrix=bad_k)

        for bad_dist in (
            [0.0, 0.0, 0.0],
            [0.0] * 6,
            [0.0] * 7,
            [True, 0.0, 0.0, 0.0],
            [float("inf"), 0.0, 0.0, 0.0],
        ):
            with self.subTest(bad_dist=bad_dist):
                with self.assertRaises(ValueError):
                    RoadCalibrator(
                        self.base_img,
                        6.0,
                        15.0,
                        camera_matrix=self.camera_matrix,
                        dist_coeffs=bad_dist,
                    )


class TestConstantAccelerationKalman2D(unittest.TestCase):
    def test_cv_trajectory_convergence_and_predict_at(self):
        kf = ConstantAccelerationKalman2D(category="vehicle")
        self.assertEqual(kf.jerk_std_ms3, 2.5)
        self.assertEqual(ConstantAccelerationKalman2D("pedestrian").jerk_std_ms3, 1.5)

        t = 0.0
        dts = [0.04, 0.06, 0.05, 0.05] * 6
        vx_true, vy_true = 6.0, 2.0
        for idx in range(25):
            x_true = 1.0 + vx_true * t
            y_true = 2.0 + vy_true * t
            accepted = kf.update(t, x_true, y_true)
            self.assertTrue(accepted)
            if idx < len(dts):
                t += dts[idx]

        vx, vy = kf.velocity_xy
        ax, ay = kf.acceleration_xy
        self.assertAlmostEqual(vx, vx_true, delta=0.05)
        self.assertAlmostEqual(vy, vy_true, delta=0.05)
        self.assertAlmostEqual(ax, 0.0, delta=0.05)
        self.assertAlmostEqual(ay, 0.0, delta=0.05)

        px, py = kf.position_xy
        fx, fy, cov_future = kf.predict_at(0.5)
        self.assertAlmostEqual(fx, px + vx_true * 0.5, delta=0.1)
        self.assertAlmostEqual(fy, py + vy_true * 0.5, delta=0.1)
        self.assertTrue(np.allclose(cov_future, cov_future.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(cov_future) > 0.0))
        self.assertGreater(float(np.trace(cov_future)), float(np.trace(kf.pos_cov_2x2)))

        # clone() 必须完全独立
        cloned = kf.clone()
        cloned.update(t + 0.1, 100.0, 100.0)
        self.assertAlmostEqual(kf.position_xy[0], px, places=9)

    def test_accelerating_and_braking_trajectories_converge(self):
        # 匀加速: v0=2.0 m/s, a=+2.5 m/s^2
        kf_acc = ConstantAccelerationKalman2D(category="vehicle")
        for i in range(30):
            t = i * 0.1
            y = 2.0 * t + 0.5 * 2.5 * t * t
            kf_acc.update(t, 3.0, y)
        self.assertAlmostEqual(kf_acc.velocity_xy[1], 2.0 + 2.5 * 2.9, delta=0.2)
        self.assertAlmostEqual(kf_acc.acceleration_xy[1], 2.5, delta=0.25)

        # 匀减速制动: v0=12.0 m/s, a=-3.5 m/s^2
        kf_brk = ConstantAccelerationKalman2D(category="vehicle")
        for i in range(30):
            t = i * 0.1
            y = 12.0 * t - 0.5 * 3.5 * t * t
            kf_brk.update(t, 3.0, y)
        self.assertAlmostEqual(kf_brk.velocity_xy[1], 12.0 - 3.5 * 2.9, delta=0.25)
        self.assertAlmostEqual(kf_brk.acceleration_xy[1], -3.5, delta=0.3)

    def test_adaptive_covariance_from_jacobian_and_invalid_fallback(self):
        cal = RoadCalibrator(
            [(400.0, 300.0), (880.0, 300.0), (1150.0, 680.0), (130.0, 680.0)],
            6.0,
            15.0,
        )
        kf = ConstantAccelerationKalman2D(category="vehicle", default_pos_std_m=0.35)
        for i in range(10):
            t = i * 0.04
            y_true = 2.0 + 5.0 * t
            uv = cal.ground_to_pixel_checked(3.0, y_true)
            cov = cal.ground_covariance_from_pixel(float(uv[0]), float(uv[1]))
            self.assertTrue(kf.update(t, 3.0, y_true, obs_cov_m2=cov))

        self.assertTrue(np.allclose(kf.pos_cov_2x2, kf.pos_cov_2x2.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(kf.pos_cov_2x2) > 0.0))
        self.assertGreater(kf.pos_std_m, 0.0)

        # 传入非正定/非对称/NaN/bool/嵌套bool/极大奇异协方差时应安全回退默认 R，不污染滤波状态且无警告
        bad_cov_cases = [
            np.array([[float("nan"), 0.0], [0.0, 0.1]]),
            np.array([[float("inf"), 0.0], [0.0, 0.1]]),
            np.array([[-0.1, 0.0], [0.0, 0.1]]),
            np.array([[0.1, 0.5], [0.0, 0.1]]),
            np.array([[1.0, 1e12], [-1e12, 1.0]]),
            np.eye(3),
            [[True, 0.0], [0.0, 0.1]],
            [[np.bool_(True), 0.0], [0.0, 0.1]],
            np.array([[True, 0.0], [0.0, 0.1]], dtype=object),
            [np.array([True, False]), np.array([False, True])],
            (np.array([1.0, 0.0], dtype=object), [0.0, np.bool_(False)]),
            np.array([[1e160, 2e160], [2e160, 1e160]]),
            np.array([[1e13, 0.0], [0.0, 1e13]]),
            np.array([[1e-13, 0.0], [0.0, 1e-13]]),
            np.array([[1.0, 1.0], [1.0, 1.0 + 1e-13]]),
        ]
        for bad_cov in bad_cov_cases:
            t_next = 0.4 + 0.04
            kf_clone = kf.clone()
            with self.subTest(bad_cov=repr(bad_cov)):
                self.assertTrue(kf_clone.update(t_next, 3.0, 2.0 + 5.0 * t_next, obs_cov_m2=bad_cov))
                self.assertTrue(np.isfinite(kf_clone._x).all())
                self.assertTrue(np.isfinite(kf_clone._P).all())
                self.assertTrue(np.all(np.linalg.eigvalsh(kf_clone._P) > 0.0))

        # 首帧即传入极值奇异或 object-bool 协方差也必须严格回退为 _default_R
        for bad_init_cov in (
            np.array([[True, 0.0], [0.0, 0.1]], dtype=object),
            [np.array([True, False]), np.array([False, True])],
            np.array([[1e160, 2e160], [2e160, 1e160]]),
        ):
            kf_init = ConstantAccelerationKalman2D(default_pos_std_m=0.35)
            self.assertTrue(kf_init.update(0.0, 0.0, 0.0, obs_cov_m2=bad_init_cov))
            self.assertTrue(np.allclose(kf_init.pos_cov_2x2, kf_init._default_R))
            self.assertTrue(kf_init.update(0.1, 1.0, 1.0, obs_cov_m2=bad_init_cov))
            self.assertTrue(np.all(np.linalg.eigvalsh(kf_init._P) > 0.0))

    def test_mahalanobis_outlier_gate_and_maneuver_reset(self):
        kf = ConstantAccelerationKalman2D(category="vehicle", default_pos_std_m=0.35, gate_chi2=25.0)
        for i in range(6):
            t = i * 0.1
            self.assertTrue(kf.update(t, 3.0, 6.0 * t))

        # 单帧 15m 跳变野值必须被门控拒收，位置/速度保持在真实轨迹附近
        accepted_spike = kf.update(0.6, 18.0, 18.6)
        self.assertFalse(accepted_spike)
        self.assertEqual(kf.consecutive_rejects, 1)
        self.assertAlmostEqual(kf.position_xy[0], 3.0, delta=0.1)
        self.assertAlmostEqual(kf.position_xy[1], 3.6, delta=0.1)
        self.assertAlmostEqual(kf.velocity_xy[1], 6.0, delta=0.2)

        # 下一帧恢复正常观测，顺利通过门控
        self.assertTrue(kf.update(0.7, 3.0, 4.2))
        self.assertEqual(kf.consecutive_rejects, 0)

        # 连续 3 帧大位移突变（真实急变道/重定位）：前两帧拒收，第 3 帧自动重置接管，防止锁死
        self.assertFalse(kf.update(0.8, 20.0, 10.0))
        self.assertFalse(kf.update(0.9, 20.5, 10.0))
        self.assertEqual(kf.consecutive_rejects, 2)
        self.assertTrue(kf.update(1.0, 21.0, 10.0))
        self.assertEqual(kf.consecutive_rejects, 0)
        self.assertAlmostEqual(kf.position_xy[0], 21.0, delta=0.05)
        self.assertAlmostEqual(kf.velocity_xy[0], 5.0, delta=0.2)

    def test_physical_clamping_and_validation(self):
        kf = ConstantAccelerationKalman2D(
            category="vehicle",
            default_pos_std_m=0.1,
            max_accel_ms2=6.0,
            max_speed_ms=33.3,
        )
        kf.update(0.0, 0.0, 0.0)
        # 在极小 dt 下制造极大位置跳跃触发限幅
        kf.update(0.1, 10.0, 10.0)
        vx, vy = kf.velocity_xy
        ax, ay = kf.acceleration_xy
        self.assertLessEqual(math.hypot(vx, vy), 33.3 + 1e-6)
        self.assertLessEqual(math.hypot(ax, ay), 6.0 + 1e-6)

        with self.assertRaises(ValueError):
            kf.update(0.1, 1.0, 1.0)
        with self.assertRaises(ValueError):
            kf.update(True, 1.0, 1.0)
        with self.assertRaises(ValueError):
            kf.update(0.2, float("nan"), 1.0)
        with self.assertRaises(ValueError):
            kf.predict_at(-0.1)
        with self.assertRaises(ValueError):
            ConstantAccelerationKalman2D(jerk_std_ms3=True)

    def test_extreme_coordinates_and_dt_never_overflow_or_poison_state(self):
        # 1. 预热前极值坐标 (1e308) 上溢不得触发 RuntimeWarning 或 NaN 污染状态
        kf_coord = ConstantAccelerationKalman2D()
        self.assertTrue(kf_coord.update(0.0, 0.0, 0.0))
        kf_coord.update(0.1, 1e308, 0.0)
        self.assertTrue(np.isfinite(kf_coord._x).all())
        self.assertTrue(np.isfinite(kf_coord._P).all())
        self.assertTrue(np.all(np.linalg.eigvalsh(kf_coord._P) > 0.0))
        self.assertTrue(kf_coord.update(0.2, 1.0, 0.0))
        self.assertTrue(np.isfinite(kf_coord._x).all())

        # 2. 极端时间步长 dt (1e80) 预测协方差溢出时安全重置且状态恒有限
        kf_dt = ConstantAccelerationKalman2D()
        self.assertTrue(kf_dt.update(0.0, 0.0, 0.0))
        self.assertTrue(kf_dt.update(1e80, 1.0, 1.0))
        self.assertEqual(kf_dt.position_xy, (1.0, 1.0))
        self.assertTrue(np.isfinite(kf_dt._x).all())
        self.assertTrue(np.isfinite(kf_dt._P).all())
        self.assertTrue(np.all(np.linalg.eigvalsh(kf_dt._P) > 0.0))
        with self.assertRaises(ValueError):
            kf_dt.predict_at(1e80)

        # 3. 连续拒收触发机动重置时跨正负 1e308 差值不得产生 NaN
        kf_reset = ConstantAccelerationKalman2D(min_updates_for_gate=3, max_consecutive_rejects=3)
        for i in range(4):
            self.assertTrue(kf_reset.update(i * 0.1, 0.0, 0.0))
        self.assertFalse(kf_reset.update(0.4, -1e308, 0.0))
        self.assertFalse(kf_reset.update(0.5, -1e308, 0.0))
        self.assertTrue(kf_reset.update(0.6, 1e308, 0.0))
        self.assertTrue(np.isfinite(kf_reset._x).all())
        self.assertTrue(np.isfinite(kf_reset._P).all())
        self.assertLessEqual(math.hypot(*kf_reset.velocity_xy), kf_reset.max_speed_ms + 1e-6)

        # 4. _clamp_physical_kinematics 对 1e308 分量保持方向并避免 hypot 溢出
        kf_clamp = ConstantAccelerationKalman2D(max_speed_ms=20.0, max_accel_ms2=5.0)
        kf_clamp._x[2:] = (1e308, 1e308, 1e308, -1e308)
        kf_clamp._clamp_physical_kinematics()
        self.assertTrue(np.isfinite(kf_clamp._x).all())
        self.assertAlmostEqual(kf_clamp.velocity_xy[0], kf_clamp.velocity_xy[1], places=6)
        self.assertAlmostEqual(math.hypot(*kf_clamp.velocity_xy), 20.0, places=6)
        self.assertAlmostEqual(math.hypot(*kf_clamp.acceleration_xy), 5.0, places=6)


class TestSpeedEstimatorKalmanAndTransfer(unittest.TestCase):
    def test_estimator_mode_validation_and_ema_backward_compat(self):
        for bad_mode in ("ukf", "", None, True, 123):
            with self.subTest(bad_mode=bad_mode):
                with self.assertRaises(ValueError):
                    SpeedEstimator(estimator_mode=bad_mode)
                with self.assertRaises(ValueError):
                    TrackMotionTracker(1, "vehicle", estimator_mode=bad_mode)

        ema = SpeedEstimator()
        self.assertEqual(ema.estimator_mode, "ema")
        for i in range(5):
            r = ema.update(1, "vehicle", i * 0.1, 3.0, i * 0.5)
        self.assertEqual(r.ax_ms2, 0.0)
        self.assertEqual(r.ay_ms2, 0.0)
        self.assertIsNone(r.pos_std_m)
        self.assertIsNone(r.pos_cov_m2)

    def test_kalman_mode_populates_kinematics_and_uncertainty(self):
        cal = RoadCalibrator(
            [(400.0, 300.0), (880.0, 300.0), (1150.0, 680.0), (130.0, 680.0)],
            6.0,
            15.0,
        )
        est = SpeedEstimator(
            speed_limit_kmh=30.0,
            reliable_min_span_s=0.2,
            estimator_mode="kalman",
        )
        final = None
        for i in range(15):
            t = i * 0.1
            y_true = 1.0 + 2.0 * t + 0.5 * 2.0 * t * t
            uv = cal.ground_to_pixel_checked(3.0, y_true)
            cov = cal.ground_covariance_from_pixel(float(uv[0]), float(uv[1]))
            final = est.update(
                1, "vehicle", t, 3.0, y_true, object_kind="car", obs_cov_m2=cov
            )

        self.assertIsNotNone(final)
        self.assertTrue(final.is_reliable)
        self.assertFalse(final.is_stationary)
        self.assertAlmostEqual(final.vy_ms, 2.0 + 2.0 * 1.4, delta=0.2)
        self.assertAlmostEqual(final.ay_ms2, 2.0, delta=0.35)
        self.assertIsNotNone(final.pos_std_m)
        self.assertGreater(final.pos_std_m, 0.0)
        self.assertIsNotNone(final.pos_cov_m2)
        self.assertEqual(len(final.pos_cov_m2), 4)
        self.assertAlmostEqual(final.pos_cov_m2[1], final.pos_cov_m2[2], places=10)
        # 可严格 JSON 序列化（不含 NaN / np.float64 问题）
        json.dumps(asdict(final), allow_nan=False)

    def test_kalman_mode_preserves_safety_contracts(self):
        # 1. 预热与时间门契约
        est = SpeedEstimator(stationary_min_frames=5, reliable_min_span_s=0.2, estimator_mode="kalman")
        r0 = est.update(1, "vehicle", 0.0, 3.0, 5.0)
        r1 = est.update(1, "vehicle", 0.05, 3.0, 5.0)
        self.assertFalse(r0.is_reliable)
        self.assertFalse(r1.is_reliable)

        # 2. 静止帧计数与起步即时恢复
        parked = [
            est.update(2, "vehicle", i * 0.1, 3.0, 5.0) for i in range(10)
        ]
        self.assertFalse(parked[5].is_stationary)
        self.assertTrue(parked[-1].is_stationary)
        restarted = est.update(2, "vehicle", 1.0, 3.0, 5.5)
        self.assertFalse(restarted.is_stationary)

        # 3. 微动截断与蠕行豁免防护
        est_ped = SpeedEstimator(estimator_mode="kalman")
        r_ped = None
        for i in range(8):
            r_ped = est_ped.update(3, "pedestrian", i * 0.1, 1.0, (i % 2) * 0.01)
        self.assertEqual(r_ped.speed_kmh, 0.0)

        est_crawl = SpeedEstimator(stationary_min_frames=5, estimator_mode="kalman")
        r_crawl = None
        for i in range(12):
            r_crawl = est_crawl.update(4, "vehicle", i * 0.1, 3.0, i * 0.02)
        self.assertTrue(r_crawl.is_reliable)
        self.assertFalse(r_crawl.is_stationary)

        # 4. 非有限坐标/时间戳不污染滤波器，且时间回退显式报错
        est_nan = SpeedEstimator(estimator_mode="kalman")
        for i in range(5):
            est_nan.update(5, "vehicle", i * 0.1, 3.0, i * 0.6)
        r_bad = est_nan.update(5, "vehicle", 0.5, float("nan"), float("inf"))
        self.assertFalse(r_bad.is_reliable)
        self.assertTrue(math.isfinite(r_bad.x_m) and math.isfinite(r_bad.y_m))
        r_ok = est_nan.update(5, "vehicle", 0.6, 3.0, 3.6)
        self.assertAlmostEqual(r_ok.vy_ms, 6.0, delta=0.2)

        with self.assertRaises(ValueError):
            est_nan.update(5, "vehicle", 0.6, 3.0, 4.0)

    def test_transfer_track_state(self):
        est = SpeedEstimator(estimator_mode="kalman", reliable_min_span_s=0.2)
        for i in range(6):
            r = est.update(10, "vehicle", i * 0.1, 3.0, i * 0.6, object_kind="car")
        self.assertTrue(r.is_reliable)

        self.assertFalse(est.transfer_track_state(999, 20))
        with self.assertRaises(ValueError):
            est.transfer_track_state(True, 20)
        with self.assertRaises(ValueError):
            est.transfer_track_state(10, -1)

        self.assertTrue(est.transfer_track_state(10, 42))
        self.assertNotIn(10, est.tracks)
        self.assertIn(42, est.tracks)
        self.assertEqual(est.tracks[42].track_id, 42)

        # 迁移后无缝延续速度、可信状态与车型先验，无需重新预热
        r_next = est.update(42, "vehicle", 0.6, 3.0, 3.6)
        self.assertEqual(r_next.track_id, 42)
        self.assertEqual(r_next.object_kind, "car")
        self.assertTrue(r_next.is_reliable)
        self.assertAlmostEqual(r_next.vy_ms, 6.0, delta=0.25)

        # 同 ID 迁移幂等
        self.assertTrue(est.transfer_track_state(42, 42))
        self.assertEqual(est.tracks[42].track_id, 42)

    def test_observe_category_is_atomic_and_consistent_across_modes(self):
        # ema 与 kalman 模式均在构造与 observe_category 时拒绝空白/非字符串类别且保持原子性
        for mode in ("ema", "kalman"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    TrackMotionTracker(1, "   ", estimator_mode=mode)

                tracker = TrackMotionTracker(1, "vehicle", estimator_mode=mode)
                for bad_cat in ("   ", "", None, 123):
                    with self.assertRaises(ValueError):
                        for _ in range(3):
                            tracker.observe_category(bad_cat, confirm_frames=3)
                    self.assertEqual(tracker.category, "vehicle")
                    self.assertIsNone(tracker._pending_category)
                    self.assertEqual(tracker._pending_count, 0)

        # kalman 模式下若 _kalman.set_category 抛出异常，self.category 必须保持不变
        kf_tracker = TrackMotionTracker(2, "vehicle", estimator_mode="kalman")
        with mock.patch.object(
            kf_tracker._kalman, "set_category", side_effect=RuntimeError("mock fail")
        ):
            kf_tracker.observe_category("pedestrian", confirm_frames=2)
            with self.assertRaises(RuntimeError):
                kf_tracker.observe_category("pedestrian", confirm_frames=2)
        self.assertEqual(kf_tracker.category, "vehicle")
        self.assertEqual(kf_tracker._kalman.category, "vehicle")

        # 正常确认切换应同步更新 tracker 与卡尔曼先验参数
        kf_ok = TrackMotionTracker(3, "vehicle", estimator_mode="kalman")
        kf_ok.observe_category("pedestrian", confirm_frames=2)
        self.assertEqual(kf_ok.category, "vehicle")
        kf_ok.observe_category("pedestrian", confirm_frames=2)
        self.assertEqual(kf_ok.category, "pedestrian")
        self.assertEqual(kf_ok._kalman.category, "pedestrian")
        self.assertEqual(kf_ok._kalman.jerk_std_ms3, 1.5)


if __name__ == "__main__":
    unittest.main()
