# -*- coding: utf-8 -*-
"""2D 锚点偏置补偿、弯道 CTRV 卡尔曼滤波及弯道消融/场景单元测试。"""

from dataclasses import asdict
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from analysis.ablation import (
    main as ablation_main,
    run_all_ablations,
    run_default_ablation,
    run_turn_speed_ablation,
)
from analysis.calibration import RoadCalibrator
from analysis.kalman import (
    ConstantTurnRateKalman2D,
    _ctrv_predict_and_jacobian,
)
from analysis.scenarios import (
    ctrv_turn_scenarios,
    default_scenarios,
    main as scenarios_main,
    obb_ssm_scenarios,
    run_ctrv_turn_scenarios,
    run_obb_ssm_scenarios,
    run_scenarios,
)
from analysis.schema import load_json_strict, validate_ablation, validate_scenarios
from analysis.speed import SpeedEstimator, SpeedReading, TrackMotionTracker


def _build_pinhole_calibrator(cam_pos_xyz, dist_coeffs=None) -> RoadCalibrator:
    """用真实针孔相机外参与地面矩形角点构建单应性标定器。"""
    k_mat = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cam_pos = np.array(cam_pos_xyz, dtype=np.float64)
    target = np.array([3.0, 7.5, 0.0], dtype=np.float64)
    zc = target - cam_pos
    zc /= np.linalg.norm(zc)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    xc = np.cross(zc, up)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    yc /= np.linalg.norm(yc)
    r_mat = np.vstack([xc, yc, zc])
    t_vec = -r_mat @ cam_pos
    rvec, _ = cv2.Rodrigues(r_mat)

    ground_pts = [(0.0, 0.0), (6.0, 0.0), (6.0, 15.0), (0.0, 15.0)]
    obj = np.array([[gx, gy, 0.0] for gx, gy in ground_pts], dtype=np.float64)
    dist_np = np.array(dist_coeffs, dtype=np.float64) if dist_coeffs is not None else None
    proj, _ = cv2.projectPoints(obj, rvec, t_vec, k_mat, dist_np)
    img_pts = [(float(p[0]), float(p[1])) for p in proj.reshape(-1, 2)]
    return RoadCalibrator(
        image_points=img_pts,
        width_m=6.0,
        length_m=15.0,
        name="pinhole_cam",
        camera_matrix=k_mat.tolist(),
        dist_coeffs=dist_coeffs,
    )


class TestVehicleAnchorBiasCompensation(unittest.TestCase):
    def setUp(self):
        self.trapezoid = [
            (400.0, 300.0),
            (880.0, 300.0),
            (1150.0, 680.0),
            (130.0, 680.0),
        ]
        self.cal_trap = RoadCalibrator(
            image_points=self.trapezoid,
            width_m=6.0,
            length_m=15.0,
            name="trap_cam",
        )

    def test_estimate_camera_ground_xy_with_intrinsics_decomposition(self):
        # 无畸变与带径向畸变针孔标定下均应反算出精确光心地面投影 (cx, cy)
        cal_front = _build_pinhole_calibrator((3.0, -8.0, 5.0))
        cam_front = cal_front.estimate_camera_ground_xy()
        self.assertIsNotNone(cam_front)
        self.assertAlmostEqual(cam_front[0], 3.0, delta=1e-3)
        self.assertAlmostEqual(cam_front[1], -8.0, delta=1e-3)

        cal_dist = _build_pinhole_calibrator(
            (-2.0, -10.0, 6.5), dist_coeffs=[-0.22, 0.06, 0.001, -0.001, 0.0]
        )
        cam_dist = cal_dist.estimate_camera_ground_xy()
        self.assertIsNotNone(cam_dist)
        self.assertAlmostEqual(cam_dist[0], -2.0, delta=1e-2)
        self.assertAlmostEqual(cam_dist[1], -10.0, delta=1e-2)

    def test_estimate_camera_ground_xy_without_intrinsics_and_degenerate_safety(self):
        # 无内参时沿标定四边形底边中点向画面下方外推，应落在近端 (y > 15) 中心线附近
        cam_cw = self.cal_trap.estimate_camera_ground_xy()
        self.assertIsNotNone(cam_cw)
        self.assertAlmostEqual(cam_cw[0], 3.0, delta=0.5)
        self.assertGreater(cam_cw[1], 15.0)

        # 逆时针角点顺序把图像底边映射至 y=0，相机侧地面原点应镜像落在 y < 0（距底边同距）
        cal_ccw = RoadCalibrator(list(reversed(self.trapezoid)), 6.0, 15.0, name="ccw")
        cam_ccw = cal_ccw.estimate_camera_ground_xy()
        self.assertIsNotNone(cam_ccw)
        self.assertAlmostEqual(cam_ccw[0], cam_cw[0], places=6)
        self.assertLess(cam_ccw[1], 0.0)
        self.assertAlmostEqual(-cam_ccw[1], cam_cw[1] - 15.0, places=6)

        # 退化/非有限/奇异单应性矩阵必须安全返回 None 而不抛异常或产生警告
        cal_bad = RoadCalibrator(self.trapezoid, 6.0, 15.0, name="bad_h")
        cal_bad.H = np.full((3, 3), float("nan"), dtype=np.float64)
        self.assertIsNone(cal_bad.estimate_camera_ground_xy())

        cal_zero = RoadCalibrator(self.trapezoid, 6.0, 15.0, name="zero_h")
        cal_zero.H = np.zeros((3, 3), dtype=np.float64)
        self.assertIsNone(cal_zero.estimate_camera_ground_xy())

        # 光心贴地 (Z_c == 0) 或外参分解奇异时安全返回 None
        cal_ground_cam = _build_pinhole_calibrator((3.0, -8.0, 5.0))
        cal_ground_cam._camera_matrix_inv_np = np.zeros((3, 3), dtype=np.float64)
        self.assertIsNone(cal_ground_cam.estimate_camera_ground_xy())

    def test_compensate_vehicle_anchor_direction_projection_clamp_and_fallback(self):
        # 相机在 (3.0, -8.0)，车辆位于 (3.0, 6.0)，相机->目标视线方向为 (0, +1)
        cal_cam = _build_pinhole_calibrator((3.0, -8.0, 5.0))
        length_m, width_m = 4.5, 1.9

        # 无 heading_deg（None / NaN / bool）按 0.5 * width_m 沿视线推进
        for bad_hdg in (None, float("nan"), float("inf"), True):
            cx, cy = cal_cam.compensate_vehicle_anchor(
                3.0, 6.0, length_m, width_m, heading_deg=bad_hdg
            )
            self.assertAlmostEqual(cx, 3.0, places=4)
            self.assertAlmostEqual(cy, 6.0 + 0.5 * width_m, places=4)

        # 车头朝向与视线平行 (heading=90 deg): 推进 0.5 * length_m
        cx_l, cy_l = cal_cam.compensate_vehicle_anchor(
            3.0, 6.0, length_m, width_m, heading_deg=90.0
        )
        self.assertAlmostEqual(cx_l, 3.0, places=4)
        self.assertAlmostEqual(cy_l, 6.0 + 0.5 * length_m, places=4)

        # 车头朝向与视线垂直 (heading=0 deg): 推进 0.5 * width_m
        cx_w, cy_w = cal_cam.compensate_vehicle_anchor(
            3.0, 6.0, length_m, width_m, heading_deg=0.0
        )
        self.assertAlmostEqual(cx_w, 3.0, places=4)
        self.assertAlmostEqual(cy_w, 6.0 + 0.5 * width_m, places=4)

        # 45 度斜角：支持投影原本超过 0.5*length_m，必须被钳位不超过 0.5*max(L, W)
        cx_45, cy_45 = cal_cam.compensate_vehicle_anchor(
            3.0, 6.0, length_m, width_m, heading_deg=45.0
        )
        shift_45 = math.hypot(cx_45 - 3.0, cy_45 - 6.0)
        self.assertLessEqual(shift_45, 0.5 * max(length_m, width_m) + 1e-9)

        # 无内参梯形相机位于近端 (y > 15)，目标 (3.0, 8.0) 应向 -Y（车体纵深）补偿
        tx, ty = self.cal_trap.compensate_vehicle_anchor(
            3.0, 8.0, length_m, width_m, heading_deg=90.0
        )
        self.assertAlmostEqual(tx, 3.0, delta=0.1)
        self.assertLess(ty, 8.0)

        # 相机与目标重合或相机原点不可用时，安全原样返回 (x_m, y_m)
        cam_xy = cal_cam.estimate_camera_ground_xy()
        self.assertEqual(
            cal_cam.compensate_vehicle_anchor(cam_xy[0], cam_xy[1], length_m, width_m),
            (float(cam_xy[0]), float(cam_xy[1])),
        )
        with mock.patch.object(cal_cam, "estimate_camera_ground_xy", return_value=None):
            self.assertEqual(
                cal_cam.compensate_vehicle_anchor(3.0, 6.0, length_m, width_m, heading_deg=90.0),
                (3.0, 6.0),
            )

        # 拒绝 bool / NaN / Inf / 非正尺寸，极值 1e308 不上溢或报警告
        for args in (
            (True, 6.0, 4.5, 1.9),
            (3.0, float("nan"), 4.5, 1.9),
            (3.0, 6.0, False, 1.9),
            (3.0, 6.0, 0.0, 1.9),
            (3.0, 6.0, 4.5, -1.0),
        ):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    cal_cam.compensate_vehicle_anchor(*args)

        ex_x, ex_y = cal_cam.compensate_vehicle_anchor(1e308, 6.0, 4.5, 1.9, heading_deg=0.0)
        self.assertTrue(math.isfinite(ex_x) and math.isfinite(ex_y))

    def test_ransac_rejected_outlier_does_not_alter_camera_ground_estimation_or_anchor(self):
        inlier_img = [
            (400.0, 300.0), (880.0, 300.0), (1150.0, 680.0),
            (130.0, 680.0), (640.0, 490.0), (760.0, 560.0),
        ]
        base = RoadCalibrator(
            [(400.0, 300.0), (880.0, 300.0), (1150.0, 680.0), (130.0, 680.0)],
            6.0, 15.0,
        )
        inlier_gd = [base.pixel_to_ground(u, v) for u, v in inlier_img]
        cal_clean = RoadCalibrator.from_correspondences(inlier_img, inlier_gd, use_ransac=True)
        cal_outlier = RoadCalibrator.from_correspondences(
            inlier_img + [(5000.0, 5000.0)],
            inlier_gd + [(1.0, 1.0)],
            use_ransac=True,
        )
        self.assertEqual(cal_clean.inlier_count, 6)
        self.assertEqual(cal_outlier.inlier_count, 6)
        self.assertEqual(cal_clean.estimate_camera_ground_xy(), cal_outlier.estimate_camera_ground_xy())
        self.assertEqual(
            cal_clean.compensate_vehicle_anchor(3.0, 8.0, 4.5, 1.9, heading_deg=90.0),
            cal_outlier.compensate_vehicle_anchor(3.0, 8.0, 4.5, 1.9, heading_deg=90.0),
        )


class TestConstantTurnRateKalman2D(unittest.TestCase):
    def test_ctrv_jacobian_matches_finite_differences_across_regimes(self):
        dt = 0.04
        eps = 1e-7
        for omega in (0.0, 5e-6, -7e-6, 2e-5, -0.45, 0.75):
            state = np.array([3.0, 5.0, 8.5, 0.7, omega], dtype=np.float64)
            with self.subTest(omega=omega):
                res = _ctrv_predict_and_jacobian(state, dt)
                self.assertIsNotNone(res)
                _, jac = res
                jac_fd = np.zeros((5, 5), dtype=np.float64)
                for col in range(5):
                    sp = state.copy()
                    sp[col] += eps
                    sm = state.copy()
                    sm[col] -= eps
                    xp, _ = _ctrv_predict_and_jacobian(sp, dt)
                    xm, _ = _ctrv_predict_and_jacobian(sm, dt)
                    jac_fd[:, col] = (xp - xm) / (2.0 * eps)
                self.assertLess(float(np.max(np.abs(jac - jac_fd))), 1e-6)

    def test_straight_line_limit_and_circular_arc_convergence(self):
        # 1. 直线极限轨迹 (omega = 0)
        kf_line = ConstantTurnRateKalman2D(category="vehicle")
        vx_t, vy_t = 6.0, 2.0
        for i in range(25):
            t = i * 0.05
            self.assertTrue(kf_line.update(t, 1.0 + vx_t * t, 2.0 + vy_t * t))

        self.assertAlmostEqual(kf_line.speed_ms, math.hypot(vx_t, vy_t), delta=0.08)
        self.assertAlmostEqual(kf_line.velocity_xy[0], vx_t, delta=0.1)
        self.assertAlmostEqual(kf_line.velocity_xy[1], vy_t, delta=0.1)
        self.assertAlmostEqual(kf_line.yaw_rate_rad_s, 0.0, delta=0.03)
        self.assertAlmostEqual(kf_line.accel_xy[0], 0.0, delta=0.2)
        self.assertAlmostEqual(kf_line.accel_xy[1], 0.0, delta=0.2)

        px, py = kf_line.position_xy
        fx, fy, cov_f = kf_line.predict_at(0.5)
        self.assertAlmostEqual(fx, px + vx_t * 0.5, delta=0.15)
        self.assertAlmostEqual(fy, py + vy_t * 0.5, delta=0.15)
        self.assertTrue(np.allclose(cov_f, cov_f.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(cov_f) > 0.0))
        self.assertGreater(float(np.trace(cov_f)), float(np.trace(kf_line.pos_cov_2x2)))

        # 2. 弯道圆弧轨迹（含负向角速度与航向跨越 +/-pi 边界）
        for w_true in (0.45, -0.35):
            kf_arc = ConstantTurnRateKalman2D(category="vehicle")
            v_true = 8.0
            x_t, y_t, psi_t = 2.0, 3.0, 2.8 if w_true > 0 else -2.8
            dt = 0.04
            for i in range(60):
                t = i * dt
                if i > 0:
                    psi1 = psi_t + w_true * dt
                    x_t += (v_true / w_true) * (math.sin(psi1) - math.sin(psi_t))
                    y_t += (v_true / w_true) * (-math.cos(psi1) + math.cos(psi_t))
                    psi_t = math.atan2(math.sin(psi1), math.cos(psi1))
                self.assertTrue(kf_arc.update(t, x_t, y_t))

            self.assertAlmostEqual(kf_arc.speed_ms, v_true, delta=0.15)
            self.assertAlmostEqual(kf_arc.yaw_rate_rad_s, w_true, delta=0.04)
            self.assertGreaterEqual(kf_arc._x[3], -math.pi - 1e-12)
            self.assertLessEqual(kf_arc._x[3], math.pi + 1e-12)
            dpsi = math.atan2(
                math.sin(math.radians(kf_arc.heading_deg) - psi_t),
                math.cos(math.radians(kf_arc.heading_deg) - psi_t),
            )
            self.assertLess(abs(math.degrees(dpsi)), 2.0)

            # 向心加速度公式验证: (-v*omega*sin(psi), v*omega*cos(psi))
            ax_exp = -kf_arc.speed_ms * kf_arc.yaw_rate_rad_s * math.sin(kf_arc._x[3])
            ay_exp = kf_arc.speed_ms * kf_arc.yaw_rate_rad_s * math.cos(kf_arc._x[3])
            self.assertAlmostEqual(kf_arc.accel_xy[0], ax_exp, places=10)
            self.assertAlmostEqual(kf_arc.accel_xy[1], ay_exp, places=10)

        # clone() 深拷贝互不干扰
        cloned = kf_line.clone()
        cloned.update(25 * 0.05, 50.0, 50.0)
        self.assertAlmostEqual(kf_line.position_xy[0], px, places=9)

    def test_obs_cov_m2_mahalanobis_gate_and_maneuver_reset(self):
        cal = RoadCalibrator(
            [(400.0, 300.0), (880.0, 300.0), (1150.0, 680.0), (130.0, 680.0)],
            6.0,
            15.0,
        )
        kf = ConstantTurnRateKalman2D(category="vehicle", gate_chi2=25.0)
        for i in range(8):
            t = i * 0.05
            y_t = 2.0 + 6.0 * t
            uv = cal.ground_to_pixel_checked(3.0, y_t)
            cov = cal.ground_covariance_from_pixel(float(uv[0]), float(uv[1]))
            self.assertTrue(kf.update(t, 3.0, y_t, obs_cov_m2=cov))

        # 非正定 / bool / object / 1e160 观测协方差必须安全回退 _default_R
        for bad_cov in (
            [[True, 0.0], [0.0, 0.1]],
            np.array([[True, 0.0], [0.0, 0.1]], dtype=object),
            np.array([[1e160, 2e160], [2e160, 1e160]]),
            np.array([[-1.0, 0.0], [0.0, 1.0]]),
        ):
            kf_c = kf.clone()
            self.assertTrue(kf_c.update(0.4, 3.0, 4.4, obs_cov_m2=bad_cov))
            self.assertTrue(np.isfinite(kf_c._x).all())
            self.assertTrue(np.all(np.linalg.eigvalsh(kf_c._P) > 0.0))

        # 单帧跳变野值应被门控拒收
        self.assertFalse(kf.update(0.45, 20.0, 20.0))
        self.assertEqual(kf.consecutive_rejects, 1)
        self.assertFalse(kf.last_accepted)

        # 恢复正常点
        self.assertTrue(kf.update(0.50, 3.0, 5.0))
        self.assertEqual(kf.consecutive_rejects, 0)

        # 连续 3 帧大位移突变自动触发重置
        self.assertFalse(kf.was_reset)
        self.assertEqual(kf.reset_count, 0)
        self.assertFalse(kf.update(0.60, 25.0, 10.0))
        self.assertFalse(kf.was_reset)
        self.assertFalse(kf.update(0.70, 25.5, 10.0))
        self.assertFalse(kf.was_reset)
        self.assertTrue(kf.update(0.80, 26.0, 10.0))
        self.assertTrue(kf.was_reset)
        self.assertEqual(kf.reset_count, 1)
        self.assertEqual(kf.consecutive_rejects, 0)
        self.assertAlmostEqual(kf.position_xy[0], 26.0, delta=0.05)
        self.assertAlmostEqual(kf.speed_ms, 5.0, delta=0.2)

        # 重置后的下一次正常更新应将 was_reset 复原为 False，reset_count 保持累计值
        self.assertTrue(kf.update(0.90, 26.5, 10.0))
        self.assertFalse(kf.was_reset)
        self.assertEqual(kf.reset_count, 1)

    def test_ctrv_mahalanobis_reset_clears_reliability_and_suppresses_transient_speeding(self):
        from analysis.conflict import ConflictAnalyzer, RiskLevel

        # 1. 构造高帧率 (30fps) 已收敛 CTRV 目标，随后注入与 Sherbrooke ID 5 同构的 3 帧异常跳变
        est = SpeedEstimator(
            speed_limit_kmh=30.0,
            reliable_min_span_s=0.2,
            estimator_mode="ctrv",
        )
        analyzer = ConflictAnalyzer(speed_limit_kmh=30.0, conflict_model="obb_ssm")
        cov = np.diag([0.095 ** 2, 0.24 ** 2])
        dt = 1.0 / 30.0

        # 前 10 帧以约 5 km/h 缓慢移动并达到 is_reliable=True
        last_r = None
        for idx in range(10):
            t = idx * dt
            last_r = est.update(
                5, "vehicle", t, 5.7 + 0.03 * idx, -0.5 + 0.025 * idx,
                object_kind="car", obs_cov_m2=cov,
            )
        self.assertIsNotNone(last_r)
        self.assertTrue(last_r.is_reliable)
        kf = est.tracks[5]._kalman
        self.assertEqual(kf.reset_count, 0)

        # 帧 10-12: 连续 3 帧马氏门控异常跳变（后两帧步差 0.34m/0.033s = 36.9 km/h，加速度违背物理一致性）
        r10 = est.update(5, "vehicle", 10 * dt, 5.856, 1.350, object_kind="car", obs_cov_m2=cov)
        self.assertFalse(kf.last_accepted)
        self.assertFalse(kf.was_reset)
        r11 = est.update(5, "vehicle", 11 * dt, 5.837, 1.560, object_kind="car", obs_cov_m2=cov)
        self.assertFalse(kf.last_accepted)
        self.assertFalse(kf.was_reset)
        r12 = est.update(5, "vehicle", 12 * dt, 5.802, 1.900, object_kind="car", obs_cov_m2=cov)

        # 第 3 次拒收触发重置：was_reset=True, reset_count=1，可靠性窗口清空，不得报可信超速
        self.assertTrue(kf.was_reset)
        self.assertEqual(kf.reset_count, 1)
        self.assertFalse(r12.is_reliable)
        self.assertFalse(r12.is_speeding)
        self.assertLess(r12.speed_kmh, 30.0)
        res12 = analyzer.analyze_frame([r12], timestamp_s=12 * dt)
        self.assertEqual(len(res12), 1)
        self.assertEqual(res12[0].risk_level, RiskLevel.LEVEL_0_SAFE)

        # 重置后 5 帧内（跨度 < 0.2s）必须保持 is_reliable=False 且不触发瞬时超速；
        # 超过 0.2s 干净后验跨度后恢复 is_reliable=True 且速度收敛到真实 ~16.2 km/h (4.5 m/s)
        post_readings = []
        for k in range(1, 10):
            t = (12 + k) * dt
            r_post = est.update(
                5, "vehicle", t, 5.802, 1.900 + 4.5 * (k * dt),
                object_kind="car", obs_cov_m2=cov,
            )
            post_readings.append(r_post)
            self.assertFalse(kf.was_reset)
            if not r_post.is_reliable:
                self.assertFalse(r_post.is_speeding)
                a_post = analyzer.analyze_frame([r_post], timestamp_s=t)
                self.assertEqual(a_post[0].risk_level, RiskLevel.LEVEL_0_SAFE)

        self.assertFalse(post_readings[4].is_reliable)  # 5*dt = 0.167s < 0.2s
        self.assertTrue(post_readings[6].is_reliable)   # 7*dt = 0.233s >= 0.2s
        self.assertAlmostEqual(post_readings[-1].speed_ms, 4.5, delta=0.4)

        # 2. 即使 3 帧连续被拒观测构成一致高速重定位，重置后也必须先清零可靠性再重新累积跨度
        est_fast = SpeedEstimator(
            speed_limit_kmh=30.0,
            reliable_min_span_s=0.2,
            estimator_mode="ctrv",
        )
        for i in range(8):
            est_fast.update(9, "vehicle", i * 0.1, 0.0, 0.5 * i, object_kind="car")
        est_fast.update(9, "vehicle", 0.8, 25.0, 10.0, object_kind="car")
        est_fast.update(9, "vehicle", 0.9, 26.2, 10.0, object_kind="car")
        r_reloc = est_fast.update(9, "vehicle", 1.0, 27.4, 10.0, object_kind="car")
        self.assertTrue(est_fast.tracks[9]._kalman.was_reset)
        self.assertFalse(r_reloc.is_reliable)
        self.assertFalse(r_reloc.is_speeding)
        self.assertEqual(
            analyzer.analyze_frame([r_reloc], timestamp_s=1.0)[0].risk_level,
            RiskLevel.LEVEL_0_SAFE,
        )

    def test_extreme_values_1e308_dt_1e80_and_bool_safety(self):
        # 参数校验拒绝 bool / 非正
        for bad_kw in (
            {"accel_std_ms2": True},
            {"yaw_accel_std_rad_s2": 0.0},
            {"max_yaw_rate_rad_s": -1.0},
            {"category": "   "},
        ):
            with self.assertRaises(ValueError):
                ConstantTurnRateKalman2D(**bad_kw)

        kf = ConstantTurnRateKalman2D(max_speed_ms=25.0, max_yaw_rate_rad_s=2.0)
        self.assertTrue(kf.update(0.0, 0.0, 0.0))
        # 1e308 极值坐标不触发 RuntimeWarning 或 NaN
        kf.update(0.1, 1e308, 0.0)
        self.assertTrue(np.isfinite(kf._x).all())
        self.assertTrue(np.isfinite(kf._P).all())
        self.assertTrue(np.all(np.linalg.eigvalsh(kf._P) > 0.0))
        self.assertLessEqual(kf.speed_ms, 25.0 + 1e-6)
        self.assertLessEqual(abs(kf.yaw_rate_rad_s), 2.0 + 1e-6)

        # 极端 dt=1e80 安全重置，predict_at(1e80) 明确报错
        kf_dt = ConstantTurnRateKalman2D()
        self.assertTrue(kf_dt.update(0.0, 0.0, 0.0))
        self.assertTrue(kf_dt.update(0.1, 1.0, 0.0))
        self.assertTrue(kf_dt.update(1e80, 2.0, 0.0))
        self.assertEqual(kf_dt.position_xy, (2.0, 0.0))
        self.assertTrue(np.isfinite(kf_dt._x).all())
        self.assertTrue(np.all(np.linalg.eigvalsh(kf_dt._P) > 0.0))
        with self.assertRaises(ValueError):
            kf_dt.predict_at(1e80)

        # -1e308 与 +1e308 连续拒收重置不产生 NaN
        kf_jump = ConstantTurnRateKalman2D(min_updates_for_gate=3, max_consecutive_rejects=3)
        for i in range(4):
            kf_jump.update(i * 0.1, 0.0, 0.0)
        self.assertFalse(kf_jump.update(0.4, -1e308, 0.0))
        self.assertFalse(kf_jump.update(0.5, -1e308, 0.0))
        self.assertTrue(kf_jump.update(0.6, 1e308, 0.0))
        self.assertTrue(np.isfinite(kf_jump._x).all())
        self.assertTrue(np.isfinite(kf_jump._P).all())

        # bool / NaN / 时间回退拒绝
        with self.assertRaises(ValueError):
            kf_jump.update(True, 0.0, 0.0)
        with self.assertRaises(ValueError):
            kf_jump.update(0.7, float("nan"), 0.0)
        with self.assertRaises(ValueError):
            kf_jump.update(0.6, 0.0, 0.0)

        # 负速度自动转正并保持航向规范在 [-pi, pi]
        kf_neg = ConstantTurnRateKalman2D()
        kf_neg._x[:] = (0.0, 0.0, -5.0, 3.0 * math.pi, 10.0)
        kf_neg._clamp_ctrv_state()
        self.assertAlmostEqual(kf_neg.speed_ms, 5.0, places=9)
        self.assertGreaterEqual(kf_neg._x[3], -math.pi - 1e-12)
        self.assertLessEqual(kf_neg._x[3], math.pi + 1e-12)
        self.assertAlmostEqual(kf_neg.yaw_rate_rad_s, kf_neg.max_yaw_rate_rad_s, places=9)


class TestSpeedEstimatorCTRVAndAblationScenarios(unittest.TestCase):
    def test_speed_estimator_ctrv_mode_and_speed_reading_backward_compat(self):
        # 旧式 SpeedReading 构造保持 100% 兼容，yaw_rate_rad_s 默认 0.0
        legacy = SpeedReading(
            track_id=1, category="vehicle",
            x_m=1.0, y_m=2.0, vx_ms=0.0, vy_ms=5.0,
            speed_ms=5.0, speed_kmh=18.0, is_speeding=False,
            heading_deg=90.0, is_reliable=True,
        )
        self.assertEqual(legacy.yaw_rate_rad_s, 0.0)

        est_ctrv = SpeedEstimator(estimator_mode="ctrv", reliable_min_span_s=0.2)
        v_true = 8.0
        w_true = 0.4
        x_t, y_t, psi_t = 3.0, 2.0, 0.4
        dt = 0.04
        final = None
        for i in range(45):
            t = i * dt
            if i > 0:
                psi1 = psi_t + w_true * dt
                x_t += (v_true / w_true) * (math.sin(psi1) - math.sin(psi_t))
                y_t += (v_true / w_true) * (-math.cos(psi1) + math.cos(psi_t))
                psi_t = math.atan2(math.sin(psi1), math.cos(psi1))
            final = est_ctrv.update(1, "vehicle", t, x_t, y_t, object_kind="car")

        self.assertIsNotNone(final)
        self.assertTrue(final.is_reliable)
        self.assertFalse(final.is_stationary)
        self.assertAlmostEqual(final.speed_ms, v_true, delta=0.2)
        self.assertAlmostEqual(final.yaw_rate_rad_s, w_true, delta=0.05)
        self.assertIsNotNone(final.pos_std_m)
        self.assertIsNotNone(final.pos_cov_m2)
        json.dumps(asdict(final), allow_nan=False)

        # 非有限观测回填与类别原子切换一致
        r_nan = est_ctrv.update(1, "vehicle", 2.0, float("nan"), 0.0)
        self.assertFalse(r_nan.is_reliable)
        self.assertTrue(math.isfinite(r_nan.x_m) and math.isfinite(r_nan.y_m))
        self.assertTrue(math.isfinite(r_nan.yaw_rate_rad_s))

        tracker = TrackMotionTracker(2, "vehicle", estimator_mode="ctrv")
        tracker.observe_category("pedestrian", confirm_frames=2)
        tracker.observe_category("pedestrian", confirm_frames=2)
        self.assertEqual(tracker.category, "pedestrian")
        self.assertEqual(tracker._kalman.category, "pedestrian")

    def test_turn_speed_ablation_and_schema_compatibility(self):
        rows = run_turn_speed_ablation()
        self.assertEqual(len(rows), 12)
        for row in rows:
            for key in (
                "estimator_mode",
                "target_speed_kmh",
                "turn_rate_rad_s",
                "mean_abs_error_kmh",
                "max_abs_error_kmh",
            ):
                self.assertIn(key, row)
            self.assertIn(row["estimator_mode"], ("ema", "kalman", "ctrv"))
            self.assertIs(type(row["mean_abs_error_kmh"]), float)
            self.assertIs(type(row["max_abs_error_kmh"]), float)
            self.assertTrue(math.isfinite(row["mean_abs_error_kmh"]))
            self.assertTrue(math.isfinite(row["max_abs_error_kmh"]))

        # 按同组速度与转率对比：CTRV 平均测速误差应低于 EMA 与 CA Kalman
        grouped = {}
        for r in rows:
            grouped.setdefault((r["target_speed_kmh"], r["turn_rate_rad_s"]), {})[
                r["estimator_mode"]
            ] = r
        for key, by_mode in grouped.items():
            with self.subTest(group=key):
                self.assertLess(
                    by_mode["ctrv"]["mean_abs_error_kmh"],
                    by_mode["kalman"]["mean_abs_error_kmh"],
                )
                self.assertLess(
                    by_mode["ctrv"]["mean_abs_error_kmh"],
                    by_mode["ema"]["mean_abs_error_kmh"],
                )

        ab_def = run_default_ablation()
        ab_all = run_all_ablations()
        self.assertEqual(ab_def.keys(), ab_all.keys())
        for req_key in ("schema_version", "speed", "conflict", "algorithm_comparison", "turn_speed_ablation"):
            self.assertIn(req_key, ab_all)
        validate_ablation(ab_all)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "ablation.json"
            self.assertEqual(ablation_main(["--output", str(out_path)]), 0)
            loaded = load_json_strict(out_path)
            validate_ablation(loaded)
            self.assertIn("turn_speed_ablation", loaded)

    def test_ctrv_turn_scenarios_and_existing_ten_scenarios_preserved(self):
        self.assertEqual(len(default_scenarios()), 5)
        self.assertEqual(len(obb_ssm_scenarios()), 5)
        self.assertGreaterEqual(len(ctrv_turn_scenarios()), 2)

        self.assertTrue(all(r["passed"] for r in run_scenarios()))
        self.assertTrue(all(r["passed"] for r in run_obb_ssm_scenarios()))
        ctrv_res = run_ctrv_turn_scenarios()
        self.assertTrue(all(r["passed"] for r in ctrv_res))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "scenarios.json"
            self.assertEqual(scenarios_main(["--output", str(out_path)]), 0)
            payload = load_json_strict(out_path)
            validate_scenarios(payload)
            self.assertEqual(len(payload["results"]), 10)
            self.assertTrue(all(r["passed"] for r in payload["results"]))
            self.assertIn("ctrv_turn_results", payload)
            self.assertTrue(all(r["passed"] for r in payload["ctrv_turn_results"]))

    def test_anchor_bias_flip_between_vehicle_and_pedestrian_resets_reliability(self):
        from analysis.pipeline import ConflictPerceptionPipeline
        import csv as _csv

        cal = _build_pinhole_calibrator([0.0, -25.0, 9.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calib_path = root / "calib.yaml"
            cal.to_yaml(calib_path)

            # 构造 motorcycle 在第 0..5 帧稳定行驶并开启锚点补偿，第 6..8 帧类别切为 person（确认翻转）
            tracks_path = root / "tracks.jsonl"
            rows = []
            for idx in range(16):
                cls_name = "motorcycle" if idx < 6 else "person"
                cat = "vehicle" if idx < 6 else "pedestrian"
                gx = -0.5 + 0.15 * idx
                gy = 8.0
                # 通过 H^-1 投回真实地面角点，使得底边锚点连续移动且触发补偿偏移切换
                pt_arr = cal.H_inv @ np.array([[gx], [gy], [1.0]], dtype=np.float64)
                u = float(pt_arr[0, 0] / pt_arr[2, 0])
                v = float(pt_arr[1, 0] / pt_arr[2, 0])
                rows.append({
                    "schema_version": 1,
                    "session_id": "test_anchor_flip",
                    "frame_index": idx,
                    "timestamp_s": idx * 0.05,
                    "width": 1280,
                    "height": 720,
                    "tracks": [{
                        "track_id": 380,
                        "class_name": cls_name,
                        "category": cat,
                        "confidence": 0.9,
                        "bbox_xyxy": [u - 15.0, v - 50.0, u + 15.0, v],
                        "anchor_xy": [u, v],
                    }],
                })
            tracks_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            out_dir = root / "out"
            pipe = ConflictPerceptionPipeline(
                calibrator=RoadCalibrator.from_yaml(calib_path),
                speed_model="ctrv",
                conflict_model="obb_ssm",
                compensate_anchor_bias=True,
                include_cyclist_vru=True,
                category_confirm_frames=3,
                reliable_min_span_s=0.2,
                window_size=8,
            )
            pipe.process_tracks(tracks_path, out_dir)
            with (out_dir / "speeds.csv").open(encoding="utf-8", newline="") as f:
                csv_rows = list(_csv.DictReader(f))
            # 第 8 帧（person 第 3 帧）完成类别切换：参考点由补偿中心切回锚点，必须重置可靠度并标记为不可靠
            by_frame = {int(r["frame_idx"]): r for r in csv_rows}
            self.assertEqual(by_frame[7]["is_reliable"], "1")
            self.assertEqual(by_frame[8]["category"], "pedestrian")
            self.assertEqual(by_frame[8]["is_reliable"], "0")
            self.assertEqual(by_frame[8]["is_speeding"], "0")
            # 经过 0.2s (4*0.05s) 干净观测后重新收敛为可靠
            self.assertEqual(by_frame[13]["is_reliable"], "1")


if __name__ == "__main__":
    unittest.main()
