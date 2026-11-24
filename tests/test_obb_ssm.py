# -*- coding: utf-8 -*-
"""OBB 旋转矩形与多维替代安全指标 (MTTC / DRAC / PET / 碰撞概率) 单元测试。"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.conflict import ConflictAnalyzer, ConflictAssessment, RiskLevel
from analysis.policy import POLICIES, ObjectPolicy, read_object_policy_config
from analysis.schema import validate_conflict_record
from analysis.speed import SpeedReading
from analysis.ssm import (
    OBBCollisionResult,
    PETTracker,
    compute_drac,
    compute_mttc,
    compute_obb_swept_ttc,
    estimate_collision_probability,
)


def _mk_reading(
    track_id: int,
    category: str,
    x_m: float,
    y_m: float,
    vx_ms: float,
    vy_ms: float,
    *,
    object_kind: str = "",
    heading_deg: float | None = None,
    is_reliable: bool = True,
    is_stationary: bool = False,
    ax_ms2: float = 0.0,
    ay_ms2: float = 0.0,
    pos_cov_m2=None,
) -> SpeedReading:
    speed_ms = math.hypot(vx_ms, vy_ms)
    if heading_deg is None:
        heading_deg = (
            (math.degrees(math.atan2(vy_ms, vx_ms)) + 360.0) % 360.0
            if speed_ms > 1e-6
            else 0.0
        )
    reading = SpeedReading(
        track_id=track_id,
        category=category,
        x_m=x_m,
        y_m=y_m,
        vx_ms=vx_ms,
        vy_ms=vy_ms,
        speed_ms=speed_ms,
        speed_kmh=speed_ms * 3.6,
        is_speeding=speed_ms * 3.6 > 30.0,
        heading_deg=heading_deg,
        is_reliable=is_reliable,
        is_stationary=is_stationary,
        object_kind=object_kind,
    )
    reading.ax_ms2 = ax_ms2
    reading.ay_ms2 = ay_ms2
    reading.pos_cov_m2 = pos_cov_m2
    return reading


class TestPolicyDimensions(unittest.TestCase):
    def test_positional_backward_compat_and_defaults(self):
        legacy = ObjectPolicy("walker", 0.8, "行人")
        self.assertEqual((legacy.length_m, legacy.width_m), (4.5, 1.9))

        expected = {
            "person": (0.8, 0.8),
            "pedestrian": (0.8, 0.8),
            "bicycle": (1.8, 0.7),
            "motorcycle": (2.1, 0.8),
            "car": (4.5, 1.9),
            "bus": (10.0, 2.55),
            "truck": (8.0, 2.5),
            "vehicle": (4.5, 1.9),
            "motor_vehicle": (4.5, 1.9),
        }
        loaded = read_object_policy_config()
        for key, (exp_len, exp_wid) in expected.items():
            self.assertAlmostEqual(POLICIES[key].length_m, exp_len)
            self.assertAlmostEqual(POLICIES[key].width_m, exp_wid)
            self.assertAlmostEqual(loaded[key].length_m, exp_len)
            self.assertAlmostEqual(loaded[key].width_m, exp_wid)

    def test_legacy_yaml_fallback_and_invalid_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_yaml = root / "legacy.yaml"
            legacy_yaml.write_text(
                "schema_version: 1\npolicies:\n"
                "  person: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                "  pedestrian: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                "  bus: {kind: bus, safe_radius_veh_m: 2.8, label: b}\n"
                "  vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n"
                "  motor_vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n",
                encoding="utf-8",
            )
            loaded = read_object_policy_config(legacy_yaml)
            self.assertEqual((loaded["pedestrian"].length_m, loaded["pedestrian"].width_m), (0.8, 0.8))
            self.assertEqual((loaded["bus"].length_m, loaded["bus"].width_m), (10.0, 2.55))
            self.assertEqual((loaded["vehicle"].length_m, loaded["vehicle"].width_m), (4.5, 1.9))

            for bad_field in ("length_m: 0", "length_m: -1.0", "width_m: true", "width_m: .nan", "length_m: .inf"):
                bad_yaml = root / "bad_dim.yaml"
                bad_yaml.write_text(
                    "schema_version: 1\npolicies:\n"
                    "  person: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                    "  pedestrian: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                    f"  vehicle: {{kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v, {bad_field}}}\n"
                    "  motor_vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n",
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    read_object_policy_config(bad_yaml)


class TestSSMPrimitives(unittest.TestCase):
    def test_obb_swept_ttc_head_on_and_rotated(self):
        # 朝 45 度方向行驶：相对距离 10m，车长 4.5m -> hx = 2.25 + 0.4 + 0.4 = 3.05m
        theta = math.radians(45.0)
        ux, uy = math.cos(theta), math.sin(theta)
        speed = 5.0
        res = compute_obb_swept_ttc(
            veh_xy=(0.0, 0.0),
            veh_v=(speed * ux, speed * uy),
            heading_rad=theta,
            veh_length_m=4.5,
            veh_width_m=1.9,
            ped_xy=(10.0 * ux, 10.0 * uy),
            ped_v=(0.0, 0.0),
            ped_radius_m=0.4,
            horizon_s=4.0,
        )
        self.assertIsInstance(res, OBBCollisionResult)
        self.assertTrue(res.collides)
        self.assertAlmostEqual(res.t_enter, (10.0 - 3.05) / 5.0, places=6)
        self.assertAlmostEqual(res.t_exit, (10.0 + 3.05) / 5.0, places=6)
        self.assertAlmostEqual(res.min_clearance_m, 0.0, places=6)

        # 已经处于扩展包络内时 t_enter == 0.0
        inside = compute_obb_swept_ttc(
            veh_xy=(0.0, 0.0),
            veh_v=(5.0, 0.0),
            heading_rad=0.0,
            veh_length_m=4.5,
            veh_width_m=1.9,
            ped_xy=(2.9, 0.0),
            ped_v=(0.0, 0.0),
            ped_radius_m=0.4,
            horizon_s=3.5,
        )
        self.assertTrue(inside.collides)
        self.assertEqual(inside.t_enter, 0.0)
        self.assertGreater(inside.t_exit, 0.0)

    def test_obb_exact_min_clearance_matches_dense_sampling(self):
        # 倾斜角掠过车角：校验解析最小净距与高密度数值采样一致
        res = compute_obb_swept_ttc(
            veh_xy=(1.0, 2.0),
            veh_v=(4.0, 0.0),
            heading_rad=0.0,
            veh_length_m=4.5,
            veh_width_m=1.9,
            ped_xy=(8.0, 6.5),
            ped_v=(0.0, -0.8),
            ped_radius_m=0.4,
            horizon_s=3.0,
        )
        self.assertFalse(res.collides)
        hx_unm = 0.5 * 4.5 + 0.4
        hy_unm = 0.5 * 1.9 + 0.4
        ts = np.linspace(0.0, 3.0, 30001)
        px = 7.0 - 4.0 * ts
        py = 4.5 - 0.8 * ts
        dx_b = np.maximum(0.0, np.abs(px) - hx_unm)
        dy_b = np.maximum(0.0, np.abs(py) - hy_unm)
        dists = np.hypot(dx_b, dy_b)
        min_idx = int(np.argmin(dists))
        self.assertAlmostEqual(res.min_clearance_m, float(dists[min_idx]), delta=1e-4)
        self.assertAlmostEqual(res.t_min_clearance, float(ts[min_idx]), delta=1e-3)

    def test_mttc_acceleration_surge_and_braking_stop(self):
        # 匀速场景：d=15, crit=3 -> d_eff=12, v_c=6 -> 2.0s
        t_const = compute_mttc(15.0, 0.0, -6.0, 0.0, 0.0, 0.0, 3.0, 3.5)
        self.assertAlmostEqual(t_const, 2.0, places=6)

        # 加速急冲：a_c = +4.0 -> 2t^2 + 6t - 12 = 0 -> (-3 + sqrt(33))/2 ~ 1.3723s
        t_surge = compute_mttc(15.0, 0.0, -6.0, 0.0, -4.0, 0.0, 3.0, 3.5)
        self.assertIsNotNone(t_surge)
        self.assertLess(t_surge, t_const)
        self.assertAlmostEqual(t_surge, (-3.0 + math.sqrt(33.0)) / 2.0, places=6)

        # 制动刹停：v_c=6, a_c=-3 -> 刹停距离 6^2/(2*3)=6m < d_eff=12m -> None
        t_stop = compute_mttc(15.0, 0.0, -6.0, 0.0, 3.0, 0.0, 3.0, 3.5)
        self.assertIsNone(t_stop)

        # 制动力不足以在碰撞前停下：v_c=10, a_c=-2 -> 刹停距离 25m > d_eff=12m -> 返回首个碰撞根
        t_fail_brake = compute_mttc(15.0, 0.0, -10.0, 0.0, 2.0, 0.0, 3.0, 3.5)
        self.assertIsNotNone(t_fail_brake)
        self.assertAlmostEqual(t_fail_brake, 5.0 - math.sqrt(13.0), places=6)

    def test_drac_and_collision_probability_contracts(self):
        # DRAC: v_c=10, d_eff=10 -> 100/20 = 5.0
        self.assertAlmostEqual(compute_drac(13.0, 0.0, -10.0, 0.0, 3.0), 5.0, places=6)
        # 背向/零趋近速度返回 None
        self.assertIsNone(compute_drac(10.0, 0.0, 2.0, 0.0, 3.0))
        self.assertIsNone(compute_drac(10.0, 0.0, 0.0, 0.0, 3.0))
        # d_eff == 0 时分母夹紧至 0.1
        self.assertAlmostEqual(compute_drac(2.0, 0.0, -4.0, 0.0, 3.0), 80.0, places=6)

        # 碰撞概率：各向同性解析 Rayleigh CDF 对比
        t_eval = 1.0
        safe_r = 1.0
        sig2 = 2.0 * (0.25 ** 2) + (0.15 * t_eval) ** 2
        exact = 1.0 - math.exp(-(safe_r ** 2) / (2.0 * sig2))
        prob = estimate_collision_probability(0.0, 0.0, 0.0, 0.0, t_eval, safe_r)
        self.assertIsNotNone(prob)
        self.assertAlmostEqual(prob, exact, places=9)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

        # 支持 4 元组和 2x2 ndarray 协方差输入，结果确定且有限
        p_tuple = estimate_collision_probability(
            1.5, 0.0, -1.0, 0.0, 1.0, 1.5,
            pos_cov_veh_2x2=(0.4, 0.05, 0.05, 0.2),
            pos_cov_ped_2x2=np.array([[0.1, 0.0], [0.0, 0.1]]),
        )
        p_far = estimate_collision_probability(15.0, 0.0, 0.0, 0.0, 1.0, 1.5)
        self.assertIsNotNone(p_tuple)
        self.assertGreater(p_tuple, 0.5)
        self.assertLess(p_far, 1e-6)
        self.assertIsNone(estimate_collision_probability(0.0, 0.0, 0.0, 0.0, None, 1.0))

    def test_ssm_hardening_rejects_bool_and_non_finite(self):
        with self.assertRaises(ValueError):
            compute_obb_swept_ttc((0.0, 0.0), (1.0, 0.0), True, 4.5, 1.9, (5.0, 0.0), (0.0, 0.0), 0.4, 3.5)
        with self.assertRaises(ValueError):
            compute_obb_swept_ttc((0.0, float("nan")), (1.0, 0.0), 0.0, 4.5, 1.9, (5.0, 0.0), (0.0, 0.0), 0.4, 3.5)
        with self.assertRaises(ValueError):
            compute_mttc(5.0, 0.0, -1.0, 0.0, 0.0, 0.0, True, 3.5)
        self.assertIsNone(compute_mttc(float("nan"), 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 3.5))
        with self.assertRaises(ValueError):
            estimate_collision_probability(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, pos_cov_veh_2x2=(1.0, 0.0, 0.0, -2.0))
        with self.assertRaises(ValueError):
            PETTracker(cell_size_m=True)


class TestOBBSSMConflictAnalyzer(unittest.TestCase):
    def setUp(self):
        self.cvm = ConflictAnalyzer(conflict_model="cvm_circle")
        self.obb = ConflictAnalyzer(conflict_model="obb_ssm")

    def test_narrow_parallel_passing_safe_under_obb_vs_false_alarm_under_cvm(self):
        # 窄校园道路会车：汽车沿 +Y 在 x=3.0m 行驶，行人在路边 x=5.0m（横向中心距 2.0m < 3.0m 圆阈值）
        # 物理边沿净距 = 2.0 - 0.95 - 0.4 = 0.65m (> 0.3m 侧向安全裕度)
        car = _mk_reading(1, "vehicle", 3.0, 0.0, 0.0, 8.0, object_kind="car", heading_deg=90.0)
        ped = _mk_reading(2, "pedestrian", 5.0, 8.0, 0.0, 1.0, object_kind="pedestrian")

        cvm_res = self.cvm.evaluate_pair(car, ped)
        obb_res = self.obb.evaluate_pair(car, ped)
        self.assertIsNotNone(cvm_res)
        self.assertEqual(cvm_res.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertIsNone(obb_res)

        frame_res = self.obb.analyze_frame([car, ped], timestamp_s=0.0)
        self.assertEqual(len(frame_res), 1)
        self.assertEqual(frame_res[0].risk_level, RiskLevel.LEVEL_0_SAFE)
        self.assertEqual(frame_res[0].conflict_model, "obb_ssm")

    def test_head_on_and_perpendicular_crossing_trigger_critical_with_accurate_t_enter(self):
        # 正面碰撞：车 (3, 0) 以 10m/s 向北，行人 (3, 13.05) 静止
        # hx = 2.25 + 0.4 + 0.4 = 3.05m -> t_enter = (13.05 - 3.05) / 10.0 = 1.00s
        car = _mk_reading(1, "vehicle", 3.0, 0.0, 0.0, 10.0, object_kind="car", heading_deg=90.0)
        ped_front = _mk_reading(2, "pedestrian", 3.0, 13.05, 0.0, 0.0, object_kind="pedestrian")
        res_head = self.obb.evaluate_pair(car, ped_front)
        self.assertIsNotNone(res_head)
        self.assertEqual(res_head.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertAlmostEqual(res_head.ttc_seconds, 1.0, places=5)
        self.assertEqual(res_head.conflict_model, "obb_ssm")
        self.assertIsNotNone(res_head.collision_prob)
        self.assertGreaterEqual(res_head.collision_prob, 0.0)
        self.assertLessEqual(res_head.collision_prob, 1.0)
        self.assertGreater(res_head.drac_ms2, 3.4)

        # 垂直横穿：车 (3, 0) vy=10, 行人 (5, 10) vx=-2 -> 在 t=0.695s 前保险杠裕度到达 y=10
        ped_cross = _mk_reading(3, "pedestrian", 5.0, 10.0, -2.0, 0.0, object_kind="pedestrian")
        res_cross = self.obb.evaluate_pair(car, ped_cross)
        self.assertIsNotNone(res_cross)
        self.assertEqual(res_cross.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertAlmostEqual(res_cross.ttc_seconds, 0.695, places=3)

        # 严格 JSON 序列化与冲突记录格式兼容校验
        rec = {
            "schema_version": 1,
            "frame_index": 0,
            "timestamp_s": 0.0,
            "event_id": None,
            "assessment": asdict(res_cross),
        }
        roundtrip = json.loads(json.dumps(rec, allow_nan=False))
        validate_conflict_record(roundtrip)

    def test_braking_to_stop_downgrades_risk_via_mttc(self):
        # 车辆 5.5 m/s、距离 12.5m (t_enter = 1.718s <= 1.8s，等速下为 CRITICAL)
        ped = _mk_reading(2, "pedestrian", 3.0, 12.5, 0.0, 0.0, object_kind="pedestrian")
        car_const = _mk_reading(
            1, "vehicle", 3.0, 0.0, 0.0, 5.5,
            object_kind="car", heading_deg=90.0, ay_ms2=0.0,
        )
        car_braking = _mk_reading(
            1, "vehicle", 3.0, 0.0, 0.0, 5.5,
            object_kind="car", heading_deg=90.0, ay_ms2=-2.0,
        )
        res_const = self.obb.evaluate_pair(car_const, ped)
        res_brake = self.obb.evaluate_pair(car_braking, ped)
        self.assertEqual(res_const.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertIsNotNone(res_brake)
        self.assertIsNone(res_brake.mttc_seconds)
        self.assertLess(res_brake.drac_ms2, self.obb.drac_caution_ms2)
        self.assertEqual(res_brake.risk_level, RiskLevel.LEVEL_2_CAUTION)

        # 稍远距离下（等速本应 CAUTION），安全刹停应直接降级为安全 (None)
        ped_far = _mk_reading(2, "pedestrian", 3.0, 15.0, 0.0, 0.0, object_kind="pedestrian")
        self.assertEqual(self.obb.evaluate_pair(car_const, ped_far).risk_level, RiskLevel.LEVEL_2_CAUTION)
        self.assertIsNone(self.obb.evaluate_pair(car_braking, ped_far))

    def test_acceleration_surge_shortens_mttc_and_raises_risk(self):
        # 慢速行驶 (2.0 m/s) 距离 9.0m (d_eff=6.35m)：等速 TTC = 2.975s (CAUTION)，
        # 急加速 ay=+4.0 m/s^2 后 MTTC 缩短至 ~1.35s -> 升级为 LEVEL_3_CRITICAL
        ped = _mk_reading(2, "pedestrian", 3.0, 9.0, 0.0, 0.0, object_kind="pedestrian")
        car_const = _mk_reading(
            1, "vehicle", 3.0, 0.0, 0.0, 2.0,
            object_kind="car", heading_deg=90.0, ay_ms2=0.0,
        )
        car_surge = _mk_reading(
            1, "vehicle", 3.0, 0.0, 0.0, 2.0,
            object_kind="car", heading_deg=90.0, ay_ms2=4.0,
        )
        res_const = self.obb.evaluate_pair(car_const, ped)
        res_surge = self.obb.evaluate_pair(car_surge, ped)
        self.assertEqual(res_const.risk_level, RiskLevel.LEVEL_2_CAUTION)
        self.assertEqual(res_surge.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertLess(res_surge.mttc_seconds, res_const.ttc_seconds)
        self.assertAlmostEqual(res_surge.ttc_seconds, res_surge.mttc_seconds, places=6)

    def test_drac_thresholds_drive_severity(self):
        # 设小 TTC 阈值以隔离测试 DRAC 分级门槛：d_eff = 20m, v=12 m/s -> DRAC = 3.6 >= 3.4 -> CRITICAL
        analyzer = ConflictAnalyzer(
            conflict_model="obb_ssm",
            ttc_critical_s=0.5,
            ttc_caution_s=1.0,
            horizon_s=3.5,
            drac_critical_ms2=3.4,
            drac_caution_ms2=2.0,
        )
        ped_fast = _mk_reading(2, "pedestrian", 3.0, 22.65, 0.0, 0.0, object_kind="pedestrian")
        car_fast = _mk_reading(1, "vehicle", 3.0, 0.0, 0.0, 12.0, object_kind="car", heading_deg=90.0)
        res_crit = analyzer.evaluate_pair(car_fast, ped_fast)
        self.assertEqual(res_crit.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertGreaterEqual(res_crit.drac_ms2, 3.4)

        # v=10 m/s, d_eff = 20m -> DRAC = 2.5 (介于 2.0 ~ 3.4) -> CAUTION
        car_med = _mk_reading(1, "vehicle", 3.0, 0.0, 0.0, 10.0, object_kind="car", heading_deg=90.0)
        res_caut = analyzer.evaluate_pair(car_med, ped_fast)
        self.assertEqual(res_caut.risk_level, RiskLevel.LEVEL_2_CAUTION)
        self.assertAlmostEqual(res_caut.drac_ms2, 2.5, places=3)

    def test_pet_captures_just_cleared_pedestrian_near_miss(self):
        # t=0.0s: 行人经过 (3.5, 10.5)；t=0.8s: 行人已离开至 (6.5, 10.5) 且正在远离，
        # 车辆从 (3.5, 2.5) 抵达 (3.5, 10.5) 紧随擦过（于 t=0.75s 跨入 [10,11) 格，仅靠 PET 捕获险情）
        ped_t0 = _mk_reading(2, "pedestrian", 3.5, 10.5, 3.5, 0.0, object_kind="pedestrian")
        car_t0 = _mk_reading(1, "vehicle", 3.5, 2.5, 0.0, 10.0, object_kind="car", heading_deg=90.0)
        self.obb.analyze_frame([car_t0, ped_t0], timestamp_s=0.0)

        ped_t1 = _mk_reading(2, "pedestrian", 6.5, 10.5, 3.5, 0.0, object_kind="pedestrian")
        car_t1 = _mk_reading(1, "vehicle", 3.5, 10.5, 0.0, 10.0, object_kind="car", heading_deg=90.0)
        res_t1 = self.obb.analyze_frame([car_t1, ped_t1], timestamp_s=0.8)
        self.assertEqual(len(res_t1), 1)
        self.assertEqual(res_t1[0].veh_track_id, 1)
        self.assertEqual(res_t1[0].ped_track_id, 2)
        self.assertEqual(res_t1[0].risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertAlmostEqual(res_t1[0].pet_seconds, 0.75, places=4)
        self.assertIsNone(res_t1[0].ttc_seconds)

        # PET 介于 1.0s ~ 2.0s 之间应触发 LEVEL_2_CAUTION
        obb_caut = ConflictAnalyzer(conflict_model="obb_ssm")
        obb_caut.analyze_frame([ped_t0], timestamp_s=0.0)
        res_caut = obb_caut.analyze_frame([car_t1, ped_t1], timestamp_s=1.5)
        self.assertEqual(res_caut[0].risk_level, RiskLevel.LEVEL_2_CAUTION)
        self.assertAlmostEqual(res_caut[0].pet_seconds, 1.5, places=4)

    def test_analyzer_validation_and_cvm_backward_compat(self):
        for bad_kwargs in (
            {"conflict_model": "invalid"},
            {"conflict_model": True},
            {"drac_critical_ms2": 1.0, "drac_caution_ms2": 2.0},
            {"drac_caution_ms2": float("nan")},
            {"pet_critical_s": 2.5, "pet_caution_s": 1.0},
            {"pet_critical_s": True},
        ):
            with self.assertRaises(ValueError):
                ConflictAnalyzer(**bad_kwargs)

        with self.assertRaises(ValueError):
            self.obb.analyze_frame([], timestamp_s=-0.1)
        with self.assertRaises(ValueError):
            self.obb.analyze_frame([], timestamp_s=True)

        # 非有限加速度在 obb_ssm 模式下安全拒绝
        car_nan_acc = _mk_reading(
            1, "vehicle", 3.0, 0.0, 0.0, 8.0,
            object_kind="car", heading_deg=90.0, ay_ms2=float("nan"),
        )
        ped = _mk_reading(2, "pedestrian", 3.0, 8.0, 0.0, 0.0, object_kind="pedestrian")
        self.assertIsNone(self.obb.evaluate_pair(car_nan_acc, ped))

        # Positional ConflictAssessment 仍保持 100% 兼容
        legacy = ConflictAssessment(
            1, 2, RiskLevel.LEVEL_2_CAUTION, "warn", 20.0, 1.0, 8.0, 1.5, 2.0, (3.0, 6.0)
        )
        self.assertEqual(legacy.conflict_model, "cvm_circle")
        self.assertIsNone(legacy.mttc_seconds)

        # cvm_circle 保持默认且不写入 obb_ssm 衍生字段
        cvm_res = self.cvm.evaluate_pair(
            _mk_reading(1, "vehicle", 3.0, 0.0, 0.0, 10.0, object_kind="car", heading_deg=90.0),
            ped,
        )
        self.assertIsNotNone(cvm_res)
        self.assertEqual(cvm_res.conflict_model, "cvm_circle")
        self.assertIsNone(cvm_res.mttc_seconds)
        self.assertIsNone(cvm_res.drac_ms2)
        self.assertIsNone(cvm_res.collision_prob)

    def test_pet_tracker_pruning_and_vehicle_size_sensitivity(self):
        # 1. PETTracker 超出 retention_s 后清理历史占用，并豁免可靠静止车辆
        tracker = PETTracker(cell_size_m=1.0, max_pet_s=4.0, retention_s=3.0)
        ped = _mk_reading(2, "pedestrian", 2.5, 5.5, 0.0, 0.0, object_kind="pedestrian")
        parked = _mk_reading(
            3, "vehicle", 2.5, 5.5, 0.0, 0.0,
            object_kind="car", is_stationary=True, is_reliable=True,
        )
        self.assertEqual(tracker.update([ped, parked], 0.0), {})
        # 平行相距 2.0m 的车辆不触发同一 1m 网格
        far_car = _mk_reading(1, "vehicle", 4.5, 5.5, 0.0, 5.0, object_kind="car")
        self.assertEqual(tracker.update([far_car], 1.0), {})
        # 超过 retention_s=3.0 后行人旧记录已被清理
        car_late = _mk_reading(1, "vehicle", 2.5, 5.5, 0.0, 5.0, object_kind="car")
        self.assertEqual(tracker.update([car_late], 3.5), {})

        # 2. 公交车 (10.0m x 2.55m) 与小汽车 (4.5m x 1.9m) 沿纵向车距一致时，公交车更早碰撞
        ped_ahead = _mk_reading(5, "pedestrian", 3.0, 10.0, 0.0, 0.0, object_kind="pedestrian")
        bus = _mk_reading(10, "vehicle", 3.0, 0.0, 0.0, 5.0, object_kind="bus", heading_deg=90.0)
        car = _mk_reading(11, "vehicle", 3.0, 0.0, 0.0, 5.0, object_kind="car", heading_deg=90.0)
        res_bus = self.obb.evaluate_pair(bus, ped_ahead)
        res_car = self.obb.evaluate_pair(car, ped_ahead)
        self.assertIsNotNone(res_bus)
        self.assertIsNotNone(res_car)
        self.assertLess(res_bus.ttc_seconds, res_car.ttc_seconds)

        # 3. 低速 (<= 0.2 m/s) 使用速度矢量 atan2(vy, vx) 而非不可靠 heading_deg
        slow_veh = _mk_reading(
            12, "vehicle", 3.0, 0.0, 0.0, 0.15,
            object_kind="car", heading_deg=0.0, ay_ms2=3.0,
        )
        ped_close = _mk_reading(13, "pedestrian", 3.0, 3.0, 0.0, 0.0, object_kind="pedestrian")
        res_slow = self.obb.evaluate_pair(slow_veh, ped_close)
        self.assertIsNotNone(res_slow)
        self.assertEqual(res_slow.risk_level, RiskLevel.LEVEL_3_CRITICAL)

        # 4. 确认全程未导入 scipy
        import sys
        self.assertNotIn("scipy", sys.modules)


if __name__ == "__main__":
    unittest.main()
