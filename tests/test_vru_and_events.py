# -*- coding: utf-8 -*-
"""VRU 骑行者冲突、AlertEvent SSM 峰值聚合与 Schema 兼容校验单元测试。"""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from analysis.conflict import (
    VRU_CYCLIST_KINDS,
    ConflictAnalyzer,
    ConflictAssessment,
    RiskLevel,
    build_conflict_maps,
)
from analysis.events import AlertEvent, ConflictEventTracker
from analysis.schema import (
    SchemaValidationError,
    validate_configuration,
    validate_conflict_record,
    validate_event_record,
    validate_run,
    validate_summary,
)
from analysis.speed import SpeedReading


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
) -> SpeedReading:
    speed_ms = math.hypot(vx_ms, vy_ms)
    if heading_deg is None:
        heading_deg = (
            (math.degrees(math.atan2(vy_ms, vx_ms)) + 360.0) % 360.0
            if speed_ms > 1e-6
            else 0.0
        )
    return SpeedReading(
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
        ax_ms2=ax_ms2,
        ay_ms2=ay_ms2,
    )


def _base_configuration() -> dict:
    unavailable = {"path": None, "status": "unavailable", "sha256": None, "size_bytes": None}
    return {
        "schema_version": 1,
        "weights": "weights.pt",
        "tracker_config": "tracker.yaml",
        "object_policy_config": "policy.yaml",
        "calibration": {"name": "test", "synthetic": False, "width_m": 6.0, "length_m": 15.0},
        "object_policies": {"vehicle": {"kind": "motor_vehicle", "safe_radius_veh_m": 2.2, "label": "vehicle"}},
        "fps_used": 25.0,
        "speed_limit_kmh": 30.0,
        "ema_alpha": 0.35,
        "window_size_frames": 8,
        "ttc_critical_s": 1.8,
        "ttc_caution_s": 3.5,
        "horizon_s": 3.5,
        "stationary_speed_ms": 0.1,
        "stationary_min_frames": 75,
        "alert_confirm_frames": 3,
        "alert_cooldown_frames": 12,
        "reproducibility": {
            "generated_at": "2026-11-24T12:00:00+08:00",
            "environment": {"python": "3.11.9", "packages": {}},
            "git": {"commit": None, "dirty": None, "status": "unavailable"},
            "artifacts": {
                key: dict(unavailable)
                for key in ("weights", "tracker_config", "calibration", "object_policy")
            },
        },
    }


def _base_summary() -> dict:
    return {
        "schema_version": 1,
        "status": "ok",
        "stop_reason": "eof",
        "frames_processed": 1,
        "elapsed_seconds": 0.1,
        "annotated_output": "a.mp4",
        "conflicts_jsonl": "conflicts.jsonl",
        "events_jsonl": "events.jsonl",
        "speeds_csv": "speeds.csv",
        "configuration_json": "configuration.json",
        "total_alert_events": 0,
        "total_speeding_alerts": 0,
        "total_caution_events": 0,
        "total_critical_conflicts": 0,
    }


class TestCyclistVRUConflict(unittest.TestCase):
    def test_init_validation_and_backward_compat(self):
        # 校验 VRU 常量定义与构造开关严格布尔类型检查。
        self.assertEqual(VRU_CYCLIST_KINDS, ("bicycle", "motorcycle"))
        for bad_flag in (1, 0, "true", "false", None, np.bool_(True)):
            with self.subTest(bad_flag=bad_flag):
                with self.assertRaises(ValueError):
                    ConflictAnalyzer(include_cyclist_vru=bad_flag)  # type: ignore[arg-type]

        # 验证 ConflictAssessment 位置构造向后兼容，新增字段位于末尾并默认 pedestrian。
        legacy = ConflictAssessment(
            1, 2, RiskLevel.LEVEL_2_CAUTION, "warn", 20.0, 1.0, 8.0, 1.5, 2.0, (3.0, 6.0)
        )
        self.assertEqual(legacy.target_kind, "pedestrian")

    def test_disabled_by_default_does_not_alert_car_vs_cyclists(self):
        # 默认关闭 include_cyclist_vru 时，汽车与自行车/摩托车均不触发冲突告警。
        car = _mk_reading(1, "vehicle", 3.0, 0.0, 0.0, 8.0, object_kind="car", heading_deg=90.0)
        bike = _mk_reading(2, "vehicle", 3.0, 8.0, 0.0, 0.0, object_kind="bicycle", heading_deg=0.0)
        moto = _mk_reading(3, "vehicle", 3.0, 10.0, 0.0, 0.0, object_kind="motorcycle", heading_deg=0.0)

        for model in ("cvm_circle", "obb_ssm"):
            analyzer = ConflictAnalyzer(conflict_model=model)
            self.assertFalse(analyzer.include_cyclist_vru)
            res = analyzer.analyze_frame([car, bike, moto], timestamp_s=0.0)
            self.assertEqual(len(res), 3)
            self.assertTrue(all(item.risk_level == RiskLevel.LEVEL_0_SAFE for item in res))
            self.assertTrue(all(item.ped_track_id is None for item in res))

    def test_cvm_circle_car_vs_bicycle_and_motorcycle_and_no_car_car_dup(self):
        analyzer = ConflictAnalyzer(conflict_model="cvm_circle", include_cyclist_vru=True)
        car = _mk_reading(1, "vehicle", 3.0, 0.0, 0.0, 8.0, object_kind="car", heading_deg=90.0)
        car2 = _mk_reading(2, "vehicle", 3.0, 10.0, 0.0, -5.0, object_kind="car", heading_deg=270.0)
        bike = _mk_reading(3, "vehicle", 6.2, 8.0, 0.0, 0.0, object_kind="bicycle", heading_deg=0.0)
        moto = _mk_reading(4, "vehicle", 6.8, 16.0, 0.0, 0.0, object_kind="motorcycle", heading_deg=0.0)

        # 汽车 vs 汽车不得互相配对告警。
        self.assertEqual(
            [a.risk_level for a in analyzer.analyze_frame([car, car2])],
            [RiskLevel.LEVEL_0_SAFE, RiskLevel.LEVEL_0_SAFE],
        )

        # 横向偏置 3.2m：行人半径 0.8 -> 阈值 3.0m 仅触发 1.5x 横穿注意；
        # 自行车半径 1.2 -> 阈值 3.4m 且 t_cpa=1.0s，直接升级为 CRITICAL 并称呼“骑行者”。
        res_bike = analyzer.evaluate_pair(car, bike)
        self.assertIsNotNone(res_bike)
        self.assertEqual(res_bike.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertEqual(res_bike.target_kind, "bicycle")
        self.assertIn("骑行者", res_bike.risk_desc)

        # 摩托车半径 1.8 -> 阈值 4.0m，横向偏置 3.8m 且 t_cpa=2.0s -> CAUTION 且称呼“骑行者”。
        res_moto = analyzer.evaluate_pair(car, moto)
        self.assertIsNotNone(res_moto)
        self.assertEqual(res_moto.risk_level, RiskLevel.LEVEL_2_CAUTION)
        self.assertEqual(res_moto.target_kind, "motorcycle")
        self.assertIn("骑行者", res_moto.risk_desc)

        # analyze_frame 对 car+bike 仅产生 1 条冲突（car->bike），骑行者不反向对车重复报警。
        frame_res = analyzer.analyze_frame([car, bike])
        conflicts = [a for a in frame_res if a.risk_level > RiskLevel.LEVEL_0_SAFE]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual((conflicts[0].veh_track_id, conflicts[0].ped_track_id), (1, 3))
        self.assertEqual(conflicts[0].target_kind, "bicycle")

        # 验证 build_conflict_maps 让处于危险中的骑行者保持高风险而非被车辆侧 LEVEL_0_SAFE 遮蔽。
        veh_map, ped_map = build_conflict_maps(frame_res)
        self.assertEqual((veh_map.get(3) or ped_map.get(3)).risk_level, RiskLevel.LEVEL_3_CRITICAL)

        # 常规行人继续保持 target_kind="pedestrian" 与“行人”称呼不变（t_cpa ~ 2.2s -> LEVEL_2_CAUTION）。
        ped = _mk_reading(5, "pedestrian", 5.0, 18.0, -1.0, 0.0, object_kind="pedestrian")
        res_ped = analyzer.evaluate_pair(car, ped)
        self.assertIsNotNone(res_ped)
        self.assertEqual(res_ped.risk_level, RiskLevel.LEVEL_2_CAUTION)
        self.assertEqual(res_ped.target_kind, "pedestrian")
        self.assertIn("行人", res_ped.risk_desc)
        self.assertNotIn("骑行者", res_ped.risk_desc)

    def test_obb_ssm_car_vs_bicycle_and_motorcycle_radii_and_cyclist_vs_ped_only(self):
        analyzer = ConflictAnalyzer(conflict_model="obb_ssm", include_cyclist_vru=True)
        car = _mk_reading(1, "vehicle", 3.0, 0.0, 0.0, 10.0, object_kind="car", heading_deg=90.0)
        # bicycle: 0.5 * max(1.8, 0.7) = 0.9m -> hx = 2.25 + 0.9 + 0.4 = 3.55m -> t_enter = (13.55-3.55)/10 = 1.0s
        bike = _mk_reading(2, "vehicle", 3.0, 13.55, 0.0, 0.0, object_kind="bicycle", heading_deg=0.0)
        res_bike = analyzer.evaluate_pair(car, bike)
        self.assertIsNotNone(res_bike)
        self.assertEqual(res_bike.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertAlmostEqual(res_bike.ttc_seconds, 1.0, places=5)
        self.assertEqual(res_bike.target_kind, "bicycle")
        self.assertIn("骑行者", res_bike.risk_desc)
        self.assertIsNotNone(res_bike.drac_ms2)
        self.assertIsNotNone(res_bike.collision_prob)

        # motorcycle: 0.5 * max(2.1, 0.8) = 1.05m -> hx = 2.25 + 1.05 + 0.4 = 3.70m -> t_enter = (13.70-3.70)/10 = 1.0s
        moto = _mk_reading(3, "vehicle", 3.0, 13.70, 0.0, 0.0, object_kind="motorcycle", heading_deg=0.0)
        res_moto = analyzer.evaluate_pair(car, moto)
        self.assertIsNotNone(res_moto)
        self.assertEqual(res_moto.risk_level, RiskLevel.LEVEL_3_CRITICAL)
        self.assertAlmostEqual(res_moto.ttc_seconds, 1.0, places=5)
        self.assertEqual(res_moto.target_kind, "motorcycle")
        self.assertIn("骑行者", res_moto.risk_desc)

        # 骑行者与骑行者不互判，骑行者仍能正常预警前方行人。
        bike2 = _mk_reading(4, "vehicle", 3.0, 6.0, 0.0, 5.0, object_kind="bicycle", heading_deg=90.0)
        ped = _mk_reading(5, "pedestrian", 3.0, 12.0, 0.0, 0.0, object_kind="pedestrian")
        res_cyc_pair = analyzer.analyze_frame([bike, bike2], timestamp_s=0.0)
        self.assertTrue(all(a.risk_level == RiskLevel.LEVEL_0_SAFE for a in res_cyc_pair))

        res_cyc_ped = analyzer.analyze_frame([bike2, ped], timestamp_s=0.1)
        self.assertEqual(len(res_cyc_ped), 1)
        self.assertEqual((res_cyc_ped[0].veh_track_id, res_cyc_ped[0].ped_track_id), (4, 5))
        self.assertEqual(res_cyc_ped[0].vehicle_kind, "bicycle")
        self.assertEqual(res_cyc_ped[0].target_kind, "pedestrian")

    def test_obb_pet_clean_with_cyclists_and_non_finite_readings_skipped(self):
        analyzer = ConflictAnalyzer(conflict_model="obb_ssm", include_cyclist_vru=True)
        # 含非有限字段（heading_deg/ax_ms2/speed_kmh 为 NaN 或 bool）的目标必须被过滤，不崩溃不落盘 NaN。
        bad_car = _mk_reading(10, "vehicle", 3.0, 0.0, 0.0, 14.0, object_kind="car", heading_deg=float("nan"))
        bad_bike = _mk_reading(11, "vehicle", 3.0, 5.0, 0.0, 0.0, object_kind="bicycle", ax_ms2=float("inf"))
        bad_ped = _mk_reading(12, "pedestrian", 3.0, 8.0, 0.0, 0.0, object_kind="pedestrian", ay_ms2=float("nan"))
        self.assertEqual(analyzer.analyze_frame([bad_car, bad_bike, bad_ped], timestamp_s=0.0), [])

        # PETTracker 在汽车+骑行者+行人同场时应干净工作，不因骑行者出现而崩溃或产生 car-bike 伪 PET。
        analyzer.reset()
        ped_t0 = _mk_reading(2, "pedestrian", 3.5, 10.5, 3.5, 0.0, object_kind="pedestrian")
        car_t0 = _mk_reading(1, "vehicle", 3.5, 2.5, 0.0, 10.0, object_kind="car", heading_deg=90.0)
        bike_t0 = _mk_reading(3, "vehicle", 3.5, 10.5, 6.25, 0.0, object_kind="bicycle", heading_deg=0.0)
        analyzer.analyze_frame([car_t0, bike_t0, ped_t0], timestamp_s=0.0)

        ped_t1 = _mk_reading(2, "pedestrian", 6.5, 10.5, 3.5, 0.0, object_kind="pedestrian")
        car_t1 = _mk_reading(1, "vehicle", 3.5, 10.5, 0.0, 10.0, object_kind="car", heading_deg=90.0)
        bike_t1 = _mk_reading(3, "vehicle", 8.5, 10.5, 6.25, 0.0, object_kind="bicycle", heading_deg=0.0)
        res_t1 = analyzer.analyze_frame([car_t1, bike_t1, ped_t1], timestamp_s=0.8)
        car_res = [r for r in res_t1 if r.veh_track_id == 1][0]
        self.assertEqual(car_res.ped_track_id, 2)
        self.assertEqual(car_res.target_kind, "pedestrian")
        self.assertAlmostEqual(car_res.pet_seconds, 0.75, places=4)

        # 仅汽车与自行车依次经过同一网格（无行人）时，PETTracker 不应错误生成汽车-自行车 PET。
        analyzer.reset()
        analyzer.analyze_frame([car_t0, bike_t0], timestamp_s=0.0)
        res_no_ped = analyzer.analyze_frame([car_t1, bike_t1], timestamp_s=0.8)
        car_only_res = [r for r in res_no_ped if r.veh_track_id == 1][0]
        self.assertEqual(car_only_res.risk_level, RiskLevel.LEVEL_1_SPEEDING)
        self.assertIsNone(car_only_res.pet_seconds)


class TestAlertEventSSMAggregation(unittest.TestCase):
    def _mk_conflict(
        self,
        *,
        veh: int = 1,
        ped: int | None = 2,
        level: RiskLevel = RiskLevel.LEVEL_2_CAUTION,
        ttc: float | None = 2.0,
        mttc: float | None = None,
        drac: float | None = None,
        pet: float | None = None,
        prob: float | None = None,
        target_kind: str = "pedestrian",
    ) -> ConflictAssessment:
        return ConflictAssessment(
            veh_track_id=veh,
            ped_track_id=ped,
            risk_level=level,
            risk_desc="x",
            veh_speed_kmh=25.0,
            ped_speed_ms=1.5 if ped is not None else 0.0,
            current_distance_m=6.0,
            min_distance_m=0.5,
            ttc_seconds=ttc,
            conflict_ground_xy=(3.0, 8.0) if ped is not None else None,
            mttc_seconds=mttc,
            drac_ms2=drac,
            pet_seconds=pet,
            collision_prob=prob,
            conflict_model="obb_ssm",
            target_kind=target_kind,
        )

    def test_aggregates_ssm_peaks_across_pending_active_and_cooldown(self):
        # 覆盖挂起确认帧、存续告警帧以及冷却吸收后恢复帧的 4 项 SSM 极值聚合与 to_dict 落盘。
        tracker = ConflictEventTracker(confirm_frames=2, cooldown_frames=3)

        # 挂起第 1 帧 (frame 0) 与确认第 2 帧 (frame 1)
        tracker.update(
            [
                self._mk_conflict(
                    level=RiskLevel.LEVEL_2_CAUTION,
                    ttc=2.4,
                    mttc=2.1234,
                    drac=2.1,
                    pet=1.4567,
                    prob=0.31234,
                    target_kind="bicycle",
                )
            ],
            0,
            0.0,
        )
        ann = tracker.update(
            [
                self._mk_conflict(
                    level=RiskLevel.LEVEL_3_CRITICAL,
                    ttc=1.3,
                    mttc=1.0556,
                    drac=4.8129,
                    pet=None,
                    prob=0.84567,
                    target_kind="bicycle",
                )
            ],
            1,
            0.1,
        )
        self.assertEqual(ann, {(1, 2): 1})

        # 冷却窗内静默 2 帧后再次出现更小 PET 与更高概率 (frame 4)
        tracker.update([], 2, 0.2)
        tracker.update([], 3, 0.3)
        tracker.update(
            [
                self._mk_conflict(
                    level=RiskLevel.LEVEL_2_CAUTION,
                    ttc=2.0,
                    mttc=float("nan"),  # 非有限字段应被忽略，不覆盖既有极值
                    drac=3.0,
                    pet=0.4126,
                    prob=0.91236,
                    target_kind="bicycle",
                )
            ],
            4,
            0.4,
        )
        tracker.flush()
        (ev,) = tracker.pop_finalized()

        self.assertAlmostEqual(ev.min_mttc_seconds, 1.0556, places=4)
        self.assertAlmostEqual(ev.max_drac_ms2, 4.8129, places=4)
        self.assertAlmostEqual(ev.min_pet_seconds, 0.4126, places=4)
        self.assertAlmostEqual(ev.max_collision_prob, 0.91236, places=5)
        self.assertEqual(ev.target_kind, "bicycle")

        d = ev.to_dict()
        serialized = json.dumps(d, ensure_ascii=False, allow_nan=False)
        roundtrip = json.loads(serialized)
        self.assertEqual(roundtrip["min_mttc_seconds"], 1.056)
        self.assertEqual(roundtrip["max_drac_ms2"], 4.813)
        self.assertEqual(roundtrip["min_pet_seconds"], 0.413)
        self.assertEqual(roundtrip["max_collision_prob"], 0.9124)
        self.assertEqual(roundtrip["target_kind"], "bicycle")
        validate_event_record(roundtrip)

    def test_interrupted_pending_resets_ssm_peaks_and_none_preserved(self):
        # 挂起中断后需彻底重置极值；全 None SSM 时 to_dict 保留 None 且位置构造兼容。
        legacy_ev = AlertEvent(
            event_id=1,
            veh_track_id=1,
            ped_track_id=None,
            start_frame=0,
            start_timestamp_s=0.0,
            end_frame=1,
            end_timestamp_s=0.1,
            peak_risk_level=RiskLevel.LEVEL_1_SPEEDING,
            min_ttc_seconds=None,
            max_veh_speed_kmh=35.0,
            alarm_frames=2,
        )
        self.assertEqual(legacy_ev.target_kind, "pedestrian")
        self.assertIsNone(legacy_ev.min_mttc_seconds)

        tracker = ConflictEventTracker(confirm_frames=2, cooldown_frames=1)
        tracker.update(
            [self._mk_conflict(mttc=0.5, drac=8.0, pet=0.2, prob=0.99, target_kind="motorcycle")],
            0,
            0.0,
        )
        tracker.update([], 1, 0.1)  # 中断挂起，清空候选统计
        tracker.update([self._mk_conflict(level=RiskLevel.LEVEL_1_SPEEDING, ped=None, ttc=None)], 2, 0.2)
        tracker.update([self._mk_conflict(level=RiskLevel.LEVEL_1_SPEEDING, ped=None, ttc=None)], 3, 0.3)
        tracker.flush()
        (ev,) = tracker.pop_finalized()
        d = ev.to_dict()
        self.assertIsNone(d["min_mttc_seconds"])
        self.assertIsNone(d["max_drac_ms2"])
        self.assertIsNone(d["min_pet_seconds"])
        self.assertIsNone(d["max_collision_prob"])
        self.assertEqual(d["target_kind"], "pedestrian")
        validate_event_record(d)


class TestSchemaVruAndEventCompatibility(unittest.TestCase):
    def test_conflict_record_target_kind_validation_and_backward_compat(self):
        # 旧版无 target_kind 的冲突记录必须 100% 兼容。
        legacy = {
            "schema_version": 1,
            "frame_index": 0,
            "timestamp_s": 0.0,
            "event_id": None,
            "assessment": {
                "veh_track_id": 1,
                "ped_track_id": 2,
                "risk_level": 2,
                "ttc_seconds": 2.0,
                "vehicle_kind": "car",
            },
        }
        validate_conflict_record(legacy)

        for ok_kind in (None, "pedestrian", "bicycle", "motorcycle"):
            rec = json.loads(json.dumps(legacy))
            rec["assessment"]["target_kind"] = ok_kind
            validate_conflict_record(rec)

        for bad_kind in ("", "   ", 123, True):
            rec = json.loads(json.dumps(legacy))
            rec["assessment"]["target_kind"] = bad_kind
            with self.assertRaises(SchemaValidationError):
                validate_conflict_record(rec)

    def test_event_record_new_fields_validation_and_backward_compat(self):
        legacy_event = {
            "schema_version": 1,
            "event_id": 1,
            "kind": "conflict",
            "veh_track_id": 1,
            "ped_track_id": 2,
            "start_frame": 0,
            "end_frame": 2,
            "start_timestamp_s": 0.0,
            "end_timestamp_s": 0.08,
            "peak_risk_level": 3,
            "min_ttc_seconds": 1.2,
            "end_reason": "eof",
        }
        validate_event_record(legacy_event)

        valid_new = dict(legacy_event)
        valid_new.update({
            "target_kind": "bicycle",
            "min_mttc_seconds": 0.95,
            "max_drac_ms2": 4.2,
            "min_pet_seconds": 0.0,
            "max_collision_prob": 0.85,
        })
        validate_event_record(valid_new)

        valid_nulls = dict(legacy_event)
        valid_nulls.update({
            "target_kind": "pedestrian",
            "min_mttc_seconds": None,
            "max_drac_ms2": None,
            "min_pet_seconds": None,
            "max_collision_prob": None,
        })
        validate_event_record(valid_nulls)

        invalid_mutations = [
            {"target_kind": None},
            {"target_kind": ""},
            {"target_kind": "   "},
            {"target_kind": True},
            {"min_mttc_seconds": -0.1},
            {"min_mttc_seconds": True},
            {"max_drac_ms2": -1.0},
            {"min_pet_seconds": -0.01},
            {"max_collision_prob": -0.01},
            {"max_collision_prob": 1.01},
            {"max_collision_prob": False},
        ]
        for patch in invalid_mutations:
            bad = dict(valid_new)
            bad.update(patch)
            with self.subTest(patch=patch):
                with self.assertRaises(SchemaValidationError):
                    validate_event_record(bad)

    def test_configuration_and_summary_optional_fields_and_run_roundtrip(self):
        cfg = _base_configuration()
        validate_configuration(cfg)
        cfg.update({
            "include_cyclist_vru": True,
            "live_reconnect_attempts": 3,
            "live_reconnect_delay_s": 1.5,
        })
        validate_configuration(cfg)

        for bad_cfg in (
            {"include_cyclist_vru": 1},
            {"include_cyclist_vru": "true"},
            {"live_reconnect_attempts": -1},
            {"live_reconnect_attempts": True},
            {"live_reconnect_attempts": 1.5},
            {"live_reconnect_delay_s": -0.1},
            {"live_reconnect_delay_s": True},
            {"live_reconnect_delay_s": float("nan")},
        ):
            mutated = dict(cfg)
            mutated.update(bad_cfg)
            with self.subTest(bad_cfg=bad_cfg):
                with self.assertRaises(SchemaValidationError):
                    validate_configuration(mutated)

        summary = _base_summary()
        validate_summary(summary)
        summary["live_reconnect_count"] = 2
        validate_summary(summary)
        for bad_cnt in (-1, True, 1.5, "2"):
            mutated_s = dict(summary)
            mutated_s["live_reconnect_count"] = bad_cnt
            with self.subTest(bad_cnt=bad_cnt):
                with self.assertRaises(SchemaValidationError):
                    validate_summary(mutated_s)

        # 端到端构造完整运行目录并调用 validate_run 校验落盘结果。
        analyzer = ConflictAnalyzer(conflict_model="obb_ssm", include_cyclist_vru=True)
        ev_tracker = ConflictEventTracker(confirm_frames=1, cooldown_frames=1)
        car = _mk_reading(1, "vehicle", 3.0, 0.0, 0.0, 10.0, object_kind="car", heading_deg=90.0)
        bike = _mk_reading(2, "vehicle", 3.0, 12.0, 0.0, 0.0, object_kind="bicycle", heading_deg=0.0)
        frame_conflicts = analyzer.analyze_frame([car, bike], timestamp_s=0.0)
        event_ids = ev_tracker.update(frame_conflicts, 0, 0.0)
        ev_tracker.flush()
        final_events = ev_tracker.pop_finalized()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary["total_alert_events"] = len(final_events)
            summary["total_critical_conflicts"] = len(final_events)
            (root / "summary.json").write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
            (root / "configuration.json").write_text(json.dumps(cfg, allow_nan=False), encoding="utf-8")
            (root / "conflicts.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "frame_index": 0,
                            "timestamp_s": 0.0,
                            "event_id": event_ids.get((c.veh_track_id, c.ped_track_id)),
                            "assessment": asdict(c),
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    for c in frame_conflicts
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "events.jsonl").write_text(
                "\n".join(json.dumps(ev.to_dict(), ensure_ascii=False, allow_nan=False) for ev in final_events) + "\n",
                encoding="utf-8",
            )
            with (root / "speeds.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted({
                    "schema_version", "frame_idx", "timestamp_s", "track_id", "category",
                    "speed_kmh", "risk_level", "event_id", "is_reliable", "is_stationary",
                }))
                writer.writeheader()
                writer.writerow({
                    "schema_version": 1, "frame_idx": 0, "timestamp_s": 0.0, "track_id": 1,
                    "category": "vehicle", "speed_kmh": 36.0, "risk_level": "LEVEL_3_CRITICAL",
                    "event_id": 1, "is_reliable": 1, "is_stationary": 0,
                })
            res = validate_run(root)
            self.assertTrue(res["valid"])
            self.assertEqual(res["counts"]["events"], 1)


if __name__ == "__main__":
    unittest.main()
