# -*- coding: utf-8 -*-
"""短时遮挡断轨缝合、端到端 Kalman+OBB-SSM+Stitching 复盘与消融/场景集成测试。"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from analysis import (
    ConstantAccelerationKalman2D,
    OBBCollisionResult,
    PETTracker,
    RoadCalibrator,
    StitchEvent,
    TrackObservation,
    TrackletStitcher,
)
from analysis.__main__ import main as analysis_main
from analysis.ablation import main as ablation_main, run_algorithm_comparison, run_default_ablation
from analysis.conflict import RiskLevel
from analysis.pipeline import ConflictPerceptionPipeline
from analysis.scenarios import (
    default_scenarios,
    main as scenarios_main,
    obb_ssm_scenarios,
    run_obb_ssm_scenarios,
    run_scenarios,
)
from analysis.schema import (
    SchemaValidationError,
    load_json_strict,
    load_jsonl_strict,
    validate_ablation,
    validate_configuration,
    validate_conflict_record,
    validate_run,
    validate_scenarios,
    validate_summary,
)


def _perspective_calibrator() -> RoadCalibrator:
    return RoadCalibrator(
        image_points=[(300.0, 120.0), (700.0, 120.0), (920.0, 560.0), (80.0, 560.0)],
        width_m=6.0,
        length_m=24.0,
        name="stitch_integration_road",
    )


def _project_ground_to_pixel(calib: RoadCalibrator, x_m: float, y_m: float) -> tuple[float, float]:
    p = calib.H_inv @ np.array([x_m, y_m, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


def _make_track_target(
    track_id: int,
    u: float,
    v: float,
    *,
    class_name: str = "car",
    category: str = "vehicle",
) -> dict:
    return {
        "track_id": track_id,
        "class_id": 2 if category == "vehicle" else 0,
        "class_name": class_name,
        "category": category,
        "confidence": 0.92,
        "bbox_xyxy": [u - 18.0, v - 36.0, u + 18.0, v],
        "anchor_xy": [u, v],
        "anchor_type": "bbox_bottom_center",
        "observation": "detected",
    }


class TestTrackletStitcher(unittest.TestCase):
    def test_parameter_and_observation_validation(self):
        for bad_kw in (
            {"max_gap_s": 0.0},
            {"max_gap_s": -1.0},
            {"max_gap_s": True},
            {"max_gap_s": float("nan")},
            {"max_distance_m": 0.0},
            {"max_distance_m": float("inf")},
            {"max_mahalanobis": False},
            {"max_heading_diff_deg": -5.0},
            {"velocity_noise_ms": 0.0},
            {"max_lost_pool": 0},
            {"max_lost_pool": True},
        ):
            with self.assertRaises(ValueError, msg=str(bad_kw)):
                TrackletStitcher(**bad_kw)

        st = TrackletStitcher()
        with self.assertRaises(ValueError):
            st.resolve_frame([], True)
        with self.assertRaises(ValueError):
            st.resolve_frame([], float("nan"))
        st.resolve_frame([], 1.0)
        with self.assertRaises(ValueError):
            st.resolve_frame([], 0.5)
        with self.assertRaises(ValueError):
            st.resolve_frame([(True, "vehicle", "car", 1.0, 2.0)], 1.1)
        with self.assertRaises(ValueError):
            st.resolve_frame([(1, "vehicle", "car", True, 2.0)], 1.1)
        with self.assertRaises(ValueError):
            st.resolve_frame(
                [(1, "vehicle", "car", 1.0, 2.0), (1, "vehicle", "car", 2.0, 2.0)],
                1.1,
            )

        # 非有限地面坐标观测安全跳过，不崩溃
        id_map, pairs = st.resolve_frame([(2, "vehicle", "car", float("nan"), 3.0)], 1.2)
        self.assertEqual(id_map, {})
        self.assertEqual(pairs, [])

    def test_occlusion_recovery_and_multihop_stitching(self):
        st = TrackletStitcher(max_gap_s=1.5, max_distance_m=3.5, max_mahalanobis=3.5)
        # 支持多种观测结构：TrackObservation、元组、SimpleNamespace
        obs0 = [
            TrackObservation(1, "vehicle", "car", 3.0, 0.0),
            (10, "pedestrian", "person", 5.5, 10.0),
        ]
        id_map, pairs = st.resolve_frame(obs0, 0.0)
        self.assertEqual(id_map, {1: 1, 10: 10})
        self.assertEqual(pairs, [])
        st.update_kinematics(
            1,
            SimpleNamespace(
                x_m=3.0, y_m=0.0, vx_ms=0.0, vy_ms=8.0,
                ax_ms2=0.0, ay_ms2=0.0, is_reliable=True,
                category="vehicle", object_kind="car",
                pos_cov_m2=(0.04, 0.0, 0.0, 0.16),
            ),
            0.0,
        )

        # 遮挡 0.6s（车辆缺席，行人继续可见）
        id_map, pairs = st.resolve_frame([SimpleNamespace(raw_track_id=10, category="pedestrian", object_kind="person", x_m=5.5, y_m=10.3)], 0.3)
        self.assertEqual(id_map, {10: 10})
        self.assertEqual(st.lost_count, 1)

        # t=0.6s 车辆以 raw_id=5 重新出现于外推点 (3.0, 4.8) 附近
        id_map, pairs = st.resolve_frame(
            [
                (5, "vehicle", "car", 3.1, 4.9),
                (10, "pedestrian", "person", 5.5, 10.6),
            ],
            0.6,
        )
        self.assertEqual(id_map, {5: 1, 10: 10})
        self.assertEqual(pairs, [(5, 1)])
        self.assertEqual(st.stitched_track_count, 1)
        self.assertIsInstance(st.stitch_events[0], StitchEvent)
        self.assertEqual(st.stitch_events[0].canonical_id, 1)
        self.assertEqual(st.stitch_events[0].raw_track_id, 5)

        # 后续帧 raw_id=5 直接映射回 canonical_id=1，不重复触发缝合事件
        id_map, pairs = st.resolve_frame([(5, "vehicle", "car", 3.1, 5.7)], 0.7)
        self.assertEqual(id_map, {5: 1})
        self.assertEqual(pairs, [])
        self.assertEqual(st.stitched_track_count, 1)

        # 二次遮挡并再切换为 raw_id=9，依然缝合回 canonical_id=1
        st.update_kinematics(
            1,
            SimpleNamespace(
                x_m=3.1, y_m=5.7, vx_ms=0.0, vy_ms=8.0,
                ax_ms2=0.0, ay_ms2=0.0, is_reliable=True,
                category="vehicle", object_kind="car",
                pos_cov_m2=(0.04, 0.0, 0.0, 0.16),
            ),
            0.7,
        )
        st.resolve_frame([], 0.9)
        id_map, pairs = st.resolve_frame([(9, "vehicle", "car", 3.1, 8.9)], 1.1)
        self.assertEqual(id_map, {9: 1})
        self.assertEqual(pairs, [(9, 1)])
        self.assertEqual(st.stitched_track_count, 2)

    def test_gates_reject_category_gap_mahalanobis_heading_and_bound_pool(self):
        # 1) 类别不一致与航向反向均不得缝合
        st = TrackletStitcher(max_gap_s=1.0, max_distance_m=5.0, max_mahalanobis=6.0, max_heading_diff_deg=45.0)
        st.resolve_frame([(1, "vehicle", "car", 3.0, 5.0)], 0.0)
        st.update_kinematics(
            1,
            SimpleNamespace(
                x_m=3.0, y_m=5.0, vx_ms=0.0, vy_ms=5.0,
                is_reliable=True, category="vehicle", object_kind="car",
            ),
            0.0,
        )
        st.resolve_frame([], 0.1)
        # 类别不同
        id_map, pairs = st.resolve_frame([(2, "pedestrian", "person", 3.0, 6.0)], 0.2)
        self.assertEqual(pairs, [])
        self.assertEqual(id_map, {2: 2})
        # 反向航向 (位移向 -Y 与历史 +Y 夹角 180 deg > 45 deg)
        id_map, pairs = st.resolve_frame([(3, "vehicle", "car", 3.0, 3.0)], 0.25)
        self.assertEqual(pairs, [])
        self.assertEqual(id_map, {3: 3})

        # 2) 各向异性协方差下横向大偏移被马氏门控拒绝（即使在欧氏门限内）
        st_maha = TrackletStitcher(max_gap_s=1.5, max_distance_m=3.5, max_mahalanobis=3.5, velocity_noise_ms=1.0)
        st_maha.resolve_frame([(1, "vehicle", "car", 3.0, 0.0)], 0.0)
        st_maha.update_kinematics(
            1,
            SimpleNamespace(
                x_m=3.0, y_m=0.0, vx_ms=0.0, vy_ms=5.0,
                is_reliable=True, category="vehicle", object_kind="car",
                pos_cov_m2=(0.02, 0.0, 0.0, 0.50),
            ),
            0.0,
        )
        st_maha.resolve_frame([], 0.05)
        # 横向偏移 1.8m (sigma_x ≈ sqrt(0.02 + 0.01) = 0.173 -> d_M > 10 > 3.5) 应拒绝
        _, pairs = st_maha.resolve_frame([(2, "vehicle", "car", 4.8, 0.5)], 0.1)
        self.assertEqual(pairs, [])

        # 3) 超过 max_gap_s 过期失效，且丢失池严格受 max_lost_pool 约束
        st_pool = TrackletStitcher(max_gap_s=0.5, max_lost_pool=2)
        st_pool.resolve_frame([(i, "vehicle", "car", float(i * 10), 0.0) for i in range(1, 5)], 0.0)
        st_pool.resolve_frame([], 0.1)
        self.assertEqual(st_pool.lost_count, 2)
        _, pairs_expired = st_pool.resolve_frame([(99, "vehicle", "car", 40.0, 0.0)], 0.7)
        self.assertEqual(pairs_expired, [])
        self.assertEqual(st_pool.lost_count, 0)


class TestEndToEndKalmanOBBStitchingReplay(unittest.TestCase):
    def _write_synthetic_tracks_with_occlusion(self, tracks_path: Path, calib: RoadCalibrator):
        fps = 25.0
        dt = 1.0 / fps
        rows = []
        # 75 帧 (3.0s): 车辆以 8.0 m/s 沿车道 y=2..26m 前进，行人稳定在前方 (3.2, 22.0)
        # 帧 25..39 (1.00s..1.56s, 0.60s) 发生遮挡消失，帧 40 起以新 track_id=2 恢复
        for idx in range(75):
            ts = round(idx * dt, 6)
            targets = []
            if not (25 <= idx < 40):
                raw_veh_id = 1 if idx < 25 else 2
                y_veh = 2.0 + 8.0 * ts
                if calib.is_in_zone(3.0, y_veh, margin_m=2.0):
                    u_v, v_v = _project_ground_to_pixel(calib, 3.0, y_veh)
                    targets.append(_make_track_target(raw_veh_id, u_v, v_v, class_name="car", category="vehicle"))
            u_p, v_p = _project_ground_to_pixel(calib, 3.2, 22.0)
            targets.append(_make_track_target(10, u_p, v_p, class_name="person", category="pedestrian"))
            rows.append({
                "schema_version": 1,
                "session_id": "sess-stitch",
                "source": "synthetic.mp4",
                "frame_index": idx,
                "timestamp_s": ts,
                "timestamp_source": "video_pts",
                "width": 1000,
                "height": 640,
                "tracks": targets,
            })
        tracks_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    def test_tracks_replay_default_vs_kalman_obb_stitching(self):
        calib = _perspective_calibrator()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calib_path = root / "calib.yaml"
            calib.to_yaml(calib_path)
            tracks_path = root / "tracks.jsonl"
            self._write_synthetic_tracks_with_occlusion(tracks_path, calib)

            out_default = root / "run_default"
            rc_def = analysis_main([
                "--tracks", str(tracks_path),
                "--calib", str(calib_path),
                "--output", str(out_default),
                "--alert-cooldown-s", "0.8",
            ])
            self.assertEqual(rc_def, 0)
            self.assertTrue(validate_run(out_default)["valid"])
            cfg_def = load_json_strict(out_default / "configuration.json")
            sum_def = load_json_strict(out_default / "summary.json")
            self.assertEqual(cfg_def["speed_model"], "ema")
            self.assertEqual(cfg_def["conflict_model"], "cvm_circle")
            self.assertIs(cfg_def["enable_stitching"], False)
            self.assertEqual(sum_def["stitched_track_count"], 0)

            out_adv = root / "run_adv"
            rc_adv = analysis_main([
                "--tracks", str(tracks_path),
                "--calib", str(calib_path),
                "--output", str(out_adv),
                "--speed-model", "kalman",
                "--conflict-model", "obb_ssm",
                "--enable-stitching",
                "--alert-cooldown-s", "0.8",
            ])
            self.assertEqual(rc_adv, 0)
            self.assertTrue(validate_run(out_adv)["valid"])
            cfg_adv = load_json_strict(out_adv / "configuration.json")
            sum_adv = load_json_strict(out_adv / "summary.json")
            self.assertEqual(cfg_adv["speed_model"], "kalman")
            self.assertEqual(cfg_adv["conflict_model"], "obb_ssm")
            self.assertIs(cfg_adv["enable_stitching"], True)
            self.assertGreaterEqual(sum_adv["stitched_track_count"], 1)

            # 验证 speeds.csv 中车辆始终归并为 canonical_id=1（不再出现碎片 ID=2）
            with (out_adv / "speeds.csv").open(encoding="utf-8", newline="") as f:
                veh_ids = {
                    int(row["track_id"])
                    for row in csv.DictReader(f)
                    if row["category"] == "vehicle"
                }
            self.assertEqual(veh_ids, {1})

            # 验证 events.jsonl 未被 0.6s 遮挡拆分成多条冲突事件
            adv_events = load_jsonl_strict(out_adv / "events.jsonl")
            conflict_events = [e for e in adv_events if e["kind"] == "conflict"]
            self.assertEqual(len(conflict_events), 1)
            self.assertEqual(conflict_events[0]["veh_track_id"], 1)

            # 验证 conflicts.jsonl 记录了 obb_ssm 衍生安全指标且全部有限合法
            conflicts = load_jsonl_strict(out_adv / "conflicts.jsonl")
            obb_items = [c["assessment"] for c in conflicts if c["assessment"]["ped_track_id"] is not None]
            self.assertTrue(obb_items)
            self.assertTrue(all(a["conflict_model"] == "obb_ssm" for a in obb_items))
            self.assertTrue(any(a["drac_ms2"] is not None and a["collision_prob"] is not None for a in obb_items))

    def test_tracks_replay_with_video_rendering_under_stitching(self):
        calib = _perspective_calibrator()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calib_path = root / "calib.yaml"
            calib.to_yaml(calib_path)
            tracks_path = root / "tracks.jsonl"
            video_path = root / "sync.mp4"

            rows = []
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1000, 640))
            self.assertTrue(writer.isOpened())
            for idx in range(25):
                writer.write(np.zeros((640, 1000, 3), dtype=np.uint8))
                ts = round(idx / 25.0, 6)
                targets = []
                if not (10 <= idx < 14):
                    raw_id = 1 if idx < 10 else 2
                    u, v = _project_ground_to_pixel(calib, 3.0, 2.0 + 8.0 * ts)
                    targets.append(_make_track_target(raw_id, u, v, class_name="car", category="vehicle"))
                u_p, v_p = _project_ground_to_pixel(calib, 3.2, 12.0)
                targets.append(_make_track_target(10, u_p, v_p, class_name="person", category="pedestrian"))
                rows.append({
                    "schema_version": 1,
                    "session_id": "sess-render",
                    "source": str(video_path),
                    "frame_index": idx,
                    "timestamp_s": ts,
                    "timestamp_source": "video_pts",
                    "width": 1000,
                    "height": 640,
                    "tracks": targets,
                })
            writer.release()
            tracks_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )

            out_dir = root / "out_render"
            badge_calls = []
            from analysis.visualization import EnhancedConflictVisualizer

            orig_badge = EnhancedConflictVisualizer.draw_speed_badge

            def _spy_badge(viz_self, frame, box_xyxy, reading, conflict=None, event_id=None):
                badge_calls.append((reading.track_id, reading.y_m, conflict, event_id))
                return orig_badge(viz_self, frame, box_xyxy, reading, conflict, event_id=event_id)

            with mock.patch.object(EnhancedConflictVisualizer, "draw_speed_badge", new=_spy_badge):
                rc = analysis_main([
                    "--tracks", str(tracks_path),
                    "--source", str(video_path),
                    "--calib", str(calib_path),
                    "--output", str(out_dir),
                    "--speed-model", "kalman",
                    "--conflict-model", "obb_ssm",
                    "--enable-stitching",
                    "--alert-cooldown-s", "0.5",
                ])
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "annotated_conflict.mp4").is_file())
            self.assertTrue(validate_run(out_dir)["valid"])

            # 验证缝合换 ID 后（canonical_id=1，y >= 6.4m 处于 raw_id=2 复现阶段），
            # draw_speed_badge 仍能匹配到非空冲突评估与 event_id
            post_switch_badges = [
                (tid, y_m, c, eid)
                for tid, y_m, c, eid in badge_calls
                if tid == 1 and y_m >= 6.4 and c is not None and c.risk_level >= RiskLevel.LEVEL_2_CAUTION and eid is not None
            ]
            self.assertTrue(post_switch_badges)
            # 确认缝合确实在视频渲染运行中触发
            self.assertGreaterEqual(load_json_strict(out_dir / "summary.json")["stitched_track_count"], 1)

    def test_invalid_pipeline_and_cli_model_options_rejected(self):
        calib = _perspective_calibrator()
        for bad_kw in (
            {"speed_model": "ukf"},
            {"speed_model": True},
            {"conflict_model": "unknown"},
            {"conflict_model": 123},
            {"enable_stitching": 1},
            {"enable_stitching": "true"},
        ):
            with self.assertRaises(ValueError):
                ConflictPerceptionPipeline(calibrator=calib, **bad_kw)

        for bad_cli in (["--speed-model", "invalid"], ["--conflict-model", "invalid"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    analysis_main(["--source", "x.mp4", *bad_cli])


class TestSchemaAblationScenariosIntegration(unittest.TestCase):
    def test_ablation_and_scenarios_validation(self):
        ablation = run_default_ablation()
        validate_ablation(ablation)
        cmp = ablation["algorithm_comparison"]
        self.assertTrue(cmp["speed_estimation"]["kalman_improves_mae"])
        self.assertLess(
            cmp["speed_estimation"]["kalman"]["mae_ms"],
            cmp["speed_estimation"]["ema"]["mae_ms"],
        )

        by_scen = {row["scenario"]: row for row in cmp["conflict_detection"]}
        self.assertEqual(by_scen["parallel_pass"]["cvm_circle"]["risk_level"], "LEVEL_3_CRITICAL")
        self.assertEqual(by_scen["parallel_pass"]["obb_ssm"]["risk_level"], "LEVEL_0_SAFE")
        self.assertEqual(by_scen["braking_yield"]["cvm_circle"]["risk_level"], "LEVEL_3_CRITICAL")
        self.assertEqual(by_scen["braking_yield"]["obb_ssm"]["risk_level"], "LEVEL_2_CAUTION")

        self.assertTrue(cmp["tracklet_stitching"]["stitching_preserved_continuity"])
        self.assertEqual(cmp["tracklet_stitching"]["stitched"]["unique_track_ids"], 1)
        self.assertEqual(cmp["tracklet_stitching"]["stitched"]["alert_event_count"], 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ab_path = root / "ablation.json"
            sc_path = root / "scenarios.json"
            self.assertEqual(ablation_main(["--output", str(ab_path)]), 0)
            self.assertEqual(scenarios_main(["--output", str(sc_path)]), 0)
            ab_loaded = load_json_strict(ab_path)
            sc_loaded = load_json_strict(sc_path)
            validate_ablation(ab_loaded)
            validate_scenarios(sc_loaded)
            self.assertEqual(
                len(sc_loaded["results"]),
                len(default_scenarios()) + len(obb_ssm_scenarios()),
            )
            self.assertTrue(all(r["passed"] for r in sc_loaded["results"]))

        obb_res = run_obb_ssm_scenarios()
        self.assertTrue(all(r["passed"] for r in obb_res))
        self.assertTrue(all(r["passed"] for r in run_scenarios()))

    def test_schema_strict_checks_for_new_optional_fields(self):
        unavailable = {"path": None, "status": "unavailable", "sha256": None, "size_bytes": None}
        cfg = {
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
                "environment": {"python": "3.12", "packages": {}},
                "git": {"commit": None, "dirty": None, "status": "unavailable"},
                "artifacts": {
                    key: dict(unavailable)
                    for key in ("weights", "tracker_config", "calibration", "object_policy")
                },
            },
        }
        # 旧版（不含新字段）必须保持兼容通过
        validate_configuration(cfg)

        cfg["speed_model"] = "kalman"
        cfg["conflict_model"] = "obb_ssm"
        cfg["enable_stitching"] = True
        validate_configuration(cfg)

        for key, bad_val in (
            ("speed_model", "invalid"),
            ("conflict_model", 1),
            ("enable_stitching", 1),
        ):
            bad_cfg = dict(cfg)
            bad_cfg[key] = bad_val
            with self.assertRaises(SchemaValidationError):
                validate_configuration(bad_cfg)

        rec = {
            "schema_version": 1,
            "frame_index": 0,
            "timestamp_s": 0.0,
            "event_id": None,
            "assessment": {
                "veh_track_id": 1,
                "ped_track_id": 2,
                "risk_level": 3,
                "ttc_seconds": 1.0,
                "vehicle_kind": "car",
                "mttc_seconds": 0.9,
                "drac_ms2": 4.2,
                "pet_seconds": None,
                "collision_prob": 0.75,
                "conflict_model": "obb_ssm",
            },
        }
        validate_conflict_record(rec)
        for field, bad_val in (
            ("mttc_seconds", -0.1),
            ("drac_ms2", True),
            ("collision_prob", 1.2),
            ("conflict_model", "bad"),
        ):
            bad_rec = json.loads(json.dumps(rec))
            bad_rec["assessment"][field] = bad_val
            with self.assertRaises(SchemaValidationError):
                validate_conflict_record(bad_rec)

        bad_sum = {
            "schema_version": 1,
            "status": "failed",
            "stop_reason": "error",
            "frames_processed": 0,
            "elapsed_seconds": 0.1,
            "stitched_track_count": True,
        }
        with self.assertRaises(SchemaValidationError):
            validate_summary(bad_sum)

        # 确保公开导出和零 scipy 依赖
        self.assertIsNotNone(ConstantAccelerationKalman2D)
        self.assertIsNotNone(OBBCollisionResult)
        self.assertIsNotNone(PETTracker)
        self.assertNotIn("scipy", sys.modules)


if __name__ == "__main__":
    unittest.main()
