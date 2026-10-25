import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from analysis.ablation import main as ablation_main, run_conflict_ablation, run_speed_ablation
from analysis.calibration import RoadCalibrator
from analysis.conflict import ConflictAnalyzer, RiskLevel
from analysis.evaluate import evaluate_run, main as evaluate_main, to_markdown
from analysis.pipeline import ConflictPerceptionPipeline
from analysis.policy import object_policy, read_object_policy_config
from analysis.scenarios import default_scenarios, main as scenarios_main, run_scenarios
from analysis.schema import validate_run
from analysis.speed import SpeedEstimator, SpeedReading


class AnalysisToolTests(unittest.TestCase):
    def test_evaluate_run_summarizes_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(
                json.dumps({"frames_processed": 12, "average_fps": 8.0}), encoding="utf-8"
            )
            (root / "conflicts.jsonl").write_text(
                json.dumps({"frame_index": 1, "event_id": 3,
                             "assessment": {"risk_level": 3}}) + "\n",
                encoding="utf-8",
            )
            (root / "events.jsonl").write_text(
                json.dumps({"event_id": 3, "kind": "conflict",
                            "peak_risk_name": "LEVEL_3_CRITICAL",
                            "duration_s": 1.2, "min_ttc_seconds": 1.0}) + "\n",
                encoding="utf-8",
            )
            with (root / "speeds.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["speed_kmh", "is_reliable", "is_stationary", "event_id"]
                )
                writer.writeheader()
                writer.writerow(
                    {"speed_kmh": "18.0", "is_reliable": "1", "is_stationary": "0", "event_id": "3"}
                )

            result = evaluate_run(root)
            self.assertFalse(result["has_ground_truth"])
            self.assertEqual(result["frames"]["processed"], 12)
            self.assertEqual(result["frames"]["confirmed_event_rows"], 1)
            self.assertEqual(result["events"]["count"], 1)
            self.assertEqual(result["events"]["by_kind"]["conflict"], 1)
            self.assertEqual(result["events"]["by_peak_level"]["LEVEL_3_CRITICAL"], 1)
            self.assertEqual(result["events"]["duration_s"]["mean"], 1.2)
            self.assertEqual(result["events"]["min_ttc_s"]["min"], 1.0)
            self.assertEqual(result["risk_counts"]["LEVEL_3"], 1)
            self.assertEqual(result["speed_rows"]["speed_kmh"]["mean"], 18.0)
            self.assertEqual(result["speed_rows"]["confirmed_event_rows"], 1)
            self.assertEqual(result["speed_rows"]["reliable_values"]["1"], 1)
            self.assertEqual(result["speed_rows"]["stationary_values"]["0"], 1)
            self.assertIn("ground truth", to_markdown(result))

            md_out = root / "eval.md"
            self.assertEqual(
                evaluate_main(["--run", str(root), "--markdown", "--output", str(md_out)]),
                0,
            )
            self.assertIn("ground truth", md_out.read_text(encoding="utf-8"))

    def test_policy_mappings_and_speed_reading_compatibility(self):
        self.assertEqual(object_policy("person", "pedestrian").kind, "pedestrian")
        self.assertEqual(object_policy("bicycle", "vehicle").kind, "bicycle")
        self.assertEqual(object_policy("bicycle", "vehicle").safe_radius_veh_m, 1.2)
        self.assertEqual(object_policy("motorcycle", "vehicle").kind, "motorcycle")
        self.assertEqual(object_policy("motorcycle", "vehicle").safe_radius_veh_m, 1.8)
        self.assertEqual(object_policy("car", "vehicle").kind, "car")
        self.assertEqual(object_policy("car", "vehicle").safe_radius_veh_m, 2.2)
        self.assertEqual(object_policy("bus", "vehicle").kind, "bus")
        self.assertEqual(object_policy("bus", "vehicle").safe_radius_veh_m, 2.8)
        self.assertEqual(object_policy("truck", "vehicle").kind, "truck")
        self.assertEqual(object_policy("truck", "vehicle").safe_radius_veh_m, 2.8)
        self.assertEqual(object_policy("unknown_veh", "vehicle").kind, "motor_vehicle")
        self.assertEqual(object_policy("unknown_veh", "vehicle").safe_radius_veh_m, 2.2)

        estimator = SpeedEstimator()
        reading = estimator.update(1, "vehicle", 0.0, 3.0, 0.0, object_kind="motorcycle")
        self.assertEqual(reading.category, "vehicle")
        self.assertEqual(reading.object_kind, "motorcycle")
        reading2 = estimator.update(1, "vehicle", 0.1, 3.0, 0.5)
        self.assertEqual(reading2.object_kind, "motorcycle")

    def test_policy_yaml_can_override_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.yaml"
            path.write_text(
                "schema_version: 1\npolicies:\n"
                "  person: {kind: pedestrian, safe_radius_veh_m: 0.8, label: person}\n"
                "  pedestrian: {kind: pedestrian, safe_radius_veh_m: 0.8, label: pedestrian}\n"
                "  vehicle: {kind: motor_vehicle, safe_radius_veh_m: 4.2, label: custom}\n"
                "  motor_vehicle: {kind: motor_vehicle, safe_radius_veh_m: 4.2, label: custom}\n",
                encoding="utf-8",
            )
            policies = read_object_policy_config(path)
            self.assertEqual(policies["vehicle"].safe_radius_veh_m, 4.2)
            calibrator = RoadCalibrator(
                [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)], 10.0, 10.0
            )
            pipeline = ConflictPerceptionPipeline(
                calibrator=calibrator, object_policy_config=path
            )
            self.assertEqual(
                pipeline.speed_estimator.object_policies["vehicle"].safe_radius_veh_m, 4.2
            )
            # 验证自定义安全半径 (4.2m + 0.8m = 5.0m) 实际影响 ConflictAnalyzer 风险分级：
            # 当 D_min = 3.5m, TTC = 1.0s 时，默认汽车 (3.0m) 为 LEVEL_2_CAUTION，
            # 而自定义 4.2m 半径下 (3.5m <= 5.0m) 升级为 LEVEL_3_CRITICAL
            veh = SpeedReading(
                track_id=1, category="vehicle", object_kind="motor_vehicle",
                x_m=3.0, y_m=0.0, vx_ms=0.0, vy_ms=6.0,
                speed_ms=6.0, speed_kmh=21.6, is_speeding=False,
                heading_deg=90.0, is_reliable=True,
            )
            ped = SpeedReading(
                track_id=2, category="pedestrian", object_kind="pedestrian",
                x_m=6.5, y_m=6.0, vx_ms=0.0, vy_ms=0.0,
                speed_ms=0.0, speed_kmh=0.0, is_speeding=False,
                heading_deg=0.0, is_reliable=True,
            )
            default_res = ConflictAnalyzer().evaluate_pair(veh, ped)
            custom_res = pipeline.conflict_analyzer.evaluate_pair(veh, ped)
            self.assertEqual(default_res.risk_level, RiskLevel.LEVEL_2_CAUTION)
            self.assertEqual(custom_res.risk_level, RiskLevel.LEVEL_3_CRITICAL)

    def test_invalid_policy_configs_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                read_object_policy_config(root / "missing.yaml")

            bad_cases = [
                "not: [valid: yaml",
                "schema_version: 2\npolicies: {}\n",
                "schema_version: 1\npolicies: {}\n",
                (
                    "schema_version: 1\npolicies:\n"
                    "  person: {kind: pedestrian, safe_radius_veh_m: -1.0, label: p}\n"
                    "  pedestrian: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                    "  vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n"
                    "  motor_vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n"
                ),
                (
                    "schema_version: 1\npolicies:\n"
                    "  person: {kind: pedestrian, safe_radius_veh_m: true, label: p}\n"
                    "  pedestrian: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                    "  vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n"
                    "  motor_vehicle: {kind: motor_vehicle, safe_radius_veh_m: 2.2, label: v}\n"
                ),
                (
                    "schema_version: 1\npolicies:\n"
                    "  person: {kind: pedestrian, safe_radius_veh_m: 0.8, label: p}\n"
                ),
            ]
            for idx, text in enumerate(bad_cases):
                p = root / f"bad_{idx}.yaml"
                p.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError, msg=f"case {idx} should fail"):
                    read_object_policy_config(p)

    def test_synthetic_scenario_library_passes(self):
        results = run_scenarios(default_scenarios())
        self.assertEqual(len(results), 5)
        self.assertTrue(all(result["passed"] for result in results))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "scenarios.json"
            self.assertEqual(scenarios_main(["--output", str(output)]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(all(result["passed"] for result in payload["results"]))

    def test_vehicle_safe_radius_changes_conflict_risk_level(self):
        analyzer = ConflictAnalyzer()
        pedestrian = SpeedReading(
            track_id=2, category="pedestrian",
            x_m=5.5, y_m=6.0, vx_ms=0.0, vy_ms=0.0,
            speed_ms=0.0, speed_kmh=0.0, is_speeding=False,
            heading_deg=0.0, is_reliable=True,
        )
        # 在 t_cpa=1.0s 时车辆到达 (3.0, 6.0)，与行人距离 D_min=2.5m：
        # bicycle 临界阈值 = 1.2 + 0.8 = 2.0m (< 2.5m <= 3.0m) -> LEVEL_2_CAUTION
        # car 临界阈值 = 2.2 + 0.8 = 3.0m (>= 2.5m 且 TTC=1.0s <= 1.8s) -> LEVEL_3_CRITICAL
        bicycle = SpeedReading(
            track_id=1, category="vehicle", object_kind="bicycle",
            x_m=3.0, y_m=0.0, vx_ms=0.0, vy_ms=6.0,
            speed_ms=6.0, speed_kmh=21.6, is_speeding=False,
            heading_deg=90.0, is_reliable=True,
        )
        car = SpeedReading(
            track_id=1, category="vehicle", object_kind="car",
            x_m=3.0, y_m=0.0, vx_ms=0.0, vy_ms=6.0,
            speed_ms=6.0, speed_kmh=21.6, is_speeding=False,
            heading_deg=90.0, is_reliable=True,
        )
        res_bike = analyzer.evaluate_pair(bicycle, pedestrian)
        res_car = analyzer.evaluate_pair(car, pedestrian)
        self.assertIsNotNone(res_bike)
        self.assertIsNotNone(res_car)
        self.assertEqual(res_bike.vehicle_kind, "bicycle")
        self.assertEqual(res_car.vehicle_kind, "car")
        self.assertEqual(res_bike.risk_level, RiskLevel.LEVEL_2_CAUTION)
        self.assertEqual(res_car.risk_level, RiskLevel.LEVEL_3_CRITICAL)

    def test_configuration_json_and_summary_json_linked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video_path = root / "v.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64)
            )
            for _ in range(3):
                writer.write(np.zeros((64, 64, 3), np.uint8))
            writer.release()

            calibrator = RoadCalibrator(
                [(10.0, 10.0), (60.0, 10.0), (60.0, 60.0), (10.0, 60.0)], 6.0, 15.0
            )
            pipeline = ConflictPerceptionPipeline(calibrator=calibrator, window_seconds=0.4)
            fake_session = mock.Mock()
            fake_rec = mock.Mock()
            fake_rec.tracks = []
            fake_session.process.return_value = fake_rec
            fake_session.retained_ids.return_value = set()

            out_dir = root / "out"
            with mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session):
                summary = pipeline.process_video(video_path, out_dir)

            config_path = out_dir / "configuration.json"
            self.assertTrue(config_path.exists())
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(Path(summary["configuration_json"]), config_path.resolve())
            self.assertEqual(summary["configuration"], config)
            self.assertEqual(config["fps_used"], 25.0)
            self.assertEqual(config["window_size_frames"], 10)
            self.assertEqual(config["window_seconds"], 0.4)
            self.assertIn("bicycle", config["object_policies"])
            self.assertEqual(config["object_policies"]["bicycle"]["safe_radius_veh_m"], 1.2)
            self.assertTrue(validate_run(out_dir)["valid"])

    def test_speed_ablation_covers_parameter_grid(self):
        rows = run_speed_ablation(
            speeds_ms=(2.0,), window_sizes=(4, 8), ema_alphas=(0.2, 0.6), frames=20
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["absolute_error_ms"] < 0.01 for row in rows))
        self.assertTrue(any(row["is_reliable"] for row in rows))
        self.assertTrue(all(not row["is_stationary"] for row in rows))

        with tempfile.TemporaryDirectory() as temporary:
            out_file = Path(temporary) / "ablation.json"
            self.assertEqual(ablation_main(["--output", str(out_file)]), 0)
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertIn("speed", payload)
            self.assertIn("conflict", payload)

    def test_conflict_ablation_skips_invalid_threshold_pairs(self):
        rows = run_conflict_ablation(critical_thresholds=(1.0, 3.0), horizons=(2.0, 4.0))
        self.assertEqual(len(rows), 3)
        self.assertTrue(any(row["risk_level"] == "LEVEL_3_CRITICAL" for row in rows))


if __name__ == "__main__":
    unittest.main()
