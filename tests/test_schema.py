import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis.provenance import collect_provenance, file_fingerprint, git_state, sha256_file
from analysis.schema import (
    SchemaValidationError,
    load_json_strict,
    load_jsonl_strict,
    validate_ablation,
    validate_configuration,
    validate_run,
    validate_scenarios,
    validate_speeds_csv,
    main as schema_main,
)


def valid_configuration():
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
            "generated_at": "2026-10-25T12:00:00+08:00",
            "environment": {"python": "3.11.9", "packages": {}},
            "git": {"commit": None, "dirty": None, "status": "unavailable"},
            "artifacts": {key: dict(unavailable) for key in
                          ("weights", "tracker_config", "calibration", "object_policy")},
        },
    }


class ProvenanceTests(unittest.TestCase):
    def test_sha256_is_chunked_and_matches_known_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.bin"
            payload = b"abc" * 10000
            path.write_bytes(payload)
            self.assertEqual(sha256_file(path, chunk_size=17), hashlib.sha256(payload).hexdigest())
            fingerprint = file_fingerprint(path)
            self.assertEqual(fingerprint["status"], "ok")
            self.assertEqual(fingerprint["size_bytes"], len(payload))

    def test_missing_file_and_git_are_nonfatal(self):
        missing = file_fingerprint("definitely-missing.file")
        self.assertEqual(missing["status"], "missing")
        self.assertIsNone(missing["sha256"])
        with mock.patch("analysis.provenance.subprocess.run", side_effect=OSError("no git")):
            state = git_state(".")
        self.assertEqual(state["status"], "unavailable")
        self.assertIsNone(state["commit"])

    def test_collect_provenance_has_no_environment_variable_dump(self):
        result = collect_provenance(
            ".", weights=None, tracker_config=None, calibration=None, object_policy=None
        )
        self.assertIn("python", result["environment"])
        self.assertIn("packages", result["environment"])
        self.assertNotIn("environment_variables", result)


class SchemaTests(unittest.TestCase):
    def test_valid_configuration_and_missing_field(self):
        validate_configuration(valid_configuration())
        invalid = valid_configuration()
        del invalid["weights"]
        with self.assertRaises(SchemaValidationError):
            validate_configuration(invalid)
        bool_version = valid_configuration()
        bool_version["schema_version"] = True
        with self.assertRaises(SchemaValidationError):
            validate_configuration(bool_version)
        bad_ttc = valid_configuration()
        bad_ttc["ttc_critical_s"] = 4.0
        with self.assertRaises(SchemaValidationError):
            validate_configuration(bad_ttc)

    def test_strict_json_rejects_nan_and_jsonl_reports_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_json = root / "bad.json"
            bad_json.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaises(SchemaValidationError):
                load_json_strict(bad_json)
            overflow_json = root / "overflow.json"
            overflow_json.write_text('{"nested": [1e309]}', encoding="utf-8")
            with self.assertRaises(SchemaValidationError):
                load_json_strict(overflow_json)
            bad_jsonl = root / "bad.jsonl"
            bad_jsonl.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(SchemaValidationError, "第 2 行"):
                load_jsonl_strict(bad_jsonl)

    def test_csv_missing_required_column_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "speeds.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(["frame_idx", "timestamp_s"])
            with self.assertRaises(SchemaValidationError):
                validate_speeds_csv(path)

    def test_scenario_and_ablation_contracts(self):
        validate_scenarios({"schema_version": 1, "results": [{
            "name": "x", "expected_levels": ["LEVEL_0_SAFE"],
            "actual_levels": ["LEVEL_0_SAFE"], "passed": True,
        }]})
        validate_ablation({"schema_version": 1, "speed": [{
            "target_speed_ms": 2.0, "window_size": 4, "ema_alpha": 0.2,
            "measured_speed_ms": 2.0, "absolute_error_ms": 0.0,
            "is_reliable": True, "is_stationary": False,
        }], "conflict": [{
            "ttc_critical_s": 1.8, "horizon_s": 3.5, "ttc_s": 1.0,
            "min_distance_m": 0.0, "risk_level": "LEVEL_3_CRITICAL",
        }]})
        with self.assertRaises(SchemaValidationError):
            validate_scenarios({"schema_version": True, "results": []})
        with self.assertRaises(SchemaValidationError):
            validate_ablation({"schema_version": 1, "speed": [{}], "conflict": []})

    def test_complete_run_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = {
                "schema_version": 1, "status": "ok", "stop_reason": "eof", "frames_processed": 1,
                "elapsed_seconds": 0.1, "annotated_output": "a.mp4",
                "conflicts_jsonl": "conflicts.jsonl", "events_jsonl": "events.jsonl",
                "speeds_csv": "speeds.csv", "configuration_json": "configuration.json",
                "total_alert_events": 0, "total_speeding_alerts": 0,
                "total_caution_events": 0, "total_critical_conflicts": 0,
            }
            (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (root / "configuration.json").write_text(
                json.dumps(valid_configuration()), encoding="utf-8"
            )
            with self.assertRaisesRegex(SchemaValidationError, "conflicts.jsonl"):
                validate_run(root)
            conflict = {
                "schema_version": 1, "frame_index": 0, "timestamp_s": 0.0, "event_id": None,
                "assessment": {"veh_track_id": 1, "ped_track_id": None,
                               "risk_level": 0, "ttc_seconds": None,
                               "vehicle_kind": "car"},
            }
            (root / "conflicts.jsonl").write_text(json.dumps(conflict) + "\n", encoding="utf-8")
            (root / "events.jsonl").write_text("", encoding="utf-8")
            with (root / "speeds.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=sorted({
                    "schema_version", "frame_idx", "timestamp_s", "track_id", "category", "speed_kmh",
                    "risk_level", "event_id", "is_reliable", "is_stationary",
                }))
                writer.writeheader()
                writer.writerow({"schema_version": 1, "frame_idx": 0, "timestamp_s": 0.0, "track_id": 1,
                                 "category": "vehicle", "speed_kmh": 0.0,
                                 "risk_level": "LEVEL_0_SAFE", "event_id": "",
                                 "is_reliable": 0, "is_stationary": 0})
            result = validate_run(root)
            self.assertTrue(result["valid"])
            self.assertEqual(result["counts"]["speed_rows"], 1)

            bad_conflict = dict(conflict)
            bad_conflict["frame_index"] = -1
            (root / "conflicts.jsonl").write_text(
                json.dumps(conflict) + "\n\n" + json.dumps(bad_conflict) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SchemaValidationError, "第 3 行"):
                validate_run(root)

    def test_ci_workflow_contains_required_checks_and_no_push(self):
        workflow = Path(".github/workflows/tests.yml").read_text(encoding="utf-8")
        self.assertIn("python -m unittest discover -s tests -q", workflow)
        self.assertIn("python -m compileall -q analysis detection tracking tests", workflow)
        self.assertNotIn("git push", workflow)
        self.assertNotIn("secrets.", workflow)

    def test_schema_cli_returns_nonzero_for_invalid_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SystemExit) as context:
                schema_main(["--run", temporary])
            self.assertEqual(context.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
