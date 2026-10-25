"""用已知坐标与时间验证图像运动数学关系；不依赖检测精度或校园数据。"""

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from tracking.export import RunExports
from tracking.motion import DEFAULT_MOTION_CONFIG, MotionConfig, MotionEstimator, read_motion_config
from tracking.motion_visualization import draw_motion
from tracking.records import FrameRecord, TargetRecord
from tracking.visualization import annotate


def target(x=100.0, y=100.0, track_id=1, class_id=0, class_name="person"):
    return TargetRecord(track_id, class_id, class_name, "pedestrian" if class_name == "person" else "vehicle",
                        0.9, (x - 5, y - 10, x + 5, y), (x, y))


def frame(index, timestamp, targets=None, session="a", kind="video_pts"):
    return FrameRecord(session, "video.mp4", index, timestamp, kind, 2000, 1000,
                       [target()] if targets is None else targets)


def unsmoothed(**overrides):
    return MotionConfig(position_tau_s=0, velocity_tau_s=0, **overrides)


class MotionMathTests(unittest.TestCase):
    def test_nonuniform_time_intervals_produce_correct_displacement_and_px_per_second(self):
        estimator = MotionEstimator(unsmoothed())
        first = estimator.update(frame(0, 0)).tracks[0]
        self.assertFalse(first.velocity_valid)
        self.assertIsNone(first.speed_px_s)
        second = estimator.update(frame(1, 0.5, [target(103, 104)])).tracks[0]
        self.assertEqual(second.raw_displacement_xy_px, (3, 4))
        self.assertEqual(second.displacement_px, 5)
        self.assertEqual(second.velocity_xy_px_s, (6, 8))
        self.assertEqual(second.speed_px_s, 10)
        self.assertEqual(second.direction_xy, (0.6, 0.8))
        self.assertAlmostEqual(second.direction_deg, 53.13010235415598)
        third = estimator.update(frame(2, 1.5, [target(106, 108)])).tracks[0]
        self.assertEqual(third.dt_s, 1)
        self.assertEqual(third.speed_px_s, 5)

    def test_direction_uses_image_x_right_y_down_and_clockwise_degrees(self):
        for dx, dy, angle in ((10, 0, 0), (0, 10, 90), (-10, 0, 180), (0, -10, 270)):
            with self.subTest(angle=angle):
                estimator = MotionEstimator(unsmoothed())
                estimator.update(frame(0, 0))
                result = estimator.update(frame(1, 1, [target(100 + dx, 100 + dy)])).tracks[0]
                self.assertEqual(result.direction_deg, angle)
                self.assertEqual(result.speed_px_s, 10)

    def test_stationary_target_has_valid_zero_speed_and_no_arbitrary_direction(self):
        estimator = MotionEstimator(unsmoothed())
        estimator.update(frame(0, 0))
        result = estimator.update(frame(1, 0.1)).tracks[0]
        self.assertTrue(result.velocity_valid)
        self.assertEqual(result.speed_px_s, 0)
        self.assertEqual(result.image_motion_state, "stationary")
        self.assertIsNone(result.direction_xy)
        self.assertIsNone(result.direction_deg)

    def test_time_adaptive_position_ema_matches_analytic_step_response(self):
        estimator = MotionEstimator(MotionConfig(position_tau_s=0.5, velocity_tau_s=0, max_dt_s=2))
        estimator.update(frame(0, 0))
        second = estimator.update(frame(1, 0.5, [target(110)])).tracks[0]
        # 一个时间常数后应到达输入阶跃的 63.212%，不使用实现中的 ema() 作为期望值。
        self.assertAlmostEqual(second.smoothed_anchor_xy[0], 106.32120558828558)
        self.assertAlmostEqual(second.velocity_xy_px_s[0], 12.64241117657116)
        self.assertEqual(second.raw_velocity_xy_px_s, (20, 0))
        third = estimator.update(frame(2, 1.5, [target(110)])).tracks[0]
        self.assertAlmostEqual(third.smoothed_anchor_xy[0], 109.50212931632136)
        self.assertEqual(third.target.anchor_xy, (110, 100))

    def test_velocity_ema_uses_elapsed_seconds(self):
        estimator = MotionEstimator(MotionConfig(position_tau_s=0, velocity_tau_s=1, max_dt_s=2))
        estimator.update(frame(0, 0))
        estimator.update(frame(1, 1, [target(110)]))  # 第一对有效观测得到 10 px/s。
        third = estimator.update(frame(2, 1.5, [target(120)])).tracks[0]
        self.assertEqual(third.raw_velocity_xy_px_s, (20, 0))
        self.assertAlmostEqual(third.speed_px_s, 13.934693402873666)

    def test_smoothing_reduces_stationary_detection_jitter(self):
        raw, smooth = MotionEstimator(unsmoothed()), MotionEstimator(MotionConfig(position_tau_s=0.2, velocity_tau_s=0.3))
        raw_values, smooth_values = [], []
        for index in range(160):
            record = frame(index, index * 0.05, [target(100 + (2 if index % 2 else -2))])
            a, b = raw.update(record).tracks[0], smooth.update(record).tracks[0]
            if index >= 40:
                raw_values.append(a.speed_px_s)
                smooth_values.append(b.speed_px_s)
        self.assertLess(sum(smooth_values) / len(smooth_values), sum(raw_values) / len(raw_values) * 0.3)


class MotionContinuityTests(unittest.TestCase):
    def test_missing_frames_produce_no_predictions_and_recovery_restarts_velocity(self):
        estimator = MotionEstimator(unsmoothed())
        first = estimator.update(frame(0, 0), {1}).tracks[0]
        second = estimator.update(frame(1, 0.1, [target(101)]), {1}).tracks[0]
        before = len(estimator.history_for(1))
        missing = estimator.update(frame(2, 0.2, []), {1})
        self.assertEqual(missing.tracks, [])
        self.assertEqual(len(estimator.history_for(1)), before)
        recovered = estimator.update(frame(3, 0.3, [target(150)]), {1}).tracks[0]
        self.assertEqual(recovered.reset_reason, "observation_gap")
        self.assertEqual(recovered.gap_frames, 1)
        self.assertIsNone(recovered.speed_px_s)
        self.assertNotEqual(recovered.segment_id, first.segment_id)
        self.assertEqual([len(segment) for segment in estimator.segments(1)], [2, 1])
        self.assertTrue(second.velocity_valid)
        after = estimator.update(frame(4, 0.4, [target(151)]), {1}).tracks[0]
        self.assertAlmostEqual(after.speed_px_s, 10)

    def test_new_id_does_not_inherit_previous_id_velocity(self):
        estimator = MotionEstimator(unsmoothed())
        old = estimator.update(frame(0, 0)).tracks[0]
        new = estimator.update(frame(1, 0.1, [target(800, track_id=2)]), {2}).tracks[0]
        self.assertEqual(new.reset_reason, "new_track")
        self.assertIsNone(new.speed_px_s)
        self.assertNotEqual(old.segment_id, new.segment_id)
        self.assertEqual(estimator.history_for(1), ())

    def test_untracked_detection_retains_observation_but_no_history_or_velocity(self):
        estimator = MotionEstimator()
        result = estimator.update(frame(0, 0, [target(track_id=None)])).tracks[0]
        self.assertEqual(result.image_motion_state, "untracked")
        self.assertIsNone(result.segment_id)
        self.assertIsNone(result.speed_px_s)
        self.assertEqual(estimator.active_track_count, 0)
        self.assertEqual(result.target.anchor_xy, (100, 100))

    def test_discontinuities_reset_position_filter_and_split_trail(self):
        cases = (
            (frame(1, 0.0001, [target(101)]), "dt_too_small"),
            (frame(1, 2, [target(101)]), "dt_too_large"),
            (frame(1, 0.1, [target(800)]), "position_jump"),
            (frame(1, 0.1, [target(101, class_id=2, class_name="car")]), "class_changed"),
            (frame(1, 0.1, [target(101)], kind="fps_estimate"), "timestamp_source_changed"),
            (frame(2, 0.1, [target(101)]), "observation_gap"),
        )
        for record, reason in cases:
            with self.subTest(reason=reason):
                estimator = MotionEstimator()
                estimator.update(frame(0, 0))
                item = estimator.update(record).tracks[0]
                self.assertEqual(item.reset_reason, reason)
                self.assertFalse(item.velocity_valid)
                self.assertIsNone(item.displacement_px)
                self.assertEqual(item.smoothed_anchor_xy, item.target.anchor_xy)
                self.assertEqual(len(estimator.segments(1)), 2)

    def test_invalid_time_does_not_mutate_existing_state(self):
        estimator = MotionEstimator()
        estimator.update(frame(0, 1))
        before = estimator.history_for(1)
        for timestamp in (1, 0.5, -1, float("nan"), float("inf"), None, True):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                estimator.update(frame(1, timestamp, [target(101)]))
            self.assertEqual(estimator.history_for(1), before)
        valid = estimator.update(frame(1, 1.1, [target(101)])).tracks[0]
        self.assertTrue(valid.velocity_valid)

    def test_duplicate_ids_predicted_boxes_and_invalid_frame_order_are_rejected(self):
        estimator = MotionEstimator()
        estimator.update(frame(0, 0))
        invalid = (frame(1, 0.1, [target(), target(101)]),
                   frame(1, 0.1, [replace(target(), observation="predicted")]),
                   frame(0, 0.1), frame(1, 0.1, [target(-10)]))
        for record in invalid:
            with self.assertRaises(ValueError):
                estimator.update(record)
            self.assertEqual(len(estimator.history_for(1)), 1)

    def test_session_change_and_explicit_reset_discard_old_state(self):
        estimator = MotionEstimator(unsmoothed())
        estimator.update(frame(0, 0))
        estimator.update(frame(1, 0.1, [target(110)]))
        fresh = estimator.update(frame(0, 0, [target(800)], session="b")).tracks[0]
        self.assertIsNone(fresh.speed_px_s)
        self.assertEqual(len(estimator.history_for(1)), 1)
        estimator.reset()
        self.assertEqual(estimator.active_track_count, 0)
        self.assertEqual(estimator.history_for(1), ())

    def test_history_and_idle_tracks_remain_bounded_during_long_input(self):
        estimator = MotionEstimator(unsmoothed(history_length=3), max_idle_frames=2)
        for index in range(1000):
            estimator.update(frame(index, index / 10, [target(track_id=index + 1)]))
            self.assertLessEqual(estimator.active_track_count, 4)
        estimator.update(frame(1000, 100, []), set())
        self.assertEqual(estimator.active_track_count, 0)

    def test_history_length_is_a_source_frame_window_and_segment_ids_are_not_reused(self):
        estimator = MotionEstimator(unsmoothed(history_length=3), max_idle_frames=0)
        first = estimator.update(frame(0, 0)).tracks[0]
        for index in range(1, 10):
            estimator.update(frame(index, index / 10, [target(100 + index)]))
        self.assertEqual([p.frame_index for p in estimator.history_for(1)], [7, 8, 9])
        estimator.update(frame(10, 1, []), set())
        reused = estimator.update(frame(11, 1.1)).tracks[0]
        self.assertGreater(reused.segment_id, first.segment_id)
        self.assertFalse(reused.velocity_valid)


class MotionConfigAndOutputTests(unittest.TestCase):
    def test_config_validation_and_disabling_smoothing(self):
        settings = read_motion_config(DEFAULT_MOTION_CONFIG)
        self.assertEqual(settings.history_length, 60)
        self.assertEqual(unsmoothed(max_step_speed_px_s=None).position_tau_s, 0)
        for override in ({"history_length": True}, {"position_tau_s": -1}, {"velocity_tau_s": float("nan")},
                         {"min_dt_s": 0}, {"max_dt_s": 0.0001}, {"arrow_max_length_px": 0}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                replace(settings, **override)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.yaml"
            path.write_text(yaml.safe_dump({"history_length": 3}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_motion_config(path)

    def test_export_units_raw_observations_and_missing_derivatives(self):
        import json
        estimator = MotionEstimator(unsmoothed())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = RunExports(root)
            try:
                for record in (frame(0, 0), frame(1, 0.5, [target(103, 104)]), frame(2, 0.6, [])):
                    motion = estimator.update(record, {1})
                    export.write(record, {1}, motion)
                export.finish("eof")
            finally:
                export.close()
            frames = [json.loads(line) for line in (root / "motion.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(frames[0]["speed_unit"], "px/s")
            self.assertIsNone(frames[0]["tracks"][0]["speed_px_s"])
            self.assertEqual(frames[1]["tracks"][0]["speed_px_s"], 10)
            self.assertEqual(frames[2]["tracks"], [])
            original = [json.loads(line) for line in (root / "tracks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertNotIn("speed_px_s", original[1]["tracks"][0])
            self.assertEqual(original[1]["schema_version"], 1)
            with (root / "motion.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertNotIn(None, rows[1])  # 防止 CSV 列错位或多出匿名列。
            self.assertEqual(rows[0]["speed_px_s"], "")
            self.assertEqual(rows[1]["speed_unit"], "px/s")
            self.assertEqual(float(rows[1]["vx_px_s"]), 6)
            self.assertEqual(float(rows[1]["vy_px_s"]), 8)
            self.assertEqual(float(rows[1]["anchor_x"]), 103)

    def test_visualization_uses_saved_motion_and_preserves_input_frame(self):
        estimator = MotionEstimator(unsmoothed())
        estimator.update(frame(0, 0))
        record = frame(1, 0.1, [target(110)])
        motion = estimator.update(record)
        image = np.zeros((1000, 2000, 3), dtype=np.uint8)
        before = estimator.history_for(1)
        with patch("cv2.arrowedLine", wraps=cv2.arrowedLine) as arrow, patch("cv2.putText", wraps=cv2.putText) as text:
            output = annotate(image, record, estimator, 10, 20, 2, ["person"], motion=motion,
                              motion_config=estimator.config)
        self.assertEqual(arrow.call_args.args[1:3], ((110, 100), (160, 100)))
        self.assertTrue(any("px/s" in call.args[1] for call in text.call_args_list))
        self.assertFalse(np.any(image))
        self.assertTrue(np.any(output))
        self.assertEqual(estimator.history_for(1), before)
        self.assertIs(draw_motion(output, motion, estimator.config), output)
        self.assertIs(draw_motion(output, motion), output)


if __name__ == "__main__":
    unittest.main()
