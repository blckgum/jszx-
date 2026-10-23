"""已知几何关系、独立检查点和流式文件契约；不是校园实测精度评估。"""

import csv
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis.calibration import RoadCalibrator
from calibration.__main__ import main
from calibration.core import MappingConfig, RoadMapper
from calibration.demo import run_demo, synthetic_config
from calibration.geometry import estimate_homography, project_points
from calibration.io import import_legacy, load_mapping, mapping_from_document, resolved_document, write_json
from calibration.lens import LensModel
from calibration.pipeline import map_tracks
from calibration.visualization import render_grid


def example_mapper(**changes):
    config, _ = synthetic_config()
    return RoadMapper(replace(config, **changes))


def track_frame(index=0, timestamp=0.0, targets=True):
    config, truth = synthetic_config()
    pixels, _ = project_points(truth, [[3 + index * 0.1, 10]])
    x, y = pixels[0].tolist()
    return {"schema_version": 1, "session_id": "test", "source": "controlled", "frame_index": index,
            "timestamp_s": timestamp, "timestamp_source": "fps_estimate", "width": 1280, "height": 720,
            "tracks": [{"track_id": 7, "segment_id": 2, "class_id": 0, "class_name": "person",
                        "confidence": 0.9, "bbox_xyxy": [x-5, y-20, x+5, y], "anchor_xy": [x, y],
                        "anchor_type": "bbox_bottom_center", "observation": "detected"}] if targets else []}


def save_frames(path, frames):
    path.write_text("".join(json.dumps(frame) + "\n" for frame in frames), encoding="utf-8")


class HomographyTests(unittest.TestCase):
    def test_known_pixel_to_metre_map_is_invariant_to_matrix_scale(self):
        matrix = np.array([[0.01, 0, -1], [0, 0.05, -5], [0, 0, 1]])
        for scale in (1, 1e-18, -2000):
            mapped, valid = project_points(matrix * scale, [[100, 100], [400, 300]])
            self.assertTrue(valid.all())
            np.testing.assert_allclose(mapped, [[0, 0], [3, 10]], atol=1e-9)

    def test_projection_at_infinity_and_nonfinite_input_are_not_fake_zero_points(self):
        matrix = np.array([[1, 0, 0], [0, 1, 0], [0.001, 0, -1]])
        mapped, valid = project_points(matrix, [[1000, 2], [float("nan"), 2], [500, 2]])
        self.assertEqual(valid.tolist(), [False, False, True])
        self.assertTrue(np.isnan(mapped[:2]).all())
        np.testing.assert_allclose(mapped[2], [-1000, -4])

    def test_known_projective_scene_recovers_independent_ground_positions(self):
        config, truth = synthetic_config()
        mapper = RoadMapper(config)
        ground = [[1.3, 3.7], [5.8, 18.2]]
        pixels, _ = project_points(truth, ground)
        mapped = mapper.map_points(pixels, camera_id=config.camera_id, image_size=config.image_size)
        np.testing.assert_allclose([point.ground_position_m for point in mapped], ground, atol=1e-5)
        roundtrip, valid = mapper.ground_to_image(ground)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(roundtrip, pixels, atol=1e-4)

    def test_rectangle_solver_reuses_existing_calibrator(self):
        image = [[100, 100], [700, 100], [700, 500], [100, 500]]
        ground = [[0, 0], [6, 0], [6, 20], [0, 20]]
        matrix, _, inliers = estimate_homography(image, ground, method="rectangle")
        old = RoadCalibrator(image, 6, 20)
        mapped, valid = project_points(matrix, [[400, 300]])
        self.assertTrue(valid.all() and inliers.all())
        np.testing.assert_allclose(mapped[0], old.pixel_to_ground(400, 300), atol=1e-7)

    def test_degenerate_reference_layouts_are_rejected(self):
        valid = [[0, 0], [2, 0], [2, 2], [0, 2]]
        invalid = ([[0, 0], [0, 0], [2, 2], [0, 2]],
                   [[0, 0], [1, 1], [2, 2], [3, 3]],
                   [[0, 0], [1, 0], [2, 0], [0, 2]])
        for points in invalid:
            with self.subTest(points=points), self.assertRaises(ValueError):
                estimate_homography(points, valid)
        with self.assertRaises(ValueError):
            estimate_homography(valid, valid, method="ransac")

    def test_ransac_excludes_an_outlier_and_uses_inlier_coverage_for_roi(self):
        config, _ = synthetic_config()
        config = replace(config, method="ransac", ransac_threshold_m=0.01,
                         image_points=config.image_points + [[30, 30]],
                         ground_points_m=config.ground_points_m + [[100, 100]])
        mapper = RoadMapper(config)
        self.assertFalse(mapper.inliers[-1])
        self.assertEqual(int(mapper.inliers.sum()), 9)
        self.assertLess(mapper.report["independent_checks"]["rmse_m"], 1e-4)
        outside = mapper.map_point(30, 30, camera_id=config.camera_id, image_size=config.image_size)
        self.assertFalse(outside.mapping_valid)
        self.assertEqual(outside.invalid_reason, "outside_roi")

    def test_wrong_correspondence_order_cannot_cross_infinity_inside_roi(self):
        with self.assertRaises(ValueError):
            RoadMapper(MappingConfig(camera_id="bad", image_size=[800, 600], source_type="synthetic",
                                     measurement_reference="unit test", method="rectangle",
                                     image_points=[[100,100],[700,500],[700,100],[100,500]],
                                     ground_points_m=[[0,0],[6,0],[6,20],[0,20]]))


class MappingValidityTests(unittest.TestCase):
    def test_synthetic_never_becomes_real_calibration_even_with_zero_error(self):
        mapper = example_mapper()
        self.assertTrue(mapper.report["independent_check_passed"])
        self.assertFalse(mapper.calibration_valid)
        self.assertFalse(mapper.report["calibration_valid"])

    def test_declared_measured_fixture_requires_independent_checks(self):
        # 只验证状态规则；测试夹具不是实地标定证据。
        mapper = example_mapper(source_type="measured", measurement_reference="unit-test declared provenance")
        self.assertTrue(mapper.calibration_valid)
        missing = example_mapper(source_type="measured", check_image_points=[], check_ground_points_m=[])
        self.assertFalse(missing.calibration_valid)
        shifted = (np.asarray(mapper.config.check_ground_points_m) + [1, 0]).tolist()
        bad = example_mapper(source_type="measured", check_ground_points_m=shifted)
        self.assertFalse(bad.calibration_valid)
        self.assertGreater(bad.report["independent_checks"]["rmse_m"], 0.5)
        restricted = example_mapper(source_type="measured", roi_image_points=[[600,400],[680,400],[680,450],[600,450]])
        self.assertFalse(restricted.calibration_valid)
        self.assertEqual(restricted.report["independent_checks"]["count"], 4)
        self.assertIsNone(restricted.report["independent_checks"]["rmse_m"])

    def test_training_points_cannot_be_reused_as_independent_checks(self):
        config, _ = synthetic_config()
        with self.assertRaisesRegex(ValueError, "独立验证"):
            replace(config, check_image_points=config.image_points[:3], check_ground_points_m=config.ground_points_m[:3])

    def test_invalid_context_and_outside_roi_return_explicit_reasons(self):
        mapper = example_mapper()
        cases = ((640, 400, "wrong", [1280,720], "camera_mismatch"),
                 (640, 400, mapper.config.camera_id, [640,360], "image_size_mismatch"),
                 (10, 10, mapper.config.camera_id, [1280,720], "outside_roi"),
                 (-1, 10, mapper.config.camera_id, [1280,720], "outside_image"),
                 (float("nan"), 10, mapper.config.camera_id, [1280,720], "nonfinite_input"))
        for x, y, camera, size, reason in cases:
            with self.subTest(reason=reason):
                result = mapper.map_point(x, y, camera_id=camera, image_size=size)
                self.assertFalse(result.mapping_valid)
                self.assertIsNone(result.ground_position_m)
                self.assertEqual(result.invalid_reason, reason)

    def test_roi_cannot_expand_beyond_reference_coverage(self):
        with self.assertRaises(ValueError):
            example_mapper(roi_image_points=[[0,0],[1280,0],[1280,720],[0,720]])

    def test_invalid_configuration_types_are_rejected(self):
        config, _ = synthetic_config()
        for change in ({"image_size":[True,720]}, {"source_type":"unknown"},
                       {"measurement_reference":""}, {"max_check_rmse_m":float("nan")}, {"schema_version":True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(config, **change)


class LensTests(unittest.TestCase):
    def parameters(self):
        return {"model":"opencv_pinhole", "camera_matrix":[[800,0,640],[0,800,360],[0,0,1]],
                "distortion_coefficients":[-0.2,0.04,0.001,-0.001,0]}

    def test_rectification_returns_pixels_and_roundtrips_distortion(self):
        lens = LensModel(self.parameters())
        points = [[200,200],[900,500],[640,360]]
        distorted = lens.distort(points)
        np.testing.assert_allclose(lens.rectify(distorted), points, atol=1e-7)
        np.testing.assert_allclose(lens.rectify([[640,360]]), [[640,360]], atol=1e-8)

    def test_same_lens_transform_is_applied_to_reference_and_target_points(self):
        config, truth = synthetic_config()
        lens = LensModel(self.parameters())
        config = replace(config, lens=self.parameters(),
                         image_points=lens.distort(config.image_points).tolist(),
                         check_image_points=lens.distort(config.check_image_points).tolist())
        mapper = RoadMapper(config)
        pixels, _ = project_points(truth, [[3,10]])
        raw = lens.distort(pixels)[0]
        result = mapper.map_point(*raw, camera_id=config.camera_id, image_size=config.image_size)
        np.testing.assert_allclose(result.ground_position_m, [3,10], atol=1e-5)
        self.assertEqual(mapper.report["distortion_status"], "opencv_pinhole_corrected")

    def test_no_lens_is_explicitly_uncorrected_and_bad_models_are_rejected(self):
        self.assertEqual(LensModel().status, "uncorrected")
        for parameters in ({"model":"fisheye"}, {**self.parameters(),"distortion_coefficients":[0,0]},
                           {**self.parameters(),"camera_matrix":[[0,0,640],[0,800,360],[0,0,1]]}):
            with self.assertRaises(ValueError):
                LensModel(parameters)


class ArtifactAndPipelineTests(unittest.TestCase):
    def test_bad_target_fields_and_resolution_do_not_export_partial_frame(self):
        mapper = example_mapper()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, change in (("bbox", {"bbox_xyxy":[1,2,3]}), ("score", {"confidence":True}),
                                 ("anchor", {"anchor_xy":[640,0]}), ("prediction", {"observation":"predicted"})):
                record=track_frame()
                record["tracks"][0].update(change)
                source=root/f"{name}.jsonl"
                save_frames(source,[record])
                with self.subTest(name=name), self.assertRaises(ValueError):
                    map_tracks(source,mapper,camera_id=mapper.config.camera_id,output=root/name,allow_synthetic=True)
                summary=json.loads((root/name/"summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["frames_processed"],0)
            record=track_frame(); record["width"]=640
            source=root/"resolution.jsonl"; save_frames(source,[record])
            with self.assertRaisesRegex(ValueError,"分辨率"):
                map_tracks(source,mapper,camera_id=mapper.config.camera_id,output=root/"resolution",allow_synthetic=True)

    def test_synthetic_track_source_cannot_inherit_measured_calibration_validity(self):
        mapper=example_mapper(source_type="measured")
        self.assertTrue(mapper.calibration_valid)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/"tracks.jsonl"
            record=track_frame(); record["synthetic"]=True
            save_frames(source,[record])
            output=map_tracks(source,mapper,camera_id=mapper.config.camera_id,output=root/"out",allow_synthetic=True)
            point=json.loads((output/"ground_tracks.jsonl").read_text(encoding="utf-8"))["tracks"][0]
            self.assertTrue(point["mapping_valid"] and point["synthetic"])
            self.assertFalse(point["calibration_valid"])

    def test_interrupt_closes_files_and_records_completed_frames(self):
        mapper=example_mapper()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/"tracks.jsonl"
            save_frames(source,[track_frame(0,0),track_frame(1,.1)])
            first=mapper.map_points([track_frame()["tracks"][0]["anchor_xy"]],
                                    camera_id=mapper.config.camera_id,image_size=mapper.config.image_size)
            with patch.object(mapper,"map_points",side_effect=[first,KeyboardInterrupt]):
                output=map_tracks(source,mapper,camera_id=mapper.config.camera_id,output=root/"out",allow_synthetic=True)
            summary=json.loads((output/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"],"interrupted")
            self.assertEqual(summary["frames_processed"],1)
            self.assertEqual(len((output/"ground_tracks.jsonl").read_text(encoding="utf-8").splitlines()),1)

    def test_saved_derived_validity_is_recomputed_and_matrix_edits_are_rejected(self):
        mapper = example_mapper()
        artifact = resolved_document(mapper)
        artifact["validation"]["calibration_valid"] = True
        self.assertFalse(mapping_from_document(artifact).calibration_valid)
        artifact["image_to_ground_h"][0][0] += 1
        with self.assertRaises(ValueError):
            mapping_from_document(artifact)

    def test_old_four_point_format_and_source_guards_remain_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"legacy.yaml"
            old = RoadCalibrator([[100,100],[700,100],[700,500],[100,500]],6,20,synthetic=True)
            old.to_yaml(path)
            before = path.read_bytes()
            mapper = import_legacy(path,camera_id="legacy",image_size=[800,600])
            self.assertTrue(mapper.synthetic)
            self.assertFalse(mapper.calibration_valid)
            self.assertEqual(path.read_bytes(), before)
            path.write_text(path.read_text(encoding="utf-8").replace("synthetic: true\n",""),encoding="utf-8")
            with self.assertWarns(UserWarning), self.assertRaises(ValueError):
                import_legacy(path,camera_id="legacy",image_size=[800,600])

    def test_stream_export_keeps_empty_frames_ids_times_and_original_coordinates(self):
        mapper = example_mapper()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/"tracks.jsonl"
            frames = [track_frame(0,0),track_frame(1,0.15,False),track_frame(2,0.4)]
            save_frames(source,frames)
            output = map_tracks(source,mapper,camera_id=mapper.config.camera_id,output=root/"out",allow_synthetic=True)
            records = [json.loads(line) for line in (output/"ground_tracks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records),3)
            self.assertEqual(records[1]["tracks"],[])
            self.assertEqual([f["timestamp_s"] for f in records],[0,0.15,0.4])
            point = records[0]["tracks"][0]
            self.assertEqual(point["track_id"],7)
            self.assertEqual(point["segment_id"],2)
            self.assertEqual(point["anchor_xy"],frames[0]["tracks"][0]["anchor_xy"])
            np.testing.assert_allclose(point["ground_position_m"],[3,10],atol=1e-5)
            self.assertNotIn("speed_kmh",point)
            with (output/"ground_tracks.csv").open(encoding="utf-8",newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),2)
            self.assertNotIn(None,rows[0])
            self.assertEqual(rows[0]["calibration_valid"],"False")

    def test_anomalous_time_preserves_completed_partial_output_and_error_summary(self):
        mapper = example_mapper()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/"bad.jsonl"
            save_frames(source,[track_frame(0,1),track_frame(1,1)])
            with self.assertRaisesRegex(ValueError,"递增"):
                map_tracks(source,mapper,camera_id=mapper.config.camera_id,output=root/"out",allow_synthetic=True)
            summary=json.loads((root/"out"/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"],"failed")
            self.assertEqual(summary["frames_processed"],1)
            self.assertEqual(len((root/"out"/"ground_tracks.jsonl").read_text(encoding="utf-8").splitlines()),1)

    def test_synthetic_opt_in_and_camera_check_happen_before_creating_output(self):
        mapper=example_mapper()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/"tracks.jsonl"
            save_frames(source,[track_frame()])
            for camera,allowed in ((mapper.config.camera_id,False),("wrong",True)):
                with self.assertRaises(ValueError):
                    map_tracks(source,mapper,camera_id=camera,output=root/"out",allow_synthetic=allowed)
                self.assertFalse((root/"out").exists())

    def test_smoothed_input_selection_is_explicit_and_never_automatic(self):
        mapper=example_mapper()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/"motion.jsonl"
            record=track_frame()
            config,truth=synthetic_config()
            smoothed,_=project_points(truth,[[4,10]])
            record["tracks"][0]["smoothed_anchor_xy"]=smoothed[0].tolist()
            save_frames(source,[record])
            for selection,expected_x in (("raw",3),("smoothed",4)):
                output=map_tracks(source,mapper,camera_id=config.camera_id,output=root/selection,
                                  allow_synthetic=True,position_source=selection)
                target=json.loads((output/"ground_tracks.jsonl").read_text(encoding="utf-8"))["tracks"][0]
                self.assertAlmostEqual(target["ground_position_m"][0],expected_x,places=4)

    def test_visualization_and_demo_use_no_real_dataset_or_display(self):
        mapper=example_mapper()
        image=np.zeros((720,1280,3),dtype=np.uint8)
        rendered=render_grid(mapper,image)
        self.assertFalse(image.any())
        self.assertTrue(rendered.any())
        with tempfile.TemporaryDirectory() as temporary,patch("sys.stdout",new_callable=io.StringIO):
            output=Path(temporary)/"demo"
            main(["demo","--output",str(output)])
            report=json.loads((output/"demo_checks.json").read_text(encoding="utf-8"))
            self.assertEqual(report["frames"],12)
            self.assertEqual(report["observations"],24)
            self.assertFalse(report["real_camera_validated"])
            self.assertLess(report["max_synthetic_position_error_m"],1e-4)


if __name__ == "__main__":
    unittest.main()
