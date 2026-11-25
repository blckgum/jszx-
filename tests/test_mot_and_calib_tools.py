# -*- coding: utf-8 -*-
"""MOT 多目标跟踪评测器与交互式多点/停车区/棋盘格标定扩展工具单元测试。"""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import yaml

from analysis import evaluate_mot, evaluate_mot_files
from analysis.calibration import RoadCalibrator
from analysis.interactive_calibrator import (
    calibrate_camera_chessboard,
    load_intrinsics_yaml,
    main as calib_main,
    run_interactive_calibration,
)
from analysis.mot_eval import (
    load_mot_tracks,
    main as mot_main,
    mot_report_to_markdown,
)


def _render_chessboard_view(
    board_img: np.ndarray,
    k_mat: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    square_size_m: float,
    cell_px: int,
    out_size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """将标准黑白棋盘格纹理按给定相机外参透视投影到图像平面（供测试动态生成标定图）。"""
    h_b2w = np.array(
        [
            [square_size_m / cell_px, 0.0, -square_size_m],
            [0.0, square_size_m / cell_px, -square_size_m],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rot, _ = cv2.Rodrigues(rvec)
    proj = k_mat @ np.column_stack([rot[:, :2], tvec])
    h_b2i = proj @ h_b2w
    return cv2.warpPerspective(board_img, h_b2i, out_size, borderValue=(200, 200, 200))


class TestMotEval(unittest.TestCase):
    def test_perfect_tracking_metrics_are_unity(self):
        gt = {
            0: [
                {"track_id": 1, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
                {"track_id": 2, "bbox_xyxy": [100.0, 20.0, 140.0, 80.0]},
            ],
            1: [
                {"track_id": 1, "bbox_xyxy": [12.0, 10.0, 52.0, 50.0]},
                {"track_id": 2, "bbox_xyxy": [102.0, 20.0, 142.0, 80.0]},
            ],
            2: [
                {"track_id": 1, "bbox_xyxy": [14.0, 10.0, 54.0, 50.0]},
                {"track_id": 2, "bbox_xyxy": [104.0, 20.0, 144.0, 80.0]},
            ],
        }
        pred = {
            0: [
                {"track_id": 10, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
                {"track_id": 20, "bbox_xyxy": [100.0, 20.0, 140.0, 80.0]},
            ],
            1: [
                {"track_id": 10, "bbox_xyxy": [12.0, 10.0, 52.0, 50.0]},
                {"track_id": 20, "bbox_xyxy": [102.0, 20.0, 142.0, 80.0]},
            ],
            2: [
                {"track_id": 10, "bbox_xyxy": [14.0, 10.0, 54.0, 50.0]},
                {"track_id": 20, "bbox_xyxy": [104.0, 20.0, 144.0, 80.0]},
            ],
        }
        for use_lap in (True, False):
            with self.subTest(use_lap=use_lap):
                res = evaluate_mot(gt, pred, iou_threshold=0.5, _use_lap=use_lap)
                self.assertEqual(res["gt_objects"], 6)
                self.assertEqual(res["pred_objects"], 6)
                self.assertEqual(res["tp"], 6)
                self.assertEqual(res["fp"], 0)
                self.assertEqual(res["fn"], 0)
                self.assertEqual(res["id_switches"], 0)
                self.assertEqual(res["fragmentations"], 0)
                self.assertEqual((res["mt"], res["pt"], res["ml"]), (2, 0, 0))
                self.assertAlmostEqual(res["mota"], 1.0, places=6)
                self.assertAlmostEqual(res["motp"], 1.0, places=6)
                self.assertAlmostEqual(res["idf1"], 1.0, places=6)
                self.assertAlmostEqual(res["hota"], 1.0, places=6)
                self.assertAlmostEqual(res["deta"], 1.0, places=6)
                self.assertAlmostEqual(res["assa"], 1.0, places=6)

    def test_clear_mot_maximizes_match_cardinality_before_iou_sum(self):
        gt_boxes = [[i * 3.2, 0.0, i * 3.2 + 10.0, 10.0] for i in range(5)]
        pred_boxes = [[(i - 1) * 3.2, 0.0, (i - 1) * 3.2 + 10.0, 10.0] for i in range(5)]
        gt = {0: [{"track_id": i + 1, "bbox_xyxy": b} for i, b in enumerate(gt_boxes)]}
        pred = {0: [{"track_id": i + 11, "bbox_xyxy": b} for i, b in enumerate(pred_boxes)]}
        for use_lap in (True, False):
            with self.subTest(use_lap=use_lap):
                res = evaluate_mot(gt, pred, iou_threshold=0.5, _use_lap=use_lap)
                self.assertEqual((res["tp"], res["fp"], res["fn"]), (5, 0, 0))
                self.assertAlmostEqual(res["mota"], 1.0, places=6)

    def test_clear_mot_preserves_valid_previous_frame_identity_matches(self):
        # 首帧严格对齐：GT1=P1(x=0..10), GT2=P2(x=2..12)
        # 次帧两目标靠近交叉：原身份 IoU=(8/12)=0.667 > 0.5，交叉 IoU=1.0；CLEAR 应延续原配对实现 IDSW=0, MOTA=1.0
        gt = {
            0: [
                {"track_id": 1, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
                {"track_id": 2, "bbox_xyxy": [2.0, 0.0, 12.0, 10.0]},
            ],
            1: [
                {"track_id": 1, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
                {"track_id": 2, "bbox_xyxy": [2.0, 0.0, 12.0, 10.0]},
            ],
        }
        pred = {
            0: [
                {"track_id": 1, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
                {"track_id": 2, "bbox_xyxy": [2.0, 0.0, 12.0, 10.0]},
            ],
            1: [
                {"track_id": 1, "bbox_xyxy": [2.0, 0.0, 12.0, 10.0]},
                {"track_id": 2, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
            ],
        }
        for use_lap in (True, False):
            with self.subTest(use_lap=use_lap):
                res = evaluate_mot(gt, pred, iou_threshold=0.5, _use_lap=use_lap)
                self.assertEqual((res["tp"], res["fp"], res["fn"]), (4, 0, 0))
                self.assertEqual(res["id_switches"], 0)
                self.assertAlmostEqual(res["mota"], 1.0, places=6)

        # 非连续帧间隔（跳帧 0 -> 2）不得把 t-2 旧匹配锁作上一相邻帧延续
        gt_gap = {0: gt[0], 2: gt[1]}
        pred_gap = {0: pred[0], 2: pred[1]}
        res_gap = evaluate_mot(gt_gap, pred_gap, iou_threshold=0.5, _use_lap=False)
        self.assertEqual(res_gap["id_switches"], 2)

    def test_hota_matches_trackeval_reference_on_ambiguous_frame(self):
        # 4 个 10x10 框存在歧义重叠，核对 TrackEval 两遍式 HOTA/DetA/AssA 数值
        gt = {
            0: [
                {"track_id": 1, "bbox_xyxy": [10.0, 0.0, 20.0, 10.0]},
                {"track_id": 2, "bbox_xyxy": [14.0, 0.0, 24.0, 10.0]},
            ]
        }
        pred = {
            0: [
                {"track_id": 1, "bbox_xyxy": [11.0, 0.0, 21.0, 10.0]},
                {"track_id": 2, "bbox_xyxy": [6.0, 0.0, 16.0, 10.0]},
            ]
        }
        for use_lap in (True, False):
            with self.subTest(use_lap=use_lap):
                res = evaluate_mot(gt, pred, _use_lap=use_lap)
                self.assertAlmostEqual(res["deta"], 0.350877, places=5)
                self.assertAlmostEqual(res["assa"], 0.842105, places=5)
                self.assertAlmostEqual(res["hota"], 0.530679, places=5)

    def test_single_id_switch_penalizes_idsw_idf1_and_hota(self):
        # 单条 10 帧真值轨迹：前 6 帧分配 ID=1，后 4 帧发生 1 次 ID 切换变为 ID=2（框完全重合）
        gt = {f: [{"track_id": 1, "bbox_xyxy": [0.0, 0.0, 20.0, 20.0]}] for f in range(10)}
        pred = {
            f: [{"track_id": 1 if f < 6 else 2, "bbox_xyxy": [0.0, 0.0, 20.0, 20.0]}]
            for f in range(10)
        }
        res_lap = evaluate_mot(gt, pred, _use_lap=True)
        res_py = evaluate_mot(gt, pred, _use_lap=False)
        self.assertEqual(res_lap, res_py)

        self.assertEqual(res_lap["tp"], 10)
        self.assertEqual(res_lap["fp"], 0)
        self.assertEqual(res_lap["fn"], 0)
        self.assertEqual(res_lap["id_switches"], 1)
        self.assertEqual(res_lap["fragmentations"], 0)
        self.assertAlmostEqual(res_lap["mota"], 0.9, places=6)
        self.assertAlmostEqual(res_lap["motp"], 1.0, places=6)
        # 全局 ID 匹配仅能选最长段 (ID=1, 6 帧)，故 id_tp=6, idf1=12/20=0.6
        self.assertEqual(res_lap["id_tp"], 6)
        self.assertEqual(res_lap["id_fp"], 4)
        self.assertEqual(res_lap["id_fn"], 4)
        self.assertAlmostEqual(res_lap["idf1"], 0.6, places=6)
        self.assertLess(res_lap["idf1"], 1.0)
        # HOTA：DetA=1.0, AssA=(6*(6/10)+4*(4/10))/10=0.52, HOTA=sqrt(0.52)
        self.assertAlmostEqual(res_lap["deta"], 1.0, places=6)
        self.assertAlmostEqual(res_lap["assa"], 0.52, places=6)
        self.assertAlmostEqual(res_lap["hota"], math.sqrt(0.52), places=5)
        self.assertLess(res_lap["hota"], 1.0)

    def test_missed_detections_false_positives_fragmentation_and_mt_pt_ml(self):
        # 3 条真值轨迹各 5 帧：
        # - GT 1: 5 帧全匹配 (MT)
        # - GT 2: 帧 0,1 匹配, 帧 2 漏检, 帧 3,4 匹配 -> 4/5=80% (MT) 且触发 1 次 fragmentation
        # - GT 3: 帧 0 偏移匹配 (IoU < 0.5 未命中), 仅帧 1 命中 1/5=20% (PT)？改为 0 帧命中 -> ML
        gt = {}
        pred = {}
        for f in range(5):
            gt[f] = [
                {"track_id": 1, "bbox_xyxy": [0.0, 0.0, 20.0, 20.0]},
                {"track_id": 2, "bbox_xyxy": [50.0, 0.0, 70.0, 20.0]},
                {"track_id": 3, "bbox_xyxy": [100.0, 0.0, 120.0, 20.0]},
            ]
            p_frame = [{"track_id": 1, "bbox_xyxy": [0.0, 0.0, 20.0, 20.0]}]
            if f != 2:
                p_frame.append({"track_id": 2, "bbox_xyxy": [50.0, 0.0, 70.0, 20.0]})
            if f == 0:
                # 1 个纯误检 FP
                p_frame.append({"track_id": 99, "bbox_xyxy": [300.0, 300.0, 320.0, 320.0]})
            pred[f] = p_frame

        # 再加 GT 4 (共 5 帧命中 2 帧 = 40% -> PT)
        for f in range(5):
            gt[f].append({"track_id": 4, "bbox_xyxy": [200.0, 0.0, 220.0, 20.0]})
            if f in (0, 1):
                pred[f].append({"track_id": 4, "bbox_xyxy": [200.0, 0.0, 220.0, 20.0]})

        res = evaluate_mot(gt, pred, iou_threshold=0.5)
        # GT 共 20 框；Pred 匹配: GT1(5) + GT2(4) + GT4(2) = 11 TP, 1 FP (id=99), 9 FN
        self.assertEqual(res["gt_objects"], 20)
        self.assertEqual(res["pred_objects"], 12)
        self.assertEqual(res["tp"], 11)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 9)
        self.assertEqual(res["id_switches"], 0)
        self.assertEqual(res["fragmentations"], 1)
        # GT1(5/5>=80%), GT2(4/5>=80%) -> MT=2; GT4(2/5=40%) -> PT=1; GT3(0/5<20%) -> ML=1
        self.assertEqual((res["mt"], res["pt"], res["ml"]), (2, 1, 1))
        self.assertAlmostEqual(res["mota"], 1.0 - (1 + 9 + 0) / 20.0, places=6)

        # 验证 pure-Python Kuhn-Munkres 与 lap.lapjv 在稀疏非方阵上一致
        with mock.patch("analysis.mot_eval._lap", None):
            res_fallback = evaluate_mot(gt, pred, iou_threshold=0.5)
        self.assertEqual(res, res_fallback)

    def test_load_mot_txt_and_tracks_jsonl_alignment_category_filter_and_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # 1) 标准 1-based MOTChallenge TXT（含空行与 # 注释行），帧号 1, 2
            gt_txt = root / "gt.txt"
            gt_txt.write_text(
                "# MOTChallenge 1-based ground truth\n"
                "\n"
                "1, 1, 10.0, 20.0, 40.0, 60.0, 1, pedestrian, 1.0\n"
                "1, 2, 100.0, 20.0, 50.0, 50.0, 1, car, 1.0\n"
                "2, 1, 12.0, 20.0, 40.0, 60.0, 1, pedestrian, 1.0\n"
                "2, 2, 104.0, 20.0, 50.0, 50.0, 1, car, 1.0\n",
                encoding="utf-8",
            )

            # 2) 仓库 0-based tracks.jsonl，帧号 0, 1（还包含一个未分配 track_id=None 的纯检测框）
            pred_jsonl = root / "tracks.jsonl"
            rows = [
                {
                    "schema_version": 1,
                    "session_id": "s1",
                    "source": "demo.mp4",
                    "frame_index": 0,
                    "timestamp_s": 0.0,
                    "timestamp_source": "fps",
                    "width": 640,
                    "height": 480,
                    "tracks": [
                        {
                            "track_id": 11,
                            "class_id": 0,
                            "class_name": "person",
                            "category": "pedestrian",
                            "confidence": 0.9,
                            "bbox_xyxy": [10.0, 20.0, 50.0, 80.0],
                            "anchor_xy": [30.0, 80.0],
                        },
                        {
                            "track_id": 12,
                            "class_id": 2,
                            "class_name": "car",
                            "category": "vehicle",
                            "confidence": 0.85,
                            "bbox_xyxy": [100.0, 20.0, 150.0, 70.0],
                            "anchor_xy": [125.0, 70.0],
                        },
                        {
                            "track_id": None,
                            "class_id": 2,
                            "class_name": "car",
                            "category": "vehicle",
                            "confidence": 0.4,
                            "bbox_xyxy": [300.0, 300.0, 320.0, 320.0],
                            "anchor_xy": [310.0, 320.0],
                        },
                    ],
                },
                {
                    "schema_version": 1,
                    "session_id": "s1",
                    "source": "demo.mp4",
                    "frame_index": 1,
                    "timestamp_s": 0.04,
                    "timestamp_source": "fps",
                    "width": 640,
                    "height": 480,
                    "tracks": [
                        {
                            "track_id": 11,
                            "class_id": 0,
                            "class_name": "person",
                            "category": "pedestrian",
                            "confidence": 0.92,
                            "bbox_xyxy": [12.0, 20.0, 52.0, 80.0],
                            "anchor_xy": [32.0, 80.0],
                        },
                        {
                            "track_id": 12,
                            "class_id": 2,
                            "class_name": "car",
                            "category": "vehicle",
                            "confidence": 0.88,
                            "bbox_xyxy": [104.0, 20.0, 154.0, 70.0],
                            "anchor_xy": [129.0, 70.0],
                        },
                    ],
                },
            ]
            pred_jsonl.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )

            # 自动将 1-based gt.txt 对齐至 0-based tracks.jsonl
            loaded_gt = load_mot_tracks(gt_txt)
            self.assertEqual(sorted(loaded_gt.keys()), [0, 1])

            # 显式关闭 1-based 归一化则保留帧号 1, 2
            loaded_gt_raw = load_mot_tracks(gt_txt, one_based=False)
            self.assertEqual(sorted(loaded_gt_raw.keys()), [1, 2])

            report = evaluate_mot_files(gt_txt, pred_jsonl)
            self.assertEqual(report["gt_objects"], 4)
            self.assertEqual(report["pred_objects"], 4)
            self.assertAlmostEqual(report["mota"], 1.0, places=6)
            self.assertAlmostEqual(report["idf1"], 1.0, places=6)
            self.assertAlmostEqual(report["hota"], 1.0, places=6)

            # 按 pedestrian 类别过滤后仅保留 2 帧行人目标
            ped_report = evaluate_mot_files(gt_txt, pred_jsonl, category_filter="pedestrian")
            self.assertEqual(ped_report["gt_objects"], 2)
            self.assertEqual(ped_report["pred_objects"], 2)
            self.assertAlmostEqual(ped_report["mota"], 1.0, places=6)

            # CLI JSON 与 Markdown 写出校验（禁止 NaN）
            out_json = root / "mot_report.json"
            out_md = root / "mot_report.md"
            self.assertEqual(
                mot_main([
                    "--gt", str(gt_txt),
                    "--pred", str(pred_jsonl),
                    "--category", "pedestrian",
                    "--output", str(out_json),
                ]),
                0,
            )
            parsed = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertAlmostEqual(parsed["mota"], 1.0, places=6)

            self.assertEqual(
                mot_main([
                    "--gt", str(gt_txt),
                    "--pred", str(pred_jsonl),
                    "--markdown",
                    "--output", str(out_md),
                ]),
                0,
            )
            md_text = out_md.read_text(encoding="utf-8")
            self.assertIn("# MOT Evaluation Report", md_text)
            self.assertIn("HOTA", md_text)
            self.assertIn("IDF1", mot_report_to_markdown(report))

    def test_mot_eval_rejects_blank_paths_bad_thresholds_and_malformed_inputs(self):
        for bad_path in ("", "   ", Path("   "), None, True, float("nan")):
            with self.subTest(bad_path=bad_path):
                with self.assertRaises(ValueError):
                    load_mot_tracks(bad_path)
                with self.assertRaises(ValueError):
                    evaluate_mot_files(bad_path, "dummy.jsonl")

        for bad_thresh in (0.0, -0.1, 1.01, True, False, float("nan"), float("inf")):
            with self.subTest(bad_thresh=bad_thresh):
                with self.assertRaises(ValueError):
                    evaluate_mot({}, {}, iou_threshold=bad_thresh)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_nan_txt = root / "bad_nan.txt"
            bad_nan_txt.write_text("1, 1, nan, 20, 30, 40\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mot_tracks(bad_nan_txt)

            bad_size_txt = root / "bad_size.txt"
            bad_size_txt.write_text("1, 1, 10, 20, 0, 40\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mot_tracks(bad_size_txt)

            bad_dup_txt = root / "bad_dup.txt"
            bad_dup_txt.write_text("1, 1, 10, 20, 30, 40\n1, 1, 15, 25, 30, 40\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mot_tracks(bad_dup_txt)

            bad_jsonl = root / "bad.jsonl"
            bad_jsonl.write_text(
                '{"frame_index": 0, "tracks": [{"track_id": 1, "bbox_xyxy": [0, 0, NaN, 10]}]}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_mot_tracks(bad_jsonl)

            # CLI 空白路径与非法阈值拒斥
            good_txt = root / "ok.txt"
            good_txt.write_text("1, 1, 10, 20, 30, 40\n", encoding="utf-8")
            for bad_argv in (
                ["--gt", "   ", "--pred", str(good_txt)],
                ["--gt", str(good_txt), "--pred", ""],
                ["--gt", str(good_txt), "--pred", str(good_txt), "--output", "   "],
                ["--gt", str(good_txt), "--pred", str(good_txt), "--iou-threshold", "0"],
                ["--gt", str(good_txt), "--pred", str(good_txt), "--iou-threshold", "1.5"],
            ):
                with self.subTest(bad_argv=bad_argv):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            mot_main(bad_argv)


class TestInteractiveCalibratorExtensions(unittest.TestCase):
    def setUp(self):
        self.base_img = [
            (400.0, 300.0),
            (880.0, 300.0),
            (1150.0, 680.0),
            (130.0, 680.0),
        ]
        self.ref_cal = RoadCalibrator(self.base_img, 6.0, 15.0)
        self.camera_matrix = [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ]
        self.dist_coeffs = [-0.22, 0.06, 0.001, -0.001, 0.0]

    def _project_ground_points(self, gd_pts):
        img_pts = []
        for gx, gy in gd_pts:
            p = self.ref_cal.H_inv @ np.array([gx, gy, 1.0], dtype=np.float64)
            img_pts.append((float(p[0] / p[2]), float(p[1] / p[2])))
        return img_pts

    def test_cli_multipoint_ransac_and_parking_zones(self):
        gd_pts = [
            (0.0, 0.0),
            (6.0, 0.0),
            (6.0, 15.0),
            (0.0, 15.0),
            (3.0, 7.5),
            (1.5, 11.0),
        ]
        img_pts = self._project_ground_points(gd_pts)
        # 在第 6 个地面点注入显著外点（仍在相机前方视场内），验证默认开启 RANSAC 能准确剔除而不被拉偏
        gd_corrupted = list(gd_pts)
        gd_corrupted[5] = (4.5, 11.0)

        # 基于理想点投影构造两个地面停车区对应的像素多边形
        zone1_gd = [(0.5, 2.0), (2.0, 2.0), (2.0, 5.0), (0.5, 5.0)]
        zone2_gd = [(4.0, 8.0), (5.5, 8.0), (5.5, 11.0), (4.0, 11.0)]
        zone1_px = self._project_ground_points(zone1_gd)
        zone2_px = self._project_ground_points(zone2_gd)

        pts_str = ",".join(f"{u:.4f},{v:.4f}" for u, v in img_pts)
        gd_str = ",".join(f"{x:.4f},{y:.4f}" for x, y in gd_corrupted)
        z1_str = ",".join(f"{u:.4f},{v:.4f}" for u, v in zone1_px)
        z2_str = ",".join(f"{u:.4f},{v:.4f}" for u, v in zone2_px)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_path = root / "校园路面.png"
            ok, enc = cv2.imencode(".png", np.zeros((720, 1280, 3), dtype=np.uint8))
            self.assertTrue(ok)
            enc.tofile(str(img_path))

            out_ransac = root / "calib_ransac.yaml"
            self.assertEqual(
                calib_main([
                    "--source", str(img_path),
                    "--points", pts_str,
                    "--ground-points", gd_str,
                    "--parking-zone", z1_str,
                    "--parking-zone", z2_str,
                    "--output", str(out_ransac),
                ]),
                0,
            )
            loaded = RoadCalibrator.from_yaml(out_ransac)
            self.assertEqual(loaded.inlier_count, 5)
            self.assertLess(loaded.reprojection_rmse_m, 1e-3)
            self.assertAlmostEqual(loaded.width_m, 6.0, places=5)
            self.assertAlmostEqual(loaded.length_m, 15.0, places=5)
            self.assertEqual(len(loaded.parking_zones), 2)
            self.assertTrue(loaded.is_in_parking_zone(1.2, 3.5))
            self.assertTrue(loaded.is_in_parking_zone(4.8, 9.5))
            self.assertFalse(loaded.is_in_parking_zone(3.0, 7.5))

            # 对比 --no-ransac：最小二乘会被外点拉偏且保留全部 6 个点
            out_ls = root / "calib_ls.yaml"
            self.assertEqual(
                calib_main([
                    "--source", str(img_path),
                    "--points", pts_str,
                    "--ground-points", gd_str,
                    "--no-ransac",
                    "--output", str(out_ls),
                ]),
                0,
            )
            loaded_ls = RoadCalibrator.from_yaml(out_ls)
            self.assertEqual(loaded_ls.inlier_count, 6)
            self.assertGreater(loaded_ls.reprojection_rmse_m, 1.0)

    def test_chessboard_calibration_and_intrinsics_injection(self):
        cols, rows = 6, 4
        sq_size = 0.03
        cell_px = 40
        board = np.full(((rows + 3) * cell_px, (cols + 3) * cell_px, 3), 255, dtype=np.uint8)
        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    y0 = (r + 1) * cell_px
                    x0 = (c + 1) * cell_px
                    board[y0:y0 + cell_px, x0:x0 + cell_px] = 0

        k_true = np.array([[780.0, 0.0, 320.0], [0.0, 780.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        views = [
            (np.array([[0.10], [0.05], [0.02]]), np.array([[-0.10], [-0.07], [0.50]])),
            (np.array([[-0.12], [0.08], [-0.05]]), np.array([[-0.11], [-0.06], [0.55]])),
            (np.array([[0.05], [-0.15], [0.10]]), np.array([[-0.09], [-0.08], [0.60]])),
            (np.array([[-0.08], [-0.10], [-0.08]]), np.array([[-0.10], [-0.07], [0.52]])),
            (np.array([[0.15], [-0.05], [0.04]]), np.array([[-0.10], [-0.07], [0.58]])),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cb_dir = root / "棋盘格目录"
            cb_dir.mkdir()
            cb_paths = []
            for idx, (rvec, tvec) in enumerate(views):
                img = _render_chessboard_view(board, k_true, rvec, tvec, sq_size, cell_px)
                p = cb_dir / f"视图_{idx}.png"
                ok, enc = cv2.imencode(".png", img)
                self.assertTrue(ok)
                enc.tofile(str(p))
                cb_paths.append(p)

            # 1) 直接测试 calibrate_camera_chessboard 函数（含中文路径）
            res = calibrate_camera_chessboard(cb_paths, board_size=(cols, rows), square_size_m=sq_size)
            self.assertEqual(res["used_images"], 5)
            self.assertEqual(res["total_images"], 5)
            self.assertEqual(res["image_size"], [640, 480])
            self.assertLess(res["reprojection_rms_px"], 0.5)
            self.assertAlmostEqual(res["camera_matrix"][0][0], 780.0, delta=30.0)
            self.assertEqual(len(res["dist_coeffs"]), 5)

            # 2) 失败帧不足 (< 3) 必须明确报错
            bad_dir = root / "bad_cb"
            bad_dir.mkdir()
            bad_paths = []
            for idx in range(2):
                p = bad_dir / f"blank_{idx}.png"
                ok, enc = cv2.imencode(".png", np.zeros((480, 640, 3), dtype=np.uint8))
                self.assertTrue(ok)
                enc.tofile(str(p))
                bad_paths.append(p)
            with self.assertRaises(ValueError):
                calibrate_camera_chessboard(bad_paths, board_size=(cols, rows), square_size_m=sq_size)

            # 3) 仅传 --chessboard-dir + --output（不传 --source）导出内参 YAML
            intr_yaml = root / "intrinsics.yaml"
            self.assertEqual(
                calib_main([
                    "--chessboard-dir", str(cb_dir),
                    "--board-size", f"{cols},{rows}",
                    "--square-size", str(sq_size),
                    "--output", str(intr_yaml),
                ]),
                0,
            )
            self.assertTrue(intr_yaml.is_file())
            intr_data = yaml.safe_load(intr_yaml.read_text(encoding="utf-8"))
            self.assertEqual(intr_data["used_images"], 5)
            loaded_k, loaded_d = load_intrinsics_yaml(intr_yaml)
            self.assertEqual(loaded_k, intr_data["camera_matrix"])
            self.assertEqual(loaded_d, intr_data["dist_coeffs"])

            # 4) --source + --intrinsics-yaml + --points 自动注入相机内参与畸变系数
            road_img = root / "road.png"
            ok, enc_road = cv2.imencode(".png", np.zeros((720, 1280, 3), dtype=np.uint8))
            self.assertTrue(ok)
            enc_road.tofile(str(road_img))

            calib_from_intr = root / "road_with_intrinsics.yaml"
            self.assertEqual(
                calib_main([
                    "--source", str(road_img),
                    "--width", "6.0",
                    "--length", "15.0",
                    "--points", "400,300,880,300,1150,680,130,680",
                    "--intrinsics-yaml", str(intr_yaml),
                    "--output", str(calib_from_intr),
                ]),
                0,
            )
            cal_intr = RoadCalibrator.from_yaml(calib_from_intr)
            self.assertEqual(cal_intr.camera_matrix, loaded_k)
            self.assertEqual(cal_intr.dist_coeffs, loaded_d)

            # 5) --source + --chessboard-dir + 多点 --ground-points + --parking-zone 全流程联动
            calib_full = root / "road_full.yaml"
            gd5 = [(0.0, 0.0), (6.0, 0.0), (6.0, 15.0), (0.0, 15.0), (3.0, 7.5)]
            px5 = self._project_ground_points(gd5)
            self.assertEqual(
                calib_main([
                    "--source", str(road_img),
                    "--points", ",".join(f"{u:.3f},{v:.3f}" for u, v in px5),
                    "--ground-points", ",".join(f"{x:.3f},{y:.3f}" for x, y in gd5),
                    "--parking-zone", "420,340,620,340,650,480,390,480",
                    "--chessboard-dir", str(cb_dir),
                    "--board-size", f"{cols},{rows}",
                    "--square-size", str(sq_size),
                    "--output", str(calib_full),
                ]),
                0,
            )
            cal_full = RoadCalibrator.from_yaml(calib_full)
            self.assertIsNotNone(cal_full.camera_matrix)
            self.assertIsNotNone(cal_full.dist_coeffs)
            self.assertEqual(len(cal_full.parking_zones), 1)

    def test_calibrator_extensions_reject_invalid_cli_and_blank_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            road_img = root / "road.png"
            ok, enc = cv2.imencode(".png", np.zeros((720, 1280, 3), dtype=np.uint8))
            self.assertTrue(ok)
            enc.tofile(str(road_img))
            out_yaml = root / "out.yaml"

            for bad_argv in (
                ["--source", "   ", "--output", str(out_yaml)],
                ["--source", str(road_img), "--output", "   "],
                ["--source", str(road_img), "--chessboard-dir", "   ", "--output", str(out_yaml)],
                ["--source", str(road_img), "--intrinsics-yaml", "   ", "--output", str(out_yaml)],
                ["--source", str(road_img), "--validation-output", "   ", "--output", str(out_yaml)],
                # 既无 --source 也无 --chessboard-dir
                ["--output", str(out_yaml)],
                # --chessboard-dir 与 --intrinsics-yaml 互斥
                ["--source", str(road_img), "--chessboard-dir", str(root), "--intrinsics-yaml", str(out_yaml), "--output", str(out_yaml)],
                # --ground-points 少于 4 对坐标
                ["--source", str(road_img), "--ground-points", "0,0,6,0,6,15", "--output", str(out_yaml)],
                # --points 与 --ground-points 点数不一致
                ["--source", str(road_img), "--points", "400,300,880,300,1150,680,130,680", "--ground-points", "0,0,6,0,6,15,0,15,3,7.5", "--output", str(out_yaml)],
                # --parking-zone 少于 3 个顶点
                ["--source", str(road_img), "--points", "400,300,880,300,1150,680,130,680", "--parking-zone", "100,100,200,200", "--output", str(out_yaml)],
                # --board-size 非法值
                ["--chessboard-dir", str(root), "--board-size", "2,6", "--output", str(out_yaml)],
            ):
                with self.subTest(bad_argv=bad_argv):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            calib_main(bad_argv)

            for bad_blank in ("", "   ", Path("   "), None, True):
                with self.subTest(bad_blank=bad_blank):
                    with self.assertRaises(ValueError):
                        load_intrinsics_yaml(bad_blank)
                    with self.assertRaises(ValueError):
                        run_interactive_calibration(bad_blank, 6.0, 15.0, save_output=False)


if __name__ == "__main__":
    unittest.main()
