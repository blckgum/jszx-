# -*- coding: utf-8 -*-
"""圆角 OBB Minkowski 求交、整车 OBB 栅格化 PET 与 HSV 外观辅助缝合单元测试。"""

from __future__ import annotations

import math
import sys
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from analysis.conflict import ConflictAnalyzer, RiskLevel
from analysis.policy import POLICIES
from analysis.speed import SpeedReading
from analysis.ssm import (
    PETTracker,
    compute_obb_swept_ttc,
)
from analysis.stitching import (
    TrackObservation,
    TrackletStitcher,
    appearance_bhattacharyya_distance,
    extract_appearance_histogram,
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
    length_m: float | None = None,
    width_m: float | None = None,
) -> SpeedReading:
    """构造测试用物理状态读数，可选附加显式车体尺寸。"""
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
    if length_m is not None:
        reading.length_m = length_m
    if width_m is not None:
        reading.width_m = width_m
    return reading


def _solid_bgr_patch(
    bgr: tuple[int, int, int],
    *,
    height: int = 80,
    width: int = 80,
    border_bgr: tuple[int, int, int] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """生成带可选外圈干扰背景边框的合成 BGR 测试帧与检测框。"""
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    x1, y1, x2, y2 = 40, 40, 40 + width, 40 + height
    if border_bgr is not None:
        frame[y1:y2, x1:x2] = border_bgr
        # 中心 60% 区域填充目标真实颜色，验证内缩裁剪能避开外圈 20% 背景干扰
        mx = int(round(0.2 * width))
        my = int(round(0.2 * height))
        frame[y1 + my : y2 - my, x1 + mx : x2 - mx] = bgr
    else:
        frame[y1:y2, x1:x2] = bgr
    return frame, [float(x1), float(y1), float(x2), float(y2)]


class TestRoundedOBBMinkowski(unittest.TestCase):
    def test_corner_grazing_distinguishes_rounded_from_sharp_box(self):
        # 核心带裕度半宽/半长：hx_core = 2.25 + 0.4 = 2.65, hy_core = 0.95 + 0.3 = 1.25
        # 圆角半径 ped_r = 0.4 -> 原直角包围盒外扩角点在 (3.05, 1.65)
        # 取位于原直角角部内部、但距圆心 (2.65, 1.25) 距离为 sqrt(0.32^2 + 0.32^2) ≈ 0.4525 > 0.4 的斜切线轨迹
        cx_core = 2.65
        cy_core = 1.25
        ped_r = 0.4
        offset = 0.32  # 0.8 * r -> sqrt(2)*0.32 = 0.4525 > 0.4（直角误报区）
        # 相对轨迹沿 (-1, 1) 斜掠过角部，t=1.0 时最接近角点 (cx_core + offset, cy_core + offset)
        res_miss = compute_obb_swept_ttc(
            veh_xy=(0.0, 0.0),
            veh_v=(1.0, -1.0),
            heading_rad=0.0,
            veh_length_m=4.5,
            veh_width_m=1.9,
            ped_xy=(cx_core + offset + 1.0, cy_core + offset - 1.0),
            ped_v=(0.0, 0.0),
            ped_radius_m=ped_r,
            horizon_s=2.0,
        )
        self.assertFalse(res_miss.collides)
        self.assertIsNone(res_miss.t_enter)
        self.assertIsNone(res_miss.t_exit)
        # 物理净距应等于相对线段到裸车角 (2.25, 0.95) 的最小欧氏距离减 ped_r
        # 直线 p(t) = (cx_core + offset + 1 - t, cy_core + offset - 1 + t) 距 (2.25, 0.95) 最近点在 t=1.05
        t_bare_star = 1.05
        px_star = cx_core + offset + 1.0 - t_bare_star
        py_star = cy_core + offset - 1.0 + t_bare_star
        expected_bare_dist = math.hypot(px_star - 2.25, py_star - 0.95) - ped_r
        self.assertAlmostEqual(res_miss.min_clearance_m, expected_bare_dist, places=6)
        self.assertAlmostEqual(res_miss.t_min_clearance, t_bare_star, places=6)

        # 相对静止位于角部 ~0.41r 误报带内时同样不误报碰撞
        res_static_corner = compute_obb_swept_ttc(
            veh_xy=(0.0, 0.0),
            veh_v=(0.0, 0.0),
            heading_rad=0.0,
            veh_length_m=4.5,
            veh_width_m=1.9,
            ped_xy=(cx_core + 0.31, cy_core + 0.31),
            ped_v=(0.0, 0.0),
            ped_radius_m=ped_r,
            horizon_s=2.0,
        )
        self.assertFalse(res_static_corner.collides)

        # 真正侵入角部圆 (dist_to_corner = sqrt(0.25^2 + 0.25^2) = 0.3536 < 0.4) 时准确触发碰撞并解析求交
        res_hit = compute_obb_swept_ttc(
            veh_xy=(0.0, 0.0),
            veh_v=(1.0, -1.0),
            heading_rad=0.0,
            veh_length_m=4.5,
            veh_width_m=1.9,
            ped_xy=(cx_core + 0.25 + 1.0, cy_core + 0.25 - 1.0),
            ped_v=(0.0, 0.0),
            ped_radius_m=ped_r,
            horizon_s=2.0,
        )
        self.assertTrue(res_hit.collides)
        self.assertIsNotNone(res_hit.t_enter)
        self.assertIsNotNone(res_hit.t_exit)
        # 圆 (cx_core, cy_core, 0.4) 与直线上最近点距离 d_perp = 0.25 * sqrt(2)，弦半长 sqrt(0.4^2 - d_perp^2) / sqrt(2)
        d_perp = 0.25 * math.sqrt(2.0)
        half_dt = math.sqrt(ped_r * ped_r - d_perp * d_perp) / math.sqrt(2.0)
        self.assertAlmostEqual(res_hit.t_enter, 1.0 - half_dt, places=6)
        self.assertAlmostEqual(res_hit.t_exit, 1.0 + half_dt, places=6)

    def test_rotated_corner_clearance_precision_and_zero_radius(self):
        # 旋转 37 度后校验角部擦过的旋转不变性与物理净距精度
        theta = math.radians(37.0)
        c, s = math.cos(theta), math.sin(theta)

        def _to_world(lx: float, ly: float) -> tuple[float, float]:
            return (c * lx - s * ly + 10.0, s * lx + c * ly - 5.0)

        ped_r = 0.5
        # 局部轨迹从 (2.65 + 0.4 + 2.0, 1.25 + 0.4) 沿 (-2, 0) 经过角点附近，最近距角圆心 y 偏移 0.4 -> 刚好在直角内、圆外
        # 改为经过局部 (2.65 + 0.39, 1.25 + 0.39) 斜线以保持物理净距 > 0
        p0_w = _to_world(2.65 + 0.39 + 1.0, 1.25 + 0.39 - 1.0)
        v_w = (c * (-1.0) - s * 1.0, s * (-1.0) + c * 1.0)
        res = compute_obb_swept_ttc(
            veh_xy=(10.0, -5.0),
            veh_v=(0.0, 0.0),
            heading_rad=theta,
            veh_length_m=4.5,
            veh_width_m=1.9,
            ped_xy=p0_w,
            ped_v=v_w,
            ped_radius_m=ped_r,
            horizon_s=2.0,
        )
        self.assertFalse(res.collides)
        ts = np.linspace(0.0, 2.0, 40001)
        lpx, lpy = res.local_pos_xy
        lvx, lvy = res.local_vel_xy
        px = lpx + lvx * ts
        py = lpy + lvy * ts
        d_car = np.hypot(np.maximum(0.0, np.abs(px) - 2.25), np.maximum(0.0, np.abs(py) - 0.95))
        dists = np.maximum(0.0, d_car - ped_r)
        self.assertAlmostEqual(res.min_clearance_m, float(dists.min()), delta=1e-4)

        # ped_radius_m == 0.0 时退化为精确带裕度核心矩形并无除零错误
        res_zero_r = compute_obb_swept_ttc(
            veh_xy=(0.0, 0.0),
            veh_v=(2.0, 0.0),
            heading_rad=0.0,
            veh_length_m=4.5,
            veh_width_m=1.9,
            ped_xy=(4.65, 0.5),
            ped_v=(0.0, 0.0),
            ped_radius_m=0.0,
            horizon_s=2.0,
        )
        self.assertTrue(res_zero_r.collides)
        self.assertAlmostEqual(res_zero_r.t_enter, 1.0, places=6)

        # ConflictAnalyzer(obb_ssm) 在斜角擦过角外安全带时不再产生误报警
        analyzer = ConflictAnalyzer(conflict_model="obb_ssm")
        car = _mk_reading(1, "vehicle", 0.0, 0.0, 1.0, -1.0, object_kind="car", heading_deg=0.0)
        ped = _mk_reading(
            2, "pedestrian", 2.65 + 0.32 + 1.0, 1.25 + 0.32 - 1.0, 0.0, 0.0, object_kind="pedestrian"
        )
        self.assertIsNone(analyzer.evaluate_pair(car, ped))


class TestVehicleOBBRasterizedPET(unittest.TestCase):
    def test_bus_side_rear_corner_sweep_triggers_rasterized_pet_order_independent(self):
        # 公交车 (10.0m x 2.55m) 先在左侧 x=2.3 行驶（不触及 x=4 列），随后向右上变道甩尾，
        # 其右侧后角扫过行人刚离开的 (4.2, 4.5) 所在 cell(4, 4)；中心点始终在 x<=3.1 不入 (4, 4)，
        # 必须依靠整车 OBB 栅格化才能捕获该擦碰险情，且读数换序完全一致。
        results = []
        for order in ("ped_first", "bus_first"):
            tracker = PETTracker(cell_size_m=1.0, max_pet_s=4.0, retention_s=8.0)
            ped_t0 = _mk_reading(2, "pedestrian", 4.2, 4.5, 4.0, 0.0, object_kind="pedestrian")
            bus_t0 = _mk_reading(1, "vehicle", 2.3, 1.0, 0.0, 8.0, object_kind="bus", heading_deg=90.0)
            ped_t1 = _mk_reading(2, "pedestrian", 6.2, 4.5, 4.0, 0.0, object_kind="pedestrian")
            bus_t1 = _mk_reading(1, "vehicle", 2.3, 5.0, 0.0, 8.0, object_kind="bus", heading_deg=90.0)
            ped_t2 = _mk_reading(2, "pedestrian", 8.2, 4.5, 4.0, 0.0, object_kind="pedestrian")
            # t=1.0s 时大巴向右转向航向 105 度，车尾右侧角扫入 cell(4, 4)
            bus_t2 = _mk_reading(1, "vehicle", 3.1, 8.5, 1.6, 7.0, object_kind="bus", heading_deg=105.0)

            seq = (
                [([ped_t0, bus_t0], 0.0), ([ped_t1, bus_t1], 0.5), ([ped_t2, bus_t2], 1.0)]
                if order == "ped_first"
                else [([bus_t0, ped_t0], 0.0), ([bus_t1, ped_t1], 0.5), ([bus_t2, ped_t2], 1.0)]
            )
            res = {}
            for items, ts in seq:
                res = tracker.update(items, ts, object_policies=POLICIES)
            results.append(res)

        self.assertIn((1, 2), results[0])
        self.assertEqual(results[0], results[1])
        self.assertGreater(results[0][(1, 2)], 0.3)
        self.assertLess(results[0][(1, 2)], 1.0)

        # 未传 object_policies 且无显式尺寸时，同一场景保持点栅格兼容（center x<=3.1 不触及 cell x=4）
        point_tracker = PETTracker(cell_size_m=1.0, max_pet_s=4.0, retention_s=8.0)
        ped_t0 = _mk_reading(2, "pedestrian", 4.2, 4.5, 4.0, 0.0, object_kind="pedestrian")
        bus_t0 = _mk_reading(1, "vehicle", 2.3, 1.0, 0.0, 8.0, object_kind="bus", heading_deg=90.0)
        ped_t1 = _mk_reading(2, "pedestrian", 6.2, 4.5, 4.0, 0.0, object_kind="pedestrian")
        bus_t1 = _mk_reading(1, "vehicle", 2.3, 5.0, 0.0, 8.0, object_kind="bus", heading_deg=90.0)
        ped_t2 = _mk_reading(2, "pedestrian", 8.2, 4.5, 4.0, 0.0, object_kind="pedestrian")
        bus_t2 = _mk_reading(1, "vehicle", 3.1, 8.5, 1.6, 7.0, object_kind="bus", heading_deg=105.0)
        point_tracker.update([ped_t0, bus_t0], 0.0)
        point_tracker.update([ped_t1, bus_t1], 0.5)
        self.assertEqual(point_tracker.update([ped_t2, bus_t2], 1.0), {})

    def test_rotated_explicit_size_simultaneous_overlap_and_retention_pruning(self):
        # 显式携带 length_m/width_m 与旋转 45 度航向：行人在大巴车尾侧角格内同时重叠应返回 PET=0.0s
        for order in ("ped_first", "bus_first"):
            tracker = PETTracker(cell_size_m=1.0, max_pet_s=4.0, retention_s=2.0)
            bus_t0 = _mk_reading(
                1, "vehicle", 0.0, 0.0, 0.0, 0.0,
                object_kind="bus", heading_deg=45.0, length_m=10.0, width_m=2.55,
            )
            # 沿局部 (-4.0, -1.0) 映射到世界坐标，位于大巴左后角实体内部
            c, s = math.cos(math.radians(45.0)), math.sin(math.radians(45.0))
            px = c * (-4.0) - s * (-1.0)
            py = s * (-4.0) + c * (-1.0)
            ped_t0 = _mk_reading(2, "pedestrian", px, py, 0.0, 0.0, object_kind="pedestrian")
            items = [ped_t0, bus_t0] if order == "ped_first" else [bus_t0, ped_t0]
            res = tracker.update(items, 0.0)
            self.assertEqual(res, {(1, 2): 0.0})

            # 超过 retention_s=2.0 后，全部 OBB 占格记录被清理，控制内存有界
            self.assertEqual(tracker.update([], 2.5), {})
            self.assertEqual(len(tracker._cell_visits), 0)
            self.assertEqual(len(tracker._last_track_pos), 0)
            self.assertEqual(len(tracker._last_track_heading), 0)

    def test_conflict_analyzer_passes_object_policies_to_pet_tracker(self):
        analyzer = ConflictAnalyzer(conflict_model="obb_ssm", object_policies=POLICIES)
        ped_t0 = _mk_reading(2, "pedestrian", 4.1, 4.5, 3.0, 0.0, object_kind="pedestrian")
        bus_t0 = _mk_reading(1, "vehicle", 3.0, 1.0, 0.0, 10.0, object_kind="bus", heading_deg=90.0)
        analyzer.analyze_frame([bus_t0, ped_t0], timestamp_s=0.0)

        ped_t1 = _mk_reading(2, "pedestrian", 6.5, 4.5, 3.0, 0.0, object_kind="pedestrian")
        bus_t1 = _mk_reading(1, "vehicle", 3.0, 9.0, 0.0, 10.0, object_kind="bus", heading_deg=90.0)
        res = analyzer.analyze_frame([bus_t1, ped_t1], timestamp_s=0.8)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].veh_track_id, 1)
        self.assertEqual(res[0].ped_track_id, 2)
        self.assertIsNotNone(res[0].pet_seconds)
        self.assertGreaterEqual(res[0].risk_level, RiskLevel.LEVEL_2_CAUTION)


class TestAppearanceStitching(unittest.TestCase):
    def test_histogram_extraction_inset_and_bhattacharyya_distance(self):
        # 中心红色 (BGR 0,0,255) + 外围 20% 绿色干扰背景 (BGR 0,255,0)，验证内缩 60% 忽略边缘背景
        frame_red_bordered, bbox = _solid_bgr_patch((0, 0, 255), border_bgr=(0, 255, 0))
        frame_red_clean, _ = _solid_bgr_patch((0, 0, 255))
        frame_blue, _ = _solid_bgr_patch((255, 0, 0))

        hist_bordered = extract_appearance_histogram(frame_red_bordered, bbox)
        hist_clean = extract_appearance_histogram(frame_red_clean, bbox)
        hist_blue = extract_appearance_histogram(frame_blue, bbox)

        self.assertIsNotNone(hist_bordered)
        self.assertIsNotNone(hist_clean)
        self.assertIsNotNone(hist_blue)
        self.assertEqual(hist_bordered.shape, (16, 8))
        self.assertAlmostEqual(float(hist_bordered.sum()), 1.0, places=6)
        self.assertTrue(np.all(hist_bordered >= 0.0))

        # 与 OpenCV HISTCMP_BHATTACHARYYA 口径一致，值域 [0.0, 1.0]
        d_same = appearance_bhattacharyya_distance(hist_bordered, hist_clean)
        d_diff = appearance_bhattacharyya_distance(hist_clean, hist_blue)
        self.assertIsNotNone(d_same)
        self.assertIsNotNone(d_diff)
        self.assertAlmostEqual(d_same, 0.0, places=6)
        self.assertAlmostEqual(d_diff, 1.0, places=6)
        cv_dist = cv2.compareHist(
            hist_clean.astype(np.float32), hist_blue.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA
        )
        self.assertAlmostEqual(d_diff, float(cv_dist), places=6)

        # 无效或尺寸过小输入安全返回 None
        self.assertIsNone(extract_appearance_histogram(None, bbox))
        self.assertIsNone(extract_appearance_histogram(np.zeros((64, 64), dtype=np.uint8), bbox))
        self.assertIsNone(extract_appearance_histogram(frame_red_clean, [10, 10, 13, 13]))
        self.assertIsNone(extract_appearance_histogram(frame_red_clean, [50, 50, 20, 60]))
        self.assertIsNone(extract_appearance_histogram(frame_red_clean, [-30, -30, -5, -5]))
        self.assertIsNone(extract_appearance_histogram(frame_red_clean, [10, 10, 40, True]))
        self.assertIsNone(extract_appearance_histogram(frame_red_clean, [10, 10, float("nan"), 40]))

        self.assertIsNone(appearance_bhattacharyya_distance(None, hist_clean))
        self.assertIsNone(appearance_bhattacharyya_distance(hist_clean, np.zeros((16, 8))))
        self.assertIsNone(appearance_bhattacharyya_distance(hist_clean, np.ones((8, 4))))
        self.assertIsNone(appearance_bhattacharyya_distance(hist_clean, np.full((16, 8), float("nan"))))

    def test_stitcher_disambiguation_gating_ema_and_none_fallback(self):
        for bad_kw in (
            {"max_appearance_dist": 0.0},
            {"max_appearance_dist": 1.2},
            {"max_appearance_dist": True},
            {"appearance_weight": -0.5},
            {"appearance_weight": float("nan")},
        ):
            with self.assertRaises(ValueError):
                TrackletStitcher(**bad_kw)

        frame_red, bbox = _solid_bgr_patch((0, 0, 255))
        frame_blue, _ = _solid_bgr_patch((255, 0, 0))
        frame_orange, _ = _solid_bgr_patch((0, 128, 255))
        hist_red = extract_appearance_histogram(frame_red, bbox)
        hist_blue = extract_appearance_histogram(frame_blue, bbox)
        hist_orange = extract_appearance_histogram(frame_orange, bbox)
        assert hist_red is not None and hist_blue is not None and hist_orange is not None

        # 1) 双车并行遮挡后位置小幅交错：仅靠运动学容易错配，外观直方图精确消歧（支持 6 元组、dict、TrackObservation）
        st = TrackletStitcher(
            max_gap_s=1.5,
            max_distance_m=3.5,
            max_mahalanobis=4.0,
            max_appearance_dist=0.55,
            appearance_weight=2.0,
        )
        obs_t0 = [
            (1, "vehicle", "car", 2.0, 0.0, hist_red),
            {"raw_track_id": 2, "category": "vehicle", "object_kind": "car", "x_m": 4.0, "y_m": 0.0, "appearance_hist": hist_blue},
        ]
        id_map, pairs = st.resolve_frame(obs_t0, 0.0)
        self.assertEqual(id_map, {1: 1, 2: 2})
        self.assertEqual(pairs, [])

        # 活跃帧混合橘红直方图验证 EMA 更新，且 None 帧不丢失已有直方图
        st.resolve_frame(
            [
                TrackObservation(1, "vehicle", "car", 2.0, 2.0, hist_orange),
                (2, "vehicle", "car", 4.0, 2.0, None),
            ],
            0.25,
        )
        tr1_hist = st._active_tracks[1].appearance_hist
        self.assertIsNotNone(tr1_hist)
        self.assertAlmostEqual(float(tr1_hist.sum()), 1.0, places=6)
        self.assertGreater(appearance_bhattacharyya_distance(tr1_hist, hist_red), 0.0)
        self.assertIsNotNone(st._active_tracks[2].appearance_hist)

        st.update_kinematics(
            1, SimpleNamespace(x_m=2.0, y_m=2.0, vx_ms=0.0, vy_ms=8.0, category="vehicle", object_kind="car"), 0.25
        )
        st.update_kinematics(
            2, SimpleNamespace(x_m=4.0, y_m=2.0, vx_ms=0.0, vy_ms=8.0, category="vehicle", object_kind="car"), 0.25
        )

        # 遮挡 0.4s，重新出现时空间坐标偏向互换，但外观门控/融合确保红车归 1、蓝车归 2
        st.resolve_frame([], 0.45)
        id_map_re, pairs_re = st.resolve_frame(
            [
                (11, "vehicle", "car", 2.7, 5.2, hist_blue),
                (12, "vehicle", "car", 3.3, 5.2, hist_red),
            ],
            0.65,
        )
        self.assertEqual(id_map_re, {11: 2, 12: 1})
        self.assertEqual(set(pairs_re), {(11, 2), (12, 1)})

        # 2) 外观差异超过 max_appearance_dist 门限时，即便物理位置零误差也拒绝错误缝合
        st_gate = TrackletStitcher(max_gap_s=1.0, max_distance_m=3.5, max_appearance_dist=0.55)
        st_gate.resolve_frame([(1, "vehicle", "car", 3.0, 1.0, hist_red)], 0.0)
        st_gate.resolve_frame([], 0.2)
        id_map_rej, pairs_rej = st_gate.resolve_frame(
            [(5, "vehicle", "car", 3.0, 1.0, hist_blue)], 0.4
        )
        self.assertEqual(pairs_rej, [])
        self.assertEqual(id_map_rej, {5: 5})

        # 3) 无视频/appearance_hist=None 时 100% 退化为纯运动学（同坐标可正常缝合，且结果与旧 5 元组完全一致）
        st_none = TrackletStitcher()
        st_legacy = TrackletStitcher()
        seq_none = [
            ([(1, "vehicle", "car", 3.0, 0.0, None)], 0.0),
            ([(1, "vehicle", "car", 3.0, 2.0, None)], 0.25),
            ([], 0.45),
            ([(5, "vehicle", "car", 3.0, 5.2, None)], 0.65),
        ]
        seq_legacy = [
            ([(1, "vehicle", "car", 3.0, 0.0)], 0.0),
            ([(1, "vehicle", "car", 3.0, 2.0)], 0.25),
            ([], 0.45),
            ([(5, "vehicle", "car", 3.0, 5.2)], 0.65),
        ]
        for (obs_a, ts_a), (obs_b, ts_b) in zip(seq_none, seq_legacy):
            self.assertEqual(st_none.resolve_frame(obs_a, ts_a), st_legacy.resolve_frame(obs_b, ts_b))
        self.assertEqual(st_none.stitch_events, st_legacy.stitch_events)
        self.assertEqual(st_none.stitched_track_count, 1)

        # 确认全程未引入 scipy
        self.assertNotIn("scipy", sys.modules)


if __name__ == "__main__":
    unittest.main()
