# -*- coding: utf-8 -*-
"""
相机空间标定与地面物理坐标单应性变换模块 (Camera Calibration & Homography Mapping)
=============================================================================
将图像中斜向下俯视的像素坐标 (u, v)，通过 4 点地面标定射影变换矩阵 H，
准确映射到校园道路的实际物理平面坐标 (X, Y)（单位：米）。
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


def _is_finite_real(value) -> bool:
    """判断输入是否为非布尔的有限实数（兼容 Python 与 NumPy 整数/浮点标量）。"""
    # Python 中 bool 继承自 int，需先显式排除 bool 与 np.bool_
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _validate_finite_float(value, label: str) -> float:
    """校验有限浮点数，显式拒绝布尔值、NaN 和 Inf。"""
    if type(value) is bool or isinstance(value, np.bool_):
        raise ValueError(f"{label}不能为布尔值")
    if not _is_finite_real(value):
        raise ValueError(f"{label}必须为有限数值")
    return float(value)


def _compute_ground_bounds(
    points: List[Tuple[float, float]], label: str = "ground_points"
) -> Tuple[float, float, float, float, float, float]:
    """计算二维地面点包络 (min_x, max_x, min_y, max_y, width_m, length_m) 并校验非退化跨度。"""
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width_m = max_x - min_x
    length_m = max_y - min_y
    if (
        not (math.isfinite(width_m) and math.isfinite(length_m))
        or width_m <= 1e-6
        or length_m <= 1e-6
    ):
        raise ValueError(f"{label} 的物理跨度必须为正数（不可全部共线或重合）")
    return float(min_x), float(max_x), float(min_y), float(max_y), float(width_m), float(length_m)


def _validate_point_2d(pt, label: str) -> Tuple[float, float]:
    is_seq = isinstance(pt, (list, tuple)) or (
        isinstance(pt, np.ndarray) and pt.ndim == 1
    )
    if not is_seq or len(pt) != 2:
        raise ValueError(f"{label}必须是长度为 2 的坐标对")
    x, y = pt[0], pt[1]
    return (
        _validate_finite_float(x, f"{label} X "),
        _validate_finite_float(y, f"{label} Y "),
    )


def _validate_camera_matrix(
    camera_matrix,
) -> Tuple[List[List[float]], np.ndarray, np.ndarray]:
    """校验 3x3 相机内参矩阵 K 并返回标准列表、K 与 K^-1。"""
    if isinstance(camera_matrix, np.ndarray):
        if camera_matrix.ndim != 2 or camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix 必须是 3x3 矩阵")
        if camera_matrix.dtype == bool or np.issubdtype(camera_matrix.dtype, np.bool_):
            raise ValueError("camera_matrix 不能包含布尔值")
        rows = camera_matrix.tolist()
    elif isinstance(camera_matrix, (list, tuple)):
        if len(camera_matrix) != 3:
            raise ValueError("camera_matrix 必须是 3x3 矩阵")
        rows = camera_matrix
    else:
        raise ValueError("camera_matrix 必须是 3x3 矩阵")

    parsed: List[List[float]] = []
    for r_idx, row in enumerate(rows):
        is_row_seq = isinstance(row, (list, tuple)) or (
            isinstance(row, np.ndarray) and row.ndim == 1
        )
        if not is_row_seq or len(row) != 3:
            raise ValueError(f"camera_matrix 第 {r_idx + 1} 行长度必须为 3")
        parsed.append([
            _validate_finite_float(row[c_idx], f"camera_matrix[{r_idx}][{c_idx}] ")
            for c_idx in range(3)
        ])

    k_np = np.array(parsed, dtype=np.float64)
    if k_np[0, 0] <= 0.0 or k_np[1, 1] <= 0.0 or k_np[2, 2] <= 0.0:
        raise ValueError("camera_matrix 焦距与齐次缩放对角项必须大于 0")
    if abs(float(np.linalg.det(k_np))) <= 1e-9:
        raise ValueError("camera_matrix 奇异不可逆")
    k_inv = np.linalg.inv(k_np)
    if not np.isfinite(k_inv).all():
        raise ValueError("camera_matrix 求逆包含非有限数值")
    return parsed, k_np, k_inv


def _validate_dist_coeffs(dist_coeffs) -> Tuple[List[float], np.ndarray]:
    """校验 OpenCV 镜头畸变系数（长度必须为 4/5/8/12/14 且全为有限数值）。"""
    valid_lengths = (4, 5, 8, 12, 14)
    if isinstance(dist_coeffs, np.ndarray):
        if dist_coeffs.ndim != 1:
            raise ValueError("dist_coeffs 必须是一维序列")
        if dist_coeffs.dtype == bool or np.issubdtype(dist_coeffs.dtype, np.bool_):
            raise ValueError("dist_coeffs 不能包含布尔值")
        seq = dist_coeffs.tolist()
    elif isinstance(dist_coeffs, (list, tuple)):
        seq = dist_coeffs
    else:
        raise ValueError("dist_coeffs 必须是列表或一维数组")

    if len(seq) not in valid_lengths:
        raise ValueError(
            f"dist_coeffs 长度必须为 {valid_lengths} 之一，当前为 {len(seq)}"
        )
    parsed = [
        _validate_finite_float(val, f"dist_coeffs[{idx}] ")
        for idx, val in enumerate(seq)
    ]
    return parsed, np.array(parsed, dtype=np.float64)


def _homography_jacobian(H: np.ndarray, u: float, v: float) -> Optional[np.ndarray]:
    """计算平面射影变换 H 在点 (u, v) 处的解析 2x2 雅可比矩阵。"""
    p_in = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        p_out = H @ p_in
        w = float(p_out[2, 0])
        if not math.isfinite(w) or abs(w) < 1e-8:
            return None
        x = float(p_out[0, 0] / w)
        y = float(p_out[1, 0] / w)
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        jac = np.array(
            [
                [(H[0, 0] - x * H[2, 0]) / w, (H[0, 1] - x * H[2, 1]) / w],
                [(H[1, 0] - y * H[2, 0]) / w, (H[1, 1] - y * H[2, 1]) / w],
            ],
            dtype=np.float64,
        )
    if not np.isfinite(jac).all():
        return None
    return jac


def _validate_polygon_2d(poly, name: str = "多边形") -> List[Tuple[float, float]]:
    if not isinstance(poly, (list, tuple)) or len(poly) < 3:
        raise ValueError(f"{name}必须至少包含 3 个二维顶点")
    return [_validate_point_2d(pt, f"{name}第 {idx + 1} 个顶点") for idx, pt in enumerate(poly)]


def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    if not (_is_finite_real(x) and _is_finite_real(y)) or len(polygon) < 3:
        return False
    # 统一转为 Python float，确保 NumPy 标量（如 np.float32）可正常参与几何判定
    px, py = float(x), float(y)
    n = len(polygon)
    # 先做边段最近点距离检查：落在多边形边界上（1e-9 米容差内）一律视为在区域内，
    # 避免射线法在顶点或水平边上的浮点边界歧义。
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-18:
            if math.hypot(px - x1, py - y1) <= 1e-9:
                return True
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
            if math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)) <= 1e-9:
                return True
    # 标准射线法（Ray-Casting）：仅当边跨越水平线 y 时才求交点，因此必有 yj != yi，不会除零。
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@dataclass
class CalibrationConfig:
    """标定参数配置数据契约"""
    name: str
    width_m: float
    length_m: float
    # 四个图像像素角点：左上、右上、右下、左下 [(u1,v1), (u2,v2), (u3,v3), (u4,v4)]
    image_points: List[Tuple[float, float]]
    # 真实道路米制坐标：[(0, 0), (width, 0), (width, length), (0, length)]
    ground_points: List[Tuple[float, float]]
    homography_matrix: List[List[float]]
    inv_homography_matrix: List[List[float]]
    # 与 RoadCalibrator 对齐的溯源字段：合成占位标记与旧版缺字段标记
    synthetic: bool = False
    synthetic_field_missing: bool = False
    camera_matrix: Optional[List[List[float]]] = None
    dist_coeffs: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)


class RoadCalibrator:
    """道路空间几何单应性标定与坐标转换引擎"""

    @staticmethod
    def _validate_common_params(
        name: str,
        synthetic: bool,
        parking_zones: Optional[List[List[Tuple[float, float]]]],
        camera_matrix: Optional[List[List[float]]],
        dist_coeffs: Optional[List[float]],
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("标定名称 name 必须是非空字符串")
        if type(synthetic) is not bool:
            raise ValueError("synthetic 必须是布尔值")

        parsed_parking_zones: List[List[Tuple[float, float]]] = []
        if parking_zones is not None:
            if not isinstance(parking_zones, (list, tuple)):
                raise ValueError("parking_zones 必须是多边形列表")
            for idx, zone in enumerate(parking_zones):
                parsed_parking_zones.append(
                    _validate_polygon_2d(zone, name=f"第 {idx + 1} 个停车区")
                )

        cam_list = None
        cam_np = None
        cam_inv_np = None
        if camera_matrix is not None:
            cam_list, cam_np, cam_inv_np = _validate_camera_matrix(camera_matrix)

        dist_list = None
        dist_np = None
        if dist_coeffs is not None:
            if cam_np is None:
                raise ValueError("提供 dist_coeffs 时必须同时提供 3x3 camera_matrix")
            dist_list, dist_np = _validate_dist_coeffs(dist_coeffs)

        return parsed_parking_zones, cam_list, cam_np, cam_inv_np, dist_list, dist_np

    def _init_intrinsics_attrs(
        self,
        cam_list: Optional[List[List[float]]],
        cam_np: Optional[np.ndarray],
        cam_inv_np: Optional[np.ndarray],
        dist_list: Optional[List[float]],
        dist_np: Optional[np.ndarray],
    ):
        self.camera_matrix: Optional[List[List[float]]] = cam_list
        self.dist_coeffs: Optional[List[float]] = dist_list
        self._camera_matrix_np: Optional[np.ndarray] = cam_np
        self._camera_matrix_inv_np: Optional[np.ndarray] = cam_inv_np
        self._dist_coeffs_np: Optional[np.ndarray] = dist_np

    def _undistort_uv(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """若存在相机内参与畸变参数，将原始像素 (u, v) 去畸变为针孔像素坐标。"""
        if self._camera_matrix_np is None or self._dist_coeffs_np is None:
            return float(u), float(v)
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-13)
        try:
            undist = cv2.undistortPoints(
                pts,
                self._camera_matrix_np,
                self._dist_coeffs_np,
                P=self._camera_matrix_np,
                criteria=criteria,
            )
        except cv2.error:
            return None
        u_p = float(undist[0, 0, 0])
        v_p = float(undist[0, 0, 1])
        if not (math.isfinite(u_p) and math.isfinite(v_p)):
            return None
        return u_p, v_p

    def _undistort_points_array(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """批量将像素坐标列表去畸变，返回 (N, 2) float64 数组。"""
        pts = np.array(points, dtype=np.float64).reshape(-1, 2)
        if self._camera_matrix_np is None or self._dist_coeffs_np is None:
            return pts
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-13)
        undist = cv2.undistortPoints(
            pts.reshape(-1, 1, 2),
            self._camera_matrix_np,
            self._dist_coeffs_np,
            P=self._camera_matrix_np,
            criteria=criteria,
        )
        out = undist.reshape(-1, 2).astype(np.float64)
        if not np.isfinite(out).all():
            raise ValueError("像素去畸变后包含非有限数值，请检查相机内参或畸变系数")
        return out

    def _finalize_homography(
        self,
        h_mat: Optional[np.ndarray],
        cheirality_ground_points: List[Tuple[float, float]],
    ):
        """校验单应性矩阵可逆性、条件数与相机前方 cheirality 齐次符号。"""
        if h_mat is None or h_mat.shape != (3, 3) or not np.isfinite(h_mat).all():
            raise ValueError("无法求解有效的单应性矩阵，请检查标定点是否退化共线")

        self.H = h_mat.astype(np.float64)
        if not np.isfinite(self.H).all():
            raise ValueError("单应性矩阵包含非有限数值")
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            raise ValueError(
                "单应性矩阵不可逆，请检查标定点是否退化共线或重复选取"
            ) from None
        if not np.isfinite(self.H_inv).all():
            raise ValueError("逆单应性矩阵包含非有限数值")

        # 矩阵条件数过大意味着几何极度狭长或接近奇异，微小的像素框抖动会被放大成巨大米制误差。
        cond_h = float(np.linalg.cond(self.H))
        if not math.isfinite(cond_h) or cond_h >= 1e12:
            raise ValueError(f"单应性矩阵条件数过大 ({cond_h:.3e})，请检查标定点是否退化")

        # 记录相机前方地面对应的逆投影齐次尺度 w 符号（cheirality 参考符号）：
        # 当外推点越过地平线落到相机主平面后方时，w 会反号并把像素折返到画面上方天空，
        # 后续 ground_to_pixel_checked 据此过滤相机后方伪影点。
        w_signs: List[int] = []
        for gx, gy in cheirality_ground_points:
            w = float((self.H_inv @ np.array([gx, gy, 1.0], dtype=np.float64))[2])
            if not math.isfinite(w) or abs(w) < 1e-6:
                raise ValueError("逆单应性投影齐次尺度分量过小或非有限，请检查标定点")
            w_signs.append(1 if w > 0 else -1)
        if len(set(w_signs)) != 1:
            raise ValueError("标定地面角点齐次尺度符号不一致，无法构成有效单向视场")
        self._front_w_sign: int = w_signs[0]

    def __init__(
        self,
        image_points: List[Tuple[float, float]],
        width_m: float,
        length_m: float,
        name: str = "campus_road",
        synthetic: bool = False,
        parking_zones: Optional[List[List[Tuple[float, float]]]] = None,
        camera_matrix: Optional[List[List[float]]] = None,
        dist_coeffs: Optional[List[float]] = None,
    ):
        (
            parsed_parking_zones,
            cam_list,
            cam_np,
            cam_inv_np,
            dist_list,
            dist_np,
        ) = self._validate_common_params(
            name, synthetic, parking_zones, camera_matrix, dist_coeffs
        )
        if not isinstance(image_points, (list, tuple)) or len(image_points) != 4:
            raise ValueError(
                f"地面标定必须严格提供 4 个角点，当前提供了 {len(image_points) if hasattr(image_points, '__len__') else 0} 个"
            )
        if (
            type(width_m) not in (int, float)
            or type(length_m) not in (int, float)
            or not (math.isfinite(width_m) and math.isfinite(length_m))
            or width_m <= 0
            or length_m <= 0
        ):
            raise ValueError("道路的实际物理宽度和长度必须为大于 0 的有限正数（单位：米）")

        parsed_points: List[Tuple[float, float]] = [
            _validate_point_2d(pt, f"第 {idx + 1} 个标定角点")
            for idx, pt in enumerate(image_points)
        ]

        # 连续相邻两边的二维叉积符号必须严格一致（全正或全负）：
        # 既兼容顺时针/逆时针两种标定顺序，又能拦截共线、重合、凹四边形和蝴蝶自交叉退化。
        crosses: List[float] = []
        for i in range(4):
            x_i, y_i = parsed_points[i]
            x_n, y_n = parsed_points[(i + 1) % 4]
            x_nn, y_nn = parsed_points[(i + 2) % 4]
            cross = (x_n - x_i) * (y_nn - y_n) - (y_n - y_i) * (x_nn - x_n)
            if not math.isfinite(cross):
                raise ValueError("标定角点几何计算溢出，请检查像素坐标范围")
            crosses.append(cross)
        if not (all(c > 1e-3 for c in crosses) or all(c < -1e-3 for c in crosses)):
            raise ValueError("4 个标定角点必须构成严格凸四边形（不可共线、重复、凹陷或自交叉）")

        # 鞋带公式计算像素多边形面积，拦截极微小退化选区。
        area = 0.5 * abs(
            sum(
                parsed_points[i][0] * parsed_points[(i + 1) % 4][1]
                - parsed_points[(i + 1) % 4][0] * parsed_points[i][1]
                for i in range(4)
            )
        )
        if not math.isfinite(area) or area < 1.0:
            raise ValueError(f"4 个标定角点围成的多边形面积过小或无效 ({area})")

        self.name = name
        # 合成占位标定标记：非 --demo 模式下拒绝加载，防止演示参数被当真实标定
        self.synthetic = synthetic
        # 旧版标定文件可能缺失 synthetic 溯源字段，由 from_yaml 置位
        self.synthetic_field_missing = False
        self.source_path: Optional[str] = None
        self.width_m = float(width_m)
        self.length_m = float(length_m)
        self._zone_bounds: Tuple[float, float, float, float] = (
            0.0, self.width_m, 0.0, self.length_m
        )
        self.image_points = parsed_points
        self.parking_zones = parsed_parking_zones
        self._init_intrinsics_attrs(cam_list, cam_np, cam_inv_np, dist_list, dist_np)
        self._used_ransac = False
        self._ransac_reproj_threshold_m = 0.5

        # 定义地面标准矩形（顺时针：左上 -> 右上 -> 右下 -> 左下）
        self.ground_points = [
            (0.0, 0.0),
            (self.width_m, 0.0),
            (self.width_m, self.length_m),
            (0.0, self.length_m),
        ]

        src_pts = self._undistort_points_array(self.image_points).astype(np.float32)
        dst_pts = np.array(self.ground_points, dtype=np.float32)

        # 求解 3x3 单应性透视变换矩阵 H
        try:
            h_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        except cv2.error as exc:
            raise ValueError(f"无法求解有效的单应性矩阵: {exc}") from None

        self._finalize_homography(h_mat, self.ground_points)

        # 校验 4 个拟合角点的地面投影精度，确保 float32 求解的 H 数值稳定。
        sq_errors: List[float] = []
        for (u, v), (gx, gy) in zip(self.image_points, self.ground_points):
            px, py = self.pixel_to_ground(u, v)
            err = math.hypot(px - gx, py - gy) if (math.isfinite(px) and math.isfinite(py)) else float("nan")
            if not math.isfinite(err) or err >= 1e-2:
                raise ValueError("标定角点往返投影误差超过 1e-2 米，单应性变换数值无效")
            sq_errors.append(err * err)
        self.reprojection_rmse_m: float = math.sqrt(sum(sq_errors) / len(sq_errors))
        self.inlier_count: int = 4

    @classmethod
    def from_correspondences(
        cls,
        image_points: List[Tuple[float, float]],
        ground_points: List[Tuple[float, float]],
        *,
        name: str = "campus_road",
        synthetic: bool = False,
        parking_zones: Optional[List[List[Tuple[float, float]]]] = None,
        use_ransac: bool = False,
        ransac_reproj_threshold_m: float = 0.5,
        camera_matrix: Optional[List[List[float]]] = None,
        dist_coeffs: Optional[List[float]] = None,
    ) -> "RoadCalibrator":
        """通过 N >= 4 组对应点（最小二乘或 RANSAC）构建道路单应性标定器。"""
        (
            parsed_parking_zones,
            cam_list,
            cam_np,
            cam_inv_np,
            dist_list,
            dist_np,
        ) = cls._validate_common_params(
            name, synthetic, parking_zones, camera_matrix, dist_coeffs
        )
        if type(use_ransac) is not bool:
            raise ValueError("use_ransac 必须是布尔值")
        thresh_m = _validate_finite_float(ransac_reproj_threshold_m, "ransac_reproj_threshold_m ")
        if thresh_m <= 0.0:
            raise ValueError("ransac_reproj_threshold_m 必须为大于 0 的有限正数")

        if not isinstance(image_points, (list, tuple, np.ndarray)) or not isinstance(
            ground_points, (list, tuple, np.ndarray)
        ):
            raise ValueError("image_points 和 ground_points 必须是二维坐标点序列")
        if len(image_points) < 4:
            raise ValueError(f"至少需要 4 组对应点，当前仅提供 {len(image_points)} 组")
        if len(image_points) != len(ground_points):
            raise ValueError("image_points 与 ground_points 数量必须一致")

        parsed_img = [
            _validate_point_2d(pt, f"第 {idx + 1} 个图像对应点")
            for idx, pt in enumerate(image_points)
        ]
        parsed_gd = [
            _validate_point_2d(pt, f"第 {idx + 1} 个地面对应点")
            for idx, pt in enumerate(ground_points)
        ]

        min_x, max_x, min_y, max_y, width_m, length_m = _compute_ground_bounds(
            parsed_gd, label="ground_points"
        )

        obj = cls.__new__(cls)
        obj.name = name
        obj.synthetic = synthetic
        obj.synthetic_field_missing = False
        obj.source_path = None
        obj.width_m = width_m
        obj.length_m = length_m
        obj._zone_bounds = (min_x, max_x, min_y, max_y)
        obj.image_points = parsed_img
        obj.ground_points = parsed_gd
        obj.parking_zones = parsed_parking_zones
        obj._init_intrinsics_attrs(cam_list, cam_np, cam_inv_np, dist_list, dist_np)
        obj._used_ransac = use_ransac and len(parsed_img) > 4
        obj._ransac_reproj_threshold_m = float(thresh_m)

        src_pts = obj._undistort_points_array(parsed_img)
        dst_pts = np.array(parsed_gd, dtype=np.float64)

        try:
            if obj._used_ransac:
                h_mat, mask = cv2.findHomography(
                    src_pts,
                    dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=float(thresh_m),
                )
            else:
                h_mat, mask = cv2.findHomography(src_pts, dst_pts, method=0)
        except cv2.error as exc:
            raise ValueError(f"无法求解有效的单应性矩阵: {exc}") from None

        if obj._used_ransac and mask is not None:
            inlier_indices = [i for i, flag in enumerate(mask.ravel().tolist()) if int(flag) > 0]
            if len(inlier_indices) < 4:
                raise ValueError(
                    f"RANSAC 内点数量不足 4 个（当前 {len(inlier_indices)} 个），无法构建有效标定"
                )
        else:
            inlier_indices = list(range(len(parsed_img)))

        inlier_ground_pts = [parsed_gd[i] for i in inlier_indices]
        # 必须仅依据 RANSAC 留存内点重算物理宽高与区域边界，防止外点撑大 is_in_zone() 与配置快照
        in_min_x, in_max_x, in_min_y, in_max_y, in_width_m, in_length_m = _compute_ground_bounds(
            inlier_ground_pts, label="RANSAC 内点地面坐标"
        )
        obj.width_m = in_width_m
        obj.length_m = in_length_m
        obj._zone_bounds = (in_min_x, in_max_x, in_min_y, in_max_y)
        obj._finalize_homography(h_mat, inlier_ground_pts)

        sq_errors: List[float] = []
        for idx in inlier_indices:
            u, v = parsed_img[idx]
            gx, gy = parsed_gd[idx]
            px, py = obj.pixel_to_ground(u, v)
            if not (math.isfinite(px) and math.isfinite(py)):
                raise ValueError("标定对应点投影退化为非有限地面坐标，单应性变换无效")
            err = math.hypot(px - gx, py - gy)
            if len(parsed_img) == 4 and err >= 1e-2:
                raise ValueError("4 点标定重投影误差超过 1e-2 米，请检查对应点是否退化")
            sq_errors.append(err * err)

        obj.reprojection_rmse_m = math.sqrt(sum(sq_errors) / len(sq_errors))
        obj.inlier_count = len(inlier_indices)
        return obj

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "RoadCalibrator":
        """从已保存的 YAML 配置文件载入标定矩阵"""
        # 拒绝空白路径（含 Path("   ")），防止 Path("") 解析成当前目录 '.' 触发目录读取异常。
        if (isinstance(yaml_path, str) and not yaml_path.strip()) or (
            isinstance(yaml_path, os.PathLike)
            and isinstance(os.fspath(yaml_path), str)
            and not os.fspath(yaml_path).strip()
        ):
            raise ValueError("标定配置文件路径不能为空")
        path = Path(yaml_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"未找到标定配置文件: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"标定文件 {path} YAML 格式错误: {exc}") from None

        if not isinstance(data, dict):
            raise ValueError(f"标定文件 {path} 内容必须是键值映射")
        for req_key in ("image_points", "width_m", "length_m"):
            if req_key not in data:
                raise ValueError(f"标定文件 {path} 缺少必需字段: {req_key}")
        if not isinstance(data["image_points"], (list, tuple)):
            raise ValueError(f"标定文件 {path} 的 image_points 必须是列表")

        img_pts = [
            tuple(p) if isinstance(p, (list, tuple)) else p
            for p in data["image_points"]
        ]

        # synthetic 是布尔溯源字段而非可选字符串：
        # bool("false") == True 会把误写字符串的文件错判为占位标定；
        # 旧版文件缺失该字段时无法区分实测/合成，按实测加载但提示重新标定
        if "synthetic" in data:
            if type(data["synthetic"]) is not bool:
                raise ValueError(
                    f"标定文件 {path} 的 synthetic 字段必须是 true/false 布尔值；"
                    "请修正字段或重新标定"
                )
            synthetic = data["synthetic"]
            field_missing = False
        else:
            warnings.warn(
                f"标定文件 {path} 缺少 synthetic 溯源字段（旧版格式），"
                "已按实测标定加载；建议用 interactive_calibrator 重新标定以写入完整字段",
                UserWarning,
                stacklevel=2,
            )
            synthetic = False
            field_missing = True

        raw_zones = data.get("parking_zones")
        if raw_zones is not None and not isinstance(raw_zones, (list, tuple)):
            raise ValueError(f"标定文件 {path} 的 parking_zones 必须是列表")

        raw_name = data.get("name", "campus_road")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"标定文件 {path} 的 name 必须是非空字符串")

        raw_camera_matrix = data.get("camera_matrix")
        raw_dist_coeffs = data.get("dist_coeffs")
        # 无论进入 4 点还是 >4 点对应点分支，均先校验 width_m 与 length_m 为非 bool 有限正数
        w_val = _validate_finite_float(data["width_m"], "width_m ")
        l_val = _validate_finite_float(data["length_m"], "length_m ")
        if w_val <= 0.0 or l_val <= 0.0:
            raise ValueError("道路的实际物理宽度和长度必须为大于 0 的有限正数（单位：米）")

        raw_ground_points = data.get("ground_points")
        use_corresp = False
        if raw_ground_points is not None:
            if not isinstance(raw_ground_points, (list, tuple)):
                raise ValueError(f"标定文件 {path} 的 ground_points 必须是列表")
            gd_pts = [
                tuple(p) if isinstance(p, (list, tuple)) else p
                for p in raw_ground_points
            ]
            if len(gd_pts) != len(img_pts):
                raise ValueError(f"标定文件 {path} 的 image_points 与 ground_points 数量不一致")
            if len(img_pts) > 4:
                use_corresp = True
            elif len(img_pts) == 4:
                # 显式提供 4 个 ground_points 时直接校验并抛出异常，避免静默吞错掩盖损坏字段
                expected_rect = [(0.0, 0.0), (w_val, 0.0), (w_val, l_val), (0.0, l_val)]
                parsed_gd = [_validate_point_2d(p, "ground_points") for p in gd_pts]
                if any(
                    math.hypot(gx - ex, gy - ey) > 1e-6
                    for (gx, gy), (ex, ey) in zip(parsed_gd, expected_rect)
                ):
                    use_corresp = True

        if use_corresp:
            raw_ransac = data.get("use_ransac", False)
            if type(raw_ransac) is not bool:
                raise ValueError(f"标定文件 {path} 的 use_ransac 必须是布尔值")
            raw_thresh = data.get("ransac_reproj_threshold_m", 0.5)
            calibrator = cls.from_correspondences(
                image_points=img_pts,
                ground_points=gd_pts,
                name=raw_name,
                synthetic=synthetic,
                parking_zones=raw_zones,
                use_ransac=raw_ransac,
                ransac_reproj_threshold_m=raw_thresh,
                camera_matrix=raw_camera_matrix,
                dist_coeffs=raw_dist_coeffs,
            )
        else:
            calibrator = cls(
                image_points=img_pts,
                width_m=data["width_m"],
                length_m=data["length_m"],
                name=raw_name,
                synthetic=synthetic,
                parking_zones=raw_zones,
                camera_matrix=raw_camera_matrix,
                dist_coeffs=raw_dist_coeffs,
            )
        calibrator.synthetic_field_missing = field_missing
        calibrator.source_path = str(path)
        return calibrator

    def to_yaml(self, save_path: str | Path) -> Path:
        """持久化标定参数至 YAML 文件"""
        # 拒绝空白路径（含 Path("   ")），防止 Path("") 解析成当前目录 '.' 并尝试覆写目录。
        if (isinstance(save_path, str) and not save_path.strip()) or (
            isinstance(save_path, os.PathLike)
            and isinstance(os.fspath(save_path), str)
            and not os.fspath(save_path).strip()
        ):
            raise ValueError("标定配置文件保存路径不能为空")
        path = Path(save_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "name": self.name,
            "synthetic": self.synthetic,
            "width_m": self.width_m,
            "length_m": self.length_m,
            "image_points": [list(pt) for pt in self.image_points],
            "ground_points": [list(pt) for pt in self.ground_points],
            "homography_matrix": self.H.tolist(),
            "inv_homography_matrix": self.H_inv.tolist(),
        }
        if self.parking_zones:
            config_data["parking_zones"] = [
                [list(pt) for pt in zone] for zone in self.parking_zones
            ]
        if self.camera_matrix is not None:
            config_data["camera_matrix"] = [list(row) for row in self.camera_matrix]
        if self.dist_coeffs is not None:
            config_data["dist_coeffs"] = list(self.dist_coeffs)
        if self._used_ransac:
            config_data["use_ransac"] = True
            config_data["ransac_reproj_threshold_m"] = self._ransac_reproj_threshold_m

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        self.source_path = str(path)

        return path

    def _project_undistorted_pixel(
        self, u_p: float, v_p: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """将针孔像素坐标通过 K^-1 转归一化射线并调用 cv2.projectPoints，返回 (proj, jac)。"""
        with np.errstate(over="ignore", invalid="ignore"):
            ray = self._camera_matrix_inv_np @ np.array([u_p, v_p, 1.0], dtype=np.float64).reshape(3, 1)
            rz = float(ray[2, 0])
            if not math.isfinite(rz) or abs(rz) < 1e-8:
                return None
            rx = float(ray[0, 0] / rz)
            ry = float(ray[1, 0] / rz)
            if not (math.isfinite(rx) and math.isfinite(ry)):
                return None
            obj_pt = np.array([[[rx, ry, 1.0]]], dtype=np.float64)
            try:
                proj, cv_jac = cv2.projectPoints(
                    obj_pt,
                    np.zeros((3, 1), dtype=np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    self._camera_matrix_np,
                    self._dist_coeffs_np,
                )
            except cv2.error:
                return None
        return proj, cv_jac

    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """
        核心几何变换：将图像像素坐标 (u, v) 映射为道路实际米制坐标 (X, Y)
        """
        if not (_is_finite_real(u) and _is_finite_real(v)):
            return float("nan"), float("nan")
        undist = self._undistort_uv(float(u), float(v))
        if undist is None:
            return float("nan"), float("nan")
        u_p, v_p = undist
        p_img = np.array([u_p, v_p, 1.0], dtype=np.float64).reshape(3, 1)
        with np.errstate(over="ignore", invalid="ignore"):
            p_ground = self.H @ p_img
        scale = float(p_ground[2, 0])
        # 地平线附近齐次分量趋近于 0，透视除法会发散，返回 NaN 交给上游安全丢弃。
        if not math.isfinite(scale) or abs(scale) < 1e-8:
            return float("nan"), float("nan")

        with np.errstate(over="ignore", invalid="ignore"):
            x_m = float(p_ground[0, 0] / scale)
            y_m = float(p_ground[1, 0] / scale)
        if not (math.isfinite(x_m) and math.isfinite(y_m)):
            return float("nan"), float("nan")
        return x_m, y_m

    def pixel_to_ground_jacobian(self, u: float, v: float) -> Optional[np.ndarray]:
        """计算 pixel_to_ground(u, v) 关于输入像素 (u, v) 的解析 2x2 雅可比矩阵 J。"""
        if not (_is_finite_real(u) and _is_finite_real(v)):
            return None

        undist = self._undistort_uv(float(u), float(v))
        if undist is None:
            return None
        u_p, v_p = undist

        j_h = _homography_jacobian(self.H, u_p, v_p)
        if j_h is None:
            return None

        if self._camera_matrix_np is not None and self._dist_coeffs_np is not None:
            # 反函数定理求去畸变雅可比 d(u', v')/d(u, v) = [d(u, v)/d(u', v')]^{-1}
            j_kinv = _homography_jacobian(self._camera_matrix_inv_np, u_p, v_p)
            if j_kinv is None:
                return None
            projected = self._project_undistorted_pixel(u_p, v_p)
            if projected is None:
                return None
            _, cv_jac = projected
            # projectPoints 雅可比第 3..4 列对应平移动态 [tx, ty] (Z=1 时即归一化射线坐标导数)
            j_dist_ray = cv_jac[:, 3:5].astype(np.float64)
            j_fwd = j_dist_ray @ j_kinv
            if not np.isfinite(j_fwd).all() or abs(float(np.linalg.det(j_fwd))) < 1e-12:
                return None
            j_undist = np.linalg.inv(j_fwd)
            j_total = j_h @ j_undist
            if not np.isfinite(j_total).all():
                return None
            return j_total

        return j_h

    def ground_covariance_from_pixel(
        self,
        u: float,
        v: float,
        sigma_u_px: float = 2.0,
        sigma_v_px: float = 3.0,
        min_var_m2: float = 1e-4,
    ) -> Optional[np.ndarray]:
        """利用一阶雅可比传播像素定位方差，返回地面 2x2 对称正定协方差矩阵 R。"""
        su = _validate_finite_float(sigma_u_px, "sigma_u_px ")
        sv = _validate_finite_float(sigma_v_px, "sigma_v_px ")
        min_var = _validate_finite_float(min_var_m2, "min_var_m2 ")
        if su < 0.0 or sv < 0.0:
            raise ValueError("sigma_u_px 与 sigma_v_px 必须为非负有限数")
        if min_var <= 0.0:
            raise ValueError("min_var_m2 必须为大于 0 的有限正数")

        jac = self.pixel_to_ground_jacobian(u, v)
        if jac is None:
            return None

        cov_px = np.diag([su * su, sv * sv])
        with np.errstate(over="ignore", invalid="ignore"):
            cov_gd = jac @ cov_px @ jac.T + min_var * np.eye(2, dtype=np.float64)
            cov_gd = 0.5 * (cov_gd + cov_gd.T)
        if not np.isfinite(cov_gd).all():
            return None
        return cov_gd

    def ground_to_pixel_checked(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """逆向几何变换：将道路米制坐标 (X, Y) 投射回图像像素坐标 (u, v)；
        对非有限输入、相机主平面后方点及近地平线溢出坐标返回 None。
        """
        if not (_is_finite_real(x) and _is_finite_real(y)):
            return None
        p_g = np.array([float(x), float(y), 1.0], dtype=np.float64).reshape(3, 1)
        with np.errstate(over="ignore", invalid="ignore"):
            p_img = self.H_inv @ p_g
        scale = float(p_img[2, 0])
        # 既拦截近地平线奇异点 (|w| < 1e-6)，也拦截跨越相机主平面跑到相机后方的点 (w 与前方反号)。
        if (
            not math.isfinite(scale)
            or abs(scale) < 1e-6
            or scale * self._front_w_sign <= 0
        ):
            return None

        with np.errstate(over="ignore", invalid="ignore"):
            u_f = float(p_img[0, 0] / scale)
            v_f = float(p_img[1, 0] / scale)
        # 限制绝对像素范围，防止超大坐标在传入 OpenCV int32 绘图接口时触发溢出崩溃。
        if (
            not (math.isfinite(u_f) and math.isfinite(v_f))
            or abs(u_f) > 1_000_000
            or abs(v_f) > 1_000_000
        ):
            return None

        if self._camera_matrix_np is not None and self._dist_coeffs_np is not None:
            projected = self._project_undistorted_pixel(u_f, v_f)
            if projected is None:
                return None
            proj, _ = projected
            u_f = float(proj[0, 0, 0])
            v_f = float(proj[0, 0, 1])
            if (
                not (math.isfinite(u_f) and math.isfinite(v_f))
                or abs(u_f) > 1_000_000
                or abs(v_f) > 1_000_000
            ):
                return None

        return int(round(u_f)), int(round(v_f))

    def ground_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """逆向几何变换：将道路米制坐标 (X, Y) 投射回图像像素坐标 (u, v)"""
        pt = self.ground_to_pixel_checked(x, y)
        # 保持旧接口契约：无效点及画幅左/上方负像素坐标统一返回 (-1, -1)。
        if pt is None or pt[0] < 0 or pt[1] < 0:
            return -1, -1
        return pt

    def is_in_zone(self, x: float, y: float, margin_m: float = 2.0) -> bool:
        """判断目标地面坐标是否处于重点受控监测区域内（带冗余边距缓冲）"""
        if not (_is_finite_real(x) and _is_finite_real(y) and _is_finite_real(margin_m)):
            return False
        # 统一转为 Python float，兼容 NumPy 标量输入
        fx, fy, fm = float(x), float(y), float(margin_m)
        min_x, max_x, min_y, max_y = self._zone_bounds
        return (
            (min_x - fm <= fx <= max_x + fm)
            and (min_y - fm <= fy <= max_y + fm)
        )

    def add_parking_zone_pixels(
        self, pixel_polygon: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """将图像像素多边形投射为地面米制停车区并登记。"""
        # 停车区以图像像素多边形录入后，立即通过单应性 H 转换为地面米制坐标留档：
        # 这样无论后续是否缩放渲染分辨率，物理多边形边界与车辆接地坐标始终处于同一米制坐标系。
        parsed = _validate_polygon_2d(pixel_polygon, name="像素停车区")
        ground_poly: List[Tuple[float, float]] = []
        for u, v in parsed:
            gx, gy = self.pixel_to_ground(u, v)
            if not (math.isfinite(gx) and math.isfinite(gy)):
                raise ValueError("停车区顶点投射至地面后出现非有限坐标，请检查像素选点")
            ground_poly.append((gx, gy))
        self.parking_zones.append(ground_poly)
        return ground_poly

    def is_in_parking_zone(self, x: float, y: float) -> bool:
        """判断地面坐标 (x, y) 是否落在已标定的路侧停车区多边形内。"""
        return any(_point_in_polygon(x, y, poly) for poly in self.parking_zones)

    def evaluate_check_points(
        self,
        check_image_points: List[Tuple[float, float]],
        check_ground_points: List[Tuple[float, float]],
    ) -> dict:
        """使用独立实测检查点验证单应性变换的地面空间精度（拒绝重合/重复点）。"""
        if (
            not isinstance(check_image_points, (list, tuple))
            or not isinstance(check_ground_points, (list, tuple))
            or not check_image_points
            or len(check_image_points) != len(check_ground_points)
        ):
            raise ValueError("check_image_points 与 check_ground_points 必须为非空且等长的坐标列表")

        parsed_img: List[Tuple[float, float]] = [
            _validate_point_2d(pt, f"第 {idx + 1} 个检查图像点")
            for idx, pt in enumerate(check_image_points)
        ]
        parsed_gd: List[Tuple[float, float]] = [
            _validate_point_2d(pt, f"第 {idx + 1} 个检查地面点")
            for idx, pt in enumerate(check_ground_points)
        ]

        # 独立检查点不得与参与求解 H 的 4 个标定角点重合，也不得自身重复，
        # 否则测得的是训练角点拟合残差而非真实地面投影误差。
        for idx, (u, v) in enumerate(parsed_img):
            if any(math.hypot(u - fu, v - fv) < 1e-6 for fu, fv in self.image_points) or any(
                math.hypot(u - pu, v - pv) < 1e-6 for pu, pv in parsed_img[:idx]
            ):
                raise ValueError("独立检查点不得重复或与标定拟合角点重合")
        for idx, (gx, gy) in enumerate(parsed_gd):
            if any(math.hypot(gx - fx, gy - fy) < 1e-6 for fx, fy in self.ground_points) or any(
                math.hypot(gx - px, gy - py) < 1e-6 for px, py in parsed_gd[:idx]
            ):
                raise ValueError("独立检查点不得重复或与标定拟合角点重合")

        errors: List[float] = []
        point_reports: List[dict] = []
        for (u, v), (gx, gy) in zip(parsed_img, parsed_gd):
            px, py = self.pixel_to_ground(u, v)
            if not (math.isfinite(px) and math.isfinite(py)):
                raise ValueError("检查点投影退化为非有限坐标")
            err_m = math.hypot(px - gx, py - gy)
            errors.append(err_m)
            point_reports.append({
                "image_xy": [u, v],
                "expected_ground_m": [gx, gy],
                "projected_ground_m": [px, py],
                "error_m": err_m,
            })

        return {
            "count": len(errors),
            "mean_error_m": sum(errors) / len(errors),
            "rmse_m": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "max_error_m": max(errors),
            "points": point_reports,
        }

    def estimate_camera_ground_xy(self) -> Optional[Tuple[float, float]]:
        """估算相机光心在地面坐标系下的投影位置 (X, Y)；无效或奇异时安全返回 None。"""
        h_mat = getattr(self, "H", None)
        h_inv = getattr(self, "H_inv", None)
        if (
            not isinstance(h_mat, np.ndarray)
            or not isinstance(h_inv, np.ndarray)
            or h_mat.shape != (3, 3)
            or h_inv.shape != (3, 3)
            or not (np.isfinite(h_mat).all() and np.isfinite(h_inv).all())
        ):
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            det_h = float(np.linalg.det(h_mat))
            det_hinv = float(np.linalg.det(h_inv))
        if not (math.isfinite(det_h) and math.isfinite(det_hinv)) or abs(det_h) <= 1e-12 or abs(det_hinv) <= 1e-12:
            return None

        gd_pts = getattr(self, "ground_points", None)
        img_pts = getattr(self, "image_points", None)
        if not isinstance(gd_pts, (list, tuple)) or not isinstance(img_pts, (list, tuple)) or len(gd_pts) < 4 or len(img_pts) < 4:
            return None
        if not all(_is_finite_real(px) and _is_finite_real(py) for px, py in gd_pts):
            return None
        g_cx = sum(float(px) for px, _ in gd_pts) / len(gd_pts)
        g_cy = sum(float(py) for _, py in gd_pts) / len(gd_pts)

        # 若已配置相机内参 K，优先尝试单应性矩阵分解 K^-1 * H_inv = lambda * [r1, r2, t]
        k_mat = getattr(self, "_camera_matrix_np", None)
        k_inv = getattr(self, "_camera_matrix_inv_np", None)
        if k_mat is not None:
            if (
                not isinstance(k_inv, np.ndarray)
                or k_inv.shape != (3, 3)
                or not np.isfinite(k_inv).all()
            ):
                return None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if abs(float(np.linalg.det(k_inv))) <= 1e-12:
                    return None
                m_mat = k_inv @ h_inv
                if not np.isfinite(m_mat).all():
                    return None
                m1, m2, m3 = m_mat[:, 0], m_mat[:, 1], m_mat[:, 2]
                n1 = float(np.linalg.norm(m1))
                n2 = float(np.linalg.norm(m2))
                if not (math.isfinite(n1) and math.isfinite(n2)) or n1 <= 1e-9 or n2 <= 1e-9:
                    return None
                # 仅当内参 K 与地面单应性尺度/正交性物理自洽时直接采用外参反算光心；
                # 若二者尺幅严重失配（如手工梯形配固定示范 K），自动退回图像底边射线外推。
                scale_ratio = max(n1, n2) / min(n1, n2)
                cos_orth = abs(float(np.dot(m1 / n1, m2 / n2)))
                if scale_ratio <= 1.8 and cos_orth <= 0.35:
                    lam = 0.5 * (n1 + n2)
                    r1, r2, t_vec = m1 / lam, m2 / lam, m3 / lam
                    # 用标定地面中心点的相机深度符号校正齐次尺度正负（保证目标位于相机前方 Z_c > 0）
                    if float(r1[2] * g_cx + r2[2] * g_cy + t_vec[2]) < 0.0:
                        r1, r2, t_vec = -r1, -r2, -t_vec
                    r3 = np.cross(r1, r2)
                    r_approx = np.column_stack([r1, r2, r3])
                    if not np.isfinite(r_approx).all() or abs(float(np.linalg.det(r_approx))) <= 1e-9:
                        return None
                    u_mat, _, vt_mat = np.linalg.svd(r_approx)
                    r_mat = u_mat @ vt_mat
                    if float(np.linalg.det(r_mat)) < 0.0:
                        u_mat[:, -1] *= -1.0
                        r_mat = u_mat @ vt_mat
                    cam_world = -r_mat.T @ t_vec
                    if not np.isfinite(cam_world).all():
                        return None
                    # 光心紧贴地面平面（高度接近 0）属于退化共面奇异配置，安全返回 None
                    if abs(float(cam_world[2])) <= 1e-4:
                        return None
                    cx_w, cy_w = float(cam_world[0]), float(cam_world[1])
                    if math.hypot(cx_w - g_cx, cy_w - g_cy) > 1e-6:
                        return cx_w, cy_w
                    return None

        # 未提供匹配内参时：从标定区域底边中点沿画面下方中心射线外推估算相机侧地面原点
        try:
            pts_uv = self._undistort_points_array(list(img_pts))
        except (ValueError, cv2.error):
            return None
        if pts_uv.ndim != 2 or pts_uv.shape[0] < 4 or pts_uv.shape[1] != 2 or not np.isfinite(pts_uv).all():
            return None

        if len(pts_uv) == 4:
            best_edge = max(
                range(4),
                key=lambda i: 0.5 * (float(pts_uv[i, 1]) + float(pts_uv[(i + 1) % 4, 1])),
            )
            p_a = pts_uv[best_edge]
            p_b = pts_uv[(best_edge + 1) % 4]
            u_bot = 0.5 * (float(p_a[0]) + float(p_b[0]))
            v_bot = 0.5 * (float(p_a[1]) + float(p_b[1]))
        else:
            order = np.argsort(pts_uv[:, 1])[::-1]
            u_bot = 0.5 * (float(pts_uv[order[0], 0]) + float(pts_uv[order[1], 0]))
            v_bot = 0.5 * (float(pts_uv[order[0], 1]) + float(pts_uv[order[1], 1]))

        span_v = float(np.max(pts_uv[:, 1]) - np.min(pts_uv[:, 1]))
        span_u = float(np.max(pts_uv[:, 0]) - np.min(pts_uv[:, 0]))
        if not (math.isfinite(span_v) and math.isfinite(span_u)) or span_v <= 1e-6:
            return None
        step_v = max(span_v, 0.25 * span_u)
        front_sign = getattr(self, "_front_w_sign", 1)

        for scale_k in (2.0, 1.5, 1.0, 0.5, 0.25):
            u_ex = u_bot
            v_ex = v_bot + scale_k * step_v
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                p_g = h_mat @ np.array([u_ex, v_ex, 1.0], dtype=np.float64)
                w = float(p_g[2])
                if not math.isfinite(w) or abs(w) < 1e-8:
                    continue
                gx = float(p_g[0] / w)
                gy = float(p_g[1] / w)
                if not (math.isfinite(gx) and math.isfinite(gy)):
                    continue
                w_inv = float((h_inv @ np.array([gx, gy, 1.0], dtype=np.float64))[2])
            # 外推原点必须位于相机主平面前方且与标定区中心不重合
            if math.isfinite(w_inv) and w_inv * front_sign > 1e-8 and math.hypot(gx - g_cx, gy - g_cy) > 1e-6:
                return gx, gy
        return None

    def compensate_vehicle_anchor(
        self,
        x_m: float,
        y_m: float,
        length_m: float,
        width_m: float,
        heading_deg: Optional[float] = None,
    ) -> Tuple[float, float]:
        """将 2D 检测框底边地面投影点沿视线向车体内部推进一阶有效半纵深，抑制几何中心偏置。"""
        if not _is_finite_real(x_m) or not _is_finite_real(y_m):
            raise ValueError("x_m 与 y_m 必须为有限实数")
        if not _is_finite_real(length_m) or not _is_finite_real(width_m):
            raise ValueError("length_m 与 width_m 必须为有限实数")
        fx, fy = float(x_m), float(y_m)
        flen, fwid = float(length_m), float(width_m)
        if flen <= 0.0 or fwid <= 0.0:
            raise ValueError("length_m 与 width_m 必须大于 0")

        cam_xy = self.estimate_camera_ground_xy()
        if cam_xy is None:
            return fx, fy
        cam_x, cam_y = cam_xy
        if not (_is_finite_real(cam_x) and _is_finite_real(cam_y)):
            return fx, fy

        with np.errstate(over="ignore", invalid="ignore"):
            dx = fx - float(cam_x)
            dy = fy - float(cam_y)
            if not (math.isfinite(dx) and math.isfinite(dy)):
                # 极值坐标差值上溢时先减半缩放再归一化求单位方向
                dx = 0.5 * fx - 0.5 * float(cam_x)
                dy = 0.5 * fy - 0.5 * float(cam_y)
            max_c = max(abs(dx), abs(dy))
            if not math.isfinite(max_c) or max_c <= 1e-9:
                return fx, fy
            nx = dx / max_c
            ny = dy / max_c
            unit_norm = math.hypot(nx, ny)
            if not math.isfinite(unit_norm) or unit_norm <= 1e-12:
                return fx, fy
            if max_c <= 1e-6 and max_c * unit_norm <= 1e-6:
                return fx, fy
            ux = nx / unit_norm
            uy = ny / unit_norm

        max_half = 0.5 * max(flen, fwid)
        if _is_finite_real(heading_deg):
            psi = math.radians(float(heading_deg))
            cos_p = math.cos(psi)
            sin_p = math.sin(psi)
            # 有限航向角下取车体矩形沿视线方向 (ux, uy) 的半支撑投影
            proj_half = 0.5 * flen * abs(ux * cos_p + uy * sin_p) + 0.5 * fwid * abs(-ux * sin_p + uy * cos_p)
        else:
            proj_half = 0.5 * fwid
        offset_m = min(proj_half, max_half)

        with np.errstate(over="ignore", invalid="ignore"):
            comp_x = fx + offset_m * ux
            comp_y = fy + offset_m * uy
        if not (math.isfinite(comp_x) and math.isfinite(comp_y)):
            return fx, fy
        return float(comp_x), float(comp_y)
