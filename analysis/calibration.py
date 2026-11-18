# -*- coding: utf-8 -*-
"""
相机空间标定与地面物理坐标单应性变换模块 (Camera Calibration & Homography Mapping)
=============================================================================
将图像中斜向下俯视的像素坐标 (u, v)，通过 4 点地面标定射影变换矩阵 H，
准确映射到校园道路的实际物理平面坐标 (X, Y)（单位：米）。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


def _validate_point_2d(pt, label: str) -> Tuple[float, float]:
    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
        raise ValueError(f"{label}必须是长度为 2 的坐标对")
    x, y = pt
    if (
        type(x) not in (int, float)
        or type(y) not in (int, float)
        or not (math.isfinite(x) and math.isfinite(y))
    ):
        raise ValueError(f"{label}坐标必须为有限数值")
    return (float(x), float(y))


def _validate_polygon_2d(poly, name: str = "多边形") -> List[Tuple[float, float]]:
    if not isinstance(poly, (list, tuple)) or len(poly) < 3:
        raise ValueError(f"{name}必须至少包含 3 个二维顶点")
    return [_validate_point_2d(pt, f"{name}第 {idx + 1} 个顶点") for idx, pt in enumerate(poly)]


def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    if not (math.isfinite(x) and math.isfinite(y)) or len(polygon) < 3:
        return False
    n = len(polygon)
    # 边界点视为在区域内
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq <= 1e-18:
            if math.hypot(x - x1, y - y1) <= 1e-9:
                return True
        else:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg_len_sq))
            if math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)) <= 1e-9:
                return True
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
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

    def to_dict(self) -> dict:
        return asdict(self)


class RoadCalibrator:
    """道路空间几何单应性标定与坐标转换引擎"""

    def __init__(
        self,
        image_points: List[Tuple[float, float]],
        width_m: float,
        length_m: float,
        name: str = "campus_road",
        synthetic: bool = False,
        parking_zones: Optional[List[List[Tuple[float, float]]]] = None,
    ):
        if type(synthetic) is not bool:
            raise ValueError("synthetic 必须是布尔值")
        if len(image_points) != 4:
            raise ValueError(f"地面标定必须严格提供 4 个角点，当前提供了 {len(image_points)} 个")
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

        parsed_parking_zones: List[List[Tuple[float, float]]] = []
        if parking_zones is not None:
            if not isinstance(parking_zones, (list, tuple)):
                raise ValueError("parking_zones 必须是多边形列表")
            for idx, zone in enumerate(parking_zones):
                parsed_parking_zones.append(
                    _validate_polygon_2d(zone, name=f"第 {idx + 1} 个停车区")
                )

        self.name = name
        # 合成占位标定标记：非 --demo 模式下拒绝加载，防止演示参数被当真实标定
        self.synthetic = synthetic
        # 旧版标定文件可能缺失 synthetic 溯源字段，由 from_yaml 置位
        self.synthetic_field_missing = False
        self.source_path: Optional[str] = None
        self.width_m = float(width_m)
        self.length_m = float(length_m)
        self.image_points = parsed_points
        self.parking_zones = parsed_parking_zones

        # 定义地面标准矩形（顺时针：左上 -> 右上 -> 右下 -> 左下）
        self.ground_points = [
            (0.0, 0.0),
            (self.width_m, 0.0),
            (self.width_m, self.length_m),
            (0.0, self.length_m),
        ]

        src_pts = np.array(self.image_points, dtype=np.float32)
        dst_pts = np.array(self.ground_points, dtype=np.float32)

        # 求解 3x3 单应性透视变换矩阵 H
        h_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        if h_mat is None or np.isnan(h_mat).any():
            raise RuntimeError("无法求解有效的单应性矩阵，请检查 4 个角点是否退化共线")

        self.H = h_mat.astype(np.float64)
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                "单应性矩阵不可逆，请检查 4 个角点是否退化共线或重复选取"
            ) from None

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "RoadCalibrator":
        """从已保存的 YAML 配置文件载入标定矩阵"""
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

        calibrator = cls(
            image_points=img_pts,
            width_m=data["width_m"],
            length_m=data["length_m"],
            name=data.get("name", "campus_road"),
            synthetic=synthetic,
            parking_zones=raw_zones,
        )
        calibrator.synthetic_field_missing = field_missing
        calibrator.source_path = str(path)
        return calibrator

    def to_yaml(self, save_path: str | Path) -> Path:
        """持久化标定参数至 YAML 文件"""
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

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        self.source_path = str(path)

        return path

    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """
        核心几何变换：将图像像素坐标 (u, v) 映射为道路实际米制坐标 (X, Y)
        """
        p_img = np.array([u, v, 1.0], dtype=np.float64).reshape(3, 1)
        p_ground = self.H @ p_img
        scale = p_ground[2, 0]
        if abs(scale) < 1e-8:
            return float("nan"), float("nan")

        x_m = float(p_ground[0, 0] / scale)
        y_m = float(p_ground[1, 0] / scale)
        return x_m, y_m

    def ground_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """逆向几何变换：将道路米制坐标 (X, Y) 投射回图像像素坐标 (u, v)"""
        p_g = np.array([x, y, 1.0], dtype=np.float64).reshape(3, 1)
        p_img = self.H_inv @ p_g
        scale = p_img[2, 0]
        if abs(scale) < 1e-8:
            return -1, -1

        u = int(round(p_img[0, 0] / scale))
        v = int(round(p_img[1, 0] / scale))
        return u, v

    def is_in_zone(self, x: float, y: float, margin_m: float = 2.0) -> bool:
        """判断目标地面坐标是否处于重点受控监测区域内（带冗余边距缓冲）"""
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        return (
            (-margin_m <= x <= self.width_m + margin_m)
            and (-margin_m <= y <= self.length_m + margin_m)
        )

    def add_parking_zone_pixels(
        self, pixel_polygon: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """将图像像素多边形投射为地面米制停车区并登记。"""
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
