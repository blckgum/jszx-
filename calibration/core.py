"""带机位、单位与来源约束的道路坐标映射接口，不输出实际速度或风险。"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json

import cv2
import numpy as np

from .geometry import estimate_homography, points_array, project_points
from .lens import LensModel


@dataclass
class MappingConfig:
    camera_id: str
    image_size: list[int]  # [width, height]，与参考点、跟踪接地点所在原图一致。
    source_type: str  # measured 或 synthetic，不能由求解成功与否自动推断。
    measurement_reference: str
    image_points: list
    ground_points_m: list
    method: str = "direct"
    ransac_threshold_m: float = 0.2
    lens: dict | None = None
    roi_image_points: list | None = None
    check_image_points: list = field(default_factory=list)
    check_ground_points_m: list = field(default_factory=list)
    max_check_rmse_m: float = 0.25
    max_check_error_m: float = 0.5
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("不支持的道路映射配置版本")
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("必须明确 camera_id")
        if (not isinstance(self.image_size, (list, tuple)) or len(self.image_size) != 2
                or any(type(v) is not int or v <= 0 for v in self.image_size)):
            raise ValueError("image_size 必须是 [width, height] 正整数")
        if self.source_type not in {"measured", "synthetic"}:
            raise ValueError("source_type 必须明确为 measured 或 synthetic")
        if not isinstance(self.measurement_reference, str) or not self.measurement_reference.strip():
            raise ValueError("必须记录真实测量来源或合成生成方法")
        for name in ("ransac_threshold_m", "max_check_rmse_m", "max_check_error_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数，单位米")
        image, ground = points_array(self.image_points, 4), points_array(self.ground_points_m, 4)
        checks, references = points_array(self.check_image_points), points_array(self.check_ground_points_m)
        if len(image) != len(ground) or len(checks) != len(references):
            raise ValueError("拟合点、检查点分别必须与对应的米制坐标等长")
        for points in (image, checks):
            if np.any(points < 0) or np.any(points > np.asarray(self.image_size)):
                raise ValueError("参考点超出声明的原图尺寸")
        # 拟合点零残差并不能证明准确性；检查点必须是另外的物理位置。
        for a, b in ((checks, image), (references, ground)):
            for i, point in enumerate(a):
                if (np.any(np.linalg.norm(b - point, axis=1) < 1e-6)
                        or np.any(np.linalg.norm(a[i + 1:] - point, axis=1) < 1e-6)):
                    raise ValueError("检查点重复，或与拟合参考点重合，不能作为独立验证")
        self.image_size = list(self.image_size)
        self.image_points, self.ground_points_m = image.tolist(), ground.tolist()
        self.check_image_points, self.check_ground_points_m = checks.tolist(), references.tolist()
        if self.roi_image_points is not None:
            self.roi_image_points = points_array(self.roi_image_points, 3).tolist()


@dataclass(frozen=True)
class MappedPoint:
    image_position_px: tuple | None
    rectified_position_px: tuple | None
    ground_position_m: tuple | None
    mapping_valid: bool
    calibration_valid: bool
    invalid_reason: str | None
    calibration_id: str
    synthetic: bool

    def to_dict(self):
        return asdict(self)


class RoadMapper:
    def __init__(self, config: MappingConfig):
        # 调用方后续修改列表时，不应改变已经求解的标定含义。
        self.config = deepcopy(config)
        self.lens = LensModel(self.config.lens)
        if self.lens.K is not None:
            self.config.lens = {"model": "opencv_pinhole", "camera_matrix": self.lens.K.tolist(),
                                "distortion_coefficients": self.lens.D.tolist()}
        raw = points_array(self.config.image_points, 4)
        rectified = self.lens.rectify(raw)
        self.H, self.H_inv, self.inliers = estimate_homography(
            rectified, self.config.ground_points_m, self.config.method, self.config.ransac_threshold_m)
        roi = self.config.roi_image_points
        if roi is None:
            # 排除 RANSAC 离群点后再确定默认有效区域，避免坏点扩张可用道路范围。
            roi = cv2.convexHull(raw[self.inliers].astype(np.float32)).reshape(-1, 2)
        self.roi = points_array(roi, 3).astype(np.float32)
        if (np.any(self.roi < 0) or np.any(self.roi > self.config.image_size)
                or not cv2.isContourConvex(self.roi) or cv2.contourArea(self.roi) < 1e-6):
            raise ValueError("ROI 必须是原图内不自交、非退化的凸多边形，按边界顺序输入")
        hull = cv2.convexHull(raw[self.inliers].astype(np.float32))
        if any(cv2.pointPolygonTest(hull, tuple(map(float, point)), True) < -1e-4 for point in self.roi):
            raise ValueError("ROI 不能扩张到标定内点覆盖范围之外")
        # 平坦有限道路区域不应跨越投影无穷远。错误的四点对应顺序常造成分母换号。
        edge_samples = np.concatenate([np.linspace(a, b, 12) for a, b in zip(self.roi, np.roll(self.roi, -1, axis=0))])
        corrected_edges = self.lens.rectify(edge_samples)
        denominators = np.column_stack((corrected_edges, np.ones(len(edge_samples)))) @ self.H[2]
        if not np.isfinite(denominators).all() or not (np.all(denominators > 0) or np.all(denominators < 0)):
            raise ValueError("投影无穷远穿过有效区域，请检查点的对应关系和顺序")
        canonical = json.dumps(asdict(self.config), sort_keys=True, ensure_ascii=False, allow_nan=False)
        self.calibration_id = hashlib.sha256(canonical.encode()).hexdigest()[:20]
        from .validation import validate_mapping
        self.report = validate_mapping(self)

    @property
    def synthetic(self):
        return self.config.source_type == "synthetic"

    @property
    def calibration_valid(self):
        # 数值拟合成功 ≠ 实景标定通过；合成数据即使零误差也不能输出真实标定有效。
        return not self.synthetic and self.report["independent_check_passed"]

    def project(self, values):
        """批量原图点 -> 去畸变像素 -> 米制点，返回坐标及几何有效掩码。"""
        raw = points_array(values, finite=False)
        rectified = self.lens.rectify(raw)
        ground, valid = project_points(self.H, rectified)
        valid &= np.isfinite(raw).all(axis=1)
        valid &= (raw >= 0).all(axis=1) & (raw <= self.config.image_size).all(axis=1)
        for i in np.flatnonzero(valid):
            if cv2.pointPolygonTest(self.roi, tuple(map(float, raw[i])), True) < -1e-4:
                valid[i] = False
        return rectified, ground, valid

    def map_points(self, values, *, camera_id, image_size):
        raw = points_array(values, finite=False)
        rectified, ground, valid = self.project(raw)
        context_error = None
        if camera_id != self.config.camera_id:
            context_error = "camera_mismatch"
        elif tuple(image_size) != tuple(self.config.image_size):
            context_error = "image_size_mismatch"
        results = []
        for i, point in enumerate(raw):
            reason = context_error
            if reason is None and not np.isfinite(point).all():
                reason = "nonfinite_input"
            elif reason is None and (np.any(point < 0) or np.any(point > self.config.image_size)):
                reason = "outside_image"
            elif reason is None and not np.isfinite(ground[i]).all():
                reason = "invalid_projection"
            elif reason is None and not valid[i]:
                reason = "outside_roi"
            results.append(MappedPoint(
                tuple(map(float, point)) if np.isfinite(point).all() else None,
                tuple(map(float, rectified[i])) if np.isfinite(rectified[i]).all() else None,
                tuple(map(float, ground[i])) if reason is None else None,
                reason is None, self.calibration_valid, reason, self.calibration_id, self.synthetic))
        return results

    def map_point(self, u, v, *, camera_id, image_size):
        return self.map_points([(u, v)], camera_id=camera_id, image_size=image_size)[0]

    def ground_to_image(self, values):
        rectified, valid = project_points(self.H_inv, values)
        raw = self.lens.distort(rectified)
        valid &= np.isfinite(raw).all(axis=1)
        return raw, valid
