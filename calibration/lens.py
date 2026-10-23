"""可选 OpenCV 针孔镜头参数；参考点和目标点统一转换到同一去畸变像素系。"""

import cv2
import numpy as np

from .geometry import points_array


class LensModel:
    def __init__(self, parameters=None):
        self.parameters = parameters
        self.K = self.D = None
        if parameters is None:
            return
        if (not isinstance(parameters, dict) or set(parameters) != {"model", "camera_matrix", "distortion_coefficients"}
                or parameters["model"] != "opencv_pinhole"):
            raise ValueError("lens 仅支持明确的 opencv_pinhole 内参/畸变配置；鱼眼需单独模型")
        self.K = np.asarray(parameters["camera_matrix"], dtype=np.float64)
        self.D = np.asarray(parameters["distortion_coefficients"], dtype=np.float64)
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.allclose(self.K[2], [0, 0, 1])
                or abs(self.K[0, 1]) > 1e-12 or abs(self.K[1, 0]) > 1e-12):
            raise ValueError("camera_matrix 必须是零倾斜、正焦距的有限针孔内参矩阵")
        if self.D.ndim != 1 or len(self.D) not in {4, 5, 8, 12, 14} or not np.isfinite(self.D).all():
            raise ValueError("畸变系数必须为有限的 4/5/8/12/14 元素列表")

    @property
    def status(self):
        return "uncorrected" if self.K is None else "opencv_pinhole_corrected"

    def rectify(self, values):
        points = points_array(values, finite=False)
        if self.K is None or not len(points):
            return points.copy()
        output = np.full_like(points, np.nan)
        finite = np.isfinite(points).all(axis=1)
        if finite.any():
            # P=K 至关重要：输出仍是去畸变图像中的像素，不是归一化相机射线坐标。
            # 当前固定的 OpenCV 5 将迭代终止条件直接放在 undistortPoints 中。
            output[finite] = cv2.undistortPoints(
                points[finite].reshape(-1, 1, 2), self.K, self.D, P=self.K,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12),
            ).reshape(-1, 2)
        return output

    def distort(self, values):
        points = points_array(values, finite=False)
        if self.K is None or not len(points):
            return points.copy()
        output = np.full_like(points, np.nan)
        finite = np.isfinite(points).all(axis=1)
        if finite.any():
            rays = np.column_stack((points[finite], np.ones(finite.sum()))) @ np.linalg.inv(self.K).T
            # 零旋转/平移仅应用镜头模型，便于将地面网格叠加回原始畸变图像。
            projected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), self.K, self.D)
            output[finite] = projected.reshape(-1, 2)
        return output
