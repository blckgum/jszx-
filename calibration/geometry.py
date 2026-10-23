"""单应性求解与批量齐次投影。坐标与阈值的单位必须由调用方明确指定。"""

from itertools import combinations

import cv2
import numpy as np


def points_array(values, minimum=0, finite=True):
    points = np.asarray(values, dtype=np.float64)
    if points.size == 0:
        points = points.reshape(0, 2)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f"点集必须为 N×2，且 N 至少为 {minimum}")
    if finite and not np.isfinite(points).all():
        raise ValueError("坐标必须为有限数值")
    return points


def check_layout(points):
    """在归一化尺度检查重复、共线，避免像素和米的量纲影响退化判断。"""
    points = points_array(points, minimum=4)
    scale = np.ptp(points, axis=0).max()
    if scale <= 0:
        raise ValueError("参考点全部重合")
    normalized = (points - points.mean(axis=0)) / scale
    for i, point in enumerate(normalized):
        if np.any(np.linalg.norm(normalized[i + 1:] - point, axis=1) < 1e-9):
            raise ValueError("参考点重复或过度接近")
    if np.linalg.matrix_rank(normalized, tol=1e-9) < 2:
        raise ValueError("参考点共线或分布退化")
    if len(points) == 4:
        for a, b, c in combinations(normalized, 3):
            if abs(np.linalg.det(np.stack((b - a, c - a)))) < 1e-8:
                raise ValueError("四点标定不能有三点共线")


def normalize_homography(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.max(np.abs(matrix)) == 0:
        raise ValueError("单应性矩阵必须为非零、有限的 3×3 矩阵")
    # H 与任意非零倍数表示相同投影，先消除任意整体比例。
    matrix = matrix / np.max(np.abs(matrix))
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("单应性矩阵不可逆") from None
    if not np.isfinite(inverse).all():
        raise ValueError("单应性逆矩阵不是有限值")
    return matrix


def project_points(matrix, values):
    """返回 N×2 浮点坐标及有效掩码；内部无效位置用 NaN，导出层转换成 null。"""
    matrix = normalize_homography(matrix)
    matrix = matrix / np.linalg.norm(matrix[2])
    points = points_array(values, finite=False)
    output = np.full((len(points), 2), np.nan)
    valid = np.isfinite(points).all(axis=1)
    if not len(points):
        return output, valid
    homogeneous = np.column_stack((points, np.ones(len(points))))
    with np.errstate(invalid="ignore", over="ignore"):
        denominator = homogeneous @ matrix[2]
        bound = np.linalg.norm(homogeneous, axis=1) * np.linalg.norm(matrix[2])
    # 分母相对第三行及输入点的尺度过小时，点接近投影无穷远；不要硬夹分母造假位置。
    valid &= np.isfinite(denominator) & (np.abs(denominator) > 1e-10 * bound)
    ordinary = valid & (np.abs(denominator) >= 1e-6)
    if ordinary.any():
        output[ordinary] = cv2.perspectiveTransform(points[ordinary].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    near = valid & ~ordinary
    if near.any():
        # OpenCV 对很小的分母可能直接返回零；通过相对尺度检查后，明确做齐次除法。
        output[near] = (homogeneous[near] @ matrix[:2].T) / denominator[near, None]
    valid &= np.isfinite(output).all(axis=1)
    output[~valid] = np.nan
    return output, valid


def estimate_homography(image_points, ground_points, method="direct", ransac_threshold_m=0.2):
    image = points_array(image_points, minimum=4)
    ground = points_array(ground_points, minimum=4)
    if len(image) != len(ground):
        raise ValueError("图像点与地面点必须一一对应")
    check_layout(image)
    check_layout(ground)
    if method not in {"direct", "ransac", "rectangle"}:
        raise ValueError("method 必须为 direct、ransac 或 rectangle")
    if not np.isfinite(ransac_threshold_m) or ransac_threshold_m <= 0:
        raise ValueError("RANSAC 阈值必须为正数，单位米")
    if method == "rectangle":
        width, length = ground[1, 0], ground[2, 1]
        expected = np.array([[0, 0], [width, 0], [width, length], [0, length]])
        if len(ground) != 4 or width <= 0 or length <= 0 or not np.allclose(ground, expected):
            raise ValueError("rectangle 模式要求按边界顺序提供原点起始的四个矩形米制角点")
        # 复用朋友已经调试的四点矩形求解，不改其 API、配置语义或异常行为。
        from analysis.calibration import RoadCalibrator
        matrix = RoadCalibrator(image.tolist(), float(width), float(length)).H
        inliers = np.ones(4, dtype=bool)
    else:
        if method == "ransac" and len(image) < 5:
            raise ValueError("RANSAC 至少需要 5 组点才有排除离群点的冗余")
        # 目标坐标是米，因此这里的 reprojection threshold 是米，不是像素。
        matrix, mask = cv2.findHomography(image, ground, cv2.RANSAC if method == "ransac" else 0,
                                           ransac_threshold_m, maxIters=5000, confidence=0.995)
        if matrix is None or mask is None:
            raise ValueError("无法估计单应性矩阵，请检查点分布及对应关系")
        inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4:
        raise ValueError("有效拟合内点不足四个")
    check_layout(image[inliers])
    check_layout(ground[inliers])
    matrix = normalize_homography(matrix)
    prediction, valid = project_points(matrix, image)
    if not valid[inliers].all():
        raise ValueError("拟合参考点出现无效投影")
    return matrix, np.linalg.inv(matrix), inliers
