"""区分拟合残差与独立检查点误差；只评估位置，不虚构速度真值。"""

import numpy as np

from .geometry import points_array, project_points


def error_metrics(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if not len(errors) or not np.isfinite(errors).all():
        return {"count": len(errors), "mean_error_m": None, "rmse_m": None, "max_error_m": None}
    return {"count": len(errors), "mean_error_m": float(errors.mean()),
            "rmse_m": float(np.sqrt(np.mean(errors ** 2))), "max_error_m": float(errors.max())}


def validate_mapping(mapper):
    config = mapper.config
    predicted, valid = project_points(mapper.H, mapper.lens.rectify(config.image_points))
    residual = np.linalg.norm(predicted - points_array(config.ground_points_m), axis=1)
    checks = points_array(config.check_image_points)
    expected = points_array(config.check_ground_points_m)
    _, mapped, check_valid = mapper.project(checks)
    errors = np.linalg.norm(mapped - expected, axis=1)
    # 无效检查点不能静默丢弃后再计算“漂亮误差”；本次检查直接不通过。
    metrics = error_metrics(errors if check_valid.all() else np.full(len(checks), np.nan))
    spread_ok = len(checks) >= 3 and np.linalg.matrix_rank(checks - checks.mean(axis=0)) == 2
    passed = bool(spread_ok and check_valid.all() and metrics["rmse_m"] is not None
                  and metrics["rmse_m"] <= config.max_check_rmse_m
                  and metrics["max_error_m"] <= config.max_check_error_m)
    return {
        "schema_version": 1, "source_type": config.source_type, "synthetic": mapper.synthetic,
        "distortion_status": mapper.lens.status, "fit_point_count": len(residual),
        "fit_inlier_count": int(mapper.inliers.sum()),
        "fit_inliers": error_metrics(residual[mapper.inliers]), "fit_all_points": error_metrics(residual),
        "independent_checks": metrics, "check_points_valid": check_valid.tolist(),
        "check_errors_m": [float(e) if good and np.isfinite(e) else None for e, good in zip(errors, check_valid)],
        "independent_check_passed": passed, "calibration_valid": bool(passed and not mapper.synthetic),
        "note": "位置验证，不是速度误差报告；合成零误差不代表真实道路测量可靠。",
    }
