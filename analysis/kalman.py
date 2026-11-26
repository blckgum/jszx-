# -*- coding: utf-8 -*-
"""
二维匀加速卡尔曼滤波器模块 (2D Constant-Acceleration Kalman Filter)
=============================================================================
基于 6 维状态向量 x = [x, y, vx, vy, ax, ay]^T 与连续白噪声加加速度 (CWNA / White-Noise Jerk)
过程噪声模型，结合相机单应性雅可比传播的各向异性观测协方差 R_k、马氏距离新息门控、
Joseph 形式协方差稳定更新与物理量软限幅，在变帧率与透视噪声下平滑估计位置、速度与加速度。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


_H_OBS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
_I6 = np.eye(6, dtype=np.float64)
_H_OBS_CTRV = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
_I5 = np.eye(5, dtype=np.float64)


def _contains_bool_value(value) -> bool:
    """递归检查 ndarray（含 object 类型）与嵌套列表/元组中是否包含 Python bool 或 np.bool_。"""
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == bool or np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype.hasobject:
            # object 数组内部可能混入 Python bool、np.bool_ 或嵌套列表/ndarray，需逐元素递归排查
            return any(_contains_bool_value(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_bool_value(item) for item in value)
    return False


def _validate_positive_float(value, label: str) -> float:
    """校验有限正浮点数，显式拒绝布尔值、NaN 和 Inf。"""
    if type(value) is bool or isinstance(value, np.bool_):
        raise ValueError(f"{label} 不能为布尔值")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} 必须为有限正数")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0:
        raise ValueError(f"{label} 必须为有限正数")
    return fval


def _validate_finite_scalar(value, label: str) -> float:
    """校验有限浮点数，显式拒绝布尔值、NaN 和 Inf。"""
    if type(value) is bool or isinstance(value, np.bool_):
        raise ValueError(f"{label} 不能为布尔值")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} 必须为有限数值")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{label} 必须为有限数值")
    return fval


def _transition_matrix(dt_s: float) -> np.ndarray:
    """构建变步长 dt 下的二维匀加速状态转移矩阵 F(dt)。"""
    f_mat = _I6.copy()
    half_dt2 = 0.5 * dt_s * dt_s
    f_mat[0, 2] = dt_s
    f_mat[1, 3] = dt_s
    f_mat[0, 4] = half_dt2
    f_mat[1, 5] = half_dt2
    f_mat[2, 4] = dt_s
    f_mat[3, 5] = dt_s
    return f_mat


def _process_noise_matrix(dt_s: float, jerk_std_ms3: float) -> np.ndarray:
    """构建连续白噪声加加速度 (White-Noise Jerk) 离散化过程噪声矩阵 Q(dt)。"""
    if dt_s <= 0.0:
        return np.zeros((6, 6), dtype=np.float64)
    q_var = jerk_std_ms3 * jerk_std_ms3
    dt2 = dt_s * dt_s
    dt3 = dt2 * dt_s
    dt4 = dt3 * dt_s
    dt5 = dt4 * dt_s

    q00 = q_var * dt5 / 20.0
    q01 = q_var * dt4 / 8.0
    q02 = q_var * dt3 / 6.0
    q11 = q_var * dt3 / 3.0
    q12 = q_var * dt2 / 2.0
    q22 = q_var * dt_s

    q_mat = np.zeros((6, 6), dtype=np.float64)
    # X 分量索引 (0, 2, 4)，Y 分量索引 (1, 3, 5)
    for offset in (0, 1):
        i0, i1, i2 = offset, 2 + offset, 4 + offset
        q_mat[i0, i0] = q00
        q_mat[i0, i1] = q_mat[i1, i0] = q01
        q_mat[i0, i2] = q_mat[i2, i0] = q02
        q_mat[i1, i1] = q11
        q_mat[i1, i2] = q_mat[i2, i1] = q12
        q_mat[i2, i2] = q22
    return q_mat


def _default_jerk_for_category(category: str) -> float:
    return 1.5 if category in ("pedestrian", "person") else 2.5


def _default_max_accel_for_category(category: str) -> float:
    return 4.0 if category in ("pedestrian", "person") else 8.0


def _default_init_vel_std(category: str) -> float:
    return 4.0 if category in ("pedestrian", "person") else 10.0


def _default_init_accel_std(category: str) -> float:
    return 2.0 if category in ("pedestrian", "person") else 4.0


def _default_ctrv_accel_std(category: str) -> float:
    return 1.5 if category in ("pedestrian", "person") else 2.0


def _default_ctrv_yaw_accel_std(category: str) -> float:
    return 0.8 if category in ("pedestrian", "person") else 0.8


def _wrap_angle_rad(angle_rad: float) -> float:
    """将有限航向角规范化至 [-pi, pi] 区间。"""
    if not math.isfinite(angle_rad):
        return 0.0
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _clamp_vec2_components(vx: float, vy: float, max_norm: float) -> Optional[Tuple[float, float]]:
    """按矢量模长限幅二维分量，先归一化最大分量以防 hypot 上溢导致 inf*0=nan。"""
    if not (math.isfinite(vx) and math.isfinite(vy)):
        return None
    max_c = max(abs(vx), abs(vy))
    if max_c <= 0.0:
        return vx, vy
    nx, ny = vx / max_c, vy / max_c
    unit_mag = math.hypot(nx, ny)
    if max_c > max_norm / unit_mag:
        scale = max_norm / unit_mag
        return nx * scale, ny * scale
    return vx, vy


def _safe_diff_velocity(
    prev_x: float,
    prev_y: float,
    zx: float,
    zy: float,
    dt_s: float,
    max_speed_ms: float,
) -> Tuple[float, float]:
    """根据相邻两次有限观测差分计算初速度矢量，溢出时按缩放方向安全限幅。"""
    with np.errstate(over="ignore", invalid="ignore"):
        vx0 = (zx - prev_x) / dt_s
        vy0 = (zy - prev_y) / dt_s
    if not (math.isfinite(vx0) and math.isfinite(vy0)):
        # 极值坐标差分上溢时先减半求差并按限速缩放方向，防止 inf*0=nan
        dx_half = 0.5 * zx - 0.5 * prev_x
        dy_half = 0.5 * zy - 0.5 * prev_y
        clamped_v = _clamp_vec2_components(dx_half, dy_half, max_speed_ms)
        return (0.0, 0.0) if clamped_v is None else clamped_v
    clamped_v = _clamp_vec2_components(vx0, vy0, max_speed_ms)
    return (0.0, 0.0) if clamped_v is None else clamped_v


def _estimate_reset_velocity(
    rejected_obs: list[Tuple[float, float, float]],
    ts: float,
    zx: float,
    zy: float,
    max_speed_ms: float,
    max_accel_ms2: float,
) -> Tuple[float, float, bool]:
    """根据连续被拒观测序列估计重置初速度；若跳变速率或加速度违背物理一致性则归零重收敛。"""
    if not rejected_obs:
        return 0.0, 0.0, False
    points = [*rejected_obs, (ts, zx, zy)]
    t_first, x_first, y_first = points[0]
    dt_span = ts - t_first
    if not math.isfinite(dt_span) or dt_span <= 1e-4:
        return 0.0, 0.0, False

    seg_vels: list[Tuple[float, float, float]] = []
    with np.errstate(over="ignore", invalid="ignore"):
        for (t0, x0, y0), (t1, x1, y1) in zip(points, points[1:]):
            dt_seg = t1 - t0
            if not math.isfinite(dt_seg) or dt_seg <= 1e-4:
                return 0.0, 0.0, False
            vx_seg = (x1 - x0) / dt_seg
            vy_seg = (y1 - y0) / dt_seg
            if not (math.isfinite(vx_seg) and math.isfinite(vy_seg)):
                return 0.0, 0.0, False
            seg_speed = math.hypot(vx_seg, vy_seg)
            if not math.isfinite(seg_speed) or seg_speed > max_speed_ms:
                return 0.0, 0.0, False
            seg_vels.append((0.5 * (t0 + t1), vx_seg, vy_seg))

        max_reset_accel = 2.0 * max_accel_ms2
        for (tm0, vx0, vy0), (tm1, vx1, vy1) in zip(seg_vels, seg_vels[1:]):
            dt_mid = tm1 - tm0
            if not math.isfinite(dt_mid) or dt_mid <= 1e-4:
                return 0.0, 0.0, False
            dv = math.hypot(vx1 - vx0, vy1 - vy0)
            if not math.isfinite(dv) or (dv / dt_mid) > max_reset_accel:
                return 0.0, 0.0, False

    vx_span, vy_span = _safe_diff_velocity(
        x_first, y_first, zx, zy, dt_span, max_speed_ms
    )
    return vx_span, vy_span, True


def _resolve_obs_cov_2x2(obs_cov_m2: Optional[np.ndarray], default_r: np.ndarray) -> np.ndarray:
    """校验 2x2 观测协方差是否有限、对称且正定；否则回退各向同性默认协方差。"""
    if obs_cov_m2 is None:
        return default_r.copy()
    # 在 np.asarray(..., dtype=np.float64) 前递归拦截 object 数组及嵌套序列/ndarray 中的布尔值
    if _contains_bool_value(obs_cov_m2):
        return default_r.copy()
    try:
        arr = np.asarray(obs_cov_m2, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return default_r.copy()

    if arr.shape != (2, 2) or not np.isfinite(arr).all():
        return default_r.copy()

    # 拦截极小或超大尺度协方差，避免行列式上溢为 inf-inf=nan 绕过正定检查
    max_abs = float(np.max(np.abs(arr)))
    if not math.isfinite(max_abs) or max_abs <= 1e-12 or max_abs > 1e12:
        return default_r.copy()

    with np.errstate(over="ignore", invalid="ignore"):
        scale_ref = max(1.0, abs(float(arr[0, 1])), abs(float(arr[1, 0])))
        diff_off = abs(float(arr[0, 1] - arr[1, 0]))
        if not math.isfinite(diff_off) or diff_off > 1e-6 * scale_ref:
            return default_r.copy()
        r_mat = 0.5 * (arr + arr.T)
        if not np.isfinite(r_mat).all():
            return default_r.copy()
        # 先按 max_abs 归一化再计算 scaled_det，抑制病态与大尺度数值误差
        scaled = r_mat / max_abs
        scaled_det = float(scaled[0, 0] * scaled[1, 1] - scaled[0, 1] * scaled[1, 0])
        det = float(r_mat[0, 0] * r_mat[1, 1] - r_mat[0, 1] * r_mat[1, 0])

    if (
        r_mat[0, 0] <= 1e-12
        or r_mat[1, 1] <= 1e-12
        or not (math.isfinite(scaled_det) and scaled_det > 1e-12)
        or not (math.isfinite(det) and det > 1e-18)
    ):
        return default_r.copy()
    return r_mat


def _compute_innovation_and_d2(
    zx: float,
    zy: float,
    x_pred: np.ndarray,
    p_pred: np.ndarray,
    r_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """计算二维位置新息、新息协方差 S 及马氏距离平方 d^2。"""
    with np.errstate(over="ignore", invalid="ignore"):
        z_vec = np.array([zx, zy], dtype=np.float64)
        y_res = z_vec - x_pred[:2]
        s_mat = p_pred[:2, :2] + r_mat
        s_mat = 0.5 * (s_mat + s_mat.T)

        if np.isfinite(y_res).all() and np.isfinite(s_mat).all():
            try:
                s_inv_y = np.linalg.solve(s_mat, y_res)
                d2 = float(y_res @ s_inv_y)
            except np.linalg.LinAlgError:
                d2 = float("inf")
        else:
            d2 = float("inf")
    if not math.isfinite(d2):
        d2 = float("inf")
    return y_res, s_mat, d2


def _joseph_update_state(
    x_pred: np.ndarray,
    p_pred: np.ndarray,
    y_res: np.ndarray,
    s_mat: np.ndarray,
    r_mat: np.ndarray,
    h_obs: np.ndarray,
    eye_n: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """执行数值稳定的 Joseph 形式卡尔曼状态与协方差更新。"""
    with np.errstate(over="ignore", invalid="ignore"):
        ph_t = p_pred @ h_obs.T
        try:
            k_gain = np.linalg.solve(s_mat.T, ph_t.T).T
        except np.linalg.LinAlgError:
            k_gain = ph_t @ np.linalg.pinv(s_mat)

        i_kh = eye_n - k_gain @ h_obs
        p_upd = i_kh @ p_pred @ i_kh.T + k_gain @ r_mat @ k_gain.T
        p_upd = 0.5 * (p_upd + p_upd.T)
        x_upd = x_pred + k_gain @ y_res
    return x_upd, p_upd


class ConstantAccelerationKalman2D:
    """二维平面匀加速卡尔曼滤波器 (状态: [x, y, vx, vy, ax, ay]^T)。"""

    def __init__(
        self,
        category: str = "vehicle",
        jerk_std_ms3: Optional[float] = None,
        default_pos_std_m: float = 0.35,
        gate_chi2: float = 25.0,
        min_updates_for_gate: int = 3,
        max_consecutive_rejects: int = 3,
        max_accel_ms2: Optional[float] = None,
        max_speed_ms: float = 33.3,
        init_vel_std_ms: Optional[float] = None,
        init_accel_std_ms2: Optional[float] = None,
    ):
        if type(category) is not str or not category.strip():
            raise ValueError("category 必须是非空字符串")
        self.category = category.strip()

        self._custom_jerk = jerk_std_ms3 is not None
        self._custom_max_accel = max_accel_ms2 is not None
        self.jerk_std_ms3 = (
            _validate_positive_float(jerk_std_ms3, "jerk_std_ms3")
            if jerk_std_ms3 is not None
            else _default_jerk_for_category(self.category)
        )
        self.default_pos_std_m = _validate_positive_float(
            default_pos_std_m, "default_pos_std_m"
        )
        self.gate_chi2 = _validate_positive_float(gate_chi2, "gate_chi2")
        if type(min_updates_for_gate) is not int or min_updates_for_gate < 1:
            raise ValueError("min_updates_for_gate 必须是 >= 1 的整数")
        if type(max_consecutive_rejects) is not int or max_consecutive_rejects < 1:
            raise ValueError("max_consecutive_rejects 必须是 >= 1 的整数")
        self.min_updates_for_gate = min_updates_for_gate
        self.max_consecutive_rejects = max_consecutive_rejects

        self.max_accel_ms2 = (
            _validate_positive_float(max_accel_ms2, "max_accel_ms2")
            if max_accel_ms2 is not None
            else _default_max_accel_for_category(self.category)
        )
        self.max_speed_ms = _validate_positive_float(max_speed_ms, "max_speed_ms")
        self.init_vel_std_ms = (
            _validate_positive_float(init_vel_std_ms, "init_vel_std_ms")
            if init_vel_std_ms is not None
            else _default_init_vel_std(self.category)
        )
        self.init_accel_std_ms2 = (
            _validate_positive_float(init_accel_std_ms2, "init_accel_std_ms2")
            if init_accel_std_ms2 is not None
            else _default_init_accel_std(self.category)
        )

        self._default_R = (self.default_pos_std_m ** 2) * np.eye(2, dtype=np.float64)
        self._x = np.zeros(6, dtype=np.float64)
        self._P = np.eye(6, dtype=np.float64)
        self._last_timestamp_s: Optional[float] = None
        self._update_count: int = 0
        self._consecutive_rejects: int = 0
        self._rejected_obs: list[Tuple[float, float, float]] = []
        self._last_rejected_obs: Optional[Tuple[float, float, float]] = None
        self._reset_count: int = 0
        self._was_reset: bool = False
        self.last_d2: Optional[float] = None
        self.last_accepted: bool = False

    def set_category(self, category: str):
        """在类别确认切换时同步更新默认过程噪声与加速度限幅先验。"""
        if type(category) is not str or not category.strip():
            raise ValueError("category 必须是非空字符串")
        self.category = category.strip()
        if not self._custom_jerk:
            self.jerk_std_ms3 = _default_jerk_for_category(self.category)
        if not self._custom_max_accel:
            self.max_accel_ms2 = _default_max_accel_for_category(self.category)

    def _resolve_obs_cov(self, obs_cov_m2: Optional[np.ndarray]) -> np.ndarray:
        """校验 2x2 观测协方差是否有限、对称且正定；否则回退各向同性默认协方差。"""
        return _resolve_obs_cov_2x2(obs_cov_m2, self._default_R)

    def _build_reset_cov(self, r_mat: np.ndarray) -> np.ndarray:
        p_mat = np.diag(
            [
                1.0,
                1.0,
                self.init_vel_std_ms ** 2,
                self.init_vel_std_ms ** 2,
                self.init_accel_std_ms2 ** 2,
                self.init_accel_std_ms2 ** 2,
            ]
        ).astype(np.float64)
        p_mat[:2, :2] = r_mat
        return p_mat

    def _reset_to_observation(
        self,
        ts: float,
        zx: float,
        zy: float,
        r_mat: np.ndarray,
        *,
        count_reset: bool = False,
    ) -> bool:
        """将滤波器安全重置为以当前有限观测为中心的初始状态与协方差。"""
        self._x[:] = (zx, zy, 0.0, 0.0, 0.0, 0.0)
        self._P = self._build_reset_cov(r_mat)
        self._last_timestamp_s = ts
        self._update_count = 1
        self._consecutive_rejects = 0
        self._rejected_obs.clear()
        self._last_rejected_obs = None
        if count_reset:
            self._reset_count += 1
            self._was_reset = True
        self.last_d2 = 0.0
        self.last_accepted = True
        return True

    @staticmethod
    def _clamp_vec2(vx: float, vy: float, max_norm: float) -> Optional[Tuple[float, float]]:
        """按矢量模长限幅二维分量，先归一化最大分量以防 hypot 上溢导致 inf*0=nan。"""
        return _clamp_vec2_components(vx, vy, max_norm)

    def _clamp_physical_kinematics(self, state: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """按矢量模长软限幅速度与加速度，保持运动方向不变；若传入候选状态含非有限值则返回 None。"""
        target = self._x if state is None else state
        with np.errstate(over="ignore", invalid="ignore"):
            v_clamped = self._clamp_vec2(float(target[2]), float(target[3]), self.max_speed_ms)
            a_clamped = self._clamp_vec2(float(target[4]), float(target[5]), self.max_accel_ms2)
        if v_clamped is None or a_clamped is None:
            if state is None:
                # 兜底清理就地状态中的非有限分量，防止污染后续步
                target[2], target[3] = (0.0, 0.0) if v_clamped is None else v_clamped
                target[4], target[5] = (0.0, 0.0) if a_clamped is None else a_clamped
                return target
            return None
        target[2], target[3] = v_clamped
        target[4], target[5] = a_clamped
        return target

    def update(
        self,
        timestamp_s: float,
        x_m: float,
        y_m: float,
        obs_cov_m2: Optional[np.ndarray] = None,
    ) -> bool:
        """融合新一帧地面坐标观测，返回本帧观测是否被接受（未触发门控拒绝）。"""
        ts = _validate_finite_scalar(timestamp_s, "timestamp_s")
        zx = _validate_finite_scalar(x_m, "x_m")
        zy = _validate_finite_scalar(y_m, "y_m")

        if self._last_timestamp_s is not None and ts <= self._last_timestamp_s:
            raise ValueError("timestamp_s 必须逐帧严格递增")

        r_mat = self._resolve_obs_cov(obs_cov_m2)
        self._was_reset = False

        if self._update_count == 0 or self._last_timestamp_s is None:
            return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=False)

        dt_s = ts - self._last_timestamp_s
        with np.errstate(over="ignore", invalid="ignore"):
            f_mat = _transition_matrix(dt_s)
            q_mat = _process_noise_matrix(dt_s, self.jerk_std_ms3)

            x_pred = f_mat @ self._x
            p_pred = f_mat @ self._P @ f_mat.T + q_mat
            p_pred = 0.5 * (p_pred + p_pred.T)

        # 极端 dt（如 1e80）导致预测状态或协方差溢出时，先验失效，安全重置到当前有限观测
        if not (np.isfinite(x_pred).all() and np.isfinite(p_pred).all()):
            return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)

        y_res, s_mat, d2 = _compute_innovation_and_d2(zx, zy, x_pred, p_pred, r_mat)
        self.last_d2 = d2

        # 马氏距离新息门控：连续拒收达到上限时自动重置，避免真实剧烈机动锁死滤波器
        if self._update_count >= self.min_updates_for_gate and (
            not math.isfinite(d2) or d2 > self.gate_chi2
        ):
            self._consecutive_rejects += 1
            if self._consecutive_rejects < self.max_consecutive_rejects:
                x_cand = self._clamp_physical_kinematics(x_pred.copy())
                if x_cand is None or not (np.isfinite(x_cand).all() and np.isfinite(p_pred).all()):
                    return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
                self._rejected_obs.append((ts, zx, zy))
                self._last_rejected_obs = (ts, zx, zy)
                self._x = x_cand
                self._P = p_pred
                self._last_timestamp_s = ts
                self.last_accepted = False
                return False

            # 连续剧烈偏移超过容忍帧数：仅在连续被拒轨迹运动学一致时用跨度差分初始化速度，否则归零重收敛
            vx0, vy0, has_vel_init = _estimate_reset_velocity(
                self._rejected_obs,
                ts,
                zx,
                zy,
                self.max_speed_ms,
                self.max_accel_ms2,
            )
            x_cand = np.array([zx, zy, vx0, vy0, 0.0, 0.0], dtype=np.float64)
            x_cand = self._clamp_physical_kinematics(x_cand)
            p_cand = self._build_reset_cov(r_mat)
            if x_cand is None or not (np.isfinite(x_cand).all() and np.isfinite(p_cand).all()):
                return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
            self._x = x_cand
            self._P = p_cand
            self._last_timestamp_s = ts
            self._consecutive_rejects = 0
            self._rejected_obs.clear()
            self._last_rejected_obs = None
            self._update_count = 2 if has_vel_init else 1
            self._reset_count += 1
            self._was_reset = True
            self.last_accepted = True
            return True

        self._consecutive_rejects = 0
        self._rejected_obs.clear()
        self._last_rejected_obs = None

        if not (np.isfinite(y_res).all() and np.isfinite(s_mat).all()):
            # 极值坐标导致残差或新息矩阵溢出时安全保留有限预测状态，不污染滤波器
            x_cand = self._clamp_physical_kinematics(x_pred.copy())
            if x_cand is None or not np.isfinite(s_mat).all():
                return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
            self._x = x_cand
            self._P = p_pred
            self._last_timestamp_s = ts
            self.last_accepted = False
            return False

        # Joseph 形式协方差更新: P = (I - K H) P (I - K H)^T + K R K^T
        x_upd, p_upd = _joseph_update_state(
            x_pred, p_pred, y_res, s_mat, r_mat, _H_OBS, _I6
        )

        x_cand = self._clamp_physical_kinematics(x_upd)
        # 提交前校验有限性：极端观测导致状态上溢则退回有限预测，协方差上溢则安全重置
        if not np.isfinite(p_upd).all():
            return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
        if x_cand is None or not np.isfinite(x_cand).all():
            x_fallback = self._clamp_physical_kinematics(x_pred.copy())
            if x_fallback is None or not np.isfinite(x_fallback).all():
                return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
            self._x = x_fallback
            self._P = p_pred
            self._last_timestamp_s = ts
            self.last_accepted = False
            return False

        self._x = x_cand
        self._P = p_upd
        self._last_timestamp_s = ts
        self._update_count += 1
        self.last_accepted = True
        return True

    def predict_at(self, dt_s: float) -> Tuple[float, float, np.ndarray]:
        """在不改变当前滤波状态的前提下，前向预测 dt_s 秒后的位置与 2x2 位置协方差。"""
        dt = _validate_finite_scalar(dt_s, "dt_s")
        if dt < 0.0:
            raise ValueError("dt_s 必须为非负有限数")
        if self._update_count == 0:
            raise RuntimeError("卡尔曼滤波器尚未接收到任何观测，无法执行预测")
        if dt == 0.0:
            return float(self._x[0]), float(self._x[1]), self._P[:2, :2].copy()

        with np.errstate(over="ignore", invalid="ignore"):
            f_mat = _transition_matrix(dt)
            q_mat = _process_noise_matrix(dt, self.jerk_std_ms3)
            x_pred = f_mat @ self._x
            p_pred = f_mat @ self._P @ f_mat.T + q_mat
            pos_cov = 0.5 * (p_pred[:2, :2] + p_pred[:2, :2].T)
        px, py = float(x_pred[0]), float(x_pred[1])
        if not (math.isfinite(px) and math.isfinite(py) and np.isfinite(pos_cov).all()):
            raise ValueError("dt_s 过大导致前向预测状态数值溢出")
        return px, py, pos_cov

    @property
    def position_xy(self) -> Tuple[float, float]:
        return float(self._x[0]), float(self._x[1])

    @property
    def velocity_xy(self) -> Tuple[float, float]:
        return float(self._x[2]), float(self._x[3])

    @property
    def acceleration_xy(self) -> Tuple[float, float]:
        return float(self._x[4]), float(self._x[5])

    @property
    def pos_cov_2x2(self) -> np.ndarray:
        return self._P[:2, :2].copy()

    @property
    def pos_std_m(self) -> float:
        """二维欧氏位置标准差 sqrt(P_xx + P_yy)（单位：米）。"""
        # 用各分量平方根求 hypot 避免超大有限方差直接相加时触发上溢
        return math.hypot(
            math.sqrt(max(0.0, float(self._P[0, 0]))),
            math.sqrt(max(0.0, float(self._P[1, 1]))),
        )

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def consecutive_rejects(self) -> int:
        return self._consecutive_rejects

    @property
    def was_reset(self) -> bool:
        """最近一次 update 是否触发了滤波器状态重置（不含首帧初始化）。"""
        return self._was_reset

    @property
    def reset_count(self) -> int:
        """累计触发滤波器状态重置的次数（不含首帧初始化）。"""
        return self._reset_count

    def clone(self) -> "ConstantAccelerationKalman2D":
        """返回与当前滤波器状态完全独立的深拷贝实例。"""
        copy_obj = ConstantAccelerationKalman2D(
            category=self.category,
            jerk_std_ms3=self.jerk_std_ms3 if self._custom_jerk else None,
            default_pos_std_m=self.default_pos_std_m,
            gate_chi2=self.gate_chi2,
            min_updates_for_gate=self.min_updates_for_gate,
            max_consecutive_rejects=self.max_consecutive_rejects,
            max_accel_ms2=self.max_accel_ms2 if self._custom_max_accel else None,
            max_speed_ms=self.max_speed_ms,
            init_vel_std_ms=self.init_vel_std_ms,
            init_accel_std_ms2=self.init_accel_std_ms2,
        )
        copy_obj._x = self._x.copy()
        copy_obj._P = self._P.copy()
        copy_obj._last_timestamp_s = self._last_timestamp_s
        copy_obj._update_count = self._update_count
        copy_obj._consecutive_rejects = self._consecutive_rejects
        copy_obj._rejected_obs = list(self._rejected_obs)
        copy_obj._last_rejected_obs = self._last_rejected_obs
        copy_obj._reset_count = self._reset_count
        copy_obj._was_reset = self._was_reset
        copy_obj.last_d2 = self.last_d2
        copy_obj.last_accepted = self.last_accepted
        return copy_obj


def _ctrv_predict_and_jacobian(
    state: np.ndarray, dt_s: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """计算 CTRV 非线性状态转移与解析 5x5 雅可比矩阵 F_k；溢出时返回 None。"""
    if not np.isfinite(state).all() or not math.isfinite(dt_s):
        return None
    px, py, v, psi, w = (float(state[i]) for i in range(5))
    f_mat = _I5.copy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = w * dt_s
        psi1 = psi + delta
        if not (math.isfinite(delta) and math.isfinite(psi1)):
            return None
        s0, c0 = math.sin(psi), math.cos(psi)
        dt2 = dt_s * dt_s
        if not math.isfinite(dt2):
            return None

        if abs(w) < 1e-5:
            # |omega| 极小时采用一阶泰勒极限直线分支防止除零，同时保留对 omega 的非零解析导数
            c_eff = dt_s * c0 - 0.5 * w * dt2 * s0
            s_eff = dt_s * s0 + 0.5 * w * dt2 * c0
            nx = px + v * c_eff
            ny = py + v * s_eff
            f_mat[0, 2] = c_eff
            f_mat[0, 3] = -v * s_eff
            f_mat[0, 4] = -0.5 * v * dt2 * s0
            f_mat[1, 2] = s_eff
            f_mat[1, 3] = v * c_eff
            f_mat[1, 4] = 0.5 * v * dt2 * c0
        elif abs(delta) < 1e-2:
            # 小转角区域 (|omega*dt| < 1e-2) 用高阶泰勒级数规避 (sin1 - sin0)/w^2 相消误差
            d2 = delta * delta
            f1 = 1.0 - d2 / 6.0 + (d2 * d2) / 120.0
            f2 = 0.5 * delta - (delta * d2) / 24.0 + (delta * d2 * d2) / 720.0
            g1 = -delta / 3.0 + (delta * d2) / 30.0 - (delta * d2 * d2) / 840.0
            g2 = 0.5 - d2 / 8.0 + (d2 * d2) / 144.0
            c_eff = dt_s * (c0 * f1 - s0 * f2)
            s_eff = dt_s * (s0 * f1 + c0 * f2)
            nx = px + v * c_eff
            ny = py + v * s_eff
            f_mat[0, 2] = c_eff
            f_mat[0, 3] = -v * s_eff
            f_mat[0, 4] = v * dt2 * (c0 * g1 - s0 * g2)
            f_mat[1, 2] = s_eff
            f_mat[1, 3] = v * c_eff
            f_mat[1, 4] = v * dt2 * (s0 * g1 + c0 * g2)
        else:
            s1, c1 = math.sin(psi1), math.cos(psi1)
            inv_w = 1.0 / w
            inv_w2 = inv_w * inv_w
            ds = s1 - s0
            dc = c1 - c0
            nx = px + v * ds * inv_w
            ny = py + v * (-dc) * inv_w
            f_mat[0, 2] = ds * inv_w
            f_mat[0, 3] = v * dc * inv_w
            f_mat[0, 4] = v * (delta * c1 - ds) * inv_w2
            f_mat[1, 2] = (-dc) * inv_w
            f_mat[1, 3] = v * ds * inv_w
            f_mat[1, 4] = v * (delta * s1 + dc) * inv_w2

        f_mat[3, 4] = dt_s
        npsi = _wrap_angle_rad(psi1)
        x_pred = np.array([nx, ny, v, npsi, w], dtype=np.float64)
    if not (np.isfinite(x_pred).all() and np.isfinite(f_mat).all()):
        return None
    return x_pred, f_mat


def _ctrv_process_noise(
    dt_s: float,
    psi: float,
    accel_std_ms2: float,
    yaw_accel_std_rad_s2: float,
) -> Optional[np.ndarray]:
    """构建 CTRV 过程噪声协方差 Q_k（纵向加速度噪声 + 横摆角加速度噪声 + 微小横向扩散底噪）。"""
    if dt_s <= 0.0:
        return np.zeros((5, 5), dtype=np.float64)
    if not (math.isfinite(dt_s) and math.isfinite(psi)):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        dt2 = dt_s * dt_s
        half_dt2 = 0.5 * dt2
        c0, s0 = math.cos(psi), math.sin(psi)
        g_mat = np.array(
            [
                [half_dt2 * c0, 0.0],
                [half_dt2 * s0, 0.0],
                [dt_s, 0.0],
                [0.0, half_dt2],
                [0.0, dt_s],
            ],
            dtype=np.float64,
        )
        q_u = np.diag([accel_std_ms2 * accel_std_ms2, yaw_accel_std_rad_s2 * yaw_accel_std_rad_s2])
        q_mat = g_mat @ q_u @ g_mat.T
        # 补充微小正定底噪防止侧向零方差退化，保证任意 dt > 0 时预测协方差严格正定
        q_mat[0, 0] += 1e-3 * dt_s
        q_mat[1, 1] += 1e-3 * dt_s
        q_mat[2, 2] += 1e-6 * dt_s
        q_mat[3, 3] += 1e-4 * dt_s
        q_mat[4, 4] += 1e-6 * dt_s
    if not np.isfinite(q_mat).all():
        return None
    return q_mat


class ConstantTurnRateKalman2D:
    """二维恒定转率与速度扩展卡尔曼滤波器 (CTRV EKF，状态: [x, y, v, psi, omega]^T)。"""

    def __init__(
        self,
        category: str = "vehicle",
        accel_std_ms2: Optional[float] = None,
        yaw_accel_std_rad_s2: Optional[float] = None,
        default_pos_std_m: float = 0.35,
        gate_chi2: float = 25.0,
        min_updates_for_gate: int = 3,
        max_consecutive_rejects: int = 3,
        max_speed_ms: float = 33.3,
        max_yaw_rate_rad_s: float = 3.0,
        init_vel_std_ms: Optional[float] = None,
        init_psi_std_rad: float = math.pi,
        init_yaw_rate_std_rad_s: float = 0.8,
    ):
        if type(category) is not str or not category.strip():
            raise ValueError("category 必须是非空字符串")
        self.category = category.strip()

        self._custom_accel_std = accel_std_ms2 is not None
        self._custom_yaw_accel_std = yaw_accel_std_rad_s2 is not None
        self.accel_std_ms2 = (
            _validate_positive_float(accel_std_ms2, "accel_std_ms2")
            if accel_std_ms2 is not None
            else _default_ctrv_accel_std(self.category)
        )
        self.yaw_accel_std_rad_s2 = (
            _validate_positive_float(yaw_accel_std_rad_s2, "yaw_accel_std_rad_s2")
            if yaw_accel_std_rad_s2 is not None
            else _default_ctrv_yaw_accel_std(self.category)
        )
        self.default_pos_std_m = _validate_positive_float(
            default_pos_std_m, "default_pos_std_m"
        )
        self.gate_chi2 = _validate_positive_float(gate_chi2, "gate_chi2")
        if type(min_updates_for_gate) is not int or min_updates_for_gate < 1:
            raise ValueError("min_updates_for_gate 必须是 >= 1 的整数")
        if type(max_consecutive_rejects) is not int or max_consecutive_rejects < 1:
            raise ValueError("max_consecutive_rejects 必须是 >= 1 的整数")
        self.min_updates_for_gate = min_updates_for_gate
        self.max_consecutive_rejects = max_consecutive_rejects

        self.max_speed_ms = _validate_positive_float(max_speed_ms, "max_speed_ms")
        self.max_yaw_rate_rad_s = _validate_positive_float(
            max_yaw_rate_rad_s, "max_yaw_rate_rad_s"
        )
        self.init_vel_std_ms = (
            _validate_positive_float(init_vel_std_ms, "init_vel_std_ms")
            if init_vel_std_ms is not None
            else _default_init_vel_std(self.category)
        )
        self.init_psi_std_rad = _validate_positive_float(
            init_psi_std_rad, "init_psi_std_rad"
        )
        self.init_yaw_rate_std_rad_s = _validate_positive_float(
            init_yaw_rate_std_rad_s, "init_yaw_rate_std_rad_s"
        )

        self._default_R = (self.default_pos_std_m ** 2) * np.eye(2, dtype=np.float64)
        self._x = np.zeros(5, dtype=np.float64)
        self._P = np.eye(5, dtype=np.float64)
        self._last_timestamp_s: Optional[float] = None
        self._update_count: int = 0
        self._consecutive_rejects: int = 0
        self._rejected_obs: list[Tuple[float, float, float]] = []
        self._last_rejected_obs: Optional[Tuple[float, float, float]] = None
        self._reset_count: int = 0
        self._was_reset: bool = False
        self.last_d2: Optional[float] = None
        self.last_accepted: bool = False

    def set_category(self, category: str):
        """在类别确认切换时同步更新 CTRV 默认纵向/角加速度过程噪声先验。"""
        if type(category) is not str or not category.strip():
            raise ValueError("category 必须是非空字符串")
        self.category = category.strip()
        if not self._custom_accel_std:
            self.accel_std_ms2 = _default_ctrv_accel_std(self.category)
        if not self._custom_yaw_accel_std:
            self.yaw_accel_std_rad_s2 = _default_ctrv_yaw_accel_std(self.category)

    def _resolve_obs_cov(self, obs_cov_m2: Optional[np.ndarray]) -> np.ndarray:
        """复用 2x2 观测协方差校验与默认回退逻辑。"""
        return _resolve_obs_cov_2x2(obs_cov_m2, self._default_R)

    def _build_reset_cov(
        self,
        r_mat: np.ndarray,
        *,
        vel_initialized: bool = False,
    ) -> np.ndarray:
        psi_std = 0.6 if vel_initialized else self.init_psi_std_rad
        vel_std = min(self.init_vel_std_ms, 4.0) if vel_initialized else self.init_vel_std_ms
        p_mat = np.diag(
            [
                1.0,
                1.0,
                vel_std ** 2,
                psi_std ** 2,
                self.init_yaw_rate_std_rad_s ** 2,
            ]
        ).astype(np.float64)
        p_mat[:2, :2] = r_mat
        return p_mat

    def _reset_to_observation(
        self,
        ts: float,
        zx: float,
        zy: float,
        r_mat: np.ndarray,
        *,
        count_reset: bool = False,
    ) -> bool:
        """将 CTRV 滤波器安全重置为以当前有限观测为中心的初始状态。"""
        self._x[:] = (zx, zy, 0.0, 0.0, 0.0)
        self._P = self._build_reset_cov(r_mat, vel_initialized=False)
        self._last_timestamp_s = ts
        self._update_count = 1
        self._consecutive_rejects = 0
        self._rejected_obs.clear()
        self._last_rejected_obs = None
        if count_reset:
            self._reset_count += 1
            self._was_reset = True
        self.last_d2 = 0.0
        self.last_accepted = True
        return True

    def _clamp_ctrv_state(self, state: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """规范航向角至 [-pi, pi] 并限幅速度与横摆角速度；遇到非有限值返回 None。"""
        target = self._x if state is None else state
        if not np.isfinite(target).all():
            if state is None:
                target[2:] = 0.0
                return target
            return None
        v = float(target[2])
        psi = float(target[3])
        omega = float(target[4])
        # 若更新后速度取负，将其反转至正速率并使航向同步偏转 pi，保持速度矢量方向一致
        if v < 0.0:
            v = -v
            psi = psi + math.pi
        v = min(v, self.max_speed_ms)
        psi = _wrap_angle_rad(psi)
        omega = max(-self.max_yaw_rate_rad_s, min(self.max_yaw_rate_rad_s, omega))
        target[2], target[3], target[4] = v, psi, omega
        return target

    def _velocity_psi_from_vec(self, vx: float, vy: float) -> Tuple[float, float]:
        with np.errstate(over="ignore", invalid="ignore"):
            v0 = math.hypot(vx, vy)
            if not math.isfinite(v0):
                clamped = _clamp_vec2_components(vx, vy, self.max_speed_ms)
                if clamped is None:
                    return 0.0, 0.0
                vx, vy = clamped
                v0 = math.hypot(vx, vy)
            v0 = min(v0, self.max_speed_ms)
            psi0 = _wrap_angle_rad(math.atan2(vy, vx)) if v0 > 1e-6 else 0.0
        return float(v0), float(psi0)

    def update(
        self,
        timestamp_s: float,
        x_m: float,
        y_m: float,
        obs_cov_m2: Optional[np.ndarray] = None,
    ) -> bool:
        """融合新一帧地面观测，执行非线性预测、马氏门控与 Joseph 协方差更新。"""
        ts = _validate_finite_scalar(timestamp_s, "timestamp_s")
        zx = _validate_finite_scalar(x_m, "x_m")
        zy = _validate_finite_scalar(y_m, "y_m")

        if self._last_timestamp_s is not None and ts <= self._last_timestamp_s:
            raise ValueError("timestamp_s 必须逐帧严格递增")

        r_mat = self._resolve_obs_cov(obs_cov_m2)
        self._was_reset = False

        if self._update_count == 0 or self._last_timestamp_s is None:
            return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=False)

        dt_s = ts - self._last_timestamp_s

        # 第 2 帧观测到达时：若速度仍未建立，通过两点差分引导初始航向与速率，打破 v=0 时 psi 不可观奇异
        if self._update_count == 1:
            gate_reset = False
            if self._update_count >= self.min_updates_for_gate:
                with np.errstate(over="ignore", invalid="ignore"):
                    iso_var = (self.init_vel_std_ms * dt_s) ** 2
                    p_iso = self._P[:2, :2] + iso_var * np.eye(2, dtype=np.float64)
                    _, _, d2_init = _compute_innovation_and_d2(
                        zx, zy, self._x, p_iso, r_mat
                    )
                self.last_d2 = d2_init
                if not math.isfinite(d2_init) or d2_init > self.gate_chi2:
                    self._consecutive_rejects += 1
                    if self._consecutive_rejects < self.max_consecutive_rejects:
                        self._rejected_obs.append((ts, zx, zy))
                        self._last_rejected_obs = (ts, zx, zy)
                        self._last_timestamp_s = ts
                        self.last_accepted = False
                        return False
                    gate_reset = True
            if gate_reset:
                vx0, vy0, has_vel_init = _estimate_reset_velocity(
                    self._rejected_obs,
                    ts,
                    zx,
                    zy,
                    self.max_speed_ms,
                    _default_max_accel_for_category(self.category),
                )
                v0, psi0 = (
                    self._velocity_psi_from_vec(vx0, vy0) if has_vel_init else (0.0, 0.0)
                )
                x_cand = self._clamp_ctrv_state(
                    np.array([zx, zy, v0, psi0, 0.0], dtype=np.float64)
                )
                p_cand = self._build_reset_cov(
                    r_mat, vel_initialized=(has_vel_init and v0 > 1e-6)
                )
                if x_cand is None or not (np.isfinite(x_cand).all() and np.isfinite(p_cand).all()):
                    return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
                self._x = x_cand
                self._P = p_cand
                self._last_timestamp_s = ts
                # 统一置为 2 使下一帧走带观测协方差 R_k 阻尼的 EKF 更新，避免 _update_count==1 退化为无滤波单帧两点差分
                self._update_count = 2
                self._consecutive_rejects = 0
                self._rejected_obs.clear()
                self._last_rejected_obs = None
                self._reset_count += 1
                self._was_reset = True
                self.last_accepted = True
                return True
            vx0, vy0 = _safe_diff_velocity(
                float(self._x[0]), float(self._x[1]), zx, zy, dt_s, self.max_speed_ms
            )
            v0, psi0 = self._velocity_psi_from_vec(vx0, vy0)
            x_cand = self._clamp_ctrv_state(np.array([zx, zy, v0, psi0, 0.0], dtype=np.float64))
            p_cand = self._build_reset_cov(r_mat, vel_initialized=(v0 > 1e-6))
            if x_cand is None or not (np.isfinite(x_cand).all() and np.isfinite(p_cand).all()):
                return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
            self._x = x_cand
            self._P = p_cand
            self._last_timestamp_s = ts
            self._update_count = 2
            self._consecutive_rejects = 0
            self._rejected_obs.clear()
            self._last_rejected_obs = None
            self.last_d2 = 0.0
            self.last_accepted = True
            return True

        x_work = self._x.copy()
        # 静止重启瞬间速率近似 0 时，先用有限位移对齐先验航向，使速度雅可比与真实运动方向一致
        if abs(float(x_work[2])) < 1e-3:
            with np.errstate(over="ignore", invalid="ignore"):
                dx_w = zx - float(x_work[0])
                dy_w = zy - float(x_work[1])
                dist_w = math.hypot(dx_w, dy_w)
                if math.isfinite(dist_w) and dist_w > 1e-3 and (dist_w / max(dt_s, 1e-4)) <= self.max_speed_ms:
                    x_work[3] = _wrap_angle_rad(math.atan2(dy_w, dx_w))

        step_res = _ctrv_predict_and_jacobian(x_work, dt_s)
        q_mat = _ctrv_process_noise(
            dt_s, float(x_work[3]), self.accel_std_ms2, self.yaw_accel_std_rad_s2
        )
        if step_res is None or q_mat is None:
            return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
        x_pred, f_mat = step_res
        with np.errstate(over="ignore", invalid="ignore"):
            p_pred = f_mat @ self._P @ f_mat.T + q_mat
            p_pred = 0.5 * (p_pred + p_pred.T)

        if not (np.isfinite(x_pred).all() and np.isfinite(p_pred).all()):
            return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)

        y_res, s_mat, d2 = _compute_innovation_and_d2(zx, zy, x_pred, p_pred, r_mat)
        self.last_d2 = d2

        # 马氏距离门控与连续拒收自动重置
        if self._update_count >= self.min_updates_for_gate and (
            not math.isfinite(d2) or d2 > self.gate_chi2
        ):
            self._consecutive_rejects += 1
            if self._consecutive_rejects < self.max_consecutive_rejects:
                x_cand = self._clamp_ctrv_state(x_pred.copy())
                if x_cand is None or not (np.isfinite(x_cand).all() and np.isfinite(p_pred).all()):
                    return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
                self._rejected_obs.append((ts, zx, zy))
                self._last_rejected_obs = (ts, zx, zy)
                self._x = x_cand
                self._P = p_pred
                self._last_timestamp_s = ts
                self.last_accepted = False
                return False

            vx0, vy0, has_vel_init = _estimate_reset_velocity(
                self._rejected_obs,
                ts,
                zx,
                zy,
                self.max_speed_ms,
                _default_max_accel_for_category(self.category),
            )
            v0, psi0 = (
                self._velocity_psi_from_vec(vx0, vy0) if has_vel_init else (0.0, 0.0)
            )
            x_cand = self._clamp_ctrv_state(
                np.array([zx, zy, v0, psi0, 0.0], dtype=np.float64)
            )
            p_cand = self._build_reset_cov(
                r_mat, vel_initialized=(has_vel_init and v0 > 1e-6)
            )
            if x_cand is None or not (np.isfinite(x_cand).all() and np.isfinite(p_cand).all()):
                return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
            self._x = x_cand
            self._P = p_cand
            self._last_timestamp_s = ts
            self._consecutive_rejects = 0
            self._rejected_obs.clear()
            self._last_rejected_obs = None
            # 统一置为 2 使重置后首帧即通过 R_k 加权 Joseph EKF 平滑收敛，避免单帧两点差分放大跳变尖峰
            self._update_count = 2
            self._reset_count += 1
            self._was_reset = True
            self.last_accepted = True
            return True

        self._consecutive_rejects = 0
        self._rejected_obs.clear()
        self._last_rejected_obs = None

        if not (np.isfinite(y_res).all() and np.isfinite(s_mat).all()):
            x_cand = self._clamp_ctrv_state(x_pred.copy())
            if x_cand is None or not np.isfinite(s_mat).all():
                return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
            self._x = x_cand
            self._P = p_pred
            self._last_timestamp_s = ts
            self.last_accepted = False
            return False

        x_upd, p_upd = _joseph_update_state(
            x_pred, p_pred, y_res, s_mat, r_mat, _H_OBS_CTRV, _I5
        )
        x_cand = self._clamp_ctrv_state(x_upd)
        if not np.isfinite(p_upd).all():
            return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
        if x_cand is None or not np.isfinite(x_cand).all():
            x_fallback = self._clamp_ctrv_state(x_pred.copy())
            if x_fallback is None or not np.isfinite(x_fallback).all():
                return self._reset_to_observation(ts, zx, zy, r_mat, count_reset=True)
            self._x = x_fallback
            self._P = p_pred
            self._last_timestamp_s = ts
            self.last_accepted = False
            return False

        self._x = x_cand
        self._P = p_upd
        self._last_timestamp_s = ts
        self._update_count += 1
        self.last_accepted = True
        return True

    def predict_at(self, dt_s: float) -> Tuple[float, float, np.ndarray]:
        """在不改变当前状态的前提下，按 CTRV 非线性运动模型预测 dt_s 秒后位置与 2x2 协方差。"""
        dt = _validate_finite_scalar(dt_s, "dt_s")
        if dt < 0.0:
            raise ValueError("dt_s 必须为非负有限数")
        if self._update_count == 0:
            raise RuntimeError("卡尔曼滤波器尚未接收到任何观测，无法执行预测")
        if dt == 0.0:
            return float(self._x[0]), float(self._x[1]), self._P[:2, :2].copy()

        step_res = _ctrv_predict_and_jacobian(self._x, dt)
        q_mat = _ctrv_process_noise(
            dt, float(self._x[3]), self.accel_std_ms2, self.yaw_accel_std_rad_s2
        )
        if step_res is None or q_mat is None:
            raise ValueError("dt_s 过大导致前向预测状态数值溢出")
        x_pred, f_mat = step_res
        with np.errstate(over="ignore", invalid="ignore"):
            p_pred = f_mat @ self._P @ f_mat.T + q_mat
            pos_cov = 0.5 * (p_pred[:2, :2] + p_pred[:2, :2].T)
        px, py = float(x_pred[0]), float(x_pred[1])
        if not (math.isfinite(px) and math.isfinite(py) and np.isfinite(pos_cov).all()):
            raise ValueError("dt_s 过大导致前向预测状态数值溢出")
        return px, py, pos_cov

    @property
    def position_xy(self) -> Tuple[float, float]:
        return float(self._x[0]), float(self._x[1])

    @property
    def velocity_xy(self) -> Tuple[float, float]:
        v = float(self._x[2])
        psi = float(self._x[3])
        return float(v * math.cos(psi)), float(v * math.sin(psi))

    @property
    def accel_xy(self) -> Tuple[float, float]:
        """由向心加速度公式 (-v*omega*sin(psi), v*omega*cos(psi)) 计算平面加速度分量。"""
        v = float(self._x[2])
        psi = float(self._x[3])
        omega = float(self._x[4])
        return float(-v * omega * math.sin(psi)), float(v * omega * math.cos(psi))

    @property
    def acceleration_xy(self) -> Tuple[float, float]:
        """向后兼容 CA 滤波器的加速度属性别名。"""
        return self.accel_xy

    @property
    def pos_cov_2x2(self) -> np.ndarray:
        return self._P[:2, :2].copy()

    @property
    def pos_std_m(self) -> float:
        """二维欧氏位置标准差 sqrt(P_xx + P_yy)（单位：米）。"""
        return math.hypot(
            math.sqrt(max(0.0, float(self._P[0, 0]))),
            math.sqrt(max(0.0, float(self._P[1, 1]))),
        )

    @property
    def speed_ms(self) -> float:
        return float(self._x[2])

    @property
    def heading_deg(self) -> float:
        return (math.degrees(float(self._x[3])) + 360.0) % 360.0

    @property
    def yaw_rate_rad_s(self) -> float:
        return float(self._x[4])

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def consecutive_rejects(self) -> int:
        return self._consecutive_rejects

    @property
    def was_reset(self) -> bool:
        """最近一次 update 是否触发了滤波器状态重置（不含首帧初始化）。"""
        return self._was_reset

    @property
    def reset_count(self) -> int:
        """累计触发滤波器状态重置的次数（不含首帧初始化）。"""
        return self._reset_count

    def clone(self) -> "ConstantTurnRateKalman2D":
        """返回与当前 CTRV 滤波器状态完全独立的深拷贝实例。"""
        copy_obj = ConstantTurnRateKalman2D(
            category=self.category,
            accel_std_ms2=self.accel_std_ms2 if self._custom_accel_std else None,
            yaw_accel_std_rad_s2=self.yaw_accel_std_rad_s2 if self._custom_yaw_accel_std else None,
            default_pos_std_m=self.default_pos_std_m,
            gate_chi2=self.gate_chi2,
            min_updates_for_gate=self.min_updates_for_gate,
            max_consecutive_rejects=self.max_consecutive_rejects,
            max_speed_ms=self.max_speed_ms,
            max_yaw_rate_rad_s=self.max_yaw_rate_rad_s,
            init_vel_std_ms=self.init_vel_std_ms,
            init_psi_std_rad=self.init_psi_std_rad,
            init_yaw_rate_std_rad_s=self.init_yaw_rate_std_rad_s,
        )
        copy_obj._x = self._x.copy()
        copy_obj._P = self._P.copy()
        copy_obj._last_timestamp_s = self._last_timestamp_s
        copy_obj._update_count = self._update_count
        copy_obj._consecutive_rejects = self._consecutive_rejects
        copy_obj._rejected_obs = list(self._rejected_obs)
        copy_obj._last_rejected_obs = self._last_rejected_obs
        copy_obj._reset_count = self._reset_count
        copy_obj._was_reset = self._was_reset
        copy_obj.last_d2 = self.last_d2
        copy_obj.last_accepted = self.last_accepted
        return copy_obj
