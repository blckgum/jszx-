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
        self._last_rejected_obs: Optional[Tuple[float, float, float]] = None
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
        if obs_cov_m2 is None:
            return self._default_R.copy()
        # 在 np.asarray(..., dtype=np.float64) 前递归拦截 object 数组及嵌套序列/ndarray 中的布尔值
        if _contains_bool_value(obs_cov_m2):
            return self._default_R.copy()
        try:
            arr = np.asarray(obs_cov_m2, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return self._default_R.copy()

        if arr.shape != (2, 2) or not np.isfinite(arr).all():
            return self._default_R.copy()

        # 拦截极小或超大尺度协方差，避免行列式上溢为 inf-inf=nan 绕过正定检查
        max_abs = float(np.max(np.abs(arr)))
        if not math.isfinite(max_abs) or max_abs <= 1e-12 or max_abs > 1e12:
            return self._default_R.copy()

        with np.errstate(over="ignore", invalid="ignore"):
            scale_ref = max(1.0, abs(float(arr[0, 1])), abs(float(arr[1, 0])))
            diff_off = abs(float(arr[0, 1] - arr[1, 0]))
            if not math.isfinite(diff_off) or diff_off > 1e-6 * scale_ref:
                return self._default_R.copy()
            r_mat = 0.5 * (arr + arr.T)
            if not np.isfinite(r_mat).all():
                return self._default_R.copy()
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
            return self._default_R.copy()
        return r_mat

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
    ) -> bool:
        """将滤波器安全重置为以当前有限观测为中心的初始状态与协方差。"""
        self._x[:] = (zx, zy, 0.0, 0.0, 0.0, 0.0)
        self._P = self._build_reset_cov(r_mat)
        self._last_timestamp_s = ts
        self._update_count = 1
        self._consecutive_rejects = 0
        self._last_rejected_obs = None
        self.last_d2 = 0.0
        self.last_accepted = True
        return True

    @staticmethod
    def _clamp_vec2(vx: float, vy: float, max_norm: float) -> Optional[Tuple[float, float]]:
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

        if self._update_count == 0 or self._last_timestamp_s is None:
            return self._reset_to_observation(ts, zx, zy, r_mat)

        dt_s = ts - self._last_timestamp_s
        with np.errstate(over="ignore", invalid="ignore"):
            f_mat = _transition_matrix(dt_s)
            q_mat = _process_noise_matrix(dt_s, self.jerk_std_ms3)

            x_pred = f_mat @ self._x
            p_pred = f_mat @ self._P @ f_mat.T + q_mat
            p_pred = 0.5 * (p_pred + p_pred.T)

        # 极端 dt（如 1e80）导致预测状态或协方差溢出时，先验失效，安全重置到当前有限观测
        if not (np.isfinite(x_pred).all() and np.isfinite(p_pred).all()):
            return self._reset_to_observation(ts, zx, zy, r_mat)

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
        self.last_d2 = d2

        # 马氏距离新息门控：连续拒收达到上限时自动重置，避免真实剧烈机动锁死滤波器
        if self._update_count >= self.min_updates_for_gate and (
            not math.isfinite(d2) or d2 > self.gate_chi2
        ):
            self._consecutive_rejects += 1
            if self._consecutive_rejects < self.max_consecutive_rejects:
                x_cand = self._clamp_physical_kinematics(x_pred.copy())
                if x_cand is None or not (np.isfinite(x_cand).all() and np.isfinite(p_pred).all()):
                    return self._reset_to_observation(ts, zx, zy, r_mat)
                self._last_rejected_obs = (ts, zx, zy)
                self._x = x_cand
                self._P = p_pred
                self._last_timestamp_s = ts
                self.last_accepted = False
                return False

            # 连续剧烈偏移超过容忍帧数：重置位置并利用最近两次被拒观测初始化速度
            vx0, vy0 = 0.0, 0.0
            has_vel_init = False
            if self._last_rejected_obs is not None:
                prev_ts, prev_x, prev_y = self._last_rejected_obs
                dt_rej = ts - prev_ts
                if dt_rej > 1e-4:
                    with np.errstate(over="ignore", invalid="ignore"):
                        vx0 = (zx - prev_x) / dt_rej
                        vy0 = (zy - prev_y) / dt_rej
                    if not (math.isfinite(vx0) and math.isfinite(vy0)):
                        # 极值坐标差分上溢时先减半求差并按限速缩放方向，防止 inf*0=nan
                        dx_half = 0.5 * zx - 0.5 * prev_x
                        dy_half = 0.5 * zy - 0.5 * prev_y
                        clamped_v = self._clamp_vec2(dx_half, dy_half, self.max_speed_ms)
                        vx0, vy0 = (0.0, 0.0) if clamped_v is None else clamped_v
                    has_vel_init = True
            x_cand = np.array([zx, zy, vx0, vy0, 0.0, 0.0], dtype=np.float64)
            x_cand = self._clamp_physical_kinematics(x_cand)
            p_cand = self._build_reset_cov(r_mat)
            if x_cand is None or not (np.isfinite(x_cand).all() and np.isfinite(p_cand).all()):
                return self._reset_to_observation(ts, zx, zy, r_mat)
            self._x = x_cand
            self._P = p_cand
            self._last_timestamp_s = ts
            self._consecutive_rejects = 0
            self._last_rejected_obs = None
            self._update_count = 2 if has_vel_init else 1
            self.last_accepted = True
            return True

        self._consecutive_rejects = 0
        self._last_rejected_obs = None

        if not (np.isfinite(y_res).all() and np.isfinite(s_mat).all()):
            # 极值坐标导致残差或新息矩阵溢出时安全保留有限预测状态，不污染滤波器
            x_cand = self._clamp_physical_kinematics(x_pred.copy())
            if x_cand is None or not np.isfinite(s_mat).all():
                return self._reset_to_observation(ts, zx, zy, r_mat)
            self._x = x_cand
            self._P = p_pred
            self._last_timestamp_s = ts
            self.last_accepted = False
            return False

        # Joseph 形式协方差更新: P = (I - K H) P (I - K H)^T + K R K^T
        with np.errstate(over="ignore", invalid="ignore"):
            ph_t = p_pred @ _H_OBS.T
            try:
                k_gain = np.linalg.solve(s_mat.T, ph_t.T).T
            except np.linalg.LinAlgError:
                k_gain = ph_t @ np.linalg.pinv(s_mat)

            i_kh = _I6 - k_gain @ _H_OBS
            p_upd = i_kh @ p_pred @ i_kh.T + k_gain @ r_mat @ k_gain.T
            p_upd = 0.5 * (p_upd + p_upd.T)
            x_upd = x_pred + k_gain @ y_res

        x_cand = self._clamp_physical_kinematics(x_upd)
        # 提交前校验有限性：极端观测导致状态上溢则退回有限预测，协方差上溢则安全重置
        if not np.isfinite(p_upd).all():
            return self._reset_to_observation(ts, zx, zy, r_mat)
        if x_cand is None or not np.isfinite(x_cand).all():
            x_fallback = self._clamp_physical_kinematics(x_pred.copy())
            if x_fallback is None or not np.isfinite(x_fallback).all():
                return self._reset_to_observation(ts, zx, zy, r_mat)
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
        copy_obj._last_rejected_obs = self._last_rejected_obs
        copy_obj.last_d2 = self.last_d2
        copy_obj.last_accepted = self.last_accepted
        return copy_obj
