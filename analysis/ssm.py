# -*- coding: utf-8 -*-
"""
旋转矩形碰撞检测与替代安全指标 (OBB Swept TTC & Surrogate Safety Measures)
=============================================================================
提供纯 Python 标准库 + NumPy 实现的高精度时空冲突替代安全指标（不依赖 SciPy）：
1. 基于旋转矩形 (OBB) 扫掠与 Slab 射线求交的精确进入/退出时间及最小净距；
2. 考虑相对加速度的修正碰撞时间 (MTTC)；
3. 避险所需减速度 (DRAC)；
4. 基于相对位置协方差与高斯-勒让德积分的确定性碰撞概率估计；
5. 基于时空网格占用的侵入后时间 (PET) 跟踪器。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

# 预计算 48 节点高斯-勒让德积分节点与权重，用于碰撞概率的一维无奇异积分。
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(48)
_GL_PAIRS: Tuple[Tuple[float, float], ...] = tuple(
    (float(x), float(w)) for x, w in zip(_GL_NODES, _GL_WEIGHTS)
)
_DEFAULT_POS_VAR_M2 = 0.25 * 0.25


def _is_finite_number(value) -> bool:
    """判断是否为有限实数，显式排除 Python/NumPy 布尔值与 NaN/Inf。"""
    if type(value) is bool or isinstance(value, np.bool_):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return False


def _validate_positive_float(value, name: str) -> float:
    if not _is_finite_number(value) or float(value) <= 0.0:
        raise ValueError(f"{name} 必须为有限正数")
    return float(value)


def _validate_non_negative_float(value, name: str) -> float:
    if not _is_finite_number(value) or float(value) < 0.0:
        raise ValueError(f"{name} 必须为有限非负数")
    return float(value)


def _validate_vec2(vec: Sequence[float], name: str) -> Tuple[float, float]:
    if isinstance(vec, (str, bytes)) or not isinstance(vec, (Sequence, np.ndarray)):
        raise ValueError(f"{name} 必须是长度为 2 的有限二维向量")
    if len(vec) != 2:
        raise ValueError(f"{name} 长度必须为 2")
    x, y = vec[0], vec[1]
    if not (_is_finite_number(x) and _is_finite_number(y)):
        raise ValueError(f"{name} 分量必须为有限数值")
    return (float(x), float(y))


def _parse_cov_2x2(cov) -> np.ndarray:
    """将 None、4 元组或 2x2 矩阵解析并校验为对称半正定 2x2 协方差矩阵。"""
    if cov is None:
        return np.eye(2, dtype=float) * _DEFAULT_POS_VAR_M2

    if isinstance(cov, (tuple, list)) and len(cov) == 4 and not isinstance(cov[0], (tuple, list, np.ndarray)):
        if not all(_is_finite_number(v) for v in cov):
            raise ValueError("pos_cov_2x2 四元组必须全部为有限数值")
        mat = np.array([[float(cov[0]), float(cov[1])], [float(cov[2]), float(cov[3])]], dtype=float)
    else:
        if isinstance(cov, np.ndarray):
            if cov.dtype == np.bool_ or cov.shape != (2, 2):
                raise ValueError("pos_cov_2x2 必须为 2x2 数值矩阵")
            flat = cov.ravel()
            if not all(_is_finite_number(v) for v in flat):
                raise ValueError("pos_cov_2x2 包含非有限值")
            mat = cov.astype(float, copy=True)
        elif isinstance(cov, (list, tuple)) and len(cov) == 2:
            rows = []
            for row in cov:
                if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 2:
                    raise ValueError("pos_cov_2x2 必须为 2x2 数值矩阵")
                if not (_is_finite_number(row[0]) and _is_finite_number(row[1])):
                    raise ValueError("pos_cov_2x2 包含非有限或布尔值")
                rows.append([float(row[0]), float(row[1])])
            mat = np.array(rows, dtype=float)
        else:
            raise ValueError("pos_cov_2x2 格式无效，需为 None、4 元组或 2x2 矩阵")

    sym = 0.5 * (mat + mat.T)
    eigvals = np.linalg.eigvalsh(sym)
    if float(eigvals[0]) < -1e-6:
        raise ValueError("pos_cov_2x2 必须为半正定协方差矩阵")
    return sym


@dataclass(frozen=True)
class OBBCollisionResult:
    """旋转矩形（OBB）扫掠碰撞检测与净距分析结果。"""
    collides: bool
    t_enter: Optional[float]
    t_exit: Optional[float]
    min_clearance_m: float
    t_min_clearance: float
    local_pos_xy: Tuple[float, float]
    local_vel_xy: Tuple[float, float]
    expanded_half_extents_xy: Tuple[float, float]


def _ray_aabb_slab_interval(
    px: float,
    py: float,
    vx: float,
    vy: float,
    hx: float,
    hy: float,
    horizon_s: float,
) -> Optional[Tuple[float, float]]:
    """计算相对射线 p + v*t 与轴对齐矩形 [-hx, hx] x [-hy, hy] 在 [0, horizon_s] 内的交集时间段。"""
    t_min = 0.0
    t_max = horizon_s
    for p_i, v_i, h_i in ((px, vx, hx), (py, vy, hy)):
        if abs(v_i) <= 1e-12:
            if abs(p_i) > h_i:
                return None
            continue
        inv_v = 1.0 / v_i
        t1 = (-h_i - p_i) * inv_v
        t2 = (h_i - p_i) * inv_v
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
        if t2 < t_max:
            t_max = t2
        if t_min > t_max:
            return None
    return (t_min, t_max)


def _segment_aabb_min_clearance(
    px: float,
    py: float,
    vx: float,
    vy: float,
    hx: float,
    hy: float,
    horizon_s: float,
) -> Tuple[float, float]:
    """解析求解线段 p + v*t (t in [0, horizon_s]) 到矩形 [-hx, hx] x [-hy, hy] 的精确最小欧氏距离与时刻。"""
    hit = _ray_aabb_slab_interval(px, py, vx, vy, hx, hy, horizon_s)
    if hit is not None:
        return (0.0, hit[0])

    # 距离平方函数分段二次且处处凸，分段边界/断点仅出现在穿过边界 x=+-hx 或 y=+-hy 处。
    raw_knots = [0.0, horizon_s]
    for p_i, v_i, h_i in ((px, vx, hx), (py, vy, hy)):
        if abs(v_i) > 1e-12:
            for boundary in (-h_i, h_i):
                t_b = (boundary - p_i) / v_i
                if 0.0 < t_b < horizon_s:
                    raw_knots.append(t_b)
    raw_knots.sort()
    knots: list[float] = []
    for t_k in raw_knots:
        if not knots or t_k - knots[-1] > 1e-12:
            knots.append(t_k)

    def _dist_sq(t: float) -> float:
        dx_box = max(0.0, abs(px + vx * t) - hx)
        dy_box = max(0.0, abs(py + vy * t) - hy)
        return dx_box * dx_box + dy_box * dy_box

    best_t = 0.0
    best_d2 = _dist_sq(0.0)

    def _consider(t: float) -> None:
        nonlocal best_t, best_d2
        d2 = _dist_sq(t)
        if d2 < best_d2 - 1e-12 or (abs(d2 - best_d2) <= 1e-12 and t < best_t):
            best_d2 = d2
            best_t = t

    for t_k in knots:
        _consider(t_k)

    for idx in range(len(knots) - 1):
        t_a = knots[idx]
        t_b = knots[idx + 1]
        t_mid = 0.5 * (t_a + t_b)
        xm = px + vx * t_mid
        ym = py + vy * t_mid

        if xm > hx:
            ax, bx = vx, px - hx
        elif xm < -hx:
            ax, bx = vx, px + hx
        else:
            ax, bx = 0.0, 0.0

        if ym > hy:
            ay, by = vy, py - hy
        elif ym < -hy:
            ay, by = vy, py + hy
        else:
            ay, by = 0.0, 0.0

        denom = ax * ax + ay * ay
        if denom > 1e-18:
            t_star = -(ax * bx + ay * by) / denom
            if t_a < t_star < t_b:
                _consider(t_star)

    return (math.sqrt(max(0.0, best_d2)), best_t)


def compute_obb_swept_ttc(
    veh_xy: Sequence[float],
    veh_v: Sequence[float],
    heading_rad: float,
    veh_length_m: float,
    veh_width_m: float,
    ped_xy: Sequence[float],
    ped_v: Sequence[float],
    ped_radius_m: float,
    horizon_s: float,
    lateral_margin_m: float = 0.3,
    longitudinal_margin_m: float = 0.4,
) -> OBBCollisionResult:
    """在车体局部坐标系中做精确射线-AABB 扫掠碰撞检测，并计算无安全裕度包络下的最小净距。"""
    vx_pos, vy_pos = _validate_vec2(veh_xy, "veh_xy")
    vv_x, vv_y = _validate_vec2(veh_v, "veh_v")
    px_pos, py_pos = _validate_vec2(ped_xy, "ped_xy")
    pv_x, pv_y = _validate_vec2(ped_v, "ped_v")
    if not _is_finite_number(heading_rad):
        raise ValueError("heading_rad 必须为有限数值")
    theta = float(heading_rad)
    length = _validate_positive_float(veh_length_m, "veh_length_m")
    width = _validate_positive_float(veh_width_m, "veh_width_m")
    ped_r = _validate_non_negative_float(ped_radius_m, "ped_radius_m")
    horizon = _validate_positive_float(horizon_s, "horizon_s")
    lat_margin = _validate_non_negative_float(lateral_margin_m, "lateral_margin_m")
    lon_margin = _validate_non_negative_float(longitudinal_margin_m, "longitudinal_margin_m")

    # 转换到车体局部坐标系：x 沿车头航向正方向，y 为左法向。
    rx = px_pos - vx_pos
    ry = py_pos - vy_pos
    dvx = pv_x - vv_x
    dvy = pv_y - vv_y

    cos_h = math.cos(theta)
    sin_h = math.sin(theta)
    local_px = cos_h * rx + sin_h * ry
    local_py = -sin_h * rx + cos_h * ry
    local_vx = cos_h * dvx + sin_h * dvy
    local_vy = -sin_h * dvx + cos_h * dvy

    hx_unmargined = 0.5 * length + ped_r
    hy_unmargined = 0.5 * width + ped_r
    hx = hx_unmargined + lon_margin
    hy = hy_unmargined + lat_margin

    slab = _ray_aabb_slab_interval(local_px, local_py, local_vx, local_vy, hx, hy, horizon)
    min_clearance, t_min_clearance = _segment_aabb_min_clearance(
        local_px, local_py, local_vx, local_vy, hx_unmargined, hy_unmargined, horizon
    )

    if slab is not None:
        return OBBCollisionResult(
            collides=True,
            t_enter=slab[0],
            t_exit=slab[1],
            min_clearance_m=min_clearance,
            t_min_clearance=t_min_clearance,
            local_pos_xy=(local_px, local_py),
            local_vel_xy=(local_vx, local_vy),
            expanded_half_extents_xy=(hx, hy),
        )
    return OBBCollisionResult(
        collides=False,
        t_enter=None,
        t_exit=None,
        min_clearance_m=min_clearance,
        t_min_clearance=t_min_clearance,
        local_pos_xy=(local_px, local_py),
        local_vel_xy=(local_vx, local_vy),
        expanded_half_extents_xy=(hx, hy),
    )


def _solve_quadratic_nonneg(
    c2: float, c1: float, c0: float, t_lo: float, t_hi: float
) -> list[Tuple[float, float]]:
    """求解区间 [t_lo, t_hi] 内满足 c2*t^2 + c1*t + c0 >= 0 的闭合时间区间集合。"""
    if t_lo > t_hi:
        return []
    if abs(c2) <= 1e-12:
        if abs(c1) <= 1e-12:
            # 退化为常数项，允许极微小的浮点舍入误差落在边界上。
            return [(t_lo, t_hi)] if c0 >= -1e-12 else []
        t_root = (-c0 / c1) + 0.0
        if c1 > 0.0:
            lo = max(t_lo, t_root)
            return [(lo, t_hi)] if lo <= t_hi else []
        hi = min(t_hi, t_root)
        return [(t_lo, hi)] if t_lo <= hi else []

    disc = c1 * c1 - 4.0 * c2 * c0
    tol = 1e-12 * max(1.0, c1 * c1, abs(4.0 * c2 * c0))
    if -tol <= disc < 0.0:
        disc = 0.0
    if disc < 0.0:
        # 无实根时抛物线不变号：开口向上恒满足，开口向下恒不满足。
        return [(t_lo, t_hi)] if c2 > 0.0 else []

    if disc == 0.0:
        r1 = r2 = (-c1 / (2.0 * c2)) + 0.0
    else:
        sqrt_disc = math.sqrt(disc)
        # 采用数值稳定求根形式，防止小加速度系数下发生有效数字相消。
        q = -0.5 * (c1 + math.copysign(sqrt_disc, c1))
        if abs(q) > 1e-18:
            r_a = (q / c2) + 0.0
            r_b = (c0 / q) + 0.0
        else:
            r_a = ((-c1 - sqrt_disc) / (2.0 * c2)) + 0.0
            r_b = ((-c1 + sqrt_disc) / (2.0 * c2)) + 0.0
        r1, r2 = (r_a, r_b) if r_a <= r_b else (r_b, r_a)

    if c2 > 0.0:
        # 开口向上对应两段外侧区间 (-inf, r1] U [r2, +inf)，相接时合并为单区间。
        intervals: list[Tuple[float, float]] = []
        if t_lo <= r1:
            intervals.append((t_lo, min(t_hi, r1)))
        if r2 <= t_hi:
            right_lo = max(t_lo, r2)
            if intervals and right_lo <= intervals[-1][1] + 1e-12:
                intervals[-1] = (intervals[-1][0], t_hi)
            else:
                intervals.append((right_lo, t_hi))
        return intervals

    # 开口向下对应单段闭区间 [r1, r2]。
    lo = max(t_lo, r1)
    hi = min(t_hi, r2)
    return [(lo, hi)] if lo <= hi else []


def _intersect_interval_lists(
    a_list: Sequence[Tuple[float, float]],
    b_list: Sequence[Tuple[float, float]],
) -> list[Tuple[float, float]]:
    """计算两组有序不相交闭区间列表的交集，返回有序闭区间列表。"""
    result: list[Tuple[float, float]] = []
    i = 0
    j = 0
    while i < len(a_list) and j < len(b_list):
        a0, a1 = a_list[i]
        b0, b1 = b_list[j]
        lo = max(a0, b0)
        hi = min(a1, b1)
        if lo <= hi:
            result.append((lo, hi))
        if a1 < b1:
            i += 1
        else:
            j += 1
    return result


def compute_obb_accel_entry_time(
    local_pos_xy: Sequence[float],
    local_vel_xy: Sequence[float],
    local_acc_xy: Sequence[float],
    half_extents_xy: Sequence[float],
    horizon_s: float,
) -> Optional[float]:
    """在 [0, horizon_s] 内闭合解析求解二阶相对轨迹进入 [-hx, hx] x [-hy, hy] 的最早时刻。"""
    px, py = _validate_vec2(local_pos_xy, "local_pos_xy")
    vx, vy = _validate_vec2(local_vel_xy, "local_vel_xy")
    ax, ay = _validate_vec2(local_acc_xy, "local_acc_xy")
    hx_raw, hy_raw = _validate_vec2(half_extents_xy, "half_extents_xy")
    hx = _validate_non_negative_float(hx_raw, "half_extents_xy[0]")
    hy = _validate_non_negative_float(hy_raw, "half_extents_xy[1]")
    horizon = _validate_positive_float(horizon_s, "horizon_s")

    # 分别求解 X/Y 轴二次不等式 |p + v*t + 0.5*a*t^2| <= h 在 [0, horizon] 内的区间并集（各至多两段）。
    x_lower = _solve_quadratic_nonneg(0.5 * ax, vx, px + hx, 0.0, horizon)
    x_upper = _solve_quadratic_nonneg(-0.5 * ax, -vx, hx - px, 0.0, horizon)
    x_intervals = _intersect_interval_lists(x_lower, x_upper)
    if not x_intervals:
        return None

    y_lower = _solve_quadratic_nonneg(0.5 * ay, vy, py + hy, 0.0, horizon)
    y_upper = _solve_quadratic_nonneg(-0.5 * ay, -vy, hy - py, 0.0, horizon)
    y_intervals = _intersect_interval_lists(y_lower, y_upper)
    if not y_intervals:
        return None

    # 求 X/Y 区间集合交集，返回最早侵入时刻。
    xy_intervals = _intersect_interval_lists(x_intervals, y_intervals)
    if not xy_intervals:
        return None
    return xy_intervals[0][0] + 0.0


def compute_mttc(
    dx: float,
    dy: float,
    dvx: float,
    dvy: float,
    dax: float,
    day: float,
    critical_dist_m: float,
    horizon_s: float,
) -> Optional[float]:
    """沿视线方向求解二阶修正碰撞时间 MTTC；若正在制动且刹停距离小于净距则返回 None。"""
    crit = _validate_non_negative_float(critical_dist_m, "critical_dist_m")
    horizon = _validate_positive_float(horizon_s, "horizon_s")
    if not all(_is_finite_number(v) for v in (dx, dy, dvx, dvy, dax, day)):
        return None

    dx_f, dy_f = float(dx), float(dy)
    dvx_f, dvy_f = float(dvx), float(dvy)
    dax_f, day_f = float(dax), float(day)

    dist = math.hypot(dx_f, dy_f)
    if dist <= 1e-9:
        if math.hypot(dvx_f, dvy_f) > 1e-6 or math.hypot(dax_f, day_f) > 1e-6:
            return 0.0
        return None

    ux = dx_f / dist
    uy = dy_f / dist
    d_eff = max(0.0, dist - crit)
    # 视线趋近速度 v_c > 0 表示双方正在靠近；a_c < 0 表示相对减速度（制动）。
    v_c = -(dvx_f * ux + dvy_f * uy)
    a_c = -(dax_f * ux + day_f * uy)

    if d_eff <= 0.0:
        if v_c > 1e-6 or (v_c >= 0.0 and a_c > 1e-6):
            return 0.0
        return None

    # 若当前正在接近并持续制动，且刹停所需距离小于剩余净距，则不会发生碰撞。
    if v_c > 0.0 and a_c < 0.0:
        stopping_dist = -(v_c * v_c) / (2.0 * a_c)
        if stopping_dist < d_eff:
            return None

    if abs(a_c) <= 1e-9:
        if v_c <= 1e-9:
            return None
        t_lin = d_eff / v_c
        return t_lin if 0.0 < t_lin <= horizon else None

    disc = v_c * v_c + 2.0 * a_c * d_eff
    if disc < 0.0:
        # 仅在极微小的浮点误差下视作切线相碰，否则无实根。
        if disc > -1e-12 and v_c > 0.0 and a_c < 0.0:
            disc = 0.0
        else:
            return None

    sqrt_disc = math.sqrt(disc)
    # 使用数值稳定的求根形式，避免 |a_c| 较小时产生有效数字抵消。
    half_a = 0.5 * a_c
    neg_c = -d_eff
    if v_c >= 0.0:
        q = -0.5 * (v_c + sqrt_disc)
    else:
        q = -0.5 * (v_c - sqrt_disc)

    roots = []
    if abs(q) > 1e-15:
        roots.append(neg_c / q)
        roots.append(q / half_a)
    else:
        roots.append((-v_c - sqrt_disc) / a_c)
        roots.append((-v_c + sqrt_disc) / a_c)

    valid = [t for t in roots if math.isfinite(t) and 0.0 < t <= horizon]
    return min(valid) if valid else None


def compute_drac(
    dx: float,
    dy: float,
    dvx: float,
    dvy: float,
    critical_dist_m: float,
) -> Optional[float]:
    """计算避险所需减速度 DRAC (m/s^2)；无有效趋近速度时返回 None。"""
    crit = _validate_non_negative_float(critical_dist_m, "critical_dist_m")
    if not all(_is_finite_number(v) for v in (dx, dy, dvx, dvy)):
        return None

    dx_f, dy_f = float(dx), float(dy)
    dvx_f, dvy_f = float(dvx), float(dvy)
    dist = math.hypot(dx_f, dy_f)
    if dist <= 1e-9:
        return None

    ux = dx_f / dist
    uy = dy_f / dist
    v_c = -(dvx_f * ux + dvy_f * uy)
    if v_c <= 1e-3:
        return None

    d_eff = max(0.0, dist - crit)
    return (v_c * v_c) / (2.0 * max(d_eff, 0.1))


def estimate_collision_probability(
    dx: float,
    dy: float,
    dvx: float,
    dvy: float,
    t_eval: Optional[float],
    safe_radius_m: float,
    pos_cov_veh_2x2=None,
    pos_cov_ped_2x2=None,
    dax: float = 0.0,
    day: float = 0.0,
) -> Optional[float]:
    """基于相对位置协方差与特征正交分解+高斯-勒让德积分，计算 [0, 1] 确定性碰撞概率。"""
    safe_r = _validate_positive_float(safe_radius_m, "safe_radius_m")
    if t_eval is None:
        return None
    if not all(_is_finite_number(v) for v in (dx, dy, dvx, dvy, dax, day, t_eval)):
        return None
    t_f = float(t_eval)
    if t_f < 0.0:
        raise ValueError("t_eval 不能为负数")

    sigma_veh = _parse_cov_2x2(pos_cov_veh_2x2)
    sigma_ped = _parse_cov_2x2(pos_cov_ped_2x2)
    motion_var = (0.15 * t_f) ** 2
    sigma_rel = sigma_veh + sigma_ped + motion_var * np.eye(2, dtype=float)

    # 纳入相对加速度的二阶外推均值位置，避免急加速/制动场景沿用一阶等速位置导致概率失真。
    half_t2 = 0.5 * (t_f ** 2)
    mu = np.array(
        [
            float(dx) + float(dvx) * t_f + float(dax) * half_t2,
            float(dy) + float(dvy) * t_f + float(day) * half_t2,
        ],
        dtype=float,
    )
    eigvals, eigvecs = np.linalg.eigh(sigma_rel)
    if float(eigvals[1]) <= 1e-12:
        return 1.0 if float(np.hypot(mu[0], mu[1])) <= safe_r else 0.0

    eigvals = np.maximum(eigvals, 1e-8)
    s1 = math.sqrt(float(eigvals[0]))
    s2 = math.sqrt(float(eigvals[1]))
    mu_rot = eigvecs.T @ mu
    m1 = float(mu_rot[0])
    m2 = float(mu_rot[1])

    # 将主要质量区间 [-safe_r, safe_r] ∩ [m1 - 8.5*s1, m1 + 8.5*s1] 映射到角度变量 theta，
    # 消除积分边界 sqrt(safe_r^2 - z1^2) 处的导数奇异性。
    lo = max(-safe_r, m1 - 8.5 * s1)
    hi = min(safe_r, m1 + 8.5 * s1)
    if lo >= hi:
        return 0.0

    th_lo = math.asin(max(-1.0, min(1.0, lo / safe_r)))
    th_hi = math.asin(max(-1.0, min(1.0, hi / safe_r)))
    half_span = 0.5 * (th_hi - th_lo)
    mid = 0.5 * (th_hi + th_lo)
    if half_span <= 0.0:
        return 0.0

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    total = 0.0
    for xk, wk in _GL_PAIRS:
        th = mid + half_span * xk
        z1 = safe_r * math.sin(th)
        rad = safe_r * math.cos(th)
        u1 = (z1 - m1) / s1
        pdf1 = math.exp(-0.5 * u1 * u1) * inv_sqrt2pi / s1
        hi2 = (rad - m2) / s2
        lo2 = (-rad - m2) / s2
        cdf_diff = 0.5 * (math.erf(hi2 * inv_sqrt2) - math.erf(lo2 * inv_sqrt2))
        total += wk * pdf1 * cdf_diff * rad

    prob = total * half_span
    return max(0.0, min(1.0, float(prob)))


@dataclass
class _CellVisit:
    """单个目标在某网格单元内一次连续驻留的起止时间戳。"""
    category: str
    first_t: float
    last_t: float


class PETTracker:
    """基于空间网格占用记录的侵入后时间 (Post-Encroachment Time, PET) 跟踪器。"""

    def __init__(
        self,
        cell_size_m: float = 1.0,
        max_pet_s: float = 4.0,
        retention_s: float = 8.0,
    ):
        self.cell_size_m = _validate_positive_float(cell_size_m, "cell_size_m")
        self.max_pet_s = _validate_positive_float(max_pet_s, "max_pet_s")
        self.retention_s = _validate_positive_float(retention_s, "retention_s")
        # cell -> {track_id: _CellVisit}
        self._cell_visits: Dict[Tuple[int, int], Dict[int, _CellVisit]] = {}
        # track_id -> 上一帧占用过的网格集合，用于区分连续滞留与再次进入
        self._active_cells_by_track: Dict[int, set[Tuple[int, int]]] = {}
        # track_id -> (timestamp_s, x_m, y_m)，用于帧间轨迹插值防止高速目标越格漏检
        self._last_track_pos: Dict[int, Tuple[float, float, float]] = {}
        self._last_timestamp_s: Optional[float] = None

    def reset(self) -> None:
        """清空所有网格占用与轨迹缓存状态，保留配置参数以便实例复用。"""
        self._cell_visits.clear()
        self._active_cells_by_track.clear()
        self._last_track_pos.clear()
        self._last_timestamp_s = None

    def _grid_cell(self, x_m: float, y_m: float) -> Tuple[int, int]:
        return (math.floor(x_m / self.cell_size_m), math.floor(y_m / self.cell_size_m))

    def _sample_cells(
        self, track_id: int, x_m: float, y_m: float, timestamp_s: float
    ) -> list[Tuple[Tuple[int, int], float, float]]:
        prev = self._last_track_pos.get(track_id)
        cells_with_ts: Dict[Tuple[int, int], Tuple[float, float]] = {}
        if prev is not None:
            prev_t, prev_x, prev_y = prev
            dt = timestamp_s - prev_t
            dist = math.hypot(x_m - prev_x, y_m - prev_y)
            if 0.0 < dt <= 1.0 and 0.0 < dist <= 15.0:
                steps = max(1, int(math.ceil(dist / (0.5 * self.cell_size_m))))
                for idx in range(1, steps + 1):
                    alpha = idx / steps
                    sx = prev_x + (x_m - prev_x) * alpha
                    sy = prev_y + (y_m - prev_y) * alpha
                    st = prev_t + dt * alpha
                    cell = self._grid_cell(sx, sy)
                    if cell not in cells_with_ts:
                        cells_with_ts[cell] = (st, st)
                    else:
                        first_st, _ = cells_with_ts[cell]
                        cells_with_ts[cell] = (first_st, st)
        end_cell = self._grid_cell(x_m, y_m)
        if end_cell not in cells_with_ts:
            cells_with_ts[end_cell] = (timestamp_s, timestamp_s)
        else:
            first_st, _ = cells_with_ts[end_cell]
            cells_with_ts[end_cell] = (first_st, timestamp_s)
        return [(cell, span[0], span[1]) for cell, span in cells_with_ts.items()]

    def update(
        self, readings: Iterable, timestamp_s: float
    ) -> Dict[Tuple[int, int], float]:
        """更新当前帧所有有效目标的网格占用，返回本帧跨类别交叉侵入 {(veh_id, ped_id): min_pet_s}。"""
        ts = _validate_non_negative_float(timestamp_s, "timestamp_s")
        if self._last_timestamp_s is not None and ts < self._last_timestamp_s:
            raise ValueError("PETTracker timestamp_s 不能倒退")
        self._last_timestamp_s = ts

        # 清理超出保留期 retention_s 的历史占用与过期轨迹坐标，控制内存有界。
        cutoff = ts - self.retention_s
        for cell, visits in list(self._cell_visits.items()):
            for tid, visit in list(visits.items()):
                if visit.last_t < cutoff:
                    del visits[tid]
            if not visits:
                del self._cell_visits[cell]
        for tid in list(self._last_track_pos.keys()):
            if self._last_track_pos[tid][0] < cutoff:
                del self._last_track_pos[tid]
                self._active_cells_by_track.pop(tid, None)

        near_misses: Dict[Tuple[int, int], float] = {}
        current_cells_by_track: Dict[int, set[Tuple[int, int]]] = {}
        # 第一阶段：遍历本帧全部有效目标，更新网格占用并记录本帧步进触及的时间区间（与输入顺序无关）。
        frame_intervals: Dict[
            Tuple[int, int], list[Tuple[int, str, float, float, float]]
        ] = {}

        for r in readings:
            cat = getattr(r, "category", None)
            if cat == "vehicle":
                if getattr(r, "is_stationary", False) and getattr(r, "is_reliable", False):
                    continue
                norm_cat = "vehicle"
            elif cat in ("pedestrian", "person"):
                norm_cat = "pedestrian"
            else:
                continue

            tid = getattr(r, "track_id", None)
            x_m = getattr(r, "x_m", None)
            y_m = getattr(r, "y_m", None)
            if type(tid) is not int or tid < 1:
                continue
            if not (_is_finite_number(x_m) and _is_finite_number(y_m)):
                continue
            x_f, y_f = float(x_m), float(y_m)

            prev_active = self._active_cells_by_track.get(tid, set())
            touched_cells = self._sample_cells(tid, x_f, y_f, ts)
            curr_active: set[Tuple[int, int]] = set()

            for cell, first_cell_ts, last_cell_ts in touched_cells:
                curr_active.add(cell)
                visits = self._cell_visits.setdefault(cell, {})
                existing = visits.get(tid)
                if existing is not None and cell in prev_active and existing.category == norm_cat:
                    existing.last_t = last_cell_ts
                    entry_t = existing.first_t
                else:
                    visits[tid] = _CellVisit(
                        category=norm_cat, first_t=first_cell_ts, last_t=last_cell_ts
                    )
                    entry_t = first_cell_ts
                frame_intervals.setdefault(cell, []).append(
                    (tid, norm_cat, first_cell_ts, last_cell_ts, entry_t)
                )

            current_cells_by_track[tid] = curr_active
            self._last_track_pos[tid] = (ts, x_f, y_f)

        # 第二阶段：与输入顺序无关地比对同帧区间与历史访客占用。
        for cell, entries in frame_intervals.items():
            visits = self._cell_visits[cell]
            frame_tids = {item[0] for item in entries}

            # (a) 本帧同时触及同一网格的异类目标：以本帧连续占用区间 [entry, last] 对称求间隔，
            # 既兼容跨帧驻留未离格目标与过路目标的 0.0s 重叠，也保证 [ped, car] 与 [car, ped] 结果完全一致。
            n_entries = len(entries)
            for idx_a in range(n_entries):
                tid_a, cat_a, _first_a, last_a, entry_a = entries[idx_a]
                for idx_b in range(idx_a + 1, n_entries):
                    tid_b, cat_b, _first_b, last_b, entry_b = entries[idx_b]
                    if cat_a == cat_b or tid_a == tid_b:
                        continue
                    pet = max(0.0, max(entry_a, entry_b) - min(last_a, last_b))
                    if 0.0 <= pet <= self.max_pet_s:
                        pair = (tid_a, tid_b) if cat_a == "vehicle" else (tid_b, tid_a)
                        prev_min = near_misses.get(pair)
                        if prev_min is None or pet < prev_min:
                            near_misses[pair] = pet

            # (b) 历史访客（本帧未触及该网格）：仅计算先到离格后另一方稍晚进入的 PET，
            # 并跳过 other_visit.last_t > entry_t 的过期重叠条目。
            for tid, norm_cat, _first_cell_ts, _last_cell_ts, entry_t in entries:
                for other_tid, other_visit in visits.items():
                    if other_tid in frame_tids or other_visit.category == norm_cat:
                        continue
                    if other_visit.last_t <= entry_t:
                        pet = entry_t - other_visit.last_t
                        if 0.0 <= pet <= self.max_pet_s:
                            pair = (tid, other_tid) if norm_cat == "vehicle" else (other_tid, tid)
                            prev_min = near_misses.get(pair)
                            if prev_min is None or pet < prev_min:
                                near_misses[pair] = pet

        # 未在本帧出现的轨迹清除活跃格标记，若稍后重入同一网格则开启新一次驻留计数。
        self._active_cells_by_track = current_cells_by_track
        return near_misses
