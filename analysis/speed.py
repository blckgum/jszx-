# -*- coding: utf-8 -*-
"""
物理空间平滑测速与 30 km/h 限速研判引擎 (Physical Speed Estimation & Speeding Detection)
=============================================================================
利用时间窗口内的地面实际米制坐标序列进行微分求解，
集成指数移动平均 (EMA) 滤波以彻底消除目标边界框脚底接触点抖动，
输出高可信度的瞬时速度矢量 (vx, vy) 与标量车速 (km/h)。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .kalman import ConstantAccelerationKalman2D, ConstantTurnRateKalman2D
from .policy import ObjectPolicy, object_policy


def _is_finite_number(value) -> bool:
    # Python 中 bool 是 int 子类，需先排除布尔值避免 True/False 当数值通过校验。
    return (
        type(value) is not bool
        and not isinstance(value, np.bool_)
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _validate_estimator_mode(estimator_mode: str) -> str:
    if type(estimator_mode) is not str or estimator_mode not in ("ema", "kalman", "ctrv"):
        raise ValueError("estimator_mode 必须是 'ema'、'kalman' 或 'ctrv'")
    return estimator_mode


def _validate_category(category: str) -> str:
    """校验目标类别必须为非空白字符串，统一 ema 与 kalman 模式校验口径。"""
    if type(category) is not str or not category.strip():
        raise ValueError("category 必须是非空字符串")
    return category.strip()


def _validate_motion_params(
    window_size: int,
    ema_alpha: float,
    speed_limit_kmh: float,
    stationary_speed_ms: float,
    stationary_min_frames: int,
    reliable_min_span_s: float,
    estimator_mode: str,
) -> str:
    """统一校验 TrackMotionTracker 与 SpeedEstimator 共享的运动学滤波器参数。"""
    if type(window_size) is not int or window_size < 3:
        raise ValueError("window_size 必须是 >= 3 的整数帧数")
    if not _is_finite_number(ema_alpha) or not (0 < ema_alpha <= 1):
        raise ValueError("ema_alpha 必须在 (0, 1] 区间")
    if not _is_finite_number(speed_limit_kmh) or speed_limit_kmh <= 0:
        raise ValueError("speed_limit_kmh 必须为有限正数")
    if not _is_finite_number(stationary_speed_ms) or stationary_speed_ms <= 0:
        raise ValueError("stationary_speed_ms 必须为有限正数")
    if type(stationary_min_frames) is not int or stationary_min_frames < 1:
        raise ValueError("stationary_min_frames 必须是 >= 1 的整数帧数")
    if not _is_finite_number(reliable_min_span_s) or reliable_min_span_s < 0:
        raise ValueError("reliable_min_span_s 必须为有限非负数")
    return _validate_estimator_mode(estimator_mode)


@dataclass
class SpeedReading:
    """单目标速度研判数据模型"""
    track_id: int
    category: str              # "vehicle" | "pedestrian"
    x_m: float                 # 当前地面 X (米)
    y_m: float                 # 当前地面 Y (米)
    vx_ms: float               # 平滑 X 轴速度分量 (米/秒)
    vy_ms: float               # 平滑 Y 轴速度分量 (米/秒)
    speed_ms: float            # 标量速度 (米/秒)；微动截断后可为 0，vx/vy 矢量分量不受影响
    speed_kmh: float           # 标量时速 (公里/小时)
    is_speeding: bool          # 是否超过校园限速 (30 km/h)
    heading_deg: float         # 运动方向角 (0~360度)
    is_reliable: bool          # 速度估计是否已收敛可信 (需要足够的连续观测点)
    is_stationary: bool = False  # 是否持续静止（泊车/停靠），需连续多帧低速确认
    in_parking_zone: bool = False  # 是否位于标定的路侧停车区多边形内
    object_kind: str = ""      # pedestrian | bicycle | motorcycle | car | bus | truck | motor_vehicle
    ax_ms2: float = 0.0        # X 轴加速度分量 (米/秒^2)
    ay_ms2: float = 0.0        # Y 轴加速度分量 (米/秒^2)
    pos_std_m: Optional[float] = None  # 滤波位置标准差 (米)
    pos_cov_m2: Optional[Tuple[float, float, float, float]] = None  # 2x2 位置协方差展开 (cxx, cxy, cyx, cyy)
    yaw_rate_rad_s: float = 0.0  # 横摆角速度 (弧度/秒，CTRV 模式下有效)

    def __post_init__(self):
        if not self.object_kind:
            self.object_kind = object_policy(None, self.category).kind


class TrackMotionTracker:
    """单个跟踪目标的连续空间运动状态滤波器"""

    def __init__(
        self,
        track_id: int,
        category: str,
        window_size: int = 8,
        ema_alpha: float = 0.35,
        speed_limit_kmh: float = 30.0,
        stationary_speed_ms: float = 0.1,
        stationary_min_frames: int = 8,
        reliable_min_span_s: float = 0.2,
        object_kind: str | None = None,
        object_policies: dict[str, ObjectPolicy] | None = None,
        estimator_mode: str = "ema",
    ):
        _validate_motion_params(
            window_size=window_size,
            ema_alpha=ema_alpha,
            speed_limit_kmh=speed_limit_kmh,
            stationary_speed_ms=stationary_speed_ms,
            stationary_min_frames=stationary_min_frames,
            reliable_min_span_s=reliable_min_span_s,
            estimator_mode=estimator_mode,
        )
        clean_category = _validate_category(category)

        self.track_id = track_id
        self.category = clean_category
        self.window_size = window_size
        self.ema_alpha = float(ema_alpha)
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.stationary_speed_ms = float(stationary_speed_ms)
        self.stationary_min_frames = stationary_min_frames
        self.reliable_min_span_s = float(reliable_min_span_s)
        self.object_policies = object_policies
        self.estimator_mode = estimator_mode
        # 类别为车辆时，过滤误传入的 person/pedestrian 标签（含 Mock 占位对象），
        # 避免把车辆的碰撞安全半径错误缩减成行人半径。
        if clean_category in ("pedestrian", "person"):
            self.object_kind = object_policy(None, clean_category, object_policies).kind
        else:
            safe_kind = (
                object_kind
                if isinstance(object_kind, str)
                and object_kind.strip()
                and object_kind.strip().lower() not in ("person", "pedestrian")
                else None
            )
            self.object_kind = object_policy(safe_kind, clean_category, object_policies).kind

        # 历史物理轨迹窗口: deque of (timestamp_s, x_m, y_m)
        self.history: deque[Tuple[float, float, float]] = deque(maxlen=window_size)

        # 平滑后的速度状态 (米/秒)
        self.smoothed_vx = 0.0
        self.smoothed_vy = 0.0
        self.has_valid_velocity = False

        # 卡尔曼滤波器实例（kalman 使用 6 维 CA，ctrv 使用 5 维 CTRV EKF）
        if estimator_mode == "kalman":
            self._kalman: Optional[ConstantAccelerationKalman2D | ConstantTurnRateKalman2D] = (
                ConstantAccelerationKalman2D(category=clean_category)
            )
        elif estimator_mode == "ctrv":
            self._kalman = ConstantTurnRateKalman2D(category=clean_category)
        else:
            self._kalman = None

        # 最后一个有限观测位置/时间戳与持续静止帧计数
        self._last_valid_xy: Optional[Tuple[float, float]] = None
        self._last_timestamp_s: Optional[float] = None
        self._stationary_streak = 0
        self._last_kalman_reset_count = 0

        # 类别滞回状态：新类别需连续多帧确认才生效
        self._pending_category: Optional[str] = None
        self._pending_count = 0

    def observe_category(self, category: str, confirm_frames: int = 3):
        """按连续 N 帧确认机制更新类别，抑制 person/vehicle 检测抖动"""
        # 提前统一校验非空白字符串类别，确保 ema/kalman 行为一致且异常时不污染待确认状态
        clean_category = _validate_category(category)
        if clean_category == self.category:
            self._pending_category, self._pending_count = None, 0
            return
        if clean_category == self._pending_category:
            self._pending_count += 1
        else:
            self._pending_category, self._pending_count = clean_category, 1
        if self._pending_count >= confirm_frames:
            # 先校验并同步更新卡尔曼先验再修改 self.category，保障异常时状态原子不变
            if self._kalman is not None:
                self._kalman.set_category(clean_category)
            self.category = clean_category
            self._pending_category, self._pending_count = None, 0

    def _kalman_extras(
        self,
    ) -> Tuple[float, float, Optional[float], Optional[Tuple[float, float, float, float]], float]:
        """读取卡尔曼加速度、位置不确定度与横摆角速度；未初始化或 EMA 模式返回默认值。"""
        if self._kalman is None or self._kalman.update_count == 0:
            return 0.0, 0.0, None, None, 0.0
        ax, ay = self._kalman.acceleration_xy
        cov = self._kalman.pos_cov_2x2
        pos_cov_tuple = (
            float(cov[0, 0]),
            float(cov[0, 1]),
            float(cov[1, 0]),
            float(cov[1, 1]),
        )
        yaw_rate = float(getattr(self._kalman, "yaw_rate_rad_s", 0.0))
        return ax, ay, self._kalman.pos_std_m, pos_cov_tuple, yaw_rate

    def _apply_micro_truncation(self, mag_ms: float) -> Tuple[float, float]:
        """应用行人与车辆微动速度显示截断，返回 (speed_ms, speed_kmh)。"""
        speed_ms = mag_ms
        speed_kmh = speed_ms * 3.6
        if self.category in ("pedestrian", "person") and speed_kmh < 0.8:
            return 0.0, 0.0
        if self.category == "vehicle" and speed_kmh < 1.0:
            return 0.0, 0.0
        return speed_ms, speed_kmh

    def _finalize_kinematics_reading(
        self,
        x_m: float,
        y_m: float,
        vx_ms: float,
        vy_ms: float,
        dt_span_s: float,
        ax_ms2: float = 0.0,
        ay_ms2: float = 0.0,
        pos_std_m: Optional[float] = None,
        pos_cov_m2: Optional[Tuple[float, float, float, float]] = None,
        yaw_rate_rad_s: float = 0.0,
    ) -> SpeedReading:
        """统一计算静止滞回、微动截断、航向角、超速与可信度判定。"""
        # 静止判定必须使用截断前的平滑速度幅值：
        # 微动截断（<1 km/h 显示为 0）只影响展示标量，
        # 若用截断后值判定，0.2 m/s 蠕行车辆会被误豁免为"泊车"导致碰撞漏报
        smoothed_mag = math.hypot(vx_ms, vy_ms)
        if len(self.history) >= 3:
            if smoothed_mag < self.stationary_speed_ms:
                self._stationary_streak += 1
            else:
                self._stationary_streak = 0
        is_stationary = self._stationary_streak >= self.stationary_min_frames

        speed_ms, speed_kmh = self._apply_micro_truncation(smoothed_mag)
        # 静止且 CTRV 已建立先验航向时沿用滤波航向，避免 0 矢量下 atan2 退化为 0
        if (
            smoothed_mag <= 1e-9
            and self.estimator_mode == "ctrv"
            and self._kalman is not None
            and self._kalman.update_count >= 2
        ):
            heading = float(self._kalman.heading_deg)
        else:
            heading = (math.degrees(math.atan2(vy_ms, vx_ms)) + 360.0) % 360.0
        min_samples = min(4, self.window_size)
        is_reliable = (
            len(self.history) >= min_samples and dt_span_s >= self.reliable_min_span_s
        )
        is_speeding = (
            is_reliable
            and (self.category == "vehicle")
            and (speed_kmh > self.speed_limit_kmh)
        )
        return SpeedReading(
            track_id=self.track_id,
            category=self.category,
            x_m=x_m,
            y_m=y_m,
            vx_ms=vx_ms,
            vy_ms=vy_ms,
            speed_ms=speed_ms,
            speed_kmh=speed_kmh,
            is_speeding=is_speeding,
            heading_deg=heading,
            is_reliable=is_reliable,
            is_stationary=is_stationary,
            object_kind=self.object_kind,
            ax_ms2=ax_ms2,
            ay_ms2=ay_ms2,
            pos_std_m=pos_std_m,
            pos_cov_m2=pos_cov_m2,
            yaw_rate_rad_s=float(yaw_rate_rad_s),
        )

    def update(
        self,
        timestamp_s: float,
        x_m: float,
        y_m: float,
        obs_cov_m2: Optional[np.ndarray] = None,
    ) -> SpeedReading:
        """输入当前帧物理坐标与时间戳，更新运动滤波器状态"""
        # 非有限观测（如地平线附近的退化投影点）不进入历史，
        # 否则 NaN 经差分进入 EMA/Kalman 后会使平滑速度永久不可恢复；
        # 读数回填最后有效位置，不把 NaN 暴露给下游渲染/研判
        if not (_is_finite_number(timestamp_s) and _is_finite_number(x_m) and _is_finite_number(y_m)):
            x_out, y_out = self._last_valid_xy if self._last_valid_xy is not None else (0.0, 0.0)
            ax_out, ay_out, std_out, cov_out, yaw_out = self._kalman_extras()
            return SpeedReading(
                track_id=self.track_id,
                category=self.category,
                x_m=x_out,
                y_m=y_out,
                vx_ms=self.smoothed_vx,
                vy_ms=self.smoothed_vy,
                speed_ms=math.hypot(self.smoothed_vx, self.smoothed_vy),
                speed_kmh=math.hypot(self.smoothed_vx, self.smoothed_vy) * 3.6,
                is_speeding=False,
                heading_deg=0.0,
                is_reliable=False,
                is_stationary=self._stationary_streak >= self.stationary_min_frames,
                object_kind=self.object_kind,
                ax_ms2=ax_out,
                ay_ms2=ay_out,
                pos_std_m=std_out,
                pos_cov_m2=cov_out,
                yaw_rate_rad_s=yaw_out,
            )

        ts = float(timestamp_s)
        xm = float(x_m)
        ym = float(y_m)

        # 时间戳必须严格递增：回退/重复观测会让窗口差分 dt <= 0，
        # 静默写历史会得到失真速度读数，宁可响亮失败
        if self._last_timestamp_s is not None and ts <= self._last_timestamp_s:
            raise ValueError("timestamp_s 必须逐帧严格递增")

        if self.estimator_mode in ("kalman", "ctrv"):
            assert self._kalman is not None
            self._last_timestamp_s = ts
            accepted = self._kalman.update(ts, xm, ym, obs_cov_m2=obs_cov_m2)
            reset_count = self._kalman.reset_count
            if self._kalman.was_reset or reset_count != self._last_kalman_reset_count:
                # 滤波器触发连续拒收或数值溢出重置后，清空旧历史窗口与静止计数，
                # 要求重置后重新累积足够跨度的干净观测才恢复 is_reliable。
                self._last_kalman_reset_count = reset_count
                self.history.clear()
                self.history.append((ts, xm, ym))
                self._stationary_streak = 0
            elif accepted:
                self.history.append((ts, xm, ym))

            kf_x, kf_y = self._kalman.position_xy
            kf_vx, kf_vy = self._kalman.velocity_xy
            kf_ax, kf_ay, pos_std, pos_cov_tuple, yaw_rate = self._kalman_extras()
            self._last_valid_xy = (kf_x, kf_y)
            self.smoothed_vx = kf_vx
            self.smoothed_vy = kf_vy
            self.has_valid_velocity = len(self.history) >= 2
            dt_span = (
                self.history[-1][0] - self.history[0][0]
                if len(self.history) >= 2
                else 0.0
            )
            return self._finalize_kinematics_reading(
                x_m=kf_x,
                y_m=kf_y,
                vx_ms=kf_vx,
                vy_ms=kf_vy,
                dt_span_s=dt_span,
                ax_ms2=kf_ax,
                ay_ms2=kf_ay,
                pos_std_m=pos_std,
                pos_cov_m2=pos_cov_tuple,
                yaw_rate_rad_s=yaw_rate,
            )

        self._last_timestamp_s = ts
        self._last_valid_xy = (xm, ym)
        self.history.append((ts, xm, ym))

        # 样本少于 3 个点时处于预热状态，速度不作为强判断
        if len(self.history) < 3:
            return SpeedReading(
                track_id=self.track_id,
                category=self.category,
                x_m=xm,
                y_m=ym,
                vx_ms=0.0,
                vy_ms=0.0,
                speed_ms=0.0,
                speed_kmh=0.0,
                is_speeding=False,
                heading_deg=0.0,
                is_reliable=False,
                object_kind=self.object_kind,
            )

        # 采用窗口两端平滑区间求导，抵御单帧目标框边缘抖动
        t_first, x_first, y_first = self.history[0]
        t_last, x_last, y_last = self.history[-1]
        dt = t_last - t_first

        # 窗口首末时间戳无跨度时（重复投影位置挤占窗口等）dt≈0，
        # 差分会放大成无穷速度：按零速度处理而不除以微小 dt
        if dt <= 1e-4:
            raw_vx, raw_vy = 0.0, 0.0
        else:
            raw_vx = (x_last - x_first) / dt
            raw_vy = (y_last - y_first) / dt

        # 过滤极端异常噪点（例如跨帧漂移超现实速度 > 120 km/h）
        raw_speed = math.hypot(raw_vx, raw_vy)
        if raw_speed > 33.3:  # 超过 120 km/h 视为跳帧噪点予以抑制
            raw_vx, raw_vy = self.smoothed_vx, self.smoothed_vy

        # 指数平滑滤波 (EMA)
        if not self.has_valid_velocity:
            self.smoothed_vx = raw_vx
            self.smoothed_vy = raw_vy
            self.has_valid_velocity = True
        else:
            self.smoothed_vx = (
                self.ema_alpha * raw_vx + (1.0 - self.ema_alpha) * self.smoothed_vx
            )
            self.smoothed_vy = (
                self.ema_alpha * raw_vy + (1.0 - self.ema_alpha) * self.smoothed_vy
            )

        return self._finalize_kinematics_reading(
            x_m=xm,
            y_m=ym,
            vx_ms=self.smoothed_vx,
            vy_ms=self.smoothed_vy,
            dt_span_s=dt,
        )


class SpeedEstimator:
    """多目标全局速度估算与生命周期管理器"""

    def __init__(
        self,
        speed_limit_kmh: float = 30.0,
        window_size: int = 8,
        ema_alpha: float = 0.35,
        stationary_speed_ms: float = 0.1,
        stationary_min_frames: int = 8,
        category_confirm_frames: int = 3,
        object_policies: dict[str, ObjectPolicy] | None = None,
        reliable_min_span_s: float = 0.2,
        estimator_mode: str = "ema",
    ):
        _validate_motion_params(
            window_size=window_size,
            ema_alpha=ema_alpha,
            speed_limit_kmh=speed_limit_kmh,
            stationary_speed_ms=stationary_speed_ms,
            stationary_min_frames=stationary_min_frames,
            reliable_min_span_s=reliable_min_span_s,
            estimator_mode=estimator_mode,
        )
        if type(category_confirm_frames) is not int or category_confirm_frames < 1:
            raise ValueError("category_confirm_frames 必须是 >= 1 的整数帧数")

        self.speed_limit_kmh = float(speed_limit_kmh)
        self.window_size = window_size
        self.ema_alpha = float(ema_alpha)
        self.stationary_speed_ms = float(stationary_speed_ms)
        self.stationary_min_frames = stationary_min_frames
        self.category_confirm_frames = category_confirm_frames
        self.object_policies = object_policies
        self.reliable_min_span_s = float(reliable_min_span_s)
        self.estimator_mode = estimator_mode
        self.tracks: Dict[int, TrackMotionTracker] = {}

    def update(
        self,
        track_id: int,
        category: str,
        timestamp_s: float,
        x_m: float,
        y_m: float,
        object_kind: str | None = None,
        obs_cov_m2: Optional[np.ndarray] = None,
    ) -> SpeedReading:
        """更新单个目标并输出速度测算"""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackMotionTracker(
                track_id=track_id,
                category=category,
                window_size=self.window_size,
                ema_alpha=self.ema_alpha,
                speed_limit_kmh=self.speed_limit_kmh,
                stationary_speed_ms=self.stationary_speed_ms,
                stationary_min_frames=self.stationary_min_frames,
                reliable_min_span_s=self.reliable_min_span_s,
                object_kind=object_kind,
                object_policies=self.object_policies,
                estimator_mode=self.estimator_mode,
            )
        else:
            tracker = self.tracks[track_id]
            # 无效观测（非有限时间戳/坐标）在 tracker 内会早退不写历史，
            # 此处同样不得推进类别滞回计数——拒绝与忽略都必须无副作用
            if _is_finite_number(timestamp_s) and _is_finite_number(x_m) and _is_finite_number(y_m):
                if (
                    tracker._last_timestamp_s is not None
                    and timestamp_s <= tracker._last_timestamp_s
                ):
                    raise ValueError("timestamp_s 必须逐帧严格递增")
                # 跟踪 ID 沿用期间检测类别可能被纠正（如 person -> car），
                # 但每帧观测可能抖动，需连续确认后才切换类别
                tracker.observe_category(category, self.category_confirm_frames)
                # 只有已确认类别切换后才同步刷新细分子类 object_kind，
                # 防止单帧误检（如 car 闪现成 person）提前缩减车辆安全半径。
                if tracker.category in ("pedestrian", "person"):
                    tracker.object_kind = object_policy(
                        None, tracker.category, tracker.object_policies
                    ).kind
                else:
                    has_valid_veh_kind = (
                        isinstance(object_kind, str)
                        and bool(object_kind.strip())
                        and object_kind.strip().lower() not in ("person", "pedestrian")
                    )
                    ped_kind = object_policy(
                        None, "pedestrian", tracker.object_policies
                    ).kind
                    if category.strip() == tracker.category and has_valid_veh_kind:
                        tracker.object_kind = object_policy(
                            object_kind, tracker.category, tracker.object_policies
                        ).kind
                    # 若确认类别已从行人切回车辆而本帧未带合法车型子类，需把原先的行人 kind
                    # 恢复为默认机动车策略（同时兼容自定义策略 YAML 里的行人 kind 名称）。
                    elif tracker.object_kind in ("pedestrian", ped_kind):
                        safe_kind = object_kind if has_valid_veh_kind else None
                        tracker.object_kind = object_policy(
                            safe_kind, tracker.category, tracker.object_policies
                        ).kind
        if obs_cov_m2 is not None:
            return self.tracks[track_id].update(
                timestamp_s, x_m, y_m, obs_cov_m2=obs_cov_m2
            )
        return self.tracks[track_id].update(timestamp_s, x_m, y_m)

    def transfer_track_state(self, old_track_id: int, new_track_id: int) -> bool:
        """将已有轨迹的滤波状态迁移至 new_track_id 并同步更新 tracker.track_id。"""
        if type(old_track_id) is not int or type(new_track_id) is not int:
            raise ValueError("old_track_id 与 new_track_id 必须是整数")
        if old_track_id < 0 or new_track_id < 0:
            raise ValueError("track_id 不能为负数")
        tracker = self.tracks.get(old_track_id)
        if tracker is None:
            return False
        if old_track_id == new_track_id:
            tracker.track_id = new_track_id
            return True
        self.tracks.pop(old_track_id)
        tracker.track_id = new_track_id
        self.tracks[new_track_id] = tracker
        return True

    def prune_stale_tracks(self, live_track_ids: set[int]):
        """清理已终结或离场的轨迹实例，防止内存泄漏"""
        for tid in list(self.tracks.keys()):
            if tid not in live_track_ids:
                del self.tracks[tid]
