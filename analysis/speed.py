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
from typing import Dict, List, Optional, Tuple

from .policy import object_policy


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
    object_kind: str = ""      # pedestrian | bicycle | motorcycle | car | bus | truck | motor_vehicle

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
    ):
        if type(window_size) is not int or window_size < 3:
            raise ValueError("window_size 必须是 >= 3 的整数帧数")
        if not math.isfinite(ema_alpha) or not (0 < ema_alpha <= 1):
            raise ValueError("ema_alpha 必须在 (0, 1] 区间")
        if not math.isfinite(speed_limit_kmh) or speed_limit_kmh <= 0:
            raise ValueError("speed_limit_kmh 必须为有限正数")
        if not math.isfinite(stationary_speed_ms) or stationary_speed_ms <= 0:
            raise ValueError("stationary_speed_ms 必须为有限正数")
        if type(stationary_min_frames) is not int or stationary_min_frames < 1:
            raise ValueError("stationary_min_frames 必须是 >= 1 的整数帧数")
        if not math.isfinite(reliable_min_span_s) or reliable_min_span_s < 0:
            raise ValueError("reliable_min_span_s 必须为有限非负数")

        self.track_id = track_id
        self.category = category
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.speed_limit_kmh = speed_limit_kmh
        self.stationary_speed_ms = stationary_speed_ms
        self.stationary_min_frames = stationary_min_frames
        self.reliable_min_span_s = reliable_min_span_s
        self.object_kind = object_policy(object_kind, category).kind

        # 历史物理轨迹窗口: deque of (timestamp_s, x_m, y_m)
        self.history: deque[Tuple[float, float, float]] = deque(maxlen=window_size)

        # 平滑后的速度状态 (米/秒)
        self.smoothed_vx = 0.0
        self.smoothed_vy = 0.0
        self.has_valid_velocity = False

        # 最后一个有限观测位置/时间戳与持续静止帧计数
        self._last_valid_xy: Optional[Tuple[float, float]] = None
        self._last_timestamp_s: Optional[float] = None
        self._stationary_streak = 0

        # 类别滞回状态：新类别需连续多帧确认才生效
        self._pending_category: Optional[str] = None
        self._pending_count = 0

    def observe_category(self, category: str, confirm_frames: int = 3):
        """按连续 N 帧确认机制更新类别，抑制 person/vehicle 检测抖动"""
        if category == self.category:
            self._pending_category, self._pending_count = None, 0
            return
        if category == self._pending_category:
            self._pending_count += 1
        else:
            self._pending_category, self._pending_count = category, 1
        if self._pending_count >= confirm_frames:
            self.category = category
            self._pending_category, self._pending_count = None, 0

    def update(self, timestamp_s: float, x_m: float, y_m: float) -> SpeedReading:
        """输入当前帧物理坐标与时间戳，更新运动滤波器状态"""
        # 非有限观测（如地平线附近的退化投影点）不进入历史，
        # 否则 NaN 经差分进入 EMA 后会使平滑速度永久不可恢复；
        # 读数回填最后有效位置，不把 NaN 暴露给下游渲染/研判
        if not (math.isfinite(timestamp_s) and math.isfinite(x_m) and math.isfinite(y_m)):
            x_out, y_out = self._last_valid_xy if self._last_valid_xy is not None else (x_m, y_m)
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
            )

        # 时间戳必须严格递增：回退/重复观测会让窗口差分 dt <= 0，
        # 静默写历史会得到失真速度读数，宁可响亮失败
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise ValueError("timestamp_s 必须逐帧严格递增")
        self._last_timestamp_s = timestamp_s
        self._last_valid_xy = (x_m, y_m)
        self.history.append((timestamp_s, x_m, y_m))

        # 样本少于 3 个点时处于预热状态，速度不作为强判断
        if len(self.history) < 3:
            return SpeedReading(
                track_id=self.track_id,
                category=self.category,
                x_m=x_m,
                y_m=y_m,
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

        # 静止判定必须使用截断前的平滑速度幅值：
        # 微动截断（<1 km/h 显示为 0）只影响展示标量，
        # 若用截断后值判定，0.2 m/s 蠕行车辆会被误豁免为"泊车"导致碰撞漏报
        smoothed_mag = math.hypot(self.smoothed_vx, self.smoothed_vy)
        if smoothed_mag < self.stationary_speed_ms:
            self._stationary_streak += 1
        else:
            self._stationary_streak = 0
        is_stationary = self._stationary_streak >= self.stationary_min_frames

        speed_ms = smoothed_mag
        speed_kmh = speed_ms * 3.6

        # 行人微动阈值截断（静止站立摆动不计为移动速度）
        if self.category == "pedestrian" and speed_kmh < 0.8:
            speed_ms, speed_kmh = 0.0, 0.0
        elif self.category == "vehicle" and speed_kmh < 1.0:
            speed_ms, speed_kmh = 0.0, 0.0

        # 方向角计算 (0~360度，以地面 +X 轴为 0 度，朝 +Y 轴正向)
        heading = (math.degrees(math.atan2(self.smoothed_vy, self.smoothed_vx)) + 360.0) % 360.0

        # 超速研判：车辆且车速超过 30 km/h (可设容差如 1.0 km/h)
        is_speeding = (self.category == "vehicle") and (speed_kmh > self.speed_limit_kmh)

        # 可信判定需同时满足样本数与时间跨度下限：
        # 仅按样本数判定会让高 FPS 视频的 4 帧（仅 0.12s）过早收敛、
        # 低 FPS 的稀疏观测却长期不可信；跨度下限使跨帧率语义一致
        min_samples = min(4, self.window_size)
        is_reliable = (
            len(self.history) >= min_samples and dt >= self.reliable_min_span_s
        )

        return SpeedReading(
            track_id=self.track_id,
            category=self.category,
            x_m=x_m,
            y_m=y_m,
            vx_ms=self.smoothed_vx,
            vy_ms=self.smoothed_vy,
            speed_ms=speed_ms,
            speed_kmh=speed_kmh,
            is_speeding=is_speeding,
            heading_deg=heading,
            is_reliable=is_reliable,
            is_stationary=is_stationary,
            object_kind=self.object_kind,
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
        reliable_min_span_s: float = 0.2,
    ):
        if not math.isfinite(ema_alpha) or not (0 < ema_alpha <= 1):
            raise ValueError("ema_alpha 必须在 (0, 1] 区间")
        if type(window_size) is not int or window_size < 3:
            raise ValueError("window_size 必须是 >= 3 的整数帧数")
        if not math.isfinite(speed_limit_kmh) or speed_limit_kmh <= 0:
            raise ValueError("speed_limit_kmh 必须为有限正数")
        if not math.isfinite(stationary_speed_ms) or stationary_speed_ms <= 0:
            raise ValueError("stationary_speed_ms 必须为有限正数")
        if type(stationary_min_frames) is not int or stationary_min_frames < 1:
            raise ValueError("stationary_min_frames 必须是 >= 1 的整数帧数")
        if type(category_confirm_frames) is not int or category_confirm_frames < 1:
            raise ValueError("category_confirm_frames 必须是 >= 1 的整数帧数")
        if not math.isfinite(reliable_min_span_s) or reliable_min_span_s < 0:
            raise ValueError("reliable_min_span_s 必须为有限非负数")
        self.speed_limit_kmh = speed_limit_kmh
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.stationary_speed_ms = stationary_speed_ms
        self.stationary_min_frames = stationary_min_frames
        self.category_confirm_frames = category_confirm_frames
        self.reliable_min_span_s = reliable_min_span_s
        self.tracks: Dict[int, TrackMotionTracker] = {}

    def update(
        self, track_id: int, category: str, timestamp_s: float, x_m: float, y_m: float,
        object_kind: str | None = None,
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
            )
        else:
            tracker = self.tracks[track_id]
            # 无效观测（非有限时间戳/坐标）在 tracker 内会早退不写历史，
            # 此处同样不得推进类别滞回计数——拒绝与忽略都必须无副作用
            if math.isfinite(timestamp_s) and math.isfinite(x_m) and math.isfinite(y_m):
                if (
                    tracker._last_timestamp_s is not None
                    and timestamp_s <= tracker._last_timestamp_s
                ):
                    raise ValueError("timestamp_s 必须逐帧严格递增")
                # 跟踪 ID 沿用期间检测类别可能被纠正（如 person -> car），
                # 但每帧观测可能抖动，需连续确认后才切换类别
                tracker.observe_category(category, self.category_confirm_frames)
                if tracker.category in ("pedestrian", "person"):
                    tracker.object_kind = "pedestrian"
                elif object_kind is not None or tracker.object_kind == "pedestrian":
                    tracker.object_kind = object_policy(object_kind, tracker.category).kind
        return self.tracks[track_id].update(timestamp_s, x_m, y_m)

    def prune_stale_tracks(self, live_track_ids: set[int]):
        """清理已终结或离场的轨迹实例，防止内存泄漏"""
        for tid in list(self.tracks.keys()):
            if tid not in live_track_ids:
                del self.tracks[tid]
