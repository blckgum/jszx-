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


@dataclass
class SpeedReading:
    """单目标速度研判数据模型"""
    track_id: int
    category: str              # "vehicle" | "pedestrian"
    x_m: float                 # 当前地面 X (米)
    y_m: float                 # 当前地面 Y (米)
    vx_ms: float               # 平滑 X 轴速度分量 (米/秒)
    vy_ms: float               # 平滑 Y 轴速度分量 (米/秒)
    speed_ms: float            # 标量速度 (米/秒)
    speed_kmh: float           # 标量时速 (公里/小时)
    is_speeding: bool          # 是否超过校园限速 (30 km/h)
    heading_deg: float         # 运动方向角 (0~360度)
    is_reliable: bool          # 速度估计是否已收敛可信 (需要足够的连续观测点)


class TrackMotionTracker:
    """单个跟踪目标的连续空间运动状态滤波器"""

    def __init__(
        self,
        track_id: int,
        category: str,
        window_size: int = 8,
        ema_alpha: float = 0.35,
        speed_limit_kmh: float = 30.0,
    ):
        self.track_id = track_id
        self.category = category
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.speed_limit_kmh = speed_limit_kmh

        # 历史物理轨迹窗口: deque of (timestamp_s, x_m, y_m)
        self.history: deque[Tuple[float, float, float]] = deque(maxlen=window_size)

        # 平滑后的速度状态 (米/秒)
        self.smoothed_vx = 0.0
        self.smoothed_vy = 0.0
        self.has_valid_velocity = False

    def update(self, timestamp_s: float, x_m: float, y_m: float) -> SpeedReading:
        """输入当前帧物理坐标与时间戳，更新运动滤波器状态"""
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
            )

        # 采用窗口两端平滑区间求导，抵御单帧目标框边缘抖动
        t_first, x_first, y_first = self.history[0]
        t_last, x_last, y_last = self.history[-1]
        dt = t_last - t_first

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

        speed_ms = math.hypot(self.smoothed_vx, self.smoothed_vy)
        speed_kmh = speed_ms * 3.6

        # 行人微动阈值截断（静止站立摆动不计为移动速度）
        if self.category == "pedestrian" and speed_kmh < 0.8:
            speed_ms, speed_kmh = 0.0, 0.0
        elif self.category == "vehicle" and speed_kmh < 1.0:
            speed_ms, speed_kmh = 0.0, 0.0

        # 方向角计算 (0~360度，正北为0度，顺时针)
        heading = (math.degrees(math.atan2(self.smoothed_vy, self.smoothed_vx)) + 360.0) % 360.0

        # 超速研判：车辆且车速超过 30 km/h (可设容差如 1.0 km/h)
        is_speeding = (self.category == "vehicle") and (speed_kmh > self.speed_limit_kmh)

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
            is_reliable=len(self.history) >= 4,
        )


class SpeedEstimator:
    """多目标全局速度估算与生命周期管理器"""

    def __init__(self, speed_limit_kmh: float = 30.0, window_size: int = 8):
        self.speed_limit_kmh = speed_limit_kmh
        self.window_size = window_size
        self.tracks: Dict[int, TrackMotionTracker] = {}

    def update(
        self, track_id: int, category: str, timestamp_s: float, x_m: float, y_m: float
    ) -> SpeedReading:
        """更新单个目标并输出速度测算"""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackMotionTracker(
                track_id=track_id,
                category=category,
                window_size=self.window_size,
                speed_limit_kmh=self.speed_limit_kmh,
            )
        return self.tracks[track_id].update(timestamp_s, x_m, y_m)

    def prune_stale_tracks(self, active_track_ids: set[int]):
        """清理已终结或离场的轨迹实例，防止内存泄漏"""
        for tid in list(self.tracks.keys()):
            if tid not in active_track_ids:
                del self.tracks[tid]
