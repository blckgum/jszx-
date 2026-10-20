# -*- coding: utf-8 -*-
"""
冲突风险感知与车速预警增强可视化引擎 (Enhanced Spatial Risk Overlay)
=============================================================================
在视频画面上实时渲染：
1. 受控监测路面标定多边形网格；
2. 目标未来 2~3 秒等速外推预测轨迹虚线；
3. 真实物理时速 (km/h) 状态徽章与超速红框；
4. 车辆与行人冲突连线、预测碰撞热区圈及顶部 HUD 分级预警看板。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration import RoadCalibrator
from .conflict import ConflictAssessment, RiskLevel, build_conflict_maps
from .speed import SpeedReading


def trajectory_color(conflict: Optional[ConflictAssessment]) -> Tuple[int, int, int]:
    """按冲突等级给预测轨迹配色：L3 红 / L2 橙 / 其他绿 (BGR)"""
    if conflict is None:
        return (0, 255, 0)
    if conflict.risk_level == RiskLevel.LEVEL_3_CRITICAL:
        return (0, 0, 255)
    if conflict.risk_level == RiskLevel.LEVEL_2_CAUTION:
        return (0, 140, 255)
    return (0, 255, 0)


class EnhancedConflictVisualizer:
    """视觉增强渲染引擎"""

    def __init__(self, calibrator: RoadCalibrator, speed_limit_kmh: float = 30.0):
        self.calibrator = calibrator
        self.speed_limit_kmh = speed_limit_kmh

        # 调色板定义 (BGR 格式)
        self.COLOR_SAFE = (50, 205, 50)       # 绿色 (正常)
        self.COLOR_SPEEDING = (0, 215, 255)   # 黄色 (超速)
        self.COLOR_CAUTION = (0, 140, 255)    # 橙色 (注意)
        self.COLOR_CRITICAL = (0, 0, 255)     # 红色 (紧急碰撞)
        self.COLOR_ZONE = (255, 191, 0)       # 天蓝/深黄色 (标定区域)

    def draw_calibrated_ground_zone(self, frame: np.ndarray, alpha: float = 0.25):
        """在路面上半透明叠加渲染受控标定多边形区域"""
        pts = np.array(self.calibrator.image_points, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (240, 180, 0))
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 220, 255), thickness=2)

    def draw_future_trajectory(
        self,
        frame: np.ndarray,
        reading: SpeedReading,
        horizon_s: float = 2.5,
        steps: int = 5,
        color: Tuple[int, int, int] = (0, 255, 0),
    ):
        """在地面物理平面上外推并投射未来运动轨迹虚线"""
        if not reading.is_reliable or reading.speed_ms < 0.3:
            return

        dt = horizon_s / steps
        # 出画点将轨迹拆成独立线段，避免跨越不可见区域画出弦线
        segments: List[List[Tuple[int, int]]] = []
        current: List[Tuple[int, int]] = []
        last_visible: Optional[Tuple[int, int]] = None
        for i in range(steps + 1):
            t = i * dt
            future_x = reading.x_m + reading.vx_ms * t
            future_y = reading.y_m + reading.vy_ms * t
            u, v = self.calibrator.ground_to_pixel(future_x, future_y)
            if 0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]:
                current.append((u, v))
                last_visible = (u, v)
            else:
                if len(current) >= 2:
                    segments.append(current)
                current = []
        if len(current) >= 2:
            segments.append(current)

        for seg in segments:
            for j in range(len(seg) - 1):
                cv2.line(frame, seg[j], seg[j + 1], color, 2, cv2.LINE_AA)
        # 绘制终点箭头圆圈
        if last_visible is not None:
            cv2.circle(frame, last_visible, 4, color, -1)

    def draw_speed_badge(
        self,
        frame: np.ndarray,
        box_xyxy: Tuple[float, float, float, float],
        reading: SpeedReading,
        conflict: Optional[ConflictAssessment] = None,
    ):
        """在目标检测框上方绘制测速与状态徽章"""
        x1, y1, x2, y2 = map(int, box_xyxy)

        # 确定框体与文字颜色
        if conflict and conflict.risk_level == RiskLevel.LEVEL_3_CRITICAL:
            color = self.COLOR_CRITICAL
            label = f"ID:{reading.track_id} {reading.category.upper()} | {reading.speed_kmh:.1f} km/h [COLLISION ALERT!]"
        elif conflict and conflict.risk_level == RiskLevel.LEVEL_2_CAUTION:
            color = self.COLOR_CAUTION
            label = f"ID:{reading.track_id} | {reading.speed_kmh:.1f} km/h [CAUTION]"
        elif reading.is_speeding:
            color = self.COLOR_SPEEDING
            label = f"ID:{reading.track_id} | {reading.speed_kmh:.1f} km/h [SPEEDING!]"
        else:
            color = self.COLOR_SAFE
            label = f"ID:{reading.track_id} | {reading.speed_kmh:.1f} km/h"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制文本背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def draw_hud_banner(
        self, frame: np.ndarray, conflicts: List[ConflictAssessment]
    ):
        """在画面顶部渲染全息状态与预警看板 (HUD Banner)"""
        h, w = frame.shape[:2]

        # 确定当前帧最高风险等级
        max_level = RiskLevel.LEVEL_0_SAFE
        primary_conflict: Optional[ConflictAssessment] = None
        for c in conflicts:
            if c.risk_level > max_level:
                max_level = c.risk_level
                primary_conflict = c

        # 头部横幅尺寸
        banner_h = 48
        overlay = frame.copy()

        if max_level == RiskLevel.LEVEL_3_CRITICAL:
            bg_color = (0, 0, 180)
            banner_text = f"CRITICAL CONFLICT ALERT: Veh #{primary_conflict.veh_track_id} <-> Ped #{primary_conflict.ped_track_id} | TTC: {primary_conflict.ttc_seconds:.1f}s | BRAKE NOW!"
        elif max_level == RiskLevel.LEVEL_2_CAUTION:
            bg_color = (0, 120, 220)
            banner_text = f"CAUTION: Trajectory Convergence (Veh #{primary_conflict.veh_track_id} <-> Ped #{primary_conflict.ped_track_id} | TTC: {primary_conflict.ttc_seconds:.1f}s)"
        elif max_level == RiskLevel.LEVEL_1_SPEEDING:
            bg_color = (0, 180, 210)
            banner_text = f"SPEED LIMIT WARNING: Veh #{primary_conflict.veh_track_id} @ {primary_conflict.veh_speed_kmh:.1f} km/h (Limit: {self.speed_limit_kmh:g} km/h)"
        else:
            bg_color = (40, 140, 40)
            banner_text = f"CAMPUS ROAD TRAFFIC MONITOR: Normal Flow | Speed Limit: {self.speed_limit_kmh:g} km/h | Safe"

        cv2.rectangle(overlay, (0, 0), (w, banner_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(
            frame,
            banner_text,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def draw_conflict_link(
        self, frame: np.ndarray, conflict: ConflictAssessment, reading_map: Dict[int, SpeedReading]
    ):
        """在冲突车辆与行人之间连线，并在预测碰撞点画出靶标圈"""
        if conflict.risk_level < RiskLevel.LEVEL_2_CAUTION or conflict.ped_track_id is None:
            return

        veh_r = reading_map.get(conflict.veh_track_id)
        ped_r = reading_map.get(conflict.ped_track_id)
        if not veh_r or not ped_r:
            return

        # 车辆与行人图像底边中心点
        u_v, v_v = self.calibrator.ground_to_pixel(veh_r.x_m, veh_r.y_m)
        u_p, v_p = self.calibrator.ground_to_pixel(ped_r.x_m, ped_r.y_m)

        color = (
            self.COLOR_CRITICAL
            if conflict.risk_level == RiskLevel.LEVEL_3_CRITICAL
            else self.COLOR_CAUTION
        )

        # 连虚线
        cv2.line(frame, (u_v, v_v), (u_p, v_p), color, 2, cv2.LINE_AA)

        # 预测交点投射到图像并画警示光圈
        if conflict.conflict_ground_xy:
            cx, cy = conflict.conflict_ground_xy
            cu, cv_pt = self.calibrator.ground_to_pixel(cx, cy)
            if 0 <= cu < frame.shape[1] and 0 <= cv_pt < frame.shape[0]:
                cv2.circle(frame, (cu, cv_pt), 14, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (cu, cv_pt), 4, color, -1)
                cv2.putText(
                    frame,
                    f"TTC {conflict.ttc_seconds:.1f}s",
                    (cu + 8, cv_pt - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: list,
        speed_readings: list[SpeedReading] | dict[int, SpeedReading],
        conflicts: list[ConflictAssessment],
    ) -> np.ndarray:
        """一站式将标定面、未来轨迹、速度徽章、冲突连线及顶部 HUD 渲染到图像"""
        render_frame = frame.copy()
        self.draw_calibrated_ground_zone(render_frame)

        if isinstance(speed_readings, dict):
            reading_map = speed_readings
            readings_list = list(speed_readings.values())
        else:
            reading_map = {r.track_id: r for r in speed_readings}
            readings_list = speed_readings

        conflict_map, ped_conflict_map = build_conflict_maps(conflicts)

        for r in readings_list:
            c = conflict_map.get(r.track_id) or ped_conflict_map.get(r.track_id)
            self.draw_future_trajectory(render_frame, r, color=trajectory_color(c))

        for t in tracks:
            tid = getattr(t, "track_id", None)
            if tid is None:
                continue
            r = reading_map.get(tid)
            if r:
                c = conflict_map.get(tid) or ped_conflict_map.get(tid)
                bbox = getattr(t, "bbox_xyxy", None)
                if bbox is not None:
                    self.draw_speed_badge(render_frame, bbox, r, c)

        for c in conflicts:
            self.draw_conflict_link(render_frame, c, reading_map)

        self.draw_hud_banner(render_frame, conflicts)
        return render_frame

