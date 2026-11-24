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

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration import RoadCalibrator
from .conflict import ConflictAssessment, RiskLevel, build_conflict_maps
from .speed import SpeedReading


def _is_finite_real(value) -> bool:
    """判断值是否为有限实数（显式排除 bool 与 np.bool_）。"""
    if value is None or type(value) is bool or isinstance(value, np.bool_):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return False


def _sanitize_finite_or_zero(value) -> float:
    """将可选物理量（如加速度）过滤为有限 float，无效值安全回退 0.0。"""
    return float(value) if _is_finite_real(value) else 0.0


def _resolve_heading_rad(reading) -> float:
    """解析车辆航向角（弧度）：优先使用有效 heading_deg，无效时按速度矢量或 0 回退。"""
    raw_heading = getattr(reading, "heading_deg", None)
    if _is_finite_real(raw_heading):
        return math.radians(float(raw_heading))
    vx = _sanitize_finite_or_zero(getattr(reading, "vx_ms", 0.0))
    vy = _sanitize_finite_or_zero(getattr(reading, "vy_ms", 0.0))
    # 静止或微速噪声下 atan2 方向不稳定，回退默认 0 弧度。
    if math.hypot(vx, vy) > 1e-6:
        return math.atan2(vy, vx)
    return 0.0


def _conflict_target_vru_kind(conflict: Optional[ConflictAssessment]) -> Optional[str]:
    """当冲突对象非传统行人（如骑行者 VRU）时返回其类型名，否则返回 None。"""
    if conflict is None:
        return None
    raw_kind = getattr(conflict, "target_kind", "pedestrian")
    if isinstance(raw_kind, str):
        clean = raw_kind.strip()
        if clean and clean != "pedestrian":
            return clean
    return None


def _format_ssm_summary(conflict: Optional[ConflictAssessment]) -> str:
    """提取并格式化可选 SSM 替代安全指标为紧凑单行摘要；全 None 时返回空字符串。"""
    if conflict is None:
        return ""
    parts: List[str] = []
    mttc = getattr(conflict, "mttc_seconds", None)
    if _is_finite_real(mttc):
        parts.append(f"MTTC:{float(mttc):.1f}s")
    drac = getattr(conflict, "drac_ms2", None)
    if _is_finite_real(drac):
        parts.append(f"DRAC:{float(drac):.1f}")
    pet = getattr(conflict, "pet_seconds", None)
    if _is_finite_real(pet):
        parts.append(f"PET:{float(pet):.1f}s")
    prob = getattr(conflict, "collision_prob", None)
    if _is_finite_real(prob):
        pct = int(round(max(0.0, min(1.0, float(prob))) * 100.0))
        parts.append(f"P:{pct}%")
    return "  ".join(parts)


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

    def _project_ground_point(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """优先调用 ground_to_pixel_checked 投射地面点，缺失时安全回退 ground_to_pixel。"""
        if not (_is_finite_real(x) and _is_finite_real(y)):
            return None
        fx, fy = float(x), float(y)
        # 优先使用带相机前方检查与地平线溢出防护的接口，防止折返画到天空或触发 OpenCV int32 溢出；
        # 同时也兼容仅提供旧版 ground_to_pixel() 的自定义/Mock 标定对象。
        checked_fn = getattr(self.calibrator, "ground_to_pixel_checked", None)
        if callable(checked_fn):
            pt = checked_fn(fx, fy)
            if pt is None or not (_is_finite_real(pt[0]) and _is_finite_real(pt[1])):
                return None
            u, v = int(pt[0]), int(pt[1])
            if abs(u) > 1_000_000 or abs(v) > 1_000_000:
                return None
            return (u, v)

        pt = self.calibrator.ground_to_pixel(fx, fy)
        if pt is None or not (_is_finite_real(pt[0]) and _is_finite_real(pt[1])):
            return None
        u, v = int(pt[0]), int(pt[1])
        # 旧版接口把无效点和负坐标都编码成 (-1, -1)，这里统一拦截。
        if u < 0 or v < 0 or abs(u) > 1_000_000 or abs(v) > 1_000_000:
            return None
        return (u, v)

    def draw_calibrated_ground_zone(self, frame: np.ndarray, alpha: float = 0.25):
        """在路面上半透明叠加渲染受控标定多边形区域与路侧停车区"""
        pts = np.array(self.calibrator.image_points, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (240, 180, 0))
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 220, 255), thickness=2)

        for zone in getattr(self.calibrator, "parking_zones", ()):
            # 只要停车区在相机前方且未近地平线溢出就照常绘制（允许局部越过左/上画幅边界）；
            # 若任意顶点投射为 None（落在地平线后方或严重溢出），整块跳过以防几何错乱。
            px_pts = [self._project_ground_point(gx, gy) for gx, gy in zone]
            if any(pt is None for pt in px_pts):
                continue
            arr = np.array(px_pts, dtype=np.int32)
            cv2.polylines(frame, [arr], isClosed=True, color=(180, 180, 80), thickness=2)
            label_x = max(0, min(frame.shape[1] - 60, px_pts[0][0]))
            label_y = max(16, min(frame.shape[0] - 4, px_pts[0][1] - 4))
            cv2.putText(
                frame, "PARKING", (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 80), 1, cv2.LINE_AA,
            )

    def draw_future_trajectory(
        self,
        frame: np.ndarray,
        reading: SpeedReading,
        horizon_s: float = 2.5,
        steps: int = 5,
        color: Tuple[int, int, int] = (0, 255, 0),
    ):
        """在地面物理平面上外推并投射未来运动轨迹虚线（支持二阶加速度外推）"""
        if not reading.is_reliable or reading.speed_ms < 0.3:
            return
        if not (
            _is_finite_real(reading.x_m)
            and _is_finite_real(reading.y_m)
            and _is_finite_real(reading.vx_ms)
            and _is_finite_real(reading.vy_ms)
        ):
            return

        x0 = float(reading.x_m)
        y0 = float(reading.y_m)
        vx = float(reading.vx_ms)
        vy = float(reading.vy_ms)
        # 仅在提供有限加速度（如 Kalman 模式）时叠加 0.5*a*t^2；EMA 或缺失/无效值退化为 0.0。
        ax_ms2 = _sanitize_finite_or_zero(getattr(reading, "ax_ms2", 0.0))
        ay_ms2 = _sanitize_finite_or_zero(getattr(reading, "ay_ms2", 0.0))

        dt = horizon_s / steps
        # 出画点将轨迹拆成独立线段，避免跨越不可见区域画出弦线
        segments: List[List[Tuple[int, int]]] = []
        current: List[Tuple[int, int]] = []
        last_visible: Optional[Tuple[int, int]] = None
        h, w = frame.shape[:2]
        for i in range(steps + 1):
            t = i * dt
            half_t2 = 0.5 * t * t
            future_x = x0 + vx * t + ax_ms2 * half_t2
            future_y = y0 + vy * t + ay_ms2 * half_t2
            pt = self._project_ground_point(future_x, future_y)
            if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
                current.append(pt)
                last_visible = pt
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

    def draw_vehicle_obb(
        self,
        frame: np.ndarray,
        reading: SpeedReading,
        length_m: float,
        width_m: float,
        color: Tuple[int, int, int] = (0, 220, 180),
    ):
        """依据车辆接地中心与航向角绘制地面旋转包围盒 (OBB) 与车头朝向指示线。"""
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return
        if not (
            _is_finite_real(getattr(reading, "x_m", None))
            and _is_finite_real(getattr(reading, "y_m", None))
            and _is_finite_real(length_m)
            and _is_finite_real(width_m)
        ):
            return
        length_f = float(length_m)
        width_f = float(width_m)
        if length_f <= 0.0 or width_f <= 0.0:
            return

        cx = float(reading.x_m)
        cy = float(reading.y_m)
        heading_rad = _resolve_heading_rad(reading)
        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        half_l = 0.5 * length_f
        half_w = 0.5 * width_f

        # 车体局部基向量：前向 (cos_h, sin_h)，左法向 (-sin_h, cos_h)
        corners_ground = (
            (cx + half_l * cos_h - half_w * sin_h, cy + half_l * sin_h + half_w * cos_h),
            (cx + half_l * cos_h + half_w * sin_h, cy + half_l * sin_h - half_w * cos_h),
            (cx - half_l * cos_h + half_w * sin_h, cy - half_l * sin_h - half_w * cos_h),
            (cx - half_l * cos_h - half_w * sin_h, cy - half_l * sin_h + half_w * cos_h),
        )
        front_mid_ground = (cx + half_l * cos_h, cy + half_l * sin_h)

        corner_pts = [self._project_ground_point(gx, gy) for gx, gy in corners_ground]
        center_pt = self._project_ground_point(cx, cy)
        front_mid_pt = self._project_ground_point(*front_mid_ground)
        all_pts = [*corner_pts, center_pt, front_mid_pt]
        if any(pt is None for pt in all_pts):
            return

        h, w = frame.shape[:2]
        # 允许车体角点局部擦边出画，但拦截落点严重越界或近地平线畸变。
        if any(pt[0] < -w or pt[0] > 2 * w or pt[1] < -h or pt[1] > 2 * h for pt in all_pts):
            return

        poly = np.array(corner_pts, dtype=np.int32)
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        if center_pt != front_mid_pt:
            cv2.arrowedLine(
                frame,
                center_pt,
                front_mid_pt,
                color,
                2,
                line_type=cv2.LINE_AA,
                tipLength=0.3,
            )

    def draw_speed_badge(
        self,
        frame: np.ndarray,
        box_xyxy: Tuple[float, float, float, float],
        reading: SpeedReading,
        conflict: Optional[ConflictAssessment] = None,
        event_id: Optional[int] = None,
    ):
        """在目标检测框上方绘制测速与状态徽章"""
        x1, y1, x2, y2 = map(int, box_xyxy)

        vru_kind = _conflict_target_vru_kind(conflict)
        # 确认与未确认告警视觉语义区分：已确认 "!"+E#n，未确认瞬时量 "?"
        mark = "!" if event_id is not None else "?"
        if conflict and conflict.risk_level == RiskLevel.LEVEL_3_CRITICAL:
            color = self.COLOR_CRITICAL
            label = f"ID:{reading.track_id} {reading.category.upper()} | {reading.speed_kmh:.1f} km/h [COLLISION ALERT{mark}]"
        elif conflict and conflict.risk_level == RiskLevel.LEVEL_2_CAUTION:
            color = self.COLOR_CAUTION
            label = f"ID:{reading.track_id} | {reading.speed_kmh:.1f} km/h [CAUTION{mark}]"
        elif reading.is_speeding or (
            conflict is not None and conflict.risk_level == RiskLevel.LEVEL_1_SPEEDING
        ):
            color = self.COLOR_SPEEDING
            label = f"ID:{reading.track_id} | {reading.speed_kmh:.1f} km/h [SPEEDING{mark}]"
        else:
            color = self.COLOR_SAFE
            label = f"ID:{reading.track_id} | {reading.speed_kmh:.1f} km/h"

        # 涉及非行人弱势交通参与者（如骑行者）时标注 [VRU]。
        if vru_kind is not None:
            label += " [VRU]"

        # 已通过事件状态机确认的告警携带 event_id，与未确认瞬时帧区分
        if event_id is not None:
            label += f" E#{event_id}"

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
        self,
        frame: np.ndarray,
        conflicts: List[ConflictAssessment],
        event_ids: Optional[Dict[Tuple[int, Optional[int]], int]] = None,
    ):
        """在画面顶部渲染全息状态与预警看板 (HUD Banner)

        event_ids: 事件状态机给出的 {(veh_id, ped_id): event_id}；
        提供时按 provisional（未确认瞬时帧）/confirmed（已确认事件）区分标注"""
        h, w = frame.shape[:2]

        # 确定当前帧最高风险等级：依次按 风险级别 -> 是否已通过事件确认 -> 最小 TTC -> 最小 PET 排序，
        # 确保已确认告警不会被同等级、刚冒头的瞬时更小 TTC 闪断掩盖。
        max_level = RiskLevel.LEVEL_0_SAFE
        primary_conflict: Optional[ConflictAssessment] = None
        best_key: Optional[Tuple[int, int, float, float]] = None
        for c in conflicts:
            if c.risk_level <= RiskLevel.LEVEL_0_SAFE:
                continue
            eid = (event_ids or {}).get((c.veh_track_id, c.ped_track_id))
            pet_val = getattr(c, "pet_seconds", None)
            key = (
                int(c.risk_level),
                1 if eid is not None else 0,
                -(c.ttc_seconds if c.ttc_seconds is not None else 999.0),
                -(float(pet_val) if _is_finite_real(pet_val) else 999.0),
            )
            if best_key is None or key > best_key:
                best_key = key
                primary_conflict = c
                max_level = c.risk_level

        ssm_summary = _format_ssm_summary(primary_conflict)
        vru_kind = _conflict_target_vru_kind(primary_conflict)
        if primary_conflict is not None and vru_kind is not None:
            target_desc = f"[VRU] {vru_kind} #{primary_conflict.ped_track_id}"
        elif primary_conflict is not None:
            target_desc = f"Ped #{primary_conflict.ped_track_id}"
        else:
            target_desc = ""

        # 头部横幅尺寸：存在 SSM 指标时扩展至双行以展示紧凑汇总，未启用时保持原 48px 布局。
        banner_h = 74 if ssm_summary else 48
        overlay = frame.copy()

        if max_level == RiskLevel.LEVEL_3_CRITICAL:
            bg_color = (0, 0, 180)
            ttc_str = f"{primary_conflict.ttc_seconds:.1f}s" if primary_conflict.ttc_seconds is not None else "N/A"
            banner_text = f"CRITICAL CONFLICT ALERT: Veh #{primary_conflict.veh_track_id} <-> {target_desc} | TTC: {ttc_str} | BRAKE NOW!"
        elif max_level == RiskLevel.LEVEL_2_CAUTION:
            bg_color = (0, 120, 220)
            ttc_str = f"{primary_conflict.ttc_seconds:.1f}s" if primary_conflict.ttc_seconds is not None else "N/A"
            banner_text = f"CAUTION: Trajectory Convergence (Veh #{primary_conflict.veh_track_id} <-> {target_desc} | TTC: {ttc_str})"
        elif max_level == RiskLevel.LEVEL_1_SPEEDING:
            bg_color = (0, 180, 210)
            banner_text = f"SPEED LIMIT WARNING: Veh #{primary_conflict.veh_track_id} @ {primary_conflict.veh_speed_kmh:.1f} km/h (Limit: {self.speed_limit_kmh:g} km/h)"
            if vru_kind is not None:
                banner_text += " [VRU]"
        else:
            bg_color = (40, 140, 40)
            banner_text = f"CAMPUS ROAD TRAFFIC MONITOR: Normal Flow | Speed Limit: {self.speed_limit_kmh:g} km/h | Safe"

        # 区分已通过确认帧数的 confirmed 事件与仍可能被去抖过滤的瞬时帧：
        # 下游声光/GPIO/网络通知应只消费 confirmed（events.jsonl），
        # 画面上的 provisional 前缀提示该告警可能闪断
        if max_level > RiskLevel.LEVEL_0_SAFE and primary_conflict is not None:
            eid = (event_ids or {}).get(
                (primary_conflict.veh_track_id, primary_conflict.ped_track_id)
            )
            prefix = f"[EVT#{eid} CONFIRMED] " if eid is not None else "[PROVISIONAL] "
            banner_text = prefix + banner_text

        cv2.rectangle(overlay, (0, 0), (w, min(h, banner_h)), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        main_y = 30 if ssm_summary else 32
        cv2.putText(
            frame,
            banner_text,
            (16, main_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if ssm_summary:
            cv2.putText(
                frame,
                ssm_summary,
                (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 245, 255),
                1,
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
        veh_pt = self._project_ground_point(veh_r.x_m, veh_r.y_m)
        ped_pt = self._project_ground_point(ped_r.x_m, ped_r.y_m)

        color = (
            self.COLOR_CRITICAL
            if conflict.risk_level == RiskLevel.LEVEL_3_CRITICAL
            else self.COLOR_CAUTION
        )

        # 连虚线
        if veh_pt is not None and ped_pt is not None:
            cv2.line(frame, veh_pt, ped_pt, color, 2, cv2.LINE_AA)

        # 预测交点投射到图像并画警示光圈
        if conflict.conflict_ground_xy:
            cx, cy = conflict.conflict_ground_xy
            conflict_pt = self._project_ground_point(cx, cy)
            if (
                conflict_pt is not None
                and 0 <= conflict_pt[0] < frame.shape[1]
                and 0 <= conflict_pt[1] < frame.shape[0]
            ):
                cu, cv_pt = conflict_pt
                cv2.circle(frame, (cu, cv_pt), 14, color, 2, cv2.LINE_AA)
                cv2.circle(frame, (cu, cv_pt), 4, color, -1)
                ttc_label = (
                    f"TTC {conflict.ttc_seconds:.1f}s"
                    if conflict.ttc_seconds is not None
                    else "TTC N/A"
                )
                cv2.putText(
                    frame,
                    ttc_label,
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

