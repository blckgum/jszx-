# -*- coding: utf-8 -*-
"""
车辆—行人时空轨迹冲突预测与 TTC 决策引擎 (Trajectory Conflict Analysis & TTC Decision Engine)
=============================================================================
利用二阶运动学与等速外推模型 (Constant Velocity Model, CVM)，
在未来短时间视界 (0 ~ 3.5 秒) 内求解最小相对距离到达时刻 (t_cpa) 与极小间距 D_min，
并与校园 30 km/h 独立限速机制结合，构建严格的分级预警决策体系。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from .speed import SpeedReading
from .policy import object_policy


class RiskLevel(IntEnum):
    """预警严重等级枚举"""
    LEVEL_0_SAFE = 0       # 绿色：正常受控行驶，无冲突
    LEVEL_1_SPEEDING = 1   # 黄色：车速超过 30 km/h，但当前与行人无冲突轨迹
    LEVEL_2_CAUTION = 2    # 橙色：车辆与行人轨迹汇聚，TTC 介于 1.8s ~ 3.5s
    LEVEL_3_CRITICAL = 3   # 红色：紧急危险！TTC <= 1.8s 且预测最小间距小于安全阈值


@dataclass
class ConflictAssessment:
    """车辆—行人冲突评估输出结果"""
    veh_track_id: int
    ped_track_id: Optional[int]
    risk_level: RiskLevel
    risk_desc: str
    veh_speed_kmh: float
    ped_speed_ms: float
    current_distance_m: float
    min_distance_m: float          # 预测未来最小距离 D_min (米)
    ttc_seconds: Optional[float]   # 碰撞时间 TTC (秒)
    conflict_ground_xy: Optional[Tuple[float, float]]  # 预测冲突点地面坐标 (米)
    vehicle_kind: str = "motor_vehicle"


def _conflict_sort_key(c: ConflictAssessment):
    """冲突项比较键：先按风险等级，同等级按 TTC 越小越紧急"""
    return (int(c.risk_level), -(c.ttc_seconds or 999.0))


def build_conflict_maps(
    conflicts: List[ConflictAssessment],
) -> Tuple[Dict[int, ConflictAssessment], Dict[int, ConflictAssessment]]:
    """构建车辆侧与行人侧的 track_id -> 最高风险冲突映射；
    同一行人被多车关联时保留最高 risk_level、同级取最小 TTC"""
    veh_map = {c.veh_track_id: c for c in conflicts}
    ped_map: Dict[int, ConflictAssessment] = {}
    for c in conflicts:
        if c.ped_track_id is None:
            continue
        cur = ped_map.get(c.ped_track_id)
        if cur is None or _conflict_sort_key(c) > _conflict_sort_key(cur):
            ped_map[c.ped_track_id] = c
    return veh_map, ped_map


class ConflictAnalyzer:
    """车辆—行人时空交互冲突研判与 TTC 决策核心"""

    def __init__(
        self,
        speed_limit_kmh: float = 30.0,
        safe_radius_veh_m: float = 2.2,
        safe_radius_ped_m: float = 0.8,
        ttc_critical_s: float = 1.8,
        ttc_caution_s: float = 3.5,
        horizon_s: float = 3.5,
    ):
        if not math.isfinite(speed_limit_kmh) or speed_limit_kmh <= 0:
            raise ValueError("speed_limit_kmh 必须为有限正数")
        if not (math.isfinite(safe_radius_veh_m) and safe_radius_veh_m > 0
                and math.isfinite(safe_radius_ped_m) and safe_radius_ped_m > 0):
            raise ValueError("安全半径必须为有限正数")
        if not (math.isfinite(ttc_critical_s) and math.isfinite(ttc_caution_s)
                and math.isfinite(horizon_s)
                and 0 < ttc_critical_s <= ttc_caution_s <= horizon_s):
            raise ValueError("TTC 阈值需满足 0 < ttc_critical <= ttc_caution <= horizon")
        self.speed_limit_kmh = speed_limit_kmh
        self.safe_radius_veh_m = safe_radius_veh_m
        self.safe_radius_ped_m = safe_radius_ped_m
        self.critical_dist_m = safe_radius_veh_m + safe_radius_ped_m  # 默认机动车 3.0 米
        self.ttc_critical_s = ttc_critical_s
        self.ttc_caution_s = ttc_caution_s
        self.horizon_s = horizon_s

    def evaluate_pair(
        self, veh: SpeedReading, ped: SpeedReading
    ) -> Optional[ConflictAssessment]:
        """评估单一机动车与单一横穿行人之间的潜在几何冲突"""
        dx = ped.x_m - veh.x_m
        dy = ped.y_m - veh.y_m
        current_dist = math.hypot(dx, dy)

        # 超过 35 米开外不构成即时校园道路冲突威胁
        if current_dist > 35.0:
            return None

        # 持续静止且速度估计已收敛的车辆（泊车/停靠）不参与碰撞研判：
        # 行人走近停靠车辆不构成交通冲突，否则 CVM 外推会对路边停靠目标
        # 持续误报；缓慢蠕行或速度未收敛的车辆不满足该条件，仍正常评估。
        # is_reliable 兜底：--parked-frames 1 或低 FPS 时静止标记可能在
        # 收敛前置位，未收敛目标一律保守参与研判，不做豁免
        if veh.is_stationary and veh.is_reliable:
            return None

        vehicle_policy = object_policy(veh.object_kind, veh.category)
        veh_radius_m = (
            self.safe_radius_veh_m
            if vehicle_policy.kind == "motor_vehicle"
            else vehicle_policy.safe_radius_veh_m
        )
        critical_dist_m = veh_radius_m + self.safe_radius_ped_m

        # 相对速度矢量：v_rel = v_ped - v_veh
        dvx = ped.vx_ms - veh.vx_ms
        dvy = ped.vy_ms - veh.vy_ms
        rel_speed_sq = dvx * dvx + dvy * dvy

        # 相对运动过微（几乎无相对速度差）
        if rel_speed_sq < 1e-4:
            return None

        # 判定两者是否处于汇聚靠近态：r · v_rel < 0
        dot_product = dx * dvx + dy * dvy
        if dot_product >= 0:
            # 内积 >= 0 说明两者正在背向运动，距离正在拉大，TTC 视为无穷大安全
            return None

        # 计算最接近点时间 t_cpa (即 TTC)
        t_cpa = -dot_product / rel_speed_sq

        # 只关注未来预测视界内的有效冲突
        if t_cpa <= 0 or t_cpa > self.horizon_s:
            return None

        # 计算在 t_cpa 时刻两者的预测间距 D_min
        rx_cpa = dx + dvx * t_cpa
        ry_cpa = dy + dvy * t_cpa
        min_dist = math.hypot(rx_cpa, ry_cpa)

        # 预测空间交点坐标（基于车辆外推位置）
        conflict_x = veh.x_m + veh.vx_ms * t_cpa
        conflict_y = veh.y_m + veh.vy_ms * t_cpa

        # 分级风险研判
        if min_dist <= critical_dist_m:
            if t_cpa <= self.ttc_critical_s:
                level = RiskLevel.LEVEL_3_CRITICAL
                desc = f"紧急碰撞危险! (TTC={t_cpa:.1f}s, 预计间距={min_dist:.1f}m)"
            else:
                level = RiskLevel.LEVEL_2_CAUTION
                desc = f"注意减速礼让行人 (TTC={t_cpa:.1f}s, 轨迹汇聚)"
        elif min_dist <= critical_dist_m * 1.5 and t_cpa <= self.ttc_caution_s:
            level = RiskLevel.LEVEL_2_CAUTION
            desc = f"注意横穿行人 (预计间距={min_dist:.1f}m)"
        else:
            return None

        return ConflictAssessment(
            veh_track_id=veh.track_id,
            ped_track_id=ped.track_id,
            risk_level=level,
            risk_desc=desc,
            veh_speed_kmh=veh.speed_kmh,
            ped_speed_ms=ped.speed_ms,
            current_distance_m=current_dist,
            min_distance_m=min_dist,
            ttc_seconds=t_cpa,
            conflict_ground_xy=(conflict_x, conflict_y),
            vehicle_kind=vehicle_policy.kind,
        )

    def analyze_frame(
        self, readings: List[SpeedReading]
    ) -> List[ConflictAssessment]:
        """全场景多目标批量冲突评估"""
        vehicles = [r for r in readings if r.category == "vehicle"]
        pedestrians = [r for r in readings if r.category == "pedestrian"]

        assessments: List[ConflictAssessment] = []

        # 遍历所有车辆
        for veh in vehicles:
            highest_conflict: Optional[ConflictAssessment] = None

            # 评估与视野内所有行人的交互
            for ped in pedestrians:
                conflict = self.evaluate_pair(veh, ped)
                if conflict is not None:
                    if (
                        highest_conflict is None
                        or conflict.risk_level > highest_conflict.risk_level
                        or (
                            conflict.risk_level == highest_conflict.risk_level
                            and (conflict.ttc_seconds or 999) < (highest_conflict.ttc_seconds or 999)
                        )
                    ):
                        highest_conflict = conflict

            veh_kind = object_policy(veh.object_kind, veh.category).kind
            if highest_conflict is not None:
                assessments.append(highest_conflict)
            elif veh.category == "vehicle" and veh.speed_kmh > self.speed_limit_kmh:
                # 用本分析器自身的限速值重判而非信任 reading.is_speeding：
                # 独立使用 ConflictAnalyzer 时两侧限速参数可能不一致
                # 独立超速判定分支：车辆超速 30 km/h，但前方暂无相交行人
                assessments.append(
                    ConflictAssessment(
                        veh_track_id=veh.track_id,
                        ped_track_id=None,
                        risk_level=RiskLevel.LEVEL_1_SPEEDING,
                        risk_desc=f"超速提醒: 时速 {veh.speed_kmh:.1f} km/h (限速 {self.speed_limit_kmh:g} km/h)",
                        veh_speed_kmh=veh.speed_kmh,
                        ped_speed_ms=0.0,
                        current_distance_m=0.0,
                        min_distance_m=0.0,
                        ttc_seconds=None,
                        conflict_ground_xy=None,
                        vehicle_kind=veh_kind,
                    )
                )
            else:
                # 正常安全通行
                assessments.append(
                    ConflictAssessment(
                        veh_track_id=veh.track_id,
                        ped_track_id=None,
                        risk_level=RiskLevel.LEVEL_0_SAFE,
                        risk_desc=f"正常行驶 ({veh.speed_kmh:.1f} km/h)",
                        veh_speed_kmh=veh.speed_kmh,
                        ped_speed_ms=0.0,
                        current_distance_m=0.0,
                        min_distance_m=0.0,
                        ttc_seconds=None,
                        conflict_ground_xy=None,
                        vehicle_kind=veh_kind,
                    )
                )

        return assessments
