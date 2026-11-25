# -*- coding: utf-8 -*-
"""
车辆—行人时空轨迹冲突预测与 TTC 决策引擎 (Trajectory Conflict Analysis & TTC Decision Engine)
=============================================================================
支持两种冲突研判模型：
1. cvm_circle（默认基线）：二阶等速圆包络外推，求解 t_cpa 与 D_min；
2. obb_ssm：车体旋转矩形 (OBB) 扫掠 TTC + MTTC + DRAC + PET + 碰撞概率综合研判，
   并与校园 30 km/h 独立限速机制结合，构建严格的分级预警决策体系。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .policy import ObjectPolicy, object_policy
from .speed import SpeedReading
from .ssm import (
    PETTracker,
    compute_drac,
    compute_mttc,
    compute_obb_accel_entry_time,
    compute_obb_swept_ttc,
    estimate_collision_probability,
)


VRU_CYCLIST_KINDS = ("bicycle", "motorcycle")


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
    min_distance_m: float          # 预测未来最小距离 D_min 或最小净距 (米)
    ttc_seconds: Optional[float]   # 碰撞时间 TTC (秒)
    conflict_ground_xy: Optional[Tuple[float, float]]  # 预测冲突点地面坐标 (米)
    vehicle_kind: str = "motor_vehicle"
    mttc_seconds: Optional[float] = None
    drac_ms2: Optional[float] = None
    pet_seconds: Optional[float] = None
    collision_prob: Optional[float] = None
    conflict_model: str = "cvm_circle"
    target_kind: str = "pedestrian"


def _is_finite_number(value) -> bool:
    # 显式排除 bool，防止 True/False 作为 int 子类通过参数校验。
    if type(value) is bool or isinstance(value, np.bool_):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return False


def _conflict_sort_key(c: ConflictAssessment):
    """冲突项比较键：先按风险等级，同等级先按 TTC、再按 PET 越小越紧急"""
    # 用 is not None 判断而非布尔真值，确保 ttc_seconds/pet_seconds == 0.0 不会被误回退成 999.0。
    return (
        int(c.risk_level),
        -(c.ttc_seconds if c.ttc_seconds is not None else 999.0),
        -(c.pet_seconds if c.pet_seconds is not None else 999.0),
    )


def build_conflict_maps(
    conflicts: List[ConflictAssessment],
) -> Tuple[Dict[int, ConflictAssessment], Dict[int, ConflictAssessment]]:
    """构建车辆侧与行人侧的 track_id -> 最高风险冲突映射；
    同一行人被多车关联时保留最高 risk_level、同级取最小 TTC"""
    veh_map: Dict[int, ConflictAssessment] = {}
    ped_map: Dict[int, ConflictAssessment] = {}
    for c in conflicts:
        cur_v = veh_map.get(c.veh_track_id)
        if cur_v is None or _conflict_sort_key(c) > _conflict_sort_key(cur_v):
            veh_map[c.veh_track_id] = c
        if c.ped_track_id is None:
            continue
        cur = ped_map.get(c.ped_track_id)
        if cur is None or _conflict_sort_key(c) > _conflict_sort_key(cur):
            ped_map[c.ped_track_id] = c
    # 若骑行者同时作为车辆侧与被撞目标侧出现且被撞风险更高，移除低优车辆侧条目以免下游 or 遮蔽高风险预警。
    for tid, ped_conflict in ped_map.items():
        veh_conflict = veh_map.get(tid)
        if veh_conflict is not None and _conflict_sort_key(ped_conflict) > _conflict_sort_key(veh_conflict):
            del veh_map[tid]
    return veh_map, ped_map


class ConflictAnalyzer:
    """车辆—行人时空交互冲突研判与 TTC/SSM 决策核心"""

    def __init__(
        self,
        speed_limit_kmh: float = 30.0,
        safe_radius_veh_m: float = 2.2,
        safe_radius_ped_m: float = 0.8,
        ttc_critical_s: float = 1.8,
        ttc_caution_s: float = 3.5,
        horizon_s: float = 3.5,
        object_policies: dict[str, ObjectPolicy] | None = None,
        parking_zone_max_speed_kmh: float = 5.0,
        conflict_model: str = "cvm_circle",
        drac_critical_ms2: float = 3.4,
        drac_caution_ms2: float = 2.0,
        pet_critical_s: float = 1.0,
        pet_caution_s: float = 2.0,
        include_cyclist_vru: bool = False,
    ):
        if not _is_finite_number(speed_limit_kmh) or speed_limit_kmh <= 0:
            raise ValueError("speed_limit_kmh 必须为有限正数")
        if not (
            _is_finite_number(safe_radius_veh_m)
            and safe_radius_veh_m > 0
            and _is_finite_number(safe_radius_ped_m)
            and safe_radius_ped_m > 0
        ):
            raise ValueError("安全半径必须为有限正数")
        if not (
            _is_finite_number(ttc_critical_s)
            and _is_finite_number(ttc_caution_s)
            and _is_finite_number(horizon_s)
            and 0 < ttc_critical_s <= ttc_caution_s <= horizon_s
        ):
            raise ValueError("TTC 阈值需满足 0 < ttc_critical <= ttc_caution <= horizon")
        if not _is_finite_number(parking_zone_max_speed_kmh) or parking_zone_max_speed_kmh < 0:
            raise ValueError("parking_zone_max_speed_kmh 必须为有限非负数")
        if not isinstance(conflict_model, str) or conflict_model not in ("cvm_circle", "obb_ssm"):
            raise ValueError("conflict_model 必须是 'cvm_circle' 或 'obb_ssm'")
        if not (
            _is_finite_number(drac_caution_ms2)
            and _is_finite_number(drac_critical_ms2)
            and 0 < drac_caution_ms2 <= drac_critical_ms2
        ):
            raise ValueError("DRAC 阈值需满足 0 < drac_caution_ms2 <= drac_critical_ms2")
        if not (
            _is_finite_number(pet_critical_s)
            and _is_finite_number(pet_caution_s)
            and 0 < pet_critical_s <= pet_caution_s
        ):
            raise ValueError("PET 阈值需满足 0 < pet_critical_s <= pet_caution_s")
        # 严格用精确 type() 校验，避免 int(0/1) 或字符串混入布尔开关。
        if type(include_cyclist_vru) is not bool:
            raise ValueError("include_cyclist_vru 必须是布尔值")

        self.speed_limit_kmh = float(speed_limit_kmh)
        self.safe_radius_veh_m = float(safe_radius_veh_m)
        self.safe_radius_ped_m = float(safe_radius_ped_m)
        self.critical_dist_m = self.safe_radius_veh_m + self.safe_radius_ped_m  # 默认机动车 3.0 米
        self.ttc_critical_s = float(ttc_critical_s)
        self.ttc_caution_s = float(ttc_caution_s)
        self.horizon_s = float(horizon_s)
        self.object_policies = object_policies
        self.parking_zone_max_speed_kmh = float(parking_zone_max_speed_kmh)
        self.conflict_model = conflict_model
        self.drac_critical_ms2 = float(drac_critical_ms2)
        self.drac_caution_ms2 = float(drac_caution_ms2)
        self.pet_critical_s = float(pet_critical_s)
        self.pet_caution_s = float(pet_caution_s)
        self.include_cyclist_vru = include_cyclist_vru
        self.pet_tracker = PETTracker() if self.conflict_model == "obb_ssm" else None

    def reset(self) -> None:
        """重置冲突分析器内部状态（如 PET 网格历史），供复用实例跨运行隔离状态。"""
        if self.pet_tracker is not None:
            self.pet_tracker.reset()

    def _resolve_target_meta(self, ped: SpeedReading) -> Tuple[ObjectPolicy, bool, str, str]:
        """解析弱势目标策略、是否骑行者、落盘 target_kind 及告警称呼。"""
        target_policy = object_policy(ped.object_kind, ped.category, self.object_policies)
        is_cyclist = target_policy.kind in VRU_CYCLIST_KINDS
        target_kind = target_policy.kind if is_cyclist else "pedestrian"
        target_label = "骑行者" if is_cyclist else "行人"
        return target_policy, is_cyclist, target_kind, target_label

    def _is_reading_finite(self, reading: SpeedReading) -> bool:
        """校验参与冲突评估的读数运动学字段是否全部为有限数值（含 obb_ssm 朝向与加速度）。"""
        if not all(
            _is_finite_number(v)
            for v in (
                reading.x_m,
                reading.y_m,
                reading.vx_ms,
                reading.vy_ms,
                reading.speed_ms,
                reading.speed_kmh,
            )
        ):
            return False
        if self.conflict_model == "obb_ssm":
            # obb_ssm 模式下同步校验朝向与加速度，None 按缺失回退默认逻辑兼容，NaN/Inf/bool 则整目标跳过。
            raw_heading = getattr(reading, "heading_deg", None)
            raw_ax = getattr(reading, "ax_ms2", 0.0)
            raw_ay = getattr(reading, "ay_ms2", 0.0)
            heading_chk = 0.0 if raw_heading is None else raw_heading
            ax_chk = 0.0 if raw_ax is None else raw_ax
            ay_chk = 0.0 if raw_ay is None else raw_ay
            if not all(_is_finite_number(v) for v in (heading_chk, ax_chk, ay_chk)):
                return False
        return True

    def _is_vehicle_exempt(self, veh: SpeedReading) -> bool:
        """判断车辆是否因确认静止或停车区内低速挪动而豁免冲突研判。"""
        # is_reliable 兜底：--parked-frames 1 或低 FPS 时静止标记可能在
        # 收敛前置位，未收敛目标一律保守参与研判，不做豁免。
        if veh.is_stationary and veh.is_reliable:
            return True
        if veh.in_parking_zone and veh.is_reliable and veh.speed_kmh <= self.parking_zone_max_speed_kmh:
            return True
        return False

    def _evaluate_pair_cvm(
        self, veh: SpeedReading, ped: SpeedReading
    ) -> Optional[ConflictAssessment]:
        """原有等速圆包络外推 (cvm_circle) 冲突研判逻辑，保持完全向后兼容。"""
        if not all(
            _is_finite_number(v)
            for v in (
                veh.x_m, veh.y_m, veh.vx_ms, veh.vy_ms, veh.speed_kmh,
                ped.x_m, ped.y_m, ped.vx_ms, ped.vy_ms, ped.speed_ms,
            )
        ):
            return None

        dx = ped.x_m - veh.x_m
        dy = ped.y_m - veh.y_m
        current_dist = math.hypot(dx, dy)

        if current_dist > 35.0:
            return None
        if self._is_vehicle_exempt(veh):
            return None

        vehicle_policy = object_policy(veh.object_kind, veh.category, self.object_policies)
        target_policy, is_cyclist, target_kind, target_label = self._resolve_target_meta(ped)
        veh_radius_m = (
            self.safe_radius_veh_m
            if (self.object_policies is None and vehicle_policy.kind == "motor_vehicle")
            else vehicle_policy.safe_radius_veh_m
        )
        # 骑行者 (bicycle/motorcycle) 按其策略半径参与交互，常规行人沿用安全半径配置。
        ped_radius_m = (
            target_policy.safe_radius_veh_m if is_cyclist else self.safe_radius_ped_m
        )
        critical_dist_m = veh_radius_m + ped_radius_m

        dvx = ped.vx_ms - veh.vx_ms
        dvy = ped.vy_ms - veh.vy_ms
        rel_speed_sq = dvx * dvx + dvy * dvy
        dot_product = dx * dvx + dy * dvy

        if current_dist <= critical_dist_m and (rel_speed_sq < 1e-4 or dot_product >= 0):
            # 安全包络当前已经相交且未相对趋近时，直接按 t=0s 触发三级危急告警。
            min_dist = current_dist
            ttc_out = 0.0
            conflict_x = veh.x_m
            conflict_y = veh.y_m
            level = RiskLevel.LEVEL_3_CRITICAL
            crit_prefix = f"紧急碰撞危险({target_label})!" if is_cyclist else "紧急碰撞危险!"
            desc = f"{crit_prefix} (TTC={ttc_out:.1f}s, 预计间距={min_dist:.1f}m)"
        else:
            if rel_speed_sq < 1e-4 or dot_product >= 0:
                return None

            t_cpa = -dot_product / rel_speed_sq
            if t_cpa <= 0:
                return None

            rx_cpa = dx + dvx * t_cpa
            ry_cpa = dy + dvy * t_cpa
            min_dist = math.hypot(rx_cpa, ry_cpa)

            if min_dist <= critical_dist_m:
                # 用相对速度与弦长精确求圆包络首次接触时刻，避免 t_cpa 越出视界时漏报或误判 TTC。
                half_chord = math.sqrt(max(0.0, critical_dist_m * critical_dist_m - min_dist * min_dist))
                t_contact = max(0.0, t_cpa - half_chord / math.sqrt(rel_speed_sq))
                if t_contact > self.horizon_s:
                    return None
                ttc_out = t_contact
                conflict_x = veh.x_m + veh.vx_ms * t_contact
                conflict_y = veh.y_m + veh.vy_ms * t_contact
                if t_contact <= self.ttc_critical_s:
                    level = RiskLevel.LEVEL_3_CRITICAL
                    crit_prefix = f"紧急碰撞危险({target_label})!" if is_cyclist else "紧急碰撞危险!"
                    desc = f"{crit_prefix} (TTC={t_contact:.1f}s, 预计间距={min_dist:.1f}m)"
                elif t_contact <= self.ttc_caution_s:
                    level = RiskLevel.LEVEL_2_CAUTION
                    desc = f"注意减速礼让{target_label} (TTC={t_contact:.1f}s, 轨迹汇聚)"
                else:
                    return None
            elif (
                min_dist <= critical_dist_m * 1.5
                and t_cpa <= self.horizon_s
                and t_cpa <= self.ttc_caution_s
            ):
                # 擦肩近距预警分支中两包络并未相交，TTC 置空并仅在描述中展示预计最小间距。
                ttc_out = None
                conflict_x = veh.x_m + veh.vx_ms * t_cpa
                conflict_y = veh.y_m + veh.vy_ms * t_cpa
                level = RiskLevel.LEVEL_2_CAUTION
                desc = f"注意横穿{target_label} (预计间距={min_dist:.1f}m)"
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
            ttc_seconds=ttc_out,
            conflict_ground_xy=(conflict_x, conflict_y),
            vehicle_kind=vehicle_policy.kind,
            conflict_model="cvm_circle",
            target_kind=target_kind,
        )

    def _evaluate_pair_obb_ssm(
        self, veh: SpeedReading, ped: SpeedReading
    ) -> Optional[ConflictAssessment]:
        """基于旋转矩形扫掠与多维替代安全指标 (OBB + MTTC + DRAC + 概率) 的冲突评估。"""
        raw_veh_ax = getattr(veh, "ax_ms2", 0.0)
        raw_veh_ay = getattr(veh, "ay_ms2", 0.0)
        raw_ped_ax = getattr(ped, "ax_ms2", 0.0)
        raw_ped_ay = getattr(ped, "ay_ms2", 0.0)
        raw_veh_heading = getattr(veh, "heading_deg", None)
        raw_ped_heading = getattr(ped, "heading_deg", None)
        veh_ax = 0.0 if raw_veh_ax is None else raw_veh_ax
        veh_ay = 0.0 if raw_veh_ay is None else raw_veh_ay
        ped_ax = 0.0 if raw_ped_ax is None else raw_ped_ax
        ped_ay = 0.0 if raw_ped_ay is None else raw_ped_ay
        # heading_deg 为 None 时允许回退速度矢量方向，但 NaN/Inf/bool 等无效值必须拦截。
        veh_heading_chk = 0.0 if raw_veh_heading is None else raw_veh_heading
        ped_heading_chk = 0.0 if raw_ped_heading is None else raw_ped_heading

        if not all(
            _is_finite_number(v)
            for v in (
                veh.x_m, veh.y_m, veh.vx_ms, veh.vy_ms, veh.speed_kmh, veh_heading_chk,
                ped.x_m, ped.y_m, ped.vx_ms, ped.vy_ms, ped.speed_ms, ped_heading_chk,
                veh_ax, veh_ay, ped_ax, ped_ay,
            )
        ):
            return None

        dx = float(ped.x_m) - float(veh.x_m)
        dy = float(ped.y_m) - float(veh.y_m)
        current_dist = math.hypot(dx, dy)
        if current_dist > 35.0:
            return None
        if self._is_vehicle_exempt(veh):
            return None

        vehicle_policy = object_policy(veh.object_kind, veh.category, self.object_policies)
        ped_policy, is_cyclist, target_kind, target_label = self._resolve_target_meta(ped)
        veh_length_m = vehicle_policy.length_m
        veh_width_m = vehicle_policy.width_m
        # 骑行者在 OBB-SSM 下使用自身外包半长作为交互半径，常规行人保持既有半径回退。
        if is_cyclist:
            ped_radius_m = 0.5 * max(ped_policy.length_m, ped_policy.width_m)
        else:
            ped_radius_m = (
                self.safe_radius_ped_m
                if abs(self.safe_radius_ped_m - 0.8) > 1e-9
                else 0.5 * max(ped_policy.length_m, ped_policy.width_m)
            )

        veh_vx, veh_vy = float(veh.vx_ms), float(veh.vy_ms)
        ped_vx, ped_vy = float(ped.vx_ms), float(ped.vy_ms)
        # 保留静止/刚起步车辆上游提供的有效 heading_deg，仅在缺失 (None) 时回退速度矢量方向。
        heading_rad = (
            math.radians(float(raw_veh_heading))
            if raw_veh_heading is not None
            else math.atan2(veh_vy, veh_vx)
        )

        obb = compute_obb_swept_ttc(
            veh_xy=(float(veh.x_m), float(veh.y_m)),
            veh_v=(veh_vx, veh_vy),
            heading_rad=heading_rad,
            veh_length_m=veh_length_m,
            veh_width_m=veh_width_m,
            ped_xy=(float(ped.x_m), float(ped.y_m)),
            ped_v=(ped_vx, ped_vy),
            ped_radius_m=ped_radius_m,
            horizon_s=self.horizon_s,
        )

        dvx = ped_vx - veh_vx
        dvy = ped_vy - veh_vy
        dax = float(ped_ax) - float(veh_ax)
        day = float(ped_ay) - float(veh_ay)

        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        hx_unm = 0.5 * veh_length_m + ped_radius_m
        hy_unm = 0.5 * veh_width_m + ped_radius_m

        if current_dist > 1e-9:
            ux = dx / current_dist
            uy = dy / current_dist
            ulx = cos_h * ux + sin_h * uy
            uly = -sin_h * ux + cos_h * uy
            # 沿当前视线方向截取未加裕度 OBB 的支撑半径，兼顾长车身纵向与窄车身横向特性。
            critical_dist_obb = min(
                hx_unm / max(abs(ulx), 1e-9),
                hy_unm / max(abs(uly), 1e-9),
            )
            a_c = -(dax * ux + day * uy)
        else:
            critical_dist_obb = hy_unm
            a_c = 0.0

        mttc = compute_mttc(
            dx, dy, dvx, dvy, dax, day, critical_dist_obb, self.horizon_s
        )
        drac = compute_drac(dx, dy, dvx, dvy, critical_dist_obb)

        # 若等速外推未撞击且存在相对加速度，在 [0, horizon_s] 内精确求解二阶轨迹进入圆角扩展 OBB 的最早时刻。
        accel_entry_t: Optional[float] = None
        if not obb.collides and (dax * dax + day * day) > 1e-6:
            alx = cos_h * dax + sin_h * day
            aly = -sin_h * dax + cos_h * day
            accel_entry_t = compute_obb_accel_entry_time(
                obb.local_pos_xy,
                obb.local_vel_xy,
                (alx, aly),
                obb.expanded_half_extents_xy,
                self.horizon_s,
                corner_radius_m=ped_radius_m,
            )
        accel_enters_obb = accel_entry_t is not None

        if not obb.collides and not accel_enters_obb:
            return None

        # 安全制动降级：已明显减速且刹停距离足够（mttc=None 且二阶未侵入 OBB）同时所需减速度低于注意阈值。
        is_safe_braking = (
            mttc is None
            and not accel_enters_obb
            and a_c < -0.5
            and drac is not None
            and drac < self.drac_caution_ms2
        )
        min_dist_out = 0.0 if accel_enters_obb else obb.min_clearance_m
        if is_safe_braking:
            if obb.collides and obb.t_enter is not None and obb.t_enter <= self.ttc_critical_s:
                level = RiskLevel.LEVEL_2_CAUTION
                t_eff: Optional[float] = obb.t_enter
                brake_prefix = f"减速制动礼让{target_label}中" if is_cyclist else "减速制动礼让中"
                desc = (
                    f"{brake_prefix} (MTTC安全, DRAC={drac:.2f}m/s^2, "
                    f"净距={min_dist_out:.2f}m)"
                )
            else:
                return None
        else:
            if obb.collides:
                assert obb.t_enter is not None
                t_eff = mttc if (mttc is not None and mttc < obb.t_enter) else obb.t_enter
            else:
                assert accel_entry_t is not None
                t_eff = accel_entry_t

            if t_eff <= self.ttc_critical_s:
                level = RiskLevel.LEVEL_3_CRITICAL
            elif t_eff <= self.ttc_caution_s:
                level = RiskLevel.LEVEL_2_CAUTION
            else:
                level = RiskLevel.LEVEL_0_SAFE

            if drac is not None:
                if drac >= self.drac_critical_ms2:
                    level = RiskLevel.LEVEL_3_CRITICAL
                elif drac >= self.drac_caution_ms2 and level < RiskLevel.LEVEL_2_CAUTION:
                    level = RiskLevel.LEVEL_2_CAUTION

            if level == RiskLevel.LEVEL_0_SAFE:
                return None

            drac_txt = f"{drac:.2f}" if drac is not None else "0.00"
            if level == RiskLevel.LEVEL_3_CRITICAL:
                crit_prefix = f"紧急碰撞危险({target_label})!" if is_cyclist else "紧急碰撞危险!"
                desc = (
                    f"{crit_prefix} (OBB-TTC={t_eff:.2f}s, "
                    f"DRAC={drac_txt}m/s^2, 净距={min_dist_out:.2f}m)"
                )
            else:
                desc = (
                    f"注意减速礼让{target_label} (OBB-TTC={t_eff:.2f}s, "
                    f"DRAC={drac_txt}m/s^2, 净距={min_dist_out:.2f}m)"
                )

        if accel_enters_obb:
            assert accel_entry_t is not None
            # 二阶加速侵入场景至少推进到侵入时刻，避免 1D MTTC 提前在 OBB 外部评估碰撞概率。
            t_eval = mttc if (mttc is not None and mttc >= accel_entry_t) else accel_entry_t
        else:
            t_eval = mttc if mttc is not None else (t_eff if t_eff is not None else obb.t_min_clearance)
        conflict_x = float(veh.x_m) + veh_vx * t_eval + 0.5 * float(veh_ax) * (t_eval ** 2)
        conflict_y = float(veh.y_m) + veh_vy * t_eval + 0.5 * float(veh_ay) * (t_eval ** 2)

        veh_cov = getattr(veh, "pos_cov_m2", None)
        ped_cov = getattr(ped, "pos_cov_m2", None)
        try:
            prob = estimate_collision_probability(
                dx, dy, dvx, dvy, t_eval, critical_dist_obb, veh_cov, ped_cov,
                dax=dax, day=day,
            )
        except ValueError:
            prob = estimate_collision_probability(
                dx, dy, dvx, dvy, t_eval, critical_dist_obb, None, None,
                dax=dax, day=day,
            )

        return ConflictAssessment(
            veh_track_id=veh.track_id,
            ped_track_id=ped.track_id,
            risk_level=level,
            risk_desc=desc,
            veh_speed_kmh=float(veh.speed_kmh),
            ped_speed_ms=float(ped.speed_ms),
            current_distance_m=current_dist,
            min_distance_m=min_dist_out,
            ttc_seconds=t_eff,
            conflict_ground_xy=(conflict_x, conflict_y),
            vehicle_kind=vehicle_policy.kind,
            mttc_seconds=mttc,
            drac_ms2=drac,
            pet_seconds=None,
            collision_prob=prob,
            conflict_model="obb_ssm",
            target_kind=target_kind,
        )

    def evaluate_pair(
        self, veh: SpeedReading, ped: SpeedReading
    ) -> Optional[ConflictAssessment]:
        """评估单一机动车与单一横穿行人之间的潜在几何冲突"""
        if self.conflict_model == "obb_ssm":
            return self._evaluate_pair_obb_ssm(veh, ped)
        return self._evaluate_pair_cvm(veh, ped)

    def analyze_frame(
        self,
        readings: List[SpeedReading],
        *,
        timestamp_s: Optional[float] = None,
    ) -> List[ConflictAssessment]:
        """全场景多目标批量冲突评估；obb_ssm 模式下传入 timestamp_s 可叠加 PET 近距险情判定。"""
        if timestamp_s is not None:
            if not _is_finite_number(timestamp_s) or float(timestamp_s) < 0.0:
                raise ValueError("timestamp_s 必须为有限非负数")

        # 无论默认模式还是 VRU 模式，所有进入冲突评估与 PET 网格的读数均先做有限字段校验。
        valid_readings = [r for r in readings if self._is_reading_finite(r)]
        pedestrians = [r for r in valid_readings if r.category in ("pedestrian", "person")]
        cyclists: List[SpeedReading] = []
        eval_entries: List[Tuple[SpeedReading, bool]] = []
        for r in valid_readings:
            if r.category == "vehicle":
                pol = object_policy(r.object_kind, r.category, self.object_policies)
                is_cyc = self.include_cyclist_vru and pol.kind in VRU_CYCLIST_KINDS
                if is_cyc:
                    cyclists.append(r)
                eval_entries.append((r, is_cyc))
            elif self.include_cyclist_vru and r.category in VRU_CYCLIST_KINDS:
                cyclists.append(r)
                eval_entries.append((r, True))

        ped_by_id = {p.track_id: p for p in pedestrians}
        # 开启骑行者 VRU 时，非骑行者机动车与行人+骑行者评估；骑行者仅与行人评估以免与机动车双向重复告警。
        vru_targets = pedestrians + cyclists if self.include_cyclist_vru else pedestrians

        pet_map: Dict[Tuple[int, int], float] = {}
        if self.conflict_model == "obb_ssm" and self.pet_tracker is not None and timestamp_s is not None:
            # 传入对象策略字典，使 PETTracker 能按车型物理尺寸做整车旋转占地栅格化
            pet_map = self.pet_tracker.update(
                valid_readings, float(timestamp_s), object_policies=self.object_policies
            )

        assessments: List[ConflictAssessment] = []

        for veh, is_cyclist_actor in eval_entries:
            highest_conflict: Optional[ConflictAssessment] = None
            targets = pedestrians if is_cyclist_actor else vru_targets

            for ped in targets:
                if ped.track_id == veh.track_id:
                    continue
                conflict = self.evaluate_pair(veh, ped)
                if conflict is not None:
                    if (
                        highest_conflict is None
                        or _conflict_sort_key(conflict) > _conflict_sort_key(highest_conflict)
                    ):
                        highest_conflict = conflict

            # 叠加 PET 网格跨轨迹侵入近险事件（捕捉行人刚撤离后车辆紧随擦过的险情）。
            if pet_map and not self._is_vehicle_exempt(veh):
                veh_kind_tmp = object_policy(veh.object_kind, veh.category, self.object_policies).kind
                for (v_id, p_id), pet_s in pet_map.items():
                    if v_id != veh.track_id:
                        continue
                    if (
                        highest_conflict is not None
                        and highest_conflict.ped_track_id == p_id
                        and highest_conflict.target_kind == "pedestrian"
                    ):
                        highest_conflict.pet_seconds = pet_s
                        if pet_s <= self.pet_critical_s and highest_conflict.risk_level < RiskLevel.LEVEL_3_CRITICAL:
                            highest_conflict.risk_level = RiskLevel.LEVEL_3_CRITICAL
                            highest_conflict.risk_desc = f"紧急近距擦碰险情 (PET={pet_s:.2f}s)"
                        continue

                    if pet_s > self.pet_caution_s:
                        continue

                    pet_level = (
                        RiskLevel.LEVEL_3_CRITICAL
                        if pet_s <= self.pet_critical_s
                        else RiskLevel.LEVEL_2_CAUTION
                    )
                    ped_obj = ped_by_id.get(p_id)
                    cur_dist = (
                        math.hypot(ped_obj.x_m - veh.x_m, ped_obj.y_m - veh.y_m)
                        if ped_obj is not None and _is_finite_number(ped_obj.x_m) and _is_finite_number(ped_obj.y_m)
                        else 0.0
                    )
                    ped_spd = (
                        float(ped_obj.speed_ms)
                        if ped_obj is not None and _is_finite_number(ped_obj.speed_ms)
                        else 0.0
                    )
                    pet_desc = (
                        f"紧急近距擦碰险情 (PET={pet_s:.2f}s)"
                        if pet_level == RiskLevel.LEVEL_3_CRITICAL
                        else f"注意近距跨迹险情 (PET={pet_s:.2f}s)"
                    )
                    pet_candidate = ConflictAssessment(
                        veh_track_id=veh.track_id,
                        ped_track_id=p_id,
                        risk_level=pet_level,
                        risk_desc=pet_desc,
                        veh_speed_kmh=float(veh.speed_kmh),
                        ped_speed_ms=ped_spd,
                        current_distance_m=cur_dist,
                        min_distance_m=0.0,
                        ttc_seconds=None,
                        conflict_ground_xy=(float(veh.x_m), float(veh.y_m)),
                        vehicle_kind=veh_kind_tmp,
                        pet_seconds=pet_s,
                        conflict_model=self.conflict_model,
                    )
                    if (
                        highest_conflict is None
                        or pet_level > highest_conflict.risk_level
                        or (
                            pet_level == highest_conflict.risk_level
                            and highest_conflict.ttc_seconds is None
                            and (
                                highest_conflict.pet_seconds is None
                                or pet_s < highest_conflict.pet_seconds
                            )
                        )
                    ):
                        highest_conflict = pet_candidate

            veh_kind = object_policy(veh.object_kind, veh.category, self.object_policies).kind
            if highest_conflict is not None:
                assessments.append(highest_conflict)
            elif (
                (veh.category == "vehicle" or veh.category in VRU_CYCLIST_KINDS)
                and _is_finite_number(veh.speed_kmh)
                and veh.speed_kmh > self.speed_limit_kmh
            ):
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
                        conflict_model=self.conflict_model,
                    )
                )
            else:
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
                        conflict_model=self.conflict_model,
                    )
                )

        return assessments
