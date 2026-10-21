# -*- coding: utf-8 -*-
"""
告警事件状态机与去抖引擎 (Debounced Alert Event State Machine)
=============================================================================
逐帧冲突研判 (ConflictAssessment) 是瞬时量：单帧检测抖动、短暂遮挡或 TTC
临界振荡都会在相邻帧间产生闪断，若按帧计数告警，一段持续 2 秒的危险会被
记成约 50 条"事件"。本模块把逐帧评估收敛为生命周期完整的事件：

- 连续 N 帧确认 (confirm_frames)：告警条件需持续满足才开立事件，
  单帧闪烁不会产生事件记录；
- 等级滞回：事件存续期间 peak_risk_level 只升不降，
  L2 升级为 L3 后即使瞬时回落仍按 L3 归档；
- 冷却合并 (cooldown_frames)：告警消失未超过冷却窗即复现，
  视为同一事件的延续（遮挡/噪声闪断），超过才归档并允许开新事件。

事件键 (key) = (veh_track_id, ped_track_id)；超速独立事件 ped_track_id=None。
同一车辆先后与不同行人冲突会产生不同键，各自独立计事件；
同一车辆"先超速、后冲突"也属于两个事件键，互不吞并。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .conflict import ConflictAssessment, RiskLevel


EventKey = Tuple[int, Optional[int]]


@dataclass
class AlertEvent:
    """一次已确认告警事件的完整生命周期记录"""
    event_id: int
    veh_track_id: int
    ped_track_id: Optional[int]
    start_frame: int               # 告警条件首次满足的帧（含确认前的挂起帧）
    start_timestamp_s: float
    end_frame: int                 # 告警条件最后一次满足的帧
    end_timestamp_s: float
    peak_risk_level: RiskLevel     # 事件期间最高风险等级（滞回：只升不降）
    min_ttc_seconds: Optional[float]
    max_veh_speed_kmh: float
    alarm_frames: int              # 告警条件实际满足的帧数
    gap_frames: int = 0            # 事件存续期间被冷却窗吸收的静默帧总数
    end_reason: str = ""           # "cooldown" | "eof" | "max_frames" | "user_quit" | "interrupt" | "exception"

    @property
    def kind(self) -> str:
        return "conflict" if self.ped_track_id is not None else "speeding"

    @property
    def duration_s(self) -> float:
        return self.end_timestamp_s - self.start_timestamp_s

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "event": "alert",
            "event_id": self.event_id,
            "kind": self.kind,
            "veh_track_id": self.veh_track_id,
            "ped_track_id": self.ped_track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_timestamp_s": round(self.start_timestamp_s, 3),
            "end_timestamp_s": round(self.end_timestamp_s, 3),
            "duration_s": round(self.duration_s, 3),
            "alarm_frames": self.alarm_frames,
            "gap_frames": self.gap_frames,
            "peak_risk_level": int(self.peak_risk_level),
            "peak_risk_name": self.peak_risk_level.name,
            "min_ttc_seconds": (
                round(self.min_ttc_seconds, 3)
                if self.min_ttc_seconds is not None
                else None
            ),
            "max_veh_speed_kmh": round(self.max_veh_speed_kmh, 1),
            "end_reason": self.end_reason,
        }


@dataclass
class _PendingStreak:
    """确认前的候选告警连续帧统计"""
    streak: int = 0
    first_frame: int = 0
    first_timestamp_s: float = 0.0
    peak: RiskLevel = RiskLevel.LEVEL_0_SAFE
    min_ttc: Optional[float] = None
    max_speed_kmh: float = 0.0

    def reset(self):
        self.streak = 0
        self.peak = RiskLevel.LEVEL_0_SAFE
        self.min_ttc = None
        self.max_speed_kmh = 0.0


@dataclass
class _KeyState:
    pending: _PendingStreak
    event: Optional[AlertEvent] = None
    silent_streak: int = 0         # 事件开立后的连续静默帧


class ConflictEventTracker:
    """把逐帧 ConflictAssessment 收敛为去抖后的告警事件流"""

    def __init__(self, confirm_frames: int = 3, cooldown_frames: int = 12):
        if type(confirm_frames) is not int or confirm_frames < 1:
            raise ValueError("confirm_frames 必须是 >= 1 的整数帧数")
        if type(cooldown_frames) is not int or cooldown_frames < 0:
            raise ValueError("cooldown_frames 必须是 >= 0 的整数帧数")
        self.confirm_frames = confirm_frames
        self.cooldown_frames = cooldown_frames
        self._states: Dict[EventKey, _KeyState] = {}
        self._finalized: List[AlertEvent] = []
        self._next_event_id = 1
        self._last_frame_idx: Optional[int] = None
        self._last_timestamp_s: Optional[float] = None

    @staticmethod
    def _key_of(c: ConflictAssessment) -> EventKey:
        return (c.veh_track_id, c.ped_track_id)

    def update(
        self,
        conflicts: List[ConflictAssessment],
        frame_idx: int,
        timestamp_s: float,
    ) -> Dict[EventKey, int]:
        """输入一帧的冲突评估，返回 {事件键: event_id}（仅已确认事件）。

        确认中的挂起帧不返回注解；事件确认后其 start_frame 已回溯包含挂起帧，
        下游可通过事件记录还原完整区间。
        """
        if not (math.isfinite(timestamp_s) and frame_idx >= 0):
            raise ValueError("frame_idx/timestamp_s 必须为有效帧序与有限时间戳")
        if self._last_frame_idx is not None and frame_idx <= self._last_frame_idx:
            # 每帧只允许一次 update：重复或回退帧序会让 alarm/gap 计数失真
            raise ValueError("frame_idx 必须逐帧严格递增")
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            # 时间戳不递增会产生负的 start/end 跨度与事件时长
            raise ValueError("timestamp_s 必须逐帧严格递增")
        self._last_frame_idx = frame_idx
        self._last_timestamp_s = timestamp_s

        alarmed: Dict[EventKey, ConflictAssessment] = {}
        for c in conflicts:
            if c.risk_level > RiskLevel.LEVEL_0_SAFE:
                alarmed[self._key_of(c)] = c

        annotations: Dict[EventKey, int] = {}

        # 1) 本帧告警满足的键：推进挂起计数或并入存续事件
        for key, c in alarmed.items():
            st = self._states.get(key)
            if st is None:
                st = _KeyState(pending=_PendingStreak())
                self._states[key] = st

            if st.event is not None:
                # 存续事件（含冷却窗内静默期）再次告警：同一事件延续，
                # 把刚吸收的静默帧计入 gap_frames，峰值等级滞回不降
                ev = st.event
                ev.gap_frames += st.silent_streak
                st.silent_streak = 0
                ev.end_frame = frame_idx
                ev.end_timestamp_s = timestamp_s
                if c.risk_level > ev.peak_risk_level:
                    ev.peak_risk_level = c.risk_level
                if c.ttc_seconds is not None and (
                    ev.min_ttc_seconds is None or c.ttc_seconds < ev.min_ttc_seconds
                ):
                    ev.min_ttc_seconds = c.ttc_seconds
                ev.max_veh_speed_kmh = max(ev.max_veh_speed_kmh, c.veh_speed_kmh)
                ev.alarm_frames += 1
                annotations[key] = ev.event_id
            else:
                # 挂起累计：需 confirm_frames 连续告警才开立事件
                p = st.pending
                if p.streak == 0:
                    p.first_frame, p.first_timestamp_s = frame_idx, timestamp_s
                p.streak += 1
                if c.risk_level > p.peak:
                    p.peak = c.risk_level
                if c.ttc_seconds is not None and (
                    p.min_ttc is None or c.ttc_seconds < p.min_ttc
                ):
                    p.min_ttc = c.ttc_seconds
                p.max_speed_kmh = max(p.max_speed_kmh, c.veh_speed_kmh)

                if p.streak >= self.confirm_frames:
                    st.event = AlertEvent(
                        event_id=self._next_event_id,
                        veh_track_id=c.veh_track_id,
                        ped_track_id=c.ped_track_id,
                        start_frame=p.first_frame,
                        start_timestamp_s=p.first_timestamp_s,
                        end_frame=frame_idx,
                        end_timestamp_s=timestamp_s,
                        peak_risk_level=p.peak,
                        min_ttc_seconds=p.min_ttc,
                        max_veh_speed_kmh=p.max_speed_kmh,
                        alarm_frames=p.streak,
                    )
                    self._next_event_id += 1
                    p.reset()
                    annotations[key] = st.event.event_id

        # 2) 本帧未告警的键：挂起中断清零；存续事件累计静默，超冷却窗归档
        for key, st in list(self._states.items()):
            if key in alarmed:
                continue
            st.pending.reset()
            if st.event is not None:
                st.silent_streak += 1
                if st.silent_streak > self.cooldown_frames:
                    st.event.end_reason = "cooldown"
                    self._finalized.append(st.event)
                    del self._states[key]
            else:
                # 仅挂起过就静默的键整键移除：从未确认的事件不占内存，
                # 防止一闪而过的干扰让 _states 无限膨胀
                del self._states[key]

        return annotations

    def pop_finalized(self) -> List[AlertEvent]:
        """取走自上次调用以来归档的事件（冷却超时确认结束）"""
        out, self._finalized = self._finalized, []
        return out

    def flush(self, reason: str = "eof"):
        """视频/输入结束时强制归档全部存续事件。

        仍处于冷却窗内的事件按"可能被截断"语义一并落盘；
        从未达到确认帧数的挂起记录不产生事件（确为闪烁，丢弃）。
        """
        for st in self._states.values():
            if st.event is not None:
                st.event.end_reason = reason
                self._finalized.append(st.event)
        self._states = {
            k: st for k, st in self._states.items() if st.event is None
        }
