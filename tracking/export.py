"""流式保存观测与轨迹生命周期；不把丢失期间的预测当作真实检测。"""

# 配套笔记：docs/modules/02_多目标跟踪_ByteTrack_解释文档.md，第 6.4、8、11 节。
# 本文件记录“看到了什么”和“何时丢失/恢复”，不负责决定两个框是不是同一目标。

from __future__ import annotations

import csv
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field

from .records import FrameRecord, TargetRecord


def json_line(stream, value):
    # 每行一个完整 JSON，便于长视频逐行读取；拒绝 NaN/Infinity，避免下游解析歧义。
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


@dataclass
class TrackStatistics:
    # 只保存一条活动 ID 的累计量；这里的 last_frame 是最后观测帧，不是最后预测帧。
    first_frame: int
    first_time: float
    last_frame: int
    last_time: float
    last_class: str
    observations: int = 0
    confidence_sum: float = 0.0
    recoveries: int = 0
    max_gap_frames: int = 0
    class_changes: int = 0
    missing: bool = False
    class_counts: Counter = field(default_factory=Counter)


class RunExports:
    """一帧一行 JSONL、一目标一行 CSV，结束的轨迹立即写盘并释放统计。"""

    COLUMNS = (
        "session_id", "source", "frame_index", "timestamp_s", "timestamp_source",
        "width", "height", "track_id", "class_id", "class_name", "category", "confidence",
        "x1", "y1", "x2", "y2", "anchor_x", "anchor_y", "anchor_type", "observation",
    )

    def __init__(self, output):
        self.resources = ExitStack()
        try:
            # 多文件构造中任一步失败，都关闭此前已打开的流，避免留下文件句柄。
            self.frames = self.resources.enter_context((output / "tracks.jsonl").open("w", encoding="utf-8"))
            self.events = self.resources.enter_context((output / "events.jsonl").open("w", encoding="utf-8"))
            self.tracklets = self.resources.enter_context((output / "tracklets.jsonl").open("w", encoding="utf-8"))
            # newline="" 让 csv 模块处理换行，避免 Windows 下产生多余空行。
            stream = self.resources.enter_context((output / "observations.csv").open("w", encoding="utf-8", newline=""))
            self.csv = csv.writer(stream)
            self.csv.writerow(self.COLUMNS)
        except BaseException:
            self.resources.close()
            raise
        self.active: dict[int, TrackStatistics] = {}
        self.event_counts = Counter()
        self.closed_tracks = 0
        self.last_record = None

    def _event(self, record, track_id, event, **extra):
        json_line(self.events, {
            "schema_version": 1, "session_id": record.session_id, "track_id": track_id,
            "event": event, "frame_index": record.frame_index, "timestamp_s": record.timestamp_s,
            **extra,
        })
        self.event_counts[event] += 1

    def _observe(self, record: FrameRecord, target: TargetRecord):
        track_id = target.track_id
        stats = self.active.get(track_id)
        if stats is None:
            stats = TrackStatistics(record.frame_index, record.timestamp_s, record.frame_index,
                                    record.timestamp_s, target.class_name)
            self.active[track_id] = stats
            self._event(record, track_id, "started")
        elif stats.missing:
            # 例如帧 0 和帧 3 有观测，中间缺帧 1、2：gap = 3 - 0 - 1 = 2。
            gap = record.frame_index - stats.last_frame - 1
            stats.recoveries += 1
            stats.max_gap_frames = max(stats.max_gap_frames, gap)
            self._event(record, track_id, "recovered", gap_frames=gap,
                        previous_observed_frame=stats.last_frame)
        if stats.last_class != target.class_name:
            # 标准 ByteTrack 可能出现类别抖动；记录变化，不擅自改写检测类别或拆分 ID。
            stats.class_changes += 1
            self._event(record, track_id, "class_changed", previous_class=stats.last_class,
                        class_name=target.class_name)
        stats.last_frame, stats.last_time = record.frame_index, record.timestamp_s
        stats.last_class, stats.missing = target.class_name, False
        stats.observations += 1
        stats.confidence_sum += target.confidence
        stats.class_counts[target.class_name] += 1

    def write(self, record: FrameRecord, retained_ids: set[int]):
        """retained_ids 来自真实跟踪器的存活池，包含暂时 Lost 的轨迹。"""
        json_line(self.frames, record.to_dict())
        # 先保存整帧，空帧也有一行；CSV 只保存目标，因此不能单靠 CSV 重建空帧时间轴。
        observed = set()
        for target in record.tracks:
            self.csv.writerow((
                record.session_id, record.source, record.frame_index, record.timestamp_s,
                record.timestamp_source, record.width, record.height, target.track_id,
                target.class_id, target.class_name, target.category, target.confidence,
                *target.bbox_xyxy, *target.anchor_xy, target.anchor_type, target.observation,
            ))
            if target.track_id is not None:
                # pending 框仍写入 CSV，但没有有效身份，不能进入按 ID 聚合的统计。
                observed.add(target.track_id)
                self._observe(record, target)
        for track_id, stats in list(self.active.items()):
            if track_id in observed:
                continue
            if track_id not in retained_ids:
                # 可能是超时删除，也可能是库内部去重；不能一概解释成超过 buffer。
                self._close_track(record, track_id, "tracker_removed")
            elif not stats.missing:
                # 连续丢失只记录一次；再次真正观测到这个 ID 时，由 _observe 记录恢复。
                stats.missing = True
                self._event(record, track_id, "lost", last_observed_frame=stats.last_frame)
        self.last_record = record

    def _close_track(self, record, track_id, reason):
        # 关闭后写出汇总并从 active 移除，避免所有历史轨迹统计随视频长度一直驻留。
        stats = self.active.pop(track_id)
        self._event(record, track_id, "closed", reason=reason, last_observed_frame=stats.last_frame)
        json_line(self.tracklets, {
            "schema_version": 1, "session_id": record.session_id, "track_id": track_id,
            "first_frame_index": stats.first_frame, "last_frame_index": stats.last_frame,
            "first_timestamp_s": stats.first_time, "last_timestamp_s": stats.last_time,
            # 这是首末观测的时间差，包含中间缺口；不等于每一瞬间都有有效检测。
            "observed_span_s": stats.last_time - stats.first_time,
            "observations": stats.observations,
            "unobserved_frames_within_span": stats.last_frame - stats.first_frame + 1 - stats.observations,
            "recoveries": stats.recoveries, "max_recovered_gap_frames": stats.max_gap_frames,
            "class_counts": dict(stats.class_counts), "class_changes": stats.class_changes,
            "last_class_name": stats.last_class,
            "mean_confidence": stats.confidence_sum / stats.observations,
            "close_frame_index": record.frame_index, "close_timestamp_s": record.timestamp_s,
            "close_reason": reason,
        })
        self.closed_tracks += 1

    def finish(self, reason):
        # 视频结束只是本次观测停止，不等于目标离开道路，也不等于跟踪器判定 Removed。
        if self.last_record is not None:
            for track_id in list(self.active):
                self._close_track(self.last_record, track_id, f"session_end:{reason}")

    def close(self):
        self.resources.close()
