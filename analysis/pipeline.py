# -*- coding: utf-8 -*-
"""
校园道路车辆—行人冲突感知与测速预警完整流水线 (End-to-End Conflict Perception Pipeline)
=============================================================================
集成 TrackingSession 目标追踪、RoadCalibrator 单应性投影、
SpeedEstimator 平滑测速、ConflictAnalyzer 碰撞决策与 EnhancedVisualizer 可视化，
实现从视频输入到结构化风险告警的端到端技术闭环。
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional

import cv2

from detection.core import DEFAULT_WEIGHTS
from tracking.core import (
    DEFAULT_TRACKER,
    LIVE_STREAM_PREFIXES,
    TrackingSession,
    VideoClock,
    read_tracker_config,
    redact_stream_url,
)
from .calibration import RoadCalibrator
from .conflict import (
    ConflictAnalyzer,
    ConflictAssessment,
    RiskLevel,
    build_conflict_maps,
)
from .events import ConflictEventTracker
from .policy import DEFAULT_OBJECT_POLICY_CONFIG, read_object_policy_config
from .provenance import collect_provenance
from .speed import SpeedEstimator, SpeedReading
from .visualization import EnhancedConflictVisualizer, trajectory_color


def is_live_source(source: str | int | Path | None) -> bool:
    """判断输入源是否为摄像头设备索引或实时网络视频流。"""
    if source is None or type(source) is bool:
        return False
    if type(source) is int:
        return source >= 0
    if isinstance(source, str):
        stripped = source.strip()
        if not stripped:
            return False
        if stripped.isascii() and stripped.isdigit():
            return True
        lower = stripped.lower()
        if lower.startswith("camera:"):
            idx_part = stripped.split(":", 1)[1].strip()
            return idx_part.isascii() and idx_part.isdigit()
        return lower.startswith(LIVE_STREAM_PREFIXES)
    return False


def parse_video_source(video_path: str | int | Path) -> tuple[int | str, str, bool]:
    """解析视频输入源，返回 (VideoCapture 参数, 脱敏日志标识, 是否实时流)。"""
    if type(video_path) is bool:
        raise ValueError("视频输入源不能为布尔值")
    if type(video_path) is int:
        if video_path < 0:
            raise ValueError("摄像头索引必须是非负整数")
        return video_path, f"camera:{video_path}", True
    if isinstance(video_path, str):
        stripped = video_path.strip()
        if not stripped:
            raise ValueError("视频输入源不能为空")
        if stripped.isascii() and stripped.isdigit():
            idx = int(stripped)
            return idx, f"camera:{idx}", True
        lower = stripped.lower()
        if lower.startswith("camera:"):
            idx_part = stripped.split(":", 1)[1].strip()
            if not (idx_part.isascii() and idx_part.isdigit()):
                raise ValueError(f"无效的摄像头标识: {video_path}")
            idx = int(idx_part)
            return idx, f"camera:{idx}", True
        if lower.startswith(LIVE_STREAM_PREFIXES):
            return stripped, redact_stream_url(stripped), True
    resolved = Path(video_path).expanduser().resolve()
    return str(resolved), str(resolved), False


class LiveStreamClock:
    """实时流单调时钟：优先使用递增 PTS，缺少或重复时回退到墙钟时间。"""

    def __init__(self):
        self._anchor_wall: Optional[float] = None
        self._anchor_ts: float = 0.0
        self.previous: float = -1.0

    def timestamp(self, frame_index: int, milliseconds: float) -> tuple[float, str]:
        now = time.perf_counter()
        if self._anchor_wall is None:
            self._anchor_wall = now
            self._anchor_ts = 0.0
        pts_s = milliseconds / 1000.0 if milliseconds is not None else float("nan")
        if math.isfinite(pts_s) and pts_s >= 0 and pts_s > self.previous and (
            frame_index > 0 or pts_s > 0
        ):
            self._anchor_wall = now
            self._anchor_ts = pts_s
            self.previous = pts_s
            return pts_s, "video_pts"
        if frame_index == 0 and self.previous < 0:
            self._anchor_wall = now
            self._anchor_ts = 0.0
            self.previous = 0.0
            return 0.0, "wall_clock"
        wall_s = self._anchor_ts + (now - self._anchor_wall)
        ts = max(wall_s, math.nextafter(self.previous, math.inf), self.previous + 1e-4)
        self.previous = ts
        return ts, "wall_clock"


class ConflictPerceptionPipeline:
    """冲突感知与预警全流程管道执行器"""

    def __init__(
        self,
        calibrator: RoadCalibrator,
        weights: str = str(DEFAULT_WEIGHTS),
        tracker_config: str = str(DEFAULT_TRACKER),
        speed_limit_kmh: float = 30.0,
        device: Optional[str] = None,
        conf: Optional[float] = None,
        iou: float = 0.7,
        imgsz: int = 640,
        ema_alpha: float = 0.35,
        window_size: int = 8,
        window_seconds: Optional[float] = None,
        zone_margin_m: float = 2.0,
        ttc_critical_s: float = 1.8,
        ttc_caution_s: float = 3.5,
        horizon_s: float = 3.5,
        stationary_speed_ms: float = 0.1,
        stationary_min_frames: int = 8,
        stationary_seconds: Optional[float] = None,
        category_confirm_frames: int = 3,
        reliable_min_span_s: float = 0.2,
        alert_confirm_frames: int = 3,
        alert_confirm_seconds: Optional[float] = None,
        alert_cooldown_frames: int = 12,
        alert_cooldown_seconds: Optional[float] = None,
        fps: Optional[float] = None,
        object_policy_config: str | Path = DEFAULT_OBJECT_POLICY_CONFIG,
        parking_zone_max_speed_kmh: float = 5.0,
    ):
        if type(speed_limit_kmh) not in (int, float) or not math.isfinite(speed_limit_kmh) or speed_limit_kmh <= 0:
            raise ValueError("speed_limit_kmh 必须为有限正数")
        if type(iou) not in (int, float) or not math.isfinite(iou) or not (0 < iou <= 1):
            raise ValueError("iou 必须在 (0, 1] 区间")
        if fps is not None and (type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0):
            raise ValueError("fps 必须是有限正数")
        if type(imgsz) is not int or imgsz <= 0:
            raise ValueError("imgsz 必须是正整数")
        if type(ema_alpha) not in (int, float) or not math.isfinite(ema_alpha) or not (0 < ema_alpha <= 1):
            raise ValueError("ema_alpha 必须在 (0, 1] 区间")
        if type(window_size) is not int or window_size < 3:
            raise ValueError("window_size 必须是 >= 3 的整数帧数")
        if type(zone_margin_m) not in (int, float) or not math.isfinite(zone_margin_m) or zone_margin_m < 0:
            raise ValueError("zone_margin_m 必须为有限非负数")
        for ttc_name, ttc_val in (
            ("ttc_critical_s", ttc_critical_s),
            ("ttc_caution_s", ttc_caution_s),
            ("horizon_s", horizon_s),
        ):
            if type(ttc_val) not in (int, float) or not math.isfinite(ttc_val) or ttc_val <= 0:
                raise ValueError(f"{ttc_name} 必须为有限正数")
        if not (ttc_critical_s <= ttc_caution_s <= horizon_s):
            raise ValueError("TTC 阈值需满足 0 < ttc_critical_s <= ttc_caution_s <= horizon_s")
        if type(stationary_speed_ms) not in (int, float) or not math.isfinite(stationary_speed_ms) or stationary_speed_ms <= 0:
            raise ValueError("stationary_speed_ms 必须为有限正数")
        if type(stationary_min_frames) is not int or stationary_min_frames < 1:
            raise ValueError("stationary_min_frames 必须是 >= 1 的整数帧数")
        if type(category_confirm_frames) is not int or category_confirm_frames < 1:
            raise ValueError("category_confirm_frames 必须是 >= 1 的整数帧数")
        if type(reliable_min_span_s) not in (int, float) or not math.isfinite(reliable_min_span_s) or reliable_min_span_s < 0:
            raise ValueError("reliable_min_span_s 必须为有限非负数")
        if window_seconds is not None and (
            type(window_seconds) not in (int, float) or not math.isfinite(window_seconds) or window_seconds <= 0
        ):
            raise ValueError("window_seconds 必须为有限正数")
        if stationary_seconds is not None and (
            type(stationary_seconds) not in (int, float) or not math.isfinite(stationary_seconds) or stationary_seconds <= 0
        ):
            raise ValueError("stationary_seconds 必须为有限正数")
        if type(alert_confirm_frames) is not int or alert_confirm_frames < 1:
            raise ValueError("alert_confirm_frames 必须是 >= 1 的整数帧数")
        if alert_confirm_seconds is not None and (
            type(alert_confirm_seconds) not in (int, float) or not math.isfinite(alert_confirm_seconds) or alert_confirm_seconds <= 0
        ):
            raise ValueError("alert_confirm_seconds 必须为有限正数")
        if type(alert_cooldown_frames) is not int or alert_cooldown_frames < 0:
            raise ValueError("alert_cooldown_frames 必须是 >= 0 的整数帧数")
        if alert_cooldown_seconds is not None and (
            type(alert_cooldown_seconds) not in (int, float) or not math.isfinite(alert_cooldown_seconds) or alert_cooldown_seconds < 0
        ):
            raise ValueError("alert_cooldown_seconds 必须为有限非负数")
        if (
            type(parking_zone_max_speed_kmh) not in (int, float)
            or not math.isfinite(parking_zone_max_speed_kmh)
            or parking_zone_max_speed_kmh < 0
        ):
            raise ValueError("parking_zone_max_speed_kmh 必须为有限非负数")

        self.calibrator = calibrator
        self.weights = weights
        self.tracker_config = tracker_config
        self.speed_limit_kmh = speed_limit_kmh
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.zone_margin_m = zone_margin_m
        self.window_seconds = window_seconds
        self.stationary_seconds = stationary_seconds
        self.alert_confirm_frames = alert_confirm_frames
        self.alert_confirm_seconds = alert_confirm_seconds
        self.alert_cooldown_frames = alert_cooldown_frames
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.fps = fps
        self.object_policy_config = str(Path(object_policy_config).expanduser().resolve())
        self.object_policies = read_object_policy_config(object_policy_config)
        self.parking_zone_max_speed_kmh = float(parking_zone_max_speed_kmh)
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.ttc_critical_s = ttc_critical_s
        self.ttc_caution_s = ttc_caution_s
        self.horizon_s = horizon_s
        self.stationary_speed_ms = stationary_speed_ms
        self.stationary_min_frames = stationary_min_frames
        self.category_confirm_frames = category_confirm_frames
        self.reliable_min_span_s = reliable_min_span_s

        # 每次 process_video 按实际 FPS 重建 estimator，跨视频不串状态
        self._estimator_kwargs = dict(
            speed_limit_kmh=speed_limit_kmh,
            window_size=window_size,
            ema_alpha=ema_alpha,
            stationary_speed_ms=stationary_speed_ms,
            stationary_min_frames=stationary_min_frames,
            category_confirm_frames=category_confirm_frames,
            reliable_min_span_s=reliable_min_span_s,
            object_policies=self.object_policies,
        )
        self.speed_estimator = SpeedEstimator(**self._estimator_kwargs)
        self.event_tracker = ConflictEventTracker(
            confirm_frames=alert_confirm_frames,
            cooldown_frames=alert_cooldown_frames,
        )
        self.conflict_analyzer = ConflictAnalyzer(
            speed_limit_kmh=speed_limit_kmh,
            ttc_critical_s=ttc_critical_s,
            ttc_caution_s=ttc_caution_s,
            horizon_s=horizon_s,
            object_policies=self.object_policies,
            parking_zone_max_speed_kmh=self.parking_zone_max_speed_kmh,
        )
        self.visualizer = EnhancedConflictVisualizer(
            calibrator=calibrator, speed_limit_kmh=speed_limit_kmh
        )

    def _configuration_snapshot(
        self,
        fps: float,
        *,
        input_mode: str = "video",
        source_tracks: Optional[str] = None,
    ) -> dict:
        """返回可复现实验所需的实际运行配置，而不依赖命令行现场。"""
        snapshot = {
            "schema_version": 1,
            "input_mode": input_mode,
            "weights": str(Path(self.weights).expanduser().resolve()),
            "tracker_config": str(Path(self.tracker_config).expanduser().resolve()),
            "object_policy_config": self.object_policy_config,
            "device": self.device,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "calibration": {
                "name": self.calibrator.name,
                "synthetic": self.calibrator.synthetic,
                "width_m": self.calibrator.width_m,
                "length_m": self.calibrator.length_m,
                "parking_zones_count": len(getattr(self.calibrator, "parking_zones", ())),
            },
            "object_policies": {
                name: {
                    "kind": policy.kind,
                    "safe_radius_veh_m": policy.safe_radius_veh_m,
                    "label": policy.label,
                }
                for name, policy in self.object_policies.items()
            },
            "fps_used": fps,
            "speed_limit_kmh": self.speed_limit_kmh,
            "ema_alpha": self.ema_alpha,
            "window_size_frames": self.speed_estimator.window_size,
            "window_seconds": self.window_seconds,
            "zone_margin_m": self.zone_margin_m,
            "parking_zone_max_speed_kmh": self.parking_zone_max_speed_kmh,
            "ttc_critical_s": self.ttc_critical_s,
            "ttc_caution_s": self.ttc_caution_s,
            "horizon_s": self.horizon_s,
            "stationary_speed_ms": self.stationary_speed_ms,
            "stationary_min_frames": self.speed_estimator.stationary_min_frames,
            "stationary_seconds": self.stationary_seconds,
            "category_confirm_frames": self.category_confirm_frames,
            "reliable_min_span_s": self.reliable_min_span_s,
            "alert_confirm_frames": self.event_tracker.confirm_frames,
            "alert_confirm_seconds": self.alert_confirm_seconds,
            "alert_cooldown_frames": self.event_tracker.cooldown_frames,
            "alert_cooldown_seconds": self.alert_cooldown_seconds,
            "reproducibility": collect_provenance(
                Path(__file__).resolve().parents[1],
                weights=self.weights,
                tracker_config=self.tracker_config,
                calibration=self.calibrator.source_path,
                object_policy=self.object_policy_config,
            ),
        }
        if source_tracks is not None:
            snapshot["source_tracks"] = source_tracks
        return snapshot

    def _configure_speed_estimator(self, fps: float):
        """按视频实际 FPS 重建速度估计器：换算秒级窗口，且不继承上一视频状态"""
        kwargs = dict(self._estimator_kwargs)
        if self.window_seconds is not None:
            # 以秒表达的测速窗口：按实际 FPS 换算帧数，跨帧率视频保持一致时间窗
            kwargs["window_size"] = max(3, int(round(fps * self.window_seconds)))
        if self.stationary_seconds is not None:
            kwargs["stationary_min_frames"] = max(1, int(round(fps * self.stationary_seconds)))
        self.speed_estimator = SpeedEstimator(**kwargs)

    def _configure_event_tracker(self, fps: float):
        """按视频实际 FPS 重建告警事件跟踪器：换算秒级窗口，且不继承上一视频状态"""
        confirm_frames = self.alert_confirm_frames
        if self.alert_confirm_seconds is not None:
            # 秒级确认窗：60fps 的 3 帧只有 50ms，按帧计数跨帧率语义不一致
            confirm_frames = max(1, int(round(fps * self.alert_confirm_seconds)))
        cooldown_frames = self.alert_cooldown_frames
        if self.alert_cooldown_seconds is not None:
            cooldown_frames = max(0, int(round(fps * self.alert_cooldown_seconds)))
        self.event_tracker = ConflictEventTracker(
            confirm_frames=confirm_frames,
            cooldown_frames=cooldown_frames,
        )

    @staticmethod
    def _write_summary_file(out_dir: Path, payload: dict):
        (out_dir / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    def _drain_alert_events_into(self, events_file, counts: Dict[str, int]):
        for evt in self.event_tracker.pop_finalized():
            if evt.peak_risk_level == RiskLevel.LEVEL_3_CRITICAL:
                counts["critical"] += 1
            elif evt.peak_risk_level == RiskLevel.LEVEL_2_CAUTION:
                counts["caution"] += 1
            elif evt.peak_risk_level == RiskLevel.LEVEL_1_SPEEDING:
                counts["speeding"] += 1
            events_file.write(
                json.dumps(evt.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
            )

    @staticmethod
    def _load_validated_tracks(tracks_file: Path) -> list[dict]:
        """读取并严格校验离线 tracks.jsonl 帧序列。"""
        from .schema import load_jsonl_strict

        if not tracks_file.is_file():
            raise FileNotFoundError(f"未找到跟踪轨迹文件: {tracks_file}")
        rows = load_jsonl_strict(tracks_file)
        expected_session: Optional[str] = None
        expected_shape: Optional[tuple[int, int]] = None
        prev_frame = -1
        prev_ts = -1.0
        for line_idx, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError(f"tracks.jsonl 第 {line_idx} 行必须是 JSON 对象")
            if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
                raise ValueError(f"tracks.jsonl 第 {line_idx} 行 schema_version 必须为 1")
            sid = row.get("session_id")
            if not isinstance(sid, str) or not sid:
                raise ValueError(f"tracks.jsonl 第 {line_idx} 行缺少有效 session_id")
            if expected_session is None:
                expected_session = sid
            elif sid != expected_session:
                raise ValueError("tracks.jsonl 包含多个 session_id，单次复盘仅支持同一会话")
            f_idx = row.get("frame_index")
            ts = row.get("timestamp_s")
            w = row.get("width")
            h = row.get("height")
            if type(f_idx) is not int or f_idx < 0 or f_idx <= prev_frame:
                raise ValueError(f"tracks.jsonl 第 {line_idx} 行 frame_index 必须非负且严格递增")
            if type(ts) not in (int, float) or not math.isfinite(ts) or ts < 0 or ts <= prev_ts:
                raise ValueError(f"tracks.jsonl 第 {line_idx} 行 timestamp_s 必须非负有限且严格递增")
            if type(w) is not int or type(h) is not int or w <= 0 or h <= 0:
                raise ValueError(f"tracks.jsonl 第 {line_idx} 行 width/height 必须为正整数")
            if expected_shape is None:
                expected_shape = (w, h)
            elif (w, h) != expected_shape:
                raise ValueError(
                    f"tracks.jsonl 第 {line_idx} 行分辨率 ({w}x{h}) 与首帧 ({expected_shape[0]}x{expected_shape[1]}) 不一致"
                )
            raw_tracks = row.get("tracks")
            if not isinstance(raw_tracks, list):
                raise ValueError(f"tracks.jsonl 第 {line_idx} 行 tracks 必须是列表")
            seen_ids: set[int] = set()
            for target in raw_tracks:
                if not isinstance(target, dict):
                    raise ValueError(f"tracks.jsonl 第 {line_idx} 行包含无效目标对象")
                tid = target.get("track_id")
                if tid is not None:
                    if type(tid) is not int or tid <= 0 or tid in seen_ids:
                        raise ValueError(f"tracks.jsonl 第 {line_idx} 行 track_id 必须为不重复正整数")
                    seen_ids.add(tid)
                cname = target.get("class_name")
                cat = target.get("category")
                if not isinstance(cname, str) or not cname or not isinstance(cat, str) or not cat:
                    raise ValueError(f"tracks.jsonl 第 {line_idx} 行缺少 class_name 或 category")
                bbox = target.get("bbox_xyxy")
                anchor = target.get("anchor_xy")
                if (
                    not isinstance(bbox, (list, tuple))
                    or len(bbox) != 4
                    or not isinstance(anchor, (list, tuple))
                    or len(anchor) != 2
                    or not all(
                        type(v) in (int, float) and math.isfinite(v)
                        for v in (*bbox, *anchor)
                    )
                ):
                    raise ValueError(f"tracks.jsonl 第 {line_idx} 行坐标必须为有限数值")
            prev_frame = f_idx
            prev_ts = float(ts)
        return rows

    def _process_single_frame(
        self,
        tracks: Iterable,
        *,
        resolve_live_ids,
        frame_idx: int,
        timestamp_s: float,
        jsonl_file,
        csv_writer,
        drain_events_cb,
        frame=None,
        writer=None,
    ):
        """执行单帧标定投影、速度估计、冲突预测、去抖落盘与可选视频渲染。"""
        speed_readings: List[SpeedReading] = []
        reading_map: Dict[int, SpeedReading] = {}
        out_of_zone_ids: set[int] = set()
        frame_max_speed = 0.0

        for t in tracks:
            if t.track_id is None:
                continue

            u_foot, v_foot = t.anchor_xy
            x_m, y_m = self.calibrator.pixel_to_ground(u_foot, v_foot)
            if not (math.isfinite(x_m) and math.isfinite(y_m)):
                # 地平线附近退化投影点：不产生读数，防止 NaN 污染速度历史
                continue

            # 标定四边形以外的坐标是外推伪影：先于滤波器更新就门控
            if not self.calibrator.is_in_zone(x_m, y_m, margin_m=self.zone_margin_m):
                out_of_zone_ids.add(t.track_id)
                continue

            reading = self.speed_estimator.update(
                track_id=t.track_id,
                category=t.category,
                timestamp_s=timestamp_s,
                x_m=x_m,
                y_m=y_m,
                object_kind=t.class_name,
            )
            reading.in_parking_zone = self.calibrator.is_in_parking_zone(x_m, y_m)
            reading_map[t.track_id] = reading
            speed_readings.append(reading)

            if reading.speed_kmh > frame_max_speed:
                frame_max_speed = reading.speed_kmh

        live_ids = resolve_live_ids(out_of_zone_ids)
        self.speed_estimator.prune_stale_tracks(live_ids)

        conflicts = self.conflict_analyzer.analyze_frame(speed_readings)
        conflict_map, ped_conflict_map = build_conflict_maps(conflicts)

        event_ids = self.event_tracker.update(conflicts, frame_idx, timestamp_s)
        drain_events_cb()

        for c in conflicts:
            json_item = {
                "schema_version": 1,
                "frame_index": frame_idx,
                "timestamp_s": round(timestamp_s, 3),
                "event_id": event_ids.get((c.veh_track_id, c.ped_track_id)),
                "assessment": asdict(c),
            }
            jsonl_file.write(
                json.dumps(json_item, ensure_ascii=False, allow_nan=False) + "\n"
            )

        for r in speed_readings:
            c = conflict_map.get(r.track_id) or ped_conflict_map.get(r.track_id)
            risk_lvl = c.risk_level.name if c else "LEVEL_0_SAFE"
            ttc_val = round(c.ttc_seconds, 2) if (c and c.ttc_seconds) else ""
            csv_event_id = (
                event_ids.get((c.veh_track_id, c.ped_track_id))
                if c is not None else None
            )
            csv_writer.writerow([
                1,
                frame_idx, round(timestamp_s, 3), r.track_id, r.category,
                round(r.x_m, 2), round(r.y_m, 2), round(r.speed_kmh, 1),
                int(r.is_speeding), risk_lvl, ttc_val,
                csv_event_id if csv_event_id is not None else "",
                int(r.is_reliable), int(r.is_stationary)
            ])

        render_frame = None
        if frame is not None:
            render_frame = frame.copy()
            self.visualizer.draw_calibrated_ground_zone(render_frame)

            for r in speed_readings:
                c = conflict_map.get(r.track_id) or ped_conflict_map.get(r.track_id)
                self.visualizer.draw_future_trajectory(
                    render_frame, r, color=trajectory_color(c)
                )

            for t in tracks:
                if t.track_id is None:
                    continue
                r = reading_map.get(t.track_id)
                if r:
                    c = conflict_map.get(t.track_id) or ped_conflict_map.get(t.track_id)
                    eid = (
                        event_ids.get((c.veh_track_id, c.ped_track_id))
                        if c is not None else None
                    )
                    self.visualizer.draw_speed_badge(
                        render_frame, t.bbox_xyxy, r, c, event_id=eid
                    )

            for c in conflicts:
                self.visualizer.draw_conflict_link(render_frame, c, reading_map)

            self.visualizer.draw_hud_banner(render_frame, conflicts, event_ids)
            if writer is not None:
                writer.write(render_frame)

        return frame_max_speed, render_frame

    def process_video(
        self,
        video_path: str | int | Path,
        output_dir: str | Path,
        show_window: bool = False,
        max_frames: Optional[int] = None,
    ) -> dict:
        """端到端处理本地视频、USB 摄像头或 RTSP/HTTP 实时流，输出渲染视频与告警日志"""
        if max_frames is not None and (type(max_frames) is not int or max_frames < 0):
            raise ValueError("max_frames 必须是 >= 0 的整数")

        capture_arg, src_label, is_live_stream = parse_video_source(video_path)
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        init_started_at = time.perf_counter()
        out_video_path = out_dir / "annotated_conflict.mp4"
        jsonl_path = out_dir / "conflicts.jsonl"
        events_path = out_dir / "events.jsonl"
        csv_path = out_dir / "speeds.csv"
        configuration_path = out_dir / "configuration.json"

        def _write_initialization_failure(exc: BaseException):
            """初始化阶段失败时留下最小可消费摘要，不掩盖原始异常。"""
            is_interrupt = isinstance(exc, KeyboardInterrupt)
            try:
                self._write_summary_file(out_dir, {
                    "schema_version": 1,
                    "status": "interrupted" if is_interrupt else "failed",
                    "stop_reason": "interrupt" if is_interrupt else "initialization",
                    "error": "KeyboardInterrupt" if is_interrupt else f"{type(exc).__name__}: {exc}",
                    "source_video": src_label,
                    "frames_processed": 0,
                    "elapsed_seconds": round(max(0.0, time.perf_counter() - init_started_at), 2),
                    "annotated_output": str(out_video_path),
                    "conflicts_jsonl": str(jsonl_path),
                    "events_jsonl": str(events_path),
                    "speeds_csv": str(csv_path),
                    "configuration_json": str(configuration_path),
                })
            except Exception:
                pass

        try:
            cap = cv2.VideoCapture(capture_arg)
        except BaseException as exc:
            _write_initialization_failure(exc)
            raise
        if not cap.isOpened():
            exc = RuntimeError(f"无法打开视频输入源: {src_label}")
            cap.release()
            _write_initialization_failure(exc)
            raise exc

        # ExitStack 统一登记资源生命周期：初始化阶段（模型加载、编码器、
        # 文件打开）与主循环异常都保证释放，不存在半初始化泄漏路径
        resources = ExitStack()
        resources.callback(cap.release)
        try:
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            if is_live_stream:
                source_fps = raw_fps if (math.isfinite(raw_fps) and 0 < raw_fps <= 240) else None
                fps = self.fps if self.fps is not None else (source_fps or 25.0)
                video_clock = (
                    VideoClock(fps=fps, force_fps=True)
                    if self.fps is not None
                    else LiveStreamClock()
                )
            else:
                # 离线视频源 FPS 缺失/无效时必须显式 --fps，避免建立在错误时间基准上
                source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
                fps = self.fps if self.fps is not None else source_fps
                if fps is None:
                    raise RuntimeError("视频缺少有效 FPS；确认帧率后通过 --fps 指定")
                video_clock = VideoClock(fps=fps, force_fps=self.fps is not None)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self._configure_speed_estimator(fps)
            self._configure_event_tracker(fps)
            session = TrackingSession(
                source=src_label,
                weights=self.weights,
                tracker_path=self.tracker_config,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
            )
            configuration = self._configuration_snapshot(
                fps, input_mode="live_stream" if is_live_stream else "video"
            )
            configuration_path.write_text(
                json.dumps(configuration, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
            resources.callback(writer.release)
            if not writer.isOpened():
                raise RuntimeError(f"无法创建渲染输出视频，请检查编码器与路径: {out_video_path}")

            jsonl_file = resources.enter_context(open(jsonl_path, "w", encoding="utf-8"))
            events_file = resources.enter_context(open(events_path, "w", encoding="utf-8"))
            csv_file = resources.enter_context(open(csv_path, "w", encoding="utf-8", newline=""))

            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "schema_version",
                "frame_idx", "timestamp_s", "track_id", "category",
                "x_m", "y_m", "speed_kmh", "is_speeding", "risk_level", "ttc_s",
                "event_id", "is_reliable", "is_stationary"
            ])
        except BaseException as exc:
            try:
                resources.close()
            except Exception:
                pass
            _write_initialization_failure(exc)
            raise

        counts = {"critical": 0, "caution": 0, "speeding": 0}
        max_observed_speed_kmh = 0.0

        def _drain_alert_events():
            self._drain_alert_events_into(events_file, counts)

        def _resolve_live_ids(out_of_zone: set[int]) -> set[int]:
            try:
                live = set(session.retained_ids())
            except Exception as exc:
                raise RuntimeError(
                    "无法读取跟踪器存活池 (tracked/lost stracks)，"
                    "当前 Ultralytics 版本接口不兼容；请固定 ultralytics==8.4.182 或更新适配"
                ) from exc
            live -= out_of_zone
            return live

        frame_idx = 0
        t_start = time.perf_counter()
        src_display = src_label if is_live_stream else Path(src_label).name

        print("=" * 60)
        print("[*] 启动校园道路冲突感知与测速预警系统:")
        print(f"    输入源:   {src_display} ({width}x{height} @ {fps:.1f} FPS)")
        print(f"    标定路网: {self.calibrator.name} (宽 {self.calibrator.width_m}m, 长 {self.calibrator.length_m}m)")
        print(f"    限速标准: {self.speed_limit_kmh} km/h")
        print("=" * 60)

        loop_state = "eof"
        loop_error = ""
        main_loop_failed = False
        close_exc: Optional[BaseException] = None
        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    loop_state = "max_frames"
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_s, time_kind = video_clock.timestamp(frame_idx, pts_ms)

                frame_record = session.process(frame, timestamp_s, time_kind)
                frame_max_speed, render_frame = self._process_single_frame(
                    frame_record.tracks,
                    resolve_live_ids=_resolve_live_ids,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    jsonl_file=jsonl_file,
                    csv_writer=csv_writer,
                    drain_events_cb=_drain_alert_events,
                    frame=frame,
                    writer=writer,
                )
                if frame_max_speed > max_observed_speed_kmh:
                    max_observed_speed_kmh = frame_max_speed

                frame_idx += 1

                if show_window and render_frame is not None:
                    cv2.imshow("Campus Vehicle-Pedestrian Conflict System", render_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        loop_state = "user_quit"
                        break

                if frame_idx % 30 == 0:
                    elapsed_now = time.perf_counter() - t_start
                    fps_proc = frame_idx / elapsed_now if elapsed_now > 0 else 0.0
                    print(f"[*] 已处理 {frame_idx} 帧 | 处理速率: {fps_proc:.1f} FPS | 最高时速: {max_observed_speed_kmh:.1f} km/h")

        except KeyboardInterrupt:
            main_loop_failed = True
            loop_state = "interrupt"
            loop_error = "KeyboardInterrupt"
            raise
        except BaseException as exc:
            main_loop_failed = True
            loop_state = "exception"
            loop_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                self.event_tracker.flush(reason=loop_state)
                _drain_alert_events()
            except Exception:
                pass
            try:
                resources.close()
            except BaseException as exc:
                close_exc = exc
                if not main_loop_failed:
                    loop_state = "exception"
                    loop_error = f"{type(exc).__name__}: {exc}"
            if loop_state in ("interrupt", "exception"):
                try:
                    self._write_summary_file(out_dir, {
                        "schema_version": 1,
                        "status": "interrupted" if loop_state == "interrupt" else "failed",
                        "stop_reason": loop_state,
                        "error": loop_error,
                        "source_video": src_label,
                        "annotated_output": str(out_video_path),
                        "conflicts_jsonl": str(jsonl_path),
                        "events_jsonl": str(events_path.resolve()),
                        "speeds_csv": str(csv_path),
                        "configuration_json": str(configuration_path),
                        "configuration": configuration,
                        "frames_processed": frame_idx,
                        "elapsed_seconds": round(max(0.0, time.perf_counter() - t_start), 2),
                        "max_observed_speed_kmh": round(max_observed_speed_kmh, 1),
                    })
                except Exception:
                    pass
            if show_window:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if close_exc is not None and not main_loop_failed:
                raise close_exc

        elapsed = max(0.0, time.perf_counter() - t_start)
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0

        summary = {
            "schema_version": 1,
            "status": "ok",
            "stop_reason": loop_state,
            "source_video": src_label,
            "annotated_output": str(out_video_path),
            "conflicts_jsonl": str(jsonl_path),
            "events_jsonl": str(events_path),
            "speeds_csv": str(csv_path),
            "configuration_json": str(configuration_path),
            "configuration": configuration,
            "frames_processed": frame_idx,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(proc_fps, 1),
            "speed_limit_kmh": self.speed_limit_kmh,
            "max_observed_speed_kmh": round(max_observed_speed_kmh, 1),
            "total_alert_events": (
                counts["speeding"] + counts["caution"] + counts["critical"]
            ),
            "total_speeding_alerts": counts["speeding"],
            "total_caution_events": counts["caution"],
            "total_critical_conflicts": counts["critical"],
        }

        self._write_summary_file(out_dir, summary)

        print("=" * 60)
        print("[OK] 冲突感知与测速研判处理完毕！")
        print(f"    渲染视频已保存: {out_video_path}")
        print(f"    告警明细已落盘: {jsonl_path} | {events_path} | {csv_path}")
        print(f"    最高车速: {max_observed_speed_kmh:.1f} km/h | 紧急冲突事件数: {counts['critical']}")
        print("=" * 60)

        return summary

    def process_tracks(
        self,
        tracks_path: str | Path,
        output_dir: str | Path,
        *,
        video_path: Optional[str | Path] = None,
        show_window: bool = False,
        max_frames: Optional[int] = None,
    ) -> dict:
        """从已有 tracks.jsonl 离线复盘测速、TTC 与事件去抖（无需重跑检测与跟踪）。"""
        if max_frames is not None and (type(max_frames) is not int or max_frames < 0):
            raise ValueError("max_frames 必须是 >= 0 的整数")

        tracks_file = Path(tracks_path).expanduser().resolve()
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        init_started_at = time.perf_counter()
        video_src = Path(video_path).expanduser().resolve() if video_path is not None else None
        out_video_path = out_dir / "annotated_conflict.mp4"
        jsonl_path = out_dir / "conflicts.jsonl"
        events_path = out_dir / "events.jsonl"
        csv_path = out_dir / "speeds.csv"
        configuration_path = out_dir / "configuration.json"

        def _write_initialization_failure(exc: BaseException):
            is_interrupt = isinstance(exc, KeyboardInterrupt)
            try:
                self._write_summary_file(out_dir, {
                    "schema_version": 1,
                    "input_mode": "tracks_jsonl",
                    "status": "interrupted" if is_interrupt else "failed",
                    "stop_reason": "interrupt" if is_interrupt else "initialization",
                    "error": "KeyboardInterrupt" if is_interrupt else f"{type(exc).__name__}: {exc}",
                    "source_tracks": str(tracks_file),
                    "source_video": str(video_src) if video_src is not None else str(tracks_file),
                    "frames_processed": 0,
                    "elapsed_seconds": round(max(0.0, time.perf_counter() - init_started_at), 2),
                    "annotated_output": str(out_video_path) if video_src is not None else None,
                    "conflicts_jsonl": str(jsonl_path),
                    "events_jsonl": str(events_path),
                    "speeds_csv": str(csv_path),
                    "configuration_json": str(configuration_path),
                })
            except Exception:
                pass

        resources = ExitStack()
        cap = None
        writer = None
        try:
            raw_frames = self._load_validated_tracks(tracks_file)
            track_buffer = read_tracker_config(self.tracker_config)["track_buffer"]

            if video_src is not None:
                cap = cv2.VideoCapture(str(video_src))
                resources.callback(cap.release)
                if not cap.isOpened():
                    raise RuntimeError(f"无法打开同步渲染视频: {video_src}")
                raw_fps = cap.get(cv2.CAP_PROP_FPS)
                source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
                fps = self.fps if self.fps is not None else source_fps
                if fps is None:
                    raise RuntimeError("视频缺少有效 FPS；确认帧率后通过 --fps 指定")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
                resources.callback(writer.release)
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建渲染输出视频，请检查编码器与路径: {out_video_path}")
            else:
                if self.fps is not None:
                    fps = self.fps
                elif len(raw_frames) >= 2:
                    dts = [
                        float(raw_frames[i]["timestamp_s"]) - float(raw_frames[i - 1]["timestamp_s"])
                        for i in range(1, len(raw_frames))
                    ]
                    median_dt = statistics.median(dts)
                    if not math.isfinite(median_dt) or median_dt <= 0:
                        raise ValueError("离线轨迹时间间隔无效，无法推算 FPS；请通过 --fps 显式指定")
                    raw_inferred_fps = 1.0 / median_dt
                    if not math.isfinite(raw_inferred_fps) or not (1e-3 <= raw_inferred_fps <= 1000.0):
                        raise ValueError(
                            f"离线轨迹推算 FPS ({raw_inferred_fps}) 超出允许范围 [1e-3, 1000.0]；请通过 --fps 显式指定"
                        )
                    fps = round(raw_inferred_fps, 6)
                    if not math.isfinite(fps) or not (1e-3 <= fps <= 1000.0):
                        raise ValueError(
                            f"离线轨迹推算 FPS ({fps}) 超出允许范围 [1e-3, 1000.0]；请通过 --fps 显式指定"
                        )
                else:
                    raise RuntimeError("离线轨迹不足 2 帧，无法推算 FPS；请通过 --fps 显式指定")

            self._configure_speed_estimator(fps)
            self._configure_event_tracker(fps)
            configuration = self._configuration_snapshot(
                fps, input_mode="tracks_jsonl", source_tracks=str(tracks_file)
            )
            configuration_path.write_text(
                json.dumps(configuration, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )

            jsonl_file = resources.enter_context(open(jsonl_path, "w", encoding="utf-8"))
            events_file = resources.enter_context(open(events_path, "w", encoding="utf-8"))
            csv_file = resources.enter_context(open(csv_path, "w", encoding="utf-8", newline=""))
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "schema_version",
                "frame_idx", "timestamp_s", "track_id", "category",
                "x_m", "y_m", "speed_kmh", "is_speeding", "risk_level", "ttc_s",
                "event_id", "is_reliable", "is_stationary"
            ])
        except BaseException as exc:
            try:
                resources.close()
            except Exception:
                pass
            _write_initialization_failure(exc)
            raise

        counts = {"critical": 0, "caution": 0, "speeding": 0}
        max_observed_speed_kmh = 0.0
        last_seen_frame: Dict[int, int] = {}
        source_video_record = (
            str(video_src)
            if video_src is not None
            else (str(raw_frames[0]["source"]) if raw_frames and raw_frames[0].get("source") else str(tracks_file))
        )

        def _drain_alert_events():
            self._drain_alert_events_into(events_file, counts)

        frames_processed = 0
        next_video_frame = 0
        t_start = time.perf_counter()
        loop_state = "eof"
        loop_error = ""
        main_loop_failed = False
        close_exc: Optional[BaseException] = None
        try:
            for row in raw_frames:
                if max_frames is not None and frames_processed >= max_frames:
                    loop_state = "max_frames"
                    break

                frame_idx = row["frame_index"]
                timestamp_s = float(row["timestamp_s"])
                frame = None
                if cap is not None:
                    while next_video_frame < frame_idx:
                        if not cap.grab():
                            raise RuntimeError("同步渲染视频帧数少于 tracks.jsonl 记录数")
                        next_video_frame += 1
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("同步渲染视频帧数少于 tracks.jsonl 记录数")
                    next_video_frame += 1
                    if (frame.shape[1], frame.shape[0]) != (row["width"], row["height"]):
                        raise RuntimeError("同步渲染视频分辨率与 tracks.jsonl 记录不一致")

                parsed_tracks = [
                    SimpleNamespace(
                        track_id=item["track_id"],
                        category=item["category"],
                        class_name=item["class_name"],
                        bbox_xyxy=tuple(item["bbox_xyxy"]),
                        anchor_xy=tuple(item["anchor_xy"]),
                    )
                    for item in row["tracks"]
                ]

                def _resolve_live_ids(out_of_zone: set[int]) -> set[int]:
                    for t in parsed_tracks:
                        if t.track_id is not None:
                            last_seen_frame[t.track_id] = frame_idx
                    for tid in list(last_seen_frame):
                        if frame_idx - last_seen_frame[tid] > track_buffer + 1 or tid in out_of_zone:
                            del last_seen_frame[tid]
                    return set(last_seen_frame)

                frame_max_speed, render_frame = self._process_single_frame(
                    parsed_tracks,
                    resolve_live_ids=_resolve_live_ids,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                    jsonl_file=jsonl_file,
                    csv_writer=csv_writer,
                    drain_events_cb=_drain_alert_events,
                    frame=frame,
                    writer=writer,
                )
                if frame_max_speed > max_observed_speed_kmh:
                    max_observed_speed_kmh = frame_max_speed

                frames_processed += 1

                if show_window and render_frame is not None:
                    cv2.imshow("Campus Vehicle-Pedestrian Conflict System", render_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        loop_state = "user_quit"
                        break
        except KeyboardInterrupt:
            main_loop_failed = True
            loop_state = "interrupt"
            loop_error = "KeyboardInterrupt"
            raise
        except BaseException as exc:
            main_loop_failed = True
            loop_state = "exception"
            loop_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                self.event_tracker.flush(reason=loop_state)
                _drain_alert_events()
            except Exception:
                pass
            try:
                resources.close()
            except BaseException as exc:
                close_exc = exc
                if not main_loop_failed:
                    loop_state = "exception"
                    loop_error = f"{type(exc).__name__}: {exc}"
            if loop_state in ("interrupt", "exception"):
                try:
                    self._write_summary_file(out_dir, {
                        "schema_version": 1,
                        "input_mode": "tracks_jsonl",
                        "status": "interrupted" if loop_state == "interrupt" else "failed",
                        "stop_reason": loop_state,
                        "error": loop_error,
                        "source_tracks": str(tracks_file),
                        "source_video": source_video_record,
                        "annotated_output": str(out_video_path) if video_src is not None else None,
                        "conflicts_jsonl": str(jsonl_path),
                        "events_jsonl": str(events_path.resolve()),
                        "speeds_csv": str(csv_path),
                        "configuration_json": str(configuration_path),
                        "configuration": configuration,
                        "frames_processed": frames_processed,
                        "elapsed_seconds": round(max(0.0, time.perf_counter() - t_start), 2),
                        "max_observed_speed_kmh": round(max_observed_speed_kmh, 1),
                    })
                except Exception:
                    pass
            if show_window and video_src is not None:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if close_exc is not None and not main_loop_failed:
                raise close_exc

        elapsed = max(0.0, time.perf_counter() - t_start)
        proc_fps = frames_processed / elapsed if elapsed > 0 else 0
        summary = {
            "schema_version": 1,
            "input_mode": "tracks_jsonl",
            "status": "ok",
            "stop_reason": loop_state,
            "source_tracks": str(tracks_file),
            "source_video": source_video_record,
            "annotated_output": str(out_video_path) if video_src is not None else None,
            "conflicts_jsonl": str(jsonl_path),
            "events_jsonl": str(events_path),
            "speeds_csv": str(csv_path),
            "configuration_json": str(configuration_path),
            "configuration": configuration,
            "frames_processed": frames_processed,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(proc_fps, 1),
            "speed_limit_kmh": self.speed_limit_kmh,
            "max_observed_speed_kmh": round(max_observed_speed_kmh, 1),
            "total_alert_events": (
                counts["speeding"] + counts["caution"] + counts["critical"]
            ),
            "total_speeding_alerts": counts["speeding"],
            "total_caution_events": counts["caution"],
            "total_critical_conflicts": counts["critical"],
        }
        self._write_summary_file(out_dir, summary)
        return summary
