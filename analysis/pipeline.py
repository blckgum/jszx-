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
import numpy as np

from detection.core import DEFAULT_WEIGHTS, is_blank_path
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
from .policy import DEFAULT_OBJECT_POLICY_CONFIG, object_policy, read_object_policy_config
from .provenance import collect_provenance
from .speed import SpeedEstimator, SpeedReading
from .stitching import TrackletStitcher
from .visualization import EnhancedConflictVisualizer, trajectory_color


def is_live_source(source: str | int | Path | None) -> bool:
    """判断输入源是否为摄像头设备索引或实时网络视频流。"""
    # 显式排除 bool，防止 True/False 作为 int 子类被当作摄像头索引 1/0。
    if source is None or type(source) is bool:
        return False
    if type(source) is int:
        return source >= 0
    if isinstance(source, str):
        stripped = source.strip()
        if not stripped:
            return False
        # 仅将 ASCII 纯数字看作摄像头索引，避免 Unicode 上标数字（如 '²'）被误判。
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
    # Python 中 bool 是 int 的子类，必须提前拒绝，避免 True/False 被当作摄像头索引 1/0。
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
        # 统一纯 ASCII 数字与 camera:N 两种摄像头写法，返回整数设备号给 VideoCapture。
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
            # 传给 VideoCapture 的保留原始认证 URL 以便拉流；写入摘要与日志的则脱敏账号密码。
            return stripped, redact_stream_url(stripped), True
    # 拦截纯空白 PathLike（如 Path("   ")），防止 Windows 下退化为当前目录 '.'。
    if is_blank_path(video_path):
        raise ValueError("视频输入源不能为空")
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
        # OpenCV 在无 PTS 的摄像头/RTSP 流首帧常返回 0.0：因此首帧 pts_s == 0 不视为有效 PTS，
        # 只有首帧自带正 PTS 或后续帧 PTS 严格递增时才采信 video_pts 并刷新墙钟锚点。
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
        # 用 math.nextafter 保底严格递增：当异常大 PTS（如 1e13s）超出 float64 1e-4 分辨率时，
        # previous + 1e-4 会舍入回原值，nextafter 可确保下游差分 dt 永远 > 0。
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
        speed_model: str = "ema",
        conflict_model: str = "cvm_circle",
        enable_stitching: bool = False,
        include_cyclist_vru: bool = False,
        live_reconnect_attempts: int = 3,
        live_reconnect_delay_s: float = 1.0,
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
        # 拒绝空白路径（含 Path("   ")），防止后续 Path("") 解析为当前工作目录 '.'。
        if is_blank_path(weights):
            raise ValueError("模型权重路径不能为空")
        if is_blank_path(tracker_config):
            raise ValueError("跟踪配置路径不能为空")
        if is_blank_path(object_policy_config):
            raise ValueError("对象策略配置路径不能为空")
        if type(speed_model) is not str or speed_model not in ("ema", "kalman"):
            raise ValueError("speed_model 必须是 'ema' 或 'kalman'")
        if type(conflict_model) is not str or conflict_model not in ("cvm_circle", "obb_ssm"):
            raise ValueError("conflict_model 必须是 'cvm_circle' 或 'obb_ssm'")
        if type(enable_stitching) is not bool:
            raise ValueError("enable_stitching 必须是布尔值")
        if type(include_cyclist_vru) is not bool:
            raise ValueError("include_cyclist_vru 必须是布尔值")
        if type(live_reconnect_attempts) is not int or live_reconnect_attempts < 0:
            raise ValueError("live_reconnect_attempts 必须是 >= 0 的整数")
        if (
            type(live_reconnect_delay_s) not in (int, float)
            or not math.isfinite(live_reconnect_delay_s)
            or live_reconnect_delay_s < 0
        ):
            raise ValueError("live_reconnect_delay_s 必须为有限非负数")

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
        self.speed_model = speed_model
        self.conflict_model = conflict_model
        self.enable_stitching = enable_stitching
        self.include_cyclist_vru = include_cyclist_vru
        self.live_reconnect_attempts = live_reconnect_attempts
        self.live_reconnect_delay_s = float(live_reconnect_delay_s)
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
            estimator_mode=speed_model,
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
            conflict_model=self.conflict_model,
            include_cyclist_vru=self.include_cyclist_vru,
        )
        self.stitcher: Optional[TrackletStitcher] = (
            TrackletStitcher() if self.enable_stitching else None
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
        # 在进入逐帧处理前冻结本次运行实效参数（含按 FPS 换算后的帧窗与文件指纹），
        # 后续修改 configs/*.yaml 不会污染历史实验记录。
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
            "speed_model": self.speed_model,
            "conflict_model": self.conflict_model,
            "enable_stitching": self.enable_stitching,
            "include_cyclist_vru": self.include_cyclist_vru,
            "live_reconnect_attempts": self.live_reconnect_attempts,
            "live_reconnect_delay_s": self.live_reconnect_delay_s,
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

    def _configure_conflict_and_stitcher(self):
        """每次处理新视频或新轨迹流前重置冲突分析器与缝合器状态，防止跨运行串状态。"""
        # 优先复用既有 conflict_analyzer 实例并调用 reset()，保留调用方或单测预先挂载的 mock
        if self.conflict_analyzer is None:
            self.conflict_analyzer = ConflictAnalyzer(
                speed_limit_kmh=self.speed_limit_kmh,
                ttc_critical_s=self.ttc_critical_s,
                ttc_caution_s=self.ttc_caution_s,
                horizon_s=self.horizon_s,
                object_policies=self.object_policies,
                parking_zone_max_speed_kmh=self.parking_zone_max_speed_kmh,
                conflict_model=self.conflict_model,
                include_cyclist_vru=self.include_cyclist_vru,
            )
        else:
            reset_fn = getattr(self.conflict_analyzer, "reset", None)
            if callable(reset_fn):
                reset_fn()

        # 开启缝合时复用并重置既有 stitcher 实例；关闭缝合时置为 None
        if self.enable_stitching:
            if self.stitcher is None:
                self.stitcher = TrackletStitcher()
            else:
                self.stitcher.reset()
        else:
            self.stitcher = None

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
        # 先把全文件按严格 JSON 规则读入并完成契约校验，确认时间轴/分辨率/ID 一致后再创建输出文件，
        # 避免写到半途因输入损坏报错而留下残缺 CSV/MP4。
        rows = load_jsonl_strict(tracks_file)
        expected_session: Optional[str] = None
        expected_shape: Optional[tuple[int, int]] = None
        prev_frame = -1
        prev_ts = -1.0
        # 逐帧校验与在线 TrackingSession / ConflictEventTracker 同等的不变量：
        # 单一 session_id、严格递增帧号与时间戳、恒定分辨率、帧内唯一正整数 track_id 及有限坐标。
        for line_idx, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError(f"tracks.jsonl 第 {line_idx} 行必须是 JSON 对象")
            if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
                raise ValueError(f"tracks.jsonl 第 {line_idx} 行 schema_version 必须为 1")
            sid = row.get("session_id")
            if not isinstance(sid, str) or not sid.strip():
                raise ValueError(f"tracks.jsonl 第 {line_idx} 行缺少有效 session_id")
            if expected_session is None:
                expected_session = sid
            elif sid != expected_session:
                raise ValueError("tracks.jsonl 包含多个 session_id，单次复盘仅支持同一会话")
            f_idx = row.get("frame_index")
            ts = row.get("timestamp_s")
            w = row.get("width")
            h = row.get("height")
            # 用精确 type() 检查避免 bool 被当作 0/1 通过整数校验。
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
                # 拒绝纯空白字符串，防止写出非法 speeds.csv 行并在 Schema 校验阶段失败。
                if (
                    not isinstance(cname, str)
                    or not cname.strip()
                    or not isinstance(cat, str)
                    or not cat.strip()
                ):
                    raise ValueError(f"tracks.jsonl 第 {line_idx} 行缺少 class_name 或 category")
                bbox = target.get("bbox_xyxy")
                anchor = target.get("anchor_xy")
                # 严禁 NaN/Inf/bool 混入框与锚点坐标，防止后续投影和绘图崩溃。
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
        valid_tracks = []
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

            valid_tracks.append((t, u_foot, v_foot, x_m, y_m))

        id_map: Dict[int, int] = {}
        if self.enable_stitching and self.stitcher is not None:
            observations = [
                (t.track_id, t.category, t.class_name, x_m, y_m)
                for t, _, _, x_m, y_m in valid_tracks
            ]
            id_map, _ = self.stitcher.resolve_frame(observations, timestamp_s)

        speed_readings: List[SpeedReading] = []
        reading_map: Dict[int, SpeedReading] = {}
        cov_fn = (
            getattr(self.calibrator, "ground_covariance_from_pixel", None)
            if self.speed_model == "kalman"
            else None
        )

        for t, u_foot, v_foot, x_m, y_m in valid_tracks:
            eff_id = id_map.get(t.track_id, t.track_id) if self.enable_stitching else t.track_id
            if self.speed_model == "kalman" and callable(cov_fn):
                obs_cov_m2 = cov_fn(u_foot, v_foot)
                reading = self.speed_estimator.update(
                    track_id=eff_id,
                    category=t.category,
                    timestamp_s=timestamp_s,
                    x_m=x_m,
                    y_m=y_m,
                    object_kind=t.class_name,
                    obs_cov_m2=obs_cov_m2,
                )
            else:
                reading = self.speed_estimator.update(
                    track_id=eff_id,
                    category=t.category,
                    timestamp_s=timestamp_s,
                    x_m=x_m,
                    y_m=y_m,
                    object_kind=t.class_name,
                )
            reading.in_parking_zone = self.calibrator.is_in_parking_zone(reading.x_m, reading.y_m)
            if self.enable_stitching and self.stitcher is not None:
                self.stitcher.update_kinematics(eff_id, reading, timestamp_s)
            reading_map[eff_id] = reading
            speed_readings.append(reading)

            if reading.speed_kmh > frame_max_speed:
                frame_max_speed = reading.speed_kmh

        # 根据存活池剔除已移出跟踪器或越出标定区的目标，防止速度状态字典长期残留。
        live_raw_ids = resolve_live_ids(out_of_zone_ids)
        if self.enable_stitching and self.stitcher is not None:
            mapped_raw = {self.stitcher.resolve_canonical_id(rid) for rid in live_raw_ids}
            live_ids = mapped_raw | self.stitcher.retained_canonical_ids(timestamp_s)
        else:
            live_ids = live_raw_ids
        self.speed_estimator.prune_stale_tracks(live_ids)

        conflicts = self.conflict_analyzer.analyze_frame(
            speed_readings, timestamp_s=timestamp_s
        )
        conflict_map, ped_conflict_map = build_conflict_maps(conflicts)

        # 事件去抖状态机推进后立刻弹出已冷却归档的事件写入 events.jsonl。
        event_ids = self.event_tracker.update(conflicts, frame_idx, timestamp_s)
        drain_events_cb()

        # conflicts.jsonl 保存逐帧原始评估（confirmed 帧回填 event_id，未确认瞬时帧为 null）。
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

        # speeds.csv 按目标逐行落盘：ttc_s 用 is not None 判断，避免 TTC=0.0 时被当假值留空。
        for r in speed_readings:
            c = conflict_map.get(r.track_id) or ped_conflict_map.get(r.track_id)
            risk_lvl = c.risk_level.name if c else "LEVEL_0_SAFE"
            ttc_val = round(c.ttc_seconds, 2) if (c and c.ttc_seconds is not None) else ""
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

            # OBB-SSM 模式下先渲染路面车辆旋转包围盒，避免遮挡后续轨迹与标签。
            if self.conflict_model == "obb_ssm":
                for r in speed_readings:
                    if r.category != "vehicle":
                        continue
                    pol = object_policy(r.object_kind, r.category, self.object_policies)
                    if pol.kind == "pedestrian":
                        continue
                    c = conflict_map.get(r.track_id) or ped_conflict_map.get(r.track_id)
                    obb_color = (
                        trajectory_color(c)
                        if c is not None and c.risk_level >= RiskLevel.LEVEL_2_CAUTION
                        else (0, 220, 180)
                    )
                    self.visualizer.draw_vehicle_obb(
                        render_frame,
                        r,
                        length_m=pol.length_m,
                        width_m=pol.width_m,
                        color=obb_color,
                    )

            for r in speed_readings:
                c = conflict_map.get(r.track_id) or ped_conflict_map.get(r.track_id)
                self.visualizer.draw_future_trajectory(
                    render_frame, r, color=trajectory_color(c)
                )

            for t in tracks:
                if t.track_id is None:
                    continue
                if self.enable_stitching and self.stitcher is not None:
                    # 仅对本帧通过标定区筛选并完成缝合映射的区内目标渲染徽章，禁止区外旧 raw_id 回退撞名
                    if t.track_id not in id_map:
                        continue
                    eff_id = id_map[t.track_id]
                else:
                    # 未开启缝合时同样跳过已进入 out_of_zone_ids 的标定区外目标
                    if t.track_id in out_of_zone_ids:
                        continue
                    eff_id = t.track_id
                r = reading_map.get(eff_id)
                if r:
                    c = conflict_map.get(eff_id) or ped_conflict_map.get(eff_id)
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

    def _reconnect_live_capture(
        self,
        cap_holder: list,
        capture_arg: int | str,
        src_label: str,
        expected_width: int,
        expected_height: int,
    ) -> tuple[bool, Optional[object]]:
        """实时流断流后连续重连，校验打开状态、首帧解码及分辨率匹配。"""
        for attempt in range(1, self.live_reconnect_attempts + 1):
            old_cap = cap_holder[0]
            if old_cap is not None:
                try:
                    old_cap.release()
                finally:
                    cap_holder[0] = None

            if self.live_reconnect_delay_s > 0:
                time.sleep(self.live_reconnect_delay_s)

            print(
                f"[!] 实时流断流，尝试第 {attempt}/{self.live_reconnect_attempts} 次重连: {src_label}"
            )
            new_cap = cv2.VideoCapture(capture_arg)
            cap_holder[0] = new_cap
            if not new_cap.isOpened():
                try:
                    new_cap.release()
                finally:
                    cap_holder[0] = None
                continue

            ok_re, frame_re = new_cap.read()
            if (
                not ok_re
                or frame_re is None
                or not hasattr(frame_re, "shape")
                or len(frame_re.shape) < 2
                or (frame_re.shape[1], frame_re.shape[0]) != (expected_width, expected_height)
            ):
                try:
                    new_cap.release()
                finally:
                    cap_holder[0] = None
                continue

            return True, frame_re
        return False, None

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
        # 拒绝空白输出目录（含 Path("   ")），防止 Path("") 解析为当前工作目录 '.' 并污染工作区。
        if is_blank_path(output_dir):
            raise ValueError("输出目录不能为空")

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

        # ExitStack 统一登记资源生命周期：通过可变句柄容器跟踪重连后的最新采集对象，
        # 确保初始化阶段与主循环异常都能正确释放当前存活的 cap。
        cap_holder = [cap]

        def _release_active_cap():
            active_cap = cap_holder[0]
            if active_cap is not None:
                active_cap.release()

        resources = ExitStack()
        resources.callback(_release_active_cap)
        try:
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            if is_live_stream:
                # 实时流经常缺省 FPS 或返回 0/异常大值；合理区间外保守按 25 FPS 换算秒级窗，
                # 实际时间戳仍由 LiveStreamClock（PTS 优先、墙钟兜底）单调推进。
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

            # 在加载 PyTorch 模型与创建 VideoWriter 前先拦截无效的 0x0 或 NaN 容器元数据。
            raw_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            raw_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if not (math.isfinite(raw_w) and math.isfinite(raw_h)) or raw_w <= 0 or raw_h <= 0:
                raise RuntimeError("视频分辨率元数据无效")
            width = int(raw_w)
            height = int(raw_h)
            if width <= 0 or height <= 0:
                raise RuntimeError("视频分辨率元数据无效")

            self._configure_speed_estimator(fps)
            self._configure_event_tracker(fps)
            self._configure_conflict_and_stitcher()
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
        live_reconnect_count = 0
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
                    # 仅实时流在断流后触发有界重连；本地视频文件遇 EOF 或坏帧直接结束。
                    if is_live_stream and self.live_reconnect_attempts > 0:
                        reconnected, frame_re = self._reconnect_live_capture(
                            cap_holder, capture_arg, src_label, width, height
                        )
                        if not reconnected or frame_re is None:
                            break
                        cap = cap_holder[0]
                        frame = frame_re
                        live_reconnect_count += 1
                    else:
                        break
                # OpenCV VideoWriter 遇到与初始化尺寸不一致的帧只会警告并静默丢帧，
                # 必须显式校验解码帧分辨率与容器元数据一致，避免悄悄写出空视频。
                if not hasattr(frame, "shape") or len(frame.shape) < 2 or (frame.shape[1], frame.shape[0]) != (width, height):
                    raise RuntimeError("实际解码帧分辨率与视频元数据不一致")

                raw_pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                pts_ms = (
                    float(raw_pts_ms)
                    if (
                        type(raw_pts_ms) is not bool
                        and not isinstance(raw_pts_ms, np.bool_)
                        and isinstance(raw_pts_ms, (int, float, np.integer, np.floating))
                    )
                    else float("nan")
                )
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
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
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
            # 必须在 resources.close() 关闭文件句柄前刷出仍在冷却/存续中的事件，
            # 确保中途异常或 Ctrl+C 中断时已确认告警事件同样完整落盘。
            try:
                self.event_tracker.flush(reason=loop_state)
                _drain_alert_events()
            except Exception:
                pass
            try:
                resources.close()
            except BaseException as exc:
                # 若主循环本身未出错而收尾关闭失败（如 VideoWriter 封装失败），记录关闭异常；
                # 若主循环已抛错，则保留原始主异常以免被次生关闭异常掩盖。
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
                        "live_reconnect_count": live_reconnect_count,
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
            "stitched_track_count": (
                self.stitcher.stitched_track_count if self.stitcher is not None else 0
            ),
            "live_reconnect_count": live_reconnect_count,
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
        # 拒绝空白输出目录（含 Path("   ")），防止 Path("") 解析为当前工作目录 '.' 并污染工作区。
        if is_blank_path(output_dir):
            raise ValueError("输出目录不能为空")
        # 拒绝空白轨迹路径（含 Path("   ")），避免 Path("") 解析成当前工作目录 '.'。
        if is_blank_path(tracks_path):
            raise ValueError("tracks.jsonl 路径不能为空")
        if video_path is not None and (
            type(video_path) is bool
            or is_live_source(video_path)
            or is_blank_path(video_path)
        ):
            raise ValueError("离线轨迹复盘仅支持非空本地视频文件同步渲染，不支持摄像头索引或实时推流")

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
                raw_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                raw_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if not (math.isfinite(raw_w) and math.isfinite(raw_h)) or raw_w <= 0 or raw_h <= 0:
                    raise RuntimeError("同步渲染视频分辨率元数据无效")
                width = int(raw_w)
                height = int(raw_h)
                if width <= 0 or height <= 0:
                    raise RuntimeError("同步渲染视频分辨率元数据无效")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
                resources.callback(writer.release)
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建渲染输出视频，请检查编码器与路径: {out_video_path}")
            else:
                if self.fps is not None:
                    fps = self.fps
                elif len(raw_frames) >= 2:
                    # 无视频且未传 --fps 时，用相邻帧时间戳差的中位数反推 FPS（抗偶发跳帧抖动），
                    # 并将有效范围约束在 [1e-3, 1000.0] 防止极端时间戳触发溢出或舍入归零。
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
            self._configure_conflict_and_stitcher()
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
                    # 若 tracks.jsonl 的 frame_index 存在跳帧，用 cap.grab() 仅推进解复用器而不做完整解码，
                    # 确保绘制叠加的画面与轨迹帧号严格对齐。
                    while next_video_frame < frame_idx:
                        if not cap.grab():
                            raise RuntimeError("同步渲染视频帧数少于 tracks.jsonl 记录数")
                        next_video_frame += 1
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("同步渲染视频帧数少于 tracks.jsonl 记录数")
                    next_video_frame += 1
                    if not hasattr(frame, "shape") or len(frame.shape) < 2 or (frame.shape[1], frame.shape[0]) != (row["width"], row["height"]):
                        raise RuntimeError("同步渲染视频分辨率与 tracks.jsonl 记录不一致")
                    # VideoWriter 不会对尺寸不匹配的帧抛异常，需主动校验以防输出静音/空帧 MP4。
                    if (frame.shape[1], frame.shape[0]) != (width, height):
                        raise RuntimeError("同步渲染视频实际帧分辨率与容器元数据不一致")

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
                    # 复现 Ultralytics BYTETracker 在线存活池边界：
                    # lost_stracks 在移除边界帧 (gap == track_buffer + 1) 仍保留 ID 参与次帧重关联，
                    # 仅当缺席跨度严格大于 track_buffer + 1 或越出标定区时才从速度滤波器剪枝。
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
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
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
            # 必须在 resources.close() 关闭文件句柄前刷出仍在冷却/存续中的事件，
            # 确保中途异常或 Ctrl+C 中断时已确认告警事件同样完整落盘。
            try:
                self.event_tracker.flush(reason=loop_state)
                _drain_alert_events()
            except Exception:
                pass
            try:
                resources.close()
            except BaseException as exc:
                # 若主循环本身未出错而收尾关闭失败（如 VideoWriter 封装失败），记录关闭异常；
                # 若主循环已抛错，则保留原始主异常以免被次生关闭异常掩盖。
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
            "stitched_track_count": (
                self.stitcher.stitched_track_count if self.stitcher is not None else 0
            ),
            "live_reconnect_count": 0,
        }
        self._write_summary_file(out_dir, summary)
        return summary
