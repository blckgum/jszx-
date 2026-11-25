# -*- coding: utf-8 -*-
"""
实时视频流丢旧帧缓冲、双端结构化预警推送与后端 HTTP/WebSocket 服务模块
=============================================================================
提供：
1. LatestFrameCapture：后台守护线程持续抓取最新帧并丢弃积压旧帧；
2. format_dual_end_alerts：按驾驶员端与路旁行人端拆解结构化预警；
3. AnalysisService：历史运行安全查询与实时后台分析任务调度（含最新 JPEG 与告警队列）；
4. create_http_server：基于标准库 ThreadingHTTPServer 的零三方 Web 依赖默认服务；
5. create_fastapi_app：可选懒导入 FastAPI + WebSocket 扩展接口。
"""

from __future__ import annotations

import argparse
import base64
from bisect import bisect_left
from collections import deque
import csv
from dataclasses import dataclass, field
import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

import cv2
import numpy as np

from detection.core import DEFAULT_WEIGHTS, is_blank_path
from tracking.core import DEFAULT_TRACKER
from .calibration import RoadCalibrator
from .conflict import RiskLevel
from .pipeline import ConflictPerceptionPipeline, parse_video_source
from .policy import DEFAULT_OBJECT_POLICY_CONFIG
from .schema import (
    SchemaValidationError,
    _reject_non_finite_tree,
    load_json_strict,
    load_jsonl_strict,
)


DEFAULT_WEB_ROOT = Path(__file__).resolve().parent / "web"

_WEB_MIME_TYPES: Dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
_VIDEO_PRESET_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_CALIB_PRESET_EXTS = {".yaml", ".yml"}
_NON_CALIB_CONFIG_NAMES = {
    "bytetrack.yaml",
    "motion.yaml",
    "object_policy.yaml",
    "ground_truth.example.yaml",
    "traffic.example.yaml",
}
_ALLOWED_RUN_EXPORT_FILES: Dict[str, str] = {
    "summary.json": "application/json; charset=utf-8",
    "speeds.csv": "text/csv; charset=utf-8",
    "conflicts.jsonl": "application/x-ndjson; charset=utf-8",
    "events.jsonl": "application/x-ndjson; charset=utf-8",
    "annotated_conflict.mp4": "video/mp4",
}
_ALLOWED_DIALOG_KINDS = ("video", "calib", "folder")
_ALLOWED_JOB_REQUEST_KEYS = frozenset({
    "source",
    "calib_path",
    "demo",
    "allow_legacy_calib",
    "output_dir",
    "max_frames",
    "speed_limit_kmh",
    "speed_model",
    "conflict_model",
    "enable_stitching",
    "include_cyclist_vru",
    "compensate_anchor_bias",
    "live_drop_stale_frames",
    "live_reconnect_attempts",
    "live_reconnect_delay_s",
    "weights",
    "tracker_config",
    "object_policy_config",
    "fps",
    "device",
    "conf",
})
_DIALOG_LOCK = threading.Lock()
_VIDEO_STREAM_CHUNK_SIZE = 65536


class _ServiceForbiddenError(PermissionError):
    """未开启关闭权限等禁止操作异常（映射 HTTP 403）。"""


class _ServiceConflictError(RuntimeError):
    """运行目录正被活动任务占用导致的冲突异常（映射 HTTP 409）。"""


class _DialogUnavailableError(RuntimeError):
    """当前环境无法打开图形文件对话框（映射 HTTP 501）。"""


def _validate_single_segment_filename(filename: Any, *, label: str = "filename") -> str:
    """校验单层文件名不含目录分隔符、空字符或相对穿越片段。"""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError(f"{label} 不能为空")
    stripped = filename.strip()
    decoded = unquote(stripped)
    twice = unquote(decoded)
    for candidate in (filename, stripped, decoded, twice):
        if (
            not candidate.strip()
            or candidate in (".", "..")
            or candidate.startswith(".")
            or ".." in candidate
            or "/" in candidate
            or "\\" in candidate
            or ":" in candidate
            or "\x00" in candidate
            or Path(candidate).name != candidate
        ):
            raise ValueError(f"非法的 {label}（禁止目录穿越）: {filename}")
    return stripped


def _decode_and_validate_asset_subpath(raw_subpath: str) -> str:
    """逐段解码并校验 /assets/ 子路径，在路径拼接前阻断目录穿越与非法分隔符。"""
    if not isinstance(raw_subpath, str) or not raw_subpath.strip():
        raise ValueError("静态资源路径不能为空")
    if "\x00" in raw_subpath or "\\" in raw_subpath or ":" in raw_subpath or raw_subpath.startswith("/"):
        raise ValueError(f"非法的静态资源路径: {raw_subpath}")

    decoded_parts: List[str] = []
    for raw_seg in raw_subpath.split("/"):
        seg = unquote(raw_seg)
        twice = unquote(seg)
        if (
            not seg
            or not seg.strip()
            or seg in (".", "..")
            or twice in (".", "..")
            or ".." in seg
            or ".." in twice
            or "/" in seg
            or "/" in twice
            or "\\" in seg
            or "\\" in twice
            or ":" in seg
            or ":" in twice
            or "\x00" in seg
            or "\x00" in twice
        ):
            raise ValueError(f"非法的静态资源路径: {raw_subpath}")
        decoded_parts.append(seg)
    return "/".join(decoded_parts)


def _parse_http_byte_range(range_header: str, total_size: int) -> Optional[tuple[int, int]]:
    """解析单个 HTTP Range 字节区间；不可满足或格式非法返回 None（对应 416）。"""
    if total_size <= 0 or not isinstance(range_header, str):
        return None
    stripped = range_header.strip()
    if not stripped.lower().startswith("bytes="):
        return None
    spec = stripped[6:].strip()
    if not spec or "," in spec or "-" not in spec:
        return None
    start_str, _, end_str = spec.partition("-")
    if "-" in end_str:
        return None
    start_str = start_str.strip()
    end_str = end_str.strip()
    if not start_str and not end_str:
        return None
    if not start_str:
        if not (end_str.isascii() and end_str.isdigit()):
            return None
        suffix_len = int(end_str)
        if suffix_len <= 0:
            return None
        start = max(0, total_size - suffix_len)
        return start, total_size - 1
    if not (start_str.isascii() and start_str.isdigit()):
        return None
    start = int(start_str)
    if start >= total_size:
        return None
    if not end_str:
        return start, total_size - 1
    if not (end_str.isascii() and end_str.isdigit()):
        return None
    end = int(end_str)
    if start > end:
        return None
    return start, min(end, total_size - 1)


def _iter_file_range(
    path: Path, start: int, end: int, chunk_size: int = _VIDEO_STREAM_CHUNK_SIZE
):
    """按固定块大小流式读取文件闭区间 [start, end]。"""
    remaining = max(0, end - start + 1)
    if remaining == 0:
        return
    with path.open("rb") as stream:
        stream.seek(start)
        while remaining > 0:
            chunk = stream.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _is_finite_real(value: Any) -> bool:
    """判断输入是否为非布尔的有限实数（兼容 Python 与 NumPy 标量）。"""
    if value is None or type(value) is bool or isinstance(value, np.bool_):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return False


def _safe_round(value: Any, digits: int = 3) -> Optional[float]:
    """将有限实数舍入为标准 float，非有限或布尔值安全返回 None。"""
    if not _is_finite_real(value):
        return None
    return round(float(value), digits)


def _json_dumps(payload: Any) -> str:
    """严格禁止 NaN/Infinity 的 JSON 序列化封装。"""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"标准 JSON 不允许非有限常量: {value}")


def _iso_now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class LatestFrameCapture:
    """后台守护线程持续抓取 VideoCapture 仅保留最新帧，防止实时流队列积压延迟。"""

    def __init__(self, capture: Any, *, read_timeout_s: float = 2.0):
        # 严格校验超时参数，防止传入 bool/NaN/非正数导致条件变量等待行为异常。
        if not _is_finite_real(read_timeout_s) or float(read_timeout_s) <= 0.0:
            raise ValueError("read_timeout_s 必须为有限正数")

        # 既支持包装已打开的 cv2.VideoCapture / Mock 对象，也支持直接传入视频源标识。
        if hasattr(capture, "read") and hasattr(capture, "isOpened"):
            self._cap = capture
        else:
            if type(capture) is bool:
                raise ValueError("视频输入源不能为布尔值")
            if is_blank_path(capture):
                raise ValueError("视频输入源不能为空")
            self._cap = cv2.VideoCapture(capture)

        self._read_timeout_s = float(read_timeout_s)
        self.dropped_frames: int = 0
        self._cond = threading.Condition()
        self._stop_event = threading.Event()
        self._latest_frame: Any = None
        self._latest_pts_ms: float = float("nan")
        self._last_read_pts_ms: float = float("nan")
        self._has_unread: bool = False
        self._ended: bool = False
        self._released: bool = False
        self._reader_error: Optional[Exception] = None

        # 守护线程随进程退出，并在 release() 时主动停止。
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="LatestFrameCaptureReader",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _parse_pts_ms(raw_pts: Any) -> float:
        if (
            type(raw_pts) is not bool
            and not isinstance(raw_pts, np.bool_)
            and isinstance(raw_pts, (int, float, np.integer, np.floating))
        ):
            val = float(raw_pts)
            if math.isfinite(val):
                return val
        return float("nan")

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ok, frame = self._cap.read()
                pts_ms = self._parse_pts_ms(self._cap.get(cv2.CAP_PROP_POS_MSEC)) if (ok and frame is not None) else float("nan")
            except Exception as exc:
                with self._cond:
                    self._reader_error = exc
                    self._ended = True
                    self._cond.notify_all()
                break

            if self._stop_event.is_set():
                break

            with self._cond:
                if not ok or frame is None:
                    self._ended = True
                    self._cond.notify_all()
                    break
                # 若上一帧尚未被消费者取走即被新帧覆盖，则记为一次丢弃旧帧。
                if self._has_unread:
                    self.dropped_frames += 1
                self._latest_frame = frame
                self._latest_pts_ms = pts_ms
                self._has_unread = True
                self._cond.notify_all()

    def read_with_pts(self, timeout_s: Optional[float] = None) -> tuple[bool, Any, float]:
        """原子获取最新可用帧及捕获时刻匹配的 PTS(ms)；无可读帧时返回 (False, None, nan)。"""
        if timeout_s is None:
            wait_limit = self._read_timeout_s
        else:
            if not _is_finite_real(timeout_s) or float(timeout_s) <= 0.0:
                raise ValueError("timeout_s 必须为有限正数")
            wait_limit = float(timeout_s)

        deadline = time.monotonic() + wait_limit
        with self._cond:
            while not self._has_unread and not self._ended and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False, None, float("nan")
                self._cond.wait(timeout=remaining)

            # 优先交付已捕获的最后一帧，确保有限测试流末尾不会漏掉最新帧。
            if self._has_unread and not self._stop_event.is_set():
                frame = self._latest_frame
                pts_ms = self._latest_pts_ms
                self._latest_frame = None
                self._latest_pts_ms = float("nan")
                self._has_unread = False
                self._last_read_pts_ms = pts_ms
                return True, frame, pts_ms
            return False, None, float("nan")

    def read(self, timeout_s: Optional[float] = None) -> tuple[bool, Any]:
        """获取最新可用帧；若尚无新帧则在超时范围内阻塞等待，已结束且无未读帧时返回 (False, None)。"""
        ok, frame, _ = self.read_with_pts(timeout_s=timeout_s)
        return ok, frame

    def get(self, prop_id: int) -> Any:
        """透传底层采集对象属性查询；对 PTS 查询返回最近消费帧匹配的原子时间戳。"""
        if self._released:
            return 0.0
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            with self._cond:
                return self._last_read_pts_ms
        return self._cap.get(prop_id)

    def isOpened(self) -> bool:
        """查询底层采集对象是否保持打开且未主动释放。"""
        if self._released or self._stop_event.is_set():
            return False
        return bool(self._cap.isOpened())

    def release(self) -> None:
        """幂等停止后台抓帧线程并释放底层采集句柄。"""
        with self._cond:
            if self._released:
                return
            self._released = True
            self._stop_event.set()
            self._latest_frame = None
            self._has_unread = False
            self._cond.notify_all()

        # 先等待抓帧线程短时退出再 release 底层 VideoCapture，避免并发释放 C++ 对象。
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=0.5)
        self._cap.release()

    def __enter__(self) -> "LatestFrameCapture":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _extract_conflict_field(conflict: Any, name: str, default: Any = None) -> Any:
    if isinstance(conflict, dict):
        return conflict.get(name, default)
    return getattr(conflict, name, default)


def _normalize_risk_level(raw_level: Any) -> RiskLevel:
    if isinstance(raw_level, RiskLevel):
        return raw_level
    if type(raw_level) is int and 0 <= raw_level <= 3:
        return RiskLevel(raw_level)
    if isinstance(raw_level, str):
        stripped = raw_level.strip().upper()
        for member in RiskLevel:
            if stripped == member.name:
                return member
    return RiskLevel(int(raw_level))


def format_dual_end_alerts(
    conflicts: Iterable[Any],
    event_ids: Optional[dict] = None,
    timestamp_s: float = 0.0,
) -> dict:
    """将单帧冲突评估拆解为驾驶员端 (driver_alerts) 与路旁行人端 (pedestrian_alerts) 双端预警。"""
    if not _is_finite_real(timestamp_s) or float(timestamp_s) < 0.0:
        raise ValueError("timestamp_s 必须为非负有限数值")

    ts_rounded = round(float(timestamp_s), 3)
    raw_items: List[tuple[tuple[int, float], Any]] = []
    for item in conflicts or ():
        level = _normalize_risk_level(_extract_conflict_field(item, "risk_level", 0))
        if level <= RiskLevel.LEVEL_0_SAFE:
            continue
        raw_ttc = _extract_conflict_field(item, "ttc_seconds", None)
        ttc_sort = float(raw_ttc) if _is_finite_real(raw_ttc) else 9999.0
        raw_items.append(((-int(level), ttc_sort), item))

    # 按高风险等级优先、最小 TTC 优先排序，保证双端播报优先级一致。
    raw_items.sort(key=lambda pair: pair[0])

    driver_alerts: List[dict] = []
    pedestrian_alerts: List[dict] = []
    highest_level = RiskLevel.LEVEL_0_SAFE

    for _, c in raw_items:
        level = _normalize_risk_level(_extract_conflict_field(c, "risk_level", 0))
        if level > highest_level:
            highest_level = level

        veh_id = int(_extract_conflict_field(c, "veh_track_id"))
        raw_ped_id = _extract_conflict_field(c, "ped_track_id", None)
        ped_id = int(raw_ped_id) if raw_ped_id is not None else None
        veh_kind = str(_extract_conflict_field(c, "vehicle_kind", "motor_vehicle") or "motor_vehicle")
        target_kind = str(_extract_conflict_field(c, "target_kind", "pedestrian") or "pedestrian")
        veh_speed_kmh = _safe_round(_extract_conflict_field(c, "veh_speed_kmh", None), 1)
        ttc_s = _safe_round(_extract_conflict_field(c, "ttc_seconds", None), 2)
        min_dist_m = _safe_round(_extract_conflict_field(c, "min_distance_m", None), 2)
        cur_dist_m = _safe_round(_extract_conflict_field(c, "current_distance_m", None), 2)
        mttc_s = _safe_round(_extract_conflict_field(c, "mttc_seconds", None), 2)
        drac_ms2 = _safe_round(_extract_conflict_field(c, "drac_ms2", None), 2)
        pet_s = _safe_round(_extract_conflict_field(c, "pet_seconds", None), 2)
        prob = _safe_round(_extract_conflict_field(c, "collision_prob", None), 3)

        event_id = None
        if event_ids is not None:
            raw_eid = event_ids.get((veh_id, ped_id))
            if type(raw_eid) is int and raw_eid >= 1:
                event_id = raw_eid

        target_label = "骑行者" if target_kind in ("bicycle", "motorcycle") else "行人"
        speed_txt = f"{veh_speed_kmh:.1f} km/h" if veh_speed_kmh is not None else "未知车速"
        ttc_txt = f"TTC={ttc_s:.2f}s" if ttc_s is not None else "TTC 未知"

        base_fields = {
            "event_id": event_id,
            "veh_track_id": veh_id,
            "ped_track_id": ped_id,
            "vehicle_kind": veh_kind,
            "target_kind": target_kind,
            "risk_level": int(level),
            "risk_name": level.name,
            "veh_speed_kmh": veh_speed_kmh,
            "ttc_seconds": ttc_s,
            "min_distance_m": min_dist_m,
            "current_distance_m": cur_dist_m,
            "mttc_seconds": mttc_s,
            "drac_ms2": drac_ms2,
            "pet_seconds": pet_s,
            "collision_prob": prob,
        }

        if level == RiskLevel.LEVEL_1_SPEEDING:
            driver_alerts.append({
                **base_fields,
                "role": "driver",
                "message": f"超速提示：车辆 #{veh_id} ({veh_kind}) 当前车速 {speed_txt}，已超出校园限速。",
                "advice": "请立即减速慢行，将车速控制在校内限速以内。",
            })
        elif level == RiskLevel.LEVEL_2_CAUTION:
            driver_alerts.append({
                **base_fields,
                "role": "driver",
                "message": f"注意预警：车辆 #{veh_id} ({speed_txt}) 与{target_label} #{ped_id} 轨迹汇聚（{ttc_txt}）。",
                "advice": "请提前备刹减速，注意观察前方弱势交通参与者动态。",
            })
            if ped_id is not None:
                pedestrian_alerts.append({
                    **base_fields,
                    "role": "pedestrian",
                    "message": f"路旁提示：{target_label} #{ped_id} 请注意车辆 #{veh_id} ({speed_txt}) 正在接近（{ttc_txt}）。",
                    "advice": "请停步观望或沿安全区域通行，避免盲目横穿。",
                })
        elif level == RiskLevel.LEVEL_3_CRITICAL:
            driver_alerts.append({
                **base_fields,
                "role": "driver",
                "message": f"紧急碰撞预警：车辆 #{veh_id} ({speed_txt}) 与{target_label} #{ped_id} 距离过近（{ttc_txt}）。",
                "advice": "请立即紧急制动停车避让。",
            })
            if ped_id is not None:
                pedestrian_alerts.append({
                    **base_fields,
                    "role": "pedestrian",
                    "message": f"紧急避险预警：{target_label} #{ped_id} 请立即注意车辆 #{veh_id} ({speed_txt}) 快速逼近（{ttc_txt}）。",
                    "advice": "请立即止步并向路侧安全区域退让避险。",
                })

    return {
        "schema_version": 1,
        "timestamp_s": ts_rounded,
        "highest_risk_level": int(highest_level),
        "highest_risk_name": highest_level.name,
        "alert_count": len(driver_alerts) + len(pedestrian_alerts),
        "driver_alerts": driver_alerts,
        "pedestrian_alerts": pedestrian_alerts,
    }


@dataclass
class _ServiceJob:
    """后台实时分析任务运行时状态与帧/告警缓存。"""

    job_id: str
    source: str
    output_dir: str
    status: str = "queued"
    created_at: str = field(default_factory=_iso_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    frames_processed: int = 0
    latest_timestamp_s: Optional[float] = None
    latest_jpeg: Optional[bytes] = None
    latest_twin: Optional[dict] = None
    frame_seq: int = 0
    alert_seq: int = 0
    alert_queue: deque = field(default_factory=deque)
    summary: Optional[dict] = None
    error: Optional[str] = None
    speed_model: str = "ema"
    conflict_model: str = "cvm_circle"
    enable_stitching: bool = False
    include_cyclist_vru: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)
    cond: threading.Condition = field(default_factory=threading.Condition)
    thread: Optional[threading.Thread] = None


class AnalysisService:
    """历史分析产物只读查询与实时视频流作业调度服务。"""

    def __init__(
        self,
        runs_root: str | Path = "runs/analysis",
        *,
        default_calib: str | Path = "configs/calibration.yaml",
        web_root: Optional[str | Path] = None,
        max_alert_queue: int = 200,
        desktop_mode: bool = False,
        allow_shutdown: bool = False,
        shutdown_callback: Optional[Callable[[], None]] = None,
    ):
        if is_blank_path(runs_root) or type(runs_root) is bool or not isinstance(runs_root, (str, os.PathLike)):
            raise ValueError("runs_root 必须是非空目录路径")
        if is_blank_path(default_calib) or type(default_calib) is bool or not isinstance(default_calib, (str, os.PathLike)):
            raise ValueError("default_calib 必须是非空文件路径")
        if web_root is not None and (
            is_blank_path(web_root) or type(web_root) is bool or not isinstance(web_root, (str, os.PathLike))
        ):
            raise ValueError("web_root 必须是非空目录路径")
        if type(max_alert_queue) is not int or max_alert_queue < 1:
            raise ValueError("max_alert_queue 必须是 >= 1 的整数")
        if type(desktop_mode) is not bool:
            raise ValueError("desktop_mode 必须是布尔值")
        if type(allow_shutdown) is not bool:
            raise ValueError("allow_shutdown 必须是布尔值")
        if shutdown_callback is not None and not callable(shutdown_callback):
            raise ValueError("shutdown_callback 必须可调用")

        self.runs_root = Path(runs_root).expanduser().resolve()
        self.default_calib = Path(default_calib).expanduser().resolve()
        self.web_root = (
            Path(web_root).expanduser().resolve() if web_root is not None else DEFAULT_WEB_ROOT
        )
        self.max_alert_queue = max_alert_queue
        self.desktop_mode = desktop_mode
        self.allow_shutdown = allow_shutdown
        self._shutdown_callback = shutdown_callback
        self._lock = threading.Lock()
        self._jobs: Dict[str, _ServiceJob] = {}
        self._active_output_dirs: Dict[str, tuple[str, Path]] = {}

    def resolve_web_asset(self, rel_path: str) -> tuple[Path, str]:
        """严格校验并解析 Web 控制台静态资源路径与 MIME 类型。"""
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ValueError("静态资源路径不能为空")

        decoded = unquote(rel_path)
        twice = unquote(decoded)
        for candidate in (rel_path, decoded, twice):
            if (
                not candidate.strip()
                or candidate.startswith("/")
                or "\\" in candidate
                or ":" in candidate
                or "\x00" in candidate
                or ".." in candidate
            ):
                raise ValueError(f"非法的静态资源路径: {rel_path}")

        parts = decoded.split("/")
        if any(not part or part in (".", "..") for part in parts):
            raise ValueError(f"非法的静态资源路径: {rel_path}")

        base_root = self.web_root.resolve()
        target = (base_root / "/".join(parts)).resolve()
        # 确认解析后的真实路径仍在 web_root 目录内部且不是根目录自身。
        if target == base_root or not target.is_relative_to(base_root):
            raise ValueError(f"静态资源路径越出根目录: {rel_path}")

        media_type = _WEB_MIME_TYPES.get(target.suffix.lower())
        if media_type is None:
            raise ValueError(f"不支持的静态资源类型: {target.suffix or '<none>'}")
        if not target.is_file():
            raise FileNotFoundError(f"未找到静态资源文件: {rel_path}")
        return target, media_type

    def _resolve_run_dir(self, run_id: str) -> Path:
        """校验 run_id 为单层安全目录名，彻底阻止目录穿越与空白路径。"""
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id 不能为空")
        if (
            run_id in (".", "..")
            or "/" in run_id
            or "\\" in run_id
            or "\x00" in run_id
            or ":" in run_id
            or Path(run_id).name != run_id
        ):
            raise ValueError(f"非法的 run_id（禁止路径穿越）: {run_id}")

        candidate = (self.runs_root / run_id).resolve()
        # 双重边界检查：确保符号链接或相对片段不会逃逸到 runs_root 之外。
        if candidate == self.runs_root or not candidate.is_relative_to(self.runs_root):
            raise ValueError(f"非法的 run_id（越出运行根目录）: {run_id}")
        if not candidate.is_dir():
            raise FileNotFoundError(f"未找到指定运行目录: {run_id}")
        return candidate

    def list_runs(self) -> List[dict]:
        """扫描 runs_root 下包含 summary.json 的历史运行记录（按修改时间倒序）。"""
        if not self.runs_root.is_dir():
            return []
        items: List[tuple[float, dict]] = []
        for entry in self.runs_root.iterdir():
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            summary_path = entry / "summary.json"
            if not summary_path.is_file():
                continue
            mtime = entry.stat().st_mtime
            try:
                summary = load_json_strict(summary_path)
            except (SchemaValidationError, OSError):
                items.append((
                    mtime,
                    {
                        "run_id": entry.name,
                        "status": "invalid",
                        "frames_processed": 0,
                    },
                ))
                continue
            if not isinstance(summary, dict):
                continue
            items.append((
                mtime,
                {
                    "run_id": entry.name,
                    "status": summary.get("status", "unknown"),
                    "stop_reason": summary.get("stop_reason"),
                    "source_video": summary.get("source_video"),
                    "frames_processed": summary.get("frames_processed", 0),
                    "elapsed_seconds": summary.get("elapsed_seconds"),
                    "max_observed_speed_kmh": summary.get("max_observed_speed_kmh"),
                    "total_alert_events": summary.get("total_alert_events", 0),
                    "total_critical_conflicts": summary.get("total_critical_conflicts", 0),
                    "has_video": (entry / "annotated_conflict.mp4").is_file(),
                },
            ))
        items.sort(key=lambda pair: (pair[0], pair[1]["run_id"]), reverse=True)
        return [item for _, item in items]

    def get_run_summary(self, run_id: str) -> dict:
        """严格读取指定运行目录的 summary.json。"""
        run_dir = self._resolve_run_dir(run_id)
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"运行目录缺少 summary.json: {run_id}")
        data = load_json_strict(summary_path)
        if not isinstance(data, dict):
            raise SchemaValidationError("summary.json 顶层必须是 JSON 对象")
        frame_map = self._load_video_frame_map(run_dir)
        out = {
            "run_id": run_id,
            "has_video": (run_dir / "annotated_conflict.mp4").is_file(),
            "has_video_frame_map": frame_map is not None,
            **data,
        }
        if frame_map is not None:
            out["video_frame_count"] = len(frame_map)
        return out

    @staticmethod
    def _validate_frame_window(
        frame_start: Optional[int], frame_end: Optional[int]
    ) -> tuple[Optional[int], Optional[int]]:
        if frame_start is not None and (type(frame_start) is not int or frame_start < 0):
            raise ValueError("frame_start 必须是非负整数")
        if frame_end is not None and (type(frame_end) is not int or frame_end < 0):
            raise ValueError("frame_end 必须是非负整数")
        if frame_start is not None and frame_end is not None and frame_end < frame_start:
            raise ValueError("frame_end 不能小于 frame_start")
        return frame_start, frame_end

    @staticmethod
    def _load_video_frame_map(run_dir: Path) -> Optional[List[int]]:
        map_path = run_dir / "video_frame_map.json"
        if not map_path.is_file():
            return None
        payload = load_json_strict(map_path)
        if not isinstance(payload, dict):
            raise SchemaValidationError("video_frame_map.json 顶层必须是 JSON 对象")
        raw_frames = payload.get("source_frames")
        if not isinstance(raw_frames, list):
            raise SchemaValidationError("video_frame_map.json 缺少合法的 source_frames 列表")
        parsed: List[int] = []
        prev = -1
        for item in raw_frames:
            if type(item) is not int or item < 0 or item <= prev:
                raise SchemaValidationError("video_frame_map.json 的 source_frames 必须是严格递增非负整数")
            parsed.append(item)
            prev = item
        return parsed

    def get_run_events(self, run_id: str, *, limit: Optional[int] = None) -> List[dict]:
        """严格读取指定运行目录的 events.jsonl 告警事件序列。"""
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit 必须是非负整数")
        run_dir = self._resolve_run_dir(run_id)
        events_path = run_dir / "events.jsonl"
        if not events_path.is_file():
            return []
        rows = load_jsonl_strict(events_path)
        return rows[:limit] if limit is not None else rows

    def get_run_speeds_preview(
        self,
        run_id: str,
        *,
        limit: int = 100,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
    ) -> dict:
        """读取指定运行目录的 speeds.csv 预览或指定帧区间样本（确保无 NaN/Inf）。"""
        if type(limit) is not int or limit < 1:
            raise ValueError("limit 必须是 >= 1 的整数")
        self._validate_frame_window(frame_start, frame_end)
        run_dir = self._resolve_run_dir(run_id)
        csv_path = run_dir / "speeds.csv"
        if not csv_path.is_file():
            return {
                "run_id": run_id,
                "columns": [],
                "preview_count": 0,
                "total_rows": 0,
                "truncated": False,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "rows": [],
            }

        rows: List[dict] = []
        total_rows = 0
        matched_rows = 0
        has_window = frame_start is not None or frame_end is not None
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = list(reader.fieldnames or [])
            for raw_row in reader:
                total_rows += 1
                if has_window:
                    try:
                        f_idx = int(raw_row.get("frame_idx", -1))
                    except (TypeError, ValueError):
                        continue
                    if frame_start is not None and f_idx < frame_start:
                        continue
                    if frame_end is not None and f_idx > frame_end:
                        continue
                matched_rows += 1
                if len(rows) >= limit:
                    continue
                clean_row = {}
                for col in columns:
                    val = raw_row.get(col, "")
                    if val in ("nan", "NaN", "inf", "Infinity", "-inf", "-Infinity"):
                        clean_row[col] = None
                    else:
                        clean_row[col] = val
                rows.append(clean_row)
        denom = matched_rows if has_window else total_rows
        return {
            "run_id": run_id,
            "columns": columns,
            "preview_count": len(rows),
            "total_rows": total_rows,
            "matched_rows": matched_rows,
            "truncated": denom > len(rows),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "rows": rows,
        }

    def get_run_conflicts_window(
        self,
        run_id: str,
        *,
        limit: Optional[int] = 300,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
    ) -> dict:
        """按可选帧区间读取 conflicts.jsonl 并返回完整总数与截断标记。"""
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit 必须是非负整数")
        self._validate_frame_window(frame_start, frame_end)
        run_dir = self._resolve_run_dir(run_id)
        conflicts_path = run_dir / "conflicts.jsonl"
        if not conflicts_path.is_file():
            return {
                "run_id": run_id,
                "count": 0,
                "total_count": 0,
                "truncated": False,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "conflicts": [],
            }
        rows = load_jsonl_strict(conflicts_path)
        has_window = frame_start is not None or frame_end is not None
        if has_window:
            filtered = []
            for rec in rows:
                f_idx = rec.get("frame_index")
                if type(f_idx) is not int:
                    continue
                if frame_start is not None and f_idx < frame_start:
                    continue
                if frame_end is not None and f_idx > frame_end:
                    continue
                filtered.append(rec)
        else:
            filtered = rows
        sliced = filtered[:limit] if limit is not None else filtered
        return {
            "run_id": run_id,
            "count": len(sliced),
            "total_count": len(filtered),
            "truncated": len(filtered) > len(sliced),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "conflicts": sliced,
        }

    def get_run_conflicts(
        self,
        run_id: str,
        *,
        limit: Optional[int] = 300,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
    ) -> List[dict]:
        """严格读取指定运行目录的 conflicts.jsonl 帧级冲突序列。"""
        return self.get_run_conflicts_window(
            run_id, limit=limit, frame_start=frame_start, frame_end=frame_end
        )["conflicts"]

    def get_run_video_path(self, run_id: str) -> Path:
        """校验并返回指定运行目录中标注输出视频 annotated_conflict.mp4 的安全路径。"""
        run_dir = self._resolve_run_dir(run_id)
        video_path = (run_dir / "annotated_conflict.mp4").resolve()
        if video_path == run_dir or not video_path.is_relative_to(run_dir):
            raise ValueError(f"非法的视频路径（越出运行目录）: {run_id}")
        if not video_path.is_file():
            raise FileNotFoundError(f"运行目录缺少 annotated_conflict.mp4: {run_id}")
        return video_path

    def get_run_frame_jpeg(self, run_id: str, frame_index: int = 0) -> bytes:
        """从历史运行输出视频按源帧号截取一帧并编码为 JPEG 字节流（兼容稀疏映射）。"""
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index 必须是非负整数")
        run_dir = self._resolve_run_dir(run_id)
        frame_map = self._load_video_frame_map(run_dir)
        target_video_pos = frame_index
        if frame_map is not None:
            pos = bisect_left(frame_map, frame_index)
            if pos >= len(frame_map) or frame_map[pos] != frame_index:
                raise FileNotFoundError(f"未找到源帧 {frame_index} 对应的输出视频帧")
            target_video_pos = pos
        video_path = self.get_run_video_path(run_id)
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"无法打开运行视频文件: {video_path.name}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_video_pos))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise FileNotFoundError(f"未找到指定视频帧: {frame_index}")
            ok_enc, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86]
            )
            if not ok_enc:
                raise RuntimeError("视频帧 JPEG 编码失败")
            return bytes(buf)
        finally:
            cap.release()

    @staticmethod
    def _format_preset_label(path: Path, *, roots: Iterable[Path]) -> str:
        """优先返回相对仓库/工作区根的 POSIX 显示标签，回退至文件名。"""
        for base in roots:
            try:
                return path.relative_to(base).as_posix()
            except ValueError:
                continue
        return path.name

    def list_presets(self) -> dict:
        """发现本地预置测试视频与标定配置文件，返回供前端一键选用的规范条目。"""
        cwd_resolved = Path.cwd().resolve()
        label_roots = (cwd_resolved, self.runs_root.parent)

        video_dirs = (
            self.runs_root.parent / "tracking-fixtures",
            cwd_resolved / "runs" / "tracking-fixtures",
        )
        seen_videos: Dict[Path, dict] = {}
        for vdir in video_dirs:
            if not vdir.is_dir():
                continue
            for entry in vdir.rglob("*"):
                try:
                    rel_parts = entry.relative_to(vdir).parts
                except ValueError:
                    continue
                if any(part.startswith(".") for part in rel_parts):
                    continue
                if not entry.is_file() or entry.suffix.lower() not in _VIDEO_PRESET_EXTS:
                    continue
                resolved = entry.resolve()
                if resolved not in seen_videos:
                    rel_label = self._format_preset_label(resolved, roots=label_roots)
                    is_fixture = "tracking-fixtures" in resolved.parts
                    seen_videos[resolved] = {
                        "label": f"测试固件 · {resolved.name} (镜像静态图，非实拍)" if is_fixture else rel_label,
                        "path": str(resolved),
                        "is_fixture": is_fixture,
                    }

        calib_dir = self.default_calib.parent
        seen_calibs: Dict[Path, dict] = {}
        if calib_dir.is_dir():
            for entry in calib_dir.rglob("*"):
                try:
                    rel_parts = entry.relative_to(calib_dir).parts
                except ValueError:
                    continue
                if any(part.startswith(".") for part in rel_parts):
                    continue
                if not entry.is_file() or entry.suffix.lower() not in _CALIB_PRESET_EXTS:
                    continue
                resolved = entry.resolve()
                if (
                    entry.name.lower() in _NON_CALIB_CONFIG_NAMES
                    and resolved != self.default_calib
                ):
                    continue
                if resolved not in seen_calibs:
                    seen_calibs[resolved] = {
                        "label": self._format_preset_label(resolved, roots=label_roots),
                        "path": str(resolved),
                    }

        videos = sorted(seen_videos.values(), key=lambda item: (item["label"], item["path"]))
        calibs = sorted(seen_calibs.values(), key=lambda item: (item["label"], item["path"]))
        return {
            "videos": videos,
            "calibs": calibs,
            "default_calib": str(self.default_calib),
        }

    def _resolve_job_calibrator(
        self,
        *,
        calibrator: Optional[RoadCalibrator],
        calib_path: Optional[str | Path],
        demo: bool,
        allow_legacy_calib: bool,
    ) -> RoadCalibrator:
        if type(demo) is not bool:
            raise ValueError("demo 必须是布尔值")
        if type(allow_legacy_calib) is not bool:
            raise ValueError("allow_legacy_calib 必须是布尔值")
        if calibrator is not None:
            if not isinstance(calibrator, RoadCalibrator):
                raise ValueError("calibrator 必须是 RoadCalibrator 实例")
            return calibrator

        target_path: Optional[Path] = None
        if calib_path is not None:
            if is_blank_path(calib_path) or type(calib_path) is bool:
                raise ValueError("calib_path 不能为空")
            target_path = Path(calib_path).expanduser().resolve()
        elif self.default_calib.is_file():
            target_path = self.default_calib

        if target_path is not None and target_path.is_file():
            loaded = RoadCalibrator.from_yaml(target_path)
            if loaded.synthetic_field_missing and not allow_legacy_calib:
                raise ValueError(
                    f"标定文件 {target_path} 缺少 synthetic 字段，请重新标定或显式设置 allow_legacy_calib=True"
                )
            if loaded.synthetic and not demo:
                raise ValueError(
                    f"标定文件 {target_path} 为合成占位标定 (synthetic: true)，仅演示模式 demo=True 下可用"
                )
            return loaded

        if target_path is not None and calib_path is not None and not demo:
            raise FileNotFoundError(f"未找到标定配置文件: {target_path}")
        if not demo:
            raise FileNotFoundError("未找到有效实测标定配置，请指定 calib_path 或设置 demo=True")

        return RoadCalibrator(
            image_points=[(400.0, 300.0), (880.0, 300.0), (1150.0, 680.0), (130.0, 680.0)],
            width_m=6.0,
            length_m=15.0,
            name="default_campus_road",
            synthetic=True,
        )

    @staticmethod
    def _job_to_dict(job: _ServiceJob) -> dict:
        with job.cond:
            return {
                "job_id": job.job_id,
                "source": job.source,
                "output_dir": job.output_dir,
                "status": job.status,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "frames_processed": job.frames_processed,
                "latest_timestamp_s": job.latest_timestamp_s,
                "has_latest_frame": job.latest_jpeg is not None,
                "latest_twin": job.latest_twin,
                "frame_seq": job.frame_seq,
                "alert_seq": job.alert_seq,
                "queued_alerts": len(job.alert_queue),
                "error": job.error,
                "summary": job.summary,
                "speed_model": job.speed_model,
                "conflict_model": job.conflict_model,
                "enable_stitching": job.enable_stitching,
                "include_cyclist_vru": job.include_cyclist_vru,
            }

    def _get_job_obj(self, job_id: str) -> _ServiceJob:
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id 不能为空")
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"未找到分析任务: {job_id}")
        return job

    def start_job(
        self,
        source: str | int | Path,
        *,
        calib_path: Optional[str | Path] = None,
        calibrator: Optional[RoadCalibrator] = None,
        demo: bool = False,
        allow_legacy_calib: bool = False,
        output_dir: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
        speed_limit_kmh: float = 30.0,
        speed_model: str = "ema",
        conflict_model: str = "cvm_circle",
        enable_stitching: bool = False,
        include_cyclist_vru: bool = False,
        compensate_anchor_bias: bool = False,
        live_drop_stale_frames: bool = False,
        live_reconnect_attempts: int = 3,
        live_reconnect_delay_s: float = 1.0,
        weights: str = str(DEFAULT_WEIGHTS),
        tracker_config: str = str(DEFAULT_TRACKER),
        object_policy_config: str | Path = DEFAULT_OBJECT_POLICY_CONFIG,
        fps: Optional[float] = None,
        device: Optional[str] = None,
        conf: Optional[float] = None,
        pipeline_factory: Optional[Callable[..., Any]] = None,
    ) -> dict:
        """校验参数并在后台线程启动实时/离线视频冲突感知作业。"""
        _, src_label, _ = parse_video_source(source)
        if max_frames is not None and (type(max_frames) is not int or max_frames < 0):
            raise ValueError("max_frames 必须是 >= 0 的整数")
        if conf is not None and (
            type(conf) not in (int, float) or not math.isfinite(conf) or not (0 < conf <= 1)
        ):
            raise ValueError("conf 必须在 (0, 1] 区间")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("device 必须为非空字符串或 None")
        if output_dir is not None and (is_blank_path(output_dir) or type(output_dir) is bool):
            raise ValueError("output_dir 不能为空")

        resolved_calib = self._resolve_job_calibrator(
            calibrator=calibrator,
            calib_path=calib_path,
            demo=demo,
            allow_legacy_calib=allow_legacy_calib,
        )

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"job-{stamp}-{uuid4().hex[:8]}"
        if output_dir is not None:
            out_dir = Path(output_dir).expanduser().resolve()
        else:
            out_dir = (self.runs_root / job_id).resolve()

        factory = pipeline_factory or ConflictPerceptionPipeline
        pipeline = factory(
            calibrator=resolved_calib,
            weights=weights,
            tracker_config=tracker_config,
            speed_limit_kmh=speed_limit_kmh,
            device=device,
            conf=conf,
            fps=fps,
            object_policy_config=object_policy_config,
            speed_model=speed_model,
            conflict_model=conflict_model,
            enable_stitching=enable_stitching,
            include_cyclist_vru=include_cyclist_vru,
            compensate_anchor_bias=compensate_anchor_bias,
            live_drop_stale_frames=live_drop_stale_frames,
            live_reconnect_attempts=live_reconnect_attempts,
            live_reconnect_delay_s=live_reconnect_delay_s,
        )

        out_dir_key = os.path.normcase(str(out_dir))
        job = _ServiceJob(
            job_id=job_id,
            source=src_label,
            output_dir=str(out_dir),
            status="queued",
            speed_model=str(speed_model),
            conflict_model=str(conflict_model),
            enable_stitching=bool(enable_stitching),
            include_cyclist_vru=bool(include_cyclist_vru),
        )
        with self._lock:
            for owner_id, active_path in self._active_output_dirs.values():
                if (
                    out_dir == active_path
                    or out_dir.is_relative_to(active_path)
                    or active_path.is_relative_to(out_dir)
                ):
                    raise _ServiceConflictError(
                        f"输出目录正被活动任务 {owner_id} 占用: {out_dir}"
                    )
            self._active_output_dirs[out_dir_key] = (job_id, out_dir)
            self._jobs[job_id] = job

        def _on_frame(frame_idx: int, timestamp_s: float, render_frame: Optional[np.ndarray]) -> bool:
            jpeg_bytes: Optional[bytes] = None
            if render_frame is not None and isinstance(render_frame, np.ndarray):
                ok, encoded = cv2.imencode(
                    ".jpg", render_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if ok:
                    jpeg_bytes = encoded.tobytes()
            twin_snap = getattr(pipeline, "_latest_frame_snapshot", None)
            with job.cond:
                job.frames_processed = frame_idx + 1
                if _is_finite_real(timestamp_s):
                    job.latest_timestamp_s = round(float(timestamp_s), 3)
                if isinstance(twin_snap, dict):
                    job.latest_twin = twin_snap
                if jpeg_bytes is not None:
                    job.latest_jpeg = jpeg_bytes
                    job.frame_seq += 1
                job.cond.notify_all()
            return not job.stop_event.is_set()

        def _on_alert(alert_payload: dict) -> None:
            if not isinstance(alert_payload, dict):
                return
            with job.cond:
                job.alert_seq += 1
                item = {"seq": job.alert_seq, **alert_payload}
                job.alert_queue.append(item)
                while len(job.alert_queue) > self.max_alert_queue:
                    job.alert_queue.popleft()
                job.cond.notify_all()

        def _worker() -> None:
            with job.cond:
                job.status = "running"
                job.started_at = _iso_now()
                job.cond.notify_all()
            try:
                summary = pipeline.process_video(
                    video_path=source,
                    output_dir=out_dir,
                    show_window=False,
                    max_frames=max_frames,
                    frame_callback=_on_frame,
                    alert_callback=_on_alert,
                    stop_event=job.stop_event,
                )
                with job.cond:
                    job.summary = summary
                    job.status = "stopped" if job.stop_event.is_set() else "completed"
                    job.finished_at = _iso_now()
                    job.cond.notify_all()
            except Exception as exc:
                with job.cond:
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.status = "failed"
                    job.finished_at = _iso_now()
                    job.cond.notify_all()
            finally:
                with self._lock:
                    owner = self._active_output_dirs.get(out_dir_key)
                    if owner is not None and owner[0] == job_id:
                        del self._active_output_dirs[out_dir_key]

        worker = threading.Thread(target=_worker, name=f"AnalysisJob-{job_id}", daemon=True)
        job.thread = worker
        try:
            worker.start()
        except Exception:
            with self._lock:
                owner = self._active_output_dirs.get(out_dir_key)
                if owner is not None and owner[0] == job_id:
                    del self._active_output_dirs[out_dir_key]
            raise
        return self._job_to_dict(job)

    def get_job_status(self, job_id: str) -> dict:
        """获取指定实时/后台作业的当前状态快照。"""
        return self._job_to_dict(self._get_job_obj(job_id))

    def list_jobs(self) -> List[dict]:
        """列出服务实例管理的全部作业状态（按创建顺序倒序）。"""
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.reverse()
        return [self._job_to_dict(job) for job in jobs]

    def stop_job(self, job_id: str, *, wait_timeout_s: float = 5.0) -> dict:
        """请求停止运行中的作业并等待其安全收尾。"""
        if not _is_finite_real(wait_timeout_s) or float(wait_timeout_s) < 0.0:
            raise ValueError("wait_timeout_s 必须为非负有限数")
        job = self._get_job_obj(job_id)
        job.stop_event.set()
        with job.cond:
            job.cond.notify_all()
        if job.thread is not None and job.thread.is_alive():
            job.thread.join(timeout=float(wait_timeout_s))
        return self._job_to_dict(job)

    def get_job_alerts(
        self, job_id: str, *, since_seq: int = 0, limit: int = 100
    ) -> dict:
        """按增量序号拉取作业的双端告警推送队列。"""
        if type(since_seq) is not int or since_seq < 0:
            raise ValueError("since_seq 必须是非负整数")
        if type(limit) is not int or limit < 1:
            raise ValueError("limit 必须是 >= 1 的整数")
        job = self._get_job_obj(job_id)
        with job.cond:
            alerts = [
                item for item in job.alert_queue if item["seq"] > since_seq
            ][:limit]
            return {
                "job_id": job.job_id,
                "status": job.status,
                "latest_seq": job.alert_seq,
                "count": len(alerts),
                "alerts": alerts,
            }

    def get_job_latest_jpeg(self, job_id: str) -> Optional[bytes]:
        """获取作业最近一帧渲染画面的 JPEG 字节流。"""
        job = self._get_job_obj(job_id)
        with job.cond:
            return job.latest_jpeg

    def iter_job_mjpeg_frames(
        self,
        job_id: str,
        *,
        max_frames: Optional[int] = None,
        wait_timeout_s: float = 1.0,
    ):
        """迭代生成 multipart/x-mixed-replace MJPEG 字节分块；任务结束后自动优雅闭合。"""
        job = self._get_job_obj(job_id)
        last_seq = -1
        emitted = 0

        while True:
            if max_frames is not None and emitted >= max_frames:
                break
            with job.cond:
                while (
                    (job.latest_jpeg is None or job.frame_seq == last_seq)
                    and job.status in ("queued", "running")
                    and not job.stop_event.is_set()
                ):
                    if not job.cond.wait(timeout=wait_timeout_s):
                        break

                has_new = job.latest_jpeg is not None and job.frame_seq != last_seq
                jpeg_bytes = job.latest_jpeg if has_new else None
                cur_seq = job.frame_seq
                finished = job.status in ("completed", "stopped", "failed") or job.stop_event.is_set()

            if jpeg_bytes is not None:
                last_seq = cur_seq
                emitted += 1
                header = (
                    f"--frame\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
                ).encode("ascii")
                yield header + jpeg_bytes + b"\r\n"

            if finished:
                break

    def _discover_weight_files(self) -> List[dict]:
        """扫描仓库根目录、weights/ 与 checkpoints/ 下的 .pt 模型权重文件。"""
        repo_root = Path(__file__).resolve().parents[1]
        cwd_resolved = Path.cwd().resolve()
        scan_bases = list(dict.fromkeys((repo_root, cwd_resolved, self.runs_root.parent)))
        label_roots = tuple(scan_bases)
        seen: Dict[Path, dict] = {}

        def _register_weight(path_obj: Path) -> None:
            if not path_obj.is_file() or path_obj.suffix.lower() != ".pt":
                return
            resolved = path_obj.resolve()
            if resolved not in seen:
                seen[resolved] = {
                    "label": self._format_preset_label(resolved, roots=label_roots),
                    "path": str(resolved),
                }

        for base in scan_bases:
            if not base.is_dir():
                continue
            for entry in sorted(base.glob("*.pt")):
                if not entry.name.startswith("."):
                    _register_weight(entry)
            for sub_name in ("weights", "checkpoints"):
                sub_dir = base / sub_name
                if not sub_dir.is_dir():
                    continue
                for entry in sorted(sub_dir.rglob("*.pt")):
                    try:
                        rel_parts = entry.relative_to(sub_dir).parts
                    except ValueError:
                        continue
                    if any(part.startswith(".") for part in rel_parts):
                        continue
                    _register_weight(entry)

        if DEFAULT_WEIGHTS.is_file():
            _register_weight(DEFAULT_WEIGHTS)

        return sorted(seen.values(), key=lambda item: (item["label"], item["path"]))

    def get_system_info(self) -> dict:
        """探测运行环境、图形文件对话框能力、CUDA 可用性及本地权重清单。"""
        can_browse = False
        try:
            if importlib.util.find_spec("tkinter") is not None:
                if sys.platform.startswith("linux"):
                    can_browse = bool(
                        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
                    )
                else:
                    can_browse = True
        except Exception:
            can_browse = False

        cuda_available = False
        try:
            if importlib.util.find_spec("torch") is not None:
                import torch  # 懒加载，严禁在模块顶层导入

                cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

        jobs = self.list_jobs()
        active_jobs = sum(1 for item in jobs if item["status"] in ("queued", "running"))
        total_runs = len(self.list_runs())

        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "desktop_mode": bool(self.desktop_mode),
            "allow_shutdown": bool(self.allow_shutdown),
            "can_browse_files": bool(can_browse),
            "cuda_available": bool(cuda_available),
            "runs_root": str(self.runs_root),
            "default_calib": str(self.default_calib),
            "weights": self._discover_weight_files(),
            "active_jobs": active_jobs,
            "total_runs": total_runs,
        }

    def stop_active_jobs(self, *, wait_timeout_s: float = 2.0) -> List[dict]:
        """停止全部处于 queued/running 状态的分析任务。"""
        if not _is_finite_real(wait_timeout_s) or float(wait_timeout_s) < 0.0:
            raise ValueError("wait_timeout_s 必须为非负有限数")
        with self._lock:
            jobs = list(self._jobs.values())
        active: List[_ServiceJob] = []
        for job in jobs:
            with job.cond:
                if job.status in ("queued", "running"):
                    job.stop_event.set()
                    job.cond.notify_all()
                    active.append(job)
        deadline = time.monotonic() + float(wait_timeout_s)
        for job in active:
            if job.thread is not None and job.thread.is_alive():
                rem = max(0.0, deadline - time.monotonic())
                job.thread.join(timeout=rem)
        return [self._job_to_dict(job) for job in active]

    def request_shutdown(self) -> dict:
        """授权模式下停止活动任务并异步触发桌面宿主关闭回调。"""
        if not self.allow_shutdown:
            raise _ServiceForbiddenError("当前服务未开启关闭权限 (allow_shutdown=False)")
        self.stop_active_jobs(wait_timeout_s=2.0)
        cb = self._shutdown_callback
        if cb is not None:
            def _run_cb() -> None:
                try:
                    cb()
                except Exception:
                    pass

            timer = threading.Timer(0.15, _run_cb)
            timer.daemon = True
            timer.start()
        return {"status": "shutting_down"}

    def browse_file_dialog(self, kind: str) -> dict:
        """通过懒导入 tkinter 打开系统原生文件/目录选择框（线程锁保护）。"""
        if not isinstance(kind, str) or kind not in _ALLOWED_DIALOG_KINDS:
            raise ValueError(f"kind 必须是 {_ALLOWED_DIALOG_KINDS} 之一")

        with _DIALOG_LOCK:
            try:
                import tkinter as tk  # 懒加载，兼容无 GUI 的 Ubuntu CI
                from tkinter import filedialog
            except Exception as exc:
                raise _DialogUnavailableError(f"当前环境未安装 tkinter 图形界面支持: {exc}") from exc

            tcl_error_cls = getattr(tk, "TclError", RuntimeError)
            root = None
            try:
                root = tk.Tk()
                root.withdraw()
                try:
                    root.attributes("-topmost", True)
                except Exception:
                    pass
                try:
                    root.update_idletasks()
                except Exception:
                    pass

                if kind == "folder":
                    init_dir = str(self.runs_root if self.runs_root.is_dir() else Path.cwd())
                    selected = filedialog.askdirectory(
                        parent=root,
                        title="选择目录",
                        initialdir=init_dir,
                    )
                elif kind == "calib":
                    init_dir = str(
                        self.default_calib.parent
                        if self.default_calib.parent.is_dir()
                        else Path.cwd()
                    )
                    selected = filedialog.askopenfilename(
                        parent=root,
                        title="选择道路标定 YAML 文件",
                        initialdir=init_dir,
                        filetypes=[
                            ("YAML 标定文件", "*.yaml *.yml"),
                            ("所有文件", "*.*"),
                        ],
                    )
                else:
                    init_dir = str(
                        self.runs_root.parent
                        if self.runs_root.parent.is_dir()
                        else Path.cwd()
                    )
                    selected = filedialog.askopenfilename(
                        parent=root,
                        title="选择视频文件",
                        initialdir=init_dir,
                        filetypes=[
                            ("视频文件", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                            ("所有文件", "*.*"),
                        ],
                    )
            except Exception as exc:
                if (
                    isinstance(exc, (tcl_error_cls, RuntimeError))
                    or "tcl" in type(exc).__name__.lower()
                    or "display" in str(exc).lower()
                ):
                    raise _DialogUnavailableError(f"无法初始化系统文件对话框: {exc}") from exc
                raise
            finally:
                if root is not None:
                    try:
                        root.destroy()
                    except Exception:
                        pass

        clean_path = (
            str(selected).strip()
            if isinstance(selected, (str, os.PathLike)) and str(selected).strip()
            else None
        )
        return {
            "cancelled": clean_path is None,
            "path": clean_path,
        }

    def resolve_run_export_file(self, run_id: str, filename: str) -> tuple[Path, str]:
        """校验导出文件名在白名单内并解析 run_dir 下的导出文件路径与 MIME 类型。"""
        clean_name = _validate_single_segment_filename(filename, label="导出文件名")
        media_type = _ALLOWED_RUN_EXPORT_FILES.get(clean_name)
        if media_type is None:
            raise ValueError(f"不允许导出的文件名: {filename}")
        run_dir = self._resolve_run_dir(run_id)
        target = (run_dir / clean_name).resolve()
        if target == run_dir or not target.is_relative_to(run_dir):
            raise ValueError(f"非法的导出文件路径: {filename}")
        if not target.is_file():
            raise FileNotFoundError(f"运行目录缺少导出文件 {clean_name}: {run_id}")
        return target, media_type

    def delete_run(self, run_id: str) -> dict:
        """安全删除指定历史运行目录；若仍有活动任务占用则返回 409 冲突。"""
        run_dir = self._resolve_run_dir(run_id)
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            with job.cond:
                is_active = job.status in ("queued", "running")
                job_out = Path(job.output_dir).resolve()
            if is_active and (
                job_out == run_dir
                or job_out.is_relative_to(run_dir)
                or run_dir.is_relative_to(job_out)
            ):
                raise _ServiceConflictError(
                    f"运行目录 {run_id} 正被活动任务 {job.job_id} 占用，无法删除"
                )
        shutil.rmtree(run_dir)
        return {"deleted": True, "run_id": run_id}

    def generate_run_charts(self, run_id: str) -> dict:
        """懒导入 analysis.charts 为指定运行记录生成车速与风险时间轴 PNG 图表。"""
        run_dir = self._resolve_run_dir(run_id)
        charts_dir = (run_dir / "charts").resolve()
        if charts_dir == run_dir or not charts_dir.is_relative_to(run_dir):
            raise ValueError(f"非法的图表目录路径: {run_id}")
        from .charts import plot_run_timeline

        rendered_paths = plot_run_timeline(run_dir, charts_dir)
        charts = []
        for item in rendered_paths:
            name = Path(item).name
            charts.append({
                "name": name,
                "url": f"/api/runs/{run_id}/charts/{name}",
            })
        return {
            "run_id": run_id,
            "charts": charts,
        }

    def resolve_run_chart_file(self, run_id: str, filename: str) -> Path:
        """严格校验并返回 run_dir/charts 下单层 .png 图表路径。"""
        clean_name = _validate_single_segment_filename(filename, label="图表文件名")
        if Path(clean_name).suffix.lower() != ".png":
            raise ValueError(f"仅允许访问 .png 图表文件: {filename}")
        run_dir = self._resolve_run_dir(run_id)
        charts_root = (run_dir / "charts").resolve()
        if charts_root == run_dir or not charts_root.is_relative_to(run_dir):
            raise ValueError(f"非法的图表目录路径: {run_id}")
        target = (charts_root / clean_name).resolve()
        if target == charts_root or not target.is_relative_to(charts_root):
            raise ValueError(f"非法的图表路径（越出 charts 目录）: {filename}")
        if not target.is_file():
            raise FileNotFoundError(f"未找到指定图表文件: {clean_name}")
        return target

    def open_run_folder(self, run_id: str) -> dict:
        """在桌面文件管理器中打开指定运行目录。"""
        run_dir = self._resolve_run_dir(run_id)
        startfile_fn = getattr(os, "startfile", None)
        if sys.platform.startswith("win") and callable(startfile_fn):
            startfile_fn(str(run_dir))
        else:
            cmd_name = "open" if sys.platform == "darwin" else "xdg-open"
            opener = shutil.which(cmd_name)
            if opener:
                subprocess.Popen(
                    [opener, str(run_dir)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        return {"opened": True, "path": str(run_dir)}

    def extract_calibration_frame(self, source: Any, frame_index: int = 0) -> dict:
        """从本地视频文件抽取指定帧并编码为 JPEG Base64 Data URL。"""
        if type(source) is not str or is_blank_path(source):
            raise ValueError("source 必须是非空本地视频文件路径")
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index 必须是非负整数")

        src_path = Path(source).expanduser().resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"未找到视频文件: {src_path}")

        cap = cv2.VideoCapture(str(src_path))
        try:
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {src_path.name}")
            raw_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(raw_total) if _is_finite_real(raw_total) and raw_total > 0 else 0
            fps = round(float(raw_fps), 3) if _is_finite_real(raw_fps) and raw_fps > 0 else 25.0

            if frame_index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                if frame_index == 0:
                    raise ValueError(f"无法从视频文件解码图像帧: {src_path.name}")
                raise FileNotFoundError(f"未找到指定帧号 {frame_index}（总帧数: {total_frames}）")

            height, width = frame.shape[:2]
            ok_enc, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
            )
            if not ok_enc:
                raise RuntimeError("标定视频帧 JPEG 编码失败")
            data_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            return {
                "width": int(width),
                "height": int(height),
                "frame_count": int(total_frames),
                "fps": float(fps),
                "frame_index": int(frame_index),
                "image_data_url": f"data:image/jpeg;base64,{data_b64}",
            }
        finally:
            cap.release()

    @staticmethod
    def _build_calibrator_from_payload(payload: dict, *, synthetic: bool = False) -> RoadCalibrator:
        """从预览/保存请求体构造并校验 RoadCalibrator（含像素停车区投射）。"""
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        if "image_points" not in payload:
            raise ValueError("请求体缺少必需字段: image_points")

        image_points = payload["image_points"]
        ground_points = payload.get("ground_points")
        parking_zones = payload.get("parking_zones")
        parking_zones_image = payload.get("parking_zones_image")
        use_ransac = True if payload.get("use_ransac") is None else payload.get("use_ransac")
        name = "campus_road" if payload.get("name") is None else payload.get("name")
        ransac_thresh = (
            0.5
            if payload.get("ransac_reproj_threshold_m") is None
            else payload.get("ransac_reproj_threshold_m")
        )

        if ground_points is not None:
            calibrator = RoadCalibrator.from_correspondences(
                image_points=image_points,
                ground_points=ground_points,
                name=name,
                synthetic=synthetic,
                parking_zones=parking_zones,
                use_ransac=use_ransac,
                ransac_reproj_threshold_m=ransac_thresh,
            )
        else:
            if "width_m" not in payload or "length_m" not in payload:
                raise ValueError("未提供 ground_points 时必须提供 width_m 与 length_m")
            calibrator = RoadCalibrator(
                image_points=image_points,
                width_m=payload["width_m"],
                length_m=payload["length_m"],
                name=name,
                synthetic=synthetic,
                parking_zones=parking_zones,
            )

        if parking_zones_image is not None:
            if not isinstance(parking_zones_image, (list, tuple)):
                raise ValueError("parking_zones_image 必须是像素多边形列表")
            for poly in parking_zones_image:
                calibrator.add_parking_zone_pixels(poly)

        return calibrator

    def preview_calibration(self, payload: dict) -> dict:
        """求解并返回道路单应性标定预览结果（重投影 RMSE、内点数与地面坐标）。"""
        calibrator = self._build_calibrator_from_payload(payload, synthetic=False)
        result = {
            "valid": True,
            "reprojection_rmse_m": round(float(calibrator.reprojection_rmse_m), 6),
            "inlier_count": int(calibrator.inlier_count),
            "width_m": round(float(calibrator.width_m), 6),
            "length_m": round(float(calibrator.length_m), 6),
            "ground_points": [
                [round(float(x), 6), round(float(y), 6)]
                for x, y in calibrator.ground_points
            ],
            "parking_zones": [
                [
                    [round(float(x), 6), round(float(y), 6)]
                    for x, y in zone
                ]
                for zone in calibrator.parking_zones
            ],
        }
        _reject_non_finite_tree(result, "calibration_preview")
        return result

    def save_calibration(self, payload: dict) -> dict:
        """校验标定几何与文件名，并将实测标定 (synthetic=False) 持久化到 default_calib 所在目录。"""
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        raw_filename = payload.get("filename")
        clean_name = _validate_single_segment_filename(raw_filename, label="filename")
        suffix = Path(clean_name).suffix.lower()
        if suffix not in _CALIB_PRESET_EXTS or not Path(clean_name).stem:
            raise ValueError("标定文件名必须以 .yaml 或 .yml 结尾")
        if clean_name.lower() in _NON_CALIB_CONFIG_NAMES:
            raise ValueError(f"禁止覆盖非标定系统配置文件: {clean_name}")

        calib_dir = self.default_calib.parent.resolve()
        target = (calib_dir / clean_name).resolve()
        if target == calib_dir or not target.is_relative_to(calib_dir):
            raise ValueError(f"非法的标定保存路径: {clean_name}")

        calibrator = self._build_calibrator_from_payload(payload, synthetic=False)
        saved_path = calibrator.to_yaml(target)
        result = {
            "saved": True,
            "path": str(saved_path),
            "label": saved_path.name,
            "reprojection_rmse_m": round(float(calibrator.reprojection_rmse_m), 6),
        }
        _reject_non_finite_tree(result, "calibration_save")
        return result

    def run_scenarios_tool(self) -> dict:
        """懒导入并执行全部 12 项确定性冲突场景回归验证。"""
        from .scenarios import (
            run_ctrv_turn_scenarios,
            run_obb_ssm_scenarios,
            run_scenarios,
        )

        results = (
            run_scenarios()
            + run_obb_ssm_scenarios()
            + run_ctrv_turn_scenarios()
        )
        total = len(results)
        passed = sum(1 for item in results if bool(item.get("passed")))
        payload = {
            "total": total,
            "passed": passed,
            "all_passed": passed == total,
            "results": results,
        }
        _reject_non_finite_tree(payload, "scenarios")
        return payload

    def run_ablation_tool(self) -> dict:
        """懒导入并运行全量确定性算法消融实验。"""
        from .ablation import run_default_ablation

        result = run_default_ablation()
        _reject_non_finite_tree(result, "ablation")
        return result


_WEB_ASSET_HEADERS: Dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-cache",
}


def _prepare_video_stream(
    service: AnalysisService, run_id: str, range_header: Optional[str]
) -> tuple[int, Dict[str, str], Any]:
    """准备视频流或 Byte-Range 响应元信息与 64KB 分块迭代器。"""
    video_path = service.get_run_video_path(run_id)
    total = video_path.stat().st_size
    if not range_header:
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(total),
        }
        body_iter = _iter_file_range(video_path, 0, total - 1) if total > 0 else iter(())
        return int(HTTPStatus.OK), headers, body_iter

    parsed_range = _parse_http_byte_range(range_header, total)
    if parsed_range is None:
        err_body = _json_dumps({"error": "Range Not Satisfiable"}).encode("utf-8")
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes */{total}",
            "Content-Length": str(len(err_body)),
        }
        return int(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE), headers, err_body

    start, end = parsed_range
    length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{total}",
        "Content-Length": str(length),
    }
    return int(HTTPStatus.PARTIAL_CONTENT), headers, _iter_file_range(video_path, start, end)


class _AnalysisThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_http_server(
    service: AnalysisService,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """创建基于 Python 标准库 ThreadingHTTPServer 的实时监测与历史查询 HTTP 服务。"""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host 不能为空")
    if type(port) is not int or not (0 <= port <= 65535):
        raise ValueError("port 必须是 0..65535 范围内的整数")

    class _RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            # 保持默认静默输出，避免单测与终端日志刷屏。
            return

        def _send_json(self, status_code: int, payload: Any) -> None:
            body = _json_dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(
            self,
            status_code: int,
            body: bytes,
            content_type: str,
            *,
            extra_headers: Optional[Dict[str, str]] = None,
        ) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if extra_headers:
                for key, val in extra_headers.items():
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(body)

        def _send_file_stream(
            self,
            status_code: int,
            path: Path,
            content_type: str,
            *,
            extra_headers: Optional[Dict[str, str]] = None,
        ) -> None:
            total = path.stat().st_size
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(total))
            if extra_headers:
                for key, val in extra_headers.items():
                    self.send_header(key, val)
            self.end_headers()
            if total <= 0:
                return
            try:
                for chunk in _iter_file_range(path, 0, total - 1):
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True

        def _read_json_body(self) -> dict:
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                return {}
            try:
                content_len = int(raw_len)
            except ValueError:
                raise ValueError("无效的 Content-Length 请求头") from None
            if content_len < 0 or content_len > 1_048_576:
                raise ValueError("请求体大小超出允许范围")
            if content_len == 0:
                return {}
            raw = self.rfile.read(content_len)
            data = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
            _reject_non_finite_tree(data, "request_body")
            if not isinstance(data, dict):
                raise ValueError("请求体必须是 JSON 对象")
            return data

        @staticmethod
        def _parse_int_query(query: dict[str, list[str]], key: str, default: Optional[int]) -> Optional[int]:
            vals = query.get(key)
            if not vals:
                return default
            try:
                return int(vals[0])
            except ValueError:
                raise ValueError(f"查询参数 {key} 必须是整数") from None

        def _handle_error(self, exc: Exception) -> None:
            if isinstance(exc, _DialogUnavailableError):
                self._send_json(
                    HTTPStatus.NOT_IMPLEMENTED,
                    {"cancelled": True, "error": str(exc)},
                )
            elif isinstance(exc, _ServiceForbiddenError) or isinstance(exc, PermissionError):
                self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
            elif isinstance(exc, _ServiceConflictError):
                self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
            elif isinstance(exc, (ValueError, json.JSONDecodeError, SchemaValidationError)):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            elif isinstance(exc, (FileNotFoundError, KeyError)):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            else:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            raw_path = parsed.path
            query = parse_qs(parsed.query)
            segments = [unquote(seg) for seg in raw_path.strip("/").split("/") if seg]

            try:
                if raw_path in ("/", "/index.html"):
                    asset_path, media_type = service.resolve_web_asset("index.html")
                    self._send_bytes(
                        HTTPStatus.OK,
                        asset_path.read_bytes(),
                        media_type,
                        extra_headers=_WEB_ASSET_HEADERS,
                    )
                    return

                if raw_path == "/favicon.ico":
                    asset_path, media_type = service.resolve_web_asset("assets/favicon.ico")
                    self._send_bytes(
                        HTTPStatus.OK,
                        asset_path.read_bytes(),
                        media_type,
                        extra_headers=_WEB_ASSET_HEADERS,
                    )
                    return

                if raw_path.startswith("/assets/"):
                    subpath = _decode_and_validate_asset_subpath(raw_path[len("/assets/"):])
                    asset_path, media_type = service.resolve_web_asset(f"assets/{subpath}")
                    self._send_bytes(
                        HTTPStatus.OK,
                        asset_path.read_bytes(),
                        media_type,
                        extra_headers=_WEB_ASSET_HEADERS,
                    )
                    return

                if raw_path == "/api/health":
                    jobs = service.list_jobs()
                    active = sum(1 for item in jobs if item["status"] in ("queued", "running"))
                    self._send_json(HTTPStatus.OK, {
                        "status": "ok",
                        "backend": "stdlib",
                        "runs_root": str(service.runs_root),
                        "active_jobs": active,
                    })
                    return

                if raw_path == "/api/system/info":
                    self._send_json(HTTPStatus.OK, service.get_system_info())
                    return

                if raw_path == "/api/presets":
                    self._send_json(HTTPStatus.OK, service.list_presets())
                    return

                if raw_path == "/api/runs":
                    runs = service.list_runs()
                    self._send_json(HTTPStatus.OK, {"runs": runs, "count": len(runs)})
                    return

                if len(segments) >= 4 and segments[0] == "api" and segments[1] == "runs" and segments[3] in ("export", "charts"):
                    run_id = segments[2]
                    sub = segments[3]
                    if len(segments) != 5:
                        raise ValueError("非法的运行资源路径（禁止空路径或多级目录穿越）")
                    filename = segments[4]
                    if sub == "export":
                        export_path, media_type = service.resolve_run_export_file(run_id, filename)
                        self._send_file_stream(
                            HTTPStatus.OK,
                            export_path,
                            media_type,
                            extra_headers={
                                "Content-Disposition": f'attachment; filename="{run_id}_{export_path.name}"',
                                "X-Content-Type-Options": "nosniff",
                                "Cache-Control": "no-store",
                            },
                        )
                        return
                    chart_path = service.resolve_run_chart_file(run_id, filename)
                    self._send_bytes(
                        HTTPStatus.OK,
                        chart_path.read_bytes(),
                        "image/png",
                        extra_headers={
                            "X-Content-Type-Options": "nosniff",
                            "Cache-Control": "no-cache",
                        },
                    )
                    return

                if len(segments) == 4 and segments[0] == "api" and segments[1] == "runs":
                    run_id = segments[2]
                    sub = segments[3]
                    if sub == "summary":
                        self._send_json(HTTPStatus.OK, service.get_run_summary(run_id))
                        return
                    if sub == "events":
                        limit = self._parse_int_query(query, "limit", None)
                        events = service.get_run_events(run_id, limit=limit)
                        self._send_json(HTTPStatus.OK, {
                            "run_id": run_id,
                            "count": len(events),
                            "events": events,
                        })
                        return
                    if sub == "speeds":
                        limit = self._parse_int_query(query, "limit", 100)
                        frame_start = self._parse_int_query(query, "frame_start", None)
                        frame_end = self._parse_int_query(query, "frame_end", None)
                        assert limit is not None
                        self._send_json(
                            HTTPStatus.OK,
                            service.get_run_speeds_preview(
                                run_id,
                                limit=limit,
                                frame_start=frame_start,
                                frame_end=frame_end,
                            ),
                        )
                        return
                    if sub == "conflicts":
                        limit = self._parse_int_query(query, "limit", 300)
                        frame_start = self._parse_int_query(query, "frame_start", None)
                        frame_end = self._parse_int_query(query, "frame_end", None)
                        self._send_json(
                            HTTPStatus.OK,
                            service.get_run_conflicts_window(
                                run_id,
                                limit=limit,
                                frame_start=frame_start,
                                frame_end=frame_end,
                            ),
                        )
                        return
                    if sub == "frame":
                        frame_index = self._parse_int_query(query, "index", 0)
                        assert frame_index is not None
                        jpeg_bytes = service.get_run_frame_jpeg(run_id, frame_index=frame_index)
                        self._send_bytes(
                            HTTPStatus.OK,
                            jpeg_bytes,
                            "image/jpeg",
                            extra_headers={"Cache-Control": "public, max-age=60"},
                        )
                        return
                    if sub == "video":
                        status_code, headers, body = _prepare_video_stream(
                            service, run_id, self.headers.get("Range")
                        )
                        if status_code == int(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE):
                            self._send_bytes(
                                status_code,
                                body,
                                "application/json; charset=utf-8",
                                extra_headers={
                                    "Accept-Ranges": headers["Accept-Ranges"],
                                    "Content-Range": headers["Content-Range"],
                                },
                            )
                            return
                        self.send_response(status_code)
                        self.send_header("Content-Type", "video/mp4")
                        for k, v in headers.items():
                            self.send_header(k, v)
                        self.end_headers()
                        try:
                            for chunk in body:
                                self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            self.close_connection = True
                        return

                if raw_path == "/api/jobs":
                    jobs = service.list_jobs()
                    self._send_json(HTTPStatus.OK, {"jobs": jobs, "count": len(jobs)})
                    return

                if len(segments) == 3 and segments[0] == "api" and segments[1] == "jobs":
                    self._send_json(HTTPStatus.OK, service.get_job_status(segments[2]))
                    return

                if len(segments) == 4 and segments[0] == "api" and segments[1] == "jobs":
                    job_id = segments[2]
                    sub = segments[3]
                    if sub == "alerts":
                        since_seq = self._parse_int_query(query, "since", 0)
                        limit = self._parse_int_query(query, "limit", 100)
                        assert since_seq is not None and limit is not None
                        self._send_json(HTTPStatus.OK, service.get_job_alerts(job_id, since_seq=since_seq, limit=limit))
                        return
                    if sub == "mjpeg":
                        max_frames = self._parse_int_query(query, "max_frames", None)
                        if max_frames is not None and max_frames < 1:
                            raise ValueError("max_frames 必须是 >= 1 的整数")
                        # 先校验 job_id 存在再发送流式响应头，否则返回标准 404 JSON。
                        service.get_job_status(job_id)
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        try:
                            for chunk in service.iter_job_mjpeg_frames(job_id, max_frames=max_frames):
                                self.wfile.write(chunk)
                                self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            pass
                        self.close_connection = True
                        return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"未找到路由: {raw_path}"})
            except Exception as exc:
                self._handle_error(exc)

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            raw_path = parsed.path
            segments = [unquote(seg) for seg in raw_path.strip("/").split("/") if seg]
            try:
                if raw_path == "/api/system/shutdown":
                    self._send_json(HTTPStatus.OK, service.request_shutdown())
                    return

                if raw_path == "/api/dialog/browse":
                    body = self._read_json_body()
                    self._send_json(HTTPStatus.OK, service.browse_file_dialog(body.get("kind")))
                    return

                if raw_path == "/api/calibration/frame":
                    body = self._read_json_body()
                    frame_idx = 0 if body.get("frame_index") is None else body.get("frame_index")
                    self._send_json(
                        HTTPStatus.OK,
                        service.extract_calibration_frame(body.get("source"), frame_index=frame_idx),
                    )
                    return

                if raw_path == "/api/calibration/preview":
                    body = self._read_json_body()
                    self._send_json(HTTPStatus.OK, service.preview_calibration(body))
                    return

                if raw_path == "/api/calibration/save":
                    body = self._read_json_body()
                    self._send_json(HTTPStatus.OK, service.save_calibration(body))
                    return

                if raw_path == "/api/tools/scenarios":
                    self._send_json(HTTPStatus.OK, service.run_scenarios_tool())
                    return

                if raw_path == "/api/tools/ablation":
                    self._send_json(HTTPStatus.OK, service.run_ablation_tool())
                    return

                if len(segments) == 4 and segments[0] == "api" and segments[1] == "runs":
                    run_id = segments[2]
                    sub = segments[3]
                    if sub == "charts":
                        self._send_json(HTTPStatus.OK, service.generate_run_charts(run_id))
                        return
                    if sub == "open-folder":
                        self._send_json(HTTPStatus.OK, service.open_run_folder(run_id))
                        return

                if raw_path == "/api/jobs":
                    body = self._read_json_body()
                    if "source" not in body:
                        raise ValueError("请求体缺少必需字段: source")
                    unknown = set(body.keys()) - _ALLOWED_JOB_REQUEST_KEYS
                    if unknown:
                        raise ValueError(f"包含不支持的作业参数: {', '.join(sorted(unknown))}")
                    job = service.start_job(**body)
                    self._send_json(HTTPStatus.CREATED, job)
                    return

                if len(segments) == 4 and segments[0] == "api" and segments[1] == "jobs" and segments[3] == "stop":
                    self._send_json(HTTPStatus.OK, service.stop_job(segments[2]))
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"未找到路由: {raw_path}"})
            except Exception as exc:
                self._handle_error(exc)

        def do_DELETE(self) -> None:
            parsed = urlsplit(self.path)
            raw_path = parsed.path
            segments = [unquote(seg) for seg in raw_path.strip("/").split("/") if seg]
            try:
                if len(segments) == 3 and segments[0] == "api" and segments[1] == "runs":
                    self._send_json(HTTPStatus.OK, service.delete_run(segments[2]))
                    return
                if len(segments) == 3 and segments[0] == "api" and segments[1] == "jobs":
                    self._send_json(HTTPStatus.OK, service.stop_job(segments[2]))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"未找到路由: {raw_path}"})
            except Exception as exc:
                self._handle_error(exc)

    return _AnalysisThreadingHTTPServer((host, port), _RequestHandler)


def create_fastapi_app(service: Optional[AnalysisService] = None):
    """懒导入 FastAPI 并构建与标准库服务同路径的 REST + MJPEG + WebSocket 应用。"""
    try:
        import asyncio
        from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import Response, StreamingResponse
    except ImportError as exc:
        raise ImportError(
            "未安装 FastAPI 可选依赖，无法创建 FastAPI 应用；"
            "默认请使用标准库服务 (--backend stdlib) 或安装 fastapi 与 uvicorn。"
        ) from exc

    svc = service if service is not None else AnalysisService()
    app = FastAPI(title="Campus Conflict Perception Service", version="1.0.0")

    def _json_response(payload: Any, status_code: int = 200) -> Response:
        return Response(
            content=_json_dumps(payload),
            status_code=status_code,
            media_type="application/json; charset=utf-8",
        )

    def _raise_http(exc: Exception) -> None:
        if isinstance(exc, _ServiceForbiddenError) or isinstance(exc, PermissionError):
            raise HTTPException(status_code=403, detail=str(exc)) from None
        if isinstance(exc, _ServiceConflictError):
            raise HTTPException(status_code=409, detail=str(exc)) from None
        if isinstance(exc, (ValueError, json.JSONDecodeError, SchemaValidationError)):
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if isinstance(exc, (FileNotFoundError, KeyError)):
            raise HTTPException(status_code=404, detail=str(exc)) from None
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from None

    async def _read_fastapi_json(request: Request) -> dict:
        raw = await request.body()
        if not raw:
            return {}
        body = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
        _reject_non_finite_tree(body, "request_body")
        if not isinstance(body, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return body

    @app.get("/")
    @app.get("/index.html")
    def web_index():
        try:
            asset_path, media_type = svc.resolve_web_asset("index.html")
            return Response(
                content=asset_path.read_bytes(),
                media_type=media_type,
                headers=dict(_WEB_ASSET_HEADERS),
            )
        except Exception as exc:
            _raise_http(exc)

    @app.get("/favicon.ico")
    def web_favicon():
        try:
            asset_path, media_type = svc.resolve_web_asset("assets/favicon.ico")
            return Response(
                content=asset_path.read_bytes(),
                media_type=media_type,
                headers=dict(_WEB_ASSET_HEADERS),
            )
        except Exception as exc:
            _raise_http(exc)

    @app.get("/assets/{subpath:path}")
    def web_asset(subpath: str):
        try:
            clean_sub = _decode_and_validate_asset_subpath(subpath)
            asset_path, media_type = svc.resolve_web_asset(f"assets/{clean_sub}")
            return Response(
                content=asset_path.read_bytes(),
                media_type=media_type,
                headers=dict(_WEB_ASSET_HEADERS),
            )
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/health")
    def health():
        jobs = svc.list_jobs()
        active = sum(1 for item in jobs if item["status"] in ("queued", "running"))
        return _json_response({
            "status": "ok",
            "backend": "fastapi",
            "runs_root": str(svc.runs_root),
            "active_jobs": active,
        })

    @app.get("/api/system/info")
    def system_info():
        try:
            return _json_response(svc.get_system_info())
        except Exception as exc:
            _raise_http(exc)

    @app.post("/api/system/shutdown")
    def system_shutdown():
        try:
            return _json_response(svc.request_shutdown())
        except _ServiceForbiddenError as exc:
            return _json_response({"error": str(exc)}, status_code=403)
        except Exception as exc:
            _raise_http(exc)

    @app.post("/api/dialog/browse")
    async def dialog_browse(request: Request):
        try:
            body = await _read_fastapi_json(request)
            return _json_response(svc.browse_file_dialog(body.get("kind")))
        except _DialogUnavailableError as exc:
            return _json_response({"cancelled": True, "error": str(exc)}, status_code=501)
        except Exception as exc:
            _raise_http(exc)

    @app.post("/api/calibration/frame")
    async def calibration_frame(request: Request):
        try:
            body = await _read_fastapi_json(request)
            frame_idx = 0 if body.get("frame_index") is None else body.get("frame_index")
            return _json_response(
                svc.extract_calibration_frame(body.get("source"), frame_index=frame_idx)
            )
        except Exception as exc:
            _raise_http(exc)

    @app.post("/api/calibration/preview")
    async def calibration_preview(request: Request):
        try:
            body = await _read_fastapi_json(request)
            return _json_response(svc.preview_calibration(body))
        except Exception as exc:
            _raise_http(exc)

    @app.post("/api/calibration/save")
    async def calibration_save(request: Request):
        try:
            body = await _read_fastapi_json(request)
            return _json_response(svc.save_calibration(body))
        except Exception as exc:
            _raise_http(exc)

    @app.post("/api/tools/scenarios")
    def tools_scenarios():
        try:
            return _json_response(svc.run_scenarios_tool())
        except Exception as exc:
            _raise_http(exc)

    @app.post("/api/tools/ablation")
    def tools_ablation():
        try:
            return _json_response(svc.run_ablation_tool())
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/presets")
    def list_presets():
        return _json_response(svc.list_presets())

    @app.get("/api/runs")
    def list_runs():
        runs = svc.list_runs()
        return _json_response({"runs": runs, "count": len(runs)})

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        try:
            return _json_response(svc.delete_run(run_id))
        except _ServiceConflictError as exc:
            return _json_response({"error": str(exc)}, status_code=409)
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/runs/{run_id}/export/{filename:path}")
    def run_export(run_id: str, filename: str):
        try:
            export_path, media_type = svc.resolve_run_export_file(run_id, filename)
            total = export_path.stat().st_size
            headers = {
                "Content-Disposition": f'attachment; filename="{run_id}_{export_path.name}"',
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
                "Content-Length": str(total),
            }
            if total == 0:
                return Response(content=b"", media_type=media_type, headers=headers)
            return StreamingResponse(
                _iter_file_range(export_path, 0, total - 1),
                media_type=media_type,
                headers=headers,
            )
        except Exception as exc:
            _raise_http(exc)

    @app.post("/api/runs/{run_id}/charts")
    def run_charts_generate(run_id: str):
        try:
            return _json_response(svc.generate_run_charts(run_id))
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/runs/{run_id}/charts/{filename:path}")
    def run_chart_file(run_id: str, filename: str):
        try:
            chart_path = svc.resolve_run_chart_file(run_id, filename)
            return Response(
                content=chart_path.read_bytes(),
                media_type="image/png",
                headers={
                    "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "no-cache",
                },
            )
        except Exception as exc:
            _raise_http(exc)

    @app.post("/api/runs/{run_id}/open-folder")
    def run_open_folder(run_id: str):
        try:
            return _json_response(svc.open_run_folder(run_id))
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str):
        try:
            return _json_response(svc.get_run_summary(run_id))
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str, limit: Optional[int] = None):
        try:
            events = svc.get_run_events(run_id, limit=limit)
            return _json_response({"run_id": run_id, "count": len(events), "events": events})
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/runs/{run_id}/speeds")
    def run_speeds(
        run_id: str,
        limit: int = 100,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
    ):
        try:
            return _json_response(
                svc.get_run_speeds_preview(
                    run_id,
                    limit=limit,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )
            )
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/runs/{run_id}/conflicts")
    def run_conflicts(
        run_id: str,
        limit: Optional[int] = 300,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
    ):
        try:
            return _json_response(
                svc.get_run_conflicts_window(
                    run_id,
                    limit=limit,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )
            )
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/runs/{run_id}/frame")
    def run_frame(run_id: str, index: int = 0):
        try:
            jpeg_bytes = svc.get_run_frame_jpeg(run_id, frame_index=index)
            return Response(
                content=jpeg_bytes,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=60"},
            )
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/runs/{run_id}/video")
    def run_video(run_id: str, request: Request):
        try:
            status_code, headers, body = _prepare_video_stream(
                svc, run_id, request.headers.get("range")
            )
            if status_code == int(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE):
                return Response(
                    content=body,
                    status_code=status_code,
                    media_type="application/json; charset=utf-8",
                    headers=headers,
                )
            return StreamingResponse(
                body,
                status_code=status_code,
                media_type="video/mp4",
                headers=headers,
            )
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/jobs")
    def list_jobs():
        jobs = svc.list_jobs()
        return _json_response({"jobs": jobs, "count": len(jobs)})

    @app.post("/api/jobs")
    async def create_job(request: Request):
        try:
            raw = await request.body()
            body = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
            _reject_non_finite_tree(body, "request_body")
            if not isinstance(body, dict) or "source" not in body:
                raise ValueError("请求体必须是包含 source 字段的 JSON 对象")
            unknown = set(body.keys()) - _ALLOWED_JOB_REQUEST_KEYS
            if unknown:
                raise ValueError(f"包含不支持的作业参数: {', '.join(sorted(unknown))}")
            return _json_response(svc.start_job(**body), status_code=201)
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        try:
            return _json_response(svc.get_job_status(job_id))
        except Exception as exc:
            _raise_http(exc)

    @app.post("/api/jobs/{job_id}/stop")
    @app.delete("/api/jobs/{job_id}")
    def stop_job(job_id: str):
        try:
            return _json_response(svc.stop_job(job_id))
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/jobs/{job_id}/alerts")
    def job_alerts(job_id: str, since: int = 0, limit: int = 100):
        try:
            return _json_response(svc.get_job_alerts(job_id, since_seq=since, limit=limit))
        except Exception as exc:
            _raise_http(exc)

    @app.get("/api/jobs/{job_id}/mjpeg")
    def job_mjpeg(job_id: str, max_frames: Optional[int] = None):
        try:
            svc.get_job_status(job_id)
            return StreamingResponse(
                svc.iter_job_mjpeg_frames(job_id, max_frames=max_frames),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )
        except Exception as exc:
            _raise_http(exc)

    @app.websocket("/ws/jobs/{job_id}/alerts")
    async def ws_job_alerts(websocket: WebSocket, job_id: str):
        try:
            svc.get_job_status(job_id)
        except Exception:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        cursor = 0
        try:
            while True:
                batch = svc.get_job_alerts(job_id, since_seq=cursor, limit=100)
                for item in batch["alerts"]:
                    await websocket.send_text(_json_dumps(item))
                    cursor = item["seq"]
                if batch["status"] in ("completed", "stopped", "failed") and not batch["alerts"]:
                    break
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return

    return app


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="校园道路冲突感知与双端预警实时后端服务"
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs/analysis",
        help="历史运行结果根目录，默认 runs/analysis",
    )
    parser.add_argument(
        "--default-calib",
        type=str,
        default="configs/calibration.yaml",
        help="默认标定 YAML 配置路径，默认 configs/calibration.yaml",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="HTTP 服务监听主机地址，默认 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP 服务监听端口，默认 8000",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("stdlib", "fastapi"),
        default="stdlib",
        help="Web 服务实现后端：stdlib（标准库，默认）或 fastapi（需可选依赖）",
    )
    args = parser.parse_args(argv)

    if not args.runs_root.strip():
        parser.error("--runs-root 不能为空")
    if not args.default_calib.strip():
        parser.error("--default-calib 不能为空")
    if not args.host.strip():
        parser.error("--host 不能为空")
    if not (0 <= args.port <= 65535):
        parser.error("--port 必须是 0..65535 范围内的整数")

    service = AnalysisService(
        runs_root=args.runs_root,
        default_calib=args.default_calib,
    )

    if args.backend == "fastapi":
        try:
            import uvicorn
        except ImportError as exc:
            parser.exit(
                1,
                "未安装 uvicorn/fastapi 可选依赖，请改用 --backend stdlib 或安装 fastapi 与 uvicorn。\n",
            )
            raise exc
        app = create_fastapi_app(service)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    server = create_http_server(service, host=args.host, port=args.port)
    bound_host, bound_port = server.server_address[:2]
    print(f"[*] 冲突感知后端服务已启动 (stdlib): http://{bound_host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
