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
from collections import deque
import csv
from dataclasses import dataclass, field
import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
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

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ok, frame = self._cap.read()
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
                self._has_unread = True
                self._cond.notify_all()

    def read(self, timeout_s: Optional[float] = None) -> tuple[bool, Any]:
        """获取最新可用帧；若尚无新帧则在超时范围内阻塞等待，已结束且无未读帧时返回 (False, None)。"""
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
                    return False, None
                self._cond.wait(timeout=remaining)

            # 优先交付已捕获的最后一帧，确保有限测试流末尾不会漏掉最新帧。
            if self._has_unread and not self._stop_event.is_set():
                frame = self._latest_frame
                self._latest_frame = None
                self._has_unread = False
                return True, frame
            return False, None

    def get(self, prop_id: int) -> Any:
        """透传底层采集对象属性查询。"""
        if self._released:
            return 0.0
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
    frame_seq: int = 0
    alert_seq: int = 0
    alert_queue: deque = field(default_factory=deque)
    summary: Optional[dict] = None
    error: Optional[str] = None
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
        max_alert_queue: int = 200,
    ):
        if is_blank_path(runs_root) or type(runs_root) is bool or not isinstance(runs_root, (str, os.PathLike)):
            raise ValueError("runs_root 必须是非空目录路径")
        if is_blank_path(default_calib) or type(default_calib) is bool or not isinstance(default_calib, (str, os.PathLike)):
            raise ValueError("default_calib 必须是非空文件路径")
        if type(max_alert_queue) is not int or max_alert_queue < 1:
            raise ValueError("max_alert_queue 必须是 >= 1 的整数")

        self.runs_root = Path(runs_root).expanduser().resolve()
        self.default_calib = Path(default_calib).expanduser().resolve()
        self.max_alert_queue = max_alert_queue
        self._lock = threading.Lock()
        self._jobs: Dict[str, _ServiceJob] = {}

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
        return {"run_id": run_id, **data}

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

    def get_run_speeds_preview(self, run_id: str, *, limit: int = 100) -> dict:
        """读取指定运行目录的 speeds.csv 前 N 行预览（确保无 NaN/Inf）。"""
        if type(limit) is not int or limit < 1:
            raise ValueError("limit 必须是 >= 1 的整数")
        run_dir = self._resolve_run_dir(run_id)
        csv_path = run_dir / "speeds.csv"
        if not csv_path.is_file():
            return {
                "run_id": run_id,
                "columns": [],
                "preview_count": 0,
                "total_rows": 0,
                "truncated": False,
                "rows": [],
            }

        rows: List[dict] = []
        total_rows = 0
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = list(reader.fieldnames or [])
            for raw_row in reader:
                total_rows += 1
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
        return {
            "run_id": run_id,
            "columns": columns,
            "preview_count": len(rows),
            "total_rows": total_rows,
            "truncated": total_rows > len(rows),
            "rows": rows,
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
                "frame_seq": job.frame_seq,
                "alert_seq": job.alert_seq,
                "queued_alerts": len(job.alert_queue),
                "error": job.error,
                "summary": job.summary,
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

        job = _ServiceJob(
            job_id=job_id,
            source=src_label,
            output_dir=str(out_dir),
            status="queued",
        )
        with self._lock:
            self._jobs[job_id] = job

        def _on_frame(frame_idx: int, timestamp_s: float, render_frame: Optional[np.ndarray]) -> bool:
            jpeg_bytes: Optional[bytes] = None
            if render_frame is not None and isinstance(render_frame, np.ndarray):
                ok, encoded = cv2.imencode(
                    ".jpg", render_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if ok:
                    jpeg_bytes = encoded.tobytes()
            with job.cond:
                job.frames_processed = frame_idx + 1
                if _is_finite_real(timestamp_s):
                    job.latest_timestamp_s = round(float(timestamp_s), 3)
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

        worker = threading.Thread(target=_worker, name=f"AnalysisJob-{job_id}", daemon=True)
        job.thread = worker
        worker.start()
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
            if isinstance(exc, (ValueError, json.JSONDecodeError, SchemaValidationError)):
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

                if raw_path == "/api/runs":
                    runs = service.list_runs()
                    self._send_json(HTTPStatus.OK, {"runs": runs, "count": len(runs)})
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
                        assert limit is not None
                        self._send_json(HTTPStatus.OK, service.get_run_speeds_preview(run_id, limit=limit))
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
                if raw_path == "/api/jobs":
                    body = self._read_json_body()
                    if "source" not in body:
                        raise ValueError("请求体缺少必需字段: source")
                    allowed_keys = {
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
                    }
                    unknown = set(body.keys()) - allowed_keys
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
        if isinstance(exc, (ValueError, json.JSONDecodeError, SchemaValidationError)):
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if isinstance(exc, (FileNotFoundError, KeyError)):
            raise HTTPException(status_code=404, detail=str(exc)) from None
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from None

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

    @app.get("/api/runs")
    def list_runs():
        runs = svc.list_runs()
        return _json_response({"runs": runs, "count": len(runs)})

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
    def run_speeds(run_id: str, limit: int = 100):
        try:
            return _json_response(svc.get_run_speeds_preview(run_id, limit=limit))
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
