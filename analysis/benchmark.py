# -*- coding: utf-8 -*-
"""
分阶段耗时基准与吞吐统计工具 (Pipeline Stage Latency Profiler & Benchmark)
=============================================================================
在临时运行目录或无持久磁盘负担下，逐帧统计冲突感知流水线的 6 个阶段耗时：
  decode / detect_track / geometry_speed / conflict_events / render_write / total_frame
以及一次性模型加载耗时 model_load_ms 与端到端处理帧率 end_to_end_fps。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from detection.core import DEFAULT_WEIGHTS, ROOT, is_blank_path
from tracking.core import DEFAULT_TRACKER, VideoClock
from .calibration import RoadCalibrator
from .policy import DEFAULT_OBJECT_POLICY_CONFIG

# 流水线分阶段耗时固定顺序（前 5 项为子阶段，第 6 项为端到端单帧总耗时）
SUBSTAGE_NAMES: tuple[str, ...] = (
    "decode",
    "detect_track",
    "geometry_speed",
    "conflict_events",
    "render_write",
)
STAGE_NAMES: tuple[str, ...] = SUBSTAGE_NAMES + ("total_frame",)


def _validate_finite_non_neg(value: Any, label: str) -> float:
    """校验有限非负浮点数，显式排除 bool 与 np.bool_。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} 不能为布尔值")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} 必须为数字")
    num = float(value)
    if not math.isfinite(num) or num < 0.0:
        raise ValueError(f"{label} 必须为有限非负数")
    return num


def _reject_non_finite_tree(value: Any, label: str) -> None:
    """递归校验 JSON 报告树，阻止 NaN/Inf 或布尔误混入数值字段。"""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} 包含非有限数值")
    elif isinstance(value, dict):
        for key, val in value.items():
            _reject_non_finite_tree(val, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _reject_non_finite_tree(item, f"{label}[{idx}]")


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """对已升序排列的非空列表计算线性插值百分位数 (0..100)。"""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos = (pct / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _compute_stages_and_fps(
    per_stage: Mapping[str, Sequence[float]],
    n_frames: int,
) -> tuple[float, dict[str, dict[str, float]]]:
    """针对给定帧切片计算各阶段均值/分位数/占比与帧内吞吐 FPS。"""
    if n_frames <= 0:
        empty_stages = {
            name: {
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "max_ms": 0.0,
                "share_pct": 0.0,
            }
            for name in STAGE_NAMES
        }
        return 0.0, empty_stages

    total_vals = per_stage["total_frame"]
    total_sum_ms = sum(total_vals)
    total_mean_ms = total_sum_ms / n_frames
    fps = (1000.0 * n_frames / total_sum_ms) if total_sum_ms > 0.0 else 0.0

    stages_summary: dict[str, dict[str, float]] = {}
    for name in STAGE_NAMES:
        vals = sorted(per_stage[name])
        mean_val = sum(vals) / n_frames
        if name == "total_frame":
            share_pct = 100.0 if total_mean_ms > 0.0 else 0.0
        else:
            share_pct = (100.0 * mean_val / total_mean_ms) if total_mean_ms > 0.0 else 0.0
        stages_summary[name] = {
            "mean_ms": round(mean_val, 3),
            "p50_ms": round(_percentile(vals, 50.0), 3),
            "p95_ms": round(_percentile(vals, 95.0), 3),
            "max_ms": round(vals[-1], 3),
            "share_pct": round(share_pct, 2),
        }

    return round(fps, 3), stages_summary


def _empty_timing_summary() -> dict:
    """返回 0 帧基准的零值统计结构，避免空样本触发除零或 RuntimeWarning。"""
    zero_fps, stages = _compute_stages_and_fps({}, 0)
    _, steady_stages = _compute_stages_and_fps({}, 0)
    return {
        "schema_version": 1,
        "frames_profiled": 0,
        "end_to_end_fps": zero_fps,
        "frame_processing_fps": zero_fps,
        "stages": stages,
        "steady_state": {
            "warmup_frames_excluded": 0,
            "frames_evaluated": 0,
            "steady_state_fps": None,
            "stages": {},
        },
    }


def summarize_stage_timings(stage_samples: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]]) -> dict:
    """汇总各阶段逐帧耗时（毫秒），输出 mean/p50/p95/max 及端到端吞吐 FPS。"""
    if stage_samples is None or isinstance(stage_samples, (str, bytes, bool, int, float)):
        raise ValueError("stage_samples 必须是逐帧样本列表或阶段耗时序列映射")

    per_stage: dict[str, list[float]] = {name: [] for name in STAGE_NAMES}
    n_frames = 0

    if isinstance(stage_samples, Mapping):
        if not stage_samples:
            return _empty_timing_summary()
        known_keys = [k for k in STAGE_NAMES if k in stage_samples]
        if not known_keys:
            raise ValueError(f"stage_samples 未包含任何已知阶段键: {STAGE_NAMES}")
        lengths: set[int] = set()
        for key in known_keys:
            raw_seq = stage_samples[key]
            if isinstance(raw_seq, (str, bytes)) or not isinstance(raw_seq, (Sequence, np.ndarray)):
                raise ValueError(f"stage_samples[{key!r}] 必须是数值序列")
            vals = [
                _validate_finite_non_neg(v, f"stage_samples[{key!r}][{i}]")
                for i, v in enumerate(raw_seq)
            ]
            per_stage[key] = vals
            lengths.add(len(vals))
        if len(lengths) > 1:
            raise ValueError("stage_samples 各阶段样本长度必须一致")
        n_frames = next(iter(lengths))
        if n_frames == 0:
            return _empty_timing_summary()
        # 仅提供子阶段而未提供 total_frame 时，按各帧子阶段之和自动合成总耗时
        if "total_frame" not in stage_samples:
            per_stage["total_frame"] = [
                sum(per_stage[sub][idx] for sub in SUBSTAGE_NAMES if per_stage[sub])
                for idx in range(n_frames)
            ]
        for name in STAGE_NAMES:
            if not per_stage[name]:
                per_stage[name] = [0.0] * n_frames
    elif isinstance(stage_samples, Sequence):
        n_frames = len(stage_samples)
        if n_frames == 0:
            return _empty_timing_summary()
        for idx, row in enumerate(stage_samples):
            if not isinstance(row, Mapping):
                raise ValueError(f"stage_samples[{idx}] 必须是键值映射")
            present_known = [k for k in STAGE_NAMES if k in row]
            if not present_known:
                raise ValueError(f"stage_samples[{idx}] 未包含任何已知阶段键")
            frame_sub_sum = 0.0
            for sub in SUBSTAGE_NAMES:
                val = _validate_finite_non_neg(row.get(sub, 0.0), f"stage_samples[{idx}][{sub!r}]")
                per_stage[sub].append(val)
                frame_sub_sum += val
            if "total_frame" in row:
                tot = _validate_finite_non_neg(
                    row["total_frame"], f"stage_samples[{idx}]['total_frame']"
                )
            else:
                tot = frame_sub_sum
            per_stage["total_frame"].append(tot)
    else:
        raise ValueError("stage_samples 必须是逐帧样本列表或阶段耗时序列映射")

    frame_processing_fps, stages_summary = _compute_stages_and_fps(per_stage, n_frames)

    if n_frames > 1:
        steady_slice = {name: per_stage[name][1:] for name in STAGE_NAMES}
        steady_fps, steady_stages = _compute_stages_and_fps(steady_slice, n_frames - 1)
        steady_state = {
            "warmup_frames_excluded": 1,
            "frames_evaluated": n_frames - 1,
            "steady_state_fps": steady_fps,
            "stages": steady_stages,
        }
    else:
        steady_state = {
            "warmup_frames_excluded": 0,
            "frames_evaluated": 0,
            "steady_state_fps": None,
            "stages": {},
        }

    return {
        "schema_version": 1,
        "frames_profiled": n_frames,
        "end_to_end_fps": frame_processing_fps,
        "frame_processing_fps": frame_processing_fps,
        "stages": stages_summary,
        "steady_state": steady_state,
    }


def _create_synthetic_demo_video(dest: Path, *, frames: int = 10, fps: float = 10.0) -> Path:
    """在缺少样例视频时于临时目录生成轻量构造视频，保障 --demo 零外部依赖运行。"""
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (128, 128))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"无法创建临时演示视频: {dest}")
    try:
        for idx in range(frames):
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            x0 = 10 + (idx * 6) % 80
            cv2.rectangle(img, (x0, 48), (x0 + 24, 96), (200, 200, 200), -1)
            writer.write(img)
    finally:
        writer.release()
    return dest


def _resolve_calibrator(
    *,
    calibrator: Optional[RoadCalibrator],
    calib_path: Optional[str | Path],
    demo: bool,
    allow_legacy_calib: bool,
) -> RoadCalibrator:
    """解析或构造标定器，严格保留与主入口一致的 synthetic 安全校验。"""
    if calibrator is not None:
        if not hasattr(calibrator, "pixel_to_ground") or not hasattr(calibrator, "is_in_zone"):
            raise ValueError("calibrator 必须是有效的 RoadCalibrator 实例")
        return calibrator

    if calib_path is not None:
        if not isinstance(calib_path, (str, os.PathLike)) or is_blank_path(calib_path):
            raise ValueError("标定文件路径不能为空")
        resolved_calib = Path(calib_path).expanduser().resolve()
    else:
        resolved_calib = (ROOT / "configs" / "calibration.yaml").resolve()

    if resolved_calib.is_file():
        loaded = RoadCalibrator.from_yaml(resolved_calib)
        if getattr(loaded, "synthetic_field_missing", False) and not allow_legacy_calib:
            raise ValueError(
                f"标定文件 {resolved_calib} 缺少 synthetic 溯源字段，请加 --allow-legacy-calib 或使用实测标定"
            )
        if getattr(loaded, "synthetic", False) and not demo:
            raise ValueError(
                f"标定文件 {resolved_calib} 为合成占位标定 (synthetic: true)，请显式开启 demo=True/--demo"
            )
        return loaded

    if not demo:
        raise FileNotFoundError(
            f"未找到标定文件 {resolved_calib}；请提供有效标定文件或开启 demo=True/--demo"
        )
    return RoadCalibrator(
        image_points=[(400.0, 300.0), (880.0, 300.0), (1150.0, 680.0), (130.0, 680.0)],
        width_m=6.0,
        length_m=15.0,
        name="default_campus_road",
        synthetic=True,
    )


def profile_pipeline_stages(
    video_path: Optional[str | int | Path] = None,
    *,
    calibrator: Optional[RoadCalibrator] = None,
    calib_path: Optional[str | Path] = None,
    demo: bool = False,
    allow_legacy_calib: bool = False,
    weights: str | Path = str(DEFAULT_WEIGHTS),
    tracker_config: str | Path = str(DEFAULT_TRACKER),
    object_policy_config: str | Path = DEFAULT_OBJECT_POLICY_CONFIG,
    device: Optional[str] = None,
    conf: Optional[float] = None,
    iou: float = 0.7,
    imgsz: int = 640,
    fps: Optional[float] = None,
    max_frames: Optional[int] = 30,
    speed_model: str = "ema",
    conflict_model: str = "cvm_circle",
    enable_stitching: bool = False,
    include_cyclist_vru: bool = False,
    render_video: bool = True,
) -> dict:
    """在临时目录中运行流水线并统计模型加载及逐帧 6 阶段耗时，产出 schema_version: 1 报告。"""
    from . import pipeline as _pipeline_mod

    for flag_name, flag_val in (
        ("demo", demo),
        ("allow_legacy_calib", allow_legacy_calib),
        ("enable_stitching", enable_stitching),
        ("include_cyclist_vru", include_cyclist_vru),
        ("render_video", render_video),
    ):
        if type(flag_val) is not bool:
            raise ValueError(f"{flag_name} 必须是布尔值")

    if max_frames is not None and (type(max_frames) is not int or max_frames < 0):
        raise ValueError("max_frames 必须是 >= 0 的整数或 None")
    if type(speed_model) is not str or speed_model not in ("ema", "kalman", "ctrv"):
        raise ValueError("speed_model 必须是 'ema'、'kalman' 或 'ctrv'")
    if type(conflict_model) is not str or conflict_model not in ("cvm_circle", "obb_ssm"):
        raise ValueError("conflict_model 必须是 'cvm_circle' 或 'obb_ssm'")
    if is_blank_path(weights):
        raise ValueError("模型权重路径不能为空")
    if is_blank_path(tracker_config):
        raise ValueError("跟踪配置路径不能为空")
    if is_blank_path(object_policy_config):
        raise ValueError("对象策略配置路径不能为空")

    if video_path is None:
        if not demo:
            raise ValueError("必须指定 video_path，或开启 demo=True/--demo")
    elif type(video_path) is bool or is_blank_path(video_path):
        raise ValueError("视频输入源不能为空或布尔值")

    resolved_calibrator = _resolve_calibrator(
        calibrator=calibrator,
        calib_path=calib_path,
        demo=demo,
        allow_legacy_calib=allow_legacy_calib,
    )

    pipe = _pipeline_mod.ConflictPerceptionPipeline(
        calibrator=resolved_calibrator,
        weights=str(weights),
        tracker_config=str(tracker_config),
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        fps=fps,
        object_policy_config=object_policy_config,
        speed_model=speed_model,
        conflict_model=conflict_model,
        enable_stitching=enable_stitching,
        include_cyclist_vru=include_cyclist_vru,
    )

    stage_samples: list[dict[str, float]] = []
    model_load_ms = 0.0
    width = 0
    height = 0
    fps_used = 0.0
    src_label = ""
    wall_start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        effective_source = video_path
        if effective_source is None:
            fixture_video = ROOT / "runs" / "tracking-fixtures" / "sample_01" / "sample_a.mp4"
            if fixture_video.is_file():
                effective_source = fixture_video
            else:
                effective_source = _create_synthetic_demo_video(tmp_root / "demo_input.mp4")

        capture_arg, src_label, is_live_stream = _pipeline_mod.parse_video_source(effective_source)
        out_video_path = tmp_root / "bench_annotated.mp4"
        jsonl_path = tmp_root / "conflicts.jsonl"
        events_path = tmp_root / "events.jsonl"
        csv_path = tmp_root / "speeds.csv"

        with ExitStack() as resources:
            cap = _pipeline_mod.cv2.VideoCapture(capture_arg)
            resources.callback(cap.release)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频输入源: {src_label}")

            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            if is_live_stream:
                source_fps = raw_fps if (math.isfinite(raw_fps) and 0 < raw_fps <= 240) else None
                fps_used = pipe.fps if pipe.fps is not None else (source_fps or 25.0)
                video_clock = (
                    VideoClock(fps=fps_used, force_fps=True)
                    if pipe.fps is not None
                    else _pipeline_mod.LiveStreamClock()
                )
            else:
                source_fps = raw_fps if (math.isfinite(raw_fps) and raw_fps > 0) else None
                fps_used = pipe.fps if pipe.fps is not None else source_fps
                if fps_used is None:
                    raise RuntimeError("视频缺少有效 FPS；请通过 --fps 显式指定")
                video_clock = VideoClock(fps=fps_used, force_fps=pipe.fps is not None)

            raw_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            raw_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if not (math.isfinite(raw_w) and math.isfinite(raw_h)) or raw_w <= 0 or raw_h <= 0:
                raise RuntimeError("视频分辨率元数据无效")
            width = int(raw_w)
            height = int(raw_h)
            if width <= 0 or height <= 0:
                raise RuntimeError("视频分辨率元数据无效")

            pipe._configure_speed_estimator(fps_used)
            pipe._configure_event_tracker(fps_used)
            pipe._configure_conflict_and_stitcher()

            t_load_0 = time.perf_counter()
            session = _pipeline_mod.TrackingSession(
                source=src_label,
                weights=pipe.weights,
                tracker_path=pipe.tracker_config,
                conf=pipe.conf,
                iou=pipe.iou,
                imgsz=pipe.imgsz,
                device=pipe.device,
            )
            model_load_ms = (time.perf_counter() - t_load_0) * 1000.0

            writer = None
            if render_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = _pipeline_mod.cv2.VideoWriter(
                    str(out_video_path), fourcc, fps_used, (width, height)
                )
                resources.callback(writer.release)
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建临时渲染输出视频: {out_video_path}")

            jsonl_file = resources.enter_context(open(jsonl_path, "w", encoding="utf-8"))
            events_file = resources.enter_context(open(events_path, "w", encoding="utf-8"))
            csv_file = resources.enter_context(open(csv_path, "w", encoding="utf-8", newline=""))
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "schema_version",
                "frame_idx", "timestamp_s", "track_id", "category",
                "x_m", "y_m", "speed_kmh", "is_speeding", "risk_level", "ttc_s",
                "event_id", "is_reliable", "is_stationary",
            ])

            counts = {"critical": 0, "caution": 0, "speeding": 0}

            def _drain_alert_events() -> None:
                pipe._drain_alert_events_into(events_file, counts)

            def _resolve_live_ids(out_of_zone: set[int]) -> set[int]:
                live = set(session.retained_ids())
                live -= out_of_zone
                return live

            # 仅在局部管线实例上挂接分阶段计时包装，零侵入 pipeline.py 主源码且无全局副作用
            frame_marks: dict[str, float] = {}
            orig_analyze = pipe.conflict_analyzer.analyze_frame
            orig_draw_zone = pipe.visualizer.draw_calibrated_ground_zone

            def _timed_analyze(*args, **kwargs):
                if "conflict_start" not in frame_marks:
                    frame_marks["conflict_start"] = time.perf_counter()
                return orig_analyze(*args, **kwargs)

            def _timed_draw_zone(*args, **kwargs):
                if "render_start" not in frame_marks:
                    frame_marks["render_start"] = time.perf_counter()
                return orig_draw_zone(*args, **kwargs)

            pipe.conflict_analyzer.analyze_frame = _timed_analyze
            pipe.visualizer.draw_calibrated_ground_zone = _timed_draw_zone
            try:
                frame_idx = 0
                while True:
                    if max_frames is not None and frame_idx >= max_frames:
                        break

                    frame_marks.clear()
                    t_frame_start = time.perf_counter()
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    if (
                        not hasattr(frame, "shape")
                        or len(frame.shape) < 2
                        or (frame.shape[1], frame.shape[0]) != (width, height)
                    ):
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
                    t_decode_end = time.perf_counter()

                    frame_record = session.process(frame, timestamp_s, time_kind)
                    t_detect_end = time.perf_counter()

                    frame_marks["geo_start"] = t_detect_end
                    pipe._process_single_frame(
                        frame_record.tracks,
                        resolve_live_ids=_resolve_live_ids,
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        jsonl_file=jsonl_file,
                        csv_writer=csv_writer,
                        drain_events_cb=_drain_alert_events,
                        frame=(frame if render_video else None),
                        writer=writer,
                    )
                    t_frame_end = time.perf_counter()

                    t_conflict_start = frame_marks.get("conflict_start", t_frame_end)
                    t_render_start = frame_marks.get("render_start", t_frame_end)

                    decode_ms = max(0.0, (t_decode_end - t_frame_start) * 1000.0)
                    detect_ms = max(0.0, (t_detect_end - t_decode_end) * 1000.0)
                    geo_ms = max(0.0, (t_conflict_start - t_detect_end) * 1000.0)
                    conflict_ms = max(0.0, (t_render_start - t_conflict_start) * 1000.0)
                    render_ms = max(0.0, (t_frame_end - t_render_start) * 1000.0)
                    total_ms = max(0.0, (t_frame_end - t_frame_start) * 1000.0)

                    stage_samples.append({
                        "decode": round(decode_ms, 4),
                        "detect_track": round(detect_ms, 4),
                        "geometry_speed": round(geo_ms, 4),
                        "conflict_events": round(conflict_ms, 4),
                        "render_write": round(render_ms, 4),
                        "total_frame": round(total_ms, 4),
                    })
                    frame_idx += 1

                pipe.event_tracker.flush(reason="eof")
                _drain_alert_events()
            finally:
                pipe.conflict_analyzer.analyze_frame = orig_analyze
                pipe.visualizer.draw_calibrated_ground_zone = orig_draw_zone

    wall_elapsed_ms = max(0.0, (time.perf_counter() - wall_start) * 1000.0)
    timing_summary = summarize_stage_timings(stage_samples)
    frames_profiled = timing_summary["frames_profiled"]
    wall_fps = (
        frames_profiled / (wall_elapsed_ms / 1000.0)
        if (frames_profiled > 0 and wall_elapsed_ms > 0.0)
        else 0.0
    )

    report = {
        "schema_version": 1,
        "source": src_label,
        "demo": demo,
        "calibration_name": resolved_calibrator.name,
        "calibration_synthetic": bool(resolved_calibrator.synthetic),
        "weights": str(Path(weights).expanduser().resolve()),
        "device": device,
        "speed_model": speed_model,
        "conflict_model": conflict_model,
        "enable_stitching": enable_stitching,
        "include_cyclist_vru": include_cyclist_vru,
        "render_video": render_video,
        "fps_used": float(fps_used),
        "resolution": [width, height],
        "max_frames": max_frames,
        "frames_profiled": frames_profiled,
        "model_load_ms": round(model_load_ms, 3),
        "wall_elapsed_ms": round(wall_elapsed_ms, 3),
        "end_to_end_fps": timing_summary["end_to_end_fps"],
        "frame_processing_fps": timing_summary["frame_processing_fps"],
        "wall_fps": round(wall_fps, 3),
        "stages": timing_summary["stages"],
        "steady_state": timing_summary["steady_state"],
    }
    _reject_non_finite_tree(report, "benchmark_report")
    return report


def benchmark_to_markdown(report: Mapping[str, Any]) -> str:
    """将分阶段耗时基准报告格式化为适合实验记录的 Markdown 表格。"""
    if not isinstance(report, Mapping) or "stages" not in report or not isinstance(report["stages"], Mapping):
        raise ValueError("report 必须是包含 stages 的基准报告字典")
    _reject_non_finite_tree(dict(report), "report")

    frames_profiled = report.get("frames_profiled", 0)
    frame_fps = float(report.get("frame_processing_fps", report.get("end_to_end_fps", 0.0)))
    raw_wall_fps = report.get("wall_fps")
    wall_ms = report.get("wall_elapsed_ms")
    if raw_wall_fps is not None and not isinstance(raw_wall_fps, (bool, np.bool_)):
        wall_fps_opt: Optional[float] = float(raw_wall_fps)
    elif (
        type(frames_profiled) is int
        and frames_profiled > 0
        and wall_ms is not None
        and not isinstance(wall_ms, (bool, np.bool_))
        and float(wall_ms) > 0.0
    ):
        wall_fps_opt = frames_profiled / (float(wall_ms) / 1000.0)
    else:
        wall_fps_opt = None

    lines = [
        "# Pipeline Stage Benchmark Report",
        "",
        f"- Source: `{report.get('source', 'unknown')}`",
        (
            f"- Mode: speed=`{report.get('speed_model', 'ema')}`, "
            f"conflict=`{report.get('conflict_model', 'cvm_circle')}`, "
            f"stitching=`{bool(report.get('enable_stitching', False))}`"
        ),
        f"- Frames profiled: {frames_profiled}",
        f"- Model load time: {float(report.get('model_load_ms', 0.0)):.3f} ms",
        f"- Frame processing throughput (excl. model load): {frame_fps:.3f} FPS",
    ]
    if wall_fps_opt is not None:
        lines.append(f"- Wall-clock throughput (incl. model load): {wall_fps_opt:.3f} FPS")
    steady = report.get("steady_state")
    if isinstance(steady, Mapping) and "steady_state_fps" in steady:
        warmup_excl = int(steady.get("warmup_frames_excluded", 0))
        frames_eval = int(steady.get("frames_evaluated", 0))
        steady_fps_raw = steady.get("steady_state_fps")
        steady_str = (
            f"{float(steady_fps_raw):.3f} FPS"
            if steady_fps_raw is not None and frames_eval > 0
            else "N/A (insufficient frames)"
        )
        lines.append(
            f"- Steady-state throughput (warmup excluded={warmup_excl}, frames={frames_eval}): "
            f"{steady_str}"
        )
    lines.extend([
        "",
        "| Stage | Mean (ms) | P50 (ms) | P95 (ms) | Max (ms) | Share (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    stages = report["stages"]
    for stage in STAGE_NAMES:
        if stage not in stages:
            continue
        row = stages[stage]
        lines.append(
            f"| {stage} | {float(row['mean_ms']):.3f} | {float(row['p50_ms']):.3f} | "
            f"{float(row['p95_ms']):.3f} | {float(row['max_ms']):.3f} | {float(row['share_pct']):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="分阶段耗时基准与吞吐统计 CLI")
    parser.add_argument("--source", type=str, default=None, help="输入视频路径或流地址")
    parser.add_argument("--calib", type=str, default=None, help="空间单应性标定配置文件路径 (YAML)")
    parser.add_argument("--demo", action="store_true", help="使用合成占位标定与默认样例视频运行基准测试")
    parser.add_argument("--allow-legacy-calib", action="store_true", help="允许加载无 synthetic 字段的旧版标定文件")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="YOLOv8 检测权重路径")
    parser.add_argument("--tracker", type=str, default=str(DEFAULT_TRACKER), help="ByteTrack 配置文件路径")
    parser.add_argument("--object-policy", type=str, default=str(DEFAULT_OBJECT_POLICY_CONFIG), help="类别策略 YAML")
    parser.add_argument("--device", type=str, default=None, help="推理设备 (如 cpu 或 0)")
    parser.add_argument("--conf", type=float, default=None, help="检测置信度阈值")
    parser.add_argument("--fps", type=float, default=None, help="强制指定视频帧率")
    parser.add_argument("--max-frames", type=int, default=30, help="最多剖析帧数，默认 30")
    parser.add_argument(
        "--speed-model",
        type=str,
        choices=("ema", "kalman", "ctrv"),
        default="ema",
        help="物理测速模型: ema / kalman / ctrv",
    )
    parser.add_argument(
        "--conflict-model",
        type=str,
        choices=("cvm_circle", "obb_ssm"),
        default="cvm_circle",
        help="冲突研判模型: cvm_circle / obb_ssm",
    )
    parser.add_argument("--enable-stitching", action="store_true", help="开启短时遮挡断轨缝合")
    parser.add_argument("--include-cyclist-vru", action="store_true", help="纳入骑行者 VRU 冲突评估")
    parser.add_argument("--no-render", action="store_true", help="跳过临时视频渲染与写盘以测试纯算法耗时")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 或 Markdown 文件路径")
    parser.add_argument("--markdown", action="store_true", help="以 Markdown 表格输出而非 JSON")
    args = parser.parse_args(argv)

    # 必须在 Path() 解析和加载前拦截所有空白路径参数，防止退化为当前工作目录 '.'
    if args.source is not None and not args.source.strip():
        parser.error("--source 不能为空")
    if args.calib is not None and not args.calib.strip():
        parser.error("--calib 不能为空")
    if not args.weights.strip():
        parser.error("--weights 不能为空")
    if not args.tracker.strip():
        parser.error("--tracker 不能为空")
    if not args.object_policy.strip():
        parser.error("--object-policy 不能为空")
    if args.output is not None and not args.output.strip():
        parser.error("--output 不能为空")
    if args.source is None and not args.demo:
        parser.error("必须指定 --source，或开启 --demo")
    if args.max_frames is not None and args.max_frames < 0:
        parser.error("--max-frames 必须是 >= 0 的整数")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps 必须是有限正数")
    if args.conf is not None and (not math.isfinite(args.conf) or not (0.0 < args.conf <= 1.0)):
        parser.error("--conf 必须在 (0, 1] 区间内")

    report = profile_pipeline_stages(
        video_path=args.source,
        calib_path=args.calib,
        demo=args.demo,
        allow_legacy_calib=args.allow_legacy_calib,
        weights=args.weights,
        tracker_config=args.tracker,
        object_policy_config=args.object_policy,
        device=args.device,
        conf=args.conf,
        fps=args.fps,
        max_frames=args.max_frames,
        speed_model=args.speed_model,
        conflict_model=args.conflict_model,
        enable_stitching=args.enable_stitching,
        include_cyclist_vru=args.include_cyclist_vru,
        render_video=not args.no_render,
    )

    content = (
        benchmark_to_markdown(report)
        if args.markdown
        else json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    )
    if args.output is not None:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
