# -*- coding: utf-8 -*-
"""
多视频目录批量冲突感知与测速运行器 (Directory Batch Analyzer)
=============================================================================
顺序扫描指定目录中的本地视频，为每个视频创建独立子目录并重建独立管线会话
（独立 ByteTrack ID、测速滤波器与告警事件状态机），汇总写入 batch_summary.json。
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

from detection.core import VIDEO_SUFFIXES
from .calibration import RoadCalibrator
from .pipeline import ConflictPerceptionPipeline


def _write_batch_summary(path: Path, summary: dict):
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def analyze_directory(
    source_dir: str | Path,
    *,
    calibrator: RoadCalibrator,
    output_dir: Optional[str | Path] = None,
    max_frames: Optional[int] = None,
    show_window: bool = False,
    **pipeline_kwargs,
) -> dict:
    """顺序分析目录下所有支持的视频文件，各自隔离状态并在异常视频处容错继续。"""
    src_dir = Path(source_dir).expanduser().resolve()
    if not src_dir.is_dir():
        raise ValueError(f"批量分析输入必须是已存在的本地目录: {src_dir}")

    videos = sorted(
        (p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda p: p.name,
    )
    if not videos:
        raise ValueError(f"目录中没有支持的视频文件: {src_dir}")

    if output_dir is not None:
        out_root = Path(output_dir).expanduser().resolve()
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = (Path("runs") / "analysis" / f"batch-{timestamp}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    batch_summary = {
        "schema_version": 1,
        "kind": "analysis_batch",
        "source_dir": str(src_dir),
        "output_dir": str(out_root),
        "status": "running",
        "videos_discovered": len(videos),
        "videos_not_processed": len(videos),
        "total_alert_events": 0,
        "total_speeding_alerts": 0,
        "total_caution_events": 0,
        "total_critical_conflicts": 0,
        "results": [],
    }
    summary_path = out_root / "batch_summary.json"
    _write_batch_summary(summary_path, batch_summary)

    for idx, video in enumerate(videos):
        dest = out_root / f"{idx + 1:04d}_{video.stem}"
        entry: dict = {"source": str(video), "output": str(dest)}
        try:
            # 每段视频独立实例化管道，确保测速滑窗、告警去抖状态机和 Track ID 完全隔离
            pipeline = ConflictPerceptionPipeline(calibrator=calibrator, **pipeline_kwargs)
            child = pipeline.process_video(
                video_path=video,
                output_dir=dest,
                show_window=show_window,
                max_frames=max_frames,
            )
            entry.update({
                "status": child["status"],
                "stop_reason": child["stop_reason"],
                "frames_processed": child["frames_processed"],
                "total_alert_events": child["total_alert_events"],
                "total_speeding_alerts": child["total_speeding_alerts"],
                "total_caution_events": child["total_caution_events"],
                "total_critical_conflicts": child["total_critical_conflicts"],
                "max_observed_speed_kmh": child["max_observed_speed_kmh"],
            })
            batch_summary["total_alert_events"] += child["total_alert_events"]
            batch_summary["total_speeding_alerts"] += child["total_speeding_alerts"]
            batch_summary["total_caution_events"] += child["total_caution_events"]
            batch_summary["total_critical_conflicts"] += child["total_critical_conflicts"]
        except KeyboardInterrupt:
            entry.update({"status": "interrupted", "stop_reason": "interrupt"})
            batch_summary["results"].append(entry)
            batch_summary["status"] = "interrupted"
            batch_summary["videos_not_processed"] = len(videos) - len(batch_summary["results"])
            _write_batch_summary(summary_path, batch_summary)
            break
        except Exception as exc:
            entry.update({
                "status": "failed",
                "stop_reason": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })

        batch_summary["results"].append(entry)
        if entry.get("stop_reason") == "user_quit":
            batch_summary["status"] = "stopped"
        batch_summary["videos_not_processed"] = len(videos) - len(batch_summary["results"])
        _write_batch_summary(summary_path, batch_summary)
        if batch_summary["status"] == "stopped":
            break

    if batch_summary["status"] == "running":
        has_failures = any(item["status"] == "failed" for item in batch_summary["results"])
        batch_summary["status"] = "completed_with_errors" if has_failures else "completed"
    batch_summary["videos_not_processed"] = len(videos) - len(batch_summary["results"])
    _write_batch_summary(summary_path, batch_summary)
    return batch_summary
