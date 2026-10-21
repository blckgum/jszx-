"""顺序处理目录中的本地视频；每段使用独立会话，失败信息写入批次清单。"""

import json
from pathlib import Path

from detection.core import VIDEO_SUFFIXES, new_run, write_json
from .core import DEFAULT_TRACKER, read_tracker_config, track_video
from .motion import DEFAULT_MOTION_CONFIG, read_motion_config


def track_directory(source, *, output=None, **options):
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("批量输入必须是本地目录")
    # 启动时固定清单，按文件名排序；不递归，不把旧输出里的视频再次当输入。
    videos = sorted((path for path in source.iterdir()
                     if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES), key=lambda path: path.name)
    if not videos:
        raise ValueError("目录中没有支持的视频文件")
    read_tracker_config(options.get("tracker", DEFAULT_TRACKER))
    read_motion_config(options.get("motion_config", DEFAULT_MOTION_CONFIG))
    output = new_run(output, "track-batch")
    summary = {"schema_version": 1, "source": str(source), "videos_discovered": len(videos),
               "status": "running", "results": []}
    summary_path = output / "batch_summary.json"
    write_json(summary_path, summary)
    for index, video in enumerate(videos):
        destination = output / f"{index + 1:04d}_{video.stem}"
        result = {"source": str(video), "output": str(destination)}
        try:
            # 复用单视频入口，让帧缓存、YOLO predictor、Kalman 状态和 ID 完全分开。
            track_video(video, output=destination, **options)
            child = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            result.update(status=child["status"], stop_reason=child["stop_reason"],
                          frames_processed=child["frames_processed"], session_id=child["session_id"],
                          reported_frame_count_mismatch=child.get("reported_frame_count_mismatch", False))
        except KeyboardInterrupt:
            result.update(status="interrupted", stop_reason="interrupted")
        except Exception as exc:
            # 坏视频不阻塞后续文件；批次 CLI 最终以非零状态提示存在失败。
            result.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        summary["results"].append(result)
        stopped = result.get("stop_reason") in {"interrupted", "user_quit"}
        if stopped:
            summary["status"] = "interrupted" if result["stop_reason"] == "interrupted" else "stopped"
        # 每个文件完成后更新清单；中途被强制终止时，先前的视频结果仍可追溯。
        write_json(summary_path, summary)
        if stopped:
            break
    if summary["status"] == "running":
        summary["status"] = "completed_with_errors" if any(r["status"] == "failed" for r in summary["results"]) else "completed"
    summary["videos_not_processed"] = len(videos) - len(summary["results"])
    write_json(summary_path, summary)
    return output
