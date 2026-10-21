"""python -m tracking：独立跟踪入口，不改变现有 detection 命令。"""

# 原理笔记：docs/modules/02_多目标跟踪_ByteTrack_解释文档.md，第 8 节。
# CLI 将参数整理为关键字传给单视频或目录入口，再把汇总状态转成进程退出码。

import argparse
import json
from pathlib import Path

from detection.core import DEFAULT_WEIGHTS
from .core import DEFAULT_TRACKER, track_video
from .batch import track_directory


def positive_int(text):
    number = int(text)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack 车辆行人视频跟踪")
    parser.add_argument("--source", required=True, help="本地视频文件或视频目录（不递归）")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--tracker", default=str(DEFAULT_TRACKER), help="独立 ByteTrack YAML")
    parser.add_argument("--output", help="尚不存在的输出目录；默认自动创建")
    parser.add_argument("--classes", nargs="+", help="按名称筛选，如 person car bus truck")
    parser.add_argument("--device", default=None, help="cpu 或 GPU 编号，例如 0")
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--conf", type=float, default=None, help="默认跟随 track_low_thresh，不能高于它")
    parser.add_argument("--iou", type=float, default=0.7, help="检测 NMS 阈值，不是跟踪匹配阈值")
    parser.add_argument("--trail-length", type=positive_int, default=60, help="最近多少源帧的观测轨迹")
    parser.add_argument("--max-frames", type=positive_int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="覆盖输出 FPS 和固定间隔时间轴")
    parser.add_argument("--show", action="store_true", help="本地桌面显示，支持 Space/P 暂停、Q/Esc 退出")
    args = parser.parse_args(argv)
    try:
        is_batch = Path(args.source).expanduser().is_dir()
        # vars(args) 的键与函数参数对应；一个目录作为一个批次，各视频内部状态独立。
        output = (track_directory if is_batch else track_video)(**vars(args))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        notes = "".join(f"{note}\n" for note in getattr(exc, "__notes__", []))
        parser.exit(1, f"跟踪失败：{exc}\n{notes}")
    if is_batch:
        summary = json.loads((output / "batch_summary.json").read_text(encoding="utf-8"))
        print(f"已处理 {len(summary['results'])}/{summary['videos_discovered']} 段视频，状态：{summary['status']}")
        print(f"结果已保存：{output}")
        if any(r.get("reported_frame_count_mismatch") for r in summary["results"]):
            print("注意：部分视频的解码帧数与元数据不一致，请核对批次清单及源视频。")
        if summary["status"] == "interrupted":
            # 130 对应主动中断；打印了输出位置，也不代表全部视频已成功处理。
            raise SystemExit(130)
        if summary["status"] == "completed_with_errors" or any(r["status"] == "failed" for r in summary["results"]):
            raise SystemExit(1)
        return
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print(f"已处理 {summary['frames_processed']} 帧，结束原因：{summary['stop_reason']}")
    print(f"结果已保存：{output}")
    if summary["reported_frame_count_mismatch"]:
        print("注意：实际解码帧数与视频元数据不一致，请检查源文件与解码日志。")
    if summary["stop_reason"] == "interrupted":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
