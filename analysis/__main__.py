# -*- coding: utf-8 -*-
"""
校园道路车辆—行人冲突感知与测速预警命令行执行入口
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from detection.core import DEFAULT_WEIGHTS
from tracking.core import DEFAULT_TRACKER
from .calibration import RoadCalibrator
from .pipeline import ConflictPerceptionPipeline


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="校园道路车辆—行人冲突感知与 30 km/h 测速预警系统"
    )
    parser.add_argument("--source", type=str, required=True, help="输入视频路径 (MP4/AVI)")
    parser.add_argument(
        "--calib",
        type=str,
        default="configs/calibration.yaml",
        help="道路空间单应性标定配置文件路径 (YAML)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="YOLOv8 检测权重路径 (默认 checkpoints/yolo/yolov8n.pt 或 weights/yolov8n.pt)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=str(DEFAULT_TRACKER),
        help="ByteTrack 配置文件路径 (默认 configs/bytetrack.yaml)",
    )
    parser.add_argument(
        "--speed-limit",
        type=float,
        default=30.0,
        help="校园道路限速阈值 (km/h)，默认 30.0",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果保存目录 (默认 runs/analysis/run-*)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备 (cpu 或 GPU 编号如 0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="检测置信度阈值",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最大处理帧数限制 (测试用)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.35,
        help="速度 EMA 平滑系数 (0~1)，越小越平滑、响应越慢，默认 0.35",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=8,
        help="测速滑窗帧数，默认 8",
    )
    parser.add_argument(
        "--ttc-critical",
        type=float,
        default=1.8,
        help="紧急碰撞预警 TTC 阈值 (秒)，默认 1.8",
    )
    parser.add_argument(
        "--ttc-caution",
        type=float,
        default=3.5,
        help="注意预警 TTC 阈值 (秒)，默认 3.5",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=3.5,
        help="轨迹外推预测视界 (秒)，默认 3.5",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否开启本地桌面实时播放窗口",
    )
    args = parser.parse_args(argv)

    calib_path = Path(args.calib).expanduser().resolve()
    if not calib_path.exists():
        # 若默认配置不存在，自动生成一个标准校园参考标定配置；
        # 该参数为占位合成值，真实测速前必须用 interactive_calibrator 重新标定
        print(f"[!] 警告: 标定文件 {calib_path} 不存在，自动生成的合成标定仅供流程演示，")
        print(f"[!] 其物理测速与 TTC 结果不可信！请先用 python -m analysis.interactive_calibrator 实测标定。")
        sample_calib = RoadCalibrator(
            image_points=[(400, 300), (880, 300), (1150, 680), (130, 680)],
            width_m=6.0,
            length_m=15.0,
            name="default_campus_road",
        )
        sample_calib.to_yaml(calib_path)
        calibrator = sample_calib
    else:
        calibrator = RoadCalibrator.from_yaml(calib_path)

    out_dir = args.output
    if not out_dir:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / "analysis" / f"run-{timestamp}"

    pipeline = ConflictPerceptionPipeline(
        calibrator=calibrator,
        weights=args.weights,
        tracker_config=args.tracker,
        speed_limit_kmh=args.speed_limit,
        device=args.device,
        conf=args.conf,
        ema_alpha=args.ema_alpha,
        window_size=args.window,
        ttc_critical_s=args.ttc_critical,
        ttc_caution_s=args.ttc_caution,
        horizon_s=args.horizon,
    )

    summary = pipeline.process_video(
        video_path=args.source,
        output_dir=out_dir,
        show_window=args.show,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
