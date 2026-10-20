# -*- coding: utf-8 -*-
"""
校园道路车辆—行人冲突感知与测速预警命令行执行入口
"""

from __future__ import annotations

import argparse
import math
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
        "--window-s",
        type=float,
        default=None,
        help="测速滑窗时长 (秒)，按视频 FPS 自动换算帧数；设置后覆盖 --window",
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
        "--parked-speed",
        type=float,
        default=0.1,
        help="判定持续静止/泊车的速度阈值 (m/s)，默认 0.1",
    )
    parser.add_argument(
        "--parked-frames",
        type=int,
        default=8,
        help="连续多少帧低于 --parked-speed 判定为静止车辆，默认 8",
    )
    parser.add_argument(
        "--parked-seconds",
        type=float,
        default=None,
        help="按秒指定静止确认时长，按视频 FPS 换算帧数；设置后覆盖 --parked-frames",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="标定文件缺失时允许使用合成占位标定（仅演示，测速结果不可信）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否开启本地桌面实时播放窗口",
    )
    args = parser.parse_args(argv)

    if not math.isfinite(args.ema_alpha) or not (0 < args.ema_alpha <= 1):
        parser.error("--ema-alpha 必须在 (0, 1] 区间")
    if args.window < 3:
        parser.error("--window 必须是 >= 3 的整数")
    if args.window_s is not None and (not math.isfinite(args.window_s) or args.window_s <= 0):
        parser.error("--window-s 必须为正数")
    if not (math.isfinite(args.ttc_critical) and math.isfinite(args.ttc_caution)
            and math.isfinite(args.horizon)
            and 0 < args.ttc_critical <= args.ttc_caution <= args.horizon):
        parser.error("TTC 阈值需满足 0 < --ttc-critical <= --ttc-caution <= --horizon")
    if not math.isfinite(args.parked_speed) or args.parked_speed <= 0:
        parser.error("--parked-speed 必须为正数")
    if args.parked_frames < 1:
        parser.error("--parked-frames 必须是 >= 1 的整数")
    if args.parked_seconds is not None and (not math.isfinite(args.parked_seconds) or args.parked_seconds <= 0):
        parser.error("--parked-seconds 必须为正数")
    if not math.isfinite(args.speed_limit) or args.speed_limit <= 0:
        parser.error("--speed-limit 必须为正数")

    calib_path = Path(args.calib).expanduser().resolve()
    synthetic_calib = None
    if calib_path.exists():
        calibrator = RoadCalibrator.from_yaml(calib_path)
        if calibrator.synthetic:
            if not args.demo:
                parser.error(
                    f"标定文件 {calib_path} 为合成占位标定 (synthetic: true)，仅供演示；"
                    "请用 python -m analysis.interactive_calibrator 实测标定，或显式加 --demo"
                )
            print("[!] 警告: 当前使用合成占位标定，物理测速与 TTC 结果不可信！")
    else:
        if not args.demo:
            parser.error(
                f"标定文件 {calib_path} 不存在。"
                "请先运行 python -m analysis.interactive_calibrator 实测标定，"
                "或显式加 --demo 使用合成占位标定（仅演示）"
            )
        # 合成标定仅供流程演示；其物理测速与 TTC 结果不可信。
        # 仅在内存中生成，留档写入运行输出目录，绝不写回正式标定路径
        print(f"[!] 警告: 标定文件 {calib_path} 不存在，自动生成的合成标定仅供流程演示，")
        print(f"[!] 其物理测速与 TTC 结果不可信！请先用 python -m analysis.interactive_calibrator 实测标定。")
        synthetic_calib = RoadCalibrator(
            image_points=[(400, 300), (880, 300), (1150, 680), (130, 680)],
            width_m=6.0,
            length_m=15.0,
            name="default_campus_road",
            synthetic=True,
        )
        calibrator = synthetic_calib

    out_dir = args.output
    if not out_dir:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / "analysis" / f"run-{timestamp}"
    if synthetic_calib is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        synthetic_calib.to_yaml(Path(out_dir) / "calibration.synthetic.yaml")

    pipeline = ConflictPerceptionPipeline(
        calibrator=calibrator,
        weights=args.weights,
        tracker_config=args.tracker,
        speed_limit_kmh=args.speed_limit,
        device=args.device,
        conf=args.conf,
        ema_alpha=args.ema_alpha,
        window_size=args.window,
        window_seconds=args.window_s,
        ttc_critical_s=args.ttc_critical,
        ttc_caution_s=args.ttc_caution,
        horizon_s=args.horizon,
        stationary_speed_ms=args.parked_speed,
        stationary_min_frames=args.parked_frames,
        stationary_seconds=args.parked_seconds,
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
