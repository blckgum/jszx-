# -*- coding: utf-8 -*-
"""
交互式 4 点地面标定工具 (Interactive Road Calibrator CLI)
=============================================================================
支持在校园路面照片或视频首帧上，使用鼠标依次点击 4 个角点：
[1] 左上 -> [2] 右上 -> [3] 右下 -> [4] 左下，
输入实际道路物理长宽，一键求解并导出 configs/calibration.yaml。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .calibration import RoadCalibrator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _read_calibration_image(src: Path) -> np.ndarray:
    """读取标定图片，兼容 Windows 含中文路径并校验解码结果。"""
    try:
        raw = np.fromfile(str(src), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size > 0 else None
    except Exception:
        image = None
    if image is None:
        image = cv2.imread(str(src))
    if image is None:
        raise RuntimeError(f"无法解码读取标定图片文件: {src}")
    return image


def run_interactive_calibration(
    source_path: str | Path,
    width_m: float,
    length_m: float,
    output_path: str | Path = "configs/calibration.yaml",
    predefined_points: Optional[List[Tuple[float, float]]] = None,
) -> RoadCalibrator:
    """执行交互式或参数化标定流程"""
    if (
        type(width_m) not in (int, float)
        or type(length_m) not in (int, float)
        or not (math.isfinite(width_m) and math.isfinite(length_m))
        or width_m <= 0
        or length_m <= 0
    ):
        raise ValueError("道路的实际物理宽度和长度必须为大于 0 的有限正数（单位：米）")

    src = Path(source_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"未找到标定底图或视频文件: {src}")

    # 支持直接从图像或视频中抓取第 1 帧
    if src.suffix.lower() in IMAGE_SUFFIXES:
        image = _read_calibration_image(src)
    else:
        cap = cv2.VideoCapture(str(src))
        ok, image = cap.read()
        cap.release()
        if not ok or image is None:
            raise RuntimeError(f"无法从视频中解码读取首帧用于标定: {src}")

    if predefined_points is not None:
        if len(predefined_points) != 4:
            raise ValueError(f"地面标定必须严格提供 4 个角点，当前提供了 {len(predefined_points)} 个")
        points = list(predefined_points)
        print(f"[*] 使用预设参数标定 4 点: {points}")
    else:
        # 交互式窗口标定
        points: List[Tuple[float, float]] = []
        window_name = "Campus Road Calibrator (Click 4 ground corners: TL -> TR -> BR -> BL)"

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((float(x), float(y)))
                print(f"[*] 选取第 {len(points)}/4 个角点: 像素坐标 ({x}, {y})")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

        print("=" * 60)
        print("[*] 交互式空间标定窗口已打开:")
        print("    请用鼠标左键按顺时针依次点击道路地面的 4 个角点:")
        print("    [1] 左上角 -> [2] 右上角 -> [3] 右下角 -> [4] 左下角")
        print("    按键盘 'R' 重置选点，点满 4 点后按 'Enter' 或 'Space' 保存退出，'Q' 取消。")
        print("=" * 60)

        while True:
            display = image.copy()
            # 绘制已点击的点
            for idx, pt in enumerate(points):
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(display, (px, py), 6, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    f"P{idx+1}",
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            # 绘制闭合多边形轮廓
            if len(points) >= 2:
                pts_arr = np.array(points, dtype=np.int32)
                cv2.polylines(display, [pts_arr], isClosed=(len(points) == 4), color=(0, 255, 0), thickness=2)

            # 顶部操作提示
            cv2.putText(
                display,
                f"Selected: {len(points)}/4 | Width: {width_m}m Length: {length_m}m",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or key == ord("q"):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("用户取消了交互式标定")
            elif key == ord("r"):
                points.clear()
                print("[*] 已重置标定点，请重新选取 4 个角点。")
            elif (key == 13 or key == 32) and len(points) == 4:
                break

        cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError("标定未完成，必须准确选取 4 个地面角点")

    calibrator = RoadCalibrator(
        image_points=points,
        width_m=width_m,
        length_m=length_m,
        name=src.stem,
    )
    saved_yaml = calibrator.to_yaml(output_path)
    print(f"[OK] 空间透视单应性标定成功完成！参数已持久化保存至: {saved_yaml}")
    return calibrator


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="校园道路 4 点空间单应性标定工具")
    parser.add_argument("--source", type=str, required=True, help="校园道路空镜照片或视频路径")
    parser.add_argument("--width", type=float, default=6.0, help="道路实际物理宽度 (米)，默认 6.0m")
    parser.add_argument("--length", type=float, default=15.0, help="标定路段纵向实际物理长度 (米)，默认 15.0m")
    parser.add_argument("--output", type=str, default="configs/calibration.yaml", help="YAML 输出配置路径")
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="非交互模式直接传入 4 点像素坐标: u1,v1,u2,v2,u3,v3,u4,v4",
    )
    args = parser.parse_args(argv)

    if not (math.isfinite(args.width) and args.width > 0):
        parser.error("--width 必须为有限正数")
    if not (math.isfinite(args.length) and args.length > 0):
        parser.error("--length 必须为有限正数")

    pts = None
    if args.points is not None:
        try:
            raw_nums = [float(x.strip()) for x in args.points.split(",")]
        except ValueError:
            parser.error("--points 格式错误，必须提供 8 个有限数字: u1,v1,u2,v2,u3,v3,u4,v4")
        if len(raw_nums) != 8 or not all(math.isfinite(x) for x in raw_nums):
            parser.error("--points 必须提供 8 个有限数字构成 4 对坐标 (u1,v1,u2,v2,u3,v3,u4,v4)")
        pts = [(raw_nums[i], raw_nums[i + 1]) for i in range(0, 8, 2)]

    run_interactive_calibration(
        source_path=args.source,
        width_m=args.width,
        length_m=args.length,
        output_path=args.output,
        predefined_points=pts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
