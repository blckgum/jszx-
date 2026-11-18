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
import json
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .calibration import RoadCalibrator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _read_calibration_image(src: Path) -> np.ndarray:
    """读取标定图片，兼容 Windows 含中文路径并校验解码结果。"""
    # Windows 下 cv2.imread 基于 C 接口窄字符路径，目录或文件名含中文时常返回 None；
    # 改由 np.fromfile 读入字节流再经 cv2.imdecode 解码可稳定兼容中文路径。
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
    save_output: bool = True,
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
    # 必须在 Path() 解析前拦截空白路径（含 Path("   ")），防止退化为当前目录 '.' 或误写空白文件。
    if (isinstance(source_path, str) and not source_path.strip()) or (
        isinstance(source_path, os.PathLike)
        and isinstance(os.fspath(source_path), str)
        and not os.fspath(source_path).strip()
    ):
        raise ValueError("标定输入源路径不能为空")
    if save_output and (
        (isinstance(output_path, str) and not output_path.strip())
        or (
            isinstance(output_path, os.PathLike)
            and isinstance(os.fspath(output_path), str)
            and not os.fspath(output_path).strip()
        )
    ):
        raise ValueError("标定输出路径不能为空")

    src = Path(source_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"未找到标定底图或视频文件: {src}")

    # 支持直接从图像或视频中抓取第 1 帧；视频无论读帧成功与否都必须释放句柄。
    if src.suffix.lower() in IMAGE_SUFFIXES:
        image = _read_calibration_image(src)
    else:
        cap = cv2.VideoCapture(str(src))
        try:
            ok, image = cap.read()
        finally:
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

        print("=" * 60)
        print("[*] 交互式空间标定窗口已打开:")
        print("    请用鼠标左键按顺时针依次点击道路地面的 4 个角点:")
        print("    [1] 左上角 -> [2] 右上角 -> [3] 右下角 -> [4] 左下角")
        print("    按键盘 'R' 重置选点，点满 4 点后按 'Enter' 或 'Space' 保存退出，'Q' 取消。")
        print("=" * 60)

        # 将窗口创建、鼠标回调绑定与主刷新循环整体放入 try...finally：
        # 无论用户按 Q/Esc、点击窗口右上角关闭按钮还是途中出错，都确保销毁 GUI 窗口不挂起进程。
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, on_mouse)
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
                # 用户点击窗口右上角关闭按钮不会向 waitKey 发送按键，需同时轮询 WND_PROP_VISIBLE < 1。
                if (
                    key in (27, ord("q"), ord("Q"))
                    or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
                ):
                    raise KeyboardInterrupt("用户取消了交互式标定")
                elif key in (ord("r"), ord("R")):
                    points.clear()
                    print("[*] 已重置标定点，请重新选取 4 个角点。")
                elif (key == 13 or key == 32) and len(points) == 4:
                    break
        finally:
            cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError("标定未完成，必须准确选取 4 个地面角点")

    calibrator = RoadCalibrator(
        image_points=points,
        width_m=width_m,
        length_m=length_m,
        name=src.stem,
    )
    if save_output:
        saved_yaml = calibrator.to_yaml(output_path)
        print(f"[OK] 空间透视单应性标定成功完成！参数已持久化保存至: {saved_yaml}")
    return calibrator


def _parse_coord_pairs(raw: str, flag_name: str, expected_pairs: Optional[int] = None) -> List[Tuple[float, float]]:
    # 将命令行逗号分隔坐标解析为成对有限浮点数，确保既满足偶数配对又拒绝 NaN/Inf。
    try:
        nums = [float(x.strip()) for x in raw.split(",")]
    except ValueError:
        raise ValueError(f"{flag_name} 格式错误，必须由逗号分隔的有限数字组成") from None
    if len(nums) < 2 or len(nums) % 2 != 0 or not all(math.isfinite(x) for x in nums):
        raise ValueError(f"{flag_name} 必须包含偶数个有限数字以构成二维坐标对")
    if expected_pairs is not None and len(nums) != expected_pairs * 2:
        raise ValueError(f"{flag_name} 必须提供 {expected_pairs * 2} 个有限数字构成 {expected_pairs} 对坐标")
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


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
    parser.add_argument(
        "--check-image-points",
        type=str,
        default=None,
        help="可选独立检查点的像素坐标序列: u1,v1,u2,v2,...",
    )
    parser.add_argument(
        "--check-ground-points",
        type=str,
        default=None,
        help="可选独立检查点对应的地面米制坐标序列: x1,y1,x2,y2,...",
    )
    parser.add_argument(
        "--validation-output",
        type=str,
        default=None,
        help="可选将独立检查点误差报告保存为 JSON 文件",
    )
    args = parser.parse_args(argv)

    # 在 Path() 解析与 GUI/写盘前拦截空白 CLI 路径参数，防止退化为当前目录 '.'。
    if not args.source.strip():
        parser.error("--source 不能为空")
    if not args.output.strip():
        parser.error("--output 不能为空")
    if args.validation_output is not None and not args.validation_output.strip():
        parser.error("--validation-output 不能为空")
    if not (math.isfinite(args.width) and args.width > 0):
        parser.error("--width 必须为有限正数")
    if not (math.isfinite(args.length) and args.length > 0):
        parser.error("--length 必须为有限正数")
    if (args.check_image_points is None) != (args.check_ground_points is None):
        parser.error("--check-image-points 与 --check-ground-points 必须成对提供")
    if args.validation_output is not None and args.check_image_points is None:
        parser.error("--validation-output 必须与 --check-image-points 和 --check-ground-points 一起使用")
    if args.validation_output is not None:
        # 防止误把验证报告路径设为标定 YAML 同路径，导致后续 JSON 写出覆写掉标定配置。
        if Path(args.validation_output).expanduser().resolve() == Path(args.output).expanduser().resolve():
            parser.error("--validation-output 不能与 --output 指向同一文件")

    pts = None
    if args.points is not None:
        try:
            pts = _parse_coord_pairs(args.points, "--points", expected_pairs=4)
        except ValueError as exc:
            parser.error(str(exc))

    check_img_pts = check_gd_pts = None
    if args.check_image_points is not None:
        try:
            check_img_pts = _parse_coord_pairs(args.check_image_points, "--check-image-points")
            check_gd_pts = _parse_coord_pairs(args.check_ground_points, "--check-ground-points")
        except ValueError as exc:
            parser.error(str(exc))
        if len(check_img_pts) != len(check_gd_pts):
            parser.error("--check-image-points 与 --check-ground-points 点数必须一致")

    # 提供了独立检查点时先不写盘：待误差检查全部通过后再持久化 YAML，
    # 防止非法/重合检查点报错时提前覆写已有标定文件。
    calibrator = run_interactive_calibration(
        source_path=args.source,
        width_m=args.width,
        length_m=args.length,
        output_path=args.output,
        predefined_points=pts,
        save_output=(check_img_pts is None),
    )
    if check_img_pts is not None:
        report = calibrator.evaluate_check_points(check_img_pts, check_gd_pts)
        print(
            f"[OK] 独立检查点验证 ({report['count']} 点): "
            f"RMSE = {report['rmse_m']:.3f} m | Max = {report['max_error_m']:.3f} m"
        )
        saved_yaml = calibrator.to_yaml(args.output)
        print(f"[OK] 空间透视单应性标定成功完成！参数已持久化保存至: {saved_yaml}")
        if args.validation_output:
            val_path = Path(args.validation_output).expanduser().resolve()
            val_path.parent.mkdir(parents=True, exist_ok=True)
            val_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
