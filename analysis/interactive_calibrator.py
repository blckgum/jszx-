# -*- coding: utf-8 -*-
"""
交互式地面与相机内参标定工具 (Interactive Road & Camera Calibrator CLI)
=============================================================================
支持：
1. 4 点地面角点交互/非交互单应性标定；
2. 多点地面对应点（--ground-points）与 RANSAC 抗外点标定；
3. 路侧停车区像素多边形（可重复 --parking-zone）映射持久化；
4. 棋盘格相机内参畸变标定（--chessboard-dir / --intrinsics-yaml）及自动去畸变注入。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from detection.core import IMAGE_SUFFIXES, is_blank_path
from .calibration import (
    RoadCalibrator,
    _validate_camera_matrix,
    _validate_dist_coeffs,
    _validate_finite_float,
)


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


def _parse_board_size(raw: str) -> Tuple[int, int]:
    """解析棋盘格内角点尺寸字符串（如 '9,6' -> (9, 6)）。"""
    parts = [tok.strip() for tok in raw.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("--board-size 格式错误，必须为 'cols,rows' 两个正整数（例如 9,6）")
    try:
        cols = int(parts[0])
        rows = int(parts[1])
    except ValueError:
        raise ValueError("--board-size 必须由整数构成（例如 9,6）") from None
    if cols < 3 or rows < 3:
        raise ValueError("--board-size 棋盘格内角点行列数均必须 >= 3")
    return cols, rows


def calibrate_camera_chessboard(
    image_paths: Iterable[str | Path],
    board_size: Tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
) -> dict:
    """使用多视角棋盘格图片标定相机内参矩阵与镜头畸变系数。"""
    if isinstance(image_paths, (str, bytes, os.PathLike)):
        raise ValueError("image_paths 必须是图片路径序列，而非单个路径")
    raw_paths = list(image_paths)
    if not raw_paths:
        raise ValueError("未提供棋盘格标定图片")

    if (
        not isinstance(board_size, (tuple, list))
        or len(board_size) != 2
        or isinstance(board_size[0], (bool, np.bool_))
        or isinstance(board_size[1], (bool, np.bool_))
        or not isinstance(board_size[0], (int, np.integer))
        or not isinstance(board_size[1], (int, np.integer))
    ):
        raise ValueError("board_size 必须是长度为 2 的整数元组 (cols, rows)")
    cols, rows = int(board_size[0]), int(board_size[1])
    if cols < 3 or rows < 3:
        raise ValueError("board_size 棋盘格内角点行列数均必须 >= 3")

    sq_size = _validate_finite_float(square_size_m, "square_size_m ")
    if sq_size <= 0.0:
        raise ValueError("square_size_m 必须为大于 0 的有限正数")

    resolved_paths: List[Path] = []
    for idx, raw_p in enumerate(raw_paths):
        if not isinstance(raw_p, (str, os.PathLike)) or is_blank_path(raw_p):
            raise ValueError(f"第 {idx + 1} 个棋盘格图片路径不能为空")
        resolved = Path(raw_p).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"未找到棋盘格图片文件: {resolved}")
        resolved_paths.append(resolved)

    # 生成 Z=0 棋盘格平面角点三维物理坐标（单位：米）
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(sq_size)

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    expected_size: Optional[Tuple[int, int]] = None
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    for img_path in resolved_paths:
        image = _read_calibration_image(img_path)
        h, w = image.shape[:2]
        cur_size = (int(w), int(h))
        if expected_size is None:
            expected_size = cur_size
        elif cur_size != expected_size:
            raise ValueError(
                f"棋盘格图片分辨率不一致: {img_path} 为 {cur_size[0]}x{cur_size[1]}，"
                f"与首张 {expected_size[0]}x{expected_size[1]} 不符"
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags=find_flags)
        if not ok or corners is None:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
        obj_points.append(objp)
        img_points.append(refined)

    total_images = len(resolved_paths)
    used_images = len(obj_points)
    # 少于 3 帧不同视角难以稳定约束焦距与径向畸变，必须明确报错阻断
    if used_images < 3 or expected_size is None:
        raise ValueError(
            f"有效棋盘格角点检测图像不足: 仅 {used_images}/{total_images} 帧成功检测到 ({cols}, {rows}) 角点，至少需要 3 帧"
        )

    try:
        rms, cam_mat, dist_coeffs, _, _ = cv2.calibrateCamera(
            obj_points, img_points, expected_size, None, None
        )
    except cv2.error as exc:
        raise ValueError(f"棋盘格相机内参标定失败: {exc}") from None

    rms_val = _validate_finite_float(rms, "reprojection_rms_px ")
    if rms_val < 0.0:
        raise ValueError("重投影均方根误差不能为负数")
    cam_list, _, _ = _validate_camera_matrix(cam_mat)
    dist_list, _ = _validate_dist_coeffs(np.asarray(dist_coeffs, dtype=np.float64).ravel())

    return {
        "camera_matrix": cam_list,
        "dist_coeffs": dist_list,
        "reprojection_rms_px": float(rms_val),
        "used_images": int(used_images),
        "total_images": int(total_images),
        "image_size": [expected_size[0], expected_size[1]],
    }


def load_intrinsics_yaml(
    yaml_path: str | Path,
) -> Tuple[List[List[float]], Optional[List[float]]]:
    """从 YAML 文件加载并校验相机内参矩阵 camera_matrix 与可选畸变系数 dist_coeffs。"""
    if not isinstance(yaml_path, (str, os.PathLike)) or is_blank_path(yaml_path):
        raise ValueError("内参配置文件路径不能为空")
    path = Path(yaml_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"未找到相机内参配置文件: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"内参文件 {path} YAML 格式错误: {exc}") from None

    if not isinstance(data, dict):
        raise ValueError(f"内参文件 {path} 内容必须是键值映射")
    if "camera_matrix" not in data:
        raise ValueError(f"内参文件 {path} 缺少必需字段: camera_matrix")

    cam_list, _, _ = _validate_camera_matrix(data["camera_matrix"])
    dist_list = None
    if data.get("dist_coeffs") is not None:
        dist_list, _ = _validate_dist_coeffs(data["dist_coeffs"])
    return cam_list, dist_list


def run_interactive_calibration(
    source_path: str | Path,
    width_m: float = 6.0,
    length_m: float = 15.0,
    output_path: str | Path = "configs/calibration.yaml",
    predefined_points: Optional[List[Tuple[float, float]]] = None,
    save_output: bool = True,
    *,
    ground_points: Optional[List[Tuple[float, float]]] = None,
    use_ransac: bool = True,
    ransac_reproj_threshold_m: float = 0.5,
    parking_zones: Optional[List[List[Tuple[float, float]]]] = None,
    camera_matrix: Optional[List[List[float]]] = None,
    dist_coeffs: Optional[List[float]] = None,
) -> RoadCalibrator:
    """执行交互式或参数化地面空间标定流程（兼容 4 点矩形与 N>=4 对应点 RANSAC）。"""
    if (
        type(width_m) not in (int, float)
        or type(length_m) not in (int, float)
        or not (math.isfinite(width_m) and math.isfinite(length_m))
        or width_m <= 0
        or length_m <= 0
    ):
        raise ValueError("道路的实际物理宽度和长度必须为大于 0 的有限正数（单位：米）")
    # 必须在 Path() 解析前拦截空白路径（含 Path("   ")），防止退化为当前目录 '.' 或误写空白文件。
    if not isinstance(source_path, (str, os.PathLike)) or is_blank_path(source_path):
        raise ValueError("标定输入源路径不能为空")
    if save_output and (
        not isinstance(output_path, (str, os.PathLike)) or is_blank_path(output_path)
    ):
        raise ValueError("标定输出路径不能为空")
    if type(use_ransac) is not bool:
        raise ValueError("use_ransac 必须是布尔值")

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

    if ground_points is not None:
        if predefined_points is None:
            raise ValueError("提供 ground_points 时必须同时提供 predefined_points")
        if len(predefined_points) < 4:
            raise ValueError(f"多点对应标定至少需要 4 个角点，当前提供了 {len(predefined_points)} 个")
        if len(predefined_points) != len(ground_points):
            raise ValueError("predefined_points 与 ground_points 数量必须一致")
        points = list(predefined_points)
        calibrator = RoadCalibrator.from_correspondences(
            image_points=points,
            ground_points=list(ground_points),
            name=src.stem,
            use_ransac=use_ransac,
            ransac_reproj_threshold_m=ransac_reproj_threshold_m,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
    else:
        if predefined_points is not None:
            if len(predefined_points) != 4:
                raise ValueError(f"地面标定必须严格提供 4 个角点，当前提供了 {len(predefined_points)} 个")
            points = list(predefined_points)
            print(f"[*] 使用预设参数标定 4 点: {points}")
        else:
            # 交互式窗口标定
            points = []
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

                    if len(points) >= 2:
                        pts_arr = np.array(points, dtype=np.int32)
                        cv2.polylines(display, [pts_arr], isClosed=(len(points) == 4), color=(0, 255, 0), thickness=2)

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
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )

    if parking_zones:
        for poly in parking_zones:
            calibrator.add_parking_zone_pixels(poly)

    if save_output:
        saved_yaml = calibrator.to_yaml(output_path)
        print(f"[OK] 空间透视单应性标定成功完成！参数已持久化保存至: {saved_yaml}")
    return calibrator


def _parse_coord_pairs(
    raw: str,
    flag_name: str,
    expected_pairs: Optional[int] = None,
    min_pairs: int = 1,
) -> List[Tuple[float, float]]:
    # 将命令行逗号分隔坐标解析为成对有限浮点数，确保既满足偶数配对又拒绝 NaN/Inf。
    try:
        nums = [float(x.strip()) for x in raw.split(",")]
    except ValueError:
        raise ValueError(f"{flag_name} 格式错误，必须由逗号分隔的有限数字组成") from None
    if len(nums) < 2 or len(nums) % 2 != 0 or not all(math.isfinite(x) for x in nums):
        raise ValueError(f"{flag_name} 必须包含偶数个有限数字以构成二维坐标对")
    pair_count = len(nums) // 2
    if pair_count < min_pairs:
        raise ValueError(f"{flag_name} 至少需要 {min_pairs} 对坐标，当前仅提供 {pair_count} 对")
    if expected_pairs is not None and pair_count != expected_pairs:
        raise ValueError(f"{flag_name} 必须提供 {expected_pairs * 2} 个有限数字构成 {expected_pairs} 对坐标")
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="校园道路空间单应性及相机内参标定工具")
    parser.add_argument("--source", type=str, default=None, help="校园道路空镜照片或视频路径")
    parser.add_argument("--width", type=float, default=6.0, help="道路实际物理宽度 (米)，默认 6.0m")
    parser.add_argument("--length", type=float, default=15.0, help="标定路段纵向实际物理长度 (米)，默认 15.0m")
    parser.add_argument("--output", type=str, default="configs/calibration.yaml", help="YAML 输出配置路径")
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="非交互模式像素坐标: u1,v1,u2,v2,... (默认 4 点；配合 --ground-points 支持 >=4 点)",
    )
    parser.add_argument(
        "--ground-points",
        type=str,
        default=None,
        help="可选与 --points 一一对应的地面米制坐标序列: x1,y1,x2,y2,... (>=4 对)",
    )
    parser.add_argument(
        "--no-ransac",
        action="store_true",
        help="多点对应标定时关闭 RANSAC，改用全量最小二乘拟合单应性",
    )
    parser.add_argument(
        "--parking-zone",
        action="append",
        default=None,
        help="可选路侧停车区像素多边形 (可重复传入): u1,v1,u2,v2,u3,v3,...",
    )
    parser.add_argument(
        "--chessboard-dir",
        type=str,
        default=None,
        help="可选棋盘格图像目录：执行相机内参及畸变标定；单独使用时直接将内参写入 --output",
    )
    parser.add_argument(
        "--board-size",
        type=str,
        default="9,6",
        help="棋盘格内部角点行列数 'cols,rows'，默认 '9,6'",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.025,
        help="棋盘格单格物理边长 (米)，默认 0.025",
    )
    parser.add_argument(
        "--intrinsics-yaml",
        type=str,
        default=None,
        help="可选已有相机内参 YAML 路径，将其 camera_matrix 与 dist_coeffs 注入地面标定",
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

    # 在 Path() 解析与 GUI/写盘前拦截所有空白 CLI 路径参数，防止退化为当前目录 '.'。
    if args.source is not None and not args.source.strip():
        parser.error("--source 不能为空")
    if not args.output.strip():
        parser.error("--output 不能为空")
    if args.validation_output is not None and not args.validation_output.strip():
        parser.error("--validation-output 不能为空")
    if args.chessboard_dir is not None and not args.chessboard_dir.strip():
        parser.error("--chessboard-dir 不能为空")
    if args.intrinsics_yaml is not None and not args.intrinsics_yaml.strip():
        parser.error("--intrinsics-yaml 不能为空")

    if args.source is None and args.chessboard_dir is None:
        parser.error("必须至少指定 --source 或 --chessboard-dir")
    if args.chessboard_dir is not None and args.intrinsics_yaml is not None:
        parser.error("--chessboard-dir 与 --intrinsics-yaml 不能同时使用")

    if not (math.isfinite(args.width) and args.width > 0):
        parser.error("--width 必须为有限正数")
    if not (math.isfinite(args.length) and args.length > 0):
        parser.error("--length 必须为有限正数")
    if not (math.isfinite(args.square_size) and args.square_size > 0):
        parser.error("--square-size 必须为有限正数")
    try:
        board_size = _parse_board_size(args.board_size)
    except ValueError as exc:
        parser.error(str(exc))

    if (args.check_image_points is None) != (args.check_ground_points is None):
        parser.error("--check-image-points 与 --check-ground-points 必须成对提供")
    if args.validation_output is not None and args.check_image_points is None:
        parser.error("--validation-output 必须与 --check-image-points 和 --check-ground-points 一起使用")
    if args.validation_output is not None:
        # 防止误把验证报告路径设为标定 YAML 同路径，导致后续 JSON 写出覆写掉标定配置。
        if Path(args.validation_output).expanduser().resolve() == Path(args.output).expanduser().resolve():
            parser.error("--validation-output 不能与 --output 指向同一文件")

    if args.source is None and any(
        val is not None
        for val in (
            args.points,
            args.ground_points,
            args.parking_zone,
            args.check_image_points,
            args.check_ground_points,
            args.validation_output,
        )
    ):
        parser.error("仅传 --chessboard-dir 进行内参标定时不能使用地面标点、停车区或检查点参数")

    if args.ground_points is not None and args.points is None:
        parser.error("--ground-points 必须与 --points 一起提供")

    pts = None
    ground_pts = None
    if args.ground_points is not None:
        try:
            pts = _parse_coord_pairs(args.points, "--points", min_pairs=4)
            ground_pts = _parse_coord_pairs(args.ground_points, "--ground-points", min_pairs=4)
        except ValueError as exc:
            parser.error(str(exc))
        if len(pts) != len(ground_pts):
            parser.error("--points 与 --ground-points 点数必须一致")
    elif args.points is not None:
        try:
            pts = _parse_coord_pairs(args.points, "--points", expected_pairs=4)
        except ValueError as exc:
            parser.error(str(exc))

    parking_zones: List[List[Tuple[float, float]]] = []
    if args.parking_zone:
        for raw_zone in args.parking_zone:
            try:
                parking_zones.append(_parse_coord_pairs(raw_zone, "--parking-zone", min_pairs=3))
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

    # 当同时传 --source 与 --chessboard-dir 时，先确认源路径存在，避免标定完棋盘格才发现底图缺失
    if args.source is not None:
        src_check = Path(args.source).expanduser().resolve()
        if not src_check.is_file():
            raise FileNotFoundError(f"未找到标定底图或视频文件: {src_check}")

    camera_matrix = None
    dist_coeffs = None
    intrinsics_result = None
    if args.chessboard_dir is not None:
        cb_dir = Path(args.chessboard_dir).expanduser().resolve()
        if not cb_dir.is_dir():
            raise FileNotFoundError(f"未找到棋盘格图像目录: {cb_dir}")
        cb_images = sorted(
            p for p in cb_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not cb_images:
            raise ValueError(f"棋盘格目录中没有支持的图片文件: {cb_dir}")
        intrinsics_result = calibrate_camera_chessboard(
            cb_images, board_size=board_size, square_size_m=args.square_size
        )
        camera_matrix = intrinsics_result["camera_matrix"]
        dist_coeffs = intrinsics_result["dist_coeffs"]
        print(
            f"[OK] 棋盘格相机内参标定完成 ({intrinsics_result['used_images']}/{intrinsics_result['total_images']} 帧): "
            f"RMS = {intrinsics_result['reprojection_rms_px']:.4f} px"
        )
    elif args.intrinsics_yaml is not None:
        camera_matrix, dist_coeffs = load_intrinsics_yaml(args.intrinsics_yaml)

    # 仅传 --chessboard-dir + --output（不传 --source）时，直接输出相机内参 YAML 并结束
    if args.source is None:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "board_size": [board_size[0], board_size[1]],
            "square_size_m": float(args.square_size),
            **(intrinsics_result or {}),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"[OK] 相机内参标定参数已保存至: {out_path}")
        return 0

    # 提供了独立检查点时先不写盘：待误差检查全部通过后再持久化 YAML，
    # 防止非法/重合检查点报错时提前覆写已有标定文件。
    calibrator = run_interactive_calibration(
        source_path=args.source,
        width_m=args.width,
        length_m=args.length,
        output_path=args.output,
        predefined_points=pts,
        save_output=(check_img_pts is None),
        ground_points=ground_pts,
        use_ransac=not args.no_ransac,
        parking_zones=parking_zones,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
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
