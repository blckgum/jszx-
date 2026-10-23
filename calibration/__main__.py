"""python -m calibration：离线标定、独立验证和轨迹坐标映射，不开启桌面窗口。"""

import argparse

import cv2

from .demo import run_demo
from .io import import_legacy, load_mapping, new_output, resolved_document, write_json
from .pipeline import map_tracks
from .visualization import render_grid


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定机位道路标定与像素→米制平面映射")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="生成已知投影的合成样例并验证，不需要数据集")
    demo.add_argument("--output")
    solve = commands.add_parser("solve", help="从参考点配置求解并保存独立验证报告")
    solve.add_argument("--config", required=True)
    solve.add_argument("--image", help="可选原图，仅用于网格叠加预览")
    solve.add_argument("--output")
    validate = commands.add_parser("validate", help="重新计算保存产物的拟合与检查点报告")
    validate.add_argument("--calibration", required=True)
    validate.add_argument("--output")
    tracks = commands.add_parser("map-tracks", help="映射已有跟踪/运动 JSONL，不重跑 YOLO")
    tracks.add_argument("--tracks", required=True)
    tracks.add_argument("--calibration", required=True)
    tracks.add_argument("--camera-id", required=True)
    tracks.add_argument("--position-source", choices=("raw", "smoothed"), default="raw")
    tracks.add_argument("--allow-synthetic", action="store_true")
    tracks.add_argument("--output")
    legacy = commands.add_parser("import-legacy", help="适配旧四点配置；原文件保留，不直接认定验证通过")
    legacy.add_argument("--config", required=True)
    legacy.add_argument("--camera-id", required=True)
    legacy.add_argument("--image-size", nargs=2, type=int, required=True, metavar=("WIDTH", "HEIGHT"))
    legacy.add_argument("--allow-legacy", action="store_true", help="明确允许转换缺少 synthetic 字段的旧文件")
    legacy.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            output = run_demo(args.output)
        elif args.command == "map-tracks":
            output = map_tracks(args.tracks, load_mapping(args.calibration), camera_id=args.camera_id,
                                output=args.output, allow_synthetic=args.allow_synthetic,
                                position_source=args.position_source)
        else:
            if args.command == "import-legacy":
                mapper = import_legacy(args.config, camera_id=args.camera_id, image_size=args.image_size,
                                       allow_legacy=args.allow_legacy)
            else:
                mapper = load_mapping(args.config if args.command == "solve" else args.calibration)
            image = None
            if args.command == "solve" and args.image:
                image = cv2.imread(args.image)
                if image is None:
                    raise ValueError("无法读取标定预览底图")
            preview = render_grid(mapper, image)
            output = new_output(args.output)
            write_json(output / "mapping.json", resolved_document(mapper))
            write_json(output / "validation.json", mapper.report)
            if not cv2.imwrite(str(output / "ground_grid.png"), preview):
                raise RuntimeError("无法保存地面网格预览")
            print(f"数值检查通过：{mapper.report['independent_check_passed']}；实测标定有效：{mapper.calibration_valid}")
        print(f"结果已保存：{output}")
        if args.command == "map-tracks":
            import json
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            if summary["status"] == "interrupted":
                raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        notes = "".join(f"{note}\n" for note in getattr(exc, "__notes__", []))
        parser.exit(1, f"标定/映射失败：{exc}\n{notes}")


if __name__ == "__main__":
    main()
