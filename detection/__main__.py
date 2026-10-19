"""命令行：python -m detection detect / train / val。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import DEFAULT_WEIGHTS, detect, load_model, new_run, prepare_runtime, write_json


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def probability(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("必须大于 0 且不超过 1")
    return number


def run_training_or_validation(args):
    import yaml

    from .dataset import audit_dataset

    splits = ("train", "val") if args.command == "train" else (args.split,)
    data, audit = audit_dataset(args.data, splits)
    model = load_model(args.weights)
    if args.command == "val":
        # 不允许拿 COCO 的类别顺序直接评估重排后的 6 类，避免错误分数。
        model_names = {int(k): v for k, v in model.names.items()}
        if model_names != data["names"]:
            raise ValueError("权重 names 与数据集 names 不一致；请使用该数据集训练产生的 best.pt 评估。")
    output = new_run(args.output, args.command)
    config = output / "dataset.resolved.yaml"
    config.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(output / "dataset_audit.json", audit)
    common = dict(data=str(config), imgsz=args.imgsz, device=args.device,
                  project=str(output.parent), name=output.name, exist_ok=True,
                  workers=0, plots=False)
    if args.command == "train":
        metrics = model.train(**common, epochs=args.epochs, batch=args.batch,
                              seed=args.seed, deterministic=True, amp=False)
    else:
        metrics = model.val(**common, split=args.split, batch=args.batch)
    result = {name: float(value) for name, value in metrics.results_dict.items()}
    per_class = []
    for index, class_id in enumerate(metrics.box.ap_class_index):
        per_class.append({
            "class_id": int(class_id), "class_name": data["names"][int(class_id)],
            "precision": float(metrics.box.p[index]), "recall": float(metrics.box.r[index]),
            "mAP50": float(metrics.box.ap50[index]), "mAP50_95": float(metrics.box.ap[index]),
        })
    write_json(output / "metrics.json", {"metrics": result, "dataset": audit,
                                         "initial_weights": str(Path(args.weights).expanduser().resolve()),
                                         "evaluated_weights": str(output / "weights" / "best.pt")
                                         if args.command == "train" else str(Path(args.weights).expanduser().resolve()),
                                         "per_class": per_class,
                                         "note": "per_class 仅包含评估集有标注的类别；训练后指标来自验证集。"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="YOLOv8 行人车辆检测、训练与评估")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("detect", "train", "val"):
        sub = commands.add_parser(command)
        sub.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLOv8 检测权重路径")
        sub.add_argument("--output", help="新输出目录；默认在 runs/yolo/ 下创建独立目录")
        sub.add_argument("--imgsz", type=positive_int, default=640)
        sub.add_argument("--device", default=None, help="cpu、0 等；默认由 PyTorch 自动选择")
        if command == "detect":
            sub.add_argument("--source", required=True, help="本地图片、图片目录或视频")
            sub.add_argument("--classes", nargs="+", help="按名称筛选，如 person car bus truck")
            sub.add_argument("--conf", type=probability, default=0.25)
            sub.add_argument("--iou", type=probability, default=0.7)
            sub.add_argument("--max-frames", type=positive_int, help="仅处理前 N 帧/张图片")
            sub.add_argument("--fps", type=float, help="覆盖视频 FPS 与时间戳；缺少元数据时使用")
        else:
            sub.add_argument("--data", required=True, help="本地 YOLO 数据集 YAML")
            sub.add_argument("--batch", type=positive_int, default=8)
            if command == "train":
                sub.add_argument("--epochs", type=positive_int, default=50)
                sub.add_argument("--seed", type=int, default=42)
            else:
                sub.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args(argv)
    if args.command == "detect" and args.fps is not None:
        import math
        if not math.isfinite(args.fps) or args.fps <= 0:
            parser.error("--fps 必须是有限正数")
    try:
        prepare_runtime()
        output = detect(args) if args.command == "detect" else run_training_or_validation(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"执行失败：{exc}\n")
    print(f"结果已保存：{output}")


if __name__ == "__main__":
    main()
