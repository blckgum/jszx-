"""命令行：python -m detection detect / train / val。"""

# python -m detection 会执行本文件，三条子命令对应不同任务：
# detect：用已有模型预测类别和位置，不需要人工标签。
# train：用图片及人工标签更新模型参数，并在验证集上评价训练进展。
# val：用人工标签核对模型预测，计算指标，不更新模型参数；也支持选择 test 集合。
# 三者共用 YOLOv8，但输入要求、输出和用途不同。

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import DEFAULT_WEIGHTS, detect, load_model, new_run, prepare_runtime, write_json


def positive_int(value):
    # 在启动模型前检查轮数、批大小和图像尺寸，避免耗时操作之后才发现参数无效。
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def probability(value):
    # 命令行使用 0～1 的小数，例如 --conf 0.25；不能把 25 当成 25%。
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("必须大于 0 且不超过 1")
    return number


def run_training_or_validation(args):
    import yaml

    from .dataset import audit_dataset

    # train 用于学习参数，val 用于选模型和调参，test 留给最终评估。
    # 训练时同时检查 train/val；独立评估时只要求所选集合可用。
    splits = ("train", "val") if args.command == "train" else (args.split,)
    # 先审计数据再加载模型，文件缺失或标注有误时可尽早结束。
    data, audit = audit_dataset(args.data, splits)
    model = load_model(args.weights)
    if args.command == "val":
        # 不允许拿 COCO 的类别顺序直接评估重排后的 6 类，避免错误分数。
        # 指标比较的是类别 ID：即便两份数据都有 bus，ID 不一致也会把正确预测判错。
        # 训练可以根据新数据调整检测头；评估则必须使用类别语义一致的权重。
        model_names = {int(k): v for k, v in model.names.items()}
        if model_names != data["names"]:
            raise ValueError("权重 names 与数据集 names 不一致；请使用该数据集训练产生的 best.pt 评估。")
    output = new_run(args.output, args.command)
    # 将解析后的绝对路径另存为本次实验配置，保留用户原 YAML 不动，便于核对实验输入。
    config = output / "dataset.resolved.yaml"
    config.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(output / "dataset_audit.json", audit)
    # project/name 合起来就是本次输出目录。exist_ok=True 允许框架写入已保存审计文件的目录；
    # new_run 已保证它是本次新建的目录，不会因此覆盖用户已有实验。
    # workers=0 使用主进程加载数据，减少不同操作系统下的多进程配置差异。
    # plots=False 不额外绘制训练图；数值指标仍会保存到 metrics.json。
    common = dict(data=str(config), imgsz=args.imgsz, device=args.device,
                  project=str(output.parent), name=output.name, exist_ok=True,
                  workers=0, plots=False)
    if args.command == "train":
        # 这里的 model 是 Ultralytics YOLO 对象，train() 会执行完整训练循环；
        # 它不同于原生 torch.nn.Module.train() 仅切换训练模式的行为。
        # epochs：遍历训练集多少轮；batch：每批多少张图片，不是线程数。
        # seed/deterministic 减少随机差异，但不能保证跨硬件、跨版本逐位复现。
        # amp=False 关闭自动混合精度，使用较直接的数值计算路径进行当前基线实验。
        metrics = model.train(**common, epochs=args.epochs, batch=args.batch,
                              seed=args.seed, deterministic=True, amp=False)
    else:
        # 即使 --split test，仍调用 val() 这个评估接口，由 split 决定实际评估的数据。
        # 测试集不参与模型选择，避免反复根据测试成绩调参后产生乐观偏差。
        metrics = model.val(**common, split=args.split, batch=args.batch)
    # results_dict 包含整体指标；转成 Python float 后便于保存为普通 JSON。
    result = {name: float(value) for name, value in metrics.results_dict.items()}
    per_class = []
    # 有些集合没有 bus 等类别的标注，指标数组只包含实际有标注的类别。
    # index 是指标数组位置，class_id 才是模型类别 ID，两者不能直接视为相同数字。
    for index, class_id in enumerate(metrics.box.ap_class_index):
        # Precision：预测为目标的结果中有多少正确；Recall：真实目标中有多少被找出。
        # AP 综合不同置信度阈值下的精确率和召回率；mAP 再对类别取平均。
        # AP50 在 IoU=0.5 的匹配条件下评估，AP50:95 再平均 0.5～0.95 的多个 IoU 条件。
        # 下面各行是单类的指标；JSON 字段沿用 mAP50/mAP50_95 命名，整体 mAP 在 metrics 中。
        per_class.append({
            "class_id": int(class_id), "class_name": data["names"][int(class_id)],
            "precision": float(metrics.box.p[index]), "recall": float(metrics.box.r[index]),
            "mAP50": float(metrics.box.ap50[index]), "mAP50_95": float(metrics.box.ap[index]),
        })
    # 区分训练起点 initial_weights 与实际评估的 evaluated_weights：
    # 训练后指标来自该次训练选出的 best.pt，不能误认为是原始预训练模型的成绩。
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
        # 先定义共用参数，再定义各任务的参数；例如检测不需要 --data 标注配置。
        sub = commands.add_parser(command)
        sub.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLOv8 检测权重路径")
        sub.add_argument("--output", help="新输出目录；默认在 runs/yolo/ 下创建独立目录")
        # imgsz 控制预处理尺寸及计算量，不会改变导出检测框使用的原图坐标系。
        sub.add_argument("--imgsz", type=positive_int, default=640)
        sub.add_argument("--device", default=None, help="cpu、0 等；默认由 PyTorch 自动选择")
        if command == "detect":
            sub.add_argument("--source", required=True, help="本地图片、图片目录或视频")
            sub.add_argument("--classes", nargs="+", help="按名称筛选，如 person car bus truck")
            # conf 筛分数，iou 控制 NMS 对重复框的抑制；这两个阈值解决不同问题。
            # 后续接 ByteTrack 时应联合调整 conf 与跟踪阈值，保留用于关联的低分候选。
            sub.add_argument("--conf", type=probability, default=0.25)
            sub.add_argument("--iou", type=probability, default=0.7)
            sub.add_argument("--max-frames", type=positive_int, help="仅处理前 N 帧/张图片")
            # fps 描述视频时间轴，不是要求模型达到某个处理速度，也不会加速模型。
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
        # 返回非零退出码，方便终端脚本判断任务确实失败，而不是仅打印错误后报告成功。
        parser.exit(1, f"执行失败：{exc}\n")
    print(f"结果已保存：{output}")


if __name__ == "__main__":
    main()
