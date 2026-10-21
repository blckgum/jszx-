"""检查本地 YOLO 标注，防止缺标签或类别映射错误被当成正常训练。"""

# 目标检测属于监督学习：图片告诉模型“看什么”，人工标注提供目标类别及正确位置。
# 本文件检查输入数据的约定，真正的图像读取、数据增强和张量批处理由 Ultralytics 完成。
# 标签格式检查通过不代表标注内容一定正确，例如漏标、框错目标仍需要人工抽查。
# 原理笔记：docs/modules/01_目标检测_YOLOv8与PyTorch_解释文档.md，第 8、13 节。

from __future__ import annotations

import math
from pathlib import Path

import yaml

from .core import IMAGE_SUFFIXES, TARGET_CLASSES


def audit_dataset(config: str | Path, required_splits: tuple[str, ...]):
    # 返回两部分：可交给 Ultralytics 的统一配置，以及每个划分的图片数和目标数。
    # required_splits 指定本次必须检查的集合：训练时检查 train/val，评估时检查所选集合。
    config = Path(config).expanduser().resolve()
    try:
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"数据配置 YAML 格式错误：{config}\n{exc}") from None
    if not isinstance(data, dict):
        raise ValueError("数据配置必须是 YAML 映射。")
    names = data.get("names")
    # YAML 支持 names: [person, car] 或 names: {0: person, 1: car}，统一为 ID -> 名称。
    # 这些 ID 决定检测头输出通道的语义；仅改名字而不调整标签，会让模型学到错误对应关系。
    if isinstance(names, list):
        names = dict(enumerate(names))
    if (not isinstance(names, dict) or any(type(k) is not int for k in names)
            or sorted(names) != list(range(len(names)))):
        raise ValueError("names 必须使用从 0 开始连续的类别 ID。")
    if (not names or any(not isinstance(n, str) for n in names.values())
            or len(set(names.values())) != len(names)
            or any(n not in TARGET_CLASSES for n in names.values())):
        raise ValueError(f"names 必须是互不重复的道路类别：{', '.join(TARGET_CLASSES)}")
    # 相对路径以 YAML 所在目录为基准，不依赖运行命令时终端的当前目录。
    root = (config.parent / str(data.get("path") or ".")).resolve()
    # Ultralytics 即使仅评估 test 也要求 train/val 两个键存在。
    # 不需要的集合可以为 None；本次实际使用的集合仍会在下面检查，不能靠占位键通过审计。
    normalized = {"path": str(root), "names": names, "train": None, "val": None}
    for split in ("train", "val", "test"):
        if isinstance(data.get(split), str):
            normalized[split] = str((root / data[split]).resolve())
    report, split_files = {}, {}
    for split in required_splits:
        relative = data.get(split)
        if not isinstance(relative, str):
            raise ValueError(f"{split} 需要指向本地图片目录，本入口不接受下载脚本或列表文件。")
        folder = (root / relative).resolve()
        if not folder.is_dir() or "images" not in folder.parts:
            raise ValueError(f"{split} 图片目录不存在或不符合 images/ 目录结构：{folder}")
        # 允许 images/train 下继续按场景建子目录，标签也必须保持相同的相对目录结构。
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"{split} 没有图片：{folder}")
        split_files[split] = {p.resolve() for p in files}
        objects = 0
        for image in files:
            # YOLO 的默认配对规则：images/train/a.jpg -> labels/train/a.txt。
            # 替换最后一个完整的 images 路径组件，避免误改目录名中包含 images 的其他文字。
            parts = list(image.parts)
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
            if not label.is_file():
                # 有些框架把缺少标签视为“无目标”，容易把漏交标签的数据当作负样本。
                # 此入口要求无目标图片也有空 txt，从而区分明确的负样本与文件遗漏。
                raise ValueError(f"缺少标签：{label}；无目标图片也请提供空 txt。")
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    values = [float(v) for v in line.split()]
                    # 当前仅接收矩形检测标签，五列为：类别、中心 x、中心 y、宽、高。
                    # 分割多边形或关键点标注的列数不同，不能直接用于这里的检测训练。
                    if len(values) != 5 or not all(math.isfinite(v) for v in values):
                        raise ValueError
                    cls, x, y, width, height = values
                    if not cls.is_integer() or int(cls) not in names:
                        raise ValueError
                    # x/宽按图像宽度归一化，y/高按图像高度归一化，都不是米制距离。
                    # 例如 640x480 图中的标签 0 0.5 0.5 0.25 0.5：
                    # 类别为 0，中心为 (320,240)，框宽高为 (160,240)，xyxy 为 (240,120,400,360)。
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        raise ValueError
                except ValueError:
                    raise ValueError(f"标签格式错误：{label}:{line_no}；应为 class_id x_center y_center width height，坐标归一化。") from None
                objects += 1
        if objects == 0:
            # 单张负样本合法，但整个训练/评估集合都没有目标标注，无法形成有效检测验证。
            raise ValueError(f"{split} 没有任何目标标注，无法验证行人车辆检测。")
        normalized[split] = str(folder)
        report[split] = {"images": len(files), "objects": objects}
    for index, first in enumerate(required_splits):
        for second in required_splits[index + 1:]:
            if split_files[first] & split_files[second]:
                # 同一图片不能同时用于训练与验证，否则分数会高估模型对未见数据的能力。
                # 这里只查相同解析路径；图片副本和相邻视频帧仍需在数据划分时人工避免泄漏。
                raise ValueError(f"{first} 与 {second} 包含相同图片路径，请重新划分数据。")
    return normalized, report
