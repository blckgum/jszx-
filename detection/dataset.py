"""检查本地 YOLO 标注，防止缺标签或类别映射错误被当成正常训练。"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from .core import IMAGE_SUFFIXES, TARGET_CLASSES


def audit_dataset(config: str | Path, required_splits: tuple[str, ...]):
    config = Path(config).expanduser().resolve()
    try:
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"数据配置 YAML 格式错误：{config}\n{exc}") from None
    if not isinstance(data, dict):
        raise ValueError("数据配置必须是 YAML 映射。")
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if (not isinstance(names, dict) or any(type(k) is not int for k in names)
            or sorted(names) != list(range(len(names)))):
        raise ValueError("names 必须使用从 0 开始连续的类别 ID。")
    if (not names or any(not isinstance(n, str) for n in names.values())
            or len(set(names.values())) != len(names)
            or any(n not in TARGET_CLASSES for n in names.values())):
        raise ValueError(f"names 必须是互不重复的道路类别：{', '.join(TARGET_CLASSES)}")
    root = (config.parent / str(data.get("path") or ".")).resolve()
    # Ultralytics 即使仅评估 test 也要求 train/val 两个键存在。
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
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"{split} 没有图片：{folder}")
        split_files[split] = {p.resolve() for p in files}
        objects = 0
        for image in files:
            parts = list(image.parts)
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"缺少标签：{label}；无目标图片也请提供空 txt。")
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    values = [float(v) for v in line.split()]
                    if len(values) != 5 or not all(math.isfinite(v) for v in values):
                        raise ValueError
                    cls, x, y, width, height = values
                    if not cls.is_integer() or int(cls) not in names:
                        raise ValueError
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        raise ValueError
                except ValueError:
                    raise ValueError(f"标签格式错误：{label}:{line_no}；应为 class_id x_center y_center width height，坐标归一化。") from None
                objects += 1
        if objects == 0:
            raise ValueError(f"{split} 没有任何目标标注，无法验证行人车辆检测。")
        normalized[split] = str(folder)
        report[split] = {"images": len(files), "objects": objects}
    for index, first in enumerate(required_splits):
        for second in required_splits[index + 1:]:
            if split_files[first] & split_files[second]:
                raise ValueError(f"{first} 与 {second} 包含相同图片路径，请重新划分数据。")
    return normalized, report
