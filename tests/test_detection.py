"""验证最易导致错误训练和评估的类别映射、标签及数据划分问题。"""

# 这些是数据与工程接口的回归测试，不是检测精度实验。
# 它们不用下载模型或运行神经网络，主要检查“错误数据是否会被及时发现”。
# YOLO 的效果需要真实图片、人工标注和独立测试集，不能由这六个测试的通过率代替。

import tempfile
import unittest
from pathlib import Path

import yaml

from detection.core import new_run, select_classes
from detection.dataset import audit_dataset


class DetectionContractTests(unittest.TestCase):
    def test_class_ids_follow_model_not_coco_order(self):
        # 故意把 person 放在 ID=1，检查代码是否真的读取模型 names，而不是写死 person=0。
        self.assertEqual(select_classes({0: "truck", 1: "person", 2: "car"}, ["person", "car"]), [1, 2])
        self.assertEqual(select_classes({0: "person", 1: "car", 2: "dog"}), [0, 1])
        with self.assertRaisesRegex(ValueError, "缺少"):
            select_classes({0: "person"}, ["car"])

    def test_existing_output_is_preserved(self):
        # 模拟已有实验目录，确认开始新任务时不会覆盖用户保存的结果。
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "user.txt"
            marker.write_text("user result")
            with self.assertRaises(FileExistsError):
                new_run(temporary, "detect")
            self.assertEqual(marker.read_text(), "user result")

    def make_dataset(self, root):
        # 构造最小的“两类、训练/验证各一张图”的目录，仅用于验证标签和路径规则。
        # TemporaryDirectory 让每个测试使用独立数据，不影响仓库中的真实数据集。
        for split in ("train", "val"):
            images = root / "images" / split
            labels = root / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            # 审计只检查文件配对与标注；图像解码由实际 YOLO 运行验证。
            (images / "sample.jpg").touch()
            (labels / "sample.txt").write_text("0 0.5 0.5 0.3 0.4\n")
        config = root / "data.yaml"
        config.write_text(yaml.safe_dump({"path": ".", "train": "images/train", "val": "images/val",
                                          "names": ["person", "car"]}))
        return config

    def test_relative_paths_resolve_from_yaml_location(self):
        # 换一个终端工作目录不应改变 YAML 指向的数据，数据根目录由配置文件的位置决定。
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, report = audit_dataset(self.make_dataset(root), ("train", "val"))
            self.assertEqual(Path(data["path"]).resolve(), root.resolve())
            self.assertEqual(report["val"], {"images": 1, "objects": 1})

    def test_missing_labels_are_not_silently_negative_samples(self):
        # 删除标签模拟漏交文件；它与“明确提供空标签的无目标图片”具有不同含义。
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_dataset(root)
            (root / "labels/train/sample.txt").unlink()
            with self.assertRaisesRegex(ValueError, "缺少标签"):
                audit_dataset(config, ("train", "val"))

    def test_invalid_class_and_nonfinite_boxes_are_rejected(self):
        # 分别模拟越界类别 ID、NaN 坐标和负宽度，防止不合法标注进入训练损失计算。
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_dataset(root)
            for line in ("2 0.5 0.5 0.2 0.3", "0 nan 0.5 0.2 0.3", "0 0.5 0.5 -0.2 0.3"):
                with self.subTest(line=line):
                    (root / "labels/train/sample.txt").write_text(line)
                    with self.assertRaisesRegex(ValueError, "标签格式错误"):
                        audit_dataset(config, ("train", "val"))

    def test_train_val_same_images_are_rejected(self):
        # 验证集如果直接复用训练图片，得到的成绩无法说明模型对未见数据的泛化能力。
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_dataset(root)
            content = yaml.safe_load(config.read_text())
            content["val"] = content["train"]
            config.write_text(yaml.safe_dump(content))
            with self.assertRaisesRegex(ValueError, "相同图片"):
                audit_dataset(config, ("train", "val"))


if __name__ == "__main__":
    unittest.main()
