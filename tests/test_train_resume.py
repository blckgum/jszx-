# -*- coding: utf-8 -*-
"""训练断点续训保留历史最佳 best.pt 与恢复权重契约测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

import config
import train as train_mod


class _TinyNet(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class TestTrainResumeBestCheckpoint(unittest.TestCase):
    def test_resume_preserves_higher_disk_best_pt_until_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_dir = root / "checkpoints"
            runs_dir = root / "runs"
            best_path = ckpt_dir / "best.pt"
            last_path = ckpt_dir / "last.pt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)

            sample_model = _TinyNet(10)
            sample_opt = torch.optim.AdamW(sample_model.parameters(), lr=1e-3)
            torch.save(
                {
                    "epoch": 5,
                    "model_name": "SimpleCNN",
                    "model_state": sample_model.state_dict(),
                    "optimizer_state": sample_opt.state_dict(),
                    "val_acc": 90.0,
                    "val_loss": 0.25,
                    "class_names": [str(i) for i in range(10)],
                },
                best_path,
            )
            torch.save(
                {
                    "epoch": 5,
                    "model_name": "SimpleCNN",
                    "model_state": sample_model.state_dict(),
                    "optimizer_state": sample_opt.state_dict(),
                    "val_acc": 85.0,
                    "class_names": [str(i) for i in range(10)],
                },
                last_path,
            )

            eval_seq = iter([
                (0.45, 70.0),  # epoch 6: worse than existing 90% best.pt
                (0.20, 88.0),  # test eval after epoch 6
                (0.15, 95.0),  # epoch 7: beats 90% best.pt
                (0.12, 94.0),  # test eval after epoch 7
            ])

            def _fake_loader(*args, **kwargs):
                xs = torch.randn(4, 3, 32, 32)
                ys = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                ds = torch.utils.data.TensorDataset(xs, ys)
                loader = torch.utils.data.DataLoader(ds, batch_size=4)
                return loader, loader, loader, [str(i) for i in range(10)]

            with mock.patch.object(config, "CKPT_DIR", ckpt_dir), \
                    mock.patch.object(config, "RUNS_DIR", runs_dir), \
                    mock.patch.object(config, "BEST_MODEL_PATH", best_path), \
                    mock.patch.object(config, "LAST_MODEL_PATH", last_path), \
                    mock.patch.object(config, "USE_SCHEDULER", False), \
                    mock.patch.object(train_mod, "get_device", return_value=torch.device("cpu")), \
                    mock.patch.object(train_mod, "build_dataloaders", side_effect=_fake_loader), \
                    mock.patch.object(train_mod, "build_model", side_effect=lambda *a, **kw: _TinyNet(10)), \
                    mock.patch.object(train_mod, "evaluate", side_effect=lambda *a, **kw: next(eval_seq)):
                # Run 1: resume to epoch 6 with 70% val_acc -> best.pt must stay at epoch 5 (90.0%)
                with mock.patch.object(sys, "argv", ["train.py", "--fake", "--epochs", "6", "--resume", str(last_path)]):
                    train_mod.main()

                kept_best = torch.load(best_path, map_location="cpu", weights_only=False)
                self.assertEqual(kept_best["epoch"], 5)
                self.assertAlmostEqual(kept_best["val_acc"], 90.0, places=4)
                hist1 = json.loads((runs_dir / "history.json").read_text(encoding="utf-8"))
                self.assertEqual(hist1["best_epoch"], 5)
                self.assertAlmostEqual(hist1["best_val_acc"], 90.0, places=4)

                # Run 2: resume from updated last.pt to epoch 7 with 95% val_acc -> updates best.pt to epoch 7
                with mock.patch.object(sys, "argv", ["train.py", "--fake", "--epochs", "7", "--resume", str(last_path)]):
                    train_mod.main()

                new_best = torch.load(best_path, map_location="cpu", weights_only=False)
                self.assertEqual(new_best["epoch"], 7)
                self.assertAlmostEqual(new_best["val_acc"], 95.0, places=4)

    def test_resume_cosine_scheduler_preserves_extended_t_max_and_aligns_plot_epochs(self):
        import math

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_dir = root / "checkpoints"
            runs_dir = root / "runs"
            best_path = ckpt_dir / "best.pt"
            last_path = ckpt_dir / "last.pt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)

            # 模拟第一阶段 5 轮余弦退火跑满后的检查点（T_max=5, last_epoch=5, lr 归零）
            base_lr = 1e-3
            sample_model = _TinyNet(10)
            sample_opt = torch.optim.AdamW(sample_model.parameters(), lr=base_lr)
            sample_sched = torch.optim.lr_scheduler.CosineAnnealingLR(sample_opt, T_max=5)
            for _ in range(5):
                sample_opt.step()
                sample_sched.step()
            self.assertEqual(sample_sched.T_max, 5)
            self.assertEqual(sample_sched.last_epoch, 5)

            torch.save(
                {
                    "epoch": 4,
                    "model_name": "SimpleCNN",
                    "model_state": sample_model.state_dict(),
                    "optimizer_state": sample_opt.state_dict(),
                    "val_acc": 80.0,
                    "val_loss": 0.4,
                    "class_names": [str(i) for i in range(10)],
                },
                best_path,
            )
            torch.save(
                {
                    "epoch": 5,
                    "model_name": "SimpleCNN",
                    "model_state": sample_model.state_dict(),
                    "optimizer_state": sample_opt.state_dict(),
                    "scheduler_state": sample_sched.state_dict(),
                    "val_acc": 79.0,
                    "best_val_acc": 80.0,
                    "best_epoch": 4,
                    "class_names": [str(i) for i in range(10)],
                },
                last_path,
            )

            eval_seq = iter([
                (0.35, 82.0),  # epoch 6
                (0.30, 86.0),  # epoch 7
                (0.22, 91.0),  # epoch 8 (new best)
                (0.25, 89.0),  # epoch 9
                (0.26, 88.0),  # epoch 10
                (0.20, 90.0),  # final test eval
            ])

            def _fake_loader(*args, **kwargs):
                xs = torch.randn(4, 3, 32, 32)
                ys = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                ds = torch.utils.data.TensorDataset(xs, ys)
                loader = torch.utils.data.DataLoader(ds, batch_size=4)
                return loader, loader, loader, [str(i) for i in range(10)]

            captured_plots: list[tuple[list[int], list[float]]] = []
            captured_vlines: list[float] = []

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.axes

            orig_plot = matplotlib.axes.Axes.plot
            orig_axvline = matplotlib.axes.Axes.axvline

            def _spy_plot(ax_self, x, y, *a, **kw):
                captured_plots.append((list(x), list(y)))
                return orig_plot(ax_self, x, y, *a, **kw)

            def _spy_axvline(ax_self, x=0, *a, **kw):
                captured_vlines.append(x)
                return orig_axvline(ax_self, x=x, *a, **kw)

            with mock.patch.object(config, "CKPT_DIR", ckpt_dir), \
                    mock.patch.object(config, "RUNS_DIR", runs_dir), \
                    mock.patch.object(config, "BEST_MODEL_PATH", best_path), \
                    mock.patch.object(config, "LAST_MODEL_PATH", last_path), \
                    mock.patch.object(config, "USE_SCHEDULER", True), \
                    mock.patch.object(train_mod, "get_device", return_value=torch.device("cpu")), \
                    mock.patch.object(train_mod, "build_dataloaders", side_effect=_fake_loader), \
                    mock.patch.object(train_mod, "build_model", side_effect=lambda *a, **kw: _TinyNet(10)), \
                    mock.patch.object(train_mod, "evaluate", side_effect=lambda *a, **kw: next(eval_seq)), \
                    mock.patch.object(matplotlib.axes.Axes, "plot", _spy_plot), \
                    mock.patch.object(matplotlib.axes.Axes, "axvline", _spy_axvline), \
                    mock.patch.object(
                        sys, "argv",
                        ["train.py", "--fake", "--epochs", "10", "--lr", str(base_lr), "--resume", str(last_path)],
                    ):
                train_mod.main()

            updated_last = torch.load(last_path, map_location="cpu", weights_only=False)
            sched_state = updated_last["scheduler_state"]
            self.assertEqual(sched_state["T_max"], 10)
            self.assertEqual(sched_state["last_epoch"], 10)

            hist = json.loads((runs_dir / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(hist["start_epoch"], 6)
            self.assertEqual(hist["best_epoch"], 8)
            self.assertEqual(hist["history"]["epoch"], [6, 7, 8, 9, 10])

            # 验证每个续训 epoch 的学习率严格单调递减，且精确匹配 T_max=10 的余弦退火曲线
            lrs = hist["history"]["lr"]
            self.assertEqual(len(lrs), 5)
            for idx, observed_lr in enumerate(lrs):
                step_num = 5 + idx  # epoch 6 对应 last_epoch=5 ... epoch 10 对应 last_epoch=9
                expected_lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * step_num / 10))
                self.assertAlmostEqual(observed_lr, expected_lr, places=10)
            for earlier, later in zip(lrs, lrs[1:]):
                self.assertGreater(earlier, later)

            # 验证 plot_curves 使用绝对 epoch [6..10] 作为横轴，与 best_epoch=8 对齐
            self.assertTrue((runs_dir / "training_curves.png").is_file())
            self.assertEqual(len(captured_plots), 4)
            for xs, _ in captured_plots:
                self.assertEqual(xs, [6, 7, 8, 9, 10])
            self.assertEqual(captured_vlines, [8, 8])

            # 兼容旧版仅含 start_epoch 或无 epoch 字段的 history.json
            self.assertEqual(
                train_mod._resolve_history_epochs({"history": {"train_loss": [1.0, 0.8]}, "start_epoch": 6}),
                [6, 7],
            )
            self.assertEqual(
                train_mod._resolve_history_epochs({"history": {"train_loss": [1.0, 0.8]}}),
                [1, 2],
            )

    def test_resume_avoids_crashing_on_incompatible_global_best_checkpoint(self):
        class _OtherNet(torch.nn.Module):
            def __init__(self, num_classes: int = 10):
                super().__init__()
                self.other = torch.nn.Linear(3 * 32 * 32, num_classes)

            def forward(self, x):
                return self.other(x.view(x.size(0), -1))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt_dir = root / "checkpoints"
            runs_dir = root / "runs"
            best_path = ckpt_dir / "best.pt"
            last_path = ckpt_dir / "last.pt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)

            # 全局 best.pt 属于另一种架构，且保留 90% 历史最高；本轮续训 SimpleCNN 仅取得 85%，不得加载异构 best.pt 崩溃
            other_model = _OtherNet(10)
            torch.save(
                {
                    "epoch": 3,
                    "model_name": "ResNet18",
                    "model_state": other_model.state_dict(),
                    "val_acc": 92.0,
                    "class_names": [str(i) for i in range(10)],
                },
                best_path,
            )
            cnn_model = _TinyNet(10)
            cnn_opt = torch.optim.AdamW(cnn_model.parameters(), lr=1e-3)
            torch.save(
                {
                    "epoch": 5,
                    "model_name": "SimpleCNN",
                    "model_state": cnn_model.state_dict(),
                    "optimizer_state": cnn_opt.state_dict(),
                    "val_acc": 80.0,
                    "best_val_acc": 90.0,
                    "best_epoch": 4,
                    "class_names": [str(i) for i in range(10)],
                },
                last_path,
            )

            eval_seq = iter([(0.28, 85.0), (0.25, 86.0)])

            def _fake_loader(*args, **kwargs):
                xs = torch.randn(4, 3, 32, 32)
                ys = torch.tensor([0, 1, 2, 3], dtype=torch.long)
                ds = torch.utils.data.TensorDataset(xs, ys)
                loader = torch.utils.data.DataLoader(ds, batch_size=4)
                return loader, loader, loader, [str(i) for i in range(10)]

            with mock.patch.object(config, "CKPT_DIR", ckpt_dir), \
                    mock.patch.object(config, "RUNS_DIR", runs_dir), \
                    mock.patch.object(config, "BEST_MODEL_PATH", best_path), \
                    mock.patch.object(config, "LAST_MODEL_PATH", last_path), \
                    mock.patch.object(config, "USE_SCHEDULER", False), \
                    mock.patch.object(train_mod, "get_device", return_value=torch.device("cpu")), \
                    mock.patch.object(train_mod, "build_dataloaders", side_effect=_fake_loader), \
                    mock.patch.object(train_mod, "build_model", side_effect=lambda *a, **kw: _TinyNet(10)), \
                    mock.patch.object(train_mod, "evaluate", side_effect=lambda *a, **kw: next(eval_seq)), \
                    mock.patch.object(sys, "argv", ["train.py", "--fake", "--model", "SimpleCNN", "--epochs", "6", "--resume", str(last_path)]):
                train_mod.main()

            preserved_best = torch.load(best_path, map_location="cpu", weights_only=False)
            self.assertEqual(preserved_best["model_name"], "ResNet18")


if __name__ == "__main__":
    unittest.main()
