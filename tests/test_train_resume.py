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


if __name__ == "__main__":
    unittest.main()
