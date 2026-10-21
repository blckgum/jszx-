# 原理笔记：docs/modules/03_图像分类实验_原理笔记.md。
# 本文件属于原有分类学习模块；校园道路检测与跟踪使用 detection/、tracking/ 入口。

"""
train.py —— 完整训练流程

这是整个项目最重要的文件。看懂它，PyTorch 就通了 80%。

────────────────────────────────────────────────────────
训练循环五步曲（背下来）：

    for images, labels in train_loader:

        images, labels = 搬到GPU(images, labels)   # 0. 数据上显卡
        optimizer.zero_grad()                      # 1. 清空上轮梯度
        outputs = model(images)                    # 2. 前向传播：算预测
        loss = criterion(outputs, labels)          # 3. 算损失：错了多少
        loss.backward()                            # 4. 反向传播：算梯度
        optimizer.step()                           # 5. 更新参数

前向传播是"猜"，反向传播是"反思"，优化器是"改进"。循环往复。

────────────────────────────────────────────────────────
怎么运行：

    # 先快速验证代码对不对（离线合成数据，1 分钟）
    python train.py --fake --epochs 5

    # 再用真实数据训练（CIFAR-10，首次要下载 170MB）
    python train.py --epochs 30

    # 接着上次继续训练
    python train.py --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

import config
from common import (AverageMeter, banner, count_parameters, ensure_dir, get_device,
                    human_number, set_seed, Timer)
from dataset import build_dataloaders
from model import build_model


# ============================================================
# 一、训练一个 epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    把训练集完整跑一遍。

    返回: (平均损失, 准确率)

    ★ 为什么训练和验证要写成两个函数？
      因为它们的逻辑完全不同：
        训练: 要算梯度、要更新参数、Dropout 要开启
        验证: 不算梯度、不更新参数、Dropout 要关闭
      混在一个函数里容易出错。
    """
    # ★ 切换到训练模式。
    #   这会开启 Dropout 和 BatchNorm 的"训练行为"（用当前 batch 的统计量）。
    #   忘了写这行，Dropout 就不起作用 → 容易过拟合。
    model.train()

    loss_meter = AverageMeter()   # 累计损失
    correct = 0                   # 预测对的样本数
    total = 0                     # 总样本数

    timer = Timer()

    for step, (images, labels) in enumerate(loader):
        # ---- 0. 把数据搬到和模型同一个设备上 ----
        # ★ 必须搬！否则报 "Expected all tensors to be on the same device"
        #   non_blocking=True 让传输异步进行，配合 pin_memory 能快一点
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ---- 1. 清空上一轮的梯度 ----
        # ★★ 这是最容易被忽略的一行！
        #    PyTorch 的梯度是**累加**的：backward() 会把新梯度加到 .grad 上。
        #    不清零的话，梯度会越来越大，几步之后直接爆炸(NaN)。
        #    设计成累加是为了支持"梯度累积"等高级技巧，但绝大多数场景都要手动清零。
        optimizer.zero_grad()

        # ---- 2. 前向传播 ----
        # 数据流过网络，得到每个类别的原始分数 logits
        outputs = model(images)          # [N, 10]

        # ---- 3. 计算损失 ----
        # ★ CrossEntropyLoss 内部自带 Softmax，所以直接喂 logits
        #   如果先手动 softmax 再喂进去，就等于做了两次，结果全错
        loss = criterion(outputs, labels)

        # ---- 4. 反向传播 ----
        # 从 loss 出发，用链式法则一路往回算每个参数的梯度，
        # 结果存到 param.grad 里
        loss.backward()

        # ★ 梯度裁剪（可选但推荐）：
        #   如果某个梯度特别大，参数会被"一脚踹飞"，训练直接崩。
        #   把梯度的总范数限制在 1.0 以内，防止这种情况。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ---- 5. 更新参数 ----
        # 执行 θ = θ - lr * grad，这一步才真正改变模型
        optimizer.step()

        # ---- 统计 ----
        # ★ 计算准确率必须在 no_grad 下做，否则会在计算图里留下记录、白费显存
        with torch.no_grad():
            # argmax(dim=1) 取每行最大值的**下标**，也就是预测的类别号
            preds = outputs.argmax(dim=1)          # [N]
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # loss 是张量，.item() 取出 Python 浮点数（不带计算图，不占显存）
        # ★ 一定要 .item()！否则会把整个计算图累积下来，显存直接爆
        loss_meter.update(loss.item(), n=labels.size(0))

        # 每 20 个 batch 打印一次进度
        if step % 20 == 0 or step == len(loader) - 1:
            print(f"\r  训练 Epoch {epoch}/{total_epochs} "
                  f"[{step + 1}/{len(loader)}] "
                  f"loss={loss_meter.avg:.4f} "
                  f"acc={100.0 * correct / max(1, total):.1f}% "
                  f"已用 {timer.elapsed_str()}", end="", flush=True)

    print()  # 换行，结束这一行的进度刷新

    avg_loss = loss_meter.avg
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy


# ============================================================
# 二、在验证集上评估
# ============================================================

@torch.no_grad()   # ★ 装饰器：整个函数的代码都在"不算梯度"模式下运行
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    在验证集/测试集上评估模型。不更新任何参数。

    ★ 三个关键点：
      1. @torch.no_grad()  —— 不算梯度，省显存提速（能省 30%~50%）
      2. model.eval()       —— 关闭 Dropout，固定 BatchNorm 统计量
      3. 不调用 optimizer.step() —— 绝不更新参数

    少任何一个，结果都不可信。
    """
    model.eval()   # ★ 极易遗漏！忘了它，验证准确率会莫名其妙地低且抖动

    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loss_meter.update(loss.item(), n=labels.size(0))

    return loss_meter.avg, 100.0 * correct / max(1, total)


# ============================================================
# 三、主训练流程
# ============================================================

def main() -> None:
    # ---------- 解析命令行参数 ----------
    parser = argparse.ArgumentParser(description="PyTorch 训练脚本")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help=f"训练轮数（默认 {config.EPOCHS}）")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help=f"批大小（默认 {config.BATCH_SIZE}）")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help=f"学习率（默认 {config.LEARNING_RATE}）")
    parser.add_argument("--fake", action="store_true",
                        help="用离线合成数据快速验证代码（不联网）")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME,
                        choices=["SimpleCNN", "ResNet18"],
                        help="模型结构")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点继续训练，例如 checkpoints/last.pt")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE,
                        help="早停耐心值：验证 loss 连续 N 轮不降就停")
    parser.add_argument("--seed", type=int, default=config.SEED, help="随机种子")
    args = parser.parse_args()

    # ---------- 初始化 ----------
    banner("PyTorch 图像分类训练")

    # ★ 第一步永远是固定随机种子，否则实验结果没法复现
    set_seed(args.seed)
    device = get_device()

    print(f"\n配置: 模型={args.model}  轮数={args.epochs}  "
          f"批大小={args.batch_size}  学习率={args.lr}")

    # ---------- 数据 ----------
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        use_synthetic=args.fake,
        batch_size=args.batch_size,
    )
    num_classes = len(class_names)

    # ---------- 模型 ----------
    model = build_model(args.model, num_classes=num_classes).to(device)

    n_params = count_parameters(model)
    print(f"\n模型参数量: {human_number(n_params)}")

    # 粗略估算模型大小：每个参数 4 字节(float32)
    print(f"模型大小  : {n_params * 4 / 1024**2:.2f} MB")

    # ★ 数据量 vs 参数量，是判断过拟合风险的关键
    # 经验法则：每个参数至少要有 1~10 个训练样本，否则必然过拟合
    n_train = len(train_loader.dataset)
    print(f"训练样本数: {n_train}")
    print(f"样本/参数比: 1 : {n_params / max(1, n_train):.1f}", end="")
    if n_params / max(1, n_train) > 50:
        print("  ← 偏高，建议用数据增强 + Dropout，或者换更小的模型")
    else:
        print("  ← 还算合理")

    # ---------- 损失函数 ----------
    # ★ 分类任务标配。它内部 = LogSoftmax + NLLLoss，所以模型输出直接给 logits
    criterion = nn.CrossEntropyLoss()

    # ---------- 优化器 ----------
    # AdamW = Adam + 正确的权重衰减实现，收敛快，对学习率不敏感，新手首选
    optimizer = torch.optim.AdamW(
        model.parameters(),      # ★ 必须传模型参数，否则什么都更新不了！
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,   # AdamW 的解耦权重衰减
    )

    # ---------- 学习率调度器 ----------
    # 余弦退火：学习率按余弦曲线从 lr 降到 0
    # 前期大步走快速下降，后期小步走精细收敛
    scheduler = None
    if config.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,   # 一个完整余弦周期的长度
        )

    # ---------- 恢复训练（可选） ----------
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n从检查点恢复: {resume_path}")
            # ★ weights_only=False 才能加载含优化器状态的完整检查点
            #   PyTorch 2.6+ 默认 weights_only=True，加载自己存的字典会报错
            # 当前仅恢复模型、优化器和轮号；调度器、随机状态、历史最佳值均未恢复，
            # 因此它支持继续优化，不保证与一次不中断的训练得到相同轨迹。
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"从第 {start_epoch} 轮继续")
        else:
            print(f"[警告] 找不到检查点 {resume_path}，从头开始训练")

    # ---------- 准备保存目录 ----------
    # 分类脚本使用固定 best.pt/last.pt/history.json，重新训练会覆盖同名产物。
    # 做对比实验前另存旧结果；检测/跟踪入口则通过 new_run 自动隔离目录。
    ensure_dir(config.CKPT_DIR)
    ensure_dir(config.RUNS_DIR)

    # ---------- 开始训练 ----------
    banner("开始训练")

    best_val_acc = 0.0          # 历史最佳验证准确率
    best_epoch = 0
    epochs_no_improve = 0       # 连续多少轮没进步（用于早停）
    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }
    total_timer = Timer()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'─' * 60}")
        current_lr = optimizer.param_groups[0]["lr"]   # 读当前学习率
        print(f"Epoch {epoch}/{args.epochs}   学习率 = {current_lr:.6f}")
        print(f"{'─' * 60}")

        # ---- 训练一轮 ----
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        # ---- 验证 ----
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # ---- 更新学习率 ----
        if scheduler is not None:
            scheduler.step()   # ★ 注意顺序：一般在 epoch 结束后调用

        # ---- 记录历史（后面画曲线用）----
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # ---- 打印本轮结果 ----
        print(f"\n  训练: loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  验证: loss={val_loss:.4f}  acc={val_acc:.2f}%")

        # ★ 诊断过拟合 / 欠拟合（这是最实用的技巧）
        gap = train_acc - val_acc
        if gap > 10:
            print(f"  [!] 训练-验证差距 {gap:.1f}% → 过拟合了："
                  f"加强数据增强 / 增大 Dropout / 早停")
        elif train_acc < 60:
            print(f"  [!] 两者都低 → 欠拟合："
                  f"模型太小 / 学习率不对 / 训练轮数不够")
        else:
            print(f"  训练-验证差距 {gap:.1f}% → 正常范围")

        # ---- 保存最佳模型 ----
        # ★ 判断标准用"验证集准确率"，绝不能用训练集！
        #   用训练集选模型 = 在选"背书背得最好的"，而不是"真会做题的"
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0

            # 保存检查点。
            # ★ 为什么要存这么多东西？
            #   model_state    —— 推断时只需要这个
            #   optimizer_state —— 想接着训练就需要它（Adam 内部有动量等状态）
            #   epoch          —— 从哪一轮继续
            #   class_names    —— 推理时要把下标变回类别名，必须存！
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "class_names": class_names,
                    "config": {
                        "num_classes": num_classes,
                        "image_size": config.IMAGE_SIZE,
                        "in_channels": config.IN_CHANNELS,
                    },
                },
                config.BEST_MODEL_PATH,
            )
            print(f"  [OK] 新的最佳模型已保存 (val_acc={val_acc:.2f}%)"
                  f" → {config.BEST_MODEL_PATH}")
        else:
            epochs_no_improve += 1
            print(f"  · 未提升 ({epochs_no_improve}/{args.patience})  "
                  f"历史最佳 {best_val_acc:.2f}% (第 {best_epoch} 轮)")

        # ---- 每轮保存 last.pt，供下一次从已写好的检查点继续优化 ----
        torch.save(
            {
                "epoch": epoch,
                "model_name": args.model,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
            },
            config.LAST_MODEL_PATH,
        )

        # ---- 早停 ----
        # 当前实现结合两个条件：本轮 val_loss 高于此前最小值，且验证准确率
        # 已连续 patience 轮没有提高；不能解读成单独按 loss 计数的标准早停。
        if len(history["val_loss"]) > 1:
            if val_loss > min(history["val_loss"][:-1]) and epochs_no_improve >= args.patience:
                print(f"\n[早停] 验证损失连续 {args.patience} 轮没有改善，停止训练")
                print(f"       最佳成绩在第 {best_epoch} 轮: val_acc={best_val_acc:.2f}%")
                break

    # ---------- 训练结束，用最佳模型测测试集 ----------
    banner("加载最佳模型，在测试集上做最终评估")

    if not config.BEST_MODEL_PATH.exists():
        print("[错误] 没有找到最佳模型，训练可能失败了")
        return

    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    print(f"最佳模型来自第 {ckpt['epoch']} 轮，验证集准确率 {ckpt['val_acc']:.2f}%")

    # ★★ 测试集全程只用这一次（数据泄漏的防护纪律）
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'=' * 60}")
    print(f"  最终测试集结果")
    print(f"{'=' * 60}")
    print(f"  测试损失  : {test_loss:.4f}")
    print(f"  测试准确率: {test_acc:.2f}%")
    print(f"  总耗时    : {total_timer.elapsed_str()}")
    print(f"{'=' * 60}")
    print("\n  ★ 这个数字才是可以写进课题报告的成绩。")
    print("    千万不要用它来反复调参，那样报出来的分数是虚高的。")

    # ---------- 保存训练历史 ----------
    history_path = config.RUNS_DIR / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "args": vars(args),
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n训练历史已保存: {history_path}")

    # ---------- 画曲线图（可选）----------
    plot_curves(history_path)


def plot_curves(history_path: Path) -> None:
    """
    画出 loss 和准确率曲线。判断过拟合最直观的方法就是看这两张图。

    ★ 看图技巧：
      左图（loss）：两条线一起下降最好。如果验证 loss 掉头往上升，
                   说明从那一刻起开始过拟合，最佳模型应该在拐点附近。
      右图（acc） ：训练 acc 一直涨，验证 acc 早早平了 → 也是过拟合。

    需要 matplotlib，没装就跳过。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # 不弹窗口，直接存文件（服务器/终端环境必须）
        import matplotlib.pyplot as plt
    except ImportError:
        print("[提示] 没装 matplotlib，跳过画图。想画图执行:")
        print("       .venv\\Scripts\\pip install matplotlib")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data["history"]
    epochs = range(1, len(h["train_loss"]) + 1)
    best_epoch = data.get("best_epoch", 0)

    # 中文字体设置，否则中文会显示成方块
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False   # 负号正常显示

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 左图：损失曲线
    ax1.plot(epochs, h["train_loss"], "o-", label="训练损失", markersize=3)
    ax1.plot(epochs, h["val_loss"], "s-", label="验证损失", markersize=3)
    if best_epoch:
        # 标出最佳模型的位置
        ax1.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5,
                    label=f"最佳模型 (第{best_epoch}轮)")
    ax1.set_xlabel("Epoch（训练轮数）")
    ax1.set_ylabel("Loss（损失）")
    ax1.set_title("损失曲线 —— 验证损失开始上升就是过拟合的信号")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 右图：准确率曲线
    ax2.plot(epochs, h["train_acc"], "o-", label="训练准确率", markersize=3)
    ax2.plot(epochs, h["val_acc"], "s-", label="验证准确率", markersize=3)
    if best_epoch:
        ax2.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5,
                    label=f"最佳模型 (第{best_epoch}轮)")
    ax2.set_xlabel("Epoch（训练轮数）")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("准确率曲线 —— 两条线差距越大，过拟合越严重")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = history_path.parent / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"训练曲线已保存: {out_path}  ← 打开看看，这是写报告的好素材")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    main()
