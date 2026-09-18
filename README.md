# PyTorch 图像识别框架（学习 + 项目骨架）

一个**带详细中文注释**的 PyTorch 项目骨架。每个文件都在解释"为什么这么写"，
而不只是"怎么写"。



---

## 快速开始

```powershell
# 1. 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 2. 环境自检 —— 确认 GPU 能用（★ 第一步永远是这个）
python common.py

# 3. 看网络结构和形状变化
python model.py

# 4. 快速跑通训练（离线合成数据，1 分钟，不联网）
python train.py --fake --epochs 5

# 5. 用自己的数据训练（首次会下载 CIFAR-10 约 170MB）
python train.py --epochs 30

# 6. 推理测试
python predict.py --demo
```

---

## 目录结构

```
pytorch-lab/
├── config.py               ← 所有超参数都在这，调参只改这个文件
├── common.py               ← 设备选择 / 随机种子 / 计时器 / 环境自检
├── dataset.py              ← 数据加载 + 数据增强（重点看注释）
├── model.py                ← 网络结构，标明了每一层的形状变化
├── train.py         ★      ← 训练循环，整个项目的心脏
├── predict.py              ← 推理：输出类别 + 置信度 + 可视化
│
├── lessons/                ← 动手理解原理的实验（建议按顺序跑）
│   ├── 01_tensor_autograd.py    张量、自动求导、计算图
│   ├── 02_gradient_descent.py   手写梯度下降、学习率实验
│   └── 03_convolution.py        手工算卷积、池化、参数量
│
├── docs/
│   └── PyTorch原理笔记.md   ← 理论讲解，配合 lessons 一起看
│
├── data/                   ← 数据集（自动下载）
├── checkpoints/            ← best.pt / last.pt
└── runs/                   ← 训练历史 JSON + 曲线图
```

---

## 学习路线（建议顺序）

| 步骤 | 命令 | 学到什么 |
|---|---|---|
| 1 | `python common.py` | 确认 GPU 可用，认识设备管理 |
| 2 | `python lessons/01_tensor_autograd.py` | **自动求导**：`backward()` 到底做了什么 |
| 3 | `python lessons/02_gradient_descent.py` | **梯度下降**：模型怎么"学"；学习率的影响 |
| 4 | `python lessons/03_convolution.py` | **卷积**：手工算一遍，理解参数量为什么这么少 |
| 5 | `python model.py` | 看懂网络结构，追踪每一层的形状 |
| 6 | `python train.py --fake --epochs 5` | **训练循环五步曲** ← 最重要 |
| 7 | `python train.py --epochs 30` | 用真实数据训练，看曲线判断过拟合 |
| 8 | `python predict.py --image xxx.jpg` | 推理、置信度、可视化 |

**理论部分**看 `docs/PyTorch原理笔记.md`，12 节，从张量讲到 YOLO。

---

## 训练循环：整个项目的核心

```python
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)  # 0. 数据上显卡

    optimizer.zero_grad()            # 1. 清空上轮梯度（★ 别忘了！）
    outputs = model(images)          # 2. 前向传播：算预测
    loss = criterion(outputs, labels)# 3. 算损失：错了多少
    loss.backward()                  # 4. 反向传播：算梯度
    optimizer.step()                 # 5. 更新参数：改进
```

这五行就是深度学习的全部。YOLO、ResNet、GPT 都是这个循环，
区别只在 `model` 长什么样。

---

## 常用命令

```powershell
# 快速验证代码（离线，1分钟）
python train.py --fake --epochs 5

# 正式训练
python train.py --epochs 50 --batch-size 128 --lr 0.001

# 从断点继续训练
python train.py --resume checkpoints/last.pt

# 换模型（迁移学习）
python train.py --model ResNet18 --epochs 10

# 推理
python predict.py --demo                    # 随机输入演示
python predict.py --image samples/cat.jpg   # 单张图
python predict.py --folder samples/         # 整个文件夹

# 查看当前配置
python config.py
```

---

## 你的课题怎么接进来

等你确定识别内容后，改动点只有三处：

### 1. 改类别（`config.py`）
```python
CLASSES = ["可回收物", "厨余垃圾", "有害垃圾", "其他垃圾"]
NUM_CLASSES = len(CLASSES)
```

### 2. 改数据集（`dataset.py`）
按文件夹结构组织数据（ImageFolder 格式）：
```
data/your_dataset/
├── train/
│   ├── 可回收物/  *.jpg
│   ├── 厨余垃圾/  *.jpg
│   └── ...
└── val/
    └── ...
```
然后用 `torchvision.datasets.ImageFolder` 替换 CIFAR-10 部分。

### 3. 数据不够就用迁移学习
```powershell
python train.py --model ResNet18 --epochs 10
```
几百张图也能出效果。

---

## 从分类到检测（架构图的最终形态）

你原来的架构图是**目标检测**（YOLOv8）。分类和检测的关系：

| | 分类 | 检测 |
|---|---|---|
| 回答的问题 | 这张图**是什么** | 图里**有什么 + 在哪** |
| 输出 | 一个类别 | 一堆框 + 每框的类别 |
| 标注成本 | 低（整图一个标签） | 高（每个物体画框） |

**建议路线**：
1. 先用这套代码把**分类**跑通 —— 你已经完成了 80% 的学习
2. 再换 YOLOv8 做检测：
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   results = model.train(data="data.yaml", epochs=100, imgsz=640)
   ```
   Ultralytics 把损失函数、数据增强、后处理全封装好了，核心代码只有几行。
   但**你得先懂这套代码里的原理**，否则报错了完全不知道怎么调。

---

## 环境说明

| 项 | 值 | 备注 |
|---|---|---|
| 显卡 | RTX 5060 8GB (sm_120) | Blackwell 架构 |
| Python | 3.13.7 | 虚拟环境在 `.venv/` |
| PyTorch | 2.11.0 + **cu128** | ★ 必须是 cu128 及以上 |

> [!] **RTX 50 系显卡的坑**：
> 老版本 PyTorch（cu121/cu124）编译时只支持到 sm_89/sm_90，
> 在 5060 上会报 `CUDA error: no kernel image is available`。
> 必须装 CUDA 12.8+ 的版本。具体的版本要求见下方"环境重建"。

### 环境重建（如果 `.venv` 坏了）

```powershell
# 用 Python 3.13 建环境（★ 不要用 3.14，PyTorch 支持滞后）
py -3.13 -m venv .venv

# 装 CUDA 版 PyTorch（★ 必须指定 cu128 索引）
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 装其他依赖
.\.venv\Scripts\pip install -r requirements.txt
```

---

## 排错速查

报错里 90% 是下面这几个。参考 `docs/PyTorch原理笔记.md` 第 1 节。

| 报错 / 现象 | 原因 | 解决 |
|---|---|---|
| `Expected all tensors on the same device` | 数据和模型不在同一设备 | 检查是否都 `.to(device)` |
| `no kernel image is available` | PyTorch 版本不支持新显卡 | 装 cu128+ 版本 |
| `RuntimeError: start a new process` | Windows 上 `num_workers > 0` | 设 `NUM_WORKERS = 0` |
| loss 变成 `NaN` | 学习率太大 / 梯度爆炸 | 调小 lr，加梯度裁剪 |
| 训练准确率 99%，验证只有 60% | **过拟合** | 数据增强、Dropout、早停 |
| 每次预测结果都不一样 | 忘了 `model.eval()` | 推理前必须切换 |
| 加载检查点报错 | PyTorch 2.6+ 默认 `weights_only=True` | 加 `weights_only=False` |
| 验证准确率莫名偏低 | 忘了 `model.eval()` | 同上 |
| 显存不够 OOM | `batch_size` 太大 | 调小 batch_size |

---

## 写课题报告能用的数据

跑完 `train.py` 后，`runs/` 目录里会有：

- `history.json` —— 每轮的 loss 和准确率（可以直接做成表格）
- `training_curves.png` —— **损失曲线 + 准确率曲线，写报告的好素材**

报告里要写清楚这几个概念（`docs/PyTorch原理笔记.md` 第 9~11 节有详细说明）：

- **训练集 / 验证集 / 测试集** 的区别和为什么必须分开
- **过拟合** 怎么判断（看两条 loss 曲线的分叉）
- **准确率为什么会骗人**（类别不均衡时要用精确率/召回率/F1）
