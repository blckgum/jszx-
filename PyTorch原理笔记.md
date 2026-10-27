# PyTorch 原理笔记

> 这份笔记配合 `lessons/` 里的脚本一起看。**先读一节，再跑对应的脚本**，比只看书快十倍。

---

## 0. 先看全局地图

你在架构图里画的流程，拆开看其实只有 6 个零件：

```
数据 (Dataset)
   ↓  批处理
DataLoader  →  images, labels          张量 Tensor
   ↓
模型 nn.Module  →  outputs (logits)    前向传播 forward
   ↓
损失函数 loss_fn(outputs, labels)     算"错了多少"
   ↓
loss.backward()                        反向传播 backward，算梯度
   ↓
optimizer.step()                       按梯度更新参数
   ↓
循环 N 轮  →  best.pt  →  推理
```

**整个深度学习，就是在重复上面这个圈。** YOLO、ResNet、GPT，全都是这个圈，区别只在"模型"长什么样。

---

## 1. 张量 Tensor

张量 = **带 GPU 加速 + 自动求导** 的 numpy 数组。

```python
import torch
a = torch.tensor([[1., 2.], [3., 4.]])
a.shape      # torch.Size([2, 2])  形状
a.dtype      # torch.float32       数据类型
a.device     # cpu 还是 cuda:0     存在哪块内存
```

三个属性记住就行：**形状、类型、设备**。90% 的 PyTorch 报错都出在这三个上。

常见的形状约定（很重要）：

| 数据 | 形状 | 含义 |
|---|---|---|
| 一组图片 | `[N, C, H, W]` | N 张图，C 通道，高 H，宽 W |
| 分类标签 | `[N]` | N 个整数，取值 `0 ~ num_classes-1` |
| 检测框 | `[N, 4]` | 每个框 `(x中心, y中心, 宽, 高)` |

> [!] PyTorch 图片是 **通道在前** `[N,C,H,W]`，而 OpenCV / PIL 读进来是 `[H,W,C]`。
> 转换要用 `permute`，这是新手最容易踩的坑，`predict.py` 里有处理。

---

## 2. 自动求导 autograd（核心中的核心）

### 2.1 为什么需要它

训练模型 = 找出让损失最小的参数。要知道往哪个方向调，就得求**损失对每个参数的偏导数**（梯度）。

一个网络可能有一亿个参数。手推公式？不可能。所以 PyTorch 帮你做。

### 2.2 计算图

只要一个张量 `requires_grad=True`，PyTorch 就会记录你对它做的**每一步运算**，形成一张有向图：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x      # PyTorch 悄悄记下：这里做了 平方 和 乘3加
y.backward()            # 从 y 往回走，一路求导
print(x.grad)           # dy/dx = 2x + 3 = 7
```

### 2.3 链式法则是怎么滚起来的

假设 $y = f(g(h(x)))$，中间量 $u=h(x),\ v=g(u),\ y=f(v)$。

链式法则：

$$\frac{dy}{dx} = \frac{dy}{dv}\cdot\frac{dv}{du}\cdot\frac{du}{dx}$$

`backward()` 就是从输出 $y$ 出发，**反向**把每一层的局部导数**乘**起来，一直乘到输入。

- **前向传播 (forward)**：从输入算到输出，顺手把中间结果存下来。
- **反向传播 (backward)**：从损失算到参数梯度，用链式法则逐层往回乘。

这就是"反向传播"名字的由来，也是为什么它比朴素求导快无数倍 —— 中间结果复用了。

### 2.4 两个必知细节

```python
# ① 梯度是累加的！不清零会越加越大
optimizer.zero_grad()   # 所以每轮必须先清零

# ② 推理时不需要梯度，能省显存提速
with torch.no_grad():
    outputs = model(images)
```

> 为什么设计成累加？因为有些高级场景（比如超大 batch 拆成几次算）需要故意累积。默认累加，用的时候自己清。

---

## 3. 梯度下降

有了梯度，就往**梯度的反方向**挪一小步：

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L$$

- $\theta$：模型参数
- $\eta$：学习率 `lr`，步子迈多大
- $\nabla_\theta L$：损失对参数的梯度，指向"损失上升最快的方向"

**所以往反方向走。** $\eta$ 太大 → 震荡飞出去；太小 → 走得慢。

三种批量方式：

| 方式 | batch 大小 | 特点 |
|---|---|---|
| 批量梯度下降 BGD | 全部数据 | 稳，但慢且费显存 |
| 随机梯度下降 SGD | 1 条 | 快，但抖得厉害 |
| **小批量 Mini-batch** | 32/64/128 | **实际都用这个**，兼顾稳定和速度 |

---

## 4. nn.Module —— 零件都装在它里面

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()              # 必须！初始化父类
        self.fc = nn.Linear(784, 10)    # 声明"有哪些参数"

    def forward(self, x):               # 声明"数据怎么流过去"
        return self.fc(x)

model = Net()
model.to("cuda")        # 把所有参数搬到显卡
model.train()           # 训练模式（Dropout/BatchNorm 行为不同）
model.eval()            # 评估模式（必须切！否则结果不稳定）
```

**两个模式的区别**一定要懂：

- `Dropout`：训练时随机丢神经元，评估时全部保留（并缩放）
- `BatchNorm`：训练时用当前 batch 的均值方差，评估时用训练期累计的滑动平均

忘了 `model.eval()` 是最常见的"训练准确率 99%，测试只有 60%"的原因之一。

---

## 5. 损失函数

衡量"预测和答案差多远"，是反向传播的起点。

| 任务 | 损失函数 | 说明 |
|---|---|---|
| 分类 | `nn.CrossEntropyLoss` | **内部自带 Softmax**，所以模型输出直接给 logits（原始分数），不要再加 Softmax！ |
| 回归 | `nn.MSELoss` | 均方误差 |
| 检测 | `CIoU Loss` + `BCE` | YOLO 内部已封装好 |

Softmax 把原始分数变成概率：

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

交叉熵：

$$L = -\sum_i y_i \log(\hat{p}_i)$$

$y$ 是 one-hot 真实标签，$\hat p$ 是预测概率。预测越接近真实，损失越小。

---

## 6. 优化器

`optimizer` 负责执行 `参数 = 参数 - lr * 梯度` 这一步。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

| 优化器 | 特点 |
|---|---|
| `SGD` + momentum | 经典，泛化有时更好，需要调 lr |
| `Adam` | 自适应学习率，收敛快，**新手首选** |
| `AdamW` | Adam + 正确的权重衰减，现在的主流默认 |

> 一个铁律：**建优化器时要传入 `model.parameters()`**，否则参数根本不会被更新。

配套的**学习率调度器**（可选项）：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

前期用大 lr 快速下降，后期用小 lr 精修。就像找路：先大步走，接近目标再小步挪。

---

## 7. 卷积 —— 图像为什么不用全连接

一张 `224×224×3` 的图展平是 150528 个数。接一个 1000 神经元的全连接层 → **1.5 亿个参数**。还没算多层。

卷积用两个假设把这个数量砍掉：

1. **局部性**：相邻像素才有关系，不用看全图 → 卷积核只看 $k\times k$ 小窗口
2. **平移不变性**：猫在左上角和右下角都是猫 → **同一个卷积核滑过整张图**（参数共享）

### 7.1 卷积核在干嘛

一个 $3\times3$ 卷积核就是 9 个数，在图上滑动，每次和窗口内的像素做**点积**：

$$\text{输出}(i,j) = \sum_{m}\sum_{n} \text{输入}(i+m,\ j+n) \cdot \text{核}(m,n) + b$$

不同的核 = 不同的特征提取器：边缘、角点、纹理……

**关键点：这些核的值不是人手工设计的，是训练学出来的。**

### 7.2 输出尺寸公式（面试和调参必背）

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1$$

- $K$：卷积核大小 `kernel_size`
- $P$：边缘补零 `padding`
- $S$：步长 `stride`

例：$32$ 输入，$K=3, P=1, S=1$ → $(32 + 2 - 3)/1 + 1 = 32$。**`padding=1, k=3` 尺寸不变**，这是最常用的组合。

### 7.3 一个经典 CNN 的结构节奏

```
卷积 → 激活(ReLU) → 池化 →  尺寸减半，通道翻倍
卷积 → 激活(ReLU) → 池化 →  再减半
...
展平 → 全连接 → 输出类别数
```

- **ReLU**：$f(x)=\max(0,x)$。引入非线性，否则多层线性叠加还是线性的，等于白搭。
- **池化 MaxPool**：取窗口内最大值，尺寸变小 → 感受野变大 → 参数变少。

**随着网络变深：空间尺寸越来越小，通道数越来越多。**
直观理解：从"看像素" → "看边缘" → "看纹理" → "看物体部件" → "看整体是什么"。

---

## 8. 训练循环：五步曲

```python
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)  # 0. 搬数据到 GPU

    optimizer.zero_grad()                    # 1. 清空上轮梯度
    outputs = model(images)                  # 2. 前向：算预测
    loss = criterion(outputs, labels)        # 3. 算损失
    loss.backward()                          # 4. 反向：算梯度
    optimizer.step()                         # 5. 更新参数
```

背下来。这几行的顺序就是深度学习的全部。

验证阶段要用完全不同的写法：

```python
model.eval()                          # 切评估模式
with torch.no_grad():                 # 关梯度，省显存
    for images, labels in val_loader:
        outputs = model(images)
        ...
```

---

## 9. 过拟合：训练集表现好，测试集拉胯

模型把训练数据"背下来"了，而不是学到规律。

**判断方法**：训练 loss 一直降，验证 loss 先降后**升** → 拐点处就是过拟合开始。

**解药**（按性价比排序）：

| 方法 | 做法 | 原理 |
|---|---|---|
| **数据增强** | 随机裁剪、翻转、变色 | 让模型见不到两张一样的图，被迫学不变特征 |
| **Early Stopping** | 验证 loss 连续 N 轮不降就停 | 停在拐点前 |
| **Dropout** | 训练时随机丢神经元 | 防止神经元之间"串通作弊" |
| **权重衰减** | `weight_decay=1e-4` | 惩罚过大的权重 |
| **减小模型** | 少几层、少几个通道 | 容量不够就没法硬背 |
| **更多数据** | 最有效但最贵 | — |

> 数据量小时，**数据增强是性价比最高的一招**。`dataset.py` 里已经写好了。

---

## 10. 评估指标的坑

**准确率 Accuracy 会骗人。**

如果 100 张图里 95 张是"其他垃圾"，模型全部猜"其他"，准确率 95%，但它什么都没学会。

所以要看**混淆矩阵**和分类指标：

- **精确率 Precision** = 预测为 A 的里面，真有 A 的比例 → $\frac{TP}{TP+FP}$（"我说是 A，有几个是真 A"）
- **召回率 Recall** = 真的 A 里面，被找出来的比例 → $\frac{TP}{TP+FN}$（"真 A 我漏了几个"）
- **F1** = 两者的调和平均 → $\frac{2PR}{P+R}$
- **mAP@0.5** = 检测任务标配，IoU > 0.5 才算检测对

**IoU（交并比）**：预测框和真实框的交集面积 ÷ 并集面积。

$$\text{IoU} = \frac{S_{\text{交}}}{S_{\text{并}}}$$

---

## 11. 迁移学习：高中生做项目的最强外挂

自己从零训练一个模型需要几万张图 + 几天 GPU。**没必要。**

大模型在 ImageNet 上学会了"边缘、纹理、形状"这些通用特征。你要做的只是：

```python
from torchvision import models
import torch.nn as nn

model = models.resnet18(weights="IMAGENET1K_V1")   # 拿别人训练好的
for p in model.parameters():
    p.requires_grad = False                        # 冻结：这部分不许动
model.fc = nn.Linear(model.fc.in_features, 4)      # 换成自己的类别数
```

**只用几百张图就能出效果。** 这就是为什么你的项目完全可行。

三种策略：

1. **只训练头**：冻结全部主干，只训最后的分类层（数据极少时用）
2. **微调**：主干用小学习率，头用大学习率（推荐）
3. **从头训练**：只在有海量数据时用

> 你的项目路线：先用 CIFAR-10 把训练流程跑通 → 换成迁移学习 → 再上 YOLOv8。

---

## 12. 从分类到检测（你架构图的最终形态）

分类 = "这张图是什么"，输出一个标签。

**检测 = "图里有什么 + 在哪"，输出一堆框。**

YOLO 的思路（You Only Look Once）：

1. 把图切成 $S\times S$ 的网格
2. 每个网格预测：有没有物体（置信度）+ 框的位置（4 个数）+ 类别概率
3. 用 **NMS 非极大值抑制** 去掉重叠的重复框

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")          # n/s/m/l/x，越大越准越慢
results = model.train(data="data.yaml", epochs=100, imgsz=640)
```

YOLO 把损失函数、数据加载、增强、后处理**全部封装好了**，所以核心代码只有几行。
但**前提是你得懂上面这 12 节**，否则报错了完全不知道怎么调。

---

## 学习路线（建议按顺序）

| 阶段 | 做什么 | 对应文件 |
|---|---|---|
| 1 | 张量、自动求导 | `lessons/01_tensor_autograd.py` |
| 2 | 手写梯度下降，理解优化 | `lessons/02_gradient_descent.py` |
| 3 | 卷积到底在算什么 | `lessons/03_convolution.py` |
| 4 | 看懂网络结构 | `model.py` |
| 5 | 完整训练循环 ← **最重要** | `train.py` |
| 6 | 推理与置信度 | `predict.py` |
| 7 | 换真实数据集（CIFAR-10） | `dataset.py` |
| 8 | 换成自己的任务 + YOLOv8 | 待定 |

**先跑 4~6 步，整个 PyTorch 就通了。**
