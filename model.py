# 原理笔记：docs/modules/03_图像分类实验_原理笔记.md。
# 本文件属于原有分类学习模块；校园道路检测与跟踪使用 detection/、tracking/ 入口。

"""
model.py —— 网络结构

这个文件回答一个问题：**数据从输入到输出，中间经过了什么？**

我们把一个卷积神经网络拆成"积木"，每块积木都用注释标明了它做什么、
以及张量的形状怎么变。看懂形状变化，就看懂 CNN 了。

────────────────────────────────────────────────────────
整个网络的数据流（以 CIFAR-10 为例，图片是 3x32x32）：

    输入  [N, 3, 32, 32]
      │
      ├─ conv_block(3 → 32)    提取低级特征：边缘、颜色块
      │                        [N, 32, 32, 32]
      ├─ MaxPool(2)            尺寸减半
      │                        [N, 32, 16, 16]
      │
      ├─ conv_block(32 → 64)   提取中级特征：纹理、简单形状
      │                        [N, 64, 16, 16]
      ├─ MaxPool(2)
      │                        [N, 64, 8, 8]
      │
      ├─ conv_block(64 → 128)  提取高级特征：物体部件
      │                        [N, 128, 8, 8]
      ├─ MaxPool(2)
      │                        [N, 128, 4, 4]
      │
      ├─ AdaptiveAvgPool       每个通道压成 1 个数
      │                        [N, 128, 1, 1] → [N, 128]
      │
      └─ Linear(128 → 10)      输出 10 个类别的原始分数
                               [N, 10]

记住这个节奏：**空间尺寸越来越小，通道数越来越多。**
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


# ============================================================
# 积木 1：一个标准的卷积块
# ============================================================

class ConvBlock(nn.Module):
    """
    最常用的卷积组合：Conv → BatchNorm → ReLU

    这三个为什么要绑在一起用？
      - Conv     : 提取特征（核心运算）
      - BatchNorm: 把这一层的输出拉回标准分布，让训练更稳、可以用更大的学习率
      - ReLU     : 加非线性，让网络有能力拟合复杂函数

    ★ 为什么卷积层默认不加 bias（偏置）？
      因为紧跟着 BatchNorm 会减去均值，
      常数的偏置会被直接减掉，加了也是白加、浪费参数。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,  # 用多少个卷积核 → 输出多少张特征图
            kernel_size=3,              # 3x3 的卷积核，最常用的尺寸
            stride=1,                   # 步长 1：一个一个像素地滑
            padding=1,                  # ★ 补 1 圈 0，让输出尺寸和输入一样
                                        #   公式 (32 + 2*1 - 3)/1 + 1 = 32
            bias=False,                 # 见上面注释
        )

        # BatchNorm2d 的参数是"通道数"，它对每个通道单独做标准化
        self.bn = nn.BatchNorm2d(out_channels)

        # ReLU: f(x) = max(0, x)，负数变 0，正数不变
        # inplace=True 表示直接在原张量上改，省一点显存
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 数据依次流过三个层，形状不变（因为 padding=1, kernel=3）
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ============================================================
# 积木 2：整个网络
# ============================================================

class SimpleCNN(nn.Module):
    """
    一个够简单、但又真的能用的 CNN。

    为什么不用现成的 ResNet？
      因为我们要看懂原理。这个网络每个零件都清清楚楚，
      38 万参数，CIFAR-10 上跑几十轮就能到 85% 左右的准确率。
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        in_channels: int = config.IN_CHANNELS,
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            num_classes: 分类数。CIFAR-10 是 10；垃圾四分类改成 4。
            in_channels: 输入图片通道数。RGB 是 3，灰度是 1。
            dropout:     Dropout 概率，训练时随机丢掉多少比例的神经元。
                        0 表示不用。数据少的时候调大，比如 0.5。
        """
        super().__init__()  # ★ 必须调用！否则 nn.Module 的内部机制不会初始化

        # ---------- 特征提取部分（也叫 backbone / 主干）----------
        # nn.Sequential 就是"按顺序执行"的容器，代码更干净
        # 每阶段实际有两个 ConvBlock；文件开头的数据流图按阶段概括。
        self.features = nn.Sequential(
            # 第 1 阶段：[N,3,32,32] → [N,32,32,32] → 池化 → [N,32,16,16]
            ConvBlock(in_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 2x2 窗口取最大，尺寸减半

            # 第 2 阶段：[N,32,16,16] → [N,64,16,16] → 池化 → [N,64,8,8]
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第 3 阶段：[N,64,8,8] → [N,128,8,8] → 池化 → [N,128,4,4]
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ---------- 分类部分（也叫 head / 头）----------
        self.classifier = nn.Sequential(
            # ★ AdaptiveAvgPool2d(1)：不管输入是 4x4 还是 8x8，
            #   都对每个通道求平均，输出固定是 1x1。
            #   好处：换成不同尺寸的输入图片也不会报错。
            #   形状：[N,128,4,4] → [N,128,1,1]
            nn.AdaptiveAvgPool2d(output_size=1),

            # Flatten：把 [N,128,1,1] 压平成 [N,128]，才能接全连接层
            nn.Flatten(),

            # Dropout：训练时随机把 30% 的神经元输出置 0。
            # ★ 这是防过拟合的手段：逼网络不要依赖某几个特定神经元。
            #   注意 eval() 模式下它会自动失效，不需要手动关。
            nn.Dropout(p=dropout),

            # 全连接层：把 128 维特征映射到 num_classes 个类别的分数
            nn.Linear(128, num_classes),
        )

        # ★ 权重初始化：好的初始值能让训练更快收敛
        self._init_weights()

    def _init_weights(self) -> None:
        """
        参数初始化。

        卷积层用 Kaiming 初始化（He 初始化）：
          它考虑了 ReLU 会把一半的神经元置 0，
          所以故意把初始权重放大一点，保证信号能传得下去。

        为什么初始化重要？
          如果初始权重全为 0，所有神经元的梯度都一样，
          它们会一直保持相同 → 网络退化成一层，学不到东西。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # mode="fan_out" 表示按输出通道数缩放
                # nonlinearity="relu" 告诉它后面接的是 ReLU
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)   # 缩放系数初始为 1
                nn.init.constant_(m.bias, 0.0)     # 平移系数初始为 0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # 小方差的随机值
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义"数据怎么流过网络"。PyTorch 会在调用 model(x) 时自动执行这个函数。

        ★ 输出的 10 个数字叫 logits（原始分数），不是概率！
          因为交叉熵损失函数内部自带 Softmax，重复做会算错。
          想拿概率的话，推理时自己加 softmax，predict.py 里有。
        """
        x = self.features(x)      # 提特征：[N,3,32,32] → [N,128,4,4]
        x = self.classifier(x)    # 分类   ：[N,128,4,4] → [N,10]
        return x


# ============================================================
# 积木 3：迁移学习版本（等基础版跑通了再回来试）
# ============================================================

def build_resnet18(num_classes: int = config.NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """
    用 ImageNet 预训练的 ResNet18 做迁移学习。

    原理：
      ResNet18 在 120 万张图上学会了"边缘、纹理、形状"这些通用视觉特征。
      我们把这些知识直接拿过来，只替换最后一层去适配自己的类别数。
      → 几百张图就能有不错的效果，这是高中生做项目的最强外挂。

    两种用法：
      1. 冻结主干，只训最后的分类层（数据极少时用，训练最快）
      2. 解冻主干，用小学习率一起微调（效果更好，推荐）

    注意：CIFAR-10 只有 32x32，比 ImageNet 的 224x224 小很多。
          ResNet18 在 32x32 上跑，最后特征图只剩 1x1，效果打折。
          如果要用，建议先把图片 resize 到 224，或者用专门为小图设计的网络。
    """
    from torchvision import models

    # weights="IMAGENET1K_V1" 表示加载在 ImageNet 上训练好的权重
    # 第一次运行会自动下载约 45MB（需要联网）
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # ResNet18 原本输出 1000 类（ImageNet 的类别数），换成我们自己的
    in_features = model.fc.in_features      # 512
    # 当前没有冻结主干，训练器传入全部 model.parameters()，默认是全网络微调。
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_model(
    name: str = config.MODEL_NAME,
    num_classes: int = config.NUM_CLASSES,
) -> nn.Module:
    """
    模型工厂：根据名字造出对应的模型。

    这样写的好处：以后想换网络，只改 config.py 里的一行，其他代码都不用动。
    这叫"依赖配置而非硬编码"。
    """
    if name == "SimpleCNN":
        return SimpleCNN(num_classes=num_classes)
    if name == "ResNet18":
        return build_resnet18(num_classes=num_classes, pretrained=config.USE_PRETRAINED)
    raise ValueError(f"未知的模型名: {name}，可选 SimpleCNN / ResNet18")


# ============================================================
# 自测：把形状变化打印出来，直观感受网络在干什么
# ============================================================

if __name__ == "__main__":
    from common import banner, count_parameters, get_device, human_number

    banner("网络结构自测")
    device = get_device()

    model = build_model("SimpleCNN").to(device)
    print(f"\n模型: {config.MODEL_NAME}")
    print(f"可训练参数量: {human_number(count_parameters(model))}"
          f"  (约 {count_parameters(model) * 4 / 1024**2:.1f} MB)")

    # ------- 追踪张量形状 -------
    # 造一批假数据，形状和真实数据一样：[批大小, 通道, 高, 宽]
    dummy = torch.randn(2, 3, 32, 32, device=device)
    print(f"\n{'层':<28}{'输出形状':<24}")
    print("-" * 52)
    print(f"{'输入':<28}{str(tuple(dummy.shape)):<24}")

    x = dummy
    # named_children() 只遍历直接子模块（features 和 classifier）
    for name, layer in model.named_children():
        x = layer(x)     # 把数据真的喂进去看形状
        print(f"{name:<28}{str(tuple(x.shape)):<24}")

    print("-" * 52)
    print(f"{'最终输出':<28}{str(tuple(x.shape)):<24}")
    print(f"\n  → N=2 是批大小，10 是 10 个类别的原始分数(logits)")
    print(f"  → 加上 softmax 就是概率：{torch.softmax(x[0], dim=0).detach().cpu().numpy().round(3)}")

    # ------- 验证 loss 能不能算 -------
    banner("损失函数验证")
    criterion = nn.CrossEntropyLoss()
    labels = torch.tensor([3, 7], device=device)   # 两张图的真实类别
    loss = criterion(x, labels)
    print(f"假数据上的 loss = {loss.item():.4f}")
    print(f"理论值 ln(10) = {torch.log(torch.tensor(10.0)).item():.4f}")
    print("  → 随机初始化的网络，10 类问题 loss 应该接近 ln(10)≈2.303。")
    print("  → 如果一开始 loss 就离谱（比如 100+），说明代码有 bug。")
    print("    这是排查问题的第一个检查点！")
