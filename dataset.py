# 原理笔记：docs/modules/03_图像分类实验_原理笔记.md。
# 本文件属于原有分类学习模块；校园道路检测与跟踪使用 detection/、tracking/ 入口。

"""
dataset.py —— 数据加载与增强

深度学习中有一句话：**数据和特征决定了上限，模型只是逼近这个上限。**

这个文件解决三个问题：
  1. 图片怎么变成模型能吃的张量？
  2. 数据增强是什么，为什么有用？
  3. DataLoader 在背后偷偷干了什么？

────────────────────────────────────────────────────────
核心机制：Dataset 和 DataLoader 的分工

    Dataset  =  回答"第 i 个样本是什么"   → 单个样本，慢，但灵活
    DataLoader = 决定"怎么凑成一批"       → 批处理、打乱、并行预读

  为什么必须分批？
    一次性把 5 万张图喂进去，显存直接爆。
    而且分批（mini-batch）还自带随机性，对训练有好处。

  DataLoader 的加速秘诀：
    当 GPU 在算第 k 批时，CPU 已经在准备第 k+1、k+2 批了。
    这叫"预取"(prefetch)，让 GPU 不会闲着等数据。
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import config
from common import banner, ensure_dir


# ============================================================
# 一、图像变换：原始像素 → 模型能吃的张量
# ============================================================

def build_train_transform() -> transforms.Compose:
    """
    训练用的变换。包含**数据增强**。

    ★★ 数据增强为什么有用？（这是本文件最重要的概念）

    神经网络参数动辄几十万，很容易"背答案"—— 把训练集每张图硬记下来。
    一旦遇到没见过的图就傻眼，这叫过拟合。

    数据增强的思路：
      同一张图，每次读的时候随机变换一下（翻转、裁剪、调色），
      模型永远看不到两张完全一样的图。
      → 它没法硬背，只能去学"猫的本质特征"这种真正通用的东西。

    效果相当于把数据集免费扩充了好几倍。
    而且，翻转/裁剪后的图**标签不变**，所以不需要重新标注！
    """

    return transforms.Compose([
        # ── 增强部分（都是随机操作）──

        # 1. 随机裁剪：先在四周补 padding 像素，再随机抠出 32x32
        #    作用：让模型不依赖物体的绝对位置
        transforms.RandomCrop(
            size=config.AUG_CROP_SIZE,
            padding=config.AUG_CROP_PADDING,
        ),

        # 2. 随机水平翻转
        #    ★ 注意：这里用"水平"是有讲究的。
        #      猫左右翻转还是猫 → 可以用。
        #      但如果识别的是文字、交通标志数字，翻转后语义就变了 → 不能用！
        #      增强手段要和任务匹配，不能无脑加。
        transforms.RandomHorizontalFlip(p=config.AUG_HORIZONTAL_FLIP_P),

        # 3. 随机旋转 ±15 度
        transforms.RandomRotation(degrees=config.AUG_ROTATION_DEGREES),

        # 4. 颜色抖动：亮度、对比度、饱和度随机变化
        #    作用：让模型对光照条件不敏感（拍照环境光线肯定不一样）
        transforms.ColorJitter(
            brightness=config.AUG_COLOR_JITTER,
            contrast=config.AUG_COLOR_JITTER,
            saturation=config.AUG_COLOR_JITTER,
        ),

        # ── 转换部分（确定性操作）──

        # 5. ToTensor：PIL 图片 [H,W,C] 且取值 0-255 (uint8)
        #              → 张量 [C,H,W] 且取值 0.0-1.0 (float32)
        #    ★ 这一步同时完成了维度顺序的转换，是最容易被忽略的关键步骤
        transforms.ToTensor(),

        # 6. Normalize：逐通道做 (x - mean) / std
        #    作用：把数据拉成均值 0、标准差 1 的分布。
        #    为什么？如果输入尺度差异大，梯度会不稳定，训练容易发散。
        #    这三个 mean/std 是统计 CIFAR-10 全体图片算出来的。
        transforms.Normalize(mean=config.CIFAR10_MEAN, std=config.CIFAR10_STD),
    ])


def build_eval_transform() -> transforms.Compose:
    """
    验证/测试用的变换。**没有增强！**

    ★ 为什么评估时不能加增强？
      增强会改变图片内容。评估的目的是衡量"模型在真实数据上表现如何"，
      必须用原始分布的数据。加了随机操作，每次评估结果都不一样，没法比较。
      而且评估需要的是"确定性"——同样的输入必须得到同样的输出。

    所以这里只保留 ToTensor + Normalize 这两个确定性的转换。
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.CIFAR10_MEAN, std=config.CIFAR10_STD),
    ])


# ============================================================
# 二、离线备用数据集（没网络时用）
# ============================================================

class SyntheticDataset(Dataset):
    """
    人造数据集：不需要下载，完全离线，用来验证代码是否跑得通。

    ★ 为什么需要它？
      CIFAR-10 要下载 170MB，如果你网络不好或者想快速测试代码改动，
      用它可以在几秒内跑完整个训练流程。

    规则设计得很巧妙（这是一道"能学会的题"）：
      每张图是 3x32x32 的随机噪声，但我们偷偷在左上角
      (0:8, 0:8) 区域塞进一个均匀的色块，色块的通道决定类别。
      → 网络必须学会"看那个角落的颜色"，才能答对。
      → 如果准确率能到 100%，说明前向传播、反向传播、优化器全都正常。

    这比用纯随机标签强得多：随机标签是学不会的（准确率只能到 10%），
    那样就分不清"代码有问题"还是"这题本身无解"了。
    """

    def __init__(self, size: int = 5000, num_classes: int = config.NUM_CLASSES,
                 transform=None) -> None:
        self.size = size
        self.num_classes = num_classes
        self.transform = transform

        # 固定随机种子，保证每次生成的数据一样
        g = torch.Generator().manual_seed(config.SEED)

        # 随机噪声作为底图
        self.images = torch.rand(size, 3, 32, 32, generator=g)

        # 标签：0 ~ num_classes-1 轮流出现，保证类别均衡
        self.labels = torch.arange(size) % num_classes

        # 在左上角 8x8 区域画色块：第 k 类 → 第 k%3 个通道亮起，亮度随类别递增
        # 这样每一类的"暗号"都不一样，但信息量足够小，不会一眼看穿
        for i in range(size):
            c = int(self.labels[i].item())
            channel = c % 3                       # 用哪个通道
            brightness = 0.5 + 0.5 * (c // 3) / max(1, (num_classes - 1) // 3)
            self.images[i, channel, 0:8, 0:8] = brightness

    def __len__(self) -> int:
        """
        ★ DataLoader 靠这个知道数据集有多大。
          必须返回 int，返回别的类型会报奇怪的错。
        """
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        ★ 最关键的方法：返回第 index 个样本，格式是 (图片张量, 标签)。

        这个方法会被 DataLoader 在多个进程/线程里**同时调用**，
        所以里面不要有共享状态、不要写文件、不要 print（会刷屏）。

        注意：这里返回的标签是一个 Python 整数（不是张量）。
              collate 函数会自动把它们堆成 [N] 的张量。
        """
        img = self.images[index]                       # [3, 32, 32]
        label = int(self.labels[index].item())         # 0~9 的整数

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# ============================================================
# 三、构建数据加载器
# ============================================================

def build_dataloaders(
    use_synthetic: bool = False,
    batch_size: int = config.BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    一次性把 训练/验证/测试 三个 DataLoader 都建好。

    返回: (train_loader, val_loader, test_loader, class_names)

    ★ 严格区分三个数据集（这是做课题必须讲清楚的）：
       训练集 train      : 用来更新参数。模型"看"它。
       验证集 val        : 调超参数、选最佳模型。模型"不学"它，但我们"看"它。
       测试集 test       : 最终报告成绩用。★ 全程只许看一次！

    ★★ 为什么测试集只能看一次？
       如果你反复用测试集调参，等于变相"偷看答案"，
       最后报出来的准确率是虚高的，不能反映真实性能。
       学术上叫"数据泄漏"(data leakage)，是严重的错误。

       所以：调参看验证集，最后毕业论文上的数字用测试集。
    """
    # ---------- 准备变换 ----------
    train_tf = build_train_transform()   # 带数据增强，只给训练集用
    eval_tf = build_eval_transform()     # 不加增强，给验证集和测试集用

    # ---------- 准备数据集 ----------
    if use_synthetic:
        banner("使用合成数据集（离线调试模式）")

        # ★ 注意这里 transform=None 的原因（这是个真实的坑）：
        #   SyntheticDataset 在 __init__ 里生成的就是 torch 张量了，
        #   而 ToTensor() 只接受 PIL 图片或 numpy 数组，
        #   对张量调用它会报 "pic should be PIL Image or ndarray"。
        #   所以这里跳过整个变换流程。
        #
        #   真实数据集（CIFAR-10）读进来是 PIL 图片，就必须走变换流程。
        full = SyntheticDataset(size=5000, transform=None)

        # 6:2:2 划分
        n_total = len(full)
        n_train = int(n_total * 0.6)
        n_val = int(n_total * 0.2)
        n_test = n_total - n_train - n_val

        # random_split 用固定种子，保证每次划分一致
        generator = torch.Generator().manual_seed(config.SEED)
        train_set, val_set, test_set = random_split(
            full, [n_train, n_val, n_test], generator=generator
        )

        class_names = [f"class_{i}" for i in range(config.NUM_CLASSES)]

    else:
        banner("加载 CIFAR-10 数据集")
        from torchvision.datasets import CIFAR10

        data_root = ensure_dir(config.DATA_DIR)

        # ★ 同一个目录加载两次，但变换不同：
        #   训练集用带增强的 transform，验证/测试集用不带增强的。
        #   这是标准做法。
        try:
            # download=True 会在第一次运行时自动下载约 170MB
            train_full = CIFAR10(root=str(data_root), train=True,
                                 download=True, transform=train_tf)
            test_set = CIFAR10(root=str(data_root), train=False,
                               download=True, transform=eval_tf)
        except Exception as e:  # noqa: BLE001
            print(f"\n[错误] 下载 CIFAR-10 失败: {e}")
            print("解决办法（三选一）：")
            print("  1. 检查网络，重新运行")
            print("  2. 加参数 --fake 使用离线合成数据（推荐先这样验证代码）")
            print("  3. 手动下载后放到 data/cifar-10-batches-py/ 目录")
            raise

        # ★ 从训练集里再切出 10% 做验证集
        #   注意：验证集要用 eval_tf（无增强），所以不能直接从 train_full 切。
        #   下面重新加载训练数据的无增强视图；不从官方测试集借样本。
        n_total = len(train_full)
        n_val = int(n_total * 0.1)
        n_train = n_total - n_val

        # 用一个不带增强的 CIFAR10 实例，只为拿到干净的图片
        # （transform 在 Dataset 上，不在样本上，所以要重新建一个）
        clean_train = CIFAR10(root=str(data_root), train=True,
                              download=False, transform=eval_tf)

        generator = torch.Generator().manual_seed(config.SEED)
        # 先划出下标，两个数据集共用同一套下标，保证不重叠
        indices = torch.randperm(n_total, generator=generator).tolist()
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        # 用 Subset 按下标取子集
        from torch.utils.data import Subset
        train_set = Subset(train_full, train_idx)   # 带增强
        val_set = Subset(clean_train, val_idx)      # 不带增强

        class_names = config.CIFAR10_CLASSES
        print(f"训练集: {len(train_set)} 张 (带数据增强)")
        print(f"验证集: {len(val_set)} 张 (无增强)")
        print(f"测试集: {len(test_set)} 张 (无增强，最终评估用)")

    # ---------- 包装成 DataLoader ----------
    # 教学默认 num_workers=0，减少进程启动配置；Windows 多进程并非不可用，
    # 但需主入口保护与可序列化对象。合成数据也强制主进程加载，便于离线检查。
    num_workers = 0 if use_synthetic else config.NUM_WORKERS

    # pin_memory: 把数据先放进"锁页内存"，传到 GPU 时更快。
    # 只有用 GPU 训练时才需要开。
    use_cuda = torch.cuda.is_available()

    # 公共参数：为了不重复写三遍
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        # persistent_workers 要求 num_workers > 0，否则报错
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,      # ★ 训练集必须打乱！
                           #   否则每轮都是同样的顺序，模型会学到"顺序"这个无关信息，
                           #   而且连续读到同一类会导致梯度方向偏斜。
        drop_last=True,    # 丢掉最后不满的批次。
                           # 避免末批过小影响批统计；BatchNorm2d 是否报错还取决于空间尺寸。
                           # 注意 batch_size 大于训练集长度时，drop_last 会使整轮没有批次。
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,     # ★ 验证集不需要打乱，结果要稳定
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **common_kwargs,
    )

    return train_loader, val_loader, test_loader, class_names


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    把标准化过的图还原成能看的像素值。

    ★ 这个函数是给"人看"的：
      模型吃的是 Normalize 之后的负数和小数，
      直接存成图片会是一片黑或者花屏。
      所以可视化之前必须反标准化：
        x_原 = x_标准化 × std + mean
    """
    mean = torch.tensor(config.CIFAR10_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config.CIFAR10_STD, dtype=tensor.dtype).view(3, 1, 1)

    # 只有 Normalize 过的图才需要还原；合成数据没标准化，直接跳过
    if not hasattr(tensor, "shape") or tensor.shape[0] != 3:
        return tensor

    out = tensor * std + mean
    # 裁剪到合法范围 [0,1]，防止显示异常
    return out.clamp(0.0, 1.0)


# ============================================================
# 自测：看看数据长什么样
# ============================================================

if __name__ == "__main__":
    banner("数据集自测")

    # 先用合成数据（不联网，秒出结果）
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        use_synthetic=True, batch_size=8
    )

    images, labels = next(iter(train_loader))   # 取一个批次看看

    print(f"\n批次图片形状 : {tuple(images.shape)}")
    print("  解读: [N=8, C=3, H=32, W=32]"
          "\n        8 张图 / 3 个颜色通道 / 32 高 / 32 宽")
    print(f"批次标签形状 : {tuple(labels.shape)}")
    print(f"标签内容     : {labels.tolist()}")
    print(f"图片数值范围 : [{images.min():.3f}, {images.max():.3f}]")
    print(f"图片 dtype   : {images.dtype}")
    print("  ★ dtype 必须是 float32。如果是 uint8，网络会报类型错误。")
    print(f"\n类别名       : {class_names}")
    print(f"批次数量     : 训练 {len(train_loader)} / 验证 {len(val_loader)}"
          f" / 测试 {len(test_loader)}")

    # 验证 DataLoader 的 shuffle 真的生效了
    banner("验证训练集是否打乱")
    b1 = next(iter(train_loader))[1].tolist()
    b2 = next(iter(train_loader))[1].tolist()
    print(f"第一次取到的标签: {b1}")
    print(f"第二次取到的标签: {b2}")
    print(f"两次是否不同: {b1 != b2}")
    print("  ★ 训练集应该每次都不一样(shuffle=True)；")
    print("    验证集应该每次都一样(shuffle=False)。")
