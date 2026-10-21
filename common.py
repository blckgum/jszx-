# 原理笔记：docs/modules/03_图像分类实验_原理笔记.md。
# 本文件属于原有分类学习模块；校园道路检测与跟踪使用 detection/、tracking/ 入口。

"""
common.py —— 通用工具

这个文件放"每个脚本都要用"的东西：
  - 选设备（GPU 还是 CPU）
  - 设置随机种子（保证结果可复现）
  - 计时器、参数量统计、打印助手

★ 为什么需要"设置随机种子"？
    神经网络的初始权重是随机的，数据打乱顺序也是随机的。
    如果不固定种子，你每次跑出来的结果都不一样，就没办法判断
    "这次改动到底有没有用"。
    固定种子 = 所有人（包括未来的你）都能复现同样的结果。
"""

from __future__ import annotations  # 让类型注解可以写得更自由（Python 3.7+ 的特性）

import os
import random
import sys
import time
from pathlib import Path


# ============================================================
# 〇、依赖自检：把"看不懂的报错"变成"人话"
# ============================================================
#
# ★ 为什么需要这个？
#   如果误用了系统里那个没装 torch 的 Python（比如 C:\Python314），
#   原本会抛出这样一串东西：
#
#       Traceback (most recent call last):
#         File "D:\...\lessons\01_tensor_autograd.py", line 18, in <module>
#           import torch
#       ModuleNotFoundError: No module named 'torch'
#
#   新手看到这个完全不知道问题在哪、该怎么修。
#   所以这里主动拦下来，直接告诉你三件事：
#     1. 你现在用的是哪个 Python
#     2. 应该用哪个
#     3. 具体怎么切过去
#
#   这叫"防御性编程"：把错误变成可以照着做的操作步骤。


def _print_setup_help(missing: str) -> None:
    """缺少依赖时，打印清楚该怎么修，而不是扔一堆 traceback。"""
    venv_python = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
    script_name = Path(sys.argv[0]).name if sys.argv else "xxx.py"

    print("=" * 64)
    print(f"  [错误] 缺少依赖包：{missing}")
    print("=" * 64)
    print()
    print("  最可能的原因：你现在用的不是本项目的虚拟环境。")
    print()
    print("  当前正在用的 Python：")
    print(f"      {sys.executable}")
    print()
    print("  应该用的 Python：")
    print(f"      {venv_python}")
    print()
    print("  ---- 解决办法（二选一）----")
    print()
    print("  【方法一】让 VS Code 用对环境（推荐，一次搞定）")
    print("      1. 按 Ctrl+Shift+P")
    print("      2. 输入并选择：Python: Select Interpreter")
    print("      3. 选带 (.venv) 字样的那一项")
    print()
    print("  【方法二】在终端里用完整路径运行")
    print(f"      .\\.venv\\Scripts\\python.exe {script_name}")
    print()
    print("  ---- 如果虚拟环境根本不存在，这样重建 ----")
    print("      py -3.13 -m venv .venv")
    print("      .\\.venv\\Scripts\\pip install torch torchvision "
          "--index-url https://download.pytorch.org/whl/cu128")
    print("      .\\.venv\\Scripts\\pip install -r requirements.txt")
    print()


# ---- 真正的导入（出错就给出上面那段提示）----
try:
    import numpy as np
except ModuleNotFoundError as exc:
    _print_setup_help(exc.name or "numpy")
    raise SystemExit(1) from None

try:
    import torch
except ModuleNotFoundError as exc:
    _print_setup_help(exc.name or "torch")
    raise SystemExit(1) from None


# ============================================================
# 〇、关于 Windows 汉字编码（踩过的坑，记在这里避免重犯）
# ============================================================
#
# 【现象】脚本在终端里跑得好好的，一重定向到文件就崩：
#       python train.py > 结果.txt
#       UnicodeEncodeError: 'gbk' codec can't encode character '\xb2'
#
# 【原因】Windows 上 Python 输出文字有两种模式：
#     模式 A 直接输出到终端 → 走 Windows 控制台 API，任何字符都能显示
#     模式 B 输出被重定向     → 退化成系统默认编码，简体中文是 GBK(cp936)
#
# 【教训】本项目所有源码里的字符都必须是 GBK 能表示的。
#     所以代码里只用了 ★  [!]  [OK]  ^2  ==>  这些 GBK 安全符号，
#     没有用 ² ³ ∂ ⚠ ✅ 之类的字符（GBK 里没有它们）。
#
#     保持这个习惯，你的脚本在终端、管道、写文件三种情况下都不会出问题。
#
# 【万一以后加了特殊字符怎么办】
#     在 PowerShell 里先执行下面这行，把控制台编码切换成 UTF-8：
#         chcp 65001
#     或者干脆避免使用它。



# ============================================================
# 一、设备选择
# ============================================================

def get_device(verbose: bool = True) -> torch.device:
    """
    自动挑选可用的计算设备：cuda（NVIDIA 显卡） > mps（苹果 M 芯片） > cpu。

    为什么要挑？
        GPU 有几千个核心，矩阵乘法比 CPU 快几十倍。
        训练神经网络 99% 的时间都在做矩阵乘法，所以能用显卡就用显卡。

    返回 torch.device 对象，之后写 x.to(device) 就能把数据搬过去。
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            # 打印显卡信息，方便确认真的用上了 GPU
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[设备] 使用 GPU: {name}  ({total:.1f} GB 显存)")

            # ★ 这一行专门为你排查 Blackwell(50系) 显卡的兼容性问题准备
            #   PyTorch 编译时支持哪些显卡架构，看这个列表里有没有你的 "sm_xxx"
            arch_list = torch.cuda.get_arch_list()
            print(f"[设备] PyTorch 支持的架构: {', '.join(arch_list)}")
        return device

    if torch.backends.mps.is_available():  # 苹果 M 系列芯片
        if verbose:
            print("[设备] 使用 Apple MPS")
        return torch.device("mps")

    if verbose:
        print("[设备] 只能用 CPU（会很慢，建议检查显卡驱动）")
    return torch.device("cpu")


# ============================================================
# 二、随机种子
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    固定所有随机源，让每次运行结果一致。

    需要固定的随机源有 4 个，少一个就还是会飘：
      1. Python 内置 random
      2. numpy
      3. PyTorch CPU
      4. PyTorch GPU (cudnn)

    ★ 注意：完全确定性（deterministic=True）会让训练变慢，
      所以这里是"尽量固定"，够用即可。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 所有 GPU 都设一遍

    # cudnn 是 NVIDIA 的卷积加速库，它会在多种算法里挑最快的，
    # 挑哪个依赖运行时的测量结果 → 引入随机性。关掉它换取可复现性。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 为后续子进程提供哈希种子；当前进程的哈希随机化不变
    # 哈希种子在进程启动时读取；此设置只影响随后启动并继承环境的 Python 进程。
    # 当前进程的哈希随机化不会因此改变，跨硬件/版本复现还需单独核验。
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# 三、小工具
# ============================================================

class AverageMeter:
    """
    用来记录"平均值"的小工具，比如一个 epoch 里的平均 loss。

    为什么不能直接把每个 batch 的 loss 加起来除以 batch 数？
        因为最后一个 batch 可能不满（比如 1000 张图，batch=64，
        最后一个 batch 只有 40 张），权重不一样，要按样本数加权平均。

    用法：
        meter = AverageMeter()
        meter.update(loss.item(), n=images.size(0))   # n 是本 batch 的样本数
        print(meter.avg)
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0      # 累计总和
        self.count = 0      # 累计样本数
        self.avg = 0.0      # 加权平均
        self.val = 0.0      # 最近一次的值（未平均）

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n     # 按样本数加权累加
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class Timer:
    """
    计时器，用来算"这个 epoch 花了多久"。

    用法：
        t = Timer()
        ...  干活 ...
        print(t.elapsed_str())
    """

    def __init__(self) -> None:
        self.start = time.time()

    def reset(self) -> None:
        self.start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def elapsed_str(self) -> str:
        """把秒数格式化成 '1h23m45s' 这种好读的形式。"""
        total = int(self.elapsed)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数量。

    参数量决定了：
      - 模型有多大（存成文件是 参数数 × 4 字节）
      - 运行有多慢
      - 需要多少数据才训得起来（经验值：参数量 >> 数据量 → 必然过拟合）
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def human_number(n: int) -> str:
    """把 1234567 显示成 '1.23M'，方便阅读。"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def ensure_dir(path: str | Path) -> Path:
    """
    确保目录存在，不存在就创建。返回 Path 对象。

    注意：mkdir 要传 parents=True，才能一次创建多层目录
    （比如 runs/train/exp1，中间几层都不存在也能建）。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def banner(title: str, char: str = "=", width: int = 60) -> None:
    """打印一个醒目的分隔标题，让终端输出好读一些。"""
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)


def check_torch_env() -> None:
    """
    检查 PyTorch 环境是否正常。

    ★ 重点检查 'sm_arch' 是否包含你的显卡架构。
      你的 RTX 5060 是 Blackwell 架构，计算能力 sm_120。
      如果装的是老版本 PyTorch（比如 cu121 的包），
      编译时只支持到 sm_89 / sm_90，跑起来会报：
        "CUDA error: no kernel image is available for execution on the device"
      解决办法：装 cu128 或更高版本的 PyTorch。
    """
    banner("PyTorch 环境自检")
    print(f"PyTorch 版本  : {torch.__version__}")
    print(f"CUDA 是否可用 : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本     : {torch.version.cuda}")
        print(f"显卡          : {torch.cuda.get_device_name(0)}")
        print(f"支持的架构    : {', '.join(torch.cuda.get_arch_list())}")

        # 查询当前显卡的计算能力，比如 RTX 5060 → (12, 0)
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"sm_{major}{minor}"
        print(f"当前显卡架构  : {sm}")

        # ★ 真正的兼容性判断：显卡架构在不在支持列表里
        if sm not in torch.cuda.get_arch_list():
            print(f"\n[警告] PyTorch 没有为 {sm} 编译内核，运行时可能报 'no kernel image'！")
            print("       请改装 CUDA 12.8+ 版本的 PyTorch。")

        # 实测一次矩阵乘法，能跑通才算真的可用
        try:
            a = torch.randn(256, 256, device="cuda")
            b = torch.randn(256, 256, device="cuda")
            c = a @ b
            torch.cuda.synchronize()  # 等待 GPU 真正算完（GPU 是异步的！）
            print(f"实测矩阵乘法  : 通过 (结果形状 {tuple(c.shape)})")
        except Exception as e:  # noqa: BLE001
            print(f"实测矩阵乘法  : 失败 -> {e}")
    else:
        print("\n[提示] 没有检测到可用的 CUDA，只能用 CPU。")
        print("       如果是 NVIDIA 显卡，检查：nvidia-smi 能否正常输出")


# ============================================================
# 直接运行本文件时，做一次环境自检
# ============================================================
if __name__ == "__main__":
    check_torch_env()
