# 原理笔记：docs/modules/03_图像分类实验_原理笔记.md。
# 本文件属于原有分类学习模块；校园道路检测与跟踪使用 detection/、tracking/ 入口。

"""
predict.py —— 推理（用训练好的模型识别新图片）

这个文件回答：**训练完之后怎么用？**

────────────────────────────────────────────────────────
推理和训练的区别：

    训练                          推理
    ─────────────────────         ─────────────────────
    model.train()                 model.eval()
    要算梯度                       用 torch.no_grad() 关掉梯度
    会更新参数                     绝不更新参数
    处理整个数据集                 处理一张几张图

────────────────────────────────────────────────────────
关键概念：logits 和 置信度(confidence) 的区别

    模型最后一层输出的是 logits（原始分数），比如 [2.1, -0.5, 8.3]
        → 可能是任意实数，可正可负，加起来不等于 1

    要变成"能解释的百分比"，必须过一个 Softmax：
        → [0.0008, 0.0001, 0.9991]
        → 加起来等于 1，这才是"置信度"

    训练时不能自己加 Softmax（CrossEntropyLoss 内部已经做了一次），
    但推理时必须自己加，否则输出没法解读。

────────────────────────────────────────────────────────
怎么运行：

    # 单张图片
    python predict.py --image samples/cat.jpg

    # 整个文件夹
    python predict.py --folder samples/

    # 用合成数据快速测试
    python predict.py --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

import config
from common import banner, get_device, human_number
from model import build_model

# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ============================================================
# 一、加载模型
# ============================================================

def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """
    从检查点文件加载模型。

    ★ 为什么不能只存 state_dict（参数）？
      因为还需要知道：
        - 用的哪个模型结构（SimpleCNN 还是 ResNet18）
        - 有几个类别，类别名是什么
      这些信息都打包在检查点字典里了。

    返回: (模型, 类别名列表)
    """
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件: {ckpt_path}\n"
            f"请先运行训练: python train.py --fake --epochs 5"
        )

    # map_location=device 让保存时在 GPU 上的权重也能加载到 CPU
    # weights_only=False 才能加载我们自己存的字典（PyTorch 2.6+ 要求）
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 从检查点里恢复元信息
    model_name = ckpt.get("model_name", config.MODEL_NAME)
    class_names = ckpt.get("class_names", config.CIFAR10_CLASSES)
    num_classes = len(class_names)

    print(f"检查点      : {ckpt_path}")
    print(f"训练轮数    : 第 {ckpt.get('epoch', '?')} 轮")
    print(f"验证准确率  : {ckpt.get('val_acc', float('nan')):.2f}%")
    print(f"模型结构    : {model_name}")
    print(f"类别数      : {num_classes} {class_names}")

    # 造出同样结构的模型，再把参数灌进去
    model = build_model(model_name, num_classes=num_classes)

    # ★ load_state_dict 是把保存的参数"填"进模型结构里。
    #   如果结构对不上会报错（比如类别数不一致），这是好事——能及早发现 bug。
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # ★★ 必须切换到评估模式！
    #    否则 Dropout 会随机丢掉神经元，同一个输入每次预测结果都不一样。
    #    这是新手最常见的 bug："模型每次预测结果都不同！"
    model.eval()

    return model, class_names


# ============================================================
# 二、预处理：把任意图片变成模型要的格式
# ============================================================

def build_inference_transform() -> transforms.Compose:
    """
    推理时的预处理。

    ★ 必须和训练时的"评估变换"完全一致！
      训练时图片被缩放到 32x32 并标准化了，
      推理时如果忘了做同样的处理，模型看到的数据分布完全不同 → 结果全错。

      这个坑叫 "train-test skew"（训练-推理不一致），
      实际项目里非常常见，一定要小心。
    """
    return transforms.Compose([
        # 统一尺寸（不管原图多大都缩成 32x32）
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.CIFAR10_MEAN, std=config.CIFAR10_STD),
    ])


# ============================================================
# 三、核心：预测一张图
# ============================================================

# ============================================================
# 三、核心：预测一张图
# ============================================================

@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image: str | Path | "torch.Tensor",
    class_names: list[str],
    device: torch.device,
    top_k: int = config.TOP_K,
) -> list[tuple[str, float]]:
    """
    预测一张图片，返回 [(类别名, 置信度), ...] 按置信度从高到低排序。

    Args:
        image: 图片路径，或者已经读好的张量
        top_k: 返回前几个可能的类别

    返回示例: [('cat', 0.9231), ('dog', 0.0612), ('deer', 0.0081)]
    """
    transform = build_inference_transform()

    # ---------- 1. 读图并变成张量 ----------
    if isinstance(image, (str, Path)):
        from PIL import Image

        # ★ convert("RGB") 很重要：
        #   有些图是灰度图（1通道）、有些带透明通道（RGBA，4通道）。
        #   模型的第一个卷积层只接受 3 通道，不转换就会报错。
        img = Image.open(image).convert("RGB")
        tensor = transform(img)            # [3, 32, 32]
    else:
        # 张量输入直接使用；调用方负责尺寸、通道和训练时一致的预处理。
        # --fake 数据训练时跳过标准化；任意真实图片路径并不与该合成分布一致。
        tensor = image                     # 已经是张量了

    # ---------- 2. 加一个"批次"维度 ----------
    # 模型要求输入是 [N, C, H, W]，但单张图是 [C, H, W]。
    # unsqueeze(0) 在第 0 维插入一个长度为 1 的维度 → [1, 3, 32, 32]
    # ★ 这是极高频的操作，几乎每个推理代码里都有
    batch = tensor.unsqueeze(0).to(device)

    # ---------- 3. 前向传播 ----------
    logits = model(batch)                  # [1, 10]，原始分数

    # ---------- 4. Softmax 变概率 ----------
    # dim=1 表示在"类别"这个维度上做归一化（不是 batch 维度！）
    probabilities = F.softmax(logits, dim=1)   # [1, 10]，每行加起来 = 1

    # 去掉批次维度 → [10]
    probabilities = probabilities.squeeze(0)

    # ---------- 5. 取 Top-K ----------
    # torch.topk 返回值最大的 k 个：数值 + 下标
    k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=k)

    # 搬回 CPU 再转成 Python 数字（GPU 上的张量不能直接 .tolist() 之外的操作）
    top_probs = top_probs.cpu().tolist()
    top_indices = top_indices.cpu().tolist()

    # 把类别下标映射回类别名
    results = [(class_names[i], float(p)) for i, p in zip(top_indices, top_probs)]
    return results


# ============================================================
# 四、可视化：把结果画在图上
# ============================================================

def visualize(
    model: torch.nn.Module,
    image_path: str | Path,
    class_names: list[str],
    device: torch.device,
    save_path: str | Path | None = None,
) -> Path:
    """
    在图片上画出预测结果（类别 + 置信度），保存成新图片。

    ★ 这就是你架构图里"类别 + 置信度 + 可视化结果"那一步。
      检测任务（YOLO）也是同样的原理，只不过还要多画一个矩形框。
    """
    from PIL import Image, ImageDraw, ImageFont

    image_path = Path(image_path)
    original = Image.open(image_path).convert("RGB")

    # —— 先做预测 ——
    results = predict_image(model, image_path, class_names, device, top_k=3)
    top_class, top_conf = results[0]

    # —— 把图放大一点，方便看清 ——
    display_size = 320
    canvas = original.resize((display_size, display_size), Image.LANCZOS)

    # —— 在底部留白，用来写文字 ——
    text_area_height = 70
    final = Image.new("RGB", (display_size, display_size + text_area_height), "white")
    final.paste(canvas, (0, 0))

    draw = ImageDraw.Draw(final)

    # —— 加载字体（找不到就用默认的，中文可能会变成方块）——
    font = None
    for font_name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(font_name, 18)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    # —— 写预测结果 ——
    # 置信度用颜色区分：高置信绿色，中等橙色，低置信红色
    if top_conf >= 0.8:
        color = (0, 150, 0)
    elif top_conf >= config.CONF_THRESHOLD:
        color = (200, 120, 0)
    else:
        color = (200, 0, 0)

    draw.text((8, display_size + 6),
              f"预测: {top_class}  {top_conf * 100:.1f}%",
              fill=color, font=font)

    # —— 写 Top-3 的其余选项 ——
    others = "  ".join(f"{c}:{p * 100:.1f}%" for c, p in results[1:3])
    draw.text((8, display_size + 32), f"其他: {others}",
              fill=(100, 100, 100), font=font)

    # —— 置信度太低要给出警告 ——
    # ★ 这一点在课题里很重要：模型"不知道"的时候，
    #   让它说不确定，比强行给个错误答案好得多
    if top_conf < config.CONF_THRESHOLD:
        draw.text((8, display_size + 52), "[!] 置信度偏低，结果可能不可靠",
                  fill=(200, 0, 0), font=font)

    # —— 保存 ——
    if save_path is None:
        save_path = image_path.parent / f"{image_path.stem}_pred{image_path.suffix}"
    save_path = Path(save_path)

    # 确保输出目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(save_path)

    return save_path


# ============================================================
# 五、批量预测文件夹
# ============================================================

@torch.no_grad()
def predict_folder(
    model: torch.nn.Module,
    folder: str | Path,
    class_names: list[str],
    device: torch.device,
    save_visualization: bool = True,
) -> list[tuple[str, str, float]]:
    """
    预测整个文件夹里的所有图片。

    返回: [(文件名, 预测类别, 置信度), ...]
    """
    folder = Path(folder)

    # 找出所有图片文件（用集合做后缀匹配，比写一堆 or 简洁）
    image_files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"[警告] {folder} 里没找到图片文件")
        print(f"       支持的格式: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        return []

    print(f"\n找到 {len(image_files)} 张图片\n")
    print(f"{'文件名':<32}{'预测类别':<16}{'置信度':>10}")
    print("-" * 58)

    results = []
    for img_path in image_files:
        preds = predict_image(model, img_path, class_names, device, top_k=1)
        cls, conf = preds[0]

        # 置信度低的标个问号，方便一眼看出哪些不可靠
        flag = " [!]" if conf < config.CONF_THRESHOLD else ""
        print(f"{img_path.name:<32}{cls:<16}{conf * 100:>9.2f}%{flag}")

        results.append((img_path.name, cls, conf))

        if save_visualization:
            visualize(model, img_path, class_names, device)

    print("-" * 58)
    print(f"共 {len(results)} 张，平均置信度 "
          f"{sum(r[2] for r in results) / len(results) * 100:.2f}%")

    return results


# ============================================================
# 六、命令行入口
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="推理：识别图片")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--folder", type=str, help="图片文件夹路径")
    parser.add_argument("--ckpt", type=str, default=str(config.BEST_MODEL_PATH),
                        help=f"模型文件（默认 {config.BEST_MODEL_PATH}）")
    parser.add_argument("--top-k", type=int, default=config.TOP_K,
                        help="显示前 K 个可能的类别")
    parser.add_argument("--demo", action="store_true",
                        help="用随机张量演示（不需要图片文件）")
    args = parser.parse_args()

    if not any([args.image, args.folder, args.demo]):
        parser.print_help()
        print("\n提示：还没训练过模型的话，先跑这个：")
        print("  python train.py --fake --epochs 5")
        return

    banner("模型推理")

    device = get_device()
    model, class_names = load_model(args.ckpt, device)

    # ---------- 演示模式 ----------
    if args.demo:
        banner("演示模式：随机输入")
        # 造一张假的图，看看模型怎么反应。
        # 随机输入只演示接口；即使是域外噪声，Softmax 也可能给出很高分数。
        # 分布外数据不保证产生均匀概率，输出不能当作真实分类效果。
        fake_image = torch.randn(3, 32, 32)

        results = predict_image(model, fake_image, class_names, device, top_k=args.top_k)

        print(f"\n{'排名':<6}{'类别':<16}{'置信度':>10}   {'概率条'}")
        print("-" * 62)
        for rank, (cls, conf) in enumerate(results, 1):
            # 用字符画一个简单的条形图，比纯数字直观
            bar = "█" * int(conf * 40)
            print(f"{rank:<6}{cls:<16}{conf * 100:>9.2f}%   {bar}")

        top_cls, top_conf = results[0]

        print(f"\n  → 注意看：模型给出了 {top_conf * 100:.2f}% 的【高置信度】。")
        print("    但这几乎肯定是错的。为什么？")
        print("\n    因为这次训练用的是【合成数据集】——那批图是人造的，")
        print("    规律是'看左上角 8x8 区域的颜色'，跟真实图片毫无关系。")
        print("    所以模型对真实世界的输入会【自信地答错】。")

        print("""
  ★ 这正好演示了两个非常重要的概念：

    1. 域偏移 (domain shift)
       模型只会认它训练时见过的东西。
       训练数据（合成图）和实际使用数据（真实照片）分布不同 → 必然失效。
       → 这就是为什么数据集质量比模型结构更重要。

    2. 高置信度 ≠ 正确
       置信度只反映"模型有多确定"，不反映"它是否正确"。
       模型完全可以在错的时候表现得非常自信。
       → 这是 AI 实际落地时最大的坑之一，也是你课题里可以展开讨论的点。

  → 想看到"正常"的预测结果，请先用真实数据训练：
       python train.py --epochs 30
     然后：
       python predict.py --image 你的图片.jpg""")
        return

    # ---------- 单张图片 ----------
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[错误] 找不到图片: {img_path}")
            return

        banner(f"预测: {img_path.name}")
        results = predict_image(model, img_path, class_names, device, top_k=args.top_k)

        print(f"\n{'排名':<6}{'类别':<16}{'置信度':>10}   {'概率条'}")
        print("-" * 62)
        for rank, (cls, conf) in enumerate(results, 1):
            bar = "█" * int(conf * 40)
            print(f"{rank:<6}{cls:<16}{conf * 100:>9.2f}%   {bar}")

        top_class, top_conf = results[0]
        print(f"\n最终结论: {top_class}  (置信度 {top_conf * 100:.1f}%)")
        if top_conf < config.CONF_THRESHOLD:
            print("[!] 置信度低于阈值，模型不太确定，建议人工复核")

        # 生成可视化图片
        out = visualize(model, img_path, class_names, device)
        print(f"\n可视化结果已保存: {out}")

    # ---------- 整个文件夹 ----------
    if args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"[错误] 找不到文件夹: {folder_path}")
            return

        banner(f"批量预测: {folder_path}")
        predict_folder(model, folder_path, class_names, device)
        print("\n可视化结果已保存在原图旁边（文件名带 _pred 后缀）")


if __name__ == "__main__":
    main()
