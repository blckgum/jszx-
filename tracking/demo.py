"""生成可复现的功能样例；镜像、平移和黑帧不代表真实校园交通运动。"""

# 原理笔记：docs/modules/02_多目标跟踪_ByteTrack_解释文档.md，第 8.2、9 节。
# 两个黑帧检查“没有检测仍推进一帧”；相同内容的第二段检查“换视频重置会话”。

import argparse
import importlib.util
import shutil
from pathlib import Path

import cv2
import numpy as np

ULTRALYTICS_ROOT = Path(importlib.util.find_spec("ultralytics").origin).parent
BUS_IMAGE = ULTRALYTICS_ROOT / "assets" / "bus.jpg"


def make_sample_video(destination, frames=18):
    if type(frames) is not int or frames <= 0:
        raise ValueError("frames 必须为正整数")
    image = cv2.imread(str(BUS_IMAGE))
    if image is None:
        raise ValueError(f"缺少 Ultralytics 随包样例图片：{BUS_IMAGE}")
    image = cv2.resize(image, (405, 540))
    scene = np.concatenate((image, cv2.flip(image, 1)), axis=1)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (810, 540))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("无法创建构造视频，请检查目录和 MP4 编码器")
    try:
        for index in range(frames):
            # 仿射矩阵产生每帧一像素的横移；第 8、9 帧置黑，用于测试空帧推进与恢复。
            matrix = np.float32([[1, 0, index], [0, 1, 0]])
            frame = cv2.warpAffine(scene, matrix, (810, 540))
            if index in (8, 9):
                frame[:] = 0
            writer.write(frame)
    finally:
        writer.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成非实拍的 ByteTrack 功能检查视频，不下载数据集")
    parser.add_argument("--output", required=True, help="尚不存在的样例目录")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
        make_sample_video(output / "sample_a.mp4")
        # 两份相同内容用于验证新会话重置；不把它们描述为两个独立场景。
        shutil.copyfile(output / "sample_a.mp4", output / "sample_b.mp4")
        (output / "样例说明.txt").write_text(
            "Ultralytics bus.jpg 的镜像、拼接、平移和空白帧构造样例。\n"
            "两段视频内容相同，每段 18 帧、10 FPS、810×540，仅用于功能检查。\n"
            "没有真实速度、连续身份真值或校园场景评价意义。\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"生成失败：{exc}\n")
    print(f"构造样例已保存：{output}")


if __name__ == "__main__":
    main()
