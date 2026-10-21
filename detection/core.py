"""YOLOv8 推理与逐帧导出。坐标始终使用原图像素，供后续跟踪使用。"""

# 图像分类通常回答“整张图属于哪一类”；目标检测回答“图里有哪些目标、各在哪里”。
# 因而，一张图片的检测结果是 0～多个目标，每个目标至少包含类别、分数和矩形框。
#
# 本文件的数据流：
# OpenCV 读图/解码视频 -> YOLOv8 预处理及推理 -> 置信度筛选和 NMS
# -> 原图坐标下的检测框 -> 绘制标注 + 逐帧 JSONL。
# 检测入口不关联跨帧身份；ID 由已实现的 tracking 模块维护，实际速度和 TTC 尚待实现。
# 原理笔记：docs/modules/01_目标检测_YOLOv8与PyTorch_解释文档.md。

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# .pt 是 PyTorch 模型文件。预训练权重已经从大规模数据中学到了视觉特征，
# 因此可以先直接检测，再使用校园道路数据微调，而不必从随机参数开始训练。
DEFAULT_WEIGHTS = ROOT / "checkpoints" / "yolo" / "yolov8n.pt"
# 前六项是 COCO 中的道路类别；vehicle 用于用户自定义的 person/vehicle 二分类模型。
# 对官方 COCO 模型，默认仅选择它实际拥有的六个细分类，不会凭空新增 vehicle 类。
TARGET_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "vehicle")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def prepare_runtime() -> None:
    # 默认缓存位于已忽略的 runs/，同时尊重用户设置的路径。
    # Ultralytics 保存配置，Matplotlib 保存字体等缓存；云机器的用户目录可能只读。
    for name, suffix in (("YOLO_CONFIG_DIR", "ultralytics"), ("MPLCONFIGDIR", "matplotlib")):
        os.environ.setdefault(name, str(ROOT / "runs" / ".cache" / suffix))
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    # 缺少依赖时明确报错，依赖安装统一使用 requirements-yolo.txt。
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def load_model(weights: str | Path):
    prepare_runtime()
    # 在真正需要模型时再导入，查看命令帮助和进行数据审计无需先加载 PyTorch 模型。
    from ultralytics import YOLO

    path = Path(weights).expanduser()
    if not path.is_file():
        # n/s/m/l/x 表示 YOLOv8 的不同规模，通常越大的模型需要越多计算资源。
        # 只有标准模型名允许触发官方下载；自定义 best.pt 缺失时应提示路径错误。
        if path.name not in {f"yolov8{size}.pt" for size in "nslmx"}:
            raise FileNotFoundError(f"模型不存在：{path}；请提供本地检测权重。")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics 从官方 assets release 下载标准 YOLOv8 权重。
    # YOLO 是 Ultralytics 的封装对象，内部包含 PyTorch 网络、类别名称和推理接口。
    model = YOLO(str(path))
    # 同一个库也支持分类、分割、姿态估计；这些任务的输出结构与检测框不同。
    if model.task != "detect":
        raise ValueError("需要目标检测权重，不能使用分类、分割或姿态模型。")
    return model


def select_classes(names: dict[int, str], requested: list[str] | None = None) -> list[int]:
    """按类别名称选择 ID，兼容 COCO 的 80 类及微调后的自定义类别顺序。"""
    # model.names 给出“模型输出 ID -> 语义名称”的对应关系，不能固定套用 COCO ID。
    # 例如 COCO 的 bus 是 5；本项目六类数据的 bus 是 4、truck 才是 5。
    # 按名称找 ID，才能正确使用不同类别数量、不同类别顺序的权重。
    available = {str(name).lower(): int(index) for index, name in names.items()}
    # 未显式指定 --classes 时，选择模型中存在的道路类别；指定后则逐项校验。
    wanted = requested if requested is not None else [n for n in TARGET_CLASSES if n in available]
    if not wanted:
        raise ValueError("权重中未找到行人或车辆类别，请检查模型 names。")
    if any(n not in TARGET_CLASSES for n in wanted):
        raise ValueError(f"支持的类别：{', '.join(TARGET_CLASSES)}")
    missing = set(wanted) - available.keys()
    if missing:
        raise ValueError(f"权重中缺少指定类别：{', '.join(sorted(missing))}")
    # 这里只筛选推理输出，不会把 COCO 的 80 类检测头重新训练成六类。
    return list(dict.fromkeys(available[n] for n in wanted))


def new_run(output: str | Path | None, task: str) -> Path:
    # 每次实验独立保存，方便对应权重、阈值与结果，也避免覆盖前一次训练的 best.pt。
    if output is None:
        parent = ROOT / "runs" / "yolo" / task
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    path = Path(output).expanduser().resolve()
    # 每次运行必须使用新目录，避免覆盖已有训练或推理结果。
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value) -> None:
    # JSON 可以被后续跟踪、分析和网页模块读取；NaN/Infinity 不属于标准 JSON 数值。
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def detect(args) -> Path:
    import cv2

    # 图片目录视为一组独立图片；只有视频文件才具有连续帧与时间轴。
    source = Path(args.source).expanduser().resolve()
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
        if not images:
            raise ValueError(f"目录中没有支持的图片：{source}")
        video = False
    elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        images, video = [source], False
    elif source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        images, video = [], True
    else:
        raise ValueError("--source 必须是本地图片、图片目录或视频文件。")
    if not video and args.fps is not None:
        raise ValueError("--fps 仅适用于视频；独立图片不推断时间戳。")

    model = load_model(args.weights)
    class_ids = select_classes(model.names, args.classes)
    output = new_run(args.output, "detect")
    capture = writer = None
    count = 0
    # 这是跨帧检测框的累计数。同一辆车出现在 100 帧里，可能被累计 100 次；
    # 统计独立车辆数量需要后续跟踪 ID，不能直接使用这个计数器。
    counts = Counter()
    started = time.perf_counter()

    def process(frame, path, frame_index, timestamp, timestamp_source, stream):
        nonlocal count
        # OpenCV 的 frame 是形状为 [高, 宽, 3] 的 NumPy 数组，通常采用 BGR 通道顺序。
        # Ultralytics 会处理颜色转换、按比例缩放/填充、归一化及 PyTorch 张量转换，
        # 因此此处直接传原图，不需要手动拼成网络要求的 [批次, 通道, 高, 宽]。
        #
        # conf：先过滤分数低于阈值的候选框。提高阈值可能减少误检，也可能增加漏检。
        # iou：这里是 NMS 阈值。IoU = 交集面积 / 并集面积，用于比较两个框的重叠程度；
        #      标准 NMS 抑制同类目标的高重叠重复框，不等于评估 mAP 时的匹配阈值。
        # imgsz：预处理的目标尺寸；模型输出会再映射回原图坐标。
        result = model.predict(
            source=frame, classes=class_ids, conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, verbose=False,
        )[0]
        # predict 返回“每张输入图片一个 Results 对象”的列表。
        # 此次只传了一帧，所以 [0] 取这一帧的结果，不是只取第一个检测目标。
        detections = []
        for box in result.boxes:
            # boxes 中的数值是 PyTorch 张量，可能存放在 GPU。
            # item()/tolist() 将其转成 Python 数字，便于写入 JSON。
            class_id = int(box.cls.item())
            class_name = str(model.names[class_id])
            # xyxy = [左上角 x, 左上角 y, 右下角 x, 右下角 y]，单位为原图像素。
            # 图像原点在左上角：x 向右增大，y 向下增大。
            # 它不同于训练标签里的“归一化中心点 + 宽高”，不能直接混用。
            coords = [float(v) for v in box.xyxy[0].tolist()]
            # 分数表示模型对本次检测的确信程度，并非经过验证的实际正确率。
            # 即使分数很高，遇到校园场景分布差异时也可能误检。
            confidence = float(box.conf.item())
            category = "pedestrian" if class_name.lower() == "person" else "vehicle"
            detections.append({
                "class_id": class_id, "class_name": class_name, "category": category,
                "confidence": confidence, "bbox_xyxy": coords, "track_id": None,
            })
            # track_id 暂为 None：逐帧检测不判断“这一帧的车是不是上一帧那辆车”。
            # 需要跨帧 ID 时使用 python -m tracking，由 ByteTrack 分配并维护身份。
            counts[class_name] += 1
            # OpenCV 的绘图颜色也是 BGR。绘制时坐标取整，JSON 仍保留浮点坐标，
            # 避免将显示所需的舍入误差传给后续轨迹与几何计算。
            color = (60, 210, 60) if category == "pedestrian" else (30, 160, 255)
            x1, y1, x2, y2 = map(round, coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        record = {
            "source": str(path), "frame_index": frame_index,
            "timestamp_s": timestamp, "timestamp_source": timestamp_source,
            "width": frame.shape[1], "height": frame.shape[0],
            "detections": detections,
        }
        # 即使没有检测结果，也保留这一帧，避免后续跟踪丢失时间信息。
        # JSONL 每行是一个完整 JSON，长视频可以逐行写入和读取，无需全部放进内存。
        # 空帧记录的 detections 为 []，后续跟踪器仍可据此执行状态预测。
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        count += 1
        return frame

    try:
        with (output / "detections.jsonl").open("w", encoding="utf-8") as stream:
            if video:
                # VideoCapture 把压缩视频解码成一帧帧图像；文件存在不代表解码一定成功。
                capture = cv2.VideoCapture(str(source))
                if not capture.isOpened():
                    raise ValueError(f"无法打开视频：{source}")
                # 源视频 FPS 描述采样时间，不是模型每秒能处理多少帧。
                # 后续测速需要真实时间间隔，元数据无效时不能随意猜一个 FPS。
                fps = args.fps if args.fps is not None else capture.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("视频缺少有效 FPS，请使用 --fps 显式指定。")
                previous_timestamp = -1.0
                while args.max_frames is None or count < args.max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    # 读取解码器给出的当前帧时间位置，毫秒除以 1000 转成秒。
                    # 时间戳应递增；用帧数/FPS 兜底时明确标记其为估计值。
                    # 对可变帧率视频，这种估计可能不够准确，不能直接当作测速真值。
                    pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    timestamp_source = "video_pts"
                    if args.fps is not None or not math.isfinite(pts) or pts < 0 or pts <= previous_timestamp:
                        pts, timestamp_source = count / fps, "fps_estimate"
                    if pts <= previous_timestamp:
                        raise ValueError("视频时间戳不连续，请检查视频或用 --fps 指定固定帧率。")
                    previous_timestamp = pts
                    if writer is None:
                        # NumPy shape 的顺序是高、宽；VideoWriter 的尺寸参数是宽、高。
                        # 在读到第一帧后再创建输出，使用实际解码尺寸。
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(str(output / "annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("无法创建 MP4，请检查 OpenCV 的视频编码支持。")
                    # process 同时导出这一帧的检测数据，并返回画好检测框的图像。
                    writer.write(process(frame, source, count, pts, timestamp_source, stream))
            else:
                for path in images:
                    if args.max_frames is not None and count >= args.max_frames:
                        break
                    frame = cv2.imread(str(path))
                    if frame is None:
                        raise ValueError(f"无法读取图片：{path}")
                    # 独立图片各自从 frame_index=0 开始，不推测它们之间的时间关系。
                    process(frame, path, 0, None, None, stream)
                    # 文件名包含原扩展名，避免同目录的 a.jpg 与 a.png 输出时互相覆盖。
                    filename = f"{path.stem}_{path.suffix[1:].lower()}_det.jpg"
                    if not cv2.imwrite(str(output / filename), frame):
                        raise RuntimeError(f"保存图片失败：{filename}")
        if count == 0:
            raise ValueError("输入没有可解码的视频帧。")
    finally:
        # 无论推理成功还是中途报错都释放资源；VideoWriter 关闭时还要完成文件封装，
        # 否则磁盘上虽然出现 MP4 文件，也可能无法正常播放。
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
    # processing_fps 覆盖本次处理循环中的解码、推理、绘制与保存等开销。
    # 它不是输入视频 FPS，也不能代替尚未集成的完整预警系统端到端性能。
    write_json(output / "summary.json", {
        "weights": str(Path(args.weights).expanduser().resolve()), "source": str(source),
        "classes": {str(i): model.names[i] for i in class_ids},
        "frames_processed": count, "max_frames": args.max_frames,
        "detections_per_class": dict(counts), "elapsed_s": elapsed,
        "processing_fps": count / elapsed, "conf": args.conf, "iou": args.iou,
        "imgsz": args.imgsz,
        "timing_note": "MP4 输出使用固定 FPS；逐帧源时间戳以 detections.jsonl 为准。",
    })
    return output
