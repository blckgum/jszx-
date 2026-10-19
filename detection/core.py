"""YOLOv8 推理与逐帧导出。坐标始终使用原图像素，供后续跟踪使用。"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "checkpoints" / "yolo" / "yolov8n.pt"
TARGET_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "vehicle")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def prepare_runtime() -> None:
    # 默认缓存位于已忽略的 runs/，同时尊重用户设置的路径。
    for name, suffix in (("YOLO_CONFIG_DIR", "ultralytics"), ("MPLCONFIGDIR", "matplotlib")):
        os.environ.setdefault(name, str(ROOT / "runs" / ".cache" / suffix))
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def load_model(weights: str | Path):
    prepare_runtime()
    from ultralytics import YOLO

    path = Path(weights).expanduser()
    if not path.is_file():
        if path.name not in {f"yolov8{size}.pt" for size in "nslmx"}:
            raise FileNotFoundError(f"模型不存在：{path}；请提供本地检测权重。")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics 从官方 assets release 下载标准 YOLOv8 权重。
    model = YOLO(str(path))
    if model.task != "detect":
        raise ValueError("需要目标检测权重，不能使用分类、分割或姿态模型。")
    return model


def select_classes(names: dict[int, str], requested: list[str] | None = None) -> list[int]:
    """按类别名称选择 ID，兼容 COCO 的 80 类及微调后的自定义类别顺序。"""
    available = {str(name).lower(): int(index) for index, name in names.items()}
    wanted = requested if requested is not None else [n for n in TARGET_CLASSES if n in available]
    if not wanted:
        raise ValueError("权重中未找到行人或车辆类别，请检查模型 names。")
    if any(n not in TARGET_CLASSES for n in wanted):
        raise ValueError(f"支持的类别：{', '.join(TARGET_CLASSES)}")
    missing = set(wanted) - available.keys()
    if missing:
        raise ValueError(f"权重中缺少指定类别：{', '.join(sorted(missing))}")
    return list(dict.fromkeys(available[n] for n in wanted))


def new_run(output: str | Path | None, task: str) -> Path:
    if output is None:
        parent = ROOT / "runs" / "yolo" / task
        parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    path = Path(output).expanduser().resolve()
    # 每次运行必须使用新目录，避免覆盖已有训练或推理结果。
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def detect(args) -> Path:
    import cv2

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
    counts = Counter()
    started = time.perf_counter()

    def process(frame, path, frame_index, timestamp, timestamp_source, stream):
        nonlocal count
        result = model.predict(
            source=frame, classes=class_ids, conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(model.names[class_id])
            coords = [float(v) for v in box.xyxy[0].tolist()]
            confidence = float(box.conf.item())
            category = "pedestrian" if class_name.lower() == "person" else "vehicle"
            detections.append({
                "class_id": class_id, "class_name": class_name, "category": category,
                "confidence": confidence, "bbox_xyxy": coords, "track_id": None,
            })
            counts[class_name] += 1
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
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        count += 1
        return frame

    try:
        with (output / "detections.jsonl").open("w", encoding="utf-8") as stream:
            if video:
                capture = cv2.VideoCapture(str(source))
                if not capture.isOpened():
                    raise ValueError(f"无法打开视频：{source}")
                fps = args.fps if args.fps is not None else capture.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("视频缺少有效 FPS，请使用 --fps 显式指定。")
                previous_timestamp = -1.0
                while args.max_frames is None or count < args.max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    pts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    timestamp_source = "video_pts"
                    if args.fps is not None or not math.isfinite(pts) or pts < 0 or pts <= previous_timestamp:
                        pts, timestamp_source = count / fps, "fps_estimate"
                    if pts <= previous_timestamp:
                        raise ValueError("视频时间戳不连续，请检查视频或用 --fps 指定固定帧率。")
                    previous_timestamp = pts
                    if writer is None:
                        height, width = frame.shape[:2]
                        writer = cv2.VideoWriter(str(output / "annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("无法创建 MP4，请检查 OpenCV 的视频编码支持。")
                    writer.write(process(frame, source, count, pts, timestamp_source, stream))
            else:
                for path in images:
                    if args.max_frames is not None and count >= args.max_frames:
                        break
                    frame = cv2.imread(str(path))
                    if frame is None:
                        raise ValueError(f"无法读取图片：{path}")
                    process(frame, path, 0, None, None, stream)
                    filename = f"{path.stem}_{path.suffix[1:].lower()}_det.jpg"
                    if not cv2.imwrite(str(output / filename), frame):
                        raise RuntimeError(f"保存图片失败：{filename}")
        if count == 0:
            raise ValueError("输入没有可解码的视频帧。")
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
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
