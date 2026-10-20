"""独立视频跟踪会话：YOLOv8 检测 + Ultralytics ByteTrack，逐帧保持状态。"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path
from uuid import uuid4

import yaml

from detection.core import DEFAULT_WEIGHTS, ROOT, VIDEO_SUFFIXES, load_model, new_run, prepare_runtime, select_classes, write_json
from .records import FrameRecord, extract_tracks
from .visualization import PlaybackController, TrailHistory, annotate

DEFAULT_TRACKER = ROOT / "configs" / "bytetrack.yaml"


def read_tracker_config(path):
    path = Path(path).expanduser().resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"跟踪配置 YAML 格式错误：{exc}") from None
    required = {"tracker_type", "track_high_thresh", "track_low_thresh", "new_track_thresh",
                "track_buffer", "match_thresh", "fuse_score"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"跟踪配置应且仅应包含：{', '.join(sorted(required))}")
    if config["tracker_type"] != "bytetrack":
        raise ValueError("本入口只支持 tracker_type: bytetrack")
    for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{key} 必须大于 0 且不超过 1")
    if config["track_low_thresh"] >= config["track_high_thresh"]:
        raise ValueError("track_low_thresh 必须小于 track_high_thresh")
    if type(config["track_buffer"]) is not int or config["track_buffer"] < 0:
        raise ValueError("track_buffer 必须是非负整数帧数")
    if type(config["fuse_score"]) is not bool:
        raise ValueError("fuse_score 必须是 YAML 布尔值")
    return config


class VideoClock:
    def __init__(self, fps: float, force_fps: bool = False):
        self.fps = fps
        self.force_fps = force_fps
        self.previous = -1.0

    def timestamp(self, frame_index: int, milliseconds: float):
        timestamp = milliseconds / 1000.0
        kind = "video_pts"
        if (self.force_fps or not math.isfinite(timestamp) or timestamp < 0
                or timestamp <= self.previous):
            timestamp, kind = frame_index / self.fps, "fps_estimate"
        if timestamp <= self.previous:
            raise ValueError("视频时间戳不递增；检查输入或经确认后用 --fps 指定固定帧率")
        self.previous = timestamp
        return timestamp, kind


class TrackingSession:
    def __init__(self, source, weights, tracker_path, classes=None, conf=None, imgsz=640,
                 iou=0.7, device=None):
        config = read_tracker_config(tracker_path)
        self.conf = config["track_low_thresh"] if conf is None else conf
        if not math.isfinite(self.conf) or not 0 < self.conf <= config["track_low_thresh"]:
            raise ValueError("检测 conf 必须大于 0 且不高于 track_low_thresh，保留第二阶段关联候选")
        # 每个视频创建新的 YOLO 包装实例，避免继承上一段视频的 predictor/trackers。
        # 权重只读取，不训练或修改；该实例在本视频的所有帧之间持续使用。
        self.model = load_model(weights)
        self.class_ids = select_classes(self.model.names, classes)
        self.class_names = [self.model.names[index] for index in self.class_ids]
        self.source = str(Path(source).expanduser().resolve())
        self.session_id = uuid4().hex
        self.frame_index = 0
        self.tracker_path = str(Path(tracker_path).resolve())
        self.imgsz, self.iou, self.device = imgsz, iou, device

    def process(self, frame, timestamp, timestamp_source):
        # persist=True 仅用于同一视频会话；每个输入帧恰好调用一次，包含没有目标的帧。
        # YOLO 检测、Kalman 预测和 LAPJV 关联均由 Ultralytics 内部完成。
        result = self.model.track(
            source=frame, persist=True, tracker=self.tracker_path, conf=self.conf,
            classes=self.class_ids, imgsz=self.imgsz, iou=self.iou,
            device=self.device, batch=1, verbose=False,
        )[0]
        height, width = frame.shape[:2]
        record = FrameRecord(self.session_id, self.source, self.frame_index, timestamp,
                             timestamp_source, width, height, extract_tracks(result))
        self.frame_index += 1
        return record


def check_display():
    # Qt 在没有显示服务时可能直接终止进程，所以在调用窗口函数之前检查。
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("当前环境没有桌面显示；去掉 --show 可保存视频，本地桌面再验证按键")
    import cv2
    if not cv2.currentUIFramework():
        raise ValueError("当前 OpenCV 不支持窗口显示；使用无 --show 模式或安装支持 GUI 的构建")


def track_video(source, *, weights=DEFAULT_WEIGHTS, tracker=DEFAULT_TRACKER, output=None,
                classes=None, conf=None, imgsz=640, iou=0.7, device=None,
                trail_length=60, max_frames=None, fps=None, show=False):
    import cv2

    prepare_runtime()
    try:
        import lap  # noqa: F401；缺失时明确提示，不让框架在运行中自动安装。
    except ImportError:
        raise ImportError("缺少 ByteTrack 匹配依赖，请运行 python -m pip install -r requirements-yolo.txt") from None
    source = Path(source).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("--source 必须是本地视频文件")
    for name, value in (("imgsz", imgsz), ("trail_length", trail_length)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError("max_frames 必须是正整数")
    if not math.isfinite(iou) or not 0 < iou <= 1:
        raise ValueError("iou 必须大于 0 且不超过 1")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("fps 必须是有限正数")
    config = read_tracker_config(tracker)
    if show:
        check_display()

    capture = cv2.VideoCapture(str(source))
    writer = None
    window_open = False
    window_name = "YOLOv8 + ByteTrack"
    stop_reason = "eof"
    processed = 0
    processing_seconds = 0.0
    recent_times = deque(maxlen=30)
    observations = Counter()
    track_ids = set()
    wall_start = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError(f"无法打开视频：{source}")
        raw_fps = capture.get(cv2.CAP_PROP_FPS)
        source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
        effective_fps = fps if fps is not None else source_fps
        if effective_fps is None:
            raise ValueError("视频缺少有效 FPS；确认帧率后通过 --fps 指定")
        reported = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(reported) if math.isfinite(reported) and reported > 0 else None
        clock = VideoClock(effective_fps, force_fps=fps is not None)
        output = new_run(output, "track")
        # 保存本次实际使用的配置，后续调参不会改变旧实验的参数记录。
        frozen_tracker = output / "tracker.resolved.yaml"
        frozen_tracker.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        session = TrackingSession(source, weights, frozen_tracker, classes, conf, imgsz, iou, device)
        history = TrailHistory(trail_length, config["track_buffer"])
        playback = PlaybackController()
        if show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_open = True
        try:
            with (output / "tracks.jsonl").open("w", encoding="utf-8") as stream:
                while not playback.quit_requested:
                    if max_frames is not None and processed >= max_frames:
                        stop_reason = "max_frames"
                        break
                    # 计时覆盖解码、推理、绘制及写入，不包括按键等待、暂停和模型初始加载。
                    frame_start = time.perf_counter()
                    ok, frame = capture.read()
                    if not ok:
                        break
                    timestamp, timestamp_source = clock.timestamp(processed, capture.get(cv2.CAP_PROP_POS_MSEC))
                    record = session.process(frame, timestamp, timestamp_source)
                    history.update(record)
                    measured_fps = len(recent_times) / sum(recent_times) if recent_times else None
                    annotated = annotate(frame, record, history, source_fps, measured_fps,
                                         total_frames, session.class_names, interactive=show)
                    if writer is None:
                        writer = cv2.VideoWriter(str(output / "annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), effective_fps,
                                                 (record.width, record.height))
                        if not writer.isOpened():
                            raise RuntimeError("无法创建标注 MP4，请检查 OpenCV 编码器与输出路径")
                    writer.write(annotated)
                    stream.write(json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
                    processed += 1
                    observations.update(target.class_name for target in record.tracks)
                    track_ids.update(target.track_id for target in record.tracks if target.track_id is not None)
                    duration = time.perf_counter() - frame_start
                    processing_seconds += duration
                    recent_times.append(duration)

                    if show:
                        cv2.imshow(window_name, annotated)
                        playback.handle_key(cv2.waitKey(1))
                        # 暂停期间只刷新最后一帧，不读取、推理或写出新帧。
                        while playback.paused and not playback.quit_requested:
                            paused_frame = annotate(frame, record, history, source_fps, measured_fps,
                                                    total_frames, session.class_names, interactive=True, paused=True)
                            cv2.imshow(window_name, paused_frame)
                            playback.handle_key(cv2.waitKey(30))
                            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                                playback.quit_requested = True
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            playback.quit_requested = True
                if playback.quit_requested:
                    stop_reason = "user_quit"
        except KeyboardInterrupt:
            stop_reason = "interrupted"
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if window_open:
            cv2.destroyWindow(window_name)

    if processed == 0 and stop_reason != "interrupted":
        raise ValueError("没有成功解码的视频帧")
    write_json(output / "summary.json", {
        "schema_version": 1, "session_id": session.session_id, "source": str(source),
        "weights": str(Path(weights).expanduser().resolve()),
        "tracker": config, "classes": {str(index): session.model.names[index] for index in session.class_ids},
        "source_fps": source_fps, "effective_output_fps": effective_fps,
        "timestamp_fps_override": fps is not None, "source_frame_count_reported": total_frames,
        "reported_frame_count_mismatch": stop_reason == "eof" and total_frames is not None and processed != total_frames,
        "frames_processed": processed, "stop_reason": stop_reason,
        "processing_seconds": processing_seconds,
        "processing_fps": processed / processing_seconds if processing_seconds > 0 else None,
        "wall_seconds_including_model_load_and_pause": time.perf_counter() - wall_start,
        "observations_per_class": dict(observations), "confirmed_track_ids": sorted(track_ids),
        "conf": session.conf, "iou": iou, "imgsz": imgsz, "trail_length": trail_length,
        "note": "仅导出检测观测；ID 不是物理速度或 TTC。MP4 使用固定 FPS，源时间以 JSONL 为准。",
    })
    return output
