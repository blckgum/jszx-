"""独立视频跟踪会话：YOLOv8 检测 + Ultralytics ByteTrack，逐帧保持状态。"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import yaml
import numpy as np

from detection.core import DEFAULT_WEIGHTS, ROOT, VIDEO_SUFFIXES, load_model, new_run, prepare_runtime, select_classes, write_json
from .records import FrameRecord, extract_tracks
from .visualization import PlaybackController, annotate
from .export import RunExports
from .motion import DEFAULT_MOTION_CONFIG, MotionEstimator, read_motion_config

DEFAULT_TRACKER = ROOT / "configs" / "bytetrack.yaml"


def resolve_conf(config, conf):
    value = config["track_low_thresh"] if conf is None else conf
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= config["track_low_thresh"]:
        raise ValueError("检测 conf 必须大于 0 且不高于 track_low_thresh，保留第二阶段关联候选")
    return value


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
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps 必须是有限正数")
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
        self.conf = resolve_conf(config, conf)
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
        self.previous_time = -1.0
        self.frame_shape = None

    def process(self, frame, timestamp, timestamp_source):
        # OpenCV 图像为 H×W×3、uint8、BGR。错误输入先拒绝，避免破坏 Kalman 时间步。
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0):
            raise ValueError("帧必须是非空 H×W×3 uint8 BGR 图像")
        if self.frame_shape is not None and frame.shape != self.frame_shape:
            raise ValueError("同一视频内分辨率发生变化，请预处理为固定尺寸后重新跟踪")
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= self.previous_time:
            raise ValueError("同一会话的时间戳必须非负且严格递增")
        if timestamp_source not in {"video_pts", "fps_estimate"}:
            raise ValueError("timestamp_source 必须为 video_pts 或 fps_estimate")
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
        self.previous_time, self.frame_shape = timestamp, frame.shape
        return record

    def retained_ids(self):
        # 只读取已固定版本的 Ultralytics 状态池，不调用 update，不改变跟踪结果。
        # 8.4.182 的 lost 池在移除边界可能短暂保留 Removed 对象，下一帧仍会参与关联。
        # 因此按池成员判断是否仍可被关联，不提前按枚举关闭同一个 ID 的统计。
        tracker = self.model.predictor.trackers[0]
        return {int(track.track_id) for track in (*tracker.tracked_stracks, *tracker.lost_stracks)}


def check_display():
    # Qt 在没有显示服务时可能直接终止进程，所以在调用窗口函数之前检查。
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ValueError("当前环境没有桌面显示；去掉 --show 可保存视频，本地桌面再验证按键")
    import cv2
    if not cv2.currentUIFramework():
        raise ValueError("当前 OpenCV 不支持窗口显示；使用无 --show 模式或安装支持 GUI 的构建")


def track_video(source, *, weights=DEFAULT_WEIGHTS, tracker=DEFAULT_TRACKER, output=None,
                classes=None, conf=None, imgsz=640, iou=0.7, device=None,
                trail_length=None, max_frames=None, fps=None, show=False,
                motion_config=DEFAULT_MOTION_CONFIG):
    import cv2

    prepare_runtime()
    try:
        import lap  # noqa: F401；缺失时明确提示，不让框架在运行中自动安装。
    except ImportError:
        raise ImportError("缺少 ByteTrack 匹配依赖，请运行 python -m pip install -r requirements-yolo.txt") from None
    source = Path(source).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("--source 必须是本地视频文件")
    if type(imgsz) is not int or imgsz <= 0:
        raise ValueError("imgsz 必须是正整数")
    if trail_length is not None and (type(trail_length) is not int or trail_length <= 0):
        raise ValueError("trail_length 必须是正整数")
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError("max_frames 必须是正整数")
    if not math.isfinite(iou) or not 0 < iou <= 1:
        raise ValueError("iou 必须大于 0 且不超过 1")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("fps 必须是有限正数")
    config = read_tracker_config(tracker)
    conf = resolve_conf(config, conf)
    motion_settings = read_motion_config(motion_config)
    if trail_length is not None:
        motion_settings = replace(motion_settings, history_length=trail_length)
    trail_length = motion_settings.history_length
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
    session = exports = output_path = history = None
    source_fps = effective_fps = total_frames = None
    failure = None
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
        output_path = new_run(output, "track")
        # 保存本次实际使用的配置，后续调参不会改变旧实验的参数记录。
        frozen_tracker = output_path / "tracker.resolved.yaml"
        frozen_tracker.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (output_path / "motion.resolved.yaml").write_text(yaml.safe_dump(asdict(motion_settings)), encoding="utf-8")
        exports = RunExports(output_path)
        session = TrackingSession(source, weights, frozen_tracker, classes, conf, imgsz, iou, device)
        # 估计器同时提供有限轨迹，避免再维护一份重复的位置缓存。
        history = MotionEstimator(motion_settings, config["track_buffer"])
        playback = PlaybackController()
        if show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_open = True
        try:
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
                retained_ids = session.retained_ids()
                motion = history.update(record, retained_ids)
                measured_fps = len(recent_times) / sum(recent_times) if recent_times else None
                annotated = annotate(frame, record, history, source_fps, measured_fps,
                                     total_frames, session.class_names, interactive=show,
                                     motion=motion, motion_config=motion_settings)
                if writer is None:
                    writer = cv2.VideoWriter(str(output_path / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), effective_fps,
                                             (record.width, record.height))
                    if not writer.isOpened():
                        raise RuntimeError("无法创建标注 MP4，请检查 OpenCV 编码器与输出路径")
                writer.write(annotated)
                exports.write(record, retained_ids, motion)
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
                                                total_frames, session.class_names, interactive=True, paused=True,
                                                motion=motion, motion_config=motion_settings)
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
        if processed == 0 and stop_reason != "interrupted":
            raise ValueError("没有成功解码的视频帧")
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as exc:
        # 推理、编码或时间轴出错仍封装已完成的视频，并留下失败原因，调用方继续收到异常。
        stop_reason, failure = "error", exc
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if window_open:
            cv2.destroyWindow(window_name)
        if exports is not None:
            try:
                exports.finish(stop_reason)
            finally:
                exports.close()
        if history is not None:
            history.reset()

    if output_path is None:
        if failure is not None:
            raise failure
        raise KeyboardInterrupt
    write_json(output_path / "summary.json", {
        "schema_version": 2, "session_id": session.session_id if session else None, "source": str(source),
        "weights": str(Path(weights).expanduser().resolve()),
        "tracker": config, "classes": {str(index): session.model.names[index] for index in session.class_ids} if session else {},
        "source_fps": source_fps, "effective_output_fps": effective_fps,
        "timestamp_fps_override": fps is not None, "source_frame_count_reported": total_frames,
        "reported_frame_count_mismatch": stop_reason == "eof" and total_frames is not None and processed != total_frames,
        "frames_processed": processed, "stop_reason": stop_reason,
        "status": "failed" if failure is not None else "interrupted" if stop_reason == "interrupted" else "completed",
        "error": {"type": type(failure).__name__, "message": str(failure)} if failure is not None else None,
        "processing_seconds": processing_seconds,
        "processing_fps": processed / processing_seconds if processing_seconds > 0 else None,
        "wall_seconds_including_model_load_and_pause": time.perf_counter() - wall_start,
        "observations_per_class": dict(observations), "confirmed_track_ids": sorted(track_ids),
        "conf": session.conf if session else conf, "iou": iou, "imgsz": imgsz, "trail_length": trail_length,
        "closed_tracklets": exports.closed_tracks if exports else 0,
        "track_events": dict(exports.event_counts) if exports else {},
        "motion": {"config": asdict(motion_settings), "schema_version": 1, "coordinate_space": "image",
                   "speed_unit": "px/s", "states": dict(exports.motion_states) if exports else {},
                   "reset_reasons": dict(exports.motion_resets) if exports else {},
                   "valid_velocity_observations": exports.valid_velocity_observations if exports else 0},
        "note": "仅导出检测观测；ID 不是物理速度或 TTC。MP4 使用固定 FPS，源时间以 JSONL 为准。",
    })
    if failure is not None:
        failure.add_note(f"本次运行汇总与已完成结果：{output_path}")
        raise failure
    return output_path
