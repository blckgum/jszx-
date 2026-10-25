"""独立的观测轨迹与图像运动估计：不依赖网络、OpenCV、相机标定或 TTC。"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from .records import FrameRecord, TargetRecord

DEFAULT_MOTION_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "motion.yaml"


@dataclass(frozen=True)
class MotionConfig:
    """tau、dt、显示时域用秒；像素阈值用 px/s；位置不转换成道路坐标。"""

    history_length: int = 60
    position_tau_s: float = 0.15
    velocity_tau_s: float = 0.30
    min_dt_s: float = 0.001
    max_dt_s: float = 1.0
    stationary_speed_px_s: float = 2.0
    max_step_speed_px_s: float | None = 2000.0
    arrow_horizon_s: float = 0.5
    arrow_max_length_px: int = 60

    def __post_init__(self):
        for name in ("history_length", "arrow_max_length_px"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        for name in ("position_tau_s", "velocity_tau_s", "min_dt_s", "max_dt_s",
                     "stationary_speed_px_s", "max_step_speed_px_s", "arrow_horizon_s"):
            value = getattr(self, name)
            if name == "max_step_speed_px_s" and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须为有限非负数")
            if name in {"min_dt_s", "max_dt_s", "max_step_speed_px_s", "arrow_horizon_s"} and value == 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.max_dt_s < self.min_dt_s:
            raise ValueError("max_dt_s 不能小于 min_dt_s")


def read_motion_config(path=DEFAULT_MOTION_CONFIG):
    try:
        values = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"运动配置 YAML 格式错误：{exc}") from None
    if not isinstance(values, dict) or set(values) != {field.name for field in fields(MotionConfig)}:
        raise ValueError("运动配置字段必须与 configs/motion.yaml 一致")
    return MotionConfig(**values)


@dataclass(frozen=True)
class MotionTarget:
    # 保留本模块 EMA 之前的跟踪观测；ByteTrack 返回框本身可能已经过 Kalman 更新。
    # 后续标定可以自行选择该输入点或本模块平滑点。
    target: TargetRecord
    smoothed_anchor_xy: tuple[float, float]
    segment_id: int | None = None
    dt_s: float | None = None
    gap_frames: int = 0
    raw_displacement_xy_px: tuple[float, float] | None = None
    displacement_xy_px: tuple[float, float] | None = None
    displacement_px: float | None = None
    raw_velocity_xy_px_s: tuple[float, float] | None = None
    velocity_xy_px_s: tuple[float, float] | None = None
    speed_px_s: float | None = None
    # direction_xy 是单位向量；图像 y 向下，direction_deg 因而从向右顺时针增加。
    direction_xy: tuple[float, float] | None = None
    direction_deg: float | None = None
    velocity_valid: bool = False
    image_motion_state: str = "initializing"
    reset_reason: str | None = None

    def to_dict(self):
        value = asdict(self)
        target = value.pop("target")
        return {**target, **value}


@dataclass(frozen=True)
class MotionFrame:
    """原始跟踪帧与当前实际观测的运动量；没有目标时 tracks 为空列表。"""

    frame: FrameRecord
    tracks: list[MotionTarget]

    def to_dict(self):
        frame = self.frame
        return {
            "schema_version": 1, "session_id": frame.session_id, "source": frame.source,
            "frame_index": frame.frame_index, "timestamp_s": frame.timestamp_s,
            "timestamp_source": frame.timestamp_source, "width": frame.width, "height": frame.height,
            "coordinate_space": "image", "position_unit": "px", "speed_unit": "px/s",
            "direction_convention": "clockwise_from_image_positive_x",
            "tracks": [track.to_dict() for track in self.tracks],
        }


@dataclass(frozen=True)
class TrajectoryPoint:
    frame_index: int
    timestamp_s: float
    timestamp_source: str
    segment_id: int
    raw_anchor_xy: tuple[float, float]
    smoothed_anchor_xy: tuple[float, float]


@dataclass
class _TrackState:
    last_frame: int
    last_time: float
    timestamp_source: str
    last: MotionTarget
    points: deque


def ema(previous, current, dt_s, tau_s):
    # alpha 随真实观测间隔变化，而非假定每秒固定帧数；expm1 在 dt/tau 很小时更稳定。
    alpha = -math.expm1(-dt_s / tau_s) if tau_s else 1.0
    return tuple(old + alpha * (new - old) for old, new in zip(previous, current))


class MotionEstimator:
    """一视频一实例；每个检测帧更新一次，空帧只裁剪缓存，不补造观测。"""

    def __init__(self, config: MotionConfig | None = None, max_idle_frames: int = 30):
        self.config = config or MotionConfig()
        if type(max_idle_frames) is not int or max_idle_frames < 0:
            raise ValueError("max_idle_frames 必须为非负整数")
        self.max_idle_frames = max(self.config.history_length, max_idle_frames)
        self.reset()

    def reset(self):
        self._states: dict[int, _TrackState] = {}
        self.session_id = None
        self.source = None
        self.shape = None
        self.previous_time = -1.0
        self.previous_frame = -1
        # 片段编号在一个会话内全局递增，缓存清理后也不复用旧编号。
        self._segment_counter = 0

    @property
    def active_track_count(self):
        return len(self._states)

    def history_for(self, track_id):
        """返回有限历史快照，包含输入点和平滑点，供绘图或后续标定读取。"""
        state = self._states.get(track_id)
        return tuple(state.points) if state else ()

    def segments(self, track_id):
        # 同一个 ID 也可能经历遮挡或异常跳变；片段号变化时不跨缺口连线。
        segments, previous = [], None
        for point in self.history_for(track_id):
            if previous is None or point.segment_id != previous:
                segments.append([])
            segments[-1].append(point.smoothed_anchor_xy)
            previous = point.segment_id
        return segments

    def _validate(self, frame):
        if not isinstance(frame.session_id, str) or not frame.session_id:
            raise ValueError("运动记录必须有非空 session_id")
        if type(frame.frame_index) is not int or frame.frame_index < 0:
            raise ValueError("frame_index 必须为非负整数")
        if (type(frame.timestamp_s) not in (int, float) or not math.isfinite(frame.timestamp_s)
                or frame.timestamp_s < 0):
            raise ValueError("运动时间戳必须非负且有限")
        if frame.timestamp_source not in {"video_pts", "fps_estimate"}:
            raise ValueError("未知的时间戳来源")
        if any(type(value) is not int or value <= 0 for value in (frame.width, frame.height)):
            raise ValueError("图像宽高必须为正整数")
        if frame.session_id == self.session_id:
            if frame.timestamp_s <= self.previous_time or frame.frame_index <= self.previous_frame:
                raise ValueError("运动帧号与时间戳必须严格递增；无效帧未更新缓存")
            if (frame.width, frame.height) != self.shape or frame.source != self.source:
                raise ValueError("同一会话不能更换源视频或分辨率")
        ids = set()
        for target in frame.tracks:
            if target.observation != "detected":
                raise ValueError("运动估计只接受 detected 观测，不能混入预测框")
            if not all(math.isfinite(value) for value in (*target.bbox_xyxy, *target.anchor_xy, target.confidence)):
                raise ValueError("观测坐标与置信度必须有限")
            x1, y1, x2, y2 = target.bbox_xyxy
            x, y = target.anchor_xy
            if not (0 <= x1 < x2 <= frame.width and 0 <= y1 < y2 <= frame.height
                    and 0 <= x <= frame.width and 0 <= y <= frame.height and 0 <= target.confidence <= 1):
                raise ValueError("观测框、接地点或置信度超出有效范围")
            if target.track_id is not None:
                if type(target.track_id) is not int or target.track_id <= 0 or target.track_id in ids:
                    raise ValueError("一帧内的有效 Track ID 必须为不重复的正整数")
                ids.add(target.track_id)

    def update(self, frame: FrameRecord, retained_ids: set[int] | None = None):
        self._validate(frame)  # 先校验整帧，错误时间戳或目标数据不改变已有状态。
        if frame.session_id != self.session_id:
            self.reset()
            self.session_id, self.source = frame.session_id, frame.source
            self.shape = (frame.width, frame.height)
        tracks = [self._observe(frame, target) for target in frame.tracks]
        observed = {target.track_id for target in frame.tracks if target.track_id is not None}
        for track_id, state in list(self._states.items()):
            while state.points and state.points[0].frame_index <= frame.frame_index - self.config.history_length:
                state.points.popleft()
            if (frame.frame_index - state.last_frame > self.max_idle_frames
                    or retained_ids is not None and track_id not in retained_ids and track_id not in observed):
                del self._states[track_id]
        self.previous_time, self.previous_frame = frame.timestamp_s, frame.frame_index
        return MotionFrame(frame, tracks)

    def _observe(self, frame, target):
        raw = target.anchor_xy
        if target.track_id is None:
            # 没有 ID 无法确认相邻帧是否为同一目标，所以保留观测，但不猜测身份或速度。
            return MotionTarget(target, raw, image_motion_state="untracked", reset_reason="untracked")
        state = self._states.get(target.track_id)
        # 使用源观测的秒数差；与这台机器处理一帧花了多久无关。
        dt = frame.timestamp_s - state.last_time if state else None
        gap = frame.frame_index - state.last_frame - 1 if state else 0
        reason = "new_track" if state is None else self._reset_reason(frame, target, state, dt, gap)
        if reason is not None:
            # 恢复首帧只提供真实位置，旧滤波状态不跨片段延续；无效速度保持 None。
            self._segment_counter += 1
            motion = MotionTarget(target=target, smoothed_anchor_xy=raw, segment_id=self._segment_counter,
                                  dt_s=dt, gap_frames=gap, reset_reason=reason)
        else:
            previous = state.last
            # 差分会放大定位噪声：先平滑像素位置，再对位置差除以真实时间间隔。
            # 同时保留输入点的 raw 差分，便于比较平滑的影响。
            raw_delta = tuple(new - old for new, old in zip(raw, previous.target.anchor_xy))
            position = ema(previous.smoothed_anchor_xy, raw, dt, self.config.position_tau_s)
            delta = tuple(new - old for new, old in zip(position, previous.smoothed_anchor_xy))
            velocity = tuple(value / dt for value in delta)
            if previous.velocity_valid:
                velocity = ema(previous.velocity_xy_px_s, velocity, dt, self.config.velocity_tau_s)
            # 首个有效观测对直接初始化速度；后续才递推，避免人为从零速度缓慢起步。
            speed = math.hypot(*velocity)
            moving = speed > self.config.stationary_speed_px_s
            motion = MotionTarget(
                target=target, smoothed_anchor_xy=position, segment_id=previous.segment_id,
                dt_s=dt, gap_frames=gap, raw_displacement_xy_px=raw_delta, displacement_xy_px=delta,
                displacement_px=math.hypot(*delta), raw_velocity_xy_px_s=tuple(value / dt for value in raw_delta),
                velocity_xy_px_s=velocity, speed_px_s=speed,
                direction_xy=tuple(value / speed for value in velocity) if moving else None,
                direction_deg=math.degrees(math.atan2(velocity[1], velocity[0])) % 360 if moving else None,
                velocity_valid=True, image_motion_state="moving" if moving else "stationary",
            )
        points = state.points if state else deque(maxlen=self.config.history_length)
        # 只在有当前检测观测时追加；空帧不会在这里产生 Kalman 预测点或插值点。
        points.append(TrajectoryPoint(frame.frame_index, frame.timestamp_s, frame.timestamp_source,
                                      motion.segment_id, raw, motion.smoothed_anchor_xy))
        self._states[target.track_id] = _TrackState(frame.frame_index, frame.timestamp_s, frame.timestamp_source, motion, points)
        return motion

    def _reset_reason(self, frame, target, state, dt, gap):
        if gap:
            return "observation_gap"
        if frame.timestamp_source != state.timestamp_source:
            return "timestamp_source_changed"
        if target.class_id != state.last.target.class_id or target.class_name != state.last.target.class_name:
            return "class_changed"
        if target.anchor_type != state.last.target.anchor_type:
            return "anchor_type_changed"
        if dt < self.config.min_dt_s:
            return "dt_too_small"
        if dt > self.config.max_dt_s:
            return "dt_too_large"
        delta = tuple(new - old for new, old in zip(target.anchor_xy, state.last.target.anchor_xy))
        if self.config.max_step_speed_px_s is not None and math.hypot(*delta) / dt > self.config.max_step_speed_px_s:
            return "position_jump"
        return None
