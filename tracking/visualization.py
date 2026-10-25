"""分组配色、有限长度轨迹和运行信息；本文件不执行目标关联。"""

from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np


class TrailHistory:
    def __init__(self, length: int, max_idle_frames: int):
        self.length = length
        self.max_idle_frames = max(length, max_idle_frames)
        self.points = {}
        self.last_seen = {}

    def update(self, record):
        for target in record.tracks:
            if target.track_id is None:
                continue
            history = self.points.setdefault(target.track_id, deque(maxlen=self.length))
            history.append((record.frame_index, record.timestamp_s, *target.anchor_xy))
            self.last_seen[target.track_id] = record.frame_index
        # 用源帧号裁剪，避免目标漏检之后仍显示很久以前的轨迹。
        for track_id in list(self.points):
            history = self.points[track_id]
            while history and history[0][0] <= record.frame_index - self.length:
                history.popleft()
            if record.frame_index - self.last_seen[track_id] > self.max_idle_frames:
                del self.points[track_id]
                del self.last_seen[track_id]

    def segments(self, track_id):
        # 观测缺失时断开线段，不把跨越空帧的直线冒充真实连续观测。
        segments = []
        previous = None
        for index, timestamp, x, y in self.points.get(track_id, ()):
            if previous is None or index != previous + 1:
                segments.append([])
            segments[-1].append((x, y))
            previous = index
        return segments


class PlaybackController:
    def __init__(self):
        self.paused = False
        self.quit_requested = False

    def handle_key(self, key: int):
        key &= 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self.quit_requested = True
        elif key in (32, ord("p"), ord("P")):
            self.paused = not self.paused


def annotate(frame, record, history, source_fps, processing_fps, total_frames, class_names,
             interactive=False, paused=False, motion=None, motion_config=None):
    # 绘制副本，避免将文字或轨迹带回模型输入。坐标仍以原视频图像为基准。
    canvas = frame.copy()
    labels = []
    for target in record.tracks:
        color = (60, 210, 60) if target.category == "pedestrian" else (30, 160, 255)
        x1, y1, x2, y2 = map(round, target.bbox_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        identity = str(target.track_id) if target.track_id is not None else "pending"
        labels.append((f"{target.class_name} {target.confidence:.2f} ID:{identity}", x1, y1, color))
        if target.track_id is not None:
            for segment in history.segments(target.track_id):
                points = np.rint(segment).astype(np.int32).reshape(-1, 1, 2)
                if len(points) > 1:
                    cv2.polylines(canvas, [points], False, color, 2)
                elif len(points) == 1:
                    cv2.circle(canvas, tuple(points[0, 0]), 2, color, -1)

    if motion is not None:
        from .motion_visualization import draw_motion
        draw_motion(canvas, motion, motion_config)

    counts = Counter(target.class_name for target in record.tracks)
    confirmed = sum(target.track_id is not None for target in record.tracks)
    total = str(total_frames) if total_frames is not None else "?"
    source_label = f"{source_fps:.2f}" if source_fps is not None else "unknown"
    speed_label = f"{processing_fps:.2f}" if processing_fps is not None else "--"
    state = "PAUSED" if paused else "RUNNING"
    paragraphs = [
        f"Frame: {record.frame_index + 1}/{total} Tracked: {confirmed}/{len(record.tracks)} {state}",
        f"SourceFPS: {source_label} ProcessingFPS: {speed_label}",
        "  ".join(f"{name}: {counts[name]}" for name in class_names),
    ]
    if interactive:
        paragraphs.append("Space/P: pause/resume  Q/Esc: quit")
    # 按实际图像宽度换行，避免较小视频中的统计文字超出画面。
    lines = []
    for paragraph in paragraphs:
        current = ""
        for token in paragraph.split():
            candidate = f"{current} {token}".strip()
            width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            if current and width > canvas.shape[1] - 16:
                lines.append(current)
                current = token
            else:
                current = candidate
        lines.append(current)
    hud_bottom = min(canvas.shape[0] - 1, 20 * len(lines) + 8)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, hud_bottom),
                  (25, 25, 25), -1)
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (8, 18 + 20 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 255) if paused else (245, 245, 245), 1)
    # 标签最后绘制并尝试上下避让，减少相邻行人的文字互相覆盖。
    # 避让后用引线指向原框；极高密度或极小画面仍可能没有足够显示空间。
    occupied = [(0, 0, canvas.shape[1], hud_bottom + 2)]
    for label, x1, y1, color in labels:
        (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = min(max(0, x1), max(0, canvas.shape[1] - width - 8))
        desired_y = min(canvas.shape[0] - baseline - 3, max(height + 5, y1 - 6))
        y = desired_y
        for offset in [0] + [direction * step for step in range(1, 13) for direction in (-1, 1)]:
            candidate = desired_y + offset * (height + baseline + 9)
            rect = (x, candidate - height - 5, x + width + 6, candidate + baseline + 2)
            if rect[1] < 0 or rect[3] >= canvas.shape[0]:
                continue
            if all(rect[2] < a or rect[0] > c or rect[3] < b or rect[1] > d for a, b, c, d in occupied):
                y = candidate
                break
        rectangle = (x, y - height - 5, x + width + 6, y + baseline + 2)
        occupied.append(rectangle)
        if y != desired_y:
            cv2.line(canvas, (x + 3, y), (max(0, x1), max(0, y1)), color, 1)
        cv2.rectangle(canvas, rectangle[:2], rectangle[2:], color, -1)
        cv2.putText(canvas, label, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas
