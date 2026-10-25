"""只绘制已计算的图像运动，不在绘图或暂停刷新时重新估计速度。"""

import cv2

from .motion import MotionConfig, MotionFrame


def draw_motion(canvas, motion: MotionFrame, config: MotionConfig | None = None):
    """在已有标注副本上原地绘制；返回值仍为同一张 canvas，避免复制整帧。"""
    config = config or MotionConfig()
    height, width = canvas.shape[:2]
    for item in motion.tracks:
        if item.target.track_id is None:
            continue
        x, y = item.smoothed_anchor_xy
        start = (round(x), round(y))
        color = (60, 210, 60) if item.target.category == "pedestrian" else (30, 160, 255)
        cv2.circle(canvas, start, 3, color, -1)
        if item.direction_xy is not None:
            # 为可读性设置最短箭头，并限制最长箭头；准确数值应读取 px/s 标签和导出字段。
            length = min(config.arrow_max_length_px, max(8.0, item.speed_px_s * config.arrow_horizon_s))
            end = (round(x + item.direction_xy[0] * length), round(y + item.direction_xy[1] * length))
            cv2.arrowedLine(canvas, start, end, (0, 230, 255), 2, cv2.LINE_AA, tipLength=0.3)
        speed = f"{item.speed_px_s:.1f}" if item.velocity_valid else "--"
        suffix = "still" if item.image_motion_state == "stationary" else "init" if not item.velocity_valid else "moving"
        label = f"{speed} px/s {suffix}"
        # 接地点旁显示速度，类别与 ID 继续使用原来的顶部标签，减少长标签互相覆盖。
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        left = max(0, min(round(x) - text_width // 2, width - text_width - 6))
        bottom = min(height - baseline - 2, max(text_height + 4, round(y) + text_height + 8))
        cv2.rectangle(canvas, (left, bottom - text_height - 4),
                      (left + text_width + 5, bottom + baseline + 1), (25, 25, 25), -1)
        cv2.putText(canvas, label, (left + 2, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (245, 245, 245), 1, cv2.LINE_AA)
    return canvas
