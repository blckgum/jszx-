"""原图上的已知地面网格、拟合点和独立检查点，不绘制未经验证的车速。"""

import cv2
import numpy as np


def render_grid(mapper, image=None):
    width, height = mapper.config.image_size
    if image is None:
        image = np.full((height, width, 3), 35, dtype=np.uint8)
    if image.shape[:2] != (height, width):
        raise ValueError("底图尺寸与标定图像尺寸不一致")
    canvas = image.copy()
    ground = np.asarray(mapper.config.ground_points_m)[mapper.inliers]
    minimum, maximum = ground.min(axis=0), ground.max(axis=0)
    # 地面等距线投影回原图；畸变存在时采样成多段折线，不能仅连两端点。
    for axis in (0, 1):
        for value in np.linspace(minimum[axis], maximum[axis], 9):
            points = np.empty((40, 2))
            points[:, axis] = value
            points[:, 1 - axis] = np.linspace(minimum[1 - axis], maximum[1 - axis], 40)
            pixels, valid = mapper.ground_to_image(points)
            segment = []
            for pixel, good in zip(pixels, valid):
                visible = (good and np.all(np.abs(pixel) < 1e6)
                           and cv2.pointPolygonTest(mapper.roi, tuple(map(float, pixel)), True) >= -1e-4)
                if visible:
                    segment.append(pixel)
                elif segment:
                    cv2.polylines(canvas, [np.rint(segment).astype(np.int32)], False, (75, 140, 75), 1)
                    segment = []
            if segment:
                cv2.polylines(canvas, [np.rint(segment).astype(np.int32)], False, (75, 140, 75), 1)
    cv2.polylines(canvas, [np.rint(mapper.roi).astype(np.int32)], True, (0, 210, 255), 2)
    for prefix, points, color in (("FIT", mapper.config.image_points, (0, 160, 255)),
                                   ("CHECK", mapper.config.check_image_points, (255, 180, 30))):
        for index, point in enumerate(points):
            xy = tuple(np.rint(point).astype(int))
            excluded = prefix == "FIT" and not mapper.inliers[index]
            label = f"{prefix}{index + 1}" + (" OUT" if excluded else "")
            point_color = (30, 30, 255) if excluded else color
            cv2.circle(canvas, xy, 5, point_color, -1)
            cv2.putText(canvas, label, (xy[0] + 6, xy[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, point_color, 1, cv2.LINE_AA)
    label = "SYNTHETIC - NOT REAL METRIC VALIDATION" if mapper.synthetic else "MEASURED / CHECK CONFIG AND REPORT"
    cv2.rectangle(canvas, (0, 0), (width - 1, 38), (20, 20, 20), -1)
    cv2.putText(canvas, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
    return canvas
