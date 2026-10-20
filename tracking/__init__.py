"""YOLOv8 + ByteTrack 本地视频跟踪。"""

# 阅读顺序：__main__.py 参数 -> core.py 会话与视频循环 -> records.py 数据接口
# -> visualization.py 轨迹与显示。实际 Kalman 预测和 LAP 匹配使用 Ultralytics 实现。
