"""YOLOv8 + ByteTrack 本地视频跟踪。"""

# 阅读顺序：__main__.py 参数 -> core.py 会话与视频循环 -> records.py 数据接口
# -> visualization.py 轨迹与显示。实际 Kalman 预测和 LAP 匹配使用 Ultralytics 实现。
# export.py 记录生命周期与表格；batch.py 隔离多个视频；demo.py 生成非实拍功能样例。
# motion.py 维护观测历史并估计 px/s 运动；motion_visualization.py 只绘制，不更新算法。
