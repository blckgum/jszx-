"""固定机位道路平面标定；只处理几何映射，不执行测速或风险判断。"""

# 阅读顺序：geometry.py 的齐次投影 -> lens.py 的坐标一致性 -> core.py 的映射契约
# -> io.py 配置/旧格式适配 -> pipeline.py 轨迹文件映射 -> __main__.py 命令。
