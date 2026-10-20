# 基于视觉目标检测与多目标跟踪的校园道路车辆—行人冲突感知与实时预警系统
# 课题全景总览、工程实现规范与系统运行指引 (Master Research Blueprint & Prompt)

---

## 1. 课题核心定位与场景边界

### 1.1 课题正式名称
基于视觉目标检测与多目标跟踪的校园道路车辆—行人冲突感知与实时预警系统
(Campus Road Vehicle-Pedestrian Conflict Perception and Real-time Warning System Based on Visual Object Detection and Multi-Object Tracking)

### 1.2 核心应用场景与物理约束
* 部署形态：校园内部主要出入道路、交叉路口或林荫道旁固定式路侧摄像头（路侧边缘计算节点）。
* 限速环境：校园内部严格执行 30 km/h 最高限速标准。
* 感知目标：机动车（轿车、SUV、校车、货车等）与行人（横穿学生、沿道路行走人员）。
* 核心任务：实现从单一画面检测向时空轨迹推演与物理冲突风险研判的纵深跨越。

### 1.3 明确的系统边界定义
* 明确做的工作 (In Scope)：
  1. 基于 YOLOv8 的实时机动车与行人高精度目标检测。
  2. 基于 ByteTrack 的跨帧目标关联、稳健 ID 分配与遮挡轨迹恢复。
  3. 基于单应性矩阵 (Homography) 的相机像素到物理地面坐标 (X, Y) 映射。
  4. 基于物理位移微分与指数移动平均 (EMA) 的车辆与行人平滑测速。
  5. 基于二阶运动学外推 (CVM) 的轨迹交叉判定与碰撞时间 (TTC) 闭式求解。
  6. 结合 30 km/h 限速与 TTC 的四级预警体系建立与多端可视化联动。
  7. 结构化日志落盘 (JSONL 冲突事件、CSV 速度轨迹)。
* 坚决不做的工作 (Out of Scope)：
  1. 不做底层车辆动力学控制（如刹车执行器、方向盘转向控制等自动驾驶闭环）。
  2. 不做全量城市道路高精地图匹配与复杂多色红绿灯全套判读。
  3. 不做 LiDAR 激光雷达点云融合（纯视觉路侧方案，追求低成本与一年内落地推广）。

---

## 2. 核心数学模型与算法推导

### 2.1 空间单应性标定与逆透视映射 (Homography Calibration)
针对斜向下俯视安装的固定路旁相机，将图像二维像素坐标 $(u, v)$ 转换到实际地面物理米制坐标 $(X, Y)$：
$$
\begin{bmatrix}
x' \\
y' \\
w'
\end{bmatrix}
=
\mathbf{H}
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}
=
\begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
\begin{bmatrix}
u \\
v \\
1
\end{bmatrix}
$$
物理地面坐标为齐次归一化结果：
$$
X = \frac{x'}{w'} = \frac{h_{11}u + h_{12}v + h_{13}}{h_{31}u + h_{32}v + h_{33}}, \quad
Y = \frac{y'}{w'} = \frac{h_{21}u + h_{22}v + h_{23}}{h_{31}u + h_{32}v + h_{33}}
$$
通过在路面标定已知物理尺寸（如长 $L$ 米、宽 $W$ 米的标准矩形区域）的 4 个特征角点，利用直接线性变换 (DLT) 精确解算 $3 \times 3$ 单应性矩阵 $\mathbf{H}$ 及其逆矩阵 $\mathbf{H}^{-1}$。

### 2.2 物理时速平滑微分与抖动抑制 (Speed Estimation & EMA Filter)
设时间戳序列为 $t_k$，目标接触地面的锚点映射物理坐标为 $(X_k, Y_k)$。
采用滑窗区间求导消除单帧边界框接触点抖动：
$$
v_{x, \text{raw}} = \frac{X_k - X_{k-m}}{t_k - t_{k-m}}, \quad
v_{y, \text{raw}} = \frac{Y_k - Y_{k-m}}{t_k - t_{k-m}}
$$
引入指数移动平均 (EMA) 对速度矢量进行平滑滤波（平滑因子 $\alpha \in [0.3, 0.4]$）：
$$
\bar{v}_{x, k} = \alpha v_{x, \text{raw}} + (1 - \alpha) \bar{v}_{x, k-1}
$$
$$
\bar{v}_{y, k} = \alpha v_{y, \text{raw}} + (1 - \alpha) \bar{v}_{y, k-1}
$$
计算最终标量时速 (km/h)：
$$
v_{\text{km/h}} = 3.6 \times \sqrt{\bar{v}_{x, k}^2 + \bar{v}_{y, k}^2}
$$

### 2.3 车辆—行人时空冲突推演与 TTC 闭式解 (Conflict & TTC Model)
在未来短视界时间窗 $t \in [0, T_{\text{horizon}}]$ 内，采用等速外推模型 (Constant Velocity Model, CVM)：
$$
\mathbf{p}_{\text{veh}}(t) = \mathbf{p}_{\text{veh}}(0) + \mathbf{v}_{\text{veh}} t
$$
$$
\mathbf{p}_{\text{ped}}(t) = \mathbf{p}_{\text{ped}}(0) + \mathbf{v}_{\text{ped}} t
$$
定义相对初始位移矢量与相对运动速度矢量：
$$
\mathbf{r} = \mathbf{p}_{\text{ped}}(0) - \mathbf{p}_{\text{veh}}(0) = (x_{\text{ped}} - x_{\text{veh}}, y_{\text{ped}} - y_{\text{veh}})
$$
$$
\mathbf{v}_{\text{rel}} = \mathbf{v}_{\text{ped}} - \mathbf{v}_{\text{veh}} = (v_{x, \text{ped}} - v_{x, \text{veh}}, v_{y, \text{ped}} - v_{y, \text{veh}})
$$
未来时刻 $t$ 时两者空间欧氏距离的平方函数为：
$$
D^2(t) = \|\mathbf{r} + \mathbf{v}_{\text{rel}} t\|^2 = \|\mathbf{r}\|^2 + 2 (\mathbf{r} \cdot \mathbf{v}_{\text{rel}}) t + \|\mathbf{v}_{\text{rel}}\|^2 t^2
$$
1. 汇聚运动条件判定：
   若 $\mathbf{r} \cdot \mathbf{v}_{\text{rel}} \ge 0$，说明两目标运动趋势为彼此远离，距离随时间递增，无碰撞风险。
   若 $\mathbf{r} \cdot \mathbf{v}_{\text{rel}} < 0$，两目标正在相互逼近，存在几何汇聚冲突。
2. 碰撞时间 (TTC / $t_{\text{cpa}}$) 闭式求解：
   令 $\frac{d(D^2(t))}{dt} = 0$，求得距离极小值对应时刻：
   $$
   t_{\text{cpa}} = -\frac{\mathbf{r} \cdot \mathbf{v}_{\text{rel}}}{\|\mathbf{v}_{\text{rel}}\|^2}
   $$
3. 预测最近相遇间距 $D_{\min}$：
   $$
   D_{\min} = \sqrt{\|\mathbf{r}\|^2 - \frac{(\mathbf{r} \cdot \mathbf{v}_{\text{rel}})^2}{\|\mathbf{v}_{\text{rel}}\|^2}}
   $$

### 2.4 四级分层预警准则 (Risk Hierarchy)
| 预警级别 | 状态标识 | 触发逻辑与物理阈值 | 现场处置指导 |
| :--- | :--- | :--- | :--- |
| Level 0 | 正常绿色 | 车辆 $v \le 30 \text{ km/h}$ 且无汇聚行人，或行人背向远离 | 正常受控通行 |
| Level 1 | 超速黄色 | 车辆 $v > 30 \text{ km/h}$，但当前视场内无交互横穿行人 | 提示驾驶员减速至 30 内 |
| Level 2 | 减速橙色 | 轨迹汇聚， $1.8\text{s} < t_{\text{cpa}} \le 3.5\text{s}$ 且 $D_{\min} \le 4.5\text{m}$ | 双端预警：车内减速、路旁注意 |
| Level 3 | 紧急红色 | 紧急冲突！ $t_{\text{cpa}} \le 1.8\text{s}$ 且 $D_{\min} \le 3.0\text{m}$ | 紧急蜂鸣报警，路侧屏爆闪 |

---

## 3. 本地工程架构与核心模块清单

项目根目录位于：`D:\AI\jszx_partner`

```text
D:\AI\jszx_partner\
├── checkpoints\
│   └── yolo\
│       └── yolov8n.pt             # 官方检测基底权重
├── configs\
│   ├── bytetrack.yaml             # ByteTrack 核心追踪超参
│   └── calibration.yaml          # 道路地面单应性变换参数文件
├── detection\
│   └── core.py                    # YOLOv8 推理接口与数据结构校验
├── tracking\
│   ├── core.py                    # ByteTrack 会话跟踪引擎
│   ├── records.py                 # 跨帧轨迹状态模型
│   └── demo.py                    # 模拟场景生成器
├── analysis\                      # [本次完整构建的核心研判体系]
│   ├── calibration.py             # 4点单应性透视映射 (H矩阵求逆/正反变换)
│   ├── speed.py                   # 物理空间 EMA 平滑测速与 30km/h 限速研判
│   ├── conflict.py                # CVM 轨迹外推、TTC 计算与四级预警决策
│   ├── visualization.py           # 增强 HUD 看板、未来虚线轨迹、碰撞热区圈
│   ├── interactive_calibrator.py  # 图像路面 4 点交互式标定工具
│   ├── pipeline.py                # 跟踪+测速+冲突+渲染端到端管道
│   ├── __init__.py                # 模块导出定义
│   └── __main__.py                # 统一命令行入口
├── tests\                         # 45 项单元与集成测试（100% 通过）
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_tracking_exports.py
│   ├── test_tracking_integration.py
│   └── test_analysis.py           # 空间标定、测速与 TTC 冲突单元测试
└── docs\
    └── MASTER_RESEARCH_PROMPT.md  # 本指导书
```

---

## 4. 命令行运行指引与操作规范

使用 Python 环境：`D:\AI\plant_yolo_env\Scripts\python.exe`

### 4.1 运行所有单元测试（验证系统完好性）
```powershell
cd D:\AI\jszx_partner
D:\AI\plant_yolo_env\Scripts\python.exe -m unittest discover -s tests -v
```

### 4.2 进行路面单应性 4 点标定
若已有校园道路空景图（例如 `campus_road.jpg`），使用交互式标定工具获取实际长宽和 4 个角点：
```powershell
cd D:\AI\jszx_partner
D:\AI\plant_yolo_env\Scripts\python.exe -m analysis.interactive_calibrator --image campus_road.jpg --width 6.0 --length 15.0 --output configs/calibration.yaml
```
操作说明：在弹出窗口中依次点击路面参考矩形的四个角点：
1. 左上角 (Top-Left)
2. 右上角 (Top-Right)
3. 右下角 (Bottom-Right)
4. 左下角 (Bottom-Left)
按 `s` 保存并生成 `configs/calibration.yaml`。

### 4.3 运行全流程端到端冲突感知与预警分析
```powershell
cd D:\AI\jszx_partner
D:\AI\plant_yolo_env\Scripts\python.exe -m analysis --source 你的校园视频.mp4 --calib configs/calibration.yaml --output runs/analysis/test_result --device 0
```
如需在屏幕上实时播放预览，添加 `--show` 参数：
```powershell
D:\AI\plant_yolo_env\Scripts\python.exe -m analysis --source 你的校园视频.mp4 --show
```

---

## 5. 校园实地数据采集与拍摄规范

为了确保空间标定与速度估算的物理精度，拍摄时请遵循以下实操标准：
1. 机位选址与视角：
   - 选取校园主干道直道或丁字路口，推荐将设备架设在教学楼 2~3 层阳台、天桥栏杆或 3 米以上路灯立杆处。
   - 保持斜向下俯视角度约 15° ~ 30°，避免极端平视（否则远近车辆重叠严重、遮挡严重）。
2. 地面基准参照物测量：
   - 拍摄前使用卷尺或步测，在路面上选定一组 4 点矩形标记（例如斑马线边缘、车位标线、减速带或两处地砖接缝）。
   - 记录该矩形的实际物理宽度（如道路宽 6.0 米）和纵向物理长度（如 15.0 米）。
3. 图像拍摄要求：
   - 先拍一张路面完全无人、无车的清晰空景照（用于生成单应性标定文件）。
   - 固定好三脚架或手机支架，禁止录制过程中随意晃动或变焦（必须保持焦距与机位固定）。
   - 录制 1~3 分钟真实校园机动车通行与同学过马路的视频。

---

## 6. 研究性课题论文与答辩亮点提炼

1. 边端算力友好：单路 1080P/720P 视觉流即可完成实时解算，硬件成本极低，极适合在校园普遍推广。
2. 物理与视觉深度融合：突破了纯图像目标检测“只能画框不能测距”的瓶颈，通过单应性投影和 EMA 微分恢复现实物理速度与相对几何运动。
3. 超速与动态冲突双层解耦解算：即便车辆在 30 km/h 法定限速内行驶，只要与横穿行人轨迹产生时空交叉且 TTC 处于危险阈值，系统仍能精准触发预警，从根本上防止传统“未超速但撞人”的恶性盲区事故。
