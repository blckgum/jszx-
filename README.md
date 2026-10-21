# 基于视觉目标检测与多目标跟踪的校园道路行人冲突感知与车速预警系统

面向校园道路固定路侧摄像头，研究车辆与行人的视觉感知、连续跟踪、实际速度估计和冲突风险预警。系统以 YOLOv8 与 PyTorch 为检测基础，通过 ByteTrack 维护连续目标身份，结合道路平面标定、轨迹预测及 TTC 分析，形成从视频输入到风险提示和事件记录的完整技术链路。

**当前阶段：车辆与行人检测、本地视频 ByteTrack 跟踪、道路标定测速、冲突分析与四级预警全链路已实现。** 已在 CPU 环境验证检测训练评估、连续 ID、短暂漏检恢复、跨视频会话重置与告警事件去抖；尚未完成校园实景标定、测速误差、实景精度或实时性评估。

## 1. 研究目标与实现状态

| 模块 | 方法与工具 | 预期输出 | 当前状态 |
|---|---|---|---|
| 视频采集与数据集构建 | 固定路侧摄像头、校园自采视频；OpenCV、FFmpeg、CVAT | 视频片段、图像与标注、数据划分 | 支持本地输入；校园数据待提供 |
| 车辆与行人检测 | YOLOv8n、PyTorch、Ultralytics | 类别、置信度、原图检测框 | **已实现基线** |
| 多目标跟踪 | ByteTrack、Kalman Filter、基于 IoU 的数据关联与 LAPJV 线性分配 | 目标 ID、连续观测位置、有限历史轨迹 | **已实现视频跟踪基线**；校园场景待验证 |
| 标定与运动状态估计 | OpenCV、单应性变换、滑窗微分与 EMA 轨迹滤波 | 道路坐标、运动方向、估计速度 | **已实现**；校园标定与测速精度待实景验证 |
| 轨迹预测与冲突分析 | 匀速模型（CVM）、速度向量外推、t_cpa/D_min 闭式求解 | 预测位置、冲突点坐标、预计最小间距 | **已实现**；待实景验证 |
| TTC 与风险决策 | 相对几何 TTC、四级风险分级、独立限速判定、事件去抖状态机 | 正常 / 超速 / 注意 / 紧急，生命周期告警事件 | **已实现**；阈值与去抖参数待实景调优 |
| 实时预警与展示 | OpenCV HUD、标注视频、结构化事件日志；可选 Streamlit 或 FastAPI | 实时画面、驾驶员及路旁学生提示、危险事件记录 | **已实现完整预警管线**；桌面窗口待本地实测，CPU 吞吐不足以承诺实时 |
| 实验评价 | 检测、跟踪、测速、延迟与事件级评价；计划使用 Pandas、Matplotlib 等汇总结果 | 实验报告、论文图表和答辩材料 | 已提供检测评估入口；完整实验待开展 |

研究重点是**道路标定与实际测速、车辆—行人时空冲突分析，以及 TTC 与风险决策**。YOLOv8 和 ByteTrack 分别提供检测与跟踪结果，`analysis` 模块据此完成物理测速、冲突研判、事件去抖与分级预警。

## 2. 技术流程

```mermaid
flowchart TD
    A[固定路侧摄像头 / 校园视频] --> B[抽帧、CVAT 标注与数据划分]
    B --> C[YOLOv8 车辆与行人检测]
    C --> D[类别、置信度、检测框与时间戳]
    D --> E[ByteTrack：Kalman 预测与数据关联]
    E --> F[相机标定、道路坐标与运动估计]
    F --> G[匀速外推与冲突区域判断]
    G --> H[TTC / 到达时间差与风险分级]
    F --> I[独立判断 30 km/h 限速]
    H --> J[画面提示、事件记录与预警 Demo]
    I --> J
```

标定测速、冲突分析与风险决策已由 `analysis` 模块实现，消费跟踪输出的轨迹锚点。实时预警是项目目标：当前 CPU 环境实测吞吐约 2.6~15 FPS，定位为离线分析，端到端延迟需要在目标硬件上测量后再评估实时性。

## 3. 核心算法

### 3.1 YOLOv8 目标检测：已接入

当前采用 Ultralytics 标准 **YOLOv8n** 检测模型和 COCO 预训练权重，通过 PyTorch 执行推理与训练。主要组成包括：

- **特征提取**：C2f 模块和 SPPF 空间金字塔池化。
- **多尺度融合**：FPN/PAN 式结构融合 P3、P4、P5 三个尺度的特征。
- **检测头**：Anchor-Free 设计，分类与边界框回归采用解耦分支。
- **训练机制**：Task-Aligned Assigner 进行标签分配，BCE 用于分类，CIoU 与 Distribution Focal Loss（DFL）用于边界框回归。
- **推理后处理**：图像缩放与填充、置信度筛选、非极大值抑制（NMS）及类别筛选。当前标准 YOLOv8 路径使用 NMS。

本仓库围绕该模型提供输入处理、道路类别筛选、原图坐标导出、数据检查及训练评估入口。YOLOv8s 可作为后续精度与计算成本对比的候选模型。

默认类别为 `person`、`bicycle`、`car`、`motorcycle`、`bus`、`truck`。导出保留模型细分类，同时将 `person` 归为 `pedestrian`，其余道路类别归为 `vehicle`；自行车、摩托车可在后续阶段单独分析。

### 3.2 ByteTrack 多目标跟踪：已接入

通过 Ultralytics `model.track()` 使用 ByteTrack 的高、低置信度检测框分阶段关联机制，利用 Kalman Filter 预测目标状态，以 IoU 等关联代价构建匹配矩阵，通过 `lap.lapjv()` 求解线性分配，并管理轨迹的建立、丢失和终止。

同一视频保持 `persist=True`；每个新视频创建独立会话和模型包装实例，清空历史，避免继承前一段视频的 ID。参数放在 `configs/bytetrack.yaml`，默认检测 conf 跟随低分关联阈值。短暂漏检恢复已作功能验证，真实遮挡和交叉通行下的稳定性仍需校园数据评估。

### 3.3 摄像机标定与实际测速：已实现

固定机位下，通过 `analysis.interactive_calibrator` 在含已知尺寸参照物的路面图像或视频首帧上选取 4 个角点，使用 OpenCV 单应性变换解算图像平面到道路平面的映射，参数持久化为 `configs/calibration.yaml`。目标的地面接触位置用检测框底边中心近似，投影为米制坐标后，经滑动窗口两端微分与指数移动平均 (EMA) 滤波计算速度：

```text
v(m/s)  = ‖ p(t) − p(t−Δt) ‖ / Δt        （窗口两端位置差分）
v(km/h) = 3.6 × v(m/s)                  （米每秒换算公里每小时）
```

位置使用标定后的米制坐标，时间来自源视频时间戳（容器 PTS 优先，缺失时按 FPS 推算）。已实现的保护机制包括：非有限观测不进入历史并回填末次有效位置、观测时间戳严格单调校验、窗口内时间戳无跨度按零速度处理、超现实速度跳变抑制、展示标量微动截断与未截断矢量分离（静止判定用矢量侧）、样本数加最小时间跨度的可信度门槛、连续低速多帧确认的泊车判定，以及类别抖动滞回。镜头畸变、道路非平面、遮挡及框位置抖动对实测测速误差的影响，仍需用具有参考速度的校园视频验证。

### 3.4 冲突分析、TTC 与预警：已实现

冲突研判采用短时匀速外推 (CVM)：由车辆与行人的相对位置、相对速度闭式求解最近接近时刻 t_cpa（即 TTC）与预计最小间距 D_min，结合安全距离、预测视界输出四级风险 `LEVEL_0_SAFE / LEVEL_1_SPEEDING / LEVEL_2_CAUTION / LEVEL_3_CRITICAL`。持续静止且速度估计已收敛的车辆豁免碰撞研判（未收敛目标保守参与），超过 35 m 的远端目标不评估。

**限速提醒与碰撞风险分级独立输出。** 限速参数默认 **30 km/h**：未超速车辆也可能因接近横穿行人产生高风险；未与行人汇聚的超速车辆同样产生独立超速事件。

**告警事件去抖状态机**：逐帧瞬时评估经连续 N 帧确认（默认 3 帧，支持秒级换算）、等级滞回（峰值只升不降）与冷却合并（静默窗内复现视为同一事件）收敛为生命周期事件，消除单帧抖动产生的告警风暴。confirmed 事件与 provisional 瞬时帧在 HUD、徽章与日志中明确区分，下游声光/网络通知应只消费 `events.jsonl` 的 confirmed 事件流。风险阈值、限速容差与确认/冷却参数需要校园实景实验调优。

## 4. 当前功能

- 本地单张图片、图片目录和视频检测；按名称筛选道路目标类别。
- 保存带框图片或 MP4，以及逐帧 `detections.jsonl` 和运行汇总 `summary.json`。
- 为没有目标的帧保留记录，导出原图像素坐标、帧号及源时间信息。
- 本地 YOLO 标注数据检查：类别映射、标签格式、缺少标签及训练/验证文件路径重叠。
- 自定义数据训练、验证与测试入口，导出整体及有标注类别的检测指标。
- ByteTrack 连续视频跟踪，显示类别、置信度、ID、分组颜色、轨迹、源 FPS、处理 FPS、帧号和当前各类数量。
- 跟踪结果流式导出 JSONL、CSV、丢失/恢复事件及逐轨迹汇总；可选桌面 `--show` 模式支持 Space/P 暂停继续、Q/Esc 退出，默认无窗口保存视频。
- 目录顺序批量跟踪，每段新建会话；单个文件失败继续处理，其错误写入批次清单。
- 推理异常或 Ctrl+C 时尝试保存已完成视频与错误汇总；提供不需要数据集的构造样例生成器。
- 冲突感知与测速预警完整链路：`python -m analysis --source <video>` 一次完成标定投影、物理测速、TTC 研判、事件去抖、HUD 渲染与结构化日志落盘。
- 交互式 4 点路面标定工具 `python -m analysis.interactive_calibrator`；`--demo` 提供仅供流程演示的合成标定，非 demo 模式拒绝加载。
- 逐帧原始评估 `conflicts.jsonl`、生命周期事件 `events.jsonl`、逐目标 `speeds.csv` 与运行摘要 `summary.json`；异常/中断路径同样归档未决事件并落盘带 `status` 的摘要。
- HUD 与徽章区分 `[PROVISIONAL]`/`[CONFIRMED]` 告警语义；`--show` 桌面窗口支持 q 退出并在事件日志中如实记录结束原因。

当前输入入口为本地文件和目录，摄像头、网络流和网页预警端尚未接入。检测入口的 `track_id` 仍为 `null`；跟踪入口输出有效 ID，未获得 ID 的返回框保留 null。当前云机器没有桌面，按键控制已通过模拟事件测试，实际窗口仍需本地验证。ByteTrack 换 ID 会把同一物理事件拆成多条记录（事件日志保留 track_id 供事后归并）；停车区/行车道分区豁免与非机动车分级需真实校园素材后定案。

## 5. 安装与快速使用

已验证环境：Python 3.11.9、PyTorch 2.11.0、torchvision 0.26.0、Ultralytics 8.4.182、opencv-python 5.0.0.93、lap 0.10.0（本地为 CUDA 12.8 构建，云 CPU 环境验证过 +cpu 构建）。模型版本由 `yolov8n.pt` 指定，Ultralytics 包版本不代表 YOLO 模型版本。

跟踪新增依赖 `lap`，已加入 `requirements-yolo.txt`，用于 ByteTrack 的线性分配。

```bash
git clone https://github.com/blckgum/jszx-.git
cd jszx-
python -m venv .venv
source .venv/bin/activate

# CPU 安装示例；使用 GPU 时先按硬件安装匹配的 CUDA 版 PyTorch
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-yolo.txt
python -m pip check
```

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活虚拟环境。计划在 RTX 5060 上开展 GPU 训练和推理实验，应按 PyTorch 官方支持情况选择对应 CUDA 构建；当前云环境的验证使用 CPU。

将下面的 `--source` 替换为实际文件或目录。首次运行需要联网下载官方权重，默认保存到 `checkpoints/yolo/yolov8n.pt`；也可以通过 `--weights` 指定已有权重。

```bash
python -m detection detect --source path/to/image.jpg --device cpu
python -m detection detect --source path/to/image_folder --device cpu
python -m detection detect --source path/to/video.mp4 --device cpu

# 按类别筛选，仅处理前 100 帧
python -m detection detect --source path/to/video.mp4 --classes person car bus truck --max-frames 100 --device cpu

# 跟踪：默认无窗口保存视频和逐帧数据
python -m tracking --source path/to/video.mp4 --tracker configs/bytetrack.yaml --device cpu

# 目录批量跟踪：每段独立输出，目录只读取一层
python -m tracking --source path/to/videos --device cpu

# 尚无实拍视频时：生成非实拍功能样例，再运行真实模型和跟踪器
python -m tracking.demo --output runs/tracking-fixtures/sample_01
python -m tracking --source runs/tracking-fixtures/sample_01 --device cpu --output runs/yolo/track-batch/sample_01

# 交互式路面标定：在含已知尺寸参照物的路面图像/首帧上取 4 个角点
python -m analysis.interactive_calibrator --source path/to/road.jpg --width 6.0 --length 15.0

# 端到端冲突感知与预警（默认读取 configs/calibration.yaml）
python -m analysis --source path/to/video.mp4 --device cpu --show

# 无实测标定时的流程演示（内存合成标定，物理量不可信，仅验证流程）
python -m analysis --source path/to/video.mp4 --demo --output runs/analysis/demo
```

| 参数 | 用途 | 默认值 |
|---|---|---|
| `--weights` | 检测模型权重 | `checkpoints/yolo/yolov8n.pt` |
| `--conf` | 检测置信度阈值 | `0.25` |
| `--iou` | NMS 的 IoU 阈值 | `0.7` |
| `--imgsz` | 模型输入尺寸 | `640` |
| `--device` | `cpu` 或 GPU 编号，如 `0` | 自动选择 |
| `--output` | 指定一个尚不存在的输出目录 | 自动创建独立目录 |
| `--fps` | 显式覆盖视频 FPS 及固定间隔时间戳 | 优先使用视频元数据 |

上述参数表描述检测入口。跟踪入口的 `--conf` 默认取 `configs/bytetrack.yaml` 中的 `track_low_thresh=0.1`，并拒绝更高值，保留低分关联候选。跟踪可追加 `--show` 启用桌面窗口，当前默认无界面。

预警分析入口 `python -m analysis` 的主要参数：

| 参数 | 用途 | 默认值 |
|---|---|---|
| `--calib` | 单应性标定 YAML 路径 | `configs/calibration.yaml` |
| `--demo` | 使用内存合成占位标定（仅流程演示） | 关闭 |
| `--allow-legacy-calib` | 加载无 `synthetic` 溯源字段的旧版标定 | 拒绝 |
| `--tracker` | ByteTrack 参数 YAML | `configs/bytetrack.yaml` |
| `--fps` | 源视频无有效 FPS 时显式指定 | 使用视频元数据 |
| `--speed-limit` | 校园限速 km/h | `30` |
| `--ema-alpha` | EMA 平滑因子 | `0.35` |
| `--window` / `--window-s` | 测速滑窗帧数/秒数（秒级覆盖帧级） | `8` 帧 |
| `--ttc-critical` / `--ttc-caution` / `--horizon` | TTC 分级阈值与预测视界秒数 | `1.8` / `3.5` / `3.5` |
| `--parked-speed` | 静止判定速度阈值 m/s | `0.1` |
| `--parked-frames` / `--parked-seconds` | 静止车辆确认帧数/秒数（秒级覆盖帧级） | `3.0` 秒 |
| `--alert-confirm` / `--alert-confirm-s` | 告警确认帧数/秒数（秒级覆盖帧级） | `3` 帧 |
| `--alert-cooldown` / `--alert-cooldown-s` | 告警冷却合并帧数/秒数（秒级覆盖帧级） | `12` 帧 |
| `--reliable-span-s` | 速度可信度最小观测时间跨度 | `0.2` 秒 |
| `--max-frames` | 最大处理帧数（0 表零帧） | 不限 |
| `--show` | 桌面窗口预览（q 退出） | 关闭 |

## 6. 数据集与训练评估

数据来源计划采用 **校园固定路侧自采视频为主、公开道路数据为补充**。BDD100K 可作为候选公开数据，但其车载视角与固定路侧视角存在差异，需要通过校园数据适配与独立评估验证效果。抽帧、片段筛选、CVAT 标注及 FFmpeg 处理属于数据准备流程。

当前入口接受本地 YOLO 检测标注。COCO 或 BDD100K 的原始标注需先转换，不能直接把其 JSON 文件传入本入口。

```text
data/traffic/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

图片与标签同名，每个目标占一行：`class_id x_center y_center width height`，四个坐标值按图像宽高归一化。无目标图片提供空标签文件。默认自定义六类顺序为 `0 person / 1 bicycle / 2 car / 3 motorcycle / 4 bus / 5 truck`；它与 COCO 原始类别 ID 不同，转换时必须重新映射。

使用 [数据集配置模板](configs/traffic.example.yaml) 修改 `path` 后运行。相对 `path` 以 YAML 所在目录为基准；`train/val/test` 以该 `path` 为基准。同一视频及其相邻帧按视频、时间段或机位划分，避免跨集合泄漏。

```bash
# 训练；GPU 可将 --device cpu 改为 --device 0
python -m detection train --data configs/traffic.example.yaml --epochs 50 --batch 8 --imgsz 640 --device cpu

# 将 run-xxxx 替换为实际训练输出目录
python -m detection val --data configs/traffic.example.yaml --weights runs/yolo/train/run-xxxx/weights/best.pt --split val --device cpu
python -m detection val --data configs/traffic.example.yaml --weights runs/yolo/train/run-xxxx/weights/best.pt --split test --device cpu
```

训练权重保存为 `weights/best.pt` 和 `weights/last.pt`，指标写入 `metrics.json`。评估时要求模型与数据集的 `names` 完全一致。验证集用于选参数，测试集用于最终报告。

## 7. 输出与实验依据

检测结果默认写入 `runs/yolo/detect/run-*`。JSONL 每帧包含输入来源、原图宽高、`frame_index`、`timestamp_s`、时间戳来源和目标列表；每个目标包含 `class_id`、`class_name`、`category`、`confidence`、`bbox_xyxy`、`track_id`。

独立图片没有时间戳；视频在源时间戳不可用时使用 FPS 估计，并记录 `fps_estimate`。标注 MP4 使用固定帧率，后续运动计算以逐帧数据中的源时间信息为准。检测框累计数量不能直接作为独立目标数量。

跟踪输出位于 `runs/yolo/track/run-*`，包含 `annotated.mp4`、`tracks.jsonl`、`observations.csv`、`events.jsonl`、`tracklets.jsonl`、`summary.json` 和 `tracker.resolved.yaml`。目录输入默认输出到 `runs/yolo/track-batch/run-*`，根目录另有 `batch_summary.json`，每段视频放在独立子目录。逐帧目标还提供 ID、框底边中心像素点和观测标记，使用 `(session_id, track_id)` 区分不同视频；当前不输出实际速度或 TTC。

预警分析输出位于 `runs/analysis/run-*`（或 `--output` 指定目录），包含 `annotated_conflict.mp4`、`conflicts.jsonl`、`events.jsonl`、`speeds.csv` 与 `summary.json`，demo 模式另有 `calibration.synthetic.yaml` 留档。`conflicts.jsonl` 保留逐帧原始评估并在确认后回填 `event_id`；`events.jsonl` 每行一个生命周期事件，记录首末帧/时间戳、峰值等级、最小 TTC、最高车速与结束原因（`cooldown`/`eof`/`max_frames`/`user_quit`/`interrupt`/`exception`）；`speeds.csv` 逐目标记录位置、速度、风险、事件关联与可靠度标记；`summary.json` 汇总事件计数，异常或中断退出时同样落盘并携带 `status`/`stop_reason`/`error`/`frames_processed` 字段，凭输出目录即可区分正常完成与异常截断。

| 评价对象 | 指标与验证方法 | 状态 |
|---|---|---|
| 检测 | Precision、Recall、mAP50、mAP50-95；分目标类别及难例场景统计 | 已实现指标导出，校园数据结果待测 |
| 跟踪 | IDF1、HOTA、ID 切换数 | 已完成功能验证；定量指标待实拍视频和连续 ID 标注 |
| 测速 | MAE、RMSE；对比参考速度及不同标定位置 | 管线已实现；误差验证待实测标定与参考数据 |
| 冲突预警 | 事件级召回率、误报率、预警提前量与端到端延迟 | 管线已实现；指标待校园实景冲突事件标注 |

已完成的功能验证：官方样例检出行人、公交车、汽车、自行车和卡车；3 帧拼接视频验证空帧保留、时间戳和输出解码；COCO128 重映射小样本按 4/4/2 张划分，完成单轮训练及 val/test 入口验证；6 项回归测试通过。小样本单轮微调后的指标较低，仅用于检查流程，不能代表校园检测效果。GPU 性能、真实道路动态视频及完整预警效果尚未验证。

跟踪新增验证：3 帧视频中的公交车在空白帧前后保持 ID；两段各 18 帧的构造视频中，多车多人同时关联，公交车在两个空白帧后恢复原 ID；同进程处理第二段视频时正确新建会话并重新分配 ID。

预警分析新增验证：标定投影精度与 YAML 溯源门控、测速 EMA/预热/可靠性/类别滞回/时间戳单调性、TTC 分级与静止豁免、事件状态机确认/滞回/冷却合并/单调性校验、端到端管线（含异常落盘、初始化资源释放、max_frames/退出截断语义、CSV 事件关联）及 CLI 参数校验均有覆盖。

当前总测试数为 **107 项，全部通过、无跳过**，包含真实网络集成、长时间丢失与临界恢复、生命周期和 CSV、坏视频批量处理、异常/中断保存及模拟按键测试。命令行已实际完成两段各 18 帧构造样例的 CPU 批量处理与输出核对。构造视频不是校园实拍基准，未据此报告 IDF1/HOTA、测速精度或承诺实时性能。

```bash
python -m unittest discover -s tests -v
```

## 8. 仓库结构

```text
jszx-/
├── detection/                    # 当前目标检测模块
│   ├── __main__.py                # detect / train / val 入口
│   ├── core.py                    # 推理、可视化、逐帧导出
│   └── dataset.py                 # 数据集与标签检查
├── configs/traffic.example.yaml   # YOLO 数据配置模板
├── configs/bytetrack.yaml         # 独立 ByteTrack 参数
├── configs/calibration.yaml       # 路面单应性标定（默认合成占位，需实测替换）
├── tracking/                      # 独立会话、轨迹记录、批处理、构造样例及可视化
├── analysis/                      # 标定、测速、冲突研判、事件去抖与预警渲染
│   ├── calibration.py             # 单应性标定与 YAML 溯源
│   ├── speed.py                   # 滑窗微分 + EMA 测速、可靠性/静止判定
│   ├── conflict.py                # CVM 外推、t_cpa/D_min 闭式解与四级风险
│   ├── events.py                  # 告警事件去抖状态机（确认/滞回/冷却合并）
│   ├── pipeline.py                # 端到端处理、日志落盘与资源管理
│   ├── visualization.py           # HUD、徽章、预测轨迹与冲突连线
│   ├── interactive_calibrator.py  # 交互式 4 点标定 CLI
│   └── __main__.py                # 参数、校验与入口
├── tests/                         # 检测、跟踪机制与真实网络集成测试
├── requirements-yolo.txt          # 检测/跟踪/分析模块依赖
├── docs/                         # 课题需求、技术路线、研究蓝图及模块说明
├── common.py / config.py          # 原有分类实验工具与配置
├── model.py / dataset.py          # 原有 SimpleCNN / ResNet18 分类实验
└── train.py / predict.py          # 原有分类训练与推理入口
```

检测、训练和评估使用 `python -m detection`，视频跟踪使用 `python -m tracking`，标定与预警分析使用 `python -m analysis` 和 `python -m analysis.interactive_calibrator`。`data/`、`checkpoints/`、`runs/` 为本地数据、权重和生成结果，不随 Git 提交。

## 9. 开发路线与参考资料

实施顺序：**检测基线及场景数据 → ByteTrack 跟踪 → 道路标定和测速 → 冲突预测与 TTC 风险分级（以上均已实现）→ 校园实景标定与实测调参 → 实验评价与消融对比**。第一版已形成可验证的完整流程；当前 CPU 吞吐定位为离线分析，实时性需在目标硬件上实测后再评估。

- [课题需求与技术路线](docs/课题需求与技术路线.md)
- [课题全景蓝图与数学推导](docs/MASTER_RESEARCH_PROMPT.md)
- [检测模块实施说明](docs/车辆行人检测方案.md)
- [目标检测 YOLOv8 与 PyTorch 解释文档](docs/modules/01_目标检测_YOLOv8与PyTorch_解释文档.md)
- [多目标跟踪 ByteTrack 解释文档](docs/modules/02_多目标跟踪_ByteTrack_解释文档.md)
- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/models/yolov8/)
- [PyTorch 安装说明](https://pytorch.org/get-started/locally/)
- [ByteTrack 项目](https://github.com/FoundationVision/ByteTrack)
