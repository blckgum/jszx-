# BDD100K 与 TrackEval：数据集、评测工具和机位标定的区别

根据 2026-10-23 读取的官方项目 README、BDD100K 格式说明和 TrackEval BDD100K 适配器整理。本轮没有下载全量数据集、安装 TrackEval 或计算其跟踪指标。

## 1. 你提供的两个链接能用吗

| 项目 | 是什么 | 对本课题的用途 | 不能代替什么 |
|---|---|---|---|
| [BDD100K](https://github.com/bdd100k) | 道路场景多任务数据集及配套工具项目 | 检测训练/验证、视频目标跟踪实验、遮挡与天气场景分析 |不能替代实际固定机位的地面尺寸和对应点 |
| [TrackEval](https://github.com/JonathonLuiten/TrackEval) | 多目标跟踪评测代码 | 用带连续 ID 的真值评估 HOTA、IDF1、CLEAR/MOTA 等 | 不是数据集、不训练检测模型、不提供相机标定或真实车速真值 |

两者可以配合：BDD100K 提供数据与真值，YOLOv8+ByteTrack 产生预测，TrackEval 对齐后计算跟踪指标。当前标定开发可以独立进行，无需先下载全部 BDD100K。

GitHub 上的 BDD100K 仓库主要包含工具与说明；克隆仓库不等于下载全部影像。TrackEval 附带或链接的示例评测数据也不等于完整的检测训练素材。

## 2. BDD100K 建议分两类准备

### 检测实验

准备 detection 对应的图像和框标注，转换到当前 YOLO 类别和格式。先核对实际发布版本的类别名称，不把 COCO 的数值 ID 直接沿用成 BDD 的 ID。

训练、验证和测试应按官方划分或视频级划分，避免同一视频相邻帧进入不同集合。车载视角和校园固定路侧视角存在域差异，公开数据上的精度不等于校园部署精度。

### 跟踪实验

必须准备连续序列影像与 MOT/box-track 标注，标注中同一真实目标跨帧共享 ID。只下载单帧检测标签，不能据此算 IDF1/HOTA。

TrackEval 提供的 BDD100K 格式文档描述了 1400/200/400 个训练/验证/测试序列，约 40 秒、5 FPS、每段约 200 帧的 MOT 数据版本。实际下载时应核对官方当前发布版本和目录，不能默认所有 BDD 视频都以同一采样率标注。

建议先取验证集中的少量完整序列跑流程，再扩大实验。当前 tracking 命令主要读取本地视频；MOT 图片序列需要按官方顺序与采样时间正确封装成视频，或后续新增序列读取器。这个适配器尚未在本轮实现，不能把图片文件夹直接当作 tracking 的视频目录输入。

## 3. 类别与帧号是重点

TrackEval 的 BDD100K 适配器支持 pedestrian、rider、car、bus、truck、train、motorcycle、bicycle 八个评测类别，并按数据协议处理其他人员/车辆等干扰项与 crowd 区域。

| 当前 YOLO 类别 | BDD100K 跟踪类别 |
|---|---|
| person | pedestrian |
| car / bus / truck | 同名类别 |
| motorcycle / bicycle | 同名类别 |
| 当前未单独建模 | rider、train |

不能把 rider 自动并入 pedestrian 后仍宣称是官方完整八类评测。第一版可明确评价双方一致的六类，并报告类别范围；扩展官方全类别评测时要补齐检测与类别策略。不同版本的 detection 类别命名还可能不同，应读取实际标注确认。

本项目 frame_index 从 0 开始；外部格式可能按文件名、index 或其他规则组织。必须按**标注帧**对齐，处理采样率、缺帧、帧名、序列长度与空帧。不能把原视频 30 FPS 的逐帧结果直接当成 5 FPS 标注序列。

身份应结合 session/sequence 区分，不把每段视频的 ID 1 当成同一个全局目标。具体输出要遵循所选 TrackEval 适配器和官方提交格式，当前 tracks.jsonl/CSV 还不是可以直接交给 TrackEval 的格式。

## 4. TrackEval 能回答什么

| 指标 | 主要看什么 |
|---|---|
| HOTA / DetA / AssA | 综合跟踪，以及检测与关联贡献 |
| IDF1 / IDP / IDR | 预测身份与参考身份的一致性 |
| MOTA / 相关 CLEAR 指标 | 漏检、误检、ID 切换等影响 |

使用前需要预测与真值的类别、帧、坐标和序列对齐。应优先使用官方 BDD100K 适配器，保留其干扰项和 crowd 处理，不用自写简化公式冒充官方指标。

后续接入 TrackEval 应另建评测任务分支，完成格式转换、少量序列对照和版本记录。本轮只核对用途与文档，不把它混入标定实现，也不虚构任何 HOTA/IDF1 数值。

## 5. 为什么它们不能替代固定机位标定

BDD100K 主要是行车视角，摄像头随车辆运动。当前方案的单个道路 H 适用于固定机位、固定像素坐标系和近似平坦区域；不能给所有行车画面套用同一个静态 H，更不能借用其尺寸作为你校园道路的实测尺寸。

你真正需要补充的是：

1. 固定机位的一帧清晰画面，记录相机、分辨率、焦距与裁剪设置。
2. 多组可重复定位的地面参考点，以及实际测量的道路平面坐标，单位 m。
3. 另外的独立检查点，不参与拟合，用于验证近远处误差。
4. 镜头畸变明显时，同一镜头设置下的多姿态棋盘格图像或已有可信 K/D 参数。
5. 到实际测速阶段，再准备独立速度参考，而不是用模型自己的输出做真值。

因此当前可以先跑[合成标定演示与测试](modules/04_摄像头标定与道路坐标映射_解释文档.md)，BDD100K 后续用于感知性能实验；实景坐标精度必须回到你自己的固定机位验证。

## 6. 本次核对的官方资料

- [BDD100K 工具与资源仓库 README](https://github.com/bdd100k/bdd100k)
- [BDD100K 项目主页](https://www.bdd100k.com/)
- [TrackEval README：指标和支持的基准](https://github.com/JonathonLuiten/TrackEval)
- [TrackEval BDD100K 数据与提交格式](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/BDD100k-format.txt)
- [TrackEval BDD100K 适配器：评测类别、干扰项与数据读取](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/bdd100k.py)
