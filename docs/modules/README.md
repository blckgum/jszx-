# 模块原理笔记与源码阅读索引

面向掌握 Python 基础、正在学习 PyTorch 和计算机视觉的读者。每篇笔记按“解决什么问题 → 原理 → 代码入口 → 输入输出 → 运行与验证 → 易错点”组织，源码注释解释关键步骤及其原因。

## 1. 当前覆盖范围

这里的“已实现”指代码已有可运行基线，不表示已在校园实景上完成精度、测速或实时性验收。状态依据[项目 README](../../README.md)和现有源码整理。

| 大模块 | 已实现内容与边界 | 原理笔记 |
|---|---|---|
| 车辆与行人检测 | 图片/目录/视频检测；YOLO 标注审计、训练与验证测试入口；校园精度待评估 | [01：YOLOv8 与 PyTorch](01_目标检测_YOLOv8与PyTorch_解释文档.md) |
| 多目标跟踪 | 单视频独立会话、ByteTrack 关联、轨迹显示、JSONL/CSV 与生命周期导出、目录批处理；真实遮挡及身份指标待评估 | [02：ByteTrack](02_多目标跟踪_ByteTrack_解释文档.md) |
| 原有图像分类学习实验 | SimpleCNN / ResNet18、CIFAR-10 / 离线合成数据、训练评估、检查点及图片推理；作为独立学习流程保留 | [03：图像分类实验](03_图像分类实验_原理笔记.md) |

道路标定、米制测速、轨迹预测、TTC 风险决策和完整预警端目前处于规划阶段；技术设想见[课题需求与技术路线](../课题需求与技术路线.md)。当前不为这些阶段编写“已完成模块”笔记，也不把像素位移解释成实际车速。

## 2. 推荐阅读顺序

想理解当前道路感知主线，按下面顺序阅读：

1. 先读 01 的检测流程、坐标和类别映射，再看 `detection/core.py`。
2. 读 01 第 8、13 节，对照 `detection/dataset.py` 和 `detection/__main__.py` 理解数据审计、训练与指标。
3. 读 02 的关联机制、会话和时间轴，再看 `tracking/core.py`、`records.py`。
4. 读 02 第 6.4、8、11 节，对照 `tracking/export.py` 理解观测、事件、轨迹汇总的区别，再看批处理和可视化。

若对张量、梯度或优化器还不熟悉，可先读 03：原生 PyTorch 的训练循环更直观，再回到 Ultralytics 封装接口。两条路线不要求依次训练两种模型。

## 3. 模块与关键代码对应

| 阅读主题 | 源码入口 | 重点看什么 |
|---|---|---|
| 检测与结构化输出 | [`detection/core.py`](../../detection/core.py) | 名称筛选、原图框、空帧和视频时间 |
| 标注及数据划分审计 | [`detection/dataset.py`](../../detection/dataset.py) | images/labels 配对、归一化坐标、路径重叠与检查边界 |
| 检测训练及评估 | [`detection/__main__.py`](../../detection/__main__.py) | `train` / `val` 分派、类别一致性、按类指标 |
| 跟踪时间轴和会话 | [`tracking/core.py`](../../tracking/core.py) | `VideoClock`、`persist=True`、`retained_ids()`、异常保存 |
| 跟踪数据结构 | [`tracking/records.py`](../../tracking/records.py) | 有效 ID、像素锚点、空帧、会话键 |
| 事件和轨迹统计 | [`tracking/export.py`](../../tracking/export.py) | started/lost/recovered/closed、缺口计算、活动统计释放 |
| 可视化与暂停 | [`tracking/visualization.py`](../../tracking/visualization.py) | 有界轨迹、缺口断线、源 FPS 与处理 FPS |
| 批处理与功能样例 | [`tracking/batch.py`](../../tracking/batch.py)、[`demo.py`](../../tracking/demo.py) | 独立会话、失败继续、主动中断、构造素材边界 |
| 分类配置及基础工具 | [`config.py`](../../config.py)、[`common.py`](../../common.py) | 设备、随机源、超参数、加权平均 |
| 分类数据及模型 | [`dataset.py`](../../dataset.py)、[`model.py`](../../model.py) | 样本/批次、训练增强、张量形状、logits |
| 分类训练及推理 | [`train.py`](../../train.py)、[`predict.py`](../../predict.py) | 梯度五步、模式切换、检查点、Top-K |

根目录 `dataset.py` 是分类数据加载器，`detection/dataset.py` 是检测标注审计器；两者处理的数据格式不同。根目录 `config.py` 也不配置 YOLO/ByteTrack 命令，检测与跟踪参数来自 CLI 及 `configs/` 中的 YAML。

## 4. 后续完成模块时的文档约定

新增大模块时，复制[原理笔记模板](_模块原理笔记模板.md)，按实际实现补齐以下内容，并更新本索引与根 README 的链接：

- 说明目标、模块边界、输入输出的数据结构、坐标单位和时间单位。
- 把关键公式或算法步骤对应到具体函数；注明第三方库负责的部分。
- 在源码入口、关键状态变化和易错分支补中文注释，说明设计原因与限制；已有充分说明处按需更新。
- 给出可执行命令、代表性输入输出示例和检查办法，区分说明示例与实测结果。
- 记录实际执行的验证、通过/失败/跳过情况及尚未验证的条件。计划能力与已实现能力分别描述。

本约定用于后续开发时同步维护文档，没有增加自动执行器或 CI 强制检查。代码变化后应同步核对笔记中的默认值、字段和链接。

## 5. 如何核对笔记

现有检测、跟踪回归测试入口：

```bash
python -m unittest discover -s tests -v
```

集成测试需要本地 YOLOv8n 权重与随 Ultralytics 安装的 `bus.jpg`；缺少资源会跳过。分类基础流程可先执行 `python model.py` 检查形状，再参考 03 的离线运行方法。合成功能检查不能代替校园数据实验或 CIFAR-10 精度评价。

## 6. 本轮文档交付核验记录

2026-10-21，对照基线 `a31c390` 核对并补充中文解释，执行结果如下：

| 核验 | 本轮实际结果 |
|---|---|
| 现有回归测试 | `python -m unittest discover -s tests -v`：30 项全部通过、无跳过，包含真实 YOLOv8 + ByteTrack 构造视频集成测试 |
| 分类小样本接口检查 | CPU 上用 12 张合成图中的 8 张执行两批训练、4 张评估；确认参数更新、损失有限、检查点重载后的 Top-K 一致；另核对 ResNet18 的 `[1,10]` 输出及样本加权均值 |
| 处理逻辑核对 | 22 个 Python 文件的语法树与非注释 token 均与基线一致；本轮 16 个 Python 文件仅调整注释 |
| 文档检查 | 检查 Markdown 相对文件链接和代码围栏闭合；保留原有分类文件的 CRLF 换行，使用 `git -c core.whitespace=cr-at-eol diff --check` 检查补丁空白 |
| 检测权重 | 本地 YOLOv8n 权重的 SHA-256 在回归测试前后一致 |

分类检查调用现有训练、评估和推理接口，临时保存检查点后删除，没有运行完整 CIFAR-10 训练，也没有改写固定分类检查点。真实校园精度、速度/TTC、桌面窗口和 GPU 性能仍待对应条件下验证。
