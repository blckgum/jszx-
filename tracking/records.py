"""逐帧观测接口；保存像素位置，不在本模块推算实际速度或 TTC。"""

# 原理笔记：docs/modules/02_多目标跟踪_ByteTrack_解释文档.md，第 4、8、11 节。
# dataclass 汇集字段，asdict 转为可写入 JSON 的普通结构；frozen 禁止字段重新赋值，
# 但不会把 tracks 列表也变成不可变对象，消费方仍应把记录当作只读数据。

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TargetRecord:
    # bbox_xyxy 为原图左上/右下角浮点像素坐标；anchor_xy 为底边中心，均不是米。
    track_id: int | None
    class_id: int
    class_name: str
    category: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    anchor_xy: tuple[float, float]
    anchor_type: str = "bbox_bottom_center"
    observation: str = "detected"


@dataclass(frozen=True)
class FrameRecord:
    # ID 只在一个会话内有效。重复运行同一视频，也需要不同 session_id。
    session_id: str
    source: str
    frame_index: int  # 从 0 开始；tracks=[] 仍保留这一帧及时间信息。
    timestamp_s: float
    timestamp_source: str
    width: int
    height: int
    tracks: list[TargetRecord] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)


def extract_tracks(result) -> list[TargetRecord]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    # 一次搬回 CPU，再转 Python 数字，避免每个框分别从 GPU 读取多个标量。
    boxes = boxes.cpu()
    # 框未获得有效跟踪 ID 时保留 None，不把检测列表下标当作 ID。
    ids = boxes.id.tolist() if boxes.is_track else [None] * len(boxes)
    records = []
    for coords, score, cls, track_id in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), ids
    ):
        class_id = int(cls)
        # 必须使用当前权重的 names；自定义六类的 ID 顺序可能与 COCO 完全不同。
        name = str(result.names[class_id])
        x1, y1, x2, y2 = map(float, coords)
        records.append(TargetRecord(
            track_id=int(track_id) if track_id is not None else None,
            class_id=class_id, class_name=name,
            category="pedestrian" if name.lower() == "person" else "vehicle",
            confidence=float(score), bbox_xyxy=(x1, y1, x2, y2),
            # 底边中心只是地面接触位置的图像近似，尚未转换为米制道路坐标。
            anchor_xy=((x1 + x2) / 2, y2),
        ))
    return records
