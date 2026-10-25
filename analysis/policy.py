"""按检测类别提供可解释的车辆类型与冲突安全半径策略。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectPolicy:
    kind: str
    safe_radius_veh_m: float
    label: str


PEDESTRIAN_POLICY = ObjectPolicy("pedestrian", 0.8, "行人")

# 注：以下安全半径仅为代码策略默认值，用于区分不同车型的几何包络先验，
# 不代表真实校园道路实测结论。
POLICIES = {
    "person": PEDESTRIAN_POLICY,
    "pedestrian": PEDESTRIAN_POLICY,
    "bicycle": ObjectPolicy("bicycle", 1.2, "非机动车·自行车"),
    "motorcycle": ObjectPolicy("motorcycle", 1.8, "机动车·摩托车"),
    "car": ObjectPolicy("car", 2.2, "机动车·汽车"),
    "bus": ObjectPolicy("bus", 2.8, "机动车·公交车"),
    "truck": ObjectPolicy("truck", 2.8, "机动车·卡车"),
    "vehicle": ObjectPolicy("motor_vehicle", 2.2, "机动车·车辆"),
    "motor_vehicle": ObjectPolicy("motor_vehicle", 2.2, "机动车·车辆"),
}


def object_policy(class_name: str | None, category: str = "vehicle") -> ObjectPolicy:
    """将模型类别映射到稳定策略；未知车辆保守使用机动车半径。"""
    if category in ("pedestrian", "person"):
        return PEDESTRIAN_POLICY
    key = (class_name or "vehicle").strip().lower()
    if key in ("person", "pedestrian"):
        return PEDESTRIAN_POLICY
    return POLICIES.get(key, POLICIES["vehicle"])
