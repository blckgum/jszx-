"""按检测类别提供可解释的车辆类型与冲突安全半径策略。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import yaml


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

DEFAULT_OBJECT_POLICY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "object_policy.yaml"


def read_object_policy_config(path: str | Path = DEFAULT_OBJECT_POLICY_CONFIG) -> dict[str, ObjectPolicy]:
    """读取策略 YAML；策略值是默认先验，不等同于现场测量值。"""
    # 拒绝空白路径字符串，防止 Path("") 解析为当前工作目录 '.'。
    if isinstance(path, str) and not path.strip():
        raise ValueError("对象策略配置路径不能为空")
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"对象策略配置文件不存在: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"对象策略 YAML 格式错误: {exc}") from None
    # 精确校验 schema_version 为整数 1（排除 True/1.0），避免把配置版本错误静默放行。
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
    ):
        raise ValueError("对象策略必须包含 schema_version: 1")
    raw = data.get("policies")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("对象策略必须包含非空 policies 映射")
    policies = {}
    for name, values in raw.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(values, dict):
            raise ValueError("对象策略名称和值必须是映射")
        kind = values.get("kind")
        radius = values.get("safe_radius_veh_m")
        label = values.get("label")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"对象策略 {name} 的 kind 无效")
        # 拒绝 bool 与 NaN/Inf，确保安全半径为有限正数。
        if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"对象策略 {name} 的 safe_radius_veh_m 必须为有限正数")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"对象策略 {name} 的 label 无效")
        policies[name.strip().lower()] = ObjectPolicy(kind.strip(), float(radius), label.strip())
    # 强制保留基础行人/机动车兜底键，防止未知类别回退时找不到默认半径。
    for required in ("person", "pedestrian", "vehicle", "motor_vehicle"):
        if required not in policies:
            raise ValueError(f"对象策略缺少必需类别: {required}")
    return policies


def object_policy(
    class_name: str | None,
    category: str = "vehicle",
    policies: dict[str, ObjectPolicy] | None = None,
) -> ObjectPolicy:
    """将模型类别映射到稳定策略；未知车辆保守使用机动车半径。"""
    policies = policies or POLICIES
    # 粗分类已判定为行人或细分标签为行人时，优先走行人包络策略。
    if category in ("pedestrian", "person"):
        return policies.get("pedestrian", PEDESTRIAN_POLICY)
    key = (class_name or "vehicle").strip().lower()
    if key in ("person", "pedestrian"):
        return policies.get("pedestrian", PEDESTRIAN_POLICY)
    # 未收录的车辆子类保守回退到普通机动车半径，避免低估碰撞包络。
    return policies.get(key, policies.get("vehicle", POLICIES["vehicle"]))
