"""配置/求解产物读写和旧四点配置适配；不覆盖现有标定文件。"""

from dataclasses import asdict
from pathlib import Path
import json
import tempfile

import numpy as np
import yaml

from .core import MappingConfig, RoadMapper

ROOT = Path(__file__).resolve().parents[1]


def read_document(path):
    try:
        value = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"标定文档解析失败：{exc}") from None
    if not isinstance(value, dict):
        raise ValueError("标定文档必须是对象/映射")
    return value


def mapping_from_document(value):
    # 输入配置与已求解产物都能读取；派生矩阵、验证结论均重新计算，不盲信文件里的 true。
    resolved = "config" in value
    values = value["config"] if resolved else value
    if not isinstance(values, dict):
        raise ValueError("config 必须是配置对象")
    try:
        mapper = RoadMapper(MappingConfig(**values))
    except TypeError as exc:
        raise ValueError(f"标定配置字段错误：{exc}") from None
    if resolved:
        if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
                or value.get("calibration_id") != mapper.calibration_id):
            raise ValueError("求解产物版本或 calibration_id 与配置不一致，请重新 solve")
        stored = np.asarray(value.get("image_to_ground_h"), dtype=np.float64)
        if stored.shape != (3, 3) or not np.allclose(stored, mapper.H, rtol=1e-6, atol=1e-9):
            raise ValueError("保存的 H 与参考点重新求解结果不一致，请重新 solve")
    return mapper


def load_mapping(path):
    return mapping_from_document(read_document(path))


def resolved_document(mapper):
    return {"schema_version": 1, "calibration_id": mapper.calibration_id,
            "config": asdict(mapper.config), "image_to_ground_h": mapper.H.tolist(),
            "ground_to_image_h": mapper.H_inv.tolist(), "inlier_mask": mapper.inliers.tolist(),
            "validation": mapper.report}


def write_json(path, value):
    # x 模式拒绝覆盖旧文件；命令行运行目录本身也必须是新目录。
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def new_output(path=None):
    if path is not None:
        output = Path(path).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        return output
    parent = ROOT / "runs" / "calibration"
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=parent))


def import_legacy(path, *, camera_id, image_size, allow_legacy=False):
    from analysis.calibration import RoadCalibrator
    legacy = RoadCalibrator.from_yaml(path)
    if legacy.synthetic_field_missing and not allow_legacy:
        raise ValueError("旧配置缺少 synthetic 来源字段；需明确 --allow-legacy 后才能转换")
    # 老配置没有图像尺寸、机位和独立检查点，转换不等于实测验证完成。
    return RoadMapper(MappingConfig(
        camera_id=camera_id, image_size=image_size,
        source_type="synthetic" if legacy.synthetic else "measured",
        measurement_reference=f"Legacy import from {Path(path).name}; independent validation pending",
        image_points=legacy.image_points, ground_points_m=legacy.ground_points, method="rectangle"))
