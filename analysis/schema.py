"""Analysis 输出契约校验；通过仅表示格式正确，不表示算法精度正确。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


class SchemaValidationError(ValueError):
    pass


def _is_blank_path(value) -> bool:
    # 同时拦截空白 str 与纯空白 PathLike（如 Path("   ")），
    # 防止 Windows 下 Path("   ").resolve() 退化为当前目录 '.' 或 Linux 误建空白文件。
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, os.PathLike):
        fspath = os.fspath(value)
        return isinstance(fspath, str) and not fspath.strip()
    return False


def _reject_constant(value):
    raise SchemaValidationError(f"标准 JSON 不允许 {value}")


def _reject_non_finite_tree(value, name: str):
    # parse_constant 只会拦截 NaN / Infinity / -Infinity 字面量，而 1e309 溢出字面量会被
    # Python json.loads 直接解析为 float('inf')；因此必须在解析后递归检查整棵值树。
    # 这里用精确 type() 检查而非 isinstance，确保 bool 不会混入数值或整数校验。
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"{name} 包含非有限数值")
    elif isinstance(value, dict):
        for k, v in value.items():
            _reject_non_finite_tree(v, f"{name}.{k}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _reject_non_finite_tree(item, f"{name}[{idx}]")


def _schema_version(data: dict, name: str):
    if "schema_version" not in data:
        raise SchemaValidationError(f"{name} 缺少字段: schema_version")
    version = data["schema_version"]
    if type(version) is not int or version != 1:
        raise SchemaValidationError(f"{name}.schema_version 必须为整数 1")


def load_json_strict(path: str | Path):
    # 拒绝空白路径（含 Path("   ")），防止 Path("") 解析成当前目录 '.' 触发目录读取异常。
    if _is_blank_path(path):
        raise SchemaValidationError("文件路径不能为空")
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"{file_path}: JSON 格式错误: {exc}") from None
    _reject_non_finite_tree(data, str(file_path))
    return data


def _load_jsonl_strict_with_lines(path: str | Path) -> list[tuple[int, object]]:
    # 拒绝空白路径（含 Path("   ")），防止 Path("") 解析成当前目录 '.'。
    if _is_blank_path(path):
        raise SchemaValidationError("文件路径不能为空")
    file_path = Path(path)
    rows = []
    for line_no, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line, parse_constant=_reject_constant)
            _reject_non_finite_tree(item, f"{file_path} 第 {line_no} 行")
            rows.append((line_no, item))
        except (json.JSONDecodeError, SchemaValidationError) as exc:
            raise SchemaValidationError(f"{file_path} 第 {line_no} 行无效: {exc}") from None
    return rows


def load_jsonl_strict(path: str | Path):
    return [item for _, item in _load_jsonl_strict_with_lines(path)]


def _mapping(value, name):
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{name} 必须是对象")
    return value


def _required(value, keys, name):
    missing = [key for key in keys if key not in value]
    if missing:
        raise SchemaValidationError(f"{name} 缺少字段: {', '.join(missing)}")


def _finite_number(value, name, *, minimum=None, exclusive_minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise SchemaValidationError(f"{name} 必须是有限数值")
    if minimum is not None and value < minimum:
        raise SchemaValidationError(f"{name} 必须 >= {minimum}")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        raise SchemaValidationError(f"{name} 必须 > {exclusive_minimum}")
    if maximum is not None and value > maximum:
        raise SchemaValidationError(f"{name} 必须 <= {maximum}")


def _integer(value, name, *, minimum=None):
    if type(value) is not int:
        raise SchemaValidationError(f"{name} 必须是整数")
    if minimum is not None and value < minimum:
        raise SchemaValidationError(f"{name} 必须 >= {minimum}")


def validate_configuration(data):
    data = _mapping(data, "configuration")
    _reject_non_finite_tree(data, "configuration")
    _required(data, (
        "schema_version", "weights", "tracker_config", "object_policy_config",
        "calibration", "object_policies", "fps_used", "speed_limit_kmh",
        "ema_alpha", "window_size_frames", "ttc_critical_s", "ttc_caution_s",
        "horizon_s", "stationary_speed_ms", "stationary_min_frames",
        "alert_confirm_frames", "alert_cooldown_frames", "reproducibility",
    ), "configuration")
    _schema_version(data, "configuration")
    for key in ("weights", "tracker_config", "object_policy_config"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise SchemaValidationError(f"configuration.{key} 必须是非空字符串")
    calib = _mapping(data["calibration"], "configuration.calibration")
    _required(calib, ("name", "synthetic", "width_m", "length_m"), "configuration.calibration")
    if not isinstance(calib["name"], str) or not calib["name"].strip():
        raise SchemaValidationError("configuration.calibration.name 必须是非空字符串")
    if type(calib["synthetic"]) is not bool:
        raise SchemaValidationError("configuration.calibration.synthetic 必须是布尔值")
    _finite_number(calib["width_m"], "configuration.calibration.width_m", exclusive_minimum=0)
    _finite_number(calib["length_m"], "configuration.calibration.length_m", exclusive_minimum=0)

    policies = _mapping(data["object_policies"], "configuration.object_policies")
    if not policies:
        raise SchemaValidationError("configuration.object_policies 不能为空")
    for name, item in policies.items():
        entry = _mapping(item, f"configuration.object_policies.{name}")
        _required(entry, ("kind", "safe_radius_veh_m", "label"), f"configuration.object_policies.{name}")
        if not isinstance(entry["kind"], str) or not entry["kind"].strip():
            raise SchemaValidationError(f"configuration.object_policies.{name}.kind 必须是非空字符串")
        _finite_number(
            entry["safe_radius_veh_m"],
            f"configuration.object_policies.{name}.safe_radius_veh_m",
            exclusive_minimum=0,
        )
        if not isinstance(entry["label"], str) or not entry["label"].strip():
            raise SchemaValidationError(f"configuration.object_policies.{name}.label 必须是非空字符串")

    _finite_number(data["fps_used"], "configuration.fps_used", exclusive_minimum=0)
    _finite_number(data["speed_limit_kmh"], "configuration.speed_limit_kmh", exclusive_minimum=0)
    _finite_number(data["ema_alpha"], "configuration.ema_alpha", exclusive_minimum=0, maximum=1)
    _integer(data["window_size_frames"], "configuration.window_size_frames", minimum=3)
    _finite_number(data["ttc_critical_s"], "configuration.ttc_critical_s", exclusive_minimum=0)
    _finite_number(data["ttc_caution_s"], "configuration.ttc_caution_s", exclusive_minimum=0)
    _finite_number(data["horizon_s"], "configuration.horizon_s", exclusive_minimum=0)
    if not (data["ttc_critical_s"] <= data["ttc_caution_s"] <= data["horizon_s"]):
        raise SchemaValidationError("configuration TTC 阈值需满足 ttc_critical_s <= ttc_caution_s <= horizon_s")
    _finite_number(data["stationary_speed_ms"], "configuration.stationary_speed_ms", exclusive_minimum=0)
    _integer(data["stationary_min_frames"], "configuration.stationary_min_frames", minimum=1)
    _integer(data["alert_confirm_frames"], "configuration.alert_confirm_frames", minimum=1)
    _integer(data["alert_cooldown_frames"], "configuration.alert_cooldown_frames", minimum=0)

    reproducibility = _mapping(data["reproducibility"], "configuration.reproducibility")
    _required(reproducibility, ("generated_at", "environment", "git", "artifacts"),
              "configuration.reproducibility")
    if not isinstance(reproducibility["generated_at"], str) or not reproducibility["generated_at"].strip():
        raise SchemaValidationError("configuration.reproducibility.generated_at 必须是非空字符串")
    _mapping(reproducibility["environment"], "configuration.reproducibility.environment")
    _mapping(reproducibility["git"], "configuration.reproducibility.git")
    artifacts = _mapping(reproducibility["artifacts"], "configuration.reproducibility.artifacts")
    _required(artifacts, ("weights", "tracker_config", "calibration", "object_policy"),
              "configuration.reproducibility.artifacts")
    return data


def validate_summary(data):
    data = _mapping(data, "summary")
    _reject_non_finite_tree(data, "summary")
    _required(data, ("schema_version", "status", "stop_reason", "frames_processed", "elapsed_seconds"), "summary")
    _schema_version(data, "summary")
    if data["status"] not in {"ok", "failed", "interrupted"}:
        raise SchemaValidationError("summary.status 无效")
    if not isinstance(data["stop_reason"], str) or not data["stop_reason"].strip():
        raise SchemaValidationError("summary.stop_reason 必须是非空字符串")
    _integer(data["frames_processed"], "summary.frames_processed", minimum=0)
    _finite_number(data["elapsed_seconds"], "summary.elapsed_seconds", minimum=0)
    if data["status"] == "ok":
        _required(data, (
            "annotated_output", "conflicts_jsonl", "events_jsonl", "speeds_csv",
            "configuration_json", "total_alert_events", "total_speeding_alerts",
            "total_caution_events", "total_critical_conflicts",
        ), "summary")
        if data["annotated_output"] is not None and (
            not isinstance(data["annotated_output"], str) or not data["annotated_output"].strip()
        ):
            raise SchemaValidationError("summary.annotated_output 必须为非空字符串或 null")
        for key in ("total_alert_events", "total_speeding_alerts", "total_caution_events",
                    "total_critical_conflicts"):
            _integer(data[key], f"summary.{key}", minimum=0)
    return data


def validate_conflict_record(data):
    data = _mapping(data, "conflict record")
    _reject_non_finite_tree(data, "conflict")
    _required(data, ("schema_version", "frame_index", "timestamp_s", "event_id", "assessment"), "conflict record")
    _schema_version(data, "conflict")
    _integer(data["frame_index"], "conflict.frame_index", minimum=0)
    _finite_number(data["timestamp_s"], "conflict.timestamp_s", minimum=0)
    if data["event_id"] is not None:
        _integer(data["event_id"], "conflict.event_id", minimum=1)
    assessment = _mapping(data["assessment"], "conflict.assessment")
    _required(assessment, (
        "veh_track_id", "ped_track_id", "risk_level", "ttc_seconds", "vehicle_kind",
    ), "conflict.assessment")
    _integer(assessment["veh_track_id"], "conflict.veh_track_id", minimum=1)
    if assessment["ped_track_id"] is not None:
        _integer(assessment["ped_track_id"], "conflict.ped_track_id", minimum=1)
    _integer(assessment["risk_level"], "conflict.risk_level", minimum=0)
    if assessment["risk_level"] > 3:
        raise SchemaValidationError("conflict.risk_level 必须在 0..3")
    if assessment["ttc_seconds"] is not None:
        _finite_number(assessment["ttc_seconds"], "conflict.ttc_seconds", minimum=0)
    if not isinstance(assessment["vehicle_kind"], str) or not assessment["vehicle_kind"].strip():
        raise SchemaValidationError("conflict.vehicle_kind 必须是非空字符串")
    return data


def validate_event_record(data):
    data = _mapping(data, "event record")
    _reject_non_finite_tree(data, "event")
    _required(data, (
        "schema_version", "event_id", "kind", "veh_track_id", "ped_track_id",
        "start_frame", "end_frame", "start_timestamp_s", "end_timestamp_s",
        "peak_risk_level", "min_ttc_seconds", "end_reason",
    ), "event record")
    _schema_version(data, "event")
    if data["kind"] not in {"conflict", "speeding"}:
        raise SchemaValidationError("event.kind 必须是 conflict 或 speeding")
    for key in ("event_id", "veh_track_id"):
        _integer(data[key], f"event.{key}", minimum=1)
    if data["ped_track_id"] is not None:
        _integer(data["ped_track_id"], "event.ped_track_id", minimum=1)
    for key in ("start_frame", "end_frame"):
        _integer(data[key], f"event.{key}", minimum=0)
    if data["end_frame"] < data["start_frame"]:
        raise SchemaValidationError("event.end_frame 不能小于 start_frame")
    for key in ("start_timestamp_s", "end_timestamp_s"):
        _finite_number(data[key], f"event.{key}", minimum=0)
    if data["end_timestamp_s"] < data["start_timestamp_s"]:
        raise SchemaValidationError("event.end_timestamp_s 不能小于 start_timestamp_s")
    _integer(data["peak_risk_level"], "event.peak_risk_level", minimum=1)
    if data["peak_risk_level"] > 3:
        raise SchemaValidationError("event.peak_risk_level 必须在 1..3")
    if data["min_ttc_seconds"] is not None:
        _finite_number(data["min_ttc_seconds"], "event.min_ttc_seconds", minimum=0)
    if not isinstance(data["end_reason"], str) or not data["end_reason"].strip():
        raise SchemaValidationError("event.end_reason 必须是非空字符串")
    return data


SPEED_COLUMNS = {
    "schema_version", "frame_idx", "timestamp_s", "track_id", "category", "speed_kmh", "risk_level",
    "event_id", "is_reliable", "is_stationary",
}


def validate_speeds_csv(path: str | Path):
    # 拒绝空白路径（含 Path("   ")），防止 Path("") 解析成当前目录 '.'。
    if _is_blank_path(path):
        raise SchemaValidationError("CSV 路径不能为空")
    file_path = Path(path)
    with file_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = SPEED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise SchemaValidationError(f"{file_path} 缺少 CSV 列: {', '.join(sorted(missing))}")
        count = 0
        for line_no, row in enumerate(reader, 2):
            # csv.DictReader 会把超出表头长度的多余列放入键 None 中，需显式拦截。
            if None in row:
                raise SchemaValidationError(f"{file_path} 第 {line_no} 行包含多余未命名列")
            try:
                schema_version = int(row["schema_version"])
                _integer(schema_version, f"speeds.csv 第 {line_no} 行 schema_version", minimum=1)
                if schema_version != 1:
                    raise SchemaValidationError(
                        f"speeds.csv 第 {line_no} 行 schema_version 必须为 1"
                    )
                _integer(int(row["frame_idx"]), f"speeds.csv 第 {line_no} 行 frame_idx", minimum=0)
                _finite_number(float(row["timestamp_s"]), f"speeds.csv 第 {line_no} 行 timestamp_s", minimum=0)
                _integer(int(row["track_id"]), f"speeds.csv 第 {line_no} 行 track_id", minimum=1)
                _finite_number(float(row["speed_kmh"]), f"speeds.csv 第 {line_no} 行 speed_kmh", minimum=0)
            except (TypeError, ValueError) as exc:
                raise SchemaValidationError(f"{file_path} 第 {line_no} 行数值字段无效: {exc}") from None
            for text_col in ("category", "risk_level"):
                val = row.get(text_col)
                if not isinstance(val, str) or not val.strip():
                    raise SchemaValidationError(f"{file_path} 第 {line_no} 行 {text_col} 必须是非空字符串")
            if row["is_reliable"] not in {"0", "1"} or row["is_stationary"] not in {"0", "1"}:
                raise SchemaValidationError(f"{file_path} 第 {line_no} 行布尔字段必须是 0/1")
            if row["event_id"]:
                try:
                    _integer(int(row["event_id"]), f"speeds.csv 第 {line_no} 行 event_id", minimum=1)
                except ValueError as exc:
                    raise SchemaValidationError(str(exc)) from None
            count += 1
    return count


def validate_scenarios(data):
    data = _mapping(data, "scenarios")
    _reject_non_finite_tree(data, "scenarios")
    _required(data, ("schema_version", "results"), "scenarios")
    _schema_version(data, "scenarios")
    if not isinstance(data["results"], list):
        raise SchemaValidationError("scenarios.results 必须是列表")
    for index, row in enumerate(data["results"]):
        _mapping(row, f"scenarios.results[{index}]")
        _required(row, ("name", "expected_levels", "actual_levels", "passed"),
                  f"scenarios.results[{index}]")
        if not isinstance(row["name"], str) or not row["name"].strip():
            raise SchemaValidationError(f"scenarios.results[{index}].name 必须是非空字符串")
        for key in ("expected_levels", "actual_levels"):
            if not isinstance(row[key], list) or not all(isinstance(x, str) and x.strip() for x in row[key]):
                raise SchemaValidationError(f"scenarios.results[{index}].{key} 必须是非空字符串列表")
        if type(row["passed"]) is not bool:
            raise SchemaValidationError(f"scenarios.results[{index}].passed 必须是布尔值")
        if "assessment_count" in row:
            _integer(row["assessment_count"], f"scenarios.results[{index}].assessment_count", minimum=0)
    return data


def validate_ablation(data):
    data = _mapping(data, "ablation")
    _reject_non_finite_tree(data, "ablation")
    _required(data, ("schema_version", "speed", "conflict"), "ablation")
    _schema_version(data, "ablation")
    if not isinstance(data["speed"], list) or not isinstance(data["conflict"], list):
        raise SchemaValidationError("ablation speed/conflict 必须是列表")
    for idx, row in enumerate(data["speed"]):
        item = _mapping(row, f"ablation.speed[{idx}]")
        _required(
            item,
            ("target_speed_ms", "window_size", "ema_alpha", "measured_speed_ms",
             "absolute_error_ms", "is_reliable", "is_stationary"),
            f"ablation.speed[{idx}]",
        )
        _finite_number(item["target_speed_ms"], f"ablation.speed[{idx}].target_speed_ms", minimum=0)
        _integer(item["window_size"], f"ablation.speed[{idx}].window_size", minimum=3)
        _finite_number(item["ema_alpha"], f"ablation.speed[{idx}].ema_alpha", exclusive_minimum=0, maximum=1)
        _finite_number(item["measured_speed_ms"], f"ablation.speed[{idx}].measured_speed_ms", minimum=0)
        _finite_number(item["absolute_error_ms"], f"ablation.speed[{idx}].absolute_error_ms", minimum=0)
        if type(item["is_reliable"]) is not bool or type(item["is_stationary"]) is not bool:
            raise SchemaValidationError(f"ablation.speed[{idx}] 布尔字段无效")
    for idx, row in enumerate(data["conflict"]):
        item = _mapping(row, f"ablation.conflict[{idx}]")
        _required(
            item,
            ("ttc_critical_s", "horizon_s", "ttc_s", "min_distance_m", "risk_level"),
            f"ablation.conflict[{idx}]",
        )
        _finite_number(item["ttc_critical_s"], f"ablation.conflict[{idx}].ttc_critical_s", exclusive_minimum=0)
        _finite_number(item["horizon_s"], f"ablation.conflict[{idx}].horizon_s", exclusive_minimum=0)
        if item["ttc_s"] is not None:
            _finite_number(item["ttc_s"], f"ablation.conflict[{idx}].ttc_s", minimum=0)
        if item["min_distance_m"] is not None:
            _finite_number(item["min_distance_m"], f"ablation.conflict[{idx}].min_distance_m", minimum=0)
        if not isinstance(item["risk_level"], str) or not item["risk_level"].strip():
            raise SchemaValidationError(f"ablation.conflict[{idx}].risk_level 必须是非空字符串")
    return data


def validate_run(run_dir: str | Path) -> dict:
    # 先校验 summary.json 获取运行状态：status=="ok" 时要求完整产物齐全，
    # 失败或中断目录仍校验已落盘的局部文件（如 summary/configuration/已刷盘事件），但不强求全套文件。
    # 拒绝空白路径（含 Path("   ")），防止 Path("") 解析成当前目录 '.'。
    if _is_blank_path(run_dir):
        raise SchemaValidationError("运行目录不能为空")
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise SchemaValidationError(f"运行目录不存在: {root}")
    summary = validate_summary(load_json_strict(root / "summary.json"))
    counts = {"conflicts": 0, "events": 0, "speed_rows": 0}
    if summary["status"] == "ok":
        for required_name in ("configuration.json", "conflicts.jsonl", "events.jsonl", "speeds.csv"):
            if not (root / required_name).exists():
                raise SchemaValidationError(f"完成运行缺少 {required_name}")
    if (root / "configuration.json").exists():
        validate_configuration(load_json_strict(root / "configuration.json"))
    if (root / "conflicts.jsonl").exists():
        conflict_path = root / "conflicts.jsonl"
        rows = _load_jsonl_strict_with_lines(conflict_path)
        for line_no, row in rows:
            try:
                validate_conflict_record(row)
            except SchemaValidationError as exc:
                raise SchemaValidationError(f"{conflict_path} 第 {line_no} 行无效: {exc}") from None
        counts["conflicts"] = len(rows)
    if (root / "events.jsonl").exists():
        events_path = root / "events.jsonl"
        rows = _load_jsonl_strict_with_lines(events_path)
        for line_no, row in rows:
            try:
                validate_event_record(row)
            except SchemaValidationError as exc:
                raise SchemaValidationError(f"{events_path} 第 {line_no} 行无效: {exc}") from None
        counts["events"] = len(rows)
    if (root / "speeds.csv").exists():
        counts["speed_rows"] = validate_speeds_csv(root / "speeds.csv")
    return {"schema_version": 1, "valid": True, "run_dir": str(root), "counts": counts,
            "note": "仅表示输出格式有效，不代表算法精度或真实场景验证通过。"}


def validate_batch_summary(data):
    data = _mapping(data, "batch summary")
    _reject_non_finite_tree(data, "batch summary")
    _required(data, (
        "schema_version", "kind", "source_dir", "output_dir", "status",
        "videos_discovered", "videos_not_processed",
        "total_alert_events", "total_speeding_alerts",
        "total_caution_events", "total_critical_conflicts", "results",
    ), "batch summary")
    _schema_version(data, "batch summary")
    if data["kind"] != "analysis_batch":
        raise SchemaValidationError("batch summary.kind 必须为 analysis_batch")
    if data["status"] not in {"running", "completed", "completed_with_errors", "interrupted", "stopped"}:
        raise SchemaValidationError("batch summary.status 无效")
    for key in (
        "videos_discovered", "videos_not_processed", "total_alert_events",
        "total_speeding_alerts", "total_caution_events", "total_critical_conflicts",
    ):
        _integer(data[key], f"batch summary.{key}", minimum=0)
    if data["videos_not_processed"] > data["videos_discovered"]:
        raise SchemaValidationError(
            "batch summary.videos_not_processed 不能大于 videos_discovered"
        )
    if not isinstance(data["results"], list):
        raise SchemaValidationError("batch summary.results 必须是列表")
    for idx, item in enumerate(data["results"]):
        entry = _mapping(item, f"batch summary.results[{idx}]")
        _required(entry, ("source", "output", "status"), f"batch summary.results[{idx}]")
        if entry["status"] not in {"ok", "failed", "interrupted"}:
            raise SchemaValidationError(f"batch summary.results[{idx}].status 无效")
        # 同时拦截纯空白字符串，防止后续 validate_batch_run 中 Path("   ") 退化为当前目录。
        for s_key in ("source", "output"):
            if not isinstance(entry[s_key], str) or not entry[s_key].strip():
                raise SchemaValidationError(f"batch summary.results[{idx}].{s_key} 必须是非空字符串")
    return data


def validate_batch_run(batch_dir: str | Path) -> dict:
    # 拒绝空白路径（含 Path("   ")），防止 Path("") 解析成当前目录 '.'。
    if _is_blank_path(batch_dir):
        raise SchemaValidationError("批量运行目录不能为空")
    root = Path(batch_dir).expanduser().resolve()
    if not root.is_dir():
        raise SchemaValidationError(f"批量运行目录不存在: {root}")
    summary = validate_batch_summary(load_json_strict(root / "batch_summary.json"))
    ok_runs = 0
    # 逐项校验成功的子运行目录；失败/中断项保留在清单中即可，不要求拥有完整产物。
    for idx, item in enumerate(summary["results"]):
        if item["status"] != "ok":
            continue
        if _is_blank_path(item["output"]):
            raise SchemaValidationError(f"batch_summary.results[{idx}].output 不能为空")
        child_dir = Path(item["output"])
        # 若整个批次目录被整体移动或复制到新位置，回退按子目录名在当前批次根目录下查找。
        if not child_dir.is_dir():
            child_dir = root / child_dir.name
        try:
            validate_run(child_dir)
        except SchemaValidationError as exc:
            raise SchemaValidationError(
                f"batch_summary.results[{idx}] 子目录校验失败: {exc}"
            ) from None
        ok_runs += 1
    return {
        "schema_version": 1,
        "valid": True,
        "kind": "analysis_batch",
        "batch_dir": str(root),
        "validated_runs": ok_runs,
        "total_results": len(summary["results"]),
        "note": "仅表示批量与子目录输出格式有效，不代表算法精度或真实场景验证通过。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验 analysis 运行输出格式")
    parser.add_argument("--run", required=True, help="runs/analysis/run-* 或 batch-* 目录")
    args = parser.parse_args(argv)
    # 必须在 Path(args.run) 之前拦截空白字符串：否则 Path("") 会解析为当前工作目录 '.'，
    # 转为 Path 对象后还会绕过 validate_run / validate_batch_run 中的 isinstance(..., str) 空白拦截。
    if not args.run.strip():
        parser.error("--run 不能为空")
    try:
        root = Path(args.run).expanduser().resolve()
        if (root / "batch_summary.json").is_file() and not (root / "summary.json").is_file():
            result = validate_batch_run(root)
        else:
            result = validate_run(root)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, SchemaValidationError) as exc:
        parser.exit(1, f"Schema 校验失败: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
