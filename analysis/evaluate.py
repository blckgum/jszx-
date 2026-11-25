"""对一次 analysis 运行产物做无真值描述性评价，或结合人工真值计算精度指标。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import fmean

import numpy as np
import yaml

from .mot_eval import _hungarian_max_weight


def _is_blank_path(value) -> bool:
    # 同时拦截空白 str 与纯空白 PathLike（如 Path("   ")），
    # 防止 Windows 下 Path("   ").resolve() 退化为当前目录 '.' 或 Linux 误建空白文件。
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, os.PathLike):
        fspath = os.fspath(value)
        return isinstance(fspath, str) and not fspath.strip()
    return False


def _reject_non_finite_constant(value: str):
    raise ValueError(f"JSON 不允许非有限常量: {value}")


def _reject_non_finite_tree(value, name: str):
    # 与 analysis.schema 保持一致：json.loads 遇到 1e309 会生成 inf 且不经过 parse_constant，
    # 即使 evaluate_run(..., validate=False) 也必须拒绝非有限数据以免静默污染统计。
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} 包含非有限数值")
    elif isinstance(value, dict):
        for k, v in value.items():
            _reject_non_finite_tree(v, f"{name}.{k}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _reject_non_finite_tree(item, f"{name}[{idx}]")


def _read_json(path: Path, default):
    if not path.exists():
        return default
    data = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_non_finite_constant,
    )
    _reject_non_finite_tree(data, str(path))
    return data


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line, parse_constant=_reject_non_finite_constant)
            _reject_non_finite_tree(item, f"{path} 第 {line_no} 行")
            rows.append(item)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是有效 JSON: {exc}") from None
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _finite(values):
    out = []
    for value in values:
        if value in (None, "") or type(value) is bool:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            out.append(num)
    return out


def _risk_name(value) -> str:
    if value is None or value == "":
        return "LEVEL_0_SAFE"
    if isinstance(value, str):
        return value if value.startswith("LEVEL_") else f"LEVEL_{value}"
    return f"LEVEL_{int(value)}_SAFE" if int(value) == 0 else f"LEVEL_{int(value)}"


def _require_non_neg_number(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} 必须为有限非负数")
    return float(value)


def _validate_ground_truth(data) -> dict:
    # 校验人工真值标注契约：要求 schema_version == 1，容差、速度参考与期望事件起止时刻均非负且有限，
    # veh_track_id / ped_track_id 为 null 时表示匹配任意目标 ID。
    if not isinstance(data, dict):
        raise ValueError("真值标注文件顶层必须是键值映射")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("真值标注必须包含 schema_version: 1")
    tol = _require_non_neg_number(data.get("tolerance_s", 0.5), "tolerance_s")

    raw_speeds = data.get("speed_references", [])
    if not isinstance(raw_speeds, list):
        raise ValueError("speed_references 必须是列表")
    speed_refs = []
    for idx, item in enumerate(raw_speeds):
        if not isinstance(item, dict):
            raise ValueError(f"speed_references[{idx}] 必须是映射")
        tid = item.get("track_id")
        if type(tid) is not int or tid <= 0:
            raise ValueError(f"speed_references[{idx}].track_id 必须为正整数")
        ref_kmh = _require_non_neg_number(
            item.get("reference_speed_kmh"), f"speed_references[{idx}].reference_speed_kmh"
        )
        t_start = item.get("start_timestamp_s")
        t_end = item.get("end_timestamp_s")
        start_val = _require_non_neg_number(t_start, f"speed_references[{idx}].start_timestamp_s") if t_start is not None else None
        end_val = _require_non_neg_number(t_end, f"speed_references[{idx}].end_timestamp_s") if t_end is not None else None
        if start_val is not None and end_val is not None and end_val < start_val:
            raise ValueError(f"speed_references[{idx}] end_timestamp_s 不能小于 start_timestamp_s")
        only_reliable = item.get("only_reliable", True)
        if type(only_reliable) is not bool:
            raise ValueError(f"speed_references[{idx}].only_reliable 必须是布尔值")
        speed_refs.append({
            "track_id": tid,
            "reference_speed_kmh": ref_kmh,
            "start_timestamp_s": start_val,
            "end_timestamp_s": end_val,
            "only_reliable": only_reliable,
        })

    raw_events = data.get("expected_events", [])
    if not isinstance(raw_events, list):
        raise ValueError("expected_events 必须是列表")
    expected_events = []
    for idx, item in enumerate(raw_events):
        if not isinstance(item, dict):
            raise ValueError(f"expected_events[{idx}] 必须是映射")
        kind = item.get("kind")
        if kind not in {"conflict", "speeding"}:
            raise ValueError(f"expected_events[{idx}].kind 必须为 conflict 或 speeding")
        start_val = _require_non_neg_number(item.get("start_timestamp_s"), f"expected_events[{idx}].start_timestamp_s")
        end_val = _require_non_neg_number(item.get("end_timestamp_s"), f"expected_events[{idx}].end_timestamp_s")
        if end_val < start_val:
            raise ValueError(f"expected_events[{idx}] end_timestamp_s 不能小于 start_timestamp_s")
        veh_id = item.get("veh_track_id")
        if veh_id is not None and (type(veh_id) is not int or veh_id <= 0):
            raise ValueError(f"expected_events[{idx}].veh_track_id 必须为正整数或 null")
        ped_id = item.get("ped_track_id")
        if ped_id is not None and (type(ped_id) is not int or ped_id <= 0):
            raise ValueError(f"expected_events[{idx}].ped_track_id 必须为正整数或 null")
        moment = item.get("conflict_moment_s")
        moment_val = _require_non_neg_number(moment, f"expected_events[{idx}].conflict_moment_s") if moment is not None else None
        expected_events.append({
            "kind": kind,
            "start_timestamp_s": start_val,
            "end_timestamp_s": end_val,
            "veh_track_id": veh_id,
            "ped_track_id": ped_id,
            "conflict_moment_s": moment_val,
        })

    return {
        "schema_version": 1,
        "tolerance_s": tol,
        "speed_references": speed_refs,
        "expected_events": expected_events,
    }


def load_ground_truth(path: str | Path) -> dict:
    """加载并校验 JSON/YAML 格式的速度与事件人工真值文件。"""
    # 拒绝空白真值路径（含 Path("   ")），防止 Path("") 解析为当前工作目录 '.'。
    if _is_blank_path(path):
        raise ValueError("真值文件路径不能为空")
    gt_path = Path(path).expanduser().resolve()
    if not gt_path.is_file():
        raise FileNotFoundError(f"真值文件不存在: {gt_path}")
    text = gt_path.read_text(encoding="utf-8")
    if gt_path.suffix.lower() == ".json":
        def _reject(val):
            raise ValueError(f"真值 JSON 不允许非有限常量: {val}")
        try:
            raw = json.loads(text, parse_constant=_reject)
        except json.JSONDecodeError as exc:
            raise ValueError(f"真值 JSON 解析失败: {exc}") from None
        _reject_non_finite_tree(raw, str(gt_path))
    else:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"真值 YAML 解析失败: {exc}") from None
    return _validate_ground_truth(raw)


def evaluate_ground_truth(
    gt: dict,
    *,
    speeds: list[dict[str, str]],
    events: list[dict],
) -> dict:
    """对照人工真值计算测速误差 (MAE/RMSE) 与事件级 Precision/Recall/F1/预警提前量。"""
    parsed = _validate_ground_truth(gt)
    tol = parsed["tolerance_s"]

    # 1) 测速误差评价：按人工真值里的 track_id 与可选时间窗筛选 speeds.csv 样本，
    # 算得平均测速后与 reference_speed_kmh 对比 MAE / RMSE / 最大绝对误差 / 平均偏差。
    speed_items = []
    signed_errors = []
    abs_errors = []
    for ref in parsed["speed_references"]:
        matched_vals = []
        for row in speeds:
            if not isinstance(row, dict):
                continue
            raw_tid = row.get("track_id", "")
            raw_ts = row.get("timestamp_s", "")
            raw_spd = row.get("speed_kmh", "")
            # 防止直接传入 Python 字典时 bool (True/False) 被 int()/float() 转成 1/0.0。
            if type(raw_tid) is bool or type(raw_ts) is bool or type(raw_spd) is bool:
                continue
            try:
                tid = int(raw_tid)
                ts = float(raw_ts)
                spd = float(raw_spd)
            except (TypeError, ValueError):
                continue
            if (
                tid != ref["track_id"]
                or not (math.isfinite(ts) and math.isfinite(spd))
                or ts < 0
                or spd < 0
            ):
                continue
            # 默认只拿 is_reliable == 1 的收敛后样本参与测速精度评价，避免预热期拉低统计。
            if ref["only_reliable"] and str(row.get("is_reliable", "")) != "1":
                continue
            if ref["start_timestamp_s"] is not None and ts < ref["start_timestamp_s"]:
                continue
            if ref["end_timestamp_s"] is not None and ts > ref["end_timestamp_s"]:
                continue
            matched_vals.append(spd)

        if matched_vals:
            measured_mean = fmean(matched_vals)
            signed_err = measured_mean - ref["reference_speed_kmh"]
            abs_err = abs(signed_err)
            signed_errors.append(signed_err)
            abs_errors.append(abs_err)
            speed_items.append({
                "track_id": ref["track_id"],
                "reference_speed_kmh": ref["reference_speed_kmh"],
                "sample_count": len(matched_vals),
                "measured_mean_kmh": round(measured_mean, 4),
                "signed_error_kmh": round(signed_err, 4),
                "abs_error_kmh": round(abs_err, 4),
                "matched": True,
            })
        else:
            speed_items.append({
                "track_id": ref["track_id"],
                "reference_speed_kmh": ref["reference_speed_kmh"],
                "sample_count": 0,
                "measured_mean_kmh": None,
                "signed_error_kmh": None,
                "abs_error_kmh": None,
                "matched": False,
            })

    speed_accuracy = {
        "reference_count": len(parsed["speed_references"]),
        "matched_count": len(abs_errors),
        "unmatched_count": len(parsed["speed_references"]) - len(abs_errors),
        "mae_kmh": round(fmean(abs_errors), 4) if abs_errors else None,
        "rmse_kmh": round(math.sqrt(fmean(e * e for e in signed_errors)), 4) if signed_errors else None,
        "max_abs_error_kmh": round(max(abs_errors), 4) if abs_errors else None,
        "mean_signed_bias_kmh": round(fmean(signed_errors), 4) if signed_errors else None,
        "items": speed_items,
    }

    # 2) 事件级 1-to-1 最大基数最优匹配：先最大化 TP 匹配数量，再最大化区间重叠时长并最小化起始时刻差。
    gt_events = parsed["expected_events"]
    n_gt = len(gt_events)
    n_pred = len(events)
    matched_pairs = []
    if n_gt > 0 and n_pred > 0:
        valid = np.zeros((n_gt, n_pred), dtype=bool)
        ov_mat = np.zeros((n_gt, n_pred), dtype=np.float64)
        sd_mat = np.zeros((n_gt, n_pred), dtype=np.float64)
        max_ov = 0.0
        max_sd = 0.0
        for gi, g in enumerate(gt_events):
            for pi, p in enumerate(events):
                if not isinstance(p, dict) or p.get("kind") != g["kind"]:
                    continue
                # 用精确布尔排除，防止 Python 中 True == 1 把布尔 track_id 误配为 1 号目标。
                p_veh = p.get("veh_track_id")
                p_ped = p.get("ped_track_id")
                if g["veh_track_id"] is not None and (type(p_veh) is bool or p_veh != g["veh_track_id"]):
                    continue
                if g["ped_track_id"] is not None and (type(p_ped) is bool or p_ped != g["ped_track_id"]):
                    continue
                raw_start = p.get("start_timestamp_s")
                raw_end = p.get("end_timestamp_s")
                if type(raw_start) is bool or type(raw_end) is bool:
                    continue
                try:
                    p_start = float(raw_start)
                    p_end = float(raw_end)
                except (TypeError, ValueError):
                    continue
                # 忽略非有限、负时间戳或起止倒挂的无效预测事件，防止异常区间产生伪匹配。
                if not (math.isfinite(p_start) and math.isfinite(p_end)) or p_start < 0 or p_end < p_start:
                    continue
                ov_start = max(p_start - tol, g["start_timestamp_s"])
                ov_end = min(p_end + tol, g["end_timestamp_s"])
                if ov_end >= ov_start:
                    overlap = ov_end - ov_start
                    start_diff = abs(p_start - g["start_timestamp_s"])
                    valid[gi, pi] = True
                    ov_mat[gi, pi] = overlap
                    sd_mat[gi, pi] = start_diff
                    if overlap > max_ov:
                        max_ov = overlap
                    if start_diff > max_sd:
                        max_sd = start_diff

        if bool(np.any(valid)):
            ov_cap = max(1.0, max_ov)
            sd_cap = max(1.0, max_sd)
            card_bonus = float(min(n_gt, n_pred) + 1) * (ov_cap + 2.0)
            weights = np.where(
                valid,
                card_bonus + ov_mat - 1e-4 * (sd_mat / sd_cap),
                0.0,
            )
            pairs = _hungarian_max_weight(weights, valid)
            matched_pairs = [(gt_events[gi], events[pi]) for gi, pi in pairs]

    tp = len(matched_pairs)
    fp = len(events) - tp
    fn = len(gt_events) - tp
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
    if precision is not None and recall is not None:
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    else:
        f1 = None

    # 预警提前量 = 人工标注的潜在冲突时刻 - 系统开立事件起始时刻；正值表示提前预警。
    leads = [
        float(g["conflict_moment_s"]) - float(p["start_timestamp_s"])
        for g, p in matched_pairs
        if g["conflict_moment_s"] is not None
    ]
    event_accuracy = {
        "expected_count": len(gt_events),
        "predicted_count": len(events),
        "tolerance_s": tol,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "lead_time_s": {
            "count": len(leads),
            "min": round(min(leads), 4) if leads else None,
            "mean": round(fmean(leads), 4) if leads else None,
            "max": round(max(leads), 4) if leads else None,
        },
    }

    return {
        "speed_accuracy": speed_accuracy,
        "event_accuracy": event_accuracy,
    }


def evaluate_run(
    run_dir: str | Path,
    *,
    validate: bool = False,
    ground_truth: str | Path | dict | None = None,
) -> dict:
    """读取运行产物，返回描述性指标；提供人工真值时同时计算准确率与测速误差。"""
    # 拒绝空白目录或真值路径（含 Path("   ")），防止 Path("") 解析成当前工作目录 '.'。
    if _is_blank_path(run_dir):
        raise ValueError("运行输出目录不能为空")
    if _is_blank_path(ground_truth):
        raise ValueError("真值文件路径不能为空")
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"运行目录不存在: {root}")
    schema_result = None
    if validate:
        from .schema import validate_run
        schema_result = validate_run(root)

    summary = _read_json(root / "summary.json", {})
    conflicts = _read_jsonl(root / "conflicts.jsonl")
    events = _read_jsonl(root / "events.jsonl")
    speeds = _read_csv(root / "speeds.csv")

    risk_counts = Counter()
    event_id_count = 0
    frame_ids = set()
    for row in conflicts:
        assessment = row.get("assessment", row)
        risk_counts[_risk_name(assessment.get("risk_level"))] += 1
        if row.get("event_id") is not None:
            event_id_count += 1
        if row.get("frame_index") is not None:
            frame_ids.add(row["frame_index"])

    event_kinds = Counter(str(row.get("kind", "unknown")) for row in events)
    peak_levels = Counter(str(row.get("peak_risk_name", row.get("peak_risk_level", "unknown"))) for row in events)
    durations = _finite(row.get("duration_s") for row in events)
    ttcs = _finite(row.get("min_ttc_seconds") for row in events)
    speeds_kmh = _finite(row.get("speed_kmh") for row in speeds)
    reliable = Counter(str(row.get("is_reliable", "")) for row in speeds)
    stationary = Counter(str(row.get("is_stationary", "")) for row in speeds)
    speed_event_id_count = sum(
        1 for row in speeds if str(row.get("event_id", "")).strip() != ""
    )

    gt_eval = None
    has_gt = ground_truth is not None
    if has_gt:
        gt_data = ground_truth if isinstance(ground_truth, dict) else load_ground_truth(ground_truth)
        gt_eval = evaluate_ground_truth(gt_data, speeds=speeds, events=events)

    payload = {
        "schema_version": 1,
        "run_dir": str(root),
        "has_ground_truth": has_gt,
        "schema_validation": schema_result,
        "note": (
            "已结合人工真值完成测速误差与事件级准确率评价。"
            if has_gt
            else "描述性运行评价，无人工真值，不声称 Precision/Recall/MAE 等精度指标。"
        ),
        "summary": summary,
        "frames": {
            "processed": summary.get("frames_processed", len(frame_ids)),
            "observed_assessment_rows": len(conflicts),
            "distinct_assessment_frames": len(frame_ids),
            "confirmed_event_rows": event_id_count,
        },
        "risk_counts": dict(sorted(risk_counts.items())),
        "events": {
            "count": len(events),
            "by_kind": dict(sorted(event_kinds.items())),
            "by_peak_level": dict(sorted(peak_levels.items())),
            "duration_s": {
                "min": min(durations) if durations else None,
                "mean": fmean(durations) if durations else None,
                "max": max(durations) if durations else None,
            },
            "min_ttc_s": {
                "min": min(ttcs) if ttcs else None,
                "mean": fmean(ttcs) if ttcs else None,
                "max": max(ttcs) if ttcs else None,
            },
        },
        "speed_rows": {
            "count": len(speeds),
            "confirmed_event_rows": speed_event_id_count,
            "speed_kmh": {
                "min": min(speeds_kmh) if speeds_kmh else None,
                "mean": fmean(speeds_kmh) if speeds_kmh else None,
                "max": max(speeds_kmh) if speeds_kmh else None,
            },
            "reliable_values": dict(sorted(reliable.items())),
            "stationary_values": dict(sorted(stationary.items())),
        },
    }
    if gt_eval is not None:
        payload["ground_truth_evaluation"] = gt_eval
    return payload


def to_markdown(result: dict) -> str:
    """将评价结果转换为适合实验记录的简短 Markdown。"""
    frames = result["frames"]
    events = result["events"]
    speed = result["speed_rows"]["speed_kmh"]
    lines = [
        "# Analysis run evaluation\n",
        f"- Run: `{result['run_dir']}`",
        f"- Frames processed: {frames['processed']}",
        f"- Assessment rows: {frames['observed_assessment_rows']}",
        f"- Events: {events['count']}",
        f"- Speed km/h: min={speed['min']}, mean={speed['mean']}, max={speed['max']}",
        "",
        "## Risk counts",
        "",
        "| Level | Rows |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in result["risk_counts"].items())
    if result.get("has_ground_truth") and "ground_truth_evaluation" in result:
        gt_eval = result["ground_truth_evaluation"]
        sa = gt_eval["speed_accuracy"]
        ea = gt_eval["event_accuracy"]
        lt = ea["lead_time_s"]
        lines.extend([
            "",
            "## Ground-truth accuracy",
            "",
            f"- Speed matched: {sa['matched_count']}/{sa['reference_count']} | MAE={sa['mae_kmh']} km/h | RMSE={sa['rmse_kmh']} km/h | MaxAbs={sa['max_abs_error_kmh']} km/h",
            f"- Event matching: TP={ea['tp']}, FP={ea['fp']}, FN={ea['fn']} | Precision={ea['precision']} | Recall={ea['recall']} | F1={ea['f1']}",
            f"- Lead time (s): count={lt['count']}, min={lt['min']}, mean={lt['mean']}, max={lt['max']}",
            "",
        ])
    else:
        lines.extend([
            "",
            "> This is descriptive output without ground truth; it is not an accuracy evaluation.",
            "",
        ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评价一次 analysis 运行产物（支持可选人工真值）")
    parser.add_argument("--run", required=True, help="runs/analysis/run-* 目录")
    parser.add_argument("--ground-truth", help="可选人工真值标注文件 (.yaml/.yml/.json)")
    parser.add_argument("--output", help="输出 JSON/Markdown 文件；不指定则打印 JSON")
    parser.add_argument("--markdown", action="store_true", help="输出简短 Markdown 而不是 JSON")
    parser.add_argument("--validate", action="store_true", help="评价前严格校验当前输出契约")
    args = parser.parse_args(argv)
    # 必须在 Path() 解析和写出前拦截空白 CLI 路径参数：
    # 否则 Path("") 会把 --run / --ground-truth 解析为当前目录 '.'，而空白 --output 会静默回退或误写空白文件。
    if not args.run.strip():
        parser.error("--run 不能为空")
    if args.ground_truth is not None and not args.ground_truth.strip():
        parser.error("--ground-truth 不能为空")
    if args.output is not None and not args.output.strip():
        parser.error("--output 不能为空")
    result = evaluate_run(args.run, validate=args.validate, ground_truth=args.ground_truth)
    content = (
        to_markdown(result)
        if args.markdown
        else json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    )
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
