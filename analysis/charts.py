# -*- coding: utf-8 -*-
"""
离线无界面实验图表生成器 (Headless Matplotlib Experiment Charts)
=============================================================================
强制在导入 pyplot 之前切换 Agg 无界面后端，并全量采用标准英文/ASCII 图表标签，
确保在无图形界面、无中文字体的 Ubuntu Linux CI 与 -W error::RuntimeWarning
环境下零警告生成高质量论文/报告 PNG 图表。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from detection.core import is_blank_path, prepare_runtime

# 必须在导入 matplotlib.pyplot 之前配置可写缓存目录并强制使用无界面 Agg 后端
prepare_runtime()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .benchmark import SUBSTAGE_NAMES

# 全局统一无衬线字体与负号配置（仅使用 matplotlib 内置字体，不依赖系统外部字体包）
_BASE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.unicode_minus": False,
}

_STAGE_DISPLAY_LABELS: dict[str, str] = {
    "decode": "Decode",
    "detect_track": "Detect+Track",
    "geometry_speed": "Geom+Speed",
    "conflict_events": "Conflict+Evt",
    "render_write": "Render+Write",
}


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限常量: {value}")


def _reject_non_finite_tree(value: Any, label: str) -> None:
    """递归拦截 JSON/对象树中的 NaN/Inf，防止绘图或汇总污染。"""
    if isinstance(value, (float, np.floating)) or type(value) is int:
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} 包含非有限数值")
    elif isinstance(value, Mapping):
        for key, val in value.items():
            _reject_non_finite_tree(val, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _reject_non_finite_tree(item, f"{label}[{idx}]")


def _finite_float(value: Any, label: str, *, non_negative: bool = False) -> float:
    """严格解析有限浮点数，显式排斥 bool / np.bool_ / NaN / Inf。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} 不能为布尔值")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} 必须为数值")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{label} 必须为有限数值")
    if non_negative and num < 0.0:
        raise ValueError(f"{label} 不能为负数")
    return num


def _validate_path_arg(path: Any, label: str) -> Path:
    """校验输入/输出路径参数非空，阻断空白路径退化为当前目录 '.'。"""
    if not isinstance(path, (str, os.PathLike)) or is_blank_path(path):
        raise ValueError(f"{label} 不能为空")
    return Path(path).expanduser().resolve()


def _load_json_strict(path: Any, label: str) -> dict:
    """严格读取 JSON 对象文件并检查全树有限性。"""
    resolved = _validate_path_arg(path, label)
    if not resolved.is_file():
        raise FileNotFoundError(f"未找到 {label}: {resolved}")
    try:
        data = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{resolved} JSON 解析失败: {exc}") from None
    _reject_non_finite_tree(data, str(resolved))
    if not isinstance(data, dict):
        raise ValueError(f"{resolved} 顶层必须是 JSON 对象")
    return data


def _load_jsonl_strict(path: Path) -> list[dict]:
    """严格读取 JSONL 列表文件。"""
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_non_finite_constant)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是有效 JSON: {exc}") from None
        _reject_non_finite_tree(row, f"{path}:{line_no}")
        if not isinstance(row, dict):
            raise ValueError(f"{path} 第 {line_no} 行必须是 JSON 对象")
        rows.append(row)
    return rows


def _ensure_output_dir(output_dir: Any) -> Path:
    out_root = _validate_path_arg(output_dir, "output_dir")
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def _save_figure(fig: plt.Figure, out_path: Path) -> str:
    """在 finally 保护下写出 PNG 并释放图形句柄，防止内存或 figure 数量告警。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format="png", dpi=150, bbox_inches="tight")
    return str(out_path)


def _risk_level_num(value: Any) -> int:
    """将数字或 LEVEL_X_NAME 字符串归一化为整数风险等级 0..3。"""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return max(0, min(3, int(value)))
    if isinstance(value, str):
        upper = value.strip().upper()
        for lvl in (3, 2, 1, 0):
            if f"LEVEL_{lvl}" in upper or upper == str(lvl):
                return lvl
    return 0


def plot_ablation_charts(
    ablation_data: Mapping[str, Any] | str | os.PathLike,
    output_dir: str | os.PathLike,
) -> list[str]:
    """根据消融实验结果绘制测速模型 (EMA/Kalman/CTRV) 与冲突模型对比 PNG 图表。"""
    if isinstance(ablation_data, (str, os.PathLike)):
        data: Mapping[str, Any] = _load_json_strict(ablation_data, "ablation_data")
    elif isinstance(ablation_data, Mapping):
        data = ablation_data
        _reject_non_finite_tree(dict(data), "ablation_data")
    else:
        raise ValueError("ablation_data 必须是消融结果字典或 JSON 文件路径")

    out_root = _ensure_output_dir(output_dir)
    generated: list[str] = []

    alg_comp = data.get("algorithm_comparison") if isinstance(data.get("algorithm_comparison"), Mapping) else {}
    speed_est = alg_comp.get("speed_estimation") if isinstance(alg_comp.get("speed_estimation"), Mapping) else {}
    turn_rows = data.get("turn_speed_ablation")
    if turn_rows is None and isinstance(alg_comp, Mapping):
        turn_rows = alg_comp.get("turn_speed_ablation")
    speed_grid = data.get("speed")
    conflict_det = alg_comp.get("conflict_detection") if isinstance(alg_comp, Mapping) else None
    conflict_grid = data.get("conflict")

    # 1) 测速误差对比图：兼容直线加速 MAE/RMSE (m/s -> km/h) 与弯道 CTRV 消融对比
    has_speed_est = any(k in speed_est and isinstance(speed_est[k], Mapping) for k in ("ema", "kalman", "ctrv"))
    has_turn_rows = isinstance(turn_rows, list) and len(turn_rows) > 0
    has_speed_grid = isinstance(speed_grid, list) and len(speed_grid) > 0

    if has_speed_est or has_turn_rows or has_speed_grid:
        speed_path = out_root / "ablation_speed_mae.png"
        with plt.rc_context(_BASE_RC):
            if has_speed_est and has_turn_rows:
                fig, (ax_lin, ax_turn) = plt.subplots(1, 2, figsize=(11.0, 4.6))
            else:
                fig, ax_single = plt.subplots(figsize=(7.2, 4.6))
                ax_lin = ax_single if has_speed_est else None
                ax_turn = ax_single if has_turn_rows and not has_speed_est else None
            try:
                if ax_lin is not None:
                    modes = [m for m in ("ema", "kalman", "ctrv") if isinstance(speed_est.get(m), Mapping)]
                    labels = [m.upper() if m != "kalman" else "Kalman (CA)" for m in modes]
                    mae_kmh = []
                    rmse_kmh = []
                    has_rmse = all("rmse_ms" in speed_est[m] or "rmse_kmh" in speed_est[m] for m in modes)
                    for m in modes:
                        entry = speed_est[m]
                        if "mae_ms" in entry:
                            mae_kmh.append(
                                _finite_float(entry["mae_ms"], f"speed_estimation.{m}.mae_ms", non_negative=True) * 3.6
                            )
                        elif "mae_kmh" in entry:
                            mae_kmh.append(
                                _finite_float(entry["mae_kmh"], f"speed_estimation.{m}.mae_kmh", non_negative=True)
                            )
                        else:
                            raise ValueError(f"speed_estimation.{m} 缺少 mae_ms 或 mae_kmh 字段")
                        if has_rmse:
                            if "rmse_ms" in entry:
                                rmse_kmh.append(
                                    _finite_float(entry["rmse_ms"], f"speed_estimation.{m}.rmse_ms", non_negative=True) * 3.6
                                )
                            else:
                                rmse_kmh.append(
                                    _finite_float(entry["rmse_kmh"], f"speed_estimation.{m}.rmse_kmh", non_negative=True)
                                )
                    x = np.arange(len(modes))
                    if has_rmse:
                        width = 0.34
                        bars1 = ax_lin.bar(x - width / 2, mae_kmh, width, label="MAE (km/h)", color="#2b6cb0")
                        bars2 = ax_lin.bar(x + width / 2, rmse_kmh, width, label="RMSE (km/h)", color="#63b3ed")
                        all_bars = list(bars1) + list(bars2)
                        max_y = max([0.5, *mae_kmh, *rmse_kmh])
                    else:
                        bars1 = ax_lin.bar(x, mae_kmh, 0.45, label="MAE (km/h)", color="#2b6cb0")
                        all_bars = list(bars1)
                        max_y = max([0.5, *mae_kmh])
                    for bar in all_bars:
                        h = bar.get_height()
                        ax_lin.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 0.02 * max(1.0, max_y),
                            f"{h:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                    ax_lin.set_xticks(x)
                    ax_lin.set_xticklabels(labels)
                    ax_lin.set_ylabel("Speed MAE (km/h)")
                    ax_lin.set_title("Perspective Speed Estimation Error")
                    ax_lin.set_ylim(0.0, max_y * 1.25)
                    ax_lin.grid(axis="y", linestyle="--", alpha=0.35)
                    ax_lin.legend(frameon=False)

                if ax_turn is not None and isinstance(turn_rows, list):
                    grouped: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
                    for idx, row in enumerate(turn_rows):
                        if not isinstance(row, Mapping):
                            raise ValueError(f"turn_speed_ablation[{idx}] 必须是字典")
                        mode = str(row.get("estimator_mode", "")).strip().lower()
                        if not mode:
                            raise ValueError(f"turn_speed_ablation[{idx}].estimator_mode 不能为空")
                        mean_err = _finite_float(
                            row["mean_abs_error_kmh"],
                            f"turn_speed_ablation[{idx}].mean_abs_error_kmh",
                            non_negative=True,
                        )
                        max_err = _finite_float(
                            row.get("max_abs_error_kmh", mean_err),
                            f"turn_speed_ablation[{idx}].max_abs_error_kmh",
                            non_negative=True,
                        )
                        grouped[mode][0].append(mean_err)
                        grouped[mode][1].append(max_err)
                    ordered_modes = [m for m in ("ema", "kalman", "ctrv") if m in grouped] + [
                        m for m in sorted(grouped) if m not in ("ema", "kalman", "ctrv")
                    ]
                    t_labels = [
                        {"ema": "EMA", "kalman": "Kalman (CA)", "ctrv": "CTRV"}.get(m, m.upper())
                        for m in ordered_modes
                    ]
                    t_mae = [float(np.mean(grouped[m][0])) for m in ordered_modes]
                    t_max = [float(np.mean(grouped[m][1])) for m in ordered_modes]
                    x = np.arange(len(ordered_modes))
                    width = 0.34
                    b1 = ax_turn.bar(x - width / 2, t_mae, width, label="Mean MAE (km/h)", color="#2f855a")
                    b2 = ax_turn.bar(x + width / 2, t_max, width, label="Mean Max Err (km/h)", color="#9ae6b4")
                    for bar in list(b1) + list(b2):
                        h = bar.get_height()
                        ax_turn.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 0.02 * max([1.0, *t_mae, *t_max]),
                            f"{h:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                    ax_turn.set_xticks(x)
                    ax_turn.set_xticklabels(t_labels)
                    ax_turn.set_ylabel("Speed MAE (km/h)")
                    ax_turn.set_title("Curved Turn Speed Ablation")
                    ax_turn.set_ylim(0.0, max([0.5, *t_mae, *t_max]) * 1.25)
                    ax_turn.grid(axis="y", linestyle="--", alpha=0.35)
                    ax_turn.legend(frameon=False)

                if not has_speed_est and not has_turn_rows and isinstance(speed_grid, list):
                    by_cfg: dict[str, list[float]] = defaultdict(list)
                    for idx, row in enumerate(speed_grid):
                        if not isinstance(row, Mapping):
                            raise ValueError(f"speed[{idx}] 必须是字典")
                        cfg_label = f"W{row.get('window_size', '?')}-A{row.get('ema_alpha', '?')}"
                        err_kmh = _finite_float(
                            row["absolute_error_ms"], f"speed[{idx}].absolute_error_ms", non_negative=True
                        ) * 3.6
                        by_cfg[cfg_label].append(err_kmh)
                    cfg_keys = list(by_cfg.keys())[:12]
                    cfg_vals = [float(np.mean(by_cfg[k])) for k in cfg_keys]
                    x = np.arange(len(cfg_keys))
                    ax_single.bar(x, cfg_vals, color="#2b6cb0", width=0.55)
                    ax_single.set_xticks(x)
                    ax_single.set_xticklabels(cfg_keys, rotation=25, ha="right", fontsize=8)
                    ax_single.set_ylabel("Speed MAE (km/h)")
                    ax_single.set_title("EMA Window and Alpha Speed Error")
                    ax_single.set_ylim(0.0, max([0.2, *cfg_vals]) * 1.25)
                    ax_single.grid(axis="y", linestyle="--", alpha=0.35)

                fig.tight_layout()
                generated.append(_save_figure(fig, speed_path))
            finally:
                plt.close(fig)

    # 2) 冲突研判指标对比图：对比 CVM 与 OBB-SSM 的风险等级、TTC/MTTC、DRAC 与碰撞概率
    has_conflict_det = isinstance(conflict_det, list) and len(conflict_det) > 0
    has_conflict_grid = isinstance(conflict_grid, list) and len(conflict_grid) > 0

    if has_conflict_det or has_conflict_grid:
        conflict_path = out_root / "ablation_conflict_metrics.png"
        with plt.rc_context(_BASE_RC):
            if has_conflict_det:
                fig, (ax_risk, ax_ssm) = plt.subplots(1, 2, figsize=(11.2, 4.6))
            else:
                fig, ax_grid = plt.subplots(figsize=(7.6, 4.6))
            try:
                if has_conflict_det and isinstance(conflict_det, list):
                    scenarios: list[str] = []
                    cvm_lvls: list[int] = []
                    obb_lvls: list[int] = []
                    drac_vals: list[float] = []
                    prob_pcts: list[float] = []

                    for idx, row in enumerate(conflict_det):
                        if not isinstance(row, Mapping):
                            raise ValueError(f"conflict_detection[{idx}] 必须是字典")
                        scenarios.append(str(row.get("scenario", f"case_{idx}")))
                        cvm_sub = row.get("cvm_circle", {})
                        obb_sub = row.get("obb_ssm", {})
                        cvm_lvls.append(_risk_level_num(cvm_sub.get("risk_level")))
                        obb_lvls.append(_risk_level_num(obb_sub.get("risk_level")))
                        raw_drac = obb_sub.get("drac_ms2")
                        drac_vals.append(
                            _finite_float(raw_drac, f"conflict[{idx}].drac_ms2", non_negative=True)
                            if raw_drac is not None
                            else 0.0
                        )
                        raw_prob = obb_sub.get("collision_prob")
                        prob_pcts.append(
                            _finite_float(raw_prob, f"conflict[{idx}].collision_prob", non_negative=True) * 100.0
                            if raw_prob is not None
                            else 0.0
                        )

                    x = np.arange(len(scenarios))
                    width = 0.34
                    b_cvm = ax_risk.bar(x - width / 2, cvm_lvls, width, label="CVM Circle", color="#718096")
                    b_obb = ax_risk.bar(x + width / 2, obb_lvls, width, label="OBB-SSM", color="#dd6b20")
                    for bar in list(b_cvm) + list(b_obb):
                        h = bar.get_height()
                        ax_risk.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 0.08,
                            f"L{int(round(h))}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                    ax_risk.set_xticks(x)
                    ax_risk.set_xticklabels(scenarios, rotation=15, ha="right")
                    ax_risk.set_yticks([0, 1, 2, 3])
                    ax_risk.set_ylabel("Metric Score")
                    ax_risk.set_title("Conflict Risk Level: CVM vs OBB-SSM")
                    ax_risk.set_ylim(0.0, 3.6)
                    ax_risk.grid(axis="y", linestyle="--", alpha=0.35)
                    ax_risk.legend(frameon=False)

                    b_drac = ax_ssm.bar(x - width / 2, drac_vals, width, label="DRAC (m/s^2)", color="#c53030")
                    ax_prob = ax_ssm.twinx()
                    b_prob = ax_prob.bar(
                        x + width / 2, prob_pcts, width, label="Collision Prob (%)", color="#3182ce"
                    )
                    for bar in b_drac:
                        h = bar.get_height()
                        ax_ssm.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 0.03 * max([1.0, *drac_vals]),
                            f"{h:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                    for bar in b_prob:
                        h = bar.get_height()
                        ax_prob.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 2.0,
                            f"{h:.1f}%",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                    ax_ssm.set_xticks(x)
                    ax_ssm.set_xticklabels(scenarios, rotation=15, ha="right")
                    ax_ssm.set_ylabel("DRAC (m/s^2)")
                    ax_prob.set_ylabel("Collision Probability (%)")
                    ax_ssm.set_ylim(0.0, max([1.0, *drac_vals]) * 1.25)
                    ax_prob.set_ylim(0.0, 115.0)
                    ax_ssm.set_title("OBB-SSM Severity Indicators")
                    ax_ssm.grid(axis="y", linestyle="--", alpha=0.35)
                    h1, l1 = ax_ssm.get_legend_handles_labels()
                    h2, l2 = ax_prob.get_legend_handles_labels()
                    ax_ssm.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
                elif isinstance(conflict_grid, list):
                    labels = [
                        f"C{row.get('ttc_critical_s', '?')}/H{row.get('horizon_s', '?')}"
                        for row in conflict_grid
                    ]
                    ttcs = [
                        _finite_float(row["ttc_s"], f"conflict[{i}].ttc_s", non_negative=True)
                        if row.get("ttc_s") is not None
                        else 0.0
                        for i, row in enumerate(conflict_grid)
                    ]
                    dists = [
                        _finite_float(row["min_distance_m"], f"conflict[{i}].min_distance_m", non_negative=True)
                        if row.get("min_distance_m") is not None
                        else 0.0
                        for i, row in enumerate(conflict_grid)
                    ]
                    x = np.arange(len(labels))
                    width = 0.35
                    ax_grid.bar(x - width / 2, ttcs, width, label="TTC (s)", color="#dd6b20")
                    ax_grid.bar(x + width / 2, dists, width, label="Min Distance (m)", color="#3182ce")
                    ax_grid.set_xticks(x)
                    ax_grid.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
                    ax_grid.set_ylabel("Metric Score")
                    ax_grid.set_title("TTC Threshold Ablation")
                    ax_grid.grid(axis="y", linestyle="--", alpha=0.35)
                    ax_grid.legend(frameon=False)

                fig.tight_layout()
                generated.append(_save_figure(fig, conflict_path))
            finally:
                plt.close(fig)

    if not generated:
        raise ValueError("ablation_data 未包含可绘制的速度或冲突对比字段")
    return generated


def _resolve_speed_limit_kmh(run_root: Path) -> float:
    """从 configuration.json 或 summary.json 读取限速阈值，缺省回退 30.0 km/h。"""
    for fname in ("configuration.json", "summary.json"):
        cfg_file = run_root / fname
        if cfg_file.is_file():
            cfg = _load_json_strict(cfg_file, str(cfg_file))
            raw_limit = cfg.get("speed_limit_kmh")
            if raw_limit is not None:
                val = _finite_float(raw_limit, f"{fname}.speed_limit_kmh", non_negative=True)
                if val > 0.0:
                    return val
    return 30.0


def plot_run_timeline(
    run_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
) -> list[str]:
    """绘制运行目录内 speeds.csv 车辆时速曲线、30 km/h 限速红线与冲突事件时间窗。"""
    run_root = _validate_path_arg(run_dir, "run_dir")
    if not run_root.is_dir():
        raise FileNotFoundError(f"运行目录不存在: {run_root}")
    speeds_csv = run_root / "speeds.csv"
    if not speeds_csv.is_file():
        raise FileNotFoundError(f"运行目录缺少 speeds.csv: {speeds_csv}")

    out_root = _ensure_output_dir(output_dir)
    speed_limit = _resolve_speed_limit_kmh(run_root)

    veh_series: dict[int, list[tuple[float, float]]] = defaultdict(list)
    all_timestamps: list[float] = []
    with speeds_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row_idx, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"{speeds_csv} 第 {row_idx} 行包含多余列")
            try:
                ts = float(row["timestamp_s"])
                spd = float(row["speed_kmh"])
                tid = int(row["track_id"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"{speeds_csv} 第 {row_idx} 行数值字段无效") from None
            if not (math.isfinite(ts) and math.isfinite(spd)) or ts < 0.0 or spd < 0.0:
                raise ValueError(f"{speeds_csv} 第 {row_idx} 行包含非有限或负数值")
            all_timestamps.append(ts)
            if str(row.get("category", "")).strip().lower() == "vehicle":
                veh_series[tid].append((ts, spd))

    events = _load_jsonl_strict(run_root / "events.jsonl")
    conflicts = _load_jsonl_strict(run_root / "conflicts.jsonl")

    # 从 events.jsonl 提取冲突/超速事件时间窗；若无 events.jsonl 则从 conflicts.jsonl 提取高风险帧段
    conflict_windows: list[tuple[float, float, int, Optional[int]]] = []
    speeding_windows: list[tuple[float, float, Optional[int]]] = []
    for ev in events:
        t0 = _finite_float(ev.get("start_timestamp_s", 0.0), "event.start_timestamp_s", non_negative=True)
        t1 = _finite_float(ev.get("end_timestamp_s", t0), "event.end_timestamp_s", non_negative=True)
        if t1 < t0:
            t1 = t0
        lvl = _risk_level_num(ev.get("peak_risk_level", 2))
        eid = ev.get("event_id")
        if ev.get("kind") == "conflict":
            conflict_windows.append((t0, t1, lvl, eid if type(eid) is int else None))
        elif ev.get("kind") == "speeding":
            speeding_windows.append((t0, t1, eid if type(eid) is int else None))

    if not conflict_windows and conflicts:
        # 兼容早期无 events.jsonl 的运行目录：按连续 LEVEL_2/LEVEL_3 帧合成时间窗
        cur_start: Optional[float] = None
        cur_end: Optional[float] = None
        cur_lvl = 0
        for row in conflicts:
            ts = _finite_float(row.get("timestamp_s", 0.0), "conflict.timestamp_s", non_negative=True)
            lvl = _risk_level_num(row.get("assessment", row).get("risk_level"))
            if lvl >= 2:
                if cur_start is None:
                    cur_start = ts
                    cur_end = ts
                    cur_lvl = lvl
                else:
                    cur_end = ts
                    cur_lvl = max(cur_lvl, lvl)
            elif cur_start is not None:
                conflict_windows.append((cur_start, cur_end if cur_end is not None else cur_start, cur_lvl, None))
                cur_start = None
        if cur_start is not None:
            conflict_windows.append((cur_start, cur_end if cur_end is not None else cur_start, cur_lvl, None))

    generated: list[str] = []
    timeline_path = out_root / "run_speed_timeline.png"
    with plt.rc_context(_BASE_RC):
        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        try:
            # 对单帧零宽度时间窗做轻微可视展宽，避免 axvspan 在图上不可见
            span_pad = 0.04
            conflict_labeled = False
            for t0, t1, lvl, _eid in conflict_windows:
                lo = max(0.0, t0 - span_pad) if math.isclose(t0, t1) else t0
                hi = (t1 + span_pad) if math.isclose(t0, t1) else t1
                color = "#e53e3e" if lvl >= 3 else "#dd6b20"
                lbl = "Conflict Window" if not conflict_labeled else None
                ax.axvspan(lo, hi, color=color, alpha=0.22, label=lbl)
                conflict_labeled = True

            speeding_labeled = False
            for t0, t1, _eid in speeding_windows:
                lo = max(0.0, t0 - span_pad) if math.isclose(t0, t1) else t0
                hi = (t1 + span_pad) if math.isclose(t0, t1) else t1
                lbl = "Speeding Window" if not speeding_labeled else None
                ax.axvspan(lo, hi, color="#ecc94b", alpha=0.18, label=lbl)
                speeding_labeled = True

            # 按观测长度与峰值车速排序，避免多车流下 legend 过长触发紧排告警
            ordered_ids = sorted(
                veh_series.keys(),
                key=lambda tid: (len(veh_series[tid]), max(s for _, s in veh_series[tid])),
                reverse=True,
            )
            max_spd_seen = speed_limit
            for idx, tid in enumerate(ordered_ids):
                pts = sorted(veh_series[tid], key=lambda pair: pair[0])
                ts_vals = [p[0] for p in pts]
                sp_vals = [p[1] for p in pts]
                if sp_vals:
                    max_spd_seen = max(max_spd_seen, max(sp_vals))
                lbl = f"Vehicle #{tid}" if idx < 8 else None
                ax.plot(ts_vals, sp_vals, marker="o", markersize=2.8, linewidth=1.6, label=lbl)

            if not ordered_ids:
                ax.text(
                    0.5,
                    0.5,
                    "No vehicle speed samples recorded",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#4a5568",
                )

            ax.axhline(
                speed_limit,
                color="#d62728",
                linestyle="--",
                linewidth=1.8,
                label=f"Speed Limit ({speed_limit:.0f} km/h)",
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (km/h)")
            ax.set_title("Vehicle Speed Timeline and Conflict Windows")
            ax.set_ylim(0.0, max(speed_limit * 1.25, max_spd_seen * 1.15, 5.0))
            if all_timestamps:
                t_min, t_max = min(all_timestamps), max(all_timestamps)
                if t_max > t_min:
                    ax.set_xlim(max(0.0, t_min - 0.05 * (t_max - t_min)), t_max + 0.05 * (t_max - t_min))
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.88)
            fig.tight_layout()
            generated.append(_save_figure(fig, timeline_path))
        finally:
            plt.close(fig)

    # 若存在逐帧冲突记录，补充一张风险等级与 TTC/MTTC 演变曲线图
    if conflicts:
        risk_path = out_root / "run_risk_timeline.png"
        frame_risk: dict[float, int] = {}
        frame_ttc: dict[float, float] = {}
        for row in conflicts:
            ts = _finite_float(row.get("timestamp_s", 0.0), "conflict.timestamp_s", non_negative=True)
            ass = row.get("assessment", row)
            lvl = _risk_level_num(ass.get("risk_level"))
            frame_risk[ts] = max(frame_risk.get(ts, 0), lvl)
            raw_ttc = ass.get("mttc_seconds") if ass.get("mttc_seconds") is not None else ass.get("ttc_seconds")
            if raw_ttc is not None:
                ttc_val = _finite_float(raw_ttc, "conflict.ttc_seconds", non_negative=True)
                if ts not in frame_ttc or ttc_val < frame_ttc[ts]:
                    frame_ttc[ts] = ttc_val

        sorted_ts = sorted(frame_risk.keys())
        with plt.rc_context(_BASE_RC):
            fig, ax_r = plt.subplots(figsize=(9.6, 4.0))
            try:
                ax_r.step(
                    sorted_ts,
                    [frame_risk[t] for t in sorted_ts],
                    where="post",
                    color="#c53030",
                    linewidth=1.8,
                    label="Peak Risk Level",
                )
                ax_r.set_yticks([0, 1, 2, 3])
                ax_r.set_ylim(-0.1, 3.4)
                ax_r.set_xlabel("Time (s)")
                ax_r.set_ylabel("Risk Level (0-3)")
                ax_r.set_title("Frame-level Risk and Minimum TTC Timeline")
                ax_r.grid(True, linestyle="--", alpha=0.35)

                if frame_ttc:
                    ttc_ts = sorted(frame_ttc.keys())
                    ax_t = ax_r.twinx()
                    ax_t.plot(
                        ttc_ts,
                        [frame_ttc[t] for t in ttc_ts],
                        color="#2b6cb0",
                        linestyle="-.",
                        marker=".",
                        linewidth=1.4,
                        label="Min TTC/MTTC (s)",
                    )
                    ax_t.set_ylabel("TTC / MTTC (s)")
                    ax_t.set_ylim(0.0, max(4.0, max(frame_ttc.values()) * 1.2))
                    h1, l1 = ax_r.get_legend_handles_labels()
                    h2, l2 = ax_t.get_legend_handles_labels()
                    ax_r.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, frameon=True)
                else:
                    ax_r.legend(loc="upper right", fontsize=8, frameon=True)

                fig.tight_layout()
                generated.append(_save_figure(fig, risk_path))
            finally:
                plt.close(fig)

    return generated


def _normalize_mot_reports(reports: Any) -> list[tuple[str, Mapping[str, Any]]]:
    """统一解析 MOT 对比输入（字典映射、二元组列表或带 label 的报告列表）。"""
    items: list[tuple[str, Any]] = []
    if isinstance(reports, Mapping):
        items = [(str(k), v) for k, v in reports.items()]
    elif isinstance(reports, Sequence) and not isinstance(reports, (str, bytes)):
        for idx, entry in enumerate(reports):
            if isinstance(entry, tuple) and len(entry) == 2:
                items.append((str(entry[0]), entry[1]))
            elif isinstance(entry, Mapping):
                label = entry.get("label") or entry.get("name") or entry.get("model") or f"Run {idx + 1}"
                items.append((str(label), entry))
            else:
                raise ValueError(f"reports[{idx}] 必须是 (label, report) 元组或报告字典")
    else:
        raise ValueError("reports 必须是映射或报告序列")

    if not items:
        raise ValueError("reports 不能为空")

    resolved: list[tuple[str, Mapping[str, Any]]] = []
    for idx, (label, payload) in enumerate(items):
        clean_label = label.strip()
        if not clean_label:
            raise ValueError(f"reports 第 {idx} 项标签不能为空")
        if isinstance(payload, (str, os.PathLike)):
            rep_dict: Mapping[str, Any] = _load_json_strict(payload, f"mot_report[{clean_label}]")
        elif isinstance(payload, Mapping):
            _reject_non_finite_tree(dict(payload), f"mot_report[{clean_label}]")
            rep_dict = payload
        else:
            raise ValueError(f"mot_report[{clean_label}] 必须是报告字典或 JSON 文件路径")
        resolved.append((clean_label, rep_dict))
    return resolved


def plot_mot_comparison(
    reports: Any,
    output_path: str | os.PathLike,
) -> str:
    """绘制多组跟踪配置的 MOTA / IDF1 / HOTA / IDSW 分组柱状对比图。"""
    out_file = _validate_path_arg(output_path, "output_path")
    parsed = _normalize_mot_reports(reports)

    labels: list[str] = []
    mota_vals: list[float] = []
    idf1_vals: list[float] = []
    hota_vals: list[float] = []
    idsw_vals: list[float] = []

    for label, rep in parsed:
        for req_k in ("mota", "idf1", "hota"):
            if req_k not in rep:
                raise ValueError(f"MOT 报告 {label!r} 缺少字段 {req_k}")
        raw_idsw = rep.get("id_switches", rep.get("idsw"))
        if raw_idsw is None:
            raise ValueError(f"MOT 报告 {label!r} 缺少字段 id_switches")

        labels.append(label)
        mota_vals.append(_finite_float(rep["mota"], f"{label}.mota"))
        idf1_vals.append(_finite_float(rep["idf1"], f"{label}.idf1", non_negative=True))
        hota_vals.append(_finite_float(rep["hota"], f"{label}.hota", non_negative=True))
        idsw_vals.append(_finite_float(raw_idsw, f"{label}.id_switches", non_negative=True))

    with plt.rc_context(_BASE_RC):
        fig, (ax_scores, ax_idsw) = plt.subplots(
            1, 2, figsize=(10.8, 4.6), gridspec_kw={"width_ratios": [2.2, 1.0]}
        )
        try:
            x = np.arange(len(labels))
            width = 0.25
            b_mota = ax_scores.bar(x - width, mota_vals, width, label="MOTA", color="#2b6cb0")
            b_idf1 = ax_scores.bar(x, idf1_vals, width, label="IDF1", color="#2f855a")
            b_hota = ax_scores.bar(x + width, hota_vals, width, label="HOTA", color="#dd6b20")

            all_scores = mota_vals + idf1_vals + hota_vals
            y_min = min(0.0, min(all_scores) - 0.12)
            y_max = max(1.08, max(all_scores) + 0.14)
            for bar in list(b_mota) + list(b_idf1) + list(b_hota):
                h = bar.get_height()
                va = "bottom" if h >= 0 else "top"
                offset = 0.015 if h >= 0 else -0.02
                ax_scores.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + offset,
                    f"{h:.3f}",
                    ha="center",
                    va=va,
                    fontsize=8,
                )
            ax_scores.axhline(0.0, color="#4a5568", linewidth=0.8)
            ax_scores.set_xticks(x)
            ax_scores.set_xticklabels(labels, rotation=12 if len(labels) > 3 else 0)
            ax_scores.set_ylabel("Metric Score")
            ax_scores.set_title("MOT Accuracy Comparison (MOTA / IDF1 / HOTA)")
            ax_scores.set_ylim(y_min, y_max)
            ax_scores.grid(axis="y", linestyle="--", alpha=0.35)
            ax_scores.legend(frameon=False, loc="lower right")

            b_id = ax_idsw.bar(x, idsw_vals, 0.5, color="#c53030", label="IDSW")
            max_idsw = max(idsw_vals) if idsw_vals else 0.0
            for bar in b_id:
                h = bar.get_height()
                ax_idsw.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max(0.05, max_idsw * 0.03),
                    f"{int(round(h))}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax_idsw.set_xticks(x)
            ax_idsw.set_xticklabels(labels, rotation=12 if len(labels) > 3 else 0)
            ax_idsw.set_ylabel("ID Switches (Count)")
            ax_idsw.set_title("ID Switches (Lower is Better)")
            ax_idsw.set_ylim(0.0, max(1.5, max_idsw * 1.25))
            ax_idsw.grid(axis="y", linestyle="--", alpha=0.35)

            fig.tight_layout()
            return _save_figure(fig, out_file)
        finally:
            plt.close(fig)


def plot_benchmark_breakdown(
    bench_data: Mapping[str, Any] | str | os.PathLike,
    output_path: str | os.PathLike,
) -> str:
    """绘制流水线各阶段耗时分布（均值/P95 柱状图 + 耗时占比饼图）。"""
    out_file = _validate_path_arg(output_path, "output_path")
    if isinstance(bench_data, (str, os.PathLike)):
        payload: Mapping[str, Any] = _load_json_strict(bench_data, "bench_data")
    elif isinstance(bench_data, Mapping):
        payload = bench_data
        _reject_non_finite_tree(dict(payload), "bench_data")
    else:
        raise ValueError("bench_data 必须是基准报告字典或 JSON 文件路径")

    stages_obj = payload.get("stages") if "stages" in payload else payload
    if not isinstance(stages_obj, Mapping) or not stages_obj:
        raise ValueError("bench_data 缺少有效的 stages 耗时统计")

    stage_labels: list[str] = []
    means: list[float] = []
    p95s: list[float] = []
    present_sub = 0

    for stage_key in SUBSTAGE_NAMES:
        row = stages_obj.get(stage_key)
        if row is not None:
            if not isinstance(row, Mapping):
                raise ValueError(f"stages.{stage_key} 必须是字典")
            present_sub += 1
            mean_v = _finite_float(row.get("mean_ms", 0.0), f"stages.{stage_key}.mean_ms", non_negative=True)
            p95_v = _finite_float(row.get("p95_ms", mean_v), f"stages.{stage_key}.p95_ms", non_negative=True)
        else:
            mean_v = 0.0
            p95_v = 0.0
        stage_labels.append(_STAGE_DISPLAY_LABELS[stage_key])
        means.append(mean_v)
        p95s.append(p95_v)

    if present_sub == 0:
        raise ValueError("bench_data.stages 未包含任何已知流水线子阶段")

    total_sum = sum(means)
    fps_val = payload.get("end_to_end_fps")
    model_load_val = payload.get("model_load_ms")
    frames_val = payload.get("frames_profiled")

    with plt.rc_context(_BASE_RC):
        fig, (ax_bar, ax_pie) = plt.subplots(
            1, 2, figsize=(11.4, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]}
        )
        try:
            x = np.arange(len(stage_labels))
            width = 0.35
            b_mean = ax_bar.bar(x - width / 2, means, width, label="Mean (ms)", color="#2b6cb0")
            b_p95 = ax_bar.bar(x + width / 2, p95s, width, label="P95 (ms)", color="#90cdf4")
            max_lat = max([0.1, *means, *p95s])
            for bar in list(b_mean) + list(b_p95):
                h = bar.get_height()
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max_lat * 0.02,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(stage_labels, rotation=15, ha="right")
            ax_bar.set_ylabel("Stage Latency (ms)")
            ax_bar.set_title("Pipeline Stage Latency (Mean vs P95)")
            ax_bar.set_ylim(0.0, max_lat * 1.25)
            ax_bar.grid(axis="y", linestyle="--", alpha=0.35)
            ax_bar.legend(frameon=False)

            colors = ["#2b6cb0", "#c53030", "#2f855a", "#dd6b20", "#805ad5"]
            if total_sum > 0.0:
                # 仅对正值切片绘制饼图，防止全零数组触发 divide-by-zero RuntimeWarning
                nz_idx = [i for i, v in enumerate(means) if v > 0.0]
                pie_vals = [means[i] for i in nz_idx]
                pie_labels = [f"{stage_labels[i]} ({100.0 * means[i] / total_sum:.1f}%)" for i in nz_idx]
                pie_colors = [colors[i % len(colors)] for i in nz_idx]
                wedges, _ = ax_pie.pie(
                    pie_vals,
                    colors=pie_colors,
                    startangle=90,
                    wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
                )
                ax_pie.legend(
                    wedges,
                    pie_labels,
                    loc="center left",
                    bbox_to_anchor=(0.95, 0.5),
                    fontsize=8,
                    frameon=False,
                )
                ax_pie.set_title("Stage Time Share (%)")
            else:
                ax_pie.axis("off")
                ax_pie.text(
                    0.5,
                    0.5,
                    "Zero positive stage latency recorded",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#4a5568",
                )
                ax_pie.set_title("Stage Time Share (%)")

            subtitle_parts = []
            if frames_val is not None and type(frames_val) is int:
                subtitle_parts.append(f"Frames: {frames_val}")
            if fps_val is not None and not isinstance(fps_val, (bool, np.bool_)):
                subtitle_parts.append(f"Throughput: {float(fps_val):.2f} FPS")
            if model_load_val is not None and not isinstance(model_load_val, (bool, np.bool_)):
                subtitle_parts.append(f"Model Load: {float(model_load_val):.1f} ms")
            if subtitle_parts:
                fig.suptitle(" | ".join(subtitle_parts), fontsize=10, y=1.01)

            fig.tight_layout()
            return _save_figure(fig, out_file)
        finally:
            plt.close(fig)


def _parse_mot_cli_item(raw: str) -> tuple[str, Path]:
    """解析 CLI --mot label=path 参数并严格拦截空白标签与空白路径。"""
    if "=" not in raw:
        raise ValueError(f"--mot 参数格式必须为 label=path，当前为 {raw!r}")
    label, raw_path = raw.split("=", 1)
    if not label.strip():
        raise ValueError("--mot 标签不能为空")
    if not raw_path.strip() or is_blank_path(raw_path):
        raise ValueError(f"--mot {label.strip()!r} 路径不能为空")
    return label.strip(), Path(raw_path).expanduser().resolve()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="生成离线无界面论文/实验对比图表 (PNG)")
    parser.add_argument("--ablation", type=str, default=None, help="analysis.ablation 输出 JSON 路径")
    parser.add_argument("--run", type=str, default=None, help="analysis 运行输出目录路径 (含 speeds.csv)")
    parser.add_argument("--benchmark", type=str, default=None, help="analysis.benchmark 输出 JSON 路径")
    parser.add_argument(
        "--mot",
        action="append",
        default=None,
        help="MOT 评测对比报告，格式为 label=path.json（可重复传入）",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="图表 PNG 保存目录")
    args = parser.parse_args(argv)

    # 严格空白路径拦截，防止 Path("") / Path("   ") 退化为当前目录 '.'
    if not args.output_dir.strip():
        parser.error("--output-dir 不能为空")
    if args.ablation is not None and not args.ablation.strip():
        parser.error("--ablation 不能为空")
    if args.run is not None and not args.run.strip():
        parser.error("--run 不能为空")
    if args.benchmark is not None and not args.benchmark.strip():
        parser.error("--benchmark 不能为空")
    if args.ablation is None and args.run is None and args.benchmark is None and not args.mot:
        parser.error("至少需提供 --ablation、--run、--benchmark 或 --mot 之一")

    mot_reports: dict[str, dict] = {}
    if args.mot:
        for raw_item in args.mot:
            try:
                label, path = _parse_mot_cli_item(raw_item)
            except ValueError as exc:
                parser.error(str(exc))
            if label in mot_reports:
                parser.error(f"--mot 存在重复标签: {label}")
            mot_reports[label] = _load_json_strict(path, f"--mot[{label}]")

    out_dir = _ensure_output_dir(args.output_dir)
    generated_files: list[str] = []

    if args.ablation is not None:
        ablation_obj = _load_json_strict(args.ablation, "--ablation")
        generated_files.extend(plot_ablation_charts(ablation_obj, out_dir))

    if args.run is not None:
        generated_files.extend(plot_run_timeline(args.run, out_dir))

    if args.benchmark is not None:
        bench_obj = _load_json_strict(args.benchmark, "--benchmark")
        generated_files.append(
            plot_benchmark_breakdown(bench_obj, out_dir / "benchmark_breakdown.png")
        )

    if mot_reports:
        generated_files.append(
            plot_mot_comparison(mot_reports, out_dir / "mot_comparison.png")
        )

    summary = {
        "schema_version": 1,
        "output_dir": str(out_dir),
        "generated_files": generated_files,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
