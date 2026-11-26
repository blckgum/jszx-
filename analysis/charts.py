# -*- coding: utf-8 -*-
"""
离线无界面实验图表生成器 (Headless Matplotlib Experiment Charts)
=============================================================================
强制在导入 pyplot 之前切换 Agg 无界面后端，并通过 matplotlib.font_manager
动态探测系统可用中文字体（优雅回退至 DejaVu Sans），在 Windows 桌面端渲染清晰
中文学术图表，同时确保在无中文字体的 Linux CI 与 -W error::RuntimeWarning 环境下
零崩溃、零告警稳定生成 PNG 图表。
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import csv
import json
import math
import os
import zlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from detection.core import is_blank_path, prepare_runtime

# 必须在导入 matplotlib.pyplot 之前配置可写缓存目录并强制使用无界面 Agg 后端
prepare_runtime()
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.ft2font as ft2font
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .benchmark import SUBSTAGE_NAMES

# 按优先级动态探测系统可用 CJK 字体，末尾始终保留 DejaVu Sans 作为安全回退
CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "STHeiti",
    "SimSun",
    "Arial Unicode MS",
)
FALLBACK_FONT = "DejaVu Sans"

_STAGE_DISPLAY_LABELS: dict[str, str] = {
    "decode": "解码",
    "detect_track": "检测与跟踪",
    "geometry_speed": "投影与测速",
    "conflict_events": "冲突与事件",
    "render_write": "渲染与写盘",
}

_STAGE_DISPLAY_LABELS_EN: dict[str, str] = {
    "decode": "Decode",
    "detect_track": "Detect & Track",
    "geometry_speed": "Geom & Speed",
    "conflict_events": "Conflict & Events",
    "render_write": "Render & Write",
}

_SPEED_MODEL_LABELS: dict[str, str] = {
    "ema": "滑窗 EMA",
    "kalman": "2D CA 卡尔曼",
    "ctrv": "弯道 CTRV",
}

_SPEED_MODEL_LABELS_EN: dict[str, str] = {
    "ema": "Sliding EMA",
    "kalman": "2D CA Kalman",
    "ctrv": "Turning CTRV",
}

_SCENARIO_DISPLAY_LABELS: dict[str, str] = {
    "parallel_pass": "窄路并行",
    "braking_yield": "制动礼让",
    "perpendicular_crossing": "垂直横穿",
    "head_on": "对向交汇",
    "crossing": "人车横穿",
}

_SCENARIO_DISPLAY_LABELS_EN: dict[str, str] = {
    "parallel_pass": "Parallel Pass",
    "braking_yield": "Braking Yield",
    "perpendicular_crossing": "Perp. Crossing",
    "head_on": "Head-On",
    "crossing": "Crossing",
}

_RISK_LEVEL_LABELS: tuple[str, str, str, str] = (
    "L0 正常",
    "L1 超速",
    "L2 注意",
    "L3 紧急",
)

_RISK_LEVEL_LABELS_EN: tuple[str, str, str, str] = (
    "L0 Safe",
    "L1 Speeding",
    "L2 Caution",
    "L3 Critical",
)

_LEGEND_STYLE: dict[str, Any] = {
    "frameon": True,
    "facecolor": "white",
    "framealpha": 0.92,
    "edgecolor": "#e2e8f0",
}

_FONT_CHARMAP_CACHE: dict[str, set[int]] = {}


def detect_available_cjk_fonts() -> list[str]:
    """按预设优先级返回当前环境中已安装的 CJK 无衬线字体名称列表。"""
    ttflist = getattr(fm.fontManager, "ttflist", ()) or ()
    available = {getattr(font, "name", "") for font in ttflist}
    return [name for name in CJK_FONT_CANDIDATES if name in available]


def configure_cjk_fonts() -> list[str]:
    """动态配置 Matplotlib 中文字体链与负号显示，并返回生效字体列表。"""
    selected = detect_available_cjk_fonts()
    if FALLBACK_FONT not in selected:
        selected.append(FALLBACK_FONT)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = list(selected)
    plt.rcParams["axes.unicode_minus"] = False
    return selected


# 模块初始化即完成一次字体与负号全局配置
configure_cjk_fonts()


@contextlib.contextmanager
def _chart_rc_context() -> Iterator[bool]:
    """在绘图上下文中固定字体链并返回是否启用中文标签（无 CJK 字体时切为英文，不抑制缺字告警）。"""
    fonts = configure_cjk_fonts()
    use_cjk = any(name != FALLBACK_FONT for name in fonts)
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": fonts,
        "axes.unicode_minus": False,
    }
    with plt.rc_context(rc):
        yield use_cjk


def _fallback_font_charmap() -> Optional[set[int]]:
    """读取回退字体（DejaVu Sans）的真实字形覆盖表，用于校验非 ASCII 用户标签。"""
    ttflist = getattr(fm.fontManager, "ttflist", ()) or ()
    for entry in ttflist:
        if getattr(entry, "name", "") == FALLBACK_FONT:
            fname = getattr(entry, "fname", None)
            if isinstance(fname, str) and fname and os.path.isfile(fname):
                cached = _FONT_CHARMAP_CACHE.get(fname)
                if cached is None:
                    cached = set(ft2font.FT2Font(fname).get_charmap().keys())
                    _FONT_CHARMAP_CACHE[fname] = cached
                return cached
    return None


def _safe_label(text: str, *, use_cjk: bool, fallback: str) -> str:
    """当无 CJK 字体时，确保外部传入标签不包含当前字体缺失的字符。"""
    if use_cjk or text.isascii():
        return text
    cmap = _fallback_font_charmap()
    if cmap is not None and all(ord(ch) in cmap for ch in text):
        return text
    if fallback.isascii():
        return fallback
    if cmap is not None and all(ord(ch) in cmap for ch in fallback):
        return fallback
    ascii_only = "".join(ch for ch in fallback if ord(ch) < 128).strip()
    return ascii_only if ascii_only else "Item"


def _display_stage(stage_key: str, *, use_cjk: bool = True) -> str:
    table = _STAGE_DISPLAY_LABELS if use_cjk else _STAGE_DISPLAY_LABELS_EN
    if stage_key in table:
        return table[stage_key]
    return _safe_label(stage_key, use_cjk=use_cjk, fallback=f"Stage ({zlib.crc32(stage_key.encode('utf-8')) & 0xFFFF:04x})")


def _display_speed_model(
    mode: str, *, use_cjk: bool = True, fallback_index: Optional[int] = None
) -> str:
    key = mode.strip().lower()
    table = _SPEED_MODEL_LABELS if use_cjk else _SPEED_MODEL_LABELS_EN
    if key in table:
        return table[key]
    fallback = (
        f"Custom Model #{fallback_index}"
        if fallback_index is not None
        else f"Model ({zlib.crc32(mode.encode('utf-8')) & 0xFFFF:04x})"
    )
    return _safe_label(mode.upper(), use_cjk=use_cjk, fallback=fallback)


def _display_scenario(name: str, *, use_cjk: bool = True, fallback: str = "Scenario") -> str:
    key = name.strip().lower()
    table = _SCENARIO_DISPLAY_LABELS if use_cjk else _SCENARIO_DISPLAY_LABELS_EN
    if key in table:
        return table[key]
    return _safe_label(name, use_cjk=use_cjk, fallback=fallback)


def _risk_level_labels(*, use_cjk: bool = True) -> list[str]:
    return list(_RISK_LEVEL_LABELS if use_cjk else _RISK_LEVEL_LABELS_EN)


def _format_speed_limit(value: float) -> str:
    """格式化限速数值，保留有效小数位并去除多余尾随零（如 30.0->30, 30.5->30.5）。"""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


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
    """写出高分辨率 PNG 并确保外边距无裁切。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format="png", dpi=180, bbox_inches="tight", pad_inches=0.14)
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
        with _chart_rc_context() as use_cjk:
            if has_speed_est and has_turn_rows:
                fig, (ax_lin, ax_turn) = plt.subplots(1, 2, figsize=(11.8, 4.9))
            else:
                fig, ax_single = plt.subplots(figsize=(7.6, 4.8))
                ax_lin = ax_single if has_speed_est else None
                ax_turn = ax_single if has_turn_rows and not has_speed_est else None
            try:
                if ax_lin is not None:
                    modes = [m for m in ("ema", "kalman", "ctrv") if isinstance(speed_est.get(m), Mapping)]
                    labels = [
                        _display_speed_model(m, use_cjk=use_cjk, fallback_index=i + 1)
                        for i, m in enumerate(modes)
                    ]
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
                    mae_label = "平均绝对误差 MAE (km/h)" if use_cjk else "MAE (km/h)"
                    rmse_label = "均方根误差 RMSE (km/h)" if use_cjk else "RMSE (km/h)"
                    if has_rmse:
                        width = 0.34
                        bars1 = ax_lin.bar(
                            x - width / 2, mae_kmh, width, label=mae_label, color="#2b6cb0"
                        )
                        bars2 = ax_lin.bar(
                            x + width / 2, rmse_kmh, width, label=rmse_label, color="#63b3ed"
                        )
                        all_bars = list(bars1) + list(bars2)
                        max_y = max([0.5, *mae_kmh, *rmse_kmh])
                    else:
                        bars1 = ax_lin.bar(
                            x, mae_kmh, 0.44, label=mae_label, color="#2b6cb0"
                        )
                        all_bars = list(bars1)
                        max_y = max([0.5, *mae_kmh])
                    for bar in all_bars:
                        h = bar.get_height()
                        ax_lin.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 0.025 * max(1.0, max_y),
                            f"{h:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8.5,
                        )
                    ax_lin.set_xticks(x)
                    ax_lin.set_xticklabels(labels, fontsize=9.5)
                    ax_lin.set_ylabel("测速误差 (km/h)" if use_cjk else "Speed Error (km/h)", fontsize=10)
                    ax_lin.set_title(
                        "透视噪声下直线加减速测速误差对比"
                        if use_cjk
                        else "Linear Acceleration Speed Error Under Perspective Noise",
                        fontsize=11.5,
                        pad=10,
                    )
                    ax_lin.set_ylim(0.0, max_y * 1.34)
                    ax_lin.grid(axis="y", linestyle="--", alpha=0.35)
                    ax_lin.legend(loc="upper right", fontsize=8.8, **_LEGEND_STYLE)

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
                        _display_speed_model(m, use_cjk=use_cjk, fallback_index=i + 1)
                        for i, m in enumerate(ordered_modes)
                    ]
                    t_mae = [float(np.mean(grouped[m][0])) for m in ordered_modes]
                    t_max = [float(np.mean(grouped[m][1])) for m in ordered_modes]
                    max_turn_y = max([0.5, *t_mae, *t_max])
                    x = np.arange(len(ordered_modes))
                    width = 0.34
                    b1 = ax_turn.bar(
                        x - width / 2,
                        t_mae,
                        width,
                        label=("平均绝对误差 MAE (km/h)" if use_cjk else "MAE (km/h)"),
                        color="#2f855a",
                    )
                    b2 = ax_turn.bar(
                        x + width / 2,
                        t_max,
                        width,
                        label=("峰值误差均值 (km/h)" if use_cjk else "Mean Peak Error (km/h)"),
                        color="#68d391",
                    )
                    for bar in list(b1) + list(b2):
                        h = bar.get_height()
                        ax_turn.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 0.025 * max(1.0, max_turn_y),
                            f"{h:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8.5,
                        )
                    ax_turn.set_xticks(x)
                    ax_turn.set_xticklabels(t_labels, fontsize=9.5)
                    ax_turn.set_ylabel("测速误差 (km/h)" if use_cjk else "Speed Error (km/h)", fontsize=10)
                    ax_turn.set_title(
                        "弯道场景测速模型消融对比"
                        if use_cjk
                        else "Turning Scenario Speed Estimator Ablation",
                        fontsize=11.5,
                        pad=10,
                    )
                    ax_turn.set_ylim(0.0, max_turn_y * 1.34)
                    ax_turn.grid(axis="y", linestyle="--", alpha=0.35)
                    ax_turn.legend(loc="upper right", fontsize=8.8, **_LEGEND_STYLE)

                if not has_speed_est and not has_turn_rows and isinstance(speed_grid, list):
                    by_cfg: dict[str, list[float]] = defaultdict(list)
                    for idx, row in enumerate(speed_grid):
                        if not isinstance(row, Mapping):
                            raise ValueError(f"speed[{idx}] 必须是字典")
                        w_val = row.get("window_size", "?")
                        a_val = row.get("ema_alpha", "?")
                        cfg_label = f"窗口{w_val}/系数{a_val}" if use_cjk else f"W{w_val}/a={a_val}"
                        err_kmh = _finite_float(
                            row["absolute_error_ms"], f"speed[{idx}].absolute_error_ms", non_negative=True
                        ) * 3.6
                        by_cfg[cfg_label].append(err_kmh)
                    cfg_keys = list(by_cfg.keys())[:12]
                    cfg_vals = [float(np.mean(by_cfg[k])) for k in cfg_keys]
                    x = np.arange(len(cfg_keys))
                    ax_single.bar(x, cfg_vals, color="#2b6cb0", width=0.55)
                    ax_single.set_xticks(x)
                    ax_single.set_xticklabels(cfg_keys, rotation=25, ha="right", fontsize=8.5)
                    ax_single.set_ylabel(
                        "测速平均绝对误差 MAE (km/h)" if use_cjk else "Speed MAE (km/h)",
                        fontsize=10,
                    )
                    ax_single.set_title(
                        "滑窗长度与 EMA 平滑系数误差消融"
                        if use_cjk
                        else "Window Size & EMA Alpha Speed Error Ablation",
                        fontsize=11.5,
                        pad=10,
                    )
                    ax_single.set_ylim(0.0, max([0.2, *cfg_vals]) * 1.28)
                    ax_single.grid(axis="y", linestyle="--", alpha=0.35)

                fig.tight_layout(pad=1.5, w_pad=2.4)
                generated.append(_save_figure(fig, speed_path))
            finally:
                plt.close(fig)

    # 2) 冲突研判指标对比图：对比 CVM 圆包络与 OBB-SSM 的风险等级、DRAC 与碰撞概率
    has_conflict_det = isinstance(conflict_det, list) and len(conflict_det) > 0
    has_conflict_grid = isinstance(conflict_grid, list) and len(conflict_grid) > 0

    if has_conflict_det or has_conflict_grid:
        conflict_path = out_root / "ablation_conflict_metrics.png"
        with _chart_rc_context() as use_cjk:
            missing_lbl = "无数据" if use_cjk else "N/A"
            if has_conflict_det:
                fig, (ax_risk, ax_ssm) = plt.subplots(1, 2, figsize=(12.2, 5.0))
            else:
                fig, ax_grid = plt.subplots(figsize=(8.0, 4.8))
            try:
                if has_conflict_det and isinstance(conflict_det, list):
                    scenarios: list[str] = []
                    cvm_lvls: list[int] = []
                    obb_lvls: list[int] = []
                    drac_vals: list[Optional[float]] = []
                    prob_pcts: list[Optional[float]] = []

                    for idx, row in enumerate(conflict_det):
                        if not isinstance(row, Mapping):
                            raise ValueError(f"conflict_detection[{idx}] 必须是字典")
                        default_scn = f"场景_{idx + 1}" if use_cjk else f"Scenario {idx + 1}"
                        raw_scenario = str(row.get("scenario", default_scn))
                        scenarios.append(
                            _display_scenario(
                                raw_scenario, use_cjk=use_cjk, fallback=f"Scenario {idx + 1}"
                            )
                        )
                        cvm_sub = row.get("cvm_circle", {})
                        obb_sub = row.get("obb_ssm", {})
                        cvm_lvls.append(_risk_level_num(cvm_sub.get("risk_level")))
                        obb_lvls.append(_risk_level_num(obb_sub.get("risk_level")))
                        raw_drac = obb_sub.get("drac_ms2")
                        drac_vals.append(
                            _finite_float(raw_drac, f"conflict[{idx}].drac_ms2", non_negative=True)
                            if raw_drac is not None
                            else None
                        )
                        raw_prob = obb_sub.get("collision_prob")
                        prob_pcts.append(
                            _finite_float(raw_prob, f"conflict[{idx}].collision_prob", non_negative=True) * 100.0
                            if raw_prob is not None
                            else None
                        )

                    x = np.arange(len(scenarios))
                    width = 0.34
                    rotate_deg = (
                        15
                        if (len(scenarios) > 4 or any(len(s) > (6 if use_cjk else 10) for s in scenarios))
                        else 0
                    )
                    xtick_ha = "right" if rotate_deg else "center"

                    b_cvm = ax_risk.bar(
                        x - width / 2,
                        cvm_lvls,
                        width,
                        label=("CVM 圆包络" if use_cjk else "CVM Circle"),
                        color="#718096",
                    )
                    b_obb = ax_risk.bar(
                        x + width / 2,
                        obb_lvls,
                        width,
                        label="OBB-SSM",
                        color="#dd6b20",
                    )
                    for bar in list(b_cvm) + list(b_obb):
                        h = bar.get_height()
                        lvl_idx = max(0, min(3, int(round(h))))
                        ax_risk.text(
                            bar.get_x() + bar.get_width() / 2,
                            h + 0.08,
                            f"L{lvl_idx}",
                            ha="center",
                            va="bottom",
                            fontsize=8.8,
                        )
                    ax_risk.set_xticks(x)
                    ax_risk.set_xticklabels(scenarios, rotation=rotate_deg, ha=xtick_ha, fontsize=9.5)
                    ax_risk.set_yticks([0, 1, 2, 3])
                    ax_risk.set_yticklabels(_risk_level_labels(use_cjk=use_cjk), fontsize=9)
                    ax_risk.set_ylabel("风险等级" if use_cjk else "Risk Level", fontsize=10)
                    ax_risk.set_title(
                        "冲突风险等级对比：CVM 圆包络 vs OBB-SSM"
                        if use_cjk
                        else "Conflict Risk Level: CVM Circle vs OBB-SSM",
                        fontsize=11.5,
                        pad=10,
                    )
                    ax_risk.set_ylim(0.0, 3.95)
                    ax_risk.grid(axis="y", linestyle="--", alpha=0.35)
                    ax_risk.legend(loc="upper center", ncol=2, fontsize=9, **_LEGEND_STYLE)

                    drac_label = "DRAC (m/s²)" if use_cjk else "DRAC (m/s^2)"
                    prob_label = "碰撞概率 (%)" if use_cjk else "Collision Prob. (%)"
                    valid_drac = [v for v in drac_vals if v is not None]
                    max_drac = max([1.0, *valid_drac])

                    drac_x = [float(x[i]) - width / 2 for i, v in enumerate(drac_vals) if v is not None]
                    drac_h = [v for v in drac_vals if v is not None]
                    if drac_x:
                        ax_ssm.bar(drac_x, drac_h, width, label=drac_label, color="#c53030")

                    ax_prob = ax_ssm.twinx()
                    prob_x = [float(x[i]) + width / 2 for i, v in enumerate(prob_pcts) if v is not None]
                    prob_h = [v for v in prob_pcts if v is not None]
                    if prob_x:
                        ax_prob.bar(prob_x, prob_h, width, label=prob_label, color="#3182ce")

                    for i, v in enumerate(drac_vals):
                        cx = float(x[i]) - width / 2
                        if v is not None:
                            ax_ssm.text(
                                cx,
                                v + 0.03 * max_drac,
                                f"{v:.2f}",
                                ha="center",
                                va="bottom",
                                fontsize=8.5,
                            )
                        else:
                            ax_ssm.text(
                                cx,
                                0.03 * max_drac,
                                missing_lbl,
                                ha="center",
                                va="bottom",
                                fontsize=8.0,
                                color="#718096",
                            )
                    for i, v in enumerate(prob_pcts):
                        cx = float(x[i]) + width / 2
                        if v is not None:
                            ax_prob.text(
                                cx,
                                v + 2.2,
                                f"{v:.1f}%",
                                ha="center",
                                va="bottom",
                                fontsize=8.5,
                            )
                        else:
                            ax_prob.text(
                                cx,
                                2.2,
                                missing_lbl,
                                ha="center",
                                va="bottom",
                                fontsize=8.0,
                                color="#718096",
                            )
                    ax_ssm.set_xticks(x)
                    ax_ssm.set_xticklabels(scenarios, rotation=rotate_deg, ha=xtick_ha, fontsize=9.5)
                    ax_ssm.set_xlim(-0.6, max(0.6, len(scenarios) - 0.4))
                    ax_ssm.set_ylabel(
                        "必要减速度 DRAC (m/s²)" if use_cjk else "Required Decel. DRAC (m/s^2)",
                        fontsize=10,
                    )
                    ax_prob.set_ylabel(
                        "碰撞概率 (%)" if use_cjk else "Collision Probability (%)",
                        fontsize=10,
                    )
                    ax_ssm.set_ylim(0.0, max_drac * 1.32)
                    ax_prob.set_ylim(0.0, 122.0)
                    ax_ssm.set_title(
                        "OBB-SSM 冲突严重度量化指标"
                        if use_cjk
                        else "OBB-SSM Conflict Severity Metrics",
                        fontsize=11.5,
                        pad=10,
                    )
                    ax_ssm.grid(axis="y", linestyle="--", alpha=0.35)
                    h1, l1 = ax_ssm.get_legend_handles_labels()
                    h2, l2 = ax_prob.get_legend_handles_labels()
                    if not h1:
                        h1, l1 = [mpatches.Patch(color="#c53030")], [drac_label]
                    if not h2:
                        h2, l2 = [mpatches.Patch(color="#3182ce")], [prob_label]
                    ax_ssm.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=9, **_LEGEND_STYLE)
                elif isinstance(conflict_grid, list):
                    labels = [
                        (
                            f"临界{row.get('ttc_critical_s', '?')}s/时域{row.get('horizon_s', '?')}s"
                            if use_cjk
                            else f"Crit {row.get('ttc_critical_s', '?')}s / Hor {row.get('horizon_s', '?')}s"
                        )
                        for row in conflict_grid
                    ]
                    ttcs: list[Optional[float]] = [
                        _finite_float(row["ttc_s"], f"conflict[{i}].ttc_s", non_negative=True)
                        if row.get("ttc_s") is not None
                        else None
                        for i, row in enumerate(conflict_grid)
                    ]
                    dists: list[Optional[float]] = [
                        _finite_float(row["min_distance_m"], f"conflict[{i}].min_distance_m", non_negative=True)
                        if row.get("min_distance_m") is not None
                        else None
                        for i, row in enumerate(conflict_grid)
                    ]
                    x = np.arange(len(labels))
                    width = 0.35
                    ttc_lbl = "碰撞时间 TTC (s)" if use_cjk else "TTC (s)"
                    dist_lbl = "最小间距 (m)" if use_cjk else "Min Distance (m)"
                    valid_grid = [v for v in (*ttcs, *dists) if v is not None]
                    max_grid_y = max([0.5, *valid_grid])

                    ttc_x = [float(x[i]) - width / 2 for i, v in enumerate(ttcs) if v is not None]
                    ttc_h = [v for v in ttcs if v is not None]
                    if ttc_x:
                        ax_grid.bar(ttc_x, ttc_h, width, label=ttc_lbl, color="#dd6b20")
                    dist_x = [float(x[i]) + width / 2 for i, v in enumerate(dists) if v is not None]
                    dist_h = [v for v in dists if v is not None]
                    if dist_x:
                        ax_grid.bar(dist_x, dist_h, width, label=dist_lbl, color="#3182ce")
                    for i, v in enumerate(ttcs):
                        if v is None:
                            ax_grid.text(
                                float(x[i]) - width / 2,
                                0.03 * max_grid_y,
                                missing_lbl,
                                ha="center",
                                va="bottom",
                                fontsize=8.0,
                                color="#718096",
                            )
                    for i, v in enumerate(dists):
                        if v is None:
                            ax_grid.text(
                                float(x[i]) + width / 2,
                                0.03 * max_grid_y,
                                missing_lbl,
                                ha="center",
                                va="bottom",
                                fontsize=8.0,
                                color="#718096",
                            )
                    ax_grid.set_xticks(x)
                    ax_grid.set_xticklabels(labels, rotation=22, ha="right", fontsize=8.5)
                    ax_grid.set_xlim(-0.6, max(0.6, len(labels) - 0.4))
                    ax_grid.set_ylabel("指标数值" if use_cjk else "Metric Value", fontsize=10)
                    ax_grid.set_title(
                        "TTC 临界阈值与预测时域消融对比"
                        if use_cjk
                        else "TTC Threshold & Prediction Horizon Ablation",
                        fontsize=11.5,
                        pad=10,
                    )
                    ax_grid.set_ylim(0.0, max_grid_y * 1.28)
                    ax_grid.grid(axis="y", linestyle="--", alpha=0.35)
                    hg, lg = ax_grid.get_legend_handles_labels()
                    if not hg:
                        hg = [mpatches.Patch(color="#dd6b20"), mpatches.Patch(color="#3182ce")]
                        lg = [ttc_lbl, dist_lbl]
                    ax_grid.legend(hg, lg, loc="upper right", fontsize=9, **_LEGEND_STYLE)

                fig.tight_layout(pad=1.5, w_pad=2.6)
                generated.append(_save_figure(fig, conflict_path))
            finally:
                plt.close(fig)

    if not generated:
        raise ValueError("ablation_data 未包含可绘制的速度或冲突对比字段")
    return generated


def _resolve_speed_limit_kmh(run_root: Path) -> float:
    """从 configuration.json 或 summary.json（含嵌套 configuration）读取限速阈值，缺省回退 30.0 km/h。"""
    for fname in ("configuration.json", "summary.json"):
        cfg_file = run_root / fname
        if cfg_file.is_file():
            cfg = _load_json_strict(cfg_file, str(cfg_file))
            candidates: list[tuple[Any, str]] = [
                (cfg.get("speed_limit_kmh"), f"{fname}.speed_limit_kmh"),
            ]
            nested_cfg = cfg.get("configuration")
            if isinstance(nested_cfg, Mapping):
                candidates.append(
                    (nested_cfg.get("speed_limit_kmh"), f"{fname}.configuration.speed_limit_kmh")
                )
            for raw_limit, label in candidates:
                if raw_limit is not None:
                    val = _finite_float(raw_limit, label, non_negative=True)
                    if val > 0.0:
                        return val
    return 30.0


def _compute_median_dt(unique_sorted_ts: Sequence[float]) -> Optional[float]:
    """根据有序去重时间戳计算典型帧间隔中位数（秒）。"""
    diffs = [
        b - a
        for a, b in zip(unique_sorted_ts, unique_sorted_ts[1:])
        if (b - a) > 1e-9
    ]
    if not diffs:
        return None
    return float(np.median(diffs))


def _are_consecutive_frames(
    prev_fidx: Optional[int],
    prev_ts: float,
    curr_fidx: Optional[int],
    curr_ts: float,
    *,
    unique_speed_ts: Sequence[float],
    median_dt: Optional[float],
) -> bool:
    """判定两条按序排列的冲突帧记录是否严格连续（无跳帧或缺失中间帧）。"""
    if prev_fidx is not None and curr_fidx is not None:
        return curr_fidx == prev_fidx + 1
    if curr_ts <= prev_ts + 1e-9:
        return False
    lo_idx = bisect.bisect_right(unique_speed_ts, prev_ts + 1e-9)
    if lo_idx < len(unique_speed_ts) and unique_speed_ts[lo_idx] < curr_ts - 1e-9:
        return False
    dt = curr_ts - prev_ts
    if median_dt is not None and median_dt > 0.0:
        return dt <= median_dt * 1.5 + 1e-9
    return dt <= 0.15 + 1e-9


def _build_segmented_series(
    points: Sequence[tuple[Optional[int], float, float]],
    *,
    unique_speed_ts: Sequence[float],
    median_dt: Optional[float],
) -> tuple[list[float], list[float]]:
    """在非连续帧间隙插入 NaN 断点，避免稀疏记录被错误连成连续折线。"""
    xs: list[float] = []
    ys: list[float] = []
    prev_fidx: Optional[int] = None
    prev_ts: Optional[float] = None
    for fidx, ts, val in points:
        if prev_ts is not None and not _are_consecutive_frames(
            prev_fidx,
            prev_ts,
            fidx,
            ts,
            unique_speed_ts=unique_speed_ts,
            median_dt=median_dt,
        ):
            xs.append(math.nan)
            ys.append(math.nan)
        xs.append(ts)
        ys.append(val)
        prev_fidx = fidx
        prev_ts = ts
    return xs, ys


def _parse_conflict_frame_records(conflicts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按帧聚合 conflicts.jsonl，保留独立 TTC 与 MTTC 最小值及峰值风险等级。"""
    by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    for idx, row in enumerate(conflicts):
        ts = _finite_float(row.get("timestamp_s", 0.0), f"conflict[{idx}].timestamp_s", non_negative=True)
        raw_fi = row.get("frame_index")
        if raw_fi is not None:
            if isinstance(raw_fi, (bool, np.bool_)) or not isinstance(raw_fi, (int, np.integer)):
                raise ValueError(f"conflict[{idx}].frame_index 必须为非负整数")
            fidx: Optional[int] = int(raw_fi)
            if fidx < 0:
                raise ValueError(f"conflict[{idx}].frame_index 不能为负数")
            key: tuple[str, Any] = ("f", fidx)
        else:
            fidx = None
            key = ("t", ts)

        ass = row.get("assessment", row)
        lvl = _risk_level_num(ass.get("risk_level"))
        raw_ttc = ass.get("ttc_seconds")
        ttc_val = (
            _finite_float(raw_ttc, f"conflict[{idx}].ttc_seconds", non_negative=True)
            if raw_ttc is not None
            else None
        )
        raw_mttc = ass.get("mttc_seconds")
        mttc_val = (
            _finite_float(raw_mttc, f"conflict[{idx}].mttc_seconds", non_negative=True)
            if raw_mttc is not None
            else None
        )

        rec = by_key.get(key)
        if rec is None:
            by_key[key] = {
                "frame_index": fidx,
                "ts": ts,
                "risk": lvl,
                "ttc": ttc_val,
                "mttc": mttc_val,
            }
        else:
            rec["ts"] = min(rec["ts"], ts)
            rec["risk"] = max(rec["risk"], lvl)
            if ttc_val is not None:
                rec["ttc"] = ttc_val if rec["ttc"] is None else min(rec["ttc"], ttc_val)
            if mttc_val is not None:
                rec["mttc"] = mttc_val if rec["mttc"] is None else min(rec["mttc"], mttc_val)

    records = list(by_key.values())
    if records and all(r["frame_index"] is not None for r in records):
        records.sort(key=lambda r: (int(r["frame_index"]), r["ts"]))
    else:
        records.sort(key=lambda r: (r["ts"], -1 if r["frame_index"] is None else int(r["frame_index"])))
    return records


def plot_run_timeline(
    run_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
) -> list[str]:
    """绘制运行目录内 speeds.csv 车辆时速曲线、限速红线与冲突/超速事件时间窗。"""
    run_root = _validate_path_arg(run_dir, "run_dir")
    if not run_root.is_dir():
        raise FileNotFoundError(f"运行目录不存在: {run_root}")
    speeds_csv = run_root / "speeds.csv"
    if not speeds_csv.is_file():
        raise FileNotFoundError(f"运行目录缺少 speeds.csv: {speeds_csv}")

    out_root = _ensure_output_dir(output_dir)
    speed_limit = _resolve_speed_limit_kmh(run_root)
    speed_limit_str = _format_speed_limit(speed_limit)

    veh_series: dict[int, list[tuple[Optional[int], float, float, bool]]] = defaultdict(list)
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
            raw_fi = row.get("frame_idx")
            if raw_fi is None or not str(raw_fi).strip():
                raw_fi = row.get("frame_index")
            fidx: Optional[int] = None
            if raw_fi is not None and str(raw_fi).strip():
                try:
                    fidx = int(str(raw_fi).strip())
                except ValueError:
                    raise ValueError(f"{speeds_csv} 第 {row_idx} 行 frame_idx 无效") from None
                if fidx < 0:
                    raise ValueError(f"{speeds_csv} 第 {row_idx} 行 frame_idx 不能为负数")
            raw_rel = row.get("is_reliable")
            is_rel = True
            if raw_rel is not None and str(raw_rel).strip():
                rel_tok = str(raw_rel).strip().lower()
                is_rel = rel_tok not in ("0", "false", "no", "n")
            all_timestamps.append(ts)
            if str(row.get("category", "")).strip().lower() == "vehicle":
                veh_series[tid].append((fidx, ts, spd, is_rel))

    unique_speed_ts = sorted(set(all_timestamps))
    median_dt = _compute_median_dt(unique_speed_ts)

    events = _load_jsonl_strict(run_root / "events.jsonl")
    conflicts = _load_jsonl_strict(run_root / "conflicts.jsonl")
    frame_records = _parse_conflict_frame_records(conflicts) if conflicts else []

    # 严格区分来自 events.jsonl 的确认事件窗，与无 events.jsonl 时仅按连续帧推断的瞬时高风险窗
    confirmed_conflict_windows: list[tuple[float, float, int, Optional[int]]] = []
    inferred_risk_windows: list[tuple[float, float, int]] = []
    speeding_windows: list[tuple[float, float, Optional[int]]] = []
    for ev in events:
        t0 = _finite_float(ev.get("start_timestamp_s", 0.0), "event.start_timestamp_s", non_negative=True)
        t1 = _finite_float(ev.get("end_timestamp_s", t0), "event.end_timestamp_s", non_negative=True)
        if t1 < t0:
            t1 = t0
        lvl = _risk_level_num(ev.get("peak_risk_level", 2))
        eid = ev.get("event_id")
        if ev.get("kind") == "conflict":
            confirmed_conflict_windows.append((t0, t1, lvl, eid if type(eid) is int else None))
        elif ev.get("kind") == "speeding":
            speeding_windows.append((t0, t1, eid if type(eid) is int else None))

    if not confirmed_conflict_windows and frame_records:
        # 仅合并严格连续的 LEVEL_2/LEVEL_3 帧，绝不将跨间隙的稀疏记录并成连续时段
        cur_start: Optional[float] = None
        cur_end: Optional[float] = None
        cur_lvl = 0
        cur_last_fidx: Optional[int] = None
        cur_last_ts: Optional[float] = None
        for rec in frame_records:
            ts = rec["ts"]
            lvl = rec["risk"]
            fidx = rec["frame_index"]
            if lvl >= 2:
                if cur_start is None:
                    cur_start = ts
                    cur_end = ts
                    cur_lvl = lvl
                    cur_last_fidx = fidx
                    cur_last_ts = ts
                elif cur_last_ts is not None and _are_consecutive_frames(
                    cur_last_fidx,
                    cur_last_ts,
                    fidx,
                    ts,
                    unique_speed_ts=unique_speed_ts,
                    median_dt=median_dt,
                ):
                    cur_end = ts
                    cur_lvl = max(cur_lvl, lvl)
                    cur_last_fidx = fidx
                    cur_last_ts = ts
                else:
                    inferred_risk_windows.append(
                        (cur_start, cur_end if cur_end is not None else cur_start, cur_lvl)
                    )
                    cur_start = ts
                    cur_end = ts
                    cur_lvl = lvl
                    cur_last_fidx = fidx
                    cur_last_ts = ts
            elif cur_start is not None:
                inferred_risk_windows.append(
                    (cur_start, cur_end if cur_end is not None else cur_start, cur_lvl)
                )
                cur_start = None
                cur_end = None
                cur_last_fidx = None
                cur_last_ts = None
        if cur_start is not None:
            inferred_risk_windows.append(
                (cur_start, cur_end if cur_end is not None else cur_start, cur_lvl)
            )

    generated: list[str] = []
    timeline_path = out_root / "run_speed_timeline.png"
    with _chart_rc_context() as use_cjk:
        fig, ax = plt.subplots(figsize=(11.0, 5.0))
        try:
            # 对单帧零宽度时间窗做轻微可视展宽，避免 axvspan 在图上不可见
            if all_timestamps and max(all_timestamps) > min(all_timestamps):
                span_pad = min(0.25, max(0.04, 0.005 * (max(all_timestamps) - min(all_timestamps))))
            else:
                span_pad = 0.04

            confirmed_labeled = False
            confirmed_label_text = "确认冲突事件窗" if use_cjk else "Confirmed Conflict Event"
            for t0, t1, lvl, _eid in confirmed_conflict_windows:
                lo = max(0.0, t0 - span_pad) if math.isclose(t0, t1) else t0
                hi = (t1 + span_pad) if math.isclose(t0, t1) else t1
                fc = "#e53e3e" if lvl >= 3 else "#dd6b20"
                lbl = confirmed_label_text if not confirmed_labeled else None
                ax.axvspan(lo, hi, facecolor=fc, alpha=0.22, label=lbl)
                confirmed_labeled = True

            inferred_labeled = False
            inferred_label_text = "瞬时高风险窗（推断）" if use_cjk else "Inferred Transient Risk"
            for t0, t1, lvl in inferred_risk_windows:
                lo = max(0.0, t0 - span_pad) if math.isclose(t0, t1) else t0
                hi = (t1 + span_pad) if math.isclose(t0, t1) else t1
                fc = "#fc8181" if lvl >= 3 else "#f6ad55"
                ec = "#c53030" if lvl >= 3 else "#c05621"
                lbl = inferred_label_text if not inferred_labeled else None
                ax.axvspan(
                    lo,
                    hi,
                    facecolor=fc,
                    edgecolor=ec,
                    hatch="//",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.24,
                    label=lbl,
                )
                inferred_labeled = True

            speeding_labeled = False
            speeding_label_text = "超速事件窗" if use_cjk else "Speeding Event Window"
            for t0, t1, _eid in speeding_windows:
                lo = max(0.0, t0 - span_pad) if math.isclose(t0, t1) else t0
                hi = (t1 + span_pad) if math.isclose(t0, t1) else t1
                lbl = speeding_label_text if not speeding_labeled else None
                ax.axvspan(lo, hi, facecolor="#ecc94b", alpha=0.20, label=lbl)
                speeding_labeled = True

            # 按观测长度与峰值车速排序，避免多车流下 legend 过长触发紧排告警
            ordered_ids = sorted(
                veh_series.keys(),
                key=lambda tid: (len(veh_series[tid]), max(s for _, _, s, _ in veh_series[tid])),
                reverse=True,
            )
            max_spd_seen = speed_limit
            unreliable_labeled = False
            unreliable_label_text = "未收敛测速样本" if use_cjk else "Unreliable Sample"
            for idx, tid in enumerate(ordered_ids):
                pts = sorted(
                    veh_series[tid],
                    key=lambda item: (item[1], -1 if item[0] is None else item[0]),
                )
                sp_all = [p[2] for p in pts]
                if sp_all:
                    max_spd_seen = max(max_spd_seen, max(sp_all))
                rel_pts = [(p[0], p[1], p[2]) for p in pts if p[3]]
                unrel_pts = [(p[1], p[2]) for p in pts if not p[3]]

                if idx < 8:
                    lbl = f"车辆 #{tid}" if use_cjk else f"Vehicle #{tid}"
                else:
                    lbl = None

                if rel_pts:
                    rel_ts, rel_sp = _build_segmented_series(
                        rel_pts, unique_speed_ts=unique_speed_ts, median_dt=median_dt
                    )
                    lines = ax.plot(
                        rel_ts, rel_sp, marker="o", markersize=2.6, linewidth=1.6, label=lbl
                    )
                    line_color = lines[0].get_color()
                else:
                    # 无可靠样本时仍用空线段保持车辆图例与颜色一致，未收敛点单独空心标出
                    lines = ax.plot([], [], marker="o", markersize=2.6, linewidth=1.6, label=lbl)
                    line_color = lines[0].get_color()

                if unrel_pts:
                    ax.plot(
                        [p[0] for p in unrel_pts],
                        [p[1] for p in unrel_pts],
                        linestyle="None",
                        marker="o",
                        markerfacecolor="none",
                        markeredgecolor=line_color,
                        markeredgewidth=1.3,
                        markersize=4.4,
                        alpha=0.85,
                        label=(unreliable_label_text if not unreliable_labeled else None),
                    )
                    unreliable_labeled = True

            if not ordered_ids:
                ax.text(
                    0.5,
                    0.5,
                    "未记录到车辆有效测速样本" if use_cjk else "No valid vehicle speed samples recorded",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="#4a5568",
                )

            ax.axhline(
                speed_limit,
                color="#d62728",
                linestyle="--",
                linewidth=1.8,
                label=(
                    f"限速阈值 ({speed_limit_str} km/h)"
                    if use_cjk
                    else f"Speed Limit ({speed_limit_str} km/h)"
                ),
            )
            ax.set_xlabel("时间 (s)" if use_cjk else "Time (s)", fontsize=10)
            ax.set_ylabel("时速 (km/h)" if use_cjk else "Speed (km/h)", fontsize=10)
            ax.set_title(
                "车辆瞬时时速与冲突/超速事件时间窗"
                if use_cjk
                else "Vehicle Speed Timeline & Conflict/Speeding Windows",
                fontsize=12,
                pad=10,
            )
            ax.set_ylim(0.0, max(speed_limit * 1.25, max_spd_seen * 1.16, 5.0))
            if all_timestamps:
                t_min, t_max = min(all_timestamps), max(all_timestamps)
                if t_max > t_min:
                    ax.set_xlim(max(0.0, t_min - 0.03 * (t_max - t_min)), t_max + 0.03 * (t_max - t_min))
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.015, 1.0),
                borderaxespad=0.0,
                fontsize=8.8,
                **_LEGEND_STYLE,
            )
            fig.tight_layout(pad=1.4)
            generated.append(_save_figure(fig, timeline_path))
        finally:
            plt.close(fig)

    # 若存在逐帧冲突记录，补充一张风险等级与独立 TTC / MTTC 演变曲线图
    if frame_records:
        risk_path = out_root / "run_risk_timeline.png"
        risk_pts = [(r["frame_index"], r["ts"], float(r["risk"])) for r in frame_records]
        ttc_pts = [(r["frame_index"], r["ts"], float(r["ttc"])) for r in frame_records if r["ttc"] is not None]
        mttc_pts = [(r["frame_index"], r["ts"], float(r["mttc"])) for r in frame_records if r["mttc"] is not None]

        risk_x, risk_y = _build_segmented_series(
            risk_pts, unique_speed_ts=unique_speed_ts, median_dt=median_dt
        )
        t_candidates = list(all_timestamps) + [r["ts"] for r in frame_records]

        with _chart_rc_context() as use_cjk:
            fig, ax_r = plt.subplots(figsize=(10.6, 4.4))
            try:
                ax_r.step(
                    risk_x,
                    risk_y,
                    where="post",
                    color="#c53030",
                    linewidth=1.9,
                    marker="o",
                    markersize=4.8,
                    label=("单帧峰值风险等级" if use_cjk else "Per-Frame Peak Risk Level"),
                )
                ax_r.set_yticks([0, 1, 2, 3])
                ax_r.set_yticklabels(_risk_level_labels(use_cjk=use_cjk), fontsize=9.2)
                ax_r.set_ylim(-0.2, 3.6)
                ax_r.set_xlabel("时间 (s)" if use_cjk else "Time (s)", fontsize=10)
                ax_r.set_ylabel("风险等级" if use_cjk else "Risk Level", fontsize=10)
                ax_r.set_title(
                    "逐帧风险等级与最小 TTC/MTTC 演变曲线"
                    if use_cjk
                    else "Per-Frame Risk Level & Min TTC/MTTC Timeline",
                    fontsize=12,
                    pad=10,
                )
                if t_candidates:
                    t_lo, t_hi = min(t_candidates), max(t_candidates)
                    if t_hi > t_lo:
                        span = t_hi - t_lo
                        ax_r.set_xlim(max(0.0, t_lo - 0.03 * span), t_hi + 0.03 * span)
                    else:
                        ax_r.set_xlim(max(0.0, t_lo - 0.5), t_lo + 0.5)
                ax_r.grid(True, linestyle="--", alpha=0.35)

                if ttc_pts or mttc_pts:
                    ax_t = ax_r.twinx()
                    all_t_vals: list[float] = []
                    if ttc_pts:
                        ttc_x, ttc_y = _build_segmented_series(
                            ttc_pts, unique_speed_ts=unique_speed_ts, median_dt=median_dt
                        )
                        ax_t.plot(
                            ttc_x,
                            ttc_y,
                            color="#2b6cb0",
                            linestyle="-.",
                            marker="o",
                            markersize=4.5,
                            linewidth=1.6,
                            label=("最小 TTC (s)" if use_cjk else "Min TTC (s)"),
                        )
                        all_t_vals.extend(p[2] for p in ttc_pts)
                    if mttc_pts:
                        mttc_x, mttc_y = _build_segmented_series(
                            mttc_pts, unique_speed_ts=unique_speed_ts, median_dt=median_dt
                        )
                        ax_t.plot(
                            mttc_x,
                            mttc_y,
                            color="#805ad5",
                            linestyle=":",
                            marker="s",
                            markersize=4.2,
                            linewidth=1.6,
                            label=("最小 MTTC (s)" if use_cjk else "Min MTTC (s)"),
                        )
                        all_t_vals.extend(p[2] for p in mttc_pts)
                    ax_t.set_ylabel("最小 TTC / MTTC (s)" if use_cjk else "Min TTC / MTTC (s)", fontsize=10)
                    y_top = max(4.0, max(all_t_vals) * 1.25)
                    ax_t.set_ylim(-0.04 * y_top, y_top)
                    h1, l1 = ax_r.get_legend_handles_labels()
                    h2, l2 = ax_t.get_legend_handles_labels()
                    ax_r.legend(
                        h1 + h2,
                        l1 + l2,
                        loc="upper right",
                        ncol=min(3, len(h1) + len(h2)),
                        fontsize=8.6,
                        **_LEGEND_STYLE,
                    )
                else:
                    ax_r.legend(loc="upper right", fontsize=8.8, **_LEGEND_STYLE)

                fig.tight_layout(pad=1.4)
                generated.append(_save_figure(fig, risk_path))
            finally:
                plt.close(fig)

    return generated


def _normalize_mot_reports(reports: Any, *, use_cjk: bool = True) -> list[tuple[str, Mapping[str, Any]]]:
    """统一解析 MOT 对比输入（字典映射、二元组列表或带 label 的报告列表）。"""
    items: list[tuple[str, Any]] = []
    if isinstance(reports, Mapping):
        items = [(str(k), v) for k, v in reports.items()]
    elif isinstance(reports, Sequence) and not isinstance(reports, (str, bytes)):
        for idx, entry in enumerate(reports):
            if isinstance(entry, tuple) and len(entry) == 2:
                items.append((str(entry[0]), entry[1]))
            elif isinstance(entry, Mapping):
                default_lbl = f"配置 {idx + 1}" if use_cjk else f"Config {idx + 1}"
                label = entry.get("label") or entry.get("name") or entry.get("model") or default_lbl
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
    with _chart_rc_context() as use_cjk:
        parsed = _normalize_mot_reports(reports, use_cjk=use_cjk)

        labels: list[str] = []
        mota_vals: list[float] = []
        idf1_vals: list[float] = []
        hota_vals: list[float] = []
        idsw_vals: list[float] = []

        for idx, (label, rep) in enumerate(parsed):
            for req_k in ("mota", "idf1", "hota"):
                if req_k not in rep:
                    raise ValueError(f"MOT 报告 {label!r} 缺少字段 {req_k}")
            raw_idsw = rep.get("id_switches", rep.get("idsw"))
            if raw_idsw is None:
                raise ValueError(f"MOT 报告 {label!r} 缺少字段 id_switches")

            labels.append(_safe_label(label, use_cjk=use_cjk, fallback=f"Config {idx + 1}"))
            mota_vals.append(_finite_float(rep["mota"], f"{label}.mota"))
            idf1_vals.append(_finite_float(rep["idf1"], f"{label}.idf1", non_negative=True))
            hota_vals.append(_finite_float(rep["hota"], f"{label}.hota", non_negative=True))
            idsw_vals.append(_finite_float(raw_idsw, f"{label}.id_switches", non_negative=True))

        fig, (ax_scores, ax_idsw) = plt.subplots(
            1, 2, figsize=(11.6, 4.9), gridspec_kw={"width_ratios": [2.2, 1.0]}
        )
        try:
            x = np.arange(len(labels))
            width = 0.25
            b_mota = ax_scores.bar(x - width, mota_vals, width, label="MOTA", color="#2b6cb0")
            b_idf1 = ax_scores.bar(x, idf1_vals, width, label="IDF1", color="#2f855a")
            b_hota = ax_scores.bar(x + width, hota_vals, width, label="HOTA", color="#dd6b20")

            all_scores = mota_vals + idf1_vals + hota_vals
            y_min = min(0.0, min(all_scores) - 0.14)
            y_max = max(1.18, max(all_scores) + 0.22)
            for bar in list(b_mota) + list(b_idf1) + list(b_hota):
                h = bar.get_height()
                va = "bottom" if h >= 0 else "top"
                offset = 0.018 if h >= 0 else -0.024
                ax_scores.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + offset,
                    f"{h:.3f}",
                    ha="center",
                    va=va,
                    fontsize=8.2,
                )
            ax_scores.axhline(0.0, color="#4a5568", linewidth=0.8)
            rot = 14 if len(labels) > 3 else 0
            ha_lbl = "right" if rot else "center"
            ax_scores.set_xticks(x)
            ax_scores.set_xticklabels(labels, rotation=rot, ha=ha_lbl, fontsize=9.2)
            ax_scores.set_ylabel("指标得分" if use_cjk else "Score", fontsize=10)
            ax_scores.set_title(
                "多目标跟踪精度指标对比 (MOTA / IDF1 / HOTA)"
                if use_cjk
                else "MOT Accuracy Comparison (MOTA / IDF1 / HOTA)",
                fontsize=11.5,
                pad=10,
            )
            ax_scores.set_ylim(y_min, y_max)
            ax_scores.grid(axis="y", linestyle="--", alpha=0.35)
            ax_scores.legend(loc="upper center", ncol=3, fontsize=9, **_LEGEND_STYLE)

            b_id = ax_idsw.bar(x, idsw_vals, 0.48, color="#c53030", label="IDSW")
            max_idsw = max(idsw_vals) if idsw_vals else 0.0
            for bar in b_id:
                h = bar.get_height()
                ax_idsw.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max(0.06, max_idsw * 0.03),
                    f"{int(round(h))}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax_idsw.set_xticks(x)
            ax_idsw.set_xticklabels(labels, rotation=rot, ha=ha_lbl, fontsize=9.2)
            ax_idsw.set_ylabel("身份切换次数 (次)" if use_cjk else "ID Switches (count)", fontsize=10)
            ax_idsw.set_title(
                "身份切换次数 IDSW (越低越好)"
                if use_cjk
                else "ID Switches IDSW (Lower Is Better)",
                fontsize=11.5,
                pad=10,
            )
            ax_idsw.set_ylim(0.0, max(1.5, max_idsw * 1.28))
            ax_idsw.grid(axis="y", linestyle="--", alpha=0.35)

            fig.tight_layout(pad=1.5, w_pad=2.2)
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
        means.append(mean_v)
        p95s.append(p95_v)

    if present_sub == 0:
        raise ValueError("bench_data.stages 未包含任何已知流水线子阶段")

    total_sum = sum(means)
    frames_val = payload.get("frames_profiled")
    frame_fps_val = payload.get("frame_processing_fps", payload.get("end_to_end_fps"))
    wall_fps_val = payload.get("wall_fps")
    wall_ms_val = payload.get("wall_elapsed_ms")
    if (
        wall_fps_val is None
        and type(frames_val) is int
        and frames_val > 0
        and wall_ms_val is not None
        and not isinstance(wall_ms_val, (bool, np.bool_))
        and isinstance(wall_ms_val, (int, float, np.integer, np.floating))
        and float(wall_ms_val) > 0.0
    ):
        wall_fps_val = frames_val / (float(wall_ms_val) / 1000.0)
    steady_obj = payload.get("steady_state")
    steady_fps_val: Optional[float] = None
    if (
        isinstance(steady_obj, Mapping)
        and isinstance(steady_obj.get("warmup_frames_excluded"), (int, np.integer))
        and not isinstance(steady_obj.get("warmup_frames_excluded"), (bool, np.bool_))
        and int(steady_obj["warmup_frames_excluded"]) > 0
        and isinstance(steady_obj.get("frames_evaluated"), (int, np.integer))
        and not isinstance(steady_obj.get("frames_evaluated"), (bool, np.bool_))
        and int(steady_obj["frames_evaluated"]) > 0
        and steady_obj.get("steady_state_fps") is not None
        and not isinstance(steady_obj.get("steady_state_fps"), (bool, np.bool_))
    ):
        steady_fps_val = float(steady_obj["steady_state_fps"])
    model_load_val = payload.get("model_load_ms")

    with _chart_rc_context() as use_cjk:
        stage_labels = [_display_stage(stage_key, use_cjk=use_cjk) for stage_key in SUBSTAGE_NAMES]
        fig, (ax_bar, ax_pie) = plt.subplots(
            1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
        )
        try:
            x = np.arange(len(stage_labels))
            width = 0.35
            b_mean = ax_bar.bar(
                x - width / 2,
                means,
                width,
                label=("均值耗时 (ms)" if use_cjk else "Mean Latency (ms)"),
                color="#2b6cb0",
            )
            b_p95 = ax_bar.bar(
                x + width / 2,
                p95s,
                width,
                label=("P95 耗时 (ms)" if use_cjk else "P95 Latency (ms)"),
                color="#63b3ed",
            )
            max_lat = max([0.1, *means, *p95s])
            for bar in list(b_mean) + list(b_p95):
                h = bar.get_height()
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max_lat * 0.025,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(
                stage_labels,
                rotation=(0 if use_cjk else 15),
                ha=("center" if use_cjk else "right"),
                fontsize=(9.5 if use_cjk else 8.8),
            )
            ax_bar.set_ylabel("单帧耗时 (ms)" if use_cjk else "Per-Frame Latency (ms)", fontsize=10)
            ax_bar.set_title(
                "流水线各阶段单帧耗时分布 (均值 vs P95)"
                if use_cjk
                else "Pipeline Stage Latency Distribution (Mean vs P95)",
                fontsize=11.5,
                pad=10,
            )
            ax_bar.set_ylim(0.0, max_lat * 1.30)
            ax_bar.grid(axis="y", linestyle="--", alpha=0.35)
            ax_bar.legend(loc="upper right", fontsize=9, **_LEGEND_STYLE)

            colors = ["#2b6cb0", "#c53030", "#2f855a", "#dd6b20", "#805ad5"]
            pie_title = "各阶段平均耗时占比 (%)" if use_cjk else "Stage Mean Latency Share (%)"
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
                ax_pie.set_aspect("equal")
                ax_pie.legend(
                    wedges,
                    pie_labels,
                    loc="center left",
                    bbox_to_anchor=(0.94, 0.5),
                    fontsize=9,
                    frameon=False,
                )
                ax_pie.set_title(pie_title, fontsize=11.5, pad=10)
            else:
                ax_pie.axis("off")
                ax_pie.text(
                    0.5,
                    0.5,
                    "未记录到正值阶段耗时" if use_cjk else "No positive stage latency recorded",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="#4a5568",
                )
                ax_pie.set_title(pie_title, fontsize=11.5, pad=10)

            subtitle_parts = []
            if frames_val is not None and type(frames_val) is int:
                subtitle_parts.append(
                    f"评测帧数：{frames_val} 帧" if use_cjk else f"Frames: {frames_val}"
                )
            if frame_fps_val is not None and not isinstance(frame_fps_val, (bool, np.bool_)):
                subtitle_parts.append(
                    f"帧内吞吐：{float(frame_fps_val):.2f} FPS"
                    if use_cjk
                    else f"Frame FPS: {float(frame_fps_val):.2f}"
                )
            if steady_fps_val is not None:
                subtitle_parts.append(
                    f"稳态吞吐：{steady_fps_val:.2f} FPS"
                    if use_cjk
                    else f"Steady FPS: {steady_fps_val:.2f}"
                )
            if wall_fps_val is not None and not isinstance(wall_fps_val, (bool, np.bool_)):
                subtitle_parts.append(
                    f"墙钟吞吐：{float(wall_fps_val):.2f} FPS"
                    if use_cjk
                    else f"Wall FPS: {float(wall_fps_val):.2f}"
                )
            if model_load_val is not None and not isinstance(model_load_val, (bool, np.bool_)):
                subtitle_parts.append(
                    f"模型加载：{float(model_load_val):.1f} ms"
                    if use_cjk
                    else f"Model Load: {float(model_load_val):.1f} ms"
                )
            if subtitle_parts:
                fig.suptitle("  |  ".join(subtitle_parts), fontsize=10.0, y=0.985)
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=1.5, w_pad=2.0)
            else:
                fig.tight_layout(pad=1.5, w_pad=2.0)

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
