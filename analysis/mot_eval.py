# -*- coding: utf-8 -*-
"""
轻量级多目标跟踪评测模块 (MOT Evaluation: MOTA / MOTP / IDF1 / HOTA / IDSW)
=============================================================================
支持读取仓库 tracks.jsonl 与标准 MOTChallenge TXT/CSV 轨迹文件，
使用确定性纯 Python Kuhn-Munkres（可选对齐 lap.lapjv）完成帧级与全局二分匹配，
计算 CLEAR MOT、IDF1 与 Luiten et al. 多阈值 HOTA 指标。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from detection.core import is_blank_path

try:
    import lap as _lap
except Exception:  # pragma: no cover - 仅在未安装可选 Cython lap 依赖时触发回退
    _lap = None


# HOTA 标准 19 档定位 IoU 阈值: 0.05, 0.10, ..., 0.95
HOTA_ALPHAS: tuple[float, ...] = tuple(round(0.05 * idx, 2) for idx in range(1, 20))

# MOT17/MOT20 常见数字类别码映射，仅在 TXT 第 8 列给出数字编号时用于可选类别筛选
_MOT_CLASS_CODE_MAP: dict[int, tuple[str, str]] = {
    1: ("pedestrian", "person"),
    2: ("pedestrian", "person"),
    3: ("vehicle", "car"),
    4: ("vehicle", "bicycle"),
    5: ("vehicle", "motorcycle"),
}


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"JSON 不允许非有限常量: {value}")


def _reject_non_finite_tree(value: Any, label: str) -> None:
    # json.loads 遇到 1e309 会溢出为 float('inf') 且不走 parse_constant，
    # 读入后递归全树检查可彻底阻断 Inf/NaN 混入指标计算。
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} 包含非有限数值")
    elif isinstance(value, dict):
        for k, v in value.items():
            _reject_non_finite_tree(v, f"{label}.{k}")
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _reject_non_finite_tree(item, f"{label}[{idx}]")


def _validate_finite_float(value: Any, label: str) -> float:
    """校验有限浮点数，显式拒绝 bool 与 np.bool_。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} 不能为布尔值")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} 必须为数值")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{label} 必须为有限数值")
    return val


def _validate_iou_threshold(iou_threshold: Any) -> float:
    val = _validate_finite_float(iou_threshold, "iou_threshold")
    if val <= 0.0 or val > 1.0:
        raise ValueError("iou_threshold 必须在 (0.0, 1.0] 区间内")
    return val


def _validate_strict_int(token: str, label: str) -> int:
    """严格解析整型文本，拒绝浮点字符串、布尔文本或空串。"""
    s = token.strip()
    if not s:
        raise ValueError(f"{label} 不能为空")
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"{label} 必须为整数，当前为 {token!r}") from None


def _validate_bbox_xyxy(bbox: Any, label: str) -> tuple[float, float, float, float]:
    if isinstance(bbox, np.ndarray):
        if bbox.ndim != 1 or bbox.shape[0] != 4:
            raise ValueError(f"{label} 必须是长度为 4 的框坐标")
        seq = bbox.tolist()
    elif isinstance(bbox, (list, tuple)):
        if len(bbox) != 4:
            raise ValueError(f"{label} 必须是长度为 4 的框坐标")
        seq = bbox
    else:
        raise ValueError(f"{label} 必须是长度为 4 的框坐标")
    x1, y1, x2, y2 = (_validate_finite_float(v, f"{label}[{i}]") for i, v in enumerate(seq))
    if x2 < x1 or y2 < y1:
        raise ValueError(f"{label} 坐标倒挂 (x2 < x1 或 y2 < y1)")
    return (x1, y1, x2, y2)


def _normalize_category_filter(category_filter: Optional[str]) -> Optional[str]:
    if category_filter is None:
        return None
    if not isinstance(category_filter, str) or not category_filter.strip():
        raise ValueError("category_filter 必须是非空字符串或 None")
    return category_filter.strip().lower()


def _matches_category_filter(
    cat_filter: Optional[str],
    category: Optional[str],
    class_name: Optional[str],
) -> bool:
    if cat_filter is None:
        return True
    candidates = {
        str(val).strip().lower()
        for val in (category, class_name)
        if isinstance(val, str) and val.strip()
    }
    return cat_filter in candidates


def _resolve_track_path(path: Any, label: str = "轨迹文件路径") -> Path:
    # 必须在 Path() 解析前拦截非路径对象与空白路径（含 Path("   ")），
    # 防止 Windows 下退化为当前目录 '.'。
    if not isinstance(path, (str, os.PathLike)) or is_blank_path(path):
        raise ValueError(f"{label}不能为空")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"未找到轨迹文件: {resolved}")
    return resolved


def _detect_track_format(path: Path, text: str) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return "jsonl"
    if suffix in {".txt", ".csv", ".mot"}:
        return "mot_txt"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return "jsonl" if stripped.startswith("{") else "mot_txt"
    return "mot_txt"


def _load_tracks_jsonl(
    path: Path,
    text: str,
    cat_filter: Optional[str],
    one_based: Optional[bool],
    frame_offset: int,
) -> dict[int, list[dict]]:
    # tracks.jsonl 默认是 0-based frame_index；仅显式指定 one_based=True 时才减 1。
    shift = (-1 if one_based is True else 0) + frame_offset
    frames: dict[int, list[dict]] = {}
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_non_finite_constant)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是有效 JSON: {exc}") from None
        _reject_non_finite_tree(row, f"{path} 第 {line_no} 行")
        if not isinstance(row, dict):
            raise ValueError(f"{path} 第 {line_no} 行必须是 JSON 对象")

        raw_f = row.get("frame_index")
        if type(raw_f) is not int or raw_f < 0:
            raise ValueError(f"{path} 第 {line_no} 行 frame_index 必须是非负整数")
        if one_based is True and raw_f < 1:
            raise ValueError(f"{path} 第 {line_no} 行在 one_based=True 下 frame_index 必须 >= 1")
        frame_idx = raw_f + shift
        if frame_idx < 0:
            raise ValueError(f"{path} 第 {line_no} 行对齐后帧号不能为负 ({frame_idx})")
        if frame_idx in frames:
            raise ValueError(f"{path} 第 {line_no} 行存在重复 frame_index: {raw_f}")

        raw_tracks = row.get("tracks")
        if not isinstance(raw_tracks, list):
            raise ValueError(f"{path} 第 {line_no} 行 tracks 必须是列表")

        seen_ids: set[int] = set()
        parsed_tracks: list[dict] = []
        for t_idx, item in enumerate(raw_tracks):
            if not isinstance(item, dict):
                raise ValueError(f"{path} 第 {line_no} 行 tracks[{t_idx}] 必须是映射对象")
            tid = item.get("track_id")
            if tid is None:
                # 未分配跟踪 ID 的纯检测框不参与轨迹级 MOT 评测
                continue
            if type(tid) is not int or tid <= 0:
                raise ValueError(f"{path} 第 {line_no} 行 track_id 必须为正整数")
            if tid in seen_ids:
                raise ValueError(f"{path} 第 {line_no} 行存在重复 track_id: {tid}")
            seen_ids.add(tid)

            bbox = _validate_bbox_xyxy(
                item.get("bbox_xyxy"),
                f"{path} 第 {line_no} 行 tracks[{t_idx}].bbox_xyxy",
            )
            cname = item.get("class_name")
            cat = item.get("category")
            if cname is not None and (not isinstance(cname, str) or not cname.strip()):
                raise ValueError(f"{path} 第 {line_no} 行 class_name 必须是非空字符串")
            if cat is not None and (not isinstance(cat, str) or not cat.strip()):
                raise ValueError(f"{path} 第 {line_no} 行 category 必须是非空字符串")
            if cat is None and cname is not None:
                cat = "pedestrian" if cname.strip().lower() == "person" else "vehicle"

            conf = item.get("confidence")
            conf_val = (
                _validate_finite_float(conf, f"{path} 第 {line_no} 行 confidence")
                if conf is not None
                else None
            )
            if not _matches_category_filter(cat_filter, cat, cname):
                continue
            parsed_tracks.append({
                "track_id": tid,
                "bbox_xyxy": bbox,
                "category": cat,
                "class_name": cname,
                "confidence": conf_val,
            })

        frames[frame_idx] = parsed_tracks
    return frames


def _load_tracks_mot_txt(
    path: Path,
    text: str,
    cat_filter: Optional[str],
    one_based: Optional[bool],
    frame_offset: int,
) -> dict[int, list[dict]]:
    raw_rows: list[tuple[int, int, dict]] = []
    seen_per_frame: dict[int, set[int]] = defaultdict(set)

    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [tok.strip() for tok in stripped.split(",")]
        if len(parts) < 6:
            raise ValueError(
                f"{path} 第 {line_no} 行列数不足 6（至少需 frame,id,bb_left,bb_top,bb_width,bb_height）"
            )
        raw_frame = _validate_strict_int(parts[0], f"{path} 第 {line_no} 行 frame")
        if raw_frame < 0:
            raise ValueError(f"{path} 第 {line_no} 行 frame 不能为负数: {raw_frame}")
        tid = _validate_strict_int(parts[1], f"{path} 第 {line_no} 行 track_id")
        if tid <= 0:
            raise ValueError(f"{path} 第 {line_no} 行 track_id 必须为正整数: {tid}")
        if tid in seen_per_frame[raw_frame]:
            raise ValueError(f"{path} 第 {line_no} 行帧 {raw_frame} 存在重复 track_id: {tid}")
        seen_per_frame[raw_frame].add(tid)

        left = _parse_finite_float_token(parts[2], f"{path} 第 {line_no} 行 bb_left")
        top = _parse_finite_float_token(parts[3], f"{path} 第 {line_no} 行 bb_top")
        width = _parse_finite_float_token(parts[4], f"{path} 第 {line_no} 行 bb_width")
        height = _parse_finite_float_token(parts[5], f"{path} 第 {line_no} 行 bb_height")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path} 第 {line_no} 行 bb_width 与 bb_height 必须为正数")

        conf_val = None
        if len(parts) >= 7 and parts[6] != "":
            conf_val = _parse_finite_float_token(parts[6], f"{path} 第 {line_no} 行 confidence")

        cat: Optional[str] = None
        cname: Optional[str] = None
        if len(parts) >= 8 and parts[7] != "" and parts[7] != "-1":
            raw_cls = parts[7]
            try:
                cls_code = int(raw_cls)
            except ValueError:
                cname = raw_cls
                cat = "pedestrian" if raw_cls.lower() in {"person", "pedestrian"} else "vehicle"
            else:
                if cls_code in _MOT_CLASS_CODE_MAP:
                    cat, cname = _MOT_CLASS_CODE_MAP[cls_code]
                else:
                    cname = str(cls_code)

        # 对第 9 列及之后的附加数值列（如 visibility/3D 坐标）同样执行有限性校验
        for extra_idx in range(8, len(parts)):
            if parts[extra_idx] != "":
                _parse_finite_float_token(parts[extra_idx], f"{path} 第 {line_no} 行第 {extra_idx + 1} 列")

        raw_rows.append((
            raw_frame,
            line_no,
            {
                "track_id": tid,
                "bbox_xyxy": (left, top, left + width, top + height),
                "category": cat,
                "class_name": cname,
                "confidence": conf_val,
            },
        ))

    if not raw_rows:
        return {}

    min_frame = min(r[0] for r in raw_rows)
    if one_based is None:
        # 标准 MOTChallenge 从第 1 帧起步；当最小帧号 >= 1 时自动平移对齐至 0-based
        auto_one_based = min_frame >= 1
    else:
        if one_based and min_frame < 1:
            raise ValueError(f"{path} 包含小于 1 的帧号 ({min_frame})，无法按 1-based 归一化")
        auto_one_based = one_based

    shift = (-1 if auto_one_based else 0) + frame_offset
    frames: dict[int, list[dict]] = {}
    for raw_frame, line_no, track in raw_rows:
        frame_idx = raw_frame + shift
        if frame_idx < 0:
            raise ValueError(f"{path} 第 {line_no} 行对齐后帧号不能为负 ({frame_idx})")
        frames.setdefault(frame_idx, [])
        if _matches_category_filter(cat_filter, track["category"], track["class_name"]):
            frames[frame_idx].append(track)
    return frames


def _parse_finite_float_token(token: str, label: str) -> float:
    try:
        val = float(token)
    except ValueError:
        raise ValueError(f"{label} 必须为有限数字，当前为 {token!r}") from None
    if not math.isfinite(val):
        raise ValueError(f"{label} 必须为有限数字，当前为 {token!r}")
    return val


def load_mot_tracks(
    path: str | Path,
    category_filter: Optional[str] = None,
    *,
    one_based: Optional[bool] = None,
    frame_offset: int = 0,
) -> dict[int, list[dict]]:
    """读取 tracks.jsonl 或标准 MOTChallenge TXT/CSV 轨迹文件，返回 0-based 帧映射。"""
    resolved = _resolve_track_path(path, label="轨迹文件路径")
    cat_filter = _normalize_category_filter(category_filter)
    if one_based is not None and type(one_based) is not bool:
        raise ValueError("one_based 必须是 bool 或 None")
    if type(frame_offset) is not int:
        raise ValueError("frame_offset 必须是整数")

    text = resolved.read_text(encoding="utf-8")
    fmt = _detect_track_format(resolved, text)
    if fmt == "jsonl":
        return _load_tracks_jsonl(resolved, text, cat_filter, one_based, frame_offset)
    return _load_tracks_mot_txt(resolved, text, cat_filter, one_based, frame_offset)


def _normalize_frames_input(frames: Any, role: str) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    """统一将帧映射或 FrameRecord 序列规范化为 {frame_index: [(track_id, bbox_xyxy), ...]}。"""
    raw_map: dict[int, Iterable[Any]] = {}
    if isinstance(frames, dict):
        for f_key, items in frames.items():
            if type(f_key) is not int or f_key < 0:
                raise ValueError(f"{role} 帧索引必须为非负整数，当前为 {f_key!r}")
            raw_map[f_key] = items
    elif isinstance(frames, (list, tuple)):
        for idx, item in enumerate(frames):
            if isinstance(item, dict) and "tracks" in item:
                f_key = item.get("frame_index", idx)
                if type(f_key) is not int or f_key < 0:
                    raise ValueError(f"{role}[{idx}].frame_index 必须为非负整数")
                if f_key in raw_map:
                    raise ValueError(f"{role} 存在重复帧号 {f_key}")
                raw_map[f_key] = item["tracks"]
            elif hasattr(item, "tracks") and hasattr(item, "frame_index"):
                f_key = item.frame_index
                if type(f_key) is not int or f_key < 0:
                    raise ValueError(f"{role}[{idx}].frame_index 必须为非负整数")
                if f_key in raw_map:
                    raise ValueError(f"{role} 存在重复帧号 {f_key}")
                raw_map[f_key] = item.tracks
            elif isinstance(item, (list, tuple)):
                raw_map[idx] = item
            else:
                raise ValueError(f"{role}[{idx}] 无法解析为帧轨迹集合")
    else:
        raise ValueError(f"{role} 必须是帧号映射字典或帧列表")

    normalized: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for f_idx, items in raw_map.items():
        if not isinstance(items, (list, tuple)):
            raise ValueError(f"{role} 第 {f_idx} 帧轨迹列表必须是 list 或 tuple")
        seen_ids: set[int] = set()
        entries: list[tuple[int, tuple[float, float, float, float]]] = []
        for t_i, obj in enumerate(items):
            if isinstance(obj, dict):
                tid = obj.get("track_id", obj.get("id"))
                raw_bbox = obj.get("bbox_xyxy")
            else:
                tid = getattr(obj, "track_id", getattr(obj, "id", None))
                raw_bbox = getattr(obj, "bbox_xyxy", None)
            if tid is None:
                continue
            if isinstance(tid, (bool, np.bool_)) or not isinstance(tid, (int, np.integer)):
                raise ValueError(f"{role} 第 {f_idx} 帧第 {t_i} 个目标 track_id 必须为非布尔整数")
            tid_int = int(tid)
            if tid_int < 0:
                raise ValueError(f"{role} 第 {f_idx} 帧 track_id 不能为负: {tid_int}")
            if tid_int in seen_ids:
                raise ValueError(f"{role} 第 {f_idx} 帧存在重复 track_id: {tid_int}")
            seen_ids.add(tid_int)
            bbox = _validate_bbox_xyxy(raw_bbox, f"{role} 第 {f_idx} 帧 track {tid_int} bbox_xyxy")
            entries.append((tid_int, bbox))
        # 按 track_id 排序确保无论输入顺序如何，代价矩阵与平局打破完全确定
        entries.sort(key=lambda pair: pair[0])
        normalized[f_idx] = entries
    return normalized


def _compute_iou_matrix(
    gt_boxes: list[tuple[float, float, float, float]],
    pred_boxes: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """向量化计算 (N_gt, N_pred) 轴对齐包围盒 IoU 矩阵。"""
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float64)

    g = np.asarray(gt_boxes, dtype=np.float64)
    p = np.asarray(pred_boxes, dtype=np.float64)
    ix1 = np.maximum(g[:, None, 0], p[None, :, 0])
    iy1 = np.maximum(g[:, None, 1], p[None, :, 1])
    ix2 = np.minimum(g[:, None, 2], p[None, :, 2])
    iy2 = np.minimum(g[:, None, 3], p[None, :, 3])

    with np.errstate(over="ignore", invalid="ignore"):
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_g = np.maximum(0.0, g[:, 2] - g[:, 0]) * np.maximum(0.0, g[:, 3] - g[:, 1])
        area_p = np.maximum(0.0, p[:, 2] - p[:, 0]) * np.maximum(0.0, p[:, 3] - p[:, 1])
        union = area_g[:, None] + area_p[None, :] - inter

        iou = np.zeros((n_gt, n_pred), dtype=np.float64)
        pos = (union > 0.0) & np.isfinite(union) & np.isfinite(inter)
        iou[pos] = np.clip(inter[pos] / union[pos], 0.0, 1.0)
    return iou


def _hungarian_max_weight(weights: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """纯 Python O(n^3) Kuhn-Munkres (最短增广路) 最大权二分匹配实现。"""
    n, m = weights.shape
    if n == 0 or m == 0:
        return []

    # 有效边权重严格 > 0，无效边置 0.0 后在 max(n, m) 语义下直接求解 -score 最小代价匹配：
    # 任意无效/补零边权重均为 0，不影响最大有效权和，最终按 valid 掩码过滤即可。
    score = np.where(valid, weights, 0.0)
    transposed = n > m
    cost = (-score.T) if transposed else (-score)
    rows, cols = cost.shape

    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)

    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            row_cost = cost[i0 - 1]
            u_i0 = u[i0]
            for j in range(1, cols + 1):
                if not used[j]:
                    cur = float(row_cost[j - 1]) - u_i0 - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: list[tuple[int, int]] = []
    for j in range(1, cols + 1):
        if p[j] != 0:
            r, c = (j - 1, p[j] - 1) if transposed else (p[j] - 1, j - 1)
            if valid[r, c]:
                pairs.append((r, c))
    pairs.sort()
    return pairs


def _solve_max_weight_assignment(
    weights: np.ndarray,
    valid: np.ndarray,
    *,
    use_lap: bool = True,
) -> list[tuple[int, int]]:
    """求解仅允许 valid=True 边参与的最大权二分匹配，优先对齐 lap.lapjv 并自动回退纯 Python 实现。"""
    n, m = weights.shape
    if n == 0 or m == 0 or not bool(np.any(valid)):
        return []

    if use_lap and _lap is not None:
        # 有效边取负权 (< 0)，禁用边取 +1.0 (> cost_limit=0.0)，确保 lapjv 绝不选中无效边
        cost = np.where(valid, -weights, 1.0).astype(np.float64)
        _, row_to_col, _ = _lap.lapjv(cost, extend_cost=True, cost_limit=0.0)
        pairs: list[tuple[int, int]] = []
        for r in range(n):
            c = int(row_to_col[r])
            if c >= 0 and valid[r, c]:
                pairs.append((r, c))
        return pairs

    return _hungarian_max_weight(weights, valid)


def evaluate_mot(
    gt_frames: dict[int, list[dict]] | list[Any],
    pred_frames: dict[int, list[dict]] | list[Any],
    iou_threshold: float = 0.5,
    *,
    _use_lap: bool = True,
) -> dict:
    """评测多目标跟踪指标：CLEAR MOT (MOTA/MOTP/IDSW/FM/MT/PT/ML)、IDF1 与多阈值 HOTA。"""
    thresh = _validate_iou_threshold(iou_threshold)
    gt_norm = _normalize_frames_input(gt_frames, "gt_frames")
    pred_norm = _normalize_frames_input(pred_frames, "pred_frames")

    all_frames = sorted(set(gt_norm.keys()) | set(pred_norm.keys()))

    gt_track_len: Counter[int] = Counter()
    pred_track_len: Counter[int] = Counter()
    frame_cache: list[tuple[int, list[int], list[int], np.ndarray]] = []

    for f_idx in all_frames:
        g_entries = gt_norm.get(f_idx, [])
        p_entries = pred_norm.get(f_idx, [])
        g_ids = [tid for tid, _ in g_entries]
        p_ids = [tid for tid, _ in p_entries]
        for gid in g_ids:
            gt_track_len[gid] += 1
        for pid in p_ids:
            pred_track_len[pid] += 1
        iou_mat = _compute_iou_matrix(
            [box for _, box in g_entries],
            [box for _, box in p_entries],
        )
        frame_cache.append((f_idx, g_ids, p_ids, iou_mat))

    gt_objects = sum(gt_track_len.values())
    pred_objects = sum(pred_track_len.values())

    # 1. CLEAR MOT 逐帧最优二分匹配（优先最大化匹配基数，再最大化匹配 IoU 和）
    tp = 0
    fp = 0
    fn = 0
    matches = 0
    iou_sum = 0.0
    id_switches = 0
    fragmentations = 0
    last_matched_pred: dict[int, int] = {}
    prev_frame_matched_pred: dict[int, int] = {}
    prev_f_idx: Optional[int] = None
    ever_tracked: dict[int, bool] = {}
    was_tracked: dict[int, bool] = {}
    gt_matched_frames: Counter[int] = Counter()

    # 同步累计 IDF1 所需的全局共现频次 (IoU >= iou_threshold)
    idf1_pair_counts: Counter[tuple[int, int]] = Counter()

    eps = 1e-12
    for f_idx, g_ids, p_ids, iou_mat in frame_cache:
        valid = iou_mat >= (thresh - eps)
        if valid.any():
            nz_rows, nz_cols = np.nonzero(valid)
            for r_i, c_i in zip(nz_rows.tolist(), nz_cols.tolist()):
                idf1_pair_counts[(g_ids[r_i], p_ids[c_i])] += 1

        if prev_f_idx is None or f_idx != prev_f_idx + 1:
            prev_frame_matched_pred.clear()

        # 按 TrackEval CLEAR 规则优先延续上一相邻帧通过门限的配对，其余目标再最大化基数与 IoU 总和
        clear_pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        pred_col_by_id = {pid: c_i for c_i, pid in enumerate(p_ids)}
        for r_i, gid in enumerate(g_ids):
            prev_pid = prev_frame_matched_pred.get(gid)
            if prev_pid is None:
                continue
            c_i = pred_col_by_id.get(prev_pid)
            if c_i is not None and bool(valid[r_i, c_i]) and c_i not in used_cols:
                clear_pairs.append((r_i, c_i))
                used_rows.add(r_i)
                used_cols.add(c_i)

        rem_rows = [r_i for r_i in range(len(g_ids)) if r_i not in used_rows]
        rem_cols = [c_i for c_i in range(len(p_ids)) if c_i not in used_cols]
        if rem_rows and rem_cols:
            sub_iou = iou_mat[np.ix_(rem_rows, rem_cols)]
            sub_valid = valid[np.ix_(rem_rows, rem_cols)]
            card_bonus = float(min(len(rem_rows), len(rem_cols)) + 1)
            for sr, sc in _solve_max_weight_assignment(
                card_bonus + sub_iou, sub_valid, use_lap=_use_lap
            ):
                clear_pairs.append((rem_rows[sr], rem_cols[sc]))

        prev_frame_matched_pred = {g_ids[r_i]: p_ids[c_i] for r_i, c_i in clear_pairs}
        prev_f_idx = f_idx

        matched_gt_set: set[int] = set()
        for r_i, c_i in clear_pairs:
            gid = g_ids[r_i]
            pid = p_ids[c_i]
            matched_gt_set.add(gid)
            gt_matched_frames[gid] += 1
            iou_sum += float(iou_mat[r_i, c_i])

            if gid in last_matched_pred and last_matched_pred[gid] != pid:
                id_switches += 1
            last_matched_pred[gid] = pid

            if ever_tracked.get(gid, False) and not was_tracked.get(gid, False):
                fragmentations += 1
            ever_tracked[gid] = True
            was_tracked[gid] = True

        for gid in g_ids:
            if gid not in matched_gt_set:
                was_tracked[gid] = False

        n_match = len(clear_pairs)
        tp += n_match
        matches += n_match
        fp += len(p_ids) - n_match
        fn += len(g_ids) - n_match

    # MT (>=80%) / PT / ML (<20%) 用精确整数乘法判定，规避浮点边界误差
    mt = pt = ml = 0
    for gid, total_g in gt_track_len.items():
        m_cnt = gt_matched_frames[gid]
        if 5 * m_cnt >= 4 * total_g:
            mt += 1
        elif 5 * m_cnt < total_g:
            ml += 1
        else:
            pt += 1

    if gt_objects > 0:
        mota = 1.0 - (fp + fn + id_switches) / gt_objects
        recall = tp / gt_objects
    else:
        mota = 1.0 if pred_objects == 0 else 0.0
        recall = 1.0 if pred_objects == 0 else 0.0

    if pred_objects > 0:
        precision = tp / pred_objects
    else:
        precision = 1.0 if gt_objects == 0 else 0.0

    motp = (iou_sum / matches) if matches > 0 else 0.0

    # 2. IDF1 / IDP / IDR：全局轨迹 ID-to-ID 二分最优匹配
    sorted_gt_ids = sorted(gt_track_len.keys())
    sorted_pred_ids = sorted(pred_track_len.keys())
    gt_idx_map = {gid: i for i, gid in enumerate(sorted_gt_ids)}
    pred_idx_map = {pid: j for j, pid in enumerate(sorted_pred_ids)}

    id_mat = np.zeros((len(sorted_gt_ids), len(sorted_pred_ids)), dtype=np.float64)
    for (gid, pid), cnt in idf1_pair_counts.items():
        id_mat[gt_idx_map[gid], pred_idx_map[pid]] = float(cnt)

    id_pairs = _solve_max_weight_assignment(id_mat, id_mat > 0.0, use_lap=_use_lap)
    id_tp = int(sum(int(round(id_mat[r_i, c_i])) for r_i, c_i in id_pairs))
    id_fn = gt_objects - id_tp
    id_fp = pred_objects - id_tp

    if gt_objects + pred_objects > 0:
        idf1 = (2.0 * id_tp) / (gt_objects + pred_objects)
        idp = (id_tp / pred_objects) if pred_objects > 0 else 0.0
        idr = (id_tp / gt_objects) if gt_objects > 0 else 0.0
    else:
        idf1 = idp = idr = 1.0

    # 3. HOTA / DetA / AssA：Luiten et al. 19 档阈值 (0.05..0.95)
    deta_list: list[float] = []
    assa_list: list[float] = []
    hota_list: list[float] = []
    if gt_objects == 0 and pred_objects == 0:
        deta_list = [1.0] * len(HOTA_ALPHAS)
        assa_list = [1.0] * len(HOTA_ALPHAS)
        hota_list = [1.0] * len(HOTA_ALPHAS)
    else:
        for alpha in HOTA_ALPHAS:
            potential_counts: Counter[tuple[int, int]] = Counter()
            for _, g_ids, p_ids, iou_mat in frame_cache:
                if iou_mat.size == 0:
                    continue
                valid_a = iou_mat >= (alpha - eps)
                if valid_a.any():
                    nz_r, nz_c = np.nonzero(valid_a)
                    for r_i, c_i in zip(nz_r.tolist(), nz_c.tolist()):
                        potential_counts[(g_ids[r_i], p_ids[c_i])] += 1

            # 全球对齐关联先验：potential / (|GT_g| + |Pred_p| - potential)
            gas: dict[tuple[int, int], float] = {}
            for pair, pm in potential_counts.items():
                denom = gt_track_len[pair[0]] + pred_track_len[pair[1]] - pm
                gas[pair] = (pm / denom) if denom > 0 else 0.0

            tp_a = 0
            fp_a = 0
            fn_a = 0
            tpa_counts: Counter[tuple[int, int]] = Counter()

            for _, g_ids, p_ids, iou_mat in frame_cache:
                n_g = len(g_ids)
                n_p = len(p_ids)
                valid_a = iou_mat >= (alpha - eps)
                if valid_a.any():
                    score_mat = np.zeros((n_g, n_p), dtype=np.float64)
                    nz_r, nz_c = np.nonzero(valid_a)
                    for r_i, c_i in zip(nz_r.tolist(), nz_c.tolist()):
                        score_mat[r_i, c_i] = gas[(g_ids[r_i], p_ids[c_i])] * float(iou_mat[r_i, c_i])
                    matched_a = _solve_max_weight_assignment(score_mat, valid_a, use_lap=_use_lap)
                else:
                    matched_a = []

                for r_i, c_i in matched_a:
                    tpa_counts[(g_ids[r_i], p_ids[c_i])] += 1
                n_m = len(matched_a)
                tp_a += n_m
                fp_a += n_p - n_m
                fn_a += n_g - n_m

            det_denom = tp_a + fp_a + fn_a
            det_a = (tp_a / det_denom) if det_denom > 0 else 1.0
            if tp_a > 0:
                ass_sum = 0.0
                for (gid, pid), c_tpa in tpa_counts.items():
                    denom = gt_track_len[gid] + pred_track_len[pid] - c_tpa
                    if denom > 0:
                        ass_sum += c_tpa * (c_tpa / denom)
                ass_a = ass_sum / tp_a
            else:
                ass_a = 0.0

            hota_a = math.sqrt(max(0.0, det_a * ass_a))
            deta_list.append(det_a)
            assa_list.append(ass_a)
            hota_list.append(hota_a)

    deta = sum(deta_list) / len(HOTA_ALPHAS)
    assa = sum(assa_list) / len(HOTA_ALPHAS)
    hota = sum(hota_list) / len(HOTA_ALPHAS)

    return {
        "schema_version": 1,
        "iou_threshold": float(thresh),
        "frames_evaluated": len(all_frames),
        "gt_objects": gt_objects,
        "pred_objects": pred_objects,
        "gt_tracks": len(gt_track_len),
        "pred_tracks": len(pred_track_len),
        "matches": matches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "mt": mt,
        "pt": pt,
        "ml": ml,
        "mota": round(mota, 6),
        "motp": round(motp, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "id_tp": id_tp,
        "id_fp": id_fp,
        "id_fn": id_fn,
        "idf1": round(idf1, 6),
        "idp": round(idp, 6),
        "idr": round(idr, 6),
        "hota": round(hota, 6),
        "deta": round(deta, 6),
        "assa": round(assa, 6),
    }


def evaluate_mot_files(
    gt_path: str | Path,
    pred_path: str | Path,
    *,
    iou_threshold: float = 0.5,
    category_filter: Optional[str] = None,
    gt_one_based: Optional[bool] = None,
    pred_one_based: Optional[bool] = None,
    frame_offset: int = 0,
) -> dict:
    """从磁盘加载真值与预测轨迹文件（支持 tracks.jsonl 与 MOTChallenge TXT/CSV）并计算 MOT 指标。"""
    thresh = _validate_iou_threshold(iou_threshold)
    cat_filter = _normalize_category_filter(category_filter)
    gt_resolved = _resolve_track_path(gt_path, label="真值轨迹文件路径")
    pred_resolved = _resolve_track_path(pred_path, label="预测轨迹文件路径")

    gt_frames = load_mot_tracks(
        gt_resolved,
        category_filter=cat_filter,
        one_based=gt_one_based,
        frame_offset=frame_offset,
    )
    pred_frames = load_mot_tracks(
        pred_resolved,
        category_filter=cat_filter,
        one_based=pred_one_based,
        frame_offset=frame_offset,
    )
    report = evaluate_mot(gt_frames, pred_frames, iou_threshold=thresh)
    report["gt_path"] = str(gt_resolved)
    report["pred_path"] = str(pred_resolved)
    report["category_filter"] = cat_filter
    return report


def mot_report_to_markdown(report: dict) -> str:
    """将 evaluate_mot / evaluate_mot_files 结果格式化为简洁的 Markdown 报告。"""
    lines = [
        "# MOT Evaluation Report",
        "",
        f"- IoU threshold: `{report['iou_threshold']}`",
        f"- Frames evaluated: `{report['frames_evaluated']}`",
        f"- GT objects / tracks: `{report['gt_objects']}` / `{report['gt_tracks']}`",
        f"- Pred objects / tracks: `{report['pred_objects']}` / `{report['pred_tracks']}`",
    ]
    if report.get("category_filter") is not None:
        lines.append(f"- Category filter: `{report['category_filter']}`")
    if "gt_path" in report:
        lines.append(f"- Ground truth: `{report['gt_path']}`")
    if "pred_path" in report:
        lines.append(f"- Prediction: `{report['pred_path']}`")

    lines.extend([
        "",
        "## Core Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| MOTA | {report['mota']:.6f} |",
        f"| MOTP (mean IoU) | {report['motp']:.6f} |",
        f"| IDF1 | {report['idf1']:.6f} |",
        f"| IDP | {report['idp']:.6f} |",
        f"| IDR | {report['idr']:.6f} |",
        f"| HOTA | {report['hota']:.6f} |",
        f"| DetA | {report['deta']:.6f} |",
        f"| AssA | {report['assa']:.6f} |",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| TP / Matches | {report['tp']} |",
        f"| FP | {report['fp']} |",
        f"| FN | {report['fn']} |",
        f"| IDSW (id_switches) | {report['id_switches']} |",
        f"| FM (fragmentations) | {report['fragmentations']} |",
        f"| MT / PT / ML | {report['mt']} / {report['pt']} / {report['ml']} |",
        f"| IDTP / IDFP / IDFN | {report['id_tp']} / {report['id_fp']} / {report['id_fn']} |",
        "",
    ])
    return "\n".join(lines)


def _parse_frame_base_flag(value: str) -> Optional[bool]:
    if value == "auto":
        return None
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"frame-base 必须为 auto/0/1，当前为 {value!r}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="多目标跟踪评测工具 (MOTA / IDF1 / HOTA / IDSW)")
    parser.add_argument("--gt", type=str, required=True, help="真值轨迹文件路径 (tracks.jsonl 或 MOTChallenge .txt/.csv)")
    parser.add_argument("--pred", type=str, required=True, help="预测轨迹文件路径 (tracks.jsonl 或 MOTChallenge .txt/.csv)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="CLEAR/IDF1 匹配 IoU 阈值，默认 0.5")
    parser.add_argument("--category", type=str, default=None, help="可选类别过滤 (如 pedestrian / vehicle / car)")
    parser.add_argument("--gt-frame-base", choices=("auto", "0", "1"), default="auto", help="真值帧号起始基准 (默认 auto)")
    parser.add_argument("--pred-frame-base", choices=("auto", "0", "1"), default="auto", help="预测帧号起始基准 (默认 auto)")
    parser.add_argument("--output", type=str, default=None, help="可选输出 JSON 或 Markdown 文件路径")
    parser.add_argument("--markdown", action="store_true", help="输出 Markdown 报告而非 JSON")
    args = parser.parse_args(argv)

    if not args.gt.strip():
        parser.error("--gt 不能为空")
    if not args.pred.strip():
        parser.error("--pred 不能为空")
    if args.output is not None and not args.output.strip():
        parser.error("--output 不能为空")
    if args.category is not None and not args.category.strip():
        parser.error("--category 不能为空")
    if not math.isfinite(args.iou_threshold) or args.iou_threshold <= 0.0 or args.iou_threshold > 1.0:
        parser.error("--iou-threshold 必须在 (0.0, 1.0] 区间内")

    report = evaluate_mot_files(
        args.gt,
        args.pred,
        iou_threshold=args.iou_threshold,
        category_filter=args.category,
        gt_one_based=_parse_frame_base_flag(args.gt_frame_base),
        pred_one_based=_parse_frame_base_flag(args.pred_frame_base),
    )
    content = (
        mot_report_to_markdown(report)
        if args.markdown
        else json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    )
    if args.output is not None:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
