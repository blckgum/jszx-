# -*- coding: utf-8 -*-
"""
遮挡断轨缝合模块 (Tracklet Stitching across Short Occlusions)
=============================================================================
基于地面物理状态外推与马氏距离门控，在目标发生短暂遮挡导致底层跟踪器
分配新 raw_track_id 时，将其与未过期的丢失轨迹一对一关联并恢复统一的
canonical_id，从而避免测速滤波器重置及单次冲突事件被拆分成多条碎片记录。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .mot_eval import _hungarian_max_weight


def _is_finite_number(value: Any) -> bool:
    """精确校验有限数值，兼容 NumPy 数值标量并拒绝 bool/np.bool_ 伪装。"""
    # 显式排除 Python bool 与 np.bool_，再兼容 Python/NumPy 实数标量
    if isinstance(value, (bool, np.bool_)):
        return False
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _is_positive_int(value: Any) -> bool:
    """精确校验正整数，兼容 NumPy 整数标量并拒绝 bool/np.bool_。"""
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _normalize_category(category: Any) -> str:
    """校验并标准化目标粗类别，使 person/pedestrian 归一。"""
    if not isinstance(category, str) or not category.strip():
        raise ValueError("目标 category 必须为非空字符串")
    norm = category.strip().lower()
    if norm in ("person", "pedestrian"):
        return "pedestrian"
    return norm


def _parse_cov_2x2(cov: Any) -> Optional[np.ndarray]:
    """将 4 元组或 2x2 协方差解析并校验为对称半正定 2x2 矩阵。"""
    if cov is None:
        return None
    try:
        if isinstance(cov, (tuple, list)) and len(cov) == 4 and not isinstance(cov[0], (tuple, list, np.ndarray)):
            if any(not _is_finite_number(v) for v in cov):
                return None
            mat = np.array([[float(cov[0]), float(cov[1])], [float(cov[2]), float(cov[3])]], dtype=float)
        else:
            arr = np.asarray(cov)
            if arr.shape != (2, 2) or arr.dtype == np.bool_:
                return None
            flat = [cov_el for cov_el in arr.reshape(-1).tolist()]
            if any(not _is_finite_number(v) for v in flat):
                return None
            mat = arr.astype(float, copy=True)
    except (TypeError, ValueError):
        return None

    if not np.all(np.isfinite(mat)):
        return None
    mat = 0.5 * (mat + mat.T)
    eigvals = np.linalg.eigvalsh(mat)
    if float(eigvals.min()) < -1e-9:
        return None
    if float(eigvals.min()) < 0.0:
        mat = mat + (-float(eigvals.min()) + 1e-9) * np.eye(2, dtype=float)
    return mat


def _validate_appearance_hist(hist: Any) -> Optional[np.ndarray]:
    """校验并 L1 归一化外观直方图；无效、全零、含 NaN/Inf 或布尔数组返回 None。"""
    if hist is None or isinstance(hist, (bool, np.bool_, str, bytes)):
        return None
    try:
        arr = np.asarray(hist)
    except (TypeError, ValueError):
        return None
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        return None
    if arr.ndim < 1 or arr.size == 0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        arr_f = arr.astype(np.float64, copy=True)
        if not np.all(np.isfinite(arr_f)) or np.any(arr_f < 0.0):
            return None
        total = float(arr_f.sum())
        if not math.isfinite(total) or total <= 0.0:
            return None
        arr_f /= total
    return arr_f


def extract_appearance_histogram(frame_bgr: Any, bbox_xyxy: Any) -> Optional[np.ndarray]:
    """截取检测框中心内缩区域（约 60% 宽高避开背景边缘），返回 16x8 H-S L1 归一化二维直方图。"""
    if frame_bgr is None or isinstance(frame_bgr, (bool, np.bool_, str, bytes)):
        return None
    try:
        arr = np.asarray(frame_bgr)
    except (TypeError, ValueError):
        return None
    if (
        arr.ndim != 3
        or arr.shape[2] != 3
        or arr.shape[0] < 2
        or arr.shape[1] < 2
        or arr.dtype == np.bool_
        or not np.issubdtype(arr.dtype, np.number)
    ):
        return None

    if isinstance(bbox_xyxy, (str, bytes)) or not isinstance(bbox_xyxy, (Sequence, np.ndarray)):
        return None
    if len(bbox_xyxy) != 4:
        return None
    if not all(_is_finite_number(v) for v in bbox_xyxy):
        return None

    fx1, fy1, fx2, fy2 = (float(v) for v in bbox_xyxy)
    if fx2 <= fx1 or fy2 <= fy1:
        return None

    img_h, img_w = int(arr.shape[0]), int(arr.shape[1])
    # 将检测框裁剪至图像合法范围，过滤越界无效框或尺寸过小框
    cx1 = max(0.0, min(float(img_w), fx1))
    cy1 = max(0.0, min(float(img_h), fy1))
    cx2 = max(0.0, min(float(img_w), fx2))
    cy2 = max(0.0, min(float(img_h), fy2))
    bw = cx2 - cx1
    bh = cy2 - cy1
    if bw < 4.0 or bh < 4.0:
        return None

    # 内缩两侧各 20%，保留中心约 60% 宽高区域以减弱背景边缘干扰
    margin_x = 0.2 * bw
    margin_y = 0.2 * bh
    ix1 = max(0, int(math.floor(cx1 + margin_x)))
    iy1 = max(0, int(math.floor(cy1 + margin_y)))
    ix2 = min(img_w, int(math.ceil(cx2 - margin_x)))
    iy2 = min(img_h, int(math.ceil(cy2 - margin_y)))
    if (ix2 - ix1) < 2 or (iy2 - iy1) < 2:
        return None

    patch = arr[iy1:iy2, ix1:ix2]
    if patch.size == 0:
        return None
    if patch.dtype != np.uint8:
        with np.errstate(over="ignore", invalid="ignore"):
            if not np.all(np.isfinite(patch)):
                return None
            patch = np.clip(patch, 0, 255).astype(np.uint8)

    # 仅在提取图像外观直方图时延迟加载 cv2，保证纯运动学离线模式零额外依赖
    import cv2

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return _validate_appearance_hist(hist)


def appearance_bhattacharyya_distance(hist_a: Any, hist_b: Any) -> Optional[float]:
    """纯 NumPy 计算两个外观直方图间的有界 Bhattacharyya 距离，值域 [0.0, 1.0]。"""
    ha = _validate_appearance_hist(hist_a)
    hb = _validate_appearance_hist(hist_b)
    if ha is None or hb is None or ha.shape != hb.shape:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        bc = float(np.sum(np.sqrt(ha * hb)))
    if not math.isfinite(bc):
        return None
    bc = max(0.0, min(1.0, bc))
    dist = math.sqrt(max(0.0, 1.0 - bc))
    return max(0.0, min(1.0, dist))


@dataclass(frozen=True)
class TrackObservation:
    """单帧地面投影观测输入数据结构"""
    raw_track_id: int
    category: str
    object_kind: str
    x_m: float
    y_m: float
    appearance_hist: Optional[np.ndarray] = None


@dataclass(frozen=True)
class StitchEvent:
    """一次成功的跨遮挡轨迹缝合审计记录"""
    timestamp_s: float
    raw_track_id: int
    canonical_id: int
    category: str
    gap_s: float
    distance_m: float
    mahalanobis: float
    heading_diff_deg: Optional[float]


@dataclass
class _TrackletState:
    """轨迹物理运动状态快照（用于活跃跟踪与丢失池外推匹配）"""
    canonical_id: int
    raw_track_id: int
    category: str
    object_kind: str
    x_m: float
    y_m: float
    last_seen_s: float
    vx_ms: float = 0.0
    vy_ms: float = 0.0
    ax_ms2: float = 0.0
    ay_ms2: float = 0.0
    pos_cov: Optional[np.ndarray] = None
    is_reliable: bool = False
    frames_seen: int = 1
    _has_external_kinematics: bool = False
    appearance_hist: Optional[np.ndarray] = None


def _extract_observation(obs: Any) -> Optional[TrackObservation]:
    """兼容元组、字典或对象形式的观测输入；非有限地面坐标返回 None 跳过。"""
    raw_hist: Any = None
    if isinstance(obs, (tuple, list)):
        if len(obs) == 6:
            raw_id, category, object_kind, x_m, y_m, raw_hist = obs
        elif len(obs) == 5:
            raw_id, category, object_kind, x_m, y_m = obs
        elif len(obs) == 4:
            raw_id, category, x_m, y_m = obs
            object_kind = ""
        else:
            raise ValueError("观测元组长度必须为 4、5 或 6")
    elif isinstance(obs, dict):
        # 显式传入 raw_track_id=None 时仍按约定回退读取 track_id，保留 0/False 以触发校验报错
        raw_id = obs.get("raw_track_id")
        if raw_id is None:
            raw_id = obs.get("track_id")
        category = obs.get("category")
        object_kind = obs.get("object_kind", "")
        x_m = obs.get("x_m")
        y_m = obs.get("y_m")
        raw_hist = obs.get("appearance_hist")
    else:
        # 对象属性 raw_track_id 显式为 None 时同样回退读取 track_id
        raw_id = getattr(obs, "raw_track_id", None)
        if raw_id is None:
            raw_id = getattr(obs, "track_id", None)
        category = getattr(obs, "category", None)
        object_kind = getattr(obs, "object_kind", "")
        x_m = getattr(obs, "x_m", None)
        y_m = getattr(obs, "y_m", None)
        raw_hist = getattr(obs, "appearance_hist", None)

    if raw_id is None:
        return None
    if not _is_positive_int(raw_id):
        raise ValueError("raw_track_id 必须为正整数")
    norm_cat = _normalize_category(category)
    kind_str = object_kind.strip() if isinstance(object_kind, str) else ""
    if isinstance(x_m, (bool, np.bool_)) or isinstance(y_m, (bool, np.bool_)):
        raise ValueError("观测地面坐标 x_m/y_m 不能为布尔值")
    # 兼容 Python 与 NumPy 整数/浮点标量，转换为 float 后校验有限性
    if not (
        isinstance(x_m, (int, float, np.integer, np.floating))
        and isinstance(y_m, (int, float, np.integer, np.floating))
    ):
        raise ValueError("观测地面坐标 x_m/y_m 必须为数值")
    x_val = float(x_m)
    y_val = float(y_m)
    if not (math.isfinite(x_val) and math.isfinite(y_val)):
        return None

    appearance_hist = _validate_appearance_hist(raw_hist)

    return TrackObservation(
        raw_track_id=int(raw_id),
        category=norm_cat,
        object_kind=kind_str,
        x_m=x_val,
        y_m=y_val,
        appearance_hist=appearance_hist,
    )


class TrackletStitcher:
    """短时遮挡断轨缝合器：结合地面运动外推与马氏距离门控恢复统一 ID。"""

    # 当上游速度估计器未提供位置协方差（如 EMA 模式）时的默认位置方差基准 (m^2)
    _DEFAULT_POS_VAR_M2 = 0.25
    # 阻尼加速外推时长上限 (s)，抑制长间隔下噪点加速度二次方发散
    _ACCEL_DAMPING_CAP_S = 0.5
    # 航向一致性检查的速度与位移门限，避免低速/极短步长下方向角被投影抖动支配
    _MIN_HEADING_SPEED_MS = 0.8
    _MIN_HEADING_DISP_M = 0.5
    # 活跃帧外观直方图 EMA 平滑更新系数
    _APPEARANCE_EMA_ALPHA = 0.3

    def __init__(
        self,
        max_gap_s: float = 1.5,
        max_distance_m: float = 3.5,
        max_mahalanobis: float = 3.5,
        max_heading_diff_deg: float = 45.0,
        velocity_noise_ms: float = 1.5,
        max_lost_pool: int = 256,
        max_appearance_dist: float = 0.55,
        appearance_weight: float = 2.0,
    ):
        for name, val in (
            ("max_gap_s", max_gap_s),
            ("max_distance_m", max_distance_m),
            ("max_mahalanobis", max_mahalanobis),
            ("max_heading_diff_deg", max_heading_diff_deg),
            ("velocity_noise_ms", velocity_noise_ms),
        ):
            if not _is_finite_number(val) or val <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if not _is_positive_int(max_lost_pool):
            raise ValueError("max_lost_pool 必须为正整数")
        if (
            not _is_finite_number(max_appearance_dist)
            or float(max_appearance_dist) <= 0.0
            or float(max_appearance_dist) > 1.0
        ):
            raise ValueError("max_appearance_dist 必须在 (0, 1] 区间")
        if not _is_finite_number(appearance_weight) or float(appearance_weight) < 0.0:
            raise ValueError("appearance_weight 必须为有限非负数")

        self.max_gap_s = float(max_gap_s)
        self.max_distance_m = float(max_distance_m)
        self.max_mahalanobis = float(max_mahalanobis)
        self.max_heading_diff_deg = float(max_heading_diff_deg)
        self.velocity_noise_ms = float(velocity_noise_ms)
        self.max_lost_pool = int(max_lost_pool)
        self.max_appearance_dist = float(max_appearance_dist)
        self.appearance_weight = float(appearance_weight)

        self._active_tracks: Dict[int, _TrackletState] = {}
        self._lost_tracks: Dict[int, _TrackletState] = {}
        self._raw_to_canonical: "OrderedDict[int, int]" = OrderedDict()
        self._used_canonical_ids: set[int] = set()
        self._next_canonical_id: int = 1
        self._last_resolve_ts: Optional[float] = None
        self.stitch_events: List[StitchEvent] = []

    @property
    def active_count(self) -> int:
        return len(self._active_tracks)

    @property
    def lost_count(self) -> int:
        return len(self._lost_tracks)

    @property
    def stitched_track_count(self) -> int:
        """累计成功缝合次数。"""
        return len(self.stitch_events)

    def reset(self) -> None:
        """重置缝合器全部状态。"""
        self._active_tracks.clear()
        self._lost_tracks.clear()
        self._raw_to_canonical.clear()
        self._used_canonical_ids.clear()
        self._next_canonical_id = 1
        self._last_resolve_ts = None
        self.stitch_events.clear()

    def _allocate_canonical_id(self, preferred_raw_id: int) -> int:
        if preferred_raw_id not in self._used_canonical_ids:
            cid = preferred_raw_id
        else:
            while self._next_canonical_id in self._used_canonical_ids:
                self._next_canonical_id += 1
            cid = self._next_canonical_id
        self._used_canonical_ids.add(cid)
        if cid >= self._next_canonical_id:
            self._next_canonical_id = cid + 1
        return cid

    def _purge_expired_and_bound_lost(self, timestamp_s: float) -> None:
        """清理超期丢失轨迹，并将丢失池限制在 max_lost_pool 容量以内。"""
        for cid in list(self._lost_tracks.keys()):
            if timestamp_s - self._lost_tracks[cid].last_seen_s > self.max_gap_s:
                del self._lost_tracks[cid]
        if len(self._lost_tracks) > self.max_lost_pool:
            ordered = sorted(
                self._lost_tracks.items(),
                key=lambda item: (item[1].last_seen_s, item[0]),
            )
            excess = len(self._lost_tracks) - self.max_lost_pool
            for cid, _ in ordered[:excess]:
                del self._lost_tracks[cid]

    def _retire_missing_active(self, keep_cids: set[int], timestamp_s: float) -> None:
        """将本帧未出现的活跃轨迹移入丢失池，保留其最后实际观测时间。"""
        for cid in list(self._active_tracks.keys()):
            if cid not in keep_cids:
                self._lost_tracks[cid] = self._active_tracks.pop(cid)
        self._purge_expired_and_bound_lost(timestamp_s)

    def _bound_raw_map(self) -> None:
        """长流运行时约束 raw_id -> canonical_id 历史映射规模，优先剔除已失效映射。"""
        cap = max(4096, self.max_lost_pool * 16)
        if len(self._raw_to_canonical) <= cap:
            return
        active_or_lost = set(self._active_tracks.keys()) | set(self._lost_tracks.keys())
        for raw_id in list(self._raw_to_canonical.keys()):
            if len(self._raw_to_canonical) <= cap:
                break
            if self._raw_to_canonical[raw_id] not in active_or_lost:
                del self._raw_to_canonical[raw_id]

    @staticmethod
    def _advance_observation_state(tr: _TrackletState, obs: TrackObservation, timestamp_s: float) -> None:
        """用当前帧观测刷新轨迹位置；若无外部滤波器，则计算简单有限差分速度兜底。"""
        dt = timestamp_s - tr.last_seen_s
        if not tr._has_external_kinematics and dt > 1e-4:
            raw_vx = (obs.x_m - tr.x_m) / dt
            raw_vy = (obs.y_m - tr.y_m) / dt
            if math.hypot(raw_vx, raw_vy) <= 33.3:
                if tr.frames_seen <= 1:
                    tr.vx_ms, tr.vy_ms = raw_vx, raw_vy
                else:
                    tr.vx_ms = 0.5 * raw_vx + 0.5 * tr.vx_ms
                    tr.vy_ms = 0.5 * raw_vy + 0.5 * tr.vy_ms
        tr.raw_track_id = obs.raw_track_id
        tr.category = obs.category
        if obs.object_kind:
            tr.object_kind = obs.object_kind
        tr.x_m = obs.x_m
        tr.y_m = obs.y_m
        tr.last_seen_s = timestamp_s
        tr.frames_seen += 1
        # 若当前观测携带有效外观直方图，在活跃帧中按 EMA 平滑并维持 L1 归一化
        if obs.appearance_hist is not None:
            if tr.appearance_hist is None or tr.appearance_hist.shape != obs.appearance_hist.shape:
                tr.appearance_hist = obs.appearance_hist.copy()
            else:
                alpha = TrackletStitcher._APPEARANCE_EMA_ALPHA
                blended = (1.0 - alpha) * tr.appearance_hist + alpha * obs.appearance_hist
                total = float(blended.sum())
                if math.isfinite(total) and total > 0.0:
                    tr.appearance_hist = blended / total

    def _evaluate_stitch_candidate(
        self,
        lost: _TrackletState,
        obs: TrackObservation,
        timestamp_s: float,
    ) -> Optional[Tuple[float, float, float, float, Optional[float]]]:
        """检验丢失轨迹与新观测的时空/运动学/外观一致性，返回 (fused_cost, d_M, dist_m, dt, heading_diff)。"""
        if lost.category != obs.category:
            return None
        dt = timestamp_s - lost.last_seen_s
        if dt < 0.0 or dt > self.max_gap_s:
            return None

        # 外推地面位置：速度线性外推 + 阻尼加速度修正
        dt_damped = min(dt, self._ACCEL_DAMPING_CAP_S)
        pred_x = lost.x_m + lost.vx_ms * dt + 0.5 * lost.ax_ms2 * (dt_damped * dt_damped)
        pred_y = lost.y_m + lost.vy_ms * dt + 0.5 * lost.ay_ms2 * (dt_damped * dt_damped)

        err_x = obs.x_m - pred_x
        err_y = obs.y_m - pred_y
        dist_m = math.hypot(err_x, err_y)
        if dist_m > self.max_distance_m:
            return None

        # 马氏距离门控：传播位置协方差与速度过程噪声方差
        base_cov = (
            lost.pos_cov
            if lost.pos_cov is not None
            else self._DEFAULT_POS_VAR_M2 * np.eye(2, dtype=float)
        )
        vel_var = (self.velocity_noise_ms * dt) ** 2
        cov_prop = base_cov + (vel_var + 1e-6) * np.eye(2, dtype=float)
        diff = np.array([err_x, err_y], dtype=float)
        try:
            inv_diff = np.linalg.solve(cov_prop, diff)
        except np.linalg.LinAlgError:
            return None
        d_m2 = float(diff @ inv_diff)
        if not math.isfinite(d_m2):
            return None
        d_m = math.sqrt(max(0.0, d_m2))
        if d_m > self.max_mahalanobis:
            return None

        # 运动目标航向一致性校验：以丢失点到新观测的位移矢量比对历史速度方向
        speed0 = math.hypot(lost.vx_ms, lost.vy_ms)
        disp_x = obs.x_m - lost.x_m
        disp_y = obs.y_m - lost.y_m
        disp_norm = math.hypot(disp_x, disp_y)
        heading_diff_deg: Optional[float] = None
        if speed0 >= self._MIN_HEADING_SPEED_MS and disp_norm >= self._MIN_HEADING_DISP_M:
            cos_theta = (lost.vx_ms * disp_x + lost.vy_ms * disp_y) / (speed0 * disp_norm)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            heading_diff_deg = math.degrees(math.acos(cos_theta))
            if heading_diff_deg > self.max_heading_diff_deg:
                return None

        # 仅当双方均带有效外观直方图时启用外观门控与代价融合，否则 100% 退化为纯运动学
        fused_cost = d_m
        if lost.appearance_hist is not None and obs.appearance_hist is not None:
            app_dist = appearance_bhattacharyya_distance(
                lost.appearance_hist, obs.appearance_hist
            )
            if app_dist is not None:
                if app_dist > self.max_appearance_dist:
                    return None
                fused_cost = d_m + self.appearance_weight * app_dist

        return (fused_cost, d_m, dist_m, dt, heading_diff_deg)

    def resolve_frame(
        self,
        observations: Iterable[Any],
        timestamp_s: float,
    ) -> Tuple[Dict[int, int], List[Tuple[int, int]]]:
        """解析当前帧跟踪观测并执行跨遮挡缝合，返回 (id_map, newly_stitched_pairs)。"""
        if not _is_finite_number(timestamp_s):
            raise ValueError("timestamp_s 必须为有限数值")
        if self._last_resolve_ts is not None and timestamp_s < self._last_resolve_ts:
            raise ValueError("timestamp_s 不能小于上一帧时间戳")
        ts = float(timestamp_s)

        parsed: List[TrackObservation] = []
        seen_raw_ids: set[int] = set()
        for raw_obs in observations:
            item = _extract_observation(raw_obs)
            if item is None:
                continue
            if item.raw_track_id in seen_raw_ids:
                raise ValueError(f"同一帧内包含重复 raw_track_id: {item.raw_track_id}")
            seen_raw_ids.add(item.raw_track_id)
            parsed.append(item)

        self._purge_expired_and_bound_lost(ts)

        id_map: Dict[int, int] = {}
        newly_stitched_pairs: List[Tuple[int, int]] = []
        seen_cids_this_frame: set[int] = set()
        unknown_obs: List[TrackObservation] = []

        # 1) 已知 raw_track_id 直接映射到已有 canonical_id；若同帧碰撞或当前所有权不再归本 raw_id 则转入新轨迹处理
        for obs in parsed:
            known_cid = self._raw_to_canonical.get(obs.raw_track_id)
            active_tr = self._active_tracks.get(known_cid) if known_cid is not None else None
            lost_tr = self._lost_tracks.get(known_cid) if known_cid is not None else None
            # 核对活跃/丢失池所有权，严禁被取代的旧 raw_id 免检抢占其他 raw_id 正在或刚持有的 canonical_id
            if (
                known_cid is None
                or known_cid in seen_cids_this_frame
                or (active_tr is not None and active_tr.raw_track_id != obs.raw_track_id)
                or (lost_tr is not None and lost_tr.raw_track_id != obs.raw_track_id)
            ):
                unknown_obs.append(obs)
                continue

            seen_cids_this_frame.add(known_cid)
            self._raw_to_canonical.move_to_end(obs.raw_track_id)
            if known_cid in self._lost_tracks:
                tr = self._lost_tracks.pop(known_cid)
                self._active_tracks[known_cid] = tr
                self._advance_observation_state(tr, obs, ts)
            elif known_cid in self._active_tracks:
                self._advance_observation_state(self._active_tracks[known_cid], obs, ts)
            else:
                self._used_canonical_ids.add(known_cid)
                self._active_tracks[known_cid] = _TrackletState(
                    canonical_id=known_cid,
                    raw_track_id=obs.raw_track_id,
                    category=obs.category,
                    object_kind=obs.object_kind,
                    x_m=obs.x_m,
                    y_m=obs.y_m,
                    last_seen_s=ts,
                    appearance_hist=(
                        obs.appearance_hist.copy()
                        if obs.appearance_hist is not None
                        else None
                    ),
                )
            id_map[obs.raw_track_id] = known_cid

        # 2) 先将本帧缺席的活跃轨迹退入丢失池，支持瞬时切换及多帧遮挡后的缝合匹配
        self._retire_missing_active(seen_cids_this_frame, ts)

        # 3) 对新出现的 raw_track_id，与未过期丢失轨迹执行一对一运动学、马氏距离及外观融合匹配
        candidates: List[
            Tuple[float, float, float, float, int, int, Optional[float], TrackObservation]
        ] = []
        for obs in unknown_obs:
            for lost_cid, lost_tr in self._lost_tracks.items():
                metrics = self._evaluate_stitch_candidate(lost_tr, obs, ts)
                if metrics is None:
                    continue
                fused_cost, d_m, dist_m, gap_s, heading_diff_deg = metrics
                candidates.append(
                    (
                        fused_cost,
                        d_m,
                        dist_m,
                        gap_s,
                        obs.raw_track_id,
                        lost_cid,
                        heading_diff_deg,
                        obs,
                    )
                )

        # 使用最大基数 + 最小融合代价二分匹配进行一对一缝合，避免局部最小代价抢占造成可匹配新轨迹建重 ID
        matched_raw: set[int] = set()
        matched_lost: set[int] = set()
        assigned_rows: List[Tuple[float, float, float, float, int, int, Optional[float], TrackObservation]] = []
        if candidates:
            raw_ids = sorted({row[4] for row in candidates})
            lost_cids = sorted({row[5] for row in candidates})
            raw_pos = {rid: idx for idx, rid in enumerate(raw_ids)}
            lost_pos = {cid: idx for idx, cid in enumerate(lost_cids)}
            rank_by_pair: dict[tuple[int, int], tuple[int, tuple[float, float, float, float, int, int, Optional[float], TrackObservation]]] = {}
            ordered_cands = sorted(
                candidates,
                key=lambda row: (row[0], row[1], row[2], row[3], row[4], row[5]),
            )
            total_cands = max(1, len(ordered_cands))
            max_cost = max(0.0, max(float(row[0]) for row in ordered_cands)) + 1.0
            weights = np.zeros((len(raw_ids), len(lost_cids)), dtype=np.float64)
            valid = np.zeros((len(raw_ids), len(lost_cids)), dtype=bool)
            card_bonus = float(min(len(raw_ids), len(lost_cids)) + 1)
            for rank_idx, row in enumerate(ordered_cands):
                ri = raw_pos[row[4]]
                ci = lost_pos[row[5]]
                valid[ri, ci] = True
                # 归一化真实 fused_cost（0.45 比重）并辅以微小确定性序偏移，确保先最大化匹配数，
                # 又在同基数下最小化融合代价总和而非仅按秩次相加。
                cost_val = max(0.0, float(row[0]))
                penalty = 0.45 * (cost_val / max_cost) + 1e-6 * (rank_idx / total_cands)
                weights[ri, ci] = card_bonus - penalty
                rank_by_pair[(ri, ci)] = (rank_idx, row)

            pairs = _hungarian_max_weight(weights, valid)
            assigned_rows = [rank_by_pair[(ri, ci)][1] for ri, ci in pairs]
            assigned_rows.sort(key=lambda row: (row[4], row[5]))

        for _fused, d_m, dist_m, gap_s, raw_id, lost_cid, heading_diff_deg, obs in assigned_rows:
            if raw_id in matched_raw or lost_cid in matched_lost:
                continue
            matched_raw.add(raw_id)
            matched_lost.add(lost_cid)

            tr = self._lost_tracks.pop(lost_cid)
            self._advance_observation_state(tr, obs, ts)
            self._active_tracks[lost_cid] = tr
            # 清理所有先前指向该 lost_cid 的旧 raw_track_id 键，防止旧 ID 复活后免检复用或抢占
            stale_raw_ids = [
                old_raw
                for old_raw, mapped_cid in self._raw_to_canonical.items()
                if mapped_cid == lost_cid and old_raw != raw_id
            ]
            for old_raw in stale_raw_ids:
                del self._raw_to_canonical[old_raw]
            self._raw_to_canonical[raw_id] = lost_cid
            self._raw_to_canonical.move_to_end(raw_id)
            id_map[raw_id] = lost_cid
            newly_stitched_pairs.append((raw_id, lost_cid))
            self.stitch_events.append(
                StitchEvent(
                    timestamp_s=ts,
                    raw_track_id=raw_id,
                    canonical_id=lost_cid,
                    category=obs.category,
                    gap_s=round(gap_s, 6),
                    distance_m=round(dist_m, 6),
                    mahalanobis=round(d_m, 6),
                    heading_diff_deg=(
                        round(heading_diff_deg, 3) if heading_diff_deg is not None else None
                    ),
                )
            )

        # 4) 未匹配成功的新 raw_track_id 开立新 canonical_id
        for obs in unknown_obs:
            if obs.raw_track_id in matched_raw:
                continue
            cid = self._allocate_canonical_id(obs.raw_track_id)
            self._raw_to_canonical[obs.raw_track_id] = cid
            self._raw_to_canonical.move_to_end(obs.raw_track_id)
            self._active_tracks[cid] = _TrackletState(
                canonical_id=cid,
                raw_track_id=obs.raw_track_id,
                category=obs.category,
                object_kind=obs.object_kind,
                x_m=obs.x_m,
                y_m=obs.y_m,
                last_seen_s=ts,
                appearance_hist=(
                    obs.appearance_hist.copy()
                    if obs.appearance_hist is not None
                    else None
                ),
            )
            id_map[obs.raw_track_id] = cid

        self._bound_raw_map()
        self._last_resolve_ts = ts
        return id_map, newly_stitched_pairs

    def update_kinematics(
        self,
        canonical_id: int,
        reading: Any,
        timestamp_s: float,
    ) -> None:
        """用测速滤波器输出的最新运动学状态更新活跃轨迹，并将本帧缺席轨迹移入丢失池。"""
        if not _is_positive_int(canonical_id):
            raise ValueError("canonical_id 必须为正整数")
        if not _is_finite_number(timestamp_s):
            raise ValueError("timestamp_s 必须为有限数值")
        # 统一转为 Python int，防止 NumPy 整数键混入内部字典与集合
        cid_key = int(canonical_id)
        ts = float(timestamp_s)

        # 独立调用 update_kinematics 时，将时间戳落后于当前帧的其余活跃轨迹归入丢失池
        for cid in list(self._active_tracks.keys()):
            if cid != cid_key and self._active_tracks[cid].last_seen_s < ts:
                self._lost_tracks[cid] = self._active_tracks.pop(cid)
        self._purge_expired_and_bound_lost(ts)

        x_m = getattr(reading, "x_m", None)
        y_m = getattr(reading, "y_m", None)
        vx_ms = getattr(reading, "vx_ms", 0.0)
        vy_ms = getattr(reading, "vy_ms", 0.0)
        ax_ms2 = getattr(reading, "ax_ms2", 0.0)
        ay_ms2 = getattr(reading, "ay_ms2", 0.0)
        raw_reliable = getattr(reading, "is_reliable", False)
        # 兼容 Python bool 与 NumPy np.bool_ 标量
        is_reliable = bool(raw_reliable) if isinstance(raw_reliable, (bool, np.bool_)) else False
        pos_cov = _parse_cov_2x2(getattr(reading, "pos_cov_m2", None))

        if not all(_is_finite_number(v) for v in (x_m, y_m, vx_ms, vy_ms)):
            return

        ax_val = float(ax_ms2) if _is_finite_number(ax_ms2) else 0.0
        ay_val = float(ay_ms2) if _is_finite_number(ay_ms2) else 0.0

        raw_cat = getattr(reading, "category", None)
        norm_cat = _normalize_category(raw_cat) if isinstance(raw_cat, str) and raw_cat.strip() else "vehicle"
        raw_kind = getattr(reading, "object_kind", "")
        kind_str = raw_kind.strip() if isinstance(raw_kind, str) else ""

        if cid_key in self._active_tracks:
            tr = self._active_tracks[cid_key]
        elif cid_key in self._lost_tracks:
            tr = self._lost_tracks.pop(cid_key)
            self._active_tracks[cid_key] = tr
        else:
            self._used_canonical_ids.add(cid_key)
            if cid_key >= self._next_canonical_id:
                self._next_canonical_id = cid_key + 1
            tr = _TrackletState(
                canonical_id=cid_key,
                raw_track_id=cid_key,
                category=norm_cat,
                object_kind=kind_str,
                x_m=float(x_m),
                y_m=float(y_m),
                last_seen_s=ts,
            )
            self._active_tracks[cid_key] = tr

        tr.x_m = float(x_m)
        tr.y_m = float(y_m)
        tr.vx_ms = float(vx_ms)
        tr.vy_ms = float(vy_ms)
        tr.ax_ms2 = ax_val
        tr.ay_ms2 = ay_val
        tr.pos_cov = pos_cov
        tr.is_reliable = is_reliable
        tr.last_seen_s = ts
        tr._has_external_kinematics = True
        if isinstance(raw_cat, str) and raw_cat.strip():
            tr.category = norm_cat
        if kind_str:
            tr.object_kind = kind_str

    def resolve_canonical_id(self, raw_track_id: int) -> int:
        """查询已知 raw_track_id 对应的 canonical_id，未知时原样返回。"""
        # 兼容 NumPy 整数标量查询并统一返回 Python int
        key = (
            int(raw_track_id)
            if isinstance(raw_track_id, np.integer) and not isinstance(raw_track_id, np.bool_)
            else raw_track_id
        )
        return self._raw_to_canonical.get(key, key)

    def retained_canonical_ids(self, timestamp_s: Optional[float] = None) -> set[int]:
        """返回当前仍在活跃池或未过期丢失池中的全部 canonical_id，供测速器保活。"""
        if timestamp_s is not None:
            if not _is_finite_number(timestamp_s):
                raise ValueError("timestamp_s 必须为有限数值")
            self._purge_expired_and_bound_lost(float(timestamp_s))
        return set(self._active_tracks.keys()) | set(self._lost_tracks.keys())
