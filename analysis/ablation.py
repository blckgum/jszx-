"""无视频的确定性参数消融与算法对比：用合成米制及透视噪声轨迹对比测速、冲突与断轨缝合。"""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from .calibration import RoadCalibrator
from .conflict import ConflictAnalyzer, RiskLevel
from .events import ConflictEventTracker
from .speed import SpeedEstimator, SpeedReading
from .stitching import TrackletStitcher


def _speed_reading(
    track_id,
    category,
    x,
    y,
    vx,
    vy,
    object_kind="motor_vehicle",
    *,
    heading_deg: float | None = None,
    ax_ms2: float = 0.0,
    ay_ms2: float = 0.0,
):
    speed = math.hypot(vx, vy)
    if heading_deg is None:
        heading_deg = (
            (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
            if speed > 1e-6
            else 0.0
        )
    return SpeedReading(
        track_id=track_id,
        category=category,
        x_m=x,
        y_m=y,
        vx_ms=vx,
        vy_ms=vy,
        speed_ms=speed,
        speed_kmh=speed * 3.6,
        is_speeding=speed * 3.6 > 30.0,
        heading_deg=heading_deg,
        is_reliable=True,
        object_kind=object_kind,
        ax_ms2=ax_ms2,
        ay_ms2=ay_ms2,
    )


def run_speed_ablation(
    speeds_ms=(2.0, 5.0, 10.0),
    window_sizes=(4, 8, 12),
    ema_alphas=(0.2, 0.35, 0.6),
    fps=25.0,
    frames=40,
) -> list[dict]:
    """对匀速直线轨迹运行窗口/EMA 组合，返回末帧误差和可靠性。"""
    # 在理想无噪声米制轨迹上扫描滑窗与 EMA 参数组合，检验算法收敛精度与响应特性。
    rows = []
    for target_speed, window, alpha in product(speeds_ms, window_sizes, ema_alphas):
        estimator = SpeedEstimator(window_size=window, ema_alpha=alpha)
        final = None
        for index in range(frames):
            final = estimator.update(
                1, "vehicle", index / fps, 3.0, target_speed * index / fps
            )
        rows.append({
            "target_speed_ms": target_speed,
            "window_size": window,
            "ema_alpha": alpha,
            "measured_speed_ms": round(final.speed_ms, 6),
            "absolute_error_ms": round(abs(final.speed_ms - target_speed), 6),
            "is_reliable": final.is_reliable,
            "is_stationary": final.is_stationary,
        })
    return rows


def run_conflict_ablation(
    critical_thresholds=(1.0, 1.8, 2.5),
    horizons=(2.0, 3.5, 5.0),
) -> list[dict]:
    """对固定车辆/行人交汇轨迹运行 TTC 阈值组合。"""
    vehicle = _speed_reading(1, "vehicle", 3.0, 0.0, 0.0, 6.0)
    pedestrian = _speed_reading(2, "pedestrian", 3.0, 6.0, 0.0, 0.0, "pedestrian")
    rows = []
    for critical, horizon in product(critical_thresholds, horizons):
        # 冲突分析器契约要求 ttc_critical_s <= ttc_caution_s <= horizon_s，跳过不合法组合。
        if critical > horizon:
            continue
        caution = max(critical, min(3.5, horizon))
        analyzer = ConflictAnalyzer(
            ttc_critical_s=critical, ttc_caution_s=caution, horizon_s=horizon
        )
        result = analyzer.evaluate_pair(vehicle, pedestrian)
        rows.append({
            "ttc_critical_s": critical,
            "horizon_s": horizon,
            "ttc_s": round(result.ttc_seconds, 6) if result else None,
            "min_distance_m": round(result.min_distance_m, 6) if result else None,
            "risk_level": result.risk_level.name if result else RiskLevel.LEVEL_0_SAFE.name,
        })
    return rows


def _estimate_lag_seconds(true_vals: list[float], meas_vals: list[float], dt: float) -> float:
    """用离散时间偏移误差最小化估计测速相对真值的平均相位滞后 (秒)。"""
    n = len(true_vals)
    if n < 12:
        return 0.0
    best_shift = 0
    best_mse = float("inf")
    max_shift = min(12, n // 4)
    for shift in range(-max_shift, max_shift + 1):
        errs = []
        for idx in range(max_shift, n - max_shift):
            ref_idx = idx - shift
            errs.append((meas_vals[idx] - true_vals[ref_idx]) ** 2)
        mse = sum(errs) / len(errs)
        if mse < best_mse:
            best_mse = mse
            best_shift = shift
    return round(best_shift * dt, 4)


def _compare_speed_estimators() -> dict:
    """在真实单应性透视噪声下的加速-巡航-制动轨迹上对比 EMA 与 Kalman。"""
    calibrator = RoadCalibrator(
        image_points=[(420.0, 260.0), (860.0, 260.0), (1180.0, 680.0), (100.0, 680.0)],
        width_m=6.0,
        length_m=24.0,
        name="ablation_perspective_road",
    )
    fps = 25.0
    dt = 1.0 / fps
    n_frames = 100
    rng = np.random.default_rng(42)

    y_pos = 1.0
    v_cur = 2.0
    true_speeds: list[float] = []
    ema_speeds: list[float] = []
    kalman_speeds: list[float] = []

    ema_est = SpeedEstimator(estimator_mode="ema")
    kalman_est = SpeedEstimator(estimator_mode="kalman")

    for i in range(n_frames):
        t = i * dt
        if t < 1.6:
            accel = 3.5
        elif t < 2.4:
            accel = 0.0
        else:
            accel = -3.2

        if i > 0:
            y_pos = y_pos + v_cur * dt + 0.5 * accel * dt * dt
            v_cur = max(0.5, v_cur + accel * dt)

        p_img = calibrator.H_inv @ np.array([3.0, y_pos, 1.0], dtype=np.float64)
        u_true = float(p_img[0] / p_img[2])
        v_true = float(p_img[1] / p_img[2])

        u_noisy = u_true + float(rng.normal(0.0, 2.0))
        v_noisy = v_true + float(rng.normal(0.0, 3.0))
        gx, gy = calibrator.pixel_to_ground(u_noisy, v_noisy)
        obs_cov = calibrator.ground_covariance_from_pixel(
            u_noisy, v_noisy, sigma_u_px=2.0, sigma_v_px=3.0
        )

        r_ema = ema_est.update(1, "vehicle", t, gx, gy, object_kind="car")
        r_kal = kalman_est.update(
            1, "vehicle", t, gx, gy, object_kind="car", obs_cov_m2=obs_cov
        )
        true_speeds.append(v_cur)
        ema_speeds.append(r_ema.speed_ms)
        kalman_speeds.append(r_kal.speed_ms)

    warmup = 8
    ref = true_speeds[warmup:]

    def _metrics(meas_all: list[float]) -> dict:
        meas = meas_all[warmup:]
        abs_errs = [abs(m - r) for m, r in zip(meas, ref)]
        sq_errs = [(m - r) ** 2 for m, r in zip(meas, ref)]
        return {
            "mae_ms": round(sum(abs_errs) / len(abs_errs), 6),
            "rmse_ms": round(math.sqrt(sum(sq_errs) / len(sq_errs)), 6),
            "max_error_ms": round(max(abs_errs), 6),
            "mean_lag_s": _estimate_lag_seconds(ref, meas, dt),
        }

    ema_m = _metrics(ema_speeds)
    kal_m = _metrics(kalman_speeds)
    return {
        "ema": ema_m,
        "kalman": kal_m,
        "kalman_improves_mae": kal_m["mae_ms"] < ema_m["mae_ms"],
    }


def _compare_conflict_models() -> list[dict]:
    """对比 CVM 圆包络与 OBB+SSM 在窄路并行、制动礼让与垂直横穿场景上的研判差异。"""
    cvm = ConflictAnalyzer(conflict_model="cvm_circle")
    obb = ConflictAnalyzer(conflict_model="obb_ssm")

    cases = [
        (
            "parallel_pass",
            _speed_reading(1, "vehicle", 3.0, 0.0, 0.0, 8.0, "car", heading_deg=90.0),
            _speed_reading(2, "pedestrian", 5.0, 8.0, 0.0, 1.0, "pedestrian", heading_deg=90.0),
        ),
        (
            "braking_yield",
            _speed_reading(
                1, "vehicle", 3.0, 0.0, 0.0, 4.8, "car", heading_deg=90.0, ay_ms2=-2.8
            ),
            _speed_reading(2, "pedestrian", 3.0, 8.5, 0.0, 0.0, "pedestrian"),
        ),
        (
            "perpendicular_crossing",
            _speed_reading(1, "vehicle", 3.0, 0.0, 0.0, 10.0, "car", heading_deg=90.0),
            _speed_reading(2, "pedestrian", 5.0, 10.0, -2.0, 0.0, "pedestrian", heading_deg=180.0),
        ),
    ]

    rows = []
    for name, veh, ped in cases:
        res_cvm = cvm.evaluate_pair(veh, ped)
        res_obb = obb.evaluate_pair(veh, ped)
        rows.append({
            "scenario": name,
            "cvm_circle": {
                "risk_level": (
                    res_cvm.risk_level.name if res_cvm is not None else RiskLevel.LEVEL_0_SAFE.name
                ),
                "ttc_s": (
                    round(res_cvm.ttc_seconds, 6)
                    if res_cvm is not None and res_cvm.ttc_seconds is not None
                    else None
                ),
            },
            "obb_ssm": {
                "risk_level": (
                    res_obb.risk_level.name if res_obb is not None else RiskLevel.LEVEL_0_SAFE.name
                ),
                "ttc_s": (
                    round(res_obb.ttc_seconds, 6)
                    if res_obb is not None and res_obb.ttc_seconds is not None
                    else None
                ),
                "mttc_s": (
                    round(res_obb.mttc_seconds, 6)
                    if res_obb is not None and res_obb.mttc_seconds is not None
                    else None
                ),
                "drac_ms2": (
                    round(res_obb.drac_ms2, 6)
                    if res_obb is not None and res_obb.drac_ms2 is not None
                    else None
                ),
                "collision_prob": (
                    round(res_obb.collision_prob, 6)
                    if res_obb is not None and res_obb.collision_prob is not None
                    else None
                ),
            },
        })
    return rows


def _compare_tracklet_stitching() -> dict:
    """在 0.6s 遮挡导致底层 ID 切换的场景下对比未缝合与 TrackletStitcher 连续性。"""
    fps = 25.0
    dt = 1.0 / fps
    total_frames = 75
    occ_start, occ_end = 25, 40  # 15 帧 = 0.60s 遮挡
    v_true = 8.0

    def _run_mode(use_stitcher: bool) -> dict:
        est = SpeedEstimator(estimator_mode="kalman")
        analyzer = ConflictAnalyzer(conflict_model="obb_ssm")
        evt_tracker = ConflictEventTracker(confirm_frames=3, cooldown_frames=20)
        stitcher = TrackletStitcher() if use_stitcher else None

        seen_veh_ids: set[int] = set()
        abs_errors: list[float] = []
        last_seen: dict[int, int] = {}

        for idx in range(total_frames):
            t = idx * dt
            veh_present = not (occ_start <= idx < occ_end)
            raw_obs = []
            if veh_present:
                raw_veh_id = 1 if idx < occ_start else 2
                y_veh = 2.0 + v_true * t
                raw_obs.append((raw_veh_id, "vehicle", "car", 3.0, y_veh))
            # 行人稳定站在前方路侧斑马线边缘，形成单次连续冲突场景
            raw_obs.append((10, "pedestrian", "pedestrian", 3.2, 24.0))

            if stitcher is not None:
                id_map, _ = stitcher.resolve_frame(raw_obs, t)
            else:
                id_map = {item[0]: item[0] for item in raw_obs}

            readings: list[SpeedReading] = []
            for raw_id, cat, kind, gx, gy in raw_obs:
                eff_id = id_map[raw_id]
                last_seen[eff_id] = idx
                r = est.update(eff_id, cat, t, gx, gy, object_kind=kind)
                if stitcher is not None:
                    stitcher.update_kinematics(eff_id, r, t)
                readings.append(r)
                if cat == "vehicle":
                    seen_veh_ids.add(eff_id)
                    if idx >= 8:
                        abs_errors.append(abs(r.speed_ms - v_true))

            if stitcher is not None:
                live_ids = stitcher.retained_canonical_ids(t)
            else:
                # 底层跟踪器在超过 5 帧 (0.2s) 失跟后清理速度状态
                live_ids = {tid for tid, f in last_seen.items() if idx - f <= 5}
            est.prune_stale_tracks(live_ids)

            conflicts = analyzer.analyze_frame(readings, timestamp_s=t)
            evt_tracker.update(conflicts, idx, t)

        evt_tracker.flush(reason="eof")
        events = evt_tracker.pop_finalized()
        mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
        out = {
            "unique_track_ids": len(seen_veh_ids),
            "alert_event_count": len(events),
            "speed_mae_ms": round(mae, 6),
        }
        if stitcher is not None:
            out["stitched_count"] = stitcher.stitched_track_count
        else:
            out["id_switches"] = max(0, len(seen_veh_ids) - 1)
        return out

    unstitched = _run_mode(False)
    stitched = _run_mode(True)
    preserved = (
        stitched["unique_track_ids"] == 1
        and stitched["alert_event_count"] == 1
        and stitched["stitched_count"] == 1
        and stitched["speed_mae_ms"] <= unstitched["speed_mae_ms"]
    )
    return {
        "unstitched": unstitched,
        "stitched": stitched,
        "stitching_preserved_continuity": preserved,
    }


def run_algorithm_comparison() -> dict:
    """运行 EMA vs Kalman、CVM vs OBB-SSM、未缝合 vs TrackletStitcher 综合对比实验。"""
    return {
        "speed_estimation": _compare_speed_estimators(),
        "conflict_detection": _compare_conflict_models(),
        "tracklet_stitching": _compare_tracklet_stitching(),
    }


def run_default_ablation() -> dict:
    return {
        "schema_version": 1,
        "description": "Deterministic synthetic trajectories; no detector or real-world accuracy claim.",
        "speed": run_speed_ablation(),
        "conflict": run_conflict_ablation(),
        "algorithm_comparison": run_algorithm_comparison(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行无视频的测速/TTC 参数消融")
    parser.add_argument("--output", help="输出 JSON 文件；不指定则打印")
    args = parser.parse_args(argv)
    # 必须在写出前拦截空白 --output：否则 "" 会静默回退打印到 stdout，
    # 而 "   " 经 Path() 解析后会在工作区尝试写出空白文件名。
    if args.output is not None and not args.output.strip():
        parser.error("--output 不能为空")
    result = run_default_ablation()
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
