"""无视频的确定性参数消融：用合成米制轨迹检查测速与 TTC 策略。"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from .conflict import ConflictAnalyzer, RiskLevel
from .speed import SpeedEstimator, SpeedReading


def _speed_reading(track_id, category, x, y, vx, vy, object_kind="motor_vehicle"):
    speed = (vx * vx + vy * vy) ** 0.5
    return SpeedReading(
        track_id=track_id, category=category, x_m=x, y_m=y,
        vx_ms=vx, vy_ms=vy, speed_ms=speed, speed_kmh=speed * 3.6,
        is_speeding=speed * 3.6 > 30.0, heading_deg=0.0,
        is_reliable=True, object_kind=object_kind,
    )


def run_speed_ablation(
    speeds_ms=(2.0, 5.0, 10.0),
    window_sizes=(4, 8, 12),
    ema_alphas=(0.2, 0.35, 0.6),
    fps=25.0,
    frames=40,
) -> list[dict]:
    """对匀速直线轨迹运行窗口/EMA 组合，返回末帧误差和可靠性。"""
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


def run_default_ablation() -> dict:
    return {
        "schema_version": 1,
        "description": "Deterministic synthetic trajectories; no detector or real-world accuracy claim.",
        "speed": run_speed_ablation(),
        "conflict": run_conflict_ablation(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行无视频的测速/TTC 参数消融")
    parser.add_argument("--output", help="输出 JSON 文件；不指定则打印")
    args = parser.parse_args(argv)
    result = run_default_ablation()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
