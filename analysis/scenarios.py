"""确定性合成冲突场景库：只验证分析规则，不模拟检测器或真实视频。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .conflict import ConflictAnalyzer, RiskLevel
from .speed import SpeedReading


def _reading(track_id, category, x, y, vx, vy, *, object_kind="motor_vehicle",
             reliable=True, stationary=False, speed_kmh=None):
    speed_ms = (vx * vx + vy * vy) ** 0.5
    return SpeedReading(
        track_id=track_id, category=category, object_kind=object_kind,
        x_m=x, y_m=y, vx_ms=vx, vy_ms=vy, speed_ms=speed_ms,
        speed_kmh=speed_ms * 3.6 if speed_kmh is None else speed_kmh,
        is_speeding=speed_ms * 3.6 > 30.0, heading_deg=0.0,
        is_reliable=reliable, is_stationary=stationary,
    )


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    description: str
    readings: tuple[SpeedReading, ...]
    expected_levels: tuple[str, ...]


def default_scenarios() -> tuple[SyntheticScenario, ...]:
    return (
        SyntheticScenario(
            "critical_crossing", "汽车与横穿行人预计相撞",
            (_reading(1, "vehicle", 3, 0, 0, 10, object_kind="car"),
             _reading(2, "pedestrian", 5, 10, -2, 0, object_kind="pedestrian")),
            (RiskLevel.LEVEL_3_CRITICAL.name,),
        ),
        SyntheticScenario(
            "speeding_only", "无行人时独立超速",
            (_reading(1, "vehicle", 3, 0, 0, 10, object_kind="car"),),
            (RiskLevel.LEVEL_1_SPEEDING.name,),
        ),
        SyntheticScenario(
            "diverging_safe", "车辆和行人远离彼此",
            (_reading(1, "vehicle", 3, 10, 0, 8, object_kind="car"),
             _reading(2, "pedestrian", 3, 2, 0, -1, object_kind="pedestrian")),
            (RiskLevel.LEVEL_0_SAFE.name,),
        ),
        SyntheticScenario(
            "parked_safe", "已确认静止车辆与接近行人",
            (_reading(1, "vehicle", 3, 5, 0, 0, object_kind="car", stationary=True),
             _reading(2, "pedestrian", 7, 5, -2, 0, object_kind="pedestrian")),
            (RiskLevel.LEVEL_0_SAFE.name,),
        ),
        SyntheticScenario(
            "bicycle_caution", "自行车使用较小几何包络",
            (_reading(1, "vehicle", 3, 0, 0, 6, object_kind="bicycle"),
             _reading(2, "pedestrian", 5.5, 6, 0, 0, object_kind="pedestrian")),
            (RiskLevel.LEVEL_2_CAUTION.name,),
        ),
    )


def run_scenarios(scenarios=None, analyzer=None) -> list[dict]:
    analyzer = analyzer or ConflictAnalyzer()
    results = []
    for scenario in scenarios or default_scenarios():
        assessments = analyzer.analyze_frame(list(scenario.readings))
        actual = tuple(sorted(item.risk_level.name for item in assessments))
        expected = tuple(sorted(scenario.expected_levels))
        results.append({
            "name": scenario.name,
            "description": scenario.description,
            "expected_levels": list(expected),
            "actual_levels": list(actual),
            "passed": actual == expected,
            "assessment_count": len(assessments),
        })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行无视频的确定性冲突场景回归")
    parser.add_argument("--output", help="输出 JSON 文件；不指定则打印")
    args = parser.parse_args(argv)
    payload = {
        "schema_version": 1,
        "description": "合成规则场景，不代表真实校园检测或测速准确率。",
        "results": run_scenarios(),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
