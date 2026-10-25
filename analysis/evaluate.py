"""对一次 analysis 运行产物做无真值的描述性评价。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是有效 JSON: {exc}") from None
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _finite(values):
    return [float(value) for value in values if value not in (None, "") and math.isfinite(float(value))]


def _risk_name(value) -> str:
    if value is None or value == "":
        return "LEVEL_0_SAFE"
    if isinstance(value, str):
        return value if value.startswith("LEVEL_") else f"LEVEL_{value}"
    return f"LEVEL_{int(value)}_SAFE" if int(value) == 0 else f"LEVEL_{int(value)}"


def evaluate_run(run_dir: str | Path) -> dict:
    """读取运行产物，返回不依赖人工真值的描述性指标。"""
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"运行目录不存在: {root}")

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

    return {
        "schema_version": 1,
        "run_dir": str(root),
        "has_ground_truth": False,
        "note": "描述性运行评价，无人工真值，不声称 Precision/Recall/MAE 等精度指标。",
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


def to_markdown(result: dict) -> str:
    """将评价结果转换为适合实验记录的简短 Markdown。"""
    frames = result["frames"]
    events = result["events"]
    speed = result["speed_rows"]["speed_kmh"]
    lines = [
        f"# Analysis run evaluation\n",
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
    lines.extend([
        "",
        "> This is descriptive output without ground truth; it is not an accuracy evaluation.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评价一次 analysis 运行产物（不需要视频或人工真值）")
    parser.add_argument("--run", required=True, help="runs/analysis/run-* 目录")
    parser.add_argument("--output", help="输出 JSON/Markdown 文件；不指定则打印 JSON")
    parser.add_argument("--markdown", action="store_true", help="输出简短 Markdown 而不是 JSON")
    args = parser.parse_args(argv)
    result = evaluate_run(args.run)
    content = to_markdown(result) if args.markdown else json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
    else:
        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
