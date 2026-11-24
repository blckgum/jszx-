# -*- coding: utf-8 -*-
"""分阶段耗时基准统计 (analysis.benchmark) 与无界面实验图表生成 (analysis.charts) 单元测试。"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from analysis import (
    plot_ablation_charts,
    plot_run_timeline,
    profile_pipeline_stages,
    summarize_stage_timings,
)
from analysis.benchmark import (
    STAGE_NAMES,
    benchmark_to_markdown,
    main as bench_main,
)
from analysis.charts import (
    main as charts_main,
    plot_benchmark_breakdown,
    plot_mot_comparison,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _write_dummy_video(path: Path, frames: int = 8, width: int = 64, height: int = 64) -> Path:
    """在临时目录生成小型 MP4 测试视频。"""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
    assert writer.isOpened()
    try:
        for idx in range(frames):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :, 0] = (idx * 25) % 255
            writer.write(img)
    finally:
        writer.release()
    return path


def _mock_tracking_session():
    """构造无权重依赖的虚拟跟踪会话，避免冷跑测试访问外网下载模型。"""
    frame_idx = {"n": 0}

    def _process(frame, timestamp_s, time_kind):
        n = frame_idx["n"]
        frame_idx["n"] += 1
        veh = mock.Mock()
        veh.track_id = 1
        veh.category = "vehicle"
        veh.class_name = "car"
        veh.anchor_xy = (30.0, 15.0 + n * 2.0)
        veh.bbox_xyxy = (20.0, 5.0 + n * 2.0, 40.0, 15.0 + n * 2.0)
        ped = mock.Mock()
        ped.track_id = 2
        ped.category = "pedestrian"
        ped.class_name = "person"
        ped.anchor_xy = (32.0, 45.0)
        ped.bbox_xyxy = (28.0, 30.0, 36.0, 45.0)
        rec = mock.Mock()
        rec.tracks = [veh, ped]
        return rec

    session = mock.Mock()
    session.process.side_effect = _process
    session.retained_ids.return_value = {1, 2}
    return session


class TestBenchmark(unittest.TestCase):
    def test_lazy_package_exports_and_cli_help_under_werror(self):
        import analysis.benchmark as bench_mod
        import analysis.charts as charts_mod

        self.assertIs(profile_pipeline_stages, bench_mod.profile_pipeline_stages)
        self.assertIs(summarize_stage_timings, bench_mod.summarize_stage_timings)
        self.assertIs(plot_ablation_charts, charts_mod.plot_ablation_charts)
        self.assertIs(plot_run_timeline, charts_mod.plot_run_timeline)

        # 验证以 -m 独立执行 CLI 时不会因提前导入子模块触发 RuntimeWarning
        repo_root = Path(__file__).resolve().parents[1]
        for mod in ("analysis.benchmark", "analysis.charts"):
            proc = subprocess.run(
                [sys.executable, "-W", "error::RuntimeWarning", "-m", mod, "--help"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"{mod} --help stderr: {proc.stderr}")

    def test_summarize_stage_timings_list_and_mapping_and_empty(self):
        samples = [
            {
                "decode": 1.0,
                "detect_track": 10.0,
                "geometry_speed": 2.0,
                "conflict_events": 1.0,
                "render_write": 6.0,
            },
            {
                "decode": 3.0,
                "detect_track": 20.0,
                "geometry_speed": 4.0,
                "conflict_events": 3.0,
                "render_write": 10.0,
            },
        ]
        summary_list = summarize_stage_timings(samples)
        self.assertEqual(summary_list["frames_profiled"], 2)
        self.assertAlmostEqual(summary_list["stages"]["decode"]["mean_ms"], 2.0, places=3)
        self.assertAlmostEqual(summary_list["stages"]["total_frame"]["mean_ms"], 30.0, places=3)
        self.assertAlmostEqual(summary_list["stages"]["total_frame"]["p50_ms"], 30.0, places=3)
        self.assertAlmostEqual(summary_list["stages"]["total_frame"]["max_ms"], 40.0, places=3)
        self.assertAlmostEqual(summary_list["end_to_end_fps"], 1000.0 / 30.0, places=2)

        # 列式字典输入应得到完全一致的统计结果
        col_samples = {
            "decode": [1.0, 3.0],
            "detect_track": [10.0, 20.0],
            "geometry_speed": [2.0, 4.0],
            "conflict_events": [1.0, 3.0],
            "render_write": [6.0, 10.0],
        }
        summary_dict = summarize_stage_timings(col_samples)
        self.assertEqual(summary_list, summary_dict)

        empty_res = summarize_stage_timings([])
        self.assertEqual(empty_res["frames_profiled"], 0)
        self.assertEqual(empty_res["end_to_end_fps"], 0.0)
        for name in STAGE_NAMES:
            self.assertEqual(empty_res["stages"][name]["mean_ms"], 0.0)

    def test_summarize_stage_timings_rejects_invalid_inputs(self):
        for bad in (
            None,
            123,
            "bad",
            [{"decode": True}],
            [{"decode": float("nan")}],
            [{"decode": -0.5}],
            [{"unknown_stage": 1.0}],
            {"decode": [1.0, 2.0], "detect_track": [1.0]},
            {"decode": [np.bool_(True)]},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    summarize_stage_timings(bad)

    def test_profile_pipeline_stages_mocked_and_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = _write_dummy_video(root / "clip.mp4", frames=6)
            fake_session = _mock_tracking_session()

            with mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session):
                report = profile_pipeline_stages(
                    video,
                    demo=True,
                    max_frames=4,
                    speed_model="kalman",
                    conflict_model="obb_ssm",
                    enable_stitching=True,
                )

            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["frames_profiled"], 4)
            self.assertGreaterEqual(report["model_load_ms"], 0.0)
            self.assertGreater(report["end_to_end_fps"], 0.0)
            self.assertEqual(set(report["stages"].keys()), set(STAGE_NAMES))

            md = benchmark_to_markdown(report)
            self.assertIn("# Pipeline Stage Benchmark Report", md)
            self.assertIn("| detect_track |", md)
            self.assertIn("| total_frame |", md)

            # 验证 max_frames=0 与无渲染模式同样输出合法 schema_version: 1 报告
            with mock.patch("analysis.pipeline.TrackingSession", return_value=_mock_tracking_session()):
                zero_report = profile_pipeline_stages(
                    video, demo=True, max_frames=0, render_video=False
                )
            self.assertEqual(zero_report["frames_profiled"], 0)
            self.assertEqual(zero_report["end_to_end_fps"], 0.0)

            # CLI 输出 JSON 与 Markdown
            out_json = root / "bench.json"
            out_md = root / "bench.md"
            with mock.patch("analysis.pipeline.TrackingSession", return_value=_mock_tracking_session()):
                rc_json = bench_main([
                    "--source", str(video),
                    "--demo",
                    "--max-frames", "4",
                    "--speed-model", "ema",
                    "--conflict-model", "cvm_circle",
                    "--output", str(out_json),
                ])
            self.assertEqual(rc_json, 0)
            parsed = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema_version"], 1)
            self.assertEqual(parsed["frames_profiled"], 4)

            with mock.patch("analysis.pipeline.TrackingSession", return_value=_mock_tracking_session()):
                rc_md = bench_main([
                    "--source", str(video),
                    "--demo",
                    "--max-frames", "3",
                    "--markdown",
                    "--output", str(out_md),
                ])
            self.assertEqual(rc_md, 0)
            self.assertIn("| total_frame |", out_md.read_text(encoding="utf-8"))

    def test_benchmark_rejects_blank_paths_and_invalid_args(self):
        for bad_path in ("", "   ", Path("   "), True):
            with self.subTest(bad_path=bad_path):
                with self.assertRaises(ValueError):
                    profile_pipeline_stages(bad_path, demo=True)

        with self.assertRaises(ValueError):
            profile_pipeline_stages(None, demo=False)

        for bad_argv in (
            ["--source", "   ", "--demo"],
            ["--source", "x.mp4", "--calib", "   "],
            ["--source", "x.mp4", "--demo", "--output", "   "],
            ["--demo", "--max-frames", "-1"],
            [],
        ):
            with self.subTest(bad_argv=bad_argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        bench_main(bad_argv)


class TestCharts(unittest.TestCase):
    def _assert_valid_png(self, path_str: str) -> None:
        p = Path(path_str)
        self.assertTrue(p.is_file(), msg=f"图表文件不存在: {p}")
        data = p.read_bytes()
        self.assertGreater(len(data), 100)
        self.assertTrue(data.startswith(PNG_MAGIC), msg=f"非合法 PNG 文件头: {p}")

    def test_matplotlib_uses_agg_and_four_chart_types_emit_zero_warnings(self):
        import matplotlib

        self.assertEqual(matplotlib.get_backend().lower(), "agg")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            charts_dir = root / "charts"
            run_dir = root / "run_01"
            run_dir.mkdir(parents=True)

            # 1) 准备真实运行产物 (speeds.csv + events.jsonl + conflicts.jsonl + configuration.json)
            (run_dir / "configuration.json").write_text(
                json.dumps({"speed_limit_kmh": 30.0}), encoding="utf-8"
            )
            with (run_dir / "speeds.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "schema_version", "frame_idx", "timestamp_s", "track_id", "category",
                    "x_m", "y_m", "speed_kmh", "is_speeding", "risk_level", "ttc_s",
                    "event_id", "is_reliable", "is_stationary",
                ])
                for idx in range(10):
                    t = idx * 0.1
                    spd1 = 22.0 + idx * 1.8
                    lvl = "LEVEL_3_CRITICAL" if 4 <= idx <= 6 else "LEVEL_0_SAFE"
                    ttc = "1.2" if 4 <= idx <= 6 else ""
                    eid = "1" if 4 <= idx <= 6 else ""
                    w.writerow([1, idx, f"{t:.2f}", 1, "vehicle", "3.0", f"{idx:.1f}", f"{spd1:.1f}", int(spd1 > 30), lvl, ttc, eid, 1, 0])
                    w.writerow([1, idx, f"{t:.2f}", 2, "pedestrian", "3.2", "10.0", "4.0", 0, "LEVEL_0_SAFE", "", "", 1, 0])

            (run_dir / "events.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "event_id": 1, "kind": "conflict",
                        "start_timestamp_s": 0.4, "end_timestamp_s": 0.6,
                        "peak_risk_level": 3, "min_ttc_seconds": 1.2,
                    }),
                    json.dumps({
                        "event_id": 2, "kind": "speeding",
                        "start_timestamp_s": 0.5, "end_timestamp_s": 0.9,
                        "peak_risk_level": 1, "min_ttc_seconds": None,
                    }),
                ]) + "\n",
                encoding="utf-8",
            )
            (run_dir / "conflicts.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "frame_index": idx, "timestamp_s": idx * 0.1, "event_id": (1 if 4 <= idx <= 6 else None),
                        "assessment": {"risk_level": (3 if 4 <= idx <= 6 else 0), "ttc_seconds": (1.2 if 4 <= idx <= 6 else None)},
                    })
                    for idx in range(10)
                ]) + "\n",
                encoding="utf-8",
            )

            # 2) 准备包含 EMA / Kalman / CTRV 与冲突指标的消融数据
            ablation_data = {
                "schema_version": 1,
                "algorithm_comparison": {
                    "speed_estimation": {
                        "ema": {"mae_ms": 0.65, "rmse_ms": 0.85, "max_error_ms": 1.4},
                        "kalman": {"mae_ms": 0.35, "rmse_ms": 0.48, "max_error_ms": 0.9},
                    },
                    "conflict_detection": [
                        {
                            "scenario": "parallel_pass",
                            "cvm_circle": {"risk_level": "LEVEL_2_CAUTION", "ttc_s": 2.1},
                            "obb_ssm": {
                                "risk_level": "LEVEL_0_SAFE", "ttc_s": None,
                                "mttc_s": None, "drac_ms2": 0.0, "collision_prob": 0.05,
                            },
                        },
                        {
                            "scenario": "perpendicular_crossing",
                            "cvm_circle": {"risk_level": "LEVEL_3_CRITICAL", "ttc_s": 1.0},
                            "obb_ssm": {
                                "risk_level": "LEVEL_3_CRITICAL", "ttc_s": 0.9,
                                "mttc_s": 0.85, "drac_ms2": 4.2, "collision_prob": 0.92,
                            },
                        },
                    ],
                },
                "turn_speed_ablation": [
                    {"estimator_mode": "ema", "target_speed_kmh": 25.0, "turn_rate_rad_s": 0.4, "mean_abs_error_kmh": 2.8, "max_abs_error_kmh": 5.1},
                    {"estimator_mode": "kalman", "target_speed_kmh": 25.0, "turn_rate_rad_s": 0.4, "mean_abs_error_kmh": 1.6, "max_abs_error_kmh": 3.0},
                    {"estimator_mode": "ctrv", "target_speed_kmh": 25.0, "turn_rate_rad_s": 0.4, "mean_abs_error_kmh": 0.7, "max_abs_error_kmh": 1.4},
                ],
            }

            # 3) 准备 MOT 评测对比与 Benchmark 数据
            mot_base = {"mota": 0.78, "idf1": 0.72, "hota": 0.68, "id_switches": 6}
            mot_stitch = {"mota": 0.89, "idf1": 0.91, "hota": 0.85, "id_switches": 1}
            bench_data = summarize_stage_timings([
                {"decode": 1.2, "detect_track": 18.5, "geometry_speed": 1.8, "conflict_events": 1.5, "render_write": 7.0}
                for _ in range(5)
            ])
            bench_data["model_load_ms"] = 210.0

            # 在严格将 RuntimeWarning 与 UserWarning 均视为错误的上下文中出图，确保 CI 零字体/除零告警
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                warnings.simplefilter("error", UserWarning)

                ablation_pngs = plot_ablation_charts(ablation_data, charts_dir)
                self.assertEqual(len(ablation_pngs), 2)
                for p in ablation_pngs:
                    self._assert_valid_png(p)

                timeline_pngs = plot_run_timeline(run_dir, charts_dir)
                self.assertEqual(len(timeline_pngs), 2)
                for p in timeline_pngs:
                    self._assert_valid_png(p)

                mot_png = plot_mot_comparison(
                    {"Baseline": mot_base, "Stitched": mot_stitch},
                    charts_dir / "mot.png",
                )
                self._assert_valid_png(mot_png)

                bench_png = plot_benchmark_breakdown(bench_data, charts_dir / "bench.png")
                self._assert_valid_png(bench_png)

                # 验证全零耗时基准不会在绘制饼图时触发 divide-by-zero RuntimeWarning
                zero_bench = summarize_stage_timings([])
                zero_png = plot_benchmark_breakdown(zero_bench, charts_dir / "bench_zero.png")
                self._assert_valid_png(zero_png)

            # 4) 验证 CLI 入口联调写出 PNG
            ablation_json = root / "ablation.json"
            bench_json = root / "bench.json"
            mot_a_json = root / "mot_a.json"
            mot_b_json = root / "mot_b.json"
            ablation_json.write_text(json.dumps(ablation_data, allow_nan=False), encoding="utf-8")
            bench_json.write_text(json.dumps(bench_data, allow_nan=False), encoding="utf-8")
            mot_a_json.write_text(json.dumps(mot_base, allow_nan=False), encoding="utf-8")
            mot_b_json.write_text(json.dumps(mot_stitch, allow_nan=False), encoding="utf-8")

            cli_out = root / "cli_charts"
            stdout_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf):
                rc = charts_main([
                    "--ablation", str(ablation_json),
                    "--run", str(run_dir),
                    "--benchmark", str(bench_json),
                    "--mot", f"baseline={mot_a_json}",
                    "--mot", f"stitched={mot_b_json}",
                    "--output-dir", str(cli_out),
                ])
            self.assertEqual(rc, 0)
            manifest = json.loads(stdout_buf.getvalue())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(len(manifest["generated_files"]), 6)
            for p in manifest["generated_files"]:
                self._assert_valid_png(p)

    def test_charts_reject_blank_paths_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for bad_path in ("", "   ", Path("   "), None, True):
                with self.subTest(bad_path=bad_path):
                    with self.assertRaises(ValueError):
                        plot_ablation_charts({"speed": [{"absolute_error_ms": 0.1}]}, bad_path)
                    with self.assertRaises(ValueError):
                        plot_run_timeline(bad_path, root)
                    with self.assertRaises(ValueError):
                        plot_mot_comparison({"a": {"mota": 1.0, "idf1": 1.0, "hota": 1.0, "id_switches": 0}}, bad_path)
                    with self.assertRaises(ValueError):
                        plot_benchmark_breakdown({"stages": {"decode": {"mean_ms": 1.0}}}, bad_path)

            bad_json = root / "bad_nan.json"
            bad_json.write_text('{"stages": {"decode": {"mean_ms": NaN}}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                plot_benchmark_breakdown(bad_json, root / "out.png")

            for bad_argv in (
                ["--output-dir", "   ", "--ablation", "a.json"],
                ["--output-dir", str(root)],
                ["--output-dir", str(root), "--ablation", "   "],
                ["--output-dir", str(root), "--run", ""],
                ["--output-dir", str(root), "--benchmark", "   "],
                ["--output-dir", str(root), "--mot", "no_equals"],
                ["--output-dir", str(root), "--mot", "label=   "],
            ):
                with self.subTest(bad_argv=bad_argv):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            charts_main(bad_argv)


if __name__ == "__main__":
    unittest.main()
