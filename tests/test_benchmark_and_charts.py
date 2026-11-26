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
    CJK_FONT_CANDIDATES,
    FALLBACK_FONT,
    configure_cjk_fonts,
    detect_available_cjk_fonts,
    main as charts_main,
    plot_benchmark_breakdown,
    plot_mot_comparison,
)
from analysis.evaluate import to_markdown as evaluate_to_markdown

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
            self.assertEqual(report["frame_processing_fps"], report["end_to_end_fps"])
            self.assertGreater(report["wall_fps"], 0.0)
            self.assertEqual(report["steady_state"]["warmup_frames_excluded"], 1)
            self.assertEqual(report["steady_state"]["frames_evaluated"], 3)
            self.assertGreater(report["steady_state"]["steady_state_fps"], 0.0)
            self.assertEqual(set(report["stages"].keys()), set(STAGE_NAMES))
            self.assertEqual(set(report["steady_state"]["stages"].keys()), set(STAGE_NAMES))

            md = benchmark_to_markdown(report)
            self.assertIn("# Pipeline Stage Benchmark Report", md)
            self.assertIn("Frame processing throughput", md)
            self.assertIn("Wall-clock throughput", md)
            self.assertIn("Steady-state throughput", md)
            self.assertIn("| detect_track |", md)
            self.assertIn("| total_frame |", md)

            # 验证 max_frames=0 与无渲染模式同样输出合法 schema_version: 1 报告
            with mock.patch("analysis.pipeline.TrackingSession", return_value=_mock_tracking_session()):
                zero_report = profile_pipeline_stages(
                    video, demo=True, max_frames=0, render_video=False
                )
            self.assertEqual(zero_report["frames_profiled"], 0)
            self.assertEqual(zero_report["end_to_end_fps"], 0.0)
            self.assertEqual(zero_report["frame_processing_fps"], 0.0)
            self.assertEqual(zero_report["wall_fps"], 0.0)
            self.assertEqual(zero_report["steady_state"]["warmup_frames_excluded"], 0)
            self.assertEqual(zero_report["steady_state"]["frames_evaluated"], 0)

            # 验证 max_frames=1 时安全不丢弃唯一帧 (warmup_frames_excluded == 0)
            with mock.patch("analysis.pipeline.TrackingSession", return_value=_mock_tracking_session()):
                one_report = profile_pipeline_stages(
                    video, demo=True, max_frames=1, render_video=False
                )
            self.assertEqual(one_report["frames_profiled"], 1)
            self.assertEqual(one_report["steady_state"]["warmup_frames_excluded"], 0)
            self.assertEqual(one_report["steady_state"]["frames_evaluated"], 1)
            self.assertEqual(one_report["steady_state"]["steady_state_fps"], one_report["frame_processing_fps"])

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

    def test_summarize_stage_timings_steady_state_excludes_frame0_warmup(self):
        # 帧 0 含冷启动抖动 (200ms)，帧 1..4 为稳定态 (每帧 20ms -> 50 FPS)
        samples = [
            {"decode": 10.0, "detect_track": 170.0, "geometry_speed": 5.0, "conflict_events": 5.0, "render_write": 10.0},
            {"decode": 2.0, "detect_track": 12.0, "geometry_speed": 1.0, "conflict_events": 1.0, "render_write": 4.0},
            {"decode": 2.0, "detect_track": 12.0, "geometry_speed": 1.0, "conflict_events": 1.0, "render_write": 4.0},
            {"decode": 2.0, "detect_track": 12.0, "geometry_speed": 1.0, "conflict_events": 1.0, "render_write": 4.0},
            {"decode": 2.0, "detect_track": 12.0, "geometry_speed": 1.0, "conflict_events": 1.0, "render_write": 4.0},
        ]
        summary = summarize_stage_timings(samples)
        self.assertEqual(summary["frames_profiled"], 5)
        self.assertEqual(summary["frame_processing_fps"], summary["end_to_end_fps"])
        # 总耗时 = 200 + 80 = 280 ms -> 5000/280 = 17.857 FPS
        self.assertAlmostEqual(summary["frame_processing_fps"], 17.857, places=3)

        steady = summary["steady_state"]
        self.assertEqual(steady["warmup_frames_excluded"], 1)
        self.assertEqual(steady["frames_evaluated"], 4)
        self.assertAlmostEqual(steady["steady_state_fps"], 50.0, places=3)
        self.assertAlmostEqual(steady["stages"]["detect_track"]["mean_ms"], 12.0, places=3)
        self.assertAlmostEqual(steady["stages"]["detect_track"]["p50_ms"], 12.0, places=3)
        self.assertAlmostEqual(steady["stages"]["detect_track"]["p95_ms"], 12.0, places=3)
        self.assertAlmostEqual(steady["stages"]["detect_track"]["max_ms"], 12.0, places=3)
        self.assertAlmostEqual(steady["stages"]["total_frame"]["mean_ms"], 20.0, places=3)

        # 单帧边界：不剔除唯一帧
        single = summarize_stage_timings([samples[1]])
        self.assertEqual(single["steady_state"]["warmup_frames_excluded"], 0)
        self.assertEqual(single["steady_state"]["frames_evaluated"], 1)
        self.assertEqual(single["steady_state"]["steady_state_fps"], single["frame_processing_fps"])
        self.assertAlmostEqual(single["steady_state"]["stages"]["total_frame"]["mean_ms"], 20.0, places=3)

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

    def test_cjk_font_detection_priority_and_headless_fallback(self):
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt

        self.assertEqual(CJK_FONT_CANDIDATES[:4], (
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
            "Noto Sans CJK SC",
        ))
        self.assertEqual(FALLBACK_FONT, "DejaVu Sans")

        # 模拟仅安装部分 Linux CJK 字体的环境，验证优先级排序与回退字体保留
        linux_fonts = [
            mock.Mock(name="font_dejavu"),
            mock.Mock(name="font_wqy"),
            mock.Mock(name="font_noto"),
        ]
        linux_fonts[0].name = "DejaVu Sans"
        linux_fonts[1].name = "WenQuanYi Micro Hei"
        linux_fonts[2].name = "Noto Sans CJK SC"
        with mock.patch.object(fm.fontManager, "ttflist", linux_fonts):
            detected = detect_available_cjk_fonts()
            self.assertEqual(detected, ["Noto Sans CJK SC", "WenQuanYi Micro Hei"])
            configured = configure_cjk_fonts()
            self.assertEqual(configured, ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"])
            self.assertFalse(plt.rcParams["axes.unicode_minus"])

        # 模拟无任何中文字体的无界面 Linux CI（仅保留真实 DejaVu Sans FontEntry）：
        # 出图必须零 RuntimeWarning 与零 UserWarning
        dejavu_only = [f for f in fm.fontManager.ttflist if getattr(f, "name", "") == "DejaVu Sans"]
        self.assertTrue(dejavu_only, msg="Matplotlib 内置 DejaVu Sans 字体缺失")
        bench_data = summarize_stage_timings([
            {"decode": 1.0, "detect_track": 12.0, "geometry_speed": 0.6, "conflict_events": 0.5, "render_write": 2.5}
        ])
        ablation_sample = {
            "algorithm_comparison": {
                "speed_estimation": {
                    "ema": {"mae_ms": 0.58, "rmse_ms": 0.68},
                    "kalman": {"mae_ms": 0.36, "rmse_ms": 0.46},
                },
                "conflict_detection": [
                    {
                        "scenario": "parallel_pass",
                        "cvm_circle": {"risk_level": "LEVEL_3_CRITICAL"},
                        "obb_ssm": {"risk_level": "LEVEL_0_SAFE", "drac_ms2": None, "collision_prob": None},
                    },
                ],
            },
        }
        cache_clear = getattr(getattr(fm.fontManager, "_findfont_cached", None), "cache_clear", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            try:
                with mock.patch.object(fm.fontManager, "ttflist", dejavu_only):
                    if callable(cache_clear):
                        cache_clear()
                    self.assertEqual(detect_available_cjk_fonts(), [])
                    self.assertEqual(configure_cjk_fonts(), ["DejaVu Sans"])
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        warnings.simplefilter("error", UserWarning)
                        out_paths = plot_ablation_charts(ablation_sample, root)
                        bench_png = plot_benchmark_breakdown(bench_data, root / "bench_headless.png")
            finally:
                if callable(cache_clear):
                    cache_clear()
                configure_cjk_fonts()
            for p in [*out_paths, bench_png]:
                self._assert_valid_png(p)

    def test_chinese_chart_labels_rendered_across_chart_types(self):
        import analysis.charts as charts_mod

        captured_texts: dict[str, str] = {}
        orig_save = charts_mod._save_figure

        def _capture_save(fig, out_path):
            parts: list[str] = []
            if fig._suptitle is not None:
                parts.append(fig._suptitle.get_text())
            for ax in fig.axes:
                parts.append(ax.get_title())
                parts.append(ax.get_xlabel())
                parts.append(ax.get_ylabel())
                parts.extend(t.get_text() for t in ax.get_xticklabels())
                parts.extend(t.get_text() for t in ax.get_yticklabels())
                leg = ax.get_legend()
                if leg is not None:
                    parts.extend(t.get_text() for t in leg.get_texts())
            captured_texts[Path(out_path).name] = "\n".join(parts)
            return orig_save(fig, out_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run_cn"
            run_dir.mkdir(parents=True)
            (run_dir / "configuration.json").write_text(json.dumps({"speed_limit_kmh": 30.0}), encoding="utf-8")
            with (run_dir / "speeds.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_s", "track_id", "category", "speed_kmh"])
                w.writerow(["0.0", "1", "vehicle", "22.5"])
                w.writerow(["0.5", "1", "vehicle", "33.0"])
            (run_dir / "events.jsonl").write_text(
                json.dumps({
                    "event_id": 1, "kind": "conflict",
                    "start_timestamp_s": 0.2, "end_timestamp_s": 0.5, "peak_risk_level": 3,
                }) + "\n",
                encoding="utf-8",
            )
            (run_dir / "conflicts.jsonl").write_text(
                json.dumps({
                    "frame_index": 0, "timestamp_s": 0.0,
                    "assessment": {"risk_level": 0, "ttc_seconds": None},
                }) + "\n"
                + json.dumps({
                    "frame_index": 1, "timestamp_s": 0.5,
                    "assessment": {"risk_level": 3, "ttc_seconds": 0.9},
                }) + "\n",
                encoding="utf-8",
            )

            ablation_data = {
                "algorithm_comparison": {
                    "speed_estimation": {
                        "ema": {"mae_ms": 0.58, "rmse_ms": 0.68},
                        "kalman": {"mae_ms": 0.36, "rmse_ms": 0.46},
                    },
                    "conflict_detection": [
                        {
                            "scenario": "parallel_pass",
                            "cvm_circle": {"risk_level": "LEVEL_3_CRITICAL"},
                            "obb_ssm": {"risk_level": "LEVEL_0_SAFE", "drac_ms2": None, "collision_prob": None},
                        },
                        {
                            "scenario": "braking_yield",
                            "cvm_circle": {"risk_level": "LEVEL_3_CRITICAL"},
                            "obb_ssm": {"risk_level": "LEVEL_2_CAUTION", "drac_ms2": 1.97, "collision_prob": 0.0},
                        },
                        {
                            "scenario": "perpendicular_crossing",
                            "cvm_circle": {"risk_level": "LEVEL_3_CRITICAL"},
                            "obb_ssm": {"risk_level": "LEVEL_3_CRITICAL", "drac_ms2": 6.94, "collision_prob": 0.473},
                        },
                    ],
                },
                "turn_speed_ablation": [
                    {"estimator_mode": "ema", "mean_abs_error_kmh": 0.43, "max_abs_error_kmh": 1.53},
                    {"estimator_mode": "kalman", "mean_abs_error_kmh": 0.57, "max_abs_error_kmh": 2.14},
                    {"estimator_mode": "ctrv", "mean_abs_error_kmh": 0.23, "max_abs_error_kmh": 0.86},
                ],
            }
            bench_data = summarize_stage_timings([
                {"decode": 1.0, "detect_track": 18.2, "geometry_speed": 0.58, "conflict_events": 0.62, "render_write": 3.09},
                {"decode": 0.8, "detect_track": 14.0, "geometry_speed": 0.50, "conflict_events": 0.55, "render_write": 2.95},
            ])
            bench_data["model_load_ms"] = 1344.0
            bench_data["wall_elapsed_ms"] = 1400.0
            bench_data["wall_fps"] = round(2.0 / 1.4, 3)

            with mock.patch.object(charts_mod, "_save_figure", side_effect=_capture_save):
                plot_ablation_charts(ablation_data, root / "charts")
                plot_run_timeline(run_dir, root / "charts")
                plot_benchmark_breakdown(bench_data, root / "charts" / "benchmark_breakdown.png")
                plot_mot_comparison(
                    {
                        "基线跟踪": {"mota": 0.78, "idf1": 0.72, "hota": 0.68, "id_switches": 6},
                        "轨迹缝合": {"mota": 0.89, "idf1": 0.91, "hota": 0.85, "id_switches": 1},
                    },
                    root / "charts" / "mot_comparison.png",
                )

        speed_txt = captured_texts["ablation_speed_mae.png"]
        for expected in ("滑窗 EMA", "2D CA 卡尔曼", "弯道 CTRV", "测速误差 (km/h)", "弯道场景测速模型消融对比"):
            self.assertIn(expected, speed_txt)

        conflict_txt = captured_texts["ablation_conflict_metrics.png"]
        for expected in ("CVM 圆包络", "OBB-SSM", "窄路并行", "制动礼让", "垂直横穿", "L0 正常", "L3 紧急", "碰撞概率 (%)"):
            self.assertIn(expected, conflict_txt)

        bench_txt = captured_texts["benchmark_breakdown.png"]
        for expected in ("解码", "检测与跟踪", "投影与测速", "冲突与事件", "渲染与写盘", "单帧耗时 (ms)", "帧内吞吐", "墙钟吞吐", "模型加载"):
            self.assertIn(expected, bench_txt)

        timeline_txt = captured_texts["run_speed_timeline.png"]
        for expected in ("时间 (s)", "时速 (km/h)", "限速阈值 (30 km/h)", "确认冲突事件窗", "车辆 #1"):
            self.assertIn(expected, timeline_txt)

        risk_txt = captured_texts["run_risk_timeline.png"]
        for expected in ("风险等级", "L0 正常", "L1 超速", "L2 注意", "L3 紧急", "单帧峰值风险等级", "最小 TTC (s)"):
            self.assertIn(expected, risk_txt)

        mot_txt = captured_texts["mot_comparison.png"]
        for expected in ("多目标跟踪精度指标对比", "身份切换次数", "指标得分"):
            self.assertIn(expected, mot_txt)

    def test_dejavu_only_headless_all_six_charts_english_labels_and_glyph_coverage(self):
        import matplotlib.font_manager as fm
        import matplotlib.ft2font as ft2font
        import matplotlib.text as mtext
        import analysis.charts as charts_mod

        dejavu_only = [f for f in fm.fontManager.ttflist if getattr(f, "name", "") == "DejaVu Sans"]
        self.assertTrue(dejavu_only, msg="Matplotlib 内置 DejaVu Sans 字体缺失")
        dejavu_cmap = set(ft2font.FT2Font(dejavu_only[0].fname).get_charmap().keys())

        captured_texts: dict[str, list[str]] = {}
        orig_save = charts_mod._save_figure

        def _capture_all_texts(fig, out_path):
            fig.canvas.draw()
            texts = [
                t.get_text()
                for t in fig.findobj(mtext.Text)
                if t.get_visible() and t.get_text().strip()
            ]
            p_out = Path(out_path)
            captured_texts[f"{p_out.parent.name}/{p_out.name}"] = texts
            return orig_save(fig, out_path)

        cache_clear = getattr(getattr(fm.fontManager, "_findfont_cached", None), "cache_clear", None)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "run_en"
            run_dir.mkdir(parents=True)
            (run_dir / "configuration.json").write_text(json.dumps({"speed_limit_kmh": 30.5}), encoding="utf-8")
            with (run_dir / "speeds.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_s", "track_id", "category", "speed_kmh"])
                w.writerow(["0.0", "1", "vehicle", "22.5"])
                w.writerow(["0.5", "1", "vehicle", "33.0"])
            (run_dir / "events.jsonl").write_text(
                json.dumps({
                    "event_id": 1, "kind": "conflict",
                    "start_timestamp_s": 0.2, "end_timestamp_s": 0.5, "peak_risk_level": 3,
                }) + "\n",
                encoding="utf-8",
            )
            (run_dir / "conflicts.jsonl").write_text(
                json.dumps({
                    "frame_index": 0, "timestamp_s": 0.0,
                    "assessment": {"risk_level": 0, "ttc_seconds": None, "mttc_seconds": None},
                }) + "\n"
                + json.dumps({
                    "frame_index": 1, "timestamp_s": 0.5,
                    "assessment": {"risk_level": 3, "ttc_seconds": 0.0, "mttc_seconds": 1.8},
                }) + "\n",
                encoding="utf-8",
            )
            ablation_data = {
                "algorithm_comparison": {
                    "speed_estimation": {
                        "ema": {"mae_ms": 0.58, "rmse_ms": 0.68},
                        "kalman": {"mae_ms": 0.36, "rmse_ms": 0.46},
                    },
                    "conflict_detection": [
                        {
                            "scenario": "parallel_pass",
                            "cvm_circle": {"risk_level": "LEVEL_3_CRITICAL"},
                            "obb_ssm": {"risk_level": "LEVEL_0_SAFE", "drac_ms2": None, "collision_prob": None},
                        },
                        {
                            "scenario": "braking_yield",
                            "cvm_circle": {"risk_level": "LEVEL_3_CRITICAL"},
                            "obb_ssm": {"risk_level": "LEVEL_2_CAUTION", "drac_ms2": 1.97, "collision_prob": 0.0},
                        },
                    ],
                },
                "turn_speed_ablation": [
                    {"estimator_mode": "ema", "mean_abs_error_kmh": 0.43, "max_abs_error_kmh": 1.53},
                    {"estimator_mode": "ctrv", "mean_abs_error_kmh": 0.23, "max_abs_error_kmh": 0.86},
                ],
            }
            bench_data = summarize_stage_timings([
                {"decode": 1.5, "detect_track": 18.0, "geometry_speed": 0.6, "conflict_events": 0.5, "render_write": 3.0},
                {"decode": 1.0, "detect_track": 14.0, "geometry_speed": 0.5, "conflict_events": 0.4, "render_write": 2.5},
            ])
            bench_data["model_load_ms"] = 850.0
            bench_data["wall_elapsed_ms"] = 1000.0
            bench_data["wall_fps"] = 2.0

            try:
                with mock.patch.object(fm.fontManager, "ttflist", dejavu_only):
                    if callable(cache_clear):
                        cache_clear()
                    self.assertEqual(detect_available_cjk_fonts(), [])
                    self.assertEqual(configure_cjk_fonts(), ["DejaVu Sans"])
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        warnings.simplefilter("error", UserWarning)
                        with mock.patch.object(charts_mod, "_save_figure", side_effect=_capture_all_texts):
                            p_abl = plot_ablation_charts(ablation_data, root / "charts")
                            p_run = plot_run_timeline(run_dir, root / "charts")
                            p_mot = plot_mot_comparison(
                                {
                                    "Baseline": {"mota": 0.78, "idf1": 0.72, "hota": 0.68, "id_switches": 6},
                                    "Stitched": {"mota": 0.89, "idf1": 0.91, "hota": 0.85, "id_switches": 1},
                                },
                                root / "charts" / "mot_comparison.png",
                            )
                            p_bench = plot_benchmark_breakdown(bench_data, root / "charts" / "benchmark_breakdown.png")
                            # 额外覆盖网格消融、全零基准、无 events.jsonl 推断风险窗、以及调用方传入中文 MOT 键时的零缺字回退
                            plot_ablation_charts(
                                {
                                    "speed": [{"window_size": 5, "ema_alpha": 0.3, "absolute_error_ms": 0.4}],
                                    "conflict": [{"ttc_critical_s": 1.8, "horizon_s": 4.0, "ttc_s": None, "min_distance_m": 2.5}],
                                },
                                root / "charts_grid",
                            )
                            plot_benchmark_breakdown(summarize_stage_timings([]), root / "charts_grid" / "zero_bench.png")
                            plot_mot_comparison(
                                {
                                    "基线跟踪": {"mota": 0.78, "idf1": 0.72, "hota": 0.68, "id_switches": 6},
                                    "轨迹缝合": {"mota": 0.89, "idf1": 0.91, "hota": 0.85, "id_switches": 1},
                                },
                                root / "charts_grid" / "mot_zh_input.png",
                            )
                            run_inferred = root / "run_en_inferred"
                            run_inferred.mkdir()
                            (run_inferred / "speeds.csv").write_text(
                                "timestamp_s,track_id,category,speed_kmh\n0.0,1,vehicle,20.0\n5.0,1,vehicle,25.0\n",
                                encoding="utf-8",
                            )
                            (run_inferred / "conflicts.jsonl").write_text(
                                json.dumps({"frame_index": 20, "timestamp_s": 2.0, "assessment": {"risk_level": 3, "ttc_seconds": 0.5}}) + "\n",
                                encoding="utf-8",
                            )
                            plot_run_timeline(run_inferred, root / "charts_inferred")
            finally:
                if callable(cache_clear):
                    cache_clear()
                configure_cjk_fonts()

            all_paths = [*p_abl, *p_run, p_mot, p_bench]
            self.assertEqual(len(all_paths), 6)
            for p in all_paths:
                self._assert_valid_png(p)

        # 真实字形覆盖校验：DejaVu Sans 模式下所有 6 张图及网格/推断/零耗时/中文输入变体零 CJK 字符、且每个字符均在 DejaVu 字形表中
        self.assertTrue(
            {
                "charts/ablation_speed_mae.png",
                "charts/ablation_conflict_metrics.png",
                "charts/run_speed_timeline.png",
                "charts/run_risk_timeline.png",
                "charts/mot_comparison.png",
                "charts/benchmark_breakdown.png",
                "charts_grid/ablation_speed_mae.png",
                "charts_grid/ablation_conflict_metrics.png",
                "charts_grid/zero_bench.png",
                "charts_grid/mot_zh_input.png",
                "charts_inferred/run_speed_timeline.png",
                "charts_inferred/run_risk_timeline.png",
            }.issubset(captured_texts.keys())
        )
        for fname, text_list in captured_texts.items():
            joined = "\n".join(text_list)
            for txt in text_list:
                for ch in txt:
                    if ch.isspace():
                        continue
                    code = ord(ch)
                    self.assertFalse(
                        0x2E80 <= code <= 0x9FFF or 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF,
                        msg=f"{fname} attempted CJK char {ch!r} in DejaVu-only mode (text={txt!r})",
                    )
                    self.assertIn(
                        code,
                        dejavu_cmap,
                        msg=f"{fname} char {ch!r} (U+{code:04X}) missing from DejaVu Sans (text={txt!r})",
                    )
            self.assertTrue(joined.strip(), msg=f"{fname} produced empty text")

        self.assertIn("Sliding EMA", "\n".join(captured_texts["charts/ablation_speed_mae.png"]))
        self.assertIn("Turning CTRV", "\n".join(captured_texts["charts/ablation_speed_mae.png"]))
        self.assertIn("CVM Circle", "\n".join(captured_texts["charts/ablation_conflict_metrics.png"]))
        self.assertIn("N/A", "\n".join(captured_texts["charts/ablation_conflict_metrics.png"]))
        self.assertIn("Confirmed Conflict Event", "\n".join(captured_texts["charts/run_speed_timeline.png"]))
        self.assertIn("Speed Limit (30.5 km/h)", "\n".join(captured_texts["charts/run_speed_timeline.png"]))
        self.assertIn("Min TTC (s)", "\n".join(captured_texts["charts/run_risk_timeline.png"]))
        self.assertIn("Min MTTC (s)", "\n".join(captured_texts["charts/run_risk_timeline.png"]))
        self.assertIn("MOT Accuracy Comparison", "\n".join(captured_texts["charts/mot_comparison.png"]))
        self.assertIn("Inferred Transient Risk", "\n".join(captured_texts["charts_inferred/run_speed_timeline.png"]))
        self.assertIn("Config 1", "\n".join(captured_texts["charts_grid/mot_zh_input.png"]))
        bench_en = "\n".join(captured_texts["charts/benchmark_breakdown.png"])
        for token in ("Frame FPS:", "Wall FPS:", "Model Load:", "Detect & Track"):
            self.assertIn(token, bench_en)

    def test_conflict_ablation_preserves_none_and_distinguishes_zero(self):
        import analysis.charts as charts_mod

        inspected: dict[str, object] = {}
        orig_save = charts_mod._save_figure

        def _inspect_conflict(fig, out_path):
            if Path(out_path).name == "ablation_conflict_metrics.png":
                # fig.axes = [ax_risk, ax_ssm, ax_prob]
                ax_ssm = fig.axes[1]
                ax_prob = fig.axes[2]
                inspected["drac_bars"] = [round(p.get_height(), 3) for p in ax_ssm.patches]
                inspected["prob_bars"] = [round(p.get_height(), 3) for p in ax_prob.patches]
                inspected["drac_texts"] = [t.get_text() for t in ax_ssm.texts]
                inspected["prob_texts"] = [t.get_text() for t in ax_prob.texts]
            return orig_save(fig, out_path)

        ablation_data = {
            "algorithm_comparison": {
                "conflict_detection": [
                    {
                        "scenario": "parallel_pass",
                        "cvm_circle": {"risk_level": "LEVEL_3_CRITICAL"},
                        "obb_ssm": {"risk_level": "LEVEL_0_SAFE", "drac_ms2": None, "collision_prob": None},
                    },
                    {
                        "scenario": "braking_yield",
                        "cvm_circle": {"risk_level": "LEVEL_3_CRITICAL"},
                        "obb_ssm": {"risk_level": "LEVEL_2_CAUTION", "drac_ms2": 1.97, "collision_prob": 0.0},
                    },
                    {
                        "scenario": "perpendicular_crossing",
                        "cvm_circle": {"risk_level": "LEVEL_3_CRITICAL"},
                        "obb_ssm": {"risk_level": "LEVEL_3_CRITICAL", "drac_ms2": 6.94, "collision_prob": 0.473},
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                warnings.simplefilter("error", UserWarning)
                with mock.patch.object(charts_mod, "_save_figure", side_effect=_inspect_conflict):
                    plot_ablation_charts(ablation_data, root)

        # parallel_pass (None/None) 不绘柱、标注“无数据”；braking_yield 的真实 0.0% 必须绘柱且标注“0.0%”
        self.assertEqual(inspected["drac_bars"], [1.97, 6.94])
        self.assertEqual(inspected["prob_bars"], [0.0, 47.3])
        self.assertIn("无数据", inspected["drac_texts"])
        self.assertNotIn("0.00", inspected["drac_texts"])
        self.assertIn("无数据", inspected["prob_texts"])
        self.assertIn("0.0%", inspected["prob_texts"])
        self.assertIn("47.3%", inspected["prob_texts"])

    def test_run_timeline_separate_ttc_mttc_sparse_gaps_and_single_point(self):
        import analysis.charts as charts_mod

        captured: dict[str, dict] = {}
        orig_save = charts_mod._save_figure

        def _capture_timeline(fig, out_path):
            name = Path(out_path).name
            if name == "run_speed_timeline.png":
                ax = fig.axes[0]
                spans = []
                for p in ax.patches:
                    x0 = float(p.get_x())
                    x1 = float(p.get_x() + p.get_width())
                    spans.append((min(x0, x1), max(x0, x1)))
                leg = ax.get_legend()
                captured[name] = {
                    "spans": spans,
                    "legend": [t.get_text() for t in leg.get_texts()] if leg else [],
                }
            elif name == "run_risk_timeline.png":
                ax_r = fig.axes[0]
                ax_t = fig.axes[1] if len(fig.axes) > 1 else None
                leg = ax_r.get_legend()
                captured[name] = {
                    "risk_lines": [
                        {
                            "x": [float(v) for v in ln.get_xdata()],
                            "y": [float(v) for v in ln.get_ydata()],
                            "marker": ln.get_marker(),
                            "label": ln.get_label(),
                        }
                        for ln in ax_r.get_lines()
                    ],
                    "ttc_lines": [
                        {
                            "x": [float(v) for v in ln.get_xdata()],
                            "y": [float(v) for v in ln.get_ydata()],
                            "marker": ln.get_marker(),
                            "label": ln.get_label(),
                        }
                        for ln in (ax_t.get_lines() if ax_t else [])
                    ],
                    "legend": [t.get_text() for t in leg.get_texts()] if leg else [],
                }
            return orig_save(fig, out_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # 1) 稀疏 L3 帧（t=2s 与 t=5s，无 events.jsonl），且 TTC=0.0 与 MTTC=2.0 并存
            run_sparse = root / "run_sparse"
            run_sparse.mkdir()
            (run_sparse / "configuration.json").write_text(json.dumps({"speed_limit_kmh": 25.25}), encoding="utf-8")
            with (run_sparse / "speeds.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_s", "track_id", "category", "speed_kmh"])
                for s in range(7):
                    w.writerow([f"{s:.1f}", "1", "vehicle", f"{20.0 + s:.1f}"])
            (run_sparse / "conflicts.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "frame_index": 20, "timestamp_s": 2.0,
                        "assessment": {"risk_level": 3, "ttc_seconds": 0.0, "mttc_seconds": 2.0},
                    }),
                    json.dumps({
                        "frame_index": 50, "timestamp_s": 5.0,
                        "assessment": {"risk_level": 3, "ttc_seconds": 0.8, "mttc_seconds": 1.5},
                    }),
                ]) + "\n",
                encoding="utf-8",
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                warnings.simplefilter("error", UserWarning)
                with mock.patch.object(charts_mod, "_save_figure", side_effect=_capture_timeline):
                    out_sparse = plot_run_timeline(run_sparse, root / "out_sparse")
            self.assertEqual(len(out_sparse), 2)

            # 验证限速保留小数 25.25，且稀疏推断高风险窗为 2 个独立窄窗，未合并成 [2, 5]
            speed_info = captured["run_speed_timeline.png"]
            self.assertIn("限速阈值 (25.25 km/h)", speed_info["legend"])
            self.assertIn("瞬时高风险窗（推断）", speed_info["legend"])
            self.assertNotIn("确认冲突事件窗", speed_info["legend"])
            self.assertEqual(len(speed_info["spans"]), 2)
            self.assertLess(speed_info["spans"][0][1], 2.5)
            self.assertGreater(speed_info["spans"][1][0], 4.5)

            # 验证风险曲线插入 NaN 断点且带点标记，TTC 与 MTTC 为两条独立曲线（TTC=0.0 未被 MTTC=2.0 覆盖）
            risk_info = captured["run_risk_timeline.png"]
            risk_ln = risk_info["risk_lines"][0]
            self.assertEqual(len(risk_ln["x"]), 3)
            self.assertTrue(np.isnan(risk_ln["x"][1]))
            self.assertNotIn(risk_ln["marker"], ("", "None", None))

            self.assertEqual(len(risk_info["ttc_lines"]), 2)
            ttc_ln, mttc_ln = risk_info["ttc_lines"]
            self.assertIn("TTC", ttc_ln["label"])
            self.assertNotIn("MTTC", ttc_ln["label"])
            self.assertIn("MTTC", mttc_ln["label"])
            self.assertEqual(ttc_ln["y"][0], 0.0)
            self.assertTrue(np.isnan(ttc_ln["y"][1]))
            self.assertEqual(ttc_ln["y"][2], 0.8)
            self.assertEqual(mttc_ln["y"][0], 2.0)
            self.assertTrue(np.isnan(mttc_ln["y"][1]))
            self.assertEqual(mttc_ln["y"][2], 1.5)

            # 2) 单点 L3 记录：必须绘制带标记点，不能在图上完全消失
            captured.clear()
            run_single = root / "run_single"
            run_single.mkdir()
            (run_single / "configuration.json").write_text(json.dumps({"speed_limit_kmh": 30.5}), encoding="utf-8")
            with (run_single / "speeds.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_s", "track_id", "category", "speed_kmh"])
                w.writerow(["0.0", "1", "vehicle", "18.0"])
                w.writerow(["2.0", "1", "vehicle", "28.0"])
            (run_single / "conflicts.jsonl").write_text(
                json.dumps({
                    "frame_index": 10, "timestamp_s": 1.0,
                    "assessment": {"risk_level": 3, "ttc_seconds": 0.0},
                }) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(charts_mod, "_save_figure", side_effect=_capture_timeline):
                out_single = plot_run_timeline(run_single, root / "out_single")
            for p in out_single:
                self._assert_valid_png(p)
            self.assertIn("限速阈值 (30.5 km/h)", captured["run_speed_timeline.png"]["legend"])
            single_risk_ln = captured["run_risk_timeline.png"]["risk_lines"][0]
            self.assertEqual(single_risk_ln["x"], [1.0])
            self.assertEqual(single_risk_ln["y"], [3.0])
            self.assertNotIn(single_risk_ln["marker"], ("", "None", None))

            # 3) 缺少 configuration.json 时，从 summary.json["configuration"]["speed_limit_kmh"] 解析限速
            captured.clear()
            run_nested = root / "run_nested_summary"
            run_nested.mkdir()
            (run_nested / "summary.json").write_text(
                json.dumps({"frames_processed": 2, "configuration": {"speed_limit_kmh": 22.5}}),
                encoding="utf-8",
            )
            with (run_nested / "speeds.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_s", "track_id", "category", "speed_kmh"])
                w.writerow(["0.0", "1", "vehicle", "15.0"])
                w.writerow(["1.0", "1", "vehicle", "24.0"])
            self.assertEqual(charts_mod._resolve_speed_limit_kmh(run_nested), 22.5)
            with mock.patch.object(charts_mod, "_save_figure", side_effect=_capture_timeline):
                out_nested = plot_run_timeline(run_nested, root / "out_nested")
            for p in out_nested:
                self._assert_valid_png(p)
            self.assertIn("限速阈值 (22.5 km/h)", captured["run_speed_timeline.png"]["legend"])

            # 若同时存在 configuration.json，则优先读取 configuration.json；均缺失时回退 30.0
            (run_nested / "configuration.json").write_text(
                json.dumps({"speed_limit_kmh": 40.0}), encoding="utf-8"
            )
            self.assertEqual(charts_mod._resolve_speed_limit_kmh(run_nested), 40.0)
            run_empty_cfg = root / "run_default_limit"
            run_empty_cfg.mkdir()
            self.assertEqual(charts_mod._resolve_speed_limit_kmh(run_empty_cfg), 30.0)

    def test_evaluate_markdown_chinese_localization(self):
        sample = {
            "run_dir": "D:/runs/demo",
            "has_ground_truth": False,
            "frames": {"processed": 360, "observed_assessment_rows": 28},
            "events": {"count": 2},
            "speed_rows": {"speed_kmh": {"min": 0.0, "mean": 5.4812, "max": 35.2}},
            "risk_counts": {"LEVEL_0_SAFE": 14, "LEVEL_1": 3, "LEVEL_2": 1, "LEVEL_3": 10},
        }
        md = evaluate_to_markdown(sample)
        for token in ("# 运行产物分析评价", "已处理帧数：360", "## 风险等级分布", "L0 正常", "L1 超速", "L2 注意", "L3 紧急", "ground truth"):
            self.assertIn(token, md)

        sample_gt = dict(sample)
        sample_gt["has_ground_truth"] = True
        sample_gt["ground_truth_evaluation"] = {
            "speed_accuracy": {"matched_count": 2, "reference_count": 2, "mae_kmh": 1.2, "rmse_kmh": 1.5, "max_abs_error_kmh": 2.0},
            "event_accuracy": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "lead_time_s": {"count": 1, "min": 1.5, "mean": 1.5, "max": 1.5}},
        }
        md_gt = evaluate_to_markdown(sample_gt)
        self.assertIn("人工真值精度评价 (Ground-truth accuracy)", md_gt)
        self.assertIn("MAE=1.2 km/h", md_gt)

    def test_evaluate_markdown_edge_cases_none_backticks_and_risk_alias_merge(self):
        # 1) Ground-truth 为 None 时统一输出“无”，且不出现裸 None 或 None km/h
        sample_none_gt = {
            "run_dir": "D:/runs/foo`bar``baz",
            "has_ground_truth": True,
            "frames": {"processed": 10, "observed_assessment_rows": 9},
            "events": {"count": 0},
            "speed_rows": {"speed_kmh": {"min": None, "mean": None, "max": None}},
            "risk_counts": {
                1: 2,
                "1": 3,
                "LEVEL_1": 1,
                "LEVEL_1_SPEEDING": 4,
                "0": 2,
                "LEVEL_0_SAFE": 5,
            },
            "ground_truth_evaluation": {
                "speed_accuracy": {
                    "matched_count": 0, "reference_count": 2,
                    "mae_kmh": None, "rmse_kmh": None, "max_abs_error_kmh": None,
                },
                "event_accuracy": {
                    "tp": 0, "fp": 0, "fn": 0,
                    "precision": None, "recall": None, "f1": None,
                    "lead_time_s": {"count": 0, "min": None, "mean": None, "max": None},
                },
            },
        }
        md = evaluate_to_markdown(sample_none_gt)
        self.assertNotIn("None", md)
        for token in (
            "MAE=无",
            "RMSE=无",
            "最大绝对误差=无",
            "Precision=无",
            "Recall=无",
            "F1=无",
            "最小=无，平均=无，最大=无",
        ):
            self.assertIn(token, md)

        # 2) 动态反引号围栏：含双反引号的路径需使用三反引号包裹；首尾含反引号需加空格填充
        self.assertIn("```D:/runs/foo`bar``baz```", md)
        sample_edge_bt = dict(sample_none_gt, run_dir="`edge`")
        md_edge = evaluate_to_markdown(sample_edge_bt)
        self.assertIn("`` `edge` ``", md_edge)

        # 3) 等价风险别名合并：L0=7, L1=10，且表格中仅出现 1 行 L1 超速
        self.assertIn("| L0 正常 | 7 |", md)
        self.assertIn("| L1 超速 | 10 |", md)
        self.assertEqual(md.count("| L1 超速 |"), 1)
        self.assertEqual(md.count("| L0 正常 |"), 1)


if __name__ == "__main__":
    unittest.main()
