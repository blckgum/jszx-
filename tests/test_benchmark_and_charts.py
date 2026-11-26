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
                {"decode": 1.0, "detect_track": 18.2, "geometry_speed": 0.58, "conflict_events": 0.62, "render_write": 3.09}
            ])
            bench_data["model_load_ms"] = 1344.0

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
        for expected in ("解码", "检测与跟踪", "投影与测速", "冲突与事件", "渲染与写盘", "单帧耗时 (ms)", "端到端吞吐"):
            self.assertIn(expected, bench_txt)

        timeline_txt = captured_texts["run_speed_timeline.png"]
        for expected in ("时间 (s)", "时速 (km/h)", "限速阈值 (30 km/h)", "冲突事件窗", "车辆 #1"):
            self.assertIn(expected, timeline_txt)

        risk_txt = captured_texts["run_risk_timeline.png"]
        for expected in ("风险等级", "L0 正常", "L1 超速", "L2 注意", "L3 紧急", "单帧峰值风险等级"):
            self.assertIn(expected, risk_txt)

        mot_txt = captured_texts["mot_comparison.png"]
        for expected in ("多目标跟踪精度指标对比", "身份切换次数", "指标得分"):
            self.assertIn(expected, mot_txt)

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


if __name__ == "__main__":
    unittest.main()
