# -*- coding: utf-8 -*-
"""CampusMotion 独立桌面宿主、系统接口、文件导出/图表、标定工作台与算法实验室测试。"""

from __future__ import annotations

import ast
import base64
import builtins
import contextlib
import csv
import http.client
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import yaml

from analysis import desktop as desktop_mod
from analysis.calibration import RoadCalibrator
from analysis.scenarios import default_scenarios
from analysis.service import (
    AnalysisService,
    DEFAULT_WEB_ROOT,
    create_fastapi_app,
    create_http_server,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_dummy_video(path: Path, n_frames: int = 5, fps: float = 10.0, size=(320, 240)) -> None:
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for idx in range(n_frames):
        frame = np.full((h, w, 3), (idx * 35) % 255, dtype=np.uint8)
        cv2.putText(frame, f"F{idx}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def _create_sample_run(runs_root: Path, run_name: str = "run-studio-01") -> Path:
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ok",
                "stop_reason": "eof",
                "source_video": "demo.mp4",
                "frames_processed": 5,
                "elapsed_seconds": 0.5,
                "speed_limit_kmh": 30.0,
                "max_observed_speed_kmh": 34.2,
                "total_alert_events": 1,
                "total_speeding_alerts": 1,
                "total_caution_events": 0,
                "total_critical_conflicts": 0,
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event": "alert",
                "event_id": 1,
                "kind": "speeding",
                "veh_track_id": 1,
                "ped_track_id": None,
                "start_frame": 0,
                "end_frame": 4,
                "start_timestamp_s": 0.0,
                "end_timestamp_s": 0.4,
                "peak_risk_level": 1,
                "min_ttc_seconds": None,
                "end_reason": "eof",
            },
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "conflicts.jsonl").write_text(
        json.dumps(
            {
                "frame_index": 2,
                "timestamp_s": 0.2,
                "assessment": {
                    "veh_track_id": 1,
                    "ped_track_id": 2,
                    "risk_level": 2,
                    "veh_speed_kmh": 34.2,
                    "ttc_seconds": 2.1,
                },
            },
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with (run_dir / "speeds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["schema_version", "frame_idx", "timestamp_s", "track_id", "category", "speed_kmh"])
        for idx in range(5):
            writer.writerow([1, idx, round(idx * 0.1, 2), 1, "vehicle", 28.0 + idx * 1.5])
    _write_dummy_video(run_dir / "annotated_conflict.mp4", n_frames=5, fps=10.0, size=(160, 120))
    return run_dir


def _http_json(url: str, method: str = "GET", payload: dict | bytes | None = None) -> tuple[int, dict]:
    data = None
    headers = {}
    if isinstance(payload, dict):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    elif isinstance(payload, bytes):
        data = payload
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=20.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


class TestImportPurityAndFavicon(unittest.TestCase):
    def test_zero_forbidden_top_level_imports_in_service_and_desktop(self):
        forbidden = {
            "fastapi",
            "uvicorn",
            "scipy",
            "matplotlib",
            "tkinter",
            "torch",
            "ultralytics",
            "httpx",
        }
        for rel in ("analysis/service.py", "analysis/desktop.py"):
            src_path = REPO_ROOT / rel
            tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))
            top_imports = set()
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top_imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top_imports.add(node.module.split(".")[0])
            self.assertEqual(top_imports & forbidden, set(), msg=f"file={rel}")

    def test_favicon_exists_and_served_on_root_route(self):
        ico_path = DEFAULT_WEB_ROOT / "assets" / "favicon.ico"
        self.assertTrue(ico_path.is_file())
        ico_bytes = ico_path.read_bytes()
        self.assertTrue(ico_bytes.startswith(b"\x00\x00\x01\x00"))

        with tempfile.TemporaryDirectory() as tmp:
            svc = AnalysisService(Path(tmp))
            server = create_http_server(svc, host="127.0.0.1", port=0)
            port = server.server_address[1]
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            try:
                with urlopen(f"http://127.0.0.1:{port}/favicon.ico", timeout=5.0) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(resp.headers.get("Content-Type"), "image/x-icon")
                    self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
                    self.assertEqual(resp.read(), ico_bytes)
            finally:
                server.shutdown()
                server.server_close()


class TestDesktopSystemDialogExportChartsAndStudioApis(unittest.TestCase):
    def test_system_info_and_shutdown_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_root = root / "runs"
            runs_root.mkdir()
            _create_sample_run(runs_root, "run-sys-01")
            wdir = root / "weights"
            wdir.mkdir()
            custom_pt = wdir / "campus_custom.pt"
            custom_pt.write_bytes(b"pt-stub")

            for bad_bool in ("yes", 1):
                with self.assertRaises(ValueError):
                    AnalysisService(runs_root, desktop_mode=bad_bool)
                with self.assertRaises(ValueError):
                    AnalysisService(runs_root, allow_shutdown=bad_bool)
            with self.assertRaises(ValueError):
                AnalysisService(runs_root, shutdown_callback="not-callable")

            svc_readonly = AnalysisService(runs_root)
            info_ro = svc_readonly.get_system_info()
            for key in (
                "python_version",
                "platform",
                "desktop_mode",
                "allow_shutdown",
                "can_browse_files",
                "cuda_available",
                "runs_root",
                "default_calib",
                "weights",
                "active_jobs",
                "total_runs",
            ):
                self.assertIn(key, info_ro)
            self.assertFalse(info_ro["desktop_mode"])
            self.assertFalse(info_ro["allow_shutdown"])
            self.assertEqual(info_ro["total_runs"], 1)
            self.assertTrue(any(item["path"] == str(custom_pt.resolve()) for item in info_ro["weights"]))

            # 模拟无 tkinter 与有伪 CUDA 环境下的懒探测分支。
            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
            orig_find_spec = importlib_find_spec = __import__("importlib.util").util.find_spec

            def _mock_find_spec(name, *args, **kwargs):
                if name == "tkinter":
                    return None
                if name == "torch":
                    return object()
                return orig_find_spec(name, *args, **kwargs)

            with mock.patch("importlib.util.find_spec", side_effect=_mock_find_spec), \
                    mock.patch.dict(sys.modules, {"torch": fake_torch}):
                info_mocked = svc_readonly.get_system_info()
            self.assertFalse(info_mocked["can_browse_files"])
            self.assertTrue(info_mocked["cuda_available"])
            self.assertIsNotNone(importlib_find_spec)

            # shutdown 未授权返回 403；授权后停止活动任务并在 0.15s 后异步触发回调。
            cb_event = threading.Event()
            svc_desk = AnalysisService(
                runs_root,
                desktop_mode=True,
                allow_shutdown=True,
                shutdown_callback=cb_event.set,
            )
            srv_ro = create_http_server(svc_readonly, host="127.0.0.1", port=0)
            srv_desk = create_http_server(svc_desk, host="127.0.0.1", port=0)
            t_ro = threading.Thread(target=srv_ro.serve_forever, daemon=True)
            t_desk = threading.Thread(target=srv_desk.serve_forever, daemon=True)
            t_ro.start()
            t_desk.start()
            try:
                ro_base = f"http://127.0.0.1:{srv_ro.server_address[1]}"
                dk_base = f"http://127.0.0.1:{srv_desk.server_address[1]}"

                st_403, body_403 = _http_json(f"{ro_base}/api/system/shutdown", method="POST", payload={})
                self.assertEqual(st_403, 403)
                self.assertIn("error", body_403)

                st_info, dk_info = _http_json(f"{dk_base}/api/system/info")
                self.assertEqual(st_info, 200)
                self.assertTrue(dk_info["desktop_mode"])
                self.assertTrue(dk_info["allow_shutdown"])

                # 启动一个慢任务验证 shutdown 前会停止活动任务。
                calib_yaml = root / "calib.yaml"
                RoadCalibrator(
                    [(200.0, 120.0), (600.0, 120.0), (760.0, 520.0), (40.0, 520.0)],
                    8.0,
                    24.0,
                ).to_yaml(calib_yaml)
                slow_video = root / "slow.mp4"
                _write_dummy_video(slow_video, n_frames=12, fps=10.0)
                gate = threading.Event()
                slow_sess = mock.Mock()
                slow_sess.process.side_effect = lambda f, ts, k: (
                    gate.wait(timeout=0.05),
                    types.SimpleNamespace(tracks=[]),
                )[1]
                slow_sess.retained_ids.return_value = set()

                with mock.patch("analysis.pipeline.TrackingSession", return_value=slow_sess):
                    st_job, job_resp = _http_json(
                        f"{dk_base}/api/jobs",
                        method="POST",
                        payload={"source": str(slow_video), "calib_path": str(calib_yaml)},
                    )
                    self.assertEqual(st_job, 201)
                    time.sleep(0.08)

                    st_shut, shut_resp = _http_json(f"{dk_base}/api/system/shutdown", method="POST", payload={})
                    self.assertEqual(st_shut, 200)
                    self.assertEqual(shut_resp, {"status": "shutting_down"})
                    self.assertTrue(cb_event.wait(timeout=2.0))

                    st_after, after_job = _http_json(f"{dk_base}/api/jobs/{job_resp['job_id']}")
                    self.assertEqual(st_after, 200)
                    self.assertIn(after_job["status"], ("stopped", "completed"))
            finally:
                srv_ro.shutdown()
                srv_ro.server_close()
                srv_desk.shutdown()
                srv_desk.server_close()

    def test_dialog_browse_success_cancel_and_headless_501(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = AnalysisService(Path(tmp))
            srv = create_http_server(svc, host="127.0.0.1", port=0)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                st_bad, _ = _http_json(f"{base}/api/dialog/browse", method="POST", payload={"kind": "invalid"})
                self.assertEqual(st_bad, 400)

                root_instances = []

                class _FakeTclError(RuntimeError):
                    pass

                class _FakeRoot:
                    def __init__(self):
                        self.destroyed = False
                        self.withdrawn = False
                        root_instances.append(self)

                    def withdraw(self):
                        self.withdrawn = True

                    def attributes(self, *_args):
                        return None

                    def update_idletasks(self):
                        return None

                    def destroy(self):
                        self.destroyed = True

                fake_tk = types.ModuleType("tkinter")
                fake_tk.Tk = _FakeRoot
                fake_tk.TclError = _FakeTclError

                fake_fd = types.ModuleType("tkinter.filedialog")
                fake_fd.askopenfilename = mock.Mock(side_effect=["C:/videos/cam1.mp4", ""])
                fake_fd.askdirectory = mock.Mock(return_value="C:/runs/out")
                fake_tk.filedialog = fake_fd

                with mock.patch.dict(
                    sys.modules, {"tkinter": fake_tk, "tkinter.filedialog": fake_fd}
                ):
                    st_v, res_v = _http_json(f"{base}/api/dialog/browse", method="POST", payload={"kind": "video"})
                    self.assertEqual(st_v, 200)
                    self.assertEqual(res_v, {"cancelled": False, "path": "C:/videos/cam1.mp4"})

                    st_c, res_c = _http_json(f"{base}/api/dialog/browse", method="POST", payload={"kind": "calib"})
                    self.assertEqual(st_c, 200)
                    self.assertEqual(res_c, {"cancelled": True, "path": None})

                    st_d, res_d = _http_json(f"{base}/api/dialog/browse", method="POST", payload={"kind": "folder"})
                    self.assertEqual(st_d, 200)
                    self.assertEqual(res_d, {"cancelled": False, "path": "C:/runs/out"})

                self.assertEqual(len(root_instances), 3)
                self.assertTrue(all(r.destroyed and r.withdrawn for r in root_instances))

                # 模拟 headless TclError -> 返回 501 且依然调用 root.destroy()。
                fake_fd.askopenfilename = mock.Mock(side_effect=_FakeTclError("no display name"))
                with mock.patch.dict(
                    sys.modules, {"tkinter": fake_tk, "tkinter.filedialog": fake_fd}
                ):
                    st_501, body_501 = _http_json(f"{base}/api/dialog/browse", method="POST", payload={"kind": "video"})
                self.assertEqual(st_501, 501)
                self.assertIs(body_501["cancelled"], True)
                self.assertIn("error", body_501)
                self.assertTrue(root_instances[-1].destroyed)

                # 模拟缺少 tkinter 模块 -> 返回 501。
                orig_import = builtins.__import__

                def _no_tk(name, *args, **kwargs):
                    if name == "tkinter" or name.startswith("tkinter."):
                        raise ImportError("No module named 'tkinter'")
                    return orig_import(name, *args, **kwargs)

                with mock.patch("builtins.__import__", side_effect=_no_tk):
                    st_missing, body_missing = _http_json(
                        f"{base}/api/dialog/browse", method="POST", payload={"kind": "folder"}
                    )
                self.assertEqual(st_missing, 501)
                self.assertIs(body_missing["cancelled"], True)
                self.assertIn("error", body_missing)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_run_export_delete_charts_and_open_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_root = root / "runs"
            runs_root.mkdir()
            run_dir = _create_sample_run(runs_root, "run-manage-01")
            bare_dir = runs_root / "run-bare-02"
            bare_dir.mkdir()
            (bare_dir / "summary.json").write_text(
                json.dumps({"status": "ok", "frames_processed": 0}, allow_nan=False),
                encoding="utf-8",
            )
            (run_dir / "secret.txt").write_text("do-not-export", encoding="utf-8")

            svc = AnalysisService(runs_root)
            srv = create_http_server(svc, host="127.0.0.1", port=0)
            port = srv.server_address[1]
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            base = f"http://127.0.0.1:{port}"
            try:
                # 导出白名单全部允许项校验头与字节一致。
                for fname in ("summary.json", "speeds.csv", "conflicts.jsonl", "events.jsonl", "annotated_conflict.mp4"):
                    with urlopen(f"{base}/api/runs/run-manage-01/export/{fname}", timeout=5.0) as resp:
                        self.assertEqual(resp.status, 200)
                        self.assertEqual(
                            resp.headers.get("Content-Disposition"),
                            f'attachment; filename="run-manage-01_{fname}"',
                        )
                        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
                        self.assertEqual(resp.read(), (run_dir / fname).read_bytes())

                # 拒绝未授权文件与路径穿越（含未规范化原始路径）。
                for bad_path in (
                    "/api/runs/run-manage-01/export/secret.txt",
                    "/api/runs/run-manage-01/export/..%2Fsummary.json",
                    "/api/runs/run-manage-01/export/%252e%252e/summary.json",
                    "/api/runs/run-manage-01/export/sub/summary.json",
                    "/api/runs/run-manage-01/export/",
                ):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
                    try:
                        conn.request("GET", bad_path)
                        res = conn.getresponse()
                        res.read()
                    finally:
                        conn.close()
                    self.assertEqual(res.status, 400, msg=f"bad_path={bad_path}")

                st_missing_exp, _ = _http_json(f"{base}/api/runs/run-bare-02/export/events.jsonl")
                self.assertEqual(st_missing_exp, 404)

                # 生成并读取 run charts PNG。
                st_ch, ch_body = _http_json(f"{base}/api/runs/run-manage-01/charts", method="POST", payload={})
                self.assertEqual(st_ch, 200)
                self.assertEqual(ch_body["run_id"], "run-manage-01")
                self.assertGreaterEqual(len(ch_body["charts"]), 1)
                first_chart = ch_body["charts"][0]
                self.assertTrue(first_chart["name"].endswith(".png"))
                self.assertEqual(first_chart["url"], f"/api/runs/run-manage-01/charts/{first_chart['name']}")

                with urlopen(f"{base}{first_chart['url']}", timeout=5.0) as cresp:
                    self.assertEqual(cresp.status, 200)
                    self.assertEqual(cresp.headers.get("Content-Type"), "image/png")
                    png_bytes = cresp.read()
                self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))

                # 非 .png 或路径穿越返回 400；缺失 .png 返回 404。
                for bad_chart in (
                    "/api/runs/run-manage-01/charts/note.txt",
                    "/api/runs/run-manage-01/charts/..%2Fsummary.json",
                    "/api/runs/run-manage-01/charts/sub/foo.png",
                ):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
                    try:
                        conn.request("GET", bad_chart)
                        res = conn.getresponse()
                        res.read()
                    finally:
                        conn.close()
                    self.assertEqual(res.status, 400, msg=f"bad_chart={bad_chart}")

                st_mc, _ = _http_json(f"{base}/api/runs/run-manage-01/charts/not_found.png")
                self.assertEqual(st_mc, 404)
                st_nocsv, _ = _http_json(f"{base}/api/runs/run-bare-02/charts", method="POST", payload={})
                self.assertEqual(st_nocsv, 404)

                # open-folder 测试（Windows 下 mock os.startfile，非 Windows mock subprocess.Popen）。
                if sys.platform.startswith("win"):
                    with mock.patch("os.startfile", create=True) as m_start:
                        st_op, op_body = _http_json(
                            f"{base}/api/runs/run-manage-01/open-folder", method="POST", payload={}
                        )
                    self.assertEqual(st_op, 200)
                    self.assertEqual(op_body, {"opened": True, "path": str(run_dir.resolve())})
                    m_start.assert_called_once_with(str(run_dir.resolve()))
                else:
                    with mock.patch("analysis.service.shutil.which", return_value="/usr/bin/xdg-open"), \
                            mock.patch("analysis.service.subprocess.Popen") as m_popen:
                        st_op, op_body = _http_json(
                            f"{base}/api/runs/run-manage-01/open-folder", method="POST", payload={}
                        )
                    self.assertEqual(st_op, 200)
                    self.assertEqual(op_body, {"opened": True, "path": str(run_dir.resolve())})
                    m_popen.assert_called_once()

                # 删除正在被活动任务占用的 run 目录必须返回 409。
                busy_run = _create_sample_run(runs_root, "run-busy-03")
                calib_yaml = root / "calib.yaml"
                RoadCalibrator(
                    [(200.0, 120.0), (600.0, 120.0), (760.0, 520.0), (40.0, 520.0)],
                    8.0,
                    24.0,
                ).to_yaml(calib_yaml)
                busy_vid = root / "busy.mp4"
                _write_dummy_video(busy_vid, n_frames=10, fps=10.0)
                gate = threading.Event()
                slow_sess = mock.Mock()
                slow_sess.process.side_effect = lambda f, ts, k: (
                    gate.wait(timeout=0.08),
                    types.SimpleNamespace(tracks=[]),
                )[1]
                slow_sess.retained_ids.return_value = set()

                with mock.patch("analysis.pipeline.TrackingSession", return_value=slow_sess):
                    st_bj, bj = _http_json(
                        f"{base}/api/jobs",
                        method="POST",
                        payload={
                            "source": str(busy_vid),
                            "calib_path": str(calib_yaml),
                            "output_dir": str(busy_run),
                        },
                    )
                    self.assertEqual(st_bj, 201)
                    time.sleep(0.05)
                    st_409, body_409 = _http_json(f"{base}/api/runs/run-busy-03", method="DELETE")
                    self.assertEqual(st_409, 409)
                    self.assertIn("error", body_409)
                    self.assertTrue(busy_run.is_dir())

                    st_stop, _ = _http_json(f"{base}/api/jobs/{bj['job_id']}/stop", method="POST", payload={})
                    self.assertEqual(st_stop, 200)

                # 任务停止后允许删除。
                st_del_busy, del_busy = _http_json(f"{base}/api/runs/run-busy-03", method="DELETE")
                self.assertEqual(st_del_busy, 200)
                self.assertEqual(del_busy, {"deleted": True, "run_id": "run-busy-03"})
                self.assertFalse(busy_run.exists())

                st_del_1, del_1 = _http_json(f"{base}/api/runs/run-manage-01", method="DELETE")
                self.assertEqual(st_del_1, 200)
                self.assertEqual(del_1, {"deleted": True, "run_id": "run-manage-01"})
                self.assertFalse(run_dir.exists())

                st_del_404, _ = _http_json(f"{base}/api/runs/run-manage-01", method="DELETE")
                self.assertEqual(st_del_404, 404)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_calibration_studio_and_algorithm_lab_apis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            default_calib = cfg_dir / "calibration.yaml"
            base_pts = [(400.0, 300.0), (880.0, 300.0), (1150.0, 680.0), (130.0, 680.0)]
            ref_cal = RoadCalibrator(base_pts, 6.0, 15.0, synthetic=True)
            ref_cal.to_yaml(default_calib)

            runs_root = root / "runs"
            runs_root.mkdir()
            vpath = root / "studio_clip.mp4"
            _write_dummy_video(vpath, n_frames=6, fps=12.0, size=(640, 480))

            svc = AnalysisService(runs_root, default_calib=default_calib)
            srv = create_http_server(svc, host="127.0.0.1", port=0)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                # 1. /api/calibration/frame
                st_f, frame_res = _http_json(
                    f"{base}/api/calibration/frame",
                    method="POST",
                    payload={"source": str(vpath), "frame_index": 2},
                )
                self.assertEqual(st_f, 200)
                self.assertEqual(frame_res["width"], 640)
                self.assertEqual(frame_res["height"], 480)
                self.assertEqual(frame_res["frame_count"], 6)
                self.assertAlmostEqual(frame_res["fps"], 12.0, places=2)
                self.assertEqual(frame_res["frame_index"], 2)
                self.assertTrue(frame_res["image_data_url"].startswith("data:image/jpeg;base64,"))
                b64_data = frame_res["image_data_url"].split(",", 1)[1]
                self.assertTrue(base64.b64decode(b64_data).startswith(b"\xff\xd8\xff"))

                st_f_miss, _ = _http_json(
                    f"{base}/api/calibration/frame",
                    method="POST",
                    payload={"source": str(root / "missing.mp4"), "frame_index": 0},
                )
                self.assertEqual(st_f_miss, 404)

                st_f_blank, _ = _http_json(
                    f"{base}/api/calibration/frame",
                    method="POST",
                    payload={"source": "   ", "frame_index": 0},
                )
                self.assertEqual(st_f_blank, 400)

                st_f_oor, _ = _http_json(
                    f"{base}/api/calibration/frame",
                    method="POST",
                    payload={"source": str(vpath), "frame_index": 99},
                )
                self.assertEqual(st_f_oor, 404)

                # 2. /api/calibration/preview（含 4 点与多点对应 + 像素停车区投射）
                park_gd = [(0.5, 2.0), (2.2, 2.0), (2.2, 6.0), (0.5, 6.0)]
                park_px = []
                for gx, gy in park_gd:
                    pt = ref_cal.ground_to_pixel_checked(gx, gy)
                    assert pt is not None
                    park_px.append([pt[0], pt[1]])

                st_prev, prev_res = _http_json(
                    f"{base}/api/calibration/preview",
                    method="POST",
                    payload={
                        "image_points": [list(p) for p in base_pts],
                        "width_m": 6.0,
                        "length_m": 15.0,
                        "parking_zones_image": [park_px],
                        "name": "north_gate",
                    },
                )
                self.assertEqual(st_prev, 200)
                self.assertTrue(prev_res["valid"])
                self.assertEqual(prev_res["inlier_count"], 4)
                self.assertLess(prev_res["reprojection_rmse_m"], 0.01)
                self.assertEqual(len(prev_res["parking_zones"]), 1)
                self.assertEqual(len(prev_res["parking_zones"][0]), 4)

                # 5 点对应点 + RANSAC 预览。
                mid_px = ref_cal.ground_to_pixel_checked(3.0, 7.5)
                assert mid_px is not None
                st_p5, p5_res = _http_json(
                    f"{base}/api/calibration/preview",
                    method="POST",
                    payload={
                        "image_points": [list(p) for p in base_pts] + [[mid_px[0], mid_px[1]]],
                        "ground_points": [[0.0, 0.0], [6.0, 0.0], [6.0, 15.0], [0.0, 15.0], [3.0, 7.5]],
                        "width_m": 6.0,
                        "length_m": 15.0,
                        "use_ransac": True,
                    },
                )
                self.assertEqual(st_p5, 200)
                self.assertEqual(p5_res["inlier_count"], 5)

                # 共线非法几何与 NaN 均须返回 400。
                st_col, _ = _http_json(
                    f"{base}/api/calibration/preview",
                    method="POST",
                    payload={
                        "image_points": [[0, 0], [10, 0], [20, 0], [30, 0]],
                        "width_m": 6.0,
                        "length_m": 15.0,
                    },
                )
                self.assertEqual(st_col, 400)

                st_nan, _ = _http_json(
                    f"{base}/api/calibration/preview",
                    method="POST",
                    payload=b'{"image_points": [[0,0],[1,0],[1,1],[0,1]], "width_m": NaN, "length_m": 10}',
                )
                self.assertEqual(st_nan, 400)

                # 3. /api/calibration/save（校验单段文件名与 synthetic=False 持久化）
                st_save, save_res = _http_json(
                    f"{base}/api/calibration/save",
                    method="POST",
                    payload={
                        "image_points": [list(p) for p in base_pts],
                        "width_m": 6.0,
                        "length_m": 15.0,
                        "parking_zones_image": [park_px],
                        "name": "north_gate_live",
                        "filename": "north_gate.yaml",
                    },
                )
                self.assertEqual(st_save, 200)
                self.assertTrue(save_res["saved"])
                self.assertEqual(save_res["label"], "north_gate.yaml")
                saved_file = (cfg_dir / "north_gate.yaml").resolve()
                self.assertEqual(save_res["path"], str(saved_file))
                self.assertTrue(saved_file.is_file())

                reloaded = RoadCalibrator.from_yaml(saved_file)
                self.assertFalse(reloaded.synthetic)
                self.assertEqual(len(reloaded.parking_zones), 1)
                self.assertTrue(reloaded.is_in_parking_zone(1.3, 4.0))

                st_pre, pre_body = _http_json(f"{base}/api/presets")
                self.assertEqual(st_pre, 200)
                self.assertTrue(any(c["path"] == str(saved_file) for c in pre_body["calibs"]))

                for bad_fname in ("../evil.yaml", "sub/a.yaml", "a\\b.yaml", "bad.txt", ".yaml", "bytetrack.yaml", "  "):
                    st_bf, _ = _http_json(
                        f"{base}/api/calibration/save",
                        method="POST",
                        payload={
                            "image_points": [list(p) for p in base_pts],
                            "width_m": 6.0,
                            "length_m": 15.0,
                            "filename": bad_fname,
                        },
                    )
                    self.assertEqual(st_bf, 400, msg=f"bad_fname={bad_fname!r}")

                # 4. /api/tools/scenarios & /api/tools/ablation
                st_sc, sc_res = _http_json(f"{base}/api/tools/scenarios", method="POST", payload={})
                self.assertEqual(st_sc, 200)
                self.assertEqual(sc_res["total"], 12)
                self.assertEqual(sc_res["passed"], 12)
                self.assertIs(sc_res["all_passed"], True)
                self.assertEqual(len(sc_res["results"]), 12)

                st_ab, ab_res = _http_json(f"{base}/api/tools/ablation", method="POST", payload={})
                self.assertEqual(st_ab, 200)
                self.assertIn("speed", ab_res)
                self.assertIn("conflict", ab_res)
                self.assertIn("algorithm_comparison", ab_res)
                self.assertIn("turn_speed_ablation", ab_res)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_calibration_five_plus_points_ransac_preview_and_save_persists_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            default_calib = cfg_dir / "calibration.yaml"
            base_pts = [(400.0, 300.0), (880.0, 300.0), (1150.0, 680.0), (130.0, 680.0)]
            ref_cal = RoadCalibrator(base_pts, 6.0, 15.0, synthetic=True)
            ref_cal.to_yaml(default_calib)

            runs_root = root / "runs"
            runs_root.mkdir()
            svc = AnalysisService(runs_root, default_calib=default_calib)
            srv = create_http_server(svc, host="127.0.0.1", port=0)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                extra_ground = [[3.0, 7.5], [1.5, 11.25]]
                extra_img = []
                for gx, gy in extra_ground:
                    pt = ref_cal.ground_to_pixel_checked(gx, gy)
                    assert pt is not None
                    extra_img.append([float(pt[0]), float(pt[1])])

                img_pts = [list(p) for p in base_pts] + extra_img
                gd_pts = [[0.0, 0.0], [6.0, 0.0], [6.0, 15.0], [0.0, 15.0]] + extra_ground

                st_prev, prev_res = _http_json(
                    f"{base}/api/calibration/preview",
                    method="POST",
                    payload={
                        "image_points": img_pts,
                        "ground_points": gd_pts,
                        "use_ransac": True,
                        "ransac_reproj_threshold_m": 0.5,
                        "name": "ransac_six_pt",
                    },
                )
                self.assertEqual(st_prev, 200)
                self.assertTrue(prev_res["valid"])
                self.assertEqual(prev_res["inlier_count"], 6)
                self.assertLess(prev_res["reprojection_rmse_m"], 0.02)
                self.assertEqual(len(prev_res["ground_points"]), 6)

                st_save, save_res = _http_json(
                    f"{base}/api/calibration/save",
                    method="POST",
                    payload={
                        "image_points": img_pts,
                        "ground_points": gd_pts,
                        "width_m": 6.0,
                        "length_m": 15.0,
                        "use_ransac": True,
                        "ransac_reproj_threshold_m": 0.5,
                        "name": "ransac_six_pt",
                        "filename": "ransac_six_pt.yaml",
                    },
                )
                self.assertEqual(st_save, 200)
                self.assertTrue(save_res["saved"])
                self.assertEqual(save_res["label"], "ransac_six_pt.yaml")
                saved_file = (cfg_dir / "ransac_six_pt.yaml").resolve()
                self.assertEqual(save_res["path"], str(saved_file))
                self.assertTrue(saved_file.is_file())

                raw_yaml = yaml.safe_load(saved_file.read_text(encoding="utf-8"))
                self.assertFalse(raw_yaml["synthetic"])
                self.assertTrue(raw_yaml["use_ransac"])
                self.assertEqual(len(raw_yaml["image_points"]), 6)
                self.assertEqual(len(raw_yaml["ground_points"]), 6)

                reloaded = RoadCalibrator.from_yaml(saved_file)
                self.assertFalse(reloaded.synthetic)
                self.assertEqual(reloaded.inlier_count, 6)
                self.assertEqual(len(reloaded.image_points), 6)
                self.assertEqual(len(reloaded.ground_points), 6)
                gx_mid, gy_mid = reloaded.pixel_to_ground(extra_img[0][0], extra_img[0][1])
                self.assertAlmostEqual(gx_mid, 3.0, delta=0.05)
                self.assertAlmostEqual(gy_mid, 7.5, delta=0.05)

                # 5+ points without ground_points must return 400 instead of crashing RoadCalibrator.
                st_no_gp, _ = _http_json(
                    f"{base}/api/calibration/save",
                    method="POST",
                    payload={
                        "image_points": img_pts,
                        "width_m": 6.0,
                        "length_m": 15.0,
                        "use_ransac": True,
                        "filename": "bad_five_no_gp.yaml",
                    },
                )
                self.assertEqual(st_no_gp, 400)
            finally:
                srv.shutdown()
                srv.server_close()

    def test_scenarios_default_heading_computed_from_velocity(self):
        scenarios = {s.name: s for s in default_scenarios()}
        cc = scenarios["critical_crossing"].readings
        self.assertIs(type(cc[0].heading_deg), float)
        self.assertAlmostEqual(cc[0].heading_deg, 90.0, places=6)
        self.assertAlmostEqual(cc[1].heading_deg, 180.0, places=6)
        div = scenarios["diverging_safe"].readings
        self.assertAlmostEqual(div[0].heading_deg, 90.0, places=6)
        self.assertAlmostEqual(div[1].heading_deg, 270.0, places=6)
        parked = scenarios["parked_safe"].readings
        self.assertAlmostEqual(parked[0].heading_deg, 0.0, places=6)

    def test_fastapi_registers_and_handles_all_new_routes(self):
        registered_routes = []

        class _FakeFastAPI:
            def __init__(self, **_kwargs):
                pass

            def _reg(self, method: str, path: str):
                def _dec(fn):
                    registered_routes.append((method, path, fn))
                    return fn
                return _dec

            def get(self, path: str):
                return self._reg("GET", path)

            def post(self, path: str):
                return self._reg("POST", path)

            def delete(self, path: str):
                return self._reg("DELETE", path)

            def websocket(self, path: str):
                return self._reg("WS", path)

        fake_fastapi = types.ModuleType("fastapi")
        fake_fastapi.FastAPI = _FakeFastAPI
        fake_fastapi.HTTPException = RuntimeError
        fake_fastapi.Request = object
        fake_fastapi.WebSocket = object
        fake_fastapi.WebSocketDisconnect = RuntimeError

        fake_responses = types.ModuleType("fastapi.responses")
        fake_responses.Response = dict
        fake_responses.StreamingResponse = lambda content, **kw: {"content": content, **kw}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_root = root / "runs"
            runs_root.mkdir()
            _create_sample_run(runs_root, "run-fa-01")
            web_dir = root / "web"
            (web_dir / "assets").mkdir(parents=True)
            (web_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
            (web_dir / "assets" / "favicon.ico").write_bytes(b"\x00\x00\x01\x00ico")

            svc = AnalysisService(runs_root, web_root=web_dir, desktop_mode=True, allow_shutdown=False)
            with mock.patch.dict(
                sys.modules,
                {"fastapi": fake_fastapi, "fastapi.responses": fake_responses},
            ):
                create_fastapi_app(svc)

            route_map = {(m, p): fn for m, p, fn in registered_routes}
            expected_routes = {
                ("GET", "/favicon.ico"),
                ("GET", "/api/system/info"),
                ("POST", "/api/system/shutdown"),
                ("POST", "/api/dialog/browse"),
                ("GET", "/api/runs/{run_id}/export/{filename:path}"),
                ("DELETE", "/api/runs/{run_id}"),
                ("POST", "/api/runs/{run_id}/charts"),
                ("GET", "/api/runs/{run_id}/charts/{filename:path}"),
                ("POST", "/api/runs/{run_id}/open-folder"),
                ("POST", "/api/calibration/frame"),
                ("POST", "/api/calibration/preview"),
                ("POST", "/api/calibration/save"),
                ("POST", "/api/tools/scenarios"),
                ("POST", "/api/tools/ablation"),
            }
            self.assertTrue(expected_routes.issubset(set(route_map.keys())))

            fav = route_map[("GET", "/favicon.ico")]()
            self.assertEqual(fav["media_type"], "image/x-icon")

            sys_info = json.loads(route_map[("GET", "/api/system/info")]()["content"])
            self.assertTrue(sys_info["desktop_mode"])

            shut_403 = route_map[("POST", "/api/system/shutdown")]()
            self.assertEqual(shut_403["status_code"], 403)

            exp_resp = route_map[("GET", "/api/runs/{run_id}/export/{filename:path}")](
                "run-fa-01", "summary.json"
            )
            self.assertIn("attachment; filename=\"run-fa-01_summary.json\"", exp_resp["headers"]["Content-Disposition"])

            ch_gen = json.loads(route_map[("POST", "/api/runs/{run_id}/charts")]("run-fa-01")["content"])
            self.assertGreaterEqual(len(ch_gen["charts"]), 1)
            ch_name = ch_gen["charts"][0]["name"]
            ch_png = route_map[("GET", "/api/runs/{run_id}/charts/{filename:path}")]("run-fa-01", ch_name)
            self.assertEqual(ch_png["media_type"], "image/png")
            self.assertTrue(ch_png["content"].startswith(b"\x89PNG\r\n\x1a\n"))

            sc_resp = json.loads(route_map[("POST", "/api/tools/scenarios")]()["content"])
            self.assertTrue(sc_resp["all_passed"])

            del_resp = json.loads(route_map[("DELETE", "/api/runs/{run_id}")]("run-fa-01")["content"])
            self.assertTrue(del_resp["deleted"])


class TestStandaloneDesktopHost(unittest.TestCase):
    def test_app_data_dir_and_chromium_detection(self):
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": "C:/Users/Test/AppData/Local"}, clear=False):
            self.assertEqual(
                desktop_mod.get_app_data_dir(),
                Path("C:/Users/Test/AppData/Local") / "CampusMotion",
            )
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": ""}, clear=False):
            self.assertEqual(
                desktop_mod.get_app_data_dir(),
                Path.home() / ".campus_motion",
            )

        with tempfile.TemporaryDirectory() as tmp:
            fake_prog = Path(tmp) / "Program Files (x86)"
            edge_bin = fake_prog / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            edge_bin.parent.mkdir(parents=True)
            edge_bin.write_bytes(b"exe")
            with mock.patch.object(sys, "platform", "win32"), \
                    mock.patch.dict(
                        os.environ,
                        {"PROGRAMFILES(X86)": str(fake_prog), "PROGRAMFILES": "", "LOCALAPPDATA": ""},
                        clear=False,
                    ), \
                    mock.patch("analysis.desktop.shutil.which", return_value=None):
                self.assertEqual(desktop_mod.find_chromium_app_binary(), edge_bin)

            with mock.patch.object(sys, "platform", "linux"), \
                    mock.patch("analysis.desktop.shutil.which", return_value=None):
                self.assertIsNone(desktop_mod.find_chromium_app_binary())

    def test_port_scan_picks_first_free_port_and_handles_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = AnalysisService(Path(tmp))
            with mock.patch("analysis.desktop._is_tcp_port_in_use", side_effect=lambda h, p: p == 8000):
                srv = desktop_mod._bind_server_with_port_scan(svc, "127.0.0.1", None)
                try:
                    self.assertGreaterEqual(srv.server_address[1], 8001)
                    self.assertLessEqual(srv.server_address[1], 8020)
                finally:
                    srv.server_close()

            with mock.patch("analysis.desktop._is_tcp_port_in_use", return_value=True):
                with self.assertRaises(RuntimeError):
                    desktop_mod._bind_server_with_port_scan(svc, "127.0.0.1", None)

    def test_desktop_app_mode_single_instance_and_shutdown_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "appdata"
            runs_dir = root / "runs"
            runs_dir.mkdir()
            calib_file = root / "calib.yaml"
            RoadCalibrator(
                [(200.0, 120.0), (600.0, 120.0), (760.0, 520.0), (40.0, 520.0)],
                8.0,
                24.0,
            ).to_yaml(calib_file)

            fake_edge = root / "msedge.exe"
            fake_edge.write_bytes(b"edge")

            launched_cmds = []
            proc_started = threading.Event()

            class _FakeProc:
                def __init__(self, cmd):
                    launched_cmds.append(cmd)
                    self.returncode = None
                    self.terminated = False
                    proc_started.set()

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.terminated = True
                    self.returncode = 0

                def wait(self, timeout=None):
                    return 0

            with mock.patch("analysis.desktop.find_chromium_app_binary", return_value=fake_edge), \
                    mock.patch("analysis.desktop.subprocess.Popen", side_effect=lambda cmd, **_kw: _FakeProc(cmd)):
                host_result = {}

                def _run_host():
                    host_result["rc"] = desktop_mod.run_desktop(
                        host="127.0.0.1",
                        port=0,
                        runs_root=runs_dir,
                        default_calib=calib_file,
                        app_data_dir=app_dir,
                    )

                th = threading.Thread(target=_run_host, daemon=True)
                th.start()

                self.assertTrue(proc_started.wait(timeout=5.0))
                inst_path = app_dir / "instance.json"
                self.assertTrue(inst_path.is_file())
                inst_data = json.loads(inst_path.read_text(encoding="utf-8"))
                base_url = inst_data["url"].rstrip("/")

                # 校验 App-Mode CLI 启动参数完整。
                self.assertEqual(len(launched_cmds), 1)
                cmd = launched_cmds[0]
                self.assertEqual(cmd[0], str(fake_edge))
                self.assertEqual(cmd[1], f"--app={inst_data['url']}")
                self.assertTrue(cmd[2].startswith("--user-data-dir="))
                self.assertIn("--window-size=1440,920", cmd)
                self.assertIn("--no-first-run", cmd)
                self.assertIn("--no-default-browser-check", cmd)

                # 校验后端已开启 desktop_mode=True 与 allow_shutdown=True。
                st_info, info = _http_json(f"{base_url}/api/system/info")
                self.assertEqual(st_info, 200)
                self.assertTrue(info["desktop_mode"])
                self.assertTrue(info["allow_shutdown"])

                # 模拟第二次启动：命中单实例，打开已有 URL 后立即退出 0，不新起服务。
                opened_urls = []
                with mock.patch("analysis.desktop.webbrowser.open", side_effect=lambda u: opened_urls.append(u) or True):
                    rc_second = desktop_mod.run_desktop(
                        host="127.0.0.1",
                        port=0,
                        runs_root=runs_dir,
                        default_calib=calib_file,
                        browser_only=True,
                        app_data_dir=app_dir,
                    )
                self.assertEqual(rc_second, 0)
                self.assertEqual(opened_urls, [inst_data["url"]])
                self.assertTrue(inst_path.is_file())

                # 发送关停请求，验证宿主有序关闭、清理 instance.json 并写入日志。
                st_shut, shut = _http_json(f"{base_url}/api/system/shutdown", method="POST", payload={})
                self.assertEqual(st_shut, 200)
                self.assertEqual(shut, {"status": "shutting_down"})

                th.join(timeout=5.0)
                self.assertFalse(th.is_alive())
                self.assertEqual(host_result.get("rc"), 0)
                self.assertFalse(inst_path.exists())
                self.assertTrue((app_dir / "desktop.log").is_file())

    def test_desktop_browser_exit_and_stale_instance_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_dir = root / "appdata"
            app_dir.mkdir(parents=True)
            runs_dir = root / "runs"
            runs_dir.mkdir()
            calib_file = root / "calib.yaml"
            RoadCalibrator(
                [(200.0, 120.0), (600.0, 120.0), (760.0, 520.0), (40.0, 520.0)],
                8.0,
                24.0,
            ).to_yaml(calib_file)

            # 写入指向死端口的陈旧 instance.json，验证会自动清理并启动新实例。
            stale_inst = app_dir / "instance.json"
            stale_inst.write_text(
                json.dumps({"pid": 999999, "host": "127.0.0.1", "port": 1, "url": "http://127.0.0.1:1/"}),
                encoding="utf-8",
            )

            class _ExitQuickProc:
                def __init__(self):
                    self.calls = 0
                    self.returncode = 0

                def poll(self):
                    self.calls += 1
                    return 0 if self.calls >= 2 else None

                def terminate(self):
                    return None

                def wait(self, timeout=None):
                    return 0

            fake_edge = root / "msedge.exe"
            fake_edge.write_bytes(b"edge")
            with mock.patch("analysis.desktop.find_chromium_app_binary", return_value=fake_edge), \
                    mock.patch("analysis.desktop.subprocess.Popen", return_value=_ExitQuickProc()):
                rc = desktop_mod.run_desktop(
                    host="127.0.0.1",
                    port=0,
                    runs_root=runs_dir,
                    default_calib=calib_file,
                    app_data_dir=app_dir,
                )
            self.assertEqual(rc, 0)
            self.assertFalse(stale_inst.exists())

            # --browser-only + --no-open 回退 CLI 校验。
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                desktop_mod.main(["--host", "   "])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                desktop_mod.main(["--port", "70000"])

    def test_bind_server_rejects_already_occupied_explicit_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calib_file = root / "calib.yaml"
            RoadCalibrator(
                [(200.0, 120.0), (600.0, 120.0), (760.0, 520.0), (40.0, 520.0)],
                8.0,
                24.0,
            ).to_yaml(calib_file)
            service = AnalysisService(runs_root=root / "runs", default_calib=calib_file)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
                occupied.bind(("127.0.0.1", 0))
                occupied.listen(1)
                occupied_port = int(occupied.getsockname()[1])
                with self.assertRaises(OSError):
                    desktop_mod._bind_server_with_port_scan(service, "127.0.0.1", occupied_port)

    def test_desktop_startup_lock_serializes_concurrent_checks(self):
        import threading
        import time as time_mod

        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "instance.lock"
            active = {"cnt": 0, "max_active": 0}
            guard = threading.Lock()

            def _worker():
                with desktop_mod._startup_lock(lock_path, timeout_s=5.0):
                    with guard:
                        active["cnt"] += 1
                        active["max_active"] = max(active["max_active"], active["cnt"])
                    time_mod.sleep(0.05)
                    with guard:
                        active["cnt"] -= 1

            t1 = threading.Thread(target=_worker)
            t2 = threading.Thread(target=_worker)
            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)
            self.assertEqual(active["max_active"], 1)

    def test_frontend_web_contract_and_zero_emoji(self):
        web_dir = REPO_ROOT / "analysis" / "web"
        html = (web_dir / "index.html").read_text(encoding="utf-8")
        styles = (web_dir / "assets" / "styles.css").read_text(encoding="utf-8")
        app_js = (web_dir / "assets" / "app.js").read_text(encoding="utf-8")
        motion_js = (web_dir / "assets" / "motion.js").read_text(encoding="utf-8")
        scenarios_py = (REPO_ROOT / "analysis" / "scenarios.py").read_text(encoding="utf-8")

        self.assertIn('id="btn-replay-sample" class="btn btn-ghost" hidden', html)
        self.assertIn('id="replay-sim-canvas"', html)
        self.assertIn('aria-controls="conflicts-table-region"', html)
        self.assertIn('aria-controls="studio-panel-calib"', html)
        self.assertIn("1.80 s", html)
        self.assertNotIn("1.50 s", html)
        self.assertIn('id="sim-speed-model"', html)
        self.assertIn('id="sim-conflict-model"', html)
        self.assertIn('id="calib-corresp-wrap"', html)
        self.assertIn('id="calib-corresp-list"', html)
        self.assertEqual(html.count('class="node-box"'), 5)
        self.assertEqual(html.count('class="node-focus-ring"'), 5)
        self.assertEqual(len(re.findall(r'class="pipeline-node(?:[\s"])', html)), 5)

        self.assertNotIn("@property --beam-angle", styles)
        self.assertNotIn("beam-spin", styles)
        self.assertIn(":focus:not(:focus-visible)", styles)
        self.assertIn(".pipeline-node:focus,", styles)
        self.assertIn(".pipeline-node:focus-visible", styles)
        self.assertIn(".pipeline-node .node-focus-ring", styles)
        self.assertIn("--surface-hairline-top: inset 0 1px 0 var(--surface-hairline-top-color);", styles)
        self.assertIn("light-dark(rgba(255, 255, 255, 0.72), rgba(255, 255, 255, 0.06))", styles)
        self.assertIn(".table-jump-btn", styles)

        self.assertIn("widgets.roadTwin.setConfig", app_js)
        self.assertIn("getPipelineStageDetail", app_js)
        self.assertIn("getActivePipelineConfig", app_js)
        self.assertIn("ensureReplayFrameWindow", app_js)
        self.assertIn("makeJumpRowHeader", app_js)
        self.assertIn("dom.replayFrameImg.onerror", app_js)
        self.assertIn("clearHistoryDisplay", app_js)
        self.assertIn("isPlayableRun", app_js)
        self.assertIn("bindTablistKeyboard", app_js)
        self.assertIn("simSpeedModel", app_js)
        self.assertIn("simConflictModel", app_js)
        self.assertIn("calibGroundPoints", app_js)

        self.assertIn("CampusRoadTwinCanvas.prototype.setConfig", motion_js)
        self.assertIn("computeCvmCircleTtc", motion_js)
        self.assertIn("SCENARIO_CYCLE_S", motion_js)
        self.assertIn("rolloutVehicleTrajectory", motion_js)
        self.assertIn("resolveHeadingRad", motion_js)

        emoji_re = re.compile(
            "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0000FE0F]"
        )
        for rel, content in (
            ("index.html", html),
            ("styles.css", styles),
            ("app.js", app_js),
            ("motion.js", motion_js),
            ("scenarios.py", scenarios_py),
        ):
            self.assertIsNone(emoji_re.search(content), f"{rel} 发现未清理的 emoji 字符")

    def test_motion_js_scenarios_kinematics_and_replay_contract(self):
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node runtime not available")

        motion_path = REPO_ROOT / "analysis" / "web" / "assets" / "motion.js"
        node_script = r"""
const fs = require("fs");
const vm = require("vm");

const motionPath = process.env.MOTION_JS_PATH;
const code = fs.readFileSync(motionPath, "utf8");
const noop = () => {};
const ctx = new Proxy(
  {
    measureText: (s) => ({ width: String(s || "").length * 6.5 }),
    createLinearGradient: () => ({ addColorStop: noop })
  },
  {
    get(target, prop) {
      return prop in target ? target[prop] : noop;
    }
  }
);
const mkEl = () => ({
  style: {},
  classList: { add: noop, remove: noop, contains: () => false },
  setAttribute: noop,
  getAttribute: () => null,
  appendChild: noop,
  removeChild: noop,
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener: noop,
  removeEventListener: noop,
  getBoundingClientRect: () => ({ width: 640, height: 380 }),
  getContext: () => ctx,
  width: 640,
  height: 380
});
const sandbox = {
  performance: { now: () => 0 },
  matchMedia: () => ({ matches: false, addEventListener: noop, removeEventListener: noop }),
  getComputedStyle: () => ({ getPropertyValue: () => "", colorScheme: "dark" }),
  document: {
    documentElement: { getAttribute: () => "dark", classList: { contains: () => false } },
    body: { appendChild: noop, removeChild: noop },
    createElement: mkEl,
    createElementNS: mkEl
  },
  requestAnimationFrame: () => 1,
  cancelAnimationFrame: noop,
  setInterval: () => 1,
  clearInterval: noop,
  setTimeout: () => 1,
  clearTimeout: noop
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.runInNewContext(code, sandbox);

const CM = sandbox.CampusMotion;
if (!CM || typeof CM._buildScenarioState !== "function" || typeof CM._computeCpaAndTtc !== "function") {
  throw new Error("Missing expected CampusMotion exports");
}

function obbClearance(veh, vru) {
  const dx = vru.x - veh.x;
  const dy = vru.y - veh.y;
  const c = Math.cos(veh.headingRad);
  const s = Math.sin(veh.headingRad);
  const lx = dx * c + dy * s;
  const ly = -dx * s + dy * c;
  const halfL = (veh.lengthM || 4.5) * 0.5;
  const halfW = (veh.widthM || 1.9) * 0.5;
  const cx = Math.max(-halfL, Math.min(halfL, lx));
  const cy = Math.max(-halfW, Math.min(halfW, ly));
  return Math.hypot(lx - cx, ly - cy) - (vru.radiusM || 0.42);
}

const scenarios = ["crosswalk_braking", "cyclist_vru_cutin", "ctrv_curve_yield"];
const scenarioSummary = {};
const cycleSteps = Math.round(CM._SCENARIO_CYCLE_S * 25);

for (const sc of scenarios) {
  const twin = new CM.CampusRoadTwinCanvas(mkEl(), {
    config: { speedModel: "ctrv", conflictModel: "obb_ssm", stitching: true, cyclistVru: true }
  });
  twin.setScenario(sc);
  twin._petCells = Object.create(null);
  twin._lastPetTimeS = null;
  twin._vehTrails = Object.create(null);

  let minClear = Infinity;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let parkClips = 0;
  let l2Steps = 0;
  let l3Steps = 0;
  let sawL3 = false;
  let departureAlerts = 0;
  let lastRisk = -1;
  let endActiveVeh = 0;

  for (let i = 0; i < cycleSteps; i += 1) {
    const t = i * 0.04;
    const st = CM._buildScenarioState(sc, t, twin._config);
    twin._evaluateSimulation(st, t, 0);
    const snap = twin._latestSimSnapshot;
    lastRisk = snap.riskLevel;
    if (snap.riskLevel === 2) l2Steps += 1;
    if (snap.riskLevel === 3) {
      l3Steps += 1;
      sawL3 = true;
    }
    if (sawL3 && t >= 5.6 && snap.riskLevel >= 2) {
      departureAlerts += 1;
    }
    if (t >= CM._SCENARIO_CYCLE_S - 0.12) {
      endActiveVeh = st.vehicles.filter((v) => !v.inParkingZone).length;
    }

    for (const v of st.vehicles) {
      minX = Math.min(minX, v.x);
      maxX = Math.max(maxX, v.x);
      minY = Math.min(minY, v.y);
      if (!v.inParkingZone) {
        const c = Math.cos(v.headingRad);
        const s = Math.sin(v.headingRad);
        const hl = (v.lengthM || 4.5) * 0.5;
        const hw = (v.widthM || 1.9) * 0.5;
        for (const [lx, ly] of [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [0, 0]]) {
          const gx = v.x + lx * c - ly * s;
          const gy = v.y + lx * s + ly * c;
          for (const pz of st.parkingZones) {
            const xs = pz.map((p) => p[0]);
            const ys = pz.map((p) => p[1]);
            if (
              gx >= Math.min(...xs) &&
              gx <= Math.max(...xs) &&
              gy >= Math.min(...ys) &&
              gy <= Math.max(...ys)
            ) {
              parkClips += 1;
            }
          }
        }
        for (const u of st.vrus) {
          minClear = Math.min(minClear, obbClearance(v, u));
        }
      }
    }
  }
  twin.destroy();

  if (minClear < 0.30) {
    throw new Error(`${sc}: collision or tight pass minClear=${minClear}`);
  }
  if (minX < -2.9 || maxX > 2.9 || minY < 0.4) {
    throw new Error(`${sc}: out of road bounds x=[${minX},${maxX}] minY=${minY}`);
  }
  if (endActiveVeh !== 0) {
    throw new Error(`${sc}: expected ego vehicle off-screen before wrap, got ${endActiveVeh}`);
  }
  if (parkClips !== 0) {
    throw new Error(`${sc}: active vehicle clipped parking zone (${parkClips} hits)`);
  }
  if (l2Steps < 8 || l3Steps < 10) {
    throw new Error(`${sc}: insufficient caution/critical steps l2=${l2Steps} l3=${l3Steps}`);
  }
  if (departureAlerts !== 0 || lastRisk !== 0) {
    throw new Error(`${sc}: unexpected departure alert depAlerts=${departureAlerts} lastRisk=${lastRisk}`);
  }
  scenarioSummary[sc] = { minClear, l2Steps, l3Steps, lastRisk, endActiveVeh };
}

// 验证 predictCtrvArc 与 computeCpaAndTtc 使用完全同一的外推点（无 0.72 分裂）
const sampleVeh = { x: 0, y: 2, vx: 0, vy: 6, accelMs2: -3.0, headingRad: Math.PI * 0.5, yawRate: 0, lengthM: 4.5, widthM: 1.9 };
const arc = CM._predictCtrvArc(sampleVeh, 2.5, 50, "ctrv");
const pt2s = arc[40]; // t = 2.0s
const stationaryVruAt2s = { x: pt2s.x, y: pt2s.y + 2.25, vx: 0, vy: 0, radiusM: 0.40 };
const cpaAligned = CM._computeCpaAndTtc(sampleVeh, stationaryVruAt2s, 3.5, "ctrv");
if (cpaAligned.ttcSeconds === null || cpaAligned.ttcSeconds > 2.05) {
  throw new Error(`Expected aligned arc & TTC <= 2.05s, got ${JSON.stringify(cpaAligned)}`);
}

// 验证 PET 网格：同步占用同格返回 null，先出后进才生成正值 PET
const petTwin = new CM.CampusRoadTwinCanvas(mkEl(), {});
petTwin._petCells = Object.create(null);
petTwin._lastPetTimeS = null;
petTwin._recordPetGrid(
  [{ trackId: 1, x: 0, y: 5, speedKmh: 20, headingRad: 0, lengthM: 4.0, widthM: 2.0, inParkingZone: false }],
  [{ trackId: 2, x: 0.1, y: 5.1 }],
  1.0
);
const overlapPets = Object.values(petTwin._petCells).map((c) => c.petS).filter((v) => v !== null);
if (overlapPets.length !== 0) {
  throw new Error(`Simultaneous PET must be null, got ${JSON.stringify(overlapPets)}`);
}
petTwin._recordPetGrid(
  [{ trackId: 1, x: 10, y: 5, speedKmh: 20, headingRad: 0, lengthM: 4.0, widthM: 2.0, inParkingZone: false }],
  [],
  1.4
);
petTwin._recordPetGrid([], [{ trackId: 2, x: 0.1, y: 5.1 }], 1.8);
const exitPets = Object.values(petTwin._petCells).map((c) => c.petS).filter((v) => v !== null);
if (exitPets.length === 0 || Math.abs(exitPets[0] - 0.8) > 1e-6) {
  throw new Error(`Expected PET=0.8s after exit, got ${JSON.stringify(exitPets)}`);
}
petTwin.destroy();

// 验证 headingDeg=0 (headingRad=0) 在 BEV _renderCurrent 中的 rotate 角精确为 0（不被拧成 Math.PI/2）
const rotAngles = [];
const rotEl = mkEl();
const baseCtx = rotEl.getContext();
const spyCtx = new Proxy(baseCtx, {
  get(t, p) {
    if (p === "rotate") return (ang) => { rotAngles.push(ang); };
    return t[p];
  }
});
rotEl.getContext = () => spyCtx;
const h0Twin = new CM.CampusRoadTwinCanvas(rotEl, {});
h0Twin.setMode("replay");
h0Twin.setReplayFrame({
  frameIndex: 0,
  timestampS: 0.0,
  vehicles: [{ track_id: 1, category: "car", x_m: 0, y_m: 3, vx_ms: 4, vy_ms: 0, speed_kmh: 14.4 }],
  vrus: [],
  conflicts: []
});
if (Math.abs(h0Twin._currentRenderState.vehicles[0].headingRad - 0) > 1e-9 || !rotAngles.some((a) => Math.abs(a) < 1e-9)) {
  throw new Error(`Expected 0 rad rotation for headingDeg=0, got angles=${JSON.stringify(rotAngles)}`);
}
h0Twin.destroy();

const replayTwin = new CM.CampusRoadTwinCanvas(mkEl(), {});
replayTwin.setMode("replay");
replayTwin.setReplayFrame({
  frameIndex: 5,
  timestampS: 0.5,
  vehicles: [{ track_id: 1, category: "car", x_m: 0, y_m: 3, vx_ms: 0, vy_ms: 8, speed_kmh: 28.8 }],
  vrus: [{ track_id: 2, category: "pedestrian", x_m: 0, y_m: 8, vx_ms: 0, vy_ms: 0 }],
  conflicts: [{ veh_track_id: 1, ped_track_id: 2, risk_level: 3, ttc_seconds: 0.62 }]
});
const replayCriticalRisk = replayTwin._currentRenderState.maxRisk;
if (!(replayCriticalRisk > 0)) {
  throw new Error(`setReplayFrame expected non-zero maxRisk, got ${replayCriticalRisk}`);
}
replayTwin.setReplayFrame({
  frameIndex: 6,
  timestampS: 0.6,
  vehicles: [{ track_id: 1, category: "car", x_m: 0, y_m: 3, vx_ms: 0, vy_ms: 4, speed_kmh: 14.4 }],
  vrus: [{ track_id: 2, category: "pedestrian", x_m: 2.5, y_m: 12, vx_ms: 0, vy_ms: 0 }],
  conflicts: []
});
const replaySafeRisk = replayTwin._currentRenderState.maxRisk;
replayTwin.destroy();

const modelProfiles = {};
for (const sm of ["ctrv", "kalman", "ema"]) {
  const modelTwin = new CM.CampusRoadTwinCanvas(mkEl(), {
    config: { speedModel: sm, conflictModel: "obb_ssm", stitching: true, cyclistVru: true }
  });
  modelTwin.setScenario("ctrv_curve_yield");
  modelTwin._petCells = Object.create(null);
  let l2 = 0;
  let l3 = 0;
  const earlyTtc = [];
  for (let i = 0; i < 160; i += 1) {
    const t = i * 0.05;
    const st = CM._buildScenarioState("ctrv_curve_yield", t, modelTwin._config);
    modelTwin._evaluateSimulation(st, t, 0);
    const snap = modelTwin._latestSimSnapshot;
    if (snap.riskLevel === 2) l2 += 1;
    if (snap.riskLevel === 3) l3 += 1;
    if (i < 25) earlyTtc.push(snap.minTtcS);
  }
  modelTwin.destroy();
  const st055 = CM._buildScenarioState("ctrv_curve_yield", 0.55, { speedModel: sm });
  const cpa055 = CM._computeCpaAndTtc(st055.vehicles[0], st055.vrus[0], 3.5, sm);
  modelProfiles[sm] = JSON.stringify({
    l2,
    l3,
    earlyTtc,
    minDist055: Number(cpa055.minDistanceM.toFixed(4)),
    ttc055: cpa055.ttcSeconds
  });
}

if (
  modelProfiles.ctrv === modelProfiles.kalman ||
  modelProfiles.ctrv === modelProfiles.ema ||
  modelProfiles.kalman === modelProfiles.ema
) {
  throw new Error(`Expected distinct ctrv/kalman/ema behavior: ${JSON.stringify(modelProfiles)}`);
}

console.log(JSON.stringify({ scenarioSummary, replayCriticalRisk, replaySafeRisk, modelProfiles }));
"""
        env = os.environ.copy()
        env["MOTION_JS_PATH"] = str(motion_path)
        proc = subprocess.run(
            [node_bin, "-"],
            input=node_script,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=30,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"Node motion.js check failed:\nstdout={proc.stdout}\nstderr={proc.stderr}",
        )
        summary = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(summary["replayCriticalRisk"], 3)
        self.assertEqual(summary["replaySafeRisk"], 0)
        self.assertEqual(set(summary["scenarioSummary"].keys()), {"crosswalk_braking", "cyclist_vru_cutin", "ctrv_curve_yield"})

    def test_app_js_replay_polling_calibration_and_table_contracts(self):
        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node runtime not available")

        app_path = REPO_ROOT / "analysis" / "web" / "assets" / "app.js"
        node_script = r"""
const fs = require("fs");
const vm = require("vm");

function makeNode(tag) {
  const listeners = {};
  const attrs = {};
  const classes = new Set();
  const node = {
    tagName: String(tag || "div").toUpperCase(),
    id: "",
    textContent: "",
    value: "",
    checked: true,
    disabled: false,
    hidden: false,
    title: "",
    className: "",
    dataset: {},
    children: [],
    style: { setProperty: () => {} },
    classList: {
      add: (...c) => c.forEach((x) => classes.add(x)),
      remove: (...c) => c.forEach((x) => classes.delete(x)),
      toggle: (c, f) => { if (f === undefined ? !classes.has(c) : f) classes.add(c); else classes.delete(c); },
      contains: (c) => classes.has(c)
    },
    setAttribute: (k, v) => { attrs[k] = String(v); },
    getAttribute: (k) => (k in attrs ? attrs[k] : null),
    removeAttribute: (k) => { delete attrs[k]; },
    appendChild: (ch) => {
      if (ch && ch._isFrag) node.children.push(...ch.children);
      else node.children.push(ch);
      return ch;
    },
    removeChild: (ch) => {
      node.children = node.children.filter((x) => x !== ch);
    },
    addEventListener: (ev, fn) => { (listeners[ev] = listeners[ev] || []).push(fn); },
    dispatchEvent: (ev) => { (listeners[ev.type] || []).forEach((fn) => fn(ev)); },
    querySelectorAll: () => [],
    closest: () => null,
    focus: () => {}
  };
  return node;
}

const byId = {};
const getEl = (id) => (byId[id] = byId[id] || Object.assign(makeNode("div"), { id }));

const simSpd = getEl("sim-speed-model"); simSpd.value = "ctrv";
const simConf = getEl("sim-conflict-model"); simConf.value = "obb_ssm";
const togStitch = getEl("toggle-stitching"); togStitch.checked = true;
const togVru = getEl("toggle-cyclist-vru"); togVru.checked = true;
for (const nid of [
  "sidebar-nav", "view-title", "view-subtitle", "replay-scrubber",
  "replay-frame-label", "replay-media-caption", "replay-frame-img", "replay-video",
  "replay-sim-canvas", "twin-counts", "kpi-frames-meta", "kpi-speed-meta",
  "live-mjpeg-img", "live-media-caption", "live-job-status-text", "btn-stop-job",
  "badge-speed-model", "badge-conflict-model", "badge-vru-model", "twin-mode-label",
  "calib-valid-badge", "calib-rmse-val", "calib-inliers-val", "btn-calib-preview",
  "calib-width-m", "calib-length-m", "calib-name", "calib-use-ransac", "calib-error",
  "conflicts-tbody", "events-tbody", "speeds-tbody",
  "conflicts-table-region", "events-table-region", "speeds-table-region", "history-table-meta",
  "viewport-sim-panel", "viewport-replay-panel", "viewport-live-panel",
  "sim-controls", "replay-controls", "live-controls", "scenario-select",
  "viewport-mode-badge", "sidebar-mode-text", "sidebar-target-text", "btn-twin-pause",
  "pipeline-stage-detail", "driver-alert-banner", "driver-alert-list",
  "ped-alert-banner", "ped-alert-list", "alert-log-list", "alert-log-count"
]) {
  getEl(nid);
}
byId["calib-width-m"].value = "6";
byId["calib-length-m"].value = "15";
byId["calib-name"].value = "campus";

let fetchRoutes = {};
const fetchCalls = [];
const sandbox = {
  console,
  document: {
    readyState: "loading",
    documentElement: makeNode("html"),
    getElementById: getEl,
    querySelector: (sel) => (sel.startsWith("#") ? getEl(sel.slice(1)) : makeNode("div")),
    querySelectorAll: () => [],
    createElement: makeNode,
    createDocumentFragment: () => ({ _isFrag: true, children: [], appendChild(c) { this.children.push(c); } }),
    addEventListener: () => {}
  },
  localStorage: { getItem: () => null, setItem: () => {} },
  navigator: {},
  requestAnimationFrame: (fn) => { fn(); return 1; },
  cancelAnimationFrame: () => {},
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  fetch: (url) => {
    fetchCalls.push(url);
    const handler = fetchRoutes[url];
    if (typeof handler === "function") return handler(url);
    return Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve("{}") });
  },
  CampusMotion: {
    RollingCounter: function () { this.val = null; this.setValue = (v) => { this.val = v; }; },
    AnimatedSignalPipeline: function () { this.setConfig = () => {}; this.setRiskLevel = () => {}; this.setPaused = () => {}; },
    CampusRoadTwinCanvas: function () {
      this.frames = [];
      this.setConfig = () => {};
      this.setMode = () => {};
      this.setScenario = () => {};
      this.setPaused = () => {};
      this.setReplayFrame = (f) => { this.frames.push(f); };
      this.refreshTheme = () => {};
    },
    SimulatedCameraCanvas: function () { this.renderState = () => {}; this.refreshTheme = () => {}; },
    SpeedTimelineChart: function () { this.setData = () => {}; this.setActiveFrame = () => {}; },
    CalibrationCanvas: function () {
      this.getState = () => ({
        imagePoints: [[0,0],[100,0],[100,100],[0,100]],
        parkingZonesImage: [],
        currentParkingDraft: []
      });
    },
    AlertAudioSynth: function () { this.trigger = () => {}; }
  }
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.runInNewContext(fs.readFileSync(process.env.APP_JS_PATH, "utf8"), sandbox);

const hook = sandbox.window.__CampusMotionAppTestHook;
if (!hook) throw new Error("Missing __CampusMotionAppTestHook");

// 初始化引用 dom 并设置轻量 widgets
const dom = hook.dom;
dom.badgeSpeedModel = byId["badge-speed-model"];
dom.badgeConflictModel = byId["badge-conflict-model"];
dom.badgeVruModel = byId["badge-vru-model"];
dom.simSpeedModel = simSpd;
dom.simConflictModel = simConf;
dom.toggleStitching = togStitch;
dom.toggleCyclistVru = togVru;
dom.replayScrubber = byId["replay-scrubber"];
dom.replayFrameLabel = byId["replay-frame-label"];
dom.replayMediaCaption = byId["replay-media-caption"];
dom.replayFrameImg = byId["replay-frame-img"];
dom.replayVideo = byId["replay-video"];
dom.twinCounts = byId["twin-counts"];
dom.twinModeLabel = byId["twin-mode-label"];
dom.kpiFramesMeta = byId["kpi-frames-meta"];
dom.kpiSpeedMeta = byId["kpi-speed-meta"];
dom.calibValidBadge = byId["calib-valid-badge"];
dom.calibRmseVal = byId["calib-rmse-val"];
dom.calibInliersVal = byId["calib-inliers-val"];
dom.btnCalibPreview = byId["btn-calib-preview"];
dom.calibWidthM = byId["calib-width-m"];
dom.calibLengthM = byId["calib-length-m"];
dom.calibName = byId["calib-name"];
dom.calibUseRansac = byId["calib-use-ransac"];
dom.calibError = byId["calib-error"];
dom.conflictsTbody = byId["conflicts-tbody"];
dom.eventsTbody = byId["events-tbody"];
dom.speedsTbody = byId["speeds-tbody"];
dom.conflictsTableRegion = byId["conflicts-table-region"];
dom.eventsTableRegion = byId["events-table-region"];
dom.speedsTableRegion = byId["speeds-table-region"];
dom.historyTableMeta = byId["history-table-meta"];
dom.viewportSimPanel = byId["viewport-sim-panel"];
dom.viewportReplayPanel = byId["viewport-replay-panel"];
dom.viewportLivePanel = byId["viewport-live-panel"];
dom.simControls = byId["sim-controls"];
dom.replayControls = byId["replay-controls"];
dom.liveControls = byId["live-controls"];
dom.scenarioSelect = byId["scenario-select"];
dom.viewportModeBadge = byId["viewport-mode-badge"];
dom.sidebarModeText = byId["sidebar-mode-text"];
dom.sidebarTargetText = byId["sidebar-target-text"];
dom.btnTwinPause = byId["btn-twin-pause"];
dom.pipelineStageDetail = byId["pipeline-stage-detail"];
dom.liveMjpegImg = byId["live-mjpeg-img"];
dom.liveMediaCaption = byId["live-media-caption"];
dom.liveJobStatusText = byId["live-job-status-text"];
dom.btnStopJob = byId["btn-stop-job"];

async function runChecks() {
  // 1) 超过 5,000 行截断回放：先进入加载态并自动请求区间接口，不渲染虚假 0 目标/L0
  const ds = hook.buildRunDataset(
    "run-big",
    { frames_processed: 1500, has_video: false },
    { events: [{ event_id: 1, kind: "speeding", veh_track_id: 1, start_frame: 0, end_frame: 0, peak_risk_level: 1 }] },
    { conflicts: [{ frame_index: 0, timestamp_s: 0.0, assessment: { veh_track_id: 1, risk_level: 0 } }], total_count: 6000, truncated: true },
    { rows: [{ frame_idx: "0", timestamp_s: "0.0", track_id: "1", category: "vehicle", object_kind: "bus", length_m: "10.0", width_m: "2.55", x_m: "1.0", y_m: "2.0", speed_kmh: "18.0", risk_level: "LEVEL_0_SAFE" }], total_rows: 7500, truncated: true }
  );
  hook.state.mode = "replay";
  hook.state.replay.runId = "run-big";
  hook.state.replay.data = ds;
  fetchRoutes["/api/runs/run-big/speeds?frame_start=1000&frame_end=1249&limit=20000"] = () => Promise.resolve({
    ok: true, status: 200,
    text: () => Promise.resolve(JSON.stringify({
      rows: [
        { frame_idx: "1200", timestamp_s: "48.0", track_id: "7", category: "vehicle", object_kind: "bus", length_m: "10.0", width_m: "2.55", x_m: "2.0", y_m: "8.0", speed_kmh: "32.0", risk_level: "LEVEL_3_CRITICAL" },
        { frame_idx: "1200", timestamp_s: "48.0", track_id: "8", category: "vehicle", object_kind: "bicycle", x_m: "2.5", y_m: "9.0", speed_kmh: "12.0", risk_level: "LEVEL_2_CAUTION" }
      ]
    }))
  });
  fetchRoutes["/api/runs/run-big/conflicts?frame_start=1000&frame_end=1249&limit=20000"] = () => Promise.resolve({
    ok: true, status: 200,
    text: () => Promise.resolve(JSON.stringify({
      conflicts: [{ frame_index: 1200, timestamp_s: 48.0, assessment: { veh_track_id: 7, ped_track_id: 8, risk_level: 3, ttc_seconds: 1.1, veh_speed_kmh: 32.0 } }]
    }))
  });

  hook.renderReplayFrame(1200);
  if (dom.twinCounts.textContent.indexOf("加载中") < 0) {
    throw new Error("Expected explicit loading status for unloaded frame 1200, got: " + dom.twinCounts.textContent);
  }
  await hook.ensureReplayFrameWindow(ds, 1200, hook.state.replayReqToken);
  const frame1200 = hook.buildReplayFramePayload(ds, 1200);
  if (frame1200.vehicles.length !== 1 || frame1200.vehicles[0].kind !== "bus" || frame1200.vehicles[0].lengthM !== 10.0) {
    throw new Error("Expected bus fine class & lengthM=10, got: " + JSON.stringify(frame1200.vehicles));
  }
  if (frame1200.vrus.length !== 1 || frame1200.vrus[0].kind !== "bicycle") {
    throw new Error("Expected bicycle routed to VRU, got: " + JSON.stringify(frame1200.vrus));
  }
  if (frame1200.riskLevel !== 3) {
    throw new Error("Expected riskLevel=3 after window merge, got: " + frame1200.riskLevel);
  }

  // 2) 实时模式徽标解耦 & 串口化轮询乱序拦截
  hook.state.liveJob.jobId = "job-1";
  hook.state.liveJob.pollToken = 1;
  hook.state.liveJob.pollSeq = 0;
  hook.state.liveJob.lastAppliedPollSeq = 0;
  hook.state.liveJob.config = { speedModel: "ctrv", conflictModel: "obb_ssm", stitching: true, cyclistVru: true };
  hook.setMonitorMode("live");
  hook.updateAlgorithmBadges();
  if (dom.badgeSpeedModel.textContent !== "CTRV" || dom.badgeConflictModel.textContent !== "OBB-SSM") {
    throw new Error("Expected CTRV/OBB-SSM live badge");
  }
  // 模拟下拉仍保持旧值也不会改写实时徽标
  simSpd.value = "ema";
  hook.updateAlgorithmBadges();
  if (dom.badgeSpeedModel.textContent !== "CTRV") {
    throw new Error("Simulation select leaked into live badge: " + dom.badgeSpeedModel.textContent);
  }

  // 3) 乱序轮询 B(120) 先返回、A(100) 后返回，状态保持 120 帧
  let resA;
  const pA = new Promise((r) => { resA = r; });
  fetchRoutes["/api/jobs/job-1"] = (url) => {
    if (hook.state.liveJob.pollSeq === 1) return pA;
    return Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve(JSON.stringify({ job_id: "job-1", status: "running", frames_processed: 120, latest_timestamp_s: 4.8 })) });
  };
  fetchRoutes["/api/jobs/job-1/alerts?since=0&limit=100"] = () => Promise.resolve({
    ok: true, status: 200, text: () => Promise.resolve(JSON.stringify({ alerts: [] }))
  });
  const callA = hook.pollLiveJobState(); // seq=1
  const callB = hook.pollLiveJobState(); // seq=2
  await callB;
  resA({ ok: true, status: 200, text: () => Promise.resolve(JSON.stringify({ job_id: "job-1", status: "running", frames_processed: 100, latest_timestamp_s: 4.0 })) });
  await callA;
  if (hook.state.liveJob.lastFramesProcessed !== 120) {
    throw new Error("Stale poll overwrote frame count: " + hook.state.liveJob.lastFramesProcessed);
  }

  // 4) 标定预览期间清空点位 -> 旧响应必须被版本护卫丢弃
  hook.state.calibState = {
    imagePoints: [[0,0],[100,0],[100,100],[0,100]],
    parkingZonesImage: [],
    currentParkingDraft: []
  };
  hook.state.calibGroundPoints = [[0,0],[6,0],[6,15],[0,15]];
  let resCalib;
  fetchRoutes["/api/calibration/preview"] = () => new Promise((r) => { resCalib = r; });
  const prevProm = hook.previewCalibration();
  hook.resetCalibValidationBadge();
  resCalib({ ok: true, status: 200, text: () => Promise.resolve(JSON.stringify({ valid: true, reprojection_rmse_m: 0.01, inlier_count: 4 })) });
  await prevProm;
  if (dom.calibValidBadge.textContent !== "待校验") {
    throw new Error("Stale calibration response applied after reset: " + dom.calibValidBadge.textContent);
  }

  // 5) 历史表格渲染原生可聚焦 .table-jump-btn
  hook.renderHistoryTables(ds);
  const cBtn = dom.conflictsTbody.children[0] && dom.conflictsTbody.children[0].children[0] && dom.conflictsTbody.children[0].children[0].children[0];
  const eBtn = dom.eventsTbody.children[0] && dom.eventsTbody.children[0].children[0] && dom.eventsTbody.children[0].children[0].children[0];
  const sBtn = dom.speedsTbody.children[0] && dom.speedsTbody.children[0].children[0] && dom.speedsTbody.children[0].children[0].children[0];
  if (!cBtn || cBtn.tagName !== "BUTTON" || !cBtn.className.includes("table-jump-btn")) {
    throw new Error("Missing .table-jump-btn in conflicts table");
  }
  if (!eBtn || eBtn.tagName !== "BUTTON" || !eBtn.className.includes("table-jump-btn")) {
    throw new Error("Missing .table-jump-btn in events table");
  }
  if (!sBtn || sBtn.tagName !== "BUTTON" || !sBtn.className.includes("table-jump-btn")) {
    throw new Error("Missing .table-jump-btn in speeds table");
  }
  console.log("APP_JS_CONTRACTS_OK");
}

runChecks().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
        env = os.environ.copy()
        env["APP_JS_PATH"] = str(app_path)
        proc = subprocess.run(
            [node_bin, "-"],
            input=node_script,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=30,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"Node app.js check failed:\nstdout={proc.stdout}\nstderr={proc.stderr}",
        )
        self.assertIn("APP_JS_CONTRACTS_OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
