# -*- coding: utf-8 -*-
"""实时后端服务、LatestFrameCapture 丢旧帧、双端告警、GUI 多点/停车区标定及管线集成测试。"""

from __future__ import annotations

import builtins
import contextlib
import csv
import http.client
import inspect
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from analysis.__main__ import main as analysis_main
from analysis.ablation import run_default_ablation
from analysis.calibration import RoadCalibrator
from analysis.conflict import ConflictAssessment, RiskLevel
from analysis.interactive_calibrator import main as calib_main, run_interactive_calibration
from analysis.pipeline import ConflictPerceptionPipeline
from analysis.schema import (
    SchemaValidationError,
    load_json_strict,
    validate_ablation,
    validate_configuration,
    validate_run,
)
from analysis.service import (
    _ALLOWED_JOB_REQUEST_KEYS,
    AnalysisService,
    DEFAULT_WEB_ROOT,
    LatestFrameCapture,
    create_fastapi_app,
    create_http_server,
    format_dual_end_alerts,
    main as service_main,
)


def _sample_calibrator() -> RoadCalibrator:
    return RoadCalibrator(
        image_points=[(200.0, 120.0), (600.0, 120.0), (760.0, 520.0), (40.0, 520.0)],
        width_m=8.0,
        length_m=24.0,
        name="svc_gui_test_road",
    )


def _write_dummy_video(path: Path, n_frames: int = 4, fps: float = 10.0, size=(800, 600)) -> None:
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), (i * 30) % 255, dtype=np.uint8))
    writer.release()


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
        with urlopen(req, timeout=10.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


class TestLatestFrameCaptureAndDualAlerts(unittest.TestCase):
    def test_latest_frame_capture_drops_stale_frames_and_releases_cleanly(self):
        # 构造先快速产出 5 帧并等待消费信号后再产出最终帧、最后 EOF 的模拟采集器。
        allow_tail = threading.Event()
        frames = [np.full((16, 16, 3), idx, dtype=np.uint8) for idx in range(6)]
        read_idx = {"i": 0}

        def _fake_read():
            idx = read_idx["i"]
            read_idx["i"] += 1
            if idx < 4:
                return True, frames[idx]
            if idx == 4:
                allow_tail.wait(timeout=2.0)
                return True, frames[idx]
            return False, None

        fake_cap = mock.Mock()
        fake_cap.isOpened.return_value = True
        fake_cap.get.return_value = 25.0
        fake_cap.read.side_effect = _fake_read

        lfc = LatestFrameCapture(fake_cap, read_timeout_s=2.0)
        try:
            # 等待后台线程快速抓取完前 4 帧（索引 0..3），至少丢弃 3 帧旧帧只保留最新第 3 帧。
            deadline = time.monotonic() + 2.0
            while lfc.dropped_frames < 3 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertGreaterEqual(lfc.dropped_frames, 3)
            self.assertTrue(lfc.isOpened())
            self.assertEqual(lfc.get(cv2.CAP_PROP_FPS), 25.0)

            ok, latest = lfc.read()
            self.assertTrue(ok)
            self.assertEqual(int(latest[0, 0, 0]), 3)

            allow_tail.set()
            ok2, tail_frame = lfc.read()
            self.assertTrue(ok2)
            self.assertEqual(int(tail_frame[0, 0, 0]), 4)

            ok3, eof_frame = lfc.read()
            self.assertFalse(ok3)
            self.assertIsNone(eof_frame)
        finally:
            lfc.release()
            # 重复释放必须幂等且底层仅释放一次。
            lfc.release()

        self.assertFalse(lfc.isOpened())
        fake_cap.release.assert_called_once()

        for bad_arg in ("", "   ", True):
            with self.assertRaises(ValueError):
                LatestFrameCapture(bad_arg)
        with self.assertRaises(ValueError):
            LatestFrameCapture(fake_cap, read_timeout_s=0.0)

    def test_format_dual_end_alerts_splits_driver_and_pedestrian(self):
        safe_c = ConflictAssessment(
            veh_track_id=1,
            ped_track_id=None,
            risk_level=RiskLevel.LEVEL_0_SAFE,
            risk_desc="safe",
            veh_speed_kmh=15.0,
            ped_speed_ms=0.0,
            current_distance_m=10.0,
            min_distance_m=10.0,
            ttc_seconds=None,
            conflict_ground_xy=None,
        )
        empty = format_dual_end_alerts([safe_c], timestamp_s=1.2)
        self.assertEqual(empty["alert_count"], 0)
        self.assertEqual(empty["driver_alerts"], [])
        self.assertEqual(empty["pedestrian_alerts"], [])

        speeding_c = ConflictAssessment(
            veh_track_id=2,
            ped_track_id=None,
            risk_level=RiskLevel.LEVEL_1_SPEEDING,
            risk_desc="speeding",
            veh_speed_kmh=38.4,
            ped_speed_ms=0.0,
            current_distance_m=0.0,
            min_distance_m=0.0,
            ttc_seconds=None,
            conflict_ground_xy=None,
            vehicle_kind="car",
        )
        caution_vru = ConflictAssessment(
            veh_track_id=3,
            ped_track_id=9,
            risk_level=RiskLevel.LEVEL_2_CAUTION,
            risk_desc="caution",
            veh_speed_kmh=22.0,
            ped_speed_ms=3.2,
            current_distance_m=8.0,
            min_distance_m=1.2,
            ttc_seconds=2.4,
            conflict_ground_xy=(4.0, 12.0),
            vehicle_kind="car",
            target_kind="bicycle",
        )
        critical_ped = ConflictAssessment(
            veh_track_id=4,
            ped_track_id=5,
            risk_level=RiskLevel.LEVEL_3_CRITICAL,
            risk_desc="critical",
            veh_speed_kmh=28.5,
            ped_speed_ms=1.1,
            current_distance_m=3.1,
            min_distance_m=0.2,
            ttc_seconds=1.05,
            conflict_ground_xy=(4.0, 10.0),
            vehicle_kind="bus",
            target_kind="pedestrian",
            mttc_seconds=0.95,
            drac_ms2=4.5,
            collision_prob=0.88,
        )

        payload = format_dual_end_alerts(
            [safe_c, speeding_c, caution_vru, critical_ped],
            event_ids={(4, 5): 11, (3, 9): 12},
            timestamp_s=2.5,
        )
        self.assertEqual(payload["highest_risk_level"], 3)
        self.assertEqual(payload["highest_risk_name"], "LEVEL_3_CRITICAL")
        # 驾驶员端 3 条（L3/L2/L1），行人端 2 条（L3 行人 + L2 骑行者，L1 纯超速不下发给行人）
        self.assertEqual(len(payload["driver_alerts"]), 3)
        self.assertEqual(len(payload["pedestrian_alerts"]), 2)
        self.assertEqual(payload["driver_alerts"][0]["risk_level"], 3)
        self.assertEqual(payload["driver_alerts"][0]["event_id"], 11)
        self.assertIn("紧急制动", payload["driver_alerts"][0]["advice"])
        self.assertEqual(payload["pedestrian_alerts"][0]["role"], "pedestrian")
        self.assertIn("骑行者", payload["pedestrian_alerts"][1]["message"])
        # 严格 JSON 可序列化且不含 NaN。
        json.dumps(payload, allow_nan=False)

        for bad_ts in (-0.1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                format_dual_end_alerts([], timestamp_s=bad_ts)


class TestAnalysisServiceAndBackend(unittest.TestCase):
    def _create_sample_run(self, runs_root: Path, run_name: str = "run-sample-01") -> Path:
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(
            json.dumps({
                "schema_version": 1,
                "status": "ok",
                "stop_reason": "eof",
                "source_video": "demo.mp4",
                "frames_processed": 5,
                "elapsed_seconds": 0.4,
                "max_observed_speed_kmh": 32.5,
                "total_alert_events": 1,
                "total_speeding_alerts": 1,
                "total_caution_events": 0,
                "total_critical_conflicts": 0,
            }, allow_nan=False),
            encoding="utf-8",
        )
        (run_dir / "events.jsonl").write_text(
            json.dumps({
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
            }, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (run_dir / "speeds.csv").open("w", newline="", encoding="utf-8") as stream:
            w = csv.writer(stream)
            w.writerow(["schema_version", "frame_idx", "timestamp_s", "track_id", "category", "speed_kmh"])
            for idx in range(5):
                w.writerow([1, idx, idx * 0.1, 1, "vehicle", 32.5])
        return run_dir

    def test_service_history_queries_and_traversal_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_root = root / "runs"
            runs_root.mkdir()
            self._create_sample_run(runs_root, "run-alpha")

            for bad_root in ("", "   ", Path("   "), True):
                with self.assertRaises(ValueError):
                    AnalysisService(bad_root)

            svc = AnalysisService(runs_root)
            runs = svc.list_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["run_id"], "run-alpha")

            summary = svc.get_run_summary("run-alpha")
            self.assertEqual(summary["frames_processed"], 5)
            events = svc.get_run_events("run-alpha")
            self.assertEqual(len(events), 1)
            preview = svc.get_run_speeds_preview("run-alpha", limit=2)
            self.assertEqual(preview["preview_count"], 2)
            self.assertEqual(preview["total_rows"], 5)
            self.assertTrue(preview["truncated"])

            for bad_id in ("", "   ", ".", "..", "../runs", "..\\runs", "a/b", "C:x", "a\x00b"):
                with self.subTest(bad_id=bad_id):
                    with self.assertRaises(ValueError):
                        svc.get_run_summary(bad_id)
                    with self.assertRaises(ValueError):
                        svc.get_run_events(bad_id)
                    with self.assertRaises(ValueError):
                        svc.get_run_speeds_preview(bad_id)

            with self.assertRaises(FileNotFoundError):
                svc.get_run_summary("nonexistent-run")

    def test_stdlib_http_server_rest_jobs_mjpeg_and_stop(self):
        calib = _sample_calibrator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_root = root / "runs"
            runs_root.mkdir()
            self._create_sample_run(runs_root, "run-01")
            calib_yaml = root / "real_calib.yaml"
            calib.to_yaml(calib_yaml)

            vpath = root / "clip.mp4"
            _write_dummy_video(vpath, n_frames=4, fps=10.0, size=(800, 600))

            # 准备在视频上快速行驶的车辆与路口行人，触发双端告警与事件落盘。
            pt_ped = calib.ground_to_pixel_checked(4.0, 14.0)
            assert pt_ped is not None

            def _make_tracks(frame_idx: int):
                pt_car = calib.ground_to_pixel_checked(4.0, 2.0 + frame_idx * 1.2)
                assert pt_car is not None
                cu, cv_p = pt_car
                pu, pv_p = pt_ped
                return [
                    SimpleNamespace(
                        track_id=1,
                        class_id=2,
                        class_name="car",
                        category="vehicle",
                        confidence=0.9,
                        bbox_xyxy=(cu - 20.0, cv_p - 40.0, cu + 20.0, float(cv_p)),
                        anchor_xy=(float(cu), float(cv_p)),
                    ),
                    SimpleNamespace(
                        track_id=2,
                        class_id=0,
                        class_name="person",
                        category="pedestrian",
                        confidence=0.9,
                        bbox_xyxy=(pu - 10.0, pv_p - 30.0, pu + 10.0, float(pv_p)),
                        anchor_xy=(float(pu), float(pv_p)),
                    ),
                ]

            call_counter = {"n": 0}
            fake_session = mock.Mock()

            def _fake_process(frame, ts, kind):
                idx = call_counter["n"]
                call_counter["n"] += 1
                return SimpleNamespace(tracks=_make_tracks(idx))

            fake_session.process.side_effect = _fake_process
            fake_session.retained_ids.return_value = {1, 2}

            svc = AnalysisService(runs_root, default_calib=calib_yaml)
            server = create_http_server(svc, host="127.0.0.1", port=0)
            port = server.server_address[1]
            srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
            srv_thread.start()

            try:
                base = f"http://127.0.0.1:{port}"
                st, health = _http_json(f"{base}/api/health")
                self.assertEqual(st, 200)
                self.assertEqual(health["backend"], "stdlib")

                st, runs_resp = _http_json(f"{base}/api/runs")
                self.assertEqual(st, 200)
                self.assertEqual(runs_resp["count"], 1)

                st, sum_resp = _http_json(f"{base}/api/runs/run-01/summary")
                self.assertEqual(st, 200)
                self.assertEqual(sum_resp["run_id"], "run-01")

                st, evt_resp = _http_json(f"{base}/api/runs/run-01/events")
                self.assertEqual(st, 200)
                self.assertEqual(evt_resp["count"], 1)

                st, bad_trav = _http_json(f"{base}/api/runs/..%2F..%2Fetc/summary")
                self.assertEqual(st, 400)
                self.assertIn("error", bad_trav)

                # 非法 NaN 请求体必须返回 400。
                st, bad_nan = _http_json(f"{base}/api/jobs", method="POST", payload=b'{"source":"x.mp4","fps":NaN}')
                self.assertEqual(st, 400)
                self.assertIn("error", bad_nan)

                with mock.patch("analysis.pipeline.TrackingSession", return_value=fake_session):
                    st, created = _http_json(
                        f"{base}/api/jobs",
                        method="POST",
                        payload={
                            "source": str(vpath),
                            "calib_path": str(calib_yaml),
                            "speed_model": "kalman",
                            "conflict_model": "obb_ssm",
                        },
                    )
                    self.assertEqual(st, 201)
                    job_id = created["job_id"]

                    deadline = time.monotonic() + 10.0
                    status_payload = created
                    while status_payload["status"] in ("queued", "running") and time.monotonic() < deadline:
                        time.sleep(0.05)
                        st, status_payload = _http_json(f"{base}/api/jobs/{job_id}")
                        self.assertEqual(st, 200)

                self.assertEqual(status_payload["status"], "completed", msg=status_payload.get("error"))
                self.assertEqual(status_payload["frames_processed"], 4)
                self.assertTrue(status_payload["has_latest_frame"])

                st, alerts_resp = _http_json(f"{base}/api/jobs/{job_id}/alerts?since=0")
                self.assertEqual(st, 200)
                self.assertGreaterEqual(alerts_resp["count"], 1)
                self.assertIn("driver_alerts", alerts_resp["alerts"][0])
                self.assertIn("pedestrian_alerts", alerts_resp["alerts"][0])

                # 验证 MJPEG 接口输出有效 multipart 边界与 JPEG 魔数头 \xff\xd8\xff。
                with urlopen(f"{base}/api/jobs/{job_id}/mjpeg", timeout=5.0) as mjpeg_resp:
                    self.assertIn("multipart/x-mixed-replace", mjpeg_resp.headers.get("Content-Type", ""))
                    raw_mjpeg = mjpeg_resp.read()
                self.assertIn(b"--frame", raw_mjpeg)
                self.assertIn(b"\xff\xd8\xff", raw_mjpeg)

                # 验证运行中任务可通过 POST /api/jobs/<id>/stop 安全停止。
                gate = threading.Event()
                slow_session = mock.Mock()
                slow_session.process.side_effect = lambda f, ts, k: (
                    gate.wait(timeout=0.05),
                    SimpleNamespace(tracks=[]),
                )[1]
                slow_session.retained_ids.return_value = set()

                long_video = root / "long.mp4"
                _write_dummy_video(long_video, n_frames=20, fps=10.0, size=(800, 600))
                with mock.patch("analysis.pipeline.TrackingSession", return_value=slow_session):
                    st, slow_job = _http_json(
                        f"{base}/api/jobs",
                        method="POST",
                        payload={"source": str(long_video), "calib_path": str(calib_yaml)},
                    )
                    self.assertEqual(st, 201)
                    slow_id = slow_job["job_id"]
                    time.sleep(0.1)
                    st, stopped = _http_json(f"{base}/api/jobs/{slow_id}/stop", method="POST", payload={})
                    self.assertEqual(st, 200)
                    self.assertEqual(stopped["status"], "stopped")
            finally:
                server.shutdown()
                server.server_close()

    def test_create_fastapi_app_lazy_import_behavior(self):
        # 顶层导入 analysis.service 绝不能提前加载 fastapi/uvicorn/scipy。
        self.assertNotIn("fastapi", sys.modules)
        self.assertNotIn("uvicorn", sys.modules)

        orig_import = builtins.__import__

        def _missing_fastapi(name, *args, **kwargs):
            if name == "fastapi" or name.startswith("fastapi."):
                raise ImportError("No module named 'fastapi'")
            return orig_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=_missing_fastapi):
            with self.assertRaisesRegex(ImportError, "FastAPI"):
                create_fastapi_app()

        # 构造最小桩 fastapi 模块验证路由注册与懒导入分支逻辑。
        registered_routes = []

        class _FakeFastAPI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

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
            svc = AnalysisService(Path(tmp))
            with mock.patch.dict(
                sys.modules,
                {"fastapi": fake_fastapi, "fastapi.responses": fake_responses},
            ):
                app = create_fastapi_app(svc)
            self.assertIsInstance(app, _FakeFastAPI)
            paths = {(m, p) for m, p, _ in registered_routes}
            self.assertIn(("GET", "/"), paths)
            self.assertIn(("GET", "/index.html"), paths)
            self.assertIn(("GET", "/assets/{subpath:path}"), paths)
            self.assertIn(("GET", "/api/health"), paths)
            self.assertIn(("GET", "/api/presets"), paths)
            self.assertIn(("GET", "/api/runs/{run_id}/conflicts"), paths)
            self.assertIn(("GET", "/api/runs/{run_id}/frame"), paths)
            self.assertIn(("GET", "/api/runs/{run_id}/video"), paths)
            self.assertIn(("POST", "/api/jobs"), paths)
            self.assertIn(("GET", "/api/jobs/{job_id}/mjpeg"), paths)
            self.assertIn(("WS", "/ws/jobs/{job_id}/alerts"), paths)

            # 直接验证新注册的 FastAPI 路由处理函数（使用桩 Response/StreamingResponse）。
            route_map = {(m, p): fn for m, p, fn in registered_routes}
            web_dir = Path(tmp) / "web"
            (web_dir / "assets").mkdir(parents=True)
            (web_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")
            (web_dir / "assets" / "app.js").write_text("console.log(1);", encoding="utf-8")
            run_dir = Path(tmp) / "run-fa"
            run_dir.mkdir()
            vpath = run_dir / "annotated_conflict.mp4"
            _write_dummy_video(vpath, n_frames=3, fps=10.0, size=(64, 64))
            (run_dir / "conflicts.jsonl").write_text(
                json.dumps({"frame_index": 0, "timestamp_s": 0.0}, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            svc.web_root = web_dir

            idx_resp = route_map[("GET", "/")]()
            self.assertEqual(idx_resp["media_type"], "text/html; charset=utf-8")
            self.assertEqual(idx_resp["headers"]["X-Content-Type-Options"], "nosniff")

            js_resp = route_map[("GET", "/assets/{subpath:path}")]("app.js")
            self.assertEqual(js_resp["media_type"], "application/javascript; charset=utf-8")

            pre_resp = route_map[("GET", "/api/presets")]()
            self.assertIn("default_calib", json.loads(pre_resp["content"]))

            c_resp = route_map[("GET", "/api/runs/{run_id}/conflicts")]("run-fa", limit=10)
            self.assertEqual(json.loads(c_resp["content"])["count"], 1)

            f_resp = route_map[("GET", "/api/runs/{run_id}/frame")]("run-fa", index=0)
            self.assertEqual(f_resp["media_type"], "image/jpeg")
            self.assertTrue(f_resp["content"].startswith(b"\xff\xd8"))

            v_206 = route_map[("GET", "/api/runs/{run_id}/video")](
                "run-fa", SimpleNamespace(headers={"range": "bytes=0-15"})
            )
            self.assertEqual(v_206["status_code"], 206)
            self.assertEqual(v_206["headers"]["Accept-Ranges"], "bytes")
            self.assertEqual(len(b"".join(v_206["content"])), 16)

            v_416 = route_map[("GET", "/api/runs/{run_id}/video")](
                "run-fa", SimpleNamespace(headers={"range": "bytes=9999999-"})
            )
            self.assertEqual(v_416["status_code"], 416)

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            service_main(["--runs-root", "   "])

    def test_web_assets_conflicts_video_range_frame_and_presets(self):
        calib = _sample_calibrator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_root = root / "runs" / "analysis"
            runs_root.mkdir(parents=True)
            fixtures_dir = root / "runs" / "tracking-fixtures" / "cam01"
            fixtures_dir.mkdir(parents=True)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            web_dir = root / "web"
            (web_dir / "assets" / "sub").mkdir(parents=True)

            # 写入全 7 种允许 MIME 类型的最小静态资源以及不被允许的 .txt 文件。
            (web_dir / "index.html").write_text("<html><title>CampusMotion</title></html>", encoding="utf-8")
            (web_dir / "assets" / "styles.css").write_text("body{color:#ededed;}", encoding="utf-8")
            (web_dir / "assets" / "sub" / "foo.js").write_text("window.ok=true;", encoding="utf-8")
            (web_dir / "assets" / "icon.svg").write_text("<svg></svg>", encoding="utf-8")
            (web_dir / "assets" / "manifest.json").write_text("{}", encoding="utf-8")
            (web_dir / "assets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (web_dir / "assets" / "favicon.ico").write_bytes(b"\x00\x00\x01\x00")
            (web_dir / "assets" / "secret.txt").write_text("nope", encoding="utf-8")

            calib_yaml = cfg_dir / "calibration.yaml"
            calib.to_yaml(calib_yaml)
            alt_calib_yml = cfg_dir / "gate_b.yml"
            calib.to_yaml(alt_calib_yml)
            # 确认非标定 YAML 不会混入 calibs 预置列表。
            (cfg_dir / "bytetrack.yaml").write_text("tracker_type: bytetrack\n", encoding="utf-8")

            fixture_vid = fixtures_dir / "clip_a.mp4"
            _write_dummy_video(fixture_vid, n_frames=5, fps=10.0, size=(320, 240))

            run_dir = self._create_sample_run(runs_root, "run-web-01")
            conflict_rows = [
                {
                    "frame_index": i,
                    "timestamp_s": round(i * 0.1, 2),
                    "assessment": {
                        "veh_track_id": 1,
                        "ped_track_id": 7 if i >= 2 else None,
                        "risk_level": 3 if i == 2 else 0,
                        "veh_speed_kmh": 28.0 if i == 2 else 0.0,
                        "ttc_seconds": 1.1 if i == 2 else None,
                    },
                }
                for i in range(5)
            ]
            (run_dir / "conflicts.jsonl").write_text(
                "\n".join(json.dumps(r, allow_nan=False) for r in conflict_rows) + "\n",
                encoding="utf-8",
            )
            run_video = run_dir / "annotated_conflict.mp4"
            # 写入纹理帧以产生 > 64KB 的真实 MP4，从而实测跨多个 64KB 分块的流式响应。
            rng = np.random.default_rng(7)
            writer = cv2.VideoWriter(
                str(run_video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8))
            writer.release()
            video_bytes = run_video.read_bytes()
            self.assertGreater(len(video_bytes), 65536)

            for bad_web in ("", "   ", Path("   "), True, 123):
                with self.assertRaises(ValueError):
                    AnalysisService(runs_root, default_calib=calib_yaml, web_root=bad_web)

            default_svc = AnalysisService(runs_root, default_calib=calib_yaml)
            self.assertEqual(default_svc.web_root, DEFAULT_WEB_ROOT)

            svc = AnalysisService(runs_root, default_calib=calib_yaml, web_root=web_dir)

            # 验证 7 种合法静态资源与 MIME 映射。
            expected_mimes = {
                "index.html": "text/html; charset=utf-8",
                "assets/styles.css": "text/css; charset=utf-8",
                "assets/sub/foo.js": "application/javascript; charset=utf-8",
                "assets/icon.svg": "image/svg+xml",
                "assets/manifest.json": "application/json; charset=utf-8",
                "assets/logo.png": "image/png",
                "assets/favicon.ico": "image/x-icon",
            }
            for rel, expected_mime in expected_mimes.items():
                resolved_p, mime = svc.resolve_web_asset(rel)
                self.assertTrue(resolved_p.is_file())
                self.assertEqual(mime, expected_mime)

            # 越界/穿越/不支持后缀/缺失文件校验。
            for bad_asset in (
                None,
                123,
                "",
                "   ",
                ".",
                "..",
                "../service.py",
                "assets/../service.py",
                "assets/%2e%2e/service.py",
                "assets/%252e%252e/service.py",
                "assets\\styles.css",
                "C:styles.css",
                "assets/bad\x00.js",
                "/index.html",
                "assets/secret.txt",
                "assets/not_exist.py",
            ):
                with self.subTest(bad_asset=bad_asset), self.assertRaises(ValueError):
                    svc.resolve_web_asset(bad_asset)
            with self.assertRaises(FileNotFoundError):
                svc.resolve_web_asset("assets/missing.js")

            # 服务层新读方法单元测试。
            self.assertEqual(len(svc.get_run_conflicts("run-web-01", limit=2)), 2)
            self.assertEqual(len(svc.get_run_conflicts("run-web-01", limit=None)), 5)
            self.assertEqual(svc.get_run_conflicts("run-web-01", limit=0), [])
            for bad_limit in (-1, True, "2"):
                with self.assertRaises(ValueError):
                    svc.get_run_conflicts("run-web-01", limit=bad_limit)
            for bad_idx in (-1, True, "0"):
                with self.assertRaises(ValueError):
                    svc.get_run_frame_jpeg("run-web-01", frame_index=bad_idx)
            with self.assertRaises(FileNotFoundError):
                svc.get_run_frame_jpeg("run-web-01", frame_index=999)

            # 缺少 conflicts.jsonl / annotated_conflict.mp4 的历史目录处理。
            bare_run = self._create_sample_run(runs_root, "run-bare")
            self.assertTrue(bare_run.is_dir())
            self.assertEqual(svc.get_run_conflicts("run-bare"), [])
            with self.assertRaises(FileNotFoundError):
                svc.get_run_video_path("run-bare")

            presets = svc.list_presets()
            self.assertEqual(presets["default_calib"], str(calib_yaml.resolve()))
            self.assertTrue(any(v["path"] == str(fixture_vid.resolve()) for v in presets["videos"]))
            calib_paths = {c["path"] for c in presets["calibs"]}
            self.assertEqual(calib_paths, {str(calib_yaml.resolve()), str(alt_calib_yml.resolve())})

            # 项目 data/demo_videos 下的真实示例视频与配套标定也应被 list_presets 发现（且 is_fixture=False）
            demo_dir = root / "data" / "demo_videos"
            demo_dir.mkdir(parents=True, exist_ok=True)
            demo_mp4 = demo_dir / "real_crosswalk.mp4"
            demo_mp4.write_bytes(b"real-demo")
            demo_calib = demo_dir / "calib_real.yaml"
            demo_calib.write_text(calib_yaml.read_text(encoding="utf-8"), encoding="utf-8")
            with mock.patch("pathlib.Path.cwd", return_value=root):
                presets_with_demo = svc.list_presets()
            demo_videos = [v for v in presets_with_demo["videos"] if v["path"] == str(demo_mp4.resolve())]
            self.assertEqual(len(demo_videos), 1)
            self.assertFalse(demo_videos[0]["is_fixture"])
            self.assertIn(str(demo_calib.resolve()), {c["path"] for c in presets_with_demo["calibs"]})

            server = create_http_server(svc, host="127.0.0.1", port=0)
            port = server.server_address[1]
            srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
            srv_thread.start()
            try:
                base = f"http://127.0.0.1:{port}"

                # 静态首页与子目录资源请求。
                for url_path, exp_ct, exp_snippet in (
                    ("/", "text/html; charset=utf-8", b"CampusMotion"),
                    ("/index.html", "text/html; charset=utf-8", b"CampusMotion"),
                    ("/assets/styles.css", "text/css; charset=utf-8", b"color:#ededed"),
                    ("/assets/sub/foo.js", "application/javascript; charset=utf-8", b"window.ok=true;"),
                ):
                    with urlopen(f"{base}{url_path}", timeout=5.0) as resp:
                        self.assertEqual(resp.status, 200)
                        self.assertEqual(resp.headers.get("Content-Type"), exp_ct)
                        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
                        self.assertEqual(resp.headers.get("Cache-Control"), "no-cache")
                        self.assertIn(exp_snippet, resp.read())

                # 使用原始 HTTPConnection 发送未被客户端归一化改写的穿越路径。
                for bad_raw in (
                    "/assets/../service.py",
                    "/assets/%2e%2e/service.py",
                    "/assets/%2e%2e%2fservice.py",
                    "/assets/..%5cservice.py",
                    "/assets/sub/../styles.css",
                    "/assets/secret.txt",
                    "/assets/",
                ):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
                    try:
                        conn.request("GET", bad_raw)
                        res = conn.getresponse()
                        body = json.loads(res.read().decode("utf-8"))
                    finally:
                        conn.close()
                    self.assertEqual(res.status, 400, msg=f"path={bad_raw}")
                    self.assertIn("error", body)

                st_miss, miss_body = _http_json(f"{base}/assets/missing.js")
                self.assertEqual(st_miss, 404)
                self.assertIn("error", miss_body)

                # 新运行数据查询接口。
                st_conf, conf_body = _http_json(f"{base}/api/runs/run-web-01/conflicts?limit=2")
                self.assertEqual(st_conf, 200)
                self.assertEqual(conf_body["run_id"], "run-web-01")
                self.assertEqual(conf_body["count"], 2)
                self.assertEqual(len(conf_body["conflicts"]), 2)

                # 视频帧截取 JPEG 与越界 404。
                with urlopen(f"{base}/api/runs/run-web-01/frame?index=0", timeout=5.0) as f_resp:
                    self.assertEqual(f_resp.status, 200)
                    self.assertEqual(f_resp.headers.get("Content-Type"), "image/jpeg")
                    self.assertEqual(f_resp.headers.get("Cache-Control"), "public, max-age=60")
                    jpg = f_resp.read()
                self.assertTrue(jpg.startswith(b"\xff\xd8"))

                st_f404, _ = _http_json(f"{base}/api/runs/run-web-01/frame?index=999")
                self.assertEqual(st_f404, 404)
                st_f400, _ = _http_json(f"{base}/api/runs/run-web-01/frame?index=-1")
                self.assertEqual(st_f400, 400)

                # 视频全量 200、部分 Range 206 与越界 416。
                with urlopen(f"{base}/api/runs/run-web-01/video", timeout=5.0) as v_full:
                    self.assertEqual(v_full.status, 200)
                    self.assertEqual(v_full.headers.get("Content-Type"), "video/mp4")
                    self.assertEqual(v_full.headers.get("Accept-Ranges"), "bytes")
                    self.assertEqual(v_full.read(), video_bytes)

                req_part = Request(
                    f"{base}/api/runs/run-web-01/video",
                    headers={"Range": "bytes=0-63"},
                )
                with urlopen(req_part, timeout=5.0) as v_part:
                    self.assertEqual(v_part.status, 206)
                    self.assertEqual(v_part.headers.get("Accept-Ranges"), "bytes")
                    self.assertEqual(
                        v_part.headers.get("Content-Range"),
                        f"bytes 0-63/{len(video_bytes)}",
                    )
                    part_data = v_part.read()
                self.assertEqual(len(part_data), 64)
                self.assertEqual(part_data, video_bytes[:64])

                req_suffix = Request(
                    f"{base}/api/runs/run-web-01/video",
                    headers={"Range": "bytes=-16"},
                )
                with urlopen(req_suffix, timeout=5.0) as v_suf:
                    self.assertEqual(v_suf.status, 206)
                    self.assertEqual(v_suf.read(), video_bytes[-16:])

                for bad_range in ("bytes=9999999-", "bytes=50-10", "bytes=invalid"):
                    req_bad = Request(
                        f"{base}/api/runs/run-web-01/video",
                        headers={"Range": bad_range},
                    )
                    with self.assertRaises(HTTPError) as ctx:
                        urlopen(req_bad, timeout=5.0)
                    self.assertEqual(ctx.exception.code, 416)
                    self.assertEqual(
                        ctx.exception.headers.get("Content-Range"),
                        f"bytes */{len(video_bytes)}",
                    )
                    ctx.exception.close()

                # /api/presets 校验。
                st_pre, pre_body = _http_json(f"{base}/api/presets")
                self.assertEqual(st_pre, 200)
                self.assertTrue(any(v["path"] == str(fixture_vid.resolve()) for v in pre_body["videos"]))
                self.assertEqual(pre_body["default_calib"], str(calib_yaml.resolve()))
            finally:
                server.shutdown()
                server.server_close()


class TestInteractiveCalibratorGuiExtensions(unittest.TestCase):
    def setUp(self):
        self.base_img = [(400.0, 300.0), (880.0, 300.0), (1150.0, 680.0), (130.0, 680.0)]
        self.ref_cal = RoadCalibrator(self.base_img, 6.0, 15.0)

    def _project(self, gd_pts):
        out = []
        for gx, gy in gd_pts:
            p = self.ref_cal.H_inv @ np.array([gx, gy, 1.0], dtype=np.float64)
            out.append((float(p[0] / p[2]), float(p[1] / p[2])))
        return out

    def test_gui_parking_zone_toggle_close_reset_and_multipoint_correspondences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img_path = root / "road.png"
            ok, enc = cv2.imencode(".png", np.zeros((720, 1280, 3), dtype=np.uint8))
            self.assertTrue(ok)
            enc.tofile(str(img_path))

            # 场景 1：4 点模式内按 'P' 切停车区 -> 点 3 点 -> 'R' 重置 -> 点 4 点 -> 'N' 闭合 -> 'P' 切回 -> 点 4 角点 -> Enter。
            z_px = self._project([(0.5, 2.0), (2.0, 2.0), (2.0, 5.0), (0.5, 5.0)])
            mouse_cbs = {}
            step_state = {"step": 0}

            def _fake_set_mouse(win, cb):
                mouse_cbs["cb"] = cb

            def _fake_wait_key_4pt(_delay):
                s = step_state["step"]
                step_state["step"] += 1
                cb = mouse_cbs["cb"]
                if s == 0:
                    return ord("p")
                if s == 1:
                    for ux, vy in ((100, 100), (150, 100), (150, 150)):
                        cb(cv2.EVENT_LBUTTONDOWN, ux, vy, 0, None)
                    return ord("r")
                if s == 2:
                    for ux, vy in z_px:
                        cb(cv2.EVENT_LBUTTONDOWN, ux, vy, 0, None)
                    return ord("n")
                if s == 3:
                    return ord("P")
                if s == 4:
                    for ux, vy in self.base_img:
                        cb(cv2.EVENT_LBUTTONDOWN, ux, vy, 0, None)
                    return 13
                return ord("q")

            out_4pt = root / "gui_4pt_park.yaml"
            with mock.patch("analysis.interactive_calibrator.cv2.namedWindow"), \
                    mock.patch("analysis.interactive_calibrator.cv2.setMouseCallback", side_effect=_fake_set_mouse), \
                    mock.patch("analysis.interactive_calibrator.cv2.imshow"), \
                    mock.patch("analysis.interactive_calibrator.cv2.waitKey", side_effect=_fake_wait_key_4pt), \
                    mock.patch("analysis.interactive_calibrator.cv2.getWindowProperty", return_value=1.0), \
                    mock.patch("analysis.interactive_calibrator.cv2.destroyAllWindows"):
                cal_4pt = run_interactive_calibration(img_path, 6.0, 15.0, output_path=out_4pt)

            self.assertEqual(len(cal_4pt.parking_zones), 1)
            self.assertTrue(cal_4pt.is_in_parking_zone(1.2, 3.5))
            self.assertFalse(cal_4pt.is_in_parking_zone(4.5, 9.0))
            reloaded = RoadCalibrator.from_yaml(out_4pt)
            self.assertEqual(len(reloaded.parking_zones), 1)

            # 场景 2：CLI 传入 --ground-points (>=4 对) 且不传 --points -> GUI 多点点选 + 'P'/'N' 停车区 -> Space 完成。
            gd5 = [(0.0, 0.0), (6.0, 0.0), (6.0, 15.0), (0.0, 15.0), (3.0, 7.5)]
            px5 = self._project(gd5)
            z2_px = self._project([(4.0, 8.0), (5.5, 8.0), (5.5, 11.0), (4.0, 11.0)])
            step5 = {"step": 0}

            def _fake_wait_key_5pt(_delay):
                s = step5["step"]
                step5["step"] += 1
                cb = mouse_cbs["cb"]
                if s == 0:
                    for ux, vy in px5:
                        cb(cv2.EVENT_LBUTTONDOWN, ux, vy, 0, None)
                    return ord("p")
                if s == 1:
                    for ux, vy in z2_px:
                        cb(cv2.EVENT_LBUTTONDOWN, ux, vy, 0, None)
                    return ord("n")
                if s == 2:
                    return 32  # Space 确认
                return ord("q")

            out_5pt = root / "gui_5pt_park.yaml"
            gd5_str = ",".join(f"{x:.3f},{y:.3f}" for x, y in gd5)
            with mock.patch("analysis.interactive_calibrator.cv2.namedWindow"), \
                    mock.patch("analysis.interactive_calibrator.cv2.setMouseCallback", side_effect=_fake_set_mouse), \
                    mock.patch("analysis.interactive_calibrator.cv2.imshow"), \
                    mock.patch("analysis.interactive_calibrator.cv2.waitKey", side_effect=_fake_wait_key_5pt), \
                    mock.patch("analysis.interactive_calibrator.cv2.getWindowProperty", return_value=1.0), \
                    mock.patch("analysis.interactive_calibrator.cv2.destroyAllWindows"):
                rc = calib_main([
                    "--source", str(img_path),
                    "--ground-points", gd5_str,
                    "--output", str(out_5pt),
                ])
            self.assertEqual(rc, 0)
            cal_5pt = RoadCalibrator.from_yaml(out_5pt)
            self.assertEqual(cal_5pt.inlier_count, 5)
            self.assertLess(cal_5pt.reprojection_rmse_m, 1e-2)
            self.assertEqual(len(cal_5pt.parking_zones), 1)
            self.assertTrue(cal_5pt.is_in_parking_zone(4.8, 9.5))

            # 非法共线 ground_points 必须在打开 GUI 窗口前即拦截报错。
            with mock.patch("analysis.interactive_calibrator.cv2.namedWindow") as m_win:
                with self.assertRaises(ValueError):
                    run_interactive_calibration(
                        img_path,
                        6.0,
                        15.0,
                        ground_points=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
                        save_output=False,
                    )
            m_win.assert_not_called()


class TestPipelineCtrvAnchorAndLiveDropIntegration(unittest.TestCase):
    def test_pipeline_cli_and_schema_validation_for_new_flags(self):
        calib = _sample_calibrator()
        for bad_kw in (
            {"speed_model": "invalid"},
            {"speed_model": 1},
            {"compensate_anchor_bias": 1},
            {"compensate_anchor_bias": "true"},
            {"live_drop_stale_frames": 0},
        ):
            with self.subTest(bad_kw=bad_kw), self.assertRaises(ValueError):
                ConflictPerceptionPipeline(calibrator=calib, **bad_kw)

        pipe = ConflictPerceptionPipeline(
            calibrator=calib,
            speed_model="ctrv",
            compensate_anchor_bias=True,
            live_drop_stale_frames=True,
        )
        snap = pipe._configuration_snapshot(25.0)
        self.assertEqual(snap["speed_model"], "ctrv")
        self.assertIs(snap["compensate_anchor_bias"], True)
        self.assertIs(snap["live_drop_stale_frames"], True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vpath = root / "clip.mp4"
            _write_dummy_video(vpath, n_frames=2, fps=10.0, size=(64, 64))
            fake_pipe = mock.Mock()
            with mock.patch("analysis.__main__.ConflictPerceptionPipeline", return_value=fake_pipe) as m_pipe:
                rc = analysis_main([
                    "--source", str(vpath),
                    "--demo",
                    "--output", str(root / "out"),
                    "--speed-model", "ctrv",
                    "--compensate-anchor-bias",
                    "--live-drop-stale-frames",
                ])
            self.assertEqual(rc, 0)
            kw = m_pipe.call_args.kwargs
            self.assertEqual(kw["speed_model"], "ctrv")
            self.assertIs(kw["compensate_anchor_bias"], True)
            self.assertIs(kw["live_drop_stale_frames"], True)

        # 验证 schema 校验兼容 ctrv 与新开关并拦截非法布尔/模型名。
        from tests.test_schema import valid_configuration

        cfg = valid_configuration()
        cfg["speed_model"] = "ctrv"
        cfg["compensate_anchor_bias"] = True
        cfg["live_drop_stale_frames"] = True
        validate_configuration(cfg)
        for key, bad_val in (
            ("speed_model", "ukf"),
            ("compensate_anchor_bias", 1),
            ("live_drop_stale_frames", "yes"),
        ):
            bad_cfg = dict(cfg)
            bad_cfg[key] = bad_val
            with self.assertRaises(SchemaValidationError):
                validate_configuration(bad_cfg)

        # 验证 validate_ablation 兼容 turn_speed_ablation 并拦截非法条目。
        ab_data = run_default_ablation()
        self.assertIn("turn_speed_ablation", ab_data)
        validate_ablation(ab_data)
        bad_ab = dict(ab_data)
        bad_ab["turn_speed_ablation"] = [{"estimator_mode": "bad"}]
        with self.assertRaises(SchemaValidationError):
            validate_ablation(bad_ab)

    def test_ctrv_anchor_bias_stitching_callbacks_and_live_drop_stale(self):
        calib = _sample_calibrator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vpath = root / "turn.mp4"
            _write_dummy_video(vpath, n_frames=10, fps=10.0, size=(800, 600))
            tracks_path = root / "tracks.jsonl"
            rows = []
            # 构造车辆沿圆弧弯道接近静止行人的轨迹，同时测试 CTRV + 锚点偏置补偿 + 外观缝合 + 回调。
            ped_px = calib.ground_to_pixel_checked(5.2, 16.0)
            assert ped_px is not None
            pu, pv = ped_px
            for idx in range(10):
                theta = idx * 0.08
                gx = 2.0 + 4.0 * (1.0 - np.cos(theta))
                gy = 4.0 + 8.0 * np.sin(theta)
                car_px = calib.ground_to_pixel_checked(float(gx), float(gy))
                assert car_px is not None
                cu, cv_p = car_px
                rows.append({
                    "schema_version": 1,
                    "session_id": "s-ctrv-int",
                    "source": str(vpath),
                    "frame_index": idx,
                    "timestamp_s": round(idx * 0.1, 3),
                    "timestamp_source": "video_pts",
                    "width": 800,
                    "height": 600,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "category": "vehicle",
                            "confidence": 0.92,
                            "bbox_xyxy": [cu - 22.0, cv_p - 44.0, cu + 22.0, float(cv_p)],
                            "anchor_xy": [float(cu), float(cv_p)],
                            "anchor_type": "bbox_bottom_center",
                            "observation": "detected",
                        },
                        {
                            "track_id": 2,
                            "class_id": 0,
                            "class_name": "person",
                            "category": "pedestrian",
                            "confidence": 0.9,
                            "bbox_xyxy": [pu - 10.0, pv - 32.0, pu + 10.0, float(pv)],
                            "anchor_xy": [float(pu), float(pv)],
                            "anchor_type": "bbox_bottom_center",
                            "observation": "detected",
                        },
                    ],
                })
            tracks_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows) + "\n",
                encoding="utf-8",
            )

            seen_frames = []
            seen_alerts = []
            pipe = ConflictPerceptionPipeline(
                calibrator=calib,
                speed_model="ctrv",
                conflict_model="obb_ssm",
                enable_stitching=True,
                compensate_anchor_bias=True,
                alert_confirm_frames=2,
            )
            out_dir = root / "out_ctrv"
            with mock.patch.object(
                calib, "compensate_vehicle_anchor", wraps=calib.compensate_vehicle_anchor
            ) as spy_comp, mock.patch.object(
                pipe.stitcher, "resolve_frame", wraps=pipe.stitcher.resolve_frame
            ) as spy_stitch:
                summary = pipe.process_tracks(
                    tracks_path,
                    out_dir,
                    video_path=vpath,
                    frame_callback=lambda idx, ts, f: seen_frames.append((idx, ts, f is not None)),
                    alert_callback=lambda payload: seen_alerts.append(payload),
                )

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(len(seen_frames), 10)
            self.assertTrue(all(has_img for _, _, has_img in seen_frames))
            self.assertEqual(spy_comp.call_count, 10)
            # 确认传入 stitcher 的为 6 元组观测且第 6 项在有画面帧时为有效直方图。
            first_obs = spy_stitch.call_args_list[0].args[0]
            self.assertEqual(len(first_obs[0]), 6)
            self.assertIsNotNone(first_obs[0][5])
            self.assertGreaterEqual(len(seen_alerts), 1)
            self.assertTrue(validate_run(out_dir)["valid"])

            # 离线无视频帧复盘开启缝合时，第 6 项外观特征必须传 None 优雅退化为纯运动学。
            out_headless = root / "out_ctrv_headless"
            pipe_headless = ConflictPerceptionPipeline(
                calibrator=calib,
                speed_model="ctrv",
                enable_stitching=True,
                compensate_anchor_bias=True,
            )
            with mock.patch.object(
                pipe_headless.stitcher, "resolve_frame", wraps=pipe_headless.stitcher.resolve_frame
            ) as spy_headless_stitch:
                headless_summary = pipe_headless.process_tracks(tracks_path, out_headless)
            self.assertEqual(headless_summary["status"], "ok")
            headless_obs = spy_headless_stitch.call_args_list[0].args[0]
            self.assertEqual(len(headless_obs[0]), 6)
            self.assertIsNone(headless_obs[0][5])
            self.assertTrue(validate_run(out_headless)["valid"])

            # 测试实时流启用 live_drop_stale_frames=True 包装 LatestFrameCapture 并由 ExitStack 释放。
            raw_cap = mock.Mock()
            raw_cap.isOpened.return_value = True
            raw_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 25.0,
                cv2.CAP_PROP_FRAME_WIDTH: 64.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 64.0,
                cv2.CAP_PROP_POS_MSEC: 0.0,
            }.get(prop, 0.0)
            frame_i = {"n": 0}

            def _fast_rtsp_read():
                n = frame_i["n"]
                frame_i["n"] += 1
                if n < 5:
                    return True, np.full((64, 64, 3), n, dtype=np.uint8)
                return False, None

            raw_cap.read.side_effect = _fast_rtsp_read
            fake_sess = mock.Mock()
            fake_sess.process.return_value = SimpleNamespace(tracks=[])
            fake_sess.retained_ids.return_value = set()

            pipe_live = ConflictPerceptionPipeline(
                calibrator=calib,
                speed_model="ctrv",
                live_drop_stale_frames=True,
                live_reconnect_attempts=0,
            )
            live_out = root / "live_drop"
            cb_frames = []
            with mock.patch("analysis.pipeline.cv2.VideoCapture", return_value=raw_cap), \
                    mock.patch("analysis.pipeline.TrackingSession", return_value=fake_sess):
                live_summary = pipe_live.process_video(
                    "rtsp://admin:pw@127.0.0.1:554/stream",
                    live_out,
                    max_frames=2,
                    frame_callback=lambda idx, ts, frm: cb_frames.append(idx) or (idx < 0),
                )
            self.assertEqual(live_summary["status"], "ok")
            self.assertEqual(live_summary["stop_reason"], "stopped")
            self.assertEqual(len(cb_frames), 1)
            raw_cap.release.assert_called_once()
            self.assertTrue(validate_run(live_out)["valid"])

    def test_get_system_info_headless_linux_disables_can_browse_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calib = _sample_calibrator()
            calib_file = root / "calib.yaml"
            calib.to_yaml(calib_file)
            service = AnalysisService(runs_root=root / "runs", default_calib=calib_file)

            clean_env = dict(os.environ)
            clean_env.pop("DISPLAY", None)
            clean_env.pop("WAYLAND_DISPLAY", None)
            with (
                mock.patch("analysis.service.sys.platform", "linux"),
                mock.patch("analysis.service.importlib.util.find_spec", return_value=object()),
                mock.patch.dict(os.environ, clean_env, clear=True),
            ):
                info_headless = service.get_system_info()
            self.assertFalse(info_headless["can_browse_files"])

            with (
                mock.patch("analysis.service.sys.platform", "linux"),
                mock.patch("analysis.service.importlib.util.find_spec", return_value=object()),
                mock.patch.dict(os.environ, {**clean_env, "WAYLAND_DISPLAY": "wayland-0"}, clear=True),
            ):
                info_wayland = service.get_system_info()
            self.assertTrue(info_wayland["can_browse_files"])

    def test_job_request_allowed_keys_match_start_job_and_reject_unknown(self):
        sig_params = set(inspect.signature(AnalysisService.start_job).parameters.keys())
        self.assertTrue(_ALLOWED_JOB_REQUEST_KEYS.issubset(sig_params))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calib_file = root / "calib.yaml"
            _sample_calibrator().to_yaml(calib_file)
            service = AnalysisService(runs_root=root / "runs", default_calib=calib_file)
            server = create_http_server(service, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"
                req = Request(
                    f"{base_url}/api/jobs",
                    data=json.dumps({"source": "demo.mp4", "unexpected_kwarg": 123}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as ctx:
                    urlopen(req, timeout=5)
                self.assertEqual(ctx.exception.code, 400)
                body = json.loads(ctx.exception.read().decode("utf-8"))
                self.assertIn("unexpected_kwarg", body["error"])

                for bad_payload in (
                    {"source": "demo.mp4", "demo": True, "conf": -0.2},
                    {"source": "demo.mp4", "demo": True, "conf": 1.5},
                    {"source": "demo.mp4", "demo": True, "conf": True},
                    {"source": "demo.mp4", "demo": True, "device": True},
                    {"source": "demo.mp4", "demo": True, "device": "   "},
                ):
                    bad_req = Request(
                        f"{base_url}/api/jobs",
                        data=json.dumps(bad_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with self.assertRaises(HTTPError) as bad_ctx:
                        urlopen(bad_req, timeout=5)
                    self.assertEqual(bad_ctx.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()

    def test_runs_report_has_video_and_pipeline_exposes_latest_twin_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            no_vid_dir = runs_dir / "run-tracks-only"
            no_vid_dir.mkdir(parents=True)
            (no_vid_dir / "summary.json").write_text(
                json.dumps({"schema_version": 1, "status": "ok", "frames_processed": 2}),
                encoding="utf-8",
            )
            vid_dir = runs_dir / "run-with-video"
            vid_dir.mkdir(parents=True)
            (vid_dir / "summary.json").write_text(
                json.dumps({"schema_version": 1, "status": "ok", "frames_processed": 2}),
                encoding="utf-8",
            )
            (vid_dir / "annotated_conflict.mp4").write_bytes(b"mp4bytes")

            calib_file = root / "calib.yaml"
            cal = _sample_calibrator()
            cal.to_yaml(calib_file)
            service = AnalysisService(runs_root=runs_dir, default_calib=calib_file)
            listed = {item["run_id"]: item for item in service.list_runs()}
            self.assertFalse(listed["run-tracks-only"]["has_video"])
            self.assertTrue(listed["run-with-video"]["has_video"])
            self.assertFalse(service.get_run_summary("run-tracks-only")["has_video"])
            self.assertTrue(service.get_run_summary("run-with-video")["has_video"])

            pipe = ConflictPerceptionPipeline(calibrator=cal)
            with self.assertRaises(ValueError):
                ConflictPerceptionPipeline(calibrator=cal, conf=1.5)
            with self.assertRaises(ValueError):
                ConflictPerceptionPipeline(calibrator=cal, device=True)

            tracks_file = root / "tracks.jsonl"
            tracks_file.write_text(
                "\n".join([
                    json.dumps({
                        "schema_version": 1,
                        "session_id": "sess-twin-1",
                        "frame_index": 0,
                        "timestamp_s": 0.0,
                        "width": 800,
                        "height": 600,
                        "tracks": [{
                            "track_id": 1,
                            "category": "vehicle",
                            "class_name": "car",
                            "bbox_xyxy": [380.0, 260.0, 460.0, 340.0],
                            "anchor_xy": [420.0, 340.0],
                        }],
                    }),
                    json.dumps({
                        "schema_version": 1,
                        "session_id": "sess-twin-1",
                        "frame_index": 1,
                        "timestamp_s": 0.1,
                        "width": 800,
                        "height": 600,
                        "tracks": [{
                            "track_id": 1,
                            "category": "vehicle",
                            "class_name": "car",
                            "bbox_xyxy": [382.0, 270.0, 462.0, 350.0],
                            "anchor_xy": [422.0, 350.0],
                        }],
                    }),
                ]),
                encoding="utf-8",
            )
            pipe.process_tracks(tracks_file, root / "out_tracks")
            snap = pipe._latest_frame_snapshot
            self.assertIsInstance(snap, dict)
            self.assertEqual(snap["frameIndex"], 1)
            self.assertEqual(len(snap["vehicles"]), 1)
            self.assertEqual(snap["vehicles"][0]["trackId"], 1)

    def test_sparse_video_frame_map_and_latest_frame_pts_and_csv_object_kind(self):
        import csv as csv_mod
        import threading
        import time as time_mod
        from analysis.service import LatestFrameCapture, _ServiceConflictError

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cal = RoadCalibrator(
                [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)], 6.0, 15.0
            )
            vid = root / "src_color.mp4"
            # 构造 6 帧色彩区分视频：帧 0 蓝、帧 2 绿、帧 4 红，便于断言稀疏抽取映射精准
            colors = [
                (250, 0, 0),
                (120, 120, 120),
                (0, 250, 0),
                (120, 120, 120),
                (0, 0, 250),
                (120, 120, 120),
            ]
            writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
            self.assertTrue(writer.isOpened())
            for bgr in colors:
                img = np.zeros((96, 128, 3), dtype=np.uint8)
                img[:, :] = bgr
                writer.write(img)
            writer.release()

            tracks_path = root / "sparse_tracks.jsonl"
            sparse_frames = []
            for f_idx, ts in ((0, 0.0), (2, 0.2), (4, 0.4)):
                sparse_frames.append(json.dumps({
                    "schema_version": 1,
                    "session_id": "s-sparse",
                    "source": str(vid),
                    "frame_index": f_idx,
                    "timestamp_s": ts,
                    "timestamp_source": "video_pts",
                    "width": 128,
                    "height": 96,
                    "tracks": [
                        {
                            "track_id": 1,
                            "class_id": 5,
                            "class_name": "bus",
                            "category": "vehicle",
                            "confidence": 0.9,
                            "bbox_xyxy": [40.0, 40.0, 60.0, 70.0],
                            "anchor_xy": [50.0, 70.0],
                        },
                        {
                            "track_id": 2,
                            "class_id": 1,
                            "class_name": "bicycle",
                            "category": "vehicle",
                            "confidence": 0.9,
                            "bbox_xyxy": [10.0, 20.0, 20.0, 40.0],
                            "anchor_xy": [15.0, 40.0],
                        },
                    ],
                }))
            tracks_path.write_text("\n".join(sparse_frames) + "\n", encoding="utf-8")

            runs_root = root / "runs"
            run_out = runs_root / "run-sparse"
            pipe = ConflictPerceptionPipeline(calibrator=cal)
            summary = pipe.process_tracks(tracks_path, run_out, video_path=vid)
            self.assertEqual(summary["frames_processed"], 3)
            self.assertTrue((run_out / "video_frame_map.json").is_file())

            with (run_out / "speeds.csv").open("r", encoding="utf-8", newline="") as f_csv:
                csv_rows = list(csv_mod.DictReader(f_csv))
            self.assertEqual(csv_rows[0]["object_kind"], "bus")
            self.assertEqual(float(csv_rows[0]["length_m"]), 10.0)
            self.assertEqual(csv_rows[1]["object_kind"], "bicycle")
            self.assertEqual(float(csv_rows[1]["length_m"]), 1.8)

            svc = AnalysisService(runs_root=runs_root)
            for f_idx, exp_chan in ((0, 0), (2, 1), (4, 2)):
                jpg = svc.get_run_frame_jpeg("run-sparse", frame_index=f_idx)
                arr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                patch = arr[82:92, 108:124].mean(axis=(0, 1))
                self.assertEqual(int(np.argmax(patch)), exp_chan)
            with self.assertRaises(FileNotFoundError):
                svc.get_run_frame_jpeg("run-sparse", frame_index=1)
            with self.assertRaises(FileNotFoundError):
                svc.get_run_frame_jpeg("run-sparse", frame_index=5)

            # LatestFrameCapture 原子配对 (frame, pts)：消费 40ms 帧期间后台提前到 80ms，仍取到 40ms PTS
            adv_evt = threading.Event()
            read_call = {"n": 0}
            cur_pts = {"val": 0.0}

            class _SyncStubCap:
                def isOpened(self):
                    return True

                def read(self):
                    read_call["n"] += 1
                    n = read_call["n"]
                    if n == 1:
                        cur_pts["val"] = 40.0
                        return True, np.full((8, 8, 3), 40, dtype=np.uint8)
                    if n == 2:
                        adv_evt.wait(timeout=2.0)
                        cur_pts["val"] = 80.0
                        return True, np.full((8, 8, 3), 80, dtype=np.uint8)
                    return False, None

                def get(self, prop):
                    return cur_pts["val"]

                def release(self):
                    adv_evt.set()

            lfc = LatestFrameCapture(_SyncStubCap(), read_timeout_s=1.0)
            try:
                ok1, frm1, pts1 = lfc.read_with_pts(timeout_s=1.0)
                self.assertTrue(ok1)
                self.assertEqual(int(frm1[0, 0, 0]), 40)
                self.assertEqual(pts1, 40.0)
                adv_evt.set()
                time_mod.sleep(0.05)
                self.assertEqual(lfc.get(cv2.CAP_PROP_POS_MSEC), 40.0)
            finally:
                lfc.release()

            # start_job 活动输出目录并发互斥，返回 _ServiceConflictError (409)
            started_evt = threading.Event()
            release_evt = threading.Event()

            class _SlowPipeline:
                def __init__(self, **kwargs):
                    pass

                def process_video(self, **kwargs):
                    started_evt.set()
                    release_evt.wait(timeout=2.0)
                    return {"status": "ok", "frames_processed": 0}

            shared_out = runs_root / "shared-job-dir"
            j1 = svc.start_job(
                str(vid), calibrator=cal, output_dir=shared_out, pipeline_factory=_SlowPipeline
            )
            self.assertTrue(started_evt.wait(timeout=2.0))
            with self.assertRaises(_ServiceConflictError):
                svc.start_job(
                    str(vid), calibrator=cal, output_dir=shared_out, pipeline_factory=_SlowPipeline
                )
            release_evt.set()
            svc.stop_job(j1["job_id"], wait_timeout_s=2.0)
            j2 = svc.start_job(
                str(vid), calibrator=cal, output_dir=shared_out, pipeline_factory=_SlowPipeline
            )
            svc.stop_job(j2["job_id"], wait_timeout_s=2.0)

    def test_reused_output_dir_cleans_stale_media_and_timeline_covers_full_run(self):
        from analysis.service import create_fastapi_app, create_http_server

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cal = RoadCalibrator(
                [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)], 6.0, 15.0
            )
            vid = root / "src.mp4"
            writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
            self.assertTrue(writer.isOpened())
            for idx in range(5):
                frame = np.full((96, 128, 3), 20 + idx * 30, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            tracks_sparse = root / "sparse.jsonl"
            tracks_sparse.write_text(
                "\n".join(
                    json.dumps({
                        "schema_version": 1,
                        "session_id": "s-1",
                        "source": str(vid),
                        "frame_index": f_idx,
                        "timestamp_s": f_idx * 0.1,
                        "timestamp_source": "video_pts",
                        "width": 128,
                        "height": 96,
                        "tracks": [{
                            "track_id": 1,
                            "class_id": 2,
                            "class_name": "car",
                            "category": "vehicle",
                            "confidence": 0.9,
                            "bbox_xyxy": [40.0, 40.0, 60.0, 70.0],
                            "anchor_xy": [50.0, 70.0],
                        }],
                    })
                    for f_idx in (0, 2, 4)
                ) + "\n",
                encoding="utf-8",
            )
            tracks_only = root / "tracks_only.jsonl"
            tracks_only.write_text(
                "\n".join(
                    json.dumps({
                        "schema_version": 1,
                        "session_id": "s-2",
                        "source": "offline",
                        "frame_index": f_idx,
                        "timestamp_s": f_idx * 0.1,
                        "timestamp_source": "video_pts",
                        "width": 128,
                        "height": 96,
                        "tracks": [{
                            "track_id": 9,
                            "class_id": 2,
                            "class_name": "car",
                            "category": "vehicle",
                            "confidence": 0.9,
                            "bbox_xyxy": [42.0, 42.0, 62.0, 72.0],
                            "anchor_xy": [52.0, 72.0],
                        }],
                    })
                    for f_idx in (0, 1)
                ) + "\n",
                encoding="utf-8",
            )

            runs_root = root / "runs"
            reused = runs_root / "reused-dir"
            pipe = ConflictPerceptionPipeline(calibrator=cal)
            s1 = pipe.process_tracks(tracks_sparse, reused, video_path=vid)
            self.assertIsNotNone(s1["annotated_output"])
            self.assertIsNotNone(s1["video_frame_map"])

            svc = AnalysisService(runs_root=runs_root)
            sum1 = svc.get_run_summary("reused-dir")
            self.assertEqual(sum1["video_frame_indices"], [0, 2, 4])

            # 复用同一 output_dir 运行纯轨迹分析后，必须清理旧视频和旧映射，服务接口也必须拒绝返回旧媒体
            s2 = pipe.process_tracks(tracks_only, reused, video_path=None)
            self.assertIsNone(s2["annotated_output"])
            self.assertIsNone(s2["video_frame_map"])
            self.assertFalse((reused / "annotated_conflict.mp4").exists())
            self.assertFalse((reused / "video_frame_map.json").exists())
            sum2 = svc.get_run_summary("reused-dir")
            self.assertFalse(sum2["has_video"])
            self.assertFalse(sum2["has_video_frame_map"])
            with self.assertRaises(FileNotFoundError):
                svc.get_run_frame_jpeg("reused-dir", frame_index=2)

            # 即便磁盘上意外残留旧文件，summary 显式声明 annotated_output: null 时仍应屏蔽
            (reused / "annotated_conflict.mp4").write_bytes(b"stale")
            self.assertFalse(svc.get_run_summary("reused-dir")["has_video"])
            with self.assertRaises(FileNotFoundError):
                svc.get_run_video_path("reused-dir")
            with self.assertRaises(FileNotFoundError):
                svc.resolve_run_export_file("reused-dir", "annotated_conflict.mp4")

            # 501 帧 x 85 目标全量 /timeline 测试（覆盖 stdlib HTTP 与 FastAPI）
            dense_dir = runs_root / "dense-501"
            dense_dir.mkdir(parents=True, exist_ok=True)
            (dense_dir / "summary.json").write_text(
                json.dumps({"frames_processed": 501, "speed_limit_kmh": 30.0}),
                encoding="utf-8",
            )
            with (dense_dir / "speeds.csv").open("w", encoding="utf-8", newline="") as f_csv:
                w = csv.writer(f_csv)
                w.writerow(["frame_idx", "timestamp_s", "track_id", "category", "x_m", "y_m", "speed_kmh", "risk_level", "ttc_s"])
                for fi in range(501):
                    for oi in range(85):
                        spd = 42.0 if fi == 499 and oi == 0 else 12.0
                        risk = "LEVEL_3_CRITICAL" if fi == 499 and oi == 0 else "LEVEL_0_SAFE"
                        ttc = "1.1" if fi == 499 and oi == 0 else ""
                        w.writerow([fi, round(fi * 0.1, 3), oi + 1, "car", 1.0, 2.0, spd, risk, ttc])

            server = create_http_server(svc, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                with urlopen(f"http://127.0.0.1:{port}/api/runs/dense-501/timeline?max_points=2000", timeout=5.0) as resp:
                    tl = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(tl["total_frames"], 501)
                self.assertEqual(tl["point_count"], 501)
                self.assertEqual(tl["points"][-1]["frame_index"], 500)
                p499 = next(p for p in tl["points"] if p["frame_index"] == 499)
                self.assertAlmostEqual(p499["speed_kmh"], 42.0, places=2)
                self.assertEqual(p499["risk_level"], 3)
            finally:
                server.shutdown()
                server.server_close()

            import importlib.util
            if importlib.util.find_spec("fastapi") is not None:
                fastapi_app = create_fastapi_app(svc)
                self.assertIsNotNone(fastapi_app)
                routes = {r.path for r in fastapi_app.routes}
                self.assertIn("/api/runs/{run_id}/timeline", routes)
            else:
                src_text = Path(inspect.getfile(create_fastapi_app)).read_text(encoding="utf-8")
                self.assertIn('@app.get("/api/runs/{run_id}/timeline")', src_text)


if __name__ == "__main__":
    unittest.main()
