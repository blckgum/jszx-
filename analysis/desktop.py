# -*- coding: utf-8 -*-
"""
CampusMotion 独立桌面应用宿主模块 (Standalone Desktop App-Mode Host)
=============================================================================
提供零原生打包 (.exe) 的独立桌面启动器，兼容 `python -m analysis.desktop` 与
`pythonw -m analysis.desktop`：
1. `%LOCALAPPDATA%/CampusMotion/instance.json` 单实例检测与复用唤醒；
2. 自动扫描 `8000..8020` 可用本地端口并启动 `AnalysisService(desktop_mode=True, allow_shutdown=True)`；
3. 自动探测 Edge / Chrome 并以独立 `--app=...` 桌面窗口模式启动（支持 `--browser-only` 回退）；
4. 监听应用窗口关闭或 `POST /api/system/shutdown` 信号，有序停止后台分析任务并清理状态文件。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import List, Optional
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

from detection.core import is_blank_path
from .service import AnalysisService, create_http_server


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT_START = 8000
DEFAULT_PORT_END = 8020


def get_app_data_dir() -> Path:
    """返回本地应用运行状态与日志目录（优先使用 %LOCALAPPDATA%/CampusMotion）。"""
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data).expanduser() / "CampusMotion"
    return Path.home() / ".campus_motion"


def _ensure_stdio_for_windowless() -> None:
    """在 pythonw 无控制台窗口环境下为 None 的标准流兜底，防止下游 print 报错。"""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _create_logger(log_path: Path) -> tuple[logging.Logger, logging.FileHandler]:
    """配置独立的桌面宿主文件日志器，返回 (logger, file_handler) 以便退出时释放句柄。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"campus_motion.desktop.{os.getpid()}.{id(log_path)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)
    return logger, file_handler


def _is_health_ok(base_url: str, *, timeout_s: float = 0.8) -> bool:
    """向指定 URL 的 /api/health 发送探活请求并校验返回状态为 ok。"""
    if not isinstance(base_url, str) or not base_url.strip():
        return False
    clean_base = base_url.strip().rstrip("/")
    if not (clean_base.startswith("http://") or clean_base.startswith("https://")):
        return False
    try:
        with urlopen(f"{clean_base}/api/health", timeout=timeout_s) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
            return isinstance(payload, dict) and payload.get("status") == "ok"
    except (URLError, OSError, ValueError, json.JSONDecodeError):
        return False


def _is_tcp_port_in_use(host: str, port: int, *, timeout_s: float = 0.15) -> bool:
    """在绑定前探测本机端口是否已有监听服务，避免 Windows SO_REUSEADDR 冲突。"""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    try:
        with socket.create_connection((probe_host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def find_chromium_app_binary() -> Optional[Path]:
    """跨平台探测支持独立 --app= 窗口模式的 Microsoft Edge 或 Google Chrome 可执行文件。"""
    candidates: List[Path] = []
    if sys.platform.startswith("win"):
        roots = [
            os.environ.get("PROGRAMFILES(X86)", ""),
            os.environ.get("PROGRAMFILES", ""),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        rel_bins = (
            Path("Microsoft") / "Edge" / "Application" / "msedge.exe",
            Path("Google") / "Chrome" / "Application" / "chrome.exe",
        )
        for root_str in roots:
            clean = root_str.strip()
            if not clean:
                continue
            for rel in rel_bins:
                candidates.append(Path(clean) / rel)
        for cmd in ("msedge.exe", "msedge", "chrome.exe", "chrome"):
            found = shutil.which(cmd)
            if found:
                candidates.append(Path(found))
    elif sys.platform == "darwin":
        candidates.extend([
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        ])
    else:
        for cmd in (
            "microsoft-edge",
            "microsoft-edge-stable",
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
        ):
            found = shutil.which(cmd)
            if found:
                candidates.append(Path(found))

    for exe_path in candidates:
        try:
            if exe_path.is_file():
                return exe_path
        except OSError:
            continue
    return None


def _build_app_mode_cmd(browser_exe: Path, url: str, profile_dir: Path) -> List[str]:
    return [
        str(browser_exe),
        f"--app={url}",
        f"--user-data-dir={profile_dir}",
        "--window-size=1440,920",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def _open_url(
    url: str,
    *,
    app_data_dir: Path,
    browser_only: bool,
    logger: logging.Logger,
) -> Optional[subprocess.Popen]:
    """启动 Edge/Chrome 独立 App-Mode 窗口；若回退浏览器模式则调用 webbrowser.open。"""
    browser_exe = None if browser_only else find_chromium_app_binary()
    if browser_exe is not None:
        profile_dir = app_data_dir / "browser_profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_app_mode_cmd(browser_exe, url, profile_dir)
        logger.info("Launching browser app-mode: %s", cmd[0])
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    logger.info("Launching default system browser for %s", url)
    webbrowser.open(url)
    return None


def _check_existing_instance(instance_path: Path, logger: logging.Logger) -> Optional[str]:
    """检查 instance.json 指向的既有桌面实例是否健康存活；若已失活则清理过期文件。"""
    if not instance_path.is_file():
        return None
    try:
        data = json.loads(instance_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Ignoring unreadable instance file %s: %s", instance_path, exc)
        try:
            instance_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    if not isinstance(data, dict):
        try:
            instance_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    url = data.get("url")
    if not isinstance(url, str) or not url.strip():
        host = data.get("host")
        port = data.get("port")
        if isinstance(host, str) and type(port) is int:
            url = f"http://{host}:{port}/"
        else:
            url = ""

    if url and _is_health_ok(url):
        return url

    try:
        instance_path.unlink(missing_ok=True)
    except OSError:
        pass
    return None


def _bind_server_with_port_scan(
    service: AnalysisService,
    host: str,
    requested_port: Optional[int],
):
    """绑定指定端口或自动扫描 8000..8020 首个可用端口并返回 HTTPServer 实例。"""
    if requested_port is not None:
        if requested_port != 0 and _is_tcp_port_in_use(host, requested_port):
            raise OSError(f"指定端口 {requested_port} 已被占用")
        return create_http_server(service, host=host, port=requested_port)

    last_exc: Optional[Exception] = None
    for candidate in range(DEFAULT_PORT_START, DEFAULT_PORT_END + 1):
        if _is_tcp_port_in_use(host, candidate):
            continue
        try:
            return create_http_server(service, host=host, port=candidate)
        except OSError as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"端口 {DEFAULT_PORT_START}..{DEFAULT_PORT_END} 均已被占用，无法启动桌面服务"
    ) from last_exc


def _write_instance_file(
    instance_path: Path,
    *,
    pid: int,
    host: str,
    port: int,
    url: str,
) -> None:
    instance_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": int(pid),
        "host": host,
        "port": int(port),
        "url": url,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp_path = instance_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp_path.replace(instance_path)


def _cleanup_own_instance_file(instance_path: Path, *, pid: int, port: int) -> None:
    if not instance_path.is_file():
        return
    try:
        data = json.loads(instance_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("pid") == pid and data.get("port") == port:
            instance_path.unlink(missing_ok=True)
    except Exception:
        try:
            instance_path.unlink(missing_ok=True)
        except OSError:
            pass


def run_desktop(
    *,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    runs_root: str | Path = REPO_ROOT / "runs" / "analysis",
    default_calib: str | Path = REPO_ROOT / "configs" / "calibration.yaml",
    browser_only: bool = False,
    no_open: bool = False,
    app_data_dir: Optional[str | Path] = None,
) -> int:
    """运行独立桌面应用宿主直到浏览器窗口关闭或收到服务关停信号。"""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host 不能为空")
    if port is not None and (type(port) is not int or not (0 <= port <= 65535)):
        raise ValueError("port 必须是 0..65535 范围内的整数")
    if is_blank_path(runs_root) or type(runs_root) is bool:
        raise ValueError("runs_root 不能为空")
    if is_blank_path(default_calib) or type(default_calib) is bool:
        raise ValueError("default_calib 不能为空")

    data_dir = (
        Path(app_data_dir).expanduser().resolve()
        if app_data_dir is not None
        else get_app_data_dir().resolve()
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "desktop.log"
    instance_path = data_dir / "instance.json"

    logger, file_handler = _create_logger(log_path)
    try:
        existing_url = _check_existing_instance(instance_path, logger)
        if existing_url is not None:
            logger.info("Existing CampusMotion instance alive at %s", existing_url)
            if not no_open:
                _open_url(
                    existing_url,
                    app_data_dir=data_dir,
                    browser_only=browser_only,
                    logger=logger,
                )
            return 0

        shutdown_event = threading.Event()
        service = AnalysisService(
            runs_root=runs_root,
            default_calib=default_calib,
            desktop_mode=True,
            allow_shutdown=True,
            shutdown_callback=shutdown_event.set,
        )
        server = _bind_server_with_port_scan(service, host, port)
        bound_host, bound_port = server.server_address[:2]
        display_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::", "") else bound_host
        url = f"http://{display_host}:{bound_port}/"

        server_thread = threading.Thread(
            target=server.serve_forever,
            name="CampusMotionDesktopHTTP",
            daemon=True,
        )
        server_thread.start()

        browser_proc: Optional[subprocess.Popen] = None
        try:
            ready_deadline = time.monotonic() + 3.0
            while time.monotonic() < ready_deadline:
                if _is_health_ok(url, timeout_s=0.3):
                    break
                time.sleep(0.025)

            _write_instance_file(
                instance_path,
                pid=os.getpid(),
                host=display_host,
                port=int(bound_port),
                url=url,
            )
            logger.info("CampusMotion desktop server listening at %s", url)

            if not no_open:
                browser_proc = _open_url(
                    url,
                    app_data_dir=data_dir,
                    browser_only=browser_only,
                    logger=logger,
                )

            try:
                while not shutdown_event.is_set():
                    if browser_proc is not None and browser_proc.poll() is not None:
                        logger.info(
                            "Browser app process exited with return code %s",
                            browser_proc.returncode,
                        )
                        break
                    shutdown_event.wait(timeout=0.25)
            except KeyboardInterrupt:
                logger.info("Received KeyboardInterrupt; shutting down desktop host.")
        finally:
            if browser_proc is not None and browser_proc.poll() is None:
                try:
                    browser_proc.terminate()
                    browser_proc.wait(timeout=2.0)
                except Exception:
                    pass
            try:
                service.stop_active_jobs(wait_timeout_s=2.0)
            except Exception as exc:
                logger.warning("Error stopping active jobs: %s", exc)
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
            _cleanup_own_instance_file(
                instance_path, pid=os.getpid(), port=int(bound_port)
            )
            logger.info("CampusMotion desktop host stopped cleanly.")
        return 0
    except Exception as exc:
        logger.exception("Desktop host encountered fatal error: %s", exc)
        raise
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_stdio_for_windowless()
    parser = argparse.ArgumentParser(
        description="CampusMotion 独立桌面应用宿主 (Edge/Chrome App-Mode)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="HTTP 服务监听主机地址，默认 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP 服务监听端口；不指定时自动探测 8000..8020 空闲端口",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(REPO_ROOT / "runs" / "analysis"),
        help="历史运行结果根目录，默认 <repo>/runs/analysis",
    )
    parser.add_argument(
        "--default-calib",
        type=str,
        default=str(REPO_ROOT / "configs" / "calibration.yaml"),
        help="默认标定 YAML 配置路径，默认 <repo>/configs/calibration.yaml",
    )
    parser.add_argument(
        "--browser-only",
        action="store_true",
        help="强制回退使用系统默认浏览器标签页打开，而不启动独立 App 窗口",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="仅启动桌面模式后端服务，不自动打开浏览器窗口",
    )
    args = parser.parse_args(argv)

    if not args.host.strip():
        parser.error("--host 不能为空")
    if args.port is not None and not (0 <= args.port <= 65535):
        parser.error("--port 必须是 0..65535 范围内的整数")
    if not args.runs_root.strip():
        parser.error("--runs-root 不能为空")
    if not args.default_calib.strip():
        parser.error("--default-calib 不能为空")

    return run_desktop(
        host=args.host,
        port=args.port,
        runs_root=args.runs_root,
        default_calib=args.default_calib,
        browser_only=bool(args.browser_only),
        no_open=bool(args.no_open),
    )


if __name__ == "__main__":
    raise SystemExit(main())
