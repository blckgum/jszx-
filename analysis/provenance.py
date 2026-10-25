"""生成可复现实验所需的环境、Git 与文件指纹；任何探测失败都不阻断主流程。"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path


PACKAGE_NAMES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
    "opencv-python": "opencv-python",
    "lap": "lap",
    "numpy": "numpy",
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """分块计算文件 SHA256，避免把模型权重整体读入内存。"""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path | None) -> dict:
    if path is None:
        return {"path": None, "status": "unavailable", "sha256": None, "size_bytes": None}
    try:
        resolved = Path(path).expanduser().resolve()
    except Exception:
        return {"path": str(path), "status": "unavailable", "sha256": None, "size_bytes": None}
    try:
        if not resolved.is_file():
            return {"path": str(resolved), "status": "missing", "sha256": None, "size_bytes": None}
        return {
            "path": str(resolved),
            "status": "ok",
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    except OSError:
        return {"path": str(resolved), "status": "unreadable", "sha256": None, "size_bytes": None}


def package_versions() -> dict[str, str | None]:
    result = {}
    for label, distribution in PACKAGE_NAMES.items():
        try:
            result[label] = metadata.version(distribution)
        except Exception:
            result[label] = None
    return result


def git_state(repo_root: str | Path) -> dict:
    try:
        root = Path(repo_root).resolve()
    except Exception:
        return {"commit": None, "dirty": None, "status": "unavailable"}

    def run(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5, check=True,
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain"))
        return {"commit": commit, "dirty": dirty, "status": "ok"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "status": "unavailable"}


def accelerator_info() -> dict:
    try:
        import torch
        available = bool(torch.cuda.is_available())
        return {
            "cuda_available": available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_name": torch.cuda.get_device_name(0) if available else None,
        }
    except Exception:
        return {"cuda_available": None, "cuda_version": None, "device_name": None}


def collect_provenance(
    repo_root: str | Path,
    *,
    weights: str | Path | None,
    tracker_config: str | Path | None,
    calibration: str | Path | None,
    object_policy: str | Path | None,
) -> dict:
    """只记录公开运行元数据和指定文件指纹，不读取环境变量或秘密。"""
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine() or None,
            "processor": platform.processor() or None,
            "accelerator": accelerator_info(),
            "packages": package_versions(),
        },
        "git": git_state(repo_root),
        "artifacts": {
            "weights": file_fingerprint(weights),
            "tracker_config": file_fingerprint(tracker_config),
            "calibration": file_fingerprint(calibration),
            "object_policy": file_fingerprint(object_policy),
        },
    }
