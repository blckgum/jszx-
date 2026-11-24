# -*- coding: utf-8 -*-
"""
校园道路车辆—行人冲突感知、测速与 TTC 风险预警模块
"""

from .batch import analyze_directory
from .calibration import CalibrationConfig, RoadCalibrator
from .conflict import ConflictAnalyzer, ConflictAssessment, RiskLevel
from .kalman import ConstantAccelerationKalman2D
from .pipeline import ConflictPerceptionPipeline
from .speed import SpeedEstimator, SpeedReading, TrackMotionTracker
from .ssm import OBBCollisionResult, PETTracker
from .stitching import StitchEvent, TrackObservation, TrackletStitcher
from .visualization import EnhancedConflictVisualizer


def __getattr__(name: str):
    # 延迟导出 .mot_eval 符号，避免 python -m analysis.mot_eval 在 -W error::RuntimeWarning
    # 下触发 runpy 对 sys.modules 预加载子模块的 RuntimeWarning。
    if name in ("evaluate_mot", "evaluate_mot_files"):
        from . import mot_eval

        value = getattr(mot_eval, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CalibrationConfig",
    "RoadCalibrator",
    "ConstantAccelerationKalman2D",
    "evaluate_mot",
    "evaluate_mot_files",
    "ConflictAnalyzer",
    "ConflictAssessment",
    "RiskLevel",
    "OBBCollisionResult",
    "PETTracker",
    "SpeedEstimator",
    "SpeedReading",
    "TrackMotionTracker",
    "StitchEvent",
    "TrackObservation",
    "TrackletStitcher",
    "EnhancedConflictVisualizer",
    "ConflictPerceptionPipeline",
    "analyze_directory",
]
