# -*- coding: utf-8 -*-
"""
校园道路车辆—行人冲突感知与测速预警完整流水线 (End-to-End Conflict Perception Pipeline)
=============================================================================
集成 TrackingSession 目标追踪、RoadCalibrator 单应性投影、
SpeedEstimator 平滑测速、ConflictAnalyzer 碰撞决策与 EnhancedVisualizer 可视化，
实现从视频输入到结构化风险告警的端到端技术闭环。
"""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from detection.core import DEFAULT_WEIGHTS
from tracking.core import DEFAULT_TRACKER, TrackingSession, VideoClock
from .calibration import RoadCalibrator
from .conflict import (
    ConflictAnalyzer,
    ConflictAssessment,
    RiskLevel,
    build_conflict_maps,
)
from .events import ConflictEventTracker
from .speed import SpeedEstimator, SpeedReading
from .visualization import EnhancedConflictVisualizer, trajectory_color


class ConflictPerceptionPipeline:
    """冲突感知与预警全流程管道执行器"""

    def __init__(
        self,
        calibrator: RoadCalibrator,
        weights: str = str(DEFAULT_WEIGHTS),
        tracker_config: str = str(DEFAULT_TRACKER),
        speed_limit_kmh: float = 30.0,
        device: Optional[str] = None,
        conf: Optional[float] = None,
        iou: float = 0.7,
        imgsz: int = 640,
        ema_alpha: float = 0.35,
        window_size: int = 8,
        window_seconds: Optional[float] = None,
        zone_margin_m: float = 2.0,
        ttc_critical_s: float = 1.8,
        ttc_caution_s: float = 3.5,
        horizon_s: float = 3.5,
        stationary_speed_ms: float = 0.1,
        stationary_min_frames: int = 8,
        stationary_seconds: Optional[float] = None,
        category_confirm_frames: int = 3,
        reliable_min_span_s: float = 0.2,
        alert_confirm_frames: int = 3,
        alert_confirm_seconds: Optional[float] = None,
        alert_cooldown_frames: int = 12,
        alert_cooldown_seconds: Optional[float] = None,
        fps: Optional[float] = None,
    ):
        if not math.isfinite(iou) or not (0 < iou <= 1):
            raise ValueError("iou 必须在 (0, 1] 区间")
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("fps 必须是有限正数")
        if type(imgsz) is not int or imgsz <= 0:
            raise ValueError("imgsz 必须是正整数")
        if not math.isfinite(zone_margin_m) or zone_margin_m < 0:
            raise ValueError("zone_margin_m 必须为有限非负数")
        if window_seconds is not None and (not math.isfinite(window_seconds) or window_seconds <= 0):
            raise ValueError("window_seconds 必须为有限正数")
        if stationary_seconds is not None and (not math.isfinite(stationary_seconds) or stationary_seconds <= 0):
            raise ValueError("stationary_seconds 必须为有限正数")
        if type(alert_confirm_frames) is not int or alert_confirm_frames < 1:
            raise ValueError("alert_confirm_frames 必须是 >= 1 的整数帧数")
        if alert_confirm_seconds is not None and (
            not math.isfinite(alert_confirm_seconds) or alert_confirm_seconds <= 0
        ):
            raise ValueError("alert_confirm_seconds 必须为有限正数")
        if type(alert_cooldown_frames) is not int or alert_cooldown_frames < 0:
            raise ValueError("alert_cooldown_frames 必须是 >= 0 的整数帧数")
        if alert_cooldown_seconds is not None and (
            not math.isfinite(alert_cooldown_seconds) or alert_cooldown_seconds < 0
        ):
            raise ValueError("alert_cooldown_seconds 必须为有限非负数")

        self.calibrator = calibrator
        self.weights = weights
        self.tracker_config = tracker_config
        self.speed_limit_kmh = speed_limit_kmh
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.zone_margin_m = zone_margin_m
        self.window_seconds = window_seconds
        self.stationary_seconds = stationary_seconds
        self.alert_confirm_frames = alert_confirm_frames
        self.alert_confirm_seconds = alert_confirm_seconds
        self.alert_cooldown_frames = alert_cooldown_frames
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.fps = fps

        # 每次 process_video 按实际 FPS 重建 estimator，跨视频不串状态
        self._estimator_kwargs = dict(
            speed_limit_kmh=speed_limit_kmh,
            window_size=window_size,
            ema_alpha=ema_alpha,
            stationary_speed_ms=stationary_speed_ms,
            stationary_min_frames=stationary_min_frames,
            category_confirm_frames=category_confirm_frames,
            reliable_min_span_s=reliable_min_span_s,
        )
        self.speed_estimator = SpeedEstimator(**self._estimator_kwargs)
        self.event_tracker = ConflictEventTracker(
            confirm_frames=alert_confirm_frames,
            cooldown_frames=alert_cooldown_frames,
        )
        self.conflict_analyzer = ConflictAnalyzer(
            speed_limit_kmh=speed_limit_kmh,
            ttc_critical_s=ttc_critical_s,
            ttc_caution_s=ttc_caution_s,
            horizon_s=horizon_s,
        )
        self.visualizer = EnhancedConflictVisualizer(
            calibrator=calibrator, speed_limit_kmh=speed_limit_kmh
        )

    def _configure_speed_estimator(self, fps: float):
        """按视频实际 FPS 重建速度估计器：换算秒级窗口，且不继承上一视频状态"""
        kwargs = dict(self._estimator_kwargs)
        if self.window_seconds is not None:
            # 以秒表达的测速窗口：按实际 FPS 换算帧数，跨帧率视频保持一致时间窗
            kwargs["window_size"] = max(3, int(round(fps * self.window_seconds)))
        if self.stationary_seconds is not None:
            kwargs["stationary_min_frames"] = max(1, int(round(fps * self.stationary_seconds)))
        self.speed_estimator = SpeedEstimator(**kwargs)

    def _configure_event_tracker(self, fps: float):
        """按视频实际 FPS 重建告警事件跟踪器：换算秒级窗口，且不继承上一视频状态"""
        confirm_frames = self.alert_confirm_frames
        if self.alert_confirm_seconds is not None:
            # 秒级确认窗：60fps 的 3 帧只有 50ms，按帧计数跨帧率语义不一致
            confirm_frames = max(1, int(round(fps * self.alert_confirm_seconds)))
        cooldown_frames = self.alert_cooldown_frames
        if self.alert_cooldown_seconds is not None:
            cooldown_frames = max(0, int(round(fps * self.alert_cooldown_seconds)))
        self.event_tracker = ConflictEventTracker(
            confirm_frames=confirm_frames,
            cooldown_frames=cooldown_frames,
        )

    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        show_window: bool = False,
        max_frames: Optional[int] = None,
    ) -> dict:
        """端到端处理整段视频，输出渲染视频与结构化告警日志"""
        if max_frames is not None and (type(max_frames) is not int or max_frames < 0):
            raise ValueError("max_frames 必须是 >= 0 的整数")

        src = Path(video_path).expanduser().resolve()
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频输入文件: {src}")

        # ExitStack 统一登记资源生命周期：初始化阶段（模型加载、编码器、
        # 文件打开）与主循环异常都保证释放，不存在半初始化泄漏路径
        resources = ExitStack()
        resources.callback(cap.release)
        try:
            # 与 tracking 模块同策略：源 FPS 缺失/无效时必须显式 --fps，
            # 静默兜底 25fps 会让测速窗口与 TTC 建立在错误时间基准上
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            source_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
            fps = self.fps if self.fps is not None else source_fps
            if fps is None:
                raise RuntimeError("视频缺少有效 FPS；确认帧率后通过 --fps 指定")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            video_clock = VideoClock(fps=fps, force_fps=self.fps is not None)
            self._configure_speed_estimator(fps)
            self._configure_event_tracker(fps)
            session = TrackingSession(
                source=str(src),
                weights=self.weights,
                tracker_path=self.tracker_config,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
            )

            out_video_path = out_dir / "annotated_conflict.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
            resources.callback(writer.release)
            if not writer.isOpened():
                raise RuntimeError(f"无法创建渲染输出视频，请检查编码器与路径: {out_video_path}")

            jsonl_path = out_dir / "conflicts.jsonl"
            events_path = out_dir / "events.jsonl"
            csv_path = out_dir / "speeds.csv"

            jsonl_file = resources.enter_context(open(jsonl_path, "w", encoding="utf-8"))
            events_file = resources.enter_context(open(events_path, "w", encoding="utf-8"))
            csv_file = resources.enter_context(open(csv_path, "w", encoding="utf-8", newline=""))

            # 表头写入也在保护范围内：此刻磁盘失败不能跳过资源释放
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "frame_idx", "timestamp_s", "track_id", "category",
                "x_m", "y_m", "speed_kmh", "is_speeding", "risk_level", "ttc_s",
                "event_id", "is_reliable", "is_stationary"
            ])
        except BaseException:
            # 初始化任何一步失败：释放已登记的 cap/writer/已开文件
            resources.close()
            raise

        total_critical_events = 0
        total_caution_events = 0
        total_speeding_events = 0
        max_observed_speed_kmh = 0.0

        def _drain_alert_events():
            # 事件级计数：一次持续危险只记 1 条，峰值等级为事件归档等级
            nonlocal total_critical_events, total_caution_events, total_speeding_events
            for evt in self.event_tracker.pop_finalized():
                if evt.peak_risk_level == RiskLevel.LEVEL_3_CRITICAL:
                    total_critical_events += 1
                elif evt.peak_risk_level == RiskLevel.LEVEL_2_CAUTION:
                    total_caution_events += 1
                elif evt.peak_risk_level == RiskLevel.LEVEL_1_SPEEDING:
                    total_speeding_events += 1
                events_file.write(json.dumps(evt.to_dict(), ensure_ascii=False) + "\n")

        frame_idx = 0
        t_start = time.time()

        print("=" * 60)
        print(f"[*] 启动校园道路冲突感知与测速预警系统:")
        print(f"    输入视频: {src.name} ({width}x{height} @ {fps:.1f} FPS)")
        print(f"    标定路网: {self.calibrator.name} (宽 {self.calibrator.width_m}m, 长 {self.calibrator.length_m}m)")
        print(f"    限速标准: {self.speed_limit_kmh} km/h")
        print("=" * 60)

        loop_state = "eof"
        loop_error = ""
        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    # 人为截断与自然结束区分归档，事件日志不混淆结束原因
                    loop_state = "max_frames"
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_s, time_kind = video_clock.timestamp(frame_idx, pts_ms)

                # 1. 目标检测与 ByteTrack 多目标追踪
                frame_record = session.process(frame, timestamp_s, time_kind)
                speed_readings: List[SpeedReading] = []
                reading_map: Dict[int, SpeedReading] = {}
                out_of_zone_ids = set()

                # 2. 空间坐标单应性变换与平滑测速
                for t in frame_record.tracks:
                    if t.track_id is None:
                        continue

                    # 底边中心点转换为地面米制物理坐标
                    u_foot, v_foot = t.anchor_xy
                    x_m, y_m = self.calibrator.pixel_to_ground(u_foot, v_foot)
                    if not (math.isfinite(x_m) and math.isfinite(y_m)):
                        # 地平线附近退化投影点：不产生读数，防止 NaN 污染速度历史
                        continue

                    # 标定四边形以外的坐标是外推伪影：先于滤波器更新就门控，
                    # 不进入速度历史，也不参与冲突研判与速度统计
                    if not self.calibrator.is_in_zone(x_m, y_m, margin_m=self.zone_margin_m):
                        out_of_zone_ids.add(t.track_id)
                        continue

                    reading = self.speed_estimator.update(
                        track_id=t.track_id,
                        category=t.category,
                        timestamp_s=timestamp_s,
                        x_m=x_m,
                        y_m=y_m,
                    )
                    reading_map[t.track_id] = reading
                    speed_readings.append(reading)

                    if reading.speed_kmh > max_observed_speed_kmh:
                        max_observed_speed_kmh = reading.speed_kmh

                # ByteTrack 的 lost 池仍可能找回目标并沿用原 ID，按跟踪器存活集
                # 剪枝而非仅按本帧可见 ID，避免短暂遮挡导致速度滤波器重置；
                # 区外轨迹的滤波器状态一并清除，重回区内时重新预热
                try:
                    live_ids = session.retained_ids()
                except Exception as exc:
                    # 内部结构变化说明 Ultralytics 版本不兼容；预警系统宁可终止
                    # 也不静默降级为"按本帧可见 ID 剪枝"而恢复遮挡闪断问题
                    raise RuntimeError(
                        "无法读取跟踪器存活池 (tracked/lost stracks)，"
                        "当前 Ultralytics 版本接口不兼容；请固定 ultralytics==8.4.182 或更新适配"
                    ) from exc
                live_ids -= out_of_zone_ids
                self.speed_estimator.prune_stale_tracks(live_ids)

                # 3. 车辆与行人轨迹外推预测与 TTC 碰撞研判
                conflicts = self.conflict_analyzer.analyze_frame(speed_readings)
                conflict_map, ped_conflict_map = build_conflict_maps(conflicts)

                # 3b. 告警事件去抖：逐帧评估收敛为生命周期事件；
                # 已确认存续的事件把 event_id 回填到本帧原始评估记录上
                event_ids = self.event_tracker.update(conflicts, frame_idx, timestamp_s)
                _drain_alert_events()

                # 4. 统计与持久化记录
                for c in conflicts:
                    json_item = {
                        "frame_index": frame_idx,
                        "timestamp_s": round(timestamp_s, 3),
                        "event_id": event_ids.get((c.veh_track_id, c.ped_track_id)),
                        "assessment": asdict(c),
                    }
                    jsonl_file.write(json.dumps(json_item, ensure_ascii=False) + "\n")

                for r in speed_readings:
                    c = conflict_map.get(r.track_id) or ped_conflict_map.get(r.track_id)
                    risk_lvl = c.risk_level.name if c else "LEVEL_0_SAFE"
                    ttc_val = round(c.ttc_seconds, 2) if (c and c.ttc_seconds) else ""
                    csv_event_id = (
                        event_ids.get((c.veh_track_id, c.ped_track_id))
                        if c is not None else None
                    )
                    csv_writer.writerow([
                        frame_idx, round(timestamp_s, 3), r.track_id, r.category,
                        round(r.x_m, 2), round(r.y_m, 2), round(r.speed_kmh, 1),
                        int(r.is_speeding), risk_lvl, ttc_val,
                        csv_event_id if csv_event_id is not None else "",
                        int(r.is_reliable), int(r.is_stationary)
                    ])

                # 5. 增强可视化渲染
                render_frame = frame.copy()
                self.visualizer.draw_calibrated_ground_zone(render_frame)

                for r in speed_readings:
                    # 绘制预测轨迹虚线；冲突对中的行人同样按风险等级着色
                    c = conflict_map.get(r.track_id) or ped_conflict_map.get(r.track_id)
                    self.visualizer.draw_future_trajectory(
                        render_frame, r, color=trajectory_color(c)
                    )

                for t in frame_record.tracks:
                    if t.track_id is None:
                        continue
                    r = reading_map.get(t.track_id)
                    if r:
                        c = conflict_map.get(t.track_id) or ped_conflict_map.get(t.track_id)
                        eid = (
                            event_ids.get((c.veh_track_id, c.ped_track_id))
                            if c is not None else None
                        )
                        self.visualizer.draw_speed_badge(
                            render_frame, t.bbox_xyxy, r, c, event_id=eid
                        )

                for c in conflicts:
                    self.visualizer.draw_conflict_link(render_frame, c, reading_map)

                self.visualizer.draw_hud_banner(render_frame, conflicts, event_ids)

                writer.write(render_frame)

                # 帧处理到此已完整（日志/视频均已落盘），立即计数——
                # 放在 q 退出与异常路径之前，否则截断时最后一帧漏计
                frame_idx += 1

                if show_window:
                    cv2.imshow("Campus Vehicle-Pedestrian Conflict System", render_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        loop_state = "user_quit"
                        break

                if frame_idx % 30 == 0:
                    fps_proc = frame_idx / (time.time() - t_start)
                    print(f"[*] 已处理 {frame_idx} 帧 | 处理速率: {fps_proc:.1f} FPS | 最高时速: {max_observed_speed_kmh:.1f} km/h")

        except KeyboardInterrupt:
            loop_state = "interrupt"
            loop_error = "KeyboardInterrupt"
            raise
        except BaseException as exc:
            loop_state = "exception"
            loop_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # 无论正常结束还是异常退出，都尽力归档未决事件；
            # flush 或写盘自身的失败不得掩盖原始异常
            try:
                self.event_tracker.flush(reason=loop_state)
                _drain_alert_events()
            except Exception:
                pass
            # 统一释放 ExitStack 登记的 cap/writer/三个输出文件
            resources.close()
            # 异常退出同样落一份 summary：下游仅凭输出目录即可分辨
            # "处理了多少帧/何种失败/输出是否完整"，不必回放日志
            if loop_state in ("interrupt", "exception"):
                try:
                    (out_dir / "summary.json").write_text(
                        json.dumps({
                            "status": "interrupted" if loop_state == "interrupt" else "failed",
                            "stop_reason": loop_state,
                            "error": loop_error,
                            "frames_processed": frame_idx,
                            "elapsed_s": round(time.time() - t_start, 2),
                            "max_observed_speed_kmh": round(max_observed_speed_kmh, 1),
                            "events_jsonl": str(events_path.resolve()),
                        }, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            if show_window:
                cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0

        summary = {
            "status": "ok",
            # eof / max_frames / user_quit：与异常路径中止摘要同契约
            "stop_reason": loop_state,
            "source_video": str(src),
            "annotated_output": str(out_video_path),
            "conflicts_jsonl": str(jsonl_path),
            "events_jsonl": str(events_path),
            "speeds_csv": str(csv_path),
            "frames_processed": frame_idx,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(proc_fps, 1),
            "speed_limit_kmh": self.speed_limit_kmh,
            "max_observed_speed_kmh": round(max_observed_speed_kmh, 1),
            "total_alert_events": (
                total_speeding_events + total_caution_events + total_critical_events
            ),
            "total_speeding_alerts": total_speeding_events,
            "total_caution_events": total_caution_events,
            "total_critical_conflicts": total_critical_events,
        }

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("=" * 60)
        print(f"[OK] 冲突感知与测速研判处理完毕！")
        print(f"    渲染视频已保存: {out_video_path}")
        print(f"    告警明细已落盘: {jsonl_path} | {events_path} | {csv_path}")
        print(f"    最高车速: {max_observed_speed_kmh:.1f} km/h | 紧急冲突事件数: {total_critical_events}")
        print("=" * 60)

        return summary
