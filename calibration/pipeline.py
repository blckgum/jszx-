"""无界面、流式映射既有 tracks/motion JSONL，不重跑网络，不计算速度。"""

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from .geometry import points_array
from .io import new_output, resolved_document, write_json


def reject_constant(value):
    raise ValueError(f"JSON 不允许非有限常量：{value}")


def validate_frame(frame, previous):
    if (not isinstance(frame, dict) or type(frame.get("schema_version")) is not int
            or frame["schema_version"] != 1):
        raise ValueError("仅支持 schema_version=1 的跟踪/运动帧")
    for name in ("session_id", "source", "timestamp_source"):
        if not isinstance(frame.get(name), str) or not frame[name]:
            raise ValueError(f"缺少有效 {name}")
    for name in ("frame_index", "width", "height"):
        value = frame.get(name)
        if type(value) is not int or value < (0 if name == "frame_index" else 1):
            raise ValueError(f"无效的 {name}")
    timestamp = frame.get("timestamp_s")
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("源时间戳必须是非负、有限秒数")
    if frame["timestamp_source"] not in {"video_pts", "fps_estimate"}:
        raise ValueError("未知的源时间戳类型")
    if type(frame.get("synthetic", False)) is not bool:
        raise ValueError("synthetic 必须是布尔值")
    if previous is not None:
        if frame["session_id"] != previous["session_id"] or frame["source"] != previous["source"]:
            raise ValueError("每个输入文件只能包含同一个视频会话；不同视频请分别映射")
        if frame["frame_index"] <= previous["frame_index"] or timestamp <= previous["timestamp_s"]:
            raise ValueError("源帧号和时间戳必须严格递增")
    if not isinstance(frame.get("tracks"), list):
        raise ValueError("tracks 必须是列表，空帧保留空列表")
    ids = set()
    for target in frame["tracks"]:
        if not isinstance(target, dict) or target.get("observation") != "detected":
            raise ValueError("只映射 detected 观测，不补造预测框")
        if target.get("anchor_type") != "bbox_bottom_center":
            raise ValueError("当前入口要求明确的 bbox_bottom_center 接地点近似")
        if (type(target.get("class_id")) is not int or target["class_id"] < 0
                or not isinstance(target.get("class_name"), str) or not target["class_name"]):
            raise ValueError("类别必须有非负整数 ID 和名称")
        score = target.get("confidence")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("置信度必须为 [0,1] 的有限数值")
        bbox, anchor = target.get("bbox_xyxy"), target.get("anchor_xy")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or not isinstance(anchor, (list, tuple)) or len(anchor) != 2
                or not all(type(v) in (int, float) and math.isfinite(v) for v in (*bbox, *anchor))):
            raise ValueError("框与接地点必须有正确的维度和有限像素坐标")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= frame["width"] and 0 <= y1 < y2 <= frame["height"]):
            raise ValueError("框超出图像范围或退化")
        if abs(anchor[0] - (x1+x2)/2) > 1e-4 or abs(anchor[1] - y2) > 1e-4:
            raise ValueError("声明的 bbox_bottom_center 与原始框底边中心不一致")
        track_id = target.get("track_id")
        if track_id is not None:
            if type(track_id) is not int or track_id <= 0 or track_id in ids:
                raise ValueError("当前帧 ID 必须是不重复的正整数或 null")
            ids.add(track_id)


def map_tracks(source, mapper, *, camera_id, output=None, allow_synthetic=False, position_source="raw"):
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise ValueError("输入必须是已有 tracks.jsonl 或 motion.jsonl 文件")
    if camera_id != mapper.config.camera_id:
        raise ValueError("camera_id 与标定机位不匹配")
    if mapper.synthetic and not allow_synthetic:
        raise ValueError("合成标定只允许显式 --allow-synthetic 的功能检查")
    if position_source not in {"raw", "smoothed"}:
        raise ValueError("position_source 必须为 raw 或 smoothed")
    output = new_output(output)
    write_json(output / "mapping.resolved.json", resolved_document(mapper))
    columns = ("session_id", "source", "frame_index", "timestamp_s", "timestamp_source", "track_id", "segment_id",
               "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "image_x_px", "image_y_px",
               "rectified_x_px", "rectified_y_px", "ground_x_m", "ground_y_m", "camera_id", "calibration_id",
               "position_source", "mapping_valid", "calibration_valid", "synthetic", "invalid_reason")
    summary = {"schema_version": 1, "input": str(source), "camera_id": camera_id,
               "calibration_id": mapper.calibration_id, "position_source": position_source,
               "frames_processed": 0, "observations": 0, "mapped_positions": 0, "trusted_positions": 0,
               "invalid_reasons": {}, "status": "completed", "stop_reason": "eof", "error": None}
    previous, failure = None, None
    started = time.perf_counter()
    try:
        with source.open(encoding="utf-8") as stream, \
                (output / "ground_tracks.jsonl").open("x", encoding="utf-8") as frames, \
                (output / "ground_tracks.csv").open("x", encoding="utf-8", newline="") as table:
            writer = csv.writer(table)
            writer.writerow(columns)
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line, parse_constant=reject_constant)
                    validate_frame(frame, previous)
                    if [frame["width"], frame["height"]] != mapper.config.image_size:
                        raise ValueError("视频分辨率与标定图像尺寸不符，不能直接沿用旧 H")
                    if frame.get("synthetic", False) and not allow_synthetic:
                        raise ValueError("合成跟踪输入需要显式 --allow-synthetic")
                    anchor_field = "anchor_xy" if position_source == "raw" else "smoothed_anchor_xy"
                    selected = [target[anchor_field] for target in frame["tracks"]]
                    points_array(selected)  # 严格检查整帧数据，再写出，避免一帧写到一半才发现坏字段。
                    mapped = mapper.map_points(selected, camera_id=camera_id, image_size=mapper.config.image_size)
                    targets = []
                    for target, point in zip(frame["tracks"], mapped):
                        value = {**target, **point.to_dict()}
                        value["synthetic"] |= frame.get("synthetic", False)
                        value["calibration_valid"] &= not value["synthetic"]
                        targets.append(value)
                    result = {**frame, "tracks": targets, "camera_id": camera_id,
                              "calibration_id": mapper.calibration_id, "position_source": position_source,
                              "coordinate_space": "image_and_ground", "image_unit": "px", "ground_unit": "m",
                              "distortion_status": mapper.lens.status,
                              "synthetic": mapper.synthetic or frame.get("synthetic", False)}
                    # 先生成可序列化帧和所有 CSV 行，再写入；无效位置填空，不写 NaN 或假零坐标。
                    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
                    rows = []
                    for target in targets:
                        bbox = target["bbox_xyxy"]
                        if len(bbox) != 4:
                            raise ValueError("bbox_xyxy 必须为四个坐标")
                        rows.append((frame["session_id"], frame["source"], frame["frame_index"], frame["timestamp_s"],
                                     frame["timestamp_source"], target["track_id"], target.get("segment_id"),
                                     target["class_id"], target["class_name"], target["confidence"], *bbox,
                                     *target["image_position_px"], *(target["rectified_position_px"] or (None, None)),
                                     *(target["ground_position_m"] or (None, None)), camera_id, mapper.calibration_id,
                                     position_source, target["mapping_valid"], target["calibration_valid"],
                                     target["synthetic"], target["invalid_reason"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"输入第 {line_number} 行无效：{exc}") from exc
                frames.write(encoded)
                writer.writerows(rows)
                summary["frames_processed"] += 1
                summary["observations"] += len(targets)
                for target in targets:
                    summary["mapped_positions"] += int(target["mapping_valid"])
                    summary["trusted_positions"] += int(target["mapping_valid"] and target["calibration_valid"])
                    reason = target["invalid_reason"]
                    if reason:
                        summary["invalid_reasons"][reason] = summary["invalid_reasons"].get(reason, 0) + 1
                previous = frame
        if not summary["frames_processed"]:
            raise ValueError("没有有效跟踪帧")
    except KeyboardInterrupt:
        summary.update(status="interrupted", stop_reason="interrupted")
    except Exception as exc:
        failure = exc
        summary.update(status="failed", stop_reason="error", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        summary["elapsed_s"] = time.perf_counter() - started
        write_json(output / "summary.json", summary)
    if failure is not None:
        failure.add_note(f"已完成部分与错误汇总：{output}")
        raise failure
    return output
