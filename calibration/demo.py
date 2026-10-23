"""已知投影关系生成的功能样例，没有校园测量、真实车速或 TTC 真值。"""

from dataclasses import asdict
import json

import cv2
import numpy as np

from .core import MappingConfig, RoadMapper
from .geometry import project_points
from .io import new_output, resolved_document, write_json
from .pipeline import map_tracks
from .visualization import render_grid


def synthetic_config():
    ground_corners = np.float32([[0, 0], [8, 0], [8, 24], [0, 24]])
    image_corners = np.float32([[400, 200], [880, 200], [1180, 680], [100, 680]])
    truth = cv2.getPerspectiveTransform(ground_corners, image_corners)
    ground = np.array([[x, y] for y in (0, 12, 24) for x in (0, 4, 8)], dtype=float)
    checks = np.array([[2, 4], [6, 4], [6, 20], [2, 20]], dtype=float)
    image, _ = project_points(truth, ground)
    check_image, _ = project_points(truth, checks)
    config = MappingConfig(
        camera_id="synthetic_fixed_camera", image_size=[1280, 720], source_type="synthetic",
        measurement_reference="Known synthetic projective matrix; no field measurements",
        image_points=image.tolist(), ground_points_m=ground.tolist(),
        check_image_points=check_image.tolist(), check_ground_points_m=checks.tolist())
    return config, truth


def run_demo(output=None):
    config, truth = synthetic_config()
    mapper = RoadMapper(config)
    output = new_output(output)
    write_json(output / "input_config.json", asdict(config))
    write_json(output / "mapping.json", resolved_document(mapper))
    write_json(output / "validation.json", mapper.report)
    if not cv2.imwrite(str(output / "ground_grid.png"), render_grid(mapper)):
        raise RuntimeError("无法保存合成地面网格预览")
    # 真值来自生成场景，不从估计结果抄回；位置对照用于数值验证，不是校园测速报告。
    references = []
    with (output / "tracks.synthetic.jsonl").open("x", encoding="utf-8") as stream:
        for index in range(12):
            ground = [[1 + 0.05 * index, 10 + 0.1 * index], [5 - 0.03 * index, 18 - 0.2 * index]]
            image, valid = project_points(truth, ground)
            assert valid.all()
            targets = []
            for j, (x, y) in enumerate(image):
                targets.append({"track_id": j + 1, "segment_id": j + 1, "class_id": 0 if j == 0 else 2,
                                "class_name": "person" if j == 0 else "car",
                                "category": "pedestrian" if j == 0 else "vehicle", "confidence": 1.0,
                                "bbox_xyxy": [x - 8, y - 30, x + 8, y], "anchor_xy": [x, y],
                                "anchor_type": "bbox_bottom_center", "observation": "detected"})
                references.append({"frame_index": index, "track_id": j + 1, "ground_position_m": ground[j]})
            frame = {"schema_version": 1, "session_id": "synthetic_mapping_demo", "source": "constructed_scene",
                     "frame_index": index, "timestamp_s": index / 10, "timestamp_source": "fps_estimate",
                     "width": 1280, "height": 720, "synthetic": True, "tracks": targets}
            stream.write(json.dumps(frame, allow_nan=False) + "\n")
    write_json(output / "synthetic_ground_reference.json", references)
    result = map_tracks(output / "tracks.synthetic.jsonl", mapper, camera_id=config.camera_id,
                        output=output / "mapped_tracks", allow_synthetic=True)
    mapped = [json.loads(line) for line in (result / "ground_tracks.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = []
    for reference, target in zip(references, (t for frame in mapped for t in frame["tracks"])):
        errors.append(np.linalg.norm(np.asarray(reference["ground_position_m"]) - target["ground_position_m"]))
        assert not target["calibration_valid"] and target["synthetic"]
    write_json(output / "demo_checks.json", {"frames": len(mapped), "observations": len(errors),
                                            "max_synthetic_position_error_m": float(max(errors)),
                                            "real_camera_validated": False})
    return output
