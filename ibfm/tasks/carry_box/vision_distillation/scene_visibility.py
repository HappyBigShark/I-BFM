"""Which start heights keep the box in the head camera's view at the first tick?

With the G1 D435 pitched 47.6 deg down the box is only in view within about
1.5-2.5 m (less for high boxes), so "visible at the start" depends on the
scene (robot pose, box size) and on the random start height.  For a prepared
scene this returns the contiguous range of box-centre heights for which the
detector fires at the initial standing pose for the nominal mount and both
pitch-error bounds; collectors and the evaluation sample start heights inside it.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import joblib
import mujoco
import numpy as np

from humanoidverse.carry_goal_sim2sim import BOX_GEOM_NAME, BOX_JOINT_NAME, _build_model, _load_box_config
from humanoidverse.utils.robot_spec import load_robot_training_spec
from vision_sensor import (
    RaycastCamera,
    SimulatedPerception,
    VisionConfig,
    camera_config,
    hide_camera_invisible_geoms,
    perturb_camera_mount,
)


HEIGHT_RANGE_M = (0.15, 0.80)  # z-aware random start heights
HEIGHT_STEP_M = 0.025
MIN_WINDOW_M = 0.02


def visible_height_range(scene_dir: Path, *, edge_margin_m: float = 0.05) -> tuple[float, float] | None:
    scene = json.loads((scene_dir / "scene.json").read_text())
    size = np.asarray(scene["box_size_m"] if isinstance(scene["box_size_m"], list) else [scene["box_size_m"]] * 3)
    robot = load_robot_training_spec(scene_dir / "robot_config.yaml")
    box_cfg = dict(_load_box_config(scene_dir / "box_config.yaml"), half_size=tuple(size / 2.0))
    vision = replace(VisionConfig(), mask_miss_prob=0.0, bbox_noise_px=0.0)
    model = _build_model(
        Path(robot.robot.xml_path),
        box_cfg,
        render_size=64,
        rgbd_camera=camera_config(vision),
    )
    # As in collection: the camera sits inside the torso/head visual mesh, which it never sees.
    hide_camera_invisible_geoms(model)
    data = mujoco.MjData(model)
    motion = next(iter(joblib.load(scene_dir / "standing_front_box.pkl").values()))
    root = int(model.jnt_qposadr[model.joint(robot.robot.free_joint).id])
    data.qpos[root : root + 3] = (scene["robot_x"], scene["robot_y"], robot.init_state["pos"][2])
    heading = float(scene["robot_heading_rad"])
    data.qpos[root + 3 : root + 7] = (math.cos(heading / 2.0), 0.0, 0.0, math.sin(heading / 2.0))
    for name, angle in robot.default_joint_angles.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = angle
    box = int(model.jnt_qposadr[model.joint(BOX_JOINT_NAME).id])
    object_quat_xyzw = np.asarray(motion["object_quat"])[0]
    name = camera_config(vision).name
    camera_id = model.camera(name).id
    nominal_pos, nominal_quat = model.cam_pos[camera_id].copy(), model.cam_quat[camera_id].copy()
    box_geom = model.geom(BOX_GEOM_NAME).id
    lowest = max(HEIGHT_RANGE_M[0], float(size[2]) / 2.0)
    heights = np.arange(lowest, HEIGHT_RANGE_M[1] + 1e-9, HEIGHT_STEP_M)
    visible = [True] * len(heights)
    pitch_bound = vision.mount_angle_error_deg[1]
    for pitch_offset in (0.0, -pitch_bound, pitch_bound):
        model.cam_pos[camera_id], model.cam_quat[camera_id] = nominal_pos, nominal_quat
        mujoco.mj_forward(model, data)
        perturb_camera_mount(model, data, name, vision, np.random.default_rng(0), pitch_offset_deg=pitch_offset)
        camera = RaycastCamera(model, name, height=vision.height, width=vision.width)
        perception = SimulatedPerception(camera, box_geom, vision, np.random.default_rng(0))
        for index, height in enumerate(heights):
            data.qpos[box : box + 3] = (scene["box_x"], scene["box_y"], height)
            data.qpos[box + 3 : box + 7] = np.roll(object_quat_xyzw, 1)
            mujoco.mj_forward(model, data)
            _depth, geom = camera.capture(data)
            cells = int(((geom == box_geom) & perception.known).sum())
            detection = perception._detect(data, cells)
            visible[index] = visible[index] and detection is not None
    # Longest contiguous visible run, shrunk by a margin so the start is not on the FOV edge.
    best, start = None, None
    for index, flag in enumerate(visible + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if best is None:
        return None
    low, high = float(heights[best[0]]), float(heights[best[1] - 1])
    if best[0] > 0:
        low += edge_margin_m
    if best[1] < len(heights):
        high -= edge_margin_m
    # z-aware samples the start height strictly between min and max.
    return (low, high) if high - low >= MIN_WINDOW_M else None


def write_visibility(scene_dir: Path) -> dict:
    window = visible_height_range(scene_dir)
    payload = {"visible_height_range_m": window}
    (scene_dir / "visibility.json").write_text(json.dumps(payload) + "\n")
    return payload
