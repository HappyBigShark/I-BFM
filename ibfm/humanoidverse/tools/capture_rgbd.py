"""Capture one G1 onboard RGB-D frame in a static test scene, without weights."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco
import numpy as np
from PIL import Image

from humanoidverse.carry_goal_sim2sim import BOX_GEOM_NAME, BOX_JOINT_NAME, _build_model
from humanoidverse.envs.rgbd_camera import RgbdCamera, RgbdCameraConfig, depth_to_points
from humanoidverse.utils.robot_spec import load_robot_training_spec


def main() -> None:
    # Defaults are the official G1 head D435 mount from RgbdCameraConfig.
    defaults = RgbdCameraConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/g1_29dof.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--fovy", type=float, default=defaults.fovy_deg)
    parser.add_argument("--pitch-down", type=float, default=defaults.pitch_down_deg)
    parser.add_argument("--mount-body", default=defaults.mount_body)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=defaults.position)
    parser.add_argument("--box-pos", type=float, nargs=3, default=(1.0, 0.0, 0.15))
    parser.add_argument("--box-size", type=float, nargs=3, default=(0.30, 0.30, 0.30))
    args = parser.parse_args()
    if not np.isfinite(args.box_size).all() or min(args.box_size) <= 0 or not np.isfinite(args.box_pos).all():
        parser.error("Box size must be positive and box size/position must be finite")
    robot = load_robot_training_spec(args.robot_config)
    config = RgbdCameraConfig(
        mount_body=args.mount_body,
        base_body=robot.robot.base_body,
        position=tuple(args.camera_pos),
        pitch_down_deg=args.pitch_down,
        fovy_deg=args.fovy,
        width=args.width,
        height=args.height,
    )
    model = _build_model(
        Path(robot.robot.xml_path),
        {"half_size": np.asarray(args.box_size) / 2.0, "mass": 1.25},
        render_size=max(config.width, config.height),
        rgbd_camera=config,
    )
    data = mujoco.MjData(model)
    root_address = model.joint(robot.robot.free_joint).qposadr[0]
    data.qpos[root_address:root_address + 3] = robot.init_state["pos"]
    # MuJoCo free joints store wxyz; respect the robot config convention.
    quat = np.asarray(robot.init_state["rot"])
    data.qpos[root_address + 3:root_address + 7] = quat[[3, 0, 1, 2]] if robot.robot.root_quat_order == "xyzw" else quat
    for name, angle in robot.default_joint_angles.items():
        data.qpos[model.joint(name).qposadr[0]] = angle
    box_address = model.joint(BOX_JOINT_NAME).qposadr[0]
    data.qpos[box_address:box_address + 3] = args.box_pos
    data.qpos[box_address + 3:box_address + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)
    with RgbdCamera(model, config, box_geom=BOX_GEOM_NAME) as camera:
        frame = camera.capture(data)
    points = depth_to_points(frame["depth_m"], frame["K"], frame["depth_valid"] & frame["box_mask"])
    output_dir = args.output_dir or Path("runs") / ("rgbd_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    # Never overwrite a previous capture.
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "frame.npz", **frame, box_points_camera=points)
    Image.fromarray(frame["rgb"]).save(output_dir / "rgb.png")
    Image.fromarray(frame["box_mask"].astype(np.uint8) * 255).save(output_dir / "box_mask.png")
    preview = np.where(frame["depth_valid"], np.clip(frame["depth_m"] / 5.0, 0, 1) * 255, 0)
    Image.fromarray(preview.astype(np.uint8)).save(output_dir / "depth_preview.png")
    metadata = {
        "camera": asdict(config),
        "scene": "Static G1 default pose, no control policy or physics rollout",
        "optical_axes": "+X right, +Y down, +Z forward",
        "depth": "frame.npz/depth_m is float32 optical Z in metres; apply depth_valid",
        "preview": "depth_preview.png is ONLY an 8-bit visualization, range 0 to 5 metres",
        "mask": "Visible box pixels from simulator ground truth, not a learned detector",
        "extrinsics": "T_world_camera / T_base_camera map optical points to world / base body",
        "visible_box_pixels": int(frame["box_mask"].sum()),
        "valid_box_points": len(points),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {config.width}x{config.height} RGB-D to {output_dir.resolve()}")
    print(f"Visible box pixels: {metadata['visible_box_pixels']}; valid box points: {len(points)}")
    if not len(points):
        print("WARNING: box is outside the camera view, occluded, or clipped; adjust camera/box pose.")


if __name__ == "__main__":
    main()
