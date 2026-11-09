"""Run: MUJOCO_GL=egl .venv/bin/python -m unittest discover -s tests -p test_rgbd_camera.py"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import math
from pathlib import Path
import unittest

import mujoco
import numpy as np

from humanoidverse.envs.rgbd_camera import RgbdCamera, RgbdCameraConfig, add_rgbd_camera, depth_to_points

G1_XML = Path(__file__).resolve().parents[1] / "humanoidverse/data/robots/g1_mjlab/g1_29dof.xml"


def scene(*, occluded=False):
    spec = mujoco.MjSpec()
    mount = spec.worldbody.add_body(name="base")
    mount.add_freejoint()
    mount.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.01, 0, 0), pos=(-1, 0, 0))
    # Looking along world +X: front face is exactly two metres from the camera.
    spec.worldbody.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=(0.2, 0.4, 0.4), pos=(2.2, 0, 0), rgba=(1, 0, 0, 1))
    if occluded:
        spec.worldbody.add_geom(name="occluder", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=(0.1, 0.8, 0.8), pos=(1, 0, 0))
    config = RgbdCameraConfig(mount_body="base", base_body="base", position=(0, 0, 0),
                              pitch_down_deg=0, fovy_deg=60, width=80, height=60)
    add_rgbd_camera(spec, config)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, config


class RgbdTests(unittest.TestCase):
    def test_validation(self):
        for kw in ({"width": 0}, {"width": 3.5}, {"height": -1}, {"fovy_deg": 180},
                   {"pitch_down_deg": float("nan")}, {"position": (0, 0)}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                RgbdCameraConfig(**kw)
        with self.assertRaises(ValueError):
            add_rgbd_camera(mujoco.MjSpec(), RgbdCameraConfig(mount_body="missing"))

    def test_metric_depth_unprojection_and_sync(self):
        model, data, config = scene()
        data.time = 0.42
        qpos = data.qpos.copy()
        with RgbdCamera(model, config, box_geom="box") as sensor:
            frame = sensor.capture(data)
            # Returned frames must not alias a reused rendering buffer.
            saved_rgb = frame["rgb"].copy()
            sensor.capture(data)
            np.testing.assert_array_equal(frame["rgb"], saved_rgb)
        np.testing.assert_array_equal(data.qpos, qpos)
        self.assertEqual(float(frame["timestamp_s"]), 0.42)
        self.assertEqual(frame["rgb"].shape, (60, 80, 3))
        self.assertEqual(frame["rgb"].dtype, np.uint8)
        self.assertEqual(frame["depth_m"].dtype, np.float32)
        self.assertTrue(frame["box_mask"][30, 40])
        self.assertFalse(frame["depth_valid"][0, 0])
        mask = frame["box_mask"] & frame["depth_valid"]
        self.assertGreater(mask.sum(), 100)
        np.testing.assert_allclose(frame["depth_m"][mask], 2.0, atol=2e-5)
        points = depth_to_points(frame["depth_m"], frame["K"], mask)
        world = points @ frame["T_world_camera"][:3, :3].T + frame["T_world_camera"][:3, 3]
        np.testing.assert_allclose(world[:, 0], 2.0, atol=2e-5)
        self.assertLessEqual(np.abs(world[:, 1:]).max(), 0.42)
        # Check pixel/optical orientation, including vertical flip convention.
        selected = np.zeros_like(mask)
        selected[20, 30] = True
        point = depth_to_points(frame["depth_m"], frame["K"], selected)[0]
        self.assertLess(point[0], 0)
        self.assertLess(point[1], 0)

    def test_box_occluded(self):
        model, data, config = scene(occluded=True)
        with RgbdCamera(model, config, box_geom="box") as sensor:
            frame = sensor.capture(data)
        self.assertFalse(frame["box_mask"].any())
        self.assertTrue(frame["depth_valid"][30, 40])
        self.assertAlmostEqual(float(frame["depth_m"][30, 40]), 0.9, places=5)

    def test_off_axis_projection(self):
        model, data, config = scene()
        model.geom_pos[model.geom("box").id] = (2.2, 0.6, 0.35)
        mujoco.mj_forward(model, data)
        with RgbdCamera(model, config, box_geom="box") as sensor:
            frame = sensor.capture(data)
        # The front-face centre projects left/up from the optical principal point.
        # Exclude visible side faces by depth, then compare pixel-centre calibration.
        front = frame["box_mask"] & np.isclose(frame["depth_m"], 2.0, atol=2e-5)
        v, u = np.nonzero(front)
        self.assertGreater(len(u), 0)
        k = frame["K"]
        self.assertAlmostEqual(float(u.mean()), k[0, 2] - k[0, 0] * 0.6 / 2.0, delta=1.0)
        self.assertAlmostEqual(float(v.mean()), k[1, 2] - k[1, 1] * 0.35 / 2.0, delta=1.0)

    def test_camera_follows_base(self):
        model, data, config = scene()
        with RgbdCamera(model, config, box_geom="box") as sensor:
            before = sensor.capture(data)
            data.qpos[:3] = (0.3, -0.2, 0.5)
            data.qpos[3:7] = (np.sqrt(0.5), 0, 0, np.sqrt(0.5))
            mujoco.mj_forward(model, data)
            after = sensor.capture(data)
        np.testing.assert_allclose(before["T_base_camera"], after["T_base_camera"], atol=1e-12)
        np.testing.assert_allclose(after["T_world_camera"][:3, 3], data.qpos[:3], atol=1e-12)
        np.testing.assert_allclose(after["T_world_camera"][:3, 2], (0, 1, 0), atol=1e-12)

    def test_default_matches_g1_urdf_d435(self):
        spec = mujoco.MjSpec.from_file(str(G1_XML))
        add_rgbd_camera(spec, RgbdCameraConfig())
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        torso = model.body("torso_link").id
        torso_rot = data.xmat[torso].reshape(3, 3)
        camera = model.camera(RgbdCameraConfig().name).id
        local_pos = torso_rot.T @ (data.cam_xpos[camera] - data.xpos[torso])
        np.testing.assert_allclose(local_pos, (0.0576235, 0.01753, 0.42987), atol=1e-9)
        # URDF d435_joint rpy="0 0.8307767239493009 0": body +X rotated about +Y.
        pitch = 0.8307767239493009
        forward = torso_rot.T @ -data.cam_xmat[camera].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(forward, (math.cos(pitch), 0.0, -math.sin(pitch)), atol=1e-9)
        self.assertAlmostEqual(float(model.cam_fovy[camera]), 58.0)

    def test_empty_points_and_bad_shapes(self):
        points = depth_to_points(np.ones((2, 2)), np.eye(3), np.zeros((2, 2), dtype=bool))
        self.assertEqual(points.shape, (0, 3))
        with self.assertRaises(ValueError):
            depth_to_points(np.ones((2, 2)), np.eye(3), np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()
