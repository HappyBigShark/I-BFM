"""Synchronized, body-mounted MuJoCo RGB-D observations (no policy changes).

Set MUJOCO_GL before importing MuJoCo, e.g. ``MUJOCO_GL=egl`` for headless use.
Optical coordinates are +X right, +Y down, +Z forward. Depth is optical Z in
metres, not Euclidean range. Segmentation is simulator ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np


# Head RealSense D435i on G1, from Unitree g1_description (rev_1_0) d435_joint:
# parent torso_link, xyz="0.0576235 0.01753 0.42987", rpy="0 0.8307767239493009 0".
# The frame is the depth origin (left IR imager); positive URDF pitch looks down.
G1_D435_POSITION = (0.0576235, 0.01753, 0.42987)
G1_D435_PITCH_DOWN_DEG = math.degrees(0.8307767239493009)
# D435 depth FOV is 87 x 58 deg (H x V); 848x480 is its recommended depth mode.
G1_D435_DEPTH_FOVY_DEG = 58.0


@dataclass(frozen=True)
class RgbdCameraConfig:
    name: str = "d435"
    mount_body: str = "torso_link"
    base_body: str = "pelvis"
    position: tuple[float, float, float] = G1_D435_POSITION
    pitch_down_deg: float = G1_D435_PITCH_DOWN_DEG
    fovy_deg: float = G1_D435_DEPTH_FOVY_DEG
    width: int = 848
    height: int = 480

    def __post_init__(self) -> None:
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in (self.width, self.height)):
            raise ValueError("Camera width and height must be positive integers")
        if not math.isfinite(self.fovy_deg) or not 0.0 < self.fovy_deg < 180.0:
            raise ValueError("Camera fovy_deg must be between 0 and 180 degrees")
        if not math.isfinite(self.pitch_down_deg):
            raise ValueError("Camera pitch must be finite")
        if np.asarray(self.position).shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError("Camera position must contain three finite values in metres")


def add_rgbd_camera(spec: mujoco.MjSpec, config: RgbdCameraConfig) -> None:
    """Attach a fixed camera before spec.compile(); mount body uses +X forward.

    The default mount is the CAD pose from Unitree's URDF, not a per-robot
    calibration; replace it with measured extrinsics when available.
    """
    mount = spec.body(config.mount_body)
    if mount is None:
        raise ValueError(f"Camera mount body not found: {config.mount_body!r}")
    angle = math.radians(config.pitch_down_deg)
    mount.add_camera(
        name=config.name,
        pos=config.position,
        xyaxes=(0.0, -1.0, 0.0, math.sin(angle), 0.0, math.cos(angle)),
        fovy=config.fovy_deg,
        mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
    )


class RgbdCamera:
    """Own one renderer; use as a context manager and capture after mj_forward.

    All three passes use one scene without advancing simulation. No image or
    box state is fed to the existing Actor/B automatically. Do not share this
    object across threads or step MjData concurrently with capture().
    """

    def __init__(self, model: mujoco.MjModel, config: RgbdCameraConfig, *, box_geom: str) -> None:
        self.model = model
        self.config = config
        self.camera_id = model.camera(config.name).id
        self.base_id = model.body(config.base_body).id
        self.box_id = model.geom(box_geom).id
        if model.cam_projection[self.camera_id] != mujoco.mjtProjection.mjPROJ_PERSPECTIVE or np.any(model.cam_sensorsize[self.camera_id]):
            raise ValueError("RgbdCamera requires the fovy-based perspective camera from add_rgbd_camera")
        # Renderer requires an offscreen framebuffer at least as large as the image.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, config.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, config.height)
        focal = config.height / (2.0 * math.tan(math.radians(model.cam_fovy[self.camera_id]) / 2.0))
        self.intrinsics = np.array([
            [focal, 0.0, (config.width - 1) / 2.0],
            [0.0, focal, (config.height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.renderer = mujoco.Renderer(model, height=config.height, width=config.width)

    def capture(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Return owned arrays; caller must update kinematics after changing qpos.

        ``T_world_camera`` maps optical-camera points to world coordinates;
        ``T_base_camera`` maps them to the configured base body's coordinates.
        ``depth_valid`` excludes background and clipping planes. Always apply
        it before unprojection. ``box_mask`` includes visible pixels only.
        """
        renderer = self.renderer
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera=self.camera_id)
        try:
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            # MuJoCo already converts the depth buffer to metric optical Z.
            depth = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
            renderer.disable_segmentation_rendering()
        near = float(self.model.vis.map.znear * self.model.stat.extent)
        far = float(self.model.vis.map.zfar * self.model.stat.extent)
        valid = np.isfinite(depth) & (depth > near) & (depth < far) & (segmentation[..., 0] >= 0)
        box_mask = (segmentation[..., 0] == self.box_id) & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        world_camera = np.eye(4)
        world_camera[:3, :3] = data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        world_camera[:3, 3] = data.cam_xpos[self.camera_id]
        base_rotation = data.xmat[self.base_id].reshape(3, 3)
        base_camera = np.eye(4)
        base_camera[:3, :3] = base_rotation.T @ world_camera[:3, :3]
        base_camera[:3, 3] = base_rotation.T @ (world_camera[:3, 3] - data.xpos[self.base_id])
        return {
            "rgb": rgb,
            "depth_m": depth,
            "depth_valid": valid,
            "box_mask": box_mask,
            "K": self.intrinsics.copy(),
            "T_world_camera": world_camera,
            "T_base_camera": base_camera,
            "timestamp_s": np.asarray(data.time, dtype=np.float64),
        }

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> RgbdCamera:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def depth_to_points(depth_m: np.ndarray, intrinsics: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Unproject selected pixels to [N, 3] optical-camera coordinates in metres.

    For visible box points, pass ``frame['depth_valid'] & frame['box_mask']``.
    """
    depth_m = np.asarray(depth_m)
    intrinsics = np.asarray(intrinsics)
    valid = np.asarray(valid, dtype=bool)
    if depth_m.ndim != 2 or valid.shape != depth_m.shape or intrinsics.shape != (3, 3):
        raise ValueError("Expected depth/valid [H, W] and intrinsics [3, 3]")
    if not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("Intrinsics must be finite with positive focal lengths")
    v, u = np.nonzero(valid & np.isfinite(depth_m) & (depth_m > 0.0))
    z = depth_m[v, u]
    return np.column_stack((
        (u - intrinsics[0, 2]) * z / intrinsics[0, 0],
        (v - intrinsics[1, 2]) * z / intrinsics[1, 1],
        z,
    )).astype(np.float32)
