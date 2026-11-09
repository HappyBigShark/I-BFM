"""Student vision observation: box mask + D435 depth, simulated and deployed alike.

Simulation renders depth and segmentation by CPU ray casting (``mj_multiRay``):
the student needs no RGB, and ray casting needs no OpenGL device access.  With
the ray direction's optical-axis component set to 1 the returned distance *is*
optical Z, the same quantity a D435 reports.

Two streams run at their own rate and latency (the mask from the Jetson
benchmark of YOLO-World + NanoSAM):

* D435i depth: per-episode mount-pose error, scale and offset, maximum range and
  whole-frame outages; per-frame latency jitter; a 3x3 Gaussian blur among valid
  pixels; z^2 noise, holes on depth edges and at random, millimetre
  quantisation, 0 = invalid;
* box mask (YOLO-World + NanoSAM): detection decided on the full-resolution
  bounding box (a far box spans only a cell or two of the training grid),
  unknown outside the colour FOV once aligned to depth, missed frames,
  boundary erosion/dilation; statistics use the depth frame of the same capture.

Depth Anything is off by default (too slow for the Jetson control loop).  With
``DISTILL_DEPTH_ANYTHING=1`` (read at import, so the learner and collectors of
one run agree) a third stream adds its mask-median distance: true depth in the
mask times a per-episode scale bias and per-frame noise, every 2nd mask frame,
300 ms latency.

With ``DISTILL_GEOMETRY=1`` the mask statistics also carry the box surface point
in the root frame: the detection-centre ray scaled by the D435i mask median,
mapped through the *nominal* camera mount and the torso pose (leg/waist
kinematics), plus that ray's direction.  The network then gets the box in the
frame its auxiliary head and the controller use, instead of learning the camera
geometry; a mount error still shows up in it, as it would on the robot.

``pool_full_resolution`` / ``detection_from_bbox`` / ``backproject_to_root`` turn real full-resolution
data into the same grid and statistics so the deployed student sees the
training contract.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from humanoidverse.envs.rgbd_camera import RgbdCameraConfig


# Geom groups visible to the camera: 0 = scene (floor, supports, box), 2 = robot
# visual meshes.  Group 3 holds collision primitives; group 5 hides markers and
# robot parts the head camera never sees.
VISIBLE_GEOM_GROUPS = np.asarray((1, 0, 1, 0, 0, 0), dtype=np.uint8)
HIDDEN_GEOM_GROUP = 5
# Visual meshes of bodies behind/around the torso-mounted camera: never hit,
# but testing them costs ~2/3 of the ray-casting time.
CAMERA_INVISIBLE_BODIES = ("torso_link", "pelvis", "waist_yaw_link", "waist_roll_link")
RAY_CUTOFF_M = 100.0  # mj_multiRay ignores hits beyond this distance
# G1 head RealSense D435i as calibrated in P-BFM (pbfm/perception/instinct_direct_depth.py):
# torso_link mount, optical axis 48.0 deg below the torso x-axis, depth FOV 89.51 x 58.29 deg.
G1_D435I_POSITION = (0.0487988662332928, 0.01, 0.4378029937970051)
G1_D435I_PITCH_DOWN_DEG = 48.0
G1_D435I_VERTICAL_FOV_DEG = 58.29
CAMERA_NAME = "d435i"
USE_DEPTH_ANYTHING = os.environ.get("DISTILL_DEPTH_ANYTHING", "0") == "1"
DEPTH_ANYTHING_STATS = ("da_median", "da_valid")
USE_GEOMETRY = os.environ.get("DISTILL_GEOMETRY", "0") == "1"
GEOMETRY_STATS = ("point_root_x", "point_root_y", "point_root_z", "point_valid", "ray_root_x", "ray_root_y", "ray_root_z")
STATS_NAMES_BASE = (
    "d435_median",
    "d435_valid",
    "mask_area",
    "mask_depth_valid_fraction",
    "d435_iqr",
    "centroid_u",
    "centroid_v",
    "bbox_width",
    "bbox_height",
    "detected",
)
STATS_NAMES = (
    STATS_NAMES_BASE[:2] + DEPTH_ANYTHING_STATS + STATS_NAMES_BASE[2:] if USE_DEPTH_ANYTHING else STATS_NAMES_BASE
) + (GEOMETRY_STATS if USE_GEOMETRY else ())
SENTINEL = -1.0


@dataclass(frozen=True)
class VisionConfig:
    height: int = 60
    width: int = 106
    full_height: int = 480  # D435 depth stream the grid is pooled from
    full_width: int = 848
    depth_period_steps: int = 2  # 25 Hz at the 50 Hz control rate
    depth_latency_steps: tuple[int, int] = (1, 3)  # per-frame jitter, inclusive
    mask_period_steps: int = 8  # 6.25 Hz; YOLO-World + NanoSAM run at 5.3-6.7 FPS
    mask_latency_steps: int = 9  # ~180 ms measured wall time per frame
    da_every_masks: int = 2  # Depth Anything (optional) on every 2nd mask frame (3.1 Hz)
    da_latency_steps: int = 15
    da_scale_bias_std: float = 0.08
    da_frame_noise_std: float = 0.03
    da_dropout_prob: float = 0.0  # Depth Anything result missing although the box is detected
    age_cap_s: float = 5.0
    depth_min_m: float = 0.2
    depth_max_m: float = 4.0  # task range: beyond it a valid return is "far"
    d435_max_range_m: tuple[float, float] = (2.5, 6.0)  # per-episode range
    d435_outage_episode_prob: float = 0.5
    d435_outage_rate_hz: float = 0.15
    d435_outage_duration_s: tuple[float, float] = (0.5, 4.0)
    d435_scale_range: tuple[float, float] = (0.95, 1.05)  # per-episode depth scale
    d435_bias_std_m: float = 0.03  # per-episode depth offset
    d435_blur_sigma_px: float = 1.0  # 3x3 Gaussian among valid pixels
    d435_noise_per_m2: float = 0.004  # sigma = k z^2 (1.6 cm at 2 m)
    d435_edge_jump_m: float = 0.05
    d435_edge_hole_prob: float = 0.3
    d435_random_hole_prob: float = 0.01
    color_fov_deg: tuple[float, float] = (69.0, 42.0)  # D435i RGB, coaxial approximation
    detect_min_px: float = 12.0  # shorter full-resolution bbox side YOLO-World needs
    mask_min_cells: int = 3
    mask_miss_prob: float = 0.05
    mask_boundary_jitter_prob: float = 0.15
    bbox_noise_px: float = 3.0

    # Per-episode mount error in the camera frame (position, and roll/pitch/yaw).
    mount_position_error_m: float = 0.01
    mount_angle_error_deg: tuple[float, float, float] = (2.0, 5.0, 2.0)

    def normalise_depth(self, depth_m: np.ndarray | float) -> np.ndarray:
        return (np.asarray(depth_m, dtype=np.float32) - self.depth_min_m) / (self.depth_max_m - self.depth_min_m)


def camera_config(vision: VisionConfig) -> RgbdCameraConfig:
    """The student's head camera: G1 D435i at the training grid resolution."""

    return RgbdCameraConfig(
        name=CAMERA_NAME,
        position=G1_D435I_POSITION,
        pitch_down_deg=G1_D435I_PITCH_DOWN_DEG,
        fovy_deg=G1_D435I_VERTICAL_FOV_DEG,
        width=vision.width,
        height=vision.height,
    )


def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, dtype=np.float64), float(angle_rad))
    return quat


def perturb_camera_mount(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str,
    vision: VisionConfig,
    rng: np.random.Generator,
    *,
    pitch_offset_deg: float | None = None,
) -> dict[str, list[float]]:
    """Apply a per-episode mount error to the camera (in its own frame) and refresh its pose.

    MuJoCo camera axes: x right, y up, -z optical axis; so pitch is about x,
    yaw about y and roll about z.  ``pitch_offset_deg`` fixes the pitch error
    (used to check visibility at the error bounds) and disables the rest.
    """

    camera_id = model.camera(camera).id
    if pitch_offset_deg is None:
        offset = rng.uniform(-vision.mount_position_error_m, vision.mount_position_error_m, 3)
        roll, pitch, yaw = (rng.uniform(-limit, limit) for limit in vision.mount_angle_error_deg)
    else:
        offset, roll, pitch, yaw = np.zeros(3), 0.0, float(pitch_offset_deg), 0.0
    nominal = np.asarray(model.cam_quat[camera_id], dtype=np.float64).copy()
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, nominal)
    model.cam_pos[camera_id] += rotation.reshape(3, 3) @ offset
    delta = np.asarray((1.0, 0.0, 0.0, 0.0))
    # Positive pitch error looks further down: rotate about -x.
    for axis, angle in (((0.0, 0.0, 1.0), roll), ((-1.0, 0.0, 0.0), pitch), ((0.0, 1.0, 0.0), yaw)):
        step = np.empty(4)
        mujoco.mju_mulQuat(step, delta, _quat_from_axis_angle(np.asarray(axis), math.radians(angle)))
        delta = step
    perturbed = np.empty(4)
    mujoco.mju_mulQuat(perturbed, nominal, delta)
    model.cam_quat[camera_id] = perturbed / np.linalg.norm(perturbed)
    mujoco.mj_camlight(model, data)
    return {"position_offset_m": offset.tolist(), "roll_pitch_yaw_deg": [roll, pitch, yaw]}


def _blur_valid(depth: np.ndarray, sigma_px: float) -> np.ndarray:
    """3x3 Gaussian blur averaging only finite pixels (no-return pixels stay no-return)."""

    if sigma_px <= 0.0:
        return depth
    offsets = (-1, 0, 1)
    weights = np.asarray([[math.exp(-(dr * dr + dc * dc) / (2.0 * sigma_px**2)) for dc in offsets] for dr in offsets])
    finite = np.isfinite(depth)
    values = np.pad(np.where(finite, depth, 0.0), 1)
    mask = np.pad(finite.astype(np.float64), 1)
    total = np.zeros(depth.shape)
    norm = np.zeros(depth.shape)
    height, width = depth.shape
    for i, dr in enumerate(offsets):
        for j, dc in enumerate(offsets):
            window = (slice(1 + dr, 1 + dr + height), slice(1 + dc, 1 + dc + width))
            total += weights[i, j] * values[window]
            norm += weights[i, j] * mask[window]
    return np.where(finite, total / np.maximum(norm, 1e-12), np.inf).astype(np.float32)


def hide_camera_invisible_geoms(model: mujoco.MjModel) -> None:
    for geom in range(model.ngeom):
        if model.geom_group[geom] == 2 and model.body(int(model.geom_bodyid[geom])).name in CAMERA_INVISIBLE_BODIES:
            model.geom_group[geom] = HIDDEN_GEOM_GROUP


class RaycastCamera:
    """Optical-Z depth and geom ids for a fixed MuJoCo camera at the feature grid."""

    def __init__(self, model: mujoco.MjModel, camera: str, *, height: int, width: int):
        self.model = model
        self.camera_id = model.camera(camera).id
        self.height, self.width = height, width
        focal = height / (2.0 * math.tan(math.radians(float(model.cam_fovy[self.camera_id])) / 2.0))
        self.intrinsics = np.asarray(
            [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]]
        )
        column, row = np.meshgrid(np.arange(width), np.arange(height))
        self.optical_x = ((column - self.intrinsics[0, 2]) / focal).astype(np.float64)
        self.optical_y = ((row - self.intrinsics[1, 2]) / focal).astype(np.float64)
        # MuJoCo cameras look along -Z with +Y up; optical frame is right/down/forward.
        self._rays_camera = np.stack(
            (self.optical_x, -self.optical_y, -np.ones_like(self.optical_x)), axis=-1
        ).reshape(-1, 3)
        count = height * width
        self._geom = np.empty(count, dtype=np.int32)
        self._distance = np.empty(count, dtype=np.float64)

    def pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Camera position and optical-frame rotation (columns right, down, forward) in world."""

        rotation = np.asarray(data.cam_xmat[self.camera_id]).reshape(3, 3) @ np.diag((1.0, -1.0, -1.0))
        return np.asarray(data.cam_xpos[self.camera_id], dtype=np.float64), rotation

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        rotation = np.asarray(data.cam_xmat[self.camera_id]).reshape(3, 3)
        vectors = np.ascontiguousarray((self._rays_camera @ rotation.T).reshape(-1))
        mujoco.mj_multiRay(
            self.model,
            data,
            np.asarray(data.cam_xpos[self.camera_id], dtype=np.float64),
            vectors,
            VISIBLE_GEOM_GROUPS,
            1,
            -1,
            self._geom,
            self._distance,
            None,
            self.height * self.width,
            RAY_CUTOFF_M,
        )
        depth = self._distance.reshape(self.height, self.width).astype(np.float32)
        geom = self._geom.reshape(self.height, self.width).copy()
        depth[geom < 0] = np.inf
        return depth, geom


def color_fov_mask(camera: RaycastCamera, color_fov_deg: tuple[float, float]) -> np.ndarray:
    """Depth-grid cells covered by the colour camera after alignment to depth."""

    half_x = math.tan(math.radians(color_fov_deg[0]) / 2.0)
    half_y = math.tan(math.radians(color_fov_deg[1]) / 2.0)
    return (np.abs(camera.optical_x) <= half_x) & (np.abs(camera.optical_y) <= half_y)


def _shift_or(mask: np.ndarray) -> np.ndarray:
    grown = mask.copy()
    grown[1:] |= mask[:-1]
    grown[:-1] |= mask[1:]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown


def _erode(mask: np.ndarray) -> np.ndarray:
    return ~_shift_or(~mask)


def detection_from_bbox(bbox_xyxy, config: VisionConfig) -> dict[str, float]:
    """Full-resolution detector box (pixels, clipped to the colour FOV) -> normalised fields."""

    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    width, height = config.full_width, config.full_height
    return {
        "centroid_u": (x1 + x2) / (width - 1) - 1.0,
        "centroid_v": (y1 + y2) / (height - 1) - 1.0,
        "bbox_width": (x2 - x1) / width,
        "bbox_height": (y2 - y1) / height,
    }


def backproject_to_root(
    stats: np.ndarray,
    depth_mm: np.ndarray,
    mask: np.ndarray,
    root_from_optical: tuple[np.ndarray, np.ndarray],
    config: VisionConfig,
    window: int = 2,
) -> np.ndarray:
    """Geometry statistics (GEOMETRY_STATS) for one frame.

    ``root_from_optical`` = (R, t) maps optical-frame points (x right, y down,
    z forward) into the root (pelvis) frame: nominal camera mount composed with
    the torso pose from joint kinematics.  The depth is the median of valid
    cells around the detection centre (within the mask when it reaches there),
    so it belongs to the surface the ray hits; the whole-mask median mixes the
    top and front faces.  The point needs such a depth; the ray only a detection.
    """

    index = {name: i for i, name in enumerate(STATS_NAMES)}
    geometry = np.zeros(len(GEOMETRY_STATS), dtype=np.float32)
    if stats[index["detected"]] <= 0.0:
        return geometry
    rotation, translation = root_from_optical
    focal = config.full_height / (2.0 * math.tan(math.radians(G1_D435I_VERTICAL_FOV_DEG) / 2.0))
    pixel_x = (stats[index["centroid_u"]] + 1.0) * (config.full_width - 1) / 2.0
    pixel_y = (stats[index["centroid_v"]] + 1.0) * (config.full_height - 1) / 2.0
    ray = np.asarray(((pixel_x - (config.full_width - 1) / 2.0) / focal, (pixel_y - (config.full_height - 1) / 2.0) / focal, 1.0))
    direction = rotation @ ray
    geometry[4:7] = direction / np.linalg.norm(direction)
    height, width = depth_mm.shape
    row = int(round(pixel_y * height / config.full_height - 0.5))
    column = int(round(pixel_x * width / config.full_width - 0.5))
    rows = slice(max(row - window, 0), min(row + window + 1, height))
    columns = slice(max(column - window, 0), min(column + window + 1, width))
    patch, patch_mask = depth_mm[rows, columns], mask[rows, columns].astype(bool)
    values = patch[patch_mask & (patch > 0)] if patch_mask.any() else patch[patch > 0]
    if values.size:
        depth = float(np.median(values)) / 1000.0
        geometry[0:3] = rotation @ (depth * ray) + translation
        geometry[3] = 1.0
    return geometry


def mask_statistics(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    detection: dict[str, float] | None,
    config: VisionConfig,
) -> np.ndarray:
    """Per-mask-frame scalars shared by simulation and deployment (see STATS_NAMES).

    ``depth_mm`` is the depth frame of the same capture as the mask.
    """

    stats = np.full(len(STATS_NAMES), SENTINEL, dtype=np.float32)
    index = {name: i for i, name in enumerate(STATS_NAMES)}
    stats[index["detected"]] = float(detection is not None)
    stats[index["d435_valid"]] = 0.0
    if USE_DEPTH_ANYTHING:
        stats[index["da_valid"]] = 0.0  # filled per tick from its own stream
    stats[index["mask_area"]] = float(mask.sum()) / mask.size
    if detection is None:
        stats[index["mask_depth_valid_fraction"]] = 0.0
        return stats
    for key, value in detection.items():
        stats[index[key]] = value
    if not mask.any():
        stats[index["mask_depth_valid_fraction"]] = 0.0
        return stats
    # Box surface depth: drop the mask rim when enough interior is left.
    interior = _erode(mask)
    region = interior if interior.sum() >= 9 else mask
    values = depth_mm[region & (depth_mm > 0)].astype(np.float32) / 1000.0
    stats[index["mask_depth_valid_fraction"]] = values.size / max(int(region.sum()), 1)
    if values.size >= config.mask_min_cells:
        stats[index["d435_valid"]] = 1.0
        stats[index["d435_median"]] = float(config.normalise_depth(np.median(values)))
        low, high = np.percentile(values, (25.0, 75.0))
        stats[index["d435_iqr"]] = float((high - low) / (config.depth_max_m - config.depth_min_m))
    return stats


class SimulatedPerception:
    """Per-episode emulation of the D435i depth and the box mask."""

    def __init__(self, camera: RaycastCamera, box_geom_id: int, config: VisionConfig, rng: np.random.Generator):
        self.camera = camera
        self.box_geom_id = int(box_geom_id)
        self.config = config
        self.rng = rng
        self.known = color_fov_mask(camera, config.color_fov_deg)
        if USE_DEPTH_ANYTHING:  # drawn only then, so runs without it keep their random streams
            self.da_scale = 1.0 + float(rng.normal(0.0, config.da_scale_bias_std))
        self.depth_scale = float(rng.uniform(*config.d435_scale_range))
        self.depth_bias_m = float(rng.normal(0.0, config.d435_bias_std_m))
        self.max_range_m = float(rng.uniform(*config.d435_max_range_m))
        self.nominal_mount = None  # (torso-frame camera pos, MuJoCo quat, root body id, torso body id) for geometry
        self.outages: list[tuple[float, float]] = []
        if rng.random() < config.d435_outage_episode_prob:
            time_s = 0.0
            while True:
                time_s += float(rng.exponential(1.0 / config.d435_outage_rate_hz))
                if time_s > 120.0:
                    break
                duration = float(rng.uniform(*config.d435_outage_duration_s))
                self.outages.append((time_s, time_s + duration))
                time_s += duration
        scale = config.full_height / camera.height
        self.full_focal = camera.intrinsics[0, 0] * scale
        self.full_centre = np.asarray(((config.full_width - 1) / 2.0, (config.full_height - 1) / 2.0))
        half_x = math.tan(math.radians(config.color_fov_deg[0]) / 2.0) * self.full_focal
        half_y = math.tan(math.radians(config.color_fov_deg[1]) / 2.0) * self.full_focal
        self.known_box_px = (
            self.full_centre[0] - half_x,
            self.full_centre[1] - half_y,
            self.full_centre[0] + half_x,
            self.full_centre[1] + half_y,
        )

    def _d435(self, depth: np.ndarray, time_s: float) -> np.ndarray:
        cfg, rng = self.config, self.rng
        if any(start <= time_s < end for start, end in self.outages):
            return np.zeros(depth.shape, dtype=np.uint16)
        # Calibration error (scale, offset), then the sensor's own edge smoothing.
        depth = _blur_valid(depth * self.depth_scale + self.depth_bias_m, cfg.d435_blur_sigma_px)
        finite = np.isfinite(depth)
        safe = np.where(finite, depth, 0.0)
        noisy = depth + rng.normal(0.0, 1.0, depth.shape).astype(np.float32) * cfg.d435_noise_per_m2 * safe**2
        valid = finite & (depth >= cfg.depth_min_m) & (depth <= self.max_range_m)
        padded = np.pad(np.where(finite, depth, 1e3), 1, mode="edge")
        jump = np.zeros(depth.shape, dtype=np.float32)
        for dr, dc in ((0, 1), (2, 1), (1, 0), (1, 2)):
            jump = np.maximum(jump, np.abs(padded[dr : dr + depth.shape[0], dc : dc + depth.shape[1]] - padded[1:-1, 1:-1]))
        holes = (jump > cfg.d435_edge_jump_m) & (rng.random(depth.shape) < cfg.d435_edge_hole_prob)
        holes |= rng.random(depth.shape) < cfg.d435_random_hole_prob
        valid &= ~holes
        return np.where(valid, np.clip(np.rint(noisy * 1000.0), 1, 65535), 0).astype(np.uint16)

    def _detect(self, data: mujoco.MjData, visible_cells: int) -> dict[str, float] | None:
        """Detector bbox from the projected box corners (None when not detected)."""

        cfg, model = self.config, self.camera.model
        position, rotation = self.camera.pose(data)
        geom_rot = np.asarray(data.geom_xmat[self.box_geom_id]).reshape(3, 3)
        centre = np.asarray(data.geom_xpos[self.box_geom_id], dtype=np.float64)
        half = np.asarray(model.geom_size[self.box_geom_id], dtype=np.float64)
        signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
        corners = (centre + (signs * half) @ geom_rot.T - position) @ rotation
        self.last_centre_z = float((centre - position) @ rotation[:, 2])
        if np.any(corners[:, 2] <= 0.05):
            return None
        pixels = corners[:, :2] / corners[:, 2:3] * self.full_focal + self.full_centre
        known = self.known_box_px
        x1, y1 = max(pixels[:, 0].min(), known[0]), max(pixels[:, 1].min(), known[1])
        x2, y2 = min(pixels[:, 0].max(), known[2]), min(pixels[:, 1].max(), known[3])
        if min(x2 - x1, y2 - y1) < cfg.detect_min_px:
            return None
        if visible_cells == 0:
            # Too small for the grid: one ray towards the centre decides occlusion.
            geom = np.zeros(1, dtype=np.int32)
            mujoco.mj_ray(model, data, position, centre - position, VISIBLE_GEOM_GROUPS, 1, -1, geom)
            if int(geom[0]) != self.box_geom_id:
                return None
        if self.rng.random() < cfg.mask_miss_prob:
            return None
        noise = self.rng.normal(0.0, cfg.bbox_noise_px, 4)
        bbox = (x1 + noise[0], y1 + noise[1], max(x2 + noise[2], x1 + noise[0] + 1.0), max(y2 + noise[3], y1 + noise[1] + 1.0))
        return detection_from_bbox(bbox, cfg)

    def capture(self, data: mujoco.MjData, time_s: float, *, mask: bool, depth_anything: bool = False) -> dict:
        cfg, rng = self.config, self.rng
        depth, geom = self.camera.capture(data)
        frame: dict = {"depth_mm": self._d435(depth, time_s)}
        if not mask:
            return frame
        visible = (geom == self.box_geom_id) & self.known
        detection = self._detect(data, int(visible.sum()))
        box_mask = np.zeros_like(visible)
        if detection is not None and visible.sum() >= cfg.mask_min_cells:
            box_mask = visible
            draw = rng.random()
            if draw < cfg.mask_boundary_jitter_prob:
                box_mask = _shift_or(box_mask) & self.known
            elif draw < 2.0 * cfg.mask_boundary_jitter_prob and _erode(box_mask).sum() >= cfg.mask_min_cells:
                box_mask = _erode(box_mask)
        frame["mask"] = box_mask.astype(np.uint8)
        frame["stats"] = mask_statistics(frame["depth_mm"], box_mask, detection, cfg)
        if USE_GEOMETRY:
            frame["stats"][-len(GEOMETRY_STATS) :] = backproject_to_root(
                frame["stats"], frame["depth_mm"], frame["mask"], self._root_from_optical(data), cfg
            )
        if depth_anything:
            frame["da_median_m"] = None
            if detection is not None:
                truth = depth[box_mask & np.isfinite(depth)]
                surface = float(np.median(truth)) if truth.size else self.last_centre_z
                frame["da_median_m"] = surface * self.da_scale * (1.0 + float(rng.normal(0.0, cfg.da_frame_noise_std)))
                if rng.random() < cfg.da_dropout_prob:
                    frame["da_median_m"] = None
        return frame

    def _root_from_optical(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Nominal mount + torso pose, expressed in the root frame (what the robot can compute)."""

        position, quat, root_id, torso_id = self.nominal_mount
        mount = np.empty(9)
        mujoco.mju_quat2Mat(mount, np.asarray(quat, dtype=np.float64))
        torso_from_optical = mount.reshape(3, 3) @ np.diag((1.0, -1.0, -1.0))
        root_rotation = np.asarray(data.xmat[root_id]).reshape(3, 3)
        torso_rotation = np.asarray(data.xmat[torso_id]).reshape(3, 3)
        torso_position = np.asarray(data.xpos[torso_id]) - np.asarray(data.xpos[root_id])
        rotation = root_rotation.T @ torso_rotation @ torso_from_optical
        translation = root_rotation.T @ (torso_rotation @ np.asarray(position) + torso_position)
        return rotation, translation


def encode_frame(depth_mm: np.ndarray, mask: np.ndarray, known: np.ndarray, config: VisionConfig):
    """Student image tensors: mask encoder [mask, known], depth encoder [depth, valid, far, mask].

    Valid depth inside the task range maps to [0, 1]; a valid return beyond it
    sets ``far`` (depth clamped to 1); no return is the -1 sentinel with
    ``valid`` = 0 — unknown, never "far".  Works on [..., H, W] batches.
    """

    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = depth_mm > 0
    far = valid & (depth_m > config.depth_max_m)
    normalised = np.where(valid, np.clip(config.normalise_depth(depth_m), 0.0, 1.0), SENTINEL)
    mask = mask.astype(np.float32)
    known = np.broadcast_to(known, mask.shape).astype(np.float32)
    mask_image = np.stack((mask, known), axis=-3)
    depth_image = np.stack((normalised, valid.astype(np.float32), far.astype(np.float32), mask), axis=-3)
    return mask_image.astype(np.float32), depth_image.astype(np.float32)


def pool_full_resolution(
    depth_raw: np.ndarray,
    depth_scale_m: float,
    mask: np.ndarray,
    config: VisionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Deployment: full-resolution D435 Z16 depth + aligned mask -> training grid.

    Each grid cell takes the median of its valid (non-zero) depth pixels — never
    a blend with invalid pixels — and is masked when at least half its pixels
    are.  Returns (depth_mm uint16 with 0 = invalid, mask uint8).
    """

    rows = np.linspace(0, depth_raw.shape[0], config.height + 1).astype(int)
    columns = np.linspace(0, depth_raw.shape[1], config.width + 1).astype(int)
    depth_m = depth_raw.astype(np.float32) * float(depth_scale_m)
    depth_mm = np.zeros((config.height, config.width), dtype=np.uint16)
    pooled_mask = np.zeros((config.height, config.width), dtype=np.uint8)
    for i in range(config.height):
        for j in range(config.width):
            block = depth_m[rows[i] : rows[i + 1], columns[j] : columns[j + 1]]
            values = block[block > 0]
            if values.size:
                depth_mm[i, j] = np.clip(np.rint(np.median(values) * 1000.0), 1, 65535)
            pooled_mask[i, j] = mask[rows[i] : rows[i + 1], columns[j] : columns[j + 1]].mean() >= 0.5
    return depth_mm, pooled_mask
