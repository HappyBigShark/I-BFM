"""Perception-error injection for the Carry Box step-0 robustness diagnostic.

The frozen controller stack reads box, contact and base state straight from
MjData.  This overlay builds one perceived view of that state per control tick
and routes it to the consumers selected in ``channels``:

* ``actor``      - box, contact and goal inputs of the actor observation
* ``b``          - backward-map privileged observation (body state + object context)
* ``controller`` - phase-controller metrics and latent-selection geometry

Consumers outside ``channels`` keep reading ground truth, so each module can be
perturbed alone or all of them together.  Errors are injected upstream (box
geometric centre/frame, base estimate, contact, goal) and every derived channel
is regenerated from the same perceived state.  Physics and every ground-truth
outcome written to the summary use the real MjData.
"""

from __future__ import annotations

import atexit
import itertools
import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch


CHANNELS = ("actor", "b", "controller")
SYMMETRY_GROUPS = ("none", "rect4", "oct8", "cube24")
VELOCITY_MODES = ("truth", "finite_difference")
LIFT_CLEARANCE_M = 0.05
PICKUP_HOLD_TICKS = 10
INITIAL_BOX_Z_ENV = "UFO_Z3D_INITIAL_BOX_Z"


@dataclass
class PerceptionConfig:
    channels: tuple[str, ...] = CHANNELS
    # Box detector + tracker.  Errors act on the box geometric centre and
    # geometric frame, then map back to the model (marker) frame.
    rate_hz: float = 50.0
    latency_s: float = 0.0
    pos_noise_std_m: float = 0.0
    pos_noise_per_m2: float = 0.0
    depth_scale_bias_std: float = 0.0
    # Time-correlated (Ornstein-Uhlenbeck) centre error, e.g. a slowly
    # changing mask truncation or depth bias: stationary std and time constant.
    pos_correlated_std_m: float = 0.0
    pos_correlated_tau_s: float = 0.5
    rot_noise_std_deg: float = 0.0
    dropout_prob: float = 0.0
    occlusion_radius_m: float = 0.0
    occlusion_dropout_prob: float = 0.0
    fov_deg: tuple[float, float] | None = None
    # G1 head D435 in torso_link (Unitree g1_description rev_1_0 d435_joint);
    # keep in sync with humanoidverse.envs.rgbd_camera.
    camera_offset_m: tuple[float, float, float] = (0.0576235, 0.01753, 0.42987)
    camera_pitch_deg: float = math.degrees(0.8307767239493009)
    velocity_mode: str = "truth"
    velocity_filter_alpha: float = 0.5
    extrapolate: bool = False
    symmetry_group: str = "none"
    symmetry_initial: str = "true"
    symmetry_jump_prob: float = 0.0
    reacquire_after_s: float = 0.5
    # Hand-box contact estimate.
    contact_delay_steps: int = 0
    contact_false_negative_prob: float = 0.0
    contact_false_positive_prob: float = 0.0
    # Base state estimate.  Perceived world-frame box quantities go through the
    # same base estimate, so these errors also shift box height and velocity.
    root_height_bias_std_m: float = 0.0
    root_height_noise_std_m: float = 0.0
    root_height_drift_std_m_per_sqrt_s: float = 0.0
    base_vel_bias_std_mps: float = 0.0
    base_vel_noise_std_mps: float = 0.0
    # Goal localisation (odometry) error.
    goal_bias_std_m: float = 0.0
    goal_drift_std_m_per_sqrt_s: float = 0.0
    measure_action_deviation: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PerceptionConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known - {"description"})
        if unknown:
            raise ValueError(f"unknown perception config keys: {unknown}")
        kwargs = {key: value for key, value in values.items() if key in known}
        for key in ("channels", "fov_deg", "camera_offset_m"):
            if kwargs.get(key) is not None:
                kwargs[key] = tuple(kwargs[key])
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        if not set(self.channels) <= set(CHANNELS):
            raise ValueError(f"channels must be a subset of {CHANNELS}, got {self.channels}")
        if self.pos_correlated_tau_s <= 0.0:
            raise ValueError("pos_correlated_tau_s must be positive")
        if self.rate_hz <= 0.0 or self.latency_s < 0.0:
            raise ValueError("rate_hz must be positive and latency_s non-negative")
        if self.symmetry_group not in SYMMETRY_GROUPS:
            raise ValueError(f"symmetry_group must be one of {SYMMETRY_GROUPS}")
        if self.symmetry_initial not in ("true", "random"):
            raise ValueError("symmetry_initial must be 'true' or 'random'")
        if self.velocity_mode not in VELOCITY_MODES:
            raise ValueError(f"velocity_mode must be one of {VELOCITY_MODES}")
        if not 0.0 < self.velocity_filter_alpha <= 1.0:
            raise ValueError("velocity_filter_alpha must be in (0, 1]")
        if self.fov_deg is not None and len(self.fov_deg) != 2:
            raise ValueError("fov_deg must be (horizontal, vertical)")
        if self.contact_delay_steps < 0:
            raise ValueError("contact_delay_steps must be non-negative")


def symmetry_group(name: str) -> list[np.ndarray]:
    """Proper rotations mapping an axis-aligned box onto itself (geom frame)."""

    if name == "none":
        return [np.eye(3)]
    elements = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(permutation, signs)):
                matrix[row, column] = sign
            if np.linalg.det(matrix) < 0.0:
                continue
            if name == "rect4" and permutation != (0, 1, 2):
                continue
            if name == "oct8" and abs(matrix[2, 2]) != 1.0:
                continue
            elements.append(matrix)
    return elements


def rotation_exp(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotvec / angle
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    quat = mat_to_quat(rotation)
    if quat[0] < 0.0:
        quat = -quat
    sin_half = float(np.linalg.norm(quat[1:]))
    if sin_half < 1e-12:
        return 2.0 * quat[1:]
    return 2.0 * math.atan2(sin_half, float(quat[0])) * quat[1:] / sin_half


def quat_angle_deg(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> float:
    conjugate = np.empty(4)
    mujoco.mju_negQuat(conjugate, np.asarray(second_wxyz, dtype=np.float64))
    relative = np.empty(4)
    mujoco.mju_mulQuat(relative, np.asarray(first_wxyz, dtype=np.float64), conjugate)
    return math.degrees(2.0 * math.atan2(float(np.linalg.norm(relative[1:])), abs(float(relative[0]))))


def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat / np.linalg.norm(quat))
    return matrix.reshape(3, 3)


def mat_to_quat(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quat


def _pitch_rotation(pitch_deg: float) -> np.ndarray:
    # Positive pitch tilts the camera x-axis (optical forward) downward.
    pitch = math.radians(pitch_deg)
    return np.asarray(
        [[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0], [-math.sin(pitch), 0.0, math.cos(pitch)]]
    )


@dataclass
class BoxGeometry:
    geom_pos: np.ndarray  # geom centre in the box body frame
    geom_rot: np.ndarray  # geom frame orientation in the box body frame


@dataclass
class TruthSample:
    box_pos: np.ndarray
    box_quat_wxyz: np.ndarray
    box_rot: np.ndarray
    box_lin_vel: np.ndarray  # world frame, at the box centre of mass
    box_ang_vel: np.ndarray  # world frame
    camera_pos: np.ndarray
    camera_rot: np.ndarray
    base_xy: np.ndarray
    contact: np.ndarray


@dataclass
class PerceivedState:
    box_pos: np.ndarray
    box_quat_wxyz: np.ndarray
    box_lin_vel: np.ndarray
    box_ang_vel: np.ndarray
    contact: np.ndarray
    root_dz: float
    base_dv: np.ndarray
    goal_offset: np.ndarray
    box_age_s: float


class PerceptionModel:
    """Per-episode generator of the perceived box, contact, base and goal state."""

    def __init__(self, config: PerceptionConfig, geometry: BoxGeometry, dt: float, rng: np.random.Generator):
        self.config = config
        self.geometry = geometry
        self.dt = float(dt)
        self.rng = rng
        self.period = max(1, round(1.0 / (config.rate_hz * self.dt)))
        self.latency_steps = max(0, round(config.latency_s / self.dt))
        self.history: deque[TruthSample] = deque(maxlen=self.latency_steps + 1)
        self.contact_history: deque[np.ndarray] = deque(maxlen=config.contact_delay_steps + 1)
        self.group = symmetry_group(config.symmetry_group)
        self.symmetry = np.eye(3)
        self.depth_scale = 1.0 + float(rng.normal(0.0, config.depth_scale_bias_std))
        self.correlated_error = rng.normal(0.0, config.pos_correlated_std_m, 3)
        self.correlated_time = 0.0
        self.root_height_bias = float(rng.normal(0.0, config.root_height_bias_std_m))
        self.root_height_drift = 0.0
        self.base_vel_bias = rng.normal(0.0, config.base_vel_bias_std_mps, 3)
        self.goal_offset = np.zeros(3)
        self.goal_offset[:2] = rng.normal(0.0, config.goal_bias_std_m, 2)
        self.step = -1
        self.estimate: dict[str, Any] | None = None
        self.stats = {"measurements": 0, "dropped": 0, "acquisitions": 0, "symmetry_jumps": 0}

    def tick(self, truth: TruthSample) -> PerceivedState:
        self.step += 1
        now = self.step * self.dt
        self.history.append(truth)
        if self.step == 0:
            # Initial detection: the episode starts with one measurement in hand.
            self._measure(truth, capture_time=0.0, force_visible=True)
        elif len(self.history) == self.history.maxlen:
            capture_step = self.step - self.latency_steps
            if capture_step % self.period == 0:
                self._measure(self.history[0], capture_time=capture_step * self.dt)
        box_pos, box_rot, box_lin_vel, box_ang_vel, capture_time = self._box_output(now)

        cfg = self.config
        self.root_height_drift += float(
            self.rng.normal(0.0, cfg.root_height_drift_std_m_per_sqrt_s * math.sqrt(self.dt))
        )
        root_dz = (
            self.root_height_bias
            + self.root_height_drift
            + float(self.rng.normal(0.0, cfg.root_height_noise_std_m))
        )
        base_dv = self.base_vel_bias + self.rng.normal(0.0, cfg.base_vel_noise_std_mps, 3)
        self.goal_offset[:2] += self.rng.normal(0.0, cfg.goal_drift_std_m_per_sqrt_s * math.sqrt(self.dt), 2)

        return PerceivedState(
            box_pos=box_pos + np.asarray((0.0, 0.0, root_dz)),
            box_quat_wxyz=mat_to_quat(box_rot),
            box_lin_vel=box_lin_vel + base_dv,
            box_ang_vel=box_ang_vel,
            contact=self._contact(truth.contact),
            root_dz=root_dz,
            base_dv=base_dv,
            goal_offset=self.goal_offset.copy(),
            box_age_s=now - capture_time,
        )

    def _visible(self, sample: TruthSample) -> bool:
        cfg = self.config
        centre = sample.box_pos + sample.box_rot @ self.geometry.geom_pos
        if cfg.fov_deg is not None:
            local = sample.camera_rot.T @ (centre - sample.camera_pos)
            if local[0] <= 0.0:
                return False
            if abs(math.degrees(math.atan2(local[1], local[0]))) > 0.5 * cfg.fov_deg[0]:
                return False
            if abs(math.degrees(math.atan2(local[2], local[0]))) > 0.5 * cfg.fov_deg[1]:
                return False
        drop = cfg.dropout_prob
        if cfg.occlusion_radius_m > 0.0 and np.linalg.norm(centre[:2] - sample.base_xy) < cfg.occlusion_radius_m:
            drop = max(drop, cfg.occlusion_dropout_prob)
        return not self.rng.random() < drop

    def _measure(self, sample: TruthSample, *, capture_time: float, force_visible: bool = False) -> None:
        cfg = self.config
        if not force_visible and not self._visible(sample):
            self.stats["dropped"] += 1
            return
        self.stats["measurements"] += 1
        previous = self.estimate
        if previous is None or capture_time - previous["capture_time"] > cfg.reacquire_after_s:
            # (Re)detection picks a symmetry-equivalent geometric frame afresh.
            self.stats["acquisitions"] += 1
            if cfg.symmetry_initial == "random":
                self.symmetry = self.group[int(self.rng.integers(len(self.group)))]
        elif len(self.group) > 1 and self.rng.random() < cfg.symmetry_jump_prob:
            self.stats["symmetry_jumps"] += 1
            others = [item for item in self.group if not np.allclose(item, self.symmetry)]
            self.symmetry = others[int(self.rng.integers(len(others)))]

        geometry = self.geometry
        centre = sample.box_pos + sample.box_rot @ geometry.geom_pos
        geom_rot = sample.box_rot @ geometry.geom_rot
        ray = centre - sample.camera_pos
        noise_std = cfg.pos_noise_std_m + cfg.pos_noise_per_m2 * float(ray @ ray)
        decay = math.exp(-max(capture_time - self.correlated_time, 0.0) / cfg.pos_correlated_tau_s)
        self.correlated_error = decay * self.correlated_error + math.sqrt(1.0 - decay * decay) * self.rng.normal(
            0.0, cfg.pos_correlated_std_m, 3
        )
        self.correlated_time = capture_time
        centre = (
            sample.camera_pos
            + self.depth_scale * ray
            + self.correlated_error
            + self.rng.normal(0.0, noise_std, 3)
        )
        noise_rot = rotation_exp(self.rng.normal(0.0, math.radians(cfg.rot_noise_std_deg), 3))
        geom_rot = noise_rot @ geom_rot @ self.symmetry
        box_rot = geom_rot @ geometry.geom_rot.T
        box_pos = centre - box_rot @ geometry.geom_pos

        if cfg.velocity_mode == "truth":
            lin_vel, ang_vel = sample.box_lin_vel.copy(), sample.box_ang_vel.copy()
        elif previous is None:
            lin_vel, ang_vel = np.zeros(3), np.zeros(3)
        else:
            elapsed = max(capture_time - previous["capture_time"], self.dt)
            alpha = cfg.velocity_filter_alpha
            lin_vel = (1.0 - alpha) * previous["lin_vel"] + alpha * (box_pos - previous["pos"]) / elapsed
            ang_vel = (1.0 - alpha) * previous["ang_vel"] + alpha * rotation_log(
                box_rot @ previous["rot"].T
            ) / elapsed
        self.estimate = {
            "pos": box_pos,
            "rot": box_rot,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
            "capture_time": capture_time,
        }

    def _box_output(self, now: float):
        estimate = self.estimate
        assert estimate is not None
        pos = estimate["pos"]
        if self.config.extrapolate:
            pos = pos + estimate["lin_vel"] * (now - estimate["capture_time"])
        return pos, estimate["rot"], estimate["lin_vel"], estimate["ang_vel"], estimate["capture_time"]

    def _contact(self, truth_contact: np.ndarray) -> np.ndarray:
        cfg = self.config
        self.contact_history.append(np.asarray(truth_contact, dtype=np.float32).copy())
        contact = self.contact_history[0].copy()
        draws = self.rng.random(2)
        held = contact > 0.5
        contact[held & (draws < cfg.contact_false_negative_prob)] = 0.0
        contact[~held & (draws < cfg.contact_false_positive_prob)] = 1.0
        return contact


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


class PerceptionInjector:
    """Routes one perceived world view per control tick to the selected consumers."""

    def __init__(self, module: Any, config: PerceptionConfig, *, seed: int, summary_path: Path | None):
        self.module = module
        self.config = config
        self.seed = int(seed)
        self.summary_path = summary_path
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.view_data: mujoco.MjData | None = None
        self.observer: Any = None
        self.shadow_observer: Any = None
        self.controller: Any = None
        self.episode = -1
        self.episodes: list[dict[str, Any]] = []
        self.perception: PerceptionModel | None = None
        self.tick_time: float | None = None
        self.view_time: float | None = None
        self.state: PerceivedState | None = None
        self.truth: TruthSample | None = None
        self.pending_b_observation: dict[str, torch.Tensor] | None = None
        self._clear_tick_observations()

    # ------------------------------------------------------------------ setup
    def perceives(self, channel: str) -> bool:
        return channel in self.config.channels

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.model is model and self.data is data:
            return
        module = self.module
        self.model, self.data = model, data
        self.view_data = mujoco.MjData(model)
        self.box_body_id = model.body(module.BOX_BODY_NAME).id
        self.box_geom_id = model.geom(module.BOX_GEOM_NAME).id
        box_joint = model.joint(module.BOX_JOINT_NAME).id
        self.box_qpos = int(model.jnt_qposadr[box_joint])
        self.box_qvel = int(model.jnt_dofadr[box_joint])
        root_joints = [
            index
            for index in range(model.njnt)
            if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE and index != box_joint
        ]
        if not root_joints:
            raise RuntimeError("perception overlay could not find the robot root free joint")
        self.root_qpos = int(model.jnt_qposadr[root_joints[0]])
        self.root_qvel = int(model.jnt_dofadr[root_joints[0]])
        self.root_body_id = int(model.jnt_bodyid[root_joints[0]])
        self.torso_body_id = model.body("torso_link").id
        self.robot_bodies = np.flatnonzero(model.body_rootid == model.body_rootid[self.root_body_id])
        self.robot_geoms = np.flatnonzero(np.isin(model.geom_bodyid, self.robot_bodies))
        if int(model.geom_bodyid[self.box_geom_id]) != self.box_body_id:
            raise RuntimeError("box geom does not belong to the box body")
        self.geometry = BoxGeometry(
            geom_pos=np.asarray(model.geom_pos[self.box_geom_id], dtype=np.float64).copy(),
            geom_rot=quat_to_mat(model.geom_quat[self.box_geom_id]),
        )
        self.camera_offset = np.asarray(self.config.camera_offset_m, dtype=np.float64)
        self.camera_pitch = _pitch_rotation(self.config.camera_pitch_deg)

    def attach_observer(self, observer: Any, kwargs: dict[str, Any]) -> None:
        self.observer = observer
        self.shadow_observer = self.module._PerceptionTruthObservation(**kwargs)
        self.context_start = 2 * int(observer.num_dof) + 6

    def reset_episode(self) -> None:
        self.finish_episode()
        self.episode += 1
        rng = np.random.default_rng([self.seed, self.episode])
        self.perception = PerceptionModel(self.config, self.geometry, self.module.CONTROL_DT, rng)
        self.tick_time = None
        self.view_time = None
        self.record = {
            "episode": self.episode,
            "initial_box_pos": None,
            "true_bilateral_ticks": 0,
            "true_hold_run": 0,
            "true_pickup": False,
            "max_true_lift_m": 0.0,
            "phase_transitions": [],
            "errors": {
                "box_pos_m": [],
                "box_rot_deg": [],
                "box_age_s": [],
                "root_dz_m": [],
                "goal_offset_m": [],
                "action_l2": [],
                "action_max_abs": [],
                "actor_state_max_abs": [],
                "b_state_max_abs": [],
            },
            "contact_mismatch_ticks": 0,
            "metric_errors": {
                "horizontal_distance": [],
                "goal_distance": [],
                "object_height": [],
                "object_speed": [],
                "root_height": [],
                "lift": [],
            },
            "bilateral_mismatch_ticks": 0,
            "ticks": 0,
        }
        self._clear_tick_observations()

    def _clear_tick_observations(self) -> None:
        # Per tick: what the overlay built for each consumer and the truth counterparts.
        self.actor_state_built: torch.Tensor | None = None
        self.truth_context: torch.Tensor | None = None
        self.b_context: torch.Tensor | None = None
        self.privileged_built: torch.Tensor | None = None
        self.privileged_truth: torch.Tensor | None = None

    # ------------------------------------------------------------------ ticking
    def ensure_tick(self, true_contact: np.ndarray | None = None) -> None:
        data = self.data
        if self.tick_time == float(data.time):
            return
        if true_contact is None:
            true_contact = self.truth.contact if self.truth is not None else np.zeros(2, dtype=np.float32)
        self.truth = self._truth_sample(np.asarray(true_contact, dtype=np.float32))
        self.state = self._snap_to_truth(self.perception.tick(self.truth), self.truth)
        self.tick_time = float(data.time)
        self._clear_tick_observations()
        self._record_tick()

    def _truth_sample(self, contact: np.ndarray) -> TruthSample:
        model, data = self.model, self.data
        lin_vel, ang_vel = self.module._world_body_velocity(model, data, self.box_body_id)
        torso_rot = np.asarray(data.xmat[self.torso_body_id]).reshape(3, 3)
        box_quat = np.asarray(data.qpos[self.box_qpos + 3 : self.box_qpos + 7], dtype=np.float64).copy()
        return TruthSample(
            box_pos=np.asarray(data.qpos[self.box_qpos : self.box_qpos + 3], dtype=np.float64).copy(),
            box_quat_wxyz=box_quat,
            box_rot=quat_to_mat(box_quat),
            box_lin_vel=np.asarray(lin_vel, dtype=np.float64),
            box_ang_vel=np.asarray(ang_vel, dtype=np.float64),
            camera_pos=np.asarray(data.xpos[self.torso_body_id]) + torso_rot @ self.camera_offset,
            camera_rot=torso_rot @ self.camera_pitch,
            base_xy=np.asarray(data.xpos[self.root_body_id, :2], dtype=np.float64).copy(),
            contact=contact.copy(),
        )

    @staticmethod
    def _snap_to_truth(state: PerceivedState, truth: TruthSample) -> PerceivedState:
        """Return exact truth bits where the injected error is below round-off.

        The frame round trip (marker -> geom -> marker) costs ~1e-16; snapping
        keeps zero-error channels bit-identical to the unmodified stack.
        """

        if np.linalg.norm(state.box_pos - truth.box_pos) < 1e-9:
            state.box_pos = truth.box_pos.copy()
        if quat_angle_deg(state.box_quat_wxyz, truth.box_quat_wxyz) < 1e-9:
            state.box_quat_wxyz = truth.box_quat_wxyz.copy()
        if np.linalg.norm(state.box_lin_vel - truth.box_lin_vel) < 1e-9:
            state.box_lin_vel = truth.box_lin_vel.copy()
        if np.linalg.norm(state.box_ang_vel - truth.box_ang_vel) < 1e-9:
            state.box_ang_vel = truth.box_ang_vel.copy()
        return state

    def _record_tick(self) -> None:
        record, truth, state = self.record, self.truth, self.state
        record["ticks"] += 1
        if record["initial_box_pos"] is None:
            record["initial_box_pos"] = truth.box_pos.tolist()
        lift = float(truth.box_pos[2] - record["initial_box_pos"][2])
        record["max_true_lift_m"] = max(record["max_true_lift_m"], lift)
        bilateral = bool(truth.contact[0] > 0.5 and truth.contact[1] > 0.5)
        record["true_bilateral_ticks"] += int(bilateral)
        record["true_hold_run"] = record["true_hold_run"] + 1 if bilateral and lift >= LIFT_CLEARANCE_M else 0
        if record["true_hold_run"] >= PICKUP_HOLD_TICKS:
            record["true_pickup"] = True
        errors = record["errors"]
        errors["box_pos_m"].append(float(np.linalg.norm(state.box_pos - truth.box_pos)))
        errors["box_rot_deg"].append(quat_angle_deg(state.box_quat_wxyz, truth.box_quat_wxyz))
        errors["box_age_s"].append(float(state.box_age_s))
        errors["root_dz_m"].append(abs(float(state.root_dz)))
        errors["goal_offset_m"].append(float(np.linalg.norm(state.goal_offset)))
        record["contact_mismatch_ticks"] += int(np.any((state.contact > 0.5) != (truth.contact > 0.5)))

    def perceived_data(self) -> mujoco.MjData:
        """Real MjData with the box and base estimate replaced by the perceived ones.

        After mj_step the stored kinematics (xpos, cvel, ...) lag qpos by one
        substep and the unmodified stack consumes them as-is.  So the view is
        never recomputed with mj_kinematics: it copies the real data and
        applies only the perceived-minus-truth deltas.
        """

        if self.view_time == self.tick_time:
            return self.view_data
        model, view, data, state, truth = self.model, self.view_data, self.data, self.state, self.truth
        mujoco.mj_copyData(view, model, data)
        box, geom = self.box_body_id, self.box_geom_id
        if not np.array_equal(state.box_pos, truth.box_pos):
            shift = state.box_pos - truth.box_pos
            view.qpos[self.box_qpos : self.box_qpos + 3] = state.box_pos
            for array, index in ((view.xpos, box), (view.xipos, box), (view.subtree_com, box), (view.geom_xpos, geom)):
                array[index] += shift
        if not np.array_equal(state.box_quat_wxyz, truth.box_quat_wxyz):
            delta = quat_to_mat(state.box_quat_wxyz) @ truth.box_rot.T
            view.qpos[self.box_qpos + 3 : self.box_qpos + 7] = state.box_quat_wxyz
            view.xmat[box] = (delta @ np.asarray(data.xmat[box]).reshape(3, 3)).reshape(9)
            view.xquat[box] = mat_to_quat(np.asarray(view.xmat[box]).reshape(3, 3))
            view.geom_xmat[geom] = (delta @ np.asarray(data.geom_xmat[geom]).reshape(3, 3)).reshape(9)
        if not (
            np.array_equal(state.box_lin_vel, truth.box_lin_vel)
            and np.array_equal(state.box_ang_vel, truth.box_ang_vel)
        ):
            # The box is its own kinematic tree, so cvel is referenced at its
            # centre of mass: [angular, linear] in the world frame.
            view.cvel[box, :3] = state.box_ang_vel
            view.cvel[box, 3:] = state.box_lin_vel
        if state.root_dz != 0.0:
            view.qpos[self.root_qpos + 2] += state.root_dz
            for array in (view.xpos, view.xipos, view.subtree_com):
                array[self.robot_bodies, 2] += state.root_dz
            view.geom_xpos[self.robot_geoms, 2] += state.root_dz
        if np.any(state.base_dv != 0.0):
            # A uniform translational velocity error adds to every robot body.
            view.qvel[self.root_qvel : self.root_qvel + 3] += state.base_dv
            view.cvel[self.robot_bodies, 3:] += state.base_dv
        self.view_time = self.tick_time
        return view

    # ------------------------------------------------------------------ consumers
    def observation_kwargs(self, kwargs: dict[str, Any], *, perceived: bool) -> dict[str, Any]:
        if not perceived:
            return kwargs
        state = self.state
        shift = np.asarray((0.0, 0.0, state.root_dz))
        return dict(
            kwargs,
            object_pos=state.box_pos,
            object_quat_wxyz=state.box_quat_wxyz,
            object_lin_vel=state.box_lin_vel,
            object_ang_vel=state.box_ang_vel,
            base_pos=np.asarray(kwargs["base_pos"]) + shift,
            hand_body_pos=np.asarray(kwargs["hand_body_pos"]) + shift,
            hand_contact=state.contact,
        )

    def goal_offset_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.state.goal_offset, device=reference.device, dtype=reference.dtype).reshape(1, 3)

    def object_context(self, kwargs: dict[str, Any], *, perceived: bool) -> torch.Tensor:
        shadow = self.shadow_observer
        shadow.target_object_pos.copy_(self.observer.target_object_pos)
        if perceived:
            shadow.target_object_pos.add_(self.goal_offset_tensor(shadow.target_object_pos))
        state = shadow.build(**self.observation_kwargs(kwargs, perceived=perceived))["state"]
        return state[:, self.context_start :]

    def build_actor_observation(self, observer: Any, build: Any, kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        self.ensure_tick(true_contact=kwargs["hand_contact"])
        self.last_observation_kwargs = kwargs
        if not self.perceives("actor"):
            observation = build(**kwargs)
            self.actor_state_built = observation["state"]
            self.truth_context = observation["state"][:, self.context_start :]
            return observation
        truth_target = observer.target_object_pos.clone()
        observer.target_object_pos.add_(self.goal_offset_tensor(truth_target))
        try:
            observation = build(**self.observation_kwargs(kwargs, perceived=True))
        finally:
            observer.target_object_pos.copy_(truth_target)
        self.actor_state_built = observation["state"]
        self.truth_context = self.object_context(kwargs, perceived=False)
        self.record["errors"]["actor_state_max_abs"].append(
            float((self.truth_context - observation["state"][:, self.context_start :]).abs().max())
        )
        return observation

    def build_privileged_state(self, original: Any, model: Any, data: Any, kwargs: dict[str, Any]) -> torch.Tensor:
        self.ensure_tick()
        truth_context = self.truth_context
        if not self.perceives("actor") and not torch.allclose(
            self.object_context(self.last_observation_kwargs, perceived=False), kwargs["object_context"], atol=1e-5
        ):
            raise RuntimeError("perception overlay: shadow object context diverged from the actor observation")
        truth = original(model, data, **dict(kwargs, object_context=truth_context))
        self.privileged_truth = truth
        if not self.perceives("b"):
            self.b_context, self.privileged_built = truth_context, truth
            return truth
        self.b_context = self.object_context(self.last_observation_kwargs, perceived=True)
        perceived = original(model, self.perceived_data(), **dict(kwargs, object_context=self.b_context))
        self.privileged_built = perceived
        self.record["errors"]["b_state_max_abs"].append(float((truth - perceived).abs().max()))
        return perceived

    def controller_metrics(self, original: Any, model: Any, data: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.ensure_tick(true_contact=kwargs["contact"])
        truth = original(model, data, **kwargs)
        if not self.perceives("controller"):
            return truth
        perceived = original(
            model,
            self.perceived_data(),
            **dict(kwargs, contact=self.state.contact, goal=np.asarray(kwargs["goal"]) + self.state.goal_offset),
        )
        record = self.record
        for key, values in record["metric_errors"].items():
            values.append(abs(float(perceived[key]) - float(truth[key])))
        record["bilateral_mismatch_ticks"] += int(bool(perceived["bilateral_contact"]) != bool(truth["bilateral_contact"]))
        return perceived

    def latent_directions(self, box_direction: np.ndarray, goal_direction: np.ndarray):
        if not self.perceives("controller"):
            return box_direction, goal_direction
        self.ensure_tick()
        truth_box = self.truth.box_pos[:2]
        perceived_box = self.state.box_pos[:2]
        goal_xy = np.asarray(goal_direction) + truth_box
        return (
            np.asarray(box_direction) + perceived_box - truth_box,
            goal_xy + self.state.goal_offset[:2] - perceived_box,
        )

    def _rebase(self, observed: torch.Tensor, built: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Substitute ``target`` for ``built`` in what the controller fed the policy.

        The controller only rewrites the trailing object context: unchanged
        contexts are swapped exactly, masked (all-zero) contexts stay masked,
        and shifted ones (goal look-ahead) receive the target-minus-built delta.
        """

        width = int(self.truth_context.shape[-1])  # both state and privileged_state end with it
        result = target.clone()
        observed_context = observed[:, -width:]
        built_context = built[:, -width:]
        if torch.equal(observed_context, built_context):
            return result
        if not torch.any(observed_context):
            result[:, -width:] = 0.0
            return result
        result[:, -width:] = observed_context + (target[:, -width:] - built_context)
        return result

    def _with_context(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, self.context_start :] = context
        return result

    def act(self, original_act: Any, observation: dict[str, torch.Tensor], latent: torch.Tensor, mean: bool):
        built = self.actor_state_built
        if built is None or not (self.perceives("actor") or self.perceives("b")):
            return original_act(observation, latent, mean=mean)
        # B reads the same "state" key as the actor.  When only one of the two
        # is perturbed, give B its own state carrying the B-channel context.
        b_context = self.b_context if self.b_context is not None else self.truth_context
        if self.perceives("actor") != self.perceives("b"):
            self.pending_b_observation = dict(
                observation, state=self._rebase(observation["state"], built, self._with_context(built, b_context))
            )
        try:
            action = original_act(observation, latent, mean=mean)
        finally:
            self.pending_b_observation = None
        if not self.config.measure_action_deviation:
            return action
        # Truth counterpart at the same latent (B and actor both see truth).
        truth_observation = dict(
            observation, state=self._rebase(observation["state"], built, self._with_context(built, self.truth_context))
        )
        if self.privileged_built is not None and "privileged_state" in observation:
            truth_observation["privileged_state"] = self._rebase(
                observation["privileged_state"], self.privileged_built, self.privileged_truth
            )
        truth_action = original_act(truth_observation, latent, mean=mean)
        difference = (action - truth_action).float()
        self.record["errors"]["action_l2"].append(float(difference.norm(dim=-1).max()))
        self.record["errors"]["action_max_abs"].append(float(difference.abs().max()))
        return action

    def perceive_initial_box_height(self) -> None:
        """Route the z-aware controller's height-band choice through perception.

        The height-banded controller reads the initial box height from this
        environment variable (written with the true sampled height) when it is
        built or reset; on hardware that height comes from the first detection.
        """

        if not self.perceives("controller") or INITIAL_BOX_Z_ENV not in os.environ:
            return
        self.ensure_tick()
        perceived = float(self.state.box_pos[2])
        if self.record.get("controller_initial_box_z") is None:
            print(
                f"[perception] controller initial box height: true={self.truth.box_pos[2]:.3f}m "
                f"perceived={perceived:.3f}m",
                flush=True,
            )
        self.record["controller_initial_box_z"] = perceived
        os.environ[INITIAL_BOX_Z_ENV] = repr(perceived)

    def record_phase(self, before: str, after: str, controller: Any) -> None:
        self.controller = controller
        if before != after:
            self.record["phase_transitions"].append(
                {"time_s": round(float(self.data.time), 4), "from": before, "to": after}
            )

    # ------------------------------------------------------------------ summary
    def finish_episode(self) -> None:
        if self.perception is None or self.record["ticks"] == 0:
            return
        record, data, controller = self.record, self.data, self.controller
        ticks = record["ticks"]
        first_phase_time: dict[str, float] = {}
        for item in record["phase_transitions"]:
            first_phase_time.setdefault(item["to"], item["time_s"])
        self.episodes.append(
            {
                "episode": record["episode"],
                "initial_box_pos": record["initial_box_pos"],
                "final_box_pos": np.asarray(data.qpos[self.box_qpos : self.box_qpos + 3]).tolist(),
                "final_root_height_m": float(data.xpos[self.root_body_id, 2]),
                "controller_initial_box_z": record.get("controller_initial_box_z"),
                "ticks": ticks,
                "true_pickup": record["true_pickup"],
                "true_pickup_definition": (
                    f"truth bilateral contact with box >= {LIFT_CLEARANCE_M:.2f} m above its start "
                    f"for >= {PICKUP_HOLD_TICKS} consecutive ticks"
                ),
                "max_true_lift_m": record["max_true_lift_m"],
                "true_bilateral_fraction": record["true_bilateral_ticks"] / ticks,
                "final_phase": getattr(controller, "phase", None),
                "first_phase_time_s": first_phase_time,
                "phase_transition_count": len(record["phase_transitions"]),
                "phase_transitions": record["phase_transitions"][:200],
                "fall_event_count": int(getattr(controller, "fall_event_count", 0)),
                "recovery_count": int(getattr(controller, "recovery_count", 0)),
                "place_attempt": int(getattr(controller, "place_attempt", 0)),
                "perception_stats": dict(self.perception.stats),
                "contact_mismatch_fraction": record["contact_mismatch_ticks"] / ticks,
                "bilateral_mismatch_fraction": (
                    record["bilateral_mismatch_ticks"] / ticks if self.perceives("controller") else None
                ),
                "errors": {key: _stats(values) for key, values in record["errors"].items()},
                "metric_errors": {key: _stats(values) for key, values in record["metric_errors"].items()},
            }
        )
        record["ticks"] = 0

    def write_summary(self) -> None:
        self.finish_episode()
        if self.summary_path is None:
            return
        payload = {"seed": self.seed, "config": asdict(self.config), "episodes": self.episodes}
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _install_policy_probe(module: Any, injector: PerceptionInjector) -> None:
    """Wrap the loaded policy: separate B observation and truth-action probe."""

    original_load = module.load_model_from_checkpoint_dir

    def load_model(*args, **kwargs):
        policy = original_load(*args, **kwargs)
        original_act = policy.act
        original_actor_prompt = policy.actor_prompt

        def act(observation, latent, mean=True):
            return injector.act(original_act, observation, latent, mean)

        def actor_prompt(normalized_obs, target_q):
            if injector.pending_b_observation is not None:
                normalized_obs = policy._normalize(injector.pending_b_observation)
            return original_actor_prompt(normalized_obs, target_q)

        policy.act = act
        policy.actor_prompt = actor_prompt
        return policy

    module.load_model_from_checkpoint_dir = load_model


def install(module: Any, config: PerceptionConfig, *, seed: int, summary_path: Path | None) -> PerceptionInjector:
    """Patch the fully assembled controller stack; call after every other overlay."""

    injector = PerceptionInjector(module, config, seed=seed, summary_path=summary_path)

    original_initialise = module._initialise_state

    def initialise_state(model, data, *args, **kwargs):
        result = original_initialise(model, data, *args, **kwargs)
        injector.bind(model, data)
        injector.reset_episode()
        return result

    module._initialise_state = initialise_state

    truth_observation = module.ActorObservation

    class PerceivedActorObservation(truth_observation):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            injector.attach_observer(self, kwargs)

        def build(self, **kwargs):
            return injector.build_actor_observation(self, super().build, kwargs)

    module._PerceptionTruthObservation = truth_observation
    module.ActorObservation = PerceivedActorObservation

    original_privileged = module._build_privileged_state

    def build_privileged_state(model, data, **kwargs):
        return injector.build_privileged_state(original_privileged, model, data, kwargs)

    module._build_privileged_state = build_privileged_state

    original_metrics = module._controller_metrics

    def controller_metrics(model, data, **kwargs):
        return injector.controller_metrics(original_metrics, model, data, kwargs)

    module._controller_metrics = controller_metrics

    parent = module.PhaseController

    class PerceivedController(parent):
        def __init__(self, *args, **kwargs):
            injector.perceive_initial_box_height()
            super().__init__(*args, **kwargs)

        def reset(self):
            injector.perceive_initial_box_height()
            super().reset()

        def latent(self, policy, observation, metrics, *, base_rotation, box_direction, goal_direction, z):
            box_direction, goal_direction = injector.latent_directions(box_direction, goal_direction)
            return super().latent(
                policy,
                observation,
                metrics,
                base_rotation=base_rotation,
                box_direction=box_direction,
                goal_direction=goal_direction,
                z=z,
            )

        def update(self, metrics, *, goal_height):
            before = self.phase
            super().update(metrics, goal_height=goal_height)
            injector.record_phase(before, self.phase, self)

    module.PhaseController = PerceivedController

    _install_policy_probe(module, injector)
    atexit.register(injector.write_summary)
    return injector


def install_from_env(module: Any) -> PerceptionInjector | None:
    profile = os.environ.get("UFO_PERCEPTION_PROFILE", "").strip()
    if not profile:
        print("[perception] no UFO_PERCEPTION_PROFILE: running the unmodified stack", flush=True)
        return None
    values = json.loads(Path(profile).read_text())
    channels = os.environ.get("UFO_PERCEPTION_CHANNELS", "").strip()
    if channels:
        values["channels"] = [item.strip() for item in channels.split(",") if item.strip()]
    config = PerceptionConfig.from_dict(values)
    seed = int(os.environ.get("UFO_PERCEPTION_SEED", "0"))
    summary = os.environ.get("UFO_PERCEPTION_SUMMARY_PATH", "").strip()
    print(
        f"[perception] profile={profile} channels={','.join(config.channels) or '-'} seed={seed}",
        flush=True,
    )
    return install(module, config, seed=seed, summary_path=Path(summary) if summary else None)
