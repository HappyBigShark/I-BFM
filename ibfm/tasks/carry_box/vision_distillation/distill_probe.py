"""DAgger collection overlay: teacher labels, student observations, optional student control.

Installed last on the frozen z-aware Carry stack.  Every ``policy.act`` call
(one per control tick) records the teacher action as the label, advances the
two simulated perception streams, builds the student observation, and runs
the student whenever a checkpoint is given.  ``DISTILL_MODE`` selects whose
action is executed (teacher, or student = DAgger); ``DISTILL_CONTROLLER_SOURCE``
selects what the phase controller reads about the box:

* ``truth``    - simulator state (as in the teacher's own deployment script);
* ``estimate`` - the student's auxiliary head.  Every box-dependent metric,
  the latent-selection directions and the z-aware height band are rebuilt
  from the estimate; robot pose stays from proprioception/odometry.

The estimate used at a tick comes from the previous tick's student step (the
controller decides the latent the student itself needs).  At the first tick
there is none yet: the student is primed once with the approach latent, without
advancing its recurrent state.  Early estimates are unreliable (no recurrent
history yet), so for the first ``DISTILL_HEIGHT_ESTIMATE_TICK`` ticks (0.6 s) the
controller keeps the approach phase and makes no box-based phase decisions; the
z-aware height band starts from the controller's own no-information default and
is re-chosen from the converged estimate at that tick.  ``controller_truth_metric_calls``
in the outcome counts any metrics call answered from truth in estimate mode and
must stay 0.

Configuration comes from environment variables set by ``run_collectors.py``
and ``evaluate.py``.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import time
import uuid
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from student import AUX_CONTACT, AUX_DIM, AUX_LAYOUT, LOW_COLUMNS, LOW_DIM, load_checkpoint
from vision_sensor import (
    HIDDEN_GEOM_GROUP,
    SENTINEL,
    STATS_NAMES,
    USE_DEPTH_ANYTHING,
    USE_GEOMETRY,
    RaycastCamera,
    SimulatedPerception,
    VisionConfig,
    camera_config,
    encode_frame,
    hide_camera_invisible_geoms,
    perturb_camera_mount,
)


SUCCESS_RADIUS_M = 0.20
LIFT_CLEARANCE_M = 0.05
PICKUP_HOLD_TICKS = 10
FALLEN_ROOT_HEIGHT_M = 0.5
INITIAL_BOX_Z_ENV = "UFO_Z3D_INITIAL_BOX_Z"
# Root height from leg kinematics: pelvis height above the lowest foot point along
# gravity (IMU attitude).  On flat ground it is exact up to sole penetration
# (median -1.1 mm in simulation); the error model stands in for encoder, link
# and sole calibration errors on the robot.
FOOT_CONTACT_OFFSET_M = -0.0011
ROOT_HEIGHT_BIAS_STD_M = 0.01
ROOT_HEIGHT_NOISE_STD_M = 0.003
# z-aware's own value when no start height is known (box on the ground: baseline band).
NO_INFORMATION_BOX_Z = 0.186
AUX = {}
_offset = 0
for _name, _width in AUX_LAYOUT:
    AUX[_name] = slice(_offset, _offset + _width)
    _offset += _width
STAT = {name: index for index, name in enumerate(STATS_NAMES)}
METRIC_ERROR_KEYS = ("horizontal_distance", "goal_distance", "object_height", "object_speed", "box_upright")


def _quat_to_mat(quat_wxyz) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat / np.linalg.norm(quat))
    return matrix.reshape(3, 3)


class TorqueEstimate:
    """Motor torque as the G1 low-level state reports it (``tau_est``, from current).

    Per episode each joint gets an offset and a gain error (gearbox friction,
    current-to-torque calibration); every tick adds noise; one tick of delay.
    Returned normalised by the joint's effort limit.
    """

    def __init__(self, rng: np.random.Generator, effort: np.ndarray, *, bias=0.02, gain=0.05, noise=0.01, delay_steps=1):
        self.rng, self.effort = rng, np.asarray(effort, dtype=np.float64)
        self.bias = rng.normal(0.0, bias, self.effort.shape) * self.effort
        self.gain = 1.0 + rng.normal(0.0, gain, self.effort.shape)
        self.noise = noise
        self.history: deque[np.ndarray] = deque(maxlen=delay_steps + 1)

    def __call__(self, torque: np.ndarray) -> np.ndarray:
        self.history.append(np.asarray(torque, dtype=np.float64).copy())
        measured = self.history[0] * self.gain + self.bias + self.rng.normal(0.0, self.noise, self.effort.shape) * self.effort
        return (measured / self.effort).astype(np.float32)


class StudentRunner:
    """Recurrent CPU inference; image features are recomputed only when a frame changes."""

    def __init__(self, checkpoint: Path, known: np.ndarray, vision: VisionConfig):
        self.model, payload = load_checkpoint(checkpoint)
        self.step_id = int(payload["step"])
        columns = payload.get("low_columns")
        if columns is not None and list(columns) != LOW_COLUMNS:
            raise ValueError(
                f"{checkpoint} was trained with a different low-dim layout "
                f"(Depth Anything {'on' if 'depth_anything_age_s/0' in columns else 'off'}); set DISTILL_DEPTH_ANYTHING to match"
            )
        self.known, self.vision = known, vision
        self.hidden = None
        self.features: torch.Tensor | None = None

    @torch.no_grad()
    def act(self, images: tuple[np.ndarray, np.ndarray] | None, low: np.ndarray, latent: torch.Tensor):
        if images is not None:
            mask_image, depth_image = encode_frame(images[0], images[1], self.known, self.vision)
            self.features = self.model.encode_images(
                torch.from_numpy(mask_image)[None], torch.from_numpy(depth_image)[None]
            )
        action, aux, self.hidden = self.model.core(
            self.features[:, None],
            torch.from_numpy(low)[None, None],
            latent.float().reshape(1, 1, -1),
            self.hidden,
        )
        return action[:, 0], aux[0, 0].numpy().astype(np.float64)

    @torch.no_grad()
    def peek(self, images: tuple[np.ndarray, np.ndarray], low: np.ndarray, latent: torch.Tensor) -> np.ndarray:
        """Auxiliary estimate for these inputs without advancing the recurrent state."""

        mask_image, depth_image = encode_frame(images[0], images[1], self.known, self.vision)
        features = self.model.encode_images(torch.from_numpy(mask_image)[None], torch.from_numpy(depth_image)[None])
        _action, aux, _hidden = self.model.core(
            features[:, None], torch.from_numpy(low)[None, None], latent.float().reshape(1, 1, -1), self.hidden
        )
        return aux[0, 0].numpy().astype(np.float64)


class Stream:
    """Frames captured at one tick become usable ``latency`` ticks later."""

    def __init__(self, latency: int):
        self.latency = latency
        self.pending: deque[tuple[int, int, Any]] = deque()  # (ready tick, capture tick, payload)
        self.capture_tick: int | None = None
        self.value: Any = None

    def push(self, tick: int, payload: Any, *, immediate: bool = False, latency: int | None = None) -> None:
        delay = self.latency if latency is None else latency
        self.pending.append((tick if immediate else tick + delay, tick, payload))

    def advance(self, tick: int) -> bool:
        changed = False
        while self.pending and self.pending[0][0] <= tick:
            _, self.capture_tick, self.value = self.pending.popleft()
            changed = True
        return changed

    def age(self, tick: int, dt: float, cap: float) -> float:
        return cap if self.capture_tick is None else min((tick - self.capture_tick) * dt, cap)


class DistillationProbe:
    def __init__(self, module: Any):
        self.module = module
        # Optional per-run overrides of the perception error model (JSON), e.g. a pessimistic Depth Anything.
        self.vision = replace(VisionConfig(), **json.loads(os.environ.get("DISTILL_VISION_OVERRIDES", "{}")))
        self.camera_config = camera_config(self.vision)
        self.output_dir = Path(os.environ["DISTILL_OUTPUT_DIR"])
        self.outcomes_path = Path(os.environ.get("DISTILL_OUTCOMES", self.output_dir.parent / "outcomes.jsonl"))
        self.save_episode = os.environ.get("DISTILL_SAVE_EPISODE", "1") == "1"
        self.mode = os.environ.get("DISTILL_MODE", "teacher")
        if self.mode not in ("teacher", "student"):
            raise ValueError(f"DISTILL_MODE must be teacher or student, got {self.mode!r}")
        self.controller_source = os.environ.get("DISTILL_CONTROLLER_SOURCE", "truth")
        if self.controller_source not in ("truth", "estimate"):
            raise ValueError(f"DISTILL_CONTROLLER_SOURCE must be truth or estimate, got {self.controller_source!r}")
        checkpoint = os.environ.get("DISTILL_STUDENT_CHECKPOINT", "").strip()
        self.checkpoint = Path(checkpoint) if checkpoint else None
        if (self.mode == "student" or self.controller_source == "estimate") and self.checkpoint is None:
            raise ValueError("student control and estimated controller metrics need DISTILL_STUDENT_CHECKPOINT")
        self.seed = int(os.environ.get("DISTILL_SEED", "0"))
        self.meta = json.loads(os.environ.get("DISTILL_META", "{}"))
        self.early_stop_ticks = round(float(os.environ.get("DISTILL_COMPLETE_HOLD_S", "1000")) / module.CONTROL_DT)
        self.height_estimate_tick = int(os.environ.get("DISTILL_HEIGHT_ESTIMATE_TICK", "30"))
        # Evaluation-only sensitivity switches: turn individual error models off.  Their
        # random draws still happen, so every other noise realisation stays paired.
        self.ablate = {item for item in os.environ.get("DISTILL_ABLATE", "").split(",") if item}
        unknown = self.ablate - {"root_height", "mount", "depth_calib", "depth_latency"}
        if unknown:
            raise ValueError(f"unknown DISTILL_ABLATE entries: {sorted(unknown)}")
        self.target = np.asarray(
            [float(os.environ.get(name, default)) for name, default in (
                ("UFO_Z3D_TARGET_X", "2.5"), ("UFO_Z3D_TARGET_Y", "0.0"), ("UFO_Z3D_TARGET_BOX_Z", "0.65"))]
        )
        self.show_vision = os.environ.get("DISTILL_SHOW_VISION", "0") == "1"
        self.viewer = self.panel = None
        self.model = self.data = self.observer = self.controller = None
        self.tick = -1
        self.written = False
        self.started = time.time()

    # ------------------------------------------------------------------ setup
    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        module = self.module
        self.model, self.data = model, data
        rng = self.rng = np.random.default_rng([self.seed, 17])
        if "depth_calib" in self.ablate:
            self.vision = replace(self.vision, d435_blur_sigma_px=0.0)
        if "depth_latency" in self.ablate:
            fixed = 2
            self.vision = replace(self.vision, depth_latency_steps=(fixed, fixed))
        camera_id = model.camera(self.camera_config.name).id
        nominal_mount = model.cam_pos[camera_id].copy(), model.cam_quat[camera_id].copy()
        self.camera_mount_error = perturb_camera_mount(model, data, self.camera_config.name, self.vision, rng)
        if "mount" in self.ablate:
            model.cam_pos[camera_id], model.cam_quat[camera_id] = nominal_mount
            mujoco.mj_camlight(model, data)
        self.camera = RaycastCamera(model, self.camera_config.name, height=self.vision.height, width=self.vision.width)
        self.box_geom_id = model.geom(module.BOX_GEOM_NAME).id
        self.box_body_id = model.body(module.BOX_BODY_NAME).id
        box_joint = model.joint(module.BOX_JOINT_NAME).id
        self.box_qpos = int(model.jnt_qposadr[box_joint])
        root_joint = next(
            index for index in range(model.njnt)
            if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE and index != box_joint
        )
        self.root_body_id = int(model.jnt_bodyid[root_joint])
        self.root_qpos = int(model.jnt_qposadr[root_joint])
        self.root_qvel = int(model.jnt_dofadr[root_joint])
        self.robot_bodies = np.flatnonzero(model.body_rootid == model.body_rootid[self.root_body_id])
        self.robot_geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, self.robot_bodies))
        self.robot_geoms = set(self.robot_geom_ids.tolist())
        self.foot_geoms = [
            geom for geom in self.robot_geom_ids
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE
            and model.body(int(model.geom_bodyid[geom])).name.endswith("ankle_roll_link")
        ]
        if not self.foot_geoms:
            raise RuntimeError("no foot collision capsules found for the kinematic root height")
        self.view_data = mujoco.MjData(model)
        self.perception = SimulatedPerception(self.camera, self.box_geom_id, self.vision, rng)
        if USE_GEOMETRY:
            self.perception.nominal_mount = (*nominal_mount, self.root_body_id, model.body("torso_link").id)
        if "depth_calib" in self.ablate:
            self.perception.depth_scale, self.perception.depth_bias_m = 1.0, 0.0
        self.torque = TorqueEstimate(rng, self.effort_limits)
        self.root_height_bias = float(rng.normal(0.0, ROOT_HEIGHT_BIAS_STD_M)) * ("root_height" not in self.ablate)
        self.root_height_noise = 0.0
        self.root_height_errors: list[float] = []
        self.velocity_errors: list[float] = []  # |estimated - true| root velocity (m/s)
        self.static_false_alarms: list[float] = []  # truly still (< 0.05 m/s) but estimated > 0.15 m/s
        self.student = StudentRunner(self.checkpoint, self.perception.known, self.vision) if self.checkpoint else None
        if self.show_vision:
            from vision_overlay import VisionPanel

            self.panel = VisionPanel(self.vision, self.perception.known)
        self.tick = -1
        self.mask_captures = 0
        self.depth_stream = Stream(self.vision.depth_latency_steps[1])
        self.mask_stream = Stream(self.vision.mask_latency_steps)
        self.da_stream = Stream(self.vision.da_latency_steps)
        self.depth_frames: list[np.ndarray] = []
        self.mask_frames: list[np.ndarray] = []
        self.mask_stats: list[np.ndarray] = []
        keys = ("depth_index", "mask_index", "stats", "low", "z", "teacher", "aux", "estimate", "phase")
        self.rows: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        self.phase_names: list[str] = []
        self.initial_box_pos = None
        self.hold_run = 0
        self.true_pickup = False
        self.complete_ticks = 0
        self.estimate: np.ndarray | None = None
        self.metric_errors: dict[str, list[float]] = {key: [] for key in METRIC_ERROR_KEYS}
        self.contact_mismatch: list[float] = []
        self.height_estimate: dict[str, float] | None = None
        self.truth_metric_calls = 0
        self.prepared_tick = -1

    # ------------------------------------------------------------------ perception
    def _advance_streams(self) -> bool:
        tick, vision, dt = self.tick, self.vision, self.module.CONTROL_DT
        if tick % vision.depth_period_steps == 0:
            is_mask = tick % vision.mask_period_steps == 0
            is_da = USE_DEPTH_ANYTHING and is_mask and self.mask_captures % vision.da_every_masks == 0
            frame = self.perception.capture(self.data, tick * dt, mask=is_mask, depth_anything=is_da)
            immediate = tick == 0  # the episode starts with one capture of each stream in hand
            self.depth_frames.append(frame["depth_mm"])
            latency = int(self.rng.integers(vision.depth_latency_steps[0], vision.depth_latency_steps[1] + 1))
            self.depth_stream.push(tick, len(self.depth_frames) - 1, immediate=immediate, latency=latency)
            if is_mask:
                self.mask_captures += 1
                self.mask_frames.append(frame["mask"])
                self.mask_stats.append(frame["stats"])
                self.mask_stream.push(tick, len(self.mask_frames) - 1, immediate=immediate)
            if is_da:
                self.da_stream.push(tick, frame["da_median_m"], immediate=immediate)
        changed = self.depth_stream.advance(tick)
        changed |= self.mask_stream.advance(tick)
        self.da_stream.advance(tick)
        return changed

    def _stats(self) -> np.ndarray:
        stats = self.mask_stats[self.mask_stream.value].copy()
        if USE_DEPTH_ANYTHING:
            da = self.da_stream.value
            stats[STAT["da_valid"]] = float(da is not None)
            stats[STAT["da_median"]] = SENTINEL if da is None else float(self.vision.normalise_depth(da))
        return stats

    # ------------------------------------------------------------------ per tick
    def prepare_tick(self, kwargs: dict[str, Any], observation: dict[str, torch.Tensor]) -> None:
        """Advance perception and build the student inputs as soon as the tick's observation exists."""

        self.observation_kwargs = kwargs
        self.tick += 1
        tick, dt, cap = self.tick, self.module.CONTROL_DT, self.vision.age_cap_s
        self.images_changed = self._advance_streams()
        stats = self._stats()

        rotation = _quat_to_mat(kwargs["base_quat_wxyz"])
        base_pos = np.asarray(kwargs["base_pos"], dtype=np.float64)
        to_root = lambda world: (np.asarray(world, dtype=np.float64) - base_pos) @ rotation
        goal = self.observer.target_object_pos.detach().cpu().numpy().reshape(3)
        self.root_height_noise = float(self.rng.normal(0.0, ROOT_HEIGHT_NOISE_STD_M)) * ("root_height" not in self.ablate)
        root_height = self.root_height_estimate(self.data)
        self.root_height_errors.append(abs(root_height - float(base_pos[2])))
        estimated_base = np.asarray((base_pos[0], base_pos[1], root_height))  # x, y, yaw: odometry (truth for now)
        hands = []
        for index in range(2):
            hand_rot = _quat_to_mat(kwargs["hand_body_quat_wxyz"][index])
            hands.append(to_root(np.asarray(kwargs["hand_body_pos"][index]) + hand_rot @ self.hand_offsets[index]))
        truth_contact = np.asarray(kwargs["hand_contact"], dtype=np.float32)
        joint_torque = self.torque(self.data.actuator_force[self.actuator_indices])
        low = np.concatenate(
            (
                observation["state"][0, : 2 * 29 + 6].float().cpu().numpy(),
                observation["history_actor"][0].float().cpu().numpy(),
                observation["last_action"][0].float().cpu().numpy(),
                (goal - estimated_base) @ rotation,
                np.concatenate(hands),
                joint_torque,
                stats,
                (
                    self.depth_stream.age(tick, dt, cap),
                    self.mask_stream.age(tick, dt, cap),
                    *((self.da_stream.age(tick, dt, cap),) if USE_DEPTH_ANYTHING else ()),
                ),
            )
        ).astype(np.float32)
        if low.shape != (LOW_DIM,):
            raise RuntimeError(f"student low-dim observation has {low.shape}, expected {LOW_DIM}")
        self.low, self.stats = low, stats

        box_pos = np.asarray(self.data.qpos[self.box_qpos : self.box_qpos + 3], dtype=np.float64)
        if self.initial_box_pos is None:
            self.initial_box_pos = box_pos.copy()
        box_lin_vel, box_ang_vel = self.module._world_body_velocity(self.model, self.data, self.box_body_id)
        lift = float(box_pos[2] - self.initial_box_pos[2])
        upright = float(np.clip(np.asarray(self.data.geom_xmat[self.box_geom_id]).reshape(3, 3)[2, 2], 0.0, 1.0))
        aux = np.concatenate(
            (
                to_root(box_pos),
                np.asarray(box_lin_vel) @ rotation,
                (lift, upright, float(box_ang_vel[2])),
                self._root_velocity_heading(rotation),
                truth_contact,
                (float(self._robot_carries_box()),),
            )
        ).astype(np.float32)
        assert aux.shape == (AUX_DIM,)
        bilateral = bool(truth_contact[0] > 0.5 and truth_contact[1] > 0.5)
        self.hold_run = self.hold_run + 1 if bilateral and lift >= LIFT_CLEARANCE_M else 0
        self.true_pickup |= self.hold_run >= PICKUP_HOLD_TICKS
        self.aux_truth = aux
        self.prepared_tick = tick

    def _root_velocity_heading(self, rotation: np.ndarray) -> np.ndarray:
        """Root linear velocity: x/y in the heading (yaw-only) frame, z world-vertical."""

        velocity = np.asarray(self.data.qvel[self.root_qvel : self.root_qvel + 3], dtype=np.float64)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        cos, sin = math.cos(yaw), math.sin(yaw)
        return np.asarray((cos * velocity[0] + sin * velocity[1], -sin * velocity[0] + cos * velocity[1], velocity[2]))

    def _robot_carries_box(self) -> bool:
        """Training label: the box touches the robot and nothing else (no floor or platform)."""

        touched = False
        for contact in self.data.contact[: self.data.ncon]:
            if self.box_geom_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.box_geom_id else contact.geom1
            if int(other) not in self.robot_geoms:
                return False
            touched = True
        return touched

    def _current_images(self) -> tuple[np.ndarray, np.ndarray]:
        return self.depth_frames[self.depth_stream.value], self.mask_frames[self.mask_stream.value]

    def prime_estimate(self, latent: torch.Tensor) -> None:
        """First tick: an estimate for the controller before the student has acted."""

        if self.controller_source == "estimate" and self.estimate is None and self.prepared_tick == self.tick:
            self.estimate = self.student.peek(self._current_images(), self.low, latent)

    def act(self, original_act, observation: dict[str, torch.Tensor], latent: torch.Tensor, mean: bool):
        teacher = original_act(observation, latent, mean=mean)
        if self.prepared_tick != self.tick:
            raise RuntimeError("policy.act without a prepared tick: observation build hook did not run")
        tick, low, stats, aux = self.tick, self.low, self.stats, self.aux_truth
        images_changed, self.images_changed = self.images_changed, False
        executed = teacher
        estimate = np.full(AUX_DIM, np.nan, dtype=np.float32)
        if self.student is not None:
            images = self._current_images() if images_changed else None
            action, self.estimate = self.student.act(images, low, latent)
            estimate = self.estimate.astype(np.float32)
            velocity = AUX["root_lin_vel_heading"]
            true_velocity, estimated_velocity = aux[velocity], estimate[velocity]
            self.velocity_errors.append(float(np.linalg.norm(estimated_velocity - true_velocity)))
            if np.linalg.norm(true_velocity[:2]) < 0.05:
                self.static_false_alarms.append(float(np.linalg.norm(estimated_velocity[:2]) > 0.15))
            if self.mode == "student":
                executed = action.to(dtype=teacher.dtype, device=teacher.device)

        phase = getattr(self.controller, "phase", "starting")
        if phase not in self.phase_names:
            self.phase_names.append(phase)
        rows = self.rows
        rows["depth_index"].append(np.int32(self.depth_stream.value))
        rows["mask_index"].append(np.int32(self.mask_stream.value))
        rows["stats"].append(stats)
        rows["low"].append(low)
        rows["z"].append(latent.detach().float().cpu().numpy().reshape(-1).astype(np.float16))
        rows["teacher"].append(teacher.detach().float().cpu().numpy().reshape(-1))
        rows["aux"].append(aux)
        rows["estimate"].append(estimate)
        rows["phase"].append(np.int8(self.phase_names.index(phase)))

        if self.panel is not None:
            self._show_vision(images_changed, phase)
        if self.using_estimate() and tick == self.height_estimate_tick and phase == "approach":
            self._apply_height_estimate()
        self.complete_ticks = self.complete_ticks + 1 if phase == "complete" else 0
        if self.complete_ticks >= self.early_stop_ticks:
            self.write_episode("complete_hold")
            raise SystemExit(0)
        return executed

    def _show_vision(self, images_changed: bool, phase: str) -> None:
        from vision_overlay import describe

        if images_changed or self.panel.image is None:
            self.panel.update(*self._current_images())
        dt, cap = self.module.CONTROL_DT, self.vision.age_cap_s
        ages = tuple(stream.age(self.tick, dt, cap) for stream in (self.depth_stream, self.mask_stream))
        truth_distance = float(np.linalg.norm(self.aux_truth[AUX["box_pos_root"]][:2]))
        extra = {"phase": f"{phase} ({self.mode}, controller reads {self.controller_source})"}
        if self.estimate is not None:
            estimated = float(np.linalg.norm(self.estimate[AUX["box_pos_root"]][:2]))
            extra["box distance est / truth"] = f"{estimated:.2f} / {truth_distance:.2f} m"
            contact = self.estimate[AUX_CONTACT] > 0.0
            truth = self.aux_truth[AUX_CONTACT] > 0.5
            extra["hand contact L/R est / truth"] = (
                f"{int(contact[0])}{int(contact[1])} / {int(truth[0])}{int(truth[1])}"
            )
            load = 1.0 / (1.0 + np.exp(-self.estimate[AUX["box_load_logit"]][0]))
            extra["robot carries box est / truth"] = f"{load:.2f} / {int(self.aux_truth[AUX['box_load_logit']][0])}"
        else:
            extra["box distance truth"] = f"{truth_distance:.2f} m"
        self.panel.show(self.viewer, describe(self.stats, ages, self.vision, extra))

    # ------------------------------------------------------------------ estimated controller inputs
    def using_estimate(self) -> bool:
        return self.controller_source == "estimate" and self.estimate is not None

    def root_height_estimate(self, data: mujoco.MjData) -> float:
        """Pelvis height above the lowest foot point along gravity, plus the error model."""

        lowest = np.inf
        for geom in self.foot_geoms:
            axis_z = abs(float(data.geom_xmat[geom][8]))
            lowest = min(lowest, float(data.geom_xpos[geom][2]) - axis_z * float(self.model.geom_size[geom][1]) - float(self.model.geom_size[geom][0]))
        kinematic = float(data.xpos[self.root_body_id][2]) - lowest + FOOT_CONTACT_OFFSET_M
        return kinematic + self.root_height_bias + self.root_height_noise

    def estimated_box_world(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        base_pos = np.asarray(data.xpos[self.root_body_id], dtype=np.float64).copy()
        base_pos[2] = self.root_height_estimate(data)
        rotation = np.asarray(data.xmat[self.root_body_id]).reshape(3, 3)
        return base_pos + rotation @ self.estimate[AUX["box_pos_root"]], rotation @ self.estimate[AUX["box_lin_vel_root"]]

    def _estimate_view(self, data: mujoco.MjData) -> mujoco.MjData:
        """Real MjData with the box and root height replaced by their estimates (no kinematics recompute)."""

        view = self.view_data
        mujoco.mj_copyData(view, self.model, data)
        position, velocity = self.estimated_box_world(data)
        # The robot itself sits at the kinematic root height estimate.
        root_shift = self.root_height_estimate(data) - float(data.xpos[self.root_body_id][2])
        view.qpos[self.root_qpos + 2] += root_shift
        for array in (view.xpos, view.xipos, view.subtree_com):
            array[self.robot_bodies, 2] += root_shift
        view.geom_xpos[self.robot_geom_ids, 2] += root_shift
        box, geom = self.box_body_id, self.box_geom_id
        shift = position - np.asarray(data.qpos[self.box_qpos : self.box_qpos + 3])
        view.qpos[self.box_qpos : self.box_qpos + 3] = position
        for array, index in ((view.xpos, box), (view.xipos, box), (view.subtree_com, box), (view.geom_xpos, geom)):
            array[index] += shift
        # The box is its own kinematic tree: cvel is [angular, linear] at its centre of mass.
        view.cvel[box, :3] = (0.0, 0.0, self.estimate[AUX["box_yaw_rate"]][0])
        view.cvel[box, 3:] = velocity
        return view

    def controller_metrics(self, original, model, data, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.metrics_call = (original, model, data, kwargs)
        truth = original(model, data, **kwargs)
        if not self.using_estimate():
            self.truth_metric_calls += self.controller_source == "estimate"
            return truth
        contact = (self.estimate[AUX_CONTACT] > 0.0).astype(np.float32)
        estimated = original(model, self._estimate_view(data), **dict(kwargs, contact=contact))
        estimated["box_upright"] = float(np.clip(self.estimate[AUX["box_upright"]][0], 0.0, 1.0))
        for key in METRIC_ERROR_KEYS:
            self.metric_errors[key].append(abs(float(estimated[key]) - float(truth[key])))
        self.contact_mismatch.append(float(bool(estimated["bilateral_contact"]) != bool(truth["bilateral_contact"])))
        return estimated

    def latent_directions(self, box_direction: np.ndarray, goal_direction: np.ndarray):
        if not self.using_estimate():
            return box_direction, goal_direction
        truth_box = np.asarray(self.data.qpos[self.box_qpos : self.box_qpos + 2], dtype=np.float64)
        estimated_box = self.estimated_box_world(self.data)[0][:2]
        goal_xy = np.asarray(goal_direction) + truth_box
        return np.asarray(box_direction) + estimated_box - truth_box, goal_xy - estimated_box

    def _apply_height_estimate(self) -> None:
        """Re-run the z-aware height-band choice with the estimated start height."""

        estimated = float(self.estimated_box_world(self.data)[0][2])
        self.height_estimate = {"estimated": estimated, "truth": float(self.initial_box_pos[2])}
        os.environ[INITIAL_BOX_Z_ENV] = repr(estimated)
        self.controller.dynamic_height_reset = True
        self.controller.reset()

    # ------------------------------------------------------------------ output
    def write_episode(self, reason: str) -> None:
        if self.written or self.data is None or self.tick < 0:
            return
        self.written = True
        data = self.data
        box_pos = np.asarray(data.qpos[self.box_qpos : self.box_qpos + 3], dtype=np.float64)
        root_height = float(data.xpos[self.root_body_id, 2])
        error = float(np.linalg.norm(box_pos - self.target))
        horizon_s = (self.tick + 1) * self.module.CONTROL_DT
        outcome = {
            "episode_id": self.meta.get("episode_id", uuid.uuid4().hex),
            "mode": self.mode,
            "controller_source": self.controller_source,
            "student_step": None if self.student is None else self.student.step_id,
            "ticks": self.tick + 1,
            "finished_by": reason,
            "placement_success": error < SUCCESS_RADIUS_M,
            "final_error_3d_m": error,
            "true_pickup": bool(self.true_pickup),
            "fallen": root_height < FALLEN_ROOT_HEIGHT_M,
            "final_phase": getattr(self.controller, "phase", None),
            "d435_max_range_m": self.perception.max_range_m,
            "d435_outage_s": sum(min(end, horizon_s) - start for start, end in self.perception.outages if start < horizon_s),
            "height_estimate": self.height_estimate,
            "root_height_mae_m": float(np.mean(self.root_height_errors)) if self.root_height_errors else None,
            "root_velocity_error_mean_mps": float(np.mean(self.velocity_errors)) if self.velocity_errors else None,
            "root_velocity_error_p95_mps": float(np.percentile(self.velocity_errors, 95)) if self.velocity_errors else None,
            "root_velocity_static_false_alarm": float(np.mean(self.static_false_alarms)) if self.static_false_alarms else None,
            "camera_mount_error": self.camera_mount_error,
            "ablate": sorted(self.ablate),
            "d435_scale": self.perception.depth_scale,
            "d435_bias_m": self.perception.depth_bias_m,
            "controller_truth_metric_calls": self.truth_metric_calls if self.controller_source == "estimate" else None,
            "controller_metric_mae": {key: float(np.mean(values)) for key, values in self.metric_errors.items() if values},
            "controller_bilateral_mismatch": float(np.mean(self.contact_mismatch)) if self.contact_mismatch else None,
            "wall_time_s": time.time() - self.started,
            **self.meta,
        }
        with open(self.outcomes_path, "a") as stream:
            stream.write(json.dumps(outcome) + "\n")
        if not self.save_episode:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        final = self.output_dir / f"ep_{outcome['episode_id']}.npz"
        temporary = final.with_name(final.name + ".tmp")
        with open(temporary, "wb") as stream:
            np.savez_compressed(
                stream,
                depth_mm=np.stack(self.depth_frames),
                mask=np.stack(self.mask_frames),
                known=self.perception.known,
                **{key: np.stack(values) for key, values in self.rows.items()},
                outcome=np.asarray(json.dumps(outcome)),
                phase_names=np.asarray(json.dumps(self.phase_names)),
            )
        os.replace(temporary, final)


def install(module: Any) -> DistillationProbe:
    probe = DistillationProbe(module)

    original_build = module._build_model

    def build_model(*args, **kwargs):
        kwargs["rgbd_camera"] = probe.camera_config
        model = original_build(*args, **kwargs)
        # The goal marker is a visualisation aid, not part of the real scene.
        model.geom_group[model.geom(module.GOAL_GEOM_NAME).id] = HIDDEN_GEOM_GROUP
        hide_camera_invisible_geoms(model)
        return model

    module._build_model = build_model

    original_initialise = module._initialise_state

    def initialise_state(model, data, *args, **kwargs):
        result = original_initialise(model, data, *args, **kwargs)
        if probe.model is not None and probe.tick >= 0:
            raise RuntimeError("distillation episodes are one per process; reset is not supported")
        probe.bind(model, data)
        return result

    module._initialise_state = initialise_state

    parent_observation = module.ActorObservation

    class ProbedObservation(parent_observation):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            probe.observer = self
            probe.hand_offsets = np.asarray(kwargs["hand_contact_offsets"], dtype=np.float64).reshape(2, 3)

        def build(self, **kwargs):
            observation = super().build(**kwargs)
            probe.prepare_tick(kwargs, observation)
            return observation

    module.ActorObservation = ProbedObservation

    original_metrics = module._controller_metrics

    def controller_metrics(model, data, **kwargs):
        return probe.controller_metrics(original_metrics, model, data, kwargs)

    module._controller_metrics = controller_metrics

    parent_controller = module.PhaseController

    class ProbedController(parent_controller):
        def __init__(self, *args, **kwargs):
            if probe.controller_source == "estimate":
                # The random-height reset wrote the true start height; the controller must not see it.
                os.environ[INITIAL_BOX_Z_ENV] = repr(NO_INFORMATION_BOX_Z)
            super().__init__(*args, **kwargs)
            probe.controller = self

        def latent(self, policy, observation, metrics, *, base_rotation, box_direction, goal_direction, z):
            if probe.controller_source == "estimate" and probe.estimate is None:
                probe.prime_estimate(z["approach"])
                original, model, data, kwargs = probe.metrics_call
                metrics = probe.controller_metrics(original, model, data, kwargs)
                probe.truth_metric_calls -= 1  # the first call was superseded by this one
            box_direction, goal_direction = probe.latent_directions(box_direction, goal_direction)
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
            if probe.controller_source == "estimate" and probe.tick < probe.height_estimate_tick:
                return  # estimate warm-up: no phase decisions on unconverged estimates
            super().update(metrics, goal_height=goal_height)

    module.PhaseController = ProbedController

    original_load = module.load_model_from_checkpoint_dir

    def load_model(*args, **kwargs):
        policy = original_load(*args, **kwargs)
        original_act = policy.act

        def act(observation, latent, mean=True):
            return probe.act(original_act, observation, latent, mean)

        policy.act = act
        return policy

    module.load_model_from_checkpoint_dir = load_model

    # The motor torques need the controller's joint order and effort limits.
    original_spec = module.load_robot_training_spec

    def load_robot_training_spec(*args, **kwargs):
        spec = original_spec(*args, **kwargs)
        probe.effort_limits = np.asarray(spec.effort_limits, dtype=np.float64)
        return spec

    module.load_robot_training_spec = load_robot_training_spec
    original_actuators = module._actuator_indices

    def actuator_indices(model, joint_names):
        indices = original_actuators(model, joint_names)
        probe.actuator_indices = np.asarray(indices, dtype=int)
        return indices

    module._actuator_indices = actuator_indices

    if probe.show_vision:
        viewer_module = module.mujoco.viewer
        original_launch = viewer_module.launch_passive

        def launch_passive(*args, **kwargs):
            probe.viewer = original_launch(*args, **kwargs)
            return probe.viewer

        viewer_module.launch_passive = launch_passive
        original_annotate = module._annotate_video_frame

        def annotate_video_frame(frame, label):
            frame = probe.panel.composite(frame) if probe.panel is not None else frame
            return original_annotate(frame, label)

        module._annotate_video_frame = annotate_video_frame
    atexit.register(lambda: probe.write_episode("max_steps"))
    return probe
