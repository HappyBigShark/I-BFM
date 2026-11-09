#!/usr/bin/env python3
"""Self-checks for perception_noise.py (run with the repo venv, no pytest needed).

    cd ibfm && PYTHONPATH=$PWD .venv/bin/python tasks/carry_box/perception_diagnostic/check_perception_noise.py
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import perception_noise as pn  # noqa: E402


DT = 0.02
SCENE = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint/>
      <geom type="capsule" size="0.05 0.1" mass="10"/>
      <body name="torso_link" pos="0 0 0.2">
        <joint type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.1 0.1 0.1" mass="5"/>
      </body>
    </body>
    <body name="box" pos="1 0 0.5">
      <joint name="box_joint" type="free"/>
      <geom name="box_geom" type="box" size="0.15 0.15 0.15" pos="0.001 -0.0016 0.0064"
            quat="0.07016202 0.97153219 0.22227911 0.04236142" mass="0.7"/>
    </body>
  </worldbody>
</mujoco>
"""


def _world_body_velocity(model, data, body_id):
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), velocity, 0)
    return velocity[3:6].copy(), velocity[:3].copy()


FAKE_MODULE = types.SimpleNamespace(
    BOX_BODY_NAME="box",
    BOX_GEOM_NAME="box_geom",
    BOX_JOINT_NAME="box_joint",
    CONTROL_DT=DT,
    _world_body_velocity=_world_body_velocity,
)


def check_symmetry_groups() -> None:
    for name, size in (("none", 1), ("rect4", 4), ("oct8", 8), ("cube24", 24)):
        group = pn.symmetry_group(name)
        assert len(group) == size, (name, len(group))
        for element in group:
            assert np.allclose(element @ element.T, np.eye(3)) and np.isclose(np.linalg.det(element), 1.0)
        for first in group:
            for second in group:
                assert any(np.allclose(first @ second, item) for item in group), f"{name} not closed"
    for element in pn.symmetry_group("oct8"):
        assert abs(element[2, 2]) == 1.0


def check_rotation_maps() -> None:
    rng = np.random.default_rng(0)
    for angle in (0.0, 1e-9, 0.3, 2.0, math.pi - 1e-6, math.pi):
        axis = rng.normal(size=3)
        rotvec = angle * axis / np.linalg.norm(axis)
        recovered = pn.rotation_log(pn.rotation_exp(rotvec))
        assert np.allclose(pn.rotation_exp(recovered), pn.rotation_exp(rotvec), atol=1e-9), angle
        assert abs(np.linalg.norm(recovered) - angle) < 1e-6, angle
    quat = pn.mat_to_quat(pn.rotation_exp(np.asarray((0.1, -0.2, 0.3))))
    assert pn.quat_angle_deg(quat, quat) < 1e-12
    assert pn.quat_angle_deg(quat, -quat) < 1e-12


def _scene():
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    box_qpos = int(model.jnt_qposadr[model.joint("box_joint").id])
    data.qpos[box_qpos + 3 : box_qpos + 7] = pn.mat_to_quat(pn.rotation_exp(np.asarray((0.2, -0.4, 1.1))))
    data.qvel[:] = np.linspace(-0.5, 0.5, model.nv)
    mujoco.mj_forward(model, data)
    for _ in range(3):
        mujoco.mj_step(model, data)
    return model, data


def _injector(model, data, **config):
    injector = pn.PerceptionInjector(FAKE_MODULE, pn.PerceptionConfig(**config), seed=3, summary_path=None)
    injector.bind(model, data)
    injector.reset_episode()
    return injector


def check_identity_view_is_exact() -> None:
    model, data = _scene()
    injector = _injector(model, data)
    injector.ensure_tick(true_contact=np.ones(2, dtype=np.float32))
    view = injector.perceived_data()
    for name in ("qpos", "qvel", "xpos", "xmat", "xquat", "cvel", "geom_xmat", "geom_xpos", "subtree_com"):
        assert np.array_equal(getattr(view, name), getattr(data, name)), name


def check_view_carries_perceived_state() -> None:
    model, data = _scene()
    injector = _injector(
        model,
        data,
        pos_noise_std_m=0.05,
        rot_noise_std_deg=10.0,
        root_height_bias_std_m=0.05,
        base_vel_bias_std_mps=0.2,
    )
    injector.ensure_tick(true_contact=np.zeros(2, dtype=np.float32))
    state, view = injector.state, injector.perceived_data()
    box = injector.box_body_id
    lin, ang = _world_body_velocity(model, view, box)
    assert np.allclose(lin, state.box_lin_vel) and np.allclose(ang, state.box_ang_vel)
    assert np.allclose(view.qpos[injector.box_qpos : injector.box_qpos + 3], state.box_pos)
    # Box world height carries the base-height error; the robot-relative box does not.
    assert np.allclose(state.box_lin_vel - injector.truth.box_lin_vel, state.base_dv)
    root = injector.root_body_id
    assert np.isclose(view.xpos[root, 2] - data.xpos[root, 2], state.root_dz)
    torso = model.body("torso_link").id
    torso_lin, _ = _world_body_velocity(model, view, torso)
    torso_truth, _ = _world_body_velocity(model, data, torso)
    assert np.allclose(torso_lin - torso_truth, state.base_dv)
    # Geometric frame orientation follows the perceived marker-frame rotation.
    delta = pn.quat_to_mat(state.box_quat_wxyz) @ injector.truth.box_rot.T
    geom = injector.box_geom_id
    assert np.allclose(view.geom_xmat[geom].reshape(3, 3), delta @ data.geom_xmat[geom].reshape(3, 3))


def _truth(position, rotation, *, contact=(0.0, 0.0)) -> pn.TruthSample:
    return pn.TruthSample(
        box_pos=np.asarray(position, dtype=np.float64),
        box_quat_wxyz=pn.mat_to_quat(rotation),
        box_rot=rotation,
        box_lin_vel=np.zeros(3),
        box_ang_vel=np.zeros(3),
        camera_pos=np.asarray((0.0, 0.0, 1.2)),
        camera_rot=np.eye(3),
        base_xy=np.zeros(2),
        contact=np.asarray(contact, dtype=np.float32),
    )


def _geometry() -> pn.BoxGeometry:
    return pn.BoxGeometry(
        geom_pos=np.asarray((0.001, -0.0016, 0.0064)),
        geom_rot=pn.quat_to_mat(np.asarray((0.07016202, 0.97153219, 0.22227911, 0.04236142))),
    )


def check_latency_and_rate() -> None:
    config = pn.PerceptionConfig(rate_hz=15.0, latency_s=0.08)
    perception = pn.PerceptionModel(config, _geometry(), DT, np.random.default_rng(0))
    assert perception.period == 3 and perception.latency_steps == 4
    outputs = []
    for step in range(30):
        state = perception.tick(_truth((1.0 + step, 0.0, 0.5), np.eye(3)))
        outputs.append((state.box_pos[0] - 1.0, state.box_age_s))
    # Step 0 is the initial detection; then captures at multiples of 3 arrive 4 steps later.
    for step, (seen_step, age) in enumerate(outputs):
        if step < 4:
            assert np.isclose(seen_step, 0.0)
            continue
        expected = (step - 4) // 3 * 3
        assert np.isclose(seen_step, expected), (step, seen_step, expected)
        assert np.isclose(age, (step - expected) * DT)


def check_symmetry_preserves_geometry() -> None:
    geometry = _geometry()
    config = pn.PerceptionConfig(symmetry_group="cube24", symmetry_initial="random")
    rotation = pn.rotation_exp(np.asarray((0.3, 0.1, -0.7)))
    position = np.asarray((1.0, 0.2, 0.6))
    centre = position + rotation @ geometry.geom_pos
    changed = 0
    for seed in range(20):
        perception = pn.PerceptionModel(config, geometry, DT, np.random.default_rng(seed))
        state = perception.tick(_truth(position, rotation))
        perceived_rot = pn.quat_to_mat(state.box_quat_wxyz)
        # Same geometric centre and a symmetry-equivalent geometric frame.
        assert np.allclose(state.box_pos + perceived_rot @ geometry.geom_pos, centre)
        relative = (rotation @ geometry.geom_rot).T @ (perceived_rot @ geometry.geom_rot)
        assert any(np.allclose(relative, item, atol=1e-9) for item in pn.symmetry_group("cube24"))
        changed += int(not np.allclose(perceived_rot, rotation))
    assert changed > 10


def check_contact_delay() -> None:
    config = pn.PerceptionConfig(contact_delay_steps=3)
    perception = pn.PerceptionModel(config, _geometry(), DT, np.random.default_rng(0))
    seen = [perception.tick(_truth((1, 0, 0.5), np.eye(3), contact=(float(step >= 5),) * 2)).contact[0] for step in range(12)]
    assert seen.index(1.0) == 8, seen


def check_b_prompt_routing() -> None:
    """With the real checkpoint: B's prompt comes from the separate B observation."""

    checkpoint = Path(__file__).resolve().parents[1] / "checkpoint_ref" / "checkpoint"
    if not checkpoint.is_dir():
        print("skip check_b_prompt_routing (no checkpoint)")
        return
    import torch
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
    from humanoidverse.agents.spherical import sphere_log_map

    module = types.SimpleNamespace(load_model_from_checkpoint_dir=load_model_from_checkpoint_dir)
    injector = pn.PerceptionInjector(module, pn.PerceptionConfig(), seed=0, summary_path=None)
    pn._install_policy_probe(module, injector)
    policy = module.load_model_from_checkpoint_dir(checkpoint, device="cpu", strict=False).eval()
    assert policy.cfg.sphere_log_mode == "actor"
    generator = torch.Generator().manual_seed(0)
    observation = {
        key: torch.randn((1, *space.shape), generator=generator) for key, space in policy.obs_space.items()
    }
    b_observation = dict(observation, state=observation["state"] + 0.5 * torch.randn(observation["state"].shape, generator=generator))
    latent = policy.project_z(torch.randn((1, policy.cfg.archi.z_dim), generator=generator))
    plain = policy.act(observation, latent)
    injector.pending_b_observation = b_observation
    routed = policy.act(observation, latent)
    injector.pending_b_observation = None
    with torch.no_grad():
        normalized = policy._normalize(observation)
        prompt = sphere_log_map(policy._backward_map(policy._normalize(b_observation)), policy.project_z(latent))
        expected = policy._actor(normalized, prompt, policy.cfg.actor_std).mean.float()
    assert torch.allclose(routed, expected, atol=1e-5), float((routed - expected).abs().max())
    assert not torch.allclose(plain, routed), "B observation had no effect"
    assert torch.equal(policy.act(observation, latent), plain), "routing leaked past the call"


def main() -> None:
    checks = [
        check_symmetry_groups,
        check_rotation_maps,
        check_identity_view_is_exact,
        check_view_carries_perceived_state,
        check_latency_and_rate,
        check_symmetry_preserves_geometry,
        check_contact_delay,
        check_b_prompt_routing,
    ]
    for check in checks:
        check()
        print(f"ok  {check.__name__}")


if __name__ == "__main__":
    main()
