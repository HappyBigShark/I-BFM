"""Vision student policy for the Carry Box distillation.

Inputs per control tick (see ``LOW_DIM_LAYOUT``): proprioception and action
history, goal and hands in the root frame, the motors' estimated joint torques
(G1 ``tau_est``; the student infers hand-box contact and load from them),
per-stream mask/depth statistics with their ages, the controller's latent
``z``, and the latest mask / depth images.  Outputs the 29-D teacher action
plus auxiliary box-state predictions for a future vision phase controller.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_sensor import DEPTH_ANYTHING_STATS, STATS_NAMES, STATS_NAMES_BASE, USE_DEPTH_ANYTHING


NUM_DOF = 29
HISTORY_LENGTH = 4
LOW_DIM_LAYOUT = (
    ("proprio", 2 * NUM_DOF + 6),  # dof_pos_rel, dof_vel, projected_gravity, base_ang_vel
    ("history_actor", HISTORY_LENGTH * (3 * NUM_DOF + 6)),
    ("last_action", NUM_DOF),
    ("goal_root", 3),
    ("hands_root", 6),
    ("joint_torque", NUM_DOF),  # estimated motor torque / effort limit
    ("vision_stats", len(STATS_NAMES)),
    ("depth_age_s", 1),
    ("mask_age_s", 1),
) + ((("depth_anything_age_s", 1),) if USE_DEPTH_ANYTHING else ())
LOW_DIM = sum(width for _, width in LOW_DIM_LAYOUT)
# Truth targets for the auxiliary head — everything the phase controller reads
# about the box: position and linear velocity in the root frame, height above
# its start, uprightness (geom z-axis up-component), world yaw rate, the
# left/right hand-box contacts, and whether the robot alone carries the box
# (load), plus the root velocity.  Contacts and load are logits; truth is a
# training label only.
AUX_LAYOUT = (
    ("box_pos_root", 3),
    ("box_lin_vel_root", 3),
    ("box_height_above_start", 1),
    ("box_upright", 1),
    ("box_yaw_rate", 1),
    # Root linear velocity: x/y in the heading (yaw-only) frame, z world-vertical.
    # Supervised for a future odometry; not read by the controller or the actor.
    ("root_lin_vel_heading", 3),
    ("hand_contact_logit", 2),
    ("box_load_logit", 1),
)
AUX_DIM = sum(width for _, width in AUX_LAYOUT)
AUX_CONTACT = slice(AUX_DIM - 3, AUX_DIM - 1)
AUX_BINARY = slice(AUX_DIM - 3, AUX_DIM)  # contacts + load


@dataclass
class StudentConfig:
    image_height: int = 60
    image_width: int = 106
    z_dim: int = 256
    action_dim: int = NUM_DOF
    low_dim: int = LOW_DIM
    mask_features: int = 128
    depth_features: int = 256
    low_features: int = 512
    z_features: int = 256
    fused_features: int = 1024
    hidden: int = 512


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, features: int, height: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.net(torch.zeros(1, in_channels, height, width)).shape[-1]
        self.head = nn.Sequential(nn.Linear(flat, features), nn.ELU())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(images))


class StudentPolicy(nn.Module):
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        c = config
        self.mask_encoder = ConvEncoder(2, c.mask_features, c.image_height, c.image_width)
        self.depth_encoder = ConvEncoder(4, c.depth_features, c.image_height, c.image_width)
        self.low_mlp = nn.Sequential(
            nn.Linear(c.low_dim, c.low_features), nn.ELU(), nn.Linear(c.low_features, c.low_features), nn.ELU()
        )
        self.z_mlp = nn.Sequential(nn.Linear(c.z_dim, c.z_features), nn.ELU())
        fused_in = c.mask_features + c.depth_features + c.low_features + c.z_features
        self.fuse = nn.Sequential(nn.Linear(fused_in, c.fused_features), nn.ELU())
        self.gru = nn.GRU(c.fused_features, c.hidden, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(c.hidden + c.fused_features, 512), nn.ELU(), nn.Linear(512, c.action_dim)
        )
        self.aux_head = nn.Linear(c.hidden, AUX_DIM)
        self.register_buffer("low_mean", torch.zeros(c.low_dim))
        self.register_buffer("low_std", torch.ones(c.low_dim))

    def encode_images(self, mask_image: torch.Tensor, depth_image: torch.Tensor) -> torch.Tensor:
        """[N, 2|4, H, W] -> [N, mask_features + depth_features]."""

        return torch.cat((self.mask_encoder(mask_image), self.depth_encoder(depth_image)), dim=-1)

    def core(self, image_features, low, z, hidden):
        """Sequences [B, T, ...] -> (action [B, T, A], aux [B, T, AUX_DIM], hidden)."""

        low = (low - self.low_mean) / self.low_std
        fused = self.fuse(torch.cat((image_features, self.low_mlp(low), self.z_mlp(z)), dim=-1))
        output, hidden = self.gru(fused, hidden)
        action = self.action_head(torch.cat((output, fused), dim=-1))
        return action, self.aux_head(output), hidden

    def forward(self, mask_image, depth_image, low, z, hidden=None):
        batch, steps = low.shape[:2]
        features = self.encode_images(mask_image.flatten(0, 1), depth_image.flatten(0, 1))
        return self.core(features.view(batch, steps, -1), low, z, hidden)


def aux_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    regression = F.mse_loss(prediction[..., : AUX_BINARY.start], target[..., : AUX_BINARY.start])
    binary = F.binary_cross_entropy_with_logits(prediction[..., AUX_BINARY], target[..., AUX_BINARY])
    return regression + binary


def save_checkpoint(path, model: StudentPolicy, *, step: int, extra: dict | None = None) -> None:
    payload = {
        "model": model.state_dict(),
        "config": asdict(model.config),
        "step": int(step),
        "low_columns": LOW_COLUMNS,
        "aux_columns": AUX_COLUMNS,
        **(extra or {}),
    }
    temporary = f"{path}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def column_names(layout, stats_names=STATS_NAMES) -> list[str]:
    names = []
    for name, width in layout:
        if name == "vision_stats":
            names.extend(f"vision_stats/{stat}" for stat in stats_names)
        else:
            names.extend(f"{name}/{index}" for index in range(width))
    return names


LOW_COLUMNS = column_names(LOW_DIM_LAYOUT)
AUX_COLUMNS = column_names(AUX_LAYOUT)
# Checkpoints saved before column names were stored: 518-D input with Depth
# Anything (two stats + an age), 12-D auxiliary head without the root velocity.
_STATS_WITH_DEPTH_ANYTHING = STATS_NAMES_BASE[:2] + DEPTH_ANYTHING_STATS + STATS_NAMES_BASE[2:]
_LAYOUT_WITHOUT_DEPTH_ANYTHING = tuple(item for item in LOW_DIM_LAYOUT if item[0] != "depth_anything_age_s")
LEGACY_LOW_COLUMNS = column_names(_LAYOUT_WITHOUT_DEPTH_ANYTHING + (("depth_anything_age_s", 1),), _STATS_WITH_DEPTH_ANYTHING)
LEGACY_AUX_COLUMNS = column_names(tuple(item for item in AUX_LAYOUT if item[0] != "root_lin_vel_heading"))


def warm_start(model: StudentPolicy, path) -> list[str]:
    """Copy parameters from an earlier checkpoint; return what was left at initialisation.

    Input columns of the low-dim layer, its normaliser and the auxiliary-head
    rows are matched by name: removed channels are dropped, kept ones copied,
    new input columns start at zero (no effect) and new outputs at their
    initialisation.  Everything else must match in shape.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload["model"]
    old_low = payload.get("low_columns") or (LEGACY_LOW_COLUMNS if source["low_mean"].numel() == len(LEGACY_LOW_COLUMNS) else None)
    old_aux = payload.get("aux_columns") or (LEGACY_AUX_COLUMNS if source["aux_head.bias"].numel() == len(LEGACY_AUX_COLUMNS) else None)
    state = model.state_dict()
    loaded, skipped = {}, []

    def remap(value, old, old_names, new_names, axis, fill):
        result = torch.full_like(value, fill)
        index = {name: i for i, name in enumerate(old_names)}
        for new_i, name in enumerate(new_names):
            if name in index:
                result.select(axis, new_i).copy_(old.select(axis, index[name]))
            else:
                skipped.append(f"{name}")
        return result

    for key, value in state.items():
        old = source.get(key)
        if old is not None and old.shape == value.shape and key not in ("aux_head.weight", "aux_head.bias"):
            loaded[key] = old
        elif key == "low_mlp.0.weight" and old_low is not None:
            loaded[key] = remap(value, old, old_low, LOW_COLUMNS, 1, 0.0)
        elif key in ("low_mean", "low_std") and old_low is not None:
            loaded[key] = remap(value, old, old_low, LOW_COLUMNS, 0, 0.0 if key == "low_mean" else 1.0)
        elif key in ("aux_head.weight", "aux_head.bias") and old_aux is not None:
            fresh = value.clone()
            index = {name: i for i, name in enumerate(old_aux)}
            for new_i, name in enumerate(AUX_COLUMNS):
                if name in index:
                    fresh.select(0, new_i).copy_(old.select(0, index[name]))
                elif key == "aux_head.bias":
                    skipped.append(f"aux/{name}")
            loaded[key] = fresh
        else:
            skipped.append(key)
    model.load_state_dict(loaded, strict=False)
    return sorted(set(skipped))


def load_checkpoint(path, device: str = "cpu") -> tuple[StudentPolicy, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = StudentPolicy(StudentConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model"])
    return model.eval(), payload
