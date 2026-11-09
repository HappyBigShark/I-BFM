#!/usr/bin/env python3
"""Step-0 perception robustness diagnostic around the frozen z-aware Carry stack.

Assembles exactly the controller stack of ``z_aware_height_banded_sim2sim.py``
and then installs the perception-error overlay from ``perception_noise.py``.
Without ``UFO_PERCEPTION_PROFILE`` the stack runs unmodified.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import perception_noise


HERE = Path(__file__).resolve().parent
Z_AWARE = HERE.parent / "z_aware_height_banded_sim2sim.py"


def _load_z_aware() -> Any:
    spec = importlib.util.spec_from_file_location("_perception_z_aware", Z_AWARE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load z-aware controller: {Z_AWARE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    z_aware = _load_z_aware()
    module = z_aware._load_controller_stack(z_aware._load_rolling_guard())
    z_aware._install_platform_collision_filter(module)
    if z_aware._env_enabled("UFO_Z3D_RANDOM_HEIGHT_RESET"):
        z_aware._install_random_height_reset(module)
    z_aware._install_height_banded_controller(module)
    perception_noise.install_from_env(module)
    module.run(module.parse_args())


if __name__ == "__main__":
    main()
