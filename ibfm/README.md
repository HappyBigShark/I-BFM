# I-BFM training and interaction source

This directory contains the H200 training implementation used for the selected 166.560M-step checkpoint and the runtime code for Carry Box, Push Box, and Kick Box.

The selected checkpoint stores its training arguments in `../checkpoints/ibfm-166m/config.json`. Start a fresh H200 run with `./scripts/train_ibfm_h200.sh ibfm_reproduction` from this directory. The launcher retains the latest and lowest-EMD policies and saves replay data once at normal completion.

For installation, interaction commands, and verification, see the [repository README](../README.md). Task-specific settings are documented alongside each interactive launcher under `tasks/`.

## Simulation RGB-D camera

Capture a body-mounted RGB-D test frame without loading a policy checkpoint:

```bash
MUJOCO_GL=egl .venv/bin/python -m humanoidverse.tools.capture_rgbd
```

Outputs include RGB, metric depth, a visible-box mask, camera calibration, and a timestamp.
This is a sensor interface and static smoke test; existing policies still consume state observations.
See [仿真 RGB-D 接口说明](docs/rgbd_simulation.md) for configuration and integration details.
