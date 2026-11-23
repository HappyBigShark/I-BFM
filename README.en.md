# I-BFM

One policy for carrying, pushing, and kicking a box. This repository is the training and interaction code. Model weights are not included.

[中文](README.md)

## Run

Linux, Python 3.10, `uv`, and MuJoCo.

```bash
git clone https://github.com/HappyBigShark/I-BFM.git
cd I-BFM/ibfm
uv sync
```

From `ibfm/`:

```bash
./tasks/carry_box/run_interactive.sh
./tasks/push_box/run_interactive.sh
./tasks/kick_box/run_interactive.sh
```

All three tasks use the same 166.560M-step checkpoint.

## Train

Eight H200 GPUs:

```bash
./scripts/train_ibfm_h200.sh ibfm_reproduction
```

Use `RESUME=1` after an interruption. Without a replay buffer from that same step, the continued run will not match an uninterrupted one.

## Code

| Path | What |
| --- | --- |
| `ibfm/tasks/carry_box/` | Carry |
| `ibfm/tasks/push_box/` | Push |
| `ibfm/tasks/kick_box/` | Kick |
| `ibfm/tasks/carry_box/vision_distillation/` | Onboard-camera carry student. Weights are not included |
| `checkpoints/ibfm-166m/` | Checkpoint notes. The weight files are not in this repository |

License: [CC BY-NC 4.0](LICENSE).
