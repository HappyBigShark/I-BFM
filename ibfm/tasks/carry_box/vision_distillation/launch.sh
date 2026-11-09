#!/usr/bin/env bash
# Launch the Carry Box vision distillation: DDP learner on GPUs + CPU DAgger collectors.
#   GPUS=0,1,2,3 WORKERS=160 bash launch.sh [extra train_student.py args]
# Stop: touch <run>/STOP (collectors finish their episodes), then kill the learner
# process group: kill -- -$(cat <run>/learner.pid)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="$(cd "${here}/../../.." && pwd)/.venv/bin/python"
run_dir="${RUN_DIR:-${here}/runs/distill_$(date +%Y%m%d_%H%M%S)}"
gpus="${GPUS:-0,1,2,3}"
mkdir -p "${run_dir}"
cd "${here}"

CUDA_VISIBLE_DEVICES="${gpus}" OMP_NUM_THREADS=4 setsid nohup "${python}" -m torch.distributed.run \
  --standalone --nproc_per_node "$(tr ',' '\n' <<< "${gpus}" | wc -l)" \
  train_student.py --run-dir "${run_dir}" "$@" > "${run_dir}/learner.log" 2>&1 < /dev/null &
echo $! > "${run_dir}/learner.pid"

setsid nohup "${python}" run_collectors.py --run-dir "${run_dir}" --workers "${WORKERS:-160}" \
  > "${run_dir}/collectors.log" 2>&1 < /dev/null &
echo $! > "${run_dir}/collectors.pid"

echo "${run_dir}"
