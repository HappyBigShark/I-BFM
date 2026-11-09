#!/usr/bin/env bash
# Watch the vision student on one round-2 scene (box in view at the start, random size).
#   CHECKPOINT=checkpoints/round2_step_0150000.pt SEED=1000 bash play_student.sh
# Options (environment):
#   MODE=student|teacher             whose action is executed (default student)
#   CONTROLLER_SOURCE=estimate|truth what the phase controller reads about the box (default estimate)
#   HEADLESS=1 MAX_STEPS=1500        no viewer; prints the outcome
#   HEADLESS=1 VIDEO=1 MAX_STEPS=1500  no viewer, record <output>/session.mp4 (with the vision panel)
#   SHOW_VISION=0                    hide the student's depth/mask panel (shown bottom-right, also in the video)
# The viewer runs until the window is closed; do not press R (one episode per process).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ibfm="$(cd "${here}/../../.." && pwd)"
python="${ibfm}/.venv/bin/python"
checkpoint="$(realpath "${CHECKPOINT:-${here}/checkpoints/round2_step_0150000.pt}")"
seed="${SEED:-1000}"
out="${OUTPUT_DIR:-${here}/runs/play_$(date +%Y%m%d_%H%M%S)_seed${seed}}"
mkdir -p "${out}"

# First seed >= SEED whose scene starts with the box in view, as in training.
scene="$(cd "${here}" && PYTHONPATH="${ibfm}" "${python}" - "${out}/scenes" "${seed}" <<'EOF'
import argparse, sys
from pathlib import Path
from run_collectors import prepare_visible_scenes, scene_settings
args = argparse.Namespace(box_size_min=0.25, box_size_max=0.40, distance_min=1.0, distance_max=2.5, heading_error_max_deg=25.0)
print(prepare_visible_scenes(Path(sys.argv[1]), int(sys.argv[2]), 1, lambda seed: scene_settings(seed, args))[0])
EOF
)"
heights="$(cd "${here}" && PYTHONPATH="${ibfm}" "${python}" -c 'import sys; from pathlib import Path; from run_collectors import height_window; w = height_window(Path(sys.argv[1])); print(w["HEIGHT_MIN"], w["HEIGHT_MAX"])' "${scene}")"
read -r height_min height_max <<< "${heights}"
scene_seed="${scene##*seed_}"
box_size="$("${python}" -c 'import json, sys; print(" x ".join(f"{v:.2f}" for v in json.load(open(sys.argv[1]))["box_size_m"]))' "${scene}/scene.json")"
echo "Scene:      ${scene} (box ${box_size} m, start height ${height_min}-${height_max} m)"
echo "Student:    ${checkpoint}"
echo "Mode:       ${MODE:-student}, controller reads ${CONTROLLER_SOURCE:-estimate}"

PREPARE_SCENE=0 SCENE_DIR="${scene}" OUTPUT_DIR="${out}" \
HEIGHT_MIN="${height_min}" HEIGHT_MAX="${height_max}" HEIGHT_SEED="${scene_seed}" \
SIM2SIM_ENTRY="${here}/collect_sim2sim.py" \
DISTILL_OUTPUT_DIR="${out}/episode" DISTILL_SAVE_EPISODE=0 DISTILL_OUTCOMES="${out}/outcome.jsonl" \
DISTILL_MODE="${MODE:-student}" DISTILL_CONTROLLER_SOURCE="${CONTROLLER_SOURCE:-estimate}" \
DISTILL_STUDENT_CHECKPOINT="${checkpoint}" DISTILL_SEED="${scene_seed}" \
DISTILL_SHOW_VISION="${SHOW_VISION:-1}" \
DISTILL_META="{\"episode_id\": \"play_${scene_seed}\", \"seed\": ${scene_seed}}" \
  bash "${here}/../run_interactive.sh"

echo "Outcome:    $(cat "${out}/outcome.jsonl" 2>/dev/null || echo 'none written')"
if [[ -f "${out}/session.mp4" ]]; then echo "Video:      ${out}/session.mp4"; fi
