#!/usr/bin/env python3
"""Paired fixed-seed evaluation of the teacher and student checkpoints (CPU only).

Every mode runs the same scenes, box heights and perception noise seeds, so
success rates compare episode by episode.

    ibfm/.venv/bin/python ibfm/tasks/carry_box/vision_distillation/evaluate.py \
        --output-dir runs/eval_x --checkpoint runs/distill_x/ckpt/step_0300000.pt --episodes 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_collectors import ENTRY, LAUNCHER, REPO, height_window, prepare_scene, prepare_visible_scenes, scene_settings


def run_episode(args: argparse.Namespace, mode: str, scene: Path, outcomes: Path) -> None:
    seed = int(scene.name.removeprefix("seed_"))
    episode_id = f"{mode}_{seed}"
    scratch = args.output_dir / "scratch" / episode_id
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    env = dict(
        os.environ,
        **(height_window(scene) if args.visible_start else {}),
        CUDA_VISIBLE_DEVICES="",
        HEADLESS="1",
        MAX_STEPS=str(args.max_steps),
        PREPARE_SCENE="0",
        SCENE_DIR=str(scene),
        OUTPUT_DIR=str(scratch / "run"),
        SIM2SIM_ENTRY=str(ENTRY),
        HEIGHT_SEED=str(seed),
        DISTILL_OUTPUT_DIR=str(scratch / "episode"),
        DISTILL_OUTCOMES=str(outcomes),
        DISTILL_SAVE_EPISODE="0",
        DISTILL_MODE=mode,
        DISTILL_STUDENT_CHECKPOINT=str(args.checkpoint) if mode == "student" or args.controller_source == "estimate" else "",
        DISTILL_CONTROLLER_SOURCE=args.controller_source,
        DISTILL_SEED=str(seed),
        DISTILL_META=json.dumps({"episode_id": episode_id, "seed": seed}),
        DISTILL_COMPLETE_HOLD_S=str(args.complete_hold_s),
    )
    with open(scratch / "log.txt", "w") as log:
        result = subprocess.run(["bash", str(LAUNCHER)], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        (args.output_dir / "failures").mkdir(exist_ok=True)
        shutil.copy(scratch / "log.txt", args.output_dir / "failures" / f"{episode_id}.log")
    shutil.rmtree(scratch, ignore_errors=True)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return math.nan, math.nan
    z, p = 1.96, successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return centre - half, centre + half


def summarise(outcomes: Path, modes: list[str]) -> str:
    rows = [json.loads(line) for line in outcomes.read_text().splitlines() if line.strip()]
    by_mode = {mode: {row["seed"]: row for row in rows if row["mode"] == mode} for mode in modes}
    lines = ["| mode | n | placement | 95% CI | true pickup | fallen | mean final error m |", "|---|---|---|---|---|---|---|"]
    for mode in modes:
        items = list(by_mode[mode].values())
        if not items:
            continue
        successes = sum(row["placement_success"] for row in items)
        low, high = wilson(successes, len(items))
        lines.append(
            f"| {mode} | {len(items)} | {successes / len(items):.3f} | {low:.2f}-{high:.2f} | "
            f"{sum(row['true_pickup'] for row in items) / len(items):.3f} | "
            f"{sum(row['fallen'] for row in items) / len(items):.3f} | "
            f"{sum(row['final_error_3d_m'] for row in items) / len(items):.3f} |"
        )
    if len(modes) == 2:
        first, second = (by_mode[mode] for mode in modes)
        common = sorted(set(first) & set(second))
        lost = sum(first[seed]["placement_success"] and not second[seed]["placement_success"] for seed in common)
        gained = sum(not first[seed]["placement_success"] and second[seed]["placement_success"] for seed in common)
        lines.append(f"\nPaired over {len(common)} seeds: {modes[1]} lost {lost}, gained {gained} relative to {modes[0]}.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--modes", default="teacher,student")
    parser.add_argument("--controller-source", default="truth", choices=("truth", "estimate"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--distance-min", type=float, default=1.5)
    parser.add_argument("--distance-max", type=float, default=2.0)
    parser.add_argument("--box-size", default="0.30 0.30 0.30", help="full box dimensions (m), or 'random'")
    parser.add_argument("--box-size-min", type=float, default=0.25, help="per-axis range for --box-size random")
    parser.add_argument("--box-size-max", type=float, default=0.40)
    parser.add_argument("--visible-start", action="store_true", help="round-2 distribution: box in view at the start")
    parser.add_argument("--heading-error-max-deg", type=float, default=25.0, help="with --visible-start")
    parser.add_argument(
        "--complete-hold-s", type=float, default=1000.0,
        help="stop once the controller has held 'complete' this long (default: full horizon, as in step 0)",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    modes = [mode for mode in args.modes.split(",") if mode]
    if "student" in modes and args.checkpoint is None:
        parser.error("--checkpoint is required for student evaluation")
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    def settings(seed: int) -> dict[str, str]:
        values = scene_settings(seed, args)
        if args.box_size != "random":
            values["BOX_SIZE"] = args.box_size
        if not args.visible_start:
            values.pop("ROBOT_HEADING_ERROR_MAX_DEG")
        return values

    scene_root = args.output_dir / "scenes"
    if args.visible_start:
        scenes = prepare_visible_scenes(scene_root, args.seed, args.episodes, settings)
    else:
        scenes = [scene_root / f"seed_{seed}" for seed in range(args.seed, args.seed + args.episodes)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda scene: prepare_scene(scene, int(scene.name[5:]), **settings(int(scene.name[5:]))), scenes))
    outcomes = args.output_dir / "outcomes.jsonl"
    done = {(row["mode"], row["seed"]) for row in map(json.loads, outcomes.read_text().splitlines())} if outcomes.is_file() else set()
    jobs = [(mode, scene) for mode in modes for scene in scenes if (mode, int(scene.name[5:])) not in done]
    print(f"[evaluate] {len(jobs)} episodes, {args.workers} parallel", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda job: run_episode(args, job[0], job[1], outcomes), jobs))
    report = summarise(outcomes, modes)
    (args.output_dir / "summary.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
