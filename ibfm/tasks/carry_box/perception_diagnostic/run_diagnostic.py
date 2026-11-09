#!/usr/bin/env python3
"""Run the step-0 perception robustness matrix and aggregate the outcomes.

Every configuration sees the same episode seeds (scene layout, initial box
height and perception noise stream), so configurations are compared paired.

    ibfm/.venv/bin/python ibfm/tasks/carry_box/perception_diagnostic/run_diagnostic.py \
        --episodes 20 --jobs 16 --output-dir /tmp/step0

Re-running with the same --output-dir skips finished episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
REPO = BUNDLE.parents[1]
PROFILES = HERE / "profiles"
ENTRY = HERE / "perception_diagnostic_sim2sim.py"
LAUNCHER = BUNDLE / "run_interactive.sh"
SCENE_TOOL = BUNDLE / "tools" / "prepare_random_height_scene.py"
PYTHON = REPO / ".venv" / "bin" / "python"

PERTURBATION_PROFILES = (
    "box_pose_small",
    "box_pose_medium",
    "box_pose_large",
    "box_timing",
    "box_visibility",
    "symmetry_initial",
    "symmetry_jump",
    "symmetry_reacquire",
    "contact",
    "base_state",
    "goal",
)
FALLEN_ROOT_HEIGHT_M = 0.5
CHANNEL_SETS = {"actor": ("actor",), "b": ("b",), "controller": ("controller",), "all": ("actor", "b", "controller")}


def build_matrix(profiles: list[str], channel_sets: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = [
        # Unmodified stack, and the instrumented zero-error stack that must match it.
        {"name": "baseline", "profile": None, "channels": None},
        {"name": "identity/all", "profile": "identity", "channels": CHANNEL_SETS["all"]},
    ]
    for profile in profiles:
        for channel_set in channel_sets:
            runs.append({"name": f"{profile}/{channel_set}", "profile": profile, "channels": CHANNEL_SETS[channel_set]})
    runs.append({"name": "combined/all", "profile": "combined", "channels": CHANNEL_SETS["all"]})
    return runs


def prepare_scene(scene_dir: Path, seed: int) -> None:
    if (scene_dir / "scene.json").is_file():
        return
    env = dict(
        os.environ,
        SCENE_OUTPUT_DIR=str(scene_dir),
        SCENE_SEED=str(seed),
        PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}",
    )
    subprocess.run([str(PYTHON), str(SCENE_TOOL)], cwd=REPO, env=env, check=True, stdout=subprocess.DEVNULL)


def episode_dir(output: Path, run: dict[str, Any], seed: int) -> Path:
    return output / "runs" / run["name"].replace("/", "__") / f"seed_{seed}"


def run_episode(output: Path, run: dict[str, Any], seed: int, args: argparse.Namespace) -> Path:
    directory = episode_dir(output, run, seed)
    if (directory / "done.json").is_file():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        HEADLESS="1",
        MAX_STEPS=str(args.max_steps),
        PREPARE_SCENE="0",
        SCENE_DIR=str(output / "scenes" / f"seed_{seed}"),
        OUTPUT_DIR=str(directory),
        SIM2SIM_ENTRY=str(ENTRY),
        HEIGHT_SEED=str(seed),
        UFO_PERCEPTION_SEED=str(seed),
        UFO_PERCEPTION_SUMMARY_PATH=str(directory / "perception_summary.json"),
        UFO_PERCEPTION_PROFILE="" if run["profile"] is None else str(PROFILES / f"{run['profile']}.json"),
        UFO_PERCEPTION_CHANNELS="" if run["channels"] is None else ",".join(run["channels"]),
    )
    with open(directory / "log.txt", "w") as log:
        result = subprocess.run(["bash", str(LAUNCHER)], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not args.keep_traces:
        (directory / "session.csv").unlink(missing_ok=True)
    (directory / "done.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
    return directory


def load_episode(directory: Path, run: dict[str, Any], seed: int) -> dict[str, Any]:
    row: dict[str, Any] = {"config": run["name"], "seed": seed}
    done = json.loads((directory / "done.json").read_text())
    row["returncode"] = done["returncode"]
    trials_path = directory / "height_trials.json"
    if trials_path.is_file():
        trial = json.loads(trials_path.read_text())["trials"][0]
        row.update(
            initial_box_z_m=trial["initial_box_z_m"],
            placement_success=bool(trial["placement_success"]),
            controller_pickup=bool(trial["pickup_success"]),
            final_error_3d_m=trial["final_error_3d_m"],
            final_box_xyz=trial["final_box_xyz_m"],
        )
    summary_path = directory / "perception_summary.json"
    if summary_path.is_file():
        episode = json.loads(summary_path.read_text())["episodes"][0]
        errors = episode["errors"]

        def stat(name: str, key: str) -> float | None:
            value = errors.get(name)
            return None if value is None else value[key]

        row.update(
            true_pickup=bool(episode["true_pickup"]),
            max_true_lift_m=episode["max_true_lift_m"],
            complete_time_s=episode["first_phase_time_s"].get("complete"),
            # The controller's own fall detector reads perceived root height and
            # can fire falsely; the truth check uses the final root height.
            controller_fall=episode["fall_event_count"] > 0,
            fell=episode["final_root_height_m"] < FALLEN_ROOT_HEIGHT_M,
            recovery_count=episode["recovery_count"],
            phase_transition_count=episode["phase_transition_count"],
            action_l2_mean=stat("action_l2", "mean"),
            action_l2_p95=stat("action_l2", "p95"),
            box_pos_err_mean_m=stat("box_pos_m", "mean"),
            box_rot_err_mean_deg=stat("box_rot_deg", "mean"),
            box_age_mean_s=stat("box_age_s", "mean"),
            contact_mismatch_fraction=episode["contact_mismatch_fraction"],
            bilateral_mismatch_fraction=episode["bilateral_mismatch_fraction"],
            symmetry_jumps=episode["perception_stats"]["symmetry_jumps"],
            dropped_measurements=episode["perception_stats"]["dropped"],
        )
    return row


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    z = 1.96
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return (centre - half, centre + half)


def mean(values: list[Any]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def aggregate(rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row)
    baseline = {row["seed"]: row for row in by_config.get("baseline", [])}
    summary = []
    for run in runs:
        items = by_config.get(run["name"], [])
        valid = [row for row in items if "placement_success" in row]
        successes = sum(row["placement_success"] for row in valid)
        low, high = wilson(successes, len(valid))
        lost = sum(
            1
            for row in valid
            if baseline.get(row["seed"], {}).get("placement_success") and not row["placement_success"]
        )
        gained = sum(
            1
            for row in valid
            if row["seed"] in baseline and not baseline[row["seed"]].get("placement_success") and row["placement_success"]
        )
        matches_baseline = sum(
            1
            for row in valid
            if row["seed"] in baseline and row.get("final_box_xyz") == baseline[row["seed"]].get("final_box_xyz")
        )
        summary.append(
            {
                "config": run["name"],
                "episodes": len(valid),
                "failed_runs": len(items) - len(valid) + sum(row["returncode"] != 0 for row in valid),
                "placement_success_rate": successes / len(valid) if valid else None,
                "placement_ci95_low": low,
                "placement_ci95_high": high,
                "lost_vs_baseline": lost,
                "gained_vs_baseline": gained,
                "identical_to_baseline": matches_baseline,
                "true_pickup_rate": mean([row.get("true_pickup") for row in valid]),
                "controller_pickup_rate": mean([row.get("controller_pickup") for row in valid]),
                "mean_final_error_3d_m": mean([row.get("final_error_3d_m") for row in valid]),
                "fall_rate": mean([row.get("fell") for row in valid]),
                "controller_fall_rate": mean([row.get("controller_fall") for row in valid]),
                "mean_complete_time_s": mean([row.get("complete_time_s") for row in valid]),
                "mean_recoveries": mean([row.get("recovery_count") for row in valid]),
                "action_l2_mean": mean([row.get("action_l2_mean") for row in valid]),
                "action_l2_p95": mean([row.get("action_l2_p95") for row in valid]),
                "box_pos_err_mean_m": mean([row.get("box_pos_err_mean_m") for row in valid]),
                "box_rot_err_mean_deg": mean([row.get("box_rot_err_mean_deg") for row in valid]),
                "bilateral_mismatch_fraction": mean([row.get("bilateral_mismatch_fraction") for row in valid]),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(key for key in row if key not in keys)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, summary: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Step-0 perception robustness diagnostic",
        "",
        f"episodes/config={args.episodes}, seeds {args.seed}..{args.seed + args.episodes - 1}, "
        f"max_steps={args.max_steps} ({args.max_steps * 0.02:.0f} s). "
        "Placement success: final box centre within 0.20 m (3D) of the target. "
        "Lost/gained: paired episodes that flip relative to the baseline. "
        "Action L2: per-tick ||a(perceived) - a(truth)|| at the same latent. "
        f"Fallen (truth): final root height < {FALLEN_ROOT_HEIGHT_M} m; (controller): the controller's own fall detector fired.",
        "",
        "| config | n | placement | 95% CI | lost | gained | true pickup | fallen (truth / controller) | final err m | complete s | action L2 mean / p95 | box pos err m | bilateral mismatch |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in summary:
        lines.append(
            "| {config} | {n} | {rate} | {low}-{high} | {lost} | {gained} | {pickup} | {falls} / {controller_falls} | {err} | {time} | {l2} / {l2p} | {pos} | {mismatch} |".format(
                config=item["config"],
                n=item["episodes"],
                rate=_fmt(item["placement_success_rate"]),
                low=_fmt(item["placement_ci95_low"]),
                high=_fmt(item["placement_ci95_high"]),
                lost=item["lost_vs_baseline"],
                gained=item["gained_vs_baseline"],
                pickup=_fmt(item["true_pickup_rate"]),
                falls=_fmt(item["fall_rate"]),
                controller_falls=_fmt(item["controller_fall_rate"]),
                err=_fmt(item["mean_final_error_3d_m"], 3),
                time=_fmt(item["mean_complete_time_s"], 1),
                l2=_fmt(item["action_l2_mean"], 3),
                l2p=_fmt(item["action_l2_p95"], 3),
                pos=_fmt(item["box_pos_err_mean_m"], 3),
                mismatch=_fmt(item["bilateral_mismatch_fraction"], 3),
            )
        )
    identity = next((item for item in summary if item["config"] == "identity/all"), None)
    if identity is not None:
        lines += [
            "",
            f"Validation: identity/all reproduced the baseline final box position on "
            f"{identity['identical_to_baseline']}/{identity['episodes']} episodes (expected: all).",
        ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--profiles", default=",".join(PERTURBATION_PROFILES))
    parser.add_argument("--channel-sets", default=",".join(CHANNEL_SETS))
    parser.add_argument("--only", default="", help="Comma-separated config names to run (default: whole matrix)")
    parser.add_argument("--keep-traces", action="store_true", help="Keep the per-step session.csv of every episode")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    profiles = [item for item in args.profiles.split(",") if item]
    channel_sets = [item for item in args.channel_sets.split(",") if item]
    for profile in profiles + ["identity", "combined"]:
        if not (PROFILES / f"{profile}.json").is_file():
            parser.error(f"unknown profile: {profile}")
    for channel_set in channel_sets:
        if channel_set not in CHANNEL_SETS:
            parser.error(f"unknown channel set: {channel_set}")
    args.matrix = build_matrix(profiles, channel_sets)
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {run["name"] for run in args.matrix}
        if unknown:
            parser.error(f"unknown --only configs: {sorted(unknown)}")
        args.matrix = [run for run in args.matrix if run["name"] in wanted]
    return args


def main() -> None:
    args = parse_args()
    output: Path = args.output_dir.resolve()
    seeds = list(range(args.seed, args.seed + args.episodes))
    jobs = [(run, seed) for run in args.matrix for seed in seeds]
    if not args.aggregate_only:
        for seed in seeds:
            prepare_scene(output / "scenes" / f"seed_{seed}", seed)
        print(f"[step0] {len(args.matrix)} configs x {len(seeds)} episodes = {len(jobs)} runs, {args.jobs} parallel", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_episode, output, run, seed, args): (run, seed) for run, seed in jobs}
            for finished, future in enumerate(as_completed(futures), start=1):
                run, seed = futures[future]
                future.result()
                if finished % 10 == 0 or finished == len(jobs):
                    print(f"[step0] {finished}/{len(jobs)} done (last: {run['name']} seed {seed})", flush=True)
    rows = [
        load_episode(episode_dir(output, run, seed), run, seed)
        for run, seed in jobs
        if (episode_dir(output, run, seed) / "done.json").is_file()
    ]
    summary = aggregate(rows, args.matrix)
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_markdown(output / "summary.md", summary, args)
    print((output / "summary.md").read_text())
    failed = [row for row in rows if row["returncode"] != 0]
    if failed:
        print(f"[step0] {len(failed)} runs exited non-zero; see their log.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
