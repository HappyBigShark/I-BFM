#!/usr/bin/env python3
"""Run DAgger collection episodes in parallel (CPU only) for train_student.py.

Each worker thread loops: pick a scene and seeds, read the learner's
``status.json`` (beta, checkpoint), run one episode as teacher-executed (with
probability beta, or while no checkpoint is published) or student-executed,
and leave the episode file in ``<run>/incoming``.  Create ``<run>/STOP`` to
stop after the current episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
REPO = BUNDLE.parents[1]
PYTHON = REPO / ".venv" / "bin" / "python"
LAUNCHER = BUNDLE / "run_interactive.sh"
SCENE_TOOL = BUNDLE / "tools" / "prepare_random_height_scene.py"
ENTRY = HERE / "collect_sim2sim.py"
sys.path.insert(0, str(REPO))


def prepare_scene(scene_dir: Path, seed: int, **overrides: str) -> None:
    """Generate one scene; ``overrides`` set scene-tool variables (BOX_SIZE, ROBOT_BOX_DISTANCE_MAX, ...)."""

    if (scene_dir / "scene.json").is_file():
        return
    env = dict(
        os.environ,
        **overrides,
        SCENE_OUTPUT_DIR=str(scene_dir),
        SCENE_SEED=str(seed),
        PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}",
    )
    subprocess.run([str(PYTHON), str(SCENE_TOOL)], cwd=REPO, env=env, check=True, stdout=subprocess.DEVNULL)


def scene_settings(seed: int, args: argparse.Namespace) -> dict[str, str]:
    """Per-scene randomisation: box dimensions are unknown on the real robot."""

    rng = np.random.default_rng([seed, 99])
    size = rng.uniform(args.box_size_min, args.box_size_max, 3)
    return {
        "BOX_SIZE": " ".join(f"{value:.3f}" for value in size),
        "ROBOT_BOX_DISTANCE_MIN": str(args.distance_min),
        "ROBOT_BOX_DISTANCE_MAX": str(args.distance_max),
        "ROBOT_HEADING_ERROR_MAX_DEG": str(args.heading_error_max_deg),
    }


def prepare_visible_scenes(root: Path, first_seed: int, count: int, settings) -> list[Path]:
    """Prepare scenes from consecutive seeds until ``count`` start with the box in view.

    ``settings(seed)`` gives the scene-tool overrides.  Each kept scene stores
    the start-height window that keeps the box visible (``visibility.json``).
    """

    from scene_visibility import write_visibility

    def prepare(seed: int) -> Path | None:
        scene = root / f"seed_{seed}"
        prepare_scene(scene, seed, **settings(seed))
        visibility = scene / "visibility.json"
        payload = json.loads(visibility.read_text()) if visibility.is_file() else write_visibility(scene)
        return scene if payload["visible_height_range_m"] is not None else None

    scenes: list[Path] = []
    seed = first_seed
    with ThreadPoolExecutor(max_workers=16) as pool:
        while len(scenes) < count:
            batch = range(seed, seed + max(16, count - len(scenes)))
            scenes.extend(scene for scene in pool.map(prepare, batch) if scene is not None)
            seed = batch.stop
    return scenes[:count]


def height_window(scene: Path) -> dict[str, str]:
    low, high = json.loads((scene / "visibility.json").read_text())["visible_height_range_m"]
    return {"HEIGHT_MIN": f"{low:.4f}", "HEIGHT_MAX": f"{high:.4f}"}


def read_status(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {"beta": 1.0, "checkpoint": None}


def worker(index: int, args: argparse.Namespace, scenes: list[Path], counters: dict, lock: threading.Lock) -> None:
    run_dir = args.run_dir
    rng = np.random.default_rng([args.seed, index])
    scratch = run_dir / "scratch" / f"worker_{index:03d}"
    episode = 0
    while not (run_dir / "STOP").exists():
        if len(list((run_dir / "incoming").glob("ep_*.npz"))) > args.max_backlog:
            time.sleep(10)  # learner is behind or not running
            continue
        status = read_status(run_dir)
        checkpoint = status.get("checkpoint")
        mode = "teacher" if checkpoint is None or rng.random() < float(status.get("beta", 1.0)) else "student"
        source = "estimate" if checkpoint is not None and rng.random() < float(status.get("gamma", 0.0)) else "truth"
        scene = scenes[int(rng.integers(len(scenes)))]
        seed = int(rng.integers(2**31 - 1))
        episode_id = f"w{index:03d}_{episode:06d}_{seed:010d}"
        meta = {"episode_id": episode_id, "scene": scene.name, "seed": seed, "learner_step": status.get("step", 0)}
        key = f"{mode}_{source}"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        env = dict(
            os.environ,
            **height_window(scene),
            CUDA_VISIBLE_DEVICES="",
            HEADLESS="1",
            MAX_STEPS=str(args.max_steps),
            PREPARE_SCENE="0",
            SCENE_DIR=str(scene),
            OUTPUT_DIR=str(scratch / "run"),
            SIM2SIM_ENTRY=str(ENTRY),
            HEIGHT_SEED=str(seed),
            DISTILL_OUTPUT_DIR=str(run_dir / "incoming"),
            DISTILL_MODE=mode,
            DISTILL_CONTROLLER_SOURCE=source,
            DISTILL_STUDENT_CHECKPOINT=checkpoint or "",
            DISTILL_SEED=str(seed),
            DISTILL_META=json.dumps(meta),
        )
        with open(scratch / "log.txt", "w") as log:
            result = subprocess.run(["bash", str(LAUNCHER)], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        with lock:
            counters["episodes"] += 1
            counters[key] = counters.get(key, 0) + 1
            if result.returncode != 0:
                counters["failed"] += 1
                failures = run_dir / "failures"
                failures.mkdir(exist_ok=True)
                shutil.copy(scratch / "log.txt", failures / f"{episode_id}.log")
        episode += 1
    shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--scenes", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--distance-min", type=float, default=1.0)
    parser.add_argument("--distance-max", type=float, default=2.5)
    parser.add_argument("--heading-error-max-deg", type=float, default=25.0)
    parser.add_argument("--box-size-min", type=float, default=0.25)
    parser.add_argument("--box-size-max", type=float, default=0.40)
    parser.add_argument("--max-backlog", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    (args.run_dir / "incoming").mkdir(parents=True, exist_ok=True)
    scenes = prepare_visible_scenes(args.run_dir / "scenes", args.seed, args.scenes, lambda seed: scene_settings(seed, args))
    print(f"[collectors] {len(scenes)} scenes ready, starting {args.workers} workers", flush=True)
    counters = {"episodes": 0, "failed": 0}
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(index, args, scenes, counters, lock), daemon=True) for index in range(args.workers)]
    for thread in threads:
        thread.start()
    started = time.time()
    while any(thread.is_alive() for thread in threads):
        time.sleep(60)
        with lock:
            snapshot = dict(counters)
        print(f"[collectors] {time.time() - started:7.0f}s {json.dumps(snapshot)}", flush=True)


if __name__ == "__main__":
    main()
