#!/usr/bin/env python3
"""DDP learner for the Carry Box vision student (launch with torchrun).

Each rank claims new episode files from ``<run>/incoming`` (atomic rename, so
every file is used by exactly one rank), keeps a FIFO replay in RAM, and
trains on teacher-labelled sequences: action MSE after a recurrent burn-in,
plus the auxiliary box-state loss.  A background thread ingests episodes and
assembles batches so the GPU never waits for NumPy gathers.  Rank 0 publishes
``ckpt/latest.pt`` and ``status.json`` that the collectors poll:

* ``beta``  - probability that an episode is teacher-executed (DAgger mixing);
* ``gamma`` - probability that the phase controller reads the student's box
  estimate instead of simulator truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from student import LOW_DIM, StudentConfig, StudentPolicy, aux_loss, save_checkpoint, warm_start
from vision_sensor import SENTINEL, VisionConfig


TICK_KEYS = ("low", "z", "teacher", "aux")


class Replay:
    def __init__(self, capacity_ticks: int, sequence: int):
        self.capacity_ticks, self.sequence = capacity_ticks, sequence
        self.episodes: deque[dict[str, np.ndarray]] = deque()
        self.ticks = 0
        self.ingested = 0

    def add(self, episode: dict[str, np.ndarray]) -> None:
        if len(episode["teacher"]) < self.sequence:
            return
        self.episodes.append(episode)
        self.ticks += len(episode["teacher"])
        self.ingested += 1
        while self.ticks > self.capacity_ticks and len(self.episodes) > 1:
            self.ticks -= len(self.episodes.popleft()["teacher"])

    def sample(self, batch: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        episodes = list(self.episodes)
        weights = np.asarray([len(item["teacher"]) - self.sequence + 1 for item in episodes], dtype=np.float64)
        chosen = rng.choice(len(episodes), size=batch, p=weights / weights.sum())
        out: dict[str, list[np.ndarray]] = {key: [] for key in ("depth_mm", "mask", "known", *TICK_KEYS)}
        for index in chosen:
            episode = episodes[index]
            start = int(rng.integers(len(episode["teacher"]) - self.sequence + 1))
            window = slice(start, start + self.sequence)
            out["depth_mm"].append(episode["depth_mm"][episode["depth_index"][window]])
            out["mask"].append(episode["mask"][episode["mask_index"][window]])
            out["known"].append(episode["known"])
            for key in TICK_KEYS:
                out[key].append(episode[key][window])
        return {key: np.stack(values) for key, values in out.items()}


def claim_episodes(run_dir: Path, rank: int, limit: int) -> list[dict[str, np.ndarray]]:
    incoming = run_dir / "incoming"
    claimed = run_dir / "claimed"
    claimed.mkdir(exist_ok=True)
    episodes = []
    for path in sorted(incoming.glob("ep_*.npz"))[: limit * 4]:
        target = claimed / f"rank{rank}_{path.name}"
        try:
            os.rename(path, target)
        except OSError:
            continue  # another rank took it
        try:
            with np.load(target) as data:
                episode = {key: data[key] for key in ("depth_mm", "mask", "known", "depth_index", "mask_index", *TICK_KEYS)}
            if episode["low"].shape[-1] != LOW_DIM:
                raise ValueError(f"low-dim width {episode['low'].shape[-1]} != {LOW_DIM}")
            episodes.append(episode)
        except Exception as error:  # a corrupt or stale-format file must not stop training
            print(f"[learner {rank}] skip {target.name}: {error}", flush=True)
        target.unlink(missing_ok=True)
        if len(episodes) >= limit:
            break
    return episodes


class BatchProducer(threading.Thread):
    """Owns the replay after start(): ingests episodes and queues ready batches."""

    def __init__(self, replay: Replay, run_dir: Path, rank: int, args: argparse.Namespace, rng: np.random.Generator):
        super().__init__(daemon=True)
        self.replay, self.run_dir, self.rank, self.args, self.rng = replay, run_dir, rank, args, rng
        self.batches: queue.Queue = queue.Queue(maxsize=args.prefetch)
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            while True:
                for episode in claim_episodes(self.run_dir, self.rank, limit=self.args.ingest_per_batch):
                    self.replay.add(episode)
                self.batches.put(pin_batch(self.replay.sample(self.args.batch, self.rng)))
        except BaseException as error:  # surface in the training loop
            self.error = error
            raise

    def get(self) -> dict[str, np.ndarray]:
        while True:
            try:
                return self.batches.get(timeout=60)
            except queue.Empty:
                if self.error is not None:
                    raise RuntimeError("batch producer failed") from self.error


def pin_batch(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Producer side: page-locked tensors so the training loop only issues async copies.

    Depth travels as int16 (millimetres; the D435 model never exceeds 6 m).
    """

    pinned = {
        "depth_mm": torch.from_numpy(batch["depth_mm"].view(np.int16)),
        "mask": torch.from_numpy(batch["mask"]),
        "known": torch.from_numpy(batch["known"]),
        **{key: torch.from_numpy(np.ascontiguousarray(batch[key], dtype=np.float32)) for key in TICK_KEYS},
    }
    return {key: value.pin_memory() for key, value in pinned.items()}


def encode_batch(batch: dict[str, torch.Tensor], vision: VisionConfig, device: torch.device):
    """Torch mirror of vision_sensor.encode_frame on [B, T, H, W] batches (pinned host tensors)."""

    depth_mm = batch["depth_mm"].to(device, non_blocking=True).to(torch.int32)
    mask = batch["mask"].to(device, non_blocking=True).float()
    known = batch["known"].to(device, non_blocking=True).float()[:, None].expand_as(mask)
    depth_m = depth_mm.float() / 1000.0
    valid = depth_mm > 0
    far = valid & (depth_m > vision.depth_max_m)
    normalised = torch.where(
        valid,
        ((depth_m - vision.depth_min_m) / (vision.depth_max_m - vision.depth_min_m)).clamp(0.0, 1.0),
        torch.full_like(depth_m, SENTINEL),
    )
    mask_image = torch.stack((mask, known), dim=2)
    depth_image = torch.stack((normalised, valid.float(), far.float(), mask), dim=2)
    tensors = {key: batch[key].to(device, non_blocking=True) for key in TICK_KEYS}
    return mask_image, depth_image, tensors


def recent_outcomes(run_dir: Path, window: int = 200) -> dict[str, float | int | None]:
    path = run_dir / "outcomes.jsonl"
    if not path.is_file():
        return {}
    with open(path, "rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 1200 * window * 4))
        lines = stream.read().decode(errors="ignore").splitlines()[1:]
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    summary: dict[str, float | int | None] = {}
    for mode in ("teacher", "student"):
        for source in ("truth", "estimate"):
            items = [row for row in rows if row.get("mode") == mode and row.get("controller_source", "truth") == source][-window:]
            prefix = f"{mode}_{source}"
            summary[f"{prefix}_episodes"] = len(items)
            if items:
                summary[f"{prefix}_placement"] = float(np.mean([row["placement_success"] for row in items]))
                summary[f"{prefix}_pickup"] = float(np.mean([row["true_pickup"] for row in items]))
                summary[f"{prefix}_fallen"] = float(np.mean([row["fallen"] for row in items]))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, help="warm-start every shape-compatible parameter")
    parser.add_argument("--batch", type=int, default=96, help="sequences per rank")
    parser.add_argument("--sequence", type=int, default=48)
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--replay-ticks", type=int, default=1_500_000, help="per rank")
    parser.add_argument("--min-replay-ticks", type=int, default=60_000, help="per rank before training")
    parser.add_argument("--ingest-per-batch", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--archive-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--beta-warmup-steps", type=int, default=3000, help="teacher-only collection while the student is untrained")
    parser.add_argument("--beta-decay-steps", type=int, default=40_000)
    parser.add_argument("--beta-min", type=float, default=0.2)
    parser.add_argument("--beta-start", type=float, default=1.0, help="beta right after warm-up; decays linearly to --beta-min")
    parser.add_argument("--gamma-start-steps", type=int, default=None, help="default: --beta-warmup-steps")
    parser.add_argument("--gamma-ramp-steps", type=int, default=0, help="0 keeps the controller on truth")
    parser.add_argument("--gamma-max", type=float, default=1.0)
    args = parser.parse_args()
    if args.gamma_start_steps is None:
        args.gamma_start_steps = args.beta_warmup_steps
    return args


def beta_at(step: int, args: argparse.Namespace) -> float:
    if step < args.beta_warmup_steps:
        return 1.0
    progress = min(1.0, (step - args.beta_warmup_steps) / args.beta_decay_steps)
    return args.beta_start + (args.beta_min - args.beta_start) * progress


def gamma_at(step: int, args: argparse.Namespace) -> float:
    if args.gamma_ramp_steps <= 0 or step < args.gamma_start_steps:
        return 0.0
    return min(args.gamma_max, (step - args.gamma_start_steps) / args.gamma_ramp_steps * args.gamma_max)


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    run_dir = args.run_dir.resolve()
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234 + rank)
    torch.manual_seed(1234)
    vision = VisionConfig()
    replay = Replay(args.replay_ticks, args.sequence)
    model = StudentPolicy(StudentConfig(image_height=vision.height, image_width=vision.width)).to(device)
    if args.init_checkpoint is not None:
        skipped = warm_start(model, args.init_checkpoint)
        if rank == 0:
            print(f"[learner] warm start from {args.init_checkpoint}; re-initialised: {skipped}", flush=True)
            if args.beta_warmup_steps == 0:
                # Collect with the warm-started student from the first episode, so the
                # replay never holds teacher-only data (which made a warm start collapse).
                latest = run_dir / "ckpt" / "latest.pt"
                save_checkpoint(latest, model, step=0, extra={"vision": vision.__dict__})
                _write_status(run_dir, step=0, beta=beta_at(0, args), gamma=gamma_at(0, args), checkpoint=str(latest),
                              extra={"phase": "warm_start_published"})

    # Wait for data, then fix the low-dim normaliser from the pooled replay.
    while True:
        for episode in claim_episodes(run_dir, rank, limit=64):
            replay.add(episode)
        ready = torch.tensor([replay.ticks >= args.min_replay_ticks], device=device, dtype=torch.int32)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
        if ready.item():
            break
        if rank == 0:
            print(f"[learner] waiting for data: rank0 replay {replay.ticks} ticks", flush=True)
            if not (args.init_checkpoint is not None and args.beta_warmup_steps == 0):
                _write_status(run_dir, step=0, beta=1.0, gamma=0.0, checkpoint=None, extra={"phase": "waiting_for_data"})
        time.sleep(20)
    low = np.concatenate([episode["low"] for episode in replay.episodes]).astype(np.float64)
    moments = torch.tensor(np.stack((low.mean(0), (low**2).mean(0))), device=device)
    dist.all_reduce(moments)
    moments /= world
    mean = moments[0]
    std = (moments[1] - mean**2).clamp_min(0.0).sqrt().clamp_min(1e-2)
    model.low_mean.copy_(mean.float())
    model.low_std.copy_(std.float())

    producer = BatchProducer(replay, run_dir, rank, args, rng)
    producer.start()
    ddp = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = 1000
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / args.max_steps, 1.0)))),
    )
    started, last_log = time.time(), time.time()
    running: dict[str, float] = {}
    for step in range(1, args.max_steps + 1):
        mask_image, depth_image, tensors = encode_batch(producer.get(), vision, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action, aux, _ = ddp(mask_image, depth_image, tensors["low"], tensors["z"])
        action, aux = action.float()[:, args.burn_in :], aux.float()[:, args.burn_in :]
        action_loss = torch.nn.functional.mse_loss(action, tensors["teacher"][:, args.burn_in :])
        auxiliary = aux_loss(aux, tensors["aux"][:, args.burn_in :])
        loss = action_loss + args.aux_weight * auxiliary
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        for key, value in (("action_mse", action_loss), ("aux", auxiliary), ("grad_norm", grad_norm)):
            running[key] = running.get(key, 0.0) + float(value)

        if step % args.save_every == 0 and rank == 0:
            published = step >= args.beta_warmup_steps
            latest = run_dir / "ckpt" / "latest.pt"
            save_checkpoint(latest, model, step=step, extra={"vision": vision.__dict__})
            if step % args.archive_every == 0:
                save_checkpoint(run_dir / "ckpt" / f"step_{step:07d}.pt", model, step=step)
            _write_status(
                run_dir,
                step=step,
                beta=beta_at(step, args),
                gamma=gamma_at(step, args),
                checkpoint=str(latest) if published else None,
            )
        if step % args.log_every == 0:
            stats = torch.tensor([replay.ticks, replay.ingested, producer.batches.qsize()], device=device, dtype=torch.float64)
            dist.all_reduce(stats)
            if rank == 0:
                now = time.time()
                record = {
                    "step": step,
                    "time_h": (now - started) / 3600,
                    "steps_per_s": args.log_every / (now - last_log),
                    "lr": scheduler.get_last_lr()[0],
                    "beta": beta_at(step, args),
                    "gamma": gamma_at(step, args),
                    "replay_ticks": int(stats[0]),
                    "episodes_ingested": int(stats[1]),
                    "prefetch_queue": float(stats[2]) / world,
                    **{key: value / args.log_every for key, value in running.items()},
                    **recent_outcomes(run_dir),
                }
                last_log = now
                print("[learner] " + json.dumps(record), flush=True)
                with open(run_dir / "metrics.jsonl", "a") as stream:
                    stream.write(json.dumps(record) + "\n")
            running = {}
    dist.destroy_process_group()


def _write_status(run_dir: Path, *, step: int, beta: float, gamma: float, checkpoint: str | None, extra: dict | None = None) -> None:
    payload = {"step": step, "beta": beta, "gamma": gamma, "checkpoint": checkpoint, "time": time.time(), **(extra or {})}
    temporary = run_dir / "status.json.tmp"
    temporary.write_text(json.dumps(payload) + "\n")
    os.replace(temporary, run_dir / "status.json")


if __name__ == "__main__":
    main()
