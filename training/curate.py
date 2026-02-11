"""Curate a pool of seeds that produce levels the trained agent can solve.

Generates levels via the Bun TS generators (guaranteeing seed→level identity
with the frontend), then evaluates each with the trained PPO model.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

import numpy as np


def find_best_checkpoint(checkpoint_dir: str) -> str | None:
    pattern = os.path.join(checkpoint_dir, "boxworld_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None

    def steps(path):
        m = re.search(r"boxworld_(\d+)_steps\.zip", path)
        return int(m.group(1)) if m else 0

    return max(files, key=steps)


def generate_levels_via_bun(start: int, count: int, batch_size: int = 2000) -> list[dict]:
    """Call the Bun TS script to generate levels as JSONL."""
    script = os.path.join(os.path.dirname(__file__), "..", "visualize", "src", "generate-levels.ts")
    levels = []
    for offset in range(0, count, batch_size):
        n = min(batch_size, count - offset)
        result = subprocess.run(
            ["bun", "run", script, str(start + offset), str(n)],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().split("\n"):
            if line:
                levels.append(json.loads(line))
    return levels


def _run_episode(model, level_data: dict, deterministic: bool) -> bool:
    """Run one episode, return True if goal reached."""
    from environment import BoxworldEnv

    env = BoxworldEnv()
    env._grid = [list(row) for row in level_data["grid"]]
    env._agent_pos = list(level_data["agentStart"])
    env._has_key = False
    env._steps = 0
    env._solve_subgoals()
    obs = env._get_obs()

    for _ in range(BoxworldEnv.MAX_STEPS):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        if terminated:
            return reward > 0
        if truncated:
            return False
    return False


def evaluate_level(model, level_data: dict, max_tries: int = 3) -> bool:
    """Level must be solved by deterministic argmax AND at least once stochastically."""
    if not _run_episode(model, level_data, deterministic=True):
        return False
    for _ in range(max_tries):
        if _run_episode(model, level_data, deterministic=False):
            return True
    return False


def curate_seeds(
    checkpoint_path: str,
    db_path: str = "../data/db.sqlite",
    start_seed: int = 0,
    num_candidates: int = 10000,
    max_tries: int = 3,
) -> list[int]:
    import sqlite3

    from stable_baselines3 import PPO

    model = PPO.load(checkpoint_path)
    print(f"Loaded checkpoint: {checkpoint_path}")

    levels = generate_levels_via_bun(start_seed, num_candidates)
    print(f"Generated {len(levels)} candidate levels via Bun")

    passing_seeds: list[int] = []
    for i, level_data in enumerate(levels):
        if evaluate_level(model, level_data, max_tries):
            passing_seeds.append(level_data["seed"])
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{len(levels)}, passing: {len(passing_seeds)}")

    print(
        f"Curated {len(passing_seeds)}/{len(levels)} seeds ({len(passing_seeds) / len(levels):.0%})"
    )

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS curated_seeds (seed INTEGER PRIMARY KEY)")
    conn.execute("DELETE FROM curated_seeds")
    conn.executemany(
        "INSERT INTO curated_seeds (seed) VALUES (?)",
        [(s,) for s in passing_seeds],
    )
    conn.commit()
    conn.close()
    print(f"Wrote {len(passing_seeds)} seeds to {db_path}")

    return passing_seeds


def cmd_curate(args):
    checkpoint = args.checkpoint if hasattr(args, "checkpoint") and args.checkpoint else None
    if not checkpoint:
        checkpoint = find_best_checkpoint(args.checkpoint_dir)
    if not checkpoint:
        print("No checkpoints found — run training first", file=sys.stderr)
        sys.exit(1)

    curate_seeds(
        checkpoint_path=checkpoint,
        db_path=args.db,
        start_seed=args.start_seed,
        num_candidates=args.num_seeds,
        max_tries=args.max_tries,
    )
