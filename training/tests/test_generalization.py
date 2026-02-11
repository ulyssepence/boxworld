"""Generalization test: verifies trained agent solves random procedural levels.

Loads the best checkpoint and runs stochastic policy (3 attempts per level)
on 100 procedural levels with seeds the agent never saw during training.
Passes if solve rate >= 40%.

Uses stochastic (softmax) evaluation because deterministic (argmax) policies
get stuck in deterministic loops — same finding as in recording system.
"""

import glob
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"


def _find_best_checkpoint() -> str | None:
    """Find the checkpoint with the most training steps."""
    pattern = str(CHECKPOINT_DIR / "boxworld_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None

    def steps(path):
        m = re.search(r"boxworld_(\d+)_steps\.zip", path)
        return int(m.group(1)) if m else 0

    return max(files, key=steps)


HAS_CHECKPOINTS = _find_best_checkpoint() is not None


@pytest.mark.skipif(not HAS_CHECKPOINTS, reason="No checkpoints — run main.py all first")
def test_generalization_on_random_levels():
    """Agent must solve >= 40% of 100 unseen procedural levels (3 stochastic tries each)."""
    from stable_baselines3 import PPO

    from environment import BoxworldEnv

    checkpoint = _find_best_checkpoint()
    assert checkpoint is not None

    model = PPO.load(checkpoint)

    num_levels = 100
    seed_base = 99999
    max_tries = 3
    solved = 0

    for i in range(num_levels):
        level_solved = False
        for _ in range(max_tries):
            env = BoxworldEnv()
            obs, _ = env.reset(seed=seed_base + i)
            for _ in range(BoxworldEnv.MAX_STEPS):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                if terminated:
                    if reward > 0:
                        level_solved = True
                    break
                if truncated:
                    break
            if level_solved:
                break
        if level_solved:
            solved += 1

    solve_rate = solved / num_levels
    print(f"\nGeneralization: {solved}/{num_levels} = {solve_rate:.1%} (3 stochastic tries)")
    assert solve_rate >= 0.40, (
        f"Solve rate {solve_rate:.1%} ({solved}/{num_levels}) < 40% threshold"
    )
