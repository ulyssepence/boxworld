"""Per-generator solve rate test using the best checkpoint."""

import glob
import re
import numpy as np
from stable_baselines3 import PPO
from environment import BoxworldEnv


GENERATORS = [
    "open_room",
    "room_partition",
    "lava_field",
    "wall_segments",
    "hybrid",
    "bsp_rooms",
    "scattered_walls",
]

N_LEVELS = 50
N_TRIES = 3
MAX_STEPS = 200


def find_best_checkpoint() -> str:
    files = glob.glob("../data/checkpoints/boxworld_*_steps.zip")
    if not files:
        raise FileNotFoundError("No checkpoints found in data/checkpoints/")

    def extract_steps(path: str) -> int:
        m = re.search(r"boxworld_(\d+)_steps", path)
        return int(m.group(1)) if m else 0

    return max(files, key=extract_steps)


def try_solve(model, env: BoxworldEnv, obs: np.ndarray) -> bool:
    """Run one stochastic episode. Returns True if reward > 0."""
    total_reward = 0.0
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward > 0


def evaluate_generator(model, gen_name: str) -> tuple[int, int]:
    env = BoxworldEnv()
    env.reset()  # initialize internals
    env._difficulty = 1.0

    gen_method = getattr(env, f"_gen_{gen_name}")
    solved = 0

    for seed in range(N_LEVELS):
        rng = np.random.default_rng(seed)
        gen_method(rng)
        env._has_key = False
        env._solve_subgoals()
        base_grid = [list(row) for row in env._grid]
        base_agent = list(env._agent_pos)

        level_solved = False
        for _ in range(N_TRIES):
            # Restore level state
            env._grid = [list(row) for row in base_grid]
            env._agent_pos = list(base_agent)
            env._has_key = False
            env._steps = 0
            env._solve_subgoals()
            obs = env._get_obs()

            if try_solve(model, env, obs):
                level_solved = True
                break

        if level_solved:
            solved += 1

    return solved, N_LEVELS


def main():
    ckpt = find_best_checkpoint()
    steps_m = re.search(r"boxworld_(\d+)_steps", ckpt)
    steps_str = f"{int(steps_m.group(1)):,}" if steps_m else "?"
    print(f"Checkpoint: {ckpt} ({steps_str} steps)")
    print(f"Levels per generator: {N_LEVELS}, tries per level: {N_TRIES}\n")

    model = PPO.load(ckpt)

    results = {}
    for gen in GENERATORS:
        solved, total = evaluate_generator(model, gen)
        pct = 100 * solved / total
        results[gen] = (solved, total, pct)
        print(f"  {gen:<20s}  {solved:>3d}/{total}  ({pct:5.1f}%)")

    print()
    print(f"{'Generator':<20s}  {'Solved':>6s}  {'Rate':>6s}")
    print("-" * 38)
    total_solved = 0
    total_levels = 0
    for gen in GENERATORS:
        s, t, p = results[gen]
        total_solved += s
        total_levels += t
        print(f"{gen:<20s}  {s:>3d}/{t}  {p:5.1f}%")
    overall = 100 * total_solved / total_levels
    print("-" * 38)
    print(f"{'OVERALL':<20s}  {total_solved:>3d}/{total_levels}  {overall:5.1f}%")


if __name__ == "__main__":
    main()
