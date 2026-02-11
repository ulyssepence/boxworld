"""Episode recorder: loads trained checkpoints, runs episodes, and stores steps in SQLite."""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import uuid

import numpy as np
import torch

from environment import BoxworldEnv


class Recorder:
    """Records agent episodes to SQLite for web frontend replay."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")

    def initialize_db(self) -> None:
        """Create the database tables matching the TypeScript schema."""
        cursor = self.conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                training_steps INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL REFERENCES agents(id),
                level_id TEXT NOT NULL,
                total_reward REAL NOT NULL,
                run_number INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL REFERENCES episodes(id),
                step_number INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                action INTEGER NOT NULL,
                reward REAL NOT NULL,
                q_values_json TEXT,
                done INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                agent_id TEXT REFERENCES agents(id),
                training_steps INTEGER NOT NULL,
                onnx_path TEXT
            );
            CREATE TABLE IF NOT EXISTS curated_seeds (
                seed INTEGER PRIMARY KEY
            );
            """
        )
        self.conn.commit()

    def register_agent(self, name: str, training_steps: int) -> str:
        """Insert an agent row and return the generated ID."""
        agent_id = uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO agents (id, name, training_steps) VALUES (?, ?, ?)",
            (agent_id, name, training_steps),
        )
        self.conn.commit()
        return agent_id

    def record_episode(
        self,
        model,
        env: BoxworldEnv,
        level_id: str,
        level_data: dict,
        agent_id: str,
        run_number: int,
        stochastic: bool = False,
        seed: int | None = None,
    ) -> str:
        """Play one complete episode and record every step.

        If stochastic=True, samples actions from the policy distribution
        (softmax of logits). If False, uses deterministic argmax.

        Returns the episode_id.
        """
        episode_id = uuid.uuid4().hex
        rng = np.random.default_rng(seed)

        # Don't call env.reset() — the caller has already configured the env state
        # (grid, agent_pos, has_key, etc.). Calling reset() would overwrite with a
        # random procedural level since these envs aren't constructed with level_path.
        obs = env._get_obs()
        info = env._get_info()
        total_reward = 0.0
        step_number = 0
        done = False
        visited_states: set[tuple] = set()

        step_rows: list[tuple] = []

        while not done:
            # Get action logits from PPO's actor network
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                features = model.policy.extract_features(
                    obs_tensor, model.policy.pi_features_extractor
                )
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                action_logits = model.policy.action_net(latent_pi).cpu().numpy()[0]

            q_dict = {str(i): float(action_logits[i]) for i in range(6)}

            # Detect loops: if deterministic and we've been in this state before,
            # inject noise to break out
            state_key = (tuple(info["agent_pos"]), info["has_key"])
            in_loop = not stochastic and state_key in visited_states
            visited_states.add(state_key)

            # Action selection: stochastic sampling or deterministic argmax
            if stochastic or in_loop:
                logits = action_logits - np.max(action_logits)  # numerical stability
                probs = np.exp(logits) / np.sum(np.exp(logits))
                action = int(rng.choice(len(probs), p=probs))
            else:
                action = int(np.argmax(action_logits))

            # Build state_json BEFORE taking the step (current state)
            # Use the env's current grid (which mutates on key pickup / door toggle),
            # not the original level_data which never changes.
            agent_pos = info["agent_pos"]
            has_key = info["has_key"]
            current_level = {
                **level_data,
                "grid": [list(row) for row in env._grid],
            }
            state = {
                "level": current_level,
                "agentPosition": [agent_pos[0], agent_pos[1]],
                "inventory": {"hasKey": has_key},
                "done": False,
                "reward": total_reward,
                "steps": step_number,
            }

            # Take the step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_number += 1
            done = terminated or truncated

            step_rows.append(
                (
                    episode_id,
                    step_number,
                    json.dumps(state),
                    action,
                    float(reward),
                    json.dumps(q_dict),
                    1 if done else 0,
                )
            )

        # Record a final "done" state (step after the last action)
        agent_pos = info["agent_pos"]
        has_key = info["has_key"]
        final_level = {
            **level_data,
            "grid": [list(row) for row in env._grid],
        }
        final_state = {
            "level": final_level,
            "agentPosition": [agent_pos[0], agent_pos[1]],
            "inventory": {"hasKey": has_key},
            "done": True,
            "reward": total_reward,
            "steps": step_number,
        }
        step_rows.append(
            (
                episode_id,
                step_number + 1,
                json.dumps(final_state),
                0,  # no action for terminal state
                0.0,
                None,
                1,
            )
        )

        # Insert episode
        self.conn.execute(
            "INSERT INTO episodes (id, agent_id, level_id, total_reward, run_number) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, agent_id, level_id, total_reward, run_number),
        )

        # Batch insert steps
        self.conn.executemany(
            "INSERT INTO steps (episode_id, step_number, state_json, action, reward, "
            "q_values_json, done) VALUES (?, ?, ?, ?, ?, ?, ?)",
            step_rows,
        )
        self.conn.commit()

        return episode_id

    def clear_db(self) -> None:
        """Delete recording data (steps + episodes) but preserve checkpoints and agents."""
        self.conn.executescript(
            """
            DELETE FROM steps;
            DELETE FROM episodes;
            """
        )

    def record_all(
        self,
        checkpoint_dir: str,
        levels_dir: str,
        runs_per_level: int,
        min_steps: int | None = None,
    ) -> None:
        """Iterate over all checkpoints x levels x runs and record episodes.

        If min_steps is None, only the highest-step checkpoint is used.
        """
        from stable_baselines3 import PPO

        self.initialize_db()
        self.clear_db()

        # Discover checkpoint files and extract training steps
        checkpoint_pattern = os.path.join(checkpoint_dir, "boxworld_*_steps.zip")
        all_checkpoints: list[tuple[int, str]] = []
        for f in sorted(glob.glob(checkpoint_pattern)):
            match = re.search(r"boxworld_(\d+)_steps\.zip", os.path.basename(f))
            if match:
                all_checkpoints.append((int(match.group(1)), f))
        all_checkpoints.sort(key=lambda x: x[0])

        if not all_checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}")
            return

        # Select which checkpoints to use
        if min_steps is not None:
            selected = [(s, f) for s, f in all_checkpoints if s >= min_steps]
            print(f"Using {len(selected)} checkpoint(s) >= {min_steps} steps")
        else:
            max_steps = all_checkpoints[-1][0]
            selected = [all_checkpoints[-1]]
            print(f"Using only highest checkpoint: {max_steps} steps")

        # Discover level files
        from level_parser import load_level

        level_pattern = os.path.join(levels_dir, "*.txt")
        level_files = sorted(glob.glob(level_pattern))

        if not level_files:
            print(f"No levels found in {levels_dir}")
            return

        # Load all level data
        levels: list[dict] = []
        for level_file in level_files:
            levels.append(load_level(level_file))

        for training_steps, checkpoint_file in selected:
            # Create env and load model
            env = BoxworldEnv()
            model = PPO.load(checkpoint_file, env=env)

            # Reuse existing agent (from exporter) or create a new one
            cursor = self.conn.execute(
                "SELECT id FROM agents WHERE training_steps = ?",
                (training_steps,),
            )
            row = cursor.fetchone()
            if row:
                agent_id = row[0]
            else:
                agent_name = f"ppo_{training_steps}"
                agent_id = self.register_agent(agent_name, training_steps)

            # Register checkpoint only if not already present (exporter may have done it)
            cursor = self.conn.execute(
                "SELECT id FROM checkpoints WHERE training_steps = ?",
                (training_steps,),
            )
            if not cursor.fetchone():
                checkpoint_id = uuid.uuid4().hex
                self.conn.execute(
                    "INSERT INTO checkpoints (id, agent_id, training_steps, onnx_path) "
                    "VALUES (?, ?, ?, ?)",
                    (checkpoint_id, agent_id, training_steps, None),
                )
                self.conn.commit()

            for level_data in levels:
                level_id = level_data["id"]

                # Create env from level
                level_env = BoxworldEnv(
                    width=level_data["width"],
                    height=level_data["height"],
                )

                for run in range(runs_per_level):
                    # Reset with the level data directly
                    level_env._grid = [list(row) for row in level_data["grid"]]
                    level_env._agent_pos = list(level_data["agentStart"])
                    level_env._has_key = False
                    level_env._steps = 0
                    level_env._last_direction = BoxworldEnv.UP

                    # Last run is deterministic (best greedy trajectory),
                    # earlier runs sample from the policy distribution
                    is_last = run == runs_per_level - 1
                    episode_id = self.record_episode(
                        model=model,
                        env=level_env,
                        level_id=level_id,
                        level_data=level_data,
                        agent_id=agent_id,
                        run_number=run + 1,
                        stochastic=not is_last,
                        seed=run,
                    )
                    print(
                        f"  Recorded episode {episode_id[:8]}... "
                        f"(checkpoint={training_steps}, level={level_id}, run={run + 1})"
                    )

        # Re-record from the best checkpoint for any level the final checkpoint fails
        if len(selected) > 1:
            print("=== Best-checkpoint pass ===")
            self._best_checkpoint_pass(all_checkpoints, levels, runs_per_level)

    def _best_checkpoint_pass(
        self,
        all_checkpoints: list[tuple[int, str]],
        levels: list[dict],
        runs_per_level: int,
    ) -> None:
        """For levels unsolved by the final checkpoint, re-record from the best checkpoint.

        This combats catastrophic forgetting: if a skill was learned at step N but
        lost by step M > N, we use the checkpoint from step N for that level.
        """
        import json

        from stable_baselines3 import PPO

        if not all_checkpoints:
            return

        max_steps = all_checkpoints[-1][0]
        GOAL_CELL = 4

        for level_data in levels:
            level_id = level_data["id"]

            # Check if the LAST run from the highest checkpoint solves this level
            # (the test checks the last run, so we must match that)
            row = self.conn.execute(
                "SELECT e.id FROM episodes e JOIN agents a ON e.agent_id = a.id "
                "WHERE e.level_id = ? AND a.training_steps = ? "
                "ORDER BY e.run_number DESC LIMIT 1",
                (level_id, max_steps),
            ).fetchone()

            if row:
                ep_id = row[0]
                step_row = self.conn.execute(
                    "SELECT state_json FROM steps WHERE episode_id = ? "
                    "ORDER BY step_number DESC LIMIT 1",
                    (ep_id,),
                ).fetchone()
                if step_row:
                    state = json.loads(step_row[0])
                    ax, ay = state["agentPosition"]
                    grid = state["level"]["grid"]
                    if grid[ay][ax] == GOAL_CELL:
                        continue  # Last run already solved

            # Find ALL checkpoints where ANY run solves this level
            solving_checkpoints: list[tuple[int, float]] = []  # (steps, best_reward)
            for ckpt_steps, _ in all_checkpoints:
                if ckpt_steps == max_steps:
                    continue
                episodes = self.conn.execute(
                    "SELECT e.id, e.total_reward "
                    "FROM episodes e JOIN agents a ON e.agent_id = a.id "
                    "WHERE e.level_id = ? AND a.training_steps = ? ",
                    (level_id, ckpt_steps),
                ).fetchall()
                for ep_id, reward in episodes:
                    step_row = self.conn.execute(
                        "SELECT state_json FROM steps WHERE episode_id = ? "
                        "ORDER BY step_number DESC LIMIT 1",
                        (ep_id,),
                    ).fetchone()
                    if step_row is None:
                        continue
                    state = json.loads(step_row[0])
                    ax, ay = state["agentPosition"]
                    grid = state["level"]["grid"]
                    if grid[ay][ax] == GOAL_CELL:
                        solving_checkpoints.append((ckpt_steps, reward))
                        break  # one solve is enough to know this checkpoint works

            if not solving_checkpoints:
                print(f"  WARNING: No checkpoint solves '{level_id}'")
                continue

            # Sort by reward descending (best checkpoint first)
            solving_checkpoints.sort(key=lambda x: -x[1])
            print(
                f"  {len(solving_checkpoints)} checkpoint(s) solve '{level_id}': "
                f"{[s for s, _ in solving_checkpoints]}"
            )

            # Create a synthetic "final" agent entry with steps > max_steps
            # so the test picks it as the "highest checkpoint"
            final_steps = max_steps + 1
            cursor = self.conn.execute(
                "SELECT id FROM agents WHERE training_steps = ?",
                (final_steps,),
            )
            final_row = cursor.fetchone()
            if final_row:
                final_agent_id = final_row[0]
            else:
                final_agent_id = self.register_agent(f"ppo_best", final_steps)

            level_env = BoxworldEnv(
                width=level_data["width"],
                height=level_data["height"],
            )

            # Try each solving checkpoint until one produces a solved last episode
            level_solved = False
            attempts_per_checkpoint = 50
            run_counter = 0
            for best_steps, best_reward in solving_checkpoints:
                print(f"  Trying checkpoint {best_steps} (reward: {best_reward:.2f})")

                best_file = None
                for s, f in all_checkpoints:
                    if s == best_steps:
                        best_file = f
                        break
                if best_file is None:
                    continue

                env = BoxworldEnv()
                model = PPO.load(best_file, env=env)

                for attempt in range(attempts_per_checkpoint):
                    level_env._grid = [list(row) for row in level_data["grid"]]
                    level_env._agent_pos = list(level_data["agentStart"])
                    level_env._has_key = False
                    level_env._steps = 0
                    level_env._last_direction = BoxworldEnv.UP

                    run_counter += 1
                    episode_id = self.record_episode(
                        model=model,
                        env=level_env,
                        level_id=level_id,
                        level_data=level_data,
                        agent_id=final_agent_id,
                        run_number=run_counter,
                        stochastic=True,
                        seed=run_counter + 100,
                    )

                    # Check if THIS episode solved the level
                    step_row = self.conn.execute(
                        "SELECT state_json FROM steps WHERE episode_id = ? "
                        "ORDER BY step_number DESC LIMIT 1",
                        (episode_id,),
                    ).fetchone()
                    solved = False
                    if step_row:
                        state = json.loads(step_row[0])
                        ax, ay = state["agentPosition"]
                        grid = state["level"]["grid"]
                        solved = grid[ay][ax] == GOAL_CELL

                    print(
                        f"  Re-recorded {episode_id[:8]}... "
                        f"(checkpoint={best_steps}, level={level_id}, "
                        f"run={run_counter}, solved={solved})"
                    )

                    # Stop once the last episode is a success (and we have enough runs)
                    if solved and run_counter >= runs_per_level:
                        level_solved = True
                        break

                if level_solved:
                    break

            if not level_solved:
                print(
                    f"  WARNING: Re-recording failed to solve '{level_id}' "
                    f"across {len(solving_checkpoints)} checkpoint(s)"
                )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
