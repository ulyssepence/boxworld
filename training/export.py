"""ONNX export for SB3 DQN checkpoints."""

from __future__ import annotations

import glob
import os
import re
import sqlite3
import uuid

import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import DQN


class Exporter:
    """Exports Stable-Baselines3 DQN checkpoints to ONNX format.

    Optionally registers exported checkpoints in a SQLite database.
    """

    def __init__(self, db_path: str | None = None):
        self.conn: sqlite3.Connection | None = None
        if db_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            self.conn = sqlite3.connect(db_path)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self._initialize_db()

    def _initialize_db(self) -> None:
        """Create the database tables if they don't exist."""
        assert self.conn is not None
        self.conn.executescript(
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
            """
        )
        self.conn.commit()

    def export_checkpoint(
        self,
        sb3_model_path: str,
        output_onnx_path: str,
        obs_size: int = 103,
    ) -> str:
        """Load an SB3 DQN checkpoint, extract the Q-network, and export to ONNX.

        Returns the output ONNX path.
        """
        model = DQN.load(sb3_model_path)
        q_net = model.policy.q_net
        q_net.eval()

        dummy_input = torch.randn(1, obs_size)
        torch.onnx.export(
            q_net,
            dummy_input,
            output_onnx_path,
            input_names=["obs"],
            output_names=["q_values"],
            dynamic_axes={"obs": {0: "batch"}, "q_values": {0: "batch"}},
            external_data=False,
        )
        return output_onnx_path

    def export_all(self, checkpoint_dir: str, output_dir: str) -> list[str]:
        """Export all boxworld_*_steps.zip checkpoints to ONNX.

        Returns list of output ONNX paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        pattern = os.path.join(checkpoint_dir, "boxworld_*_steps.zip")
        checkpoint_files = sorted(glob.glob(pattern))

        output_paths: list[str] = []

        for checkpoint_file in checkpoint_files:
            basename = os.path.basename(checkpoint_file)
            stem = basename.replace(".zip", "")
            onnx_filename = stem + ".onnx"
            onnx_path = os.path.join(output_dir, onnx_filename)

            self.export_checkpoint(checkpoint_file, onnx_path)
            output_paths.append(onnx_path)

            # Register in DB if connected
            if self.conn is not None:
                match = re.search(r"boxworld_(\d+)_steps", basename)
                if match:
                    training_steps = int(match.group(1))
                    agent_name = f"dqn_{training_steps}"
                    agent_id = uuid.uuid4().hex
                    self.conn.execute(
                        "INSERT INTO agents (id, name, training_steps) VALUES (?, ?, ?)",
                        (agent_id, agent_name, training_steps),
                    )
                    checkpoint_id = uuid.uuid4().hex
                    self.conn.execute(
                        "INSERT INTO checkpoints (id, agent_id, training_steps, onnx_path) "
                        "VALUES (?, ?, ?, ?)",
                        (checkpoint_id, agent_id, training_steps, onnx_path),
                    )
                    self.conn.commit()

            print(f"Exported {basename} -> {onnx_filename}")

        return output_paths

    def verify_export(
        self,
        onnx_path: str,
        sb3_model_path: str,
        test_obs: np.ndarray | None = None,
    ) -> bool:
        """Verify that ONNX output matches the SB3 Q-network output.

        Returns True if outputs match within 1e-5 tolerance.
        """
        model = DQN.load(sb3_model_path)

        if test_obs is None:
            obs_size = model.observation_space.shape[0]
            test_obs = np.random.rand(obs_size).astype(np.float32)

        # SB3 / PyTorch side
        q_net = model.policy.q_net
        q_net.eval()
        with torch.no_grad():
            sb3_out = q_net(torch.as_tensor(test_obs).unsqueeze(0)).detach().numpy()

        # ONNX side
        session = ort.InferenceSession(onnx_path)
        onnx_out = session.run(None, {"obs": test_obs.reshape(1, -1)})[0]

        return bool(np.allclose(sb3_out, onnx_out, atol=1e-5))

    def close(self) -> None:
        """Close the database connection if open."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
