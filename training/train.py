"""DQN training pipeline using Stable-Baselines3."""

import os
from dataclasses import dataclass

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from environment import BoxworldEnv


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    batch_size: int = 64
    gamma: float = 0.99
    target_update_interval: int = 1000
    exploration_fraction: float = 0.3
    exploration_final_eps: float = 0.02


class Trainer:
    """DQN trainer wrapping Stable-Baselines3."""

    def __init__(self, env: BoxworldEnv, config: TrainerConfig | None = None):
        self.env = env
        self.config = config or TrainerConfig()
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            target_update_interval=self.config.target_update_interval,
            exploration_fraction=self.config.exploration_fraction,
            exploration_final_eps=self.config.exploration_final_eps,
            policy_kwargs={"net_arch": [128, 128]},
            verbose=1,
        )

    def train(self, total_steps: int, checkpoint_interval: int, checkpoint_dir: str) -> None:
        """Run training, saving checkpoints at regular intervals."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=checkpoint_interval,
            save_path=checkpoint_dir,
            name_prefix="boxworld",
        )
        self.model.learn(total_timesteps=total_steps, callback=callback)

    def load_checkpoint(self, path: str) -> None:
        """Load a saved model checkpoint."""
        self.model = DQN.load(path, env=self.env)
