"""PPO training pipeline using Stable-Baselines3."""

import os
from dataclasses import dataclass
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from environment import BoxworldEnv


@dataclass
class TrainerConfig:
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 8


class Trainer:
    """PPO trainer wrapping Stable-Baselines3."""

    def __init__(
        self,
        env_fn: Callable[..., BoxworldEnv] | BoxworldEnv,
        config: TrainerConfig | None = None,
        env_kwargs: dict | None = None,
    ):
        self.config = config or TrainerConfig()

        # Accept either an env factory (for make_vec_env) or a single env instance
        if callable(env_fn) and not isinstance(env_fn, BoxworldEnv):
            self.vec_env = make_vec_env(
                env_fn,
                n_envs=self.config.n_envs,
                env_kwargs=env_kwargs or {},
            )
        else:
            # Single env passed directly (e.g. from tests)
            self.vec_env = env_fn

        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128])},
            verbose=1,
        )

    def train(self, total_steps: int, checkpoint_interval: int, checkpoint_dir: str) -> None:
        """Run training, saving checkpoints at regular intervals."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        # PPO save_freq is per-env, so divide by n_envs for total-step intervals
        n_envs = getattr(self.vec_env, "num_envs", 1)
        callback = CheckpointCallback(
            save_freq=max(1, checkpoint_interval // n_envs),
            save_path=checkpoint_dir,
            name_prefix="boxworld",
        )
        self.model.learn(total_timesteps=total_steps, callback=callback)

    def load_checkpoint(self, path: str) -> None:
        """Load a saved model checkpoint."""
        self.model = PPO.load(path, env=self.vec_env)
