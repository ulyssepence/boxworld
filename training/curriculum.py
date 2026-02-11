"""Curriculum learning callback for PPO training."""

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """Ramps env difficulty from 0→1 over the first fraction of training."""

    def __init__(self, total_timesteps: int, ramp_fraction: float = 0.6, verbose: int = 0):
        super().__init__(verbose)
        self._total = total_timesteps
        self._ramp_end = int(total_timesteps * ramp_fraction)

    def _on_step(self) -> bool:
        difficulty = min(1.0, self.num_timesteps / max(1, self._ramp_end))
        for env in self.training_env.envs:
            env._difficulty = difficulty
        if self.verbose > 0 and self.num_timesteps % 50_000 < self.training_env.num_envs:
            print(f"  [curriculum] step={self.num_timesteps}, difficulty={difficulty:.3f}")
        return True
