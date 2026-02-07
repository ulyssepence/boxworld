"""Tests for the PPO training pipeline."""

import os
import tempfile

from environment import BoxworldEnv
from train import Trainer, TrainerConfig


def test_trainer_initializes_with_valid_env():
    env = BoxworldEnv()
    trainer = Trainer(env)
    assert trainer.model is not None


def test_short_training_run():
    env = BoxworldEnv()
    trainer = Trainer(env)
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(total_steps=100, checkpoint_interval=50, checkpoint_dir=tmpdir)


def test_checkpoints_saved_at_intervals():
    env = BoxworldEnv()
    trainer = Trainer(env)
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(total_steps=200, checkpoint_interval=100, checkpoint_dir=tmpdir)
        # SB3 saves checkpoints as .zip files
        checkpoint_files = [f for f in os.listdir(tmpdir) if f.endswith(".zip")]
        assert len(checkpoint_files) >= 2


def test_checkpoint_can_be_loaded():
    env = BoxworldEnv()
    trainer = Trainer(env)
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(total_steps=100, checkpoint_interval=50, checkpoint_dir=tmpdir)
        checkpoint_files = [f for f in os.listdir(tmpdir) if f.endswith(".zip")]
        assert len(checkpoint_files) > 0
        # Load the first checkpoint
        checkpoint_path = os.path.join(tmpdir, checkpoint_files[0])
        trainer.load_checkpoint(checkpoint_path)
        # Verify model can predict
        obs, _ = env.reset(seed=42)
        action, _ = trainer.model.predict(obs, deterministic=True)
        assert 0 <= action < 6


def test_custom_hyperparameters():
    env = BoxworldEnv()
    config = TrainerConfig(learning_rate=1e-3, batch_size=32)
    trainer = Trainer(env, config)
    assert trainer.config.learning_rate == 1e-3
    assert trainer.config.batch_size == 32
