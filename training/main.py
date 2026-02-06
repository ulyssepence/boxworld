"""Boxworld training CLI entry point."""

import argparse

from environment import BoxworldEnv
from train import Trainer, TrainerConfig


def cmd_train(args):
    env = BoxworldEnv()
    trainer = Trainer(env, TrainerConfig())
    trainer.train(
        total_steps=args.steps,
        checkpoint_interval=args.interval,
        checkpoint_dir=args.checkpoint_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Boxworld training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train DQN agent")
    train_parser.add_argument("--steps", type=int, default=500_000)
    train_parser.add_argument("--interval", type=int, default=10_000)
    train_parser.add_argument("--checkpoint-dir", default="../data/checkpoints")
    train_parser.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
