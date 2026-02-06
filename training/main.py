"""Boxworld training CLI entry point."""

import argparse

from environment import BoxworldEnv
from train import Trainer, TrainerConfig


def cmd_train(args):
    env = BoxworldEnv(
        levels_dir=args.levels_dir,
        designed_level_prob=0.7,
    )
    trainer = Trainer(env, TrainerConfig())
    trainer.train(
        total_steps=args.steps,
        checkpoint_interval=args.interval,
        checkpoint_dir=args.checkpoint_dir,
    )


def cmd_record(args):
    from record import Recorder

    recorder = Recorder(args.db)
    try:
        recorder.record_all(
            checkpoint_dir=args.checkpoint_dir,
            levels_dir=args.levels_dir,
            runs_per_level=args.runs_per_level,
            min_steps=args.min_steps,
        )
    finally:
        recorder.close()


def cmd_export(args):
    from export import Exporter

    exporter = Exporter(db_path=args.db)
    try:
        paths = exporter.export_all(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
        )
        print(f"Exported {len(paths)} checkpoint(s) to ONNX")
    finally:
        exporter.close()


def cmd_all(args):
    """Run the full pipeline: train -> export -> record."""
    import os

    # Ensure output_dir falls back to checkpoint_dir if not explicitly set
    if not hasattr(args, "output_dir") or args.output_dir is None:
        args.output_dir = args.checkpoint_dir

    # Use total training steps as min_steps so only the final checkpoint is recorded
    if args.min_steps is None:
        args.min_steps = args.steps

    # Clear database so we start fresh
    for suffix in ("", "-shm", "-wal"):
        path = args.db + suffix
        if os.path.exists(path):
            os.remove(path)

    # Clear old checkpoints so stale files from previous runs don't pollute recording
    if os.path.isdir(args.checkpoint_dir):
        for f in os.listdir(args.checkpoint_dir):
            fp = os.path.join(args.checkpoint_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    print("=== Step 1: Training ===")
    cmd_train(args)
    print("=== Step 2: Exporting to ONNX ===")
    cmd_export(args)
    print("=== Step 3: Recording episodes ===")
    cmd_record(args)
    print("=== Pipeline complete ===")


def main():
    parser = argparse.ArgumentParser(description="Boxworld training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train DQN agent")
    train_parser.add_argument("--steps", type=int, default=500_000)
    train_parser.add_argument("--interval", type=int, default=10_000)
    train_parser.add_argument("--checkpoint-dir", default="../data/checkpoints")
    train_parser.add_argument("--levels-dir", default="../data/levels")
    train_parser.set_defaults(func=cmd_train)

    record_parser = subparsers.add_parser("record", help="Record agent episodes to SQLite")
    record_parser.add_argument("--db", default="../data/db.sqlite")
    record_parser.add_argument("--checkpoint-dir", default="../data/checkpoints")
    record_parser.add_argument("--levels-dir", default="../data/levels")
    record_parser.add_argument("--runs-per-level", type=int, default=5)
    record_parser.add_argument(
        "--min-steps",
        type=int,
        default=None,
        help="Only record from checkpoints with at least this many training steps (default: highest checkpoint only)",
    )
    record_parser.set_defaults(func=cmd_record)

    export_parser = subparsers.add_parser("export", help="Export checkpoints to ONNX format")
    export_parser.add_argument("--checkpoint-dir", default="../data/checkpoints")
    export_parser.add_argument("--output-dir", default="../data/checkpoints")
    export_parser.add_argument("--db", default="../data/db.sqlite")
    export_parser.set_defaults(func=cmd_export)

    all_parser = subparsers.add_parser("all", help="Run full pipeline: train -> export -> record")
    all_parser.add_argument("--steps", type=int, default=500_000)
    all_parser.add_argument("--interval", type=int, default=10_000)
    all_parser.add_argument("--checkpoint-dir", default="../data/checkpoints")
    all_parser.add_argument("--output-dir", default=None)
    all_parser.add_argument("--db", default="../data/db.sqlite")
    all_parser.add_argument("--levels-dir", default="../data/levels")
    all_parser.add_argument("--runs-per-level", type=int, default=5)
    all_parser.add_argument(
        "--min-steps",
        type=int,
        default=None,
        help="Only record from checkpoints with at least this many training steps (default: total --steps)",
    )
    all_parser.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
