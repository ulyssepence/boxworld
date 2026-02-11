"""Tests for the seed curation pipeline."""

import os
import pathlib
import shutil
import sqlite3
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"


def _has_bun():
    return shutil.which("bun") is not None


def _has_checkpoint():
    from curate import find_best_checkpoint

    return find_best_checkpoint(str(CHECKPOINT_DIR)) is not None


class TestGenerateLevelsViaBun:
    @pytest.mark.skipif(not _has_bun(), reason="Bun not installed")
    def test_generates_levels(self):
        from curate import generate_levels_via_bun

        levels = generate_levels_via_bun(start=0, count=5)
        assert len(levels) >= 1
        for lv in levels:
            assert "seed" in lv
            assert "grid" in lv
            assert "agentStart" in lv
            assert "width" in lv
            assert "height" in lv

    @pytest.mark.skipif(not _has_bun(), reason="Bun not installed")
    def test_deterministic(self):
        from curate import generate_levels_via_bun

        a = generate_levels_via_bun(start=42, count=3)
        b = generate_levels_via_bun(start=42, count=3)
        assert a == b


class TestEvaluateLevel:
    @pytest.mark.skipif(not _has_checkpoint(), reason="No checkpoints")
    def test_evaluate_trivial_level(self):
        """An open room with agent next to goal should be solved."""
        from stable_baselines3 import PPO

        from curate import evaluate_level, find_best_checkpoint

        checkpoint = find_best_checkpoint(str(CHECKPOINT_DIR))
        model = PPO.load(checkpoint)

        # 10x10, walls on border, agent at (1,1), goal at (2,1)
        grid = [[1] * 10 for _ in range(10)]
        for y in range(1, 9):
            for x in range(1, 9):
                grid[y][x] = 0
        grid[1][2] = 4  # goal right next to agent
        level = {"width": 10, "height": 10, "grid": grid, "agentStart": [1, 1]}
        assert evaluate_level(model, level, max_tries=3) is True


class TestCurateSeeds:
    @pytest.mark.skipif(
        not (_has_bun() and _has_checkpoint()),
        reason="Needs Bun + checkpoints",
    )
    def test_curate_small_batch(self):
        from curate import curate_seeds, find_best_checkpoint

        checkpoint = find_best_checkpoint(str(CHECKPOINT_DIR))
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        try:
            seeds = curate_seeds(
                checkpoint_path=checkpoint,
                db_path=db_path,
                start_seed=0,
                num_candidates=20,
                max_tries=1,
            )
            assert isinstance(seeds, list)
            assert all(isinstance(s, int) for s in seeds)

            conn = sqlite3.connect(db_path)
            saved = [r[0] for r in conn.execute("SELECT seed FROM curated_seeds ORDER BY seed")]
            conn.close()
            assert sorted(seeds) == saved
        finally:
            os.unlink(db_path)
