"""E2E acceptance test: verifies trained agent solves all designed levels.

Reads data/db.sqlite produced by `main.py all`. Run the pipeline first:
    cd training && uv run python main.py all
Then:
    cd training && uv run pytest tests/test_e2e.py -v
"""

import json
import pathlib
import sqlite3

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "db.sqlite"

TRAINING_LEVELS = [
    "open_room",
    "simple_corridor",
    "lava_crossing",
    "door_key",
    "two_rooms",
    "two_keys",
    "open_shortcut",
    "three_keys",
    "zigzag_lava",
    "dead_ends",
]

HOLDOUT_LEVELS = [
    "key_lava_gauntlet",
]

HAS_DB = DB_PATH.exists()

# Cell type for goal (must match environment.py and types.ts)
GOAL_CELL = 4


def _check_level_solved(conn: sqlite3.Connection, level_id: str) -> tuple[bool, str]:
    """Check that the last recorded episode ends with the agent on the goal cell.

    "Last" = highest run_number from the highest-step checkpoint.
    Returns (solved, message) tuple.
    """
    row = conn.execute(
        "SELECT e.id, e.total_reward, e.run_number, a.training_steps "
        "FROM episodes e JOIN agents a ON e.agent_id = a.id "
        "WHERE e.level_id = ? "
        "ORDER BY a.training_steps DESC, e.run_number DESC LIMIT 1",
        (level_id,),
    ).fetchone()
    if row is None:
        return False, f"No episodes for '{level_id}'"

    ep_id, reward, run_number, steps = row

    step_row = conn.execute(
        "SELECT state_json FROM steps WHERE episode_id = ? ORDER BY step_number DESC LIMIT 1",
        (ep_id,),
    ).fetchone()
    if step_row is None:
        return False, f"No steps in last episode (reward={reward:.2f})"

    state = json.loads(step_row[0])
    ax, ay = state["agentPosition"]
    grid = state["level"]["grid"]
    if grid[ay][ax] == GOAL_CELL:
        return True, ""

    return False, f"reward={reward:.2f}, run={run_number}, checkpoint={steps}"


@pytest.mark.skipif(not HAS_DB, reason="No database — run main.py all first")
class TestTrainingLevelsSolved:
    @pytest.fixture(autouse=True)
    def _db(self):
        self.conn = sqlite3.connect(str(DB_PATH))
        yield
        self.conn.close()

    @pytest.mark.parametrize("level_id", TRAINING_LEVELS)
    def test_level_solved(self, level_id: str):
        """At least one episode must end with the agent on the goal cell."""
        solved, msg = _check_level_solved(self.conn, level_id)
        assert solved, f"'{level_id}' not solved: agent never reached the goal ({msg})"


@pytest.mark.skipif(not HAS_DB, reason="No database — run main.py all first")
class TestHoldoutLevelsSolved:
    @pytest.fixture(autouse=True)
    def _db(self):
        self.conn = sqlite3.connect(str(DB_PATH))
        yield
        self.conn.close()

    @pytest.mark.parametrize("level_id", HOLDOUT_LEVELS)
    def test_level_solved(self, level_id: str):
        """Holdout level must be solved — agent failed to generalize if not."""
        solved, msg = _check_level_solved(self.conn, level_id)
        assert solved, f"'{level_id}' failed to generalize: agent never reached the goal ({msg})"
