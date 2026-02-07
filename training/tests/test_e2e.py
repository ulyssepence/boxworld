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

DESIGNED_LEVELS = [
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

HAS_DB = DB_PATH.exists()

# Cell type for goal (must match environment.py and types.ts)
GOAL_CELL = 4


@pytest.mark.skipif(not HAS_DB, reason="No database — run main.py all first")
class TestAllLevelsSolved:
    @pytest.fixture(autouse=True)
    def _db(self):
        self.conn = sqlite3.connect(str(DB_PATH))
        yield
        self.conn.close()

    @pytest.mark.parametrize("level_id", DESIGNED_LEVELS)
    def test_level_solved(self, level_id: str):
        """At least one episode must end with the agent on the goal cell."""
        episodes = self.conn.execute(
            "SELECT id, total_reward FROM episodes WHERE level_id = ?",
            (level_id,),
        ).fetchall()
        assert episodes, f"No episodes for '{level_id}'"

        best_reward = max(r for _, r in episodes)
        reached_goal = False
        for ep_id, _ in episodes:
            # Get the final step's state (highest step_number)
            row = self.conn.execute(
                "SELECT state_json FROM steps WHERE episode_id = ? ORDER BY step_number DESC LIMIT 1",
                (ep_id,),
            ).fetchone()
            if row is None:
                continue
            state = json.loads(row[0])
            ax, ay = state["agentPosition"]
            grid = state["level"]["grid"]
            # Check if agent is standing on the goal cell
            if grid[ay][ax] == GOAL_CELL:
                reached_goal = True
                break

        assert reached_goal, (
            f"'{level_id}' not solved: agent never reached the goal "
            f"(best_reward={best_reward:.2f}, {len(episodes)} episodes)"
        )
