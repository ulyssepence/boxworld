"""Tests for the episode recorder."""

import json
import os
import sqlite3

import pytest
import torch
from stable_baselines3 import DQN

from environment import BoxworldEnv
from record import Recorder


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.sqlite")


@pytest.fixture
def recorder(db_path):
    rec = Recorder(db_path)
    rec.initialize_db()
    yield rec
    rec.close()


@pytest.fixture
def trained_model():
    """Create a DQN model with minimal training for testing."""
    env = BoxworldEnv()
    model = DQN("MlpPolicy", env, learning_starts=10, verbose=0)
    model.learn(total_timesteps=50)
    return model, env


def test_initialize_creates_tables(recorder):
    cursor = recorder.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cursor.fetchall()}
    assert "agents" in tables
    assert "episodes" in tables
    assert "steps" in tables
    assert "checkpoints" in tables


def test_register_agent(recorder):
    agent_id = recorder.register_agent("test_agent", 1000)
    assert agent_id is not None
    assert len(agent_id) == 32  # uuid4 hex

    cursor = recorder.conn.execute(
        "SELECT name, training_steps FROM agents WHERE id = ?", (agent_id,)
    )
    row = cursor.fetchone()
    assert row[0] == "test_agent"
    assert row[1] == 1000


def test_record_single_episode(recorder, trained_model):
    model, env = trained_model
    agent_id = recorder.register_agent("test_agent", 50)

    level_data = {
        "id": "test_level",
        "name": "Test Level",
        "width": 10,
        "height": 10,
        "grid": env._grid,
        "agentStart": list(env._agent_pos),
    }

    obs, info = env.reset(seed=42)

    episode_id = recorder.record_episode(
        model=model,
        env=env,
        level_id="test_level",
        level_data=level_data,
        agent_id=agent_id,
        run_number=1,
    )

    # Check episode row exists
    cursor = recorder.conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
    episode = cursor.fetchone()
    assert episode is not None

    # Check step rows exist
    cursor = recorder.conn.execute("SELECT COUNT(*) FROM steps WHERE episode_id = ?", (episode_id,))
    step_count = cursor.fetchone()[0]
    assert step_count > 0


def test_step_data_is_valid_json(recorder, trained_model):
    model, env = trained_model
    agent_id = recorder.register_agent("test_agent", 50)

    level_data = {
        "id": "test_level",
        "name": "Test Level",
        "width": 10,
        "height": 10,
        "grid": env._grid,
        "agentStart": list(env._agent_pos),
    }

    obs, info = env.reset(seed=42)

    episode_id = recorder.record_episode(
        model=model,
        env=env,
        level_id="test_level",
        level_data=level_data,
        agent_id=agent_id,
        run_number=1,
    )

    cursor = recorder.conn.execute(
        "SELECT state_json, q_values_json FROM steps WHERE episode_id = ?", (episode_id,)
    )
    rows = cursor.fetchall()
    assert len(rows) > 0

    for state_json, q_values_json in rows:
        # state_json must be valid JSON
        state = json.loads(state_json)
        assert "level" in state
        assert "agentPosition" in state
        assert "inventory" in state
        assert "done" in state
        assert "reward" in state
        assert "steps" in state

        # q_values_json may be None for the terminal step
        if q_values_json is not None:
            q = json.loads(q_values_json)
            assert isinstance(q, dict)


def test_q_values_have_six_actions(recorder, trained_model):
    model, env = trained_model
    agent_id = recorder.register_agent("test_agent", 50)

    level_data = {
        "id": "test_level",
        "name": "Test Level",
        "width": 10,
        "height": 10,
        "grid": env._grid,
        "agentStart": list(env._agent_pos),
    }

    obs, info = env.reset(seed=42)

    episode_id = recorder.record_episode(
        model=model,
        env=env,
        level_id="test_level",
        level_data=level_data,
        agent_id=agent_id,
        run_number=1,
    )

    cursor = recorder.conn.execute(
        "SELECT q_values_json FROM steps WHERE episode_id = ? AND q_values_json IS NOT NULL",
        (episode_id,),
    )
    rows = cursor.fetchall()
    assert len(rows) > 0

    for (q_values_json,) in rows:
        q = json.loads(q_values_json)
        assert len(q) == 6
        for i in range(6):
            assert str(i) in q
            assert isinstance(q[str(i)], float)


def test_episode_total_reward_matches_steps(recorder, trained_model):
    model, env = trained_model
    agent_id = recorder.register_agent("test_agent", 50)

    level_data = {
        "id": "test_level",
        "name": "Test Level",
        "width": 10,
        "height": 10,
        "grid": env._grid,
        "agentStart": list(env._agent_pos),
    }

    obs, info = env.reset(seed=42)

    episode_id = recorder.record_episode(
        model=model,
        env=env,
        level_id="test_level",
        level_data=level_data,
        agent_id=agent_id,
        run_number=1,
    )

    # Get episode total_reward
    cursor = recorder.conn.execute("SELECT total_reward FROM episodes WHERE id = ?", (episode_id,))
    total_reward = cursor.fetchone()[0]

    # Sum step rewards
    cursor = recorder.conn.execute(
        "SELECT SUM(reward) FROM steps WHERE episode_id = ?", (episode_id,)
    )
    step_reward_sum = cursor.fetchone()[0]

    assert abs(total_reward - step_reward_sum) < 1e-6


def test_deterministic_replay(recorder, trained_model):
    model, env = trained_model

    # Reset twice with same seed and compare initial states
    obs1, info1 = env.reset(seed=123)
    state1 = info1.copy()

    obs2, info2 = env.reset(seed=123)
    state2 = info2.copy()

    assert state1["agent_pos"] == state2["agent_pos"]
    assert state1["has_key"] == state2["has_key"]
    assert (obs1 == obs2).all()


def test_multiple_runs_recorded(recorder, trained_model):
    model, env = trained_model
    agent_id = recorder.register_agent("test_agent", 50)

    level_data = {
        "id": "test_level",
        "name": "Test Level",
        "width": 10,
        "height": 10,
        "grid": env._grid,
        "agentStart": list(env._agent_pos),
    }

    for run in range(3):
        env.reset(seed=run)
        recorder.record_episode(
            model=model,
            env=env,
            level_id="test_level",
            level_data=level_data,
            agent_id=agent_id,
            run_number=run + 1,
        )

    cursor = recorder.conn.execute("SELECT COUNT(*) FROM episodes WHERE agent_id = ?", (agent_id,))
    count = cursor.fetchone()[0]
    assert count == 3


# ---------------------------------------------------------------------------
# Mock model for deterministic tests (avoids DQN training overhead)
# ---------------------------------------------------------------------------


class _FakeQNet:
    """Returns fixed Q-values that always choose a specific action."""

    def __init__(self, action: int, n_actions: int = 6):
        self.action = action
        self.n_actions = n_actions

    def __call__(self, obs_tensor):
        batch = obs_tensor.shape[0]
        q = torch.full((batch, self.n_actions), -1.0)
        q[:, self.action] = 1.0
        return q


class _FakeModel:
    """Minimal stand-in for an SB3 DQN model used by Recorder.record_episode."""

    def __init__(self, action: int):
        self.device = torch.device("cpu")
        self.policy = type("_P", (), {"q_net": _FakeQNet(action)})()


def _make_corridor_env():
    """5×5 open corridor: agent at (1,1), goal at (3,1), all floor except walls."""
    # Layout (y down, x right):
    #  W W W W W
    #  W A . G W
    #  W . . . W
    #  W . . . W
    #  W . . . W
    env = BoxworldEnv(width=5, height=5)
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 4, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    env._grid = [list(row) for row in grid]
    env._agent_pos = [1, 1]
    env._has_key = False
    env._steps = 0
    env._last_direction = BoxworldEnv.UP
    return env, grid


def test_agent_position_changes_during_recording(recorder):
    """Agent must visibly move — agentPosition must differ across steps."""
    env, grid = _make_corridor_env()
    # FakeModel always picks RIGHT (action=3). Agent starts at (1,1), should move right.
    model = _FakeModel(action=BoxworldEnv.RIGHT)
    agent_id = recorder.register_agent("fake_right", 0)

    level_data = {
        "id": "corridor",
        "name": "Corridor",
        "width": 5,
        "height": 5,
        "grid": grid,
        "agentStart": [1, 1],
    }

    episode_id = recorder.record_episode(
        model=model,
        env=env,
        level_id="corridor",
        level_data=level_data,
        agent_id=agent_id,
        run_number=1,
    )

    cursor = recorder.conn.execute(
        "SELECT state_json FROM steps WHERE episode_id = ? ORDER BY step_number",
        (episode_id,),
    )
    positions = [json.loads(row[0])["agentPosition"] for row in cursor.fetchall()]

    # Agent starts at [1,1] and moves RIGHT each step → should reach [3,1] (Goal).
    # We must see at least 2 distinct positions.
    unique_positions = {tuple(p) for p in positions}
    assert len(unique_positions) >= 2, (
        f"Agent appears stationary — only positions seen: {unique_positions}"
    )


def test_grid_reflects_key_pickup(recorder):
    """After picking up a key, the grid cell must change from Key(3) to Floor(0)."""
    # 5×5: agent at (1,2), key at (2,2), goal far away at (3,1)
    # Layout:
    #  W W W W W
    #  W . . G W
    #  W A K . W
    #  W . . . W
    #  W W W W W
    env = BoxworldEnv(width=5, height=5)
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 4, 1],
        [1, 0, 3, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    env._grid = [list(row) for row in grid]
    env._agent_pos = [1, 2]
    env._has_key = False
    env._steps = 0
    env._last_direction = BoxworldEnv.UP

    # Step 1: move RIGHT → agent lands on (2,2) where key is
    # Step 2: PICKUP → key consumed, grid[2][2] becomes Floor
    # After that the model keeps moving RIGHT (action 3) toward goal
    call_count = 0

    class _PickupThenRight:
        def __call__(self, obs_tensor):
            nonlocal call_count
            batch = obs_tensor.shape[0]
            q = torch.full((batch, 6), -1.0)
            if call_count == 0:
                q[:, BoxworldEnv.RIGHT] = 1.0  # move onto key
            elif call_count == 1:
                q[:, BoxworldEnv.PICKUP] = 1.0  # pick up key
            else:
                q[:, BoxworldEnv.RIGHT] = 1.0  # continue toward goal
            call_count += 1
            return q

    model = _FakeModel(action=0)  # dummy; overridden below
    model.policy.q_net = _PickupThenRight()

    agent_id = recorder.register_agent("fake_pickup", 0)
    level_data = {
        "id": "key_test",
        "name": "Key Test",
        "width": 5,
        "height": 5,
        "grid": grid,
        "agentStart": [1, 2],
    }

    episode_id = recorder.record_episode(
        model=model,
        env=env,
        level_id="key_test",
        level_data=level_data,
        agent_id=agent_id,
        run_number=1,
    )

    cursor = recorder.conn.execute(
        "SELECT step_number, state_json FROM steps WHERE episode_id = ? ORDER BY step_number",
        (episode_id,),
    )
    rows = cursor.fetchall()

    # Find grid[2][2] (the key cell) across steps
    key_cell_values = {}
    for step_num, state_json in rows:
        state = json.loads(state_json)
        key_cell_values[step_num] = state["level"]["grid"][2][2]

    # Before pickup the cell should be KEY(3), after pickup it should be FLOOR(0)
    first_step_val = key_cell_values[min(key_cell_values)]
    assert first_step_val == 3, f"First step should show KEY(3) at grid[2][2], got {first_step_val}"

    # After pickup (step 2 records state *before* the pickup action, step 3 records
    # state *after* pickup was executed), so from step 3 onward the cell should be FLOOR.
    last_step_val = key_cell_values[max(key_cell_values)]
    assert last_step_val == 0, f"Final step should show FLOOR(0) at grid[2][2], got {last_step_val}"


def test_first_step_matches_agent_start(recorder):
    """step_number=1's agentPosition must match agentStart from level_data."""
    env, grid = _make_corridor_env()
    model = _FakeModel(action=BoxworldEnv.RIGHT)
    agent_id = recorder.register_agent("fake_start", 0)

    agent_start = [1, 1]
    level_data = {
        "id": "corridor",
        "name": "Corridor",
        "width": 5,
        "height": 5,
        "grid": grid,
        "agentStart": agent_start,
    }

    episode_id = recorder.record_episode(
        model=model,
        env=env,
        level_id="corridor",
        level_data=level_data,
        agent_id=agent_id,
        run_number=1,
    )

    cursor = recorder.conn.execute(
        "SELECT state_json FROM steps WHERE episode_id = ? AND step_number = 1",
        (episode_id,),
    )
    row = cursor.fetchone()
    assert row is not None, "No step_number=1 found"

    state = json.loads(row[0])
    assert state["agentPosition"] == agent_start, (
        f"First step agentPosition {state['agentPosition']} != agentStart {agent_start}"
    )
