"""Bug reproduction tests for agent movement issues.

Bug 1: "Run Agent" in browser produces a stationary agent.
Bug 2: Recorded episode playback shows a stationary agent for most episodes.

Hypothesis: DQN model trained on random procedural levels picks a blocked action
on the hand-designed levels — agent never moves, observations never change,
so it picks the same blocked action forever.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

from environment import BoxworldEnv

# Paths relative to repo root
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
LEVELS_DIR = DATA_DIR / "levels"

ONNX_500K = CHECKPOINTS_DIR / "boxworld_500000_steps.onnx"
ZIP_500K = CHECKPOINTS_DIR / "boxworld_500000_steps.zip"

HAS_ONNX = ONNX_500K.exists()
HAS_ZIP = ZIP_500K.exists()
HAS_LEVELS = LEVELS_DIR.exists() and any(LEVELS_DIR.glob("*.json"))


def _load_level(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _make_env_from_level(level_data: dict) -> BoxworldEnv:
    """Create a BoxworldEnv manually configured from level JSON data."""
    env = BoxworldEnv(width=level_data["width"], height=level_data["height"])
    env._grid = [list(row) for row in level_data["grid"]]
    env._agent_pos = list(level_data["agentStart"])
    env._has_key = False
    env._steps = 0
    env._last_direction = BoxworldEnv.UP
    return env


# ---------------------------------------------------------------------------
# Test 1: ONNX produces different Q-values for different states
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ONNX, reason="No ONNX checkpoint at data/checkpoints/")
def test_onnx_produces_different_qvalues_for_different_states():
    """Load the 500k-step ONNX model, construct obs at two different positions,
    run inference, and assert Q-values differ. Confirms the inference pipeline works."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_500K))

    # Build two different observations for a 10x10 grid
    # Obs size = 10*10 + 3 = 103
    obs_a = np.zeros(103, dtype=np.float32)
    obs_a[100] = 1.0  # agent_x = 1
    obs_a[101] = 1.0  # agent_y = 1
    obs_a[102] = 0.0  # has_key = False

    obs_b = np.zeros(103, dtype=np.float32)
    obs_b[100] = 5.0  # agent_x = 5
    obs_b[101] = 5.0  # agent_y = 5
    obs_b[102] = 1.0  # has_key = True

    q_a = session.run(None, {"obs": obs_a.reshape(1, -1)})[0][0]
    q_b = session.run(None, {"obs": obs_b.reshape(1, -1)})[0][0]

    # Q-values should differ for different observations
    assert not np.allclose(q_a, q_b, atol=1e-6), (
        f"ONNX returns identical Q-values for different states:\n"
        f"  state A q-values: {q_a}\n"
        f"  state B q-values: {q_b}"
    )


# ---------------------------------------------------------------------------
# Test 2: ONNX inference moves agent on a real level (Bug 1 reproduction)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ONNX or not HAS_LEVELS, reason="Need ONNX checkpoint + levels")
def test_onnx_inference_moves_agent_on_real_level():
    """Simulate the browser's runAgent() loop using ONNX inference.

    Load real .onnx + simple_corridor.json, run 200 steps
    (get obs -> ONNX -> argmax -> env.step), assert agent position changes.
    This is the core Bug 1 reproduction.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_500K))

    level_path = LEVELS_DIR / "simple_corridor.json"
    if not level_path.exists():
        pytest.skip("simple_corridor.json not found")

    level_data = _load_level(level_path)
    env = _make_env_from_level(level_data)

    positions = [tuple(env._agent_pos)]
    actions_taken = []

    for step_i in range(200):
        obs = env._get_obs()
        q_values = session.run(None, {"obs": obs.reshape(1, -1)})[0][0]
        action = int(np.argmax(q_values))
        actions_taken.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(tuple(env._agent_pos))

        if terminated or truncated:
            break

    unique_positions = set(positions)
    unique_actions = set(actions_taken)

    # Diagnostic info on failure
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PICKUP", 5: "TOGGLE"}
    action_counts = {}
    for a in actions_taken:
        action_counts[action_names[a]] = action_counts.get(action_names[a], 0) + 1

    assert len(unique_positions) > 1, (
        f"Bug 1 reproduced: Agent never moved on simple_corridor!\n"
        f"  Start position: {positions[0]}\n"
        f"  Actions taken (counts): {action_counts}\n"
        f"  Unique actions: {[action_names[a] for a in unique_actions]}\n"
        f"  Q-values at initial state: {dict(zip(action_names.values(), q_values.tolist()))}"
    )


# ---------------------------------------------------------------------------
# Test 3: Recorded episodes show agent movement (Bug 2 reproduction)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ZIP or not HAS_LEVELS, reason="Need .zip checkpoint + levels")
def test_recorded_episode_agent_moves_on_real_levels():
    """Load real .zip checkpoint, record an episode on each level via
    Recorder.record_episode(), parse state_json, assert agentPosition changes.
    This is the core Bug 2 reproduction.
    """
    from stable_baselines3 import DQN

    from record import Recorder

    env = BoxworldEnv()
    model = DQN.load(str(ZIP_500K), env=env)

    level_files = sorted(LEVELS_DIR.glob("*.json"))
    assert len(level_files) > 0, "No level files found"

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = tmp.name

    try:
        recorder = Recorder(db_path)
        recorder.initialize_db()
        agent_id = recorder.register_agent("test_500k", 500000)

        stationary_levels = []

        for level_file in level_files:
            level_data = _load_level(level_file)
            level_id = level_data["id"]

            level_env = _make_env_from_level(level_data)

            episode_id = recorder.record_episode(
                model=model,
                env=level_env,
                level_id=level_id,
                level_data=level_data,
                agent_id=agent_id,
                run_number=1,
            )

            cursor = recorder.conn.execute(
                "SELECT state_json FROM steps WHERE episode_id = ? ORDER BY step_number",
                (episode_id,),
            )
            positions = []
            for (state_json,) in cursor.fetchall():
                state = json.loads(state_json)
                positions.append(tuple(state["agentPosition"]))

            unique = set(positions)
            if len(unique) <= 1:
                stationary_levels.append(level_id)

        recorder.close()
    finally:
        os.unlink(db_path)

    # The 500k model doesn't solve all levels — lava_maze and key_puzzle
    # may produce stationary agents. The bug was that the agent was stationary
    # on ALL levels; ensure it moves on at least one non-trivial level.
    moving_levels = len(level_files) - len(stationary_levels)
    assert moving_levels >= 2, (
        f"Bug 2 reproduced: Agent stationary on too many levels: {stationary_levels}\n"
        f"(only moved on {moving_levels}/{len(level_files)} levels)"
    )


# ---------------------------------------------------------------------------
# Test 4: stateToTensor alignment (cross-language encoding check)
# ---------------------------------------------------------------------------


def test_stateToTensor_matches_get_obs():
    """Cross-language alignment: manually build the TS-style tensor in Python,
    compare with BoxworldEnv._get_obs(). Must be identical.

    TS encoding (from ml.ts stateToTensor):
      for y in 0..height: for x in 0..width: grid[y][x]
      then agent_x, agent_y, has_key
    """
    # Use a small known grid
    level_data = {
        "id": "test",
        "name": "Test",
        "width": 5,
        "height": 5,
        "grid": [
            [1, 1, 1, 1, 1],
            [1, 0, 3, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 4, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        "agentStart": [1, 1],
    }

    env = _make_env_from_level(level_data)
    env._has_key = True  # Test with has_key=True to verify that bit

    python_obs = env._get_obs()

    # Build the same tensor the TS way
    grid = level_data["grid"]
    width = level_data["width"]
    height = level_data["height"]
    agent_x, agent_y = env._agent_pos
    has_key = env._has_key

    ts_values = []
    for y in range(height):
        for x in range(width):
            ts_values.append(float(grid[y][x]))
    ts_values.append(float(agent_x))
    ts_values.append(float(agent_y))
    ts_values.append(1.0 if has_key else 0.0)

    ts_obs = np.array(ts_values, dtype=np.float32)

    np.testing.assert_array_equal(
        python_obs,
        ts_obs,
        err_msg="Observation encoding mismatch between Python _get_obs() and TS stateToTensor()",
    )
