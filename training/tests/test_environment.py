"""Tests for the BoxworldEnv gymnasium environment."""

import json
import os
import tempfile

import numpy as np
import pytest

from environment import BoxworldEnv


# ---------------------------------------------------------------------------
# Basic space / shape tests
# ---------------------------------------------------------------------------


def test_reset_returns_valid_observation():
    env = BoxworldEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_observation_space_shape():
    env = BoxworldEnv()
    assert env.observation_space.shape == (103,)  # 10*10 + 3


def test_action_space_size():
    env = BoxworldEnv()
    assert env.action_space.n == 6


def test_observation_values_in_bounds():
    env = BoxworldEnv()
    obs, _ = env.reset(seed=42)
    assert np.all(obs >= 0.0)
    # Grid cells 0-5, agent_x/y up to 9, has_key 0/1 -- all <= 9
    assert np.all(obs <= 9.0)


# ---------------------------------------------------------------------------
# Movement tests
# ---------------------------------------------------------------------------


def _make_simple_env():
    """Create an env with a known 5x5 layout for movement tests.

    Layout (grid[y][x]):
        y=0: W W W W W
        y=1: W . . . W
        y=2: W . A . W
        y=3: W . . . W
        y=4: W W W W W

    Agent at (2, 2), all interior = floor.
    """
    env = BoxworldEnv(width=5, height=5)
    env.reset(seed=0)
    # Override grid
    env._grid = [[BoxworldEnv.WALL] * 5 for _ in range(5)]
    for y in range(1, 4):
        for x in range(1, 4):
            env._grid[y][x] = BoxworldEnv.FLOOR
    env._agent_pos = [2, 2]
    env._has_key = False
    return env


def test_move_to_empty_cell():
    env = _make_simple_env()
    # Move UP from (2,2) -> (2,1)
    obs, reward, terminated, truncated, info = env.step(BoxworldEnv.UP)
    assert info["agent_pos"] == [2, 1]
    assert reward == pytest.approx(-0.01)
    assert not terminated
    assert not truncated


def test_move_down():
    env = _make_simple_env()
    env.step(BoxworldEnv.DOWN)
    assert env._agent_pos == [2, 3]


def test_move_left():
    env = _make_simple_env()
    env.step(BoxworldEnv.LEFT)
    assert env._agent_pos == [1, 2]


def test_move_right():
    env = _make_simple_env()
    env.step(BoxworldEnv.RIGHT)
    assert env._agent_pos == [3, 2]


def test_move_into_wall():
    env = _make_simple_env()
    # Move agent to (1,1), then try to move UP into the wall at y=0
    env._agent_pos = [1, 1]
    env.step(BoxworldEnv.UP)
    assert env._agent_pos == [1, 1]  # blocked


def test_move_into_wall_left():
    env = _make_simple_env()
    env._agent_pos = [1, 1]
    env.step(BoxworldEnv.LEFT)
    assert env._agent_pos == [1, 1]  # blocked


# ---------------------------------------------------------------------------
# Key / Pickup tests
# ---------------------------------------------------------------------------


def test_pickup_key():
    env = _make_simple_env()
    # Place a key at agent position
    env._grid[2][2] = BoxworldEnv.KEY
    env.step(BoxworldEnv.PICKUP)
    assert env._has_key is True
    assert env._grid[2][2] == BoxworldEnv.FLOOR


def test_pickup_without_key():
    env = _make_simple_env()
    # Agent on floor, pickup should do nothing
    assert env._grid[2][2] == BoxworldEnv.FLOOR
    env.step(BoxworldEnv.PICKUP)
    assert env._has_key is False


def test_pickup_key_only_on_key_cell():
    env = _make_simple_env()
    # Key is at (1,1), agent at (2,2) - pickup should not work
    env._grid[1][1] = BoxworldEnv.KEY
    env.step(BoxworldEnv.PICKUP)
    assert env._has_key is False
    assert env._grid[1][1] == BoxworldEnv.KEY


# ---------------------------------------------------------------------------
# Door / Toggle tests
# ---------------------------------------------------------------------------


def test_toggle_door_with_key():
    env = _make_simple_env()
    env._has_key = True
    # Place door above agent: grid[1][2] = DOOR
    env._grid[1][2] = BoxworldEnv.DOOR
    # Set last direction to UP so toggle looks at (2, 1)
    env._last_direction = BoxworldEnv.UP
    env.step(BoxworldEnv.TOGGLE)
    assert env._grid[1][2] == BoxworldEnv.FLOOR


def test_toggle_door_without_key():
    env = _make_simple_env()
    env._has_key = False
    env._grid[1][2] = BoxworldEnv.DOOR
    env._last_direction = BoxworldEnv.UP
    env.step(BoxworldEnv.TOGGLE)
    assert env._grid[1][2] == BoxworldEnv.DOOR  # door stays


def test_toggle_uses_last_direction():
    env = _make_simple_env()
    env._has_key = True
    # Place door to the RIGHT of agent at (3, 2)
    env._grid[2][3] = BoxworldEnv.DOOR
    # Set last direction to RIGHT
    env._last_direction = BoxworldEnv.RIGHT
    env.step(BoxworldEnv.TOGGLE)
    assert env._grid[2][3] == BoxworldEnv.FLOOR


def test_move_blocked_by_closed_door():
    env = _make_simple_env()
    # Place door above agent
    env._grid[1][2] = BoxworldEnv.DOOR
    env.step(BoxworldEnv.UP)
    # Agent should not have moved
    assert env._agent_pos == [2, 2]


def test_move_through_opened_door():
    env = _make_simple_env()
    env._has_key = True
    env._grid[1][2] = BoxworldEnv.DOOR
    env._last_direction = BoxworldEnv.UP
    # Toggle to open door
    env.step(BoxworldEnv.TOGGLE)
    assert env._grid[1][2] == BoxworldEnv.FLOOR
    # Now move up
    env.step(BoxworldEnv.UP)
    assert env._agent_pos == [2, 1]


# ---------------------------------------------------------------------------
# Goal and Lava tests
# ---------------------------------------------------------------------------


def test_reach_goal():
    env = _make_simple_env()
    # Place goal above agent
    env._grid[1][2] = BoxworldEnv.GOAL
    obs, reward, terminated, truncated, info = env.step(BoxworldEnv.UP)
    assert reward == pytest.approx(1.0)
    assert terminated is True
    assert truncated is False


def test_step_onto_lava():
    env = _make_simple_env()
    # Place lava below agent
    env._grid[3][2] = BoxworldEnv.LAVA
    obs, reward, terminated, truncated, info = env.step(BoxworldEnv.DOWN)
    assert reward == pytest.approx(-1.0)
    assert terminated is True
    assert truncated is False


def test_step_reward_is_negative():
    """Normal step should cost -0.01."""
    env = _make_simple_env()
    _, reward, _, _, _ = env.step(BoxworldEnv.UP)
    assert reward == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# Deterministic seed tests
# ---------------------------------------------------------------------------


def test_deterministic_seed():
    env1 = BoxworldEnv()
    obs1, _ = env1.reset(seed=123)

    env2 = BoxworldEnv()
    obs2, _ = env2.reset(seed=123)

    np.testing.assert_array_equal(obs1, obs2)


def test_different_seeds_produce_different_layouts():
    env = BoxworldEnv()
    obs1, _ = env.reset(seed=1)
    obs2, _ = env.reset(seed=2)
    # Extremely unlikely to be identical
    assert not np.array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Level loading from JSON tests
# ---------------------------------------------------------------------------


def test_load_level_from_json():
    level_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "levels", "four_rooms.json"
    )
    level_path = os.path.normpath(level_path)
    env = BoxworldEnv(level_path=level_path)
    obs, info = env.reset(seed=0)
    assert env._width == 10
    assert env._height == 10
    assert obs.shape == (103,)
    assert info["agent_pos"] == [6, 1]


def test_load_level_grid_matches_json():
    level_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "levels", "four_rooms.json"
    )
    level_path = os.path.normpath(level_path)
    env = BoxworldEnv(level_path=level_path)
    env.reset()

    with open(level_path) as f:
        data = json.load(f)

    for y in range(env._height):
        for x in range(env._width):
            assert env._grid[y][x] == data["grid"][y][x]


def test_load_custom_json_level():
    """Create a temp JSON level and load it."""
    level_data = {
        "id": "test_level",
        "name": "Test Level",
        "width": 4,
        "height": 4,
        "grid": [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 4, 1],
            [1, 1, 1, 1],
        ],
        "agentStart": [1, 1],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(level_data, f)
        tmp_path = f.name

    try:
        env = BoxworldEnv(level_path=tmp_path)
        obs, info = env.reset()
        assert env._width == 4
        assert env._height == 4
        assert obs.shape == (4 * 4 + 3,)
        assert info["agent_pos"] == [1, 1]
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Max steps / truncation tests
# ---------------------------------------------------------------------------


def test_max_steps_truncation():
    env = _make_simple_env()
    # Run 200 steps doing nothing harmful (toggle on floor with no door)
    for i in range(199):
        obs, reward, terminated, truncated, info = env.step(BoxworldEnv.PICKUP)
        assert not truncated, f"Truncated too early at step {i + 1}"
        assert not terminated

    # Step 200 should trigger truncation
    obs, reward, terminated, truncated, info = env.step(BoxworldEnv.PICKUP)
    assert truncated is True
    assert terminated is False


def test_terminated_prevents_truncation():
    """If agent reaches goal before 200 steps, truncated should be False."""
    env = _make_simple_env()
    env._grid[1][2] = BoxworldEnv.GOAL
    _, _, terminated, truncated, _ = env.step(BoxworldEnv.UP)
    assert terminated is True
    assert truncated is False


# ---------------------------------------------------------------------------
# Observation content tests
# ---------------------------------------------------------------------------


def test_observation_encodes_agent_position():
    env = _make_simple_env()
    obs = env._get_obs()
    # Last 3 values: agent_x, agent_y, has_key
    assert obs[-3] == pytest.approx(2.0)  # agent_x
    assert obs[-2] == pytest.approx(2.0)  # agent_y
    assert obs[-1] == pytest.approx(0.0)  # has_key


def test_observation_encodes_has_key():
    env = _make_simple_env()
    env._has_key = True
    obs = env._get_obs()
    assert obs[-1] == pytest.approx(1.0)


def test_observation_grid_portion():
    env = _make_simple_env()
    obs = env._get_obs()
    grid_obs = obs[: 5 * 5]
    # Check corners are walls
    assert grid_obs[0] == pytest.approx(float(BoxworldEnv.WALL))  # (0,0)
    assert grid_obs[4] == pytest.approx(float(BoxworldEnv.WALL))  # (4,0)
    # Check interior is floor
    assert grid_obs[6] == pytest.approx(float(BoxworldEnv.FLOOR))  # (1,1)


# ---------------------------------------------------------------------------
# Info dict tests
# ---------------------------------------------------------------------------


def test_info_contains_expected_keys():
    env = BoxworldEnv()
    _, info = env.reset(seed=42)
    assert "agent_pos" in info
    assert "has_key" in info
    assert "steps" in info


def test_info_steps_increments():
    env = _make_simple_env()
    _, info = env.reset(seed=0)
    # After reset, override grid to simple layout again
    env._grid = [[BoxworldEnv.WALL] * 5 for _ in range(5)]
    for y in range(1, 4):
        for x in range(1, 4):
            env._grid[y][x] = BoxworldEnv.FLOOR
    env._agent_pos = [2, 2]

    _, _, _, _, info = env.step(BoxworldEnv.UP)
    assert info["steps"] == 1
    _, _, _, _, info = env.step(BoxworldEnv.DOWN)
    assert info["steps"] == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_reset_clears_state():
    env = _make_simple_env()
    env._has_key = True
    env._steps = 100
    env.reset(seed=99)
    assert env._has_key is False
    assert env._steps == 0


def test_step_onto_key_does_not_auto_pickup():
    """Moving onto a key cell should NOT automatically pick it up."""
    env = _make_simple_env()
    env._grid[1][2] = BoxworldEnv.KEY
    env.step(BoxworldEnv.UP)  # move onto key
    assert env._agent_pos == [2, 1]
    assert env._has_key is False
    assert env._grid[1][2] == BoxworldEnv.KEY  # key still there


def test_custom_dimensions():
    env = BoxworldEnv(width=15, height=8)
    obs, _ = env.reset(seed=42)
    assert env._width == 15
    assert env._height == 8
    assert obs.shape == (15 * 8 + 3,)
    assert env.observation_space.shape == (15 * 8 + 3,)


def test_goal_distance_uses_bfs_not_manhattan():
    """BFS distance should account for walls, not just Manhattan distance."""
    env = BoxworldEnv(width=5, height=5)
    env.reset(seed=0)
    # Layout:
    #   W W W W W
    #   W A W G W
    #   W . . . W
    #   W W W W W
    #   W W W W W
    env._grid = [[BoxworldEnv.WALL] * 5 for _ in range(5)]
    env._grid[1][1] = BoxworldEnv.FLOOR  # agent
    env._grid[1][3] = BoxworldEnv.GOAL  # goal
    env._grid[2][1] = BoxworldEnv.FLOOR
    env._grid[2][2] = BoxworldEnv.FLOOR
    env._grid[2][3] = BoxworldEnv.FLOOR
    env._agent_pos = [1, 1]

    # Manhattan distance would be 2, but wall at (2,1) blocks direct path
    # BFS path: (1,1) -> (1,2) -> (2,2) -> (3,2) -> (3,1) = 4 steps
    assert env._goal_distance() == 4.0


# ---------------------------------------------------------------------------
# Door-aware BFS shaping tests (G)
# ---------------------------------------------------------------------------


def _make_door_env():
    """Create a 5x5 env with agent, door, key, and goal behind the door.

    Layout (grid[y][x]):
        y=0: W W W W W
        y=1: W A . K W
        y=2: W W D W W
        y=3: W . G . W
        y=4: W W W W W

    Agent at (1,1), Key at (3,1), Door at (2,2), Goal at (2,3).
    """
    env = BoxworldEnv(width=5, height=5)
    env.reset(seed=0)
    env._grid = [[BoxworldEnv.WALL] * 5 for _ in range(5)]
    env._grid[1][1] = BoxworldEnv.FLOOR  # agent
    env._grid[1][2] = BoxworldEnv.FLOOR
    env._grid[1][3] = BoxworldEnv.KEY
    env._grid[2][2] = BoxworldEnv.DOOR
    env._grid[3][1] = BoxworldEnv.FLOOR
    env._grid[3][2] = BoxworldEnv.GOAL
    env._grid[3][3] = BoxworldEnv.FLOOR
    env._agent_pos = [1, 1]
    env._has_key = False
    env._steps = 0
    env._last_direction = BoxworldEnv.UP
    return env


def test_goal_distance_without_key_blocked_by_door():
    """Without a key, BFS should not pass through doors."""
    env = _make_door_env()
    assert env._has_key is False
    # Goal at (2,3) is behind door at (2,2) — unreachable without key
    assert env._goal_distance() is None


def test_goal_distance_with_key_passes_through_door():
    """With a key, BFS should treat doors as passable."""
    env = _make_door_env()
    env._has_key = True
    # Agent at (1,1) -> (2,1) -> through door (2,2) -> goal (2,3) = 3 steps
    assert env._goal_distance() == 3.0


def test_key_distance_finds_nearest_key():
    """_key_distance should return BFS distance to nearest key."""
    env = _make_door_env()
    # Agent at (1,1), key at (3,1), path: (1,1) -> (2,1) -> (3,1) = 2
    assert env._key_distance() == 2.0


def test_key_distance_returns_none_when_no_key():
    """_key_distance returns None when no key exists."""
    env = _make_simple_env()
    assert env._key_distance() is None


def test_key_distance_zero_when_standing_on_key():
    """_key_distance returns 0 when agent is on a key cell."""
    env = _make_simple_env()
    env._grid[2][2] = BoxworldEnv.KEY
    assert env._key_distance() == 0.0


def test_shaping_toward_key_when_goal_blocked():
    """When goal is behind a door, shaping should guide toward the key."""
    env = _make_door_env()
    # Agent at (1,1), key at (3,1), goal blocked
    # Move RIGHT toward key: (1,1) -> (2,1)
    _, reward, _, _, _ = env.step(BoxworldEnv.RIGHT)
    # Step penalty -0.01 + shaping toward key: 0.1 * (2 - 1) = +0.1
    assert reward == pytest.approx(-0.01 + 0.1)


def test_shaping_toward_goal_when_has_key():
    """When agent has a key and goal is reachable through door, shape toward goal."""
    env = _make_door_env()
    env._has_key = True
    env._agent_pos = [2, 1]  # one step from door, 2 from goal
    # Move DOWN toward door/goal: (2,1) -> blocked by door
    # Actually the door blocks movement, so agent stays at (2,1)
    # Let's instead test from (1,1) moving right toward (2,1) which is closer to goal
    env._agent_pos = [1, 1]
    # Goal distance with key: (1,1)->(2,1)->(2,2 door)->(2,3 goal) = 3
    assert env._goal_distance() == 3.0
    _, reward, _, _, _ = env.step(BoxworldEnv.RIGHT)
    # Agent moves to (2,1). New goal dist = 2. Old was 3.
    # Step penalty -0.01 + shaping: 0.1 * (3 - 2) = +0.1
    assert reward == pytest.approx(-0.01 + 0.1)


def test_pickup_reward_bonus():
    """Picking up a key gives +0.2 bonus on top of step penalty."""
    env = _make_simple_env()
    env._grid[2][2] = BoxworldEnv.KEY
    _, reward, _, _, _ = env.step(BoxworldEnv.PICKUP)
    # Step penalty -0.01 + pickup bonus +0.2, no shaping (no goal)
    assert reward == pytest.approx(-0.01 + 0.2)
    assert env._has_key is True


def test_toggle_reward_bonus():
    """Opening a door gives +0.2 bonus and consumes the key."""
    env = _make_simple_env()
    env._has_key = True
    env._grid[1][2] = BoxworldEnv.DOOR
    env._last_direction = BoxworldEnv.UP
    _, reward, _, _, _ = env.step(BoxworldEnv.TOGGLE)
    # Step penalty -0.01 + toggle bonus +0.2, no shaping (no goal)
    assert reward == pytest.approx(-0.01 + 0.2)
    assert env._grid[1][2] == BoxworldEnv.FLOOR
    assert env._has_key is False  # key consumed


def test_toggle_consumes_key():
    """Toggle should consume the key after opening a door."""
    env = _make_simple_env()
    env._has_key = True
    env._grid[1][2] = BoxworldEnv.DOOR
    env._last_direction = BoxworldEnv.UP
    env.step(BoxworldEnv.TOGGLE)
    assert env._has_key is False


def test_toggle_without_door_no_bonus():
    """Toggle with key but no adjacent door gives no bonus."""
    env = _make_simple_env()
    env._has_key = True
    env._last_direction = BoxworldEnv.UP
    # No door above agent — just wall
    _, reward, _, _, _ = env.step(BoxworldEnv.TOGGLE)
    assert reward == pytest.approx(-0.01)
    assert env._has_key is True  # key not consumed


# ---------------------------------------------------------------------------
# Procedural maze generator tests (B)
# ---------------------------------------------------------------------------


def test_generated_maze_is_solvable():
    """Generated maze should have a reachable goal (possibly via key+door)."""
    env = BoxworldEnv()
    for seed in range(20):
        env.reset(seed=seed)
        goal = _find_cell(env, BoxworldEnv.GOAL)

        # Check if goal is directly reachable
        dist = env._bfs_distance(env._agent_pos[0], env._agent_pos[1], goal[0], goal[1])
        if dist is not None:
            continue

        # If not, check key+door solvability: key reachable, and goal reachable with key
        key_dist = env._key_distance()
        assert key_dist is not None, f"Seed {seed}: goal blocked by door but no reachable key"
        env._has_key = True
        goal_dist = env._goal_distance()
        env._has_key = False
        assert goal_dist is not None, f"Seed {seed}: goal unreachable even with key"


def test_generated_maze_has_corridors():
    """Generated maze should have wall cells in the interior (not just border)."""
    env = BoxworldEnv()
    env.reset(seed=42)
    interior_walls = 0
    for y in range(1, env._height - 1):
        for x in range(1, env._width - 1):
            if env._grid[y][x] == BoxworldEnv.WALL:
                interior_walls += 1
    # A maze should have many interior walls (open room had ~10%)
    # Recursive backtracker on 10x10 produces ~30-40 interior walls
    assert interior_walls > 10, f"Only {interior_walls} interior walls — not maze-like"


def test_generated_maze_minimum_distance():
    """Most generated mazes should have agent-to-goal BFS distance >= 5."""
    env = BoxworldEnv()
    short_count = 0
    for seed in range(50):
        env.reset(seed=seed)
        goal = _find_cell(env, BoxworldEnv.GOAL)
        dist = env._bfs_distance(env._agent_pos[0], env._agent_pos[1], goal[0], goal[1])
        if dist is not None and dist < 5:
            short_count += 1
    # Allow some fallback cases but most should meet the threshold
    assert short_count < 10, f"{short_count}/50 mazes had distance < 5"


def test_generated_maze_sometimes_has_doors():
    """Some generated mazes should include doors and keys (~30% chance)."""
    env = BoxworldEnv()
    door_count = 0
    for seed in range(100):
        env.reset(seed=seed)
        has_door = any(
            env._grid[y][x] == BoxworldEnv.DOOR
            for y in range(env._height)
            for x in range(env._width)
        )
        if has_door:
            door_count += 1
    # With 30% chance, expect ~30 but allow variance
    assert door_count >= 5, f"Only {door_count}/100 mazes had doors"


def test_generated_maze_sometimes_has_lava():
    """Some generated mazes should include lava (~20% chance)."""
    env = BoxworldEnv()
    lava_count = 0
    for seed in range(100):
        env.reset(seed=seed)
        has_lava = any(
            env._grid[y][x] == BoxworldEnv.LAVA
            for y in range(env._height)
            for x in range(env._width)
        )
        if has_lava:
            lava_count += 1
    assert lava_count >= 3, f"Only {lava_count}/100 mazes had lava"


def test_generated_maze_deterministic():
    """Same seed should produce the same maze."""
    env1 = BoxworldEnv()
    obs1, _ = env1.reset(seed=42)
    env2 = BoxworldEnv()
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


def test_generated_maze_door_has_key():
    """When a maze has a door, it should also have a key on the agent's side."""
    env = BoxworldEnv()
    for seed in range(100):
        env.reset(seed=seed)
        has_door = any(
            env._grid[y][x] == BoxworldEnv.DOOR
            for y in range(env._height)
            for x in range(env._width)
        )
        has_key = any(
            env._grid[y][x] == BoxworldEnv.KEY
            for y in range(env._height)
            for x in range(env._width)
        )
        if has_door:
            assert has_key, f"Seed {seed}: has door but no key"


def _find_cell(env: BoxworldEnv, cell_type: int) -> tuple[int, int]:
    """Find the first cell of the given type."""
    for y in range(env._height):
        for x in range(env._width):
            if env._grid[y][x] == cell_type:
                return (x, y)
    raise ValueError(f"Cell type {cell_type} not found")
