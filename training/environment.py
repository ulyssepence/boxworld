"""Boxworld grid environment for RL training."""

from __future__ import annotations

import json

import gymnasium
import numpy as np
from gymnasium import spaces


class BoxworldEnv(gymnasium.Env):
    """Boxworld grid environment for RL training.

    The grid uses row-major indexing: grid[y][x], matching the TypeScript version.

    Observation: flattened grid (width*height) + agent_x + agent_y + has_key
    Action space: UP=0, DOWN=1, LEFT=2, RIGHT=3, PICKUP=4, TOGGLE=5
    """

    metadata = {"render_modes": []}

    # Cell types
    FLOOR = 0
    WALL = 1
    DOOR = 2
    KEY = 3
    GOAL = 4
    LAVA = 5

    # Actions
    UP = 0  # y -= 1
    DOWN = 1  # y += 1
    LEFT = 2  # x -= 1
    RIGHT = 3  # x += 1
    PICKUP = 4
    TOGGLE = 5

    # Rewards
    REWARD_GOAL = 1.0
    REWARD_LAVA = -1.0
    REWARD_STEP = -0.01

    MAX_STEPS = 200

    def __init__(self, level_path: str | None = None, width: int = 10, height: int = 10):
        super().__init__()

        self._level_path = level_path
        self._width = width
        self._height = height

        # Will be set during reset
        self._grid: list[list[int]] = []
        self._agent_pos: list[int] = [0, 0]  # [x, y]
        self._has_key: bool = False
        self._steps: int = 0
        self._last_direction: int = self.UP  # default facing direction

        # If level_path provided, load it to get correct dimensions for spaces
        if level_path is not None:
            self._load_level(level_path)

        obs_size = self._width * self._height + 3
        self.observation_space = spaces.Box(low=0.0, high=5.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        If a level_path was provided at construction, the level is reloaded from JSON.
        Otherwise, a procedural level is generated using the given seed.
        """
        super().reset(seed=seed)

        self._has_key = False
        self._steps = 0
        self._last_direction = self.UP

        if self._level_path is not None:
            self._load_level(self._level_path)
        else:
            self._generate_level(seed)

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self._steps += 1
        reward = self.REWARD_STEP
        terminated = False
        truncated = False

        if action in (self.UP, self.DOWN, self.LEFT, self.RIGHT):
            reward, terminated = self._handle_move(action)
        elif action == self.PICKUP:
            self._handle_pickup()
        elif action == self.TOGGLE:
            self._handle_toggle()

        if self._steps >= self.MAX_STEPS and not terminated:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _handle_move(self, action: int) -> tuple[float, bool]:
        """Handle movement actions. Returns (reward, terminated)."""
        dx, dy = self._action_to_delta(action)
        self._last_direction = action

        new_x = self._agent_pos[0] + dx
        new_y = self._agent_pos[1] + dy

        # Bounds check
        if not (0 <= new_x < self._width and 0 <= new_y < self._height):
            return self.REWARD_STEP, False

        cell = self._grid[new_y][new_x]

        # Blocked by wall or closed door
        if cell == self.WALL or cell == self.DOOR:
            return self.REWARD_STEP, False

        # Move the agent
        self._agent_pos = [new_x, new_y]

        # Check what we stepped onto
        if cell == self.GOAL:
            return self.REWARD_GOAL, True
        elif cell == self.LAVA:
            return self.REWARD_LAVA, True

        return self.REWARD_STEP, False

    def _handle_pickup(self) -> None:
        """Handle the pickup action."""
        x, y = self._agent_pos
        if self._grid[y][x] == self.KEY:
            self._has_key = True
            self._grid[y][x] = self.FLOOR

    def _handle_toggle(self) -> None:
        """Handle the toggle action. Opens a door adjacent in the last move direction if has_key."""
        if not self._has_key:
            return

        dx, dy = self._action_to_delta(self._last_direction)
        door_x = self._agent_pos[0] + dx
        door_y = self._agent_pos[1] + dy

        # Bounds check
        if not (0 <= door_x < self._width and 0 <= door_y < self._height):
            return

        if self._grid[door_y][door_x] == self.DOOR:
            self._grid[door_y][door_x] = self.FLOOR

    def _get_obs(self) -> np.ndarray:
        """Build the observation array: flattened grid + agent_x + agent_y + has_key."""
        flat_grid = []
        for row in self._grid:
            flat_grid.extend(row)

        obs = flat_grid + [
            float(self._agent_pos[0]),
            float(self._agent_pos[1]),
            1.0 if self._has_key else 0.0,
        ]
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict:
        """Return auxiliary info dict."""
        return {
            "agent_pos": list(self._agent_pos),
            "has_key": self._has_key,
            "steps": self._steps,
        }

    def _load_level(self, path: str) -> None:
        """Load a level from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        self._width = data["width"]
        self._height = data["height"]
        # Deep copy the grid so we can mutate it freely
        self._grid = [list(row) for row in data["grid"]]
        self._agent_pos = list(data["agentStart"])

    def _generate_level(self, seed: int | None) -> None:
        """Generate a procedural level with seeded numpy RNG.

        Creates a width x height room with:
        - Walls around the border
        - Some random internal walls
        - One key, one goal placed on floor tiles
        - Agent start on a floor tile
        """
        rng = np.random.default_rng(seed)

        w, h = self._width, self._height

        # Start with all floor
        self._grid = [[self.FLOOR for _ in range(w)] for _ in range(h)]

        # Add border walls
        for x in range(w):
            self._grid[0][x] = self.WALL
            self._grid[h - 1][x] = self.WALL
        for y in range(h):
            self._grid[y][0] = self.WALL
            self._grid[y][w - 1] = self.WALL

        # Add some random internal walls (roughly 10% of interior cells)
        interior_cells = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                interior_cells.append((x, y))

        num_walls = max(1, len(interior_cells) // 10)
        wall_indices = rng.choice(len(interior_cells), size=num_walls, replace=False)
        for idx in wall_indices:
            x, y = interior_cells[idx]
            self._grid[y][x] = self.WALL

        # Collect remaining floor cells for placing objects and agent
        floor_cells = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if self._grid[y][x] == self.FLOOR:
                    floor_cells.append((x, y))

        # Need at least 3 floor cells: agent, key, goal
        if len(floor_cells) < 3:
            # Fallback: clear some walls
            for y in range(1, min(4, h - 1)):
                for x in range(1, min(4, w - 1)):
                    self._grid[y][x] = self.FLOOR
            floor_cells = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if self._grid[y][x] == self.FLOOR:
                        floor_cells.append((x, y))

        # Shuffle and assign positions
        chosen_indices = rng.choice(len(floor_cells), size=3, replace=False)
        agent_x, agent_y = floor_cells[chosen_indices[0]]
        key_x, key_y = floor_cells[chosen_indices[1]]
        goal_x, goal_y = floor_cells[chosen_indices[2]]

        self._grid[key_y][key_x] = self.KEY
        self._grid[goal_y][goal_x] = self.GOAL
        self._agent_pos = [agent_x, agent_y]

    @staticmethod
    def _action_to_delta(action: int) -> tuple[int, int]:
        """Convert an action to (dx, dy)."""
        if action == BoxworldEnv.UP:
            return (0, -1)
        elif action == BoxworldEnv.DOWN:
            return (0, 1)
        elif action == BoxworldEnv.LEFT:
            return (-1, 0)
        elif action == BoxworldEnv.RIGHT:
            return (1, 0)
        return (0, 0)
