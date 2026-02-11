"""Boxworld grid environment for RL training."""

from __future__ import annotations

import glob
import os

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
    REWARD_PICKUP = 0.2
    REWARD_TOGGLE = 0.2
    REWARD_DOOR_ADJACENT = 0.0  # disabled — was causing bonus farming

    MAX_STEPS = 200

    def __init__(
        self,
        level_path: str | None = None,
        width: int = 10,
        height: int = 10,
        levels_dir: str | None = None,
        designed_level_prob: float = 0.0,
        level_weights: dict[str, float] | None = None,
        exclude_levels: list[str] | None = None,
    ):
        super().__init__()

        self._level_path = level_path
        self._width = width
        self._height = height
        self._designed_level_prob = designed_level_prob

        # Load hand-designed levels for mixed training
        self._designed_levels: list[dict] = []
        if levels_dir is not None:
            from level_parser import load_level

            for path in sorted(glob.glob(os.path.join(levels_dir, "*.txt"))):
                data = load_level(path)
                # Only include levels matching our grid dimensions
                if data.get("width") == width and data.get("height") == height:
                    self._designed_levels.append(data)

        if exclude_levels:
            self._designed_levels = [
                lv for lv in self._designed_levels if lv["id"] not in exclude_levels
            ]

        # Compute normalized weights for designed level sampling
        self._level_weights: np.ndarray | None = None
        if self._designed_levels and level_weights:
            weights = [level_weights.get(lv["id"], 1.0) for lv in self._designed_levels]
            total = sum(weights)
            self._level_weights = np.array([w / total for w in weights])

        # Will be set during reset
        self._grid: list[list[int]] = []
        self._agent_pos: list[int] = [0, 0]  # [x, y]
        self._has_key: bool = False
        self._steps: int = 0
        self._last_direction: int = self.UP  # default facing direction
        self._subgoals: list[tuple[str, tuple[int, int]]] = []
        self._subgoal_index: int = 0
        self._difficulty: float = 1.0  # 0.0=easy, 1.0=full difficulty (curriculum)

        # If level_path provided, load it to get correct dimensions for spaces
        if level_path is not None:
            self._load_level(level_path)

        obs_size = self._width * self._height + 3
        high = np.full(obs_size, 5.0, dtype=np.float32)
        high[-3] = float(self._width - 1)  # agent_x
        high[-2] = float(self._height - 1)  # agent_y
        high[-1] = 1.0  # has_key
        self.observation_space = spaces.Box(low=0.0, high=high, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        If a level_path was provided at construction, the level is reloaded from JSON.
        Otherwise, with probability designed_level_prob, a hand-designed level is loaded.
        Otherwise, a procedural level is generated using the given seed.
        """
        super().reset(seed=seed)

        self._has_key = False
        self._steps = 0
        self._last_direction = self.UP

        if self._level_path is not None:
            self._load_level(self._level_path)
        elif self._designed_levels and self.np_random.random() < self._designed_level_prob:
            if self._level_weights is not None:
                idx = int(self.np_random.choice(len(self._designed_levels), p=self._level_weights))
            else:
                idx = int(self.np_random.integers(len(self._designed_levels)))
            level = self._designed_levels[idx]
            self._grid = [list(row) for row in level["grid"]]
            self._agent_pos = list(level["agentStart"])
        else:
            self._generate_level(seed)

        self._solve_subgoals()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self._steps += 1
        reward = self.REWARD_STEP
        terminated = False
        truncated = False

        old_x, old_y = self._agent_pos
        had_key = self._has_key
        toggled_door = False

        if action in (self.UP, self.DOWN, self.LEFT, self.RIGHT):
            reward, terminated = self._handle_move(action)
        elif action == self.PICKUP:
            self._handle_pickup()
            if not had_key and self._has_key:
                reward += self.REWARD_PICKUP
        elif action == self.TOGGLE:
            if self._handle_toggle():
                reward += self.REWARD_TOGGLE
                toggled_door = True

        # Advance subgoal index on key pickup or door toggle
        if self._subgoal_index < len(self._subgoals):
            sg_type, _ = self._subgoals[self._subgoal_index]
            if sg_type == "key" and not had_key and self._has_key:
                self._subgoal_index += 1
            elif sg_type == "door" and toggled_door:
                self._subgoal_index += 1

        # Subgoal-chain reward shaping (bidirectional: reward closer, penalize farther)
        if not terminated and self._subgoal_index < len(self._subgoals):
            _, subgoal_pos = self._subgoals[self._subgoal_index]
            old_dist = self._bfs_distance_safe(old_x, old_y, *subgoal_pos)
            new_dist = self._bfs_distance_safe(self._agent_pos[0], self._agent_pos[1], *subgoal_pos)
            if old_dist is not None and new_dist is not None and new_dist != old_dist:
                reward += 0.05 * (old_dist - new_dist)

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

    def _handle_toggle(self) -> bool:
        """Handle the toggle action. Opens any adjacent door if has_key.

        Checks all 4 cardinal directions (matching TypeScript play.ts behavior).
        Returns True if a door was opened, False otherwise.
        """
        if not self._has_key:
            return False

        ax, ay = self._agent_pos
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            door_x, door_y = ax + dx, ay + dy
            if not (0 <= door_x < self._width and 0 <= door_y < self._height):
                continue
            if self._grid[door_y][door_x] == self.DOOR:
                self._grid[door_y][door_x] = self.FLOOR
                self._has_key = False
                return True
        return False

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

    def _solve_subgoals(self) -> None:
        """Pre-solve the level to compute a subgoal chain: key→door→key→door→...→goal.

        Operates on a copy of the grid to simulate key pickups and door opens.
        Stores result in self._subgoals and resets self._subgoal_index to 0.
        """
        from collections import deque

        self._subgoals = []
        self._subgoal_index = 0

        # Work on a copy so we can simulate key pickups and door opens
        grid = [list(row) for row in self._grid]
        pos = tuple(self._agent_pos)
        w, h = self._width, self._height

        def bfs_find(
            start: tuple[int, int], target_type: int, g: list[list[int]]
        ) -> tuple[int, int] | None:
            """BFS from start to nearest cell of target_type. Doors are impassable."""
            visited = {start}
            queue = deque([start])
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if (nx, ny) in visited:
                        continue
                    cell = g[ny][nx]
                    if cell == self.WALL or cell == self.DOOR:
                        continue
                    if cell == target_type:
                        return (nx, ny)
                    visited.add((nx, ny))
                    queue.append((nx, ny))
            return None

        # Try to build subgoal chain: find keys and doors iteratively
        max_iterations = 10  # safety limit
        for _ in range(max_iterations):
            # Can we reach the goal directly?
            goal_pos = bfs_find(pos, self.GOAL, grid)
            if goal_pos is not None:
                self._subgoals.append(("goal", goal_pos))
                return

            # Can't reach goal — look for a reachable key
            key_pos = bfs_find(pos, self.KEY, grid)
            if key_pos is None:
                # No reachable key and no reachable goal — unsolvable from here
                break

            self._subgoals.append(("key", key_pos))
            # Simulate picking up the key
            grid[key_pos[1]][key_pos[0]] = self.FLOOR
            pos = key_pos

            # Now find the nearest reachable door (BFS treats doors as impassable,
            # but we want the nearest door *adjacent* to a reachable cell)
            door_pos = self._find_nearest_door(pos, grid)
            if door_pos is None:
                break

            self._subgoals.append(("door", door_pos))
            # Simulate opening the door
            grid[door_pos[1]][door_pos[0]] = self.FLOOR
            pos = door_pos

        # If we couldn't build a complete chain, try goal as fallback
        goal_pos = bfs_find(pos, self.GOAL, grid)
        if goal_pos is not None and (not self._subgoals or self._subgoals[-1][0] != "goal"):
            self._subgoals.append(("goal", goal_pos))

    def _find_nearest_door(
        self, start: tuple[int, int], grid: list[list[int]]
    ) -> tuple[int, int] | None:
        """BFS to find the nearest door adjacent to a reachable floor cell."""
        from collections import deque

        w, h = self._width, self._height
        visited = {start}
        queue = deque([start])
        seen_doors: set[tuple[int, int]] = set()
        nearest_door: tuple[int, int] | None = None
        # BFS layer by layer — first door found is nearest
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if (nx, ny) in visited:
                    continue
                cell = grid[ny][nx]
                if cell == self.DOOR and (nx, ny) not in seen_doors:
                    if nearest_door is None:
                        nearest_door = (nx, ny)
                    seen_doors.add((nx, ny))
                    continue  # don't traverse through the door
                if cell == self.WALL:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return nearest_door

    def _load_level(self, path: str) -> None:
        """Load a level from a .txt or .json file."""
        if path.endswith(".txt"):
            from level_parser import load_level

            data = load_level(path)
        else:
            import json

            with open(path) as f:
                data = json.load(f)

        self._width = data["width"]
        self._height = data["height"]
        # Deep copy the grid so we can mutate it freely
        self._grid = [list(row) for row in data["grid"]]
        self._agent_pos = list(data["agentStart"])

    def _init_border(self) -> None:
        """Create an all-floor grid with wall border."""
        w, h = self._width, self._height
        self._grid = [[self.FLOOR for _ in range(w)] for _ in range(h)]
        for x in range(w):
            self._grid[0][x] = self.WALL
            self._grid[h - 1][x] = self.WALL
        for y in range(h):
            self._grid[y][0] = self.WALL
            self._grid[y][w - 1] = self.WALL

    def _place_agent_and_goal(
        self,
        rng: np.random.Generator,
        min_dist: int = 5,
        floor_cells: list[tuple[int, int]] | None = None,
    ) -> tuple[int, int, int, int]:
        """Place agent and goal on floor cells with minimum BFS distance.

        Returns (agent_x, agent_y, goal_x, goal_y).
        """
        if floor_cells is None:
            floor_cells = [
                (x, y)
                for y in range(1, self._height - 1)
                for x in range(1, self._width - 1)
                if self._grid[y][x] == self.FLOOR
            ]

        if len(floor_cells) < 2:
            self._agent_pos = [1, 1]
            self._grid[self._height - 2][self._width - 2] = self.GOAL
            return (1, 1, self._width - 2, self._height - 2)

        max_attempts = 50
        for _ in range(max_attempts):
            indices = rng.choice(len(floor_cells), size=2, replace=False)
            ax, ay = floor_cells[int(indices[0])]
            gx, gy = floor_cells[int(indices[1])]
            dist = self._bfs_distance(ax, ay, gx, gy)
            if dist is not None and dist >= min_dist:
                break
        else:
            indices = rng.choice(len(floor_cells), size=2, replace=False)
            ax, ay = floor_cells[int(indices[0])]
            gx, gy = floor_cells[int(indices[1])]

        self._agent_pos = [ax, ay]
        self._grid[gy][gx] = self.GOAL
        return (ax, ay, gx, gy)

    def _generate_level(self, seed: int | None) -> None:
        """Generate a procedural level with varied layout styles.

        Dispatches to one of 7 generators. BSP rooms and scattered walls
        match the TS web UI generators so the agent generalizes to levels
        users see in the browser.
        """
        rng = np.random.default_rng(seed)
        roll = rng.random()
        if roll < 0.08:
            self._gen_open_room(rng)
        elif roll < 0.22:
            self._gen_room_partition(rng)
        elif roll < 0.36:
            self._gen_lava_field(rng)
        elif roll < 0.48:
            self._gen_wall_segments(rng)
        elif roll < 0.60:
            self._gen_hybrid(rng)
        elif roll < 0.80:
            self._gen_bsp_rooms(rng)
        else:
            self._gen_scattered_walls(rng)

    def _gen_open_room(self, rng: np.random.Generator) -> None:
        """Open room with 0-6 scattered wall cells."""
        self._init_border()
        w, h = self._width, self._height
        d = self._difficulty

        # Place 0-6 scattered single wall cells (scales with difficulty)
        num_walls = int(rng.integers(0, max(1, int(3 + 4 * d))))
        interior = [(x, y) for y in range(1, h - 1) for x in range(1, w - 1)]
        if num_walls > 0 and interior:
            indices = rng.choice(len(interior), size=min(num_walls, len(interior)), replace=False)
            for idx in indices:
                x, y = interior[int(idx)]
                self._grid[y][x] = self.WALL

        ax, ay, gx, gy = self._place_agent_and_goal(rng, min_dist=7)

        # Door/key (scaled by difficulty)
        roll = rng.random()
        no_door_thresh = 1.0 - 0.6 * d  # at d=0: always no doors; at d=1: 40% no doors
        if roll < no_door_thresh:
            num_doors = 0
        elif roll < no_door_thresh + 0.35 * d:
            num_doors = 1
        elif roll < no_door_thresh + 0.5 * d:
            num_doors = 2
        else:
            num_doors = 3
        for _ in range(num_doors):
            self._add_door_and_key(rng, ax, ay, gx, gy)

        # Lava (scaled by difficulty)
        if rng.random() < 0.5 * d:
            self._add_lava(rng, ax, ay, gx, gy)

    def _gen_room_partition(self, rng: np.random.Generator) -> None:
        """Rooms divided by wall partitions with door gaps."""
        self._init_border()
        w, h = self._width, self._height

        # Pick number of partitions: 40% one, 40% two, 20% three
        roll = rng.random()
        if roll < 0.40:
            num_partitions = 1
        elif roll < 0.80:
            num_partitions = 2
        else:
            num_partitions = 3

        door_positions: list[tuple[int, int]] = []
        used_positions: set[int] = set()

        for _ in range(num_partitions):
            horizontal = rng.random() < 0.5

            if horizontal:
                # Wall across a row
                valid = [y for y in range(2, h - 2) if y not in used_positions]
                if not valid:
                    continue
                pos = valid[int(rng.integers(len(valid)))]
                used_positions.add(pos)
                # Fill row with walls
                for x in range(1, w - 1):
                    self._grid[pos][x] = self.WALL
                # Cut 1 gap and place a door
                gap_x = int(rng.integers(1, w - 1))
                self._grid[pos][gap_x] = self.DOOR
                door_positions.append((gap_x, pos))
            else:
                # Wall down a column
                valid = [x for x in range(2, w - 2) if x not in used_positions]
                if not valid:
                    continue
                pos = valid[int(rng.integers(len(valid)))]
                used_positions.add(pos)
                for y in range(1, h - 1):
                    self._grid[y][pos] = self.WALL
                gap_y = int(rng.integers(1, h - 1))
                self._grid[gap_y][pos] = self.DOOR
                door_positions.append((pos, gap_y))

        # Place agent and goal
        ax, ay, gx, gy = self._place_agent_and_goal(rng, min_dist=5)

        # Place keys only for doors that survived (weren't overwritten by later partitions)
        surviving_doors = [
            (dx, dy) for (dx, dy) in door_positions if self._grid[dy][dx] == self.DOOR
        ]
        for door_x, door_y in surviving_doors:
            reachable = self._bfs_reachable(ax, ay)
            key_candidates = [
                (x, y)
                for (x, y) in reachable
                if self._grid[y][x] == self.FLOOR and (x, y) != (ax, ay) and (x, y) != (gx, gy)
            ]
            if key_candidates:
                kx, ky = key_candidates[int(rng.integers(len(key_candidates)))]
                self._grid[ky][kx] = self.KEY

        # 50% chance lava
        if rng.random() < 0.5:
            self._add_lava(rng, ax, ay, gx, gy)

    def _gen_lava_field(self, rng: np.random.Generator) -> None:
        """Open room with lava strips/patches."""
        self._init_border()
        w, h = self._width, self._height

        # Need at least 8x8 for strip/zigzag patterns, fall back to patches
        if h < 8 or w < 8:
            # Small grid: just place a few lava patches
            for _ in range(int(rng.integers(1, 4))):
                px = int(rng.integers(1, w - 1))
                py = int(rng.integers(1, h - 1))
                if self._grid[py][px] == self.FLOOR:
                    self._grid[py][px] = self.LAVA
        else:
            roll = rng.random()
            if roll < 0.30:
                # Horizontal strips: 1-3 rows of lava spanning 60-80% of width
                num_strips = int(rng.integers(1, 4))
                for _ in range(num_strips):
                    row = int(rng.integers(3, h - 3))
                    min_len = max(1, int(0.6 * (w - 2)))
                    max_len = max(min_len + 1, int(0.8 * (w - 2)) + 1)
                    strip_len = int(rng.integers(min_len, max_len))
                    start_x = int(rng.integers(1, max(2, w - 1 - strip_len)))
                    end_x = min(start_x + strip_len, w - 1)
                    for x in range(start_x, end_x):
                        self._grid[row][x] = self.LAVA
                    # Cut 1-2 gaps
                    if end_x > start_x:
                        num_gaps = int(rng.integers(1, 3))
                        for _ in range(num_gaps):
                            gap_x = int(rng.integers(start_x, end_x))
                            self._grid[row][gap_x] = self.FLOOR
            elif roll < 0.75:
                # Zigzag: 3-5 alternating strips at different rows with offset gaps
                available_rows = list(range(2, h - 2))
                num_rows = min(int(rng.integers(3, 6)), len(available_rows))
                if num_rows > 0:
                    rows = sorted(rng.choice(available_rows, size=num_rows, replace=False))
                    for i, row in enumerate(rows):
                        row = int(row)
                        for x in range(1, w - 1):
                            self._grid[row][x] = self.LAVA
                        # Offset gaps — 1-2 cells wide
                        gap_x = 1 + (i * 3) % (w - 2)
                        self._grid[row][gap_x] = self.FLOOR
                        if rng.random() < 0.5 and gap_x + 1 < w - 1:
                            self._grid[row][gap_x + 1] = self.FLOOR
            else:
                # Patches: 2-4 connected lava patches of 2-4 cells each
                num_patches = int(rng.integers(2, 5))
                for _ in range(num_patches):
                    px = int(rng.integers(2, w - 2))
                    py = int(rng.integers(2, h - 2))
                    patch_size = int(rng.integers(2, 5))
                    cells = [(px, py)]
                    for _ in range(patch_size - 1):
                        cx, cy = cells[-1]
                        dx, dy = [(0, 1), (0, -1), (1, 0), (-1, 0)][int(rng.integers(4))]
                        nx, ny = cx + dx, cy + dy
                        if 1 <= nx < w - 1 and 1 <= ny < h - 1:
                            cells.append((nx, ny))
                    for cx, cy in cells:
                        self._grid[cy][cx] = self.LAVA

        # Place agent above lava region, goal below
        top_floor = [
            (x, y)
            for y in range(1, h // 2)
            for x in range(1, w - 1)
            if self._grid[y][x] == self.FLOOR
        ]
        bot_floor = [
            (x, y)
            for y in range(h // 2, h - 1)
            for x in range(1, w - 1)
            if self._grid[y][x] == self.FLOOR
        ]

        if top_floor and bot_floor:
            ai = int(rng.integers(len(top_floor)))
            ax, ay = top_floor[ai]
            self._agent_pos = [ax, ay]
            gi = int(rng.integers(len(bot_floor)))
            gx, gy = bot_floor[gi]
            self._grid[gy][gx] = self.GOAL
        else:
            ax, ay, gx, gy = self._place_agent_and_goal(rng, min_dist=5)

        # Verify solvability, widen gaps if blocked
        for _ in range(20):
            if self._bfs_distance_safe(ax, ay, gx, gy) is not None:
                break
            # Find a lava cell and clear it
            lava_cells = [
                (x, y)
                for y in range(1, h - 1)
                for x in range(1, w - 1)
                if self._grid[y][x] == self.LAVA
            ]
            if not lava_cells:
                break
            rx, ry = lava_cells[int(rng.integers(len(lava_cells)))]
            self._grid[ry][rx] = self.FLOOR

        # 40% chance door/key — combining lava + doors is important for generalization
        if rng.random() < 0.4:
            self._add_door_and_key(rng, ax, ay, gx, gy)

    def _gen_wall_segments(self, rng: np.random.Generator) -> None:
        """Wall segments creating corridors and channels."""
        self._init_border()
        w, h = self._width, self._height

        num_segments = int(rng.integers(3, 6))
        segment_positions: list[tuple[bool, int]] = []  # (is_horizontal, position)

        for _ in range(num_segments):
            horizontal = rng.random() < 0.5

            if horizontal:
                # Check spacing from existing horizontal segments
                valid = [
                    y
                    for y in range(2, h - 2)
                    if all(abs(y - p) >= 2 for is_h, p in segment_positions if is_h)
                ]
                if not valid:
                    continue
                row = valid[int(rng.integers(len(valid)))]
                segment_positions.append((True, row))
                # Span 50-80% of grid width
                span = int(rng.integers(int(0.5 * (w - 2)), int(0.8 * (w - 2)) + 1))
                start_x = int(rng.integers(1, max(2, w - 1 - span)))
                for x in range(start_x, min(start_x + span, w - 1)):
                    self._grid[row][x] = self.WALL
            else:
                valid = [
                    x
                    for x in range(2, w - 2)
                    if all(abs(x - p) >= 2 for is_h, p in segment_positions if not is_h)
                ]
                if not valid:
                    continue
                col = valid[int(rng.integers(len(valid)))]
                segment_positions.append((False, col))
                span = int(rng.integers(int(0.5 * (h - 2)), int(0.8 * (h - 2)) + 1))
                start_y = int(rng.integers(1, max(2, h - 1 - span)))
                for y in range(start_y, min(start_y + span, h - 1)):
                    self._grid[y][col] = self.WALL

        ax, ay, gx, gy = self._place_agent_and_goal(rng, min_dist=5)

        # 40% chance door/key
        roll = rng.random()
        if roll < 0.40:
            num_doors = 0
        elif roll < 0.75:
            num_doors = 1
        elif roll < 0.90:
            num_doors = 2
        else:
            num_doors = 3
        for _ in range(num_doors):
            self._add_door_and_key(rng, ax, ay, gx, gy)

        # 50% chance lava
        if rng.random() < 0.5:
            self._add_lava(rng, ax, ay, gx, gy)

    def _gen_hybrid(self, rng: np.random.Generator) -> None:
        """Room partition base with lava patches near the goal."""
        self._init_border()
        w, h = self._width, self._height

        # 1-2 partitions with doors
        num_partitions = int(rng.integers(1, 3))
        door_positions: list[tuple[int, int]] = []
        used_positions: set[int] = set()

        for _ in range(num_partitions):
            horizontal = rng.random() < 0.5
            if horizontal:
                valid = [y for y in range(2, h - 2) if y not in used_positions]
                if not valid:
                    continue
                pos = valid[int(rng.integers(len(valid)))]
                used_positions.add(pos)
                for x in range(1, w - 1):
                    self._grid[pos][x] = self.WALL
                gap_x = int(rng.integers(1, w - 1))
                self._grid[pos][gap_x] = self.DOOR
                door_positions.append((gap_x, pos))
            else:
                valid = [x for x in range(2, w - 2) if x not in used_positions]
                if not valid:
                    continue
                pos = valid[int(rng.integers(len(valid)))]
                used_positions.add(pos)
                for y in range(1, h - 1):
                    self._grid[y][pos] = self.WALL
                gap_y = int(rng.integers(1, h - 1))
                self._grid[gap_y][pos] = self.DOOR
                door_positions.append((pos, gap_y))

        ax, ay, gx, gy = self._place_agent_and_goal(rng, min_dist=5)

        # Place keys only for doors that survived
        surviving_doors = [
            (dx, dy) for (dx, dy) in door_positions if self._grid[dy][dx] == self.DOOR
        ]
        for door_x, door_y in surviving_doors:
            reachable = self._bfs_reachable(ax, ay)
            key_candidates = [
                (x, y)
                for (x, y) in reachable
                if self._grid[y][x] == self.FLOOR and (x, y) != (ax, ay) and (x, y) != (gx, gy)
            ]
            if key_candidates:
                kx, ky = key_candidates[int(rng.integers(len(key_candidates)))]
                self._grid[ky][kx] = self.KEY

        # Add 3-7 lava cells as a patch near the goal
        num_lava = int(rng.integers(3, 8))
        cells = [(gx, gy)]
        for _ in range(num_lava):
            cx, cy = cells[-1]
            dx, dy = [(0, 1), (0, -1), (1, 0), (-1, 0)][int(rng.integers(4))]
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < w - 1 and 1 <= ny < h - 1:
                cells.append((nx, ny))
        # Place lava (skip goal cell and agent cell)
        for cx, cy in cells:
            if (cx, cy) != (gx, gy) and (cx, cy) != (ax, ay) and self._grid[cy][cx] == self.FLOOR:
                self._grid[cy][cx] = self.LAVA

        # Verify solvability, remove lava if blocked
        for _ in range(20):
            if self._bfs_distance_safe(ax, ay, gx, gy) is not None:
                break
            lava_cells = [
                (x, y)
                for y in range(1, h - 1)
                for x in range(1, w - 1)
                if self._grid[y][x] == self.LAVA
            ]
            if not lava_cells:
                break
            rx, ry = lava_cells[int(rng.integers(len(lava_cells)))]
            self._grid[ry][rx] = self.FLOOR

    # --- BSP rooms generator (matches TS play.ts) ---

    @staticmethod
    def _bsp_split(
        x: int, y: int, w: int, h: int, rng: np.random.Generator, depth: int
    ) -> list[tuple[int, int, int, int]]:
        """Recursive BSP split. Returns list of (x, y, w, h) rects."""
        min_size = 3
        if depth <= 0 or (w < min_size * 2 + 1 and h < min_size * 2 + 1):
            return [(x, y, w, h)]
        can_h = w >= min_size * 2 + 1
        can_v = h >= min_size * 2 + 1
        if not can_h and not can_v:
            return [(x, y, w, h)]
        split_h = can_h and (not can_v or rng.random() < 0.5)
        if split_h:
            s_min = x + min_size
            s_max = x + w - min_size
            at = s_min + int(rng.integers(0, max(1, s_max - s_min)))
            left = BoxworldEnv._bsp_split(x, y, at - x, h, rng, depth - 1)
            right = BoxworldEnv._bsp_split(at + 1, y, x + w - at - 1, h, rng, depth - 1)
            return left + right
        else:
            s_min = y + min_size
            s_max = y + h - min_size
            at = s_min + int(rng.integers(0, max(1, s_max - s_min)))
            top = BoxworldEnv._bsp_split(x, y, w, at - y, rng, depth - 1)
            bot = BoxworldEnv._bsp_split(x, at + 1, w, y + h - at - 1, rng, depth - 1)
            return top + bot

    def _carve_corridor(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Carve horizontal-first L-shaped corridor between two points."""
        x, y = x1, y1
        while x != x2:
            if 0 <= y < self._height and 0 <= x < self._width:
                if self._grid[y][x] == self.WALL:
                    self._grid[y][x] = self.FLOOR
            x += 1 if x < x2 else -1
        while y != y2:
            if 0 <= y < self._height and 0 <= x < self._width:
                if self._grid[y][x] == self.WALL:
                    self._grid[y][x] = self.FLOOR
            y += 1 if y < y2 else -1
        if 0 <= y < self._height and 0 <= x < self._width:
            if self._grid[y][x] == self.WALL:
                self._grid[y][x] = self.FLOOR

    def _is_chokepoint(self, x: int, y: int) -> bool:
        """Check if (x,y) is a corridor chokepoint: exactly 2 floor neighbors on opposite sides."""
        if self._grid[y][x] != self.FLOOR:
            return False
        w, h = self._width, self._height

        def passable(cx: int, cy: int) -> bool:
            return 0 <= cx < w and 0 <= cy < h and self._grid[cy][cx] != self.WALL

        horiz = (
            passable(x - 1, y)
            and passable(x + 1, y)
            and not passable(x, y - 1)
            and not passable(x, y + 1)
        )
        vert = (
            not passable(x - 1, y)
            and not passable(x + 1, y)
            and passable(x, y - 1)
            and passable(x, y + 1)
        )
        return horiz or vert

    def _get_floor_cells(self) -> list[tuple[int, int]]:
        """Get all interior floor cells."""
        return [
            (x, y)
            for y in range(1, self._height - 1)
            for x in range(1, self._width - 1)
            if self._grid[y][x] == self.FLOOR
        ]

    def _gen_bsp_rooms(self, rng: np.random.Generator) -> None:
        """BSP room-corridor layout matching TS generateLevel (useRooms=true branch)."""
        w, h = self._width, self._height
        d = self._difficulty
        # Start with all walls + border
        self._grid = [[self.WALL for _ in range(w)] for _ in range(h)]

        # BSP split interior — fewer splits at low difficulty
        max_depth = 1 if d < 0.3 else 2
        rooms = self._bsp_split(1, 1, w - 2, h - 2, rng, max_depth)

        # Carve rooms
        for rx, ry, rw, rh in rooms:
            for cy in range(ry, ry + rh):
                for cx in range(rx, rx + rw):
                    if 0 < cy < h - 1 and 0 < cx < w - 1:
                        self._grid[cy][cx] = self.FLOOR

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            ax, ay, aw, ah = rooms[i]
            bx, by, bw, bh = rooms[i + 1]
            self._carve_corridor(ax + aw // 2, ay + ah // 2, bx + bw // 2, by + bh // 2)

        floors = self._get_floor_cells()
        if len(floors) < 8:
            # Fallback to open room
            self._gen_open_room(rng)
            return

        # Place agent and goal with max BFS distance (sample pairs like TS)
        best_dist = -1
        agent_pos = floors[0]
        goal_pos = floors[-1]
        pair_tries = min(20, len(floors) * 2)
        for _ in range(pair_tries):
            a = floors[int(rng.integers(len(floors)))]
            b = floors[int(rng.integers(len(floors)))]
            if a == b:
                continue
            d = self._bfs_distance(a[0], a[1], b[0], b[1])
            if d is not None and d > best_dist:
                best_dist = d
                agent_pos = a
                goal_pos = b

        self._agent_pos = list(agent_pos)
        self._grid[goal_pos[1]][goal_pos[0]] = self.GOAL
        ax, ay = agent_pos
        gx, gy = goal_pos

        # Door on chokepoint (probability scales with difficulty)
        if rng.random() < 0.5 * d:
            chokepoints = [
                (x, y)
                for x, y in floors
                if self._is_chokepoint(x, y) and (x, y) != agent_pos and (x, y) != goal_pos
            ]
            if chokepoints:
                door_cell = chokepoints[int(rng.integers(len(chokepoints)))]
                self._grid[door_cell[1]][door_cell[0]] = self.DOOR
                # Place key in agent's reachable area
                reachable = [
                    (x, y)
                    for x, y in self._get_floor_cells()
                    if (x, y) != agent_pos
                    and (x, y) != goal_pos
                    and self._bfs_distance(ax, ay, x, y) is not None
                ]
                if reachable:
                    key_cell = reachable[int(rng.integers(len(reachable)))]
                    self._grid[key_cell[1]][key_cell[0]] = self.KEY
                else:
                    self._grid[door_cell[1]][door_cell[0]] = self.FLOOR

        # Lava (probability scales with difficulty)
        if rng.random() < 0.6 * d:
            lava_count = 2 + int(rng.integers(4))
            placed = 0
            for _ in range(lava_count * 5):
                if placed >= lava_count:
                    break
                candidates = [
                    (x, y)
                    for x, y in self._get_floor_cells()
                    if (x, y) != agent_pos and (x, y) != goal_pos
                ]
                if not candidates:
                    break
                cell = candidates[int(rng.integers(len(candidates)))]
                self._grid[cell[1]][cell[0]] = self.LAVA
                if self._bfs_distance_safe(ax, ay, gx, gy) is None:
                    self._grid[cell[1]][cell[0]] = self.FLOOR
                else:
                    placed += 1

    def _gen_scattered_walls(self, rng: np.random.Generator) -> None:
        """Open layout with scattered walls matching TS generateLevel (useRooms=false branch)."""
        self._init_border()
        w, h = self._width, self._height
        d = self._difficulty

        # Wall count scales with difficulty: 2-6 at d=0, 4-13 at d=1
        min_walls = int(2 + 2 * d)
        max_walls = int(6 + 7 * d)
        wall_count = min_walls + int(rng.integers(max(1, max_walls - min_walls + 1)))
        for _ in range(wall_count):
            wx = 1 + int(rng.integers(w - 2))
            wy = 1 + int(rng.integers(h - 2))
            self._grid[wy][wx] = self.WALL
            if rng.random() < 0.4:
                dx = 1 if rng.random() < 0.5 else 0
                dy = 0 if dx == 1 else 1
                for j in range(1, 1 + 1 + int(rng.integers(2))):
                    ex, ey = wx + dx * j, wy + dy * j
                    if 0 < ex < w - 1 and 0 < ey < h - 1:
                        self._grid[ey][ex] = self.WALL

        floors = self._get_floor_cells()
        if len(floors) < 8:
            self._gen_open_room(rng)
            return

        # Place agent and goal with max BFS distance
        best_dist = -1
        agent_pos = floors[0]
        goal_pos = floors[-1]
        pair_tries = min(20, len(floors) * 2)
        for _ in range(pair_tries):
            a = floors[int(rng.integers(len(floors)))]
            b = floors[int(rng.integers(len(floors)))]
            if a == b:
                continue
            dist = self._bfs_distance(a[0], a[1], b[0], b[1])
            if dist is not None and dist > best_dist:
                best_dist = dist
                agent_pos = a
                goal_pos = b

        self._agent_pos = list(agent_pos)
        self._grid[goal_pos[1]][goal_pos[0]] = self.GOAL
        ax, ay = agent_pos
        gx, gy = goal_pos

        # Door on chokepoint (probability scales with difficulty)
        has_door = False
        if rng.random() < 0.5 * d:
            chokepoints = [
                (x, y)
                for x, y in floors
                if self._is_chokepoint(x, y) and (x, y) != agent_pos and (x, y) != goal_pos
            ]
            if chokepoints:
                door_cell = chokepoints[int(rng.integers(len(chokepoints)))]
                self._grid[door_cell[1]][door_cell[0]] = self.DOOR
                reachable = [
                    (x, y)
                    for x, y in self._get_floor_cells()
                    if (x, y) != agent_pos
                    and (x, y) != goal_pos
                    and self._bfs_distance(ax, ay, x, y) is not None
                ]
                if reachable:
                    key_cell = reachable[int(rng.integers(len(reachable)))]
                    self._grid[key_cell[1]][key_cell[0]] = self.KEY
                    has_door = True
                else:
                    self._grid[door_cell[1]][door_cell[0]] = self.FLOOR

        # Lava (probability scales with difficulty)
        if rng.random() < 0.6 * d:
            lava_count = 2 + int(rng.integers(4))
            placed = 0
            for _ in range(lava_count * 5):
                if placed >= lava_count:
                    break
                candidates = [
                    (x, y)
                    for x, y in self._get_floor_cells()
                    if (x, y) != agent_pos and (x, y) != goal_pos
                ]
                if not candidates:
                    break
                cell = candidates[int(rng.integers(len(candidates)))]
                self._grid[cell[1]][cell[0]] = self.LAVA
                solvable = (
                    self._bfs_distance_safe(ax, ay, gx, gy) is not None
                    if has_door
                    else self._bfs_distance(ax, ay, gx, gy) is not None
                )
                if not solvable:
                    self._grid[cell[1]][cell[0]] = self.FLOOR
                else:
                    placed += 1

        # Final solvability check
        if has_door:
            if self._bfs_distance_safe(ax, ay, gx, gy) is None:
                # Unsolvable — retry as open room
                self._gen_open_room(rng)
        else:
            if self._bfs_distance(ax, ay, gx, gy) is None:
                self._gen_open_room(rng)

    def _bfs_reachable(self, sx: int, sy: int) -> set[tuple[int, int]]:
        """BFS to find all floor cells reachable from (sx, sy), not crossing walls/doors."""
        from collections import deque

        reachable = {(sx, sy)}
        queue = deque([(sx, sy)])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self._width and 0 <= ny < self._height):
                    continue
                if (nx, ny) in reachable:
                    continue
                cell = self._grid[ny][nx]
                if cell == self.WALL or cell == self.DOOR:
                    continue
                reachable.add((nx, ny))
                queue.append((nx, ny))
        return reachable

    def _bfs_distance(self, sx: int, sy: int, tx: int, ty: int) -> float | None:
        """BFS distance between two points on the current grid. Doors are impassable."""
        from collections import deque

        if (sx, sy) == (tx, ty):
            return 0.0
        visited = {(sx, sy)}
        queue = deque([((sx, sy), 0)])
        while queue:
            (cx, cy), dist = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self._width and 0 <= ny < self._height):
                    continue
                if (nx, ny) in visited:
                    continue
                if (nx, ny) == (tx, ty):
                    return float(dist + 1)
                cell = self._grid[ny][nx]
                if cell == self.WALL or cell == self.DOOR:
                    continue
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
        return None

    def _add_door_and_key(
        self,
        rng: np.random.Generator,
        agent_x: int,
        agent_y: int,
        goal_x: int,
        goal_y: int,
    ) -> None:
        """Add a door on the path between agent and goal, plus a key on the agent's side."""
        from collections import deque

        # Find the BFS path from agent to goal
        start = (agent_x, agent_y)
        goal = (goal_x, goal_y)
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        queue = deque([start])
        found = False
        while queue:
            cx, cy = queue.popleft()
            if (cx, cy) == goal:
                found = True
                break
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self._width and 0 <= ny < self._height):
                    continue
                if (nx, ny) in parent:
                    continue
                cell = self._grid[ny][nx]
                if cell == self.WALL:
                    continue
                parent[(nx, ny)] = (cx, cy)
                queue.append((nx, ny))

        if not found:
            return

        # Reconstruct path
        path = []
        node: tuple[int, int] | None = goal
        while node is not None:
            path.append(node)
            node = parent.get(node)
        path.reverse()

        # Pick a wall cell adjacent to a mid-path floor cell to become a door
        # We need: a floor cell on the path (not start/end), and it becomes a door
        # Actually simpler: pick a floor cell on the path (not agent/goal) to become a door
        candidates = [p for p in path[1:-1] if self._grid[p[1]][p[0]] == self.FLOOR]
        if not candidates:
            return

        door_pos = candidates[int(rng.integers(len(candidates)))]
        self._grid[door_pos[1]][door_pos[0]] = self.DOOR

        # Place key on a floor cell reachable from agent (BFS without crossing the door)
        reachable = set()
        q = deque([start])
        reachable.add(start)
        while q:
            cx, cy = q.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self._width and 0 <= ny < self._height):
                    continue
                if (nx, ny) in reachable:
                    continue
                cell = self._grid[ny][nx]
                if cell == self.WALL or cell == self.DOOR:
                    continue
                reachable.add((nx, ny))
                q.append((nx, ny))

        # Floor cells reachable from agent, not agent/goal/door
        key_candidates = [
            (x, y)
            for (x, y) in reachable
            if self._grid[y][x] == self.FLOOR
            and (x, y) != start
            and (x, y) != goal
            and (x, y) != door_pos
        ]
        if not key_candidates:
            # Can't place key, remove door
            self._grid[door_pos[1]][door_pos[0]] = self.FLOOR
            return

        kx, ky = key_candidates[int(rng.integers(len(key_candidates)))]
        self._grid[ky][kx] = self.KEY

    def _add_lava(
        self,
        rng: np.random.Generator,
        agent_x: int,
        agent_y: int,
        goal_x: int,
        goal_y: int,
    ) -> None:
        """Add some lava cells, ensuring the level remains solvable."""
        floor_cells = [
            (x, y)
            for y in range(1, self._height - 1)
            for x in range(1, self._width - 1)
            if self._grid[y][x] == self.FLOOR
            and (x, y) != (agent_x, agent_y)
            and (x, y) != (goal_x, goal_y)
        ]

        num_lava = min(len(floor_cells) // 5, int(rng.integers(1, 5)))
        if num_lava == 0:
            return

        indices = rng.choice(len(floor_cells), size=min(num_lava, len(floor_cells)), replace=False)
        for idx in indices:
            x, y = floor_cells[int(idx)]
            # Tentatively place lava
            self._grid[y][x] = self.LAVA
            # Check solvability — agent must be able to reach goal without stepping on lava
            if self._bfs_distance_safe(agent_x, agent_y, goal_x, goal_y) is None:
                # Revert — this lava blocks the only safe path
                self._grid[y][x] = self.FLOOR

    def _bfs_distance_safe(self, sx: int, sy: int, tx: int, ty: int) -> float | None:
        """BFS distance avoiding walls, doors, AND lava.

        The target cell itself is always reachable (e.g. a door subgoal can be
        reached even though doors are otherwise impassable).
        """
        from collections import deque

        if (sx, sy) == (tx, ty):
            return 0.0
        visited = {(sx, sy)}
        queue = deque([((sx, sy), 0)])
        while queue:
            (cx, cy), dist = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self._width and 0 <= ny < self._height):
                    continue
                if (nx, ny) in visited:
                    continue
                # Always allow reaching the target (e.g. door subgoal)
                if (nx, ny) == (tx, ty):
                    return float(dist + 1)
                cell = self._grid[ny][nx]
                if cell in (self.WALL, self.DOOR, self.LAVA):
                    continue
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
        return None

    def _goal_distance(self) -> float | None:
        """Return BFS shortest-path distance from agent to the goal.

        Door-aware: if the agent has a key, doors are treated as passable.
        Returns None if unreachable.
        """
        from collections import deque

        # Find goal position
        goal = None
        for y in range(self._height):
            for x in range(self._width):
                if self._grid[y][x] == self.GOAL:
                    goal = (x, y)
                    break
            if goal:
                break
        if goal is None:
            return None

        start = (self._agent_pos[0], self._agent_pos[1])
        if start == goal:
            return 0.0

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            (cx, cy), dist = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self._width and 0 <= ny < self._height):
                    continue
                if (nx, ny) in visited:
                    continue
                cell = self._grid[ny][nx]
                if cell == self.WALL:
                    continue
                if cell == self.DOOR and not self._has_key:
                    continue
                if (nx, ny) == goal:
                    return float(dist + 1)
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

        return None  # goal unreachable

    def _key_distance(self) -> float | None:
        """Return BFS shortest-path distance from agent to the nearest key.

        Doors are treated as impassable (keys are always on the accessible side).
        Returns None if no key on the grid.
        """
        from collections import deque

        # Check if any key exists
        keys = set()
        for y in range(self._height):
            for x in range(self._width):
                if self._grid[y][x] == self.KEY:
                    keys.add((x, y))
        if not keys:
            return None

        start = (self._agent_pos[0], self._agent_pos[1])
        if start in keys:
            return 0.0

        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            (cx, cy), dist = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self._width and 0 <= ny < self._height):
                    continue
                if (nx, ny) in visited:
                    continue
                cell = self._grid[ny][nx]
                if cell == self.WALL or cell == self.DOOR:
                    continue
                if (nx, ny) in keys:
                    return float(dist + 1)
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

        return None  # no reachable key

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
