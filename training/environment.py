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

    def _generate_level(self, seed: int | None) -> None:
        """Generate a procedural maze level using recursive backtracker.

        Produces corridor-based mazes with optional doors, keys, and lava.
        Ensures the level is solvable with a minimum BFS distance of 5.
        """
        from collections import deque

        rng = np.random.default_rng(seed)
        w, h = self._width, self._height

        # --- Maze carving via recursive backtracker (DFS) ---
        # Work on odd-indexed cells as "rooms", walls between them.
        # Interior maze grid: cells at odd coords in [1, w-2] x [1, h-2]
        self._grid = [[self.WALL for _ in range(w)] for _ in range(h)]

        # Maze cells are at odd coordinates within the border
        maze_w = (w - 1) // 2  # number of maze columns
        maze_h = (h - 1) // 2  # number of maze rows

        if maze_w < 1 or maze_h < 1:
            # Grid too small for maze, fallback to open room
            self._grid = [[self.FLOOR for _ in range(w)] for _ in range(h)]
            for x in range(w):
                self._grid[0][x] = self.WALL
                self._grid[h - 1][x] = self.WALL
            for y in range(h):
                self._grid[y][0] = self.WALL
                self._grid[y][w - 1] = self.WALL
            self._agent_pos = [1, 1]
            self._grid[h - 2][w - 2] = self.GOAL
            return

        visited_maze = [[False] * maze_w for _ in range(maze_h)]

        # Map maze coords to grid coords
        def to_grid(mx: int, my: int) -> tuple[int, int]:
            return (1 + mx * 2, 1 + my * 2)

        # Start maze from random cell
        start_mx = int(rng.integers(maze_w))
        start_my = int(rng.integers(maze_h))
        visited_maze[start_my][start_mx] = True
        gx, gy = to_grid(start_mx, start_my)
        self._grid[gy][gx] = self.FLOOR

        stack = [(start_mx, start_my)]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while stack:
            cx, cy = stack[-1]
            # Find unvisited neighbors
            neighbors = []
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < maze_w and 0 <= ny < maze_h and not visited_maze[ny][nx]:
                    neighbors.append((nx, ny, dx, dy))

            if not neighbors:
                stack.pop()
                continue

            # Pick random neighbor
            idx = int(rng.integers(len(neighbors)))
            nx, ny, dx, dy = neighbors[idx]
            visited_maze[ny][nx] = True

            # Carve passage: open the neighbor cell and the wall between
            ngx, ngy = to_grid(nx, ny)
            wall_gx = to_grid(cx, cy)[0] + dx
            wall_gy = to_grid(cx, cy)[1] + dy
            self._grid[ngy][ngx] = self.FLOOR
            self._grid[wall_gy][wall_gx] = self.FLOOR

            stack.append((nx, ny))

        # --- Collect floor cells ---
        floor_cells = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if self._grid[y][x] == self.FLOOR:
                    floor_cells.append((x, y))

        if len(floor_cells) < 3:
            # Shouldn't happen with a proper maze, but fallback
            self._agent_pos = [1, 1]
            self._grid[1][1] = self.FLOOR
            self._grid[h - 2][w - 2] = self.GOAL
            return

        # --- Place agent and goal with minimum BFS distance ---
        min_dist = 5
        max_attempts = 50
        for _ in range(max_attempts):
            indices = rng.choice(len(floor_cells), size=2, replace=False)
            ax, ay = floor_cells[int(indices[0])]
            gx, gy = floor_cells[int(indices[1])]
            # Quick BFS distance check
            dist = self._bfs_distance(ax, ay, gx, gy)
            if dist is not None and dist >= min_dist:
                break
        else:
            # Accept whatever we got
            indices = rng.choice(len(floor_cells), size=2, replace=False)
            ax, ay = floor_cells[int(indices[0])]
            gx, gy = floor_cells[int(indices[1])]

        self._agent_pos = [ax, ay]
        self._grid[gy][gx] = self.GOAL

        # --- Optionally add door/key pairs (probabilistic multi-door) ---
        # 40% no doors, 35% one, 15% two, 10% three
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

        # --- Optionally add lava (~30% chance) ---
        if rng.random() < 0.3:
            self._add_lava(rng, ax, ay, gx, gy)

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
