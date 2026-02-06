# Training Improvements Plan

**Date:** 2026-02-06
**Goal:** Improve DQN training so the agent can solve more than just `dead_end`.

## Problem

The 500k-step DQN agent solves `dead_end` but fails on `simple_corridor`, `four_rooms`, `key_puzzle`, and `lava_maze`. Three root causes:

1. **BFS shaping blind spot on doors:** `_goal_distance()` treats doors as impassable walls, so it returns `None` when the goal is behind a door. This means reward shaping provides *zero signal* on levels like `key_puzzle` and `four_rooms` — the agent has no gradient toward learning key/door mechanics.

2. **Procedural levels don't teach relevant skills:** The generator creates open rooms with ~10% random walls, a key, and a goal. No doors, no lava, no corridors. The agent trains mostly on trivially easy levels and rarely encounters the patterns it needs.

3. **`designed_level_prob=0.3` is too low:** Each of the 5 designed levels only gets ~6% of training time.

## Changes

### G. Fix door-aware BFS shaping + add key/door rewards

**File:** `training/environment.py`

1. **Make `_goal_distance()` door-aware:** When the agent has a key, BFS should treat doors as passable (cost 1 like floor). When the agent doesn't have a key, doors remain impassable but we add a *secondary* shaping signal toward the nearest key.

2. **Add a `_key_distance()` method:** BFS from agent to nearest key cell, ignoring doors (keys are always on the floor side). Returns `None` if no key on the grid.

3. **Restructure shaping in `step()`:**
   - If goal is reachable (BFS finds a path, possibly through doors when has_key): shape toward goal.
   - Else if no key held and key exists: shape toward nearest key.
   - This gives the agent a continuous gradient: approach key → pick up → approach door → open → approach goal.

4. **Add explicit pickup/toggle rewards:**
   - Picking up a key: `+0.2` bonus
   - Opening a door: `+0.2` bonus
   - These are one-time events that directly reward the desired sub-behaviors.

**Tests to add/update:**
- Test that `_goal_distance()` with `has_key=True` treats doors as passable
- Test that `_goal_distance()` with `has_key=False` still blocks on doors
- Test `_key_distance()` returns correct BFS distance
- Test that shaping reward includes key-approach signal when goal is unreachable
- Test pickup reward bonus
- Test toggle reward bonus

### B. Improve procedural level generator

**File:** `training/environment.py`

Replace `_generate_level()` with a more diverse generator that produces levels resembling the designed ones:

1. **Maze generation using recursive backtracker (DFS):** Start with all walls, carve passages. This produces connected corridors with dead ends — much more like `dead_end` and `simple_corridor`.

2. **Randomly add doors + keys:** With some probability (~30%), place a door on a passage and a key on the agent's side. Ensures the agent encounters key/door mechanics during procgen training.

3. **Randomly add lava:** With some probability (~20%), convert some floor cells to lava, ensuring the level remains solvable (BFS check after placement).

4. **Minimum BFS distance check:** Reject levels where agent-to-goal BFS distance is < 5 to avoid trivially easy levels.

**Tests to add:**
- Test that generated mazes have connected corridors (BFS from agent reaches goal)
- Test that generated mazes sometimes include doors + keys
- Test that generated mazes sometimes include lava
- Test that minimum distance constraint is respected
- Test deterministic seeding still works

### C. Increase `designed_level_prob`

**File:** `training/main.py`

Change `designed_level_prob` from `0.3` to `0.7`. The agent spends 70% of training on the hand-designed levels it needs to solve, and 30% on procedural levels for generalization.

## Files Modified

| File | Changes |
|------|---------|
| `training/environment.py` | Door-aware `_goal_distance()`, new `_key_distance()`, shaping refactor, pickup/toggle rewards, new `_generate_level()` |
| `training/main.py` | `designed_level_prob=0.3` → `0.7` |
| `training/tests/test_environment.py` | New tests for all the above |

## Risks

- Changing rewards/shaping changes the meaning of existing checkpoints and recorded episodes. The next `main.py all` run will produce new data that supersedes the old.
- The maze generator is more complex — must verify solvability.
- Existing tests that check exact reward values (e.g., `test_reach_goal` expects `reward == 1.0`, `test_step_reward_is_negative` expects `-0.01`) may need updating if shaping affects those test scenarios.
