# E2E Level-Solving Test & Training Iteration

## Goal

Create an e2e acceptance test asserting all 5 designed levels are solved by the trained PPO agent, then iterate on training config until the test passes.

## Starting point

Only 2 of 5 designed levels solved (door_key, two_rooms). The remaining 3 (open_room, simple_corridor, lava_crossing) all timed out at -2.0 reward.

## What was created

### `training/tests/test_e2e.py`

- Parametrized pytest that reads `data/db.sqlite` and checks each of the 5 designed levels
- Each level is a separate test case so pytest output shows exactly which pass/fail
- Skips gracefully if no DB exists (hasn't been trained yet)
- Assertion: at least one recorded episode must have the agent standing on the goal cell in its final state

## Training iterations

| # | Config changes | Levels solved | Key observation |
|---|---------------|---------------|-----------------|
| 1 | Baseline: 1M steps, [64,64], ent=0.05, positive-only shaping, designed_prob=0.7 | 2/5 | door_key and two_rooms only |
| 2 | 2M steps, [128,128] network, designed_prob=0.9 | 2/5 | Same 2 levels; more capacity didn't help |
| 3 | Bidirectional shaping (±0.05 per BFS step), ent_coef=0.01 | 4/5 | open_room and simple_corridor now solved; lava_crossing still fails |
| 4 | Lava-safe BFS (`_bfs_distance_safe`), shaping coefficient ±0.1 | **5/5** | All levels solved |

## Bugs found and fixed

### Bug 1: Positive-only shaping causes oscillation

- **Symptom:** Agent moves toward goal but gets stuck bouncing between two positions
- **Root cause:** Shaping only rewarded moving closer (`new_dist < old_dist`), never penalized moving away. The agent oscillated because both positions had equal "distance" and no penalty for retreating.
- **Fix:** Made shaping bidirectional — reward moving closer AND penalize moving farther (`new_dist != old_dist` instead of `new_dist < old_dist`)
- **File:** `environment.py:167-172`

### Bug 2: BFS reward shaping routed through lava

- **Symptom:** Agent on lava_crossing walks right up to the lava row then gets stuck spamming TOGGLE
- **Root cause:** `_bfs_distance()` treats lava as passable (only walls/doors block). From position (8,3), BFS said "move DOWN to (8,4) — distance 4!" but (8,4) is lava = instant death. The safe path through the gap at column 4 had BFS distance 13, so the shaping literally told the agent to walk into lava.
- **Fix:** Switched reward shaping from `_bfs_distance` to `_bfs_distance_safe` (which avoids walls, doors, AND lava)
- **File:** `environment.py:169-170`

## Final training config

- **Steps:** 2M (default, via `main.py --steps`)
- **Network:** pi=[128,128], vf=[128,128]
- **Entropy coefficient:** 0.01
- **Designed level probability:** 0.9
- **Shaping:** Bidirectional ±0.1 per BFS step, using lava-safe BFS
- **Level weights:** All 1.0 (equal sampling)

## Files changed

- `training/tests/test_e2e.py` — New file, 5 parametrized test cases
- `training/environment.py` — Bidirectional shaping + lava-safe BFS
- `training/train.py` — ent_coef 0.05→0.01, net_arch [64,64]→[128,128]
- `training/main.py` — steps 1M→2M, designed_level_prob 0.7→0.9
- `training/tests/test_environment.py` — Updated 4 reward assertions for new shaping coefficient (0.05→0.1)
- `CLAUDE.md` — Updated reward description, training config, test counts

## Verification

```
$ cd training && uv run pytest tests/test_e2e.py -v
tests/test_e2e.py::TestAllLevelsSolved::test_level_solved[open_room] PASSED
tests/test_e2e.py::TestAllLevelsSolved::test_level_solved[simple_corridor] PASSED
tests/test_e2e.py::TestAllLevelsSolved::test_level_solved[lava_crossing] PASSED
tests/test_e2e.py::TestAllLevelsSolved::test_level_solved[door_key] PASSED
tests/test_e2e.py::TestAllLevelsSolved::test_level_solved[two_rooms] PASSED
```

Full suite: 101 passed (96 existing + 5 new e2e).
