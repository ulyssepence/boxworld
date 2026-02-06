# Boxworld Implementation Task Plan

A breakdown of the Boxworld RL Visualization Spec into discrete tasks, each completable by a coding agent within a single context window. Tasks are ordered by dependency. Each task includes files to touch, verification steps, and testing strategies.

---

## Implementation Progress

| Task | Status | Commit | Notes |
|------|--------|--------|-------|
| 1. Scaffolding | **DONE** | `8180d42` | Pre-commit hook needed path fix (git returns repo-relative paths, hook cd's into subdirs) |
| 2. Types + Game Logic | **DONE** | `6d86c30` | 369 lines added; esbuild bundles play.ts in 4ms |
| 3. 3D Models | **DONE** | `9f20a32` | @gltf-transform/core v4 API changed: `setIndex` -> `setIndices`, `createPrimitive()` is on doc not mesh |
| 4. Python Gymnasium Env | **DONE** | `a35db87` | 36 pytest tests pass; added `[tool.pytest.ini_options] pythonpath = ["."]` to pyproject.toml |
| 5. Express API + SQLite | **DONE** | `0006d51` | **Switched from better-sqlite3 to bun:sqlite** -- better-sqlite3 not supported in Bun runtime |
| 6. R3F Scene + App UI | **DONE** | `4b7e65f` | 760 lines; uses inline Three.js primitives (not GLB models); dark theme sidebar |
| 7. DQN Training | **DONE** | `a6bd061` | 5 tests pass (29s); SB3 DQN with CheckpointCallback |
| 8. Episode Recording | **DONE** | | Recorder class with greedy policy playback; 8 pytest tests; state_json matches TS GameState exactly |
| 9. ONNX Export | **DONE** | | Exporter class with torch.onnx.export; verify_export confirms ONNX↔PyTorch match within 1e-5; 5 tests |
| 10. ONNX Runtime Web | **DONE** | | ml.ts with Agent class; ort loaded via script tag (avoids esbuild WASM issues); stateToTensor matches Python _get_obs() |
| 11. Seed Data Pipeline | **DONE** | | 4 new levels (simple_corridor, dead_end, key_puzzle, lava_maze); `all` subcommand runs train→export→record; e2e smoke test passed |
| 12. Level Editor | **DONE** | | Edit mode toggle, cell click cycling (Floor→Wall→Goal→Floor), Reset button restores originalLevel |
| 13. Playwright Testing | **DONE** | | 11 tests (9 smoke + 2 visual); Chromium only; baseline screenshots; webServer auto-start in config |

### Key Discoveries & Deviations

1. **bun:sqlite instead of better-sqlite3 (Task 5):** The `better-sqlite3` npm package uses native bindings that are not supported in Bun. Switched `db.ts` to use `import { Database } from 'bun:sqlite'` which has a nearly identical API. The `better-sqlite3` and `@types/better-sqlite3` packages were removed from package.json. Future tasks referencing better-sqlite3 should use bun:sqlite instead.

2. **Pre-commit hook path handling (Task 1):** `git diff --cached --name-only` returns repo-relative paths (e.g. `visualize/src/foo.ts`). The hook `cd`s into subdirectories, so paths need `sed` stripping (e.g. `sed 's|^visualize/||'`). Initial version failed on first commit attempt.

3. **@gltf-transform/core v4 breaking changes (Task 3):** The agent's initial script used `doc.createPrimitive().setAttribute().setIndex()` chain which doesn't work in v4. Fixed to: `doc.createPrimitive()` returns a standalone Primitive, methods are `setAttribute()`, `setIndices()` (not `setIndex`), and the primitive is added to a mesh via `mesh.addPrimitive(prim)`.

4. **Task 6 uses inline primitives, not GLB models:** The R3F scene renders walls as `<boxGeometry>`, floor as `<planeGeometry>`, etc. with colored `<meshStandardMaterial>`. The GLB files from Task 3 are not loaded yet. This is fine -- GLB loading can be added when hand-made Blender models replace the placeholders.

5. **Test counts exceeded spec:** Task 4 spec listed 14 tests; implementation has 36 (added edge cases, direction tests, observation content tests, info dict tests). Task 7 has all 5 specified tests.

6. **Observation encoding alignment:** Both Python (`environment.py`) and TypeScript (`play.ts`) use `grid[y][x]` row-major order. Observation is `[flattened_grid..., agent_x, agent_y, has_key]`. This must remain aligned for ONNX inference (Task 10).

7. **onnxruntime-web loaded via script tag (Task 10):** esbuild cannot bundle onnxruntime-web's WASM files. Solution: `ort.min.js` + `.wasm` files are copied to `static/` in the build script, loaded via `<script>` in index.html, and accessed as `globalThis.ort` in `ml.ts`. This avoids all bundling issues.

8. **onnxscript dependency (Task 9):** `torch>=2.10.0` requires `onnxscript>=0.6.0` for `torch.onnx.export`. This was auto-added by `uv add` during Task 9.

9. **Task 8 records a final terminal state:** The Recorder appends an extra step after the episode ends with `done=True`, `action=0`, `reward=0.0`, and no Q-values. This gives the frontend a state to display for the terminal position.

10. **Task 11 end-to-end smoke test:** `python main.py all --steps 100 --interval 50` successfully trained, exported 2 ONNX checkpoints, and recorded 50 episodes (2 checkpoints × 5 levels × 5 runs).

11. **Playwright only Chromium (Task 13):** Only Chromium browser installed to save CI time. Config can be extended to Firefox/WebKit later. Visual regression baselines are platform-specific (`-chromium-darwin.png`).

---

## Task 1: Project Scaffolding, Formatter, and Pre-commit Hook

**Files to create/modify:**
- `.prettierrc` (new) -- single Prettier config file
- `training/pyproject.toml` (modify) -- add Ruff as dev dependency, add `[tool.ruff]` section
- `.git/hooks/pre-commit` (new) -- shell script that runs formatters on staged files
- `visualize/package.json` (modify) -- add Prettier devDependency, rename project from "tic-tac-toe" to "boxworld", add `format` script
- `visualize/tsconfig.json` (new) -- for editor support (esbuild ignores it but IDEs need it)
- `visualize/static/index.html` (modify) -- fix title from "Tic Tac Toe?" to "Boxworld"
- `.gitignore` (create/modify) -- proper ignores for data/, node_modules, .venv, *.pyc, etc.

**What it accomplishes:**
- Ruff for Python (zero-config, fast, replaces black+isort)
- Prettier for TypeScript (zero-config defaults)
- Pre-commit hook script: runs `ruff format` on staged Python files and `npx prettier --write` on staged TypeScript files
- Fixes leftover "tic-tac-toe" naming from the project template

**Verification:**
- `ruff format --check training/` passes
- `npx prettier --check visualize/src/` passes
- Making a commit triggers the pre-commit hook and auto-formats staged files

**Dependencies:** None (first task)

---

## Task 2: TypeScript Types and Pure Game Logic

**Files to create/modify:**
- `visualize/src/types.ts` (rewrite) -- all shared types
- `visualize/src/play.ts` (rewrite) -- pure game logic, zero React/Three.js dependencies
- `data/levels/four_rooms.json` (new) -- converted from existing `.minigrid` ASCII format

**What it accomplishes:**

`types.ts` defines:
- Enums: `CellType` (Floor, Wall, Door, Key, Goal, Lava), `Action` (Up, Down, Left, Right, Pickup, Toggle)
- Interfaces: `Level`, `GameState`, `QValues`, `Step`, `Episode`
- API response types for the server endpoints

`play.ts` exports pure functions:
- `createInitialState(level)` -- returns a fresh GameState
- `step(state, action)` -- returns next GameState (immutable)
- `isValidMove(state, action)` -- checks if move is legal
- `getWallPositions(level)` -- returns array of [x, y] for wall cells
- `parseLevelJson(json)` -- parses level from JSON file format
- `generateLevel(seed)` -- procedural level generation with seeded PRNG

Uses qualified imports: `import * as t from './types'`

**Verification:**
- `esbuild src/play.ts --bundle --platform=node` compiles without error
- Manual checks: `createInitialState` returns `done: false`, agent at `agentStart`
- `step()` into a wall leaves position unchanged
- `step()` onto goal sets `done: true`, reward = +1

**Hypothesis-based testing for game logic:**
- Stepping into a wall leaves position unchanged
- Picking up a key sets `inventory.hasKey = true`
- Toggling a door with a key opens it (door becomes floor)
- Stepping onto goal sets `done = true` and reward = +1
- Stepping onto lava sets `done = true` and reward = -1

**Dependencies:** Task 1

---

## Task 3: 3D Placeholder Models

**Format choice: glTF Binary (.glb)**
- Industry standard for web 3D
- Compact binary format
- Native Blender export support
- Loaded by Three.js `GLTFLoader` / Drei's `useGLTF`

**Files to create:**
- `visualize/static/models/floor.glb` -- flat 1x1 plane, gray (#888)
- `visualize/static/models/wall.glb` -- 1x1x1 cube, dark gray (#444)
- `visualize/static/models/door.glb` -- 0.1x1x1 slab, brown (#8B4513)
- `visualize/static/models/key.glb` -- small torus, gold (#FFD700)
- `visualize/static/models/goal.glb` -- flat circle, green (#00FF88)
- `visualize/static/models/lava.glb` -- flat 1x1 plane, red/orange (#FF4500)
- `visualize/static/models/player.glb` -- 0.5 radius sphere, cyan (#00FFFF)

**What it accomplishes:**
Creates programmatic stand-in .glb files via a Node.js generation script using Three.js and `@gltf-transform/core`. Each is a single mesh with a distinct color. The user will later replace these with hand-made Blender models exported as .glb.

**Verification:**
- All 7 `.glb` files exist and are > 0 bytes
- Loading them with `useGLTF` in R3F produces no errors (verified in Task 6)

**Dependencies:** Task 1 (package.json updated)

---

## Task 4: Python Gymnasium Environment

**Files to create/modify:**
- `training/environment.py` (new) -- `BoxworldEnv(gymnasium.Env)` class
- `training/pyproject.toml` (modify) -- add `pytest` dev dependency
- `training/tests/__init__.py` (new)
- `training/tests/test_environment.py` (new)

**What it accomplishes:**
Implements the custom Gymnasium environment as a reusable library class:
- Configurable grid size (default 10x10)
- Cell types: Floor(0), Wall(1), Door(2), Key(3), Goal(4), Lava(5)
- 6 discrete actions: Up, Down, Left, Right, Pickup, Toggle
- Observation: flattened grid (width*height) + agent position (2) + has_key (1) = 103 floats for 10x10
- Rewards: +1 goal, -1 lava, -0.01 per step
- Level loading from JSON files
- Seeded procedural generation via `reset(seed=N)`
- Gymnasium-compliant `reset()` and `step()` methods
- Max 200 steps before truncation

**Verification:**
```
cd training && uv run pytest tests/test_environment.py -v
```

**Pytest test cases:**
- `test_reset_returns_valid_observation` -- obs shape matches observation_space, info is dict
- `test_observation_space_shape` -- for 10x10: shape is (103,)
- `test_action_space_size` -- 6 discrete actions
- `test_move_to_empty_cell` -- agent moves, reward is -0.01
- `test_move_into_wall` -- agent stays, reward is -0.01
- `test_pickup_key` -- on key tile + Pickup -> has_key = True, key removed from grid
- `test_pickup_without_key` -- Pickup on empty tile -> no change
- `test_toggle_door_with_key` -- adjacent to door + has_key + Toggle -> door becomes floor
- `test_toggle_door_without_key` -- no change
- `test_reach_goal` -- step onto goal -> reward = +1, done = True
- `test_step_onto_lava` -- reward = -1, done = True
- `test_deterministic_seed` -- same seed produces identical layout
- `test_load_level_from_json` -- load four_rooms.json, verify grid dimensions
- `test_max_steps_truncation` -- after 200 steps, truncated = True

**Dependencies:** Task 2 (JSON level format must match between Python and TypeScript)

---

## Task 5: Express API Server with SQLite

> **IMPLEMENTATION NOTE:** Switched from `better-sqlite3` to `bun:sqlite`. The `better-sqlite3` package uses native C++ bindings that Bun does not support. `bun:sqlite` has an almost identical API (`prepare().all()`, `.exec()`, etc.) and is built into the Bun runtime.

**Files to create/modify:**
- `visualize/src/server.ts` (rewrite) -- thin entrypoint: instantiate DB, mount routes, listen
- `visualize/src/db.ts` (new) -- SQLite wrapper using `bun:sqlite`
- `visualize/src/types.ts` (modify if needed) -- API response types

**What it accomplishes:**

`db.ts` exports a `DB` class:
- `constructor(dbPath)` -- opens SQLite connection with `{ create: true }`
- `initialize()` -- creates the 4 tables (agents, episodes, steps, checkpoints) if not present
- `getCheckpoints()` -- returns list of checkpoints
- `getLevels()` -- returns level metadata (reads JSON files from data/levels/)
- `getLevelWithEpisodes(levelId)` -- returns level definition + all episodes with full step data

`server.ts` becomes:
```typescript
import * as db from './db'
const database = new db.Database('../data/db.sqlite')
database.initialize()
// mount routes, serve static, listen on 8000
```

Endpoints:
- `GET /api/checkpoints` -> `{ checkpoints: [{ id, trainingSteps }] }`
- `GET /api/levels` -> `{ levels: [{ id, name, seed }] }`
- `GET /api/levels/:levelId` -> `{ level: {...}, episodes: [{...steps}] }`

Sets `Cache-Control: public, max-age=31536000, immutable` on level endpoints. Removes old WebSocket imports.

**Verification:**
```
cd visualize && npm run build && bun run src/server.ts &
curl http://localhost:8000/api/checkpoints   # -> {"checkpoints":[]}
curl http://localhost:8000/api/levels         # -> {"levels":[...]}
curl http://localhost:8000/                   # -> serves index.html
kill %1
```

**Express endpoint tests (node:test or inline script):**
- `GET /api/checkpoints` returns 200 with `{ checkpoints: [] }` when DB is empty
- `GET /api/levels` returns 200 with array matching JSON files in data/levels/
- `GET /api/levels/four_rooms` returns level data with correct grid dimensions
- `GET /api/levels/nonexistent` returns 404
- Cache-Control headers present on responses
- Static file serving works

**Dependencies:** Task 1, Task 2 (types.ts, level JSON)

---

## Task 6: React Three Fiber Scene, State Management, and Post-processing Shader

**Files to create/modify:**
- `visualize/src/render.tsx` (new) -- R3F scene components
- `visualize/src/util.tsx` (new) -- React Context, reducer, hooks
- `visualize/src/api.ts` (new) -- API client fetch wrappers
- `visualize/src/app.tsx` (new) -- main UI + entry point
- `visualize/src/client.tsx` (rewrite) -- thin entry rendering `<app.App />`

**What it accomplishes:**

`util.tsx`:
- AppContext with useReducer
- Actions: LOAD_LEVEL, LOAD_EPISODE, SEEK, PLAY, PAUSE, STEP_FORWARD, STEP_BACKWARD, SET_SPEED, TOGGLE_EDIT_MODE, EDIT_CELL
- Exports: `AppProvider`, `useApp()`, `usePlayback()` hook (auto-advances steps with useFrame)

`api.ts`:
- `fetchCheckpoints()`, `fetchLevels()`, `fetchLevel(id)` -- simple fetch wrappers

`render.tsx`:
- `Scene` -- top-level Canvas wrapper with lighting + OrbitControls
- `Grid` -- floor tile instances
- `Walls` -- wall mesh instances
- `Items` -- keys, doors, goal, lava meshes
- `Agent` -- animated mesh with lerp via useFrame + refs (bypasses React re-renders)
- `QValueArrows` -- directional indicators showing relative Q-values on grid
- **Wraps children in `<GLSLShader code="color = scene(uv);">` from existing shader.tsx** -- passthrough post-processing (user will customize later)

`app.tsx`:
- Top-level `App` component with Canvas, playback controls, level selector, checkpoint filter, Q-value side panel

`client.tsx`:
- Imports and mounts `<app.App />`

Uses qualified imports throughout: `import * as t from './types'`, `import * as play from './play'`, etc.

**Verification:**
```
cd visualize && npm run build && bun run src/server.ts
# Open http://localhost:8000 -- see 3D grid, controls, level loaded from API
```

**Hypothesis-based Three.js testing:**
- When a level with 5 wall tiles is loaded, the scene contains 5 wall meshes
- When episode has 10 steps and we SEEK to step 5, the agent mesh position matches step 5's agentPosition
- When PLAY then PAUSE dispatched, `isPlaying` toggles correctly
- GLSLShader renders without WebGL errors (no shader compilation errors in console)

**Dependencies:** Task 2, Task 5, Task 3 (optional -- inline primitives also work)

---

## Task 7: DQN Training Pipeline

**Files to create/modify:**
- `training/train.py` (new) -- `Trainer` class
- `training/main.py` (rewrite) -- thin entry point
- `training/tests/test_train.py` (new)

**What it accomplishes:**

`train.py` exports `Trainer` class:
- `__init__(env, config)` -- configures DQN with Stable-Baselines3
- `train(total_steps, checkpoint_interval, checkpoint_dir)` -- runs training loop, saves SB3 checkpoints every N steps
- `load_checkpoint(path)` -- loads saved model

Hyperparameters from spec: `learning_rate=1e-4, buffer_size=100_000, learning_starts=1000, batch_size=32, gamma=0.99, target_update_interval=1000, exploration_fraction=0.1, exploration_final_eps=0.05`

`main.py` instantiates env, trainer, and calls `train()`. Library-like: all logic in `train.py`, `main.py` is just wiring.

**Verification:**
```
cd training && uv run pytest tests/test_train.py -v
```

**Pytest test cases:**
- `test_trainer_initializes_with_valid_env` -- DQN model creates without error
- `test_short_training_run` -- train for 100 steps without crashing
- `test_checkpoints_saved_at_intervals` -- train 200 steps, interval=100 -> 2 checkpoint files
- `test_checkpoint_can_be_loaded` -- save then load, predict still works
- `test_custom_hyperparameters` -- config overrides work

**Dependencies:** Task 4

---

## Task 8: Episode Recording to SQLite

**Files to create/modify:**
- `training/record.py` (new) -- `Recorder` class
- `training/main.py` (modify) -- add `record` command
- `training/tests/test_record.py` (new)

**What it accomplishes:**

`record.py` exports `Recorder` class:
- `__init__(db_path)` -- opens SQLite, creates schema if needed
- `initialize_db()` -- creates the 4 tables (agents, episodes, steps, checkpoints)
- `register_agent(name, training_steps)` -- inserts agent row
- `record_episode(agent, env, level_seed, run_number)` -- plays one episode, records each step's state, action, reward, Q-values into SQLite
- `record_all(checkpoint_dir, level_seeds, runs_per_level)` -- iterates checkpoints x levels x runs

Per-step data stored: `state_json` (grid + agent position + inventory), `q_values_json` (all 6 actions), action, reward, done flag.

**Verification:**
```
cd training && uv run pytest tests/test_record.py -v
```

**Pytest test cases:**
- `test_initialize_creates_tables` -- all 4 tables exist after initialize_db()
- `test_register_agent` -- agent row created with correct name and training_steps
- `test_record_single_episode` -- episode row + step rows exist in DB
- `test_step_data_is_valid_json` -- state_json and q_values_json parse as valid JSON
- `test_q_values_have_six_actions` -- each step has exactly 6 Q-value entries
- `test_episode_total_reward_matches_steps` -- sum of step rewards equals episode total_reward
- `test_deterministic_replay` -- same seed produces same initial state
- `test_multiple_runs_recorded` -- recording 3 runs creates 3 episode rows

**Dependencies:** Task 4, Task 7

---

## Task 9: ONNX Export

**Files to create/modify:**
- `training/export.py` (new) -- `Exporter` class
- `training/main.py` (modify) -- add `export` command
- `training/tests/test_export.py` (new)

**What it accomplishes:**

`export.py` exports `Exporter` class:
- `export_checkpoint(sb3_model_path, output_onnx_path, obs_size)` -- loads SB3 checkpoint, extracts Q-network, exports via `torch.onnx.export()`
- `export_all(checkpoint_dir, output_dir)` -- exports all `.zip` checkpoints to `.onnx`
- `verify_export(onnx_path, sb3_model_path, test_obs)` -- runs both models on same input, asserts outputs match within tolerance

Input shape: `(1, obs_size)` float32. Output shape: `(1, 6)` float32 (Q-values).

Also registers each exported checkpoint in the SQLite `checkpoints` table.

**Verification:**
```
cd training && uv run pytest tests/test_export.py -v
```

**Pytest test cases:**
- `test_export_produces_onnx_file` -- .onnx file exists and > 0 bytes
- `test_onnx_model_loads_with_onnxruntime` -- `ort.InferenceSession(path)` succeeds
- `test_onnx_output_shape` -- output is (1, 6) for single observation
- `test_onnx_matches_pytorch` -- ONNX output matches SB3 predict within 1e-5 tolerance
- `test_export_all_checkpoints` -- given 3 checkpoints, produces 3 .onnx files

**Dependencies:** Task 7, Task 8

---

## Task 10: ONNX Runtime Web In-Browser Inference

**Files to create/modify:**
- `visualize/src/ml.ts` (new) -- ONNX inference wrapper
- `visualize/package.json` (modify) -- add `onnxruntime-web`
- `visualize/src/app.tsx` (modify) -- wire up live inference mode

**What it accomplishes:**

`ml.ts` exports:
- `class Agent` with `load(modelUrl)`, `getQValues(state)`, `selectAction(state)`
- `stateToTensor(state)` -- converts GameState to flattened float32 matching Python's observation format

When user clicks "Run Agent" in edit mode, the app loads the selected ONNX checkpoint, runs inference step-by-step, and plays the result through the existing playback system.

**Critical alignment:** The tensor format in `stateToTensor()` must exactly match `environment.py`'s observation encoding (grid flattening order, position encoding, inventory flag).

**Verification:**
```
cd visualize && npm run build  # esbuild bundles onnxruntime-web
# Browser: load app, select checkpoint, click "Run Agent"
```

**Testing:**
- `stateToTensor` produces Float32Array of correct length (103 for 10x10 grid)
- `ml.Agent` instantiates without error
- With a real ONNX model, `getQValues` returns 6 values

**Dependencies:** Task 2, Task 6, Task 9

---

## Task 11: Seed Data Pipeline (End-to-End)

**Files to create/modify:**
- `training/main.py` (modify) -- complete CLI with train/record/export modes via argparse
- `data/levels/simple_corridor.json` (new)
- `data/levels/dead_end.json` (new)
- `data/levels/key_puzzle.json` (new)
- `data/levels/lava_maze.json` (new)

**What it accomplishes:**
Creates hand-designed preset level JSON files and provides the complete CLI:
```
python main.py train --steps 500000 --interval 10000
python main.py export
python main.py record --runs-per-level 5
python main.py all  # runs train + export + record sequentially
```

`main.py` is a thin argparse wrapper that instantiates `BoxworldEnv`, `Trainer`, `Recorder`, `Exporter` and calls their methods. Library-like: all logic lives in the library modules.

**Verification:**
```
cd training
uv run python main.py train --steps 1000 --interval 500
uv run python main.py export
uv run python main.py record --levels simple_corridor --runs 1
sqlite3 ../data/db.sqlite "SELECT COUNT(*) FROM episodes;"
ls ../data/checkpoints/*.onnx
```

**Dependencies:** Tasks 4, 7, 8, 9

---

## Task 12: Level Editor Mode

**Files to modify:**
- `visualize/src/app.tsx` -- add edit mode UI (toggle button, cell type palette, Run Agent, Reset)
- `visualize/src/render.tsx` -- add onClick handlers on grid cells for editing
- `visualize/src/util.tsx` -- add EDIT_CELL action handler in reducer

**What it accomplishes:**
When "Edit Mode" toggled:
- Clicking a grid cell cycles: Floor -> Wall -> Goal -> Floor
- Grid updates immediately in the 3D view
- "Run Agent" button triggers live ONNX inference on edited level
- "Reset" button restores original level

Click detection uses R3F's built-in `onClick` on mesh components.

**Verification:**
- Open app, load level, toggle Edit Mode
- Click floor tile -> becomes wall (visual change)
- Click wall -> becomes goal
- Click "Run Agent" -> agent runs inference on modified level
- Click "Reset" -> grid returns to original

**Hypothesis-based tests:**
- Dispatching EDIT_CELL(x, y, Wall) on Floor updates `state.level.grid[y][x]` to `CellType.Wall`
- After editing, mesh count changes (new wall = new box mesh added)
- Resetting after edit restores original grid values

**Dependencies:** Task 6, Task 10

---

## Task 13: Cross-Browser Testing with Playwright

**Files to create/modify:**
- `visualize/package.json` (modify) -- add `@playwright/test` devDependency
- `visualize/playwright.config.ts` (new) -- Playwright config for Chromium, Firefox, WebKit
- `visualize/tests/e2e/smoke.spec.ts` (new) -- functional smoke tests
- `visualize/tests/e2e/visual.spec.ts` (new) -- screenshot-based visual checks

**What it accomplishes:**

`smoke.spec.ts`:
- Page loads without JS errors (check console for errors)
- Canvas element present with non-zero dimensions
- API endpoints return 200
- Playback controls visible and clickable
- Level selector contains expected levels

`visual.spec.ts`:
- Screenshots at key states: initial load, level loaded, during playback, edit mode
- Compare against baselines with `toMatchSnapshot()` and `maxDiffPixelRatio: 0.05`

**Cross-browser and visual regression strategy:**

Three approaches, in order of recommendation:

1. **Playwright screenshot comparison (recommended for this project):** Built-in `expect(page).toHaveScreenshot()` with `maxDiffPixelRatio: 0.05` threshold. Free, integrated, works in CI. Generous threshold accommodates WebGL anti-aliasing and GPU variance between environments. Run across Chromium, Firefox, and WebKit.

2. **Multimodal LLM review (novel, optional CI step):** Take Playwright screenshots, send to a vision-capable LLM with a prompt like "Does this screenshot show a rendered 3D gridworld with visible walls, floor, and agent? Are there rendering artifacts?" Useful because it's semantic rather than pixel-level -- it handles anti-aliasing differences gracefully. Good for periodic review or gating releases, not every commit. Nondeterministic and costly, so best as an optional CI step.

3. **Percy / Chromatic (skip for now):** Commercial visual regression services. Consistent cloud rendering reduces GPU variance. Industry standard for design-system-heavy apps. Overkill at this project's scale -- revisit if the team grows.

**Verification:**
```
cd visualize
npx playwright install
npx playwright test tests/e2e/smoke.spec.ts
```

**Dependencies:** Task 5, Task 6

---

## Task Dependency Graph

```
Task 1: Scaffolding, Formatter, Pre-commit
  |
  +---> Task 2: Types + Game Logic
  |       |
  |       +---> Task 4: Python Gymnasium Env
  |       |       |
  |       |       +---> Task 7: DQN Training
  |       |       |       |
  |       |       |       +---> Task 8: Episode Recording
  |       |       |       |       |
  |       |       |       |       +---> Task 9: ONNX Export
  |       |       |       |               |
  |       |       |       +---------------+
  |       |       |                       |
  |       |       |               Task 11: Seed Data Pipeline
  |       |       |
  |       +---> Task 5: Express API Server
  |       |       |
  |       +-------+---> Task 6: R3F Scene + Shader + App
  |                       |
  |                       +---> Task 10: ONNX Runtime Web
  |                       |       |
  |                       +-------+---> Task 12: Level Editor
  |                       |
  |                       +---> Task 13: Playwright Testing
  |
  +---> Task 3: 3D Placeholder Models (independent, used by Task 6)
```

## Implementation Nudge Coverage

| Nudge | Coverage |
|-------|----------|
| 1. Python testing (pytest, standard and simple) | Tasks 4, 7, 8, 9 -- specific test cases listed per task |
| 2. Web server + hypothesis-based client testing | Task 5 (endpoint tests), Tasks 6, 12 (Three.js hypothesis tests) |
| 3. 3D placeholder models (Blender -> .glb) | Task 3 -- glTF Binary format, 7 stand-in files |
| 4. Post-processing shader (`color = scene(uv)`) | Task 6 -- GLSLShader passthrough wrapping the scene |
| 5. Formatter + pre-commit hook (single config) | Task 1 -- Ruff (Python) + Prettier (TypeScript), shared pre-commit hook |
| 6. esbuild bundling (no vite) | Existing config preserved, verified in Task 6 |
| 7. Cross-browser testing (Playwright + LLM option) | Task 13 -- Playwright across 3 browsers, LLM review as optional step |
| 8. Qualified imports (`import * as`) | All TypeScript tasks -- `import * as t`, `import * as play`, etc. |
| 9. Library-like modules (thin entrypoints) | All tasks -- `main.py` is argparse wrapper, `client.tsx` mounts App, `server.ts` wires DB + routes |

---

## Future Work & Open Questions

### Discovered during implementation

1. **GLB model loading:** Task 6 renders with inline Three.js primitives (BoxGeometry, PlaneGeometry, etc.) instead of loading the GLB files from Task 3. A future task should either:
   - Switch render.tsx to use `useGLTF` from Drei to load the .glb models
   - Or keep inline primitives until hand-made Blender models are ready, then switch directly to those

2. **Python recorder SQLite schema alignment:** The Python `record.py` (Task 8) must create the exact same 4-table schema as the TypeScript `db.ts`. The schema is defined in both places -- consider a shared schema file or at minimum verify column names match exactly. Key concern: the TypeScript side reads `state_json` and `q_values_json` and parses them as `GameState` and `QValues` -- the Python side must serialize these in exactly the matching format.

3. **bun:sqlite in Task 8 recorder:** The Python recorder writes to SQLite using Python's `sqlite3` module. The TypeScript server reads the same database using `bun:sqlite`. These are compatible at the file format level (both use standard SQLite), but WAL mode must be handled carefully -- both sides use WAL which is fine for concurrent reads.

4. **Observation encoding contract:** The critical alignment between `environment.py`'s `_get_obs()` and `ml.ts`'s `stateToTensor()` (Task 10) must be tested explicitly. Both must produce: `[grid[0][0], grid[0][1], ..., grid[h-1][w-1], agent_x, agent_y, has_key]` as Float32. Consider adding a cross-language test that runs the same level through both and compares byte-for-byte.

5. **Bundle size:** The client.js bundle is 1.1MB minified. Adding `onnxruntime-web` (Task 10) will increase this significantly. Consider:
   - Lazy-loading the ONNX runtime only when "Run Agent" is clicked
   - Using the WASM backend (smaller than WebGL) if inference speed is acceptable
   - Code-splitting with esbuild's `splitting` option (requires ESM output)

6. **Shader passthrough integration:** The `GLSLShader` component from `shader.tsx` wraps the 3D scene with `color = scene(uv);`. This is currently a no-op passthrough. The shader infrastructure is ready for custom post-processing effects (e.g., edge detection, color grading, heatmap overlays for Q-values).

7. **Missing `audio.ts` cleanup:** The `audio.ts` file still exists with a Howler.js Player class and empty `Clip` type. It's unused since the client.tsx rewrite. Should be removed or repurposed for UI sounds.
