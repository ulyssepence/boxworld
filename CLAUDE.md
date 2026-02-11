# Boxworld

RL training + web visualization for a grid-based puzzle game. PPO agent learns to navigate 10x10 grids with walls, doors, keys, lava, and goals. Python backend trains with Gymnasium + Stable-Baselines3, records episodes to SQLite, exports models to ONNX. TypeScript frontend visualizes with React Three Fiber and runs live in-browser inference via ONNX Runtime Web.

## Quick Reference

```bash
# Python (from training/)
uv run python main.py train          # Train PPO agent (2M steps)
uv run python main.py export         # Export checkpoints to ONNX
uv run python main.py record         # Record episodes to SQLite
uv run python main.py all            # Full pipeline: train → export → record
uv run pytest -v                     # Run all 96 Python tests

# TypeScript (from visualize/)
npm run build                        # Bundle frontend + copy ONNX WASM files to static/
bun run src/server.ts                # Start dev server on :8000
npx playwright test                  # Run all 19 e2e tests
npx playwright test --workers 1      # Sequential (needed for visual/agent tests)
```

## Game Mechanics

The agent navigates a 10x10 grid. Cell types: Floor (walkable), Wall (blocks movement), Door (blocks until toggled with key), Key (pick up with Pickup action), Goal (reach to win), Lava (instant death). Six actions: Up, Down, Left, Right, Pickup (grab key if standing on one), Toggle (open adjacent door if holding key, consumes key).

**Rewards:** Goal = +1.0, Lava = -1.0, each step = -0.01, key pickup = +0.2, door toggle = +0.2, max 200 steps then truncated at -2.0. Reward shaping uses subgoal-chain BFS: at reset, `_solve_subgoals()` computes key→door→...→goal sequence. Bidirectional shaping: +0.05 per BFS step closer, -0.05 per BFS step farther from current subgoal. Uses `_bfs_distance_safe` (avoids lava) so shaping never incentivizes walking through lava.

**Observation:** Flat Float32 array of size `width*height + 3` = 103 for 10x10. Encoding: `[grid[0][0], ..., grid[h-1][w-1], agent_x, agent_y, has_key]`.

## Critical Alignment Rules

These must stay in sync across Python and TypeScript or inference will silently produce wrong results:

1. **Observation encoding** — `environment.py:_get_obs()` and `ml.ts:stateToTensor()` must produce identical flat Float32 arrays: `[grid[0][0], grid[0][1], ..., grid[h-1][w-1], agent_x, agent_y, has_key]`
2. **Grid indexing** — Row-major everywhere: `grid[y][x]`
3. **Enum values** — `CellType` (Floor=0, Wall=1, Door=2, Key=3, Goal=4, Lava=5) and `Action` (Up=0, Down=1, Left=2, Right=3, Pickup=4, Toggle=5) must match between `environment.py` constants and `types.ts` enums
4. **ONNX contract** — Input name `"obs"`, output name `"q_values"`, 6 actions
5. **DB schema** — `record.py:initialize_db()` and `db.ts:initialize()` must match

## File Map

### Python — `training/`

| File | Purpose | Key exports |
|------|---------|-------------|
| `environment.py` | Gymnasium env — grid world with walls, doors, keys, lava, goals | `BoxworldEnv` (reset, step, _get_obs, _generate_level, _load_level, _solve_subgoals) |
| `train.py` | PPO training with SB3 | `Trainer`, `TrainerConfig` (dataclass with hyperparams) |
| `record.py` | Play episodes with trained models, store every step to SQLite | `Recorder` (record_all, record_episode) |
| `export.py` | Extract actor network from SB3 checkpoint, export to ONNX | `Exporter`, `OnnxablePolicy` (export_checkpoint, export_all, verify_export) |
| `main.py` | Argparse CLI entry point | Subcommands: train, record, export, all |

Tests: `tests/test_environment.py` (69), `tests/test_train.py` (5), `tests/test_record.py` (11), `tests/test_export.py` (7), `tests/test_bugs.py` (4), `tests/test_e2e.py` (5)

### TypeScript — `visualize/src/`

| File | Purpose | Key exports |
|------|---------|-------------|
| `types.ts` | Shared types and enums | `CellType`, `Action`, `Level`, `GameState`, `Step`, `Episode`, `QValues` |
| `play.ts` | Pure game logic (immutable) | `createInitialState`, `step`, `isValidMove`, `getWallPositions`, `generateLevel` |
| `ml.ts` | ONNX Runtime Web inference | `stateToTensor`, `Agent` class (load, getQValues, selectAction) |
| `db.ts` | SQLite via bun:sqlite (server-side only) | `DB` class (getCheckpoints, getLevels, getLevelWithEpisodes) |
| `api.ts` | Fetch wrappers for REST endpoints | `fetchCheckpoints`, `fetchLevels`, `fetchLevel` |
| `util.tsx` | React context + reducer for app state | `AppState`, `AppAction`, `AppProvider`, `useApp`, `usePlayback` |
| `render.tsx` | R3F 3D scene components | `Grid`, `Walls`, `Items`, `Agent` (with lerp), `QValueArrows`, `Scene` |
| `shader.tsx` | GLSL post-processing for R3F | `GLSLShader` (renders scene to FBO, applies fragment shader) |
| `app.tsx` | Main UI — sidebar controls + 3D view | `App`, `GameView`, `Sidebar`, `runAgent` (live ONNX inference) |
| `server.ts` | Express server (port 8000) | Routes: `/api/checkpoints`, `/api/levels`, `/api/levels/:id`, `/checkpoints/*.onnx` |
| `client.tsx` | React root entry point | Renders `<App>` into `#root` |
| `audio.ts` | Audio wrapper (howler.js) — currently unused | `Player` class |

Tests: `tests/e2e/smoke.spec.ts` (9), `tests/e2e/visual.spec.ts` (2), `tests/e2e/agent-movement.spec.ts` (2), `tests/e2e/basic-ui-flow.spec.ts` (5)

**Unused files:** `audio.ts` (no audio clips defined), `scripts/generate-models.py` (R3F uses built-in geometries instead of .glb files)

### Data — `data/`

```
data/
├── levels/              # Hand-designed level JSON files (5 levels)
│   ├── open_room.json
│   ├── simple_corridor.json
│   ├── lava_crossing.json
│   ├── door_key.json
│   └── two_rooms.json
├── checkpoints/         # SB3 .zip + ONNX .onnx (10k to 500k steps, every 10k)
└── db.sqlite            # Recorded episodes (SQLite WAL mode)
```

**Level JSON format:**
```json
{ "id": "simple_corridor", "name": "Simple Corridor", "width": 10, "height": 10,
  "grid": [[1,1,1,...], ...], "agentStart": [1, 1] }
```

**SQLite tables:** agents, episodes, steps (with state_json, q_values_json), checkpoints

## UI State Flow

1. User selects level → `api.fetchLevel(id)` → `LOAD_LEVEL` dispatched → sets `currentLevel`, `episodes` (sorted by reward DESC), `currentEpisodeIndex: 0`, `currentStep: 0`
2. Sidebar shows episode selector + playback controls. Step Info panel shows current step's position, action, reward, Q-values.
3. Step forward/backward buttons dispatch `STEP_FORWARD`/`STEP_BACKWARD`. Range slider dispatches `SEEK`. Play button starts interval-based auto-stepping via `usePlayback` hook.
4. "Run Agent" loads ONNX model at selected checkpoint, runs `play.step()` in a loop up to 200 steps, dispatches `LOAD_INFERENCE_EPISODE` with the generated episode.
5. `GameView` reads `episode.steps[currentStep].state.agentPosition` for agent position, falls back to `level.agentStart` when no step data. The `Agent` component lerps between previous and current position.

**3D coordinate mapping:** Grid cell `(x, y)` maps to Three.js position `[x, 0, y]`. Agent sphere at `[x, 0.4, y]`, walls at `[x, 0.5, y]` (1x1x1 boxes).

## Architecture Gotchas

**ONNX Runtime Web** — Cannot be bundled with esbuild due to WASM issues. Loaded via `<script>` tag in `index.html` which sets `globalThis.ort`. The build script copies `ort.min.js` + `.wasm` files to `static/`.

**Bundler is esbuild** — Not Vite or webpack. Single `esbuild src/client.tsx --bundle --minify --outfile=static/client.js`.

**Runtime is Bun** — Server uses `bun:sqlite` (not better-sqlite3). Run server with `bun run src/server.ts`.

**Python uses uv** — Not pip. Run with `cd training && uv run pytest -v` or `uv run python main.py ...`.

**Checkpoint naming** — Pattern: `boxworld_{steps}_steps.{zip,onnx}`. The `.zip` is SB3 format, `.onnx` is for browser. URL served at `/checkpoints/boxworld_{steps}_steps.onnx`.

**Toggle door mechanics** — Both Python and TypeScript toggle ANY adjacent door (all 4 directions) when the agent has a key.

**Recording records state BEFORE action** — Each step's `state_json` is the state that led to the action, not the resulting state.

**Agent lerp animation** — `render.tsx:Agent` lerps between `prevPosition` and `position` over ~125ms using smooth step interpolation. Needs `prevPosition` prop to animate.

**Canvas preserveDrawingBuffer** — Set to `true` on the R3F Canvas so Playwright tests can read WebGL pixels via `drawImage`.

**Playwright tests** — Use Chromium only. Visual baselines are platform-specific. Agent tests use pixel centroid analysis (scan for cyan R<100, G>140, B>140). React 19 `<select>` onChange requires `__reactProps` fiber hack in tests.

**SQLite WAL mode** — Enabled by both Python (record.py) and TypeScript (db.ts) for concurrent read/write compatibility.

**record.py --min-steps** — Use `--min-steps 500000` to only record episodes from the best checkpoint, skipping early poorly-trained ones.

**Training quality** — Default is 10M steps with CNN feature extractor (one-hot grid → 2 conv layers → avg pool + FC), bidirectional lava-safe subgoal-chain shaping (±0.05), curriculum (difficulty ramps 0→1 over 40% of training), and weighted designed level sampling. Config: `designed_level_prob=0.15`, `n_steps=2048`, `batch_size=256`, `lr=3e-4`, `ent_coef=0.05`, `use_cnn=True`. All 10 designed levels + 1 holdout level solved, plus >= 40% generalization on 100 random procedural levels (verified by `test_e2e.py` and `test_generalization.py`).
