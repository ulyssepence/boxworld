# Boxworld

RL training and 3D web visualization for a grid-based puzzle game. A PPO agent with a CNN feature extractor learns to navigate 10x10 grids with walls, doors, keys, lava, and goals. Training uses Gymnasium + Stable-Baselines3 with curriculum learning and procedural level generation. Episodes are recorded to SQLite, models are exported to ONNX, and everything is visualized in a React Three Fiber frontend with live in-browser inference.

## Prerequisites

- Python 3.13+ with [uv](https://docs.astral.sh/uv/)
- [Bun](https://bun.sh/) runtime
- Node.js 18+ (for npm/npx)

## Quick Start

```bash
# Install dependencies
cd training && uv sync
cd ../visualize && npm install

# Run the full pipeline: train → export → curate seeds → record episodes
cd ../training
uv run python main.py all --steps 20000000

# Build and start the web app
cd ../visualize
npm run build
bun run src/server.ts
# Open http://localhost:8000
```

## Game Mechanics

The agent navigates a 10x10 grid with 6 cell types and 6 actions.

**Cell types:**

| Cell  | Value | Behavior |
|-------|-------|----------|
| Floor | 0     | Walkable |
| Wall  | 1     | Blocks movement |
| Door  | 2     | Blocks until toggled with a key |
| Key   | 3     | Picked up with the Pickup action |
| Goal  | 4     | Reach to win (+1.0) |
| Lava  | 5     | Instant death (-1.0) |

**Actions:** Up, Down, Left, Right, Pickup (grab key if standing on one), Toggle (open any adjacent door if holding a key; consumes the key).

**Rewards:**

| Event | Reward |
|-------|--------|
| Reach goal | +1.0 |
| Step on lava | -1.0 |
| Pick up key | +0.2 |
| Toggle door | +0.2 |
| Each step | -0.01 |
| 200 steps (truncated) | -2.0 |

**Reward shaping:** At episode reset, `_solve_subgoals()` computes a key→door→...→goal subgoal chain via BFS. Each step, the agent gets +0.05 per BFS step closer to the current subgoal and -0.05 per step farther. BFS avoids lava cells so shaping never incentivizes walking through lava.

**Observation:** Flat Float32 array of 103 values: `[grid[0][0], grid[0][1], ..., grid[9][9], agent_x, agent_y, has_key]`. Grid is row-major (`grid[y][x]`).

## Training

All commands run from the `training/` directory.

### Full pipeline

```bash
uv run python main.py all --steps 20000000
```

This runs train → export → curate → record in sequence, clearing old checkpoints and the database first.

### Individual steps

```bash
# Train PPO agent (saves checkpoints every 50k steps to ../data/checkpoints/)
uv run python main.py train --steps 20000000 --interval 50000

# Export SB3 checkpoints to ONNX (for in-browser inference)
uv run python main.py export

# Curate seeds — tests 10k random seeds against the best checkpoint,
# stores ones the agent can solve for the frontend's "Generate" button
uv run python main.py curate

# Record agent episodes to SQLite (for replay in the frontend)
uv run python main.py record --runs-per-level 5
```

### CLI reference

```
train   --steps N              Total training steps (default: 20000000)
        --interval N           Checkpoint save interval (default: 50000)
        --checkpoint-dir PATH  (default: ../data/checkpoints)
        --levels-dir PATH      (default: ../data/levels)

export  --checkpoint-dir PATH  (default: ../data/checkpoints)
        --output-dir PATH      (default: ../data/checkpoints)
        --db PATH              (default: ../data/db.sqlite)

curate  --checkpoint-dir PATH  (default: ../data/checkpoints)
        --checkpoint PATH      Specific checkpoint (default: best available)
        --db PATH              (default: ../data/db.sqlite)
        --start-seed N         (default: 0)
        --num-seeds N          Seeds to test (default: 10000)
        --max-tries N          Attempts per seed (default: 3)

record  --checkpoint-dir PATH  (default: ../data/checkpoints)
        --levels-dir PATH      (default: ../data/levels)
        --db PATH              (default: ../data/db.sqlite)
        --runs-per-level N     (default: 5)
        --min-steps N          Only record from checkpoints with >= N training steps

all     All of the above options combined.
        --steps, --interval, --checkpoint-dir, --output-dir, --db,
        --levels-dir, --runs-per-level, --min-steps, --num-seeds
```

### PPO architecture

**Hyperparameters:** lr=3e-4, n_steps=2048, batch_size=256, n_epochs=10, gamma=0.99, clip_range=0.2, ent_coef=0.05, 8 parallel environments.

**CNN feature extractor (GridCNNExtractor):**
1. One-hot encode grid cells into 6 channels (one per cell type)
2. Conv2d(6→32, 3x3, padding=1) + ReLU
3. Conv2d(32→64, 3x3, padding=1) + ReLU
4. AdaptiveAvgPool2d(1) → 64 features
5. Concatenate with [agent_x, agent_y, has_key] → 67 features
6. FC(67→128, ReLU)
7. Actor: FC(128→128, ReLU) → FC(128→6) action logits
8. Critic: FC(128→128, ReLU) → FC(128→1) value

**Curriculum learning:** `CurriculumCallback` ramps environment difficulty from 0→1 over the first 40% of training. At low difficulty, generators produce mostly open rooms with few obstacles. At full difficulty, levels have complex mazes, multiple doors, and lava fields.

### Level generation

Training uses a mix of 10 hand-designed levels (sampled 15% of the time with per-level weights) and 7 procedural generators (85%):

| Generator | Weight | Description |
|-----------|--------|-------------|
| bsp_rooms | 20% | Binary space partition — carves rooms connected by corridors, doors at chokepoints |
| scattered_walls | 20% | Open layout with scattered walls and extended wall segments |
| room_partition | 14% | 1-3 wall partitions dividing the grid, doors + keys |
| lava_field | 14% | Horizontal strips, zigzag patterns, or lava patches |
| wall_segments | 12% | Corridor-creating wall segments |
| hybrid | 12% | Room partitions combined with a lava patch near goal |
| open_room | 8% | Mostly open with 0-6 scattered walls, optional doors + lava |

All generators scale difficulty with the curriculum parameter. The same generators are implemented in both Python (for training) and TypeScript (for the web UI's "Generate" button).

### Performance

The CNN generalizes to ~40% of random procedural levels (3 stochastic attempts each), compared to ~2% for an MLP baseline. All 10 designed training levels and 1 holdout level are solved reliably. Stochastic action selection (softmax sampling) significantly outperforms deterministic argmax because argmax gets stuck in loops.

## Level format

Levels are ASCII text files in `data/levels/`:

```
##########
#A       #
#        #
#        #
#  K     #
#####D####
#        #
#        #
#       G#
##########
```

| Symbol | Cell type |
|--------|-----------|
| `#` | Wall |
| `.` or space | Floor |
| `A` | Agent start (floor) |
| `K` | Key |
| `D` | Door |
| `G` | Goal |
| `~` | Lava |

There are 10 hand-designed levels plus 1 excluded level (`key_lava_gauntlet` — too hard for current training):

open_room, simple_corridor, lava_crossing, door_key, two_rooms, two_keys, open_shortcut, three_keys, zigzag_lava, dead_ends

## Web App

### Build and run

```bash
cd visualize

# Bundle client.tsx with esbuild + copy ONNX Runtime WASM files to static/
npm run build

# Start the server on http://localhost:8000
bun run src/server.ts

# Or use watch mode (auto-rebuild + restart on file changes)
npm run watch
```

### UI overview

**Top bar:**
- Level selector dropdown (hand-designed + generated levels)
- "Generate" button — creates a random procedural level from curated seeds the agent can solve
- "Edit Level" toggle — click grid cells to cycle through cell types
- Two tabs: **Recordings** and **Live**

**Recordings tab:** Browse recorded episodes sorted by reward. Playback controls: previous/play/next step buttons, seek slider, 1–20x speed.

**Live tab:** Select an ONNX checkpoint, click "Run" to watch the agent solve the current level in real-time using in-browser ONNX inference. Supports step-by-step mode. Auto-restarts when you edit a cell in edit mode.

**3D view:** React Three Fiber scene with custom GLSL shaders. The agent lerps between positions over ~125ms. Q-value arrows show the model's action preferences as colored cones. Lava cells have animated Perlin noise and particle emitters. Orbit controls for camera.

**Info panels** (bottom-right): Current step number, action, reward, position, has_key, done status, and all 6 Q-values.

### API

```
GET  /api/checkpoints       → { checkpoints: [{ id, trainingSteps }] }
GET  /api/levels             → { levels: [{ id, name, seed? }] }
GET  /api/levels/:id         → { level, episodes: [{ id, totalReward, steps }] }
GET  /api/curated-seeds      → { seeds: [...] }
GET  /checkpoints/*.onnx     → Binary ONNX model
```

## Docker

```bash
docker build -t boxworld .
docker run -p 8000:8000 -v $(pwd)/data:/app/data boxworld
```

The Dockerfile uses a multi-stage build: `oven/bun:1` builds the frontend, `oven/bun:1-slim` runs the server. Mount `data/` to provide levels, checkpoints, and the SQLite database.

## Testing

### Python (96+ tests)

```bash
cd training
uv run pytest -v
```

Covers environment mechanics, training setup, episode recording, ONNX export/verification, regression bugs, level parsing, seed curation, generalization on random levels, and end-to-end pipeline tests.

### Playwright (19 e2e tests)

```bash
cd visualize

# Install browser (first time)
npx playwright install chromium

# Run all tests (auto-starts the server)
npm run test:e2e

# Run specific suites
npx playwright test tests/e2e/smoke.spec.ts
npx playwright test tests/e2e/visual.spec.ts

# Update visual regression baselines
npx playwright test tests/e2e/visual.spec.ts --update-snapshots
```

Tests cover page loading, API endpoints, visual regression, agent rendering (pixel centroid analysis), level selection, playback controls, and ONNX inference.

### Formatting

```bash
cd training && uv run ruff format .
cd visualize && npm run format
```

## Project Structure

```
training/
  environment.py    Gymnasium env (10x10 grid, 6 cell types, 6 actions, 103-dim obs)
  train.py          SB3 PPO trainer with CNN feature extractor
  curriculum.py     CurriculumCallback — ramps difficulty 0→1
  record.py         Episode recorder to SQLite (stochastic + deterministic runs)
  export.py         ONNX exporter (extracts actor network from SB3 policy)
  curate.py         Seed curator — finds solvable random levels for frontend
  level_parser.py   ASCII .txt → level dict parser
  main.py           CLI entry point (train, export, curate, record, all)
  tests/            96+ pytest tests

visualize/
  src/
    types.ts        Shared types and enums (CellType, Action, Level, GameState, etc.)
    play.ts         Pure game logic — step, createInitialState, 5 procedural generators
    ml.ts           ONNX inference — Agent class, stateToTensor, softmax sampling
    db.ts           SQLite wrapper (bun:sqlite, WAL mode)
    api.ts          Fetch wrappers for REST endpoints
    util.tsx        React 19 context + reducer for app state
    render.tsx      R3F 3D scene — grid, walls, items, agent (lerp), Q-value arrows
    shader.tsx      GLSL post-processing — voronoi, chromatic aberration, noise
    app.tsx         Main UI — sidebar controls, edit mode, tabs, playback
    server.ts       Express server (port 8000)
    client.tsx      React root entry point
  tests/e2e/        Playwright tests (smoke, visual, agent movement, UI flow)
  static/           HTML, CSS, bundled JS, ONNX WASM runtime, GLB models

data/
  levels/           10 hand-designed ASCII .txt level files
  checkpoints/      SB3 .zip checkpoints + exported .onnx models
  db.sqlite         Recorded episodes (SQLite WAL mode)
```

## Architecture Notes

**ONNX Runtime Web** can't be bundled with esbuild due to WASM issues. It's loaded via a `<script>` tag in `index.html` which sets `globalThis.ort`. The build script copies `ort.min.js` and `.wasm` files from node_modules to `static/`.

**Bundler is esbuild**, not Vite or webpack. Single command: `esbuild src/client.tsx --bundle --minify --outfile=static/client.js`.

**Runtime is Bun.** The server uses `bun:sqlite` for database access. Run with `bun run src/server.ts`.

**Python uses uv**, not pip. All Python commands are `uv run python ...` or `uv run pytest`.

**Observation encoding alignment** is the most critical invariant: Python's `_get_obs()` and TypeScript's `stateToTensor()` must produce identical Float32 arrays. The ONNX model expects input named `"obs"` and outputs `"q_values"` (actually action logits from the PPO actor, not Q-values).

**SQLite WAL mode** is enabled by both Python (record.py) and TypeScript (db.ts) for concurrent read/write compatibility.

**Recording stores state before action** — each step's `state_json` is the state that led to the action, not the resulting state. Of 5 runs per level, 4 use stochastic sampling and 1 uses deterministic argmax (with loop detection).

**Checkpoint naming:** `boxworld_{steps}_steps.{zip,onnx}`. The `.zip` is SB3 format for Python, the `.onnx` is for the browser. Served at `/checkpoints/boxworld_{steps}_steps.onnx`.
