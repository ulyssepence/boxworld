# Boxworld

RL training and 3D web visualization for a grid-based puzzle game. A DQN agent learns to navigate levels with walls, doors, keys, lava, and goals. Training episodes are recorded to SQLite, exported to ONNX, and replayed in a React Three Fiber frontend with live in-browser inference.

## Prerequisites

- Python 3.13+ with [uv](https://docs.astral.sh/uv/)
- [Bun](https://bun.sh/) runtime
- Node.js 18+ (for npm/npx)

## Install

```bash
# Python dependencies
cd training
uv sync

# TypeScript dependencies
cd ../visualize
npm install
```

## Training

All commands run from the `training/` directory.

```bash
cd training

# Train a DQN agent (saves checkpoints to ../data/checkpoints/)
uv run python main.py train --steps 500000 --interval 10000

# Export checkpoints to ONNX (for in-browser inference)
uv run python main.py export

# Record agent episodes to SQLite (for replay in the frontend)
uv run python main.py record --runs-per-level 5

# Run the full pipeline: train -> export -> record
uv run python main.py all --steps 500000 --interval 10000
```

### Training CLI options

```
train   --steps N              Total training steps (default: 500000)
        --interval N           Checkpoint save interval (default: 10000)
        --checkpoint-dir PATH  (default: ../data/checkpoints)

export  --checkpoint-dir PATH  (default: ../data/checkpoints)
        --output-dir PATH      (default: ../data/checkpoints)
        --db PATH              (default: ../data/db.sqlite)

record  --checkpoint-dir PATH  (default: ../data/checkpoints)
        --levels-dir PATH      (default: ../data/levels)
        --db PATH              (default: ../data/db.sqlite)
        --runs-per-level N     (default: 5)

all     All of the above options combined.
```

## Web app

### Build the client

```bash
cd visualize
npm run build
```

This bundles `src/client.tsx` with esbuild, and copies the onnxruntime-web WASM files to `static/`.

### Start the server

```bash
cd visualize
bun run src/server.ts
```

Open http://localhost:8000. The sidebar lets you select levels, replay recorded episodes, and run live ONNX inference with any exported checkpoint. Toggle Edit Mode to modify the grid and test the agent on custom layouts.

### Watch mode (rebuild + restart on changes)

```bash
cd visualize
npm run watch
```

## Testing

### Python tests

```bash
cd training
uv run pytest -v
```

Runs 54 tests across environment, training, recording, and ONNX export.

### Playwright end-to-end tests

```bash
cd visualize

# Install browser (first time only)
npx playwright install chromium

# Run all e2e tests (auto-starts the server)
npm run test:e2e

# Run subsets
npx playwright test tests/e2e/smoke.spec.ts
npx playwright test tests/e2e/visual.spec.ts

# Update visual regression baselines after intentional UI changes
npx playwright test tests/e2e/visual.spec.ts --update-snapshots
```

### Formatting

```bash
# Python
cd training && uv run ruff format .

# TypeScript
cd visualize && npm run format
```

## Project structure

```
training/
  environment.py    Gymnasium env (10x10 grid, 6 actions, 103-dim obs)
  train.py          SB3 DQN trainer
  record.py         Episode recorder (SQLite)
  export.py         ONNX exporter
  main.py           CLI entry point
  tests/            pytest suite

visualize/
  src/
    types.ts        Shared types and enums
    play.ts         Pure game logic (step, createInitialState)
    ml.ts           ONNX inference (Agent class, stateToTensor)
    db.ts           SQLite wrapper (bun:sqlite)
    api.ts          Fetch wrappers for API endpoints
    util.tsx        React context, reducer, hooks
    render.tsx      R3F 3D scene components
    app.tsx         Main UI (sidebar, controls, edit mode)
    server.ts       Express server (port 8000)
    client.tsx      Client entry point
  tests/e2e/        Playwright tests
  static/           Built assets, HTML, WASM files

data/
  levels/           Level JSON files (10x10 grids)
  checkpoints/      SB3 .zip and ONNX .onnx files
  db.sqlite         Episode recordings
```
