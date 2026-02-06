# Boxworld: Reinforcement Learning Visualization Technical Spec

A web application that visualizes pre-trained reinforcement learning agents playing gridworld environments, allowing users to inspect agent decision-making and test how well the agent's training generalizes by allowing them to edit levels.

Gridworld is an AI training game that's a 2D grid where the AI agent must move, pick up keys, bring keys to doors, go through doors, move to exit. Gridworld is an AI training game that's a 2D grid where the AI agent must move, pick up keys, bring keys to doors, go through doors, move to exit. We will use DQN (Deep Q-Network) Reinforcement Learning to train the AI. 

---

## Overview

### Core Experience

1. User loads the app → sees a 3D gridworld with a trained agent. Camera can be rotated and panned.
2. Feature 1: Agent plays through the level live, showing the "Q-values" for each action the agent can take (a Q-Value says how "good" each action feels with the current state of the game)
3. Feature 1: User can pause, scrub through the episode (instance of AI playing level) timeline, or step frame-by-frame
4. Feature 1: User edits the level (toggle walls, randomize) → agent immediately re-runs with inference
5. Feature 2: User can watch recordings of multiple generations of the agent trying to complete the same level
6. Stretch goal: Feature 3: User can switch between agents trained at different skill levels (X samples taken during the training) and run them live

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training (Offline)                       │
│              Python: Stable-Baselines3 DQN → SQLite             │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Web App (React + Three.js)                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ Game Logic  │ →  │ Game State  │ →  │ 3D Visualization    │  │
│  │ (Pure TS)   │    │             │    │ (React Three Fiber) │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         ↑                  ↑                                    │
│  ┌─────────────┐    ┌─────────────┐                             │
│  │ ONNX Model  │    │   Server    │                             │
│  │ (Inference) │    │ (Episodes)  │                             │
│  └─────────────┘    └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline (Python)

### Environment

Custom gridworld environment compatible with Gymnasium API:

```python
# Environment features
- Grid size: 8x8 to 12x12 (configurable)
- Elements: Wall, Floor, Door, Key, Goal, Lava
- Agent: Position + inventory (has_key boolean)
- Actions: [Up, Down, Left, Right, Pickup, Toggle] (6 discrete)
- Observation: Flattened grid state + agent position + inventory
- Reward: +1 goal, -1 lava, -0.01 per step (encourage efficiency)
```

### DQN Training

Using Stable-Baselines3:

```python
from stable_baselines3 import DQN

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./logs/"
)
```

### Checkpoint Strategy

Save checkpoints throughout training to visualize the full learning progression:

- **Total training**: 500k steps
- **Checkpoint interval**: Every 10k steps
- **Total checkpoints**: 50 checkpoints (010000, 020000, ... 500000)

The frontend can then:
1. Fetch the full list of available checkpoints
2. Let user select a sample size N (e.g., 5 or 10)
3. Compute evenly-spaced indices to sample from the training timeline
4. Display how the agent's behavior evolves across those checkpoints

For convenience, we still define semantic aliases:
- **Beginner**: ~050000 (10% through training)
- **Intermediate**: ~200000 (40% through training)
- **Expert**: 500000 (fully trained)

### Data Storage (SQLite)

Schema:

```sql
-- Training metadata
CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    name TEXT,              -- 'beginner', 'intermediate', 'expert'
    training_steps INTEGER,
    created_at TIMESTAMP
);

-- Episode recordings
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    level_seed INTEGER,     -- For reproducibility
    total_reward FLOAT,
    steps INTEGER,
    success BOOLEAN
);

-- Step-by-step data
CREATE TABLE steps (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER REFERENCES episodes(id),
    step_number INTEGER,
    state_json TEXT,        -- Grid state as JSON
    action INTEGER,
    reward FLOAT,
    q_values_json TEXT,     -- Q-value for each action
    done BOOLEAN
);

-- Model weights (for potential future use)
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    training_step INTEGER,
    weights_path TEXT       -- Path to .onnx file
);
```

### Episode Recording Strategy

Each checkpoint is recorded playing the same set of fixed levels. This enables two key visualizations:

1. **Cross-generation comparison**: How does an early checkpoint vs late checkpoint handle the same level?
2. **Learning progression**: Watch the same level being attempted at different points in training

**Recording schedule:**

- **10 preset levels** (fixed seeds) — hand-designed to test specific behaviors
- **50 checkpoints**: every 10k steps from 10k to 500k
- **5 runs per checkpoint per level** — captures behavioral variance (epsilon-greedy exploration means even trained agents vary)
- **Total**: 10 levels × 50 checkpoints × 5 runs = **2,500 recorded episodes**

The 5 runs per level reveal whether an agent *reliably* solves a level or just gets lucky. Late checkpoints should show consistent paths; early checkpoints may flail differently each time.

**Storage estimate:** ~50-100 steps per episode × 2,500 episodes = ~150k-250k step records. At ~500 bytes per step JSON, this is ~75-125 MB total. Acceptable for SQLite, can be compressed for transfer.

### Episode JSON Structure

```json
{
  "agentId": "expert",
  "trainingSteps": 500000,
  "levelId": "level_001",
  "levelSeed": 42,
  "runNumber": 3,
  "level": {
    "width": 8,
    "height": 8,
    "grid": [[0, 0, 1, ...], ...],  // 0=floor, 1=wall, 2=door, etc.
    "agentStart": [1, 1],
    "goalPosition": [6, 6],
    "keys": [[3, 3]],
    "doors": [[4, 4]]
  },
  "totalReward": 0.87,
  "success": true,
  "steps": [
    {
      "step": 0,
      "agentPosition": [1, 1],
      "inventory": { "hasKey": false },
      "action": 2,
      "actionName": "Right",
      "reward": -0.01,
      "qValues": {
        "Up": -0.23,
        "Down": -0.45,
        "Left": -0.67,
        "Right": 0.12,
        "Pickup": -0.89,
        "Toggle": -0.91
      },
      "done": false
    },
    // ...
  ]
}
```

---

## Backend API (Express + TypeScript)

A lightweight Express server serves episode data from SQLite, designed for CDN caching.

### Endpoints

```
GET /api/checkpoints
  → Returns list of all checkpoints with training step counts
  → Cache: long (changes only on redeploy)
  → Response: { checkpoints: [{ id: "step_010000", trainingSteps: 10000 }, ...] }

GET /api/levels
  → Returns list of level IDs and metadata
  → Cache: long
  → Response: { levels: [{ id: "level_001", name: "Simple Corridor", seed: 42 }, ...] }

GET /api/levels/:levelId
  → Returns level definition + all episodes for this level (with full step data)
  → Cache: immutable (level data never changes)
  → Response: {
      level: { width, height, grid, agentStart, ... },
      episodes: [{ checkpointId, runNumber, success, steps: [...] }, ...]
    }
```

### Caching Strategy

- **Deterministic URLs**: No query params, path segments only
- **Immutable data**: Set `Cache-Control: public, max-age=31536000, immutable` on level endpoints
- **CDN-friendly**: Any CDN (Cloudflare, Vercel Edge, etc.) will cache responses automatically

### Frontend Sampling

The frontend handles checkpoint sampling client-side:

```typescript
function sampleCheckpoints(all: Checkpoint[], n: number): Checkpoint[] {
  if (n >= all.length) return all;
  const step = (all.length - 1) / (n - 1);
  return Array.from({ length: n }, (_, i) => all[Math.round(i * step)]);
}

// User selects n=5 → get checkpoints at 10k, 132k, 255k, 377k, 500k
const sampled = sampleCheckpoints(checkpoints, 5);
```

---

## Web Application (React + TypeScript)

### Project Structure

```
/src
  types.ts      # All shared types (Level, Episode, GameState, etc.)
  play.ts       # Pure game logic: actions, state transitions, level utils
  ml.ts         # ONNX Runtime Web wrapper: ml.Agent, ml.load(), ml.getQValues()
  render.tsx    # React Three Fiber: render.Scene, render.Grid, render.Agent, etc.
  util.tsx      # React Context + hooks: util.AppProvider, util.useApp, util.usePlayback
  api.ts        # Server API client: api.fetchLevels(), api.fetchEpisodes()
  app.tsx       # Main app: UI components, controls, Q-value display, entry point
  server.ts     # Express API server
```

### Import Style

Use qualified imports for clarity and to allow short, expressive names:

```typescript
// app.tsx
import * as t from './types';
import * as play from './play';
import * as ml from './ml';
import * as render from './render';
import * as util from './util';
import * as api from './api';

// Usage examples:
const level: t.Level = await api.fetchLevel('level_001');
const state: t.GameState = play.createInitialState(level);
const qValues: t.QValues = await ml.getQValues(agent, state);
const nextState = play.step(state, t.Action.Right);
```

### Types (`types.ts`)

All shared types live in one file:

```typescript
// types.ts
export enum CellType { Floor = 0, Wall = 1, Door = 2, Key = 3, Goal = 4, Lava = 5 }
export enum Action { Up = 0, Down = 1, Left = 2, Right = 3, Pickup = 4, Toggle = 5 }

export interface Level {
  width: number;
  height: number;
  grid: CellType[][];
  agentStart: [number, number];
  goalPosition: [number, number];
  keys: [number, number][];
  doors: [number, number][];
}

export interface GameState {
  level: Level;
  agentPosition: [number, number];
  inventory: { hasKey: boolean };
  stepCount: number;
  totalReward: number;
  done: boolean;
}

export type QValues = Record<Action, number>;

export interface Step {
  step: number;
  agentPosition: [number, number];
  inventory: { hasKey: boolean };
  action: Action;
  reward: number;
  qValues: QValues;
  done: boolean;
}

export interface Episode {
  checkpointId: string;
  trainingSteps: number;
  levelId: string;
  runNumber: number;
  level: Level;
  totalReward: number;
  success: boolean;
  steps: Step[];
}
```

### Core Game Logic (`play.ts`)

Completely isolated from React/Three.js. Can be tested independently.

```typescript
// play.ts
import * as t from './types';

export function createInitialState(level: t.Level): t.GameState;
export function step(state: t.GameState, action: t.Action): t.GameState;
export function isValidMove(state: t.GameState, action: t.Action): boolean;
export function generateLevel(seed: number): t.Level;
```

### State Management (`util.tsx`)

UI state lives in React Context. Animation state lives in refs (mutated in useFrame, no re-renders).

```typescript
// util.tsx
import * as t from './types';

export interface AppState {
  level: t.Level | null;
  episode: t.Episode | null;
  currentStepIndex: number;
  isPlaying: boolean;
  playbackSpeed: number;  // 0.5x, 1x, 2x
  editMode: boolean;
}

export type AppAction =
  | { type: 'LOAD_LEVEL'; level: t.Level }
  | { type: 'LOAD_EPISODE'; episode: t.Episode }
  | { type: 'SEEK'; stepIndex: number }
  | { type: 'PLAY' }
  | { type: 'PAUSE' }
  | { type: 'STEP_FORWARD' }
  | { type: 'STEP_BACKWARD' }
  | { type: 'SET_SPEED'; speed: number }
  | { type: 'TOGGLE_EDIT_MODE' }
  | { type: 'EDIT_CELL'; x: number; y: number; cellType: t.CellType };

const AppContext = createContext<{
  state: AppState;
  dispatch: Dispatch<AppAction>;
} | null>(null);

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}

function reducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SEEK':
      return { ...state, currentStepIndex: action.stepIndex };
    case 'PLAY':
      return { ...state, isPlaying: true };
    // ...
  }
}
```

### Animation State (`render.tsx`)

For smooth 60fps animation, bypass React entirely with refs + useFrame:

```typescript
// render.tsx
import * as THREE from 'three';
import * as util from './util';

export function Agent() {
  const { state } = util.useApp();
  const meshRef = useRef<THREE.Mesh>(null);
  const targetPos = useRef<[number, number]>([0, 0]);

  // Update target when step changes (React re-render)
  useEffect(() => {
    if (state.episode && state.currentStepIndex >= 0) {
      const step = state.episode.steps[state.currentStepIndex];
      targetPos.current = step.agentPosition;
    }
  }, [state.episode, state.currentStepIndex]);

  // Interpolate every frame (no React re-render)
  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.position.x = THREE.MathUtils.lerp(
        meshRef.current.position.x,
        targetPos.current[0],
        delta * 10
      );
      meshRef.current.position.z = THREE.MathUtils.lerp(
        meshRef.current.position.z,
        targetPos.current[1],
        delta * 10
      );
    }
  });

  return (
    <mesh ref={meshRef}>
      <boxGeometry args={[0.5, 0.5, 0.5]} />
      <meshStandardMaterial color="#00ff88" />
    </mesh>
  );
}
```

### 3D Visualization (`render.tsx` continued)

```typescript
// render.tsx (continued)
import * as play from './play';

export function Scene() {
  return (
    <Canvas camera={{ position: [0, 10, 10], fov: 50 }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} />
      <Grid />
      <Walls />
      <Items />
      <Agent />
      <Goal />
      <OrbitControls />
    </Canvas>
  );
}

export function Walls() {
  const { state } = util.useApp();
  if (!state.level) return null;

  const walls = play.getWallPositions(state.level);

  return (
    <>
      {walls.map(([x, y], i) => (
        <mesh key={i} position={[x, 0.5, y]}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="#444" />
        </mesh>
      ))}
    </>
  );
}
```

### Live Inference (`ml.ts`)

Using ONNX Runtime Web for running the trained model in-browser:

```typescript
// ml.ts
import * as ort from 'onnxruntime-web';
import * as t from './types';

export class Agent {
  private session: ort.InferenceSession | null = null;

  async load(modelPath: string) {
    this.session = await ort.InferenceSession.create(modelPath);
  }

  async getQValues(state: t.GameState): Promise<t.QValues> {
    const input = stateToTensor(state);
    const results = await this.session!.run({ input });
    const data = results.output.data as Float32Array;

    return {
      [t.Action.Up]: data[0],
      [t.Action.Down]: data[1],
      [t.Action.Left]: data[2],
      [t.Action.Right]: data[3],
      [t.Action.Pickup]: data[4],
      [t.Action.Toggle]: data[5],
    };
  }

  async selectAction(state: t.GameState): Promise<t.Action> {
    const qValues = await this.getQValues(state);
    return Object.entries(qValues).reduce(
      (best, [action, value]) => value > qValues[best as t.Action] ? Number(action) as t.Action : best,
      t.Action.Up
    );
  }
}

function stateToTensor(state: t.GameState): ort.Tensor {
  // Flatten grid + position + inventory into float array
  // ...
}
```

### Q-Value Visualization (`app.tsx`)

Display Q-values as overlays or a side panel:

```typescript
// app.tsx
import * as t from './types';
import * as util from './util';
import * as ml from './ml';
import * as render from './render';

export function QValueDisplay() {
  const { state } = util.useApp();
  const liveQValues = util.useLiveInference(state);

  // Get Q-values from recorded episode or live inference
  const qValues = state.episode
    ? state.episode.steps[state.currentStepIndex].qValues
    : liveQValues;

  if (!qValues) return null;

  const maxQ = Math.max(...Object.values(qValues));

  return (
    <div className="q-value-panel">
      <h3>Q-Values (Expected Return)</h3>
      {Object.entries(qValues).map(([action, value]) => (
        <div key={action} className={value === maxQ ? 'best-action' : ''}>
          <span>{t.Action[Number(action)]}</span>
          <div className="bar" style={{ width: `${normalize(value) * 100}%` }} />
          <span>{value.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}
```

---

## User Interactions

### Playback Mode (Recorded Episodes)

1. Select agent (beginner/intermediate/expert)
2. Select episode from list
3. Watch playback with controls:
   - Play/Pause
   - Step forward/backward
   - Scrub timeline
   - Adjust speed (0.5x, 1x, 2x)
4. Q-values update at each step

### Live Mode (Edited Levels)

1. Toggle "Edit Mode"
2. Click cells to cycle through: Floor → Wall → Goal → Floor
3. Click "Run Agent" → agent uses ONNX model for inference
4. Watch agent attempt the modified level
5. Compare: does the expert generalize better than the beginner?

### Preset Levels

Curated levels demonstrating specific behaviors:

1. **Simple corridor**: Direct path to goal
2. **Dead end**: Agent must backtrack
3. **Key puzzle**: Must pickup key before opening door
4. **Lava maze**: Requires careful navigation
5. **Multi-room**: Several doors and keys
6. **Adversarial**: Designed to fool undertrained agents

---

## Dependencies

### Python (Training)

```
stable-baselines3>=2.0
gymnasium>=0.29
torch>=2.0
sqlite3 (stdlib)
onnx
onnxruntime
```

### Web App

```json
{
  "dependencies": {
    "react": "^18",
    "react-dom": "^18",
    "@react-three/fiber": "^8",
    "@react-three/drei": "^9",
    "three": "^0.160",
    "onnxruntime-web": "^1.17"
  },
  "devDependencies": {
    "typescript": "^5",
    "vite": "^5"
  }
}
```

---

## Project Structure

```
/boxworld
  /data                       # Shared data (training outputs, used by visualize)
    /checkpoints              # ONNX model files
      010000.onnx
      020000.onnx
      ...
      500000.onnx
    /levels                   # Preset level definitions
      level_001.json
      level_002.json
      ...
    db.sqlite                 # SQLite database with all episode data

  /training                   # Python training pipeline
    main.py                   # Entry point
    environment.py            # Custom Gymnasium environment
    train.py                  # DQN training script
    record.py                 # Record episodes to SQLite
    export.py                 # Export checkpoints to ONNX
    pyproject.toml
    .venv/

  /visualize                  # TypeScript web application
    /src
      types.ts                # All shared types (t.Level, t.Episode, etc.)
      play.ts                 # Pure game logic (play.step, play.createInitialState)
      ml.ts                   # ONNX inference (ml.Agent, ml.getQValues)
      render.tsx              # React Three Fiber (render.Scene, render.Agent)
      util.tsx                # Context + hooks (util.useApp, util.usePlayback)
      api.ts                  # API client (api.fetchLevels, api.fetchEpisodes)
      app.tsx                 # UI components, entry point
      server.ts               # Express API server
    /static                   # Static assets
    package.json

  /docs                       # Documentation
    *.md
```

### Data Flow

1. **Training** writes to `/data/db.sqlite` and `/data/checkpoints/`
2. **Visualize server** reads from `/data/db.sqlite`
3. **Visualize client** fetches from API, loads ONNX from `/data/checkpoints/`

---

## Verification

### Training Pipeline

1. Run `python train.py` → produces checkpoints
2. Run `python record.py` → records 50 episodes per agent
3. Run `python export.py` → generates JSON + ONNX files
4. Verify: JSON files are valid, ONNX models load in Python

### Web App

1. Run `npm run dev` → app loads without errors
2. Verify: Episode playback shows agent moving through level
3. Verify: Q-values display updates at each step
4. Verify: Level editing toggles cells correctly
5. Verify: "Run Agent" uses ONNX model and agent navigates
6. Verify: Different agents (beginner/expert) show different behaviors

### Integration

1. Train new agent → export → verify it appears in web app
2. Create adversarial level → verify expert handles it, beginner fails
3. Check ONNX inference matches Python model predictions (within floating-point tolerance)

---

## Future Extensions (Out of Scope)

- Post-processing shaders (CRT, bloom, color grading)
- Browser-based training via TensorFlow.js
- Multiplayer: compare your edits with others
- More environment types (MiniHack-style, Crafter-style)
- Attention visualization for transformer-based agents
- Export GIFs/videos of episodes

### Stretch Goal: Hidden Layer Visualization

The hidden layer activations form a learned representation space where states that require similar actions should cluster together. Potential visualizations:

1. **t-SNE/UMAP projection**: Project hidden activations to 2D, color by Q-value or success rate. Reveals clusters of "good situations" vs "doomed situations."

2. **Embedding interpolation**: Take two state embeddings, walk the line between them, decode to Q-values. Shows whether agent confidence changes smoothly or has sharp decision boundaries.

3. **Adversarial state detection**: Find small perturbations in embedding space that flip the agent's decision — reveals what the network is sensitive to.

4. **Prototype extraction**: Identify the "most key-like" or "most goal-adjacent" directions in embedding space.

This would require exporting hidden layer activations alongside Q-values during recording.

---

## Appendix A: DQN Explained

### What is DQN?

DQN (Deep Q-Network) is a reinforcement learning algorithm that learns to estimate "how good" each action is in any given state.

### Q-Values

A **Q-value** `Q(state, action)` answers: "If I'm in this state and take this action, what's my expected total future reward?"

For example:
- State: Agent at (2,3), has key, door ahead
- Q-values: `Up=0.12, Right=0.87, Left=-0.45, ...`
- Agent picks Right (highest Q-value)

### Why "Deep"?

Classic Q-learning stores Q-values in a table — one entry per (state, action) pair. This explodes combinatorially for large state spaces.

DQN replaces the table with a neural network:
- Input: state representation (flattened grid + position + inventory)
- Output: Q-value for each possible action
- The network learns to generalize across similar states

### Network Architecture

A simple MLP with ~11,000 parameters:

```
Input (103) → Linear(103×64) + ReLU → Linear(64×64) + ReLU → Linear(64×6) → Output (6 Q-values)

Layer 1: 103 × 64 + 64 bias = 6,656 parameters
Layer 2: 64 × 64 + 64 bias  = 4,160 parameters
Layer 3: 64 × 6 + 6 bias    = 390 parameters
Total: ~11,200 parameters (~44 KB)
```

The "64" is the **hidden dimension** (or hidden size/width) — the number of neurons in each hidden layer. This controls the capacity of the network's internal representation.

---

## Appendix B: The Training Loop

### Step-by-Step Process

```
1. OBSERVE current state S
   Encode: 100 grid values + 2 position + 1 inventory = 103 floats

2. FORWARD PASS through network
   Input: 103 floats → Output: 6 Q-values (one per action)

3. SELECT action (ε-greedy)
   - 10% of the time: random action (exploration)
   - 90% of the time: action with highest Q-value (exploitation)

4. EXECUTE action in environment
   Environment returns: reward, new state S', done flag

5. STORE experience in replay buffer
   Save tuple: (S, action, reward, S', done)
   Buffer holds ~100,000 recent experiences

6. SAMPLE batch from replay buffer
   Randomly pick 32 past experiences (not just the current one)

7. COMPUTE targets using Bellman equation
   If done: target = reward
   Else:    target = reward + γ × max(Q(S', all_actions))

8. COMPUTE loss
   Loss = (Q(S, action) - target)²
   Average over the batch

9. GRADIENT DESCENT
   Update weights to minimize loss

10. REPEAT
```

### Where Does the Reward Come From?

You define it in the environment. It's the only way to communicate "what I want" to the agent:

```python
if new_pos == goal:
    reward = +1.0
elif grid[new_pos] == Lava:
    reward = -1.0
else:
    reward = -0.01  # Step penalty encourages efficiency
```

The agent never sees this code — it only experiences the consequences.

### Why Use a Replay Buffer?

Without it, consecutive training samples are highly correlated. If the agent walks down a corridor for 100 steps, you'd do 100 gradient updates all saying "going right is good in corridors." The network overfits and forgets other situations.

The replay buffer samples randomly from past experiences, so each batch contains diverse situations (some corridor steps, some lava deaths, some key pickups). This prevents catastrophic forgetting and stabilizes learning.

### Can Samples Repeat?

Yes, and that's fine. Important experiences (rare successes, deaths) may need multiple passes before the network internalizes them. **Prioritized Experience Replay** is a variant that intentionally samples "surprising" experiences more often.

### What Does γ (Gamma) Do?

The discount factor γ controls how much the agent values future vs immediate rewards:

- γ = 0.99: Cares about rewards ~100 steps ahead
- γ = 0.9: Cares about rewards ~10 steps ahead
- γ = 0: Completely myopic, only sees immediate reward

The `γ × max(Q(S', ...))` term creates a chain of influence backward through time. Q-values "propagate" from terminal states (goal, lava) back to earlier states during training.

### Why "No Future" for Terminal States?

When `done=True`, there are no more states — the episode is over. The Bellman equation simplifies:

```
Q(terminal_state, action) = reward    # No future term
Q(normal_state, action) = reward + γ × max(Q(next_state, ...))
```

This isn't a penalty for finishing. The reward already reflects the outcome (+1 for goal, -1 for lava). There's simply nothing more to account for.

---

## Appendix C: Training Challenges

### Early-Episode Bias

Early in training, the replay buffer is dominated by early-episode steps because the agent keeps dying or timing out before reaching later states.

Mitigations:
1. **Large buffer** (100k experiences) — successful late-episode experiences persist and get resampled
2. **Bootstrapping** — even without reaching the goal, states *near* the goal learn positive values, which propagate backward
3. **Episode length limits** — cap at ~200 steps so even lost agents generate diverse experiences
4. **Prioritized replay** — upweight rare/surprising experiences

### Curriculum Learning

If the environment is too hard, the agent never accidentally succeeds and learning stalls. Solutions:

- **Reverse curriculum**: Start episodes closer to the goal, gradually move start position back
- **Demonstration states**: Record a successful trajectory, start new episodes from random points along it
- **Goal-proximal starts**: Randomly start 1-5 steps from goal, increase distance as success rate improves

### Reward Shaping Pitfalls

Bad reward design leads to bad behavior:
- No step penalty? Agent wanders forever
- Step penalty too high? Agent prefers quick death (-1) over long journeys (-0.01 × 100 = -1)
- Reward for getting closer? Agent oscillates to farm "getting closer" rewards
