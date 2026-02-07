import * as t from './types'

/** Returns a fresh GameState with agent at agentStart, done=false, reward=0, steps=0 */
export function createInitialState(level: t.Level): t.GameState {
  return {
    level,
    agentPosition: [level.agentStart[0], level.agentStart[1]],
    inventory: { hasKey: false },
    done: false,
    reward: 0,
    steps: 0,
  }
}

/** Deep-clone a 2D grid so mutations don't affect the original */
function cloneGrid(grid: t.CellType[][]): t.CellType[][] {
  return grid.map((row) => [...row])
}

/** Get the [dx, dy] offset for a movement action, or null for non-movement actions */
function actionDelta(action: t.Action): [number, number] | null {
  switch (action) {
    case t.Action.Up:
      return [0, -1]
    case t.Action.Down:
      return [0, 1]
    case t.Action.Left:
      return [-1, 0]
    case t.Action.Right:
      return [1, 0]
    default:
      return null
  }
}

/**
 * Returns a NEW GameState (immutable). Handles:
 * - Movement (Up/Down/Left/Right): move if target is Floor/Goal/Lava/Key, blocked by Wall/closed Door
 * - Pickup: if on Key tile and action is Pickup, set hasKey=true, remove key from grid
 * - Toggle: if adjacent to Door and hasKey, open door (change to Floor)
 * - Goal: stepping onto Goal sets done=true, reward=+1
 * - Lava: stepping onto Lava sets done=true, reward=-1
 * - Each step: reward -= 0.01, steps += 1
 */
export function step(state: t.GameState, action: t.Action): t.GameState {
  if (state.done) return state

  const [ax, ay] = state.agentPosition
  const grid = cloneGrid(state.level.grid)
  let newX = ax
  let newY = ay
  let hasKey = state.inventory.hasKey
  let reward = state.reward - 0.01
  let done = state.done

  const delta = actionDelta(action)

  if (delta !== null) {
    // Movement action
    const targetX = ax + delta[0]
    const targetY = ay + delta[1]

    if (
      targetX >= 0 &&
      targetX < state.level.width &&
      targetY >= 0 &&
      targetY < state.level.height
    ) {
      const targetCell = grid[targetY][targetX]

      if (
        targetCell === t.CellType.Floor ||
        targetCell === t.CellType.Goal ||
        targetCell === t.CellType.Lava ||
        targetCell === t.CellType.Key
      ) {
        newX = targetX
        newY = targetY

        if (targetCell === t.CellType.Goal) {
          done = true
          reward += 1
        } else if (targetCell === t.CellType.Lava) {
          done = true
          reward -= 1
        }
      }
      // Wall and Door block movement
    }
  } else if (action === t.Action.Pickup) {
    // Pick up key if standing on one
    if (grid[ay][ax] === t.CellType.Key) {
      hasKey = true
      grid[ay][ax] = t.CellType.Floor
    }
  } else if (action === t.Action.Toggle) {
    // Toggle a door adjacent to the agent (check all four directions)
    if (hasKey) {
      const directions: [number, number][] = [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
      ]
      for (const [dx, dy] of directions) {
        const adjX = ax + dx
        const adjY = ay + dy
        if (
          adjX >= 0 &&
          adjX < state.level.width &&
          adjY >= 0 &&
          adjY < state.level.height &&
          grid[adjY][adjX] === t.CellType.Door
        ) {
          grid[adjY][adjX] = t.CellType.Floor
          hasKey = false
          break
        }
      }
    }
  }

  return {
    level: { ...state.level, grid },
    agentPosition: [newX, newY],
    inventory: { hasKey },
    done,
    reward,
    steps: state.steps + 1,
  }
}

/** Returns true if the action would change the state */
export function isValidMove(state: t.GameState, action: t.Action): boolean {
  if (state.done) return false

  const next = step(state, action)

  // Compare agent position, inventory, grid, and done status
  if (next.agentPosition[0] !== state.agentPosition[0]) return true
  if (next.agentPosition[1] !== state.agentPosition[1]) return true
  if (next.inventory.hasKey !== state.inventory.hasKey) return true
  if (next.done !== state.done) return true

  // Check if any grid cell changed (e.g., door toggled)
  for (let y = 0; y < state.level.height; y++) {
    for (let x = 0; x < state.level.width; x++) {
      if (next.level.grid[y][x] !== state.level.grid[y][x]) return true
    }
  }

  return false
}

/** Returns array of [x, y] for all Wall cells */
export function getWallPositions(level: t.Level): [number, number][] {
  const walls: [number, number][] = []
  for (let y = 0; y < level.height; y++) {
    for (let x = 0; x < level.width; x++) {
      if (level.grid[y][x] === t.CellType.Wall) {
        walls.push([x, y])
      }
    }
  }
  return walls
}

/** Parses a level from the JSON file format */
export function parseLevelJson(json: any): t.Level {
  return {
    id: json.id,
    name: json.name,
    width: json.width,
    height: json.height,
    grid: json.grid as t.CellType[][],
    agentStart: json.agentStart as [number, number],
    seed: json.seed,
  }
}

/** Simple seeded PRNG using a linear congruential generator */
function createRng(seed: number): () => number {
  let state = seed
  return () => {
    // LCG parameters from Numerical Recipes
    state = (state * 1664525 + 1013904223) & 0xffffffff
    return (state >>> 0) / 0xffffffff
  }
}

/** Procedural level generation with seeded PRNG */
export function generateLevel(seed: number): t.Level {
  const rng = createRng(seed)
  const width = 10
  const height = 10

  // Start with all floors
  const grid: t.CellType[][] = Array.from({ length: height }, () =>
    Array.from({ length: width }, () => t.CellType.Floor),
  )

  // Surround with walls
  for (let x = 0; x < width; x++) {
    grid[0][x] = t.CellType.Wall
    grid[height - 1][x] = t.CellType.Wall
  }
  for (let y = 0; y < height; y++) {
    grid[y][0] = t.CellType.Wall
    grid[y][width - 1] = t.CellType.Wall
  }

  // Add some random internal walls
  const wallCount = Math.floor(rng() * 8) + 4
  for (let i = 0; i < wallCount; i++) {
    const wx = Math.floor(rng() * (width - 2)) + 1
    const wy = Math.floor(rng() * (height - 2)) + 1
    grid[wy][wx] = t.CellType.Wall
  }

  // Place a door
  const doorX = Math.floor(rng() * (width - 2)) + 1
  const doorY = Math.floor(rng() * (height - 2)) + 1
  grid[doorY][doorX] = t.CellType.Door

  // Place a key (ensure it's on a floor tile)
  let keyX: number, keyY: number
  do {
    keyX = Math.floor(rng() * (width - 2)) + 1
    keyY = Math.floor(rng() * (height - 2)) + 1
  } while (grid[keyY][keyX] !== t.CellType.Floor)
  grid[keyY][keyX] = t.CellType.Key

  // Place goal (ensure it's on a floor tile)
  let goalX: number, goalY: number
  do {
    goalX = Math.floor(rng() * (width - 2)) + 1
    goalY = Math.floor(rng() * (height - 2)) + 1
  } while (grid[goalY][goalX] !== t.CellType.Floor)
  grid[goalY][goalX] = t.CellType.Goal

  // Place agent start (ensure it's on a floor tile, not same as goal/key)
  let startX: number, startY: number
  do {
    startX = Math.floor(rng() * (width - 2)) + 1
    startY = Math.floor(rng() * (height - 2)) + 1
  } while (grid[startY][startX] !== t.CellType.Floor)

  // Optionally add lava
  if (rng() > 0.5) {
    let lavaX: number, lavaY: number
    do {
      lavaX = Math.floor(rng() * (width - 2)) + 1
      lavaY = Math.floor(rng() * (height - 2)) + 1
    } while (grid[lavaY][lavaX] !== t.CellType.Floor)
    grid[lavaY][lavaX] = t.CellType.Lava
  }

  return {
    id: `generated_${seed}`,
    name: `Generated Level (seed ${seed})`,
    width,
    height,
    grid,
    agentStart: [startX, startY],
    seed,
  }
}
