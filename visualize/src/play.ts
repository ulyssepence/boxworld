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
    state = (state * 1664525 + 1013904223) & 0xffffffff
    return (state >>> 0) / 0xffffffff
  }
}

/** BFS from start to goal on the grid. Returns distance or -1 if unreachable. */
export function bfs(
  grid: t.CellType[][],
  start: [number, number],
  goal: [number, number],
  hasKey: boolean,
): number {
  const h = grid.length
  const w = grid[0].length
  const key = (x: number, y: number) => y * w + x
  const visited = new Set<number>()
  const queue: [number, number, number][] = [[start[0], start[1], 0]]
  visited.add(key(start[0], start[1]))

  while (queue.length > 0) {
    const [x, y, dist] = queue.shift()!
    if (x === goal[0] && y === goal[1]) return dist
    for (const [dx, dy] of [
      [0, -1],
      [0, 1],
      [-1, 0],
      [1, 0],
    ] as const) {
      const nx = x + dx
      const ny = y + dy
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
      if (visited.has(key(nx, ny))) continue
      const cell = grid[ny][nx]
      if (cell === t.CellType.Wall) continue
      if (cell === t.CellType.Lava) continue
      if (cell === t.CellType.Door && !hasKey) continue
      visited.add(key(nx, ny))
      queue.push([nx, ny, dist + 1])
    }
  }
  return -1
}

interface Rect {
  x: number
  y: number
  w: number
  h: number
}

/** BSP split a rectangle into 2-4 rooms */
function bspSplit(rect: Rect, rng: () => number, depth: number): Rect[] {
  const minSize = 3
  if (depth <= 0 || (rect.w < minSize * 2 + 1 && rect.h < minSize * 2 + 1)) return [rect]

  const canSplitH = rect.w >= minSize * 2 + 1
  const canSplitV = rect.h >= minSize * 2 + 1
  if (!canSplitH && !canSplitV) return [rect]

  const splitH = canSplitH && (!canSplitV || rng() < 0.5)

  if (splitH) {
    const splitMin = rect.x + minSize
    const splitMax = rect.x + rect.w - minSize
    const splitAt = splitMin + Math.floor(rng() * (splitMax - splitMin))
    const left: Rect = { x: rect.x, y: rect.y, w: splitAt - rect.x, h: rect.h }
    const right: Rect = { x: splitAt + 1, y: rect.y, w: rect.x + rect.w - splitAt - 1, h: rect.h }
    return [...bspSplit(left, rng, depth - 1), ...bspSplit(right, rng, depth - 1)]
  } else {
    const splitMin = rect.y + minSize
    const splitMax = rect.y + rect.h - minSize
    const splitAt = splitMin + Math.floor(rng() * (splitMax - splitMin))
    const top: Rect = { x: rect.x, y: rect.y, w: rect.w, h: splitAt - rect.y }
    const bottom: Rect = { x: rect.x, y: splitAt + 1, w: rect.w, h: rect.y + rect.h - splitAt - 1 }
    return [...bspSplit(top, rng, depth - 1), ...bspSplit(bottom, rng, depth - 1)]
  }
}

/** Carve a horizontal or vertical corridor between two points */
function carveCorridor(grid: t.CellType[][], x1: number, y1: number, x2: number, y2: number) {
  let x = x1
  let y = y1
  // Go horizontal first, then vertical
  while (x !== x2) {
    if (y >= 0 && y < grid.length && x >= 0 && x < grid[0].length) {
      if (grid[y][x] === t.CellType.Wall) grid[y][x] = t.CellType.Floor
    }
    x += x < x2 ? 1 : -1
  }
  while (y !== y2) {
    if (y >= 0 && y < grid.length && x >= 0 && x < grid[0].length) {
      if (grid[y][x] === t.CellType.Wall) grid[y][x] = t.CellType.Floor
    }
    y += y < y2 ? 1 : -1
  }
  if (y >= 0 && y < grid.length && x >= 0 && x < grid[0].length) {
    if (grid[y][x] === t.CellType.Wall) grid[y][x] = t.CellType.Floor
  }
}

/** Get all floor cells in a grid */
function getFloorCells(grid: t.CellType[][]): [number, number][] {
  const cells: [number, number][] = []
  for (let y = 1; y < grid.length - 1; y++) {
    for (let x = 1; x < grid[0].length - 1; x++) {
      if (grid[y][x] === t.CellType.Floor) cells.push([x, y])
    }
  }
  return cells
}

/** Check if a cell is a corridor chokepoint (exactly 2 floor neighbors on opposite sides) */
function isChokepoint(grid: t.CellType[][], x: number, y: number): boolean {
  if (grid[y][x] !== t.CellType.Floor) return false
  const h = grid.length
  const w = grid[0].length
  const passable = (cx: number, cy: number) =>
    cx >= 0 && cx < w && cy >= 0 && cy < h && grid[cy][cx] !== t.CellType.Wall
  const horiz =
    passable(x - 1, y) && passable(x + 1, y) && !passable(x, y - 1) && !passable(x, y + 1)
  const vert =
    !passable(x - 1, y) && !passable(x + 1, y) && passable(x, y - 1) && passable(x, y + 1)
  return horiz || vert
}

/** Procedural level generation with BSP rooms + BFS solvability check */
export function generateLevel(seed: number): t.Level {
  for (let attempt = 0; attempt < 20; attempt++) {
    const level = tryGenerateLevel(seed + attempt)
    if (level) return level
  }
  // Fallback: simple open room
  return tryGenerateLevel(seed, true)!
}

function tryGenerateLevel(seed: number, fallback = false): t.Level | null {
  const rng = createRng(seed)
  // Warm up RNG to decorrelate sequential seeds
  rng()
  rng()
  rng()
  const width = 10
  const height = 10

  // Start with all floors + border walls
  const grid: t.CellType[][] = Array.from({ length: height }, () =>
    Array.from({ length: width }, () => t.CellType.Floor),
  )
  for (let x = 0; x < width; x++) {
    grid[0][x] = t.CellType.Wall
    grid[height - 1][x] = t.CellType.Wall
  }
  for (let y = 0; y < height; y++) {
    grid[y][0] = t.CellType.Wall
    grid[y][width - 1] = t.CellType.Wall
  }

  if (fallback) {
    grid[1][1] = t.CellType.Goal
    return {
      id: `generated_${seed}`,
      name: `Generated (seed ${seed})`,
      width,
      height,
      grid,
      agentStart: [width - 2, height - 2],
      seed,
    }
  }

  const useRooms = rng() < 0.35

  if (useRooms) {
    // BSP room-corridor layout: start from all walls, carve rooms
    for (let y = 1; y < height - 1; y++)
      for (let x = 1; x < width - 1; x++) grid[y][x] = t.CellType.Wall

    const interior: Rect = { x: 1, y: 1, w: width - 2, h: height - 2 }
    const rooms = bspSplit(interior, rng, 2)

    for (const room of rooms) {
      for (let y = room.y; y < room.y + room.h; y++)
        for (let x = room.x; x < room.x + room.w; x++)
          if (y > 0 && y < height - 1 && x > 0 && x < width - 1) grid[y][x] = t.CellType.Floor
    }

    for (let i = 0; i < rooms.length - 1; i++) {
      const a = rooms[i]
      const b = rooms[i + 1]
      carveCorridor(
        grid,
        Math.floor(a.x + a.w / 2),
        Math.floor(a.y + a.h / 2),
        Math.floor(b.x + b.w / 2),
        Math.floor(b.y + b.h / 2),
      )
    }
  } else {
    // Open layout with scattered walls (closer to training levels)
    const wallCount = 4 + Math.floor(rng() * 10)
    for (let i = 0; i < wallCount; i++) {
      const wx = 1 + Math.floor(rng() * (width - 2))
      const wy = 1 + Math.floor(rng() * (height - 2))
      grid[wy][wx] = t.CellType.Wall
      // 40% chance to extend into an L or line of 2-3 walls
      if (rng() < 0.4) {
        const dx = rng() < 0.5 ? 1 : 0
        const dy = dx === 0 ? 1 : 0
        for (let j = 1; j <= 1 + Math.floor(rng() * 2); j++) {
          const ex = wx + dx * j
          const ey = wy + dy * j
          if (ex > 0 && ex < width - 1 && ey > 0 && ey < height - 1) grid[ey][ex] = t.CellType.Wall
        }
      }
    }
  }

  const floors = getFloorCells(grid)
  if (floors.length < 8) return null

  // Place agent and goal in different rooms, maximizing distance
  let bestDist = -1
  let agentPos: [number, number] = floors[0]
  let goalPos: [number, number] = floors[floors.length - 1]

  // Sample random pairs and pick the one with max BFS distance
  const pairTries = Math.min(20, floors.length * 2)
  for (let i = 0; i < pairTries; i++) {
    const a = floors[Math.floor(rng() * floors.length)]
    const b = floors[Math.floor(rng() * floors.length)]
    if (a[0] === b[0] && a[1] === b[1]) continue
    const d = bfs(grid, a, b, true)
    if (d > bestDist) {
      bestDist = d
      agentPos = a
      goalPos = b
    }
  }

  grid[goalPos[1]][goalPos[0]] = t.CellType.Goal

  // Optionally place a door on a chokepoint (50% chance)
  let hasDoor = false
  if (rng() < 0.5) {
    const chokepoints = floors.filter(
      ([x, y]) =>
        isChokepoint(grid, x, y) &&
        !(x === agentPos[0] && y === agentPos[1]) &&
        !(x === goalPos[0] && y === goalPos[1]),
    )
    if (chokepoints.length > 0) {
      const doorCell = chokepoints[Math.floor(rng() * chokepoints.length)]
      grid[doorCell[1]][doorCell[0]] = t.CellType.Door

      // Place key in agent's reachable area (without key, i.e. can't pass door)
      const reachable = getFloorCells(grid).filter(([x, y]) => {
        if (x === agentPos[0] && y === agentPos[1]) return false
        if (x === goalPos[0] && y === goalPos[1]) return false
        return bfs(grid, agentPos, [x, y], false) >= 0
      })

      if (reachable.length > 0) {
        const keyCell = reachable[Math.floor(rng() * reachable.length)]
        grid[keyCell[1]][keyCell[0]] = t.CellType.Key
        hasDoor = true
      } else {
        // Can't place key — remove the door
        grid[doorCell[1]][doorCell[0]] = t.CellType.Floor
      }
    }
  }

  // Optionally place 2-5 lava cells (60% chance)
  if (rng() < 0.6) {
    const lavaCount = 2 + Math.floor(rng() * 4)
    let placed = 0
    for (let i = 0; i < lavaCount * 5 && placed < lavaCount; i++) {
      const candidates = getFloorCells(grid).filter(
        ([x, y]) =>
          !(x === agentPos[0] && y === agentPos[1]) && !(x === goalPos[0] && y === goalPos[1]),
      )
      if (candidates.length === 0) break
      const cell = candidates[Math.floor(rng() * candidates.length)]
      // Tentatively place lava, then check solvability
      grid[cell[1]][cell[0]] = t.CellType.Lava
      const solvable = hasDoor
        ? bfs(grid, agentPos, goalPos, true) >= 0
        : bfs(grid, agentPos, goalPos, false) >= 0
      if (!solvable) {
        grid[cell[1]][cell[0]] = t.CellType.Floor
      } else {
        placed++
      }
    }
  }

  // Final solvability check
  if (hasDoor) {
    let keyCell: [number, number] | null = null
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) if (grid[y][x] === t.CellType.Key) keyCell = [x, y]
    if (!keyCell || bfs(grid, agentPos, keyCell, false) < 0) return null
    if (bfs(grid, agentPos, goalPos, true) < 0) return null
  } else {
    if (bfs(grid, agentPos, goalPos, false) < 0) return null
  }

  return {
    id: `generated_${seed}`,
    name: `Generated (seed ${seed})`,
    width,
    height,
    grid,
    agentStart: agentPos,
    seed,
  }
}
