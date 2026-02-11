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

// --- Helper functions for procedural generation ---

/** BFS distance with doors impassable, lava passable. Returns distance or null. */
function bfsDistance(
  grid: t.CellType[][],
  sx: number,
  sy: number,
  tx: number,
  ty: number,
): number | null {
  const h = grid.length
  const w = grid[0].length
  if (sx === tx && sy === ty) return 0
  const key = (x: number, y: number) => y * w + x
  const visited = new Set<number>([key(sx, sy)])
  const queue: [number, number, number][] = [[sx, sy, 0]]
  while (queue.length > 0) {
    const [cx, cy, dist] = queue.shift()!
    for (const [dx, dy] of [
      [0, -1],
      [0, 1],
      [-1, 0],
      [1, 0],
    ] as const) {
      const nx = cx + dx
      const ny = cy + dy
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
      const k = key(nx, ny)
      if (visited.has(k)) continue
      if (nx === tx && ny === ty) return dist + 1
      const cell = grid[ny][nx]
      if (cell === t.CellType.Wall || cell === t.CellType.Door) continue
      visited.add(k)
      queue.push([nx, ny, dist + 1])
    }
  }
  return null
}

/** BFS distance avoiding walls, doors, AND lava. Target cell always reachable. */
function bfsDistanceSafe(
  grid: t.CellType[][],
  sx: number,
  sy: number,
  tx: number,
  ty: number,
): number | null {
  const h = grid.length
  const w = grid[0].length
  if (sx === tx && sy === ty) return 0
  const key = (x: number, y: number) => y * w + x
  const visited = new Set<number>([key(sx, sy)])
  const queue: [number, number, number][] = [[sx, sy, 0]]
  while (queue.length > 0) {
    const [cx, cy, dist] = queue.shift()!
    for (const [dx, dy] of [
      [0, -1],
      [0, 1],
      [-1, 0],
      [1, 0],
    ] as const) {
      const nx = cx + dx
      const ny = cy + dy
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
      const k = key(nx, ny)
      if (visited.has(k)) continue
      if (nx === tx && ny === ty) return dist + 1
      const cell = grid[ny][nx]
      if (cell === t.CellType.Wall || cell === t.CellType.Door || cell === t.CellType.Lava) continue
      visited.add(k)
      queue.push([nx, ny, dist + 1])
    }
  }
  return null
}

/** BFS all cells reachable from (sx,sy), not crossing walls/doors. */
function bfsReachable(grid: t.CellType[][], sx: number, sy: number): [number, number][] {
  const h = grid.length
  const w = grid[0].length
  const key = (x: number, y: number) => y * w + x
  const visited = new Set<number>([key(sx, sy)])
  const queue: [number, number][] = [[sx, sy]]
  const result: [number, number][] = [[sx, sy]]
  while (queue.length > 0) {
    const [cx, cy] = queue.shift()!
    for (const [dx, dy] of [
      [0, -1],
      [0, 1],
      [-1, 0],
      [1, 0],
    ] as const) {
      const nx = cx + dx
      const ny = cy + dy
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
      const k = key(nx, ny)
      if (visited.has(k)) continue
      const cell = grid[ny][nx]
      if (cell === t.CellType.Wall || cell === t.CellType.Door) continue
      visited.add(k)
      queue.push([nx, ny])
      result.push([nx, ny])
    }
  }
  return result
}

/** Place agent and goal on floor cells with minimum BFS distance. */
function placeAgentAndGoal(
  grid: t.CellType[][],
  rng: () => number,
  minDist: number,
): { ax: number; ay: number; gx: number; gy: number } | null {
  const floors = getFloorCells(grid)
  if (floors.length < 2) return null
  for (let attempt = 0; attempt < 50; attempt++) {
    const a = floors[Math.floor(rng() * floors.length)]
    const b = floors[Math.floor(rng() * floors.length)]
    if (a[0] === b[0] && a[1] === b[1]) continue
    const d = bfsDistance(grid, a[0], a[1], b[0], b[1])
    if (d !== null && d >= minDist) return { ax: a[0], ay: a[1], gx: b[0], gy: b[1] }
  }
  // Fallback: pick any two distinct floor cells
  const a = floors[Math.floor(rng() * floors.length)]
  let b = floors[Math.floor(rng() * floors.length)]
  if (a[0] === b[0] && a[1] === b[1])
    b = floors[(Math.floor(rng() * (floors.length - 1)) + 1) % floors.length]
  return { ax: a[0], ay: a[1], gx: b[0], gy: b[1] }
}

/** Add a door on the BFS path agent→goal, plus a key in agent-reachable area. */
function addDoorAndKey(
  grid: t.CellType[][],
  rng: () => number,
  ax: number,
  ay: number,
  gx: number,
  gy: number,
): void {
  const h = grid.length
  const w = grid[0].length
  // BFS from agent to goal to find path
  const key = (x: number, y: number) => y * w + x
  const parent = new Map<number, number>()
  const start = key(ax, ay)
  const goal = key(gx, gy)
  parent.set(start, -1)
  const queue: [number, number][] = [[ax, ay]]
  let found = false
  while (queue.length > 0) {
    const [cx, cy] = queue.shift()!
    if (cx === gx && cy === gy) {
      found = true
      break
    }
    for (const [dx, dy] of [
      [0, -1],
      [0, 1],
      [-1, 0],
      [1, 0],
    ] as const) {
      const nx = cx + dx
      const ny = cy + dy
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue
      const k = key(nx, ny)
      if (parent.has(k)) continue
      const cell = grid[ny][nx]
      if (cell === t.CellType.Wall) continue
      parent.set(k, key(cx, cy))
      queue.push([nx, ny])
    }
  }
  if (!found) return

  // Reconstruct path
  const path: [number, number][] = []
  let node = goal
  while (node !== -1) {
    path.push([node % w, Math.floor(node / w)])
    node = parent.get(node)!
  }
  path.reverse()

  // Pick a floor cell on path (not start/end) as door
  const candidates = path.slice(1, -1).filter(([x, y]) => grid[y][x] === t.CellType.Floor)
  if (candidates.length === 0) return

  const doorPos = candidates[Math.floor(rng() * candidates.length)]
  grid[doorPos[1]][doorPos[0]] = t.CellType.Door

  // Place key in agent-reachable area (BFS without crossing door)
  const reachable = bfsReachable(grid, ax, ay)
  const keyCandidates = reachable.filter(
    ([x, y]) =>
      grid[y][x] === t.CellType.Floor &&
      !(x === ax && y === ay) &&
      !(x === gx && y === gy) &&
      !(x === doorPos[0] && y === doorPos[1]),
  )
  if (keyCandidates.length === 0) {
    grid[doorPos[1]][doorPos[0]] = t.CellType.Floor
    return
  }
  const keyCell = keyCandidates[Math.floor(rng() * keyCandidates.length)]
  grid[keyCell[1]][keyCell[0]] = t.CellType.Key
}

/** Add 1-4 lava cells, ensuring solvability via bfsDistanceSafe. */
function addLava(
  grid: t.CellType[][],
  rng: () => number,
  ax: number,
  ay: number,
  gx: number,
  gy: number,
): void {
  const h = grid.length
  const w = grid[0].length
  const floors = getFloorCells(grid).filter(
    ([x, y]) => !(x === ax && y === ay) && !(x === gx && y === gy),
  )
  const numLava = Math.min(Math.floor(floors.length / 5), 1 + Math.floor(rng() * 4))
  if (numLava === 0) return
  const count = Math.min(numLava, floors.length)
  // Shuffle-pick without replacement
  const indices = Array.from({ length: floors.length }, (_, i) => i)
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[indices[i], indices[j]] = [indices[j], indices[i]]
  }
  for (let i = 0; i < count; i++) {
    const [x, y] = floors[indices[i]]
    grid[y][x] = t.CellType.Lava
    if (bfsDistanceSafe(grid, ax, ay, gx, gy) === null) {
      grid[y][x] = t.CellType.Floor
    }
  }
}

// --- Generators ---

function initBorder(width: number, height: number): t.CellType[][] {
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
  return grid
}

/** Open room with 0-6 scattered wall cells. */
function genOpenRoom(grid: t.CellType[][], rng: () => number): [number, number] | null {
  const h = grid.length
  const w = grid[0].length
  const numWalls = Math.floor(rng() * 7) // 0-6
  const interior: [number, number][] = []
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) interior.push([x, y])
  // Shuffle and pick
  for (let i = interior.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[interior[i], interior[j]] = [interior[j], interior[i]]
  }
  for (let i = 0; i < Math.min(numWalls, interior.length); i++) {
    grid[interior[i][1]][interior[i][0]] = t.CellType.Wall
  }

  const pos = placeAgentAndGoal(grid, rng, 7)
  if (!pos) return null
  const { ax, ay, gx, gy } = pos
  grid[gy][gx] = t.CellType.Goal

  // 0-3 doors (60% chance of any)
  const roll = rng()
  let numDoors: number
  if (roll < 0.4) numDoors = 0
  else if (roll < 0.75) numDoors = 1
  else if (roll < 0.9) numDoors = 2
  else numDoors = 3
  for (let i = 0; i < numDoors; i++) addDoorAndKey(grid, rng, ax, ay, gx, gy)

  // 50% lava
  if (rng() < 0.5) addLava(grid, rng, ax, ay, gx, gy)

  return [ax, ay]
}

/** Rooms divided by wall partitions with door gaps. */
function genRoomPartition(grid: t.CellType[][], rng: () => number): [number, number] | null {
  const h = grid.length
  const w = grid[0].length

  const roll = rng()
  const numPartitions = roll < 0.4 ? 1 : roll < 0.8 ? 2 : 3
  const doorPositions: [number, number][] = []
  const usedPositions = new Set<number>()

  for (let p = 0; p < numPartitions; p++) {
    const horizontal = rng() < 0.5
    if (horizontal) {
      const valid = []
      for (let y = 2; y < h - 2; y++) if (!usedPositions.has(y)) valid.push(y)
      if (valid.length === 0) continue
      const pos = valid[Math.floor(rng() * valid.length)]
      usedPositions.add(pos)
      for (let x = 1; x < w - 1; x++) grid[pos][x] = t.CellType.Wall
      const gapX = 1 + Math.floor(rng() * (w - 2))
      grid[pos][gapX] = t.CellType.Door
      doorPositions.push([gapX, pos])
    } else {
      const valid = []
      for (let x = 2; x < w - 2; x++) if (!usedPositions.has(x)) valid.push(x)
      if (valid.length === 0) continue
      const pos = valid[Math.floor(rng() * valid.length)]
      usedPositions.add(pos)
      for (let y = 1; y < h - 1; y++) grid[y][pos] = t.CellType.Wall
      const gapY = 1 + Math.floor(rng() * (h - 2))
      grid[gapY][pos] = t.CellType.Door
      doorPositions.push([pos, gapY])
    }
  }

  const pos = placeAgentAndGoal(grid, rng, 5)
  if (!pos) return null
  const { ax, ay, gx, gy } = pos
  grid[gy][gx] = t.CellType.Goal

  // Place keys for surviving doors
  for (const [dx, dy] of doorPositions) {
    if (grid[dy][dx] !== t.CellType.Door) continue
    const reachable = bfsReachable(grid, ax, ay)
    const keyCandidates = reachable.filter(
      ([x, y]) =>
        grid[y][x] === t.CellType.Floor && !(x === ax && y === ay) && !(x === gx && y === gy),
    )
    if (keyCandidates.length > 0) {
      const kc = keyCandidates[Math.floor(rng() * keyCandidates.length)]
      grid[kc[1]][kc[0]] = t.CellType.Key
    }
  }

  // 50% lava
  if (rng() < 0.5) addLava(grid, rng, ax, ay, gx, gy)

  return [ax, ay]
}

/** Open room with lava strips/patches. */
function genLavaField(grid: t.CellType[][], rng: () => number): [number, number] | null {
  const h = grid.length
  const w = grid[0].length

  const roll = rng()
  if (roll < 0.3) {
    // Horizontal strips
    const numStrips = 1 + Math.floor(rng() * 3)
    for (let s = 0; s < numStrips; s++) {
      const row = 3 + Math.floor(rng() * (h - 6))
      const minLen = Math.max(1, Math.floor(0.6 * (w - 2)))
      const maxLen = Math.max(minLen + 1, Math.floor(0.8 * (w - 2)) + 1)
      const stripLen = minLen + Math.floor(rng() * (maxLen - minLen))
      const startX = 1 + Math.floor(rng() * Math.max(1, w - 1 - stripLen))
      const endX = Math.min(startX + stripLen, w - 1)
      for (let x = startX; x < endX; x++) grid[row][x] = t.CellType.Lava
      // Cut 1-2 gaps
      if (endX > startX) {
        const numGaps = 1 + Math.floor(rng() * 2)
        for (let g = 0; g < numGaps; g++) {
          const gapX = startX + Math.floor(rng() * (endX - startX))
          grid[row][gapX] = t.CellType.Floor
        }
      }
    }
  } else if (roll < 0.75) {
    // Zigzag
    const availableRows: number[] = []
    for (let y = 2; y < h - 2; y++) availableRows.push(y)
    const numRows = Math.min(3 + Math.floor(rng() * 3), availableRows.length)
    // Pick random rows
    for (let i = availableRows.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      ;[availableRows[i], availableRows[j]] = [availableRows[j], availableRows[i]]
    }
    const rows = availableRows.slice(0, numRows).sort((a, b) => a - b)
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i]
      for (let x = 1; x < w - 1; x++) grid[row][x] = t.CellType.Lava
      // Offset gaps
      const gapX = 1 + ((i * 3) % (w - 2))
      grid[row][gapX] = t.CellType.Floor
      if (rng() < 0.5 && gapX + 1 < w - 1) grid[row][gapX + 1] = t.CellType.Floor
    }
  } else {
    // Patches
    const numPatches = 2 + Math.floor(rng() * 3)
    for (let p = 0; p < numPatches; p++) {
      const px = 2 + Math.floor(rng() * (w - 4))
      const py = 2 + Math.floor(rng() * (h - 4))
      const patchSize = 2 + Math.floor(rng() * 3)
      const cells: [number, number][] = [[px, py]]
      for (let j = 0; j < patchSize - 1; j++) {
        const [cx, cy] = cells[cells.length - 1]
        const dirs: [number, number][] = [
          [0, 1],
          [0, -1],
          [1, 0],
          [-1, 0],
        ]
        const [dx, dy] = dirs[Math.floor(rng() * 4)]
        const nx = cx + dx
        const ny = cy + dy
        if (nx >= 1 && nx < w - 1 && ny >= 1 && ny < h - 1) cells.push([nx, ny])
      }
      for (const [cx, cy] of cells) grid[cy][cx] = t.CellType.Lava
    }
  }

  // Agent top half, goal bottom half
  const topFloor: [number, number][] = []
  const botFloor: [number, number][] = []
  for (let y = 1; y < Math.floor(h / 2); y++)
    for (let x = 1; x < w - 1; x++) if (grid[y][x] === t.CellType.Floor) topFloor.push([x, y])
  for (let y = Math.floor(h / 2); y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) if (grid[y][x] === t.CellType.Floor) botFloor.push([x, y])

  let ax: number, ay: number, gx: number, gy: number
  if (topFloor.length > 0 && botFloor.length > 0) {
    const a = topFloor[Math.floor(rng() * topFloor.length)]
    ax = a[0]
    ay = a[1]
    const g = botFloor[Math.floor(rng() * botFloor.length)]
    gx = g[0]
    gy = g[1]
  } else {
    const pos = placeAgentAndGoal(grid, rng, 5)
    if (!pos) return null
    ax = pos.ax
    ay = pos.ay
    gx = pos.gx
    gy = pos.gy
  }
  grid[gy][gx] = t.CellType.Goal

  // Ensure solvability by removing lava if blocked
  for (let attempt = 0; attempt < 20; attempt++) {
    if (bfsDistanceSafe(grid, ax, ay, gx, gy) !== null) break
    const lavaCells: [number, number][] = []
    for (let y = 1; y < h - 1; y++)
      for (let x = 1; x < w - 1; x++) if (grid[y][x] === t.CellType.Lava) lavaCells.push([x, y])
    if (lavaCells.length === 0) break
    const rc = lavaCells[Math.floor(rng() * lavaCells.length)]
    grid[rc[1]][rc[0]] = t.CellType.Floor
  }

  // 40% door/key
  if (rng() < 0.4) addDoorAndKey(grid, rng, ax, ay, gx, gy)

  return [ax, ay]
}

/** Wall segments creating corridors and channels. */
function genWallSegments(grid: t.CellType[][], rng: () => number): [number, number] | null {
  const h = grid.length
  const w = grid[0].length
  const numSegments = 3 + Math.floor(rng() * 3) // 3-5
  const segmentPositions: { isH: boolean; pos: number }[] = []

  for (let s = 0; s < numSegments; s++) {
    const horizontal = rng() < 0.5
    if (horizontal) {
      const valid = []
      for (let y = 2; y < h - 2; y++) {
        if (segmentPositions.every((sp) => !sp.isH || Math.abs(y - sp.pos) >= 2)) valid.push(y)
      }
      if (valid.length === 0) continue
      const row = valid[Math.floor(rng() * valid.length)]
      segmentPositions.push({ isH: true, pos: row })
      const span =
        Math.floor(0.5 * (w - 2)) +
        Math.floor(rng() * (Math.floor(0.8 * (w - 2)) - Math.floor(0.5 * (w - 2)) + 1))
      const startX = 1 + Math.floor(rng() * Math.max(1, w - 1 - span))
      for (let x = startX; x < Math.min(startX + span, w - 1); x++) grid[row][x] = t.CellType.Wall
    } else {
      const valid = []
      for (let x = 2; x < w - 2; x++) {
        if (segmentPositions.every((sp) => sp.isH || Math.abs(x - sp.pos) >= 2)) valid.push(x)
      }
      if (valid.length === 0) continue
      const col = valid[Math.floor(rng() * valid.length)]
      segmentPositions.push({ isH: false, pos: col })
      const span =
        Math.floor(0.5 * (h - 2)) +
        Math.floor(rng() * (Math.floor(0.8 * (h - 2)) - Math.floor(0.5 * (h - 2)) + 1))
      const startY = 1 + Math.floor(rng() * Math.max(1, h - 1 - span))
      for (let y = startY; y < Math.min(startY + span, h - 1); y++) grid[y][col] = t.CellType.Wall
    }
  }

  const pos = placeAgentAndGoal(grid, rng, 5)
  if (!pos) return null
  const { ax, ay, gx, gy } = pos
  grid[gy][gx] = t.CellType.Goal

  // 0-3 doors (60% chance)
  const roll = rng()
  let numDoors: number
  if (roll < 0.4) numDoors = 0
  else if (roll < 0.75) numDoors = 1
  else if (roll < 0.9) numDoors = 2
  else numDoors = 3
  for (let i = 0; i < numDoors; i++) addDoorAndKey(grid, rng, ax, ay, gx, gy)

  // 50% lava
  if (rng() < 0.5) addLava(grid, rng, ax, ay, gx, gy)

  return [ax, ay]
}

/** Room partitions with lava patches near the goal. */
function genHybrid(grid: t.CellType[][], rng: () => number): [number, number] | null {
  const h = grid.length
  const w = grid[0].length
  const numPartitions = 1 + Math.floor(rng() * 2) // 1-2
  const doorPositions: [number, number][] = []
  const usedPositions = new Set<number>()

  for (let p = 0; p < numPartitions; p++) {
    const horizontal = rng() < 0.5
    if (horizontal) {
      const valid = []
      for (let y = 2; y < h - 2; y++) if (!usedPositions.has(y)) valid.push(y)
      if (valid.length === 0) continue
      const pos = valid[Math.floor(rng() * valid.length)]
      usedPositions.add(pos)
      for (let x = 1; x < w - 1; x++) grid[pos][x] = t.CellType.Wall
      const gapX = 1 + Math.floor(rng() * (w - 2))
      grid[pos][gapX] = t.CellType.Door
      doorPositions.push([gapX, pos])
    } else {
      const valid = []
      for (let x = 2; x < w - 2; x++) if (!usedPositions.has(x)) valid.push(x)
      if (valid.length === 0) continue
      const pos = valid[Math.floor(rng() * valid.length)]
      usedPositions.add(pos)
      for (let y = 1; y < h - 1; y++) grid[y][pos] = t.CellType.Wall
      const gapY = 1 + Math.floor(rng() * (h - 2))
      grid[gapY][pos] = t.CellType.Door
      doorPositions.push([pos, gapY])
    }
  }

  const pos = placeAgentAndGoal(grid, rng, 5)
  if (!pos) return null
  const { ax, ay, gx, gy } = pos
  grid[gy][gx] = t.CellType.Goal

  // Place keys for surviving doors
  for (const [dx, dy] of doorPositions) {
    if (grid[dy][dx] !== t.CellType.Door) continue
    const reachable = bfsReachable(grid, ax, ay)
    const keyCandidates = reachable.filter(
      ([x, y]) =>
        grid[y][x] === t.CellType.Floor && !(x === ax && y === ay) && !(x === gx && y === gy),
    )
    if (keyCandidates.length > 0) {
      const kc = keyCandidates[Math.floor(rng() * keyCandidates.length)]
      grid[kc[1]][kc[0]] = t.CellType.Key
    }
  }

  // 3-7 lava cells as patch near goal
  const numLava = 3 + Math.floor(rng() * 5)
  const lavaCells: [number, number][] = [[gx, gy]]
  for (let i = 0; i < numLava; i++) {
    const [cx, cy] = lavaCells[lavaCells.length - 1]
    const dirs: [number, number][] = [
      [0, 1],
      [0, -1],
      [1, 0],
      [-1, 0],
    ]
    const [ddx, ddy] = dirs[Math.floor(rng() * 4)]
    const nx = cx + ddx
    const ny = cy + ddy
    if (nx >= 1 && nx < w - 1 && ny >= 1 && ny < h - 1) lavaCells.push([nx, ny])
  }
  for (const [cx, cy] of lavaCells) {
    if (cx === gx && cy === gy) continue
    if (cx === ax && cy === ay) continue
    if (grid[cy][cx] === t.CellType.Floor) grid[cy][cx] = t.CellType.Lava
  }

  // Ensure solvability
  for (let attempt = 0; attempt < 20; attempt++) {
    if (bfsDistanceSafe(grid, ax, ay, gx, gy) !== null) break
    const lc: [number, number][] = []
    for (let y = 1; y < h - 1; y++)
      for (let x = 1; x < w - 1; x++) if (grid[y][x] === t.CellType.Lava) lc.push([x, y])
    if (lc.length === 0) break
    const rc = lc[Math.floor(rng() * lc.length)]
    grid[rc[1]][rc[0]] = t.CellType.Floor
  }

  return [ax, ay]
}

/** BSP room-corridor layout. */
function genBspRooms(grid: t.CellType[][], rng: () => number): [number, number] | null {
  const h = grid.length
  const w = grid[0].length
  // Fill interior with walls
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) grid[y][x] = t.CellType.Wall

  const interior: Rect = { x: 1, y: 1, w: w - 2, h: h - 2 }
  const rooms = bspSplit(interior, rng, 2)

  for (const room of rooms) {
    for (let y = room.y; y < room.y + room.h; y++)
      for (let x = room.x; x < room.x + room.w; x++)
        if (y > 0 && y < h - 1 && x > 0 && x < w - 1) grid[y][x] = t.CellType.Floor
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

  const floors = getFloorCells(grid)
  if (floors.length < 8) return null

  // Place agent and goal with max BFS distance
  let bestDist = -1
  let agentPos: [number, number] = floors[0]
  let goalPos: [number, number] = floors[floors.length - 1]
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
  const [ax, ay] = agentPos
  const [gx, gy] = goalPos

  // Door on chokepoint (50%)
  if (rng() < 0.5) {
    const chokepoints = floors.filter(
      ([x, y]) => isChokepoint(grid, x, y) && !(x === ax && y === ay) && !(x === gx && y === gy),
    )
    if (chokepoints.length > 0) {
      const doorCell = chokepoints[Math.floor(rng() * chokepoints.length)]
      grid[doorCell[1]][doorCell[0]] = t.CellType.Door
      const reachable = getFloorCells(grid).filter(([x, y]) => {
        if (x === ax && y === ay) return false
        if (x === gx && y === gy) return false
        return bfs(grid, agentPos, [x, y], false) >= 0
      })
      if (reachable.length > 0) {
        const keyCell = reachable[Math.floor(rng() * reachable.length)]
        grid[keyCell[1]][keyCell[0]] = t.CellType.Key
      } else {
        grid[doorCell[1]][doorCell[0]] = t.CellType.Floor
      }
    }
  }

  // 60% lava
  if (rng() < 0.6) {
    const lavaCount = 2 + Math.floor(rng() * 4)
    let placed = 0
    for (let i = 0; i < lavaCount * 5 && placed < lavaCount; i++) {
      const candidates = getFloorCells(grid).filter(
        ([x, y]) => !(x === ax && y === ay) && !(x === gx && y === gy),
      )
      if (candidates.length === 0) break
      const cell = candidates[Math.floor(rng() * candidates.length)]
      grid[cell[1]][cell[0]] = t.CellType.Lava
      if (bfsDistanceSafe(grid, ax, ay, gx, gy) === null) {
        grid[cell[1]][cell[0]] = t.CellType.Floor
      } else {
        placed++
      }
    }
  }

  return agentPos
}

/** Open layout with scattered walls. */
function genScatteredWalls(grid: t.CellType[][], rng: () => number): [number, number] | null {
  const h = grid.length
  const w = grid[0].length
  const wallCount = 4 + Math.floor(rng() * 10)
  for (let i = 0; i < wallCount; i++) {
    const wx = 1 + Math.floor(rng() * (w - 2))
    const wy = 1 + Math.floor(rng() * (h - 2))
    grid[wy][wx] = t.CellType.Wall
    if (rng() < 0.4) {
      const dx = rng() < 0.5 ? 1 : 0
      const dy = dx === 0 ? 1 : 0
      for (let j = 1; j <= 1 + Math.floor(rng() * 2); j++) {
        const ex = wx + dx * j
        const ey = wy + dy * j
        if (ex > 0 && ex < w - 1 && ey > 0 && ey < h - 1) grid[ey][ex] = t.CellType.Wall
      }
    }
  }

  const floors = getFloorCells(grid)
  if (floors.length < 8) return null

  // Place agent and goal with max BFS distance
  let bestDist = -1
  let agentPos: [number, number] = floors[0]
  let goalPos: [number, number] = floors[floors.length - 1]
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
  const [ax, ay] = agentPos
  const [gx, gy] = goalPos

  // Door on chokepoint (50%)
  let hasDoor = false
  if (rng() < 0.5) {
    const chokepoints = floors.filter(
      ([x, y]) => isChokepoint(grid, x, y) && !(x === ax && y === ay) && !(x === gx && y === gy),
    )
    if (chokepoints.length > 0) {
      const doorCell = chokepoints[Math.floor(rng() * chokepoints.length)]
      grid[doorCell[1]][doorCell[0]] = t.CellType.Door
      const reachable = getFloorCells(grid).filter(([x, y]) => {
        if (x === ax && y === ay) return false
        if (x === gx && y === gy) return false
        return bfs(grid, agentPos, [x, y], false) >= 0
      })
      if (reachable.length > 0) {
        const keyCell = reachable[Math.floor(rng() * reachable.length)]
        grid[keyCell[1]][keyCell[0]] = t.CellType.Key
        hasDoor = true
      } else {
        grid[doorCell[1]][doorCell[0]] = t.CellType.Floor
      }
    }
  }

  // 60% lava
  if (rng() < 0.6) {
    const lavaCount = 2 + Math.floor(rng() * 4)
    let placed = 0
    for (let i = 0; i < lavaCount * 5 && placed < lavaCount; i++) {
      const candidates = getFloorCells(grid).filter(
        ([x, y]) => !(x === ax && y === ay) && !(x === gx && y === gy),
      )
      if (candidates.length === 0) break
      const cell = candidates[Math.floor(rng() * candidates.length)]
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
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) if (grid[y][x] === t.CellType.Key) keyCell = [x, y]
    if (!keyCell || bfs(grid, agentPos, keyCell, false) < 0) return null
    if (bfs(grid, agentPos, goalPos, true) < 0) return null
  } else {
    if (bfs(grid, agentPos, goalPos, false) < 0) return null
  }

  return agentPos
}

// --- Main entry point ---

/** Procedural level generation with 7 generators matching Python training distribution */
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

  const grid = initBorder(width, height)

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

  const roll = rng()
  let agentPos: [number, number] | null
  // Weights match Python training (environment.py:448-461)
  if (roll < 0.08) agentPos = genOpenRoom(grid, rng)
  else if (roll < 0.22) agentPos = genRoomPartition(grid, rng)
  else if (roll < 0.36) agentPos = genLavaField(grid, rng)
  else if (roll < 0.48) agentPos = genWallSegments(grid, rng)
  else if (roll < 0.6) agentPos = genHybrid(grid, rng)
  else if (roll < 0.8) agentPos = genBspRooms(grid, rng)
  else agentPos = genScatteredWalls(grid, rng)

  if (!agentPos) return null

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
