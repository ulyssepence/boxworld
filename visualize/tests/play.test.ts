import { describe, test, expect } from 'bun:test'
import { createInitialState, step } from '../src/play'
import { CellType, Action, Level, Step, GameState } from '../src/types'

/** Helper: create a small level for testing */
function makeLevel(overrides?: Partial<Level>): Level {
  // 5x5 grid: walls around the border, floor inside
  const grid: CellType[][] = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
  ]
  return {
    id: 'test',
    name: 'Test',
    width: 5,
    height: 5,
    grid,
    agentStart: [1, 1],
    ...overrides,
  }
}

describe('play.step', () => {
  test('toggle door consumes key', () => {
    // Agent at (1,2), key already picked up, door at (2,2)
    const grid: CellType[][] = [
      [1, 1, 1, 1, 1],
      [1, 0, 0, 0, 1],
      [1, 0, 2, 0, 1], // door at (2,2)
      [1, 0, 0, 0, 1],
      [1, 1, 1, 1, 1],
    ]
    const level = makeLevel({ grid, agentStart: [1, 2] })
    let state = createInitialState(level)
    // Give the agent a key
    state = { ...state, inventory: { hasKey: true } }

    // Toggle should open the door AND consume the key
    const after = step(state, Action.Toggle)
    expect(after.level.grid[2][2]).toBe(CellType.Floor) // door opened
    expect(after.inventory.hasKey).toBe(false) // key consumed
  })

  test('agent cannot toggle second door without picking up another key', () => {
    // Agent at (1,2), two doors at (2,2) and (3,2), key at (1,1)
    const grid: CellType[][] = [
      [1, 1, 1, 1, 1],
      [1, 3, 0, 0, 1], // key at (1,1)
      [1, 0, 2, 2, 1], // doors at (2,2) and (3,2)
      [1, 0, 0, 0, 1],
      [1, 1, 1, 1, 1],
    ]
    const level = makeLevel({ grid, agentStart: [1, 2] })
    let state = createInitialState(level)

    // Pick up key
    state = step(state, Action.Up) // move to (1,1) where key is
    state = step(state, Action.Pickup)
    expect(state.inventory.hasKey).toBe(true)

    // Move back down next to first door
    state = step(state, Action.Down) // back to (1,2)

    // Toggle first door
    state = step(state, Action.Toggle)
    expect(state.level.grid[2][2]).toBe(CellType.Floor) // first door opened
    expect(state.inventory.hasKey).toBe(false) // key consumed

    // Try to toggle second door — should fail (no key)
    state = step(state, Action.Right) // move to (2,2) next to second door at (3,2)
    state = step(state, Action.Toggle)
    expect(state.level.grid[2][3]).toBe(CellType.Door) // second door still closed
  })

  test('pickup key changes grid (key removed)', () => {
    const grid: CellType[][] = [
      [1, 1, 1, 1, 1],
      [1, 3, 0, 0, 1], // key at (1,1)
      [1, 0, 0, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 1, 1, 1, 1],
    ]
    const level = makeLevel({ grid, agentStart: [1, 1] })
    let state = createInitialState(level)

    state = step(state, Action.Pickup)
    expect(state.inventory.hasKey).toBe(true)
    expect(state.level.grid[1][1]).toBe(CellType.Floor) // key removed from grid
  })
})

describe('inference episode replay (simulates runAgent)', () => {
  /**
   * Simulates what runAgent() does: runs play.step in a loop, records
   * prevState at each step (state BEFORE the action), then verifies the
   * recorded steps show grid changes between steps.
   */
  test('recorded steps reflect key pickup and door toggle in grid', () => {
    // Level: agent at (1,1), key at (1,3), door at (2,3), goal at (3,3)
    const grid: CellType[][] = [
      [1, 1, 1, 1, 1],
      [1, 0, 0, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 3, 2, 4, 1], // key at (1,3), door at (2,3), goal at (3,3)
      [1, 1, 1, 1, 1],
    ]
    const level = makeLevel({ grid, agentStart: [1, 1] })

    // Simulate runAgent's recording pattern
    let gameState = createInitialState(level)
    const steps: Step[] = []
    const actions = [
      Action.Down, // (1,1) -> (1,2)
      Action.Down, // (1,2) -> (1,3) onto key
      Action.Pickup, // pick up key
      Action.Right, // (1,3) -> (2,3) — blocked by door, stays at (1,3)
      Action.Toggle, // toggle door at (2,3)
      Action.Right, // (1,3) -> (2,3) through opened door
      Action.Right, // (2,3) -> (3,3) goal!
    ]

    for (const action of actions) {
      const prevState = gameState
      gameState = step(gameState, action)
      steps.push({
        state: prevState,
        action,
        reward: gameState.reward - prevState.reward,
      })
      if (gameState.done) break
    }

    // Step 0: moving down, no key yet, key visible in grid
    expect(steps[0].state.inventory.hasKey).toBe(false)
    expect(steps[0].state.level.grid[3][1]).toBe(CellType.Key)
    expect(steps[0].state.level.grid[3][2]).toBe(CellType.Door)

    // Step 2: Pickup action. State BEFORE pickup still shows key in grid.
    expect(steps[2].action).toBe(Action.Pickup)
    expect(steps[2].state.level.grid[3][1]).toBe(CellType.Key)
    expect(steps[2].state.inventory.hasKey).toBe(false)

    // Step 3: State AFTER pickup — key removed from grid, hasKey = true
    expect(steps[3].state.level.grid[3][1]).toBe(CellType.Floor)
    expect(steps[3].state.inventory.hasKey).toBe(true)

    // Step 4: Toggle action. State BEFORE toggle still shows door.
    expect(steps[4].action).toBe(Action.Toggle)
    expect(steps[4].state.level.grid[3][2]).toBe(CellType.Door)
    expect(steps[4].state.inventory.hasKey).toBe(true)

    // Step 5: State AFTER toggle — door removed, key consumed
    expect(steps[5].state.level.grid[3][2]).toBe(CellType.Floor)
    expect(steps[5].state.inventory.hasKey).toBe(false)

    // Final state: agent reached goal
    expect(gameState.done).toBe(true)
    expect(gameState.reward).toBeGreaterThan(0)
  })

  test('each recorded step has a distinct level object (not shared references)', () => {
    const grid: CellType[][] = [
      [1, 1, 1, 1, 1],
      [1, 0, 0, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 3, 2, 4, 1],
      [1, 1, 1, 1, 1],
    ]
    const level = makeLevel({ grid, agentStart: [1, 1] })

    let gameState = createInitialState(level)
    const steps: Step[] = []
    const actions = [
      Action.Down,
      Action.Down,
      Action.Pickup,
      Action.Toggle,
      Action.Right,
      Action.Right,
    ]

    for (const action of actions) {
      const prevState = gameState
      gameState = step(gameState, action)
      steps.push({ state: prevState, action, reward: 0 })
      if (gameState.done) break
    }

    // Verify no two consecutive steps share the same level object reference
    for (let i = 1; i < steps.length; i++) {
      expect(steps[i].state.level).not.toBe(steps[i - 1].state.level)
    }
  })

  test('observation encoding changes after pickup (has_key flips)', () => {
    // This simulates what stateToTensor sees: the has_key field must change
    const grid: CellType[][] = [
      [1, 1, 1, 1, 1],
      [1, 3, 0, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 1, 1, 1, 1],
    ]
    const level = makeLevel({ grid, agentStart: [1, 1] })
    let state = createInitialState(level)

    // Before pickup
    expect(state.inventory.hasKey).toBe(false)

    // Pickup
    state = step(state, Action.Pickup)
    expect(state.inventory.hasKey).toBe(true)

    // After toggle (even without a door nearby, hasKey should stay true)
    state = step(state, Action.Toggle)
    expect(state.inventory.hasKey).toBe(true) // no door to toggle, key stays

    // Now add a door scenario
    const grid2: CellType[][] = [
      [1, 1, 1, 1, 1],
      [1, 0, 2, 0, 1], // door at (2,1)
      [1, 0, 0, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 1, 1, 1, 1],
    ]
    const level2 = makeLevel({ grid: grid2, agentStart: [1, 1] })
    let state2 = createInitialState(level2)
    state2 = { ...state2, inventory: { hasKey: true } }

    // Toggle near door — key consumed
    state2 = step(state2, Action.Toggle)
    expect(state2.inventory.hasKey).toBe(false)
    expect(state2.level.grid[1][2]).toBe(CellType.Floor)
  })
})
