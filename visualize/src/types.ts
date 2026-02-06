export enum CellType {
  Floor,
  Wall,
  Door,
  Key,
  Goal,
  Lava,
}

export enum Action {
  Up,
  Down,
  Left,
  Right,
  Pickup,
  Toggle,
}

/** Level definition loaded from JSON */
export interface Level {
  id: string
  name: string
  width: number
  height: number
  grid: CellType[][] // grid[y][x]
  agentStart: [number, number] // [x, y]
  seed?: number
}

/** Runtime game state */
export interface GameState {
  level: Level
  agentPosition: [number, number]
  inventory: { hasKey: boolean }
  done: boolean
  reward: number
  steps: number
}

/** Q-values for all actions at a position */
export interface QValues {
  [Action.Up]: number
  [Action.Down]: number
  [Action.Left]: number
  [Action.Right]: number
  [Action.Pickup]: number
  [Action.Toggle]: number
}

/** Single step in an episode */
export interface Step {
  state: GameState
  action: Action
  reward: number
  qValues?: QValues
}

/** Full episode recording */
export interface Episode {
  id: string
  agentId: string
  levelId: string
  steps: Step[]
  totalReward: number
}

/** API response types */
export interface CheckpointInfo {
  id: string
  trainingSteps: number
}

export interface LevelInfo {
  id: string
  name: string
  seed?: number
}

export interface LevelWithEpisodes {
  level: Level
  episodes: Episode[]
}

/** Messages (keep for compatibility but will be replaced) */
export type ToClientMessage = any
export type ToServerMessage = any
