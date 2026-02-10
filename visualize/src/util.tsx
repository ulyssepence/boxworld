import * as React from 'react'
import * as t from './types'
import * as play from './play'
import * as ml from './ml'

export interface LiveInferenceState {
  active: boolean
  running: boolean
  gameState: t.GameState | null
  prevAgentPosition: [number, number] | null
  lastAction: t.Action | null
  lastQValues: t.QValues | null
  stepCount: number
  history: t.Step[]
  generation: number
}

const initialLiveInference: LiveInferenceState = {
  active: false,
  running: false,
  gameState: null,
  prevAgentPosition: null,
  lastAction: null,
  lastQValues: null,
  stepCount: 0,
  history: [],
  generation: 0,
}

export type ViewMode = 'recordings' | 'inference'

export interface AppState {
  levels: t.LevelInfo[]
  checkpoints: t.CheckpointInfo[]
  currentLevel: t.Level | null
  originalLevel: t.Level | null
  episodes: t.Episode[]
  currentEpisodeIndex: number
  currentStep: number
  isPlaying: boolean
  playbackSpeed: number // steps per second
  editMode: boolean
  inferenceLoading: boolean
  liveInference: LiveInferenceState
  viewMode: ViewMode
  editVersion: number
}

export type AppAction =
  | { type: 'LOAD_LEVELS'; levels: t.LevelInfo[] }
  | { type: 'LOAD_CHECKPOINTS'; checkpoints: t.CheckpointInfo[] }
  | { type: 'LOAD_LEVEL'; level: t.Level; episodes: t.Episode[] }
  | { type: 'SET_EPISODE'; index: number }
  | { type: 'SEEK'; step: number }
  | { type: 'PLAY' }
  | { type: 'PAUSE' }
  | { type: 'STEP_FORWARD' }
  | { type: 'STEP_BACKWARD' }
  | { type: 'SET_SPEED'; speed: number }
  | { type: 'TOGGLE_EDIT_MODE' }
  | { type: 'EDIT_CELL'; x: number; y: number; cellType: t.CellType }
  | { type: 'RESET_LEVEL' }
  | { type: 'LOAD_INFERENCE_EPISODE'; episode: t.Episode }
  | { type: 'SET_INFERENCE_LOADING'; loading: boolean }
  | { type: 'START_LIVE_INFERENCE' }
  | {
      type: 'LIVE_STEP'
      action: t.Action
      qValues: t.QValues
      newState: t.GameState
      preStepGrid: t.CellType[][]
      generation: number
    }
  | { type: 'LIVE_PLAY' }
  | { type: 'LIVE_PAUSE' }
  | { type: 'STOP_LIVE_INFERENCE' }
  | { type: 'SET_VIEW_MODE'; mode: ViewMode }
  | { type: 'RESTART_LIVE_INFERENCE' }
  | { type: 'GENERATE_LEVEL'; level: t.Level }

const initialState: AppState = {
  levels: [],
  checkpoints: [],
  currentLevel: null,
  originalLevel: null,
  episodes: [],
  currentEpisodeIndex: 0,
  currentStep: 0,
  isPlaying: false,
  playbackSpeed: 4,
  editMode: false,
  inferenceLoading: false,
  liveInference: initialLiveInference,
  viewMode: 'recordings',
  editVersion: 0,
}

function getMaxStep(state: AppState): number {
  const episode = state.episodes[state.currentEpisodeIndex]
  if (!episode) return 0
  return Math.max(0, episode.steps.length - 1)
}

function reducer(state: AppState, action: AppAction): AppState {
  console.log(action, state)
  switch (action.type) {
    case 'LOAD_LEVELS':
      return { ...state, levels: action.levels }

    case 'LOAD_CHECKPOINTS':
      return { ...state, checkpoints: action.checkpoints }

    case 'LOAD_LEVEL':
      return {
        ...state,
        currentLevel: action.level,
        originalLevel: {
          ...action.level,
          grid: action.level.grid.map((row) => [...row]),
        },
        episodes: action.episodes,
        currentEpisodeIndex: 0,
        currentStep: 0,
        isPlaying: false,
        editMode: false,
        liveInference: initialLiveInference,
        viewMode: 'recordings',
        editVersion: 0,
      }

    case 'SET_EPISODE':
      return {
        ...state,
        currentEpisodeIndex: action.index,
        currentStep: 0,
        isPlaying: false,
      }

    case 'SEEK':
      return { ...state, currentStep: action.step }

    case 'PLAY':
      return { ...state, isPlaying: true }

    case 'PAUSE':
      return { ...state, isPlaying: false }

    case 'STEP_FORWARD': {
      const max = getMaxStep(state)
      const next = state.currentStep + 1
      if (next > max) {
        // Loop back to start
        return { ...state, currentStep: 0 }
      }
      return { ...state, currentStep: next }
    }

    case 'STEP_BACKWARD':
      return { ...state, currentStep: Math.max(0, state.currentStep - 1) }

    case 'SET_SPEED':
      return { ...state, playbackSpeed: action.speed }

    case 'TOGGLE_EDIT_MODE':
      return { ...state, editMode: !state.editMode }

    case 'EDIT_CELL': {
      if (!state.currentLevel) return state
      // Block edits to the cell the agent is standing on during live inference
      if (state.liveInference.active && state.liveInference.gameState) {
        const [agentX, agentY] = state.liveInference.gameState.agentPosition
        if (action.x === agentX && action.y === agentY) return state
      }
      const grid = state.currentLevel.grid.map((row) => [...row])
      grid[action.y][action.x] = action.cellType
      return {
        ...state,
        currentLevel: { ...state.currentLevel, grid },
        editVersion: state.editVersion + 1,
      }
    }

    case 'RESET_LEVEL': {
      if (!state.originalLevel) return state
      return {
        ...state,
        currentLevel: {
          ...state.originalLevel,
          grid: state.originalLevel.grid.map((row) => [...row]),
        },
        episodes: [],
        currentEpisodeIndex: 0,
        currentStep: 0,
        isPlaying: false,
      }
    }

    case 'LOAD_INFERENCE_EPISODE':
      return {
        ...state,
        episodes: [action.episode],
        currentEpisodeIndex: 0,
        currentStep: 0,
        isPlaying: false,
      }

    case 'SET_INFERENCE_LOADING':
      return { ...state, inferenceLoading: action.loading }

    case 'START_LIVE_INFERENCE': {
      if (!state.currentLevel) return state
      const gameState = play.createInitialState(state.currentLevel)
      return {
        ...state,
        editMode: false,
        viewMode: 'inference',
        liveInference: {
          active: true,
          running: false,
          gameState,
          prevAgentPosition: null,
          lastAction: null,
          lastQValues: null,
          stepCount: 0,
          history: [],
          generation: state.liveInference.generation + 1,
        },
      }
    }

    case 'LIVE_STEP': {
      const li = state.liveInference
      if (!li.active || !li.gameState || !state.currentLevel) return state
      if (action.generation !== li.generation) return state
      const prevPos: [number, number] = [
        li.gameState.agentPosition[0],
        li.gameState.agentPosition[1],
      ]
      const historyStep: t.Step = {
        state: li.gameState,
        action: action.action,
        reward: action.newState.reward - li.gameState.reward,
        qValues: action.qValues,
      }
      // Apply only agent-caused grid changes (key pickup, door toggle) on top of
      // the current level grid, preserving any user edits made since the tick started.
      // Diff the tick's input grid (preStepGrid) against the output grid (newState)
      // to find agent-caused changes, then patch onto the current grid.
      const mergedGrid = state.currentLevel.grid.map((row) => [...row])
      const outputGrid = action.newState.level.grid
      for (let y = 0; y < state.currentLevel.height; y++) {
        for (let x = 0; x < state.currentLevel.width; x++) {
          if (action.preStepGrid[y][x] !== outputGrid[y][x]) {
            mergedGrid[y][x] = outputGrid[y][x]
          }
        }
      }
      // Update newState's level to use the merged grid so gameState stays consistent
      const updatedNewState: t.GameState = {
        ...action.newState,
        level: { ...state.currentLevel, grid: mergedGrid },
      }
      return {
        ...state,
        currentLevel: { ...state.currentLevel, grid: mergedGrid },
        liveInference: {
          ...li,
          gameState: updatedNewState,
          prevAgentPosition: prevPos,
          lastAction: action.action,
          lastQValues: action.qValues,
          stepCount: li.stepCount + 1,
          history: [...li.history, historyStep],
        },
      }
    }

    case 'LIVE_PLAY':
      return {
        ...state,
        liveInference: { ...state.liveInference, running: true },
      }

    case 'LIVE_PAUSE':
      return {
        ...state,
        liveInference: { ...state.liveInference, running: false },
      }

    case 'STOP_LIVE_INFERENCE':
      return {
        ...state,
        liveInference: initialLiveInference,
      }

    case 'SET_VIEW_MODE': {
      if (action.mode === 'recordings' && state.liveInference.active) {
        return {
          ...state,
          viewMode: 'recordings',
          liveInference: initialLiveInference,
        }
      }
      return { ...state, viewMode: action.mode }
    }

    case 'RESTART_LIVE_INFERENCE': {
      if (!state.currentLevel) return state
      const freshState = play.createInitialState(state.currentLevel)
      return {
        ...state,
        liveInference: {
          ...state.liveInference,
          running: true,
          gameState: freshState,
          prevAgentPosition: null,
          lastAction: null,
          lastQValues: null,
          stepCount: 0,
          history: [],
          generation: state.liveInference.generation + 1,
        },
      }
    }

    case 'GENERATE_LEVEL':
      return {
        ...state,
        currentLevel: action.level,
        originalLevel: {
          ...action.level,
          grid: action.level.grid.map((row) => [...row]),
        },
        episodes: [],
        currentEpisodeIndex: 0,
        currentStep: 0,
        isPlaying: false,
        editMode: false,
        liveInference: initialLiveInference,
        viewMode: 'inference',
        editVersion: 0,
      }

    default:
      return state
  }
}

type AppContextType = [AppState, React.Dispatch<AppAction>]

const AppContext = React.createContext<AppContextType | null>(null)

export function AppProvider({ children }: { children: React.ReactNode }) {
  const value = React.useReducer(reducer, initialState)
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppContextType {
  const ctx = React.useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}

export function usePlayback() {
  const [state, dispatch] = useApp()

  React.useEffect(() => {
    if (state.viewMode !== 'recordings') return
    if (!state.isPlaying) return

    const episode = state.episodes[state.currentEpisodeIndex]
    if (!episode) return

    const intervalMs = 1000 / state.playbackSpeed
    const id = setInterval(() => {
      dispatch({ type: 'STEP_FORWARD' })
    }, intervalMs)

    return () => clearInterval(id)
  }, [
    state.viewMode,
    state.isPlaying,
    state.playbackSpeed,
    state.currentEpisodeIndex,
    state.episodes,
    dispatch,
  ])
}

export function useLiveInference() {
  const [state, dispatch] = useApp()
  const agentRef = React.useRef<ml.Agent | null>(null)
  const stateRef = React.useRef(state)
  const tickInProgressRef = React.useRef(false)

  stateRef.current = state

  const tick = React.useCallback(async () => {
    if (tickInProgressRef.current) return
    tickInProgressRef.current = true
    try {
      const s = stateRef.current
      const agent = agentRef.current
      const gs = s.liveInference.gameState
      if (!agent || !gs || !s.currentLevel || gs.done || s.liveInference.stepCount >= 200) {
        dispatch({ type: 'LIVE_PAUSE' })
        return
      }
      // Merge: use agent's position/inventory but the current level's grid (picks up user edits)
      const merged: t.GameState = {
        ...gs,
        level: s.currentLevel,
      }
      const preStepGrid = s.currentLevel.grid
      const gen = s.liveInference.generation
      const { action, qValues } = await agent.selectAction(merged)
      const newState = play.step(merged, action)
      dispatch({ type: 'LIVE_STEP', action, qValues, newState, preStepGrid, generation: gen })
      if (newState.done || s.liveInference.stepCount + 1 >= 200) {
        dispatch({ type: 'LIVE_PAUSE' })
      }
    } finally {
      tickInProgressRef.current = false
    }
  }, [dispatch])

  // Auto-stepping interval when active && running
  React.useEffect(() => {
    const li = state.liveInference
    if (!li.active || !li.running) return

    const intervalMs = 1000 / state.playbackSpeed
    const id = setInterval(tick, intervalMs)
    return () => clearInterval(id)
  }, [state.liveInference.active, state.liveInference.running, state.playbackSpeed, tick])

  const stepOnce = React.useCallback(async () => {
    dispatch({ type: 'LIVE_PAUSE' })
    await tick()
  }, [dispatch, tick])

  return { agentRef, stepOnce }
}
