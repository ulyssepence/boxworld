import * as React from 'react'
import * as t from './types'

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
      const grid = state.currentLevel.grid.map((row) => [...row])
      grid[action.y][action.x] = action.cellType
      return {
        ...state,
        currentLevel: { ...state.currentLevel, grid },
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
    if (!state.isPlaying) return

    const episode = state.episodes[state.currentEpisodeIndex]
    if (!episode) return

    const intervalMs = 1000 / state.playbackSpeed
    const id = setInterval(() => {
      dispatch({ type: 'STEP_FORWARD' })
    }, intervalMs)

    return () => clearInterval(id)
  }, [state.isPlaying, state.playbackSpeed, state.currentEpisodeIndex, state.episodes, dispatch])
}
