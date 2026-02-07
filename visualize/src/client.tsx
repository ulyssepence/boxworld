import * as React from 'react'
import * as ReactDOM from 'react-dom/client'
import * as api from './api'
import * as util from './util'
import * as render from './render'
import * as ml from './ml'
import * as t from './types'

function cycleCellType(currentCell: t.CellType): t.CellType {
  switch (currentCell) {
    case t.CellType.Floor:
      return t.CellType.Wall
    case t.CellType.Wall:
      return t.CellType.Goal
    case t.CellType.Goal:
      return t.CellType.Floor
    default:
      return t.CellType.Floor
  }
}

function GameView() {
  const [state, dispatch] = util.useApp()
  util.usePlayback()

  const level = state.currentLevel
  if (!level) return null

  const li = state.liveInference

  if (li.active) {
    // Live inference mode: display currentLevel (shared mutable grid), agent from gameState
    const agentPos: [number, number] = li.gameState ? li.gameState.agentPosition : level.agentStart
    const prevAgentPos = li.prevAgentPosition ?? undefined

    const handleCellClick = (x: number, y: number) => {
      const currentCell = level.grid[y][x]
      dispatch({ type: 'EDIT_CELL', x, y, cellType: cycleCellType(currentCell) })
    }

    return (
      <>
        <render.Grid level={level} onCellClick={handleCellClick} />
        <render.Walls level={level} onCellClick={handleCellClick} />
        <render.Items level={level} onCellClick={handleCellClick} />
        <render.Agent
          position={agentPos}
          prevPosition={prevAgentPos}
          stepsPerSecond={state.playbackSpeed}
        />
        <render.QValueArrows qValues={li.lastQValues ?? undefined} position={agentPos} />
      </>
    )
  }

  // Replay mode (unchanged)
  const episode = state.episodes[state.currentEpisodeIndex]
  const currentStepData = episode?.steps[state.currentStep]
  const prevStepData = state.currentStep > 0 ? episode?.steps[state.currentStep - 1] : undefined

  const agentPos: [number, number] = currentStepData
    ? currentStepData.state.agentPosition
    : level.agentStart

  const prevAgentPos: [number, number] | undefined = prevStepData
    ? prevStepData.state.agentPosition
    : undefined

  const displayLevel: t.Level = state.editMode
    ? level
    : currentStepData
      ? currentStepData.state.level
      : level

  const handleCellClick = state.editMode
    ? (x: number, y: number) => {
        const currentCell = displayLevel.grid[y][x]
        dispatch({ type: 'EDIT_CELL', x, y, cellType: cycleCellType(currentCell) })
      }
    : undefined

  return (
    <>
      <render.Grid level={displayLevel} onCellClick={handleCellClick} />
      <render.Walls level={displayLevel} onCellClick={handleCellClick} />
      <render.Items level={displayLevel} onCellClick={handleCellClick} />
      <render.Agent
        position={agentPos}
        prevPosition={prevAgentPos}
        stepsPerSecond={state.playbackSpeed}
      />
      <render.QValueArrows qValues={currentStepData?.qValues} position={agentPos} />
    </>
  )
}

async function startLiveAgent(
  checkpointUrl: string,
  agentRef: React.MutableRefObject<ml.Agent | null>,
  dispatch: React.Dispatch<util.AppAction>,
) {
  dispatch({ type: 'SET_INFERENCE_LOADING', loading: true })
  try {
    const agent = new ml.Agent()
    await agent.load(checkpointUrl)
    agentRef.current = agent
    dispatch({ type: 'START_LIVE_INFERENCE' })
    dispatch({ type: 'LIVE_PLAY' })
  } catch (err) {
    console.error('Failed to load agent:', err)
  } finally {
    dispatch({ type: 'SET_INFERENCE_LOADING', loading: false })
  }
}

const ACTION_NAMES: Record<t.Action, string> = {
  [t.Action.Up]: 'Up',
  [t.Action.Down]: 'Down',
  [t.Action.Left]: 'Left',
  [t.Action.Right]: 'Right',
  [t.Action.Pickup]: 'Pickup',
  [t.Action.Toggle]: 'Toggle',
}

function QValuesPanel({ qValues }: { qValues: t.QValues }) {
  return (
    <div className="sidebar-section">
      <label>Q-Values</label>
      <div className="step-info">
        {(
          [
            [t.Action.Up, 'Up'],
            [t.Action.Down, 'Down'],
            [t.Action.Left, 'Left'],
            [t.Action.Right, 'Right'],
            [t.Action.Pickup, 'Pickup'],
            [t.Action.Toggle, 'Toggle'],
          ] as [t.Action, string][]
        ).map(([action, name]) => (
          <div key={action}>
            {name}: {qValues[action].toFixed(3)}
          </div>
        ))}
      </div>
    </div>
  )
}

function Sidebar() {
  const [state, dispatch] = util.useApp()
  const { agentRef, stepOnce } = util.useLiveInference()
  const [selectedCheckpointId, setSelectedCheckpointId] = React.useState<string>('')

  const li = state.liveInference

  // Space key toggles play/pause
  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault()
        if (li.active) {
          dispatch({ type: li.running ? 'LIVE_PAUSE' : 'LIVE_PLAY' })
        } else {
          dispatch({ type: state.isPlaying ? 'PAUSE' : 'PLAY' })
        }
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [state.isPlaying, li.active, li.running, dispatch])

  // Fetch levels on mount
  React.useEffect(() => {
    api.fetchLevels().then((levels) => {
      dispatch({ type: 'LOAD_LEVELS', levels })
    })
    api.fetchCheckpoints().then((checkpoints) => {
      dispatch({ type: 'LOAD_CHECKPOINTS', checkpoints })
    })
  }, [dispatch])

  // Auto-select the best (highest training steps) checkpoint when checkpoints load
  React.useEffect(() => {
    if (state.checkpoints.length > 0 && !selectedCheckpointId) {
      setSelectedCheckpointId(state.checkpoints[state.checkpoints.length - 1].id)
    }
  }, [state.checkpoints, selectedCheckpointId])

  const episode = state.episodes[state.currentEpisodeIndex]
  const currentStepData = episode?.steps[state.currentStep]
  const maxStep = episode ? Math.max(0, episode.steps.length - 1) : 0

  const handleLevelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value
    if (!id) return
    api.fetchLevel(id).then((data) => {
      if (!data.level) {
        console.error('Level not found:', id, data)
        return
      }
      dispatch({ type: 'LOAD_LEVEL', level: data.level, episodes: data.episodes })
    })
  }

  return (
    <div className="sidebar">
      <h2>Boxworld</h2>

      <div className="sidebar-section">
        <label>Level</label>
        <select onChange={handleLevelChange} defaultValue="">
          <option value="" disabled>
            Select a level...
          </option>
          {state.levels.map((l) => (
            <option key={l.id} value={l.id}>
              {l.name}
            </option>
          ))}
        </select>
      </div>

      {state.currentLevel && !li.active && (
        <div className="sidebar-section">
          <label>Edit Mode</label>
          <div className="controls-row">
            <button onClick={() => dispatch({ type: 'TOGGLE_EDIT_MODE' })}>
              {state.editMode ? 'Exit Edit' : 'Edit Level'}
            </button>
            {state.editMode && (
              <button onClick={() => dispatch({ type: 'RESET_LEVEL' })}>Reset</button>
            )}
          </div>
          {state.editMode && (
            <div className="step-info">Click tiles: Floor &rarr; Wall &rarr; Goal &rarr; Floor</div>
          )}
        </div>
      )}

      {state.currentLevel && !li.active && state.episodes.length > 0 && (
        <div className="sidebar-section">
          <label>Episode</label>
          <select
            value={state.currentEpisodeIndex}
            onChange={(e) => dispatch({ type: 'SET_EPISODE', index: Number(e.target.value) })}
          >
            {state.episodes.map((ep, i) => (
              <option key={ep.id} value={i}>
                {ep.trainingSteps != null
                  ? `${ep.trainingSteps.toLocaleString()} steps`
                  : `Episode ${i + 1}`}{' '}
                (reward: {ep.totalReward.toFixed(2)})
              </option>
            ))}
          </select>
        </div>
      )}

      {state.currentLevel && state.checkpoints.length > 0 && (
        <div className="sidebar-section">
          <label>Agent Inference</label>
          <select
            value={selectedCheckpointId}
            onChange={(e) => setSelectedCheckpointId(e.target.value)}
            disabled={li.active}
          >
            {state.checkpoints.map((cp) => (
              <option key={cp.id} value={cp.id}>
                {cp.trainingSteps.toLocaleString()} steps
              </option>
            ))}
          </select>
          {!li.active && (
            <button
              disabled={state.inferenceLoading || !selectedCheckpointId}
              onClick={() => {
                const cp = state.checkpoints.find((c) => c.id === selectedCheckpointId)
                if (!cp || !state.currentLevel) return
                const url = `/checkpoints/boxworld_${cp.trainingSteps}_steps.onnx`
                startLiveAgent(url, agentRef, dispatch)
              }}
            >
              {state.inferenceLoading ? 'Loading...' : 'Run Agent'}
            </button>
          )}
        </div>
      )}

      {li.active && (
        <div className="sidebar-section">
          <label>Live Inference</label>
          <div className="controls-row">
            <button
              onClick={() => dispatch({ type: li.running ? 'LIVE_PAUSE' : 'LIVE_PLAY' })}
              disabled={!!li.gameState?.done || li.stepCount >= 200}
            >
              {li.running ? '\u23F8' : '\u25B6'}
            </button>
            <button
              onClick={stepOnce}
              disabled={li.running || !!li.gameState?.done || li.stepCount >= 200}
            >
              &#9654;&#9654;
            </button>
            <button onClick={() => dispatch({ type: 'STOP_LIVE_INFERENCE' })}>Stop</button>
          </div>
          <div className="controls-row">
            <label>Speed</label>
            <input
              type="range"
              min={1}
              max={20}
              value={state.playbackSpeed}
              onChange={(e) => dispatch({ type: 'SET_SPEED', speed: Number(e.target.value) })}
            />
            <span>{state.playbackSpeed} sps</span>
          </div>
          <div className="step-info">Click tiles to edit the level while the agent runs.</div>
        </div>
      )}

      {li.active && li.gameState && (
        <div className="sidebar-section">
          <label>Step Info</label>
          <div className="step-info">
            <div>Step: {li.stepCount} / 200</div>
            {li.lastAction !== null && <div>Action: {ACTION_NAMES[li.lastAction]}</div>}
            <div>Reward: {li.gameState.reward.toFixed(3)}</div>
            <div>
              Position: ({li.gameState.agentPosition[0]}, {li.gameState.agentPosition[1]})
            </div>
            <div>Has Key: {li.gameState.inventory.hasKey ? 'Yes' : 'No'}</div>
            <div>Done: {li.gameState.done ? 'Yes' : 'No'}</div>
          </div>
        </div>
      )}

      {li.active && li.lastQValues && <QValuesPanel qValues={li.lastQValues} />}

      {!li.active && state.currentLevel && (
        <div className="sidebar-section">
          <label>Playback</label>
          <div className="controls-row">
            <button
              onClick={() => dispatch({ type: 'STEP_BACKWARD' })}
              disabled={state.currentStep <= 0}
            >
              &#9664;&#9664;
            </button>
            <button
              onClick={() => dispatch({ type: state.isPlaying ? 'PAUSE' : 'PLAY' })}
              disabled={!episode}
            >
              {state.isPlaying ? '\u23F8' : '\u25B6'}
            </button>
            <button
              onClick={() => dispatch({ type: 'STEP_FORWARD' })}
              disabled={state.currentStep >= maxStep}
            >
              &#9654;&#9654;
            </button>
          </div>
          {episode && (
            <input
              type="range"
              min={0}
              max={maxStep}
              value={state.currentStep}
              onChange={(e) => dispatch({ type: 'SEEK', step: Number(e.target.value) })}
              className="seek-slider"
            />
          )}
          <div className="controls-row">
            <label>Speed</label>
            <input
              type="range"
              min={1}
              max={20}
              value={state.playbackSpeed}
              onChange={(e) => dispatch({ type: 'SET_SPEED', speed: Number(e.target.value) })}
            />
            <span>{state.playbackSpeed} sps</span>
          </div>
        </div>
      )}

      {!li.active && currentStepData && (
        <div className="sidebar-section">
          <label>Step Info</label>
          <div className="step-info">
            <div>
              Step: {state.currentStep} / {maxStep}
            </div>
            <div>Action: {ACTION_NAMES[currentStepData.action]}</div>
            <div>Reward: {currentStepData.reward.toFixed(3)}</div>
            <div>
              Position: ({currentStepData.state.agentPosition[0]},{' '}
              {currentStepData.state.agentPosition[1]})
            </div>
            <div>Has Key: {currentStepData.state.inventory.hasKey ? 'Yes' : 'No'}</div>
            <div>Done: {currentStepData.state.done ? 'Yes' : 'No'}</div>
          </div>
        </div>
      )}

      {!li.active && currentStepData?.qValues && <QValuesPanel qValues={currentStepData.qValues} />}
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <util.AppProvider>
      <div style={{ width: '100vw', height: '100vh', display: 'flex' }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <render.Scene>
            <GameView />
          </render.Scene>
        </div>
        <Sidebar />
      </div>
    </util.AppProvider>
  </React.StrictMode>,
)
