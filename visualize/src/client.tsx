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
      return t.CellType.Door
    case t.CellType.Door:
      return t.CellType.Key
    case t.CellType.Key:
      return t.CellType.Goal
    case t.CellType.Goal:
      return t.CellType.Lava
    case t.CellType.Lava:
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

  if (li.active || state.viewMode === 'inference') {
    // Inference mode: display currentLevel (shared mutable grid), agent from gameState
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

  // Replay mode
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
    <div className="overlay-panel">
      <div className="panel-label">Q-Values</div>
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

function StepInfoPanel({
  stepLabel,
  action,
  reward,
  position,
  hasKey,
  done,
}: {
  stepLabel: string
  action: string | null
  reward: string
  position: [number, number]
  hasKey: boolean
  done: boolean
}) {
  return (
    <div className="overlay-panel">
      <div className="panel-label">Step Info</div>
      <div className="step-info">
        <div>Step: {stepLabel}</div>
        {action && <div>Action: {action}</div>}
        <div>Reward: {reward}</div>
        <div>
          Position: ({position[0]}, {position[1]})
        </div>
        <div>Has Key: {hasKey ? 'Yes' : 'No'}</div>
        <div>Done: {done ? 'Yes' : 'No'}</div>
      </div>
    </div>
  )
}

function SpeedControls() {
  const [state, dispatch] = util.useApp()
  return (
    <>
      <button
        className="speed-btn"
        onClick={() => dispatch({ type: 'SET_SPEED', speed: Math.max(1, state.playbackSpeed - 1) })}
        disabled={state.playbackSpeed <= 1}
      >
        -
      </button>
      <span className="speed-label">{state.playbackSpeed}×</span>
      <button
        className="speed-btn"
        onClick={() =>
          dispatch({ type: 'SET_SPEED', speed: Math.min(20, state.playbackSpeed + 1) })
        }
        disabled={state.playbackSpeed >= 20}
      >
        +
      </button>
    </>
  )
}

function RecordingsTab() {
  const [state, dispatch] = util.useApp()

  const episode = state.episodes[state.currentEpisodeIndex]
  const maxStep = episode ? Math.max(0, episode.steps.length - 1) : 0

  return (
    <div className="tab-content">
      {state.episodes.length > 0 && (
        <select
          className="overlay-select"
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
      )}
      <button onClick={() => dispatch({ type: 'STEP_BACKWARD' })} disabled={state.currentStep <= 0}>
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
      <SpeedControls />
    </div>
  )
}

function InferenceTab({
  agentRef,
  stepOnce,
}: {
  agentRef: React.MutableRefObject<ml.Agent | null>
  stepOnce: () => Promise<void>
}) {
  const [state, dispatch] = util.useApp()
  const [selectedCheckpointId, setSelectedCheckpointId] = React.useState<string>('')

  const li = state.liveInference

  // Auto-select best checkpoint
  React.useEffect(() => {
    if (state.checkpoints.length > 0 && !selectedCheckpointId) {
      setSelectedCheckpointId(state.checkpoints[state.checkpoints.length - 1].id)
    }
  }, [state.checkpoints, selectedCheckpointId])

  // Auto-restart inference on cell edit
  React.useEffect(() => {
    if (state.editVersion === 0) return
    if (state.viewMode !== 'inference' || !agentRef.current) return
    dispatch({ type: 'RESTART_LIVE_INFERENCE' })
  }, [state.editVersion])

  return (
    <div className="tab-content">
      {state.checkpoints.length > 0 && (
        <>
          <select
            className="overlay-select"
            value={selectedCheckpointId}
            onChange={(e) => setSelectedCheckpointId(e.target.value)}
          >
            {state.checkpoints.map((cp) => (
              <option key={cp.id} value={cp.id}>
                {cp.trainingSteps.toLocaleString()} steps
              </option>
            ))}
          </select>
          {!li.active ? (
            <button
              disabled={state.inferenceLoading || !selectedCheckpointId}
              onClick={() => {
                const cp = state.checkpoints.find((c) => c.id === selectedCheckpointId)
                if (!cp || !state.currentLevel) return
                const url = `/checkpoints/boxworld_${cp.trainingSteps}_steps.onnx`
                startLiveAgent(url, agentRef, dispatch)
              }}
            >
              {state.inferenceLoading ? 'Loading...' : 'Run'}
            </button>
          ) : (
            <button
              disabled={state.inferenceLoading || !selectedCheckpointId}
              onClick={() => {
                const cp = state.checkpoints.find((c) => c.id === selectedCheckpointId)
                if (!cp || !state.currentLevel) return
                const url = `/checkpoints/boxworld_${cp.trainingSteps}_steps.onnx`
                startLiveAgent(url, agentRef, dispatch)
              }}
            >
              Restart
            </button>
          )}
        </>
      )}
      {li.active && (
        <>
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
          <SpeedControls />
        </>
      )}
    </div>
  )
}

function AboutOverlay({ onClose }: { onClose: () => void }) {
  return (
    <div className="about-overlay" onClick={onClose}>
      <div className="about-panel" onClick={(e) => e.stopPropagation()}>
        <button className="about-close" onClick={onClose}>
          &times;
        </button>
        <h2>
          Boxworld (
          <a target="_blank" href="https://github.com/ulyssepence/boxworld">
            source
          </a>
          )
        </h2>
        <p>
          An AI agent has learned to navigate grid-based world with walls, doors, keys, lava, and
          goals. Trained with the Reinforcement Learning using{' '}
          <a target="_blank" href="https://en.wikipedia.org/wiki/Proximal_policy_optimization">
            PPO (Proximal Policy Optimization)
          </a>{' '}
          and{' '}
          <a target="_blank" href="https://stable-baselines3.readthedocs.io/en/master/">
            Stable-Baselines3
          </a>
          .
        </p>
        <p>
          Use the <strong>Recordings</strong> tab to play back recordings of the agent after
          differing number of steps/iterations of the training. Watch it stumbling around in the
          environment and slowly figuring out how to take actions that lead to better{' '}
          <strong>rewards</strong>.
        </p>
        <p>
          Use the <strong>Live</strong> tab to run the agent live in your browser after it's gone
          through different amounts of training steps. Edit the level to see how they overcome (or
          often don't) new patterns in the environment. The live <strong>inference</strong> runs the{' '}
          <a target="_blank" href="https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange">
            ONNX
          </a>{' '}
          in the 3D environment created with React Three Fiber.
        </p>
      </div>
    </div>
  )
}

function Overlay() {
  const [state, dispatch] = util.useApp()
  const { agentRef, stepOnce } = util.useLiveInference()
  const [showAbout, setShowAbout] = React.useState(false)

  const li = state.liveInference

  // Space key toggles play/pause
  React.useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault()
        if (state.viewMode === 'inference') {
          dispatch({ type: li.running ? 'LIVE_PAUSE' : 'LIVE_PLAY' })
        } else {
          dispatch({ type: state.isPlaying ? 'PAUSE' : 'PLAY' })
        }
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [state.viewMode, state.isPlaying, li.running, dispatch])

  // Fetch levels on mount, auto-load the first one
  React.useEffect(() => {
    api.fetchLevels().then((levels) => {
      dispatch({ type: 'LOAD_LEVELS', levels })
      if (levels.length > 0) {
        api.fetchLevel(levels[0].id).then((data) => {
          if (data.level) {
            dispatch({ type: 'LOAD_LEVEL', level: data.level, episodes: data.episodes })
          }
        })
      }
    })
    api.fetchCheckpoints().then((checkpoints) => {
      dispatch({ type: 'LOAD_CHECKPOINTS', checkpoints })
    })
  }, [dispatch])

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

  // Compute step info for bottom-right panel
  const episode = state.episodes[state.currentEpisodeIndex]
  const currentStepData = episode?.steps[state.currentStep]
  const maxStep = episode ? Math.max(0, episode.steps.length - 1) : 0

  const showRecordingInfo = state.viewMode === 'recordings' && currentStepData
  const showInferenceInfo = state.viewMode === 'inference' && li.active && li.gameState

  return (
    <>
      {/* Top overlay — single horizontal bar */}
      <div className="overlay-top">
        <span className="title">Boxworld</span>
        <button className="about-btn" onClick={() => setShowAbout(true)}>
          ?
        </button>
        <select
          className="overlay-select"
          onChange={handleLevelChange}
          value={state.currentLevel?.id ?? ''}
        >
          {!state.currentLevel && (
            <option value="" disabled>
              Select a level...
            </option>
          )}
          {state.levels.map((l) => (
            <option key={l.id} value={l.id}>
              {l.name}
            </option>
          ))}
        </select>
        {state.currentLevel && (
          <>
            <button
              onClick={() => {
                dispatch({ type: 'TOGGLE_EDIT_MODE' })
                if (!state.editMode) {
                  dispatch({ type: 'SET_VIEW_MODE', mode: 'inference' })
                }
              }}
            >
              {state.editMode ? 'Exit Edit' : 'Edit Level'}
            </button>
            {state.editMode && (
              <button onClick={() => dispatch({ type: 'RESET_LEVEL' })}>Reset</button>
            )}
            <div className="overlay-divider" />
            <div className="tab-strip">
              <button
                className={`tab ${state.viewMode === 'recordings' ? 'tab-active' : ''}`}
                onClick={() => dispatch({ type: 'SET_VIEW_MODE', mode: 'recordings' })}
              >
                Recordings
              </button>
              <button
                className={`tab ${state.viewMode === 'inference' ? 'tab-active' : ''}`}
                onClick={() => dispatch({ type: 'SET_VIEW_MODE', mode: 'inference' })}
              >
                Live
              </button>
            </div>
            <div className="overlay-divider" />
            {state.viewMode === 'recordings' && <RecordingsTab />}
            {state.viewMode === 'inference' && (
              <InferenceTab agentRef={agentRef} stepOnce={stepOnce} />
            )}
          </>
        )}
      </div>

      {/* Bottom-right overlay */}
      <div className="overlay-bottom-right">
        {showRecordingInfo && (
          <>
            <StepInfoPanel
              stepLabel={`${state.currentStep} / ${maxStep}`}
              action={ACTION_NAMES[currentStepData.action]}
              reward={currentStepData.reward.toFixed(3)}
              position={currentStepData.state.agentPosition}
              hasKey={currentStepData.state.inventory.hasKey}
              done={currentStepData.state.done}
            />
            {currentStepData.qValues && <QValuesPanel qValues={currentStepData.qValues} />}
          </>
        )}
        {showInferenceInfo && (
          <>
            <StepInfoPanel
              stepLabel={`${li.stepCount} / 200`}
              action={li.lastAction !== null ? ACTION_NAMES[li.lastAction] : null}
              reward={li.gameState!.reward.toFixed(3)}
              position={li.gameState!.agentPosition}
              hasKey={li.gameState!.inventory.hasKey}
              done={li.gameState!.done}
            />
            {li.lastQValues && <QValuesPanel qValues={li.lastQValues} />}
          </>
        )}
      </div>
      {showAbout && <AboutOverlay onClose={() => setShowAbout(false)} />}
    </>
  )
}

function Root() {
  const [state] = util.useApp()
  const level = state.currentLevel
  const target: [number, number, number] | undefined = level
    ? [(level.width - 1) / 2, 0, (level.height - 1) / 2]
    : undefined

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      <render.Scene target={target}>
        <GameView />
      </render.Scene>
      <Overlay />
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <util.AppProvider>
      <Root />
    </util.AppProvider>
  </React.StrictMode>,
)
