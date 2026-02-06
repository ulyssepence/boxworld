import * as React from 'react'
import * as t from './types'
import * as api from './api'
import * as util from './util'
import * as render from './render'
import * as shader from './shader'
import * as ml from './ml'
import * as play from './play'

function GameView() {
  const [state, dispatch] = util.useApp()
  util.usePlayback()

  const level = state.currentLevel
  if (!level) return null

  const episode = state.episodes[state.currentEpisodeIndex]
  const currentStepData = episode?.steps[state.currentStep]
  const prevStepData = state.currentStep > 0 ? episode?.steps[state.currentStep - 1] : undefined

  // Determine agent position: from episode step state, or level start
  const agentPos: [number, number] = currentStepData
    ? currentStepData.state.agentPosition
    : level.agentStart

  const prevAgentPos: [number, number] | undefined = prevStepData
    ? prevStepData.state.agentPosition
    : undefined

  // Use the step's game state grid if available (doors may have been toggled, keys picked up)
  const displayLevel: t.Level = currentStepData ? currentStepData.state.level : level

  const handleCellClick = state.editMode
    ? (x: number, y: number) => {
        const currentCell = displayLevel.grid[y][x]
        // Cycle: Floor -> Wall -> Goal -> Floor
        let nextType: t.CellType
        switch (currentCell) {
          case t.CellType.Floor:
            nextType = t.CellType.Wall
            break
          case t.CellType.Wall:
            nextType = t.CellType.Goal
            break
          case t.CellType.Goal:
            nextType = t.CellType.Floor
            break
          default:
            nextType = t.CellType.Floor
            break
        }
        dispatch({ type: 'EDIT_CELL', x, y, cellType: nextType })
      }
    : undefined

  return (
    <shader.GLSLShader code="color = scene(uv);">
      <render.Grid level={displayLevel} onCellClick={handleCellClick} />
      <render.Walls level={displayLevel} onCellClick={handleCellClick} />
      <render.Items level={displayLevel} onCellClick={handleCellClick} />
      <render.Agent position={agentPos} prevPosition={prevAgentPos} />
      <render.QValueArrows qValues={currentStepData?.qValues} position={agentPos} />
    </shader.GLSLShader>
  )
}

async function runAgent(
  level: t.Level,
  checkpointUrl: string,
  dispatch: React.Dispatch<util.AppAction>,
) {
  dispatch({ type: 'SET_INFERENCE_LOADING', loading: true })

  try {
    const agent = new ml.Agent()
    await agent.load(checkpointUrl)

    let gameState = play.createInitialState(level)
    const steps: t.Step[] = []

    for (let i = 0; i < 200 && !gameState.done; i++) {
      const { action, qValues } = await agent.selectAction(gameState)
      const prevState = gameState
      gameState = play.step(gameState, action)
      steps.push({
        state: prevState,
        action,
        reward: gameState.reward - prevState.reward,
        qValues,
      })
    }

    const episode: t.Episode = {
      id: 'live-inference',
      agentId: 'onnx',
      levelId: level.id,
      steps,
      totalReward: gameState.reward,
    }

    dispatch({ type: 'LOAD_INFERENCE_EPISODE', episode })
  } catch (err) {
    console.error('Inference failed:', err)
  } finally {
    dispatch({ type: 'SET_INFERENCE_LOADING', loading: false })
  }
}

function Sidebar() {
  const [state, dispatch] = util.useApp()
  const [selectedCheckpointId, setSelectedCheckpointId] = React.useState<string>('')

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
      dispatch({ type: 'LOAD_LEVEL', level: data.level, episodes: data.episodes })
    })
  }

  const actionName = (action: t.Action): string => {
    switch (action) {
      case t.Action.Up:
        return 'Up'
      case t.Action.Down:
        return 'Down'
      case t.Action.Left:
        return 'Left'
      case t.Action.Right:
        return 'Right'
      case t.Action.Pickup:
        return 'Pickup'
      case t.Action.Toggle:
        return 'Toggle'
      default:
        return '?'
    }
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

      {state.currentLevel && (
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

      {state.currentLevel && state.episodes.length > 0 && (
        <div className="sidebar-section">
          <label>Episode</label>
          <select
            value={state.currentEpisodeIndex}
            onChange={(e) => dispatch({ type: 'SET_EPISODE', index: Number(e.target.value) })}
          >
            {state.episodes.map((ep, i) => (
              <option key={ep.id} value={i}>
                Episode {i + 1} (reward: {ep.totalReward.toFixed(2)})
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
          >
            {state.checkpoints.map((cp) => (
              <option key={cp.id} value={cp.id}>
                {cp.trainingSteps.toLocaleString()} steps
              </option>
            ))}
          </select>
          <button
            disabled={state.inferenceLoading || !selectedCheckpointId}
            onClick={() => {
              const cp = state.checkpoints.find((c) => c.id === selectedCheckpointId)
              if (!cp || !state.currentLevel) return
              const url = `/checkpoints/boxworld_${cp.trainingSteps}_steps.onnx`
              runAgent(state.currentLevel, url, dispatch)
            }}
          >
            {state.inferenceLoading ? 'Running...' : 'Run Agent'}
          </button>
        </div>
      )}

      {state.currentLevel && (
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

      {currentStepData && (
        <div className="sidebar-section">
          <label>Step Info</label>
          <div className="step-info">
            <div>
              Step: {state.currentStep} / {maxStep}
            </div>
            <div>Action: {actionName(currentStepData.action)}</div>
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

      {currentStepData?.qValues && (
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
                {name}: {currentStepData.qValues![action].toFixed(3)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export function App() {
  return (
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
  )
}
