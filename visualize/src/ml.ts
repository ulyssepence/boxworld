/**
 * ONNX Runtime Web inference module.
 *
 * Loads exported DQN checkpoints and runs in-browser inference
 * to produce Q-values and select actions for the Boxworld agent.
 *
 * onnxruntime-web is loaded via a <script> tag in index.html,
 * which exposes the `ort` global. This avoids WASM bundling
 * issues with esbuild.
 */
import * as t from './types'

/** Access the ort global set by the onnxruntime-web script tag */
function getOrt(): typeof import('onnxruntime-web') {
  const ort = (globalThis as any).ort
  if (!ort) {
    throw new Error(
      'onnxruntime-web is not loaded. Ensure ort.min.js is included via a <script> tag.',
    )
  }
  return ort
}

/**
 * Convert a GameState to the flat Float32Array observation that the
 * Python environment produces.
 *
 * Encoding (must match environment.py `_get_obs()` exactly):
 *   [grid[0][0], grid[0][1], ..., grid[0][w-1],
 *    grid[1][0], ..., grid[h-1][w-1],
 *    agent_x, agent_y, has_key]
 *
 * - Grid is row-major: grid[y][x]
 * - Cell types: Floor=0, Wall=1, Door=2, Key=3, Goal=4, Lava=5
 * - agent_x, agent_y: float coordinates
 * - has_key: 1.0 or 0.0
 * - dtype: Float32
 */
export function stateToTensor(state: t.GameState): Float32Array {
  const { level, agentPosition, inventory } = state
  const { width, height, grid } = level
  const size = width * height + 3
  const tensor = new Float32Array(size)

  let idx = 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      tensor[idx++] = grid[y][x]
    }
  }

  tensor[idx++] = agentPosition[0] // agent_x
  tensor[idx++] = agentPosition[1] // agent_y
  tensor[idx++] = inventory.hasKey ? 1.0 : 0.0

  return tensor
}

/**
 * ONNX inference agent that loads an exported DQN model
 * and runs forward passes to produce Q-values.
 */
export class Agent {
  private session: any = null

  /**
   * Load an ONNX model from the given URL.
   * The URL should point to a .onnx file served by the backend
   * (e.g. `/checkpoints/boxworld_10000_steps.onnx`).
   */
  async load(modelUrl: string): Promise<void> {
    const ort = getOrt()
    this.session = await ort.InferenceSession.create(modelUrl)
  }

  /**
   * Run inference on the given game state and return Q-values
   * for all six actions.
   */
  async getQValues(state: t.GameState): Promise<t.QValues> {
    if (!this.session) {
      throw new Error('Model not loaded. Call load() first.')
    }

    const ort = getOrt()
    const tensor = stateToTensor(state)
    const ortTensor = new ort.Tensor('float32', tensor, [1, tensor.length])
    const results = await this.session.run({ obs: ortTensor })
    const output = results.q_values.data as Float32Array

    return {
      [t.Action.Up]: output[0],
      [t.Action.Down]: output[1],
      [t.Action.Left]: output[2],
      [t.Action.Right]: output[3],
      [t.Action.Pickup]: output[4],
      [t.Action.Toggle]: output[5],
    }
  }

  /**
   * Get Q-values and select the action with the highest Q-value (argmax).
   */
  async selectAction(state: t.GameState): Promise<{ action: t.Action; qValues: t.QValues }> {
    const qValues = await this.getQValues(state)

    let bestAction = t.Action.Up
    let bestValue = -Infinity

    for (let a = 0; a <= 5; a++) {
      const v = qValues[a as t.Action]
      if (v > bestValue) {
        bestValue = v
        bestAction = a as t.Action
      }
    }

    return { action: bestAction, qValues }
  }
}
