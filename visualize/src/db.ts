import { Database } from 'bun:sqlite'
import fs from 'fs'
import path from 'path'
import * as t from './types'

export class DB {
  private db: Database

  constructor(dbPath: string) {
    this.db = new Database(dbPath, { create: true })
    this.db.exec('PRAGMA journal_mode = WAL')
  }

  initialize() {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        training_steps INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL REFERENCES agents(id),
        level_id TEXT NOT NULL,
        total_reward REAL NOT NULL,
        run_number INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_id TEXT NOT NULL REFERENCES episodes(id),
        step_number INTEGER NOT NULL,
        state_json TEXT NOT NULL,
        action INTEGER NOT NULL,
        reward REAL NOT NULL,
        q_values_json TEXT,
        done INTEGER NOT NULL DEFAULT 0
      );
      CREATE TABLE IF NOT EXISTS checkpoints (
        id TEXT PRIMARY KEY,
        agent_id TEXT REFERENCES agents(id),
        training_steps INTEGER NOT NULL,
        onnx_path TEXT
      );
    `)
  }

  getCheckpoints(): t.CheckpointInfo[] {
    const rows = this.db
      .prepare('SELECT id, training_steps FROM checkpoints ORDER BY training_steps')
      .all() as { id: string; training_steps: number }[]
    return rows.map((r) => ({ id: r.id, trainingSteps: r.training_steps }))
  }

  getLevels(levelsDir: string): t.LevelInfo[] {
    if (!fs.existsSync(levelsDir)) return []

    const files = fs.readdirSync(levelsDir).filter((f) => f.endsWith('.json'))
    return files.map((f) => {
      const raw = fs.readFileSync(path.join(levelsDir, f), 'utf-8')
      const json = JSON.parse(raw)
      const info: t.LevelInfo = { id: json.id, name: json.name }
      if (json.seed !== undefined) info.seed = json.seed
      return info
    })
  }

  getLevelWithEpisodes(levelId: string, levelsDir: string): t.LevelWithEpisodes | null {
    const levelPath = path.join(levelsDir, `${levelId}.json`)
    if (!fs.existsSync(levelPath)) return null

    const raw = fs.readFileSync(levelPath, 'utf-8')
    const json = JSON.parse(raw)
    const level: t.Level = {
      id: json.id,
      name: json.name,
      width: json.width,
      height: json.height,
      grid: json.grid as t.CellType[][],
      agentStart: json.agentStart as [number, number],
      seed: json.seed,
    }

    const episodeRows = this.db
      .prepare(
        'SELECT id, agent_id, level_id, total_reward FROM episodes WHERE level_id = ? ORDER BY run_number',
      )
      .all(levelId) as {
      id: string
      agent_id: string
      level_id: string
      total_reward: number
    }[]

    const episodes: t.Episode[] = episodeRows.map((ep) => {
      const stepRows = this.db
        .prepare(
          'SELECT state_json, action, reward, q_values_json FROM steps WHERE episode_id = ? ORDER BY step_number',
        )
        .all(ep.id) as {
        state_json: string
        action: number
        reward: number
        q_values_json: string | null
      }[]

      const steps: t.Step[] = stepRows.map((s) => ({
        state: JSON.parse(s.state_json) as t.GameState,
        action: s.action as t.Action,
        reward: s.reward,
        qValues: s.q_values_json ? (JSON.parse(s.q_values_json) as t.QValues) : undefined,
      }))

      return {
        id: ep.id,
        agentId: ep.agent_id,
        levelId: ep.level_id,
        steps,
        totalReward: ep.total_reward,
      }
    })

    return { level, episodes }
  }
}
