import * as t from './types'

export async function fetchCheckpoints(): Promise<t.CheckpointInfo[]> {
  const res = await fetch('/api/checkpoints')
  const data = await res.json()
  return data.checkpoints
}

export async function fetchLevels(): Promise<t.LevelInfo[]> {
  const res = await fetch('/api/levels')
  const data = await res.json()
  return data.levels
}

export async function fetchLevel(id: string): Promise<t.LevelWithEpisodes> {
  const res = await fetch(`/api/levels/${id}`)
  return res.json()
}

export async function fetchCuratedSeeds(): Promise<number[]> {
  const res = await fetch('/api/curated-seeds')
  const data = await res.json()
  return data.seeds
}
