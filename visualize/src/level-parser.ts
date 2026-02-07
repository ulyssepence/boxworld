import { CellType, Level } from './types'

const CHAR_TO_CELL: Record<string, CellType> = {
  '#': CellType.Wall,
  ' ': CellType.Floor,
  D: CellType.Door,
  K: CellType.Key,
  G: CellType.Goal,
  '~': CellType.Lava,
  A: CellType.Floor, // Agent start treated as floor
}

export function parseLevel(text: string, levelId: string): Level {
  const lines = text.replace(/\n$/, '').split('\n')
  if (lines.length === 0) throw new Error('Empty level text')

  const height = lines.length
  const width = lines[0].length

  let agentStart: [number, number] | null = null
  const grid: CellType[][] = []

  for (let y = 0; y < height; y++) {
    const line = lines[y]
    if (line.length !== width) {
      throw new Error(`Row ${y} has length ${line.length}, expected ${width} (ragged rows)`)
    }
    const row: CellType[] = []
    for (let x = 0; x < width; x++) {
      const ch = line[x]
      if (!(ch in CHAR_TO_CELL)) {
        throw new Error(`Unknown character '${ch}' at (${x}, ${y})`)
      }
      row.push(CHAR_TO_CELL[ch])
      if (ch === 'A') {
        agentStart = [x, y]
      }
    }
    grid.push(row)
  }

  if (agentStart === null) {
    throw new Error("No agent start position ('A') found in level")
  }

  const name = levelId
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')

  return { id: levelId, name, width, height, grid, agentStart }
}
