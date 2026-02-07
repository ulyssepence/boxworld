import { describe, test, expect } from 'bun:test'
import { parseLevel } from '../src/level-parser'
import { CellType } from '../src/types'

describe('parseLevel', () => {
  test('parse simple level', () => {
    const text = '####\n#A #\n# G#\n####'
    const result = parseLevel(text, 'test')
    expect(result.width).toBe(4)
    expect(result.height).toBe(4)
    expect(result.agentStart).toEqual([1, 1])
    expect(result.grid[2][2]).toBe(CellType.Goal)
  })

  test('all cell type mappings', () => {
    const text = '#A DKG~#'
    const result = parseLevel(text, 'types')
    const row = result.grid[0]
    expect(row[0]).toBe(CellType.Wall)
    expect(row[1]).toBe(CellType.Floor) // Agent → Floor
    expect(row[2]).toBe(CellType.Floor) // Space → Floor
    expect(row[3]).toBe(CellType.Door)
    expect(row[4]).toBe(CellType.Key)
    expect(row[5]).toBe(CellType.Goal)
    expect(row[6]).toBe(CellType.Lava)
    expect(row[7]).toBe(CellType.Wall)
  })

  test('agent replaced with floor', () => {
    const text = '###\n#A#\n###'
    const result = parseLevel(text, 'test')
    expect(result.grid[1][1]).toBe(CellType.Floor)
    expect(result.agentStart).toEqual([1, 1])
  })

  test('missing agent throws', () => {
    const text = '###\n# #\n###'
    expect(() => parseLevel(text, 'test')).toThrow('No agent')
  })

  test('id and name derived from filename', () => {
    const text = '#A#'
    const result = parseLevel(text, 'my_cool_level')
    expect(result.id).toBe('my_cool_level')
    expect(result.name).toBe('My Cool Level')
  })

  test('ragged rows throws', () => {
    const text = '####\n#A#\n####'
    expect(() => parseLevel(text, 'test')).toThrow('ragged')
  })
})
