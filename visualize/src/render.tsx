import * as React from 'react'
import * as Fiber from '@react-three/fiber'
import * as Drei from '@react-three/drei'
import * as THREE from 'three'
import * as t from './types'
import * as play from './play'

// Colors
const COLORS = {
  floor: '#888888',
  wall: '#444444',
  door: '#8B4513',
  key: '#FFD700',
  goal: '#00FF88',
  lava: '#FF4500',
  agent: '#00FFFF',
}

function cellColor(cellType: t.CellType): string {
  switch (cellType) {
    case t.CellType.Floor:
      return COLORS.floor
    case t.CellType.Wall:
      return COLORS.wall
    case t.CellType.Door:
      return COLORS.door
    case t.CellType.Key:
      return COLORS.key
    case t.CellType.Goal:
      return COLORS.goal
    case t.CellType.Lava:
      return COLORS.lava
    default:
      return COLORS.floor
  }
}

/** Grid floor tiles - rendered as flat planes for each cell */
export function Grid({
  level,
  onCellClick,
}: {
  level: t.Level
  onCellClick?: (x: number, y: number) => void
}) {
  const tiles = React.useMemo(() => {
    const result: { x: number; y: number; color: string }[] = []
    for (let y = 0; y < level.height; y++) {
      for (let x = 0; x < level.width; x++) {
        const cell = level.grid[y][x]
        if (cell !== t.CellType.Wall) {
          result.push({ x, y, color: cellColor(cell) })
        }
      }
    }
    return result
  }, [level])

  return (
    <group>
      {tiles.map((tile, i) => (
        <mesh
          key={i}
          position={[tile.x, 0, tile.y]}
          rotation={[-Math.PI / 2, 0, 0]}
          onClick={
            onCellClick
              ? (e) => {
                  e.stopPropagation()
                  onCellClick(tile.x, tile.y)
                }
              : undefined
          }
        >
          <planeGeometry args={[1, 1]} />
          <meshStandardMaterial color={tile.color} />
        </mesh>
      ))}
    </group>
  )
}

/** Wall blocks - rendered as unit cubes */
export function Walls({
  level,
  onCellClick,
}: {
  level: t.Level
  onCellClick?: (x: number, y: number) => void
}) {
  const walls = React.useMemo(() => play.getWallPositions(level), [level])

  return (
    <group>
      {walls.map(([x, y], i) => (
        <mesh
          key={i}
          position={[x, 0.5, y]}
          onClick={
            onCellClick
              ? (e) => {
                  e.stopPropagation()
                  onCellClick(x, y)
                }
              : undefined
          }
        >
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color={COLORS.wall} />
        </mesh>
      ))}
    </group>
  )
}

/** Keys, doors, goals, lava - rendered as distinctive shapes on top of floor */
export function Items({
  level,
  onCellClick,
}: {
  level: t.Level
  onCellClick?: (x: number, y: number) => void
}) {
  const items = React.useMemo(() => {
    const result: { x: number; y: number; type: t.CellType }[] = []
    for (let y = 0; y < level.height; y++) {
      for (let x = 0; x < level.width; x++) {
        const cell = level.grid[y][x]
        if (
          cell === t.CellType.Door ||
          cell === t.CellType.Key ||
          cell === t.CellType.Goal ||
          cell === t.CellType.Lava
        ) {
          result.push({ x, y, type: cell })
        }
      }
    }
    return result
  }, [level])

  const clickHandler = (x: number, y: number) =>
    onCellClick
      ? (e: { stopPropagation: () => void }) => {
          e.stopPropagation()
          onCellClick(x, y)
        }
      : undefined

  return (
    <group>
      {items.map((item, i) => {
        switch (item.type) {
          case t.CellType.Door:
            // Door: tall thin box
            return (
              <mesh key={i} position={[item.x, 0.4, item.y]} onClick={clickHandler(item.x, item.y)}>
                <boxGeometry args={[0.9, 0.8, 0.9]} />
                <meshStandardMaterial color={COLORS.door} />
              </mesh>
            )
          case t.CellType.Key:
            // Key: small floating octahedron
            return (
              <mesh
                key={i}
                position={[item.x, 0.35, item.y]}
                onClick={clickHandler(item.x, item.y)}
              >
                <octahedronGeometry args={[0.2]} />
                <meshStandardMaterial
                  color={COLORS.key}
                  emissive={COLORS.key}
                  emissiveIntensity={0.3}
                />
              </mesh>
            )
          case t.CellType.Goal:
            // Goal: glowing cylinder
            return (
              <mesh
                key={i}
                position={[item.x, 0.15, item.y]}
                onClick={clickHandler(item.x, item.y)}
              >
                <cylinderGeometry args={[0.3, 0.3, 0.3, 8]} />
                <meshStandardMaterial
                  color={COLORS.goal}
                  emissive={COLORS.goal}
                  emissiveIntensity={0.5}
                />
              </mesh>
            )
          case t.CellType.Lava:
            // Lava: flat glowing plane (already colored by Grid, add emissive overlay)
            return (
              <mesh
                key={i}
                position={[item.x, 0.01, item.y]}
                rotation={[-Math.PI / 2, 0, 0]}
                onClick={clickHandler(item.x, item.y)}
              >
                <planeGeometry args={[0.95, 0.95]} />
                <meshStandardMaterial
                  color={COLORS.lava}
                  emissive={COLORS.lava}
                  emissiveIntensity={0.4}
                />
              </mesh>
            )
          default:
            return null
        }
      })}
    </group>
  )
}

/** Agent sphere with lerp animation between positions */
export function Agent({
  position,
  prevPosition,
}: {
  position: [number, number]
  prevPosition?: [number, number]
}) {
  const meshRef = React.useRef<THREE.Mesh>(null!)
  const startPos = React.useRef<THREE.Vector3>(new THREE.Vector3(position[0], 0.4, position[1]))
  const targetPos = React.useRef<THREE.Vector3>(new THREE.Vector3(position[0], 0.4, position[1]))
  const progress = React.useRef(1)

  React.useEffect(() => {
    if (prevPosition) {
      startPos.current.set(prevPosition[0], 0.4, prevPosition[1])
    } else {
      startPos.current.copy(targetPos.current)
    }
    targetPos.current.set(position[0], 0.4, position[1])
    progress.current = 0
  }, [position, prevPosition])

  Fiber.useFrame((_, delta) => {
    if (!meshRef.current) return
    if (progress.current < 1) {
      progress.current = Math.min(1, progress.current + delta * 8)
      const t = progress.current
      // Smooth step interpolation
      const smooth = t * t * (3 - 2 * t)
      meshRef.current.position.lerpVectors(startPos.current, targetPos.current, smooth)
    }
  })

  return (
    <mesh ref={meshRef} position={[position[0], 0.4, position[1]]}>
      <sphereGeometry args={[0.3, 16, 16]} />
      <meshStandardMaterial color={COLORS.agent} emissive={COLORS.agent} emissiveIntensity={0.3} />
    </mesh>
  )
}

/** Q-value direction arrows overlay */
export function QValueArrows({
  qValues,
  position,
}: {
  qValues?: t.QValues
  position: [number, number]
}) {
  if (!qValues) return null

  const directions: { action: t.Action; dx: number; dz: number; rotation: number }[] = [
    { action: t.Action.Up, dx: 0, dz: -0.4, rotation: 0 },
    { action: t.Action.Down, dx: 0, dz: 0.4, rotation: Math.PI },
    { action: t.Action.Left, dx: -0.4, dz: 0, rotation: Math.PI / 2 },
    { action: t.Action.Right, dx: 0.4, dz: 0, rotation: -Math.PI / 2 },
  ]

  // Normalize Q-values to [0, 1] range for opacity
  const values = directions.map((d) => qValues[d.action])
  const minQ = Math.min(...values)
  const maxQ = Math.max(...values)
  const range = maxQ - minQ || 1

  return (
    <group position={[position[0], 0.02, position[1]]}>
      {directions.map((dir) => {
        const normalized = (qValues[dir.action] - minQ) / range
        return (
          <mesh
            key={dir.action}
            position={[dir.dx, 0, dir.dz]}
            rotation={[-Math.PI / 2, 0, dir.rotation]}
          >
            <coneGeometry args={[0.12, 0.3, 4]} />
            <meshStandardMaterial color="#FFFF00" transparent opacity={0.3 + normalized * 0.7} />
          </mesh>
        )
      })}
    </group>
  )
}

/** Main scene wrapper with Canvas, lights, and controls */
export function Scene({ children }: { children: React.ReactNode }) {
  return (
    <Fiber.Canvas camera={{ position: [5, 10, 10], fov: 50 }}>
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />
      <Drei.OrbitControls />
      {children}
    </Fiber.Canvas>
  )
}
