import * as React from 'react'
import * as Fiber from '@react-three/fiber'
import * as Drei from '@react-three/drei'
import * as THREE from 'three'
import * as t from './types'
import * as play from './play'
import GLSLShader from './shader'

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

/** Stable key for grid content — changes when any cell changes */
function gridKey(grid: t.CellType[][]): string {
  return grid.map((row) => row.join(',')).join(';')
}

/** Grid floor tiles - rendered as flat planes for each cell */
export function Grid({
  level,
  onCellClick,
}: {
  level: t.Level
  onCellClick?: (x: number, y: number) => void
}) {
  const gk = gridKey(level.grid)
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
  }, [gk])

  return (
    <group>
      {tiles.map((tile) => (
        <mesh
          key={`${tile.x},${tile.y}`}
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
  const wk = gridKey(level.grid)
  const walls = React.useMemo(() => play.getWallPositions(level), [wk])

  return (
    <group>
      {walls.map(([x, y]) => (
        <mesh
          key={`${x},${y}`}
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
  const gk = gridKey(level.grid)
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
  }, [gk])

  const clickHandler = (x: number, y: number) =>
    onCellClick
      ? (e: { stopPropagation: () => void }) => {
          e.stopPropagation()
          onCellClick(x, y)
        }
      : undefined

  return (
    <group>
      {items.map((item) => {
        const key = `${item.x},${item.y}`
        switch (item.type) {
          case t.CellType.Door:
            // Door: tall thin box
            return (
              <mesh
                key={key}
                position={[item.x, 0.4, item.y]}
                onClick={clickHandler(item.x, item.y)}
              >
                <boxGeometry args={[0.9, 0.8, 0.9]} />
                <meshStandardMaterial color={COLORS.door} />
              </mesh>
            )
          case t.CellType.Key:
            // Key: small floating octahedron
            return (
              <mesh
                key={key}
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
                key={key}
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
                key={key}
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

/** Agent model with linear lerp animation and facing direction */
// Rotation offset if the model's "forward" doesn't align with -Z
const AGENT_ROTATION_OFFSET = 0

export function Agent({
  position,
  prevPosition,
  stepsPerSecond = 4,
}: {
  position: [number, number]
  prevPosition?: [number, number]
  stepsPerSecond?: number
}) {
  const groupRef = React.useRef<THREE.Group>(null!)
  const { scene } = Drei.useGLTF('/static/models/player.glb')
  const clonedScene = React.useMemo(() => scene.clone(), [scene])
  const startPos = React.useRef<THREE.Vector3>(new THREE.Vector3(position[0], 0, position[1]))
  const targetPos = React.useRef<THREE.Vector3>(new THREE.Vector3(position[0], 0, position[1]))
  const progress = React.useRef(1)
  const facingAngle = React.useRef(0)

  React.useEffect(() => {
    if (prevPosition) {
      startPos.current.set(prevPosition[0], 0, prevPosition[1])
    } else {
      startPos.current.copy(targetPos.current)
    }
    targetPos.current.set(position[0], 0, position[1])
    progress.current = 0

    // Compute facing direction from movement delta
    const dx = targetPos.current.x - startPos.current.x
    const dz = targetPos.current.z - startPos.current.z
    if (dx !== 0 || dz !== 0) {
      facingAngle.current = Math.atan2(dx, dz) + AGENT_ROTATION_OFFSET
    }
  }, [position, prevPosition])

  Fiber.useFrame((_, delta) => {
    if (!groupRef.current) return
    if (progress.current < 1) {
      // Lerp completes in exactly 1/stepsPerSecond seconds
      progress.current = Math.min(1, progress.current + delta * stepsPerSecond)
      groupRef.current.position.lerpVectors(startPos.current, targetPos.current, progress.current)
    }
    groupRef.current.rotation.y = facingAngle.current
  })

  return (
    <group ref={groupRef} position={[position[0], 0, position[1]]}>
      <primitive object={clonedScene} />
    </group>
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

const postProcessor = `
  float period_secs = 45.0;
  float band_size = 0.003;
  float band = mod(uv.y, band_size);
  float direction = band < (band_size / 2.0) ? 1.0 : -1.2;

  vec3 original = scene(uv);
  if (original.x == 0.0 && original.y == 0.0 && original.z == 0.0) {
    // uv = (uv + triangle(floor_to_nearest(uv.y, band_size / 4.0)) * 6.28) * 3.0;
    uv = uv * 3.0;
    color = vec3(
      scene(vec2(direction * t / period_secs, 0.0) + uv + voronoi_noise(t / 4.0 + uv *  10.0) * 0.008).x,
      scene(vec2(direction * t / period_secs, 0.0) + uv - voronoi_noise(t / 4.0 + uv *  10.0) * 0.008).y,
      scene(vec2(direction * t / period_secs, 0.0) + uv - voronoi_noise(t / 4.0 + uv * 100.0) * 0.005).z
    ) * 0.05;
  } else {
    color = vec3(
      scene(uv + voronoi_noise(t / 4.0 + uv * 10.0) * 0.008).x,
      scene(uv - voronoi_noise(t / 4.0 + uv * 10.0) * 0.004).y,
      scene(uv - voronoi_noise(t / 4.0 + uv * 100.0) * 0.005).z
    );
  }
`

function CameraController({ target }: { target: [number, number, number] }) {
  const controlsRef = React.useRef<any>(null)
  const { camera } = Fiber.useThree()
  const initialized = React.useRef(false)

  React.useEffect(() => {
    if (!controlsRef.current) return
    controlsRef.current.target.set(...target)
    controlsRef.current.update()
    if (!initialized.current) {
      camera.position.set(target[0], 10, target[2] + 6)
      initialized.current = true
    }
  }, [target[0], target[1], target[2], camera])

  return <Drei.OrbitControls ref={controlsRef} />
}

export function Scene({
  children,
  target,
}: {
  children: React.ReactNode
  target?: [number, number, number]
}) {
  const center = target ?? [4.5, 0, 4.5]
  return (
    <Fiber.Canvas
      camera={{ position: [center[0], 10, center[2] + 6], fov: 50 }}
      gl={{ preserveDrawingBuffer: true }}
    >
      <GLSLShader code={postProcessor}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={0.8} />
        <CameraController target={center} />
        {children}
      </GLSLShader>
    </Fiber.Canvas>
  )
}
