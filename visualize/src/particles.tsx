import * as React from 'react'
import * as Fiber from '@react-three/fiber'
import * as THREE from 'three'
import { GLSL_UTILITIES } from './shader'

// --- Types ---

export interface ParticlesProps {
  /** Number of particles */
  count?: number
  /** Emission origin in world coords */
  origin?: [number, number, number]
  /** Random spread around origin per axis */
  spread?: [number, number, number]
  /** Base initial velocity */
  velocity?: [number, number, number]
  /** Random spread on velocity per axis */
  velocitySpread?: [number, number, number]
  /** Constant acceleration (e.g. gravity = [0, -4, 0]) */
  gravity?: [number, number, number]
  /** [min, max] lifetime in seconds */
  lifetime?: [number, number]
  /** [min, max] size in world units (converted to pixels via perspective) */
  size?: [number, number]
  /** Base color (hex string or THREE.Color) */
  color?: string | THREE.Color
  /** [start, end] opacity over particle life */
  opacity?: [number, number]
  /** Optional texture for quads */
  texture?: THREE.Texture
  /** Optional custom fragment shader body. Available mutable locals:
   *    float t       — global time (seconds)
   *    float age     — normalized particle age [0..1]
   *    float seed    — per-particle random [0..1]
   *    vec3 color    — base color (from props)
   *    float alpha   — current alpha (pre-faded by age)
   *    vec2 uv       — UV within the quad [0..1]
   *    bool hasTexture — whether a texture was provided
   *    sampler2D tex — the texture (only valid if hasTexture)
   *  Write to `color` and `alpha` to control output. */
  fragmentShader?: string
  /** THREE.js blending mode (default: NormalBlending) */
  blending?: THREE.Blending
  /** Depth write (default: false for transparent particles) */
  depthWrite?: boolean
}

// --- Shader sources ---

// We use the built-in `position` attribute as the particle origin, so
// Three.js's modelViewMatrix transform works correctly. Additional
// per-particle data (velocity, lifetime, etc.) are custom attributes.
const VERTEX_SHADER = `
attribute float aBirthTime;
attribute float aLifetime;
attribute float aSeed;
attribute vec3 aVelocity;
attribute float aSize;

uniform float uTime;
uniform vec3 uGravity;
uniform float uSizeScale;

varying float vAge;
varying float vSeed;

void main() {
    // Each particle cycles on its own lifetime
    float elapsed = uTime - aBirthTime;
    float t = mod(elapsed, aLifetime);
    float age = t / aLifetime;

    vAge = age;
    vSeed = aSeed;

    // Kinematic position: p = origin + v*t + 0.5*a*t^2
    // 'position' is the built-in attribute — we use it as the particle origin
    vec3 pos = position + aVelocity * t + 0.5 * uGravity * t * t;

    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mvPosition;

    // Size in pixels: world-unit size scaled by perspective
    gl_PointSize = aSize * uSizeScale / -mvPosition.z;
}
`

function buildFragmentShader(customCode?: string, hasTexture?: boolean): string {
  const textureDecl = hasTexture ? 'uniform sampler2D tex;\n' : ''

  const defaultBody = hasTexture
    ? `
    vec4 texColor = texture2D(tex, uv);
    color *= texColor.rgb;
    alpha *= texColor.a;
    `
    : `
    // Soft circle falloff
    float d = length(uv - vec2(0.5));
    alpha *= smoothstep(0.5, 0.2, d);
    `

  const userBody = customCode ?? defaultBody

  return `
uniform float uTime;
uniform vec3 uColor;
uniform vec2 uOpacity;
uniform bool uHasTexture;
${textureDecl}

varying float vAge;
varying float vSeed;

${GLSL_UTILITIES}

void main() {
    // Mutable locals for user code
    float t = uTime;
    float age = vAge;
    float seed = vSeed;
    vec2 uv = gl_PointCoord;
    vec3 color = uColor;
    bool hasTexture = uHasTexture;

    // Pre-compute alpha with age-based fade
    float alpha = mix(uOpacity.x, uOpacity.y, age);

    ${userBody}

    // Discard fully transparent fragments
    if (alpha <= 0.0) discard;

    gl_FragColor = vec4(color, alpha);
}
`
}

// --- Seeded PRNG (deterministic per-particle) ---

function mulberry32(seed: number): () => number {
  let s = seed | 0
  return () => {
    s = (s + 0x6d2b79f5) | 0
    let t = Math.imul(s ^ (s >>> 15), 1 | s)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// --- Component ---

export function Particles({
  count = 200,
  origin = [0, 0, 0],
  spread = [0, 0, 0],
  velocity = [0, 2, 0],
  velocitySpread = [1, 0.5, 1],
  gravity = [0, -4, 0],
  lifetime = [0.8, 1.5],
  size = [0.05, 0.15],
  color = '#ffffff',
  opacity = [1, 0],
  texture,
  fragmentShader: customFragment,
  blending = THREE.NormalBlending,
  depthWrite = false,
}: ParticlesProps) {
  const materialRef = React.useRef<THREE.ShaderMaterial>(null!)
  const pointsRef = React.useRef<THREE.Points>(null!)

  const colorObj = React.useMemo(
    () => (color instanceof THREE.Color ? color : new THREE.Color(color)),
    [color],
  )

  // Build geometry with per-particle attributes.
  // The built-in `position` attribute stores the particle origin.
  // Dependencies are spread into primitives so new array references
  // from the parent don't cause a rebuild when values haven't changed.
  const geometry = React.useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const rng = mulberry32(42)

    const positions = new Float32Array(count * 3)
    const birthTimes = new Float32Array(count)
    const lifetimes = new Float32Array(count)
    const seeds = new Float32Array(count)
    const velocities = new Float32Array(count * 3)
    const sizes = new Float32Array(count)

    const maxLT = lifetime[1]

    for (let i = 0; i < count; i++) {
      const lt = lifetime[0] + rng() * (lifetime[1] - lifetime[0])
      lifetimes[i] = lt
      seeds[i] = rng()

      // Stagger birth times so particles are spread across the cycle
      birthTimes[i] = -(rng() * maxLT)

      // Origin stored in position attribute
      positions[i * 3 + 0] = origin[0] + (rng() - 0.5) * spread[0]
      positions[i * 3 + 1] = origin[1] + (rng() - 0.5) * spread[1]
      positions[i * 3 + 2] = origin[2] + (rng() - 0.5) * spread[2]

      velocities[i * 3 + 0] = velocity[0] + (rng() - 0.5) * velocitySpread[0]
      velocities[i * 3 + 1] = velocity[1] + (rng() - 0.5) * velocitySpread[1]
      velocities[i * 3 + 2] = velocity[2] + (rng() - 0.5) * velocitySpread[2]

      sizes[i] = size[0] + rng() * (size[1] - size[0])
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('aBirthTime', new THREE.BufferAttribute(birthTimes, 1))
    geo.setAttribute('aLifetime', new THREE.BufferAttribute(lifetimes, 1))
    geo.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1))
    geo.setAttribute('aVelocity', new THREE.BufferAttribute(velocities, 3))
    geo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1))

    return geo
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [count, ...origin, ...spread, ...velocity, ...velocitySpread, ...lifetime, ...size])

  // Disable frustum culling — particles travel beyond their origin
  // so the bounding sphere would be too tight
  React.useEffect(() => {
    if (pointsRef.current) {
      pointsRef.current.frustumCulled = false
    }
  }, [])

  const hasTexture = !!texture

  const fragmentSrc = React.useMemo(
    () => buildFragmentShader(customFragment, hasTexture),
    [customFragment, hasTexture],
  )

  // Compute size scale from camera FOV and canvas height so that
  // `size` prop maps to world units
  const camera = Fiber.useThree((s) => s.camera) as THREE.PerspectiveCamera
  const canvasHeight = Fiber.useThree((s) => s.size.height)
  const sizeScale = React.useMemo(() => {
    const fov = (camera.fov * Math.PI) / 180
    return canvasHeight / (2 * Math.tan(fov / 2))
  }, [camera.fov, canvasHeight])

  const uniforms = React.useMemo(
    () => ({
      uTime: { value: 0 },
      uGravity: { value: new THREE.Vector3(...gravity) },
      uColor: { value: colorObj },
      uOpacity: { value: new THREE.Vector2(opacity[0], opacity[1]) },
      uHasTexture: { value: hasTexture },
      uSizeScale: { value: sizeScale },
      ...(hasTexture ? { tex: { value: texture } } : {}),
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [...gravity, colorObj, ...opacity, hasTexture, texture, sizeScale],
  )

  Fiber.useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime
    }
  })

  return (
    <points ref={pointsRef} geometry={geometry}>
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={VERTEX_SHADER}
        fragmentShader={fragmentSrc}
        transparent
        blending={blending}
        depthWrite={depthWrite}
      />
    </points>
  )
}
