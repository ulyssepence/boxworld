import * as React from 'react'
import * as Fiber from '@react-three/fiber'
import * as THREE from 'three'
import { GLSL_UTILITIES } from './shader'

// --- Types ---

export interface CustomMaterialProps {
  /** Base color (hex string or THREE.Color) */
  color?: string | THREE.Color
  /** Optional texture — sampled before user code, multiplied into color */
  texture?: THREE.Texture
  /** Enable transparency (default: false) */
  transparent?: boolean
  /** THREE.js blending mode (default: NormalBlending) */
  blending?: THREE.Blending
  /** Depth write (default: true) */
  depthWrite?: boolean
  /** Face side (default: FrontSide) */
  side?: THREE.Side
  /** Custom vertex shader body. Available mutable locals:
   *    vec3  position      — object-space position (modify to deform)
   *    vec3  normal        — object-space normal
   *    vec2  uv            — texture coordinates
   *    vec3  color         — vertex color (passed to fragment)
   *    vec3  worldPosition — world-space position of original vertex (read-only input)
   *    float t             — time in seconds
   *    float dt            — delta time since last frame */
  vertexShader?: string
  /** Custom fragment shader body. Available mutable locals:
   *    float t             — time in seconds
   *    float dt            — delta time
   *    vec2  uv            — interpolated UVs
   *    vec3  normal        — interpolated + normalized normal
   *    vec3  color         — from vertex shader (base = uColor)
   *    float alpha         — opacity (starts at 1.0)
   *    vec3  worldPosition — interpolated world position
   *    bool  hasTexture    — whether texture was provided
   *    sampler2D tex       — texture (only valid if hasTexture)
   *  Write to `color` and `alpha` to control output. */
  fragmentShader?: string
}

// --- Shader builders ---

function buildVertexShader(userCode?: string): string {
  // User code refers to position/normal/uv — we do a simple text replace
  // to map them to non-colliding local names (pos/norm/uvCoord), avoiding
  // the #define approach which would corrupt Three.js's injected preamble.
  const processed = (userCode ?? '')
    .replace(/\bposition\b/g, 'pos')
    .replace(/\bnormal\b/g, 'norm')
    .replace(/\buv\b/g, 'uvCoord')

  return `
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vColor;
varying vec3 vWorldPosition;
varying vec3 vMeshOrigin;

uniform float uTime;
uniform float uDeltaTime;
uniform vec3 uColor;

${GLSL_UTILITIES}

void main() {
    vec3 worldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
    vec3 meshOrigin = modelMatrix[3].xyz;

    vec3 pos = position;
    vec3 norm = normal;
    vec2 uvCoord = uv;
    vec3 color = uColor;
    float t = uTime;
    float dt = uDeltaTime;

    ${processed}

    vUv = uvCoord;
    vNormal = normalize(normalMatrix * norm);
    vColor = color;
    vWorldPosition = (modelMatrix * vec4(pos, 1.0)).xyz;
    vMeshOrigin = meshOrigin;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`
}

function buildFragmentShader(userCode?: string): string {
  const defaultBody = `
    color = lit(color, normal);
  `

  const userBody = userCode ?? defaultBody

  return `
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vColor;
varying vec3 vWorldPosition;
varying vec3 vMeshOrigin;

uniform float uTime;
uniform float uDeltaTime;
uniform bool uHasTexture;
uniform sampler2D tex;

${GLSL_UTILITIES}

vec3 lit(vec3 c, vec3 n) {
    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
    float diffuse = max(dot(n, lightDir), 0.0);
    float ambient = 0.15;
    return c * (ambient + (1.0 - ambient) * diffuse);
}

void main() {
    float t = uTime;
    float dt = uDeltaTime;
    vec2 uv = vUv;
    vec3 normal = normalize(vNormal);
    vec3 color = vColor;
    float alpha = 1.0;
    vec3 worldPosition = vWorldPosition;
    vec3 meshOrigin = vMeshOrigin;
    bool hasTexture = uHasTexture;

    // Sample texture before user code so color already includes it
    if (hasTexture) {
        vec4 texColor = texture2D(tex, uv);
        color *= texColor.rgb;
        alpha *= texColor.a;
    }

    ${userBody}

    if (alpha <= 0.0) discard;

    gl_FragColor = vec4(color, alpha);
}
`
}

// 1x1 white texture used when no texture prop is provided
const DUMMY_TEXTURE = new THREE.DataTexture(
  new Uint8Array([255, 255, 255, 255]),
  1,
  1,
  THREE.RGBAFormat,
)
DUMMY_TEXTURE.needsUpdate = true

// --- Component ---

export function CustomMaterial({
  color = '#888888',
  texture,
  transparent = false,
  blending = THREE.NormalBlending,
  depthWrite = true,
  side = THREE.FrontSide,
  vertexShader: customVertex,
  fragmentShader: customFragment,
}: CustomMaterialProps) {
  const materialRef = React.useRef<THREE.ShaderMaterial>(null!)

  const colorObj = React.useMemo(
    () => (color instanceof THREE.Color ? color : new THREE.Color(color)),
    [color],
  )

  const hasTexture = !!texture

  const vertexSrc = React.useMemo(() => buildVertexShader(customVertex), [customVertex])

  const fragmentSrc = React.useMemo(() => buildFragmentShader(customFragment), [customFragment])

  // Update shaders when source changes
  React.useEffect(() => {
    if (materialRef.current) {
      materialRef.current.vertexShader = vertexSrc
      materialRef.current.fragmentShader = fragmentSrc
      materialRef.current.needsUpdate = true
    }
  }, [vertexSrc, fragmentSrc])

  const uniforms = React.useMemo(
    () => ({
      uTime: { value: 0 },
      uDeltaTime: { value: 0 },
      uColor: { value: colorObj },
      uHasTexture: { value: hasTexture },
      tex: { value: texture ?? DUMMY_TEXTURE },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [colorObj, hasTexture, texture],
  )

  Fiber.useFrame((state, delta) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime
      materialRef.current.uniforms.uDeltaTime.value = delta
    }
  })

  return (
    <shaderMaterial
      ref={materialRef}
      uniforms={uniforms}
      vertexShader={vertexSrc}
      fragmentShader={fragmentSrc}
      transparent={transparent}
      blending={blending}
      depthWrite={depthWrite}
      side={side}
    />
  )
}

/** Apply materials to meshes in a scene by index order.
 *  Traverses the scene, collects all Mesh children, and assigns materials[i]
 *  to the i-th mesh. If fewer materials than meshes, the last material is
 *  reused for remaining meshes. */
export function applyMaterials(scene: THREE.Object3D, materials: THREE.Material[]): void {
  if (materials.length === 0) return
  let i = 0
  scene.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.material = materials[Math.min(i, materials.length - 1)]
      i++
    }
  })
}

/** Create a THREE.ShaderMaterial imperatively (for applying to GLB meshes).
 *  Returns a stable material + a useFrame hook that updates time uniforms. */
export function useCustomMaterial(props: CustomMaterialProps = {}): THREE.ShaderMaterial {
  const {
    color = '#888888',
    texture,
    transparent = false,
    blending = THREE.NormalBlending,
    depthWrite = true,
    side = THREE.FrontSide,
    vertexShader: customVertex,
    fragmentShader: customFragment,
  } = props

  const mat = React.useMemo(() => {
    const colorObj = color instanceof THREE.Color ? color : new THREE.Color(color)
    const hasTexture = !!texture
    return new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uDeltaTime: { value: 0 },
        uColor: { value: colorObj },
        uHasTexture: { value: hasTexture },
        tex: { value: texture ?? DUMMY_TEXTURE },
      },
      vertexShader: buildVertexShader(customVertex),
      fragmentShader: buildFragmentShader(customFragment),
      transparent,
      blending,
      depthWrite,
      side,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [color, texture, customVertex, customFragment, transparent, blending, depthWrite, side])

  Fiber.useFrame((state, delta) => {
    mat.uniforms.uTime.value = state.clock.elapsedTime
    mat.uniforms.uDeltaTime.value = delta
  })

  return mat
}
