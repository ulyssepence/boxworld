# CustomMaterial shader component

`visualize/src/material.tsx` — a reusable R3F component that replaces `<meshStandardMaterial>` with a custom `ShaderMaterial`, accepting optional vertex and fragment shader body strings.

## Two ways to use it

### 1. `<CustomMaterial>` JSX component — for meshes with inline geometry

```tsx
<mesh position={[x, 0.5, y]}>
  <boxGeometry args={[1, 1, 1]} />
  <CustomMaterial
    color="#444444"
    vertexShader={`
      position.y += sin(position.x * 6.0 + t * TAU) * 0.1;
    `}
    fragmentShader={`
      color = lit(vec3(0.1), normal);
    `}
  />
</mesh>
```

### 2. `useCustomMaterial()` hook — for GLB models

GLB models have their own mesh hierarchy with embedded materials, so `<CustomMaterial>` as a JSX child won't work. Instead, create a material imperatively and traverse the scene to apply it:

```tsx
function KeyModel({ position }) {
  const glb = Drei.useGLTF('/static/models/key.glb')
  const scene = React.useMemo(() => glb.scene.clone(true), [glb.scene])
  const mat = material.useCustomMaterial({
    fragmentShader: `color = vec3(1.0, 0.8, 0.0);`,
  })

  React.useEffect(() => {
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) child.material = mat
    })
  }, [scene, mat])

  return (
    <group position={position}>
      <primitive object={scene} />
    </group>
  )
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `color` | `string \| THREE.Color` | `'#888888'` | Base color, available as `color` in shaders |
| `texture` | `THREE.Texture` | — | Optional texture, sampled before user code |
| `transparent` | `boolean` | `false` | Enable alpha blending |
| `blending` | `THREE.Blending` | `NormalBlending` | Blending mode |
| `depthWrite` | `boolean` | `true` | Write to depth buffer |
| `side` | `THREE.Side` | `FrontSide` | Which faces to render |
| `vertexShader` | `string` | — | Custom vertex shader body |
| `fragmentShader` | `string` | Lambert lighting | Custom fragment shader body |

## Vertex shader locals

User code can read and write these mutable locals:

| Local | Type | Description |
|-------|------|-------------|
| `position` | `vec3` | Object-space position (modify to deform) |
| `normal` | `vec3` | Object-space normal |
| `uv` | `vec2` | Texture coordinates |
| `color` | `vec3` | Vertex color (passed to fragment) |
| `worldPosition` | `vec3` | World-space position of original vertex (before user code) |
| `meshOrigin` | `vec3` | World-space translation of the mesh (from `modelMatrix`) |
| `t` | `float` | Time in seconds |
| `dt` | `float` | Delta time since last frame |

All functions from `GLSL_UTILITIES` (shader.tsx) are available — `sin01`, `rotate2d`, `perlin_noise`, `voronoi_noise`, `map`, `random`, `palette`, etc.

**Implementation note:** User code writes `position`, `normal`, `uv` but internally these are text-replaced to `pos`, `norm`, `uvCoord` to avoid shadowing Three.js's injected `attribute vec3 position` in the GLSL preamble.

## Fragment shader locals

| Local | Type | Description |
|-------|------|-------------|
| `t` | `float` | Time in seconds |
| `dt` | `float` | Delta time |
| `uv` | `vec2` | Interpolated UVs |
| `normal` | `vec3` | Interpolated + normalized world-space normal |
| `color` | `vec3` | Base color (from vertex shader / `color` prop) |
| `alpha` | `float` | Opacity (starts at 1.0) |
| `worldPosition` | `vec3` | Interpolated world position |
| `meshOrigin` | `vec3` | World-space translation of the mesh |
| `hasTexture` | `bool` | Whether texture was provided |
| `tex` | `sampler2D` | Texture sampler (always declared; dummy 1x1 white if no texture) |

Write to `color` and `alpha` to control output. Fragments with `alpha <= 0.0` are discarded.

## Built-in functions

### `lit(vec3 c, vec3 n) → vec3`

Lambert-like directional + ambient shading. Light direction: `normalize(vec3(1.0, 1.0, 0.5))`, ambient: 0.15.

```glsl
color = lit(vec3(0.6), normal);           // lit grey
color = lit(color, normal);               // lit base color
color = lit(vec3(1.0, 0.0, 0.0), normal); // lit red
```

### GLSL_UTILITIES (from shader.tsx)

All utilities are available in both vertex and fragment shaders. Key ones:

- `rotate2d(vec2 v, float radians)` — 2D rotation matrix
- `perlin_noise(vec2 uv)` — Perlin noise [0, 1]
- `voronoi_noise(vec2 uv)` — Voronoi cell noise
- `simple_noise(vec2 uv)` — Multi-octave value noise
- `sin01(float v)` — sin remapped to [0, 1]
- `map(float v, float min1, float max1, float min2, float max2)` — remap
- `random(vec2 uv)` — hash-based pseudo-random
- `palette(vec3 a, vec3 b, vec3 c, vec3 d, float t)` — cosine palette
- `hsv2rgb(vec3 c)` / `rgb2hsv(vec3 c)` — color space conversion
- `PI`, `TAU` is not defined (use `2.0 * PI` or define it yourself)

## Default behavior (no custom shaders)

- **Vertex:** pass-through (identity transform)
- **Fragment:** `color = lit(color, normal)` — Lambert shading with the base `color` prop

## Gotchas

- **GLSL needs semicolons** — unlike JavaScript, every statement must end with `;`
- **No `gl_VertexID`** — Three.js uses WebGL1/GLSL 100, not GLSL 300 es
- **Texture always declared** — the `tex` sampler is always present (bound to a 1x1 white dummy when no texture prop). This avoids GLSL compile errors since all code paths are compiled regardless of `if (hasTexture)`.
- **Additive blending on bright surfaces** — particles using `AdditiveBlending` are invisible against white/bright surfaces since adding to 1.0 is still 1.0
- **No `#define` in user code** — the text-replace approach means `#define` directives in user vertex code may behave unexpectedly if they reference `position`, `normal`, or `uv`
- **`meshStandardMaterial` lighting** — `CustomMaterial` does not participate in Three.js's lighting system. Use the `lit()` function for basic shading, or write your own lighting in the fragment shader.
