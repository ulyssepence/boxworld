/* This was partially written by AI... I admit it. You never saw this. */
import * as React from 'react'
import * as Fiber from '@react-three/fiber'
import * as Drei from '@react-three/drei'
import * as THREE from 'three'

export interface GLSLShaderProps {
  code: string
  children?: React.ReactNode
}

const GLSL_UTILITIES = `
#define PI 3.1415926535897932384626433832795

float one_minus (float a) {
    return 1.0 - a;
}

vec2 cartesian_to_polar_long_lat (vec3 xyz) {
    float r = length(xyz);
    float long_ = atan(xyz.y, xyz.x);
    float lat = acos(xyz.z / r);
    return vec2(long_, lat);
}

vec2 xy_to_r_theta(vec2 xy) {
    float r = length(xy);
    float theta = atan(xy.y, xy.x);
    return vec2(r, theta);
}

vec2 r_theta_to_xy(vec2 rt) {
    float r = rt.x;
    float theta = rt.y;
    return vec2(r * cos(theta), r * sin(theta));
}

float floor_to_nearest (float value, float step) {
    return floor(value / step) * step;
}

vec2 floor_to_nearest2 (vec2 value, vec2 step) {
    return floor(value / step) * step;
}

float map(float value, float min1, float max1, float min2, float max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

float map01(float value, float bottom, float top) {
    return map(value, 0.0, 1.0, bottom, top);
}

vec2 back_to_front_bottom_to_top (vec3 unit_ray) {
    vec2 uv_temp = cartesian_to_polar_long_lat(unit_ray);
    uv_temp.y /= PI;
    uv_temp.y = one_minus(uv_temp.y);

    vec2 uv2 = cartesian_to_polar_long_lat(unit_ray.xzy);
    uv2.y = one_minus(uv2.y / PI);

    return vec2(uv_temp.y, uv2.y);
}

vec3 rainbow_gradient (float value) {
    value = clamp(value, 0.0, 1.0);
    float segment = 1.0 / 6.0;
    float segment_index = floor(value / segment);
    float segment_value = fract(value / segment);

    vec3 rainbow_colors[7];
    rainbow_colors[0] = vec3(1.0, 0.0, 0.0);
    rainbow_colors[1] = vec3(1.0, 0.5, 0.0);
    rainbow_colors[2] = vec3(1.0, 1.0, 0.0);
    rainbow_colors[3] = vec3(0.0, 1.0, 0.0);
    rainbow_colors[4] = vec3(0.0, 0.0, 1.0);
    rainbow_colors[5] = vec3(0.29, 0.0, 0.51);
    rainbow_colors[6] = vec3(0.93, 0.51, 0.93);

    vec3 color1 = rainbow_colors[int(segment_index)];
    vec3 color2 = rainbow_colors[int(min(segment_index + 1.0, 6.0))];
    return mix(color1, color2, segment_value);
}

vec3 palette (vec3 a, vec3 b, vec3 c, vec3 d, float t) {
    return a + b * cos(6.28318 * (c * t + d));
}

float sin01 (float value) {
    return (sin(value) + 1.0) * 0.5;
}

float sin01ma (float t, float period_secs, float offset_secs, float m, float a) {
    return sin01((t + offset_secs) / period_secs * 2.0 * PI) * m + a;
}

float stay01 (float t, float period_secs, float offset_secs, float power) {
    float wave = sin((t + offset_secs) / period_secs * 2.0 * PI);
    float s = sign(wave);
    float a = map(s - (s * pow(1.0 - abs(wave), power)), -1.0, 1.0, 0.0, 1.0);
    return a;
}

float stay (float t, float start, float stop, float period_secs, float offset_secs, float power) {
    return stay01(t, period_secs, offset_secs, power) * (stop - start) + start;
}

float triangle (float t) {
    return 2.0 * abs(fract(t + 0.5) - 0.5);
}

float square (float t) {
    return float(0.5 < fract(t));
}

vec3 replace_rgb (vec3 old_color, vec3 new_r, vec3 new_g, vec3 new_b) {
    return new_r * old_color.x + new_g * old_color.y + new_b * old_color.z;
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec2 with_u (vec2 uv, float new_value) {
    return vec2(new_value, uv.y);
}

vec2 with_v (vec2 uv, float new_value) {
    return vec2(uv.x, new_value);
}

vec2 rotate2d(vec2 v, float radians) {
    float s = sin(radians);
    float c = cos(radians);
    mat2 m = mat2(vec2(c, s), vec2(-s, c));
    return m * v;
}

float rect_sdf (vec2 uv, vec2 center, vec2 size) {
    vec2 d = abs(uv - center) - size;
    float outside = length(max(d, 0.0));
    float inside = min(max(d.x, d.y), 0.0);
    return outside + inside;
}

float circle_sdf (vec2 uv, vec2 center, float size) {
    return length(uv - center) - size;
}

float sphere_sdf (vec3 p, float size) {
    return length(p) - size;
}

float box_sdf (vec3 p, vec3 size) {
    vec3 q = abs(p) - size;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float smin (float a, float b, float k) {
    float res = exp2(-k * a) + exp2(-k * b);
    return -log2(res) / k;
}

float random (vec2 uv) {
    return fract(sin(dot(uv, vec2(12.9898,78.233))) * 43758.5453123);
}

float simple_noise_value(vec2 uv) {
    vec2 i = floor(uv);
    vec2 f = fract(uv);
    f = f * f * (3.0 - 2.0 * f);

    vec2 c0 = i + vec2(0.0, 0.0);
    vec2 c1 = i + vec2(1.0, 0.0);
    vec2 c2 = i + vec2(0.0, 1.0);
    vec2 c3 = i + vec2(1.0, 1.0);
    float r0 = random(c0);
    float r1 = random(c1);
    float r2 = random(c2);
    float r3 = random(c3);

    float bottomOfGrid = mix(r0, r1, f.x);
    float topOfGrid = mix(r2, r3, f.x);
    return mix(bottomOfGrid, topOfGrid, f.y);
}

float simple_noise(vec2 uv) {
    float t = 0.0;
    for (int i = 0; i < 3; i++) {
        float freq = pow(2.0, float(i));
        float amp = pow(0.5, float(3 - i));
        t += simple_noise_value(uv / freq) * amp;
    }
    return t;
}

float white_noise_f(float x) {
    return fract(sin(x * 12.9898) * 43758.5453);
}

vec2 white_noise_v2(vec2 x) {
    return fract(sin(x * vec2(12.9898, 78.233)) * 43758.5453);
}

vec3 white_noise_v3(vec3 x) {
    return fract(sin(x * vec3(12.9898, 78.233, 45.164)) * 43758.5453);
}

float voronoi_noise(vec2 uv) {
    vec2 indexUV = floor(uv);
    vec2 fractUV = fract(uv);

    float minDist = 1.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            float rand = random(indexUV + neighbor);
            vec2 point = vec2(rand, rand);
            vec2 diff = neighbor + point - fractUV;
            float dist = length(diff);
            minDist = min(minDist, dist);
        }
    }

    return minDist;
}

vec4 perlin_noise_permute(vec4 x) {
    return mod((34.0 * x + 1.0) * x, 289.0);
}

vec2 perlin_noise_fade(vec2 t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float perlin_noise(vec2 P) {
    vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
    vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
    Pi = mod(Pi, 289.0);
    vec4 ix = Pi.xzxz;
    vec4 iy = Pi.yyww;
    vec4 fx = Pf.xzxz;
    vec4 fy = Pf.yyww;
    vec4 i = perlin_noise_permute(perlin_noise_permute(ix) + iy);
    vec4 gx = 2.0 * fract(i * 0.0243902) - 1.0;
    vec4 gy = abs(gx) - 0.5;
    vec4 tx = floor(gx + 0.5);
    gx = gx - tx;
    vec2 g00 = vec2(gx.x, gy.x);
    vec2 g10 = vec2(gx.y, gy.y);
    vec2 g01 = vec2(gx.z, gy.z);
    vec2 g11 = vec2(gx.w, gy.w);
    vec4 norm = 1.79284291400159 - 0.85373472095314 * vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 *= norm.x;
    g01 *= norm.y;
    g10 *= norm.z;
    g11 *= norm.w;
    float n00 = dot(g00, vec2(fx.x, fy.x));
    float n10 = dot(g10, vec2(fx.y, fy.y));
    float n01 = dot(g01, vec2(fx.z, fy.z));
    float n11 = dot(g11, vec2(fx.w, fy.w));
    vec2 fade_xy = perlin_noise_fade(Pf.xy);
    vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
    float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy * 0.5 + 0.5;
}

float turbulence_noise(vec2 uv, float octaves) {
    float result = 0.0;
    float frequency = 1.0;
    float amplitude = 0.5;
    int octaves_i = int(octaves);

    for (int i = 0; i < octaves_i; i++) {
        result += perlin_noise(uv * frequency) * amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }

    return result;
}
`

function buildFragmentShader(code: string): string {
  return `
uniform float time;
uniform sampler2D sceneTexture;
uniform vec2 resolution;
varying vec2 vUv;

${GLSL_UTILITIES}

vec3 scene(vec2 uv) {
    return texture2D(sceneTexture, fract(uv)).rgb;
}

void main() {
    float t = time;
    vec2 uv = vUv;
    vec3 color = scene(uv);
    ${code}
    // Apply gamma correction (linear to sRGB)
    color = pow(color, vec3(1.0 / 2.2));
    gl_FragColor = vec4(color, 1.0);
}
`
}

const VERTEX_SHADER = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

function PostProcessQuad({ code, sceneTexture }: { code: string; sceneTexture: THREE.Texture }) {
  const materialRef = React.useRef<THREE.ShaderMaterial>(null!)
  const size = Fiber.useThree((state) => state.size)

  // Create material once, update shader when code changes
  React.useEffect(() => {
    if (materialRef.current) {
      materialRef.current.fragmentShader = buildFragmentShader(code)
      materialRef.current.needsUpdate = true
    }
  }, [code])

  // Update uniforms each frame
  Fiber.useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.time.value = state.clock.elapsedTime
      materialRef.current.uniforms.sceneTexture.value = sceneTexture
      materialRef.current.uniforms.resolution.value.set(size.width, size.height)
    }
  })

  return (
    <mesh>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={{
          time: { value: 0 },
          sceneTexture: { value: sceneTexture },
          resolution: { value: new THREE.Vector2(size.width, size.height) },
        }}
        vertexShader={VERTEX_SHADER}
        fragmentShader={buildFragmentShader(code)}
      />
    </mesh>
  )
}

export function GLSLShader({ code, children }: GLSLShaderProps) {
  const { size, gl, scene, camera } = Fiber.useThree()

  // Create FBO to render scene into
  const fbo = Drei.useFBO(size.width, size.height, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
    type: THREE.HalfFloatType, // Higher precision to avoid banding
  })

  // Orthographic camera for fullscreen quad
  const orthoCamera = React.useMemo(() => {
    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
    return cam
  }, [])

  // Fullscreen quad
  const quadMesh = React.useMemo(() => {
    const geometry = new THREE.PlaneGeometry(2, 2)
    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        sceneTexture: { value: null },
        resolution: { value: new THREE.Vector2() },
      },
      vertexShader: VERTEX_SHADER,
      fragmentShader: buildFragmentShader(code),
    })
    return new THREE.Mesh(geometry, material)
  }, [])

  // Update shader when code changes
  React.useEffect(() => {
    const material = quadMesh.material as THREE.ShaderMaterial
    material.fragmentShader = buildFragmentShader(code)
    material.needsUpdate = true
  }, [code, quadMesh])

  // Custom render loop
  Fiber.useFrame((state) => {
    const material = quadMesh.material as THREE.ShaderMaterial

    // Save current tone mapping settings
    const originalToneMapping = gl.toneMapping

    // 1. Render the scene to FBO (with tone mapping applied)
    gl.setRenderTarget(fbo)
    gl.clear()
    gl.render(scene, camera)

    // 2. Render fullscreen quad with post-processing to screen (no tone mapping - already applied)
    gl.setRenderTarget(null)
    gl.toneMapping = THREE.NoToneMapping
    material.uniforms.time.value = state.clock.elapsedTime
    material.uniforms.sceneTexture.value = fbo.texture
    material.uniforms.resolution.value.set(size.width, size.height)
    gl.render(quadMesh, orthoCamera)

    // Restore tone mapping
    gl.toneMapping = originalToneMapping
  }, 1) // Run after scene updates

  // Render children normally into the main scene
  return <>{children}</>
}

export default GLSLShader
