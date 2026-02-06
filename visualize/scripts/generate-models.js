/**
 * generate-models.js
 *
 * Generates 7 placeholder .glb 3D model files for the Boxworld visualizer.
 * Run with: node scripts/generate-models.js
 */

import { Document, NodeIO } from "@gltf-transform/core";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = path.resolve(__dirname, "../static/models");

// ---------------------------------------------------------------------------
// Hex color helpers
// ---------------------------------------------------------------------------

/** Convert a hex color string like "#FF4500" to an RGBA array with values 0-1. */
function hexToRGBA(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return [r, g, b, 1.0];
}

// ---------------------------------------------------------------------------
// Geometry builders -- each returns { positions, normals, indices }
// All values are plain JS arrays; they get wrapped in typed arrays later.
// ---------------------------------------------------------------------------

/** Flat 1x1 quad centered at origin, lying in the XZ plane (normal up). */
function buildPlane() {
  // prettier-ignore
  const positions = [
    -0.5, 0,  0.5,
     0.5, 0,  0.5,
     0.5, 0, -0.5,
    -0.5, 0, -0.5,
  ];
  // prettier-ignore
  const normals = [
    0, 1, 0,
    0, 1, 0,
    0, 1, 0,
    0, 1, 0,
  ];
  const indices = [0, 1, 2, 0, 2, 3];
  return { positions, normals, indices };
}

/** Axis-aligned box with given half-extents (defaults to 0.5 on each axis = 1x1x1 cube). */
function buildBox(hx = 0.5, hy = 0.5, hz = 0.5) {
  const positions = [];
  const normals = [];
  const indices = [];

  // Each face: 4 vertices, 2 triangles.
  const faces = [
    // [normal, tangent, bitangent] -- used to build the 4 corners
    { n: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] }, // +Z
    { n: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] }, // -Z
    { n: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] }, // +X
    { n: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] }, // -X
    { n: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] }, // +Y
    { n: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] }, // -Y
  ];

  const half = [hx, hy, hz];
  let vi = 0;

  for (const { n, u, v } of faces) {
    // center of face
    const cx = n[0] * half[0];
    const cy = n[1] * half[1];
    const cz = n[2] * half[2];
    // tangent / bitangent scaled by half-extents
    const ux = u[0] * half[0],
      uy = u[1] * half[1],
      uz = u[2] * half[2];
    const vx = v[0] * half[0],
      vy = v[1] * half[1],
      vz = v[2] * half[2];

    // 4 corners: center +/- tangent +/- bitangent
    positions.push(
      cx - ux - vx, cy - uy - vy, cz - uz - vz,
      cx + ux - vx, cy + uy - vy, cz + uz - vz,
      cx + ux + vx, cy + uy + vy, cz + uz + vz,
      cx - ux + vx, cy - uy + vy, cz - uz + vz,
    );
    normals.push(...n, ...n, ...n, ...n);
    indices.push(vi, vi + 1, vi + 2, vi, vi + 2, vi + 3);
    vi += 4;
  }

  return { positions, normals, indices };
}

/** Torus centered at origin lying in the XZ plane. */
function buildTorus(radius = 0.2, tube = 0.05, radialSegments = 16, tubularSegments = 8) {
  const positions = [];
  const normals = [];
  const indices = [];

  for (let j = 0; j <= radialSegments; j++) {
    for (let i = 0; i <= tubularSegments; i++) {
      const u = (i / tubularSegments) * Math.PI * 2;
      const v = (j / radialSegments) * Math.PI * 2;

      const x = (radius + tube * Math.cos(v)) * Math.cos(u);
      const y = tube * Math.sin(v);
      const z = (radius + tube * Math.cos(v)) * Math.sin(u);

      positions.push(x, y, z);

      const nx = Math.cos(v) * Math.cos(u);
      const ny = Math.sin(v);
      const nz = Math.cos(v) * Math.sin(u);
      normals.push(nx, ny, nz);
    }
  }

  for (let j = 0; j < radialSegments; j++) {
    for (let i = 0; i < tubularSegments; i++) {
      const a = j * (tubularSegments + 1) + i;
      const b = a + tubularSegments + 1;
      indices.push(a, b, a + 1);
      indices.push(a + 1, b, b + 1);
    }
  }

  return { positions, normals, indices };
}

/** Flat disc (filled circle) in the XZ plane, normal up. */
function buildCircle(radius = 0.4, segments = 16) {
  const positions = [0, 0, 0]; // center
  const normals = [0, 1, 0];
  const indices = [];

  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    positions.push(radius * Math.cos(theta), 0, radius * Math.sin(theta));
    normals.push(0, 1, 0);
  }

  for (let i = 1; i <= segments; i++) {
    indices.push(0, i, i + 1);
  }

  return { positions, normals, indices };
}

/** UV sphere centered at origin. */
function buildSphere(radius = 0.5, widthSegments = 16, heightSegments = 8) {
  const positions = [];
  const normals = [];
  const indices = [];

  for (let y = 0; y <= heightSegments; y++) {
    const v = y / heightSegments;
    const phi = v * Math.PI;
    for (let x = 0; x <= widthSegments; x++) {
      const u = x / widthSegments;
      const theta = u * Math.PI * 2;

      const nx = Math.sin(phi) * Math.cos(theta);
      const ny = Math.cos(phi);
      const nz = Math.sin(phi) * Math.sin(theta);

      positions.push(radius * nx, radius * ny, radius * nz);
      normals.push(nx, ny, nz);
    }
  }

  for (let y = 0; y < heightSegments; y++) {
    for (let x = 0; x < widthSegments; x++) {
      const a = y * (widthSegments + 1) + x;
      const b = a + widthSegments + 1;
      indices.push(a, b, a + 1);
      indices.push(a + 1, b, b + 1);
    }
  }

  return { positions, normals, indices };
}

// ---------------------------------------------------------------------------
// GLB writer
// ---------------------------------------------------------------------------

/**
 * Build a glTF Document from raw geometry + color, then write it as GLB.
 */
async function writeGLB(io, filename, geometry, color) {
  const doc = new Document();
  const buffer = doc.createBuffer();

  // Material
  const material = doc.createMaterial("mat").setBaseColorFactor(color).setMetallicFactor(0).setRoughnessFactor(1);

  // Accessors
  const positionAccessor = doc
    .createAccessor("position")
    .setType("VEC3")
    .setArray(new Float32Array(geometry.positions))
    .setBuffer(buffer);

  const normalAccessor = doc
    .createAccessor("normal")
    .setType("VEC3")
    .setArray(new Float32Array(geometry.normals))
    .setBuffer(buffer);

  const indexAccessor = doc
    .createAccessor("index")
    .setType("SCALAR")
    .setArray(new Uint16Array(geometry.indices))
    .setBuffer(buffer);

  // Primitive -> Mesh
  const prim = doc.createPrimitive();
  prim.setAttribute("POSITION", positionAccessor);
  prim.setAttribute("NORMAL", normalAccessor);
  prim.setIndices(indexAccessor);
  prim.setMaterial(material);
  const mesh = doc.createMesh("mesh").addPrimitive(prim);

  // Node -> Scene
  const node = doc.createNode("root").setMesh(mesh);
  const scene = doc.createScene("scene").addChild(node);
  doc.getRoot().setDefaultScene(scene);

  const outPath = path.join(OUTPUT_DIR, filename);
  await io.write(outPath, doc);
  console.log(`  wrote ${outPath}`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const io = new NodeIO();

  console.log("Generating placeholder .glb models...\n");

  // 1. floor -- flat 1x1 plane, gray
  await writeGLB(io, "floor.glb", buildPlane(), hexToRGBA("#888888"));

  // 2. wall -- 1x1x1 cube, dark gray
  await writeGLB(io, "wall.glb", buildBox(0.5, 0.5, 0.5), hexToRGBA("#444444"));

  // 3. door -- 0.1x1x1 slab (thin on x-axis), brown
  await writeGLB(io, "door.glb", buildBox(0.05, 0.5, 0.5), hexToRGBA("#8B4513"));

  // 4. key -- small torus, gold
  await writeGLB(io, "key.glb", buildTorus(0.2, 0.05, 16, 8), hexToRGBA("#FFD700"));

  // 5. goal -- flat circle, green
  await writeGLB(io, "goal.glb", buildCircle(0.4, 16), hexToRGBA("#00FF88"));

  // 6. lava -- flat 1x1 plane, red/orange
  await writeGLB(io, "lava.glb", buildPlane(), hexToRGBA("#FF4500"));

  // 7. player -- 0.5 radius sphere, cyan
  await writeGLB(io, "player.glb", buildSphere(0.5, 16, 8), hexToRGBA("#00FFFF"));

  console.log("\nDone! All 7 models generated.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
