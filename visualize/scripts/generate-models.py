#!/usr/bin/env python3
"""
Generate 7 placeholder .glb 3D model files for the Boxworld visualizer.
Uses only Python stdlib -- writes GLB (glTF Binary) directly.
"""

import json
import math
import os
import struct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "static", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hex_to_rgba(hex_str):
    """Convert '#RRGGBB' to [r, g, b, 1.0] with values 0-1."""
    r = int(hex_str[1:3], 16) / 255.0
    g = int(hex_str[3:5], 16) / 255.0
    b = int(hex_str[5:7], 16) / 255.0
    return [r, g, b, 1.0]


def pad_to_4(data):
    """Pad bytes to 4-byte boundary with spaces (for JSON) or zeros (for BIN)."""
    remainder = len(data) % 4
    if remainder:
        data += b"\x00" * (4 - remainder)
    return data


def compute_bounds(positions):
    """Compute min/max for a flat list of [x,y,z, x,y,z, ...] floats."""
    xs = positions[0::3]
    ys = positions[1::3]
    zs = positions[2::3]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def write_glb(filename, positions, normals, indices, color):
    """
    Write a minimal GLB file with one mesh, one material.
    positions: flat list of floats (x,y,z,...)
    normals:   flat list of floats (nx,ny,nz,...)
    indices:   flat list of ints
    color:     [r, g, b, a] 0-1
    """
    # Pack binary data
    pos_bytes = struct.pack(f"<{len(positions)}f", *positions)
    norm_bytes = struct.pack(f"<{len(normals)}f", *normals)
    idx_bytes = struct.pack(f"<{len(indices)}H", *indices)

    # Buffer layout: positions | normals | indices (each section padded to 4 bytes)
    pos_offset = 0
    pos_length = len(pos_bytes)

    norm_offset = pos_length
    norm_length = len(norm_bytes)

    idx_offset = norm_offset + norm_length
    idx_length = len(idx_bytes)

    # Pad index bytes to 4-byte boundary for the total buffer
    total_bin = pos_bytes + norm_bytes + idx_bytes
    total_bin = pad_to_4(total_bin)
    buffer_byte_length = len(total_bin)

    num_vertices = len(positions) // 3
    num_indices = len(indices)
    pos_min, pos_max = compute_bounds(positions)

    # Build glTF JSON
    gltf = {
        "asset": {"version": "2.0", "generator": "boxworld-generate-models"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": num_vertices,
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max,
            },
            {
                "bufferView": 1,
                "componentType": 5126,  # FLOAT
                "count": num_vertices,
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": num_indices,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": pos_length,
                "target": 34962,  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": norm_offset,
                "byteLength": norm_length,
                "target": 34962,  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": idx_offset,
                "byteLength": idx_length,
                "target": 34963,  # ELEMENT_ARRAY_BUFFER
            },
        ],
        "buffers": [{"byteLength": buffer_byte_length}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": color,
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                }
            }
        ],
    }

    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    # Pad JSON chunk to 4 bytes with spaces (per GLB spec)
    remainder = len(json_bytes) % 4
    if remainder:
        json_bytes += b" " * (4 - remainder)

    json_chunk_length = len(json_bytes)
    bin_chunk_length = len(total_bin)

    # GLB header: magic + version + total length
    total_length = (
        12  # GLB header
        + 8 + json_chunk_length  # JSON chunk header + data
        + 8 + bin_chunk_length  # BIN chunk header + data
    )

    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "wb") as f:
        # GLB Header
        f.write(struct.pack("<I", 0x46546C67))  # magic: 'glTF'
        f.write(struct.pack("<I", 2))  # version
        f.write(struct.pack("<I", total_length))

        # JSON chunk
        f.write(struct.pack("<I", json_chunk_length))
        f.write(struct.pack("<I", 0x4E4F534A))  # 'JSON'
        f.write(json_bytes)

        # BIN chunk
        f.write(struct.pack("<I", bin_chunk_length))
        f.write(struct.pack("<I", 0x004E4942))  # 'BIN\0'
        f.write(total_bin)

    size = os.path.getsize(out_path)
    print(f"  wrote {out_path} ({size} bytes)")


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def build_plane():
    """Flat 1x1 quad in XZ plane, normal up."""
    positions = [
        -0.5, 0.0,  0.5,
         0.5, 0.0,  0.5,
         0.5, 0.0, -0.5,
        -0.5, 0.0, -0.5,
    ]
    normals = [
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
    ]
    indices = [0, 1, 2, 0, 2, 3]
    return positions, normals, indices


def build_box(hx=0.5, hy=0.5, hz=0.5):
    """Axis-aligned box with given half-extents."""
    positions = []
    normals = []
    indices = []

    faces = [
        # normal,      tangent,      bitangent
        ([0, 0, 1],  [1, 0, 0],  [0, 1, 0]),   # +Z
        ([0, 0, -1], [-1, 0, 0], [0, 1, 0]),   # -Z
        ([1, 0, 0],  [0, 0, -1], [0, 1, 0]),   # +X
        ([-1, 0, 0], [0, 0, 1],  [0, 1, 0]),   # -X
        ([0, 1, 0],  [1, 0, 0],  [0, 0, -1]),  # +Y
        ([0, -1, 0], [1, 0, 0],  [0, 0, 1]),   # -Y
    ]

    half = [hx, hy, hz]
    vi = 0

    for n, u, v in faces:
        cx = n[0] * half[0]
        cy = n[1] * half[1]
        cz = n[2] * half[2]
        ux, uy, uz = u[0]*half[0], u[1]*half[1], u[2]*half[2]
        vx, vy, vz = v[0]*half[0], v[1]*half[1], v[2]*half[2]

        positions.extend([
            cx - ux - vx, cy - uy - vy, cz - uz - vz,
            cx + ux - vx, cy + uy - vy, cz + uz - vz,
            cx + ux + vx, cy + uy + vy, cz + uz + vz,
            cx - ux + vx, cy - uy + vy, cz - uz + vz,
        ])
        normals.extend(n * 4)
        indices.extend([vi, vi+1, vi+2, vi, vi+2, vi+3])
        vi += 4

    return positions, normals, indices


def build_torus(radius=0.2, tube=0.05, radial_seg=16, tubular_seg=8):
    """Torus centered at origin in XZ plane."""
    positions = []
    normals = []
    indices = []

    for j in range(radial_seg + 1):
        for i in range(tubular_seg + 1):
            u = (i / tubular_seg) * math.pi * 2
            v = (j / radial_seg) * math.pi * 2

            x = (radius + tube * math.cos(v)) * math.cos(u)
            y = tube * math.sin(v)
            z = (radius + tube * math.cos(v)) * math.sin(u)
            positions.extend([x, y, z])

            nx = math.cos(v) * math.cos(u)
            ny = math.sin(v)
            nz = math.cos(v) * math.sin(u)
            normals.extend([nx, ny, nz])

    for j in range(radial_seg):
        for i in range(tubular_seg):
            a = j * (tubular_seg + 1) + i
            b = a + tubular_seg + 1
            indices.extend([a, b, a + 1])
            indices.extend([a + 1, b, b + 1])

    return positions, normals, indices


def build_circle(radius=0.4, segments=16):
    """Flat disc in XZ plane, normal up."""
    positions = [0.0, 0.0, 0.0]  # center
    normals = [0.0, 1.0, 0.0]
    indices = []

    for i in range(segments + 1):
        theta = (i / segments) * math.pi * 2
        positions.extend([radius * math.cos(theta), 0.0, radius * math.sin(theta)])
        normals.extend([0.0, 1.0, 0.0])

    for i in range(1, segments + 1):
        indices.extend([0, i, i + 1])

    return positions, normals, indices


def build_sphere(radius=0.5, width_seg=16, height_seg=8):
    """UV sphere centered at origin."""
    positions = []
    normals = []
    indices = []

    for y in range(height_seg + 1):
        v = y / height_seg
        phi = v * math.pi
        for x in range(width_seg + 1):
            u = x / width_seg
            theta = u * math.pi * 2

            nx = math.sin(phi) * math.cos(theta)
            ny = math.cos(phi)
            nz = math.sin(phi) * math.sin(theta)

            positions.extend([radius * nx, radius * ny, radius * nz])
            normals.extend([nx, ny, nz])

    for y in range(height_seg):
        for x in range(width_seg):
            a = y * (width_seg + 1) + x
            b = a + width_seg + 1
            indices.extend([a, b, a + 1])
            indices.extend([a + 1, b, b + 1])

    return positions, normals, indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating placeholder .glb models...\n")

    # 1. floor -- flat 1x1 plane, gray
    p, n, i = build_plane()
    write_glb("floor.glb", p, n, i, hex_to_rgba("#888888"))

    # 2. wall -- 1x1x1 cube, dark gray
    p, n, i = build_box(0.5, 0.5, 0.5)
    write_glb("wall.glb", p, n, i, hex_to_rgba("#444444"))

    # 3. door -- 0.1x1x1 slab (thin on x-axis), brown
    p, n, i = build_box(0.05, 0.5, 0.5)
    write_glb("door.glb", p, n, i, hex_to_rgba("#8B4513"))

    # 4. key -- small torus, gold
    p, n, i = build_torus(0.2, 0.05, 16, 8)
    write_glb("key.glb", p, n, i, hex_to_rgba("#FFD700"))

    # 5. goal -- flat circle, green
    p, n, i = build_circle(0.4, 16)
    write_glb("goal.glb", p, n, i, hex_to_rgba("#00FF88"))

    # 6. lava -- flat 1x1 plane, red/orange
    p, n, i = build_plane()
    write_glb("lava.glb", p, n, i, hex_to_rgba("#FF4500"))

    # 7. player -- 0.5 radius sphere, cyan
    p, n, i = build_sphere(0.5, 16, 8)
    write_glb("player.glb", p, n, i, hex_to_rgba("#00FFFF"))

    print("\nDone! All 7 models generated.")


if __name__ == "__main__":
    main()
