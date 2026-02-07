"""Parse ASCII level files (.txt) into level dicts compatible with BoxworldEnv."""

from __future__ import annotations

import os

# Character → CellType mapping
CHAR_TO_CELL = {
    "#": 1,  # Wall
    " ": 0,  # Floor
    "D": 2,  # Door
    "K": 3,  # Key
    "G": 4,  # Goal
    "~": 5,  # Lava
    "A": 0,  # Agent start (treated as Floor)
}


def parse_level(text: str, level_id: str) -> dict:
    """Parse ASCII level text into a dict with id, name, width, height, grid, agentStart."""
    lines = text.rstrip("\n").split("\n")
    if not lines:
        raise ValueError("Empty level text")

    height = len(lines)
    width = len(lines[0])

    agent_start: list[int] | None = None
    grid: list[list[int]] = []

    for y, line in enumerate(lines):
        if len(line) != width:
            raise ValueError(f"Row {y} has length {len(line)}, expected {width} (ragged rows)")
        row: list[int] = []
        for x, ch in enumerate(line):
            if ch not in CHAR_TO_CELL:
                raise ValueError(f"Unknown character '{ch}' at ({x}, {y})")
            row.append(CHAR_TO_CELL[ch])
            if ch == "A":
                agent_start = [x, y]
        grid.append(row)

    if agent_start is None:
        raise ValueError("No agent start position ('A') found in level")

    name = level_id.replace("_", " ").title()

    return {
        "id": level_id,
        "name": name,
        "width": width,
        "height": height,
        "grid": grid,
        "agentStart": agent_start,
    }


def load_level(path: str) -> dict:
    """Load a .txt level file and return the parsed dict."""
    level_id = os.path.splitext(os.path.basename(path))[0]
    with open(path) as f:
        text = f.read()
    return parse_level(text, level_id)
