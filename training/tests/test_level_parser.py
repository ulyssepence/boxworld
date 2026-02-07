"""Tests for the ASCII level parser."""

import glob
import os

import pytest

from level_parser import load_level, parse_level


def test_parse_simple_level():
    text = "####\n#A #\n# G#\n####\n"
    result = parse_level(text, "test")
    assert result["width"] == 4
    assert result["height"] == 4
    assert result["agentStart"] == [1, 1]
    assert result["grid"][2][2] == 4  # Goal


def test_parse_all_cell_types():
    text = "#A DKG~#"
    result = parse_level(text, "types")
    row = result["grid"][0]
    assert row[0] == 1  # Wall
    assert row[1] == 0  # Agent → Floor
    assert row[2] == 0  # Space → Floor
    assert row[3] == 2  # Door
    assert row[4] == 3  # Key
    assert row[5] == 4  # Goal
    assert row[6] == 5  # Lava
    assert row[7] == 1  # Wall


def test_agent_replaced_with_floor():
    text = "###\n#A#\n###\n"
    result = parse_level(text, "test")
    assert result["grid"][1][1] == 0  # Floor, not 'A'
    assert result["agentStart"] == [1, 1]


def test_missing_agent_raises():
    text = "###\n# #\n###\n"
    with pytest.raises(ValueError, match="No agent"):
        parse_level(text, "test")


def test_id_and_name_derived():
    text = "#A#"
    result = parse_level(text, "my_cool_level")
    assert result["id"] == "my_cool_level"
    assert result["name"] == "My Cool Level"


def test_ragged_rows_raises():
    text = "####\n#A#\n####\n"
    with pytest.raises(ValueError, match="ragged"):
        parse_level(text, "test")


def test_load_all_existing_levels():
    levels_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "levels")
    levels_dir = os.path.normpath(levels_dir)
    txt_files = sorted(glob.glob(os.path.join(levels_dir, "*.txt")))
    assert len(txt_files) >= 10, f"Expected at least 10 level files, found {len(txt_files)}"
    for path in txt_files:
        data = load_level(path)
        assert data["width"] > 0
        assert data["height"] > 0
        assert len(data["grid"]) == data["height"]
        assert len(data["grid"][0]) == data["width"]
        assert data["agentStart"] is not None
