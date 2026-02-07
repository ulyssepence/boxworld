"""Tests for ONNX export of SB3 PPO checkpoints."""

import os
import shutil

import numpy as np
import onnxruntime as ort
import pytest
import torch
from stable_baselines3 import PPO

from environment import BoxworldEnv
from export import Exporter


@pytest.fixture
def trained_model(tmp_path):
    """Create a minimal PPO model, train briefly, and save a checkpoint."""
    env = BoxworldEnv()
    model = PPO("MlpPolicy", env, n_steps=64, verbose=0)
    model.learn(total_timesteps=64)
    checkpoint_path = str(tmp_path / "boxworld_64_steps")
    model.save(checkpoint_path)
    # SB3 appends .zip automatically
    return model, checkpoint_path + ".zip", env


@pytest.fixture
def exported_onnx(tmp_path, trained_model):
    """Export the trained model to ONNX and return both paths."""
    model, checkpoint_path, env = trained_model
    onnx_path = str(tmp_path / "boxworld_64_steps.onnx")
    exporter = Exporter()
    exporter.export_checkpoint(checkpoint_path, onnx_path)
    exporter.close()
    return onnx_path, checkpoint_path, model, env


def test_export_produces_onnx_file(exported_onnx):
    onnx_path, _, _, _ = exported_onnx
    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 0


def test_onnx_model_loads_with_onnxruntime(exported_onnx):
    onnx_path, _, _, _ = exported_onnx
    session = ort.InferenceSession(onnx_path)
    assert session is not None
    input_names = [inp.name for inp in session.get_inputs()]
    assert "obs" in input_names


def test_onnx_output_shape(exported_onnx):
    onnx_path, _, _, _ = exported_onnx
    session = ort.InferenceSession(onnx_path)
    obs = np.random.rand(1, 103).astype(np.float32)
    result = session.run(None, {"obs": obs})
    assert result[0].shape == (1, 6)


def test_onnx_matches_pytorch(exported_onnx):
    onnx_path, checkpoint_path, _, _ = exported_onnx
    exporter = Exporter()
    try:
        assert exporter.verify_export(onnx_path, checkpoint_path)
    finally:
        exporter.close()


def test_no_external_data_file(exported_onnx):
    """ONNX export must NOT create a separate .onnx.data file (weights must be inlined)."""
    onnx_path, _, _, _ = exported_onnx
    onnx_dir = os.path.dirname(onnx_path)
    files_in_dir = os.listdir(onnx_dir)
    data_files = [f for f in files_in_dir if f.endswith(".onnx.data")]
    assert data_files == [], f"External data file(s) found — weights are NOT inlined: {data_files}"


def test_onnx_loads_from_isolated_directory(exported_onnx, tmp_path):
    """Copying just the .onnx file to a clean directory must still allow loading + inference.

    This proves the weights are fully embedded and no external files are needed.
    """
    onnx_path, _, _, _ = exported_onnx
    clean_dir = str(tmp_path / "isolated")
    os.makedirs(clean_dir)
    isolated_path = os.path.join(clean_dir, "model.onnx")
    shutil.copy2(onnx_path, isolated_path)

    # Must load without error
    session = ort.InferenceSession(isolated_path)
    obs = np.random.rand(1, 103).astype(np.float32)
    result = session.run(None, {"obs": obs})
    assert result[0].shape == (1, 6), f"Unexpected output shape: {result[0].shape}"


def test_export_all_checkpoints(tmp_path):
    """Given 3 checkpoints in a directory, export_all produces 3 ONNX files."""
    env = BoxworldEnv()
    model = PPO("MlpPolicy", env, n_steps=64, verbose=0)
    model.learn(total_timesteps=64)

    checkpoint_dir = str(tmp_path / "checkpoints")
    output_dir = str(tmp_path / "onnx_output")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save 3 checkpoints with different step counts
    for steps in [1000, 5000, 10000]:
        path = os.path.join(checkpoint_dir, f"boxworld_{steps}_steps")
        model.save(path)

    exporter = Exporter()
    try:
        paths = exporter.export_all(checkpoint_dir, output_dir)
        assert len(paths) == 3
        for p in paths:
            assert os.path.exists(p)
            assert p.endswith(".onnx")
    finally:
        exporter.close()
