"""Tests for training-only geodesic PBRS (`training/pbrs.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from doom_env import make_env
from training.maps import TRAIN_MAPS, generate_training_variations
from training.pbrs import GeodesicPBRS, _build_nav_graph
from training.wad_gen.udmf import build_grid_wad


def test_pbrs_parses_all_20_train_wads_with_reachable_goals():
    for spec in TRAIN_MAPS:
        graph = _build_nav_graph(spec.wad, spec.map_name)
        assert graph is not None, f"Failed to parse nav graph for {spec.wad}"
        assert len(graph.exit_cells) >= 1, f"No exit cells found for {spec.wad}"

        pbrs = GeodesicPBRS(spec.wad, spec.map_name, scale=2.0)
        env = make_env(spec.wad, spec.map_name, episode_ticks=200, render_buffers=False)
        try:
            env.reset(seed=0)
            priv = env.privileged_state()
            pbrs.reset(priv)
            assert pbrs._prev_stage_key != ("none",), f"No reachable initial target on {spec.wad}"
        finally:
            env.close()


def test_pbrs_parses_100_percent_of_procedural_wads_with_reachable_goals(tmp_path):
    proc_specs = generate_training_variations(tmp_path, count_per_family=2, base_seed=3000)
    assert len(proc_specs) == 8
    for spec in proc_specs:
        graph = _build_nav_graph(spec.wad, spec.map_name)
        assert graph is not None, f"Failed to parse nav graph for procedural WAD {spec.wad}"
        assert len(graph.exit_cells) >= 1, f"No exit cells found for procedural WAD {spec.wad}"

        pbrs = GeodesicPBRS(spec.wad, spec.map_name, scale=2.0)
        env = make_env(spec.wad, spec.map_name, episode_ticks=200, render_buffers=False)
        try:
            env.reset(seed=0)
            pbrs.reset(env.privileged_state())
            assert pbrs._prev_stage_key != ("none",), f"No reachable initial target on procedural WAD {spec.wad}"
        finally:
            env.close()


def test_pbrs_gives_positive_shaping_moving_toward_key_and_exit_and_resets_on_stage_switch(tmp_path):
    # Use a sealed key-return map where Red lock R truly gates the exit E
    wad = str(
        build_grid_wad(
            tmp_path / "sealed_key.wad",
            "###########\n#P..r#....#\n#....R..E.#\n#....#....#\n###########",
            cell_size=64,
            player_angle=0,
        )
    )
    pbrs = GeodesicPBRS(wad, "MAP01", scale=2.0, gamma=0.995)
    env = make_env(wad, "MAP01", episode_ticks=600, render_buffers=False)
    try:
        env.reset(seed=0)
        p0 = env.privileged_state()
        pbrs.reset(p0)
        assert pbrs._prev_stage_key is not None and pbrs._prev_stage_key[0] == "key"

        # Walking 3 steps toward Red key moves closer without reaching it yet (positive PBRS)
        fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        rewards = []
        for _ in range(3):
            env.step(fwd)
            rewards.append(pbrs.step(env.privileged_state()))
        assert sum(rewards) > 0.0
        assert pbrs._prev_stage_key[0] == "key"

        # Picking up the Red key switches stage cleanly from 'key' to 'exit' with 0.0 spike
        p_key = dict(env.privileged_state())
        assert p_key["keys"][2] < 0.5
        p_key["keys"] = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        switch_rew = pbrs.step(p_key)
        assert switch_rew == pytest.approx(0.0)
        assert pbrs._prev_stage_key == ("exit",)
    finally:
        env.close()


def test_pbrs_fails_safe_on_missing_wad():
    pbrs = GeodesicPBRS("nonexistent_map_xyz.wad", "MAP01", scale=2.0)
    assert pbrs.graph is None
    pbrs.reset({"x": 0.0, "y": 0.0, "keys": np.zeros(6)})
    assert pbrs.step({"x": 64.0, "y": 0.0, "keys": np.zeros(6)}) == 0.0
