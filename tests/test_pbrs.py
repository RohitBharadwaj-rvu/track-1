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
    proc_specs = generate_training_variations(tmp_path, count_per_family=4, base_seed=3000)
    assert len(proc_specs) == 20
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


def test_pbrs_handles_exit_switch_and_height_blocked_windows_and_cliffs():
    # 1. One-sided wall exit switch must nudge target cell inside the room so it is reachable
    sw_pbrs = GeodesicPBRS("maps/stress/exit_switch.wad", "MAP01", scale=2.0, gamma=1.0)
    assert sw_pbrs.graph is not None
    env = make_env("maps/stress/exit_switch.wad", "MAP01", episode_ticks=200, render_buffers=False)
    try:
        env.reset(seed=0)
        sw_pbrs.reset(env.privileged_state())
        assert sw_pbrs._prev_stage_key == ("exit",)
        fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        shaped = []
        for _ in range(15):
            env.step(fwd)
            shaped.append(sw_pbrs.step(env.privileged_state()))
        assert sum(shaped) > 0.0
    finally:
        env.close()

    # 2. Low-ceiling window (40u < 56u) must block PBRS routing -> longer geodesic path than open room
    p_open = GeodesicPBRS("maps/stress/key_open.wad", "MAP01", scale=2.0, gamma=1.0)
    p_win = GeodesicPBRS("maps/stress/key_window.wad", "MAP01", scale=2.0, gamma=1.0)
    init_priv = {"x": 80.0, "y": 80.0, "keys": np.zeros(6, dtype=np.float32)}
    p_open.reset(init_priv)
    p_win.reset(init_priv)
    assert p_open._prev_stage_key is not None and p_open._prev_stage_key[0] == "key"
    assert p_win._prev_stage_key is not None and p_win._prev_stage_key[0] == "key"
    assert p_win._prev_phi < p_open._prev_phi

    # 3. 48u cliff (> 24u) must block direct path and route up the 3-step staircase
    p_stairs = GeodesicPBRS("maps/stress/stairs_ledge.wad", "MAP01", scale=2.0, gamma=1.0)
    p_stairs.reset(init_priv)
    assert p_stairs._prev_stage_key is not None and p_stairs._prev_stage_key[0] == "key"
    # Moving north onto the stairs must strictly increase potential (strictly shorter detour to key)
    step_north_rew = p_stairs.step({"x": 80.0, "y": 240.0, "keys": np.zeros(6, dtype=np.float32)})
    assert step_north_rew > 0.0

    # 4. Impassable mid-grate must route around north flank
    p_grate = GeodesicPBRS("maps/stress/grate_combat.wad", "MAP01", scale=2.0, gamma=1.0)
    p_grate.reset({"x": 80.0, "y": 128.0, "keys": np.zeros(6, dtype=np.float32)})
    assert p_grate._prev_stage_key == ("exit",)
    assert p_grate.step({"x": 160.0, "y": 336.0, "keys": np.zeros(6, dtype=np.float32)}) > 0.0


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
    pbrs = GeodesicPBRS(wad, "MAP01", scale=2.0, gamma=1.0)
    env = make_env(wad, "MAP01", episode_ticks=600, render_buffers=False)
    try:
        env.reset(seed=0)
        p0 = env.privileged_state()
        pbrs.reset(p0)
        assert pbrs._prev_stage_key is not None and pbrs._prev_stage_key[0] == "key"

        # Walking toward Red key moves closer across 32u grid cells without picking up yet (positive PBRS)
        fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        rewards = []
        for _ in range(10):
            env.step(fwd)
            priv = env.privileged_state()
            if float(priv["keys"][2]) > 0.5:
                break
            rewards.append(pbrs.step(priv))
        assert len(rewards) >= 2
        assert sum(rewards) > 0.0
        assert pbrs._prev_stage_key[0] == "key"

        # Picking up the Red key switches stage cleanly from 'key' to 'exit' with 0.0 spike
        p_key = dict(env.privileged_state())
        if float(p_key["keys"][2]) < 0.5:
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


def test_pbrs_routes_around_impassable_window_and_reaches_wall_exit_switches():
    # key_window has an impassable height_ceiling=40 window between spawn and blue skull -> strictly longer geodesic than key_open
    p_open = GeodesicPBRS("maps/stress/key_open.wad", "MAP01", scale=3.0, gamma=1.0)
    p_win = GeodesicPBRS("maps/stress/key_window.wad", "MAP01", scale=3.0, gamma=1.0)
    env_open = make_env("maps/stress/key_open.wad", "MAP01", episode_ticks=200, render_buffers=False)
    env_win = make_env("maps/stress/key_window.wad", "MAP01", episode_ticks=200, render_buffers=False)
    try:
        env_open.reset(seed=0)
        env_win.reset(seed=0)
        p_open.reset(env_open.privileged_state())
        p_win.reset(env_win.privileged_state())
        assert p_open._prev_stage_key == ("key", ())
        assert p_win._prev_stage_key == ("key", ())
        # Higher detour distance (+~98u around y=160 window) -> strictly lower potential on key_window
        assert p_win._prev_phi < p_open._prev_phi - 0.05
    finally:
        env_open.close()
        env_win.close()

    # Wall exit switch must also have a reachable exit potential
    p_sw = GeodesicPBRS("maps/stress/exit_switch.wad", "MAP01", scale=3.0, gamma=1.0)
    env_sw = make_env("maps/stress/exit_switch.wad", "MAP01", episode_ticks=200, render_buffers=False)
    try:
        env_sw.reset(seed=0)
        p_sw.reset(env_sw.privileged_state())
        assert p_sw._prev_stage_key == ("exit",)
        assert p_sw._prev_phi > 0.0
    finally:
        env_sw.close()


def test_pbrs_never_rewards_standing_still(tmp_path):
    wad = str(
        build_grid_wad(
            tmp_path / "still.wad",
            "###########\n#P..r#....#\n#....R..E.#\n#....#....#\n###########",
            cell_size=64,
            player_angle=0,
        )
    )
    env = make_env(wad, "MAP01", episode_ticks=200, render_buffers=False)
    try:
        env.reset(seed=0)
        p0 = env.privileged_state()
    finally:
        env.close()

    pbrs_exact = GeodesicPBRS(wad, "MAP01", scale=3.0, gamma=1.0)
    pbrs_exact.reset(p0)
    assert pbrs_exact._prev_phi > 0.0
    assert pbrs_exact.step(p0) == pytest.approx(0.0)

    pbrs_disc = GeodesicPBRS(wad, "MAP01", scale=3.0, gamma=0.995)
    pbrs_disc.reset(p0)
    assert pbrs_disc.step(p0) <= 0.0
