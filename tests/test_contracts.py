"""Tests for the shared parallel-development contracts (Roles 1–4)."""

from __future__ import annotations

import numpy as np

from doom_env import make_env
from submission.obs import (
    MAX_ENTITIES,
    NUM_ACTIONS,
    NUM_ENTITY_FEATURES,
    NUM_RAYS,
    NUM_VITALS,
    batch_preprocessed_obs,
    preprocess_obs,
)
from submission.policy import PolicyConfig
from training.interfaces import EpisodeMetrics, MapSpec, PrivilegedEventTracker


def test_preprocess_obs_on_real_doom_env():
    env = make_env("maps/two_rooms.wad", "MAP01", episode_ticks=100, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        p0 = preprocess_obs(obs)
        assert p0["ray_dist"].shape == (NUM_RAYS,)
        assert p0["ray_type"].shape == (NUM_RAYS,)
        assert p0["ent_feat"].shape == (MAX_ENTITIES, NUM_ENTITY_FEATURES)
        assert p0["ent_type"].shape == (MAX_ENTITIES,)
        assert p0["ent_mask"].shape == (MAX_ENTITIES,)
        assert p0["vitals"].shape == (NUM_VITALS,)
        for arr in p0.values():
            assert np.all(np.isfinite(arr))

        obs1, _, _, _, _ = env.step(np.zeros(NUM_ACTIONS, dtype=np.int8))
        p1 = preprocess_obs(obs1)
        batched = batch_preprocessed_obs([p0, p1])
        assert batched["ray_dist"].shape == (2, NUM_RAYS)
        assert batched["ent_feat"].shape == (2, MAX_ENTITIES, NUM_ENTITY_FEATURES)
        assert batched["vitals"].shape == (2, NUM_VITALS)
    finally:
        env.close()


def test_preprocess_obs_empty_entities_zeroed():
    obs = {
        "rays": {
            "distance": np.ones(NUM_RAYS, dtype=np.float32),
            "hit_type": np.zeros(NUM_RAYS, dtype=np.int64),
        },
        "visible_entities": {
            "features": np.full((MAX_ENTITIES, NUM_ENTITY_FEATURES), 5.0, dtype=np.float32),
            "entity_type": np.zeros(MAX_ENTITIES, dtype=np.int64),
            "mask": np.zeros(MAX_ENTITIES, dtype=np.int8),
        },
        "vitals": np.zeros(NUM_VITALS, dtype=np.float32),
    }
    p = preprocess_obs(obs)
    assert np.all(p["ent_feat"] == 0.0)
    assert np.all(p["ent_mask"] == 0.0)


def test_policy_config_roundtrip():
    cfg = PolicyConfig(arch="telemetry_gru_v1", hidden_size=128, num_memory_slots=8)
    restored = PolicyConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_privileged_event_tracker_distinguishes_events():
    tracker = PrivilegedEventTracker()
    base = {
        "x": 0.0,
        "y": 0.0,
        "kills": 0,
        "damage_taken": 0.0,
        "keys": np.zeros(6, dtype=np.int32),
        "exit_reached": False,
        "dead": False,
        "timeout": False,
        "sector_ceiling": np.array([0.0, 0.0, 128.0]),
        "door_lock": np.array([0, 1, 0]),
    }
    tracker.reset(base)

    # Pickup key + open unlocked door (sector 0) + move to new cell
    s1 = {
        **base,
        "x": 128.0,
        "keys": np.array([1, 0, 0, 0, 0, 0], dtype=np.int32),
        "sector_ceiling": np.array([64.0, 0.0, 128.0]),
    }
    ev1, new_cell1 = tracker.update(s1)
    assert ev1.keys_picked_up == 1
    assert ev1.doors_opened == 1
    assert ev1.locked_doors_unlocked == 0
    assert new_cell1 is True
    assert tracker.unique_cells_visited == 2

    # Unlock locked door (sector 1) + kill + exit; unlocked door 0 should not double-count
    s2 = {
        **s1,
        "kills": 2,
        "damage_taken": 15.0,
        "sector_ceiling": np.array([96.0, 64.0, 128.0]),
        "exit_reached": True,
    }
    ev2, new_cell2 = tracker.update(s2)
    assert ev2.keys_picked_up == 0
    assert ev2.doors_opened == 0
    assert ev2.locked_doors_unlocked == 1
    assert ev2.kills == 2
    assert ev2.damage_taken == 15.0
    assert ev2.exit_reached is True
    assert new_cell2 is False


def test_map_spec_and_episode_metrics_defaults():
    spec = MapSpec(wad="maps/two_rooms.wad", family="door_exit", tier=1, split="dev")
    metrics = EpisodeMetrics(wad=spec.wad, map_name=spec.map_name, family=spec.family, tier=spec.tier, seed=0)
    assert spec.episode_ticks == 2100
    assert metrics.outcome == "timeout"
