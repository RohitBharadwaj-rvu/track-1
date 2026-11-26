"""Tests for exact left-right mirror symmetry with EgocentricSpatialMap & SyncVecDoomEnv."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from submission.obs import NUM_ACTIONS
from training.interfaces import MapSpec
from training.rollout import SyncVecDoomEnv
from training.symmetry import (
    mirror_action,
    mirror_aux_targets,
    mirror_preprocessed_obs,
    mirror_priv_features,
    mirror_raw_obs,
)
from training.wad_gen.udmf import build_grid_wad


def test_mirror_transforms_are_exact_involutions():
    rng = np.random.default_rng(11)
    act = rng.integers(0, 2, size=(4, NUM_ACTIONS), dtype=np.int8)
    np.testing.assert_array_equal(mirror_action(mirror_action(act)), act)

    aux = rng.normal(size=(4, 4)).astype(np.float32)
    np.testing.assert_array_equal(mirror_aux_targets(mirror_aux_targets(aux)), aux)

    priv = rng.normal(size=(4, 16)).astype(np.float32)
    np.testing.assert_array_equal(mirror_priv_features(mirror_priv_features(priv)), priv)


def test_sync_vec_env_mirror_with_ego_map_produces_exact_mirror_twin(tmp_path: Path):
    wad_path = tmp_path / "asym_ego_mirror.wad"
    build_grid_wad(
        wad_path,
        """
        #######
        #...r.#
        #.P...#
        #######
        """,
        player_angle=0,
    )
    spec = MapSpec(wad=str(wad_path), map_name="MAP01", family="multi_room", tier=2, split="train", episode_ticks=200)
    turn_left = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.int8)
    turn_right = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.int8)

    with SyncVecDoomEnv([spec], base_seed=5, num_envs=1, use_ego_map=True, mirror_prob=0.0) as env_plain, SyncVecDoomEnv(
        [spec], base_seed=5, num_envs=1, use_ego_map=True, mirror_prob=1.0
    ) as env_mirrored:
        obs_p = env_plain.reset()
        obs_m = env_mirrored.reset()

        expected_m0 = mirror_preprocessed_obs({k: v[0] for k, v in obs_p.items()})
        np.testing.assert_allclose(obs_m["ray_dist"][0], expected_m0["ray_dist"], atol=1e-5)
        np.testing.assert_allclose(obs_m["ent_feat"][0], expected_m0["ent_feat"], atol=1e-5)
        assert float(obs_p["ent_feat"][0, 0, 1]) > 0.0
        assert float(obs_m["ent_feat"][0, 0, 1]) < 0.0
        # Ego map crop left/right mirror check (key marker flips col <-> 24 - col)
        np.testing.assert_array_equal(obs_m["ego_map"][0], expected_m0["ego_map"])

        next_p, _, _, _, info_p = env_plain.step(turn_left)
        next_m, _, _, _, info_m = env_mirrored.step(turn_right)
        assert info_p[0]["mirrored"] is False
        assert info_m[0]["mirrored"] is True

        expected_m1 = mirror_preprocessed_obs({k: v[0] for k, v in next_p.items()})
        np.testing.assert_allclose(next_m["ray_dist"][0], expected_m1["ray_dist"], atol=1e-4)
        np.testing.assert_allclose(next_m["ent_feat"][0], expected_m1["ent_feat"], atol=1e-4)
        np.testing.assert_allclose(next_m["vitals"][0], expected_m1["vitals"], atol=1e-4)
