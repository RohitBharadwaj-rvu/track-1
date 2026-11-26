"""TDD tests for Role 2 vectorized envs, GAE, and rollout buffer (`training/rollout.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from fixtures import fixture_wad
from training.interfaces import EpisodeMetrics, MapSpec
from training.rollout import RolloutBuffer, SubprocVecDoomEnv, SyncVecDoomEnv, compute_gae


def test_sync_vec_env_truncation_final_obs_metrics_and_autoreset():
    wad = fixture_wad("#####\n#P..#\n#####")
    spec = MapSpec(wad=wad, map_name="MAP01", family="multi_room", tier=2, split="train", episode_ticks=40)
    noop = np.zeros((1, 9), dtype=np.int8)

    with SyncVecDoomEnv([spec], base_seed=10, render_buffers=False, num_envs=1) as vec:
        obs0 = vec.reset()
        assert obs0["ray_dist"].shape == (1, 64)
        assert obs0["vitals"].shape == (1, 17)

        for step_idx in range(1, 10):
            obs, rewards, term, trunc, infos = vec.step(noop)
            assert not term[0] and not trunc[0]
            assert "episode" not in infos[0]

        # Step 10 hits episode_ticks=40 (FRAME_SKIP=4 -> 10 decision steps)
        obs10, rewards10, term10, trunc10, infos10 = vec.step(noop)
        assert not term10[0]
        assert trunc10[0]
        assert "final_obs" in infos10[0]
        assert infos10[0]["final_obs"]["ray_dist"].shape == (64,)
        ep: EpisodeMetrics = infos10[0]["episode"]
        assert ep.outcome == "timeout"
        assert ep.steps == 10
        assert ep.unique_cells_visited >= 1
        assert obs10["ray_dist"].shape == (1, 64)

        # Step 11 runs seamlessly in the auto-reset episode
        obs11, _, term11, trunc11, infos11 = vec.step(noop)
        assert not term11[0] and not trunc11[0]
        assert obs11["ray_dist"].shape == (1, 64)


def test_subproc_vec_env_steps_and_closes_cleanly():
    spec = MapSpec(wad="maps/two_rooms.wad", map_name="MAP01", family="door_exit", tier=1, split="train", episode_ticks=200)
    rng = np.random.default_rng(0)

    with SubprocVecDoomEnv([spec], base_seed=0, render_buffers=False, num_envs=2) as vec:
        obs = vec.reset()
        assert obs["ray_dist"].shape == (2, 64)
        assert obs["ent_feat"].shape == (2, 32, 5)
        for _ in range(5):
            actions = rng.integers(0, 2, size=(2, 9), dtype=np.int8)
            obs, rewards, term, trunc, infos = vec.step(actions)
            assert rewards.shape == (2,)
            assert np.all(np.isfinite(obs["ray_dist"]))
            assert np.all(np.isfinite(rewards))
            assert len(infos) == 2


def test_compute_gae_distinguishes_termination_and_truncation():
    # Single env, 3 steps, gamma=0.9, lambda=0.5, last_value=10.0
    # Case A: unterminated -> bootstraps values[t+1] and last_value
    rewards = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    values = np.array([[0.5], [1.0], [1.5]], dtype=np.float32)
    terminated = np.array([[False], [False], [False]])
    truncated = np.array([[False], [False], [False]])
    boot = np.zeros((3, 1), dtype=np.float32)
    last_v = np.array([10.0], dtype=np.float32)

    adv_a, ret_a = compute_gae(rewards, values, terminated, truncated, boot, last_v, gamma=0.9, gae_lambda=0.5)
    # t=2: delta2 = 3.0 + 0.9*10.0 - 1.5 = 10.5 -> adv2 = 10.5
    # t=1: delta1 = 2.0 + 0.9*1.5 - 1.0 = 2.35 -> adv1 = 2.35 + 0.45*10.5 = 7.075
    # t=0: delta0 = 1.0 + 0.9*1.0 - 0.5 = 1.4  -> adv0 = 1.4 + 0.45*7.075 = 4.58375
    np.testing.assert_allclose(adv_a[:, 0], [4.58375, 7.075, 10.5], atol=1e-5)
    np.testing.assert_allclose(ret_a[:, 0], adv_a[:, 0] + values[:, 0], atol=1e-5)

    # Case B: mid-rollout termination at t=1 -> next_val=0 at t=1, AND no GAE bleed from t=2 into t=1
    term_b = np.array([[False], [True], [False]])
    adv_b, _ = compute_gae(rewards, values, term_b, truncated, boot, last_v, gamma=0.9, gae_lambda=0.5)
    # t=2: 10.5
    # t=1: delta1 = 2.0 + 0 - 1.0 = 1.0 (done=True stops t=2 GAE bleed)
    # t=0: delta0 = 1.4 + 0.45*1.0 = 1.85
    np.testing.assert_allclose(adv_b[:, 0], [1.85, 1.0, 10.5], atol=1e-5)

    # Case C: mid-rollout truncation at t=1 with bootstrap_value=4.0 -> bootstraps 4.0, still cuts GAE bleed
    trunc_c = np.array([[False], [True], [False]])
    boot_c = np.array([[0.0], [4.0], [0.0]], dtype=np.float32)
    adv_c, _ = compute_gae(rewards, values, terminated, trunc_c, boot_c, last_v, gamma=0.9, gae_lambda=0.5)
    # t=1: delta1 = 2.0 + 0.9*4.0 - 1.0 = 4.6 (done=True stops t=2 GAE bleed)
    # t=0: 1.4 + 0.45*4.6 = 3.47
    np.testing.assert_allclose(adv_c[:, 0], [3.47, 4.6, 10.5], atol=1e-5)


def test_rollout_buffer_chunk_minibatches_preserve_order_and_hidden_alignment():
    T, N, H = 8, 2, 4
    buf = RolloutBuffer(num_steps=T, num_envs=N, hidden_shape=(1, N, H))
    for t in range(T):
        # Tag each (t, env) uniquely in rewards = env * 100 + t and hidden[:, e, 0] = env * 100 + t
        rew = np.array([0 * 100 + t, 1 * 100 + t], dtype=np.float32)
        hid = np.zeros((1, N, H), dtype=np.float32)
        hid[0, 0, 0] = rew[0]
        hid[0, 1, 0] = rew[1]
        obs = {
            "ray_dist": np.full((N, 64), float(t), dtype=np.float32),
            "ray_type": np.zeros((N, 64), dtype=np.int64),
            "ent_feat": np.zeros((N, 32, 5), dtype=np.float32),
            "ent_type": np.zeros((N, 32), dtype=np.int64),
            "ent_mask": np.zeros((N, 32), dtype=np.float32),
            "vitals": np.zeros((N, 17), dtype=np.float32),
        }
        buf.add(
            step=t,
            obs_batch=obs,
            prev_action=np.zeros((N, 9), dtype=np.float32),
            action=np.zeros((N, 9), dtype=np.float32),
            log_prob=np.zeros(N, dtype=np.float32),
            reward=rew,
            value=np.zeros(N, dtype=np.float32),
            terminated=np.zeros(N, dtype=bool),
            truncated=np.zeros(N, dtype=bool),
            bootstrap_value=np.zeros(N, dtype=np.float32),
            episode_start=np.array([t == 0, t == 0], dtype=bool),
            hidden=hid,
        )
    buf.compute_returns_and_advantages(np.zeros(N, dtype=np.float32))

    rng = np.random.default_rng(42)
    batches = list(buf.iter_chunk_minibatches(chunk_len=4, num_minibatches=2, rng=rng))
    assert len(batches) == 2

    seen_starts = set()
    for mb in batches:
        assert mb["obs"]["ray_dist"].shape == (4, 2, 64)
        assert mb["init_hidden"].shape == (1, 2, H)
        for b in range(mb["init_hidden"].shape[1]):
            start_tag = float(mb["init_hidden"][0, b, 0])
            seen_starts.add(start_tag)
            # Chunk time steps must be consecutive: start_tag, start_tag+1, start_tag+2, start_tag+3
            np.testing.assert_allclose(mb["obs"]["ray_dist"][:, b, 0], np.arange(4) + (start_tag % 100))

    assert seen_starts == {0.0, 4.0, 100.0, 104.0}
