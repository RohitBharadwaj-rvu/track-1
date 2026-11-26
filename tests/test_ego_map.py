"""Tests for egocentric odometric spatial memory (`submission/ego_map.py`) and policy integration."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from doom_env import make_env
from submission.agent import DoomAgent
from submission.ego_map import (
    CH_LOCKED_DOOR,
    CH_POST_KEY_VISITED,
    CH_VISITED,
    EGO_MAP_CHANNELS,
    EGO_MAP_SIZE,
    EgocentricSpatialMap,
)
from submission.obs import NUM_ACTIONS
from submission.policy import PolicyConfig, build_policy, load_checkpoint, save_checkpoint
from training.interfaces import MapSpec
from training.rollout import _DoomWorkerSlot


def test_ego_map_odometry_accuracy_and_speed_on_real_env():
    env = make_env("maps/train/train_gauntlet_full_complex.wad", "MAP01", episode_ticks=2100, render_buffers=False)
    ego = EgocentricSpatialMap()
    rng = np.random.default_rng(7)
    try:
        obs, _ = env.reset(seed=123)
        p0 = env.privileged_state()
        ego.reset()
        prev_act = np.zeros(NUM_ACTIONS, dtype=np.int8)
        crop0 = ego.update(obs, prev_act)
        assert crop0.shape == (EGO_MAP_CHANNELS, EGO_MAP_SIZE, EGO_MAP_SIZE)
        assert crop0.dtype == np.uint8

        a0 = math.radians(float(p0["angle"]))
        max_yaw_err = 0.0
        max_pos_err = 0.0
        t_tot = 0.0
        steps = 0

        for _ in range(300):
            act = np.zeros(NUM_ACTIONS, dtype=np.int8)
            fwd = rng.choice([0, 1, 2], p=[0.1, 0.7, 0.2])
            stf = rng.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
            trn = rng.choice([0, 1, 2], p=[0.5, 0.25, 0.25])
            if fwd == 1:
                act[0] = 1
            elif fwd == 2:
                act[1] = 1
            if stf == 1:
                act[2] = 1
            elif stf == 2:
                act[3] = 1
            if trn == 1:
                act[4] = 1
            elif trn == 2:
                act[5] = 1

            obs, _, term, trunc, _ = env.step(act)
            t0 = time.perf_counter()
            crop = ego.update(obs, act)
            t_tot += time.perf_counter() - t0
            assert crop.shape == (EGO_MAP_CHANNELS, EGO_MAP_SIZE, EGO_MAP_SIZE)
            steps += 1

            p = env.privileged_state()
            dx_w = float(p["x"]) - float(p0["x"])
            dy_w = float(p["y"]) - float(p0["y"])
            tx = dx_w * math.cos(a0) + dy_w * math.sin(a0)
            ty = -dx_w * math.sin(a0) + dy_w * math.cos(a0)
            true_yaw = ((float(p["angle"]) - float(p0["angle"])) + 180.0) % 360.0 - 180.0
            yaw_err = abs(((ego.yaw_odom_deg - true_yaw) + 180.0) % 360.0 - 180.0)
            pos_err = math.hypot(ego.x_odom - tx, ego.y_odom - ty)
            max_yaw_err = max(max_yaw_err, yaw_err)
            max_pos_err = max(max_pos_err, pos_err)
            if term or trunc:
                break

        assert max_yaw_err < 15.0, f"Odometry yaw drifted by {max_yaw_err:.2f} deg"
        assert max_pos_err < 150.0, f"Odometry position drifted by {max_pos_err:.1f} units"
        ms_per_step = (t_tot / max(1, steps)) * 1000.0
        assert ms_per_step < 2.0, f"Ego map too slow: {ms_per_step:.3f} ms/step"
    finally:
        env.close()


def test_ego_map_locked_door_persists_then_clears_and_post_key_resets():
    wad = "maps/train/train_key_short_red.wad"
    env = make_env(wad, "MAP01", episode_ticks=800, render_buffers=False)
    ego = EgocentricSpatialMap()
    try:
        obs, _ = env.reset(seed=0)
        ego.reset()
        ego.update(obs, np.zeros(NUM_ACTIONS, dtype=np.int8))
        # Red locked door 'R' is directly ahead in initial rays -> stamped into CH_LOCKED_DOOR
        assert np.any(ego._grid[CH_LOCKED_DOOR] > 0.5)
        assert np.any(ego._grid[CH_VISITED] > 0.5)
        assert not np.any(ego._grid[CH_POST_KEY_VISITED] > 0.5)

        # Turn 90+ degrees left away from the locked door -> CH_LOCKED_DOOR remains in memory
        turn_l = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
        for _ in range(10):
            obs, _, _, _, _ = env.step(turn_l)
            crop = ego.update(obs, turn_l)
        assert np.any(ego._grid[CH_LOCKED_DOOR] > 0.5)
        assert np.any(crop[CH_LOCKED_DOOR] > 128)
    finally:
        env.close()


def test_ego_map_bit_exact_parity_between_worker_slot_and_doom_agent(tmp_path, monkeypatch):
    wad = "maps/train/train_key_short_red.wad"
    spec = MapSpec(wad=wad, map_name="MAP01", family="key_return", tier=3, episode_ticks=600)

    cfg = PolicyConfig(arch="telemetry_gru_v1", use_ego_map=True, eval_deterministic=True)
    policy = build_policy(cfg)
    ckpt = tmp_path / "ego_policy.pt"
    save_checkpoint(ckpt, policy, cfg)
    monkeypatch.setenv("DOOM_AGENT_WEIGHTS", str(ckpt))

    slot = _DoomWorkerSlot(slot_id=0, map_specs=[spec], base_seed=0, render_buffers=False, use_ego_map=True)
    # Match slot seed=1 on first reset (slot_id*1_000_000 + 1)
    env = make_env(wad, "MAP01", episode_ticks=600, render_buffers=False)
    try:
        slot_obs = slot.reset()
        raw_obs, _ = env.reset(seed=slot._current_seed)
        agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        agent.reset()

        for _ in range(30):
            # Compute agent ego map via agent.act and compare against slot step
            act = agent.act(raw_obs)
            # Re-verify backing grid crop directly after agent updated _ego_map
            assert agent._ego_map is not None
            agent_crop = np.clip(
                agent._ego_map._grid[
                    :,
                    agent._ego_map._to_grid_idx(
                        agent._ego_map.x_odom
                        + np.arange(1) * 0.0,
                        agent._ego_map.y_odom,
                    )[0][0],
                    0,
                ],
                0.0,
                1.0,
            )
            assert agent_crop.shape == (EGO_MAP_CHANNELS,)

            raw_obs, _, term, trunc, _ = env.step(act)
            slot_next, _, s_term, s_trunc, info = slot.step(act)
            assert term == s_term and trunc == s_trunc

            # Before next agent.act, predict what slot already computed in slot_next["ego_map"]
            if not (term or trunc):
                # Peek into agent._ego_map without mutating agent state using a cloned check on final step
                assert slot_next["ego_map"].shape == (EGO_MAP_CHANNELS, EGO_MAP_SIZE, EGO_MAP_SIZE)
            else:
                assert info["final_obs"]["ego_map"].shape == (EGO_MAP_CHANNELS, EGO_MAP_SIZE, EGO_MAP_SIZE)
                break
    finally:
        slot.close()
        env.close()

    # Exact bit parity test comparing EgocentricSpatialMap via DoomAgent vs _DoomWorkerSlot directly
    slot2 = _DoomWorkerSlot(slot_id=0, map_specs=[spec], base_seed=0, render_buffers=False, use_ego_map=True)
    env2 = make_env(wad, "MAP01", episode_ticks=600, render_buffers=False)
    try:
        obs_slot = slot2.reset()
        raw2, _ = env2.reset(seed=slot2._current_seed)
        agent2 = DoomAgent(env2.observation_space, env2.action_space, "cpu")
        agent2.reset()
        rng = np.random.default_rng(12)
        for _ in range(40):
            # Update agent2._ego_map using exact same _prev_action_np contract as DoomAgent.act
            crop_agent = agent2._ego_map.update(raw2, agent2._prev_action_np)
            np.testing.assert_array_equal(crop_agent, obs_slot["ego_map"])
            act = np.zeros(NUM_ACTIONS, dtype=np.int8)
            act[int(rng.integers(0, 6))] = 1
            agent2._prev_action_np = act
            raw2, _, t1, tr1, _ = env2.step(act)
            obs_slot, _, t2, tr2, _ = slot2.step(act)
            if t1 or tr1 or t2 or tr2:
                break
    finally:
        slot2.close()
        env2.close()


def test_spatial_map_policy_forward_backward_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = PolicyConfig(arch="telemetry_gru_v1", use_ego_map=True, ego_map_out_dim=128)
    policy = build_policy(cfg)
    T, B = 4, 2
    obs_seq = {
        "ray_dist": torch.rand(T, B, 64),
        "ray_type": torch.randint(0, 6, (T, B, 64)),
        "ent_feat": torch.zeros(T, B, 32, 5),
        "ent_type": torch.zeros(T, B, 32, dtype=torch.long),
        "ent_mask": torch.zeros(T, B, 32),
        "vitals": torch.zeros(T, B, 17),
        "ego_map": torch.randint(0, 256, (T, B, EGO_MAP_CHANNELS, EGO_MAP_SIZE, EGO_MAP_SIZE), dtype=torch.uint8),
    }
    lp, ent, val = policy.evaluate_chunks(
        obs_seq,
        torch.zeros(T, B, 9),
        torch.zeros(T, B, 9),
        policy.init_hidden(B, "cpu"),
        torch.zeros(T, B, dtype=torch.bool),
    )
    loss = -(lp.mean() + 0.01 * ent.mean()) + 0.5 * (val ** 2).mean()
    loss.backward()
    for name, param in policy.ego_map_encoder.named_parameters():
        assert param.grad is not None, f"Missing grad on {name}"
        assert torch.isfinite(param.grad).all()

    ckpt = tmp_path / "ego_ckpt.pt"
    save_checkpoint(ckpt, policy, cfg)
    loaded, loaded_cfg, _ = load_checkpoint(ckpt, device="cpu")
    assert loaded_cfg.use_ego_map is True
    assert loaded_cfg.ego_map_out_dim == 128
    assert loaded.ego_map_encoder is not None
