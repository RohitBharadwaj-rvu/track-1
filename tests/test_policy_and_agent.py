"""Tests for Role 1 telemetry policy, frozen Role 1/2 contract parity, checkpoint I/O, and DoomAgent."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env
from doom_env.sensors import RAY_LOCKED
from fixtures import fixture_wad
from submission.agent import DoomAgent
from submission.obs import (
    MAX_ENTITIES,
    NUM_ACTIONS,
    NUM_ENTITY_FEATURES,
    NUM_RAYS,
    NUM_VITALS,
    batch_preprocessed_obs,
    obs_to_torch,
    preprocess_obs,
)
from submission.policy import PolicyConfig, build_policy, load_checkpoint, save_checkpoint


def _random_preprocessed_seq(
    t_steps: int,
    batch_size: int,
    *,
    empty_steps: tuple[int, ...] = (),
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    ray_dist = torch.rand(t_steps, batch_size, NUM_RAYS, generator=gen)
    ray_type = torch.randint(0, 6, (t_steps, batch_size, NUM_RAYS), generator=gen)
    ent_feat = torch.randn(t_steps, batch_size, MAX_ENTITIES, NUM_ENTITY_FEATURES, generator=gen)
    ent_feat[..., 0] = torch.abs(ent_feat[..., 0]) * 2.0
    ent_feat[..., 1:3] = torch.clamp(ent_feat[..., 1:3], -1.0, 1.0)
    ent_type = torch.randint(1, 12, (t_steps, batch_size, MAX_ENTITIES), generator=gen)
    ent_mask = torch.randint(0, 2, (t_steps, batch_size, MAX_ENTITIES), generator=gen).to(torch.float32)
    for t in empty_steps:
        ent_mask[t] = 0.0
    ent_feat = ent_feat * ent_mask.unsqueeze(-1)
    vitals = torch.randn(t_steps, batch_size, NUM_VITALS, generator=gen)
    vitals[..., 0:2] = torch.rand(t_steps, batch_size, 2, generator=gen) * 2.0
    vitals[..., 2] = torch.randint(0, 8, (t_steps, batch_size), generator=gen).to(torch.float32) / 7.0
    vitals[..., 3:8] = torch.rand(t_steps, batch_size, 5, generator=gen) * 0.5
    vitals[..., 8:14] = torch.randint(0, 2, (t_steps, batch_size, 6), generator=gen).to(torch.float32)
    return {
        "ray_dist": ray_dist,
        "ray_type": ray_type,
        "ent_feat": ent_feat,
        "ent_type": ent_type,
        "ent_mask": ent_mask,
        "vitals": vitals,
    }


def test_preprocessor_real_env_obs_ignores_pixels():
    env = make_env(fixture_wad("#########\n#P..h..z#\n#########"), render_buffers=False)
    try:
        obs0, _ = env.reset(seed=0)
        obs1, *_ = env.step(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8))
    finally:
        env.close()

    p0 = preprocess_obs(obs0)
    assert p0["ray_dist"].shape == (NUM_RAYS,) and p0["ray_dist"].dtype == np.float32
    assert p0["ray_type"].shape == (NUM_RAYS,) and p0["ray_type"].dtype == np.int64
    assert p0["ent_feat"].shape == (MAX_ENTITIES, NUM_ENTITY_FEATURES)
    assert p0["ent_type"].shape == (MAX_ENTITIES,) and p0["ent_type"].dtype == np.int64
    assert p0["ent_mask"].shape == (MAX_ENTITIES,) and p0["ent_mask"].dtype == np.float32
    assert p0["vitals"].shape == (NUM_VITALS,) and p0["vitals"].dtype == np.float32

    obs0_no_px = {k: copy.deepcopy(v) for k, v in obs0.items() if k not in ("rgb", "depth")}
    p0_no_px = preprocess_obs(obs0_no_px)
    for k in p0:
        np.testing.assert_array_equal(p0[k], p0_no_px[k])

    batched = obs_to_torch(batch_preprocessed_obs([p0, preprocess_obs(obs1)]), device="cpu")
    assert batched["ray_dist"].shape == (2, NUM_RAYS)
    assert batched["ent_feat"].shape == (2, MAX_ENTITIES, NUM_ENTITY_FEATURES)
    assert batched["vitals"].shape == (2, NUM_VITALS)


def test_empty_and_full_entities_no_nan_forward_and_backward():
    torch.manual_seed(7)
    policy = build_policy(PolicyConfig(arch="telemetry_gru_v1"))
    T, B = 4, 2
    obs_seq = _random_preprocessed_seq(T, B, seed=7)
    obs_seq["ent_mask"][:] = 0.0
    obs_seq["ent_mask"][:, 1] = 1.0  # row 0 completely empty; row 1 completely full
    obs_seq["ent_feat"] = obs_seq["ent_feat"] * obs_seq["ent_mask"].unsqueeze(-1)

    prev_actions = torch.zeros(T, B, NUM_ACTIONS, dtype=torch.float32)
    actions = torch.randint(0, 2, (T, B, NUM_ACTIONS), dtype=torch.float32)
    starts = torch.zeros(T, B, dtype=torch.bool)
    init_hidden = policy.init_hidden(B, "cpu")

    log_prob, entropy, value = policy.evaluate_chunks(obs_seq, prev_actions, actions, init_hidden, starts)
    for tensor in (log_prob, entropy, value):
        assert torch.isfinite(tensor).all()

    loss = -(log_prob.mean() + 0.01 * entropy.mean()) + 0.5 * (value**2).mean()
    loss.backward()

    assert policy.entity_encoder.empty_embed.grad is not None
    assert torch.isfinite(policy.entity_encoder.empty_embed.grad).all()
    assert policy.entity_encoder.empty_embed.grad.abs().sum() > 0

    for name, param in policy.named_parameters():
        assert param.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_directional_left_right_sensitivity():
    torch.manual_seed(11)
    policy = build_policy(PolicyConfig(arch="telemetry_gru_v1"))
    policy.eval()

    base = _random_preprocessed_seq(1, 1, empty_steps=(0,), seed=11)
    base = {k: v.squeeze(0) for k, v in base.items()}
    base["ray_dist"].fill_(1.0)
    base["ray_type"].zero_()

    left_obs = {k: v.clone() for k, v in base.items()}
    right_obs = {k: v.clone() for k, v in base.items()}

    # Index 16 is 90 deg left; index 48 is 90 deg right.
    left_obs["ray_dist"][0, 16] = 0.25
    left_obs["ray_type"][0, 16] = RAY_LOCKED
    right_obs["ray_dist"][0, 48] = 0.25
    right_obs["ray_type"][0, 48] = RAY_LOCKED

    hidden = policy.init_hidden(1, "cpu")
    prev_action = torch.zeros(1, NUM_ACTIONS, dtype=torch.float32)
    with torch.no_grad():
        logits_left, _, _ = policy.act_step(left_obs, prev_action, hidden)
        logits_right, _, _ = policy.act_step(right_obs, prev_action, hidden)

    diff = (logits_left - logits_right).abs().max().item()
    assert diff > 1e-3, f"expected left vs right locked door logits to differ, got max diff {diff}"


def test_prev_action_affects_logits():
    torch.manual_seed(19)
    policy = build_policy(PolicyConfig(arch="telemetry_gru_v1"))
    policy.eval()

    obs = {k: v.squeeze(0) for k, v in _random_preprocessed_seq(1, 1, seed=19).items()}
    hidden = policy.init_hidden(1, "cpu")
    prev0 = torch.zeros(1, NUM_ACTIONS, dtype=torch.float32)
    prev1 = torch.zeros(1, NUM_ACTIONS, dtype=torch.float32)
    prev1[0, 7] = 1.0  # USE was pressed on the previous step

    with torch.no_grad():
        logits0, _, _ = policy.act_step(obs, prev0, hidden)
        logits1, _, _ = policy.act_step(obs, prev1, hidden)

    assert (logits0 - logits1).abs().max().item() > 1e-3


@pytest.mark.parametrize("arch", ["telemetry_gru_v1", "stub"])
@pytest.mark.parametrize("with_resets", [True, False])
def test_evaluate_chunks_matches_step_by_step_with_resets(arch: str, with_resets: bool):
    torch.manual_seed(23)
    policy = build_policy(PolicyConfig(arch=arch))
    policy.eval()

    T, B = 12, 4
    obs_seq = _random_preprocessed_seq(T, B, empty_steps=(2, 7), seed=23)

    init_hidden = policy.init_hidden(B, "cpu")
    init_hidden.normal_(mean=0.0, std=0.5)
    prev_a = torch.randint(0, 2, (B, NUM_ACTIONS), dtype=torch.float32)

    episode_starts = torch.zeros(T, B, dtype=torch.bool)
    if with_resets:
        episode_starts[0, 1] = True
        episode_starts[3, 0] = True
        episode_starts[5, 2] = True
        episode_starts[5, 3] = True
        episode_starts[11, 1] = True

    h = init_hidden
    prev_actions = []
    actions = []
    step_values = []
    step_log_probs = []
    step_entropy = []
    gen = torch.Generator().manual_seed(101)
    with torch.no_grad():
        for t in range(T):
            keep = (~episode_starts[t]).float()
            h = h * keep.view(1, B, 1)
            prev_a = prev_a * keep.view(B, 1)
            prev_actions.append(prev_a)

            obs_t = {k: v[t] for k, v in obs_seq.items()}
            logits_t, value_t, h = policy.act_step(obs_t, prev_a, h)
            act_t, lp_t = policy.sample_from_logits(logits_t, deterministic=False, generator=gen)
            lp_check, ent_t = policy.logprob_and_entropy(logits_t, act_t)
            torch.testing.assert_close(lp_t, lp_check, atol=1e-5, rtol=1e-5)

            actions.append(act_t)
            step_values.append(value_t)
            step_log_probs.append(lp_t)
            step_entropy.append(ent_t)
            prev_a = act_t

        prev_action_seq = torch.stack(prev_actions, dim=0)
        action_seq = torch.stack(actions, dim=0)
        seq_lp, seq_ent, seq_val = policy.evaluate_chunks(
            obs_seq, prev_action_seq, action_seq, init_hidden, episode_starts
        )

    torch.testing.assert_close(seq_lp, torch.stack(step_log_probs, dim=0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(seq_ent, torch.stack(step_entropy, dim=0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(seq_val, torch.stack(step_values, dim=0), atol=1e-5, rtol=1e-5)


def test_factorized_action_head_forbids_contradictory_buttons():
    torch.manual_seed(5)
    policy = build_policy(PolicyConfig(arch="telemetry_gru_v1", action_head="factorized"))
    policy.eval()
    logits = torch.randn(512, 12)
    gen = torch.Generator().manual_seed(77)
    act, lp = policy.sample_from_logits(logits, deterministic=False, generator=gen)
    assert act.shape == (512, NUM_ACTIONS)
    assert torch.isfinite(lp).all()
    # Mutually exclusive pairs must never both be 1
    assert not bool(((act[:, 0] > 0.5) & (act[:, 1] > 0.5)).any())
    assert not bool(((act[:, 2] > 0.5) & (act[:, 3] > 0.5)).any())
    assert not bool(((act[:, 4] > 0.5) & (act[:, 5] > 0.5)).any())

    # Also verify Role 2's 4-tuple sample_action(obs, prev_action, hidden) contract
    obs1 = {k: v.squeeze(0) for k, v in _random_preprocessed_seq(1, 2, seed=9).items()}
    a2, lp2, val2, h2 = policy.sample_action(obs1, torch.zeros(2, NUM_ACTIONS), policy.init_hidden(2, "cpu"))
    assert a2.shape == (2, NUM_ACTIONS) and lp2.shape == (2,) and val2.shape == (2,)
    assert h2.shape == (1, 2, policy.config.hidden_size)


def test_sign_verified_entity_kinematics_and_ammo_empty_flag():
    policy = build_policy(PolicyConfig(arch="telemetry_gru_v1"))
    enc = policy.entity_encoder
    # Entity at 90 deg LEFT (sin=1, cos=0), distance=0.25:
    # Moving RIGHT (+vel_right = +1.0) means moving TOWARD player -> positive closing speed!
    # Moving LEFT (-vel_right = -1.0) means moving AWAY -> negative closing speed!
    ent_feat = torch.zeros(2, 1, 5)
    ent_feat[0, 0] = torch.tensor([0.25, 1.0, 0.0, 0.0, 1.0])
    ent_feat[1, 0] = torch.tensor([0.25, 1.0, 0.0, 0.0, -1.0])
    ent_mask = torch.ones(2, 1)
    norm = enc._normalize_features(ent_feat, ent_mask)
    closing_toward = float(norm[0, 0, 6])
    closing_away = float(norm[1, 0, 6])
    inv_tti_toward = float(norm[0, 0, 8])
    inv_tti_away = float(norm[1, 0, 8])
    assert closing_toward > 0.5 and closing_away < -0.5
    assert inv_tti_toward > 0.5 and inv_tti_away == pytest.approx(0.0)

    # Empty ammo vs 2 shells (2/50 = 0.04) should strongly separate VitalsActionEncoder outputs
    v0 = torch.zeros(1, NUM_VITALS)
    v2 = torch.zeros(1, NUM_VITALS)
    v0[0, 0] = 1.0
    v2[0, 0] = 1.0
    v2[0, 3] = 0.04  # 2 shells / 50
    prev = torch.zeros(1, NUM_ACTIONS)
    with torch.no_grad():
        e0 = policy.vitals_encoder(v0, prev)
        e2 = policy.vitals_encoder(v2, prev)
    assert (e0 - e2).abs().max().item() > 0.05


@pytest.mark.parametrize("arch", ["telemetry_gru_v1", "stub"])
def test_checkpoint_roundtrip_and_two_consecutive_maps_reset(
    arch: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    torch.manual_seed(31)
    weights_path = tmp_path / f"{arch}.pt"
    cfg = PolicyConfig(arch=arch, eval_deterministic=True)
    policy = build_policy(cfg)
    save_checkpoint(weights_path, policy, cfg, meta={"test": "two_maps", "arch": arch})

    reloaded, loaded_cfg, meta = load_checkpoint(weights_path, device="cpu")
    assert loaded_cfg == cfg
    assert meta == {"test": "two_maps", "arch": arch}

    obs_seq = _random_preprocessed_seq(3, 2, seed=42)
    prev_a = torch.zeros(2, NUM_ACTIONS, dtype=torch.float32)
    h = policy.init_hidden(2, "cpu")
    policy.eval()
    with torch.no_grad():
        l1, v1, _ = policy.act_step({k: v[0] for k, v in obs_seq.items()}, prev_a, h)
        l2, v2, _ = reloaded.act_step({k: v[0] for k, v in obs_seq.items()}, prev_a, h)
    torch.testing.assert_close(l1, l2)
    torch.testing.assert_close(v1, v2)

    monkeypatch.setenv("DOOM_AGENT_WEIGHTS", str(weights_path))
    door_wad = fixture_wad("##########\n#P..D...E#\n##########")
    key_wad = fixture_wad("############\n#P.r..R...E#\n############")

    env1 = make_env(door_wad, episode_ticks=400, render_buffers=False)
    env2 = make_env(key_wad, episode_ticks=400, render_buffers=False)
    try:
        agent = DoomAgent(env1.observation_space, env1.action_space, "cpu")

        # Trajectory A on Map 1 (5 steps) -> reset -> Map 2 initial action
        obs1, _ = env1.reset(seed=0)
        agent.reset()
        for _ in range(5):
            act1 = agent.act(obs1)
            assert act1.shape == (NUM_ACTIONS,) and act1.dtype == np.int8 and np.all((act1 == 0) | (act1 == 1))
            obs1, _, term, trunc, _ = env1.step(act1)
            if term or trunc:
                break
        assert torch.count_nonzero(agent._hidden) > 0

        obs2, _ = env2.reset(seed=0)
        agent.reset()
        assert torch.count_nonzero(agent._hidden) == 0
        assert torch.count_nonzero(agent._prev_action) == 0
        first_act_map2_a = agent.act(obs2).copy()
        assert first_act_map2_a.shape == (NUM_ACTIONS,) and first_act_map2_a.dtype == np.int8

        # Trajectory B on Map 1 (15 steps, different seed) -> reset -> Map 2 initial action must match A
        obs1_b, _ = env1.reset(seed=2)
        agent.reset()
        for _ in range(15):
            act1_b = agent.act(obs1_b)
            obs1_b, _, term, trunc, _ = env1.step(act1_b)
            if term or trunc:
                break

        obs2_b, _ = env2.reset(seed=0)
        agent.reset()
        assert torch.count_nonzero(agent._hidden) == 0
        assert torch.count_nonzero(agent._prev_action) == 0
        first_act_map2_b = agent.act(obs2_b).copy()
        np.testing.assert_array_equal(first_act_map2_a, first_act_map2_b)
    finally:
        env1.close()
        env2.close()
