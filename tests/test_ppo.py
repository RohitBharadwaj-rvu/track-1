"""TDD tests for Role 2 recurrent PPO trainer and resumable checkpoints (`training/ppo.py`)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from submission.obs import obs_to_torch
from submission.policy import PolicyConfig, build_policy, load_checkpoint, save_checkpoint
from training.ppo import PPOConfig, RecurrentPPOTrainer, load_training_state, save_training_state
from training.rollout import RolloutBuffer


def _synthetic_buffer(T: int = 8, N: int = 2, H: int = 16, seed: int = 0) -> RolloutBuffer:
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(num_steps=T, num_envs=N, hidden_shape=(1, N, H))
    for t in range(T):
        obs = {
            "ray_dist": rng.random((N, 64), dtype=np.float32),
            "ray_type": rng.integers(0, 6, size=(N, 64), dtype=np.int64),
            "ent_feat": np.zeros((N, 32, 5), dtype=np.float32),
            "ent_type": np.zeros((N, 32), dtype=np.int64),
            "ent_mask": np.zeros((N, 32), dtype=np.float32),
            "vitals": rng.normal(0.0, 0.5, size=(N, 17)).astype(np.float32),
        }
        buf.add(
            step=t,
            obs_batch=obs,
            prev_action=rng.integers(0, 2, size=(N, 9)).astype(np.float32),
            action=rng.integers(0, 2, size=(N, 9)).astype(np.float32),
            log_prob=np.full(N, -6.2, dtype=np.float32),
            reward=rng.normal(0.0, 1.0, size=N).astype(np.float32),
            value=rng.normal(0.0, 0.5, size=N).astype(np.float32),
            terminated=np.array([t == 3 and e == 0 for e in range(N)], dtype=bool),
            truncated=np.zeros(N, dtype=bool),
            bootstrap_value=np.zeros(N, dtype=np.float32),
            episode_start=np.array([t == 0 or (t == 4 and e == 0) for e in range(N)], dtype=bool),
            hidden=np.zeros((1, N, H), dtype=np.float32),
        )
    buf.compute_returns_and_advantages(np.zeros(N, dtype=np.float32))
    return buf


def test_recurrent_ppo_update_changes_weights_and_returns_finite_metrics():
    torch.manual_seed(0)
    cfg = PolicyConfig(arch="stub", hidden_size=16)
    policy = build_policy(cfg)
    before = [p.detach().clone() for p in policy.parameters()]

    ppo_cfg = PPOConfig(lr=1e-3, update_epochs=2, chunk_len=4, num_minibatches=2, max_grad_norm=0.5, target_kl=None)
    trainer = RecurrentPPOTrainer(policy, ppo_cfg, device="cpu")
    buf = _synthetic_buffer(T=8, N=2, H=16)

    metrics = trainer.update(buf, rng=np.random.default_rng(1))
    assert metrics["updates"] > 0
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "grad_norm"):
        assert np.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"
    assert any(not torch.allclose(b, a) for b, a in zip(before, policy.parameters()))


def test_recurrent_ppo_learns_rewarded_button_on_synthetic_bandit():
    torch.manual_seed(0)
    rng = np.random.default_rng(7)
    cfg = PolicyConfig(arch="stub", hidden_size=32)
    policy = build_policy(cfg)
    ppo_cfg = PPOConfig(lr=5e-3, update_epochs=4, chunk_len=4, num_minibatches=2, ent_coef=0.005, target_kl=None)
    trainer = RecurrentPPOTrainer(policy, ppo_cfg, device="cpu")

    T, N, H = 8, 4, 32
    const_obs_single = {
        "ray_dist": np.ones((N, 64), dtype=np.float32),
        "ray_type": np.zeros((N, 64), dtype=np.int64),
        "ent_feat": np.zeros((N, 32, 5), dtype=np.float32),
        "ent_type": np.zeros((N, 32), dtype=np.int64),
        "ent_mask": np.zeros((N, 32), dtype=np.float32),
        "vitals": np.zeros((N, 17), dtype=np.float32),
    }

    for _ in range(25):
        buf = RolloutBuffer(num_steps=T, num_envs=N, hidden_shape=(1, N, H))
        hidden = policy.init_hidden(N, "cpu")
        prev_action = np.zeros((N, 9), dtype=np.float32)
        for t in range(T):
            with torch.no_grad():
                logits, val, next_h = policy.act_step(
                    obs_to_torch(const_obs_single, "cpu"),
                    torch.as_tensor(prev_action, dtype=torch.float32),
                    hidden,
                )
                act = torch.bernoulli(torch.sigmoid(logits))
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, act, reduction="none")
                lp = (-bce.sum(dim=-1)).numpy()
            act_np = act.numpy().astype(np.float32)
            # Reward button 0 being pressed (+1 vs -1)
            rew = np.where(act_np[:, 0] > 0.5, 1.0, -1.0).astype(np.float32)
            buf.add(
                step=t,
                obs_batch=const_obs_single,
                prev_action=prev_action,
                action=act_np,
                log_prob=lp,
                reward=rew,
                value=val.numpy().astype(np.float32),
                terminated=np.ones(N, dtype=bool),  # single-step bandit episodes
                truncated=np.zeros(N, dtype=bool),
                bootstrap_value=np.zeros(N, dtype=np.float32),
                episode_start=np.ones(N, dtype=bool),
                hidden=hidden.numpy(),
            )
            hidden = next_h * 0.0
            prev_action = np.zeros((N, 9), dtype=np.float32)
        buf.compute_returns_and_advantages(np.zeros(N, dtype=np.float32))
        trainer.update(buf, rng=rng)

    with torch.no_grad():
        logits, val, _ = policy.act_step(
            obs_to_torch(const_obs_single, "cpu"),
            torch.zeros(N, 9),
            policy.init_hidden(N, "cpu"),
        )
        p_btn0 = float(torch.sigmoid(logits[:, 0]).mean().item())
        mean_val = float(val.mean().item())

    assert p_btn0 > 0.75, f"Expected button 0 prob > 0.75, got {p_btn0:.3f}"
    assert mean_val > 0.2, f"Expected positive value estimate, got {mean_val:.3f}"


def test_checkpoint_save_and_resume_preserves_outputs_and_optimizer(tmp_path):
    torch.manual_seed(0)
    cfg = PolicyConfig(arch="stub", hidden_size=16)
    policy = build_policy(cfg)
    ppo_cfg = PPOConfig(lr=1e-3, update_epochs=1, chunk_len=4, num_minibatches=2, target_kl=None)
    trainer = RecurrentPPOTrainer(policy, ppo_cfg, device="cpu")
    buf = _synthetic_buffer(T=8, N=2, H=16)
    trainer.update(buf, rng=np.random.default_rng(0))

    obs_probe = obs_to_torch(
        {k: v[0] for k, v in buf.obs.items()},
        "cpu",
    )
    prev_probe = torch.zeros(2, 9)
    h_probe = policy.init_hidden(2, "cpu")
    with torch.no_grad():
        logits_before, val_before, _ = policy.act_step(obs_probe, prev_probe, h_probe)

    train_ckpt = tmp_path / "train_state.pt"
    export_ckpt = tmp_path / "export_policy.pt"
    save_training_state(train_ckpt, policy, cfg, trainer.optimizer, ppo_cfg, {"updates": 1, "steps": 16})
    save_checkpoint(export_ckpt, policy, cfg, {"updates": 1})

    resumed_policy, resumed_cfg, resumed_opt, step_info = load_training_state(train_ckpt, device="cpu")
    assert resumed_cfg == cfg
    assert step_info["steps"] == 16
    assert len(resumed_opt.state) > 0

    exported_policy, exp_cfg, exp_meta = load_checkpoint(export_ckpt, device="cpu")
    assert exp_cfg == cfg and exp_meta["updates"] == 1

    with torch.no_grad():
        l_res, v_res, _ = resumed_policy.act_step(obs_probe, prev_probe, h_probe)
        l_exp, v_exp, _ = exported_policy.act_step(obs_probe, prev_probe, h_probe)
    torch.testing.assert_close(l_res, logits_before)
    torch.testing.assert_close(v_res, val_before)
    torch.testing.assert_close(l_exp, logits_before)
    torch.testing.assert_close(v_exp, val_before)
