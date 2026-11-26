"""TDD tests for Role 2 PyTorch improvements: RND curiosity, asymmetric critic, and aux memory."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from submission.obs import obs_to_torch
from submission.policy import PolicyConfig, build_policy, load_checkpoint, save_checkpoint
from training.aux_heads import AsymmetricCritic, AuxMemoryHead, FeatureCaptureHook
from training.curiosity import RNDConfig, RNDIntrinsicCuriosity
from training.ppo import PPOConfig, RecurrentPPOTrainer, load_training_state, save_training_state
from training.rollout import AUX_TARGET_DIM, PRIV_FEATURE_DIM, RolloutBuffer


def _obs_batch(batch_size: int = 4, *, seed: int = 0, ray_val: float | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ray_dist = (
        np.full((batch_size, 64), ray_val, dtype=np.float32)
        if ray_val is not None
        else rng.random((batch_size, 64), dtype=np.float32)
    )
    return {
        "ray_dist": ray_dist,
        "ray_type": rng.integers(0, 6, size=(batch_size, 64), dtype=np.int64),
        "ent_feat": np.zeros((batch_size, 32, 5), dtype=np.float32),
        "ent_type": np.zeros((batch_size, 32), dtype=np.int64),
        "ent_mask": np.zeros((batch_size, 32), dtype=np.float32),
        "vitals": np.zeros((batch_size, 17), dtype=np.float32),
    }


def test_rnd_curiosity_decreases_reward_on_familiar_states():
    torch.manual_seed(0)
    rnd = RNDIntrinsicCuriosity(RNDConfig(coef=0.05, lr=1e-3), device="cpu")
    familiar = _obs_batch(batch_size=8, seed=1, ray_val=0.25)
    familiar["ray_type"][:] = 1
    novel = _obs_batch(batch_size=8, seed=2, ray_val=0.90)
    novel["ray_type"][:] = 3

    seq_familiar = {k: np.repeat(v[None, ...], 4, axis=0) for k, v in familiar.items()}
    first_loss = rnd.update(seq_familiar)
    for _ in range(35):
        last_loss = rnd.update(seq_familiar)

    r_familiar = float(np.mean(rnd.compute_reward(familiar)))
    r_novel = float(np.mean(rnd.compute_reward(novel)))

    assert last_loss < first_loss * 0.5
    assert r_novel > r_familiar


def test_feature_hook_asymmetric_critic_and_aux_head_gradients(tmp_path):
    torch.manual_seed(0)
    cfg = PolicyConfig(arch="telemetry_gru_v1")
    policy = build_policy(cfg)
    hook = FeatureCaptureHook(policy)
    asym = AsymmetricCritic(hidden_dim=cfg.hidden_size, priv_dim=PRIV_FEATURE_DIM)
    aux = AuxMemoryHead(hidden_dim=cfg.hidden_size)

    try:
        # 1. Single-step feature capture
        obs = obs_to_torch(_obs_batch(batch_size=2, seed=0), "cpu")
        prev = torch.zeros(2, 9)
        h = policy.init_hidden(2, "cpu")
        _, _, _ = policy.act_step(obs, prev, h)
        feat_step = hook.pop()
        assert feat_step.shape == (2, cfg.hidden_size)
        val = asym(feat_step, torch.zeros(2, PRIV_FEATURE_DIM))
        assert val.shape == (2,)

        # 2. Sequence feature capture + gradient flow from aux_loss into policy.gru
        T, B = 6, 2
        obs_seq = {k: v.unsqueeze(0).repeat(T, *([1] * v.ndim)) for k, v in obs.items()}
        _, _, _ = policy.evaluate_chunks(
            obs_seq,
            torch.zeros(T, B, 9),
            torch.zeros(T, B, 9),
            h,
            torch.zeros(T, B, dtype=torch.bool),
        )
        feat_seq = hook.pop()
        assert feat_seq.shape == (T, B, cfg.hidden_size)

        targets = torch.tensor(
            [[[1.0, 0.0, 1.0, 0.5], [0.0, 0.0, 0.0, 0.0]]] * T,
            dtype=torch.float32,
        )
        aux_loss, aux_metrics = aux.compute_loss(feat_seq, targets)
        assert np.isfinite(float(aux_loss.item()))
        assert "aux_seen_acc" in aux_metrics

        policy.zero_grad()
        aux_loss.backward()
        gru_grad_norm = sum(
            float(p.grad.abs().sum().item())
            for p in policy.gru.parameters()
            if p.grad is not None
        )
        assert gru_grad_norm > 0.0

        # 3. Exported submission checkpoint must remain strictly blind TelemetryPolicy
        ckpt_path = tmp_path / "blind_export.pt"
        save_checkpoint(ckpt_path, policy, cfg, {"arch": "telemetry_gru_v1"})
        loaded, loaded_cfg, _ = load_checkpoint(ckpt_path, device="cpu")
        assert loaded_cfg.arch == "telemetry_gru_v1"
        assert not hasattr(loaded, "asymmetric_critic")
    finally:
        hook.remove()


def test_recurrent_ppo_with_asymmetric_critic_aux_and_rnd_roundtrip(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    cfg = PolicyConfig(arch="telemetry_gru_v1")
    policy = build_policy(cfg)
    ppo_cfg = PPOConfig(
        lr=3e-4,
        update_epochs=2,
        chunk_len=4,
        num_minibatches=2,
        target_kl=None,
        use_asymmetric_critic=True,
        aux_coef=0.2,
        curiosity_coef=0.02,
    )
    trainer = RecurrentPPOTrainer(policy, ppo_cfg, device="cpu")

    T, N, H = 8, 2, cfg.hidden_size
    buf = RolloutBuffer(num_steps=T, num_envs=N, hidden_shape=(1, N, H))
    for t in range(T):
        obs = _obs_batch(batch_size=N, seed=t)
        priv_f = rng.normal(0.0, 0.3, size=(N, PRIV_FEATURE_DIM)).astype(np.float32)
        aux_t = np.array([[1.0, 0.5, 0.866, 0.4], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        buf.add(
            step=t,
            obs_batch=obs,
            prev_action=np.zeros((N, 9), dtype=np.float32),
            action=rng.integers(0, 2, size=(N, 9)).astype(np.float32),
            log_prob=np.full(N, -6.2, dtype=np.float32),
            reward=rng.normal(0.0, 0.5, size=N).astype(np.float32),
            value=np.zeros(N, dtype=np.float32),
            terminated=np.zeros(N, dtype=bool),
            truncated=np.zeros(N, dtype=bool),
            bootstrap_value=np.zeros(N, dtype=np.float32),
            episode_start=np.array([t == 0, t == 0], dtype=bool),
            hidden=np.zeros((1, N, H), dtype=np.float32),
            priv_features=priv_f,
            aux_targets=aux_t,
        )
    buf.compute_returns_and_advantages(np.zeros(N, dtype=np.float32), gamma=ppo_cfg.gamma)
    metrics = trainer.update(buf, rng=rng)
    for k in ("policy_loss", "value_loss", "aux_loss", "aux_seen_acc", "rnd_loss"):
        assert np.isfinite(metrics[k]), f"{k} not finite: {metrics[k]}"

    ckpt = tmp_path / "train_state_full.pt"
    save_training_state(
        ckpt,
        policy,
        cfg,
        trainer.optimizer,
        ppo_cfg,
        {"updates": 1},
        asymmetric_critic=trainer.asymmetric_critic,
        aux_head=trainer.aux_head,
        curiosity=trainer.curiosity,
    )
    pol2, cfg2, opt2, info2, extras = load_training_state(ckpt, device="cpu", return_extras=True)
    assert cfg2.arch == "telemetry_gru_v1"
    assert info2["updates"] == 1
    assert "asymmetric_critic_state" in extras
    assert "aux_head_state" in extras
    assert "curiosity_state" in extras
    assert len(opt2.state) > 0
