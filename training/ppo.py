"""Recurrent PPO trainer with chunked BPTT, asymmetric critic, aux memory, and RND curiosity (Role 2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from submission.obs import obs_to_torch
from submission.policy import CHECKPOINT_FORMAT_VERSION, PolicyConfig, build_policy
from training.aux_heads import AsymmetricCritic, AuxMemoryHead, FeatureCaptureHook
from training.curiosity import RNDConfig, RNDIntrinsicCuriosity
from training.rollout import PRIV_FEATURE_DIM, RolloutBuffer


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    chunk_len: int = 64
    num_minibatches: int = 4
    target_kl: float | None = 0.03
    normalize_advantages: bool = True
    clip_vloss: bool = True
    use_huber_vloss: bool = True
    huber_delta: float = 5.0
    use_asymmetric_critic: bool = True
    aux_coef: float = 0.2
    curiosity_coef: float = 0.0
    detach_asymmetric_critic_feat: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecurrentPPOTrainer:
    """Performs recurrent PPO updates over contiguous chunks with optional training-only heads."""

    def __init__(
        self,
        policy: nn.Module,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        *,
        asymmetric_critic: AsymmetricCritic | None = None,
        aux_head: AuxMemoryHead | None = None,
        curiosity: RNDIntrinsicCuriosity | None = None,
    ):
        self.policy = policy
        self.config = config or PPOConfig()
        self._ent_coef = float(self.config.ent_coef)
        self.device = torch.device(device)
        hidden_dim = int(getattr(getattr(policy, "config", None), "hidden_size", 256))

        self.asymmetric_critic = asymmetric_critic
        if self.config.use_asymmetric_critic and self.asymmetric_critic is None:
            self.asymmetric_critic = AsymmetricCritic(hidden_dim, PRIV_FEATURE_DIM).to(self.device)

        self.aux_head = aux_head
        if self.config.aux_coef > 0.0 and self.aux_head is None:
            self.aux_head = AuxMemoryHead(hidden_dim).to(self.device)

        self.curiosity = curiosity
        if self.config.curiosity_coef > 0.0 and self.curiosity is None:
            self.curiosity = RNDIntrinsicCuriosity(
                RNDConfig(coef=self.config.curiosity_coef), device=self.device
            )

        self.feature_hook = (
            FeatureCaptureHook(self.policy)
            if (self.asymmetric_critic is not None or self.aux_head is not None) and hasattr(self.policy, "critic")
            else None
        )

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            params = list(self.policy.parameters())
            if self.asymmetric_critic is not None:
                params.extend(self.asymmetric_critic.parameters())
            if self.aux_head is not None:
                params.extend(self.aux_head.parameters())
            self.optimizer = torch.optim.Adam(params, lr=self.config.lr, eps=1e-5)

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def set_ent_coef(self, ent_coef: float) -> None:
        self._ent_coef = float(ent_coef)

    def close(self) -> None:
        if self.feature_hook is not None:
            self.feature_hook.remove()
            self.feature_hook = None

    def update(self, buffer: RolloutBuffer, rng: np.random.Generator | None = None) -> dict[str, float]:
        cfg = self.config
        self.policy.train()
        if self.asymmetric_critic is not None:
            self.asymmetric_critic.train()
        if self.aux_head is not None:
            self.aux_head.train()
        if rng is None:
            rng = np.random.default_rng()

        pg_losses, v_losses, entropies, kls, clipfracs, grad_norms = [], [], [], [], [], []
        aux_losses, aux_accs, rnd_losses = [], [], []
        epochs_run = 0
        early_stopped = False

        for epoch in range(cfg.update_epochs):
            epoch_kls = []
            for mb in buffer.iter_chunk_minibatches(cfg.chunk_len, cfg.num_minibatches, rng=rng):
                obs_seq = obs_to_torch(mb["obs"], self.device)
                prev_action_seq = torch.as_tensor(mb["prev_actions"], dtype=torch.float32, device=self.device)
                action_seq = torch.as_tensor(mb["actions"], dtype=torch.float32, device=self.device)
                old_logprob = torch.as_tensor(mb["old_log_probs"], dtype=torch.float32, device=self.device)
                old_value = torch.as_tensor(mb["old_values"], dtype=torch.float32, device=self.device)
                adv = torch.as_tensor(mb["advantages"], dtype=torch.float32, device=self.device)
                ret = torch.as_tensor(mb["returns"], dtype=torch.float32, device=self.device)
                episode_starts = torch.as_tensor(mb["episode_starts"], dtype=torch.bool, device=self.device)
                init_hidden = torch.as_tensor(mb["init_hidden"], dtype=torch.float32, device=self.device)

                if cfg.normalize_advantages and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                new_logprob, entropy, blind_value = self.policy.evaluate_chunks(
                    obs_seq, prev_action_seq, action_seq, init_hidden, episode_starts
                )

                feat_seq = self.feature_hook.pop() if self.feature_hook is not None else None
                if self.asymmetric_critic is not None and feat_seq is not None:
                    priv_seq = torch.as_tensor(mb["priv_features"], dtype=torch.float32, device=self.device)
                    critic_feat = feat_seq.detach() if cfg.detach_asymmetric_critic_feat else feat_seq
                    new_value = self.asymmetric_critic(critic_feat, priv_seq)
                else:
                    new_value = blind_value

                logratio = new_logprob - old_logprob
                ratio = torch.exp(logratio)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.use_huber_vloss:
                    if cfg.clip_vloss:
                        v_loss_unclipped = F.huber_loss(new_value, ret, reduction="none", delta=cfg.huber_delta)
                        v_clipped = old_value + torch.clamp(new_value - old_value, -cfg.clip_coef, cfg.clip_coef)
                        v_loss_clipped = F.huber_loss(v_clipped, ret, reduction="none", delta=cfg.huber_delta)
                        v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = F.huber_loss(new_value, ret, reduction="mean", delta=cfg.huber_delta)
                elif cfg.clip_vloss:
                    v_loss_unclipped = (new_value - ret) ** 2
                    v_clipped = old_value + torch.clamp(new_value - old_value, -cfg.clip_coef, cfg.clip_coef)
                    v_loss_clipped = (v_clipped - ret) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - ret) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self._ent_coef * entropy_loss + cfg.vf_coef * v_loss

                if self.aux_head is not None and cfg.aux_coef > 0.0 and feat_seq is not None:
                    aux_targets = torch.as_tensor(mb["aux_targets"], dtype=torch.float32, device=self.device)
                    aux_loss, aux_m = self.aux_head.compute_loss(feat_seq, aux_targets)
                    loss = loss + cfg.aux_coef * aux_loss
                    aux_losses.append(float(aux_loss.item()))
                    aux_accs.append(aux_m["aux_seen_acc"])

                if self.curiosity is not None:
                    rnd_losses.append(self.curiosity.update(mb["obs"]))

                all_params = list(self.policy.parameters())
                if self.asymmetric_critic is not None:
                    all_params.extend(self.asymmetric_critic.parameters())
                if self.aux_head is not None:
                    all_params.extend(self.aux_head.parameters())

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm).item())
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - logratio).mean().item())
                    clipfrac = float(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(entropy_loss.item()))
                kls.append(approx_kl)
                epoch_kls.append(approx_kl)
                clipfracs.append(clipfrac)
                grad_norms.append(grad_norm)

            epochs_run = epoch + 1
            if cfg.target_kl is not None and epoch_kls and float(np.mean(epoch_kls)) > cfg.target_kl:
                early_stopped = True
                break

        y_pred = buffer.values.reshape(-1)
        y_true = buffer.returns.reshape(-1)
        var_y = float(np.var(y_true))
        explained_var = 0.0 if var_y < 1e-8 else float(1.0 - np.var(y_true - y_pred) / var_y)

        return {
            "updates": float(len(pg_losses)),
            "epochs_run": float(epochs_run),
            "early_stopped": float(early_stopped),
            "policy_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "explained_variance": explained_var,
            "aux_loss": float(np.mean(aux_losses)) if aux_losses else 0.0,
            "aux_seen_acc": float(np.mean(aux_accs)) if aux_accs else 0.0,
            "rnd_loss": float(np.mean(rnd_losses)) if rnd_losses else 0.0,
            "mean_intrinsic_reward": float(getattr(buffer, "last_mean_intrinsic_reward", 0.0)),
        }


def save_training_state(
    path: str | Path,
    policy: nn.Module,
    policy_config: PolicyConfig,
    optimizer: torch.optim.Optimizer,
    ppo_config: PPOConfig,
    step_info: dict[str, Any] | None = None,
    *,
    asymmetric_critic: AsymmetricCritic | None = None,
    aux_head: AuxMemoryHead | None = None,
    curiosity: RNDIntrinsicCuriosity | None = None,
) -> Path:
    """Save a resumable training checkpoint containing model + optimizer + optional training heads."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": policy_config.to_dict(),
        "ppo_config": ppo_config.to_dict(),
        "state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step_info": dict(step_info or {}),
    }
    if asymmetric_critic is not None:
        payload["asymmetric_critic_state"] = asymmetric_critic.state_dict()
    if aux_head is not None:
        payload["aux_head_state"] = aux_head.state_dict()
    if curiosity is not None:
        payload["curiosity_state"] = curiosity.state_dict()
    torch.save(payload, out)
    return out


def load_training_state(
    path: str | Path,
    device: str | torch.device = "cpu",
    policy: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    return_extras: bool = False,
):
    """Restore a resumable training checkpoint saved by `save_training_state`."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    policy_config = PolicyConfig.from_dict(payload["config"])
    if policy is None:
        policy = build_policy(policy_config).to(device)
    policy.load_state_dict(payload["state_dict"])

    ppo_raw = payload.get("ppo_config", {})
    hidden_dim = int(policy_config.hidden_size)
    extras: dict[str, Any] = {}
    params = list(policy.parameters())

    asym = None
    if "asymmetric_critic_state" in payload:
        asym = AsymmetricCritic(hidden_dim, PRIV_FEATURE_DIM).to(device)
        asym.load_state_dict(payload["asymmetric_critic_state"])
        params.extend(asym.parameters())
        extras["asymmetric_critic"] = asym
        extras["asymmetric_critic_state"] = payload["asymmetric_critic_state"]

    aux_head = None
    if "aux_head_state" in payload:
        aux_head = AuxMemoryHead(hidden_dim).to(device)
        aux_head.load_state_dict(payload["aux_head_state"])
        params.extend(aux_head.parameters())
        extras["aux_head"] = aux_head
        extras["aux_head_state"] = payload["aux_head_state"]

    if "curiosity_state" in payload:
        coef = float(ppo_raw.get("curiosity_coef", 0.02))
        rnd = RNDIntrinsicCuriosity(RNDConfig(coef=coef), device=device)
        rnd.load_state_dict(payload["curiosity_state"])
        extras["curiosity"] = rnd
        extras["curiosity_state"] = payload["curiosity_state"]

    if optimizer is None:
        lr = float(ppo_raw.get("lr", 3e-4))
        optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    step_info = dict(payload.get("step_info", {}))
    if return_extras:
        return policy, policy_config, optimizer, step_info, extras
    return policy, policy_config, optimizer, step_info
