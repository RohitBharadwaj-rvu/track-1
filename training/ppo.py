"""Recurrent PPO trainer with chunked BPTT, KL early stopping, and resumable checkpoints (Role 2).

Uses only the frozen Role 1 contract (`policy.evaluate_chunks`, `PolicyConfig`, `save_checkpoint`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from submission.obs import obs_to_torch
from submission.policy import CHECKPOINT_FORMAT_VERSION, PolicyConfig, build_policy
from training.rollout import RolloutBuffer


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecurrentPPOTrainer:
    """Performs recurrent PPO updates over contiguous chunks from a `RolloutBuffer`."""

    def __init__(
        self,
        policy: nn.Module,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
    ):
        self.policy = policy
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.optimizer = optimizer or torch.optim.Adam(self.policy.parameters(), lr=self.config.lr, eps=1e-5)

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def update(self, buffer: RolloutBuffer, rng: np.random.Generator | None = None) -> dict[str, float]:
        cfg = self.config
        self.policy.train()
        if rng is None:
            rng = np.random.default_rng()

        pg_losses, v_losses, entropies, kls, clipfracs, grad_norms = [], [], [], [], [], []
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

                new_logprob, entropy, new_value = self.policy.evaluate_chunks(
                    obs_seq, prev_action_seq, action_seq, init_hidden, episode_starts
                )

                logratio = new_logprob - old_logprob
                ratio = torch.exp(logratio)

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if cfg.clip_vloss:
                    v_loss_unclipped = (new_value - ret) ** 2
                    v_clipped = old_value + torch.clamp(new_value - old_value, -cfg.clip_coef, cfg.clip_coef)
                    v_loss_clipped = (v_clipped - ret) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - ret) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm).item())
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
        }


def save_training_state(
    path: str | Path,
    policy: nn.Module,
    policy_config: PolicyConfig,
    optimizer: torch.optim.Optimizer,
    ppo_config: PPOConfig,
    step_info: dict[str, Any] | None = None,
) -> Path:
    """Save a resumable training checkpoint containing model + optimizer + step metadata."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": policy_config.to_dict(),
        "ppo_config": ppo_config.to_dict(),
        "state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step_info": dict(step_info or {}),
    }
    torch.save(payload, out)
    return out


def load_training_state(
    path: str | Path,
    device: str | torch.device = "cpu",
    policy: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[nn.Module, PolicyConfig, torch.optim.Optimizer, dict[str, Any]]:
    """Restore a resumable training checkpoint saved by `save_training_state`."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    policy_config = PolicyConfig.from_dict(payload["config"])
    if policy is None:
        policy = build_policy(policy_config).to(device)
    policy.load_state_dict(payload["state_dict"])
    if optimizer is None:
        lr = float(payload.get("ppo_config", {}).get("lr", 3e-4))
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return policy, policy_config, optimizer, dict(payload.get("step_info", {}))
