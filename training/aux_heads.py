"""Training-only asymmetric critic and auxiliary locked-door memory heads (`TRACK1_IMPROVEMENTS.md §4.3`).

Attaches cleanly to any Role 1 policy implementing `policy.critic` via a forward pre-hook,
keeping `submission/policy.py` and `submission/agent.py` strictly blind at evaluation time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.rollout import AUX_TARGET_DIM, PRIV_FEATURE_DIM


class FeatureCaptureHook:
    """Captures the latent feature tensor entering `policy.critic` without altering `submission/`."""

    def __init__(self, policy: nn.Module):
        if not hasattr(policy, "critic"):
            raise AttributeError("Policy must define a `critic` submodule for FeatureCaptureHook")
        self._last: torch.Tensor | None = None
        self._handle = policy.critic.register_forward_pre_hook(self._hook)  # type: ignore[union-attr]

    def _hook(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self._last = inputs[0]

    def pop(self) -> torch.Tensor:
        if self._last is None:
            raise RuntimeError("FeatureCaptureHook has no captured tensor; call policy forward first")
        feat = self._last
        self._last = None
        return feat

    def remove(self) -> None:
        self._handle.remove()


class AsymmetricCritic(nn.Module):
    """Training-only critic conditioned on blind recurrent features + privileged game signals."""

    def __init__(self, hidden_dim: int, priv_dim: int = PRIV_FEATURE_DIM, head_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + priv_dim, head_dim),
            nn.SiLU(),
            nn.Linear(head_dim, head_dim),
            nn.SiLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, feat: torch.Tensor, priv_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, priv_features], dim=-1)
        return self.net(x).squeeze(-1)


class AuxMemoryHead(nn.Module):
    """Auxiliary training-only head predicting locked-door sighting & ego-frame bearing."""

    def __init__(self, hidden_dim: int, out_dim: int = AUX_TARGET_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)

    def compute_loss(self, feat: torch.Tensor, aux_targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Predict `[seen_locked_door, door_sin, door_cos, door_dist_norm]`.

        Computes BCE on sighting bit + masked MSE on ego bearing/distance where `seen == 1`.
        """
        pred = self.forward(feat)
        seen_target = aux_targets[..., 0]
        seen_logits = pred[..., 0]
        seen_loss = F.binary_cross_entropy_with_logits(seen_logits, seen_target)

        mask = (seen_target > 0.5).to(dtype=pred.dtype).unsqueeze(-1)
        geom_sq = ((pred[..., 1:4] - aux_targets[..., 1:4]) ** 2) * mask
        geom_loss = geom_sq.sum() / (mask.sum() * 3.0 + 1e-6)

        with torch.no_grad():
            seen_acc = float(((seen_logits > 0.0) == (seen_target > 0.5)).float().mean().item())
        return seen_loss + geom_loss, {
            "aux_seen_loss": float(seen_loss.item()),
            "aux_geom_loss": float(geom_loss.item()),
            "aux_seen_acc": seen_acc,
        }
