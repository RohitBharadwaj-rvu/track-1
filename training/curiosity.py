"""Training-only Random Network Distillation (RND) intrinsic curiosity (`TRACK1_IMPROVEMENTS.md §3.5`).

Operates exclusively during training on normalized telemetry (`rays`, `vitals`); adds zero runtime
or parameter overhead to `submission/agent.py` or exported `submission/weights/policy.pt`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from submission.obs import NUM_RAY_HIT_TYPES, NUM_RAYS, NUM_VITALS


@dataclass(frozen=True)
class RNDConfig:
    coef: float = 0.02
    lr: float = 1e-3
    feat_dim: int = NUM_RAYS * 2 + NUM_VITALS  # 64 + 64 + 17 = 145
    hidden_dim: int = 128
    out_dim: int = 64
    max_reward: float = 5.0


class RunningRMS:
    """Tracks scalar running variance for normalizing intrinsic reward scale."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        batch_mean = float(np.mean(arr))
        batch_var = float(np.var(arr))
        batch_count = float(arr.size)

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = max(1e-6, m2 / tot)
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.mean = float(state["mean"])
        self.var = float(state["var"])
        self.count = float(state["count"])


class RNDIntrinsicCuriosity:
    """Predicts a frozen random projection of telemetry states; prediction error drives exploration."""

    def __init__(self, config: RNDConfig | None = None, device: str | torch.device = "cpu"):
        self.config = config or RNDConfig()
        self.device = torch.device(device)
        cfg = self.config

        self.target_net = nn.Sequential(
            nn.Linear(cfg.feat_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
        ).to(self.device)
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.target_net.eval()

        self.predictor_net = nn.Sequential(
            nn.Linear(cfg.feat_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=cfg.lr)
        self.rms = RunningRMS()

    def _obs_to_flat_tensor(self, obs: dict[str, Any]) -> torch.Tensor:
        ray_dist = torch.as_tensor(obs["ray_dist"], dtype=torch.float32, device=self.device)
        ray_type = torch.as_tensor(obs["ray_type"], dtype=torch.float32, device=self.device) / max(
            1, NUM_RAY_HIT_TYPES - 1
        )
        vitals = torch.as_tensor(obs["vitals"], dtype=torch.float32, device=self.device)
        x = torch.cat([ray_dist, ray_type, vitals], dim=-1)
        return x.reshape(-1, self.config.feat_dim)

    def compute_reward(self, obs_batch: dict[str, Any]) -> np.ndarray:
        """Compute clipped normalized intrinsic reward per batch item."""
        if self.config.coef <= 0.0:
            n = int(np.asarray(obs_batch["vitals"]).shape[0])
            return np.zeros(n, dtype=np.float32)

        self.predictor_net.eval()
        with torch.no_grad():
            x = self._obs_to_flat_tensor(obs_batch)
            target = self.target_net(x)
            pred = self.predictor_net(x)
            mse = ((pred - target) ** 2).mean(dim=-1).cpu().numpy().astype(np.float32)

        self.rms.update(mse)
        normed = np.clip(mse / self.rms.std, 0.0, self.config.max_reward)
        return (normed * float(self.config.coef)).astype(np.float32)

    def update(self, obs_seq: dict[str, Any]) -> float:
        """Train predictor toward frozen target on a rollout chunk; returns MSE loss."""
        self.predictor_net.train()
        x = self._obs_to_flat_tensor(obs_seq)
        with torch.no_grad():
            target = self.target_net(x)
        pred = self.predictor_net(x)
        loss = F.mse_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def state_dict(self) -> dict[str, Any]:
        return {
            "target_net": self.target_net.state_dict(),
            "predictor_net": self.predictor_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rms": self.rms.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.target_net.load_state_dict(state["target_net"])
        self.predictor_net.load_state_dict(state["predictor_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rms.load_state_dict(state["rms"])
