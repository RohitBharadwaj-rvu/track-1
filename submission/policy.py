"""Shared Policy interface, config, checkpoint I/O, and runnable stub for parallel work.

Ownership (see REVISED_PLAN.md & training/README.md):
  - Role 1 (Policy & Exported Agent) replaces `TelemetryPolicy` internals (circular-conv rays,
    masked-attention entities with empty-set output, vitals + prev_action GRU, 9-button actor/value).
  - Role 2 (PPO & Rollout Collection) calls ONLY the frozen contract below:
      * `policy.init_hidden(batch_size, device)`
      * `policy.act_step(obs_batch, prev_action, hidden)`
      * `policy.evaluate_chunks(obs_seq, prev_action_seq, action_seq, init_hidden, episode_starts)`
      * `save_checkpoint` / `load_checkpoint`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from submission.obs import (
    MAX_ENTITIES,
    NUM_ACTIONS,
    NUM_ENTITY_FEATURES,
    NUM_ENTITY_TYPES,
    NUM_RAY_HIT_TYPES,
    NUM_RAYS,
    NUM_VITALS,
)

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PolicyConfig:
    """Serializable architecture hyperparameters saved inside every checkpoint."""

    arch: str = "stub"  # Role 1 sets e.g. "telemetry_gru_v1" or "telemetry_landmark_v1"
    hidden_size: int = 256
    ray_channels: int = 32
    entity_dim: int = 64
    vitals_dim: int = 64
    num_actions: int = NUM_ACTIONS
    num_rays: int = NUM_RAYS
    num_ray_hit_types: int = NUM_RAY_HIT_TYPES
    max_entities: int = MAX_ENTITIES
    num_entity_features: int = NUM_ENTITY_FEATURES
    num_entity_types: int = NUM_ENTITY_TYPES
    num_vitals: int = NUM_VITALS
    num_memory_slots: int = 0  # >0 when landmark memory challenger is enabled

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyConfig:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def build_policy(config: PolicyConfig):
    """Factory that builds the PyTorch policy matching `config.arch`.

    Role 1 registers new architecture names here while keeping `"stub"` available
    for fast unit tests and Role 2's PPO smoke tests.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class StubRecurrentPolicy(nn.Module):
        """Lightweight GRU policy implementing the exact Stage-1 contract.

        Lets Role 2 develop and test rollout collection, GAE, and recurrent PPO
        immediately while Role 1 builds the full circular-conv + attention encoder.
        """

        def __init__(self, cfg: PolicyConfig):
            super().__init__()
            self.config = cfg
            in_dim = cfg.num_rays * 2 + cfg.num_vitals + cfg.num_actions
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden_size),
                nn.Tanh(),
            )
            self.gru = nn.GRU(cfg.hidden_size, cfg.hidden_size, batch_first=False)
            self.actor = nn.Linear(cfg.hidden_size, cfg.num_actions)
            self.critic = nn.Linear(cfg.hidden_size, 1)

        def init_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
            """Return zero recurrent state of shape (1, B, hidden_size)."""
            return torch.zeros(1, batch_size, self.config.hidden_size, device=device)

        def _encode(self, obs: dict[str, torch.Tensor], prev_action: torch.Tensor) -> torch.Tensor:
            # Supports both (B, ...) and (T, B, ...)
            ray_dist = obs["ray_dist"].float()
            ray_type = obs["ray_type"].float() / max(1, self.config.num_ray_hit_types - 1)
            vitals = obs["vitals"].float()
            x = torch.cat([ray_dist, ray_type, vitals, prev_action.float()], dim=-1)
            return self.encoder(x)

        def act_step(
            self,
            obs_batch: dict[str, torch.Tensor],
            prev_action: torch.Tensor,
            hidden: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Single-step forward for rollout collection and `DoomAgent.act()`.

            Args:
                obs_batch: dict of tensors with leading batch dim (B, ...).
                prev_action: (B, 9) float/int tensor of last executed buttons.
                hidden: (1, B, H) recurrent state.

            Returns:
                logits: (B, 9) Bernoulli logits for the 9 Doom buttons.
                value: (B,) state-value estimate.
                next_hidden: (1, B, H) updated recurrent state.
            """
            emb = self._encode(obs_batch, prev_action).unsqueeze(0)  # (1, B, H)
            out, next_hidden = self.gru(emb, hidden)
            h = out.squeeze(0)  # (B, H)
            logits = self.actor(h)
            value = self.critic(h).squeeze(-1)
            return logits, value, next_hidden

        def evaluate_chunks(
            self,
            obs_seq: dict[str, torch.Tensor],
            prev_action_seq: torch.Tensor,
            action_seq: torch.Tensor,
            init_hidden: torch.Tensor,
            episode_starts: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Evaluate contiguous (T, B, ...) rollout chunks for recurrent PPO.

            Resets hidden state at every timestep `t` where `episode_starts[t, b]` is True/1.

            Args:
                obs_seq: dict of tensors with shape (T, B, ...).
                prev_action_seq: (T, B, 9) previous executed action.
                action_seq: (T, B, 9) taken binary actions in {0, 1}.
                init_hidden: (1, B, H) hidden state before t=0 of the chunk.
                episode_starts: (T, B) bool or float mask (1.0/True = start of new episode).

            Returns:
                log_prob: (T, B) summed Bernoulli log-prob across the 9 buttons.
                entropy: (T, B) summed Bernoulli entropy across the 9 buttons.
                value: (T, B) value predictions.
            """
            emb = self._encode(obs_seq, prev_action_seq)  # (T, B, H)
            T = emb.shape[0]
            h = init_hidden
            outputs = []
            starts = episode_starts.bool()
            for t in range(T):
                keep = (~starts[t]).float().view(1, -1, 1)
                h = h * keep
                out_t, h = self.gru(emb[t : t + 1], h)
                outputs.append(out_t)
            seq_h = torch.cat(outputs, dim=0)  # (T, B, H)
            logits = self.actor(seq_h)
            value = self.critic(seq_h).squeeze(-1)

            # Independent Bernoulli over 9 binary buttons
            bce = F.binary_cross_entropy_with_logits(logits, action_seq.float(), reduction="none")
            log_prob = -bce.sum(dim=-1)
            probs = torch.sigmoid(logits)
            entropy = -(
                probs * F.logsigmoid(logits) + (1.0 - probs) * F.logsigmoid(-logits)
            ).sum(dim=-1)
            return log_prob, entropy, value

    if config.arch == "stub":
        return StubRecurrentPolicy(config)
    raise ValueError(f"Unknown policy arch: {config.arch!r}. Register it in submission/policy.py.")


def save_checkpoint(
    path: str | Path,
    model: Any,
    config: PolicyConfig,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Save a portable checkpoint compatible with `DoomAgent` and Git LFS."""
    import torch

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "meta": dict(meta or {}),
    }
    torch.save(payload, out)
    return out


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Any, PolicyConfig, dict[str, Any]]:
    """Load a checkpoint created by `save_checkpoint` onto `device` in eval mode."""
    import torch

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = PolicyConfig.from_dict(payload["config"])
    model = build_policy(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, dict(payload.get("meta", {}))
