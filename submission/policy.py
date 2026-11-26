"""Shared Policy interface, config, checkpoint I/O, and telemetry architectures.

Ownership (see REVISED_PLAN.md & training/README.md):
  - Role 1 (Policy & Exported Agent) implements `TelemetryPolicy` (`arch="telemetry_gru_v1"`):
    circular-conv rays, masked-attention entities with learned empty-room output,
    normalized vitals + prev_action GRU, and 9-button Bernoulli actor + scalar critic.
  - Role 2 (PPO & Rollout Collection) calls ONLY the frozen contract below:
      * `policy.init_hidden(batch_size, device)`
      * `policy.act_step(obs_batch, prev_action, hidden)`
      * `policy.evaluate_chunks(obs_seq, prev_action_seq, action_seq, init_hidden, episode_starts)`
      * `save_checkpoint` / `load_checkpoint`
"""

from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from submission.obs import (
        MAX_ENTITIES,
        NUM_ACTIONS,
        NUM_ENTITY_FEATURES,
        NUM_ENTITY_TYPES,
        NUM_RAY_HIT_TYPES,
        NUM_RAYS,
        NUM_VITALS,
    )
except ImportError:  # pragma: no cover - fallback when submission/ is imported directly
    from obs import (
        MAX_ENTITIES,
        NUM_ACTIONS,
        NUM_ENTITY_FEATURES,
        NUM_ENTITY_TYPES,
        NUM_RAY_HIT_TYPES,
        NUM_RAYS,
        NUM_VITALS,
    )

CHECKPOINT_FORMAT_VERSION = 1
_LOG_FIVE = math.log(5.0)


@dataclass(frozen=True)
class PolicyConfig:
    """Serializable architecture hyperparameters saved inside every checkpoint."""

    arch: str = "telemetry_gru_v1"  # Role 1: "telemetry_gru_v1"; Role 2 smoke tests: "stub"
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

    # Role 1 encoder/head/eval hyperparameters (optional with defaults for backward compat)
    ray_out_dim: int = 128
    entity_out_dim: int = 128
    entity_type_embed_dim: int = 16
    entity_attn_heads: int = 4
    entity_num_queries: int = 2
    head_hidden_dim: int = 128
    use_gru_residual: bool = True
    use_button_init_bias: float = -1.5
    weapon_switch_init_bias: float = -2.0
    eval_deterministic: bool = False
    action_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.num_rays % 4 != 0:
            raise ValueError(f"num_rays must be divisible by 4, got {self.num_rays}")
        if self.entity_dim % self.entity_attn_heads != 0:
            raise ValueError(
                f"entity_dim ({self.entity_dim}) must be divisible by "
                f"entity_attn_heads ({self.entity_attn_heads})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyConfig:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def build_policy(config: PolicyConfig):
    """Factory that builds the PyTorch policy matching `config.arch`.

    Supports both `"stub"` (Role 2 smoke tests) and `"telemetry_gru_v1"` (Role 1 full policy).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _init_orthogonal(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    class StubRecurrentPolicy(nn.Module):
        """Lightweight GRU policy implementing the exact Stage-1 contract.

        Lets Role 2 develop and test rollout collection, GAE, and recurrent PPO
        immediately alongside the full circular-conv + attention encoder.
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
            emb = self._encode(obs_batch, prev_action).unsqueeze(0)
            out, next_hidden = self.gru(emb, hidden)
            h = out.squeeze(0)
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
            emb = self._encode(obs_seq, prev_action_seq)
            t_steps = emb.shape[0]
            h = init_hidden
            outputs = []
            starts = episode_starts.bool()
            for t in range(t_steps):
                keep = (~starts[t]).float().view(1, -1, 1)
                h = h * keep
                out_t, h = self.gru(emb[t : t + 1], h)
                outputs.append(out_t)
            seq_h = torch.cat(outputs, dim=0)
            logits = self.actor(seq_h)
            value = self.critic(seq_h).squeeze(-1)

            bce = F.binary_cross_entropy_with_logits(logits, action_seq.float(), reduction="none")
            log_prob = -bce.sum(dim=-1)
            probs = torch.sigmoid(logits)
            entropy = -(
                probs * F.logsigmoid(logits) + (1.0 - probs) * F.logsigmoid(-logits)
            ).sum(dim=-1)
            return log_prob, entropy, value

    class CircularRayEncoder(nn.Module):
        """Circular 1D conv encoder preserving directional angular bins through fusion."""

        def __init__(self, cfg: PolicyConfig) -> None:
            super().__init__()
            self.num_hit_types = cfg.num_ray_hit_types
            in_channels = 2 + cfg.num_ray_hit_types
            c1 = cfg.ray_channels
            c2 = max(16, (cfg.ray_channels * 3) // 2)
            c3 = cfg.ray_channels
            self.convs = nn.Sequential(
                nn.Conv1d(in_channels, c1, kernel_size=5, stride=1, padding=2, padding_mode="circular"),
                nn.SiLU(),
                nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
                nn.SiLU(),
                nn.Conv1d(c2, c3, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
                nn.SiLU(),
            )
            out_bins = cfg.num_rays // 4
            self.proj = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(c3 * out_bins, cfg.ray_out_dim),
                nn.LayerNorm(cfg.ray_out_dim),
                nn.SiLU(),
            )

        def forward(self, ray_dist: torch.Tensor, ray_type: torch.Tensor) -> torch.Tensor:
            d = ray_dist.float()
            dist_pair = torch.stack([d, 1.0 - d], dim=1)
            hit_idx = torch.clamp(ray_type.long(), 0, self.num_hit_types - 1)
            hit_oh = F.one_hot(hit_idx, num_classes=self.num_hit_types).to(dtype=d.dtype).permute(0, 2, 1)
            return self.proj(self.convs(torch.cat([dist_pair, hit_oh], dim=1)))

    class VitalsActionEncoder(nn.Module):
        """Encodes preprocessed vitals (already scaled in submission/obs.py) + prev_action."""

        def __init__(self, cfg: PolicyConfig) -> None:
            super().__init__()
            self.num_vitals = cfg.num_vitals
            self.num_weapon_slots = 8
            vitals_encoded_dim = 24 if cfg.num_vitals == 17 else cfg.num_vitals
            in_dim = vitals_encoded_dim + cfg.num_actions
            self.net = nn.Sequential(
                nn.Linear(in_dim, cfg.vitals_dim),
                nn.LayerNorm(cfg.vitals_dim),
                nn.SiLU(),
                nn.Linear(cfg.vitals_dim, cfg.vitals_dim),
                nn.LayerNorm(cfg.vitals_dim),
                nn.SiLU(),
            )

        def forward(self, vitals: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
            v = vitals.float()
            if self.num_vitals == 17:
                # obs.py scales: health/100, armor/100, selected_weapon/7, ammo/scales, keys, vel/20
                health = torch.clamp(v[..., 0:1], 0.0, 2.0)
                armor = torch.clamp(v[..., 1:2], 0.0, 2.0)
                weapon_idx = torch.clamp(torch.round(v[..., 2] * 7.0).long(), 0, self.num_weapon_slots - 1)
                weapon_oh = F.one_hot(weapon_idx, num_classes=self.num_weapon_slots).to(dtype=v.dtype)
                ammo = torch.log1p(torch.clamp(v[..., 3:8], min=0.0))
                keys = torch.clamp(v[..., 8:14], 0.0, 1.0)
                vel = torch.tanh(v[..., 14:17])
                norm_v = torch.cat([health, armor, weapon_oh, ammo, keys, vel], dim=-1)
            else:
                norm_v = torch.tanh(v)
            return self.net(torch.cat([norm_v, prev_action.to(dtype=norm_v.dtype)], dim=-1))

    class MaskedEntityAttentionEncoder(nn.Module):
        """Masked multi-head cross-attention over visible entities with empty-room fallback."""

        def __init__(self, cfg: PolicyConfig) -> None:
            super().__init__()
            self.num_entity_types = cfg.num_entity_types
            self.num_heads = cfg.entity_attn_heads
            self.num_queries = cfg.entity_num_queries
            self.entity_dim = cfg.entity_dim
            self.head_dim = cfg.entity_dim // cfg.entity_attn_heads

            self.type_emb = nn.Embedding(cfg.num_entity_types, cfg.entity_type_embed_dim)
            entity_feat_dim = 6 if cfg.num_entity_features == 5 else cfg.num_entity_features
            self.num_entity_features = cfg.num_entity_features
            self.entity_mlp = nn.Sequential(
                nn.Linear(cfg.entity_type_embed_dim + entity_feat_dim, cfg.entity_dim),
                nn.LayerNorm(cfg.entity_dim),
                nn.SiLU(),
                nn.Linear(cfg.entity_dim, cfg.entity_dim),
                nn.SiLU(),
            )
            self.query_proj = nn.Linear(
                cfg.vitals_dim + cfg.ray_out_dim,
                cfg.entity_num_queries * cfg.entity_dim,
            )
            self.k_proj = nn.Linear(cfg.entity_dim, cfg.entity_dim)
            self.v_proj = nn.Linear(cfg.entity_dim, cfg.entity_dim)
            self.out_proj = nn.Sequential(
                nn.Linear(cfg.entity_num_queries * cfg.entity_dim, cfg.entity_out_dim),
                nn.LayerNorm(cfg.entity_out_dim),
                nn.SiLU(),
            )
            self.empty_embed = nn.Parameter(torch.empty(1, cfg.entity_out_dim))

        def _normalize_features(self, ent_feat: torch.Tensor, ent_mask: torch.Tensor) -> torch.Tensor:
            f = ent_feat.float()
            mask = ent_mask.to(dtype=f.dtype).unsqueeze(-1)
            if self.num_entity_features == 5:
                # obs.py already divides [dist/1024, sin, cos, rel_vel_fwd, rel_vel_right] by [1, 1, 1, 20, 20]
                dist = torch.clamp(f[..., 0], min=0.0)
                dist_log = torch.log1p(dist) / _LOG_FIVE
                dist_near = 1.0 / (1.0 + dist)
                sin_a = torch.clamp(f[..., 1], -1.0, 1.0)
                cos_a = torch.clamp(f[..., 2], -1.0, 1.0)
                rel_vel_fwd = torch.tanh(f[..., 3])
                rel_vel_right = torch.tanh(f[..., 4])
                out = torch.stack([dist_log, dist_near, sin_a, cos_a, rel_vel_fwd, rel_vel_right], dim=-1)
                return out * mask
            return torch.tanh(f) * mask

        def forward(
            self,
            ent_feat: torch.Tensor,
            ent_type: torch.Tensor,
            ent_mask: torch.Tensor,
            vitals_feat: torch.Tensor,
            ray_feat: torch.Tensor,
        ) -> torch.Tensor:
            batch_size, max_entities, _ = ent_feat.shape
            norm_feats = self._normalize_features(ent_feat, ent_mask)
            types = torch.clamp(ent_type.long(), 0, self.num_entity_types - 1)
            ent_h = self.entity_mlp(torch.cat([self.type_emb(types), norm_feats], dim=-1))

            cond = torch.cat([vitals_feat, ray_feat], dim=-1)
            q = (
                self.query_proj(cond)
                .view(batch_size, self.num_queries, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = self.k_proj(ent_h).view(batch_size, max_entities, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(ent_h).view(batch_size, max_entities, self.num_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
            valid = ent_mask > 0.5
            has_any = valid.any(dim=-1)

            # Guarantee >=1 unmasked key per row so softmax never produces NaNs in empty rooms.
            safe_valid = valid.clone()
            safe_valid[~has_any, 0] = True
            scores = scores.masked_fill(~safe_valid[:, None, None, :], -1e9)
            attn = torch.softmax(scores, dim=-1) * valid[:, None, None, :].to(dtype=scores.dtype)

            pooled = (
                torch.matmul(attn, v)
                .transpose(1, 2)
                .contiguous()
                .view(batch_size, self.num_queries * self.entity_dim)
            )
            out = self.out_proj(pooled)
            return torch.where(has_any.unsqueeze(-1), out, self.empty_embed.expand(batch_size, -1))

    class TelemetryPolicy(nn.Module):
        """Zero-pixel recurrent telemetry actor-critic implementing the frozen Role 1/2 contract."""

        def __init__(self, cfg: PolicyConfig) -> None:
            super().__init__()
            self.config = cfg
            self.ray_encoder = CircularRayEncoder(cfg)
            self.vitals_encoder = VitalsActionEncoder(cfg)
            self.entity_encoder = MaskedEntityAttentionEncoder(cfg)

            fused_dim = cfg.ray_out_dim + cfg.entity_out_dim + cfg.vitals_dim
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU(),
            )
            self.gru = nn.GRU(cfg.hidden_size, cfg.hidden_size, batch_first=False)
            self.post_gru_norm = nn.LayerNorm(cfg.hidden_size)
            self.actor = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.head_hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.head_hidden_dim, cfg.num_actions),
            )
            self.critic = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.head_hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.head_hidden_dim, 1),
            )
            self._init_weights()

        def _init_weights(self) -> None:
            self.apply(lambda m: _init_orthogonal(m, gain=math.sqrt(2.0)))
            nn.init.normal_(self.entity_encoder.type_emb.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.entity_encoder.empty_embed, mean=0.0, std=0.02)
            for name, param in self.gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

            actor_out: nn.Linear = self.actor[-1]  # type: ignore[assignment]
            nn.init.orthogonal_(actor_out.weight, gain=0.01)
            nn.init.zeros_(actor_out.bias)
            with torch.no_grad():
                if self.config.num_actions > 7:
                    actor_out.bias[7] = self.config.use_button_init_bias
                if self.config.num_actions > 8:
                    actor_out.bias[8] = self.config.weapon_switch_init_bias

            critic_out: nn.Linear = self.critic[-1]  # type: ignore[assignment]
            nn.init.orthogonal_(critic_out.weight, gain=1.0)
            nn.init.zeros_(critic_out.bias)

        def init_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
            """Return zero recurrent state of shape (1, B, hidden_size)."""
            return torch.zeros(1, batch_size, self.config.hidden_size, dtype=torch.float32, device=device)

        def _encode(self, obs: dict[str, torch.Tensor], prev_action: torch.Tensor) -> torch.Tensor:
            ray_feat = self.ray_encoder(obs["ray_dist"], obs["ray_type"])
            vitals_feat = self.vitals_encoder(obs["vitals"], prev_action)
            ent_feat = self.entity_encoder(
                obs["ent_feat"],
                obs["ent_type"],
                obs["ent_mask"],
                vitals_feat,
                ray_feat,
            )
            return self.fusion(torch.cat([ray_feat, ent_feat, vitals_feat], dim=-1))

        def _heads(self, gru_out: torch.Tensor, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            feat = self.post_gru_norm(gru_out + fused) if self.config.use_gru_residual else self.post_gru_norm(gru_out)
            logits = self.actor(feat)
            value = self.critic(feat).squeeze(-1)
            return logits, value

        def act_step(
            self,
            obs_batch: dict[str, torch.Tensor],
            prev_action: torch.Tensor,
            hidden: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Single-step forward for rollout collection and `DoomAgent.act()`."""
            fused = self._encode(obs_batch, prev_action)
            out, next_hidden = self.gru(fused.unsqueeze(0), hidden)
            logits, value = self._heads(out.squeeze(0), fused)
            return logits, value, next_hidden

        def evaluate_chunks(
            self,
            obs_seq: dict[str, torch.Tensor],
            prev_action_seq: torch.Tensor,
            action_seq: torch.Tensor,
            init_hidden: torch.Tensor,
            episode_starts: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Evaluate contiguous (T, B, ...) rollout chunks with exact episode-boundary resets."""
            t_steps, batch_size = obs_seq["vitals"].shape[:2]
            obs_flat = {k: v.reshape(t_steps * batch_size, *v.shape[2:]) for k, v in obs_seq.items()}
            prev_flat = prev_action_seq.reshape(t_steps * batch_size, self.config.num_actions)
            fused_seq = self._encode(obs_flat, prev_flat).reshape(t_steps, batch_size, self.config.hidden_size)

            h = init_hidden
            starts = episode_starts.to(device=h.device).bool()
            keep_h = (~starts).to(dtype=h.dtype)
            reset_steps = torch.nonzero(starts.any(dim=1), as_tuple=False).squeeze(-1).tolist()

            if not reset_steps:
                gru_out, h = self.gru(fused_seq, h)
            else:
                outs = []
                prev_t = 0
                for t in reset_steps:
                    if t > prev_t:
                        seg_out, h = self.gru(fused_seq[prev_t:t], h)
                        outs.append(seg_out)
                    h = h * keep_h[t].view(1, batch_size, 1)
                    prev_t = t
                if prev_t < t_steps:
                    seg_out, h = self.gru(fused_seq[prev_t:t_steps], h)
                    outs.append(seg_out)
                gru_out = torch.cat(outs, dim=0)

            logits, value = self._heads(gru_out, fused_seq)
            bce = F.binary_cross_entropy_with_logits(logits, action_seq.to(dtype=logits.dtype), reduction="none")
            log_prob = -bce.sum(dim=-1)
            probs = torch.sigmoid(logits)
            entropy = -(
                probs * F.logsigmoid(logits) + (1.0 - probs) * F.logsigmoid(-logits)
            ).sum(dim=-1)
            return log_prob, entropy, value

    if config.arch == "stub":
        return StubRecurrentPolicy(config)
    if config.arch == "telemetry_gru_v1":
        return TelemetryPolicy(config)
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
    cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": config.to_dict(),
        "state_dict": cpu_state_dict,
        "meta": dict(meta or {}),
    }
    torch.save(payload, out)
    return out


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Any, PolicyConfig, dict[str, Any]]:
    """Load a checkpoint created by `save_checkpoint` onto `device` in eval mode."""
    import torch

    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    except (RuntimeError, pickle.UnpicklingError):
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = PolicyConfig.from_dict(payload["config"])
    model = build_policy(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, dict(payload.get("meta", {}))
