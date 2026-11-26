"""Shared Policy interface, config, checkpoint I/O, and telemetry architectures.

Ownership (see REVISED_PLAN.md & training/README.md):
  - Role 1 (Policy & Exported Agent) implements `TelemetryPolicy` (`arch="telemetry_gru_v1"`):
    circular-conv rays, masked-attention entities with combat kinematics, category counts &
    learned empty-room output, enriched vitals + prev_action GRU, and factorized
    (3x 3-way categorical + 3x Bernoulli) or 9-Bernoulli actor + scalar critic.
  - Role 2 (PPO & Rollout Collection) calls ONLY the frozen contract below:
      * `policy.init_hidden(batch_size, device)`
      * `policy.act_step(obs_batch, prev_action, hidden)`
      * `policy.sample_action(logits, deterministic=False, threshold=0.5, generator=None)`
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
    from submission.ego_map import EGO_MAP_CHANNELS, EGO_MAP_SIZE
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
    from ego_map import EGO_MAP_CHANNELS, EGO_MAP_SIZE
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
FACTORIZED_LOGIT_DIM = 12  # 3x3 categorical (fwd/back, strafe, turn) + 3 Bernoulli (attack, use, weapon)


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
    entity_num_queries: int = 4
    head_hidden_dim: int = 128
    action_head: str = "factorized"  # "factorized" (3x3-cat + 3-bern) | "bernoulli" (9-bern)
    use_gru_residual: bool = True
    use_button_init_bias: float = -1.5
    weapon_switch_init_bias: float = -2.0
    eval_deterministic: bool = False
    eval_temperature: float = 1.0
    action_threshold: float = 0.5
    use_ego_map: bool = False
    ego_map_out_dim: int = 128
    ego_map_channels: int = EGO_MAP_CHANNELS
    ego_map_size: int = EGO_MAP_SIZE

    def __post_init__(self) -> None:
        if self.num_rays % 4 != 0:
            raise ValueError(f"num_rays must be divisible by 4, got {self.num_rays}")
        if self.entity_dim % self.entity_attn_heads != 0:
            raise ValueError(
                f"entity_dim ({self.entity_dim}) must be divisible by "
                f"entity_attn_heads ({self.entity_attn_heads})"
            )
        if self.action_head not in ("factorized", "bernoulli"):
            raise ValueError(f"action_head must be 'factorized' or 'bernoulli', got {self.action_head!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyConfig:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        # Backward compatibility for legacy checkpoints saved before `action_head` existed
        if "action_head" not in valid:
            valid["action_head"] = "bernoulli"
        if "entity_num_queries" not in valid:
            valid["entity_num_queries"] = 2
        return cls(**valid)


def _pair_to_cat_index(pos_btn, neg_btn):
    """Map mutually-exclusive binary pair (pos, neg) -> class index {0: none, 1: pos, 2: neg}."""
    import torch

    pos = pos_btn > 0.5
    neg = neg_btn > 0.5
    idx = torch.zeros_like(pos, dtype=torch.long)
    idx = torch.where(pos & ~neg, torch.ones_like(idx), idx)
    idx = torch.where(neg & ~pos, torch.full_like(idx, 2), idx)
    return idx


def sample_actions_from_logits(
    logits,
    *,
    action_head: str = "factorized",
    deterministic: bool = False,
    threshold: float = 0.5,
    temperature: float = 1.0,
    generator: Any = None,
):
    """Sample (or argmax) a 9-button binary action tensor and its exact joint log-prob.

    Supports both:
      - `"factorized"`: 12 logits = 3x3-way categoricals (fwd/back, left/right, turnL/turnR)
                        + 3 independent Bernoullis (attack, use, select_next_weapon).
      - `"bernoulli"`:  9 independent Bernoulli logits.
    """
    import torch
    import torch.nn.functional as F

    temp = max(1e-4, float(temperature))
    sample_logits = logits if (deterministic or abs(temp - 1.0) <= 1e-6) else (logits / temp)

    if action_head == "bernoulli":
        probs = torch.sigmoid(sample_logits)
        if deterministic:
            act = (probs >= threshold).to(dtype=logits.dtype)
        else:
            act = torch.bernoulli(probs, generator=generator).to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, act, reduction="none")
        log_prob = -bce.sum(dim=-1)
        return act, log_prob

    # Factorized: 3x3 categorical + 3 Bernoulli -> (..., 9) binary action
    lead_shape = logits.shape[:-1]
    cat_logits = logits[..., :9].reshape(*lead_shape, 3, 3)
    bern_logits = logits[..., 9:12]
    sample_cat_logits = sample_logits[..., :9].reshape(*lead_shape, 3, 3)
    sample_bern_logits = sample_logits[..., 9:12]

    cat_log_probs = F.log_softmax(cat_logits, dim=-1)
    if deterministic:
        cat_idx = torch.argmax(cat_logits, dim=-1)
    else:
        cat_probs = torch.softmax(sample_cat_logits, dim=-1).reshape(-1, 3)
        cat_idx = torch.multinomial(cat_probs, num_samples=1, generator=generator).reshape(*lead_shape, 3)

    cat_lp = cat_log_probs.gather(-1, cat_idx.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    bern_probs = torch.sigmoid(sample_bern_logits)
    if deterministic:
        bern_act = (torch.sigmoid(bern_logits) >= threshold).to(dtype=logits.dtype)
    else:
        bern_act = torch.bernoulli(bern_probs, generator=generator).to(dtype=logits.dtype)
    bern_lp = -F.binary_cross_entropy_with_logits(bern_logits, bern_act, reduction="none").sum(dim=-1)

    act = torch.zeros(*lead_shape, NUM_ACTIONS, dtype=logits.dtype, device=logits.device)
    for k in range(3):
        act[..., 2 * k] = (cat_idx[..., k] == 1).to(dtype=logits.dtype)
        act[..., 2 * k + 1] = (cat_idx[..., k] == 2).to(dtype=logits.dtype)
    act[..., 6:9] = bern_act
    return act, cat_lp + bern_lp


def action_logprob_and_entropy(logits, action_9d, *, action_head: str = "factorized"):
    """Compute exact joint `(log_prob, entropy)` of `action_9d` under `logits`."""
    import torch
    import torch.nn.functional as F

    act = action_9d.to(dtype=logits.dtype)
    if action_head == "bernoulli":
        bce = F.binary_cross_entropy_with_logits(logits, act, reduction="none")
        log_prob = -bce.sum(dim=-1)
        probs = torch.sigmoid(logits)
        entropy = -(probs * F.logsigmoid(logits) + (1.0 - probs) * F.logsigmoid(-logits)).sum(dim=-1)
        return log_prob, entropy

    lead_shape = logits.shape[:-1]
    cat_logits = logits[..., :9].reshape(*lead_shape, 3, 3)
    bern_logits = logits[..., 9:12]

    cat_log_probs = F.log_softmax(cat_logits, dim=-1)
    cat_probs = torch.softmax(cat_logits, dim=-1)
    cat_targets = torch.stack(
        [
            _pair_to_cat_index(act[..., 0], act[..., 1]),
            _pair_to_cat_index(act[..., 2], act[..., 3]),
            _pair_to_cat_index(act[..., 4], act[..., 5]),
        ],
        dim=-1,
    )
    cat_lp = cat_log_probs.gather(-1, cat_targets.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    cat_ent = -(cat_probs * cat_log_probs).sum(dim=-1).sum(dim=-1)

    bern_act = act[..., 6:9]
    bern_lp = -F.binary_cross_entropy_with_logits(bern_logits, bern_act, reduction="none").sum(dim=-1)
    bern_probs = torch.sigmoid(bern_logits)
    bern_ent = -(
        bern_probs * F.logsigmoid(bern_logits) + (1.0 - bern_probs) * F.logsigmoid(-bern_logits)
    ).sum(dim=-1)
    return cat_lp + bern_lp, cat_ent + bern_ent


def build_policy(config: PolicyConfig):
    """Factory that builds the PyTorch policy matching `config.arch`.

    Supports both `"stub"` (Role 2 smoke tests) and `"telemetry_gru_v1"` (Role 1 full policy).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _init_orthogonal(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    class StubRecurrentPolicy(nn.Module):
        """Lightweight GRU policy implementing the exact Stage-1 contract.

        Always uses 9-Bernoulli logits for fast smoke tests and contract checks.
        """

        def __init__(self, cfg: PolicyConfig):
            super().__init__()
            self.config = cfg
            self.action_head = "bernoulli"
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

        def sample_from_logits(
            self,
            logits: torch.Tensor,
            *,
            deterministic: bool = False,
            threshold: float = 0.5,
            temperature: float = 1.0,
            generator: Any = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return sample_actions_from_logits(
                logits,
                action_head=self.action_head,
                deterministic=deterministic,
                threshold=threshold,
                temperature=temperature,
                generator=generator,
            )

        def sample_action(
            self,
            obs_batch: dict[str, torch.Tensor],
            prev_action: torch.Tensor,
            hidden: torch.Tensor,
            *,
            deterministic: bool = False,
            threshold: float = 0.5,
            temperature: float = 1.0,
            generator: Any = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits, value, next_hidden = self.act_step(obs_batch, prev_action, hidden)
            act, log_prob = self.sample_from_logits(
                logits,
                deterministic=deterministic,
                threshold=threshold,
                temperature=temperature,
                generator=generator,
            )
            return act, log_prob, value, next_hidden

        def logprob_and_entropy(
            self, logits: torch.Tensor, action_9d: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return action_logprob_and_entropy(logits, action_9d, action_head=self.action_head)

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
            log_prob, entropy = self.logprob_and_entropy(logits, action_seq)
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
        """Encodes preprocessed vitals + explicit ammo/health/speed flags + prev_action."""

        def __init__(self, cfg: PolicyConfig) -> None:
            super().__init__()
            self.num_vitals = cfg.num_vitals
            self.num_weapon_slots = 8
            # 24 base dims + 4 explicit flags (selected_ammo_empty, has_other_ammo, low_health, horiz_speed)
            vitals_encoded_dim = 28 if cfg.num_vitals == 17 else cfg.num_vitals
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

                # Explicit binary/scalar cues for empty ammo, low HP, and wall-snag detection
                selected_ammo_empty = (v[..., 3:4] <= 1e-6).to(dtype=v.dtype)
                has_other_ammo = (v[..., 4:8].sum(dim=-1, keepdim=True) > 1e-6).to(dtype=v.dtype)
                low_health = (v[..., 0:1] < 0.35).to(dtype=v.dtype)
                horiz_speed = torch.tanh(torch.hypot(v[..., 14:15], v[..., 15:16]))

                norm_v = torch.cat(
                    [
                        health,
                        armor,
                        weapon_oh,
                        ammo,
                        keys,
                        vel,
                        selected_ammo_empty,
                        has_other_ammo,
                        low_health,
                        horiz_speed,
                    ],
                    dim=-1,
                )
            else:
                norm_v = torch.tanh(v)
            return self.net(torch.cat([norm_v, prev_action.to(dtype=norm_v.dtype)], dim=-1))

    class MaskedEntityAttentionEncoder(nn.Module):
        """Masked multi-head cross-attention over visible entities with combat kinematics & counts.

        Derives exact ego-frame kinematics consistent with `env/doom_env/sensors.py`:
          - `rel_angle` is 0 straight ahead and positive to the player's LEFT.
          - `rel_vel_right` is positive to the player's RIGHT.
          - Therefore radial unit vector toward entity is `(+cos_a, -sin_a)`:
              * closing_speed = -(rel_vel_fwd * cos_a - rel_vel_right * sin_a)
              * lateral_speed = rel_vel_fwd * sin_a + rel_vel_right * cos_a
        """

        _COUNT_SUMMARY_DIM = 8  # 7 category log1p counts + nearest threat proximity

        def __init__(self, cfg: PolicyConfig) -> None:
            super().__init__()
            self.num_entity_types = cfg.num_entity_types
            self.num_heads = cfg.entity_attn_heads
            self.num_queries = cfg.entity_num_queries
            self.entity_dim = cfg.entity_dim
            self.head_dim = cfg.entity_dim // cfg.entity_attn_heads

            self.type_emb = nn.Embedding(cfg.num_entity_types, cfg.entity_type_embed_dim)
            entity_feat_dim = 10 if cfg.num_entity_features == 5 else cfg.num_entity_features
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
                nn.Linear(cfg.entity_num_queries * cfg.entity_dim + self._COUNT_SUMMARY_DIM, cfg.entity_out_dim),
                nn.LayerNorm(cfg.entity_out_dim),
                nn.SiLU(),
            )
            self.empty_embed = nn.Parameter(torch.empty(1, cfg.entity_out_dim))

        def _normalize_features(self, ent_feat: torch.Tensor, ent_mask: torch.Tensor) -> torch.Tensor:
            f = ent_feat.float()
            mask = ent_mask.to(dtype=f.dtype).unsqueeze(-1)
            if self.num_entity_features == 5:
                # obs.py scales [dist/1024, sin, cos, rel_vel_fwd, rel_vel_right] by [1, 1, 1, 20, 20]
                dist = torch.clamp(f[..., 0], min=0.0)
                dist_log = torch.log1p(dist) / _LOG_FIVE
                dist_near = 1.0 / (1.0 + dist)
                sin_a = torch.clamp(f[..., 1], -1.0, 1.0)
                cos_a = torch.clamp(f[..., 2], -1.0, 1.0)
                raw_fwd = f[..., 3]
                raw_right = f[..., 4]
                rel_vel_fwd = torch.tanh(raw_fwd)
                rel_vel_right = torch.tanh(raw_right)

                # Sign-verified ego kinematics (sensors.py: +sin_a = LEFT, +raw_right = RIGHT)
                closing_raw = -(raw_fwd * cos_a - raw_right * sin_a)
                lateral_raw = raw_fwd * sin_a + raw_right * cos_a
                closing_speed = torch.tanh(closing_raw)
                lateral_speed = torch.tanh(lateral_raw)
                inv_tti = torch.tanh(torch.clamp(closing_raw, min=0.0) / (dist + 0.05))
                crosshair_align = torch.exp(-torch.abs(sin_a) / 0.1) * (cos_a > 0.0).to(dtype=f.dtype)

                out = torch.stack(
                    [
                        dist_log,
                        dist_near,
                        sin_a,
                        cos_a,
                        rel_vel_fwd,
                        rel_vel_right,
                        closing_speed,
                        lateral_speed,
                        inv_tti,
                        crosshair_align,
                    ],
                    dim=-1,
                )
                return out * mask
            return torch.tanh(f) * mask

        def _category_summary(
            self, ent_feat: torch.Tensor, types: torch.Tensor, valid: torch.Tensor
        ) -> torch.Tensor:
            """Compute explicit per-category log1p counts and nearest-threat proximity."""
            dtype = ent_feat.dtype
            v = valid.to(dtype=dtype)
            # Entity type IDs (env/doom_env/spaces.py):
            # 1=melee, 2=hitscan, 3=proj_enemy, 4=projectile, 5=health, 6=armor, 7=ammo, 8=weapon, 9=key
            c_melee = ((types == 1) & valid).to(dtype=dtype).sum(dim=-1)
            c_hitscan = ((types == 2) & valid).to(dtype=dtype).sum(dim=-1)
            c_proj_e = ((types == 3) & valid).to(dtype=dtype).sum(dim=-1)
            c_missile = ((types == 4) & valid).to(dtype=dtype).sum(dim=-1)
            c_heal = (((types == 5) | (types == 6)) & valid).to(dtype=dtype).sum(dim=-1)
            c_ammo = (((types == 7) | (types == 8)) & valid).to(dtype=dtype).sum(dim=-1)
            c_key = ((types == 9) & valid).to(dtype=dtype).sum(dim=-1)

            counts = torch.log1p(
                torch.stack([c_melee, c_hitscan, c_proj_e, c_missile, c_heal, c_ammo, c_key], dim=-1)
            )
            dist = torch.clamp(ent_feat[..., 0].float(), min=0.0)
            is_threat = ((types >= 1) & (types <= 4) & valid).to(dtype=dtype)
            threat_prox = (1.0 / (1.0 + dist)) * is_threat * v
            nearest_threat = threat_prox.max(dim=-1, keepdim=True).values
            return torch.cat([counts, nearest_threat], dim=-1)

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
            summary = self._category_summary(ent_feat, types, valid)
            out = self.out_proj(torch.cat([pooled, summary], dim=-1))
            return torch.where(has_any.unsqueeze(-1), out, self.empty_embed.expand(batch_size, -1))

    class SpatialMapEncoder(nn.Module):
        """Compact 2D CNN encoding the 8-channel egocentric odometric spatial memory."""

        def __init__(self, cfg: PolicyConfig) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(cfg.ego_map_channels, 24, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
            )
            out_hw = cfg.ego_map_size
            for _ in range(3):
                out_hw = (out_hw + 1) // 2
            self.proj = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(48 * out_hw * out_hw, cfg.ego_map_out_dim),
                nn.LayerNorm(cfg.ego_map_out_dim),
                nn.SiLU(),
            )

        def forward(self, ego_map: torch.Tensor) -> torch.Tensor:
            if not ego_map.is_floating_point():
                x = ego_map.float() / 255.0
            else:
                x = ego_map.float()
                if x.numel() > 0 and float(x.max().item()) > 1.5:
                    x = x / 255.0
            return self.proj(self.conv(x))

    class TelemetryPolicy(nn.Module):
        """Zero-pixel recurrent telemetry actor-critic implementing the frozen Role 1/2 contract."""

        def __init__(self, cfg: PolicyConfig) -> None:
            super().__init__()
            self.config = cfg
            self.action_head = cfg.action_head
            self.use_ego_map = bool(cfg.use_ego_map)
            self.ray_encoder = CircularRayEncoder(cfg)
            self.vitals_encoder = VitalsActionEncoder(cfg)
            self.entity_encoder = MaskedEntityAttentionEncoder(cfg)
            self.ego_map_encoder = SpatialMapEncoder(cfg) if self.use_ego_map else None

            fused_dim = cfg.ray_out_dim + cfg.entity_out_dim + cfg.vitals_dim
            if self.use_ego_map:
                fused_dim += cfg.ego_map_out_dim
            self.fusion = nn.Sequential(
                nn.Linear(fused_dim, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU(),
            )
            self.gru = nn.GRU(cfg.hidden_size, cfg.hidden_size, batch_first=False)
            self.post_gru_norm = nn.LayerNorm(cfg.hidden_size)
            actor_out_dim = FACTORIZED_LOGIT_DIM if cfg.action_head == "factorized" else cfg.num_actions
            self.actor = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.head_hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.head_hidden_dim, actor_out_dim),
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
                if self.action_head == "factorized":
                    actor_out.bias[10] = self.config.use_button_init_bias
                    actor_out.bias[11] = self.config.weapon_switch_init_bias
                else:
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
            parts = [ray_feat, ent_feat, vitals_feat]
            if self.ego_map_encoder is not None:
                ego_map = obs.get("ego_map")
                if ego_map is None:
                    b = vitals_feat.shape[0]
                    ego_map = torch.zeros(
                        b,
                        self.config.ego_map_channels,
                        self.config.ego_map_size,
                        self.config.ego_map_size,
                        dtype=vitals_feat.dtype,
                        device=vitals_feat.device,
                    )
                parts.append(self.ego_map_encoder(ego_map))
            return self.fusion(torch.cat(parts, dim=-1))

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

        def sample_from_logits(
            self,
            logits: torch.Tensor,
            *,
            deterministic: bool = False,
            threshold: float = 0.5,
            temperature: float = 1.0,
            generator: Any = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return sample_actions_from_logits(
                logits,
                action_head=self.action_head,
                deterministic=deterministic,
                threshold=threshold,
                temperature=temperature,
                generator=generator,
            )

        def sample_action(
            self,
            obs_batch: dict[str, torch.Tensor],
            prev_action: torch.Tensor,
            hidden: torch.Tensor,
            *,
            deterministic: bool = False,
            threshold: float = 0.5,
            temperature: float = 1.0,
            generator: Any = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            logits, value, next_hidden = self.act_step(obs_batch, prev_action, hidden)
            act, log_prob = self.sample_from_logits(
                logits,
                deterministic=deterministic,
                threshold=threshold,
                temperature=temperature,
                generator=generator,
            )
            return act, log_prob, value, next_hidden

        def logprob_and_entropy(
            self, logits: torch.Tensor, action_9d: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return action_logprob_and_entropy(logits, action_9d, action_head=self.action_head)

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
            log_prob, entropy = self.logprob_and_entropy(logits, action_seq)
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
