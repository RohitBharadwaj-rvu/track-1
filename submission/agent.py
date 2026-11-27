"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

try:
    from submission.ego_map import EgocentricSpatialMap
    from submission.obs import NUM_ACTIONS, obs_to_torch, preprocess_obs
    from submission.policy import PolicyConfig, load_checkpoint
except ImportError:  # pragma: no cover - fallback when submission/ is on sys.path directly
    from ego_map import EgocentricSpatialMap
    from obs import NUM_ACTIONS, obs_to_torch, preprocess_obs
    from policy import PolicyConfig, load_checkpoint

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "policy.pt"


def _make_dummy_obs(config: PolicyConfig) -> dict:
    mask = np.zeros(config.max_entities, dtype=np.int8)
    mask[0] = 1
    return {
        "vitals": np.zeros(config.num_vitals, dtype=np.float32),
        "rays": {
            "distance": np.ones(config.num_rays, dtype=np.float32),
            "hit_type": np.zeros(config.num_rays, dtype=np.int64),
        },
        "visible_entities": {
            "features": np.zeros((config.max_entities, config.num_entity_features), dtype=np.float32),
            "entity_type": np.zeros(config.max_entities, dtype=np.int64),
            "mask": mask,
        },
    }


_RAY_MIRROR_IDX = (-np.arange(64, dtype=np.int64)) % 64
_ACTION_MIRROR_IDX = np.array([0, 1, 3, 2, 5, 4, 6, 7, 8], dtype=np.int64)
_FACTORIZED_LOGIT_MIRROR_IDX = [0, 1, 2, 3, 5, 4, 6, 8, 7, 9, 10, 11]


def _mirror_raw_obs(raw_obs: dict) -> dict:
    rays = raw_obs["rays"]
    ray_dist = np.asarray(rays["distance"])
    ray_hit = np.asarray(rays["hit_type"])
    ray_idx = _RAY_MIRROR_IDX if ray_dist.shape[-1] == 64 else ((-np.arange(ray_dist.shape[-1], dtype=np.int64)) % ray_dist.shape[-1])
    ve = raw_obs["visible_entities"]
    ve_feats = np.array(ve["features"], copy=True)
    ve_feats[..., 1] = -ve_feats[..., 1]
    ve_feats[..., 4] = -ve_feats[..., 4]
    vitals = np.array(raw_obs["vitals"], copy=True)
    vitals[..., 15] = -vitals[..., 15]
    out = dict(raw_obs)
    out["rays"] = {
        "distance": np.ascontiguousarray(ray_dist[..., ray_idx]),
        "hit_type": np.ascontiguousarray(ray_hit[..., ray_idx]),
    }
    out["visible_entities"] = {
        "features": ve_feats,
        "entity_type": np.array(ve["entity_type"], copy=True),
        "mask": np.array(ve["mask"], copy=True),
    }
    out["vitals"] = vitals
    return out


def _unmirror_logits(logits: torch.Tensor, *, action_head: str = "factorized") -> torch.Tensor:
    if action_head == "factorized":
        return logits[..., _FACTORIZED_LOGIT_MIRROR_IDX]
    return logits[..., _ACTION_MIRROR_IDX]


def _obs_to_batch_tensors(
    observation: dict,
    device: str,
    ego_map: EgocentricSpatialMap | None = None,
    prev_action_np: np.ndarray | None = None,
    *,
    ego_map_mir: EgocentricSpatialMap | None = None,
    mirror_ensemble: bool = False,
) -> dict[str, torch.Tensor]:
    pa = prev_action_np if prev_action_np is not None else np.zeros(NUM_ACTIONS, dtype=np.int8)
    prep = preprocess_obs(observation)
    if ego_map is not None:
        prep["ego_map"] = ego_map.update(observation, pa)
    if not mirror_ensemble:
        batched = {k: v[None, ...] for k, v in prep.items()}
        return obs_to_torch(batched, device=device)

    mir_obs = _mirror_raw_obs(observation)
    prep_mir = preprocess_obs(mir_obs)
    if ego_map_mir is not None:
        prep_mir["ego_map"] = ego_map_mir.update(mir_obs, np.ascontiguousarray(pa[_ACTION_MIRROR_IDX]))
    batched = {k: np.stack([prep[k], prep_mir[k]], axis=0) for k in prep}
    return obs_to_torch(batched, device=device)


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once. Load weights from WEIGHTS_DIR onto `device` ("cuda" or "cpu")."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        weights_override = os.environ.get("DOOM_AGENT_WEIGHTS")
        weights_path = Path(weights_override) if weights_override else DEFAULT_WEIGHTS_PATH
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Export weights with `uv run python training/export_smoke_weights.py`."
            )

        self.policy, self.config, self.checkpoint_meta = load_checkpoint(weights_path, device=self.device)
        self.policy.eval()
        self.deterministic = bool(self.config.eval_deterministic)
        self.temperature = float(getattr(self.config, "eval_temperature", 1.0))
        self.action_threshold = float(self.config.action_threshold)
        self.mirror_ensemble = bool(getattr(self.config, "eval_mirror_ensemble", False))
        batch_size = 2 if self.mirror_ensemble else 1
        self._ego_map = EgocentricSpatialMap() if bool(getattr(self.config, "use_ego_map", False)) else None
        self._ego_map_mir = (
            EgocentricSpatialMap() if (self._ego_map is not None and self.mirror_ensemble) else None
        )
        self._hidden = self.policy.init_hidden(batch_size, self.device)
        self._prev_action = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.float32, device=self.device)
        self._prev_action_np = np.zeros(NUM_ACTIONS, dtype=np.int8)

        # Warm up both active-entity and empty-room branches in __init__ (60s budget)
        # so the first act() step never pays CUDA/PyTorch initialization latency.
        with torch.inference_mode():
            warmup_obs = _make_dummy_obs(self.config)
            self.policy.act_step(
                _obs_to_batch_tensors(
                    warmup_obs,
                    self.device,
                    self._ego_map,
                    self._prev_action_np,
                    ego_map_mir=self._ego_map_mir,
                    mirror_ensemble=self.mirror_ensemble,
                ),
                self._prev_action,
                self._hidden,
            )
            warmup_obs["visible_entities"]["mask"][0] = 0
            self.policy.act_step(
                _obs_to_batch_tensors(
                    warmup_obs,
                    self.device,
                    self._ego_map,
                    self._prev_action_np,
                    ego_map_mir=self._ego_map_mir,
                    mirror_ensemble=self.mirror_ensemble,
                ),
                self._prev_action,
                self._hidden,
            )

        self.reset()

    def reset(self) -> None:
        """Called before every map. Clear recurrent state, spatial map, and previous-action buffer."""
        batch_size = 2 if self.mirror_ensemble else 1
        self._hidden = self.policy.init_hidden(batch_size, self.device)
        self._prev_action = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.float32, device=self.device)
        self._prev_action_np = np.zeros(NUM_ACTIONS, dtype=np.int8)
        if self._ego_map is not None:
            self._ego_map.reset()
        if self._ego_map_mir is not None:
            self._ego_map_mir.reset()

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int8 array of shape (9,) with values in {0, 1}."""
        with torch.inference_mode():
            obs_t = _obs_to_batch_tensors(
                observation,
                self.device,
                self._ego_map,
                self._prev_action_np,
                ego_map_mir=self._ego_map_mir,
                mirror_ensemble=self.mirror_ensemble,
            )
            if not self.mirror_ensemble:
                action_t, _, _, next_hidden = self.policy.sample_action(
                    obs_t,
                    self._prev_action,
                    self._hidden,
                    deterministic=self.deterministic,
                    threshold=self.action_threshold,
                    temperature=self.temperature,
                )
                self._hidden = next_hidden
                self._prev_action = action_t
                act_np = action_t[0].detach().cpu().numpy().astype(np.int8, copy=False)
                self._prev_action_np = act_np
                return act_np

            logits, _, next_hidden = self.policy.act_step(obs_t, self._prev_action, self._hidden)
            action_head = getattr(self.policy, "action_head", "factorized")
            avg_logits = 0.5 * (logits[0:1] + _unmirror_logits(logits[1:2], action_head=action_head))
            action_t, _ = self.policy.sample_from_logits(
                avg_logits,
                deterministic=self.deterministic,
                threshold=self.action_threshold,
                temperature=self.temperature,
            )
            self._hidden = next_hidden
            act_np = action_t[0].detach().cpu().numpy().astype(np.int8, copy=False)
            act_mir_t = action_t[:, _ACTION_MIRROR_IDX]
            self._prev_action = torch.cat([action_t, act_mir_t], dim=0)
            self._prev_action_np = act_np
            return act_np
