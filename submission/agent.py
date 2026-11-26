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


def _obs_to_batch_tensors(
    observation: dict,
    device: str,
    ego_map: EgocentricSpatialMap | None = None,
    prev_action_np: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    prep = preprocess_obs(observation)
    if ego_map is not None:
        pa = prev_action_np if prev_action_np is not None else np.zeros(NUM_ACTIONS, dtype=np.int8)
        prep["ego_map"] = ego_map.update(observation, pa)
    batched = {k: v[None, ...] for k, v in prep.items()}
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
        self._ego_map = EgocentricSpatialMap() if bool(getattr(self.config, "use_ego_map", False)) else None
        self._hidden = self.policy.init_hidden(1, self.device)
        self._prev_action = torch.zeros(1, NUM_ACTIONS, dtype=torch.float32, device=self.device)
        self._prev_action_np = np.zeros(NUM_ACTIONS, dtype=np.int8)

        # Warm up both active-entity and empty-room branches in __init__ (60s budget)
        # so the first act() step never pays CUDA/PyTorch initialization latency.
        with torch.inference_mode():
            warmup_obs = _make_dummy_obs(self.config)
            self.policy.act_step(
                _obs_to_batch_tensors(warmup_obs, self.device, self._ego_map, self._prev_action_np),
                self._prev_action,
                self._hidden,
            )
            warmup_obs["visible_entities"]["mask"][0] = 0
            self.policy.act_step(
                _obs_to_batch_tensors(warmup_obs, self.device, self._ego_map, self._prev_action_np),
                self._prev_action,
                self._hidden,
            )

        self.reset()

    def reset(self) -> None:
        """Called before every map. Clear recurrent state, spatial map, and previous-action buffer."""
        self._hidden = self.policy.init_hidden(1, self.device)
        self._prev_action = torch.zeros(1, NUM_ACTIONS, dtype=torch.float32, device=self.device)
        self._prev_action_np = np.zeros(NUM_ACTIONS, dtype=np.int8)
        if self._ego_map is not None:
            self._ego_map.reset()

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int8 array of shape (9,) with values in {0, 1}."""
        with torch.inference_mode():
            obs_t = _obs_to_batch_tensors(observation, self.device, self._ego_map, self._prev_action_np)
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
