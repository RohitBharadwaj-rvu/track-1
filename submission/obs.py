"""Shared observation preprocessing for both training (Role 2) and submission (Role 1).

Contract: docs/CONTRACT.md §2 and env/doom_env/spaces.py.
Pure NumPy by default so it has zero heavy import cost; `obs_to_torch` converts
preprocessed dicts to PyTorch tensors on demand.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from doom_env.spaces import (
        BUTTONS,
        ENTITY_FEATURES,
        ENTITY_TYPES,
        MAX_ENTITIES,
        NUM_RAYS,
        RAY_HIT_TYPES,
        VITALS_FIELDS,
    )

    NUM_RAY_HIT_TYPES = len(RAY_HIT_TYPES)  # 6
    NUM_ENTITY_FEATURES = len(ENTITY_FEATURES)  # 5
    NUM_ENTITY_TYPES = len(ENTITY_TYPES)  # 12
    NUM_VITALS = len(VITALS_FIELDS)  # 17
    NUM_ACTIONS = len(BUTTONS)  # 9
except ImportError:  # pragma: no cover - fallback when testing policy without vizdoom
    NUM_RAYS = 64
    NUM_RAY_HIT_TYPES = 6
    MAX_ENTITIES = 32
    NUM_ENTITY_FEATURES = 5
    NUM_ENTITY_TYPES = 12
    NUM_VITALS = 17
    NUM_ACTIONS = 9

# Per-field scales matching doom_env.spaces.VITALS_FIELDS:
#   health, armor, selected_weapon, selected_weapon_ammo,
#   ammo_bullets, ammo_shells, ammo_rockets, ammo_cells,
#   6 key flags (already 0/1),
#   vel_forward, vel_right, vel_up
_VITALS_SCALES = np.array(
    [
        100.0,  # health (0-200)
        100.0,  # armor (0-200)
        7.0,    # selected_weapon (0-7)
        100.0,  # selected_weapon_ammo
        200.0,  # ammo_bullets
        50.0,   # ammo_shells
        50.0,   # ammo_rockets
        300.0,  # ammo_cells
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # key flags (0/1)
        20.0,   # vel_forward (map units/tic)
        20.0,   # vel_right
        20.0,   # vel_up
    ],
    dtype=np.float32,
)

# Scales for visible_entities["features"]:
#   [distance / 1024, sin(rel_angle), cos(rel_angle), rel_vel_forward, rel_vel_right]
_ENTITY_FEAT_SCALES = np.array([1.0, 1.0, 1.0, 20.0, 20.0], dtype=np.float32)


def preprocess_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert one raw `doom_env` observation dict into flat, normalized NumPy arrays.

    Ignores `rgb` and `depth`. Output keys and shapes (unbatched):
      - `ray_dist`:    (64,) float32 in [0, 1]
      - `ray_type`:    (64,) int64   in [0, 5]
      - `ent_feat`:    (32, 5) float32 (zeroed where mask == 0)
      - `ent_type`:    (32,) int64   in [0, 11]
      - `ent_mask`:    (32,) float32 in {0.0, 1.0}
      - `vitals`:      (17,) float32 scaled & clipped to [-10, 10]
    """
    rays = obs["rays"]
    ray_dist = np.clip(np.asarray(rays["distance"], dtype=np.float32), 0.0, 1.0)
    ray_type = np.clip(np.asarray(rays["hit_type"], dtype=np.int64), 0, NUM_RAY_HIT_TYPES - 1)

    ents = obs["visible_entities"]
    ent_mask = np.asarray(ents["mask"], dtype=np.float32)
    ent_type = np.clip(np.asarray(ents["entity_type"], dtype=np.int64), 0, NUM_ENTITY_TYPES - 1)
    ent_feat = np.asarray(ents["features"], dtype=np.float32) / _ENTITY_FEAT_SCALES
    ent_feat = np.nan_to_num(ent_feat, nan=0.0, posinf=10.0, neginf=-10.0)
    ent_feat = np.clip(ent_feat, -10.0, 10.0) * ent_mask[:, None]

    vitals = np.asarray(obs["vitals"], dtype=np.float32) / _VITALS_SCALES
    vitals = np.nan_to_num(vitals, nan=0.0, posinf=10.0, neginf=-10.0)
    vitals = np.clip(vitals, -10.0, 10.0)

    return {
        "ray_dist": ray_dist,
        "ray_type": ray_type,
        "ent_feat": ent_feat,
        "ent_type": ent_type,
        "ent_mask": ent_mask,
        "vitals": vitals,
    }


def batch_preprocessed_obs(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack a list of preprocessed observations along axis 0 -> batch shape (B, ...)."""
    if not items:
        raise ValueError("Cannot batch an empty list of observations")
    keys = items[0].keys()
    return {k: np.stack([item[k] for item in items], axis=0) for k in keys}


def obs_to_torch(obs_batch: dict[str, np.ndarray], device: str | Any) -> dict[str, Any]:
    """Convert a NumPy observation dict (unbatched, (B, ...), or (T, B, ...)) to PyTorch tensors."""
    import torch

    return {k: torch.as_tensor(v, device=device) for k, v in obs_batch.items()}
