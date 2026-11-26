"""Exact left-right ego-frame mirror symmetry transforms (training-only data augmentation).

All transformations are exact reflections across the player's longitudinal axis based on
`env/doom_env/sensors.py` and `env/doom_env/spaces.py`:
  - `rays`: 64 CCW rays starting straight ahead -> index permutation `(-i) % 64`
    (`0` ahead and `32` behind stay fixed; `16` left <-> `48` right).
  - `ent_feat` / `visible_entities["features"]`:
    `[dist, sin(rel_angle), cos(rel_angle), rel_vel_fwd, rel_vel_right]`
    -> negate indices `1` (`sin(rel_angle)`) and `4` (`rel_vel_right`).
  - `vitals`: index `15` is `vel_right` -> negate index `15`.
  - `actions` / `prev_actions`: `[0:FWD, 1:BACK, 2:MOVE_LEFT, 3:MOVE_RIGHT, 4:TURN_LEFT, 5:TURN_RIGHT, ...]`
    -> swap `(2 <-> 3)` and `(4 <-> 5)`.
  - `aux_targets`: `[0:seen_locked, 1:door_sin, 2:door_cos, 3:door_dist_norm]`
    -> negate index `1` (`door_sin`).
  - `priv_features`: `priv_features[..., 12:16]` is `aux_targets` -> negate index `13`.

Pure NumPy with zero PyTorch dependency so worker subprocesses remain fast and lightweight.
Every transform is a strict involution: `f(f(x)) == x`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from submission.obs import NUM_RAYS

RAY_MIRROR_IDX: np.ndarray = (-np.arange(NUM_RAYS, dtype=np.int64)) % NUM_RAYS
ACTION_MIRROR_IDX: np.ndarray = np.array([0, 1, 3, 2, 5, 4, 6, 7, 8], dtype=np.int64)


def mirror_raw_obs(raw_obs: dict[str, Any]) -> dict[str, Any]:
    """Return a left-right mirrored copy of a raw `doom_env` observation dict.

    Feeding mirrored raw observations (together with policy-frame mirrored actions) into
    `EgocentricSpatialMap.update` produces an exact mirror-twin spatial map crop as well.
    """
    rays = raw_obs["rays"]
    mirrored_rays = {
        "distance": np.ascontiguousarray(np.asarray(rays["distance"])[..., RAY_MIRROR_IDX]),
        "hit_type": np.ascontiguousarray(np.asarray(rays["hit_type"])[..., RAY_MIRROR_IDX]),
    }

    ve = raw_obs["visible_entities"]
    ve_feats = np.array(ve["features"], copy=True)
    ve_feats[..., 1] = -ve_feats[..., 1]
    ve_feats[..., 4] = -ve_feats[..., 4]
    mirrored_ve = {
        "features": ve_feats,
        "entity_type": np.array(ve["entity_type"], copy=True),
        "mask": np.array(ve["mask"], copy=True),
    }

    vitals = np.array(raw_obs["vitals"], copy=True)
    vitals[..., 15] = -vitals[..., 15]

    out = dict(raw_obs)
    out["rays"] = mirrored_rays
    out["visible_entities"] = mirrored_ve
    out["vitals"] = vitals
    return out


def mirror_preprocessed_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a left-right mirrored copy of a preprocessed telemetry observation dict."""
    ray_dist = np.ascontiguousarray(obs["ray_dist"][..., RAY_MIRROR_IDX])
    ray_type = np.ascontiguousarray(obs["ray_type"][..., RAY_MIRROR_IDX])

    ent_feat = np.array(obs["ent_feat"], copy=True)
    ent_feat[..., 1] = -ent_feat[..., 1]
    ent_feat[..., 4] = -ent_feat[..., 4]

    vitals = np.array(obs["vitals"], copy=True)
    vitals[..., 15] = -vitals[..., 15]

    out: dict[str, np.ndarray] = {
        "ray_dist": ray_dist,
        "ray_type": ray_type,
        "ent_feat": ent_feat,
        "ent_type": np.array(obs["ent_type"], copy=True),
        "ent_mask": np.array(obs["ent_mask"], copy=True),
        "vitals": vitals,
    }
    if "ego_map" in obs:
        # Flip horizontal axis (col 0 = left <-> col W-1 = right)
        out["ego_map"] = np.ascontiguousarray(obs["ego_map"][..., ::-1])
    return out


def mirror_action(action: np.ndarray) -> np.ndarray:
    """Swap `MOVE_LEFT <-> MOVE_RIGHT` and `TURN_LEFT <-> TURN_RIGHT` in an action array."""
    arr = np.asarray(action)
    if arr.shape[-1] != ACTION_MIRROR_IDX.shape[0]:
        raise ValueError(f"Expected action dim {ACTION_MIRROR_IDX.shape[0]}, got {arr.shape[-1]}")
    return np.ascontiguousarray(arr[..., ACTION_MIRROR_IDX])


def mirror_aux_targets(aux_targets: np.ndarray) -> np.ndarray:
    """Negate `door_sin` (index 1) in `aux_targets` `(..., 4)`."""
    out = np.array(aux_targets, copy=True)
    if out.shape[-1] < 2:
        raise ValueError(f"Expected aux_targets dim >= 2, got {out.shape[-1]}")
    out[..., 1] = -out[..., 1]
    return out


def mirror_priv_features(priv_features: np.ndarray) -> np.ndarray:
    """Negate `door_sin` (index 13 = aux[1]) in `priv_features` `(..., 16)`."""
    out = np.array(priv_features, copy=True)
    if out.shape[-1] < 14:
        raise ValueError(f"Expected priv_features dim >= 14, got {out.shape[-1]}")
    out[..., 13] = -out[..., 13]
    return out
