"""Egocentric odometric spatial memory built strictly from agent observations.

Contract & rule compliance (`docs/RULES.md` §1 & `docs/CONTRACT.md` §1-2):
  - Uses ONLY `raw_obs` (`rays`, `visible_entities`, `vitals`) and `prev_action`.
  - ZERO access to `env.privileged_state()`, global map coordinates, or pathfinding.
  - Pure NumPy (zero top-level PyTorch import) so `SubprocVecDoomEnv` workers stay lean.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

EGO_MAP_CHANNELS = 8
EGO_MAP_SIZE = 25
EGO_MAP_CELL_SIZE = 64.0  # covers 25 * 64 = 1600x1600 map units centered on player

_BACKING_SIZE = 128
_BACKING_CELL = 48.0  # 128 * 48 = 6144x6144 map units (±3072 around spawn)
_BACKING_ORIGIN = _BACKING_SIZE // 2

_NUM_RAYS = 64
_RAY_RANGE = 1024.0
_RAY_WALL = 1
_RAY_DOOR = 2
_RAY_LOCKED = 3
_RAY_ENEMY = 4

_RAY_ANGLES = np.arange(_NUM_RAYS, dtype=np.float64) * (2.0 * math.pi / _NUM_RAYS)
_RAY_COS = np.cos(_RAY_ANGLES)
_RAY_SIN = np.sin(_RAY_ANGLES)
_MARCH_FRACS = np.linspace(0.0, 1.0, 25, dtype=np.float64)[:-1]  # 24 samples (~42u max step)

# Channel indices inside the 8-channel egocentric map
CH_VISITED = 0
CH_POST_KEY_VISITED = 1
CH_WALL = 2
CH_FREE = 3
CH_DOOR = 4
CH_LOCKED_DOOR = 5
CH_ITEMS_KEYS = 6
CH_ENEMIES = 7

# Precomputed ego-centric crop offsets (row 0 = ahead, col 0 = left, center (12,12) = player)
_HALF = EGO_MAP_SIZE // 2
_R_GRID, _C_GRID = np.meshgrid(
    np.arange(EGO_MAP_SIZE, dtype=np.float64),
    np.arange(EGO_MAP_SIZE, dtype=np.float64),
    indexing="ij",
)
_LOCAL_FWD = (_HALF - _R_GRID) * EGO_MAP_CELL_SIZE
_LOCAL_LEFT = (_HALF - _C_GRID) * EGO_MAP_CELL_SIZE

# Doom turn-speed constants (360 / 65536 * 320 and * 640 degrees/tic)
_SLOW_TURN_DEG = 1.7578125
_FAST_TURN_DEG = 3.515625

# Intensity for off-crop landmarks projected onto the 25x25 crop perimeter (vs 1.0 in-crop)
_OFFCROP_BEACON_VAL = 0.65


class EgocentricSpatialMap:
    """Maintains an odometric 2D spatial memory and renders ego-aligned (8, 25, 25) uint8 crops."""

    def __init__(self) -> None:
        self._grid = np.zeros((EGO_MAP_CHANNELS, _BACKING_SIZE, _BACKING_SIZE), dtype=np.float32)
        self.x_odom = 0.0
        self.y_odom = 0.0
        self.yaw_odom_deg = 0.0
        self._prev_vel = (0.0, 0.0)
        self._prev_key_count = 0
        self._step_count = 0
        self._turnheld = 0
        self._prev_turn_dir = 0
        self._prev_pts: np.ndarray | None = None
        self._prev_valid: np.ndarray | None = None

    def reset(self) -> None:
        self._grid.fill(0.0)
        self.x_odom = 0.0
        self.y_odom = 0.0
        self.yaw_odom_deg = 0.0
        self._prev_vel = (0.0, 0.0)
        self._prev_key_count = 0
        self._step_count = 0
        self._turnheld = 0
        self._prev_turn_dir = 0
        self._prev_pts = None
        self._prev_valid = None

    def _to_grid_idx(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[Any, Any]:
        gx = np.rint(np.asarray(x) / _BACKING_CELL + _BACKING_ORIGIN).astype(np.intp)
        gy = np.rint(np.asarray(y) / _BACKING_CELL + _BACKING_ORIGIN).astype(np.intp)
        return np.clip(gx, 0, _BACKING_SIZE - 1), np.clip(gy, 0, _BACKING_SIZE - 1)

    def _update_yaw_odom(self, prev_action: np.ndarray) -> float:
        """Estimate heading delta (degrees, CCW positive) from Doom's turnheld physics."""
        td = int(prev_action[4] > 0.5) - int(prev_action[5] > 0.5)
        d_deg = 0.0
        if self._step_count == 1:
            self._turnheld = 4 if td != 0 else 0
        elif self._step_count == 2:
            if td != 0:
                if self._prev_turn_dir != 0:
                    d_deg = _FAST_TURN_DEG * td
                    self._turnheld += 4
                else:
                    d_deg = _SLOW_TURN_DEG * td
                    self._turnheld = 4
            else:
                self._turnheld = 0
        else:
            if td == 0:
                self._turnheld = 0
            else:
                for _ in range(4):
                    self._turnheld += 1
                    d_deg += (_SLOW_TURN_DEG if self._turnheld < 6 else _FAST_TURN_DEG) * td
        self._prev_turn_dir = td
        self.yaw_odom_deg = (self.yaw_odom_deg + d_deg) % 360.0
        return d_deg

    def update(self, raw_obs: dict[str, Any], prev_action: np.ndarray) -> np.ndarray:
        """Advance odometry, stamp current sensor observations, and return (8, 25, 25) uint8 map."""
        rays = raw_obs["rays"]
        ray_dist_norm = np.asarray(rays["distance"], dtype=np.float64)
        ray_dist_m = np.clip(ray_dist_norm, 0.0, 1.0) * _RAY_RANGE
        ray_type = np.asarray(rays["hit_type"], dtype=np.int64)
        vitals = np.asarray(raw_obs["vitals"], dtype=np.float64)
        vf, vr = float(vitals[14]), float(vitals[15])

        valid_wall = (ray_type == _RAY_WALL) & (ray_dist_m < 980.0)
        pts = np.stack([ray_dist_m * _RAY_COS, ray_dist_m * _RAY_SIN], axis=1).astype(np.float32)  # (64, 2)

        if self._step_count == 0:
            self._prev_vel = (vf, vr)
            self._prev_pts = pts
            self._prev_valid = valid_wall
            self._step_count = 1
        else:
            d_deg = self._update_yaw_odom(prev_action)
            df = 1.25 * self._prev_vel[0] + 2.50 * vf
            dl = -(1.25 * self._prev_vel[1] + 2.50 * vr)
            self._prev_vel = (vf, vr)

            # Refine ego displacement against previous wall rays (float32, <=32 pts for fast 1-CPU step)
            if (
                self._prev_pts is not None
                and self._prev_valid is not None
                and int(valid_wall.sum()) >= 12
                and int(self._prev_valid.sum()) >= 12
            ):
                d_rad = math.radians(d_deg)
                c_d = np.float32(math.cos(d_rad))
                s_d = np.float32(math.sin(d_rad))
                p_prev = self._prev_pts[self._prev_valid]
                p_cur = pts[valid_wall]
                p_rot = np.empty_like(p_prev)
                p_rot[:, 0] = p_prev[:, 0] * c_d + p_prev[:, 1] * s_d
                p_rot[:, 1] = -p_prev[:, 0] * s_d + p_prev[:, 1] * c_d
                row_idx = np.arange(len(p_cur))
                thresh2 = np.float32(48.0 * 48.0)
                for _ in range(2):
                    shifted = p_cur + np.array([df, dl], dtype=np.float32)
                    diff = p_rot[None, :, :] - shifted[:, None, :]
                    dist2 = (diff * diff).sum(axis=2)
                    nn_idx = np.argmin(dist2, axis=1)
                    d_nn = dist2[row_idx, nn_idx]
                    inliers = d_nn < thresh2
                    if int(inliers.sum()) >= 8:
                        delta = (p_rot[nn_idx[inliers]] - shifted[inliers]).mean(axis=0)
                        df = float(np.clip(df + float(delta[0]), -48.0, 48.0))
                        dl = float(np.clip(dl + float(delta[1]), -48.0, 48.0))

            self._prev_pts = pts
            self._prev_valid = valid_wall
            yaw_rad = math.radians(self.yaw_odom_deg)
            c_y, s_y = math.cos(yaw_rad), math.sin(yaw_rad)
            self.x_odom += df * c_y - dl * s_y
            self.y_odom += df * s_y + dl * c_y
            self._step_count += 1

        yaw_rad = math.radians(self.yaw_odom_deg)
        c_y, s_y = math.cos(yaw_rad), math.sin(yaw_rad)

        # World-frame unit vectors of all 64 rays
        world_ray_angles = yaw_rad + _RAY_ANGLES
        w_cos = np.cos(world_ray_angles)
        w_sin = np.sin(world_ray_angles)

        # 1. Visited & post-key visited channels (reset ch1 on new key pickup)
        px_i_arr, py_i_arr = self._to_grid_idx(self.x_odom, self.y_odom)
        px_i, py_i = int(px_i_arr), int(py_i_arr)
        key_count = int(np.sum(vitals[8:14] > 0.5))
        if key_count > self._prev_key_count:
            self._grid[CH_POST_KEY_VISITED].fill(0.0)
            self._prev_key_count = key_count
            # Key item can be picked up from ~36u away; clear key markers in 3x3 around player
            px_lo, px_hi = max(0, px_i - 1), min(_BACKING_SIZE, px_i + 2)
            py_lo, py_hi = max(0, py_i - 1), min(_BACKING_SIZE, py_i + 2)
            self._grid[CH_ITEMS_KEYS, px_lo:px_hi, py_lo:py_hi] = 0.0
        self._grid[CH_VISITED, px_i, py_i] = 1.0
        if key_count > 0:
            self._grid[CH_POST_KEY_VISITED, px_i, py_i] = 1.0
        # Clear pickup markers at the cell the player has already walked onto
        self._grid[CH_ITEMS_KEYS, px_i, py_i] = 0.0

        # 2. Ray-march seen free space (ch3) and clear opened/unlocked doors along pass-through rays
        march_dist = (ray_dist_m[:, None] * 0.92) * _MARCH_FRACS[None, :]
        mx = self.x_odom + march_dist * w_cos[:, None]
        my = self.y_odom + march_dist * w_sin[:, None]
        mgx, mgy = self._to_grid_idx(mx, my)
        self._grid[CH_FREE, mgx, mgy] = 1.0
        # Clear door flags well short of ray hit; never erase CH_LOCKED_DOOR while holding 0 keys
        clear_mask = march_dist < np.maximum(0.0, ray_dist_m[:, None] - 56.0)
        if np.any(clear_mask):
            cgx, cgy = mgx[clear_mask], mgy[clear_mask]
            self._grid[CH_DOOR, cgx, cgy] = 0.0
            if key_count > 0:
                self._grid[CH_LOCKED_DOOR, cgx, cgy] = 0.0

        # 3. Ray endpoints: walls (1), unlocked doors (2), locked doors (3), enemy rays (4)
        hit_x = self.x_odom + ray_dist_m * w_cos
        hit_y = self.y_odom + ray_dist_m * w_sin
        in_range = ray_dist_m < 1004.0

        wall_m = in_range & (ray_type == _RAY_WALL)
        if np.any(wall_m):
            wx_i, wy_i = self._to_grid_idx(hit_x[wall_m], hit_y[wall_m])
            self._grid[CH_WALL, wx_i, wy_i] = 1.0

        door_m = in_range & (ray_type == _RAY_DOOR)
        if np.any(door_m):
            dx_i, dy_i = self._to_grid_idx(hit_x[door_m], hit_y[door_m])
            self._grid[CH_DOOR, dx_i, dy_i] = 1.0

        lock_m = in_range & (ray_type == _RAY_LOCKED)
        if np.any(lock_m):
            lx_i, ly_i = self._to_grid_idx(hit_x[lock_m], hit_y[lock_m])
            self._grid[CH_LOCKED_DOOR, lx_i, ly_i] = 1.0

        # 4. Decay enemy channel and stamp visible entities (items/keys + enemies)
        self._grid[CH_ENEMIES] *= 0.90
        enemy_ray_m = in_range & (ray_type == _RAY_ENEMY)
        if np.any(enemy_ray_m):
            ex_i, ey_i = self._to_grid_idx(hit_x[enemy_ray_m], hit_y[enemy_ray_m])
            self._grid[CH_ENEMIES, ex_i, ey_i] = np.maximum(self._grid[CH_ENEMIES, ex_i, ey_i], 0.8)

        ve = raw_obs["visible_entities"]
        vmask = np.asarray(ve["mask"], dtype=np.float32) > 0.5
        if np.any(vmask):
            feats = np.asarray(ve["features"], dtype=np.float64)[vmask]
            etypes = np.asarray(ve["entity_type"], dtype=np.int64)[vmask]
            edist = np.clip(feats[:, 0], 0.0, 4.0) * _RAY_RANGE
            sin_a = np.clip(feats[:, 1], -1.0, 1.0)
            cos_a = np.clip(feats[:, 2], -1.0, 1.0)
            efwd = edist * cos_a
            eleft = edist * sin_a
            ex = self.x_odom + efwd * c_y - eleft * s_y
            ey = self.y_odom + efwd * s_y + eleft * c_y
            egx, egy = self._to_grid_idx(ex, ey)

            # Enemies: 1=melee, 2=hitscan, 3=proj_enemy
            is_enemy = (etypes >= 1) & (etypes <= 3)
            if np.any(is_enemy):
                self._grid[CH_ENEMIES, egx[is_enemy], egy[is_enemy]] = 1.0

            # Pickups: 9=key (1.0), 8=weapon (0.8), 5/6/7=health/armor/ammo (0.45)
            is_key = etypes == 9
            if np.any(is_key):
                self._grid[CH_ITEMS_KEYS, egx[is_key], egy[is_key]] = 1.0
            is_weapon = etypes == 8
            if np.any(is_weapon):
                self._grid[CH_ITEMS_KEYS, egx[is_weapon], egy[is_weapon]] = np.maximum(
                    self._grid[CH_ITEMS_KEYS, egx[is_weapon], egy[is_weapon]], 0.8
                )
            is_supply = (etypes >= 5) & (etypes <= 7)
            if np.any(is_supply):
                self._grid[CH_ITEMS_KEYS, egx[is_supply], egy[is_supply]] = np.maximum(
                    self._grid[CH_ITEMS_KEYS, egx[is_supply], egy[is_supply]], 0.45
                )

        # 5. Extract zero-blur ego-aligned crop (8, 25, 25)
        crop_wx = self.x_odom + _LOCAL_FWD * c_y - _LOCAL_LEFT * s_y
        crop_wy = self.y_odom + _LOCAL_FWD * s_y + _LOCAL_LEFT * c_y
        cgx, cgy = self._to_grid_idx(crop_wx, crop_wy)
        crop = self._grid[:, cgx, cgy]

        # 5b. Project off-crop stamped locked doors & keys onto the 25x25 perimeter ring so
        # long key-return routes (>800u) retain an egocentric bearing cue to already-seen locks/keys.
        for ch, thresh in ((CH_LOCKED_DOOR, 0.5), (CH_ITEMS_KEYS, 0.9)):
            gx_arr, gy_arr = np.nonzero(self._grid[ch] >= thresh)
            if gx_arr.size == 0:
                continue
            wx = (gx_arr.astype(np.float64) - _BACKING_ORIGIN) * _BACKING_CELL
            wy = (gy_arr.astype(np.float64) - _BACKING_ORIGIN) * _BACKING_CELL
            dx = wx - self.x_odom
            dy = wy - self.y_odom
            lfwd = dx * c_y + dy * s_y
            lleft = -dx * s_y + dy * c_y
            r_f = _HALF - lfwd / EGO_MAP_CELL_SIZE
            c_f = _HALF - lleft / EGO_MAP_CELL_SIZE
            outside = (r_f < 0.0) | (r_f > EGO_MAP_SIZE - 1) | (c_f < 0.0) | (c_f > EGO_MAP_SIZE - 1)
            if np.any(outside):
                dr = r_f[outside] - _HALF
                dc = c_f[outside] - _HALF
                scale = _HALF / np.maximum(np.maximum(np.abs(dr), np.abs(dc)), 1e-6)
                br = np.clip(np.rint(_HALF + dr * scale).astype(np.intp), 0, EGO_MAP_SIZE - 1)
                bc = np.clip(np.rint(_HALF + dc * scale).astype(np.intp), 0, EGO_MAP_SIZE - 1)
                crop[ch, br, bc] = np.maximum(crop[ch, br, bc], _OFFCROP_BEACON_VAL)

        return np.clip(crop * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
