"""Vectorized ViZDoom environments, GAE, and contiguous recurrent rollout buffers (Role 2).

Designed to minimize Python/torch dependency overhead inside worker processes:
`SyncVecDoomEnv`, `SubprocVecDoomEnv`, `compute_gae`, and `RolloutBuffer` are pure NumPy +
`doom_env`. `collect_rollout` imports `torch` lazily so environment tests run in `.venv` too.
"""

from __future__ import annotations

from collections import OrderedDict
import importlib
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np

from submission.obs import (
    MAX_ENTITIES,
    NUM_ACTIONS,
    NUM_ENTITY_FEATURES,
    NUM_RAYS,
    NUM_VITALS,
    batch_preprocessed_obs,
    preprocess_obs,
)
from training.interfaces import EpisodeMetrics, MapSpec
from training.rewards import RewardConfig, RewardShaper

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_MAPS = [
    MapSpec(
        wad="maps/two_rooms.wad",
        map_name="MAP01",
        family="door_exit",
        tier=1,
        split="train",
        episode_ticks=2100,
        seeds=(0,),
    )
]


def parse_map_arg(raw: str, split: str = "train") -> MapSpec:
    """Parse `wad[:map_name[:family[:tier[:episode_ticks]]]]` into a `MapSpec`."""
    parts = raw.split(":")
    # Handle Windows drive letter prefix like C:\path\map.wad:...
    if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isalpha() and parts[1].startswith(("\\", "/")):
        parts = [f"{parts[0]}:{parts[1]}", *parts[2:]]
    if len(parts) > 5:
        raise ValueError(f"Invalid --map specification: {raw!r}")
    wad = parts[0]
    map_name = parts[1].upper() if len(parts) > 1 and parts[1] else "MAP01"
    family = parts[2] if len(parts) > 2 and parts[2] else "door_exit"
    tier = int(parts[3]) if len(parts) > 3 and parts[3] else 1
    episode_ticks = int(parts[4]) if len(parts) > 4 and parts[4] else 2100
    return MapSpec(
        wad=wad,
        map_name=map_name,
        family=family,
        tier=tier,
        split=split,
        episode_ticks=episode_ticks,
    )


def resolve_train_maps(map_args: list[str] | None = None, split: str = "train") -> list[MapSpec]:
    """Resolve training maps from CLI `--map`, Role 3's `training.maps`, or fallback."""
    specs, _ = resolve_train_maps_and_weights(map_args=map_args, split=split, curriculum_stage=None)
    return specs


def resolve_train_maps_and_weights(
    map_args: list[str] | None = None,
    split: str = "train",
    curriculum_stage: str | None = None,
) -> tuple[list[MapSpec], list[float] | None]:
    """Resolve training maps and optional normalized curriculum sampling weights."""
    if map_args:
        return [parse_map_arg(m, split=split) for m in map_args], None
    try:
        maps_mod = importlib.import_module("training.maps")
    except ModuleNotFoundError:
        return list(DEFAULT_TRAIN_MAPS), None

    if curriculum_stage is not None and hasattr(maps_mod, "get_curriculum_maps"):
        specs = list(maps_mod.get_curriculum_maps(curriculum_stage))
        weights = (
            list(maps_mod.get_curriculum_weights(curriculum_stage))
            if hasattr(maps_mod, "get_curriculum_weights")
            else None
        )
        if specs:
            return specs, weights

    if hasattr(maps_mod, "get_maps"):
        specs = list(maps_mod.get_maps(split=split))
        if specs:
            return specs, None
    for attr in ("TRAIN_MAPS", "ALL_MAPS", "DEV_MAPS"):
        if hasattr(maps_mod, attr):
            specs = [s for s in getattr(maps_mod, attr) if split == "all" or s.split == split]
            if specs:
                return specs, None
    return list(DEFAULT_TRAIN_MAPS), None


def _normalize_weights(weights: list[float] | None, n: int) -> np.ndarray | None:
    if weights is None:
        return None
    if len(weights) != n:
        raise ValueError(f"map_weights length ({len(weights)}) must match map_specs length ({n})")
    arr = np.asarray(weights, dtype=np.float64)
    if np.any(arr < 0.0) or float(arr.sum()) <= 0.0:
        raise ValueError("map_weights must be non-negative with positive sum")
    return arr / float(arr.sum())


def _resolve_wad_path(wad: str) -> str:
    p = Path(wad)
    if p.is_absolute() or p.exists():
        return str(p)
    candidate = ROOT / wad
    return str(candidate if candidate.exists() else p)


class _DoomWorkerSlot:
    """Single environment stream caching `DoomEnv` per (wad, map_name, ticks) with RewardShaper."""

    def __init__(
        self,
        slot_id: int,
        map_specs: list[MapSpec],
        reward_config: RewardConfig | None = None,
        base_seed: int = 0,
        render_buffers: bool = False,
        max_cached_envs: int = 32,
        map_weights: list[float] | None = None,
    ):
        if not map_specs:
            raise ValueError("map_specs must not be empty")
        self.slot_id = slot_id
        self.map_specs = list(map_specs)
        self._map_probs = _normalize_weights(map_weights, len(self.map_specs))
        self.shaper = RewardShaper(reward_config)
        self.render_buffers = render_buffers
        self.max_cached_envs = max(1, max_cached_envs)
        self._rng = np.random.default_rng(base_seed + slot_id * 10007)
        self._episode_counter = 0
        self._envs: OrderedDict[tuple[str, str, int], Any] = OrderedDict()

        self._active_env: Any = None
        self._active_spec: MapSpec = self.map_specs[0]
        self._current_seed: int = 0
        self._prev_action: np.ndarray = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._steps = 0
        self._total_reward = 0.0
        self._keys_picked_up = 0
        self._doors_opened = 0
        self._locked_doors_unlocked = 0
        self._kills = 0
        self._damage_taken = 0.0

    def _get_or_create_env(self, spec: MapSpec):
        wad_path = _resolve_wad_path(spec.wad)
        key = (wad_path, spec.map_name.upper(), int(spec.episode_ticks))
        if key in self._envs:
            self._envs.move_to_end(key)
            return self._envs[key]
        while len(self._envs) >= self.max_cached_envs:
            _, evicted = self._envs.popitem(last=False)
            evicted.close()
        from doom_env import make_env

        env = make_env(
            wad_path,
            spec.map_name,
            episode_ticks=spec.episode_ticks,
            render_buffers=self.render_buffers,
        )
        self._envs[key] = env
        return env

    def reset(self) -> dict[str, np.ndarray]:
        if self._map_probs is not None:
            idx = int(self._rng.choice(len(self.map_specs), p=self._map_probs))
        else:
            idx = int(self._rng.integers(0, len(self.map_specs)))
        spec = self.map_specs[idx]
        env = self._get_or_create_env(spec)
        self._active_env = env
        self._active_spec = spec
        self._episode_counter += 1
        self._current_seed = int(self.slot_id * 1_000_000 + self._episode_counter)

        raw_obs, _ = env.reset(seed=self._current_seed)
        self.shaper.reset(env.privileged_state(), obs=raw_obs)
        self._prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._steps = 0
        self._total_reward = 0.0
        self._keys_picked_up = 0
        self._doors_opened = 0
        self._locked_doors_unlocked = 0
        self._kills = 0
        self._damage_taken = 0.0
        return preprocess_obs(raw_obs)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._active_env is None:
            raise RuntimeError("Call reset() before step()")
        raw_obs, _, terminated, truncated, _ = self._active_env.step(action)
        priv = self._active_env.privileged_state()
        reward, events, _ = self.shaper.step(
            priv,
            action=action,
            prev_action=self._prev_action,
            obs=raw_obs,
        )
        self._prev_action = np.asarray(action, dtype=np.int8).copy()

        self._steps += 1
        self._total_reward += reward
        self._keys_picked_up += events.keys_picked_up
        self._doors_opened += events.doors_opened
        self._locked_doors_unlocked += events.locked_doors_unlocked
        self._kills += events.kills
        self._damage_taken += events.damage_taken

        info: dict[str, Any] = {}
        if terminated or truncated:
            final_obs = preprocess_obs(raw_obs)
            if priv.get("exit_reached"):
                outcome = "exit"
            elif priv.get("dead"):
                outcome = "death"
            else:
                outcome = "timeout"
            info["final_obs"] = final_obs
            info["episode"] = EpisodeMetrics(
                wad=self._active_spec.wad,
                map_name=self._active_spec.map_name,
                family=self._active_spec.family,
                tier=self._active_spec.tier,
                seed=self._current_seed,
                steps=self._steps,
                ticks=int(priv.get("tic", 0)),
                outcome=outcome,
                keys_picked_up=self._keys_picked_up,
                doors_opened=self._doors_opened,
                locked_doors_unlocked=self._locked_doors_unlocked,
                kills=self._kills,
                damage_taken=float(self._damage_taken),
                final_health=float(priv.get("health", 0.0)),
                unique_cells_visited=self.shaper.unique_cells_visited,
                total_reward=float(self._total_reward),
            )
            next_obs = self.reset()
        else:
            next_obs = preprocess_obs(raw_obs)
        return next_obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        for env in self._envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._envs.clear()
        self._active_env = None


class SyncVecDoomEnv:
    """Synchronous vectorized Doom environment over preprocessed telemetry dicts."""

    def __init__(
        self,
        map_specs: list[MapSpec],
        reward_config: RewardConfig | None = None,
        base_seed: int = 0,
        render_buffers: bool = False,
        num_envs: int | None = None,
        max_cached_envs: int = 32,
        map_weights: list[float] | None = None,
    ):
        self.num_envs = num_envs if num_envs is not None else max(1, len(map_specs))
        self.slots = [
            _DoomWorkerSlot(
                slot_id=i,
                map_specs=map_specs,
                reward_config=reward_config,
                base_seed=base_seed,
                render_buffers=render_buffers,
                max_cached_envs=max_cached_envs,
                map_weights=map_weights,
            )
            for i in range(self.num_envs)
        ]

    def reset(self) -> dict[str, np.ndarray]:
        return batch_preprocessed_obs([slot.reset() for slot in self.slots])

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        act = np.asarray(actions, dtype=np.int8)
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        for i, slot in enumerate(self.slots):
            o, r, te, tr, info = slot.step(act[i])
            obs_list.append(o)
            rewards.append(r)
            terms.append(te)
            truncs.append(tr)
            infos.append(info)
        return (
            batch_preprocessed_obs(obs_list),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for slot in self.slots:
            slot.close()

    def __enter__(self) -> SyncVecDoomEnv:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _worker_loop(
    conn: Any,
    slot_id: int,
    map_specs: list[MapSpec],
    reward_config: RewardConfig | None,
    base_seed: int,
    render_buffers: bool,
    max_cached_envs: int,
    map_weights: list[float] | None = None,
) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    slot = _DoomWorkerSlot(
        slot_id=slot_id,
        map_specs=map_specs,
        reward_config=reward_config,
        base_seed=base_seed,
        render_buffers=render_buffers,
        max_cached_envs=max_cached_envs,
        map_weights=map_weights,
    )
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(slot.reset())
            elif cmd == "step":
                conn.send(slot.step(data))
            elif cmd == "close":
                slot.close()
                conn.send(None)
                break
            else:
                raise ValueError(f"Unknown worker command: {cmd!r}")
    except EOFError:
        pass
    finally:
        slot.close()
        conn.close()


class SubprocVecDoomEnv:
    """Multiprocess vectorized Doom environment using spawned worker processes."""

    def __init__(
        self,
        map_specs: list[MapSpec],
        reward_config: RewardConfig | None = None,
        base_seed: int = 0,
        render_buffers: bool = False,
        num_envs: int = 4,
        max_cached_envs: int = 32,
        map_weights: list[float] | None = None,
    ):
        self.num_envs = int(num_envs)
        ctx = mp.get_context("spawn")
        self._parent_conns = []
        self._procs = []
        self._closed = False
        for i in range(self.num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_loop,
                args=(
                    child_conn,
                    i,
                    map_specs,
                    reward_config,
                    base_seed,
                    render_buffers,
                    max_cached_envs,
                    map_weights,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._procs.append(proc)

    def reset(self) -> dict[str, np.ndarray]:
        for conn in self._parent_conns:
            conn.send(("reset", None))
        return batch_preprocessed_obs([conn.recv() for conn in self._parent_conns])

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        act = np.asarray(actions, dtype=np.int8)
        for i, conn in enumerate(self._parent_conns):
            conn.send(("step", act[i]))
        obs_list, rewards, terms, truncs, infos = [], [], [], [], []
        for conn in self._parent_conns:
            o, r, te, tr, info = conn.recv()
            obs_list.append(o)
            rewards.append(r)
            terms.append(te)
            truncs.append(tr)
            infos.append(info)
        return (
            batch_preprocessed_obs(obs_list),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            infos,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for conn in self._parent_conns:
            try:
                conn.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            conn.close()
        for proc in self._procs:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)

    def __enter__(self) -> SubprocVecDoomEnv:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    bootstrap_values: np.ndarray,
    last_values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation distinguishing death/exit from time truncation.

    All time-indexed inputs have shape `(T, N)`; `last_values` has shape `(N,)`.
    - When `terminated[t, i]` is True: `next_val = 0.0` (no bootstrap), GAE resets.
    - When `truncated[t, i]` is True (and not terminated): `next_val = bootstrap_values[t, i]`,
      and GAE still resets so rewards from the auto-reset next episode never bleed across.
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    done = term | trunc

    for t in range(T - 1, -1, -1):
        next_nonterminal_val = np.asarray(last_values, dtype=np.float32) if t == T - 1 else values[t + 1]
        next_val = np.where(term[t], 0.0, np.where(trunc[t], bootstrap_values[t], next_nonterminal_val))
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * np.where(done[t], 0.0, gae)
        advantages[t] = gae

    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


class RolloutBuffer:
    """Stores contiguous `(T, N, ...)` rollouts and yields shuffled `(chunk_len, B, ...)` batches."""

    def __init__(self, num_steps: int, num_envs: int, hidden_shape: tuple[int, ...]):
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.hidden_shape = tuple(hidden_shape)

        T, N = self.num_steps, self.num_envs
        self.obs: dict[str, np.ndarray] = {
            "ray_dist": np.zeros((T, N, NUM_RAYS), dtype=np.float32),
            "ray_type": np.zeros((T, N, NUM_RAYS), dtype=np.int64),
            "ent_feat": np.zeros((T, N, MAX_ENTITIES, NUM_ENTITY_FEATURES), dtype=np.float32),
            "ent_type": np.zeros((T, N, MAX_ENTITIES), dtype=np.int64),
            "ent_mask": np.zeros((T, N, MAX_ENTITIES), dtype=np.float32),
            "vitals": np.zeros((T, N, NUM_VITALS), dtype=np.float32),
        }
        self.prev_actions = np.zeros((T, N, NUM_ACTIONS), dtype=np.float32)
        self.actions = np.zeros((T, N, NUM_ACTIONS), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.terminated = np.zeros((T, N), dtype=bool)
        self.truncated = np.zeros((T, N), dtype=bool)
        self.bootstrap_values = np.zeros((T, N), dtype=np.float32)
        self.episode_starts = np.zeros((T, N), dtype=bool)
        # Hidden state BEFORE step t; shape (T, num_layers, N, hidden_size) or matching hidden_shape
        self.hiddens = np.zeros((T, *self.hidden_shape), dtype=np.float32)

        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)

    def add(
        self,
        step: int,
        obs_batch: dict[str, np.ndarray],
        prev_action: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        bootstrap_value: np.ndarray,
        episode_start: np.ndarray,
        hidden: np.ndarray,
    ) -> None:
        for k, arr in self.obs.items():
            arr[step] = obs_batch[k]
        self.prev_actions[step] = prev_action
        self.actions[step] = action
        self.log_probs[step] = log_prob
        self.rewards[step] = reward
        self.values[step] = value
        self.terminated[step] = terminated
        self.truncated[step] = truncated
        self.bootstrap_values[step] = bootstrap_value
        self.episode_starts[step] = episode_start
        self.hiddens[step] = hidden

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.advantages, self.returns = compute_gae(
            rewards=self.rewards,
            values=self.values,
            terminated=self.terminated,
            truncated=self.truncated,
            bootstrap_values=self.bootstrap_values,
            last_values=last_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def iter_chunk_minibatches(
        self,
        chunk_len: int,
        num_minibatches: int,
        rng: np.random.Generator | None = None,
    ):
        """Yield shuffled `(chunk_len, B, ...)` minibatches with exact chunk-start hidden states."""
        if self.num_steps % chunk_len != 0:
            raise ValueError(f"num_steps ({self.num_steps}) must be divisible by chunk_len ({chunk_len})")
        num_chunks_per_env = self.num_steps // chunk_len
        total_chunks = num_chunks_per_env * self.num_envs
        if total_chunks < 1:
            raise ValueError("No chunks available in buffer")
        mb_count = max(1, min(num_minibatches, total_chunks))

        # Enumerate all (chunk_idx, env_idx) pairs and shuffle across chunks, keeping time contiguous.
        pairs = [(c, e) for e in range(self.num_envs) for c in range(num_chunks_per_env)]
        order = np.arange(total_chunks)
        if rng is not None:
            rng.shuffle(order)

        for batch_indices in np.array_split(order, mb_count):
            if len(batch_indices) == 0:
                continue
            chosen = [pairs[int(idx)] for idx in batch_indices]

            def _slice_seq(arr: np.ndarray) -> np.ndarray:
                return np.stack(
                    [arr[c * chunk_len : (c + 1) * chunk_len, e] for c, e in chosen],
                    axis=1,
                )

            init_hidden = np.stack(
                [self.hiddens[c * chunk_len, :, e] for c, e in chosen],
                axis=1,
            )  # (num_layers, B, H)

            yield {
                "obs": {k: _slice_seq(v) for k, v in self.obs.items()},
                "prev_actions": _slice_seq(self.prev_actions),
                "actions": _slice_seq(self.actions),
                "old_log_probs": _slice_seq(self.log_probs),
                "old_values": _slice_seq(self.values),
                "advantages": _slice_seq(self.advantages),
                "returns": _slice_seq(self.returns),
                "episode_starts": _slice_seq(self.episode_starts),
                "init_hidden": init_hidden,
            }


def collect_rollout(
    vec_env: SyncVecDoomEnv | SubprocVecDoomEnv,
    policy: Any,
    buffer: RolloutBuffer,
    obs: dict[str, np.ndarray],
    prev_action: np.ndarray,
    hidden: Any,
    episode_starts: np.ndarray,
    device: str | Any,
    deterministic: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, Any, np.ndarray, list[EpisodeMetrics]]:
    """Collect `buffer.num_steps` transitions across `vec_env` into `buffer`.

    Handles truncation bootstrapping via `infos[i]["final_obs"]` and zeros hidden state
    plus `prev_action` on any episode boundary before returning the next rollout state.
    """
    import torch
    import torch.nn.functional as F

    from submission.obs import obs_to_torch

    completed_episodes: list[EpisodeMetrics] = []
    policy.eval()

    with torch.no_grad():
        for step in range(buffer.num_steps):
            hidden_before = hidden.detach().cpu().numpy()
            obs_t = obs_to_torch(obs, device)
            prev_t = torch.as_tensor(prev_action, dtype=torch.float32, device=device)

            logits, value, next_hidden = policy.act_step(obs_t, prev_t, hidden)
            if hasattr(policy, "sample_action"):
                act_t, log_prob_t = policy.sample_action(logits, deterministic=deterministic)
                log_prob = log_prob_t.cpu().numpy()
            else:
                if deterministic:
                    act_t = (logits > 0.0).float()
                else:
                    probs = torch.sigmoid(logits)
                    act_t = torch.bernoulli(probs)
                bce = F.binary_cross_entropy_with_logits(logits, act_t, reduction="none")
                log_prob = (-bce.sum(dim=-1)).cpu().numpy()
            action_np = act_t.cpu().numpy().astype(np.float32)
            value_np = value.cpu().numpy().astype(np.float32)

            next_obs, rewards, terminated, truncated, infos = vec_env.step(action_np.astype(np.int8))
            done = terminated | truncated

            bootstrap_values = np.zeros(vec_env.num_envs, dtype=np.float32)
            trunc_only_indices = [
                i for i in range(vec_env.num_envs) if truncated[i] and not terminated[i] and "final_obs" in infos[i]
            ]
            if trunc_only_indices:
                final_batch = batch_preprocessed_obs([infos[i]["final_obs"] for i in trunc_only_indices])
                final_obs_t = obs_to_torch(final_batch, device)
                final_prev_t = act_t[trunc_only_indices]
                final_h_t = next_hidden[:, trunc_only_indices, :]
                _, boot_vals, _ = policy.act_step(final_obs_t, final_prev_t, final_h_t)
                for idx_pos, env_idx in enumerate(trunc_only_indices):
                    bootstrap_values[env_idx] = float(boot_vals[idx_pos].item())

            for info in infos:
                if "episode" in info:
                    completed_episodes.append(info["episode"])

            buffer.add(
                step=step,
                obs_batch=obs,
                prev_action=prev_action,
                action=action_np,
                log_prob=log_prob,
                reward=rewards,
                value=value_np,
                terminated=terminated,
                truncated=truncated,
                bootstrap_value=bootstrap_values,
                episode_start=episode_starts,
                hidden=hidden_before,
            )

            # Reset recurrent state & prev_action at episode boundaries for the NEXT step.
            if np.any(done):
                keep_np = (~done).astype(np.float32)
                keep_t = torch.as_tensor(keep_np, device=device).view(1, -1, 1)
                next_hidden = next_hidden * keep_t
                next_prev_action = action_np * keep_np[:, None]
            else:
                next_prev_action = action_np

            obs = next_obs
            prev_action = next_prev_action
            hidden = next_hidden
            episode_starts = done.copy()

        # Bootstrap value for the final rollout state (step T)
        obs_last_t = obs_to_torch(obs, device)
        prev_last_t = torch.as_tensor(prev_action, dtype=torch.float32, device=device)
        _, last_values_t, _ = policy.act_step(obs_last_t, prev_last_t, hidden)
        last_values = last_values_t.cpu().numpy().astype(np.float32)

    return obs, prev_action, hidden, episode_starts, completed_episodes, last_values
