"""Staged curriculum and tier/family sampling weights for Role 2 training (`TRACK1_IMPROVEMENTS.md §2.2`)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from training.interfaces import EpisodeMetrics, MapSpec

_DEFAULT_PHASE_TIER_WEIGHTS: tuple[dict[int, float], ...] = (
    {1: 0.45, 2: 0.35, 3: 0.15, 4: 0.05},  # Phase 0 (warmup): dense short tasks + replay
    {1: 0.20, 2: 0.30, 3: 0.40, 4: 0.10},  # Phase 1 (progression): key-return focus + T1-2 replay
    {1: 0.15, 2: 0.20, 3: 0.35, 4: 0.30},  # Phase 2 (gauntlet): full difficulty mixture
)


@dataclass(frozen=True)
class CurriculumConfig:
    auto_curriculum: bool = True
    tier_weights: dict[int, float] | None = None
    family_weights: dict[str, float] | None = None
    window_size: int = 32
    min_episodes_to_promote: int = 16
    tier12_exit_promote_threshold: float = 0.40
    tier3_progress_promote_threshold: float = 0.30
    tier3_require_exit: bool = False
    adaptive_weights: bool = False
    adaptive_window: int = 16
    adaptive_floor: float = 0.25


class CurriculumScheduler:
    """Computes normalized per-map sampling weights and advances curriculum phases from episode outcomes."""

    def __init__(self, config: CurriculumConfig | None = None):
        self.config = config or CurriculumConfig()
        self.phase = 0
        self._t12_exits: deque[float] = deque(maxlen=self.config.window_size)
        self._t3_progress: deque[float] = deque(maxlen=self.config.window_size)
        self._map_exits: dict[str, deque[float]] = {}

    def current_tier_weights(self) -> dict[int, float]:
        if self.config.tier_weights is not None:
            return dict(self.config.tier_weights)
        idx = min(self.phase, len(_DEFAULT_PHASE_TIER_WEIGHTS) - 1)
        return dict(_DEFAULT_PHASE_TIER_WEIGHTS[idx])

    def compute_map_weights(self, map_specs: list[MapSpec]) -> list[float]:
        if not map_specs:
            raise ValueError("map_specs must not be empty")
        tier_w = self.current_tier_weights()
        fam_w = self.config.family_weights or {}

        # Normalize (fw * aw) within each tier so each tier preserves its exact target budget.
        tier_indices: dict[int, list[int]] = {}
        inner_weights: list[float] = []
        for idx, spec in enumerate(map_specs):
            tier_indices.setdefault(spec.tier, []).append(idx)
            fw = float(fam_w.get(spec.family, 1.0))
            if not self.config.adaptive_weights:
                aw = 1.0
            elif spec.wad in self._map_exits and len(self._map_exits[spec.wad]) > 0:
                hist = self._map_exits[spec.wad]
                exit_rate = sum(hist) / len(hist)
                aw = max(self.config.adaptive_floor, 1.0 - exit_rate + self.config.adaptive_floor)
            else:
                # Optimistic exploration weight for newly refreshed / unseen maps.
                aw = 1.0 + self.config.adaptive_floor
            inner_weights.append(max(1e-6, fw * aw))

        raw = [0.0] * len(map_specs)
        for tier, idxs in tier_indices.items():
            tb = max(0.0, float(tier_w.get(tier, 1.0)))
            inner_sum = sum(inner_weights[i] for i in idxs)
            for i in idxs:
                raw[i] = max(1e-8, tb * (inner_weights[i] / inner_sum))

        total = float(sum(raw))
        return [w / total for w in raw]

    def observe_episodes(self, episodes: list[EpisodeMetrics]) -> bool:
        """Record completed episodes and advance curriculum phase if mastery criteria are met."""
        for ep in episodes:
            if self.config.adaptive_weights:
                hist = self._map_exits.get(ep.wad)
                if hist is None:
                    hist = deque(maxlen=self.config.adaptive_window)
                    self._map_exits[ep.wad] = hist
                hist.append(1.0 if ep.outcome == "exit" else 0.0)

        if not self.config.auto_curriculum or self.config.tier_weights is not None:
            return False

        for ep in episodes:
            if ep.tier in (1, 2):
                self._t12_exits.append(1.0 if ep.outcome == "exit" else 0.0)
            elif ep.tier == 3:
                progressed = (
                    (ep.outcome == "exit")
                    if self.config.tier3_require_exit
                    else ((ep.outcome == "exit") or (ep.locked_doors_unlocked > 0))
                )
                self._t3_progress.append(1.0 if progressed else 0.0)

        min_eps = self.config.min_episodes_to_promote
        if self.phase == 0 and len(self._t12_exits) >= min_eps:
            rate = sum(self._t12_exits) / len(self._t12_exits)
            if rate >= self.config.tier12_exit_promote_threshold:
                self.phase = 1
                return True
        elif self.phase == 1 and len(self._t3_progress) >= min_eps:
            rate = sum(self._t3_progress) / len(self._t3_progress)
            if rate >= self.config.tier3_progress_promote_threshold:
                self.phase = 2
                return True
        return False
