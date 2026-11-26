"""Staged curriculum and tier/family sampling weights for Role 2 training (`TRACK1_IMPROVEMENTS.md §2.2`)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

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
    adaptive_mode: str = "legacy"  # "legacy" | "sfl_pvl"
    adaptive_window: int = 16
    adaptive_floor: float = 0.25
    sfl_weight: float = 0.55
    pvl_weight: float = 0.30
    failure_weight: float = 0.15
    staleness_coef: float = 0.10
    hedge_eta: float = 0.25
    champion_regress_boost: float = 0.60


class CurriculumScheduler:
    """Computes normalized per-map sampling weights and advances curriculum phases from episode outcomes."""

    def __init__(self, config: CurriculumConfig | None = None):
        self.config = config or CurriculumConfig()
        self.phase = 0
        self._t12_exits: deque[float] = deque(maxlen=self.config.window_size)
        self._t3_progress: deque[float] = deque(maxlen=self.config.window_size)
        self._map_exits: dict[str, deque[float]] = {}
        self._map_pvl: dict[str, deque[float]] = {}
        self._map_staleness: dict[str, int] = {}
        self._map_regressed: dict[str, float] = {}
        self._family_hedge: dict[str, float] = {}

    def observe_pvl(self, wad_pvl: dict[str, float]) -> None:
        """Record mean positive GAE / value-loss surprise per WAD for regret prioritization."""
        win = max(4, int(self.config.adaptive_window))
        for wad, pvl in wad_pvl.items():
            hist = self._map_pvl.get(wad)
            if hist is None:
                hist = deque(maxlen=win)
                self._map_pvl[wad] = hist
            hist.append(max(0.0, float(pvl)))

    def get_map_scores(self, map_specs: list[MapSpec]) -> dict[str, float]:
        """Compute UED frontier priority score per WAD (used by SFL+PVL weights and ACCEL archive)."""
        cfg = self.config
        if cfg.adaptive_mode != "sfl_pvl":
            scores: dict[str, float] = {}
            for spec in map_specs:
                hist = self._map_exits.get(spec.wad)
                if hist:
                    p = sum(hist) / float(len(hist))
                    scores[spec.wad] = max(cfg.adaptive_floor, 1.0 - p + cfg.adaptive_floor)
                else:
                    scores[spec.wad] = 1.0 + cfg.adaptive_floor
            return scores

        active_wads = {s.wad for s in map_specs}
        mean_pvls = {w: (sum(h) / float(len(h))) for w, h in self._map_pvl.items() if h and w in active_wads}
        max_pvl = max(mean_pvls.values(), default=0.0)
        scores = {}
        for spec in map_specs:
            hist = self._map_exits.get(spec.wad)
            if not hist:
                scores[spec.wad] = 1.0 + cfg.adaptive_floor
                continue
            p = sum(hist) / float(len(hist))
            sfl = 4.0 * p * (1.0 - p)
            pvl_norm = (mean_pvls.get(spec.wad, 0.0) / max_pvl) if max_pvl > 1e-8 else 0.0
            stale = self._map_staleness.get(spec.wad, 0)
            stale_bonus = min(0.5, cfg.staleness_coef * float(stale) / 10.0)
            regressed = self._map_regressed.get(spec.wad, 0.0)
            raw = (
                cfg.sfl_weight * sfl
                + cfg.pvl_weight * pvl_norm
                + cfg.failure_weight * (1.0 - p)
                + stale_bonus
                + cfg.champion_regress_boost * regressed
                + cfg.adaptive_floor
            )
            scores[spec.wad] = max(cfg.adaptive_floor, float(raw))
        return scores

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
        scores = self.get_map_scores(map_specs) if self.config.adaptive_weights else {}

        # Normalize (fw * aw) within each tier so each tier preserves its exact target budget.
        tier_indices: dict[int, list[int]] = {}
        inner_weights: list[float] = []
        for idx, spec in enumerate(map_specs):
            tier_indices.setdefault(spec.tier, []).append(idx)
            fw = float(fam_w.get(spec.family, 1.0))
            if self.config.adaptive_mode == "sfl_pvl" and spec.family in self._family_hedge:
                fw *= float(self._family_hedge[spec.family])
            if not self.config.adaptive_weights:
                aw = 1.0
            else:
                aw = float(scores.get(spec.wad, 1.0 + self.config.adaptive_floor))
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
        seen_wads: set[str] = set()
        fam_learn_sigs: dict[str, list[float]] = {}
        for ep in episodes:
            seen_wads.add(ep.wad)
            if self.config.adaptive_weights:
                hist = self._map_exits.get(ep.wad)
                if hist is None:
                    hist = deque(maxlen=self.config.adaptive_window)
                    self._map_exits[ep.wad] = hist
                hist.append(1.0 if ep.outcome == "exit" else 0.0)
                self._map_staleness[ep.wad] = 0
                extra = ep.extra or {}
                if extra.get("regressed_vs_champion"):
                    self._map_regressed[ep.wad] = 1.0
                elif ep.outcome == "exit":
                    self._map_regressed[ep.wad] = self._map_regressed.get(ep.wad, 0.0) * 0.5
                p = sum(hist) / float(len(hist))
                fam_learn_sigs.setdefault(ep.family, []).append(4.0 * p * (1.0 - p) + 0.25 * (1.0 - p))

        if self.config.adaptive_weights:
            for wad in list(self._map_staleness):
                if wad not in seen_wads:
                    self._map_staleness[wad] += 1
            if self.config.adaptive_mode == "sfl_pvl" and fam_learn_sigs:
                for fam, vals in fam_learn_sigs.items():
                    sig = sum(vals) / float(len(vals))
                    prev = self._family_hedge.get(fam, 1.0)
                    self._family_hedge[fam] = max(0.5, min(2.0, prev * math.exp(self.config.hedge_eta * (sig - 0.5))))
                mean_h = sum(self._family_hedge.values()) / float(max(1, len(self._family_hedge)))
                if mean_h > 1e-6:
                    for fam in list(self._family_hedge):
                        self._family_hedge[fam] = max(0.5, min(2.0, self._family_hedge[fam] / mean_h))

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
