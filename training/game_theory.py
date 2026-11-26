"""Game-theoretic curriculum, primal-dual CMDP rewards, ACCEL evolution, and Nash evaluation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from training.interfaces import EpisodeMetrics, MapSpec

RUBRIC_POLYTOPE_CORNERS: dict[str, dict[str, Any]] = {
    "exit_heavy": {
        "w_exit": 0.70,
        "w_eff": 0.10,
        "w_prog": 0.10,
        "w_expl": 0.05,
        "w_combat": 0.05,
        "w_early_death": 0.15,
        "tier_weights": {1: 1.0, 2: 1.5, 3: 2.5, 4: 4.0},
    },
    "balanced": {
        "w_exit": 0.50,
        "w_eff": 0.10,
        "w_prog": 0.20,
        "w_expl": 0.10,
        "w_combat": 0.10,
        "w_early_death": 0.15,
        "tier_weights": {1: 1.0, 2: 1.5, 3: 2.5, 4: 4.0},
    },
    "progression_heavy": {
        "w_exit": 0.45,
        "w_eff": 0.05,
        "w_prog": 0.30,
        "w_expl": 0.10,
        "w_combat": 0.10,
        "w_early_death": 0.15,
        "tier_weights": {1: 1.0, 2: 1.5, 3: 3.0, 4: 4.5},
    },
    "combat_survival": {
        "w_exit": 0.45,
        "w_eff": 0.10,
        "w_prog": 0.15,
        "w_expl": 0.05,
        "w_combat": 0.25,
        "w_early_death": 0.25,
        "tier_weights": {1: 1.0, 2: 1.5, 3: 2.5, 4: 4.0},
    },
    "steep_tier4": {
        "w_exit": 0.55,
        "w_eff": 0.10,
        "w_prog": 0.15,
        "w_expl": 0.10,
        "w_combat": 0.10,
        "w_early_death": 0.20,
        "tier_weights": {1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0},
    },
}


def _ep_field(ep: EpisodeMetrics | dict[str, Any], name: str, default: Any = 0) -> Any:
    if isinstance(ep, dict):
        return ep.get(name, default)
    return getattr(ep, name, default)


def compute_episode_rubric_score(
    ep: EpisodeMetrics | dict[str, Any],
    *,
    corner: str = "balanced",
    early_death_ticks: int = 700,
) -> float:
    """Compute bounded `[0, 1+]` verifiable rubric score aligned with `docs/EVALUATION.md`."""
    if corner not in RUBRIC_POLYTOPE_CORNERS:
        raise ValueError(f"Unknown rubric corner {corner!r}; expected one of {tuple(RUBRIC_POLYTOPE_CORNERS)}")
    weights = RUBRIC_POLYTOPE_CORNERS[corner]

    tier = int(_ep_field(ep, "tier", 1) or 1)
    outcome = str(_ep_field(ep, "outcome", "timeout"))
    ticks = int(_ep_field(ep, "ticks", 0) or 0)
    extra = _ep_field(ep, "extra", {}) or {}
    default_ticks = 4200 if tier >= 3 else 2100
    ep_ticks = int(_ep_field(ep, "episode_ticks", 0) or extra.get("episode_ticks", 0) or default_ticks)
    final_health = float(_ep_field(ep, "final_health", 0.0) or 0.0)
    keys_picked = int(_ep_field(ep, "keys_picked_up", 0) or 0)
    doors_opened = int(_ep_field(ep, "doors_opened", 0) or 0)
    unlocks = int(_ep_field(ep, "locked_doors_unlocked", 0) or 0)
    cells = int(_ep_field(ep, "unique_cells_visited", 0) or 0)
    kills = int(_ep_field(ep, "kills", 0) or 0)

    exit_term = 1.0 if outcome == "exit" else 0.0
    if outcome == "exit":
        time_rem = max(0.0, 1.0 - float(ticks) / float(max(1, ep_ticks)))
        hp_frac = min(1.0, max(0.0, final_health) / 100.0)
        eff_term = 0.5 * time_rem + 0.5 * hp_frac
    else:
        eff_term = 0.0

    prog_term = min(
        1.0,
        0.35 * min(keys_picked, 2) + 0.50 * min(unlocks, 2) + 0.15 * min(doors_opened, 2),
    )
    expl_term = min(1.0, max(0.0, float(cells)) / 60.0)
    combat_term = min(1.0, max(0.0, float(kills)) / 5.0)
    early_death_pen = 1.0 if (outcome == "death" and ticks <= early_death_ticks) else 0.0

    raw = (
        float(weights["w_exit"]) * exit_term
        + float(weights["w_eff"]) * eff_term
        + float(weights["w_prog"]) * prog_term
        + float(weights["w_expl"]) * expl_term
        + float(weights["w_combat"]) * combat_term
        - float(weights["w_early_death"]) * early_death_pen
    )
    return float(max(0.0, raw))


def _tier_weighted_rubric_mean(
    episodes: Sequence[EpisodeMetrics | dict[str, Any]],
    *,
    corner: str,
) -> float:
    if not episodes:
        return 0.0
    tier_weights = RUBRIC_POLYTOPE_CORNERS[corner]["tier_weights"]
    num = 0.0
    den = 0.0
    for ep in episodes:
        tier = int(_ep_field(ep, "tier", 1) or 1)
        w = float(tier_weights.get(tier, 1.0))
        num += w * compute_episode_rubric_score(ep, corner=corner)
        den += w
    return num / den if den > 0.0 else 0.0


def compute_polytope_scores(episodes: Sequence[EpisodeMetrics | dict[str, Any]]) -> dict[str, Any]:
    """Evaluate verifiable rubric scores across all 5 hidden-weight polytope corners."""
    corners: dict[str, float] = {}
    for corner in RUBRIC_POLYTOPE_CORNERS:
        corners[corner] = round(_tier_weighted_rubric_mean(episodes, corner=corner), 4)
    if not corners:
        return {"corners": {}, "nominal": 0.0, "min_corner": 0.0, "min_corner_name": "balanced", "max_corner": 0.0}
    min_name = min(corners, key=lambda k: corners[k])
    max_val = max(corners.values())
    return {
        "corners": corners,
        "nominal": corners["balanced"],
        "min_corner": corners[min_name],
        "min_corner_name": min_name,
        "max_corner": max_val,
    }


def compute_cvar(per_map_scores: dict[str, float], alpha: float = 0.25) -> float:
    """Compute lower-tail Conditional Value at Risk (`CVaR_alpha`) across per-map scores."""
    if not per_map_scores:
        return 0.0
    vals = sorted(float(v) for v in per_map_scores.values())
    k = max(1, min(len(vals), int(math.ceil(float(alpha) * len(vals)))))
    return round(sum(vals[:k]) / float(k), 4)


def solve_zero_sum_nash(payoff: np.ndarray, *, iters: int = 2000) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve approximate Nash equilibrium `(p*, q*, value)` of a row-max/col-min matrix game via Hedge."""
    m = np.asarray(payoff, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D payoff matrix, got shape {m.shape}")
    n_agents, n_maps = m.shape
    if n_agents == 1:
        p = np.ones(1, dtype=np.float64)
        min_idx = int(np.argmin(m[0]))
        q = np.zeros(n_maps, dtype=np.float64)
        q[min_idx] = 1.0
        return p, q, float(m[0, min_idx])
    if n_maps == 1:
        max_idx = int(np.argmax(m[:, 0]))
        p = np.zeros(n_agents, dtype=np.float64)
        p[max_idx] = 1.0
        q = np.ones(1, dtype=np.float64)
        return p, q, float(m[max_idx, 0])

    lo = float(np.min(m))
    span = float(np.max(m) - lo)
    if span < 1e-12:
        p = np.full(n_agents, 1.0 / n_agents, dtype=np.float64)
        q = np.full(n_maps, 1.0 / n_maps, dtype=np.float64)
        return p, q, float(lo)

    m_norm = (m - lo) / span
    t_steps = max(100, int(iters))
    eta = math.sqrt(2.0 * math.log(float(max(n_agents, n_maps))) / float(t_steps))

    p = np.full(n_agents, 1.0 / n_agents, dtype=np.float64)
    q = np.full(n_maps, 1.0 / n_maps, dtype=np.float64)
    p_sum = np.zeros(n_agents, dtype=np.float64)
    q_sum = np.zeros(n_maps, dtype=np.float64)

    for _ in range(t_steps):
        p_sum += p
        q_sum += q
        row_pay = m_norm @ q
        col_cost = p @ m_norm
        row_shift = row_pay - float(np.max(row_pay))
        col_shift = -(col_cost - float(np.min(col_cost)))
        p = p * np.exp(eta * row_shift)
        p /= float(np.sum(p))
        q = q * np.exp(eta * col_shift)
        q /= float(np.sum(q))

    p_avg = p_sum / float(np.sum(p_sum))
    q_avg = q_sum / float(np.sum(q_sum))
    value = float(p_avg @ m @ q_avg)
    return p_avg, q_avg, value


def _per_map_rubric_dict(
    report: dict[str, Any],
    *,
    corner: str = "balanced",
) -> tuple[dict[str, float], dict[str, int]]:
    buckets: dict[str, list[float]] = {}
    tiers: dict[str, int] = {}
    for ep in report.get("episodes", []):
        key = f"{Path(str(ep['wad'])).name}:{str(ep.get('map_name', 'MAP01')).upper()}"
        buckets.setdefault(key, []).append(compute_episode_rubric_score(ep, corner=corner))
        if key not in tiers:
            tiers[key] = int(_ep_field(ep, "tier", 1) or 1)
    return {k: sum(vals) / float(len(vals)) for k, vals in buckets.items() if vals}, tiers


def compare_reports_nash(
    named_reports: Sequence[tuple[str, dict[str, Any]]],
    *,
    corner: str = "balanced",
) -> dict[str, Any]:
    """Compare 2+ evaluation reports via tier-weighted Minimax-Regret Nash equilibrium, CVaR-25, and frontier maps."""
    if not named_reports:
        raise ValueError("named_reports must not be empty")
    if corner not in RUBRIC_POLYTOPE_CORNERS:
        raise ValueError(f"Unknown rubric corner {corner!r}; expected one of {tuple(RUBRIC_POLYTOPE_CORNERS)}")

    per_agent_maps: list[dict[str, float]] = []
    map_tiers: dict[str, int] = {}
    for _, rep in named_reports:
        mdict, tiers = _per_map_rubric_dict(rep, corner=corner)
        per_agent_maps.append(mdict)
        for mk, tr in tiers.items():
            map_tiers.setdefault(mk, tr)

    key_sets = [set(d.keys()) for d in per_agent_maps if d]
    common_keys = sorted(set.intersection(*key_sets)) if len(key_sets) == len(named_reports) and key_sets else []
    common_count = len(common_keys)
    map_keys = common_keys if common_keys else sorted({k for d in per_agent_maps for k in d})
    if not map_keys:
        map_keys = ["empty:MAP01"]

    payoff = np.zeros((len(named_reports), len(map_keys)), dtype=np.float64)
    for i, mdict in enumerate(per_agent_maps):
        for j, mk in enumerate(map_keys):
            payoff[i, j] = float(mdict.get(mk, 0.0))

    tier_weights = RUBRIC_POLYTOPE_CORNERS[corner]["tier_weights"]
    w_vec = np.array([float(tier_weights.get(map_tiers.get(mk, 1), 1.0)) for mk in map_keys], dtype=np.float64)
    col_best = np.max(payoff, axis=0)
    regret_mat = (col_best[None, :] - payoff) * w_vec[None, :]

    # Solve Minimax-Regret zero-sum game: row maximizes -regret (minimizes regret), col minimizes -regret.
    p_star, q_star, neg_regret_val = solve_zero_sum_nash(-regret_mat)
    game_val = float(p_star @ payoff @ q_star)

    agents_out: list[dict[str, Any]] = []
    for i, (name, rep) in enumerate(named_reports):
        eps = rep.get("episodes", [])
        poly = compute_polytope_scores(eps)
        mdict = {mk: float(payoff[i, j]) for j, mk in enumerate(map_keys)}
        raw_regrets = col_best - payoff[i]
        nash_regret = float(regret_mat[i] @ q_star)
        agents_out.append(
            {
                "name": name,
                "nash_regret": round(nash_regret, 4),
                "nash_value": round(float(payoff[i] @ q_star), 4),
                "nash_weight": round(float(p_star[i]), 4),
                "minimax_regret": round(float(np.max(raw_regrets)), 4),
                "mean_rubric": round(float(np.mean(payoff[i])), 4),
                "cvar_25": compute_cvar(mdict, 0.25),
                "polytope_min": poly["min_corner"],
            }
        )
    agents_out.sort(key=lambda row: (row["nash_regret"], -row["nash_value"], row["minimax_regret"], -row["mean_rubric"]))

    frontier_order = np.argsort(-q_star)
    frontier_maps: list[dict[str, Any]] = []
    for idx in frontier_order[:8]:
        j = int(idx)
        frontier_maps.append(
            {
                "map": map_keys[j],
                "tier": int(map_tiers.get(map_keys[j], 1)),
                "nash_weight": round(float(q_star[j]), 4),
                "scores": {name: round(float(payoff[i, j]), 4) for i, (name, _) in enumerate(named_reports)},
            }
        )

    return {
        "corner": corner,
        "common_maps": common_count,
        "total_maps": len(map_keys),
        "game_value": round(float(game_val), 4),
        "minimax_regret_game_value": round(float(-neg_regret_val), 4),
        "agents": agents_out,
        "frontier_maps": frontier_maps,
    }


@dataclass(frozen=True)
class PrimalDualConfig:
    """Configuration for Lagrangian CMDP auto-calibrated verifiable penalty weights."""

    enabled: bool = True
    lr: float = 0.15
    window_size: int = 32
    damage_target: float = 35.0
    early_death_target: float = 0.08
    post_key_death_target: float = 0.08
    use_waste_target: float = 0.08
    lambda_max: float = 2.0
    damage_penalty_bounds: tuple[float, float] = (-0.08, -0.035)
    death_penalty_bounds: tuple[float, float] = (-11.0, -5.0)
    post_key_death_bounds: tuple[float, float] = (-3.5, 0.0)
    use_waste_bounds: tuple[float, float] = (-0.020, -0.002)


class PrimalDualController:
    """Updates Lagrangian dual multipliers on verifiable behavioral cost constraints."""

    def __init__(self, config: PrimalDualConfig | None = None) -> None:
        self.config = config or PrimalDualConfig()
        win = max(4, int(self.config.window_size))
        self._dmg_hist: deque[float] = deque(maxlen=win)
        self._early_hist: deque[float] = deque(maxlen=win)
        self._pk_hist: deque[float] = deque(maxlen=win)
        self._use_hist: deque[float] = deque(maxlen=win)
        self.lambdas: dict[str, float] = {
            "damage": 0.0,
            "early_death": 0.0,
            "post_key_death": 0.0,
            "use_waste": 0.0,
        }

    def current_metrics(self) -> dict[str, float]:
        return {f"cmdp_lambda_{k}": round(float(v), 4) for k, v in self.lambdas.items()}

    def observe_episodes(self, episodes: Sequence[EpisodeMetrics], base_cfg: Any) -> tuple[Any, dict[str, float]]:
        if not self.config.enabled:
            return base_cfg, self.current_metrics()

        for ep in episodes:
            self._dmg_hist.append(float(ep.damage_taken))
            extra = ep.extra or {}
            early = bool(extra.get("early_death", ep.outcome == "death" and ep.ticks <= 700))
            pk_death = bool(extra.get("died_after_pickup", ep.outcome == "death" and ep.keys_picked_up > 0))
            had_key = ep.keys_picked_up > 0 or bool(extra.get("had_key_pickup")) or pk_death
            self._early_hist.append(1.0 if early else 0.0)
            if had_key:
                self._pk_hist.append(1.0 if pk_death else 0.0)
            if "use_waste_rate" in extra:
                self._use_hist.append(float(extra["use_waste_rate"]))

        cfg = self.config
        if len(self._dmg_hist) >= 4:
            mean_dmg = sum(self._dmg_hist) / float(len(self._dmg_hist))
            mean_early = sum(self._early_hist) / float(len(self._early_hist))
            v_dmg = float(np.clip((mean_dmg - cfg.damage_target) / max(1.0, cfg.damage_target), -1.5, 1.5))
            v_early = float(np.clip((mean_early - cfg.early_death_target) / max(0.01, cfg.early_death_target), -1.5, 1.5))
            self.lambdas["damage"] = float(np.clip(self.lambdas["damage"] + cfg.lr * v_dmg, 0.0, cfg.lambda_max))
            self.lambdas["early_death"] = float(
                np.clip(self.lambdas["early_death"] + cfg.lr * v_early, 0.0, cfg.lambda_max)
            )
        if len(self._pk_hist) >= 4:
            mean_pk = sum(self._pk_hist) / float(len(self._pk_hist))
            v_pk = float(np.clip((mean_pk - cfg.post_key_death_target) / max(0.01, cfg.post_key_death_target), -1.5, 1.5))
            self.lambdas["post_key_death"] = float(
                np.clip(self.lambdas["post_key_death"] + cfg.lr * v_pk, 0.0, cfg.lambda_max)
            )
        if len(self._use_hist) >= 4:
            mean_use = sum(self._use_hist) / float(len(self._use_hist))
            v_use = float(np.clip((mean_use - cfg.use_waste_target) / max(0.01, cfg.use_waste_target), -1.5, 1.5))
            self.lambdas["use_waste"] = float(
                np.clip(self.lambdas["use_waste"] + cfg.lr * v_use, 0.0, cfg.lambda_max)
            )

        l_max = max(1e-6, cfg.lambda_max)
        dmg_frac = self.lambdas["damage"] / l_max
        early_frac = self.lambdas["early_death"] / l_max
        pk_frac = self.lambdas["post_key_death"] / l_max
        use_frac = self.lambdas["use_waste"] / l_max

        dmg_lo, dmg_hi = cfg.damage_penalty_bounds
        death_lo, death_hi = cfg.death_penalty_bounds
        pk_lo, pk_hi = cfg.post_key_death_bounds
        use_lo, use_hi = cfg.use_waste_bounds

        dmg_pen = float(np.clip(base_cfg.damage_penalty + dmg_frac * (dmg_lo - base_cfg.damage_penalty), dmg_lo, dmg_hi))
        death_pen = float(
            np.clip(base_cfg.death_penalty + early_frac * (death_lo - base_cfg.death_penalty), death_lo, death_hi)
        )
        pk_pen = float(
            np.clip(
                base_cfg.post_key_death_penalty + pk_frac * (pk_lo - base_cfg.post_key_death_penalty),
                pk_lo,
                pk_hi,
            )
        )
        use_pen = float(
            np.clip(base_cfg.use_waste_penalty + use_frac * (use_lo - base_cfg.use_waste_penalty), use_lo, use_hi)
        )

        updated = replace(
            base_cfg,
            damage_penalty=round(dmg_pen, 5),
            death_penalty=round(death_pen, 4),
            post_key_death_penalty=round(pk_pen, 4),
            use_waste_penalty=round(use_pen, 5),
        )
        return updated, self.current_metrics()


def load_champion_baselines(report_path: str | Path, *, corner: str = "balanced") -> dict[str, dict[str, float]]:
    """Load per-WAD verifiable rubric & exit baselines keyed by `wad_name:MAP_NAME` from an eval JSON."""
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[float, float]]] = {}
    for ep in data.get("episodes", []):
        name = f"{Path(str(ep['wad'])).name}:{str(ep.get('map_name', 'MAP01')).upper()}"
        rubric = compute_episode_rubric_score(ep, corner=corner)
        exited = 1.0 if ep.get("outcome") == "exit" else 0.0
        grouped.setdefault(name, []).append((rubric, exited))
    out: dict[str, dict[str, float]] = {}
    for name, pairs in grouped.items():
        n = float(len(pairs))
        out[name] = {
            "rubric": round(sum(p[0] for p in pairs) / n, 4),
            "exit_rate": round(sum(p[1] for p in pairs) / n, 4),
        }
    return out


@dataclass
class _ArchiveEntry:
    spec: MapSpec
    genome: Any
    score: float = 1.0
    staleness: int = 0


class AccelLevelArchive:
    """ACCEL-style evolutionary archive retaining frontier levels and mutating high-regret genomes."""

    def __init__(self, *, elite_frac: float = 0.25, mutate_frac: float = 0.50) -> None:
        self.elite_frac = float(elite_frac)
        self.mutate_frac = float(mutate_frac)
        self._entries_by_wad: dict[str, _ArchiveEntry] = {}
        self._pending_unlink: list[Path] = []

    def cleanup_retired(self) -> None:
        """Delete retired WADs strictly after vec_env has switched away from them."""
        for p in self._pending_unlink:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        self._pending_unlink.clear()

    def seed_pool(self, specs_and_genomes: Sequence[tuple[MapSpec, Any]]) -> None:
        for spec, genome in specs_and_genomes:
            self._entries_by_wad[spec.wad] = _ArchiveEntry(spec=spec, genome=genome, score=1.0, staleness=0)

    def update_scores(self, score_by_wad: dict[str, float]) -> None:
        for wad, entry in self._entries_by_wad.items():
            if wad in score_by_wad:
                entry.score = float(score_by_wad[wad])
                entry.staleness = 0
            else:
                entry.staleness += 1

    def evolve_pool(
        self,
        *,
        count_per_family: int,
        families: Sequence[str],
        out_dir: str | Path,
        rng: Any,
        generation_seed: int,
        genome_cls: type[Any],
        generator_fn: Callable[..., MapSpec],
        mutate_fn: Callable[[Any, Any], Any],
        defer_unlink: bool = False,
    ) -> list[MapSpec]:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        k_fam = max(1, int(count_per_family))
        new_pool: list[MapSpec] = []
        new_entries: dict[str, _ArchiveEntry] = {}

        n_elite = max(1, int(round(k_fam * self.elite_frac))) if k_fam >= 2 else 0
        n_mutate = max(1, int(round(k_fam * self.mutate_frac))) if k_fam >= 2 else 1

        next_seed = int(generation_seed)
        for fam in families:
            fam_entries = [e for e in self._entries_by_wad.values() if e.spec.family == fam]
            fam_entries.sort(key=lambda e: -(e.score + 0.05 * min(10, e.staleness)))

            elites = fam_entries[: min(n_elite, len(fam_entries))]
            for entry in elites:
                if Path(entry.spec.wad).exists():
                    new_pool.append(entry.spec)
                    new_entries[entry.spec.wad] = _ArchiveEntry(
                        spec=entry.spec,
                        genome=entry.genome,
                        score=entry.score,
                        staleness=entry.staleness,
                    )

            target_muts = min(n_mutate, k_fam - sum(1 for s in new_pool if s.family == fam))
            for m_idx in range(max(0, target_muts)):
                next_seed += 1
                if fam_entries:
                    parent = fam_entries[m_idx % max(1, min(len(fam_entries), max(2, len(fam_entries) // 2)))]
                    cand_genome = mutate_fn(parent.genome, rng)
                    cand_genome = replace(cand_genome, seed=next_seed)
                else:
                    cand_genome = genome_cls(family=fam, seed=next_seed)
                wad_file = out_path / f"proc_accel_{fam}_g{getattr(cand_genome, 'generation', 0)}_s{next_seed}.wad"
                try:
                    spec = generator_fn(cand_genome, out_path, out_path=wad_file)
                except Exception:
                    fresh_genome = genome_cls(family=fam, seed=next_seed + 10_000)
                    wad_file = out_path / f"proc_accel_{fam}_s{fresh_genome.seed}.wad"
                    spec = generator_fn(fresh_genome, out_path, out_path=wad_file)
                    cand_genome = fresh_genome
                new_pool.append(spec)
                new_entries[spec.wad] = _ArchiveEntry(spec=spec, genome=cand_genome, score=1.25, staleness=0)

            while sum(1 for s in new_pool if s.family == fam) < k_fam:
                next_seed += 1
                fresh_genome = genome_cls(family=fam, seed=next_seed)
                wad_file = out_path / f"proc_accel_{fam}_s{next_seed}.wad"
                spec = generator_fn(fresh_genome, out_path, out_path=wad_file)
                new_pool.append(spec)
                new_entries[spec.wad] = _ArchiveEntry(spec=spec, genome=fresh_genome, score=1.25, staleness=0)

        keep_resolved = {str(Path(s.wad).resolve()) for s in new_pool}
        for wad in list(self._entries_by_wad):
            p = Path(wad)
            if p.exists() and str(p.resolve()) not in keep_resolved and p.name.startswith("proc_accel_"):
                if defer_unlink:
                    self._pending_unlink.append(p)
                else:
                    try:
                        p.unlink()
                    except OSError:
                        pass

        self._entries_by_wad = new_entries
        return new_pool
