"""Greedy Model Soup builder, B=2 mirror-equivariance tester, and multi-suite selector."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

import torch

from submission.policy import load_checkpoint, save_checkpoint
from training.eval import evaluate_suite, resolve_maps


def _compatible_state_dicts(sd_a: dict[str, torch.Tensor], sd_b: dict[str, torch.Tensor]) -> bool:
    if set(sd_a.keys()) != set(sd_b.keys()):
        return False
    return all(sd_a[k].shape == sd_b[k].shape for k in sd_a)


def _blend_checkpoints(
    paths_and_weights: list[tuple[Path, float]],
    out_path: Path,
    *,
    eval_deterministic: bool = True,
    rocket_safety_mask: bool = True,
    eval_mirror_ensemble: bool = False,
) -> Path:
    total_w = sum(w for _, w in paths_and_weights)
    if total_w <= 0.0:
        raise ValueError("Soup weights must sum to > 0")
    loaded = [load_checkpoint(p, device="cpu") for p, _ in paths_and_weights]
    base_model, base_cfg, _ = loaded[0]
    base_sd = base_model.state_dict()
    for m, _, _ in loaded[1:]:
        if not _compatible_state_dicts(base_sd, m.state_dict()):
            raise ValueError("Incompatible state dicts cannot be souped")

    avg_sd: dict[str, torch.Tensor] = {}
    for k in base_sd:
        acc = torch.zeros_like(base_sd[k], dtype=torch.float32)
        for (m, _, _), (_, w) in zip(loaded, paths_and_weights):
            acc += (float(w) / total_w) * m.state_dict()[k].float()
        avg_sd[k] = acc.to(dtype=base_sd[k].dtype)

    base_model.load_state_dict(avg_sd)
    cfg = replace(
        base_cfg,
        eval_deterministic=eval_deterministic,
        rocket_safety_mask=rocket_safety_mask,
        eval_mirror_ensemble=eval_mirror_ensemble,
    )
    meta = {
        "soup_sources": [{"path": str(p), "weight": float(w)} for p, w in paths_and_weights],
        "eval_deterministic": eval_deterministic,
        "rocket_safety_mask": rocket_safety_mask,
        "eval_mirror_ensemble": eval_mirror_ensemble,
    }
    return save_checkpoint(out_path, base_model, cfg, meta=meta)


def _evaluate_suites(
    ckpt_path: Path,
    suites: list[str],
    *,
    device: str = "cpu",
    corpus_manifest: Path | None = None,
) -> dict[str, Any]:
    prev = os.environ.get("DOOM_AGENT_WEIGHTS")
    os.environ["DOOM_AGENT_WEIGHTS"] = str(ckpt_path.resolve())
    per_suite: dict[str, Any] = {}
    all_episodes: list[dict[str, Any]] = []
    try:
        for s in suites:
            if s.startswith("manifest:") and corpus_manifest is not None:
                import json as _json
                from training.interfaces import MapSpec

                recs = [_json.loads(ln) for ln in corpus_manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
                specs = [
                    MapSpec(r["wad"], map_name=r["map_name"], family=r["family"], tier=r["tier"],
                            episode_ticks=r["episode_ticks"], split=r["split"])
                    for r in recs
                ]
                rep = evaluate_suite(specs, device=device, seeds_override=(0,), eval_mode="default")
            else:
                specs = resolve_maps(suite=s)
                rep = evaluate_suite(specs, device=device, eval_mode="default")
            per_suite[s] = rep["summary"]["overall"]
            all_episodes.extend(rep.get("episodes", []))
    finally:
        if prev is None:
            os.environ.pop("DOOM_AGENT_WEIGHTS", None)
        else:
            os.environ["DOOM_AGENT_WEIGHTS"] = prev

    from training.game_theory import compute_polytope_scores
    from training.eval import summarize_episodes
    from training.interfaces import EpisodeMetrics

    ep_objs = [
        EpisodeMetrics(
            wad=e["wad"],
            map_name=e["map_name"],
            family=e["family"],
            tier=e["tier"],
            seed=e["seed"],
            steps=e["steps"],
            ticks=e["ticks"],
            outcome=e["outcome"],
            keys_picked_up=e["keys_picked_up"],
            doors_opened=e["doors_opened"],
            locked_doors_unlocked=e["locked_doors_unlocked"],
            kills=e["kills"],
            damage_taken=e["damage_taken"],
            final_health=e["final_health"],
            unique_cells_visited=e["unique_cells_visited"],
            total_reward=e["total_reward"],
            agent_time_s=e.get("agent_time_s", 0.0),
            extra=e.get("extra", {}),
        )
        for e in all_episodes
    ]
    agg = summarize_episodes(ep_objs)["overall"] if ep_objs else {}
    poly = compute_polytope_scores(ep_objs) if ep_objs else {"min_corner": 0.0, "nominal": 0.0}
    dev_exits = int(per_suite.get("dev", {}).get("exits", 0))
    dev_total = int(per_suite.get("dev", {}).get("episodes", 0))
    return {
        "path": str(ckpt_path),
        "per_suite": per_suite,
        "overall": agg,
        "polytope_min": float(poly.get("min_corner", 0.0)),
        "dev_exits": dev_exits,
        "dev_total": dev_total,
    }


def _rank_key(row: dict[str, Any]) -> tuple:
    ov = row["overall"]
    clean = 1 if (ov.get("errors", 0) == 0 and ov.get("all_within_budget", True)) else 0
    dev_ok = 1 if (row["dev_total"] == 0 or row["dev_exits"] >= max(0, row["dev_total"] - 2)) else 0
    return (
        clean,
        dev_ok,
        round(float(row["polytope_min"]), 4),
        int(ov.get("tier34_exits", 0)),
        int(ov.get("exits", 0)),
        int(ov.get("locked_doors_unlocked", 0)),
        -int(ov.get("deaths", 0)),
        round(float(ov.get("cvar_25_rubric", 0.0)), 4),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate, soup, and select the best compatible checkpoint.")
    ap.add_argument("--base-policy", required=True, type=Path)
    ap.add_argument("--candidates", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report-out", type=Path, default=None)
    ap.add_argument("--work-dir", type=Path, default=Path("training/runs/soup_work"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suites", default="dev,final_ood,stress,freedoom")
    ap.add_argument("--fast-suites", default="dev,final_ood")
    ap.add_argument("--corpus-manifest", type=Path, default=None)
    args = ap.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    fast_suites = [s.strip() for s in args.fast_suites.split(",") if s.strip()]

    # Materialize clean eval_deterministic=True, rocket_safety_mask=True copies for base & candidates
    pool_entries: list[tuple[str, Path]] = []
    base_clean = args.work_dir / "base_floor.pt"
    _blend_checkpoints([(args.base_policy, 1.0)], base_clean)
    pool_entries.append(("s6b_floor", base_clean))

    valid_cands = [p for p in args.candidates if p.is_file()]
    clean_cands: list[tuple[str, Path]] = []
    for idx, p in enumerate(valid_cands):
        cpath = args.work_dir / f"cand_{idx}_{p.stem}.pt"
        try:
            _blend_checkpoints([(p, 1.0)], cpath)
            clean_cands.append((p.parent.name + "/" + p.name if p.name == "policy.pt" else p.stem, cpath))
        except Exception as exc:
            print(f"[soup] skipping {p}: {exc}", flush=True)

    pool_entries.extend(clean_cands)
    if clean_cands:
        last_name, last_path = clean_cands[-1]
        for w_last, tag in ((0.7, "soup_07_03"), (0.5, "soup_05_05")):
            sp = args.work_dir / f"{tag}.pt"
            _blend_checkpoints([(last_path, w_last), (base_clean, 1.0 - w_last)], sp)
            pool_entries.append((f"{tag}({last_name}+s6b)", sp))
        if len(clean_cands) >= 2:
            top3 = clean_cands[-3:]
            sp = args.work_dir / "soup_mean_last.pt"
            _blend_checkpoints([(p, 1.0) for _, p in top3], sp)
            pool_entries.append((f"soup_mean({len(top3)})", sp))

    screen_rows: list[dict[str, Any]] = []
    screen_suite_list = fast_suites if set(fast_suites) != set(suites) else suites
    for name, path in pool_entries:
        res = _evaluate_suites(path, screen_suite_list, device=args.device, corpus_manifest=args.corpus_manifest)
        res["name"] = name
        screen_rows.append(res)
        ov = res["overall"]
        print(
            f"[screen] {name:30s} exits={ov.get('exits',0)}/{ov.get('episodes',0)} "
            f"deaths={ov.get('deaths',0)} unlock={ov.get('locked_doors_unlocked',0)} "
            f"poly_min={res['polytope_min']:.4f} cvar25={ov.get('cvar_25_rubric',0.0):.4f}",
            flush=True,
        )

    screen_rows.sort(key=_rank_key, reverse=True)
    # Full confirmation on top 2 + base_floor across all suites
    confirm_names = []
    for r in screen_rows:
        if r["name"] not in confirm_names and len(confirm_names) < 2:
            confirm_names.append(r["name"])
    if "s6b_floor" not in confirm_names:
        confirm_names.append("s6b_floor")

    final_rows: list[dict[str, Any]] = []
    if set(screen_suite_list) == set(suites):
        final_rows = [r for r in screen_rows if r["name"] in confirm_names]
    else:
        by_name = dict(pool_entries)
        for name in confirm_names:
            res = _evaluate_suites(by_name[name], suites, device=args.device, corpus_manifest=args.corpus_manifest)
            res["name"] = name
            final_rows.append(res)
            ov = res["overall"]
            print(
                f"[confirm] {name:29s} exits={ov.get('exits',0)}/{ov.get('episodes',0)} "
                f"t34={ov.get('tier34_exits',0)} deaths={ov.get('deaths',0)} "
                f"unlock={ov.get('locked_doors_unlocked',0)} poly_min={res['polytope_min']:.4f}",
                flush=True,
            )

    final_rows.sort(key=_rank_key, reverse=True)
    best_row = final_rows[0]

    # Also test B=2 mirror equivariance on winner
    best_mir_path = args.work_dir / "winner_mirror_b2.pt"
    _blend_checkpoints([(Path(best_row["path"]), 1.0)], best_mir_path, eval_mirror_ensemble=True)
    mir_row = _evaluate_suites(best_mir_path, suites, device=args.device, corpus_manifest=args.corpus_manifest)
    mir_row["name"] = best_row["name"] + "+B2mirror"
    final_rows.append(mir_row)
    ov_m = mir_row["overall"]
    print(
        f"[mirror]  {mir_row['name']:29s} exits={ov_m.get('exits',0)}/{ov_m.get('episodes',0)} "
        f"deaths={ov_m.get('deaths',0)} unlock={ov_m.get('locked_doors_unlocked',0)} "
        f"poly_min={mir_row['polytope_min']:.4f} max_ms={ov_m.get('max_step_ms',0.0):.2f}",
        flush=True,
    )

    final_rows.sort(key=_rank_key, reverse=True)
    winner = final_rows[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m, c, meta = load_checkpoint(Path(winner["path"]), device="cpu")
    meta["selection_summary"] = {
        "winner": winner["name"],
        "polytope_min": winner["polytope_min"],
        "overall": winner["overall"],
        "per_suite": winner["per_suite"],
    }
    save_checkpoint(args.out, m, c, meta=meta)
    print(f"[selected] {winner['name']} -> {args.out}", flush=True)

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps({"screen": screen_rows, "final": final_rows, "winner": winner}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
