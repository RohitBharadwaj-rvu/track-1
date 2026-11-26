"""Role 2 CLI training loop, staged curriculum, throughput sweep, and Stage-1 two-map verifier."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import shutil
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from submission.obs import NUM_ACTIONS
from submission.policy import PolicyConfig, build_policy, save_checkpoint
from training.curriculum import CurriculumConfig, CurriculumScheduler
from training.game_theory import AccelLevelArchive, PrimalDualController, load_champion_baselines
from training.generate_corpus import CorpusPoolSampler
from training.interfaces import EpisodeMetrics, MapSpec
from training.ppo import PPOConfig, RecurrentPPOTrainer, load_training_state, save_training_state
from training.rewards import RewardConfig, RewardShaper
from training.rollout import (
    RolloutBuffer,
    SubprocVecDoomEnv,
    SyncVecDoomEnv,
    collect_rollout,
    resolve_train_maps,
)
from training.wad_gen.dungeon import (
    DungeonGenome,
    generate_dungeon_from_genome,
    mutate_dungeon_genome,
)
from training.wad_gen.extreme import EXTREME_ARCHETYPES, generate_extreme_wad
from training.wad_gen.procedural import generate_procedural_wad


@dataclass
class TrainConfig:
    arch: str = "telemetry_gru_v1"
    action_head: str = "factorized"  # "factorized" | "bernoulli"
    hidden_size: int = 256
    ray_channels: int = 32
    entity_dim: int = 64
    vitals_dim: int = 64
    num_memory_slots: int = 0
    use_ego_map: bool = True
    ego_map_out_dim: int = 128
    eval_deterministic: bool = False
    split: str = "train"
    stage: str | None = None
    procedural_per_family: int = 4
    procedural_dir: str | None = None
    procedural_refresh_updates: int = 20
    tier_weights: dict[int, float] | None = None
    auto_curriculum: bool = True
    adaptive_weights: bool = True
    adaptive_mode: str = "sfl_pvl"
    use_accel: bool = True
    use_primal_dual: bool = True
    champion_report: str | None = None
    corpus_manifest: str | None = None
    corpus_pool_size: int = 60
    corpus_high_diff_frac: float = 0.45
    corpus_include_freedoom: bool = True
    tier12_promote_threshold: float = 0.65
    tier3_promote_threshold: float = 0.50
    min_promote_episodes: int = 32
    promote_window_size: int = 48
    tier3_require_exit: bool = True
    mirror_prob: float = 0.5
    maps: tuple[str, ...] = ()
    num_envs: int = 4
    vec_mode: str = "subproc"  # "subproc" | "sync"
    max_cached_envs: int = 48
    rollout_steps: int = 256
    chunk_len: int = 64
    num_minibatches: int = 4
    update_epochs: int = 4
    total_steps: int = 50_000
    seed: int = 0
    device: str = "auto"
    lr: float = 3e-4
    min_lr_frac: float = 0.25
    ent_coef: float = 0.015
    ent_coef_end: float = 0.004
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    use_huber_vloss: bool = True
    huber_delta: float = 5.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    use_asymmetric_critic: bool = True
    aux_coef: float = 0.2
    curiosity_coef: float = 0.0
    pbrs_scale: float = 3.0
    anneal_lr: bool = True
    checkpoint_dir: str = "training/runs/default"
    export_path: str = "submission/weights/policy.pt"
    resume: str | None = None
    save_interval: int = 10
    save_every_seconds: float = 0.0
    max_train_seconds: float = 0.0
    log_interval: int = 1
    render_buffers: bool = False


def resolve_device(device_spec: str = "auto") -> str:
    import torch

    if device_spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_spec


def _parse_tier_weights(raw: str | None) -> dict[int, float] | None:
    if not raw:
        return None
    out: dict[int, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":", 1)
        out[int(k.strip())] = float(v.strip())
    return out or None


def _make_vec_env(
    map_specs: list[MapSpec],
    num_envs: int,
    vec_mode: str,
    seed: int,
    render_buffers: bool = False,
    reward_config: RewardConfig | None = None,
    map_weights: list[float] | None = None,
    max_cached_envs: int = 32,
    use_ego_map: bool = False,
    mirror_prob: float = 0.0,
    champion_baselines: dict[str, dict[str, float]] | None = None,
):
    cache_size = max(1, min(int(max_cached_envs), len(map_specs)))
    if vec_mode == "sync":
        return SyncVecDoomEnv(
            map_specs=map_specs,
            reward_config=reward_config,
            base_seed=seed,
            render_buffers=render_buffers,
            num_envs=num_envs,
            max_cached_envs=cache_size,
            map_weights=map_weights,
            use_ego_map=use_ego_map,
            mirror_prob=mirror_prob,
            champion_baselines=champion_baselines,
        )
    if vec_mode == "subproc":
        return SubprocVecDoomEnv(
            map_specs=map_specs,
            reward_config=reward_config,
            base_seed=seed,
            render_buffers=render_buffers,
            num_envs=num_envs,
            max_cached_envs=cache_size,
            map_weights=map_weights,
            use_ego_map=use_ego_map,
            mirror_prob=mirror_prob,
            champion_baselines=champion_baselines,
        )
    raise ValueError(f"Unsupported vec_mode: {vec_mode!r}")


def _summarize_episodes(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    if not episodes:
        return {
            "episodes": 0.0,
            "exit_rate": 0.0,
            "death_rate": 0.0,
            "timeout_rate": 0.0,
            "door_open_rate": 0.0,
            "unlock_rate": 0.0,
            "key_rate": 0.0,
            "mean_reward": 0.0,
            "mean_rubric_score": 0.0,
            "mean_cells": 0.0,
            "mean_kills": 0.0,
        }
    n = float(len(episodes))
    rubric_vals = [float((e.extra or {}).get("rubric_score", 0.0)) for e in episodes]
    return {
        "episodes": n,
        "exit_rate": sum(e.outcome == "exit" for e in episodes) / n,
        "death_rate": sum(e.outcome == "death" for e in episodes) / n,
        "timeout_rate": sum(e.outcome == "timeout" for e in episodes) / n,
        "door_open_rate": sum(e.doors_opened > 0 for e in episodes) / n,
        "unlock_rate": sum(e.locked_doors_unlocked > 0 for e in episodes) / n,
        "key_rate": sum(e.keys_picked_up > 0 for e in episodes) / n,
        "mean_reward": float(np.mean([e.total_reward for e in episodes])),
        "mean_rubric_score": float(np.mean(rubric_vals)),
        "mean_cells": float(np.mean([e.unique_cells_visited for e in episodes])),
        "mean_kills": float(np.mean([e.kills for e in episodes])),
    }


def _generate_hybrid_poly_pool(
    count_per_family: int,
    families: tuple[str, ...],
    proc_dir: Path,
    base_seed: int,
) -> list[MapSpec]:
    specs: list[MapSpec] = []
    if count_per_family <= 0:
        return specs
    seed = int(base_seed)
    for fam in families:
        for _ in range(int(count_per_family)):
            seed += 1
            if fam == "gauntlet":
                arch = EXTREME_ARCHETYPES[seed % len(EXTREME_ARCHETYPES)]
                out_wad = proc_dir / f"proc_poly_{fam}_s{seed}.wad"
                spec = generate_extreme_wad(arch, seed=seed, out_dir=proc_dir, split="train", out_path=out_wad)
            else:
                out_wad = proc_dir / f"proc_poly_{fam}_s{seed}.wad"
                spec = generate_procedural_wad(
                    fam,
                    seed=seed,
                    out_dir=proc_dir,
                    split="train",
                    with_combat=(fam == "key_return"),
                    out_path=out_wad,
                    continuous_angle=True,
                    poly_only=True,
                )
            specs.append(spec)
    return specs


def _init_accel_pool(
    cfg: TrainConfig,
    proc_dir: str,
) -> tuple[list[MapSpec], list[MapSpec], list[MapSpec], AccelLevelArchive]:
    base_specs = resolve_train_maps(
        None,
        split=cfg.split,
        stage=cfg.stage,
        procedural_per_family=0,
    )
    families = ("door_exit", "combat", "multi_room", "key_return", "gauntlet")
    proc_path = Path(proc_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    seed = cfg.seed + 1000
    pairs: list[tuple[MapSpec, DungeonGenome]] = []
    k_total = int(cfg.procedural_per_family)
    k_accel = max(1, k_total // 2) if k_total >= 2 else k_total
    k_poly = max(0, k_total - k_accel)
    for fam in families:
        for _ in range(k_accel):
            genome = DungeonGenome(family=fam, seed=seed)
            spec = generate_dungeon_from_genome(genome, proc_path)
            pairs.append((spec, genome))
            seed += 1
    poly_specs = _generate_hybrid_poly_pool(k_poly, families, proc_path, base_seed=seed + 50_000)
    archive = AccelLevelArchive()
    archive.seed_pool(pairs)
    return base_specs, [s for s, _ in pairs], poly_specs, archive


def run_training(cfg: TrainConfig, reward_config: RewardConfig | None = None) -> dict[str, Any]:
    """Run end-to-end recurrent PPO training with curriculum, asymmetric critic, aux & RND."""
    import torch

    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    proc_dir = cfg.procedural_dir or str(Path(cfg.checkpoint_dir) / "proc_wads")
    families = ("door_exit", "combat", "multi_room", "key_return", "gauntlet")
    corpus_sampler: CorpusPoolSampler | None = None
    archive: AccelLevelArchive | None = None
    base_map_specs: list[MapSpec] = []
    hybrid_poly_specs: list[MapSpec] = []

    if cfg.corpus_manifest and not cfg.maps and cfg.stage != "connect":
        base_map_specs = resolve_train_maps(
            None,
            split=cfg.split,
            stage=cfg.stage,
            procedural_per_family=0,
        )
        corpus_sampler = CorpusPoolSampler(
            cfg.corpus_manifest,
            include_freedoom=cfg.corpus_include_freedoom,
        )
        sampled = corpus_sampler.sample_pool(
            pool_size=cfg.corpus_pool_size,
            high_diff_frac=cfg.corpus_high_diff_frac,
            rng=rng,
        )
        map_specs = [*base_map_specs, *sampled]
    else:
        use_accel_pool = (
            bool(cfg.use_accel)
            and int(cfg.procedural_per_family) > 0
            and not cfg.maps
            and cfg.stage != "connect"
        )
        if use_accel_pool:
            base_map_specs, accel_specs, hybrid_poly_specs, archive = _init_accel_pool(cfg, proc_dir)
            map_specs = [*base_map_specs, *accel_specs, *hybrid_poly_specs]
        else:
            map_specs = resolve_train_maps(
                list(cfg.maps) if cfg.maps else None,
                split=cfg.split,
                stage=cfg.stage,
                procedural_per_family=cfg.procedural_per_family,
                procedural_dir=proc_dir,
                procedural_seed=cfg.seed + 1000,
            )

    champion_baselines = (
        load_champion_baselines(cfg.champion_report) if cfg.champion_report else None
    )
    base_reward_cfg = (
        reward_config
        if reward_config is not None
        else RewardConfig(pbrs_scale=cfg.pbrs_scale, pbrs_gamma=1.0)
    )
    effective_reward_cfg = base_reward_cfg
    cmdp = PrimalDualController() if cfg.use_primal_dual else None
    cmdp_metrics: dict[str, float] = cmdp.current_metrics() if cmdp is not None else {}

    scheduler = CurriculumScheduler(
        CurriculumConfig(
            auto_curriculum=cfg.auto_curriculum,
            tier_weights=cfg.tier_weights,
            window_size=cfg.promote_window_size,
            min_episodes_to_promote=cfg.min_promote_episodes,
            tier12_exit_promote_threshold=cfg.tier12_promote_threshold,
            tier3_progress_promote_threshold=cfg.tier3_promote_threshold,
            tier3_require_exit=cfg.tier3_require_exit,
            adaptive_weights=cfg.adaptive_weights,
            adaptive_mode=cfg.adaptive_mode,
        )
    )

    ppo_cfg = PPOConfig(
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_coef=cfg.clip_coef,
        clip_vloss=cfg.clip_vloss,
        use_huber_vloss=cfg.use_huber_vloss,
        huber_delta=cfg.huber_delta,
        vf_coef=cfg.vf_coef,
        ent_coef=cfg.ent_coef,
        max_grad_norm=cfg.max_grad_norm,
        update_epochs=cfg.update_epochs,
        chunk_len=cfg.chunk_len,
        num_minibatches=cfg.num_minibatches,
        target_kl=cfg.target_kl,
        use_asymmetric_critic=cfg.use_asymmetric_critic,
        aux_coef=cfg.aux_coef,
        curiosity_coef=cfg.curiosity_coef,
    )

    start_update = 0
    total_decisions = 0
    episodes_completed = 0
    if cfg.resume:
        policy, policy_cfg, optimizer, step_info, extras = load_training_state(
            cfg.resume, device=device, return_extras=True
        )
        start_update = int(step_info.get("updates", 0))
        total_decisions = int(step_info.get("total_decisions", 0))
        episodes_completed = int(step_info.get("episodes_completed", 0))
        scheduler.phase = int(step_info.get("curriculum_phase", 0))
        trainer = RecurrentPPOTrainer(
            policy,
            ppo_cfg,
            device=device,
            optimizer=optimizer,
            asymmetric_critic=extras.get("asymmetric_critic"),
            aux_head=extras.get("aux_head"),
            curiosity=extras.get("curiosity"),
        )
    else:
        policy_cfg = PolicyConfig(
            arch=cfg.arch,
            action_head=cfg.action_head,
            hidden_size=cfg.hidden_size,
            ray_channels=cfg.ray_channels,
            entity_dim=cfg.entity_dim,
            vitals_dim=cfg.vitals_dim,
            num_memory_slots=cfg.num_memory_slots,
            use_ego_map=cfg.use_ego_map,
            ego_map_out_dim=cfg.ego_map_out_dim,
            eval_deterministic=cfg.eval_deterministic,
        )
        policy = build_policy(policy_cfg).to(device)
        trainer = RecurrentPPOTrainer(policy, ppo_cfg, device=device)

    initial_weights = scheduler.compute_map_weights(map_specs)

    use_ego_map = bool(getattr(policy_cfg, "use_ego_map", False))
    steps_per_update = cfg.rollout_steps * cfg.num_envs
    target_updates = max(1, (cfg.total_steps + steps_per_update - 1) // steps_per_update)

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"
    resume_path = ckpt_dir / "latest_train_state.pt"

    all_recent_episodes: list[EpisodeMetrics] = []
    last_summary: dict[str, float] = {}
    t_start = time.perf_counter()
    last_save = t_start
    finished_update = start_update

    try:
        with _make_vec_env(
            map_specs,
            num_envs=cfg.num_envs,
            vec_mode=cfg.vec_mode,
            seed=cfg.seed,
            render_buffers=cfg.render_buffers,
            reward_config=effective_reward_cfg,
            map_weights=initial_weights,
            max_cached_envs=cfg.max_cached_envs,
            use_ego_map=use_ego_map,
            mirror_prob=cfg.mirror_prob,
            champion_baselines=champion_baselines,
        ) as vec_env:
            obs = vec_env.reset()
            prev_action = np.zeros((cfg.num_envs, NUM_ACTIONS), dtype=np.float32)
            hidden = policy.init_hidden(cfg.num_envs, device)
            episode_starts = np.ones(cfg.num_envs, dtype=bool)
            hidden_shape = tuple(hidden.shape)
            buffer = RolloutBuffer(
                cfg.rollout_steps,
                cfg.num_envs,
                hidden_shape=hidden_shape,
                use_ego_map=use_ego_map,
            )

            for local_idx in range(target_updates):
                update_idx = start_update + local_idx + 1
                frac_done = float(local_idx) / float(max(1, target_updates - 1)) if target_updates > 1 else 0.0
                if cfg.anneal_lr and target_updates > 1:
                    lr_frac = max(cfg.min_lr_frac, 1.0 - (1.0 - cfg.min_lr_frac) * frac_done)
                    trainer.set_lr(cfg.lr * lr_frac)
                if target_updates > 1:
                    cur_ent = cfg.ent_coef + (cfg.ent_coef_end - cfg.ent_coef) * frac_done
                    trainer.set_ent_coef(cur_ent)

                t_iter = time.perf_counter()
                obs, prev_action, hidden, episode_starts, ep_list, last_values = collect_rollout(
                    vec_env=vec_env,
                    policy=policy,
                    buffer=buffer,
                    obs=obs,
                    prev_action=prev_action,
                    hidden=hidden,
                    episode_starts=episode_starts,
                    device=device,
                    asymmetric_critic=trainer.asymmetric_critic,
                    feature_hook=trainer.feature_hook,
                    curiosity=trainer.curiosity,
                )
                buffer.compute_returns_and_advantages(
                    last_values=last_values,
                    gamma=cfg.gamma,
                    gae_lambda=cfg.gae_lambda,
                )
                ppo_metrics = trainer.update(buffer, rng=rng)
                iter_s = max(1e-6, time.perf_counter() - t_iter)

                scheduler.observe_pvl(buffer.compute_per_wad_pvl())
                if ep_list:
                    promoted = scheduler.observe_episodes(ep_list)
                    if cmdp is not None:
                        effective_reward_cfg, cmdp_metrics = cmdp.observe_episodes(ep_list, base_reward_cfg)
                        vec_env.set_reward_config(effective_reward_cfg)
                    if archive is not None:
                        archive.update_scores(scheduler.get_map_scores(map_specs))
                    if promoted or cfg.adaptive_weights:
                        new_weights = scheduler.compute_map_weights(map_specs)
                        vec_env.set_map_weights(new_weights)
                    if promoted:
                        print(f"[curriculum] Promoted to phase {scheduler.phase} (tier weights={scheduler.current_tier_weights()})", flush=True)

                if (
                    cfg.procedural_refresh_updates > 0
                    and (corpus_sampler is not None or cfg.procedural_per_family > 0)
                    and not cfg.maps
                    and cfg.stage != "connect"
                    and (local_idx + 1) < target_updates
                    and update_idx % cfg.procedural_refresh_updates == 0
                ):
                    refresh_seed = cfg.seed + 1000 + update_idx * 97
                    retired_poly_paths: list[Path] = []
                    if corpus_sampler is not None:
                        sampled = corpus_sampler.sample_pool(
                            pool_size=cfg.corpus_pool_size,
                            high_diff_frac=cfg.corpus_high_diff_frac,
                            rng=rng,
                            scores_by_wad=scheduler.get_map_scores(map_specs),
                        )
                        map_specs = [*base_map_specs, *sampled]
                    elif archive is not None:
                        k_total = int(cfg.procedural_per_family)
                        k_accel = max(1, k_total // 2) if k_total >= 2 else k_total
                        k_poly = max(0, k_total - k_accel)
                        accel_specs = archive.evolve_pool(
                            count_per_family=k_accel,
                            families=families,
                            out_dir=proc_dir,
                            rng=rng,
                            generation_seed=refresh_seed,
                            genome_cls=DungeonGenome,
                            generator_fn=generate_dungeon_from_genome,
                            mutate_fn=mutate_dungeon_genome,
                            defer_unlink=True,
                        )
                        retired_poly_paths = [Path(s.wad) for s in hybrid_poly_specs]
                        hybrid_poly_specs = _generate_hybrid_poly_pool(
                            k_poly, families, Path(proc_dir), base_seed=refresh_seed + 50_000
                        )
                        map_specs = [*base_map_specs, *accel_specs, *hybrid_poly_specs]
                    else:
                        map_specs = resolve_train_maps(
                            None,
                            split=cfg.split,
                            stage=cfg.stage,
                            procedural_per_family=cfg.procedural_per_family,
                            procedural_dir=proc_dir,
                            procedural_seed=refresh_seed,
                        )
                    refreshed_weights = scheduler.compute_map_weights(map_specs)
                    vec_env.set_map_specs(map_specs, refreshed_weights)
                    if archive is not None:
                        archive.cleanup_retired()
                    for rp in retired_poly_paths:
                        if rp.exists() and rp.name.startswith("proc_poly_"):
                            try:
                                rp.unlink()
                            except OSError:
                                pass

                total_decisions += steps_per_update
                episodes_completed += len(ep_list)
                all_recent_episodes.extend(ep_list)
                recent_window = all_recent_episodes[-max(20, cfg.num_envs * 4) :]
                ep_summary = _summarize_episodes(recent_window)
                sps = steps_per_update / iter_s

                record = {
                    "update": update_idx,
                    "total_decisions": total_decisions,
                    "episodes_completed": episodes_completed,
                    "decisions_per_sec": round(sps, 1),
                    "curriculum_phase": scheduler.phase,
                    **ppo_metrics,
                    **ep_summary,
                    **cmdp_metrics,
                }
                last_summary = record
                with metrics_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")

                if update_idx % max(1, cfg.log_interval) == 0:
                    print(
                        f"[update {update_idx}/{start_update + target_updates}] "
                        f"steps={total_decisions} sps={sps:.0f} p={scheduler.phase} "
                        f"eps={episodes_completed} exit={ep_summary['exit_rate']:.2f} "
                        f"door={ep_summary['door_open_rate']:.2f} key={ep_summary['key_rate']:.2f} "
                        f"unlock={ep_summary['unlock_rate']:.2f} rew={ep_summary['mean_reward']:.2f} "
                        f"aux={ppo_metrics['aux_loss']:.3f} rnd={ppo_metrics['rnd_loss']:.3f} "
                        f"kl={ppo_metrics['approx_kl']:.4f}",
                        flush=True,
                    )

                finished_update = update_idx
                now = time.perf_counter()
                timed_save = cfg.save_every_seconds > 0 and now - last_save >= cfg.save_every_seconds
                interval_save = cfg.save_interval > 0 and update_idx % cfg.save_interval == 0
                if timed_save or interval_save:
                    save_training_state(
                        resume_path,
                        policy,
                        policy_cfg,
                        trainer.optimizer,
                        ppo_cfg,
                        {
                            "updates": update_idx,
                            "total_decisions": total_decisions,
                            "episodes_completed": episodes_completed,
                            "curriculum_phase": scheduler.phase,
                        },
                        asymmetric_critic=trainer.asymmetric_critic,
                        aux_head=trainer.aux_head,
                        curiosity=trainer.curiosity,
                    )
                    if cfg.export_path:
                        plain_meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}
                        save_checkpoint(cfg.export_path, policy, policy_cfg, meta=plain_meta)
                    if timed_save:
                        snapshots = ckpt_dir / f"update_{update_idx:07d}"
                        snapshots.mkdir(exist_ok=True)
                        shutil.copy2(resume_path, snapshots / "train_state.pt")
                        if cfg.export_path:
                            shutil.copy2(cfg.export_path, snapshots / "policy.pt")
                        last_save += cfg.save_every_seconds
                if cfg.max_train_seconds > 0 and time.perf_counter() - t_start >= cfg.max_train_seconds:
                    break

            save_training_state(
                resume_path,
                policy,
                policy_cfg,
                trainer.optimizer,
                ppo_cfg,
                {
                    "updates": finished_update,
                    "total_decisions": total_decisions,
                    "episodes_completed": episodes_completed,
                    "curriculum_phase": scheduler.phase,
                },
                asymmetric_critic=trainer.asymmetric_critic,
                aux_head=trainer.aux_head,
                curiosity=trainer.curiosity,
            )
            export_out = None
            if cfg.export_path:
                plain_meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in last_summary.items()}
                export_out = str(save_checkpoint(cfg.export_path, policy, policy_cfg, meta=plain_meta))
    finally:
        trainer.close()

    elapsed_s = max(1e-6, time.perf_counter() - t_start)
    overall_sps = ((finished_update - start_update) * steps_per_update) / elapsed_s
    return {
        "device": device,
        "arch": policy_cfg.arch,
        "num_maps": len(map_specs),
        "curriculum_phase": scheduler.phase,
        "total_decisions": total_decisions,
        "optimizer_updates": finished_update,
        "episodes_completed": episodes_completed,
        "elapsed_s": elapsed_s,
        "decisions_per_sec": overall_sps,
        "export_path": export_out,
        "resume_path": str(resume_path),
        "final_metrics": last_summary,
    }


def bench_throughput(
    map_specs: list[MapSpec],
    worker_counts: list[int],
    rollout_steps: int = 128,
    arch: str = "telemetry_gru_v1",
    hidden_size: int = 256,
    device: str = "auto",
    vec_mode: str = "subproc",
    use_ego_map: bool = True,
) -> list[dict[str, Any]]:
    """Measure end-to-end decisions/sec (env rollout + PPO update) across worker counts."""
    import torch

    dev = resolve_device(device)
    policy_cfg = PolicyConfig(arch=arch, hidden_size=hidden_size, use_ego_map=use_ego_map)
    results = []
    for n_workers in worker_counts:
        policy = build_policy(policy_cfg).to(dev)
        ppo_cfg = PPOConfig(chunk_len=min(64, rollout_steps), num_minibatches=min(4, max(1, n_workers)))
        trainer = RecurrentPPOTrainer(policy, ppo_cfg, device=dev)
        rng = np.random.default_rng(0)

        try:
            with _make_vec_env(
                map_specs,
                num_envs=n_workers,
                vec_mode=vec_mode,
                seed=0,
                use_ego_map=use_ego_map,
            ) as vec_env:
                obs = vec_env.reset()
                prev_action = np.zeros((n_workers, NUM_ACTIONS), dtype=np.float32)
                hidden = policy.init_hidden(n_workers, dev)
                episode_starts = np.ones(n_workers, dtype=bool)
                buf = RolloutBuffer(
                    rollout_steps,
                    n_workers,
                    tuple(hidden.shape),
                    use_ego_map=use_ego_map,
                )

                obs, prev_action, hidden, episode_starts, _, last_v = collect_rollout(
                    vec_env, policy, buf, obs, prev_action, hidden, episode_starts, dev,
                    asymmetric_critic=trainer.asymmetric_critic,
                    feature_hook=trainer.feature_hook,
                    curiosity=trainer.curiosity,
                )
                buf.compute_returns_and_advantages(last_v, gamma=ppo_cfg.gamma)
                trainer.update(buf, rng=rng)
                if dev == "cuda":
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                n_iters = 2
                for _ in range(n_iters):
                    obs, prev_action, hidden, episode_starts, _, last_v = collect_rollout(
                        vec_env, policy, buf, obs, prev_action, hidden, episode_starts, dev,
                        asymmetric_critic=trainer.asymmetric_critic,
                        feature_hook=trainer.feature_hook,
                        curiosity=trainer.curiosity,
                    )
                    buf.compute_returns_and_advantages(last_v, gamma=ppo_cfg.gamma)
                    trainer.update(buf, rng=rng)
                if dev == "cuda":
                    torch.cuda.synchronize()
                elapsed = max(1e-6, time.perf_counter() - t0)
        finally:
            trainer.close()

        decisions = n_iters * rollout_steps * n_workers
        dps = decisions / elapsed
        entry = {
            "num_workers": n_workers,
            "rollout_steps": rollout_steps,
            "decisions": decisions,
            "elapsed_s": round(elapsed, 3),
            "decisions_per_sec": round(dps, 1),
            "device": dev,
            "vec_mode": vec_mode,
        }
        print(
            f"[bench workers={n_workers}] {decisions} decisions in {elapsed:.2f}s -> {dps:.1f} dec/s ({dev})",
            flush=True,
        )
        results.append(entry)
    return results


def verify_checkpoint_two_maps(
    checkpoint_path: str | Path,
    map_specs: list[MapSpec],
    device: str = "cpu",
    deterministic: bool = True,
) -> list[EpisodeMetrics]:
    """Reload an exported checkpoint through `DoomAgent` and play two consecutive maps with reset()."""
    from doom_env import make_env
    from doom_env.spaces import action_space as make_action_space
    from doom_env.spaces import observation_space as make_observation_space
    from submission.agent import DoomAgent
    from training.rollout import _resolve_wad_path

    if not map_specs:
        raise ValueError("map_specs must not be empty")

    prev_override = os.environ.get("DOOM_AGENT_WEIGHTS")
    os.environ["DOOM_AGENT_WEIGHTS"] = str(Path(checkpoint_path).resolve())
    try:
        agent = DoomAgent(make_observation_space(), make_action_space(), device=device)
        if hasattr(agent, "deterministic"):
            agent.deterministic = deterministic
    finally:
        if prev_override is None:
            os.environ.pop("DOOM_AGENT_WEIGHTS", None)
        else:
            os.environ["DOOM_AGENT_WEIGHTS"] = prev_override

    chosen = [map_specs[0], map_specs[1 % len(map_specs)]]
    outcomes: list[EpisodeMetrics] = []

    for idx, spec in enumerate(chosen):
        env = make_env(_resolve_wad_path(spec.wad), spec.map_name, episode_ticks=spec.episode_ticks, render_buffers=False)
        shaper = RewardShaper()
        try:
            raw_obs, _ = env.reset(seed=idx)
            agent.reset()
            shaper.reset(env.privileged_state(), obs=raw_obs)
            prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
            steps = keys = doors = unlocks = kills = 0
            dmg = total_rew = 0.0
            t0 = time.perf_counter()

            while True:
                act_np = np.asarray(agent.act(raw_obs), dtype=np.int8)
                raw_obs, _, term, trunc, _ = env.step(act_np)
                priv = env.privileged_state()
                rew, ev, _ = shaper.step(priv, action=act_np, prev_action=prev_action, obs=raw_obs)
                prev_action = act_np
                steps += 1
                total_rew += rew
                keys += ev.keys_picked_up
                doors += ev.doors_opened
                unlocks += ev.locked_doors_unlocked
                kills += ev.kills
                dmg += ev.damage_taken
                if term or trunc:
                    outcome = "exit" if priv["exit_reached"] else ("death" if priv["dead"] else "timeout")
                    outcomes.append(
                        EpisodeMetrics(
                            wad=spec.wad,
                            map_name=spec.map_name,
                            family=spec.family,
                            tier=spec.tier,
                            seed=idx,
                            steps=steps,
                            ticks=int(priv["tic"]),
                            outcome=outcome,
                            keys_picked_up=keys,
                            doors_opened=doors,
                            locked_doors_unlocked=unlocks,
                            kills=kills,
                            damage_taken=float(dmg),
                            final_health=float(priv["health"]),
                            unique_cells_visited=shaper.unique_cells_visited,
                            total_reward=float(total_rew),
                            agent_time_s=float(time.perf_counter() - t0),
                        )
                    )
                    break
        finally:
            env.close()
    return outcomes


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Role 2 Recurrent PPO Trainer & Throughput Sweeper")
    p.add_argument("--arch", default="telemetry_gru_v1")
    p.add_argument("--action-head", choices=("factorized", "bernoulli"), default="factorized")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--ray-channels", type=int, default=32)
    p.add_argument("--entity-dim", type=int, default=64)
    p.add_argument("--vitals-dim", type=int, default=64)
    p.add_argument("--num-memory-slots", type=int, default=0)
    p.add_argument("--use-ego-map", dest="use_ego_map", action="store_true", default=True)
    p.add_argument("--no-use-ego-map", dest="use_ego_map", action="store_false")
    p.add_argument("--ego-map-out-dim", type=int, default=128)
    p.add_argument("--eval-deterministic", action="store_true", default=False)
    p.add_argument("--map", dest="maps", action="append", default=[])
    p.add_argument("--split", default="train")
    p.add_argument("--stage", default=None)
    p.add_argument("--procedural-per-family", type=int, default=4)
    p.add_argument("--procedural-dir", default=None)
    p.add_argument("--procedural-refresh-updates", type=int, default=20)
    p.add_argument("--tier-weights", default=None, help="Comma-separated tier:weight, e.g. 1:0.4,2:0.3,3:0.2,4:0.1")
    p.add_argument("--auto-curriculum", dest="auto_curriculum", action="store_true", default=True)
    p.add_argument("--no-auto-curriculum", dest="auto_curriculum", action="store_false")
    p.add_argument("--adaptive-weights", dest="adaptive_weights", action="store_true", default=True)
    p.add_argument("--no-adaptive-weights", dest="adaptive_weights", action="store_false")
    p.add_argument("--adaptive-mode", choices=("legacy", "sfl_pvl"), default="sfl_pvl")
    p.add_argument("--accel", dest="use_accel", action="store_true", default=True)
    p.add_argument("--no-accel", dest="use_accel", action="store_false")
    p.add_argument("--primal-dual", dest="use_primal_dual", action="store_true", default=True)
    p.add_argument("--no-primal-dual", dest="use_primal_dual", action="store_false")
    p.add_argument("--champion-report", default=None, help="Optional eval JSON path for champion-anchored hill-climb rewards")
    p.add_argument("--corpus-manifest", default=None, help="Path to pre-generated corpus manifest.jsonl for rotating map streaming")
    p.add_argument("--corpus-pool-size", type=int, default=60)
    p.add_argument("--corpus-high-diff-frac", type=float, default=0.45)
    p.add_argument("--corpus-include-freedoom", dest="corpus_include_freedoom", action="store_true", default=True)
    p.add_argument("--no-corpus-include-freedoom", dest="corpus_include_freedoom", action="store_false")
    p.add_argument("--tier12-promote-threshold", type=float, default=0.65)
    p.add_argument("--tier3-promote-threshold", type=float, default=0.50)
    p.add_argument("--min-promote-episodes", type=int, default=32)
    p.add_argument("--promote-window-size", type=int, default=48)
    p.add_argument("--asymmetric-critic", dest="use_asymmetric_critic", action="store_true", default=True)
    p.add_argument("--no-asymmetric-critic", dest="use_asymmetric_critic", action="store_false")
    p.add_argument("--aux-coef", type=float, default=0.2)
    p.add_argument("--curiosity-coef", type=float, default=0.0)
    p.add_argument("--pbrs-scale", type=float, default=3.0)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--vec-mode", choices=("subproc", "sync"), default="subproc")
    p.add_argument("--max-cached-envs", type=int, default=48)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--chunk-len", type=int, default=64)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.25)
    p.add_argument("--ent-coef", type=float, default=0.015)
    p.add_argument("--ent-coef-end", type=float, default=0.004)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-vloss", dest="clip_vloss", action="store_true", default=True)
    p.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")
    p.add_argument("--huber-vloss", dest="use_huber_vloss", action="store_true", default=True)
    p.add_argument("--no-huber-vloss", dest="use_huber_vloss", action="store_false")
    p.add_argument("--huber-delta", type=float, default=5.0)
    p.add_argument("--mirror-prob", type=float, default=0.5)
    p.add_argument("--no-anneal-lr", action="store_true")
    p.add_argument("--checkpoint-dir", default="training/runs/default")
    p.add_argument("--export-path", default="submission/weights/policy.pt")
    p.add_argument("--resume", default=None)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--save-every-seconds", type=float, default=0.0,
                   help="Archive policy and training state at the next completed PPO update")
    p.add_argument("--max-train-seconds", type=float, default=0.0,
                   help="Finish and export at the first completed update after this wall time")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--bench-workers", default=None, help="Comma-separated worker counts, e.g. 1,4,8,12")
    p.add_argument("--verify-two-maps", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.bench_workers:
        workers = [int(x.strip()) for x in args.bench_workers.split(",") if x.strip()]
        specs = resolve_train_maps(
            args.maps or None,
            split=args.split,
            stage=args.stage,
            procedural_per_family=args.procedural_per_family,
            procedural_dir=args.procedural_dir,
        )
        res = bench_throughput(
            map_specs=specs,
            worker_counts=workers,
            rollout_steps=args.rollout_steps,
            arch=args.arch,
            hidden_size=args.hidden_size,
            device=args.device,
            vec_mode=args.vec_mode,
            use_ego_map=args.use_ego_map,
        )
        out_dir = Path(args.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "throughput_sweep.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        return

    cfg = TrainConfig(
        arch=args.arch,
        action_head=args.action_head,
        hidden_size=args.hidden_size,
        ray_channels=args.ray_channels,
        entity_dim=args.entity_dim,
        vitals_dim=args.vitals_dim,
        num_memory_slots=args.num_memory_slots,
        use_ego_map=args.use_ego_map,
        ego_map_out_dim=args.ego_map_out_dim,
        eval_deterministic=args.eval_deterministic,
        split=args.split,
        stage=args.stage,
        procedural_per_family=args.procedural_per_family,
        procedural_dir=args.procedural_dir,
        procedural_refresh_updates=args.procedural_refresh_updates,
        tier_weights=_parse_tier_weights(args.tier_weights),
        auto_curriculum=args.auto_curriculum,
        adaptive_weights=args.adaptive_weights,
        adaptive_mode=args.adaptive_mode,
        use_accel=args.use_accel,
        use_primal_dual=args.use_primal_dual,
        champion_report=args.champion_report,
        corpus_manifest=args.corpus_manifest,
        corpus_pool_size=args.corpus_pool_size,
        corpus_high_diff_frac=args.corpus_high_diff_frac,
        corpus_include_freedoom=args.corpus_include_freedoom,
        tier12_promote_threshold=args.tier12_promote_threshold,
        tier3_promote_threshold=args.tier3_promote_threshold,
        min_promote_episodes=args.min_promote_episodes,
        promote_window_size=args.promote_window_size,
        maps=tuple(args.maps),
        num_envs=args.num_envs,
        vec_mode=args.vec_mode,
        max_cached_envs=args.max_cached_envs,
        rollout_steps=args.rollout_steps,
        chunk_len=args.chunk_len,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        total_steps=args.total_steps,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        min_lr_frac=args.min_lr_frac,
        ent_coef=args.ent_coef,
        ent_coef_end=args.ent_coef_end,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_vloss=args.clip_vloss,
        use_huber_vloss=args.use_huber_vloss,
        huber_delta=args.huber_delta,
        mirror_prob=args.mirror_prob,
        use_asymmetric_critic=args.use_asymmetric_critic,
        aux_coef=args.aux_coef,
        curiosity_coef=args.curiosity_coef,
        pbrs_scale=args.pbrs_scale,
        anneal_lr=not args.no_anneal_lr,
        checkpoint_dir=args.checkpoint_dir,
        export_path=args.export_path,
        resume=args.resume,
        save_interval=args.save_interval,
        save_every_seconds=args.save_every_seconds,
        max_train_seconds=args.max_train_seconds,
        log_interval=args.log_interval,
    )
    summary = run_training(cfg)
    print("TRAIN_SUMMARY:", json.dumps(summary, default=str), flush=True)

    if args.verify_two_maps and summary["export_path"]:
        specs = resolve_train_maps(args.maps or None, split=args.split, stage=args.stage)
        eps = verify_checkpoint_two_maps(summary["export_path"], specs, device=summary["device"])
        print("TWO_MAP_VERIFY:", json.dumps([asdict(e) for e in eps]), flush=True)


if __name__ == "__main__":
    main()
