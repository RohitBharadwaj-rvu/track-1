# Track 1 improvements audit (sync 2)

*Uploaded: 2026-11-26 20:44 IST (`improvements/2026-11-26_20-44/`)*

Follow-up audit after the team merged Roles 1 to 4 into `master` (`07de9db`), landed the pre-MVP upgrades from `improvements/2026-11-26_17-42/`, and opened two memory-challenger branches (`challenger1/landmark-memory` and `hill-climb/ego-memory-v2`).

---

## 1. Current GitHub status across all 7 branches

| Branch | Latest commit | Status |
|---|---|---|
| `master` | `07de9db` | Roles 1 to 4 merged. Has factorized actions, combat kinematics, category counts, ammo/health flags, extended BPTT (`gamma=0.995`), asymmetric critic, aux door-bearing head, curiosity, curriculum scheduler, rebalanced rewards, Role 3 WAD suites, and Role 4 evaluator + benchmarks. **Still holds smoke/proof weights only.** |
| `challenger1/landmark-memory` | `df09f47` | Adds `telemetry_landmark_v1` (`LandmarkMemoryModule` with 2D ego-odometry + learned turn rate + salience cues), multi-lock aux tracking, Huber value loss (`huber_delta=5.0`), entropy annealing (`--ent-coef-end`), and `--disable-memory-read` ablation toggle. Branched before `07de9db` (drops `docs/BENCHMARK_COMPARISON.md`). |
| `hill-climb/ego-memory-v2` | `6593bb3` | Adds `EgocentricSpatialMap` (`submission/ego_map.py`, 8-channel `25x25` odometric grid + 2-iter wall-ray ICP + `SpatialMapEncoder` 2D CNN), geodesic PBRS (`training/pbrs.py`), adaptive per-map curriculum weights, expanded `procedural.py`, and Colab A100 notebook updates. |
| `role4/eval-release` | `2c454bf` | Merged into `master` (`07de9db`). Added `docs/BENCHMARK_COMPARISON.md`, `training/brief_tools.py`, and Linux+CUDA `validate.py` numbers (`156 MB` VRAM, `4.0-4.8 ms` p50). |
| `role3/maps-curriculum` | `66e43fc` | Reconciled into `master` (`07de9db`). |
| `role2/ppo-pipeline` | `3a48f8d` | Behind `master` on Role 1/3/4 merges; records completed Role 2 Colab T4 verification run (`77 passed`, `325-417 dec/s`, `100%` exit on `#P..D...E#`). |
| `role1/telemetry-policy` | `624f321` | Behind `master` (pre-factorized/pre-Role 3/4 merge). |

---

## 2. What is now solved vs. what is still blocking a winning submission

### Solved since `17:42`
- All 4 roles integrated on `master` with Role 3's Train (20), Dev (10), and Final (8) UDMF WADs and procedural generator.
- Factorized action heads, combat kinematics, threat counts, `selected_ammo_empty` / `low_health` flags, higher exploration cell cap (`250`), post-key cell reset, `USE`/door-bounce and ammo-waste penalties, asymmetric critic, aux door-bearing loss, and `gamma=0.995` are all implemented.
- Sandbox footprint verified on Linux + CUDA (`156 MB / 2,048 MB` VRAM, `1,380 MB / 8,192 MB` RAM, `4.0-4.8 ms` p50 latency on 1 CPU thread).

### Biggest remaining blocker: zero full-curriculum trained weights on any branch
- Every branch still has smoke/fixture weights in `submission/weights/policy.pt`, and `docs/BRIEF.md` still has `*TBD*` for Stage 2 MVP and Final Shipped Agent on the 10 held-out `DEV_MAPS`.
- You now have three competing architectures (`master` GRU, `challenger1/landmark-memory`, `hill-climb/ego-memory-v2`) before locking in a single trained `DEV_MAPS` checkpoint. Freeze a real trained MVP checkpoint on `DEV_MAPS` (`docs/eval_mvp_dev.json`) before adding any more features.

---

## 3. Concrete code-level issues to fix now

### 3.1 `hill-climb/ego-memory-v2` issues (partial observability, combat, and sandbox safety)

1. **Combat reward regression (`training/rewards.py`):**
   - `hill-climb` tripled `kill_reward` (`0.5 -> 1.5`) while cutting `damage_penalty` in half (`-0.035 -> -0.018`).
   - At those values, absorbing 50 HP of fireball damage (`-0.90`) to facetank one enemy (`+1.50`) is strongly net positive (`+0.60`), weakening Tier 1 and Tier 4 projectile dodging.
   - **Fix:** Restore `damage_penalty` to `-0.03` to `-0.035` (or cap `kill_reward <= 0.8`) and keep total combat reward secondary to survival and exit.
2. **Value-loss stability regression (`training/ppo.py`):**
   - `hill-climb` turned off `clip_vloss = False` without adding Huber loss, while also adding `pbrs_scale = 2.0` and raising `exit_reward = 12.0`. Large return spikes can destabilize the critic.
   - **Fix:** Cherry-pick `challenger1`'s Huber value loss (`use_huber_vloss = True`, `huber_delta = 5.0`) and `clip_vloss = True` onto `hill-climb`.
3. **1-CPU sandbox latency risk in `submission/ego_map.py`:**
   - Every `act()` step runs a 2-iteration `64x64` float64 nearest-neighbor ICP loop (`ego_map.py:165-175`), ray-marches `1,536` points, and extracts a rotated `25x25` crop on CPU. Evaluation pins the agent to **1 CPU thread**.
   - **Fix:** Benchmark `validate.py --docker` (1 CPU pinned) on `hill-climb` before committing training compute. If CPU latency rises near 15 to 20 ms/step, switch the ICP math to `float32` and subsample every 2nd valid wall ray (`32x32` instead of `64x64`, 4x faster).
4. **Backing grid edge clipping in `submission/ego_map.py`:**
   - `_BACKING_SIZE = 80`, `_BACKING_CELL = 48.0` covers only `+/-1,920` map units from spawn. Any larger Tier 3 or Tier 4 hidden map clips out-of-bounds wall/door/visited stamps onto row/col `0` or `79`.
   - **Fix:** Increase `_BACKING_SIZE` to `128` (`+/-3,072` units; only `512 KB` float32 RAM, zero extra crop cost).
5. **Verify `training/pbrs.py` coverage and non-oscillation:**
   - `pbrs.py` parses UDMF maps via `parse_udmf` and checks `_EXIT_SPECIALS = {11, 52, 243, 244}`. Verify `_build_nav_graph()` succeeds on 100% of Role 3's train and procedural WADs (silent `None` fallbacks create uneven reward scales across maps).
   - Watch Tier 1/4 combat replays to verify `pbrs_scale = 2.0` does not pull the agent straight into projectiles instead of strafing sideways.
   - Keep `training/pbrs.py` strictly under `training/` (never imported by `submission/`). In `docs/BRIEF.md`, state clearly that `submission/ego_map.py` is pure ego-centric sensory accumulation (no `A*`, no pathfinding, no global coordinates) feeding a trained CNN.

### 3.2 `challenger1/landmark-memory` issues (long-horizon training speed and merge hygiene)

1. **Python per-step BPTT loop bottleneck (`submission/policy.py`):**
   - `TelemetryLandmarkPolicy.evaluate_chunks()` loops `for t in range(t_steps):` in Python, calling single-step GRU + `memory.step()` sequentially. At `chunk_len = 64` or `128`, this launches hundreds of tiny GPU kernels per minibatch and can slow PPO updates 3x to 5x compared to fused `self.gru`.
   - **Fix:** Run `--bench-workers` on Colab A100/L4 comparing `telemetry_gru_v1` vs. `telemetry_landmark_v1` update time. If updates bottleneck rollout collection, test `torch.compile` on `memory.step` or keep `chunk_len = 64` with larger batch sizes.
2. **Preserve `docs/BENCHMARK_COMPARISON.md` on merge:**
   - `challenger1/landmark-memory` branched before `master` commit `07de9db`, so `git diff master..challenger1/landmark-memory` currently shows `docs/BENCHMARK_COMPARISON.md` deleted. Merge `origin/master` into `challenger1/landmark-memory` before opening a PR.
3. **Cherry-pick `challenger1`'s best training upgrades across branches:**
   - Multi-lock auxiliary target tracking (`_locked_doors_xy` list in `training/rollout.py` for Tier 4 two-lock maps), Huber value loss (`huber_delta = 5.0`), entropy annealing (`--ent-coef-end`), and `--disable-memory-read` diagnostic ablation are strict upgrades that benefit both `master` and `hill-climb`.

### 3.3 Remaining cross-branch gaps (all branches)

1. **Exit and switch sensory blind spot (partial observability):**
   - `env/doom_env/sensors.py` still has no `exit` or `switch` ray hit type or entity type, and collapses walls, ledges > 24 units, and openings < 56 units into `RAY_WALL`.
   - Confirm whether hidden maps can use wall-mounted exit switches or stairs. If any `dev` or `final` maps rely on switches or vertical ledges, test a tiny pooled `depth`/`rgb` feature branch within the `40 ms` budget (you currently use only `4.8 ms` p50 and `156 MB` VRAM).
2. **`docs/BRIEF.md` is stale compared to `master`:**
   - `docs/BRIEF.md` still describes `telemetry_gru_v1` as `687,098` params with *"9 independent binary button logits (`MultiBinary(9)`)"*, even though `master` now uses factorized action heads, combat kinematics, and category summaries.
   - Update `docs/BRIEF.md` (and export the final 2-page PDF via `training/brief_tools.py`) once real dev results land so judges do not see discrepancies between the PDF and `submission/policy.py`.
3. **Evaluation determinism sweep (`eval_mode`):**
   - Role 4 added `eval_mode` benchmarking to `training/eval.py`. Once your first full policy finishes training, run deterministic vs. low-temperature sampling sweeps on `DEV_MAPS` and save the winner in `PolicyConfig` before freezing weights.

---

## 4. Recommended execution plan from here

1. **Sync `challenger1/landmark-memory` and `hill-climb/ego-memory-v2` with `master` and apply the quick fixes above** (restore `damage_penalty` + Huber vloss + `_BACKING_SIZE=128` + 1-CPU `validate.py` check on `hill-climb`; merge `master` + benchmark update throughput on `challenger1`).
2. **Train and freeze a real Stage 2 MVP checkpoint first (`telemetry_gru_v1` on `master`)** on Colab A100 using Role 3's Train + procedural maps, then run:
   ```bash
   python -m training.eval --suite mvp_gate --check-mvp-gate docs/eval_baseline_dev.json
   python -m training.eval --suite dev --out docs/eval_mvp_dev.json --record-dir /tmp/replays --keep-key-replays
   python -m training.eval --compare docs/eval_baseline_dev.json docs/eval_mvp_dev.json
   ```
3. **Run fair head-to-head challenger comparisons on the exact same 10 `DEV_MAPS` (30 episodes):**
   - Incumbent: `telemetry_gru_v1` (`master`)
   - Challenger A: `telemetry_landmark_v1` (`challenger1/landmark-memory`), plus `--disable-memory-read` ablation for the 2-page brief
   - Challenger B: `ego-memory-v2` (`hill-climb/ego-memory-v2`) with PBRS and fixed combat rewards
4. **Promote the single highest-scoring Tier 3 to 4 winner to `master`:**
   - Commit final weights via Git LFS, run the untouched `FINAL_MAPS` suite once, update `docs/BRIEF.md` -> 2-page PDF, and pass `python -m training.release_check` plus `python validate.py --docker` from a fresh clone.
