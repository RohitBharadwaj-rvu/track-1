# Track 1 Project Brief: Telemetry-Driven Recurrent DOOM Agent

*Yugaantar 2026 · DevForge — Track 1 ("Can It Play DOOM?") | Shipped Checkpoint: Sprint 6B (`telemetry_gru_v1`, `884k` params, `3.55 MB` Git LFS)*

---

## Page 1: USP, Architecture, Representation & Spatial Memory

### 1. Unique Selling Proposition (USP)
Instead of relying on off-the-shelf Stable-Baselines3 pixel CNNs that overfit to wall textures and crowd the `2 GB` VRAM / `1 CPU` sandbox, we engineered a **100% custom PyTorch + NumPy stack** from scratch (`0` extra packages in `submission/requirements.txt`). Combining `doom_env` telemetry (`render_buffers=False`, `301.7` decisions/sec on A100) with a pure-NumPy odometric 2D spatial memory yields a compact `3.55 MB` agent using `<26%` of VRAM and `<19%` of the step-latency budget—while strictly obeying `docs/RULES.md` (all actions from trained neural weights; zero `if/else` action trees, zero inference pathfinding, zero aimbots, untouched `env/`).

### 2. System Architecture & Inference Pipeline (`submission/`)
`DoomAgent.__init__()` loads `submission/weights/policy.pt` (`eval_deterministic=True`, `rocket_safety_mask=True`) and warms both populated and empty-room paths to eliminate first-step CUDA jitter. `reset()` clears the GRU state, spatial grid, and `prev_action` between maps. Each step runs: `obs.py` telemetry scaling + `EgocentricSpatialMap` odometry → four parallel encoders → `Fusion (448→256)` → `Residual GRU (256)` → `Factorized Actor` + observation-only rocket mask → `MultiBinary(9)`. All training scaffolding (`AsymmetricCritic`, `AuxMemoryHead`, `GeodesicPBRS`, `RND`) is stripped at export.

### 3. Sensory & State Representation (`submission/obs.py`, `submission/policy.py`)
- **`rays` (64 CCW rays, 360°, 1024u):** Normalized distance `[d, 1 − d]` plus 6-class one-hot hit type (`none, wall, door, locked_door, enemy, item`).
- **`visible_entities` (up to 32 slots):** 12-class learned embeddings plus 10 ego-frame combat features (distance, bearing `sin/cos`, relative velocity, closing/lateral speed, inverse time-to-impact, and crosshair alignment).
- **`vitals` (17) + `prev_action` (9):** Scaled health, armor, 8-way weapon one-hot, `log1p` ammo, 6 key flags, 3D velocity, and 4 explicit cues (`empty_ammo`, `other_ammo`, `low_health`, `speed`) paired with `prev_action` for wall-snag detection and ViZDoom rising-edge `USE` / weapon switching.
- **`rgb` & `depth`:** Intentionally excluded to prevent texture overfitting and maximize training speed; residual switch/nukage aliasing is addressed in our failure analysis.

### 4. Neural Architecture (`telemetry_gru_v1`, `884,133` params)
- **Circular 1D CNN (`79.6k`):** Three circular-padded 1D convolutions over the 64 rays preserve 360° directional continuity without wraparound seams.
- **Variable-Enemy Masked Cross-Attention (`98.2k`):** Handles `0..32` arbitrary enemies permutation-invariantly via 4-head, 4-query attention conditioned on vitals and rays (with NaN-safe masking, per-category threat/item counts, and a learned empty-room embedding).
- **Factorized Actor (`34.4k`) & Rocket Mask:** Three 3-way categorical heads (move, strafe, turn — guaranteeing `0` conflicting buttons) plus three Bernoullis (`ATTACK`, `USE`, `SELECT_NEXT_WEAPON`) and an observation-only mask suppressing point-blank RocketLauncher (`wpn=5`) self-kills.

### 5. Spatial Memory Design (`submission/ego_map.py` + Residual GRU)
- **Odometric 2D Grid (`EgocentricSpatialMap`):** Pure-NumPy `128×128` grid (`48u/cell`) built solely from legal observations and `prev_action`. Integrates Doom's turn-hold physics and velocity refined with 2-step wall-ray ICP to reduce wall-slide drift.
- **Key-Return Channels & Perimeter Beacons:** Maintains 8 semantic channels—including a **post-key visited trail** that resets on key pickup to drive backtracking, and **locked doors** protected from erasure until a key is held. Extracts an ego-aligned `(8, 25, 25)` crop into a 3-layer CNN (`121.3k`), projecting off-crop locked doors and keys onto the crop border so distant wings remain reachable.
- **Residual GRU (`394.8k`) + Aux Supervision:** `LayerNorm(GRU(fused) + fused)` tracks short-term combat and button-edge timing; during training, `AuxMemoryHead` supervises nearest matching locked-door bearing (`100%` sighting accuracy).

---

## Page 2: Training Pipeline, Benchmarks, Compliance & Failure Analysis

### 6. Training & Exploration Pipeline (`training/`)
- **Custom Recurrent PPO:** Built from scratch in PyTorch (`chunk_len=64`, episode-boundary hidden masking, GAE `γ=0.995`, clipped Huber value loss, `bf16` AMP, and `50%` left-right mirror symmetry), trained for `3.26M` decisions (`1,439` updates, `9,122` episodes).
- **Privileged Shaping (training-only):** `AsymmetricCritic` conditioned on 16 privileged progress features (`0.86` explained variance) plus UDMF/binary WAD **Geodesic PBRS** computing shortest-path potentials across `spawn → key/switch → lock → exit`.
- **Anti-Stuck Exploration & Rewards:** Exit-dominant hierarchical rewards (`+12` exit + time/HP bonus, `+7` unlock, `+4` key, `+1.2` kill), diminishing cell novelty + **post-key exploration reset**, door-bounce / spin / `USE`-hold penalties, and optional RND curiosity.
- **5,000-Map Procedural Corpus & Curriculum:** Custom UDMF/dungeon generators (`two_key_join`, `remote_key_relay`, `revisit_wing`, compound gauntlets) plus 9 disjoint `freedoom2` training maps (`0` D4 geometric collisions with held-out suites), driven by a 3-phase auto-curriculum with SFL/PVL regret sampling, ACCEL mutation, and Do-No-Harm promotion gating.

### 1. Baseline vs. Final Outcomes (Held-Out Proxy Suites)
*Note: Official hidden maps and scoring weights are private; numbers below are local proxy evaluations on `submission/weights/policy.pt` (Sprint 6B, deterministic, `cbe1ca42c08d` → `06761f9a68f1`).*

| Candidate | Suite | Episodes | Exits | Keys Picked Up | Locked Doors Unlocked | Kills | Mean Cells Visited | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 0 Random (`submission/agent.py`) | 10 dev maps (30 eps) | 30 | 1 (3.3%) | 0 | 0 | 10 | 13.97 | 0 |
| Sprint 6B (`s6b_policy.pt`) | 10 dev maps (20 eps) | 20 | 19 (95.0%) | 10 | 10 | 16 | 42.35 | 0 |
| Sprint 6B (`s6b_policy.pt`) | 14 stress probes (28 eps) | 28 | 23 (82.1%) | 16 | 16 | 27 | 37.04 | 0 |
| Sprint 6B (`s6b_policy.pt`) | 5k Full-Dev (499 maps, s0) | 499 | 83 (16.6%) | 165 | 87 | 543 | 52.37 | 0 |
| Sprint 6B (`s6b_policy.pt`) | 5k Sealed-Eval (499 maps, s0) | 499 | 72 (14.4%) | 160 | 80 | 525 | 53.93 | 0 |

- **Highlights:** `0` deaths and `100%` `unlock|pickup` & `exit|unlock` on `dev` (`0.9792` proxy); `10/14` `stress` probes solved `100%`; `51.8%` `unlock|pickup` and `0` opposing-button steps across `998` unseen corpus maps (`max 8.84 ms/step`). On extreme OOD/retail suites, Sprint 6B scores `1/16` exits on `final_ood` (`32` kills) and `0/8` on `freedoom2 MAP01–08` (`31` kills, `89.4` cells).

### 2. Sandbox Compliance & Runtime (`validate.py`)
- **Linux + CUDA 1-CPU Isolated Check (`PASSED`):** `45.0 MB / 1.0 GB` repo (Git LFS `3.55 MB` weights), `0` extra packages, `2.6s / 60s` init, **`7.4 ms` p50** (`8.8 ms` p99) vs `40 ms` cap, `1.0s / 21.0s` budget (`two_rooms.wad` exit in `125` steps), **`1,478 MB / 8,192 MB` RAM**, **`532 MB / 2,048 MB` VRAM**.
- **Local Checks:** Windows RTX 3050 (`125 MB` VRAM, `16.6 ms` p50) and Windows 1-CPU (`8.9 ms` p50, `104`-step exit) both pass cleanly.

### 3. Failure Analysis & Mitigations (`SPRINT6B_FAILURE_ANALYSIS.md`)
- **Fixed — RocketLauncher Self-Kills:** Close-range splash deaths (`final_ood_t3_dropoff_skull`, steps 23–38) were eliminated by enabling observation-only `rocket_safety_mask=True` in `submission/policy.py` (`+1` exit on 3-seed `dev+final_ood`).
- **Fixed — Forgetting Prevention:** Post-6B fine-tuning showed harsh death penalties caused survival cowardice and weak KL caused drift; we added analytical teacher KL, Model Soup, and a strict Do-No-Harm promotion gate preserving Sprint 6B.
- **Known Limitation — Wall Switches & Nukage:** Telemetry rays tag wall switches as `RAY_WALL` (`0` `USE` presses over `1,575` steps on `exit_switch.wad`) and omit damaging floors; with `74%` VRAM unused, adding a tiny `30×40` `rgb`/`depth` CNN is our clear next upgrade.
