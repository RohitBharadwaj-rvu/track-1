# Track 1 Brief: Telemetry-Driven Recurrent DOOM Agent with Egocentric Spatial Memory

*Yugaantar DevForge Track 1 ("Can It Play DOOM?") | Submission Checkpoint: Sprint 6B (`s6b_policy.pt`, `884,133` params)*

---

## Page 1: USP, Architecture & Training Pipeline

### 1. Unique Selling Proposition & Telemetry Strategy
Our agent plays unseen `freedoom2.wad` (`skill 3`) levels zero-shot under strict sandbox caps (`2 GB VRAM`, `8 GB RAM`, `1 CPU thread`, `40 ms/step`). Rather than spending compute on heavy pixel CNNs that overfit to wall textures, we combine `doom_env`'s ego-centric telemetry (`rays`, `visible_entities`, `vitals`) with a pure-NumPy odometric 2D spatial memory (`EgocentricSpatialMap`) built strictly from legal observations and `prev_action`. Training with `render_buffers=False` reaches `300+` decisions/sec per worker while keeping inference under `156 MB` VRAM (`<8%` of cap) and well within the `40 ms` step budget.

### 2. Neural Architecture & Spatial Memory (`submission/policy.py`, `submission/ego_map.py`)
- **Circular Raycast Encoder:** Circular 1D convolutions over 64 ego-centric rays (normalized distance + 6-class hit type) preserve 360-degree left/right continuity without boundary wrap artifacts.
- **Combat-Aware Entity Attention:** Learned 12-class entity embeddings fused with ego-frame kinematics (closing speed, lateral speed, inverse time-to-impact, crosshair alignment) and per-category threat/item counts, pooled with 4-head, 4-query masked attention conditioned on vitals and ray context (with a deterministic empty-set fallback for empty rooms).
- **Egocentric Spatial Memory (`EgocentricSpatialMap`):** Integrates player velocity and Doom's turn-hold physics from `vitals` and `prev_action` (zero privileged coordinates, zero pathfinding) onto a `128x128` backing grid (`48u` cells). Renders an ego-aligned `(8, 25, 25)` crop (visited, post-key visited, walls, free space, doors, locked doors, items/keys, enemies) with perimeter beacons for off-crop landmarks, encoded by a compact CNN (`128`-dim output).
- **Temporal Memory & Factorized Control:** Residual GRU (`hidden_size=256`) fuses `[ray_emb, entity_emb, ego_map_emb, vitals_action_emb]` with explicit `selected_ammo_empty`, `has_any_other_ammo`, `low_health`, horizontal speed, and `prev_action` features so the policy respects ViZDoom's button-down edge semantics for `USE` and `SELECT_NEXT_WEAPON`. Three 3-way categorical heads (move, strafe, turn) and three Bernoulli heads (`ATTACK`, `USE`, `SELECT_NEXT_WEAPON`) map cleanly to `MultiBinary(9)` with zero opposing-button conflicts, paired with an observation-based close-range rocket safety mask (`rocket_safety_mask=True`).

### 3. Training & Exploration Pipeline (`training/`)
- **Custom Recurrent PPO:** Built from scratch in PyTorch (`gamma=0.995`, `chunk_len=64`, episode-boundary hidden masking, `3.26M` decisions / `1,439` updates for Sprint 6B) with a training-only asymmetric critic, auxiliary door-bearing supervision, RND intrinsic curiosity, potential-based reward shaping (PBRS), and left-right mirror symmetry augmentation.
- **Procedural Curriculum & Held-Out Splits:** Separate `train`, `dev` (10 maps), `stress` (14 targeted probes), `final_ood` (8 unseen polygon/verticality maps), and bundled `freedoom2.wad` (`MAP01..MAP08`) suites, plus a procedural progression generator covering two-key joins, remote relays, and multi-wing backtracking.

---

## Page 2: Evaluation Outcomes, Runtime & Failure Analysis

### 1. Baseline vs. Final Outcomes (Held-Out Proxy Suites)
*Note: Organizer hidden maps and exact weights are private; numbers below are local proxy evaluations on `submission/weights/policy.pt` (Sprint 6B, deterministic, source `cbe1ca42c08d`, promoted `06761f9a68f1`).*

| Candidate | Suite | Episodes | Exits | Keys Picked Up | Locked Doors Unlocked | Kills | Mean Cells Visited | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 0 Random Baseline | 10 dev maps (30 eps) | 30 | 1 (3.3%) | 0 | 0 | 10 | 13.97 | 0 |
| Sprint 6B (`s6b_policy.pt`) | 10 dev maps (20 eps) | 20 | 19 (95.0%) | 10 | 10 | 16 | 42.35 | 0 |
| Sprint 6B (`s6b_policy.pt`) | 14 stress probes (28 eps) | 28 | 23 (82.1%) | 16 | 16 | 27 | 37.04 | 0 |
| Sprint 6B (`s6b_policy.pt`) | 8 final_ood maps (16 eps) | 16 | 1 (6.2%) | 4 | 2 | 32 | 54.88 | 0 |
| Sprint 6B (`s6b_policy.pt`) | Freedoom MAP01–08 (8 eps) | 8 | 0 (0.0%) | 0 | 0 | 31 | 89.38 | 0 |

- **Long-horizon backtracking mastery on held-out synthetic tiers:** Sprint 6B achieves `100%` `unlock|pickup` and `100%` `exit|unlock` on `dev` (`0` deaths, `0.9792` proxy score) and solves `10/14` `stress` probes `100%` (including `key_window`, `junction_return`, `stairs_ledge`, `grate_combat`, `nukage_hazard`, and `dropoff_return`; `0.8750` proxy score).

### 2. Sandbox Compliance & Inference Runtime (`validate.py`)
- **Footprint & Dependencies:** `45.0 MB` tracked repo size (cap `1.0 GB`, Git LFS enabled on `submission/weights/policy.pt`), `0` extra packages in `submission/requirements.txt`, offline and read-only compliant.
- **GPU Validation (`validate.py --device cuda`, RTX 3050 6GB):** `5.5s` init (cap `60s`), `16.6 ms` p50 step latency (cap `40 ms`), `125 MB` peak VRAM (cap `2,048 MB`), exit reached on `two_rooms.wad:MAP01`.
- **Linux + CUDA 1-CPU Reference (`telemetry_gru_v1`):** `2.4s` init, `4.0–4.8 ms` p50 (`20 ms` max), `1,380 MB / 8,192 MB` RAM, `156 MB / 2,048 MB` VRAM.
- **Local CPU Validation (promoted `s6b_policy.pt` on Windows):** `4.7s` init, `first 11 ms`, `p50 8.9 ms`, `p99 13.2 ms`, `max 25 ms`, `1.0s / 21.0s` compute budget, exit reached in `104` steps on `two_rooms.wad:MAP01`.

### 3. Concrete Failure Analysis & Mitigations (`SPRINT6B_FAILURE_ANALYSIS.md`)
- **Point-blank RocketLauncher self-kills (`final_ood_t3_dropoff_skull`, 0/3, deaths at steps 23–38):** On spawn, the agent picked up a RocketLauncher (`wpn=5`) and fired at a nearby ChaingunGuy, causing fatal splash self-damage (`85 HP -> 0` in 2 steps). **Fix shipped:** enabled `rocket_safety_mask=True` in the promoted checkpoint (observation-only action mask suppressing `ATTACK` when `selected_weapon == 5` and front rays report close geometry/enemies) alongside a self-damage penalty in `training/rewards.py`.
- **Telemetry aliasing on wall switches and nukage (`stress/exit_switch` 0/3, `remote_door_switch` 0/3, Freedoom):** `doom_env` rays tag wall switches and exit switches as `RAY_WALL` and do not encode damaging nukage floors. On `exit_switch.wad`, the agent saw only `RAY_WALL` and pressed `USE` `0` times across `1,575` steps. Because our model uses `<8%` of VRAM, fusing a tiny downsampled `rgb`/`depth` CNN is our planned next upgrade for switch and hazard visibility.
