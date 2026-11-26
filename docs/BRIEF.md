# Track 1 Brief: Telemetry-Driven Recurrent DOOM Agent

*(Draft working brief — updated by Role 4 across Stage 1 -> Stage 4 milestones)*

---

## Page 1: Architecture & Memory Design

### 1. Problem & Telemetry Choice
Our agent plays unseen `freedoom2.wad` (`skill 3`) levels zero-shot under strict inference limits (`2 GB VRAM`, `8 GB RAM`, `1 CPU thread`, `40 ms/step`). Rather than spending compute on raw pixel CNNs, we operate directly on `doom_env`'s ego-centric telemetry signals (`rays`, `visible_entities`, `vitals`) with `render_buffers=False` during training.

### 2. Policy & Recurrent Memory (`submission/policy.py`, `telemetry_gru_v1`, `730,109` base params)
- **Raycast & Entity Encoders:** Circular 1D convolutions over 64 ego-centric rays (distance + hit type) paired with learned entity class embeddings, derived combat kinematics (closing/lateral speed, inverse TTI, crosshair alignment), and masked multi-query attention conditioned on vitals and ray context.
- **Vitals, Temporal & Spatial Memory:** GRU fusing `[ray_emb, entity_emb, vitals_action_emb]` with explicit ammo/health/speed and `prev_action` features (detecting wall-snags and `USE` / `SELECT_NEXT_WEAPON` edge semantics), optionally augmented with `EgocentricSpatialMap` 2D spatial memory (`submission/ego_map.py`, with off-crop objective beacons).
- **Factorized Action Heads & Training Pipeline:** Three 3-way categorical heads (move, strafe, turn) + three Bernoulli heads (`ATTACK`, `USE`, `SELECT_NEXT_WEAPON`) mapped deterministically to `MultiBinary(9)`, trained via Recurrent PPO (`gamma=0.995`, `chunk_len=64`) with a training-only asymmetric critic, non-negative PBRS (`training/pbrs.py`) supporting window/ledge/switch routing, L/R mirror symmetry augmentation (`training/symmetry.py`), and weighted curriculum sampling.

### 3. Curriculum & Evaluation Suites
- Training curriculum covers Tier 1 combat, Tier 2 multi-room exploration, Tier 3 key-and-return progression, and Tier 4 gauntlets with broadened procedural verticality, enemy roster, and angled/door variations.
- Held-out verification spans 10 `DEV_MAPS`, 10 `maps/stress/` probes (orientation, exit/remote switches, stairs/ledges, grates, key-window), and `freedoom2.wad` (`MAP01–04`); `FINAL_MAPS` stays untouched until final weight freeze.
- Memory upgrades (`EgocentricSpatialMap` and `telemetry_landmark_v1`) are promoted only when independent evaluation on identical held-out Tier 3–4 dev seeds confirms higher unlocks and exits.

---

## Page 2: Evaluation Outcomes, Runtime & Failure Analysis

### 1. Baseline vs. Final Outcomes (Local Dev Proxy Suite)
*Note: Organizer evaluation weights and hidden maps are private; local metrics below are proxy measurements.*

| Candidate | Suite | Episodes | Exits | Keys Picked Up | Locked Doors Unlocked | Kills | Mean Cells Visited | Errors |
|---|---|---|---|---|---|---|---|---|
| Stage 0 Random (`submission/agent.py`) | 10 dev maps (30 eps) | 30 | 1 | 0 | 0 | 10 | 13.97 | 0 |
| Stage 2 MVP Recurrent PPO | *TBD* | — | — | — | — | — | — | — |
| Final Shipped Agent | *TBD* | — | — | — | — | — | — | — |

### 2. Sandbox Compliance & Inference Runtime (`validate.py`)
- **Hardware target:** RTX 3090 (2 GB VRAM cap), 8 GB RAM, 1 CPU thread (`OMP_NUM_THREADS=1`), offline read-only repo.
- **Stage 0 Random Baseline (`validate.py` on Windows local):** `init 0.0s`, `first 2 ms`, `p50 0.9 ms`, `p99 1.6 ms`, `max 2 ms`, `0.5s / 21.0s` per-map compute budget.
- **Stage 1 Connect `telemetry_gru_v1` (`validate.py` on Linux + CUDA, pinned to 1 CPU):** `init 2.4s`, `p50 4.0–4.8 ms`, `p99 6.9–10.8 ms`, `max 20 ms`, `2.2–2.5s / 21.0s` compute budget, `1,380 MB / 8,192 MB` RAM, `156 MB / 2,048 MB` VRAM. Solvable door/exit proof (`#P..D...E#`): `100%` exit rate (`24` steps on two consecutive maps with `agent.reset()`).
- **Final Agent (`validate.py --docker`):** *TBD after weight freeze.*

### 3. Informative Failure Case
- *TBD: Document one concrete dev-map failure replay from `training/eval.py --record-dir` (e.g., key pickup without return unlock, or `USE` door-bounce) and how curriculum/architecture addressed it.*
