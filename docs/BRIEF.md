# Track 1 Brief: Telemetry-Driven Recurrent DOOM Agent

*(Draft working brief — updated by Role 4 across Stage 1 -> Stage 4 milestones)*

---

## Page 1: Architecture & Memory Design

### 1. Problem & Telemetry Choice
Our agent plays unseen `freedoom2.wad` (`skill 3`) levels zero-shot under strict inference limits (`2 GB VRAM`, `8 GB RAM`, `1 CPU thread`, `40 ms/step`). Rather than spending compute on raw pixel CNNs, we operate directly on `doom_env`'s ego-centric telemetry signals (`rays`, `visible_entities`, `vitals`) with `render_buffers=False` during training.

### 2. Policy & Recurrent Memory (`submission/policy.py`, `telemetry_gru_v1`, `730,109` params)
- **Raycast Encoder:** Circular 1D convolutions over the 64 ego-centric rays (normalized distance + hit type) preserving left/right directional bins across 360°.
- **Entity Encoder:** Learned class embeddings + derived ego-frame combat kinematics (closing speed, lateral speed, inverse time-to-impact, crosshair alignment) + per-category threat/item counts, pooled with masked multi-query attention conditioned on vitals and ray context (with deterministic empty-set fallback).
- **Vitals & Temporal Memory:** GRU fusing `[ray_emb, entity_emb, vitals_action_emb]` with explicit `selected_ammo_empty`, `has_any_other_ammo`, `low_health`, horizontal speed, and `prev_action` features so the policy detects wall-snags and respects ViZDoom's button-down edge semantics for `USE` and `SELECT_NEXT_WEAPON`.
- **Factorized Action Heads & Training Pipeline:** Three 3-way categorical heads (forward/backward/none, strafe left/right/none, turn left/right/none) + three independent Bernoulli heads (`ATTACK`, `USE`, `SELECT_NEXT_WEAPON`) mapped deterministically to `MultiBinary(9)`, trained via Recurrent PPO (`gamma=0.995`, `chunk_len=64`) with a training-only asymmetric critic, auxiliary door-bearing supervision, intrinsic curiosity, and weighted curriculum sampling.

### 3. Curriculum & Failure-Driven Upgrades
- Training curriculum covers Tier 1 combat, Tier 2 multi-room exploration, Tier 3 key-and-return progression, and Tier 4 gauntlets using separate train/dev/final map families plus procedural variations.
- Memory challenger branches (`telemetry_landmark_v1` persistent landmark slots and `EgocentricSpatialMap` 2D grid) are promoted only when independent evaluation on identical held-out Tier 3–4 dev seeds confirms higher unlocks and exits.

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
