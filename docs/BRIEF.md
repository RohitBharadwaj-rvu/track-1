# Track 1 Brief: Telemetry-Driven Recurrent DOOM Agent

*(Draft working brief — updated by Role 4 across Stage 1 -> Stage 4 milestones)*

---

## Page 1: Architecture & Memory Design

### 1. Problem & Telemetry Choice
Our agent plays unseen `freedoom2.wad` (`skill 3`) levels zero-shot under strict inference limits (`2 GB VRAM`, `8 GB RAM`, `1 CPU thread`, `40 ms/step`). Rather than spending compute on raw pixel CNNs, we operate directly on `doom_env`'s ego-centric telemetry signals (`rays`, `visible_entities`, `vitals`) with `render_buffers=False` during training.

### 2. Policy & Recurrent Memory (`submission/policy.py`)
- **Raycast Encoder:** Circular 1D convolutions over the 64 ego-centric rays (normalized distance + hit type) to preserve directional left/right bins across 360°.
- **Entity Encoder:** Learned class embeddings + relative geometry/velocity features `(32, 5)` pooled with masked attention conditioned on `vitals` and ray context, with a deterministic empty-set fallback.
- **Temporal & Button-Edge Memory:** A GRU fuses `[ray_emb, entity_emb, normalized_vitals, prev_action]` so the policy respects ViZDoom's button-down edge semantics for `USE` and `SELECT_NEXT_WEAPON` and tracks short-term navigation state.
- **Action & Value Heads:** 9 independent binary button logits (`MultiBinary(9)`) and a scalar value critic trained via Recurrent PPO with GAE and episode-boundary hidden-state masking.

### 3. Curriculum & Failure-Driven Upgrades
- Training curriculum covers Tier 1 combat, Tier 2 multi-room exploration, Tier 3 key-and-return progression, and Tier 4 gauntlets using separate train/dev/final map families.
- Architectural upgrades (e.g., learned landmark memory slots for long-horizon key/door backtracking) are promoted only when independent evaluation on identical held-out seeds confirms improvement on Tier 3–4 exits and unlocks.

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
- **Final Agent (`validate.py --docker`):** *TBD after weight freeze.*

### 3. Informative Failure Case
- *TBD: Document one concrete dev-map failure replay from `training/eval.py --record-dir` (e.g., key pickup without return unlock, or `USE` door-bounce) and how curriculum/architecture addressed it.*
