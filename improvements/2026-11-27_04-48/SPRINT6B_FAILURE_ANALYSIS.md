# Sprint 6B Failure Analysis: `stress`, `final_ood`, and `freedoom`

**Date:** 2026-11-27  
**Branch / Commit Evaluated:** `hill-climb/sprint-6-ood` @ `7d1fe29`  
**Checkpoint:** `training/checkpoints/s6b_policy.pt` (SHA-256 prefix `cbe1ca42c08d`, `3.56 MB`)  
**Hardware:** Local NVIDIA GeForce RTX 3050 6GB Laptop GPU (`torch==2.11.0+cu128`, `vizdoom==1.3.3`)

---

## 1. Executive Summary & Local GPU Reproduction

All evaluations were reproduced locally on GPU (`--device cuda`, `--eval-mode deterministic`) across all 3 default seeds per map (`120` total episodes across 4 suites) plus the official `validate.py` benchmark:

- **`validate.py --device cuda`:** **PASSED** (`5.5s` init, `~16.6 ms` p50 step latency, `~125 MB` peak VRAM vs `2,048 MB` cap, exit reached on both `two_rooms.wad:MAP01` runs).
- **Exact seed-matched reproduction:** Local GPU results match the committed Sprint 6B reports in `training/results/sprint6b-2026-11-27/` across every suite (`dev` 19/20, `stress` 23/28, `final_ood` 1/16, `freedoom` 0/8).

| Suite | Local GPU (All 3 Seeds) | Matched-Seed Local GPU | Saved S6B Report | Deaths | Timeouts | Mean Damage | Mean Death Step | Avg / Max Step (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`dev`** | **28/30** (`93.3%`, score `0.9722`) | 19/20 | 19/20 (`95.0%`) | 0 | 2 | 21.3 | — | 15.7 / 49.6 |
| **`stress`** | **33/42** (`78.6%`, score `0.8214`) | 23/28 | 23/28 (`82.1%`) | 2 | 7 | 15.3 | 402.0 | 15.8 / 98.0 |
| **`final_ood`** | **3/24** (`12.5%`, score `0.0625`) | 1/16 | 1/16 (`6.2%`) | **20** | 1 | **122.0** | **197.7** | 16.0 / 71.2 |
| **`freedoom`** | **0/24** (`0.0%`, score `0.0000`) | 0/8 | 0/8 (`0.0%`) | **12** | **12** | **68.1** | 410.2 | 16.0 / 69.3 |

**Bottom line:** Sprint 6B has largely mastered synthetic in-distribution navigation and multi-key backtracking (`100%` `unlock|pickup` and `100%` `exit|unlock` on `dev`; `10/14` `stress` probes solved `3/3`). Its collapse on `stress` (remaining 4 probes), `final_ood` (`20/24` deaths), and `freedoom` (`0/24` exits) stems from **six concrete, verifiable root causes** spanning sensor blindness, weapon self-damage, and combat survivability.

---

## 2. Root Cause Breakdown (With Trace Evidence)

### RC1 — Sensory Blindness to Wall Switches & Exit Switches (`RAY_WALL` Aliasing)
- **Affected maps:** `stress/exit_switch.wad` (`0/3` timeout), `stress/remote_door_switch.wad` (`0/3` timeout), `final_ood_t2_nukage_remote.wad`, `final_ood_t4_nukage_gauntlet.wad`, and Freedoom `MAP01`, `MAP03`, `MAP05`, `MAP06`.
- **Root cause in code:**
  - `env/doom_env/spaces.py` defines only 6 ray hit types: `("none", "wall", "door", "locked_door", "enemy", "item")`.
  - `env/doom_env/sensors.py` (`LevelGeometry.update`) marks only closed doors (`lock == 0` -> `RAY_DOOR`, `lock > 0` -> `RAY_LOCKED`); **wall switches, exit switches, and lifts emit `RAY_WALL`**.
  - `submission/obs.py` explicitly drops `rgb` and `depth` (`preprocess_obs` uses telemetry only).
- **Direct trace evidence:**
  - `exit_switch.wad` (seed 0): `Ray hit types seen: ['wall']`; across all 3 seeds (`1,575` steps total), the agent pressed `USE` **0 times**. To a telemetry-only policy, the exit switch wall is mathematically identical to a dead-end wall.
  - `remote_door_switch.wad` (seed 0): `Ray hit types seen: ['wall', 'door']`; the closed remote door is tagged `RAY_DOOR` while the switch in the north alcove is `RAY_WALL`. The agent pressed `USE` `63` times across 3 seeds (`12` consecutive) trying to open the remote door or adjacent walls, never discovering the invisible switch.
  - `freedoom2.wad:MAP03` & `MAP06`: `use=0` across `1,050` steps per seed despite exploring `38–67` cells; `MAP01` clears `15` enemies and opens `6` doors across 3 seeds (`127.7` cells) but times out `3/3` without triggering the switch exit.

### RC2 — Point-Blank Rocket Launcher Self-Kills
- **Affected maps:** `final_ood_t3_dropoff_skull.wad` (`0/3`, deaths at steps `23`, `38`, `24`), `final_ood_t4_nukage_gauntlet.wad` (`0/3`, deaths at steps `177`, `357`).
- **Direct trace evidence (`final_ood_t3_dropoff_skull.wad`, seed 0):**
  ```text
  s15: wpn 2->5 (ammo=7)              # Picks up RocketLauncher near spawn
  s17: hp 100->94 (wpn=5, atk=0)      # Takes light hit from ChaingunGuy
  s20: hp 94->85  (wpn=5, atk=1)      # Fires RocketLauncher
  s22: END term=True exit=False hp=85 # Instant splash-damage self-kill (85 HP -> dead)
  ```
- **Why it happens:** The policy fires `ATTACK` whenever an enemy is in front of it without conditioning on `selected_weapon == 5` (RocketLauncher) + close range to an enemy or wall. On `final_ood_t3_dropoff_skull`, a `RocketLauncher` (`ThingSpec(384, 256, 2003)`) sits right in the starting courtyard next to a `ChaingunGuy` (`256, 384`), causing instant self-destruction on every seed.

### RC3 — Severe Combat Survivability Deficit Against High-DPS / Heavy Rosters & Wide Arenas
- **Affected maps:** `stress/open_arena_horde.wad` (`1/3`), `stress/three_lock_citadel.wad` (`2/3`), **all 8 `final_ood` maps** (`20/24` deaths, mean damage `122.0`), and Freedoom `MAP02`, `MAP04`, `MAP07`, `MAP08` (`12/12` deaths).
- **Training-time evidence:** Even at the end of Sprint 6B training (`training/results/sprint6b-2026-11-27/metrics.jsonl`, update `1439`, `3.26M` decisions), the agent still had a **`43.75%` death rate** (`exit_rate=39.6%`) on its own training distribution.
- **Direct trace evidence (`final_ood_t1_open_crossfire.wad`, seed 0):**
  - The agent kept the starter **Pistol (`wpn=2`)** all the way until dying at step `93`, walking past the `Shotgun` (`320, 640`) and `GreenArmor` (`320, 480`) directly ahead of spawn.
  - Wide `>1024u` arenas (`final_ood_t1_open_crossfire` is `1408u` wide; `open_arena_horde` and Freedoom maps exceed `1024u`) produce `RAY_NONE` hits while enemies (`ChaingunGuy`, `ShotgunGuy`, `Revenant`, `Arachnotron`, `Fatso`, `BaronOfHell`, `HellKnight`) engage from long range or elevated balconies.
  - On `final_ood`, `deaths_after_pickup = 7` — the agent takes heavy attrition while collecting keys and dies during the return leg (`final_ood_t3_window_backtrack`, `final_ood_t4_three_lock_fortress`, `final_ood_t4_nukage_gauntlet`).

### RC4 — Sensory Blindness to Damaging Floors (Nukage / Hazard Sectors)
- **Affected maps:** `final_ood_t2_nukage_remote.wad` (`0/3`, deaths at steps `68`, `61`, `70`), `final_ood_t4_nukage_gauntlet.wad` (`0/3`).
- **Root cause:** Damaging sectors (`sector_special=71`, `NUKAGE1`) are **not encoded anywhere in `rays`, `visible_entities`, or `vitals`**. Without `rgb`, the agent only learns it is standing in nukage after health drops.
- **Direct trace evidence (`final_ood_t2_nukage_remote.wad`, seed 0):**
  - Health plunged `100 -> 94 -> 85 -> 71 -> 59 -> 47 -> 35 -> 26` between steps `11` and `21` (nukage damage + `ChaingunGuy` + `Spectre` in the trench), dying at step `67`.
  - Although Sprint 6B passed the simple `stress/nukage_hazard.wad` (`3/3`, no enemies), combining nukage with active hitscanners/melee + a remote switch in `final_ood_t2_nukage_remote` is fatal `100%` of the time.

### RC5 — Wasted Attacks Through Impassable Sightlines / Grates & Target-Lock Loops
- **Affected maps:** `freedoom2.wad:MAP05` (`703/703` = **`100%` wasted attacks** across 3 seeds!), `final_ood_t3_window_backtrack.wad` (`77/202` = `38%` wasted attacks), `final_ood_t4_three_lock_fortress.wad` (`53/281` wasted attacks, `121/210` consecutive `USE`).
- **Root cause:** `visible_entities` reports enemies visible through `<56u` windows, grates, or across elevated ledges (`sight_block` is false even when `move_block` is true), and the policy fires continuously without verifying hits or switching tactics when shots are blocked by geometry or out of effective range.

### RC6 — Scale, Verticality & Multi-Mechanic Shift on Freedoom (`0/24` Exits)
- **Affected maps:** All 8 Freedoom maps (`MAP01–MAP08`: `12` deaths, `12` timeouts, `0` exits, only `1` key picked up and `0` locked doors unlocked across `24` episodes).
- **Root cause:** Real Doom/Freedoom levels combine all five issues above simultaneously:
  - Large multi-floor maps (`>1024u` sightlines, lifts, stairs, secret/remote triggers) that exceed the spatial horizon of the ego-map crop and training distribution.
  - Heavy reliance on wall switches and lifts (invisible in telemetry-only observations).
  - Sustained combat attrition (`84` kills across 24 episodes, yet `12/24` deaths) with scarce health/ammo management.

---

## 3. Per-Map Failure Tables (Local RTX 3050 GPU, 3 Seeds/Map)

### `stress` (Failed 4 / 14 maps; 33/42 episodes passed)

| Map | Tier | Outcomes (s0,s1,s2) | Steps | Keys / Unlocks | Kills | Atk (Wasted) | USE (Consec) | Primary Root Cause |
|---|---:|---|---|---:|---:|---:|---:|---|
| `exit_switch.wad` | 2 | T, T, T | 525, 525, 525 | 0 / 0 | 0 | 0 (0) | 0 (0) | **RC1** (switch emits `RAY_WALL`) |
| `remote_door_switch.wad` | 2 | T, T, T | 525, 525, 525 | 0 / 0 | 0 | 0 (0) | 63 (12) | **RC1** (remote switch invisible; spams USE on door) |
| `open_arena_horde.wad` | 4 | D, E, T | 438, 270, 1050 | 0 / 0 | 20 | 1036 (92) | 0 (0) | **RC3** (wide arena horde attrition) |
| `three_lock_citadel.wad` | 4 | E, E, D | 389, 441, 366 | 9 / 9 | 8 | 360 (89) | 158 (65) | **RC3** (late-stage combat death after unlocking) |

*(All other 10 `stress` maps — including `key_window`, `junction_return`, `stairs_ledge`, `grate_combat`, `nukage_hazard`, and `dropoff_return` — passed `3/3`.)*

### `final_ood` (Failed 8 / 8 maps; 3/24 episodes passed)

| Map | Tier | Outcomes (s0,s1,s2) | Steps | Keys / Unlocks | Kills | Atk (Wasted) | USE (Consec) | Primary Root Cause |
|---|---:|---|---|---:|---:|---:|---:|---|
| `final_ood_t1_open_crossfire.wad` | 1 | D, D, D | 94, 199, 221 | 0 / 0 | 5 | 198 (17) | 0 (0) | **RC3** (keeps Pistol `wpn=2` vs ChaingunGuy/Imp/ShotgunGuy) |
| `final_ood_t1_cliff_grate_arena.wad` | 1 | D, D, E | 161, 81, 71 | 0 / 0 | 2 | 242 (8) | 1 (0) | **RC3 / RC5** (grate crossfire + Demon/ShotgunGuy) |
| `final_ood_t2_nukage_remote.wad` | 2 | D, D, D | 68, 61, 70 | 0 / 0 | 4 | 199 (17) | 1 (0) | **RC4 + RC3 + RC1** (nukage + ChaingunGuy/Spectre + remote switch) |
| `final_ood_t2_stair_labyrinth.wad` | 2 | D, E, E | 185, 182, 233 | 0 / 0 | 6 | 131 (39) | 56 (32) | **RC3** (1 death in 3-door stepped combat) |
| `final_ood_t3_dropoff_skull.wad` | 3 | D, D, D | 23, 38, 24 | 1 / 0 | 3 | 73 (13) | 0 (0) | **RC2** (point-blank RocketLauncher self-kill) |
| `final_ood_t3_window_backtrack.wad` | 3 | D, D, D | 333, 176, 45 | 3 / 2 | 5 | 202 (77) | 40 (10) | **RC3 + RC5** (Arachnotron/Fatso/Cacodemon attrition + window shots) |
| `final_ood_t4_three_lock_fortress.wad` | 4 | D, D, D | 264, 494, 883 | 2 / 2 | 8 | 281 (53) | 210 (121) | **RC3** (9-monster boss roster + post-pickup deaths) |
| `final_ood_t4_nukage_gauntlet.wad` | 4 | D, T, D | 177, 1050, 357 | 2 / 0 | 14 | 737 (113) | 12 (5) | **RC4 + RC1 + RC2 + RC3** (nukage + remote switch + RocketLauncher + 8 monsters) |

### `freedoom` (Failed 8 / 8 maps; 0/24 episodes passed)

| Map | Outcomes (s0,s1,s2) | Steps | Keys / Doors / Unlocks | Kills | Cells | Atk (Wasted) | USE | Primary Root Cause |
|---|---|---|---:|---:|---:|---:|---:|---|
| `MAP01` | T, T, T | 1050, 1050, 1050 | 0 / 6 / 0 | 15 | 127.7 | 299 (69) | 21 | **RC1 + RC6** (switch exit invisible) |
| `MAP02` | D, D, D | 234, 243, 229 | 0 / 2 / 0 | 10 | 73.3 | 509 (34) | 1 | **RC3** (combat death) |
| `MAP03` | T, T, T | 1050, 1050, 1050 | 0 / 0 / 0 | 0 | 38.0 | 24 (24) | 0 | **RC1 + RC6** (stuck; `use=0`) |
| `MAP04` | D, D, D | 573, 735, 114 | 0 / 2 / 0 | 16 | 158.3 | 631 (85) | 2 | **RC3 + RC6** (sustained combat attrition) |
| `MAP05` | T, T, T | 1050, 1050, 1050 | 0 / 3 / 0 | 6 | 48.7 | 703 (703) | 3 | **RC5 + RC1** (`100%` wasted attacks) |
| `MAP06` | T, T, T | 1050, 1050, 1050 | 0 / 0 / 0 | 0 | 67.0 | 3 (3) | 0 | **RC1 + RC6** (stuck; `use=0`) |
| `MAP07` | D, D, D | 264, 182, 747 | 1 / 7 / 0 | 16 | 93.7 | 525 (123) | 19 | **RC3 + RC6** (died after key pickup) |
| `MAP08` | D, D, D | 493, 477, 632 | 0 / 4 / 0 | 21 | 126.7 | 517 (48) | 33 | **RC3 + RC6** (heavy combat attrition) |

---

## 4. Prioritized Action Plan for Sprint 7

1. **Eliminate RocketLauncher Self-Kills (High Impact, Low Effort):**
   - Add an explicit self-damage penalty in `training/rewards.py` (detect large single-step HP drops when `selected_weapon == 5` and `ATTACK == 1`) and include `RocketLauncher` + close-quarters enemies in the procedural combat curriculum so the policy learns to switch weapons (`SELECT_NEXT_WEAPON`) or back up before firing rockets.
2. **Add Lightweight Hybrid Vision (`rgb` / `depth` Tiny CNN or Switch/Hazard Patch Encoder) (High Impact for RC1 & RC4):**
   - Current latency (`~16 ms` on RTX 3050, `~4–5 ms` on RTX 3090) and VRAM (`~125 MB` of `2,048 MB`) leave massive headroom to fuse a tiny 3-layer strided CNN on downsampled `30×40` `rgb` (and/or `depth`) into the GRU input.
   - This is the **only** principled way to make wall switches, exit switches, lifts, and damaging nukage floors observable without violating fair-play rules.
3. **Close the Combat Survivability Gap (High Impact for RC3):**
   - Sprint 6B's `43.75%` training death rate shows the policy has not yet converged on heavy-roster combat.
   - Increase Tier 1/4 combat curriculum weight with **weapon-upgrade + armor scavenging** (reward picking up Shotgun/SSG/Chaingun/Plasma/Armor when under-equipped), **heavy hitscanner/projectile strafe-dodging** (`ChaingunGuy`, `Revenant`, `Arachnotron`, `Fatso`), and **post-key-pickup ambush survival**.
4. **Penalize Blocked-Sightline Attack Spam & Stagnant Remote-Door USE Loops (Medium Impact for RC5):**
   - Strengthen wasted-attack penalties when enemy HP / kill count does not change over sustained firing bursts, and add wall-switch + remote-door procedural training maps (paired with the lightweight RGB encoder from #2) so the policy learns to search alcoves for switches when a `RAY_DOOR` fails to open on `USE`.
5. **Submission Hygiene Reminder:**
   - `submission/weights/policy.pt` currently still contains Sprint 5A weights (`f7e75f9d9d99`), not `s6b_policy.pt` (`cbe1ca42c08d`), and active work is on `hill-climb/sprint-6-ood` rather than `master`. Promote the winning checkpoint and merge to `master` before the submission deadline.
