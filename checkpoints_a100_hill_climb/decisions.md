# A100 4-Round Hill-Climb Decision Log

## Baseline (Sprint 6B Incumbent)
- **Commit:** `c9ac3a28b29506aca47a6db92cbdcb3cb7a0c282` (`hill-climb/sprint-6-ood`)
- **Policy:** `s6b_policy.pt` (`sha256=cbe1ca42c08dd37827a2571cbfb5bac97f28584d7e588f5ce67ecd5278ce8e7d`)
- **State:** `s6b_train_state.pt` (`sha256=4eaea1247606e9b7a111c770e58bf4e06a3c1212eb6d9ab647f7ea6a76fa23b8`, restored Adam LR `5e-5`)
- **Dev Panel (36 maps, seed 0, deterministic):** `5/36` exits (`2` T3–4, `3` T1–2), `9` locked-door unlocks, `19` keys picked up (`15/36` eps), `7/15` (`0.4667`) unlock|pickup, `2/7` (`0.2857`) exit|unlock, `9` deaths after pickup, `21` deaths, `10` timeouts, `0` errors, `1,680` USE (`672` consec), `3,499` attacks (`969` wasted), Nash value `0.3038`, mean rubric `0.2170`, CVaR-25 `0.0000`.
- **Sachin S6B Failure Context (`f505081`):** RC1 (telemetry-only `RAY_WALL` switch aliasing), RC2 (point-blank RocketLauncher self-kills), RC3 (heavy-roster combat attrition), RC4 (nukage), RC5 (wasted blocked-sightline attacks & USE loops).

---

## Round 1 — REJECTED (`keep_incumbent`)
- **Config:** `HIGH=0.45`, `POOL=60`, `EXTRA=[]`, `seed=202601`, `1,200s` (`1,203.0s` train wall, `206.4 dec/s`, `485` updates `1440->1924`, `~704` eps `9122->9826`, `248,320` decisions).
- **Export:** `round_1/policy.pt` (`cb7db36c...`), `round_1/latest_train_state.pt` (`904f8a59...`), actor-tensor parity verified (`64` tensors, LR `5e-5`).
- **Dev Panel (36 maps, seed 0):** `4/36` exits (`0` T3–4 [`-2`], `4` T1–2 [`+1`]), `0` unlocks [`-9`], `15` keys (`13` eps, `unlock|pickup=0.00`), `20` deaths (`key_return` deaths `4->7`, `multi_room` deaths `1->2`, `combat` exits `0->2`), `12` timeouts, `0` errors, `140` USE (`22` consec), `2,922` attacks (`316` wasted).
- **Game Theory:** Nash value `0.2005` vs `0.3038`, minimax regret `0.9000` vs `0.7750`, mean rubric `0.1798` vs `0.2170` (`-0.0417`), polytope min `0.0953` vs `0.1327` (`-0.0374`), CVaR-25 `0.0000` (`+0.0`).
- **Diagnosis:** All 4 primal-dual CMDP lambdas (`damage`, `early_death`, `post_key_death`, `use_waste`) pegged at `2.0` under `HIGH=0.45` (`45%` train death rate), suppressing `USE` (`1,680 -> 140`) and collapsing locked-door unlocks (`9 -> 0`) while causing `key_return` survival to regress (`deaths 4 -> 7`).
- **Action:** Restored Sprint 6B incumbent (`s6b_policy.pt`, `s6b_train_state.pt`, `s6b_panel_s0.json`).

---

## Round 2 — REJECTED (`keep_incumbent`)
- **Config:** `HIGH=0.35` (`-0.10`), `POOL=60`, `EXTRA=[]`, `seed=202602`, `1,200s` (`1,201.9s` train wall, `205.8 dec/s`, `483` updates `1440->1922`, `~705` eps `9122->9827`, `247,296` decisions).
- **Export:** `round_2/policy.pt` (`7dffc2fe...`), `round_2/latest_train_state.pt` (`93a3c835...`), actor-tensor parity verified (`64` tensors, LR `5e-5`).
- **Dev Panel (36 maps, seed 0):** `3/36` exits (`0` T3–4 [`-2`], `3` T1–2 [`0`]; `combat` `0->2`, `door_exit` `1->0`, `multi_room` `2->1`), `2` unlocks [`-7`], `14` keys (`11` eps, `unlock|pickup=0.0909`), `16` deaths [`-5`], `17` timeouts [`+7`], `0` errors, `mean_unique_cells` `53.61 -> 41.47`, `462` USE (`115` consec), `2,242` attacks (`207` wasted).
- **Game Theory:** Nash value `0.4459` vs `0.1918` (skewed by `c04125` combat frontier), minimax regret `0.7750` vs `0.5987`, mean rubric `0.1393` vs `0.2170` (`-0.0771`), polytope min `0.0683` vs `0.1327` (`-0.0644`), CVaR-25 `0.0000` (`+0.0`).
- **Diagnosis:** Lowering `HIGH` to `0.35` cut deaths (`21 -> 16`) and wasted attacks (`969 -> 207`) and improved `combat` (`0/3 -> 2/3`), **but** caused repeatable regression on `key_return` (`2->0` exits, `4->0` unlocks across both R1 and R2), collapsed `mean_unique_cells` (`53.61 -> 41.47`), and spiked timeouts (`10 -> 17`). All 4 CMDP lambdas still pegged at `2.0`.
- **Action:** Restored Sprint 6B incumbent (`s6b_policy.pt`, `s6b_train_state.pt`, `s6b_panel_s0.json`, `HIGH=0.45`, `POOL=60`).

---

## Round 3 — REJECTED (`keep_incumbent`)
- **Config:** `--champion-report` (`s6b_train_anchor_s01.json`, verified train-only: `40` eps across `20` train WADs, `35/40` exits), restored `HIGH=0.45`, `POOL=60`, `seed=202603`, `1,200s` (`1,202.3s` train wall, `190.8 dec/s`, `448` updates `1440->1887`, `~759` eps `9122->9881`, `229,376` decisions).
- **Export:** `round_3/policy.pt` (`236ef469...`), `round_3/latest_train_state.pt` (`6e26654e...`), actor-tensor parity verified (`64` tensors, LR `5e-5`).
- **Dev Panel (36 maps, seed 0):** `5/36` exits (`1` T3–4 [`-1`], `4` T1–2 [`+1`]; `combat` `0->2`, `door_exit` `1->1`, `key_return` `2->1`, `multi_room` `2->1`), `3` unlocks [`-6`] (`key_return` `4->3`, `gauntlet` `5->0`), `13` keys (`11` eps, `unlock|pickup=0.2727`, `exit|unlock=0.3333`), `21` deaths (`0`), `10` timeouts (`0`), `0` errors, `mean_unique_cells` `53.61 -> 40.97`, `477` USE (`141` consec), `3,788` attacks (`1,027` wasted).
- **Game Theory:** Nash value `0.3896` vs `0.3910` (Nash weight `0.509` vs `0.490`), minimax regret `0.7750` vs `0.8213`, mean rubric `0.1819` vs `0.2170`, polytope min `0.1022` vs `0.1327`, CVaR-25 `0.0000`.
- **Diagnosis:** `--champion-report` recovered `key_return` unlocks (`3`) and exits (`1`) and matched total exits (`5/36`) with lower minimax regret, **but** still fell short on `tier34_exits` (`-1`) and `locked_doors_unlocked` (`-6`) because exploration (`mean_unique_cells` `53.61 -> 40.97`) and first-milestone discovery (`episodes_with_pickup` `15 -> 11`, `gauntlet` keys `11 -> 7`) collapsed across all 3 rounds under pegged CMDP penalties (`2.0`).
- **Action:** Restored Sprint 6B incumbent (`s6b_policy.pt`, `s6b_train_state.pt`, `s6b_panel_s0.json`, `HIGH=0.45`, `POOL=60`, `EXTRA=[]`).

---

## Round 4 — REJECTED (`keep_incumbent`)
- **Config:** `--curiosity-coef 0.005` (RND intrinsic reward active: `rnd_loss=0.0017`, `mean_intrinsic_reward=0.00122`), restored `HIGH=0.45`, `POOL=60`, `seed=202604`, `1,200s` (`1,203.5s` train wall, `195.3 dec/s`, `459` updates `1440->1898`, `~755` eps `9123->9878`, `235,008` decisions).
- **Export:** `round_4/policy.pt` (`5a513ecd...`), `round_4/latest_train_state.pt` (`f9077ac8...`), actor-tensor parity verified (`64` tensors, LR `5e-5`).
- **Dev Panel (36 maps, seed 0):** `3/36` exits (`0` T3–4 [`-2`], `3` T1–2 [`0`]; `combat` `0->1`, `door_exit` `1->1`, `key_return` `2->0`, `multi_room` `2->1`), `4` unlocks [`-5`] (`key_return` `4->3`, `gauntlet` `5->1`), `16` keys (`14` eps; `key_return` keys `7->9`, `unlock|pickup=0.2857`, `exit|unlock=0.0`), `24` deaths [`+3`] (`deaths_after_pickup` `9->11`), `9` timeouts [`-1`], `0` errors, `mean_unique_cells` `53.61 -> 47.03`, `1,201` USE (`608` consec), `3,167` attacks (`986` wasted).
- **Game Theory:** Nash value `0.2388` vs `0.3351` (Nash weight `0.297` vs `0.703`), minimax regret `0.8867` vs `0.6250`, mean rubric `0.1607` vs `0.2170` (`-0.0624`), polytope min `0.0786` vs `0.1327` (`-0.0541`), CVaR-25 `0.0007` (`+0.0007`).
- **Diagnosis:** RND curiosity partially restored exploration (`mean_unique_cells 47.03` vs `40.97` in R3, `episodes_with_pickup 14` vs `11` in R3, `key_return` keys `9`), **but** raised post-pickup deaths (`9->11`) and total deaths (`21->24`) and failed to beat Sprint 6B on Tier 3–4 exits (`-2`) or unlocks (`-5`).
- **Action:** Rejected and retained pretrial Sprint 6B incumbent (`s6b_policy.pt`, `s6b_train_state.pt`) so no RND module is carried forward.

---

## Final Selection (Frozen Before Opening Sealed Eval)
- **Selected Policy:** Original **Sprint 6B** (`s6b_policy.pt`, `sha256=cbe1ca42c08dd37827a2571cbfb5bac97f28584d7e588f5ce67ecd5278ce8e7d`; `s6b_train_state.pt`, `sha256=4eaea1247606e9b7a111c770e58bf4e06a3c1212eb6d9ab647f7ea6a76fa23b8`).
- **Measured Hill-Climb Gain:** **`0`** (all 4 challengers rejected on paired Dev panel evidence).
- **Optional Methods Summary:**
  - `HIGH=0.35` (Round 2): Ran; triggered by Round 1 `key_return` survival regression and pegged CMDP lambdas; cut deaths (`21->16`) but spiked timeouts (`10->17`) and collapsed exploration (`41.47`) -> Rejected.
  - `--champion-report` (Round 3): Ran on verified train-only Sprint 6B anchor (`40` eps, `20` train WADs); triggered by repeatable `key_return`/`multi_room` regression in R1–R2; best challenger (`5/36` exits, `1` T3–4, `3` unlocks, lower minimax regret `0.775`) but still `-1` T3–4 exit / `-6` unlocks vs S6B -> Rejected.
  - `--curiosity-coef 0.005` RND (Round 4): Ran (`mean_intrinsic_reward=0.00122`); triggered by exploration/pickup contraction in R1–R3; lifted pickups (`14`) and cells (`47.03`) but raised deaths (`24`) and regressed T3–4 exits (`-2`) -> Rejected.
  - `POOL=48`: Not triggered (unlocks did not rise into timeouts).
  - ACCEL adversarial level mutation: **Deferred** (no training-family saturation observed; corpus-manifest path bypasses online ACCEL).
