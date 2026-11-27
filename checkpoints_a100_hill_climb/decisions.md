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

## Round 3 — PLANNED (Single Change: `--champion-report` Train Anchor)
- **Hypothesis:** Because both Round 1 and Round 2 showed repeatable regression on `key_return` and `multi_room` progression when resuming Sprint 6B on the 5k corpus, evaluating Sprint 6B on `--split train --seeds 0,1` (training-only anchor, never Dev) and passing `--champion-report` (with restored incumbent `HIGH=0.45`, `POOL=60`) anchors matched training-map competence against forgetting while preserving combat survivability gains, improving Tier 3–4 unlocks/exits on `dev_panel.jsonl` seed 0 without regressing Tier 1–2 exits or deaths.
- **Falsifying Metric:** No lexicographic gain over Sprint 6B (`tier34_exits <= 2` and `locked_doors_unlocked <= 9`) or persistent guardrail regression.
