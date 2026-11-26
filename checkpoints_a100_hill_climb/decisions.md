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

## Round 2 — PLANNED (Single Change: `HIGH: 0.45 -> 0.35`)
- **Hypothesis:** Reducing `HIGH` from `0.45` to `0.35` (keeping `POOL=60`, `EXTRA=[]`, resuming from `s6b_train_state.pt`) reduces lethal nightmare saturation (addressing RC3 combat attrition and preventing CMDP `use_waste`/`post_key_death` lambdas from pegging and suppressing unlocks), yielding more Tier 3–4 unlocks and exits on `dev_panel.jsonl` seed 0 while preserving Tier 1–2 exits and combat survival.
- **Falsifying Metric:** No lexicographic improvement over Sprint 6B (`tier34_exits <= 2` and `locked_doors_unlocked <= 9`) or persistent guardrail regression.
