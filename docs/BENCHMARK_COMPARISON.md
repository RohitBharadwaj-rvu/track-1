# Track 1 Benchmark Comparison: Telemetry Policy vs. Reference Baselines

*(Compiled by Role 4 from measured local/Colab runs and published ViZDoom literature profiles. Per `REVISED_PLAN.md §6`, external systems below are published reference profiles on standard ViZDoom pixel scenarios—not matched runs on `doom_env`. True apples-to-apples gameplay comparisons are measured on our identical 10 held-out `DEV_MAPS` seeds against `docs/eval_baseline_dev.json`.)*

---

## 1. Side-by-Side Architecture & Footprint Comparison

| System | Inputs | Parameters | Inference VRAM *(Cap: 2,048 MB)* | Inference Latency *(Cap: 40 ms)* | Training Throughput / Footprint | Source / Notes |
|---|---|---|---|---|---|---|
| **Ours (`telemetry_gru_v1`, measured)** | Ego-centric telemetry (`rays`, `visible_entities`, `vitals`, `prev_action`) | **`0.69M`** (`687,098`) | **`156 MB`** *(7.6% of cap)* | **`4.0–4.8 ms` p50** (`20 ms` max, 1 CPU + CUDA) | `325–417 dec/s` on 2 Colab T4 CPUs (`render_buffers=False`); `100%` exit on `#P..D...E#` in `80k` steps | Measured via `validate.py` and `training/train.py` in [training/BENCHMARKS.md](../training/BENCHMARKS.md); `1,380 MB / 8,192 MB` RAM. |
| **Official Random Starter (measured)** | None (random `MultiBinary(9)`) | `0` | `~0 MB` | `0.8–1.5 ms` p50 | None | Measured on all 10 held-out `DEV_MAPS` (30 episodes) in [docs/eval_baseline_dev.json](eval_baseline_dev.json): `1/30` exits, `0` keys, `0` unlocks, `19` deaths, `57.6%` opposing buttons. |
| **DreamerV3 *(published pixel reference)*** | Raw `64×64` RGB pixels + RSSM + decoder | `~10M–20M` *(15–30× larger)* | `~1.5–3.0 GB` *(risks 2 GB cap)* | `~2–5 ms` model-only + pixel render overhead | `6–10 GB` training VRAM | Spends capacity reconstructing wall textures; tight fit inside the 2 GB VRAM evaluation sandbox. |
| **Sample Factory APPO *(published pixel reference)*** | Raw `128×72` RGB pixels + ConvNet + RNN | Multi-million pixel CNN + RNN | `4–8 GB` training VRAM | Fast on GPU, pixel-bound | `10k–100k` steps/s on 10–72 CPU cores; `doom_battle` trains up to `4B` steps | Designed for massive multi-core servers on single pixel scenarios (`doom_basic`, `doom_battle`), not `doom_env` telemetry + keycard backtracking within 24 hours. |

---

## 2. Key Takeaways for Training & Judging

1. **Massive Sandbox Headroom (`13×` under VRAM cap, `~8×` under latency cap):**
   `telemetry_gru_v1` consumes only **`156 MB` VRAM** (out of `2,048 MB`) and **`4.0–4.8 ms` p50** (out of `40 ms/step`) while pinned to 1 CPU core. That gives our team plenty of headroom to add **learned landmark memory slots** for Tier 3–4 keycard backtracking without risking OOM or per-step timeouts.
2. **Telemetry Throughput Without Pixel Rendering:**
   Setting `render_buffers=False` eliminates ViZDoom screen-buffer rendering during rollouts, hitting **`325–417 dec/s`** even on a 2-CPU Colab T4 and scaling further on multi-CPU A100/L4 runtimes.
3. **Apples-to-Apples Progression Benchmarks (Next):**
   Once full Colab A100 training finishes on Role 3's `train` suite, Role 4 runs the exact held-out comparison against `docs/eval_baseline_dev.json`:
   ```bash
   python -m training.eval --suite mvp_gate --check-mvp-gate docs/eval_baseline_dev.json
   python -m training.eval --suite dev --out docs/eval_mvp_dev.json --record-dir /tmp/replays --keep-key-replays
   python -m training.eval --compare docs/eval_baseline_dev.json docs/eval_mvp_dev.json
   ```
