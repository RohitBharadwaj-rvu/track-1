"""Stage 1 (Connect) smoke utility: export weights and reload via DoomAgent across 2 maps."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env  # noqa: E402
from submission.agent import DEFAULT_WEIGHTS_PATH, DoomAgent  # noqa: E402
from submission.policy import PolicyConfig, build_policy, save_checkpoint  # noqa: E402


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_WEIGHTS_PATH), help="Output weights path (.pt)")
    parser.add_argument("--arch", default="telemetry_gru_v1", help="Policy architecture (telemetry_gru_v1 | stub)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for smoke policy init")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--steps", type=int, default=30, help="Steps per map for the 2-map check")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    out_path = Path(args.out).resolve()

    torch.manual_seed(args.seed)
    cfg = PolicyConfig(arch=args.arch)
    policy = build_policy(cfg)
    param_count = sum(p.numel() for p in policy.parameters())
    save_checkpoint(
        out_path,
        policy,
        cfg,
        meta={"stage": "stage1_connect_smoke", "seed": args.seed, "arch": cfg.arch},
    )
    size_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Exported {out_path} ({size_mb:.2f} MB, {param_count:,} params, arch={cfg.arch})")

    os.environ["DOOM_AGENT_WEIGHTS"] = str(out_path)
    latencies_ms: list[float] = []
    env = make_env(ROOT / "maps" / "two_rooms.wad", "MAP01", episode_ticks=2100, render_buffers=False)
    try:
        agent = DoomAgent(env.observation_space, env.action_space, device)
        for map_idx, seed in enumerate((0, 1), start=1):
            obs, _ = env.reset(seed=seed)
            agent.reset()
            assert torch.count_nonzero(agent._hidden) == 0, "GRU hidden state was not zeroed on reset()"
            assert torch.count_nonzero(agent._prev_action) == 0, "prev_action was not zeroed on reset()"
            for _ in range(args.steps):
                t0 = time.perf_counter()
                action = agent.act(obs)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                assert isinstance(action, np.ndarray) and action.shape == (9,) and action.dtype == np.int8
                assert np.all((action == 0) | (action == 1))
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
            print(f"Map run {map_idx}/2 passed ({args.steps} max steps, seed={seed}, device={device})")
    finally:
        env.close()

    arr = np.array(latencies_ms, dtype=np.float64)
    print(
        f"Latency over {len(arr)} steps on {device}: "
        f"mean={arr.mean():.2f} ms, p99={np.percentile(arr, 99):.2f} ms, max={arr.max():.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
