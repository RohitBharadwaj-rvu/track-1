"""Promote a trained checkpoint to submission/weights/policy.pt with explicit inference flags.

Usage:
    uv run python -m training.promote_checkpoint training/checkpoints/s6b_policy.pt --label "Sprint 6B"
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path

from submission.policy import load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "submission" / "weights" / "policy.pt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def promote_checkpoint(
    source: Path,
    out_path: Path = DEFAULT_OUT,
    *,
    label: str | None = None,
    eval_deterministic: bool = True,
    rocket_safety_mask: bool = True,
    eval_mirror_ensemble: bool = False,
) -> dict:
    """Copy a trained policy checkpoint to `out_path` with submission inference settings."""
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")

    src_sha = _sha256(source)
    model, cfg, meta = load_checkpoint(source, device="cpu")
    new_cfg = replace(
        cfg,
        eval_deterministic=bool(eval_deterministic),
        rocket_safety_mask=bool(rocket_safety_mask),
        eval_mirror_ensemble=bool(eval_mirror_ensemble),
    )

    new_meta = dict(meta or {})
    try:
        rel_source = str(source.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        rel_source = str(source)
    new_meta["promoted_from"] = rel_source
    new_meta["promoted_source_sha256"] = src_sha
    if label:
        new_meta["promoted_label"] = label

    save_checkpoint(out_path, model, new_cfg, meta=new_meta)
    out_sha = _sha256(out_path)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "source": rel_source,
        "source_sha256": src_sha,
        "output": str(out_path),
        "output_sha256": out_sha,
        "size_bytes": out_path.stat().st_size,
        "params": n_params,
        "arch": new_cfg.arch,
        "use_ego_map": bool(getattr(new_cfg, "use_ego_map", False)),
        "eval_deterministic": new_cfg.eval_deterministic,
        "rocket_safety_mask": new_cfg.rocket_safety_mask,
        "eval_mirror_ensemble": new_cfg.eval_mirror_ensemble,
        "label": label,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a policy checkpoint to submission/weights/policy.pt.")
    parser.add_argument("source", type=Path, help="Path to source policy checkpoint (.pt)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Destination weights path")
    parser.add_argument("--label", default=None, help="Human-readable label stored in checkpoint metadata")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set eval_deterministic in promoted checkpoint config (default: true)",
    )
    parser.add_argument(
        "--rocket-safety-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable observation-based close-range rocket safety mask (default: true)",
    )
    parser.add_argument(
        "--mirror-ensemble",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable B=2 mirror-equivariant inference ensemble (default: false)",
    )
    args = parser.parse_args(argv)

    info = promote_checkpoint(
        args.source,
        args.out,
        label=args.label,
        eval_deterministic=args.deterministic,
        rocket_safety_mask=args.rocket_safety_mask,
        eval_mirror_ensemble=args.mirror_ensemble,
    )
    print(f"Promoted {info['source']} -> {info['output']}")
    print(f"  label:                {info['label'] or '(none)'}")
    print(f"  arch / params:        {info['arch']} / {info['params']:,} ({info['size_bytes'] / 1e6:.2f} MB)")
    print(f"  use_ego_map:          {info['use_ego_map']}")
    print(f"  eval_deterministic:   {info['eval_deterministic']}")
    print(f"  rocket_safety_mask:   {info['rocket_safety_mask']}")
    print(f"  eval_mirror_ensemble: {info['eval_mirror_ensemble']}")
    print(f"  source sha256:        {info['source_sha256'][:16]}")
    print(f"  output sha256:        {info['output_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
