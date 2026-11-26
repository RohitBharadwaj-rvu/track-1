"""Role 4 release pre-flight check (Git LFS, submission cleanliness, and validate.py).

Usage:
    python -m training.release_check
    python -m training.release_check --docker
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
WEIGHT_EXTS = {".pt", ".pth", ".safetensors", ".ckpt", ".npz", ".msgpack", ".onnx"}


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


def check_lfs_and_submission() -> tuple[bool, list[str]]:
    """Verify Git LFS setup, LFS tracking for weights, and clean submission/ git status."""
    notes: list[str] = []
    ok = True

    lfs_ver = _run(["git", "lfs", "version"])
    if lfs_ver.returncode != 0:
        ok = False
        notes.append("FAIL: git-lfs is not available on PATH.")
    else:
        notes.append(f"OK: {lfs_ver.stdout.strip().splitlines()[0]}")

    lfs_files_res = _run(["git", "-C", str(ROOT), "lfs", "ls-files", "--name-only"])
    lfs_tracked = {line.strip().replace("\\", "/") for line in lfs_files_res.stdout.splitlines() if line.strip()}

    weights_dir = ROOT / "submission" / "weights"
    if weights_dir.is_dir():
        weight_files = [
            p for p in weights_dir.rglob("*")
            if p.is_file() and p.name != ".gitkeep"
        ]
        if not weight_files:
            notes.append("INFO: submission/weights/ has no weight files yet (Stage 1 baseline).")
        for wf in weight_files:
            rel = str(wf.relative_to(ROOT)).replace("\\", "/")
            if wf.suffix.lower() in WEIGHT_EXTS and rel not in lfs_tracked:
                ok = False
                notes.append(f"FAIL: Weight file '{rel}' is not tracked by Git LFS.")
            else:
                notes.append(f"OK: Weight file '{rel}' checked (LFS tracked: {rel in lfs_tracked}).")

    status_res = _run(["git", "-C", str(ROOT), "status", "--porcelain", "submission/"])
    dirty = [ln for ln in status_res.stdout.splitlines() if ln.strip()]
    if dirty:
        notes.append("WARN: Uncommitted/untracked files in submission/ (commit before final release):")
        notes.extend(f"  {ln}" for ln in dirty)
    else:
        notes.append("OK: submission/ working tree is clean.")

    return ok, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Role 4 release pre-flight check.")
    parser.add_argument("--docker", action="store_true", help="Run validate.py --docker")
    parser.add_argument("--image", default=None, help="Docker image name passed to validate.py --image")
    parser.add_argument("--map", action="append", default=None, help="Extra map(s) passed to validate.py --map")
    args = parser.parse_args(argv)

    print("=== Role 4 Release Pre-Flight ===")
    lfs_ok, notes = check_lfs_and_submission()
    for note in notes:
        print(note)

    val_cmd = [sys.executable, str(ROOT / "validate.py")]
    if args.docker:
        val_cmd.append("--docker")
    if args.image:
        val_cmd.extend(["--image", args.image])
    for m in args.map or []:
        val_cmd.extend(["--map", m])

    print(f"\n=== Running: {' '.join(val_cmd)} ===")
    val_res = subprocess.run(val_cmd, cwd=str(ROOT))
    if not lfs_ok:
        return 1
    return val_res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
