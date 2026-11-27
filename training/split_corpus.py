"""Split an audited corpus by difficulty and archetype using a fixed seed."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random


def split_corpus(manifest: Path, seed: int = 2026) -> dict[str, dict[str, int]]:
    groups = defaultdict(list)
    layouts = set()
    maps = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        identity = (record["wad"], record["map_name"])
        layout = record["d4_sig_hash"]
        if identity in maps or not layout or layout in layouts:
            raise ValueError(f"Missing or duplicate map geometry: {identity}")
        maps.add(identity)
        layouts.add(layout)
        groups[record["difficulty_band"], record["archetype"]].append(record)

    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "eval": []}
    for group in sorted(groups):
        records = sorted(groups[group], key=lambda r: r["d4_sig_hash"])
        rng.shuffle(records)
        held_out = max(1, round(len(records) * 0.1)) if len(records) >= 3 else 0
        splits["dev"].extend(records[:held_out])
        splits["eval"].extend(records[held_out:2 * held_out])
        splits["train"].extend(records[2 * held_out:])

    bands = {band for band, _ in groups}
    counts = {split: dict(Counter(r["difficulty_band"] for r in records))
              for split, records in splits.items()}
    if not bands or any(set(count) != bands for count in counts.values()):
        raise ValueError("Each difficulty band needs enough maps to appear in all three splits")

    for split, records in splits.items():
        for record in records:
            record["split"] = split
        _write_jsonl(manifest.with_name(f"{split}.jsonl"), records)
        if split == "dev":
            # Keep a fixed, relatively cheap panel while covering every available archetype and band.
            used = set()
            panel = []
            for record in records:
                key = (record["difficulty_band"], record["archetype"])
                if key not in used:
                    used.add(key)
                    panel.append(record)
            _write_jsonl(manifest.with_name("dev_panel.jsonl"), panel)
    return counts


def _write_jsonl(path, records):
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(split_corpus(args.manifest, args.seed), indent=2))
