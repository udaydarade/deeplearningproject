"""Build explicitly bounded V3.1 topological blocks from the frozen manifest."""
from __future__ import annotations
import json
import re
from pathlib import Path

MANIFEST = Path("data/v3/feature_sets.json")
SCALARS = ("totalpers", "maxpers", "entropy", "bar_count", "landscape_l1", "landscape_l2")
DYNAMICS = ("_slope5", "_std5", "_ac15", "_z250", "_velocity")


def build_blocks(manifest: dict[str, list[str]]) -> dict[str, list[str]]:
    universe = manifest["F4"]
    blocks = {}
    for cloud in ("T", "A"):
        for h in (0, 1, 2):
            pat = re.compile(rf"^tda_{cloud}_k(12|24|full)_w(60|120)_h{h}_(.+)$")
            names = []
            for c in universe:
                m = pat.match(c)
                if not m: continue
                tail=m.group(3)
                if any(tail.startswith(s) for s in SCALARS) and not any(tail.endswith(d) for d in DYNAMICS):
                    names.append(c)
            blocks[f"{cloud}_H{h}"] = sorted(set(names))
    for cloud in ("T", "A"):
        dyn = [c for c in universe if c.startswith(f"tda_{cloud}_") and any(c.endswith(s) for s in DYNAMICS)]
        windows = sorted({re.search(r"_w(\d+)_", c).group(1) for c in dyn if re.search(r"_w(\d+)_", c)})
        for w in windows:
            # The time-series groups are split by point-cloud window to respect
            # the preregistered maximum of 60 input columns per block.
            cols = sorted(c for c in dyn if f"_w{w}_" in c)
            for i in range(0, len(cols), 60):
                part = i // 60 + 1
                name = f"DYN_{cloud}_w{w}" + (f"_part{part}" if len(cols) > 60 else "")
                blocks[name] = cols[i:i+60]
    return blocks


if __name__ == "__main__":
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, cols in build_blocks(m).items():
        print(f"{name}: {len(cols)} columns")
