#!/usr/bin/env python3
"""Dataset catalog for local BRAT corpora, train-only generation seeds, and downstream splits.

Usage:
  python scripts/dataset_registry.py --list
  python scripts/dataset_registry.py --split achs
  python scripts/dataset_registry.py --seeds achs_pseudo --out ~/datasets/ACHS_pseudo_seeds --style normalize_case
"""
import argparse
import json
import pathlib
import re
import sys

import yaml

BASE = pathlib.Path(__file__).resolve().parent.parent
REGISTRY = BASE / "configs" / "dataset_registry.yaml"


def load():
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["datasets"]


def expand(p):
    return pathlib.Path(p).expanduser()


def resolve(p):
    q = pathlib.Path(p)
    return q if q.is_absolute() else BASE / q


def normalize_case(t):
    t = t.lower()
    t = re.sub(r"(?<=[.!?]\s)\w", lambda m: m.group(0).upper(), t)
    return t[0].upper() + t[1:] if t else t


def cmd_list(ds):
    for name, e in ds.items():
        ev = e.get("evaluation") or {}
        gen = e.get("generation") or {}
        split = e.get("split") or {}
        print(
            f"{name:22s} fmt={e['format']:11s} task={ev.get('task', '-'):4s} "
            f"split={split.get('mode', '-'):6s} seeds={gen.get('seed_policy', '-')}"
        )


def cmd_split(ds, name):
    f = (ds[name].get("split") or {}).get("file")
    if not f or not resolve(f).exists():
        print(f"INFO: No fixed split file for {name} (mode={(ds[name].get('split') or {}).get('mode', 'auto')})")
        return
    s = json.load(open(resolve(f), encoding="utf-8"))
    print(f"{name}: train={len(s['train'])} val={len(s['val'])} test={len(s['test'])}")


def cmd_seeds(ds, name, out, override_style=None):
    e = ds[name]
    gen = e.get("generation") or {}
    pol = gen.get("seed_policy", "all")
    src = expand(e["path"])
    if not src.exists():
        sys.exit(f"ERROR: Source seed directory does not exist: {src}")
    if pol.startswith("file:"):
        ids = set(json.load(open(resolve(pol[5:]), encoding="utf-8"))["seed_docs"])
    elif pol == "all":
        ids = None
    else:
        sys.exit(f"Unsupported seed_policy: {pol}")
    style = override_style or gen.get("style", "raw")
    out = expand(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.txt")):
        if ids is not None and p.stem not in ids:
            continue
        dst = out / p.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if style == "normalize_case":
            dst.write_text(normalize_case(p.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")
        else:
            dst.symlink_to(p.resolve())
        n += 1
    (out / "SEEDS_PROVENANCE.json").write_text(
        json.dumps(
            {
                "source": str(src),
                "seed_policy": pol,
                "style": style,
                "n_seeds": n,
                "excluded_by": "configs/seeds/achs_pseudo__train_only.json (val/test from ACHS seed42 split)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Materialized {n} train-only seeds in {out} (style={style})")


def main():
    ap = argparse.ArgumentParser(description="Inspect dataset catalog or materialize train-only seeds")
    ap.add_argument("--list", action="store_true", help="List all registered datasets")
    ap.add_argument("--split", help="Print split statistics for a dataset")
    ap.add_argument("--seeds", help="Dataset name to materialize filtered seed files")
    ap.add_argument("--out", help="Output directory for materialized seeds")
    ap.add_argument("--style", choices=["raw", "normalize_case"], help="Optional case normalization mode")
    a = ap.parse_args()
    ds = load()
    if a.list:
        cmd_list(ds)
    if a.split:
        cmd_split(ds, a.split)
    if a.seeds:
        if not a.out:
            sys.exit("--seeds requires --out")
        cmd_seeds(ds, a.seeds, a.out, a.style)


if __name__ == "__main__":
    main()
