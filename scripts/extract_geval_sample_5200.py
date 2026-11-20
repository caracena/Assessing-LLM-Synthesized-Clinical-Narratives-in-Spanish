#!/usr/bin/env python3
"""Extract a balanced stratified sample of synthetic clinical notes for blinded Jev-1.13 evaluation.

Samples `--per-source` notes per generator (`8B` and `27B`) across available seed
sources (default `--per-source 200` across all 13 sources -> N = 5,200; pass
`--per-source 10` for N = 260). Gracefully skips restricted sources (e.g.
`achs_pseudo`) if unavailable locally.
"""

import argparse
import json
import os
import pandas as pd

DEFAULT_DATASETS = [
    "achs_pseudo", "adolescere", "atadolcc", "cantemist", "cardioccc",
    "cares", "chilean_waiting_list", "e3c", "jart", "livingner",
    "meddoplace", "reccmi", "spaccc",
]


def resolve_csv_path(rel_path: str) -> str:
    if os.path.exists(rel_path):
        return rel_path
    project_root = os.environ.get("PROJECT_ROOT", ".")
    alt_path = os.path.join(project_root, rel_path)
    if os.path.exists(alt_path):
        return alt_path
    return ""


def main():
    parser = argparse.ArgumentParser(description="Extract balanced stratified synthetic notes for clinical quality evaluation")
    parser.add_argument("--per-source", type=int, default=200, help="Notes per source per generator (200 -> N=5200, 10 -> N=260)")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Source datasets to sample from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="results/geval_sample_data_5200.json", help="Output JSON path")
    parser.add_argument("--preserve-cache", default="results/geval_sample_data_260.json", help="Optional existing cache JSON to preserve")
    args = parser.parse_args()

    existing_by_key = {}
    existing_indices = {}

    if args.preserve_cache and os.path.exists(args.preserve_cache):
        with open(args.preserve_cache, "r", encoding="utf-8") as f:
            prev_samples = json.load(f)
        for s in prev_samples:
            key = (s["generator"], s["dataset"])
            existing_by_key.setdefault(key, []).append(s)
            try:
                orig_idx = int(s["id"].rsplit("_", 1)[-1])
                existing_indices.setdefault(key, set()).add(orig_idx)
            except ValueError:
                pass
        print(f"Loaded {len(prev_samples)} prior samples from {args.preserve_cache}")

    paired_datasets = []
    only_8b_datasets = []
    only_27b_datasets = []
    for ds in args.datasets:
        p8 = resolve_csv_path(f"corpus/sinteticos/sintetico_{ds}.csv")
        p27 = resolve_csv_path(f"corpus/sinteticos_qwen3.8_27b/sintetico_{ds}.csv")
        if p8 and p27:
            paired_datasets.append((ds, p8, p27))
        elif p8:
            only_8b_datasets.append((ds, p8))
        elif p27:
            only_27b_datasets.append((ds, p27))
        else:
            print(f"[SKIP] Missing synthetic CSV for '{ds}'; omitting from sample.")

    if paired_datasets:
        gen_plan = [
            ("8B", "8b", [(ds, p8) for ds, p8, _ in paired_datasets]),
            ("27B", "27b", [(ds, p27) for ds, _, p27 in paired_datasets]),
        ]
        n_sources = len(paired_datasets)
    elif only_8b_datasets:
        print(f"[INFO] Single-generator mode (8B only across {len(only_8b_datasets)} sources).")
        gen_plan = [("8B", "8b", only_8b_datasets)]
        n_sources = len(only_8b_datasets)
    elif only_27b_datasets:
        print(f"[INFO] Single-generator mode (27B only across {len(only_27b_datasets)} sources).")
        gen_plan = [("27B", "27b", only_27b_datasets)]
        n_sources = len(only_27b_datasets)
    else:
        raise FileNotFoundError("No sintetico_*.csv files found in corpus/sinteticos*.")

    samples = []
    for generator, prefix, ds_entries in gen_plan:
        for ds, csv_path in ds_entries:
            key = (generator, ds)
            kept = list(existing_by_key.get(key, []))
            used_idx = set(existing_indices.get(key, set()))
            need = args.per_source - len(kept)

            if need > 0:
                df = pd.read_csv(csv_path).dropna(subset=["text"])
                df = df[df["text"].astype(str).str.strip().str.len() > 10]
                if used_idx:
                    df = df.drop(index=[i for i in used_idx if i in df.index])
                if len(df) < need:
                    raise RuntimeError(
                        f"Insufficient rows in {csv_path}: requested {need}, available {len(df)}"
                    )
                extra_df = df.sample(n=need, random_state=args.seed)
                for idx, r in extra_df.iterrows():
                    kept.append({
                        "id": f"{prefix}_{ds}_{idx}",
                        "generator": generator,
                        "dataset": ds,
                        "text": str(r["text"])[:1500],
                    })

            samples.extend(kept[: args.per_source])

    n_8b = sum(1 for s in samples if s["generator"] == "8B")
    n_27b = sum(1 for s in samples if s["generator"] == "27B")
    print(f"Extracted {len(samples)} balanced notes across {n_sources} sources ({n_8b} from 8B, {n_27b} from 27B)")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved sample to {args.output}")


if __name__ == "__main__":
    main()
