#!/usr/bin/env python3
"""Mix a synthetic or seed clinical CSV corpus with the CoWeSe biomedical replay corpus."""

import argparse
import csv
import random
from pathlib import Path

FIELDS = ["text", "source_source", "corpus_origen"]


def mix_corpus_files(
    synth_path: Path,
    ext_path: Path,
    out_path: Path,
    sample_external: int = None,
    seed: int = 42,
) -> dict:
    """Concatenate clinical/synthetic CSV rows with CoWeSe rows into a unified training CSV."""
    with open(synth_path, "r", encoding="utf-8", newline="") as f:
        rows_synth = [
            {k: r.get(k, "") for k in FIELDS}
            for r in csv.DictReader(f)
            if (r.get("text") or "").strip()
        ]

    with open(ext_path, "r", encoding="utf-8", newline="") as f:
        rows_ext = [
            {
                "text": r.get("text", ""),
                "source_source": r.get("source_source") or "cowese",
                "corpus_origen": r.get("corpus_origen") or str(ext_path),
            }
            for r in csv.DictReader(f)
            if (r.get("text") or "").strip()
        ]

    if sample_external is not None and sample_external > 0 and sample_external < len(rows_ext):
        rows_ext = random.Random(seed).sample(rows_ext, sample_external)

    all_rows = rows_synth + rows_ext
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"Mixed corpus saved to {out_path}: "
        f"{len(rows_synth):,} clinical/synthetic + {len(rows_ext):,} CoWeSe = {len(all_rows):,} total"
    )
    return {"n_synthetic": len(rows_synth), "n_external": len(rows_ext), "n_total": len(all_rows)}


def main():
    parser = argparse.ArgumentParser(description="Mix clinical/synthetic CSV with external CoWeSe replay CSV")
    parser.add_argument("--synthetic", required=True, help="Path to synthetic or real-seed master CSV")
    parser.add_argument("--external", required=True, help="Path to prepared CoWeSe CSV")
    parser.add_argument("--output", required=True, help="Output path for combined pretraining CSV")
    parser.add_argument("--sample-external", type=int, default=None, help="Optional subsample size for CoWeSe")
    parser.add_argument("--seed", type=int, default=42, help="Random seed when subsampling")
    args = parser.parse_args()

    mix_corpus_files(
        synth_path=Path(args.synthetic).expanduser(),
        ext_path=Path(args.external).expanduser(),
        out_path=Path(args.output).expanduser(),
        sample_external=args.sample_external,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
