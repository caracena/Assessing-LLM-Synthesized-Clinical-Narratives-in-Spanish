"""Reconstruct a deduplicated master synthetic CSV from per-source JSON outputs.

Reads all `sintetico_*.json` files in the target synthetic directory, extracts
`data["synthetic"]`, deduplicates exact text strings, and writes a clean CSV
(`Pure-Synth` pretraining corpus ready for MLM or CoWeSe mixing).

Usage:
    python scripts/reconstruir_maestro.py --model qwen3.8_27b --output corpus_gigante/maestro_qwen3.8_27b.csv
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SINTETICO_DIR = BASE_DIR / "corpus" / "sinteticos"
DEFAULT_OUTPUT = BASE_DIR / "corpus_gigante" / "maestro.csv"


def reconstruir_maestro(output_path: str = None, dry_run: bool = False, sources: list = None, sintetico_dir: str = None):
    output_path = Path(output_path or DEFAULT_OUTPUT)
    sintetico_path = Path(sintetico_dir) if sintetico_dir else SINTETICO_DIR

    json_files = sorted(sintetico_path.glob("sintetico_*.json"))
    json_files = [f for f in json_files if not f.name.endswith("_checkpoint.json")]

    if sources:
        source_set = set(sources)
        json_files = [f for f in json_files if f.stem.replace("sintetico_", "", 1) in source_set]

    if not json_files:
        print(f"[WARN] No sintetico_*.json files found in {sintetico_path}")
        return []

    print(f"\n{'='*60}")
    print("RECONSTRUCTING DEDUPLICATED MASTER CSV")
    print(f"Found {len(json_files)} synthetic source files")
    print(f"{'='*60}\n")

    all_rows = []
    seen_texts = set()
    duplicates_dropped = 0
    source_counts = Counter()

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        synthetic = data.get("synthetic", [])
        corpus_input = meta.get("corpus_input", jf.stem)

        added_here = 0
        for r in synthetic:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            if text in seen_texts:
                duplicates_dropped += 1
                continue
            seen_texts.add(text)
            src = r.get("source_source", "unknown")
            if src == "camtemist":
                src = "cantemist"
            all_rows.append({
                "text": text,
                "source_source": src,
                "corpus_origen": corpus_input,
            })
            source_counts[src] += 1
            added_here += 1

        print(f"  [OK] {jf.name:<45} {added_here:>7} unique / {len(synthetic):>7} total")

    print(f"\n{'-'*60}")
    print(f"Total unique rows:          {len(all_rows)}")
    if duplicates_dropped:
        print(f"Exact duplicates dropped:   {duplicates_dropped}")
    print(f"{'-'*60}")
    print("\nBreakdown by source_source:")
    for src, cnt in source_counts.most_common():
        print(f"  {src:<30} {cnt:>8}")

    if dry_run:
        print("\nDry run -- no file written.")
        return all_rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "source_source", "corpus_origen"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[OK] Master CSV saved: {output_path}")
    print(f"  Total rows: {len(all_rows)}")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build deduplicated synthetic master CSV from JSON outputs")
    parser.add_argument(
        "--model",
        default=None,
        choices=["qwen3_8b", "qwen3.8_27b"],
        help="Automatically resolve standard input/output paths for qwen3_8b or qwen3.8_27b",
    )
    parser.add_argument("--sintetico-dir", default=None, help="Directory containing sintetico_*.json files")
    parser.add_argument("--output", default=None, help="Output master CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing CSV")
    parser.add_argument("--sources", nargs="+", default=None, help="Restrict to specific source names")
    args = parser.parse_args()

    sintetico_dir = args.sintetico_dir
    output_path = args.output

    if args.model == "qwen3.8_27b":
        sintetico_dir = sintetico_dir or (BASE_DIR / "corpus" / "sinteticos_qwen3.8_27b")
        output_path = output_path or (BASE_DIR / "corpus_gigante" / "maestro_qwen3.8_27b.csv")
    elif args.model == "qwen3_8b":
        sintetico_dir = sintetico_dir or (BASE_DIR / "corpus" / "sinteticos")
        output_path = output_path or (BASE_DIR / "corpus_gigante" / "maestro_qwen3_8b.csv")

    reconstruir_maestro(output_path, dry_run=args.dry_run, sources=args.sources, sintetico_dir=sintetico_dir)
