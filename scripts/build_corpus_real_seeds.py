#!/usr/bin/env python3
"""Build the Real Seeds control corpus from authentic clinical seed texts.

Collects all normalized seed collections (`corpus/por_source/*_normalizado.json`),
filters short entries (< min_words words), applies cross-source exact deduplication,
and writes `corpus_gigante/maestro_real_seeds.csv` (29,812 authentic texts across
the 13 paper seed collections).
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def build_real_seeds_corpus(input_dir: Path, output_file: Path, min_words: int = 15):
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*_normalizado.json"))
    if not json_files:
        raise FileNotFoundError(f"No *_normalizado.json files found in {input_dir}")

    print("=" * 65)
    print("BUILDING REAL SEEDS CONTROL CORPUS")
    print(f"Input directory: {input_dir}")
    print(f"Output CSV:      {output_file}")
    print(f"Files found:     {len(json_files)}")
    print("=" * 65)

    all_rows = []
    seen_texts = set()
    source_counts = Counter()
    short_dropped = 0
    duplicate_dropped = 0

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        source_name = data.get("metadata", {}).get("name", jf.stem.replace("_normalizado", ""))
        texts = data.get("texts", [])
        print(f"-> Reading {jf.name} ({source_name}): {len(texts)} raw seeds")

        for item in texts:
            if isinstance(item, dict):
                text_str = item.get("text", "").strip()
                item_source = item.get("source", source_name)
            else:
                text_str = str(item).strip()
                item_source = source_name

            word_count = len(text_str.split())
            if word_count < min_words:
                short_dropped += 1
                continue

            if text_str in seen_texts:
                duplicate_dropped += 1
                continue

            seen_texts.add(text_str)
            all_rows.append({
                "text": text_str,
                "source_source": item_source,
                "corpus_origen": str(jf.as_posix()),
            })
            source_counts[item_source] += 1

    print("=" * 65)
    print(f"Valid unique clinical seeds:       {len(all_rows)}")
    print(f"Dropped short (<{min_words} words):       {short_dropped}")
    print(f"Dropped cross-source duplicates:   {duplicate_dropped}")
    print("Distribution by clinical source:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {src:25s}: {cnt:>6d} texts")
    print("=" * 65)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "source_source", "corpus_origen"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[OK] Real Seeds CSV saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Build deduplicated Real Seeds control CSV")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("corpus/por_source"),
        help="Directory containing *_normalizado.json seed files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corpus_gigante/maestro_real_seeds.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=15,
        help="Minimum word count threshold per seed document (default: 15)",
    )
    args = parser.parse_args()

    build_real_seeds_corpus(args.input_dir, args.output, args.min_words)


if __name__ == "__main__":
    main()
