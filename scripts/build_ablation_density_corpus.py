#!/usr/bin/env python3
"""Build density ablation corpora (N=1, N=5, N=10 variants per seed).

Subsamples up to N valid synthetic variants per active seed document from the
generated `sintetico_*.json` files and writes a deduplicated CSV for pretraining.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def build_ablation_corpus(n_per_doc: int, input_dir: Path, output_file: Path, dry_run: bool = False):
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    json_files = sorted(input_dir.glob("sintetico_*.json"))
    json_files = [f for f in json_files if not f.name.endswith("_checkpoint.json")]

    if not json_files:
        raise FileNotFoundError(f"No sintetico_*.json files found in {input_dir}")

    print("=" * 65)
    print(f"BUILDING DENSITY ABLATION CORPUS: N={n_per_doc} VARIANTS PER SEED")
    print(f"Input directory: {input_dir}")
    print(f"Output CSV:      {output_file}")
    print(f"Files found:     {len(json_files)}")
    print("=" * 65)

    all_rows = []
    seen_texts = set()
    source_counts = Counter()
    duplicates_dropped = 0

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        corpus_input = metadata.get("corpus_input", jf.stem)
        default_source = jf.stem.replace("sintetico_", "")

        docs = data.get("documents", [])
        extracted_from_docs = 0

        if docs and isinstance(docs, list) and len(docs) > 0 and "generated_texts" in docs[0]:
            for doc in docs:
                gen_texts = doc.get("generated_texts", [])
                doc_source = doc.get("source_source") or default_source
                selected = gen_texts[:n_per_doc]
                for t in selected:
                    text_str = t.strip()
                    if not text_str:
                        continue
                    if text_str in seen_texts:
                        duplicates_dropped += 1
                        continue
                    seen_texts.add(text_str)
                    all_rows.append({
                        "text": text_str,
                        "source_source": doc_source,
                        "corpus_origen": corpus_input,
                    })
                    source_counts[doc_source] += 1
                    extracted_from_docs += 1
        else:
            synth = data.get("synthetic", [])
            by_seed = {}
            for r in synth:
                idx = r.get("source_index", -1)
                by_seed.setdefault(idx, []).append(r)

            for _, items in by_seed.items():
                for r in items[:n_per_doc]:
                    text_str = (r.get("text") or "").strip()
                    if not text_str:
                        continue
                    if text_str in seen_texts:
                        duplicates_dropped += 1
                        continue
                    doc_source = r.get("source_source") or default_source
                    seen_texts.add(text_str)
                    all_rows.append({
                        "text": text_str,
                        "source_source": doc_source,
                        "corpus_origen": corpus_input,
                    })
                    source_counts[doc_source] += 1
                    extracted_from_docs += 1

        print(f"  [OK] {jf.name:<45} -> {extracted_from_docs:>7} texts extracted")

    print("-" * 65)
    print(f"TOTAL UNIQUE TEXTS (N={n_per_doc}): {len(all_rows):,}")
    if duplicates_dropped > 0:
        print(f"Duplicates dropped:             {duplicates_dropped:,}")
    print("-" * 65)
    print("\nBreakdown by source:")
    for src, cnt in source_counts.most_common():
        print(f"  {src:<30} {cnt:>8,}")

    if dry_run:
        print("\nDry run -- no file written.")
        return all_rows

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "source_source", "corpus_origen"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[OK] Saved: {output_file}")
    return all_rows


def main():
    p = argparse.ArgumentParser(description="Build N=1, N=5, or N=10 density ablation CSV from synthetic JSONs")
    p.add_argument("--n-per-doc", type=int, required=True, choices=[1, 2, 3, 5, 10], help="Synthetic variants per seed")
    p.add_argument("--input-dir", default="corpus/sinteticos_qwen3.8_27b", help="Directory with sintetico_*.json files")
    p.add_argument("--output", default=None, help="Output CSV path (default: corpus_gigante/maestro_n{N}.csv)")
    p.add_argument("--dry-run", action="store_true", help="Print counts without writing CSV")
    args = p.parse_args()

    out = args.output or f"corpus_gigante/maestro_n{args.n_per_doc}.csv"
    build_ablation_corpus(args.n_per_doc, Path(args.input_dir), Path(out), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
