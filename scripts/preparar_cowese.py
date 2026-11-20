"""Prepare the CoWeSe Spanish biomedical web corpus as a scientific background anchor.

Groups consecutive sentences from CoWeSe.txt into 25-sentence pseudo-documents
and subsamples 251,548 documents (seed=42) to provide general biomedical replay
during continual MLM pretraining.

Usage:
    python scripts/preparar_cowese.py \
        --input data/cowese/CoWeSe.txt \
        --output data/cowese/cowese_prepared.csv \
        --target-docs 251548 \
        --sentences-per-doc 25 \
        --seed 42
"""

import argparse
import csv
import random
from pathlib import Path
from tqdm import tqdm


def group_sentences_into_docs(input_path: str, sentences_per_doc: int) -> list[str]:
    """Group consecutive lines into fixed-size sentence blocks."""
    docs = []
    buffer = []

    with open(input_path, encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Grouping sentences into pseudo-documents"):
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if len(buffer) >= sentences_per_doc:
                docs.append(" ".join(buffer))
                buffer = []

    if len(buffer) >= sentences_per_doc // 2:
        docs.append(" ".join(buffer))

    return docs


def main():
    parser = argparse.ArgumentParser(description="Prepare CoWeSe biomedical web text for continual pretraining")
    parser.add_argument("--input", required=True, help="Path to CoWeSe.txt")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--target-docs",
        type=int,
        default=251548,
        help="Number of pseudo-documents to sample (default: 251548)",
    )
    parser.add_argument(
        "--sentences-per-doc",
        type=int,
        default=25,
        help="Sentences per pseudo-document (default: 25)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Reading and grouping: {args.input}")
    docs = group_sentences_into_docs(args.input, args.sentences_per_doc)
    print(f"Generated pseudo-documents: {len(docs)}")

    if len(docs) > args.target_docs:
        print(f"Subsampling {args.target_docs} documents (seed={args.seed})...")
        docs = random.sample(docs, args.target_docs)
    else:
        print(f"[WARN] Generated {len(docs)} documents (< target {args.target_docs}); keeping all.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "source_source", "corpus_origen"])
        writer.writeheader()
        for doc in docs:
            writer.writerow({
                "text": doc,
                "source_source": "cowese",
                "corpus_origen": "CoWeSe (Spanish Biomedical Web Corpus)",
            })

    print(f"\n[OK] Saved: {output_path}")
    print(f"  {len(docs)} pseudo-documents (~{args.sentences_per_doc} sentences each)")


if __name__ == "__main__":
    main()
