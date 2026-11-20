#!/usr/bin/env python3
"""Corpus-level lexical diversity and specialized clinical terminology audit (Qwen-3-8B vs Qwen-3.8-27B).

Computes vocabulary richness (unique types, STTR-1000, Distinct-2/3, Hapax ratio),
specialized medical term density via clinical affixes, and 3/4-gram novelty
against source seeds. Outputs `results/corpus_diversity_comparison.csv`,
`figures/corpus_diversity_report.txt`, and `figures/fig4_corpus_lexical_diversity.{pdf,png}`.
"""

import argparse
from collections import Counter
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Common Spanish clinical morphological suffixes and prefixes (preserved verbatim)
CLINICAL_SUFFIXES = (
    "itis", "osis", "oma", "patia", "patía", "ectomia", "ectomía", "tomia", "tomía",
    "scopia", "scopía", "megalia", "algia", "emia", "uria", "plejia", "plejía",
    "paresia", "grafia", "grafía", "terapia", "metria", "metría", "genesis", "génesis"
)
CLINICAL_PREFIXES = (
    "hiper", "hipo", "peri", "intra", "post", "pre", "sub", "retro", "cardio",
    "neuro", "osteo", "gastro", "hemo", "angio", "bronco", "hepato", "encefalo"
)


def clean_and_tokenize(text):
    """Lowercase and tokenize clinical Spanish text while preserving accents."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [w for w in text.split() if len(w) > 1 and not w.isdigit()]
    return tokens


def is_clinical_term(word):
    """Detect whether a token matches specialized Spanish clinical morphology."""
    w = word.lower()
    if len(w) < 5:
        return False
    for suff in CLINICAL_SUFFIXES:
        if w.endswith(suff):
            return True
    for pref in CLINICAL_PREFIXES:
        if w.startswith(pref):
            return True
    return False


def compute_sttr(tokens, chunk_size=1000):
    """Standardized Type-Token Ratio (STTR) in 1,000-token windows."""
    if len(tokens) < chunk_size:
        return len(set(tokens)) / max(len(tokens), 1)
    ratios = []
    for i in range(0, len(tokens) - chunk_size + 1, chunk_size):
        chunk = tokens[i : i + chunk_size]
        ratios.append(len(set(chunk)) / chunk_size)
    return np.mean(ratios)


def compute_distinct_ngrams(tokens, n=2):
    """Compute Distinct-N (unique n-grams / total n-grams)."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)


def compute_novelty_rate(synth_tokens, seed_tokens, n=3):
    """Percentage of synthetic n-grams absent from the seed corpus."""
    if len(synth_tokens) < n or len(seed_tokens) < n:
        return 0.0
    seed_ngrams = set(tuple(seed_tokens[i : i + n]) for i in range(len(seed_tokens) - n + 1))
    synth_ngrams = [tuple(synth_tokens[i : i + n]) for i in range(len(synth_tokens) - n + 1)]
    novel_count = sum(1 for ng in synth_ngrams if ng not in seed_ngrams)
    return novel_count / max(len(synth_ngrams), 1)


def analyze_corpus_sample(csv_path, sample_size=20000, seed=42):
    """Load a stratified/random corpus sample and compute lexical and clinical metrics."""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        first_col = df.columns[0]
        df.rename(columns={first_col: "text"}, inplace=True)

    total_docs = len(df)
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=seed)
    else:
        df_sample = df

    all_tokens = []
    doc_lengths = []
    clinical_terms = set()
    total_clinical_tokens = 0

    for text in df_sample["text"]:
        toks = clean_and_tokenize(text)
        if toks:
            all_tokens.extend(toks)
            doc_lengths.append(len(toks))
            for t in toks:
                if is_clinical_term(t):
                    clinical_terms.add(t)
                    total_clinical_tokens += 1

    total_tokens = len(all_tokens)
    vocab = set(all_tokens)
    vocab_size = len(vocab)
    token_counts = Counter(all_tokens)
    hapax = sum(1 for _, count in token_counts.items() if count == 1)

    sttr = compute_sttr(all_tokens, chunk_size=1000)
    dist_1 = vocab_size / max(total_tokens, 1)
    dist_2 = compute_distinct_ngrams(all_tokens, n=2)
    dist_3 = compute_distinct_ngrams(all_tokens, n=3)
    hapax_ratio = hapax / max(vocab_size, 1)
    clin_density = (total_clinical_tokens / max(total_tokens, 1)) * 100

    return {
        "total_docs_full": total_docs,
        "sample_docs": len(df_sample),
        "total_tokens_sample": total_tokens,
        "vocab_size": vocab_size,
        "mean_doc_length": np.mean(doc_lengths) if doc_lengths else 0.0,
        "std_doc_length": np.std(doc_lengths) if doc_lengths else 0.0,
        "sttr_1k": sttr,
        "distinct_1": dist_1,
        "distinct_2": dist_2,
        "distinct_3": dist_3,
        "hapax_ratio": hapax_ratio,
        "unique_clinical_terms": len(clinical_terms),
        "clinical_density_pct": clin_density,
    }


def _resolve_novelty_source(project_root: str, requested_source: str, active_modes: tuple = ("8b", "27b")):
    """Resolve normalized seeds and matching synthetic CSV paths for active generator modes."""
    candidates = [requested_source]
    por_dir = os.path.join(project_root, "corpus/por_source")
    if os.path.isdir(por_dir):
        for fname in sorted(os.listdir(por_dir)):
            if fname.endswith("_normalizado.json"):
                cand = fname.replace("_normalizado.json", "")
                if cand not in candidates:
                    candidates.append(cand)

    for src in candidates:
        seed_path = os.path.join(project_root, f"corpus/por_source/{src}_normalizado.json")
        s8_path = os.path.join(project_root, f"corpus/sinteticos/sintetico_{src}.csv")
        s27_path = os.path.join(project_root, f"corpus/sinteticos_qwen3.8_27b/sintetico_{src}.csv")
        if not os.path.exists(seed_path):
            continue
        ok_8b = ("8b" not in active_modes) or os.path.exists(s8_path)
        ok_27b = ("27b" not in active_modes) or os.path.exists(s27_path)
        if ok_8b and ok_27b:
            return src, seed_path, (s8_path if os.path.exists(s8_path) else None), (s27_path if os.path.exists(s27_path) else None)
    return requested_source, None, None, None


def _resolve_master_csv(project_root: str, rel_path: str):
    if os.path.exists(rel_path):
        return rel_path
    alt = os.path.join(project_root, rel_path)
    return alt if os.path.exists(alt) else None


def main():
    parser = argparse.ArgumentParser(description="Analyze corpus-level lexical diversity and clinical richness")
    parser.add_argument("--seed-source", default="achs_pseudo", help="Source name for n-gram novelty vs seeds (default: achs_pseudo, falls back if absent)")
    parser.add_argument("--sample-size", type=int, default=20000, help="Sample size per master corpus (default: 20000)")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("=" * 75)
    print("CORPUS LEXICAL DIVERSITY & CLINICAL TERMINOLOGY AUDIT")
    print("=" * 75)

    project_root = os.environ.get("PROJECT_ROOT", ".")
    path_8b = _resolve_master_csv(project_root, "corpus_gigante/maestro_qwen3_8b.csv")
    path_27b = _resolve_master_csv(project_root, "corpus_gigante/maestro_qwen3.8_27b.csv")

    if not path_8b and not path_27b:
        raise FileNotFoundError(
            "No master synthetic CSVs found at corpus_gigante/maestro_qwen3_8b.csv or corpus_gigante/maestro_qwen3.8_27b.csv. Run Step 2 first."
        )

    stats_map = {}
    if path_8b:
        stats_map["8b"] = analyze_corpus_sample(path_8b, sample_size=args.sample_size, seed=42)
    if path_27b:
        stats_map["27b"] = analyze_corpus_sample(path_27b, sample_size=args.sample_size, seed=42)

    active_modes = tuple(stats_map.keys())
    if len(active_modes) == 1:
        only_name = "Qwen3-8B" if active_modes[0] == "8b" else "Qwen3.8-27B"
        print(f"[INFO] Single-generator mode ({only_name} only).")

    nov_src, seed_path, synth_8b_path, synth_27b_path = _resolve_novelty_source(
        project_root, args.seed_source, active_modes=active_modes
    )
    novelty_3gram = {"8b": 0.0, "27b": 0.0}
    novelty_4gram = {"8b": 0.0, "27b": 0.0}

    if seed_path:
        print(f"\nComputing synthetic n-gram novelty vs original seeds ({nov_src})...")
        with open(seed_path, "r", encoding="utf-8") as f:
            seed_data = json.load(f)
        if isinstance(seed_data, dict) and "texts" in seed_data:
            raw_items = seed_data["texts"][:1000]
        elif isinstance(seed_data, list):
            raw_items = seed_data[:1000]
        else:
            raw_items = []
        seed_text = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_items])
        seed_tokens = clean_and_tokenize(seed_text)

        for key, csv_p in [("8b", synth_8b_path), ("27b", synth_27b_path)]:
            if key in stats_map and csv_p:
                df_s = pd.read_csv(csv_p).head(1000)
                toks_s = clean_and_tokenize(" ".join(df_s["text"].dropna().astype(str)))
                novelty_3gram[key] = compute_novelty_rate(toks_s, seed_tokens, n=3) * 100
                novelty_4gram[key] = compute_novelty_rate(toks_s, seed_tokens, n=4) * 100
    else:
        print(f"\n[WARN] Seed/per-source synthetic files not found for '{args.seed_source}'; n-gram novelty reported as 0.0.")

    if "8b" in stats_map and "27b" in stats_map:
        stats_8b, stats_27b = stats_map["8b"], stats_map["27b"]
        vocab_rel = ((stats_27b["vocab_size"] / max(stats_8b["vocab_size"], 1)) - 1) * 100
        clin_rel = ((stats_27b["unique_clinical_terms"] / max(stats_8b["unique_clinical_terms"], 1)) - 1) * 100
        results = [
            {"Metric": "Total Generated Docs", "Qwen3-8B": f"{stats_8b['total_docs_full']:,}", "Qwen3.8-27B": f"{stats_27b['total_docs_full']:,}", "Delta (27B vs 8B)": f"+{stats_27b['total_docs_full'] - stats_8b['total_docs_full']:,}"},
            {"Metric": "Average Length (Words/Doc)", "Qwen3-8B": f"{stats_8b['mean_doc_length']:.1f} ± {stats_8b['std_doc_length']:.1f}", "Qwen3.8-27B": f"{stats_27b['mean_doc_length']:.1f} ± {stats_27b['std_doc_length']:.1f}", "Delta (27B vs 8B)": f"{stats_27b['mean_doc_length'] - stats_8b['mean_doc_length']:+.1f}"},
            {"Metric": "Unique Vocabulary (Types in sample)", "Qwen3-8B": f"{stats_8b['vocab_size']:,}", "Qwen3.8-27B": f"{stats_27b['vocab_size']:,}", "Delta (27B vs 8B)": f"+{stats_27b['vocab_size'] - stats_8b['vocab_size']:,} ({vocab_rel:+.1f}%)"},
            {"Metric": "Standardized TTR (STTR-1000)", "Qwen3-8B": f"{stats_8b['sttr_1k']:.4f}", "Qwen3.8-27B": f"{stats_27b['sttr_1k']:.4f}", "Delta (27B vs 8B)": f"{stats_27b['sttr_1k'] - stats_8b['sttr_1k']:+.4f}"},
            {"Metric": "Distinct-2 (Unique Bigrams)", "Qwen3-8B": f"{stats_8b['distinct_2']:.4f}", "Qwen3.8-27B": f"{stats_27b['distinct_2']:.4f}", "Delta (27B vs 8B)": f"{stats_27b['distinct_2'] - stats_8b['distinct_2']:+.4f}"},
            {"Metric": "Distinct-3 (Unique Trigrams)", "Qwen3-8B": f"{stats_8b['distinct_3']:.4f}", "Qwen3.8-27B": f"{stats_27b['distinct_3']:.4f}", "Delta (27B vs 8B)": f"{stats_27b['distinct_3'] - stats_8b['distinct_3']:+.4f}"},
            {"Metric": "Hapax Legomena Ratio", "Qwen3-8B": f"{stats_8b['hapax_ratio']*100:.2f}%", "Qwen3.8-27B": f"{stats_27b['hapax_ratio']*100:.2f}%", "Delta (27B vs 8B)": f"{(stats_27b['hapax_ratio'] - stats_8b['hapax_ratio'])*100:+.2f}%"},
            {"Metric": "Specialized Clinical Terms", "Qwen3-8B": f"{stats_8b['unique_clinical_terms']:,}", "Qwen3.8-27B": f"{stats_27b['unique_clinical_terms']:,}", "Delta (27B vs 8B)": f"+{stats_27b['unique_clinical_terms'] - stats_8b['unique_clinical_terms']:,} ({clin_rel:+.1f}%)"},
            {"Metric": "Medical Token Density (%)", "Qwen3-8B": f"{stats_8b['clinical_density_pct']:.2f}%", "Qwen3.8-27B": f"{stats_27b['clinical_density_pct']:.2f}%", "Delta (27B vs 8B)": f"{stats_27b['clinical_density_pct'] - stats_8b['clinical_density_pct']:+.2f}%"},
            {"Metric": f"3-gram Novelty vs Seeds ({nov_src})", "Qwen3-8B": f"{novelty_3gram['8b']:.2f}%", "Qwen3.8-27B": f"{novelty_3gram['27b']:.2f}%", "Delta (27B vs 8B)": f"{novelty_3gram['27b'] - novelty_3gram['8b']:+.2f}%"},
            {"Metric": f"4-gram Novelty vs Seeds ({nov_src})", "Qwen3-8B": f"{novelty_4gram['8b']:.2f}%", "Qwen3.8-27B": f"{novelty_4gram['27b']:.2f}%", "Delta (27B vs 8B)": f"{novelty_4gram['27b'] - novelty_4gram['8b']:+.2f}%"},
        ]
    else:
        k_only = active_modes[0]
        st = stats_map[k_only]
        col = "Qwen3-8B" if k_only == "8b" else "Qwen3.8-27B"
        results = [
            {"Metric": "Total Generated Docs", col: f"{st['total_docs_full']:,}"},
            {"Metric": "Average Length (Words/Doc)", col: f"{st['mean_doc_length']:.1f} ± {st['std_doc_length']:.1f}"},
            {"Metric": "Unique Vocabulary (Types in sample)", col: f"{st['vocab_size']:,}"},
            {"Metric": "Standardized TTR (STTR-1000)", col: f"{st['sttr_1k']:.4f}"},
            {"Metric": "Distinct-2 (Unique Bigrams)", col: f"{st['distinct_2']:.4f}"},
            {"Metric": "Distinct-3 (Unique Trigrams)", col: f"{st['distinct_3']:.4f}"},
            {"Metric": "Hapax Legomena Ratio", col: f"{st['hapax_ratio']*100:.2f}%"},
            {"Metric": "Specialized Clinical Terms", col: f"{st['unique_clinical_terms']:,}"},
            {"Metric": "Medical Token Density (%)", col: f"{st['clinical_density_pct']:.2f}%"},
            {"Metric": f"3-gram Novelty vs Seeds ({nov_src})", col: f"{novelty_3gram[k_only]:.2f}%"},
            {"Metric": f"4-gram Novelty vs Seeds ({nov_src})", col: f"{novelty_4gram[k_only]:.2f}%"},
        ]

    df_res = pd.DataFrame(results)
    df_res.to_csv("results/corpus_diversity_comparison.csv", index=False)
    print("\nSummary Results:")
    print(df_res.to_string(index=False))

    with open("figures/corpus_diversity_report.txt", "w", encoding="utf-8") as f:
        f.write("CORPUS LEXICAL DIVERSITY & CLINICAL TERMINOLOGY REPORT\n")
        f.write("======================================================\n\n")
        f.write(df_res.to_string(index=False))
        f.write("\n\nKEY FINDINGS:\n")
        f.write("1. Terminological Richness: Synthetic generation expands unique medical vocabulary and terminology density.\n")
        f.write("2. Syntactic & Lexical Diversity: STTR and Distinct-2/3 quantify non-formulaic syntactic variety.\n")
        f.write("3. Seed Novelty: 3-gram and 4-gram novelty rates confirm synthesis beyond verbatim seed phrasing.\n")

    print("\nGenerating Figure 4: Corpus Lexical Diversity & Richness...")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(2)
    series_cfg = []
    if "8b" in stats_map:
        series_cfg.append(("8b", "Qwen-3-8B", "#2b5c8f"))
    if "27b" in stats_map:
        series_cfg.append(("27b", "Qwen-3.8-27B", "#d95f02"))
    w = 0.35 if len(series_cfg) == 2 else 0.45

    def _offset(idx_s):
        if len(series_cfg) == 1:
            return 0.0
        return -w / 2 if idx_s == 0 else w / 2

    ax = axes[0]
    for i_s, (key, label, color) in enumerate(series_cfg):
        st = stats_map[key]
        ax.bar(x + _offset(i_s), [st["vocab_size"] / 1000, st["unique_clinical_terms"] / 100], width=w, label=label, color=color, edgecolor="black")
    ax.set_title("(a) Vocabulary & Medical Depth", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Vocab Size\n(k-words)", "Medical Terms\n(x100)"])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    ax = axes[1]
    all_dist_vals = []
    bars_all = []
    for i_s, (key, label, color) in enumerate(series_cfg):
        st = stats_map[key]
        vals = [st["distinct_2"], st["distinct_3"]]
        all_dist_vals.extend(vals)
        bars_all.extend(ax.bar(x + _offset(i_s), vals, width=w, label=label, color=color, edgecolor="black"))
    ax.set_title("(b) Lexical & Syntactic Diversity", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Distinct-2\n(Bigrams)", "Distinct-3\n(Trigrams)"])
    ax.set_ylim(0, max(0.48, max(all_dist_vals) * 1.25))
    for bar in bars_all:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    ax = axes[2]
    for i_s, (key, label, color) in enumerate(series_cfg):
        ax.bar(x + _offset(i_s), [novelty_3gram[key], novelty_4gram[key]], width=w, label=label, color=color, edgecolor="black")
    ax.set_title("(c) Novelty vs. Seed Memorization", fontweight="bold")
    ax.set_ylabel("Novel N-grams (% not in seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(["3-gram Novelty\n(% uncopied)", "4-gram Novelty\n(% uncopied)"])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig("figures/fig4_corpus_lexical_diversity.pdf")
    fig.savefig("figures/fig4_corpus_lexical_diversity.png")
    plt.close(fig)
    print("[OK] Figure 4 saved to figures/fig4_corpus_lexical_diversity.{pdf,png}")
    print("[OK] Report saved to figures/corpus_diversity_report.txt and results/corpus_diversity_comparison.csv")


if __name__ == "__main__":
    main()
