#!/usr/bin/env python3
"""
analyze_privacy_memorization.py
Empirical analysis of verbatim memorization and privacy preservation
in Spanish clinical synthetic text generation (Qwen-3-8B vs Qwen-3.8-27B).

Evaluates:
- k-gram overlap (k = 1, 3, 5, 7, 10, 15, 20)
- Longest Common Substring (LCS) in tokens
- Verbatim Memorization Rate (thresholds >= 5, 10, 15, 20 consecutive tokens)
- Exact Match rate
- Numerical / Date mutation rate
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def tokenize(text: str) -> List[str]:
    """Tokenize into lowercased alphanumeric words."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\w+\b', text.lower())

def get_kgrams(tokens: List[str], k: int) -> Set[Tuple[str, ...]]:
    """Return set of token k-grams."""
    if len(tokens) < k:
        return set()
    return {tuple(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}

def compute_lcs_tokens(tokens_src: List[str], tokens_syn: List[str]) -> int:
    """
    Compute length of Longest Common Substring (consecutive tokens).
    Uses efficient hash-table sliding window approach.
    """
    if not tokens_src or not tokens_syn:
        return 0
    
    # Store positions in source
    src_set = set()
    # Fast check: if no unigrams overlap, LCS is 0
    set_src = set(tokens_src)
    set_syn = set(tokens_syn)
    if not (set_src & set_syn):
        return 0
    
    # Dynamic programming with rolling array for maximum consecutive match
    # Since docs are typically 50-300 tokens, 1D DP is fast
    n_src = len(tokens_src)
    n_syn = len(tokens_syn)
    
    # Use binary search on length or suffix-based approach
    # For clinical notes (len < 500), 2D matrix or dict map is instant
    max_len = 0
    # Map token to positions in src
    token_to_idx = {}
    for i, tok in enumerate(tokens_src):
        if tok not in token_to_idx:
            token_to_idx[tok] = []
        token_to_idx[tok].append(i)
        
    dp = [0] * (n_src + 1)
    for j in range(1, n_syn + 1):
        tok_syn = tokens_syn[j - 1]
        new_dp = [0] * (n_src + 1)
        if tok_syn in token_to_idx:
            for i in token_to_idx[tok_syn]:
                src_pos = i + 1
                length = dp[src_pos - 1] + 1
                new_dp[src_pos] = length
                if length > max_len:
                    max_len = length
        dp = new_dp
        
    return max_len

def extract_dates(text: str) -> Set[str]:
    """Extract standard date patterns (DD/MM/YYYY, DD-MM-YYYY, etc.)."""
    return set(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text))

def extract_numbers(text: str) -> Set[str]:
    """Extract isolated numbers / measurements."""
    return set(re.findall(r'\b\d+(?:[.,]\d+)?\b', text))

def evaluate_pairs(pairs: List[Dict], dataset_name: str, model_name: str, max_samples: int = 5000) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate list of {'source_text': str, 'text': str} dictionaries.
    """
    if len(pairs) > max_samples:
        import random
        random.seed(42)
        eval_pairs = random.sample(pairs, max_samples)
    else:
        eval_pairs = pairs
        
    k_values = [1, 3, 5, 7, 10, 15, 20]
    records = []
    
    for i, item in enumerate(eval_pairs):
        src_text = item.get("source_text", "")
        syn_text = item.get("text", "")
        
        src_toks = tokenize(src_text)
        syn_toks = tokenize(syn_text)
        
        if not src_toks or not syn_toks:
            continue
            
        len_src = len(src_toks)
        len_syn = len(syn_toks)
        
        # k-gram overlaps
        k_overlaps = {}
        for k in k_values:
            src_kg = get_kgrams(src_toks, k)
            syn_kg = get_kgrams(syn_toks, k)
            if not syn_kg:
                overlap = 0.0
            else:
                overlap = len(syn_kg & src_kg) / len(syn_kg)
            k_overlaps[f"overlap_{k}gram"] = overlap
            
        # Longest Common Substring (consecutive tokens)
        lcs = compute_lcs_tokens(src_toks, syn_toks)
        norm_lcs = lcs / len_syn if len_syn > 0 else 0.0
        
        # Exact match
        exact_match = 1.0 if src_text.strip() == syn_text.strip() else 0.0
        
        # Date & number overlap / mutation
        src_dates = extract_dates(src_text)
        syn_dates = extract_dates(syn_text)
        date_copied = len(syn_dates & src_dates) > 0 if src_dates else False
        date_mutated = len(syn_dates - src_dates) > 0 if syn_dates else False
        
        rec = {
            "dataset": dataset_name,
            "model": model_name,
            "doc_idx": i,
            "len_src_tokens": len_src,
            "len_syn_tokens": len_syn,
            "lcs_tokens": lcs,
            "norm_lcs": norm_lcs,
            "exact_match": exact_match,
            "has_src_dates": len(src_dates) > 0,
            "date_copied": date_copied,
            "date_mutated": date_mutated,
            **k_overlaps
        }
        records.append(rec)
        
    df = pd.DataFrame(records)
    
    # Compute summary
    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "n_evaluated": len(df),
        "mean_len_src": df["len_src_tokens"].mean(),
        "mean_len_syn": df["len_syn_tokens"].mean(),
        "mean_lcs_tokens": df["lcs_tokens"].mean(),
        "median_lcs_tokens": df["lcs_tokens"].median(),
        "p95_lcs_tokens": df["lcs_tokens"].quantile(0.95),
        "mean_norm_lcs": df["norm_lcs"].mean(),
        "verbatim_ge_5": (df["lcs_tokens"] >= 5).mean(),
        "verbatim_ge_10": (df["lcs_tokens"] >= 10).mean(),
        "verbatim_ge_15": (df["lcs_tokens"] >= 15).mean(),
        "verbatim_ge_20": (df["lcs_tokens"] >= 20).mean(),
        "exact_match_rate": df["exact_match"].mean(),
    }
    for k in k_values:
        summary[f"mean_overlap_{k}gram"] = df[f"overlap_{k}gram"].mean()
        
    return df, summary

def main():
    parser = argparse.ArgumentParser(description="Analyze verbatim memorization and privacy")
    parser.add_argument("--cluster-base", type=str, default=os.environ.get("PROJECT_ROOT", "."),
                        help="Project root path")
    parser.add_argument("--max-samples", type=int, default=3000,
                        help="Max samples per dataset")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for CSVs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    datasets_to_test = [
        ("achs_pseudo", "sintetico_achs_pseudo.json"),
        ("cantemist", "sintetico_cantemist.json"),
        ("cares", "sintetico_cares.json"),
        ("spaccc", "sintetico_spaccc.json")
    ]
    
    models = [
        ("qwen3_8b", os.path.join(args.cluster_base, "corpus/sinteticos")),
        ("qwen3.8_27b", os.path.join(args.cluster_base, "corpus/sinteticos_qwen3.8_27b"))
    ]
    
    all_summaries = []
    all_detailed = []
    
    for ds_name, json_filename in datasets_to_test:
        for model_name, model_dir in models:
            json_path = os.path.join(model_dir, json_filename)
            if not os.path.exists(json_path):
                print(f"File not found: {json_path}, skipping...")
                continue
                
            print(f"Loading {model_name} on {ds_name} from {json_path}...")
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            pairs = data.get("synthetic", [])
            print(f"Loaded {len(pairs)} synthetic records. Evaluating up to {args.max_samples}...")
            
            df_det, summary = evaluate_pairs(pairs, ds_name, model_name, max_samples=args.max_samples)
            all_detailed.append(df_det)
            all_summaries.append(summary)
            
            print(f"[{model_name} | {ds_name}] Mean LCS: {summary['mean_lcs_tokens']:.2f} tok | "
                  f"Verbatim>=10: {summary['verbatim_ge_10']*100:.2f}% | "
                  f"5-gram overlap: {summary['mean_overlap_5gram']*100:.2f}% | "
                  f"10-gram overlap: {summary['mean_overlap_10gram']*100:.2f}%")
                  
    if not all_summaries:
        print("No evaluations completed.")
        return
        
    df_summary = pd.DataFrame(all_summaries)
    df_detailed_all = pd.concat(all_detailed, ignore_index=True)
    
    summary_csv = os.path.join(args.output_dir, "privacy_memorization_summary.csv")
    detailed_csv = os.path.join(args.output_dir, "privacy_memorization_detailed.csv")
    
    df_summary.to_csv(summary_csv, index=False)
    # Save head or sample of detailed to keep file size manageable
    df_detailed_all.to_csv(detailed_csv, index=False)
    print(f"\nSaved summary to {summary_csv}")
    print(f"Saved detailed records to {detailed_csv}")
    
    # Generate Figures
    generate_plots(df_summary, df_detailed_all)

def generate_plots(df_summary: pd.DataFrame, df_detailed: pd.DataFrame):
    """Generate publication-ready figures for LaTeX paper."""
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # -------------------------------------------------------------
    # Figure 7: k-gram Overlap Decay Curve (The core proof of non-verbatim)
    # -------------------------------------------------------------
    k_vals = [1, 3, 5, 7, 10, 15, 20]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    palette = {"qwen3_8b": "#1f77b4", "qwen3.8_27b": "#2ca02c"}
    labels = {"qwen3_8b": "Qwen-3-8B Generator", "qwen3.8_27b": "Qwen-3.8-27B Generator (Flagship)"}
    
    # Left subplot: ACHS specifically, or first available dataset if restricted ACHS is absent
    focus_ds = "achs_pseudo" if "achs_pseudo" in df_summary["dataset"].values else df_summary["dataset"].iloc[0]
    focus_title = "(A) ACHS Domain (Chilean Emergency Records)" if focus_ds == "achs_pseudo" else f"(A) {focus_ds} Domain"
    focus_sub = df_summary[df_summary["dataset"] == focus_ds]
    for _, row in focus_sub.iterrows():
        m = row["model"]
        y_vals = [row[f"mean_overlap_{k}gram"] * 100 for k in k_vals]
        ax1.plot(k_vals, y_vals, marker="o", linewidth=2.5, markersize=7,
                 color=palette.get(m, "gray"), label=labels.get(m, m))
                 
    # Add baseline reference line
    ax1.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax1.set_title(focus_title, fontsize=13, fontweight="bold")
    ax1.set_xlabel("k-gram Size (consecutive tokens)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Overlap with Seed Document (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(k_vals)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=10)
    
    # Right subplot: Multi-dataset Average
    mean_by_model = df_summary.groupby("model")[[f"mean_overlap_{k}gram" for k in k_vals]].mean()
    for m in mean_by_model.index:
        y_vals = [mean_by_model.loc[m, f"mean_overlap_{k}gram"] * 100 for k in k_vals]
        ax2.plot(k_vals, y_vals, marker="s", linewidth=2.5, markersize=7,
                 color=palette.get(m, "gray"), label=labels.get(m, m))
                 
    ax2.set_title("(B) Cross-Dataset Global Average (All Sources)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("k-gram Size (consecutive tokens)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Overlap with Seed Document (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(k_vals)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=10)
    
    plt.tight_layout()
    fig_path_png = "figures/fig7_privacy_verbatim_kgrams.png"
    fig_path_pdf = "figures/fig7_privacy_verbatim_kgrams.pdf"
    plt.savefig(fig_path_png, dpi=300)
    plt.savefig(fig_path_pdf)
    plt.close()
    print(f"Saved Figure 7 to {fig_path_png} and {fig_path_pdf}")
    
    # -------------------------------------------------------------
    # Figure 8: Longest Common Subsequence (LCS) Distribution & Verbatim Rates
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    focus_det_label = "achs_pseudo" if "achs_pseudo" in df_detailed["dataset"].values else df_detailed["dataset"].iloc[0]
    focus_det = df_detailed[df_detailed["dataset"] == focus_det_label]
    if not focus_det.empty:
        sns.histplot(data=focus_det, x="lcs_tokens", hue="model", palette=palette,
                     bins=range(0, 35, 1), element="step", stat="density", common_norm=False,
                     ax=ax1, alpha=0.4, linewidth=2)
        ax1.axvline(10, color="crimson", linestyle="--", linewidth=1.8, label="Carlini et al. Verbatim Bound (>=10)")
        lcs_title = "(A) Empirical LCS Distribution (ACHS)" if focus_det_label == "achs_pseudo" else f"(A) Empirical LCS Distribution ({focus_det_label})"
        ax1.set_title(lcs_title, fontsize=13, fontweight="bold")
        ax1.set_xlabel("Longest Verbatim Token Span (tokens)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax1.set_xlim(0, 30)
        ax1.legend(frameon=True, fontsize=9)
        
    # Bar plot of Verbatim Memorization Rate by threshold
    thresholds = ["verbatim_ge_5", "verbatim_ge_10", "verbatim_ge_15", "verbatim_ge_20"]
    thresh_labels = ["Span >= 5", "Span >= 10", "Span >= 15", "Span >= 20"]
    
    rates_data = []
    for _, row in df_summary.iterrows():
        for t, lab in zip(thresholds, thresh_labels):
            rates_data.append({
                "Dataset": row["dataset"],
                "Model": "27B Flagship" if "27b" in row["model"] else "8B Generator",
                "Threshold": lab,
                "Rate (%)": row[t] * 100
            })
    df_rates = pd.DataFrame(rates_data)
    
    sns.barplot(data=df_rates, x="Threshold", y="Rate (%)", hue="Model",
                palette={"8B Generator": "#1f77b4", "27B Flagship": "#2ca02c"}, ax=ax2, errorbar=None)
    ax2.set_title("(B) Memorization Incident Rate across Thresholds", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Verbatim Sequence Threshold", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Documents Matching Threshold (%)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=10)
    
    plt.tight_layout()
    fig_path_png = "figures/fig8_longest_verbatim_distribution.png"
    fig_path_pdf = "figures/fig8_longest_verbatim_distribution.pdf"
    plt.savefig(fig_path_png, dpi=300)
    plt.savefig(fig_path_pdf)
    plt.close()
    print(f"Saved Figure 8 to {fig_path_png} and {fig_path_pdf}")

if __name__ == "__main__":
    main()
