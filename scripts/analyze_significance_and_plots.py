#!/usr/bin/env python3
"""Statistical hypothesis testing (Table 10, Appendix B) and publication figure generation.

Can either read downstream benchmark CSVs produced in Step 4 (`--results-csv`)
or evaluate the exact manuscript benchmark measurements (`--use-paper-results`,
default when `--results-csv` is omitted).

Generates:
  - figures/significance_report.txt
  - figures/fig1_density_scaling.{pdf,png}
  - figures/fig2_head_to_head_baselines.{pdf,png}
  - figures/fig6_sample_efficiency_curve.{pdf,png}
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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

# Canonical manuscript multi-seed benchmark scores (Seeds 42, 123, 2024; Table 2 & Appendix B)
PAPER_MULTISEED = {
    "RigoBERTa-Clinical": {
        "achs": np.array([0.3630, 0.3901, 0.3918]),
        "carmen_i": np.array([0.7407, 0.7258, 0.7402]),
    },
    "Control (XLM-R Large)": {
        "achs": np.array([0.3824, 0.3898, 0.4063]),
        "carmen_i": np.array([0.7125, 0.7383, 0.7240]),
    },
    "CLIN-X-ES": {
        "achs": np.array([0.3882, 0.3743, 0.4259]),
        "carmen_i": np.array([0.7355, 0.7162, 0.7411]),
    },
    "Real Seeds Control": {
        "achs": np.array([0.4102, 0.3642, 0.3991]),
        "carmen_i": np.array([0.7460, 0.7239, 0.7317]),
    },
    "8B (N=1)": {
        "achs": np.array([0.4007, 0.3899, 0.3723]),
        "carmen_i": np.array([0.7256, 0.7430, 0.7421]),
    },
    "8B (N=5)": {
        "achs": np.array([0.3891, 0.3965, 0.3882]),
        "carmen_i": np.array([0.7360, 0.7493, 0.7329]),
    },
    "Flagship 8B (N=10)": {
        "achs": np.array([0.3940, 0.3981, 0.3676]),
        "carmen_i": np.array([0.7545, 0.7508, 0.7614]),
    },
    "27B (N=1)": {
        "achs": np.array([0.3654, 0.3810, 0.3654]),
        "carmen_i": np.array([0.7373, 0.7657, 0.7500]),
    },
    "27B (N=5)": {
        "achs": np.array([0.3913, 0.3921, 0.3920]),
        "carmen_i": np.array([0.7335, 0.7560, 0.7680]),
    },
    "Flagship 27B (N=10)": {
        "achs": np.array([0.3934, 0.4057, 0.4154]),
        "carmen_i": np.array([0.7677, 0.7588, 0.7693]),
    },
}

PAPER_SAMPLE_EFFICIENCY = {
    "frac_pct": [10, 25, 50, 100],
    "ctrl": [0.1256, 0.2063, 0.3098, 0.3824],
    "rigo": [0.1456, 0.2183, 0.2861, 0.3630],
    "flagship": [0.1570, 0.2124, 0.2865, 0.3934],
}


def paired_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def pooled_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / max(nx + ny - 2, 1))
    return float((np.mean(x) - np.mean(y)) / (sp + 1e-12))


MODEL_ALIASES = {
    "IIC/RigoBERTa-Clinical": "RigoBERTa-Clinical",
    "llange/xlm-roberta-large-spanish-clinical": "CLIN-X-ES",
    "FacebookAI/xlm-roberta-large": "Control (XLM-R Large)",
}


def load_multiseed_from_csv(csv_path: str, target_model: str = None):
    """Load evaluated Step 4 CSV metrics, overriding multi-seed arrays when complete 3-seed ACHS+CARMEN-I data exists."""
    df = pd.read_csv(csv_path)
    data = {k: {ds: v.copy() for ds, v in d.items()} for k, d in PAPER_MULTISEED.items()}
    target_seeds = [42, 123, 2024]

    if "train_fraction" in df.columns:
        frac = pd.to_numeric(df["train_fraction"], errors="coerce").fillna(1.0)
        df_full = df[frac >= 0.999].copy()
    else:
        df_full = df.copy()

    complete_models = {}
    for model_label, sub in df_full.groupby("model"):
        raw_label = str(model_label)
        mapped = MODEL_ALIASES.get(raw_label, raw_label)
        entry = {}
        for ds in ("achs", "carmen_i"):
            ds_rows = sub[(sub["dataset"] == ds) & (sub["seed"].isin(target_seeds))].sort_values("seed")
            if len(ds_rows) == 3:
                entry[ds] = ds_rows["f1"].astype(float).values
        if "achs" in entry and "carmen_i" in entry:
            data[mapped] = entry
            complete_models[raw_label] = (mapped, entry)

    selected_target = None
    if target_model and target_model in complete_models:
        mapped, entry = complete_models[target_model]
        data["Flagship 27B (N=10)"] = entry
        selected_target = target_model
    elif not target_model:
        baseline_keys = set(MODEL_ALIASES.values()) | {"RigoBERTa-Clinical", "CLIN-X-ES", "Control (XLM-R Large)", "Real Seeds Control"}
        for raw_label, (mapped, entry) in complete_models.items():
            if mapped not in baseline_keys:
                data["Flagship 27B (N=10)"] = entry
                selected_target = raw_label
                break

    summary_rows = []
    for (model_label, ds_name), grp in df_full.groupby(["model", "dataset"]):
        f1_vals = grp["f1"].astype(float).values
        summary_rows.append({
            "Model": str(model_label),
            "Dataset": str(ds_name),
            "N": len(f1_vals),
            "Mean_F1": round(float(np.mean(f1_vals)), 4),
            "Std_F1": round(float(np.std(f1_vals, ddof=1)), 4) if len(f1_vals) > 1 else 0.0,
            "Peak_F1": round(float(np.max(f1_vals)), 4),
        })
    eval_summary_df = pd.DataFrame(summary_rows)
    return data, selected_target, eval_summary_df


def load_sample_efficiency_override(csv_path: str, target_model: str = None):
    """Optionally load 10/25/50/100% ACHS sample efficiency points from Step 4.4 CSV."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "train_fraction" not in df.columns or "dataset" not in df.columns:
        return None
    sub = df[df["dataset"] == "achs"].copy()
    if target_model:
        sub = sub[sub["model"] == target_model]
    elif len(sub["model"].unique()) > 1:
        return None
    if sub.empty:
        return None
    sub["frac"] = pd.to_numeric(sub["train_fraction"], errors="coerce")
    vals = []
    for req in (0.10, 0.25, 0.50, 1.00):
        m = sub[np.isclose(sub["frac"], req, atol=0.02)]
        if m.empty:
            return None
        vals.append(float(m["f1"].astype(float).mean()))
    return vals


def main():
    parser = argparse.ArgumentParser(description="Compute statistical significance and generate Figures 1, 2, and 6")
    parser.add_argument("--results-csv", default=None, help="Optional Step 4 evaluation CSV to summarize dynamically")
    parser.add_argument("--target-model", default=None, help="Model ID from --results-csv to evaluate as the flagship target")
    parser.add_argument("--sample-efficiency-csv", default=None, help="Optional Step 4.4 ACHS sample-efficiency CSV for Figure 6")
    parser.add_argument("--use-paper-results", action="store_true", help="Use canonical paper multi-seed arrays (default if --results-csv is omitted)")
    parser.add_argument("--fig-dir", default="figures", help="Output directory for figures and report")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    eval_summary_df = pd.DataFrame()
    selected_target = None
    if args.results_csv and not args.use_paper_results:
        if not os.path.exists(args.results_csv):
            raise FileNotFoundError(f"Results CSV not found: {args.results_csv}")
        data_multiseed, selected_target, eval_summary_df = load_multiseed_from_csv(
            args.results_csv, target_model=args.target_model
        )
        if selected_target:
            print(f"[INFO] Using CSV target model '{selected_target}' for paired multi-seed significance & figures.")
        else:
            print(
                "[INFO] Step 4 CSV lacks complete 3-seed (42, 123, 2024) ACHS+CARMEN-I records for target comparison; "
                "using manuscript multi-seed tables for paired t-tests alongside the evaluated Step 4 CSV summary below."
            )
    else:
        data_multiseed = PAPER_MULTISEED

    flagship_label = selected_target or "27B"
    flagship_27b = data_multiseed["Flagship 27B (N=10)"]
    baselines = [
        ("RigoBERTa-Clinical", "RigoBERTa-Clinical"),
        ("CLIN-X-ES", "CLIN-X-ES"),
        ("Control (XLM-R Large)", "Control (XLM-R Large)"),
        ("Flagship 8B (N=10)", "Flagship 8B (N=10)"),
        ("Real Seeds Control", "Real Seeds Control (Appendix B)"),
    ]
    benchmarks = ["carmen_i", "achs"]
    ds_titles = {
        "carmen_i": "CARMEN-I (Hospital EHR)",
        "achs": "ACHS (Occupational Emergency)",
    }

    print("=" * 75)
    print("STATISTICAL SIGNIFICANCE ANALYSIS (MULTI-SEED SEEDS: 42, 123, 2024)")
    print("=" * 75)

    # Collect the 6 planned primary contrasts for Holm-Bonferroni check (Table 10)
    planned_contrasts = []
    report_lines = [
        f"STATISTICAL SIGNIFICANCE REPORT (TWO-TAILED PAIRED T-TEST, df=2; target={flagship_label})",
        "============================================================================",
        "",
    ]

    for ds in benchmarks:
        report_lines.append(f"--- Benchmark: {ds_titles[ds]} ---")
        print(f"\n--- Benchmark: {ds_titles[ds]} ---")
        f27 = flagship_27b[ds]
        for key, label in baselines:
            if key not in data_multiseed:
                continue
            comp = data_multiseed[key][ds]
            delta_pts = (np.mean(f27) - np.mean(comp)) * 100
            t_stat, p_val = stats.ttest_rel(f27, comp)
            dz = paired_cohens_d(f27, comp)
            dp = pooled_cohens_d(f27, comp)
            line = (
                f"  27B vs {label:<32s} | Delta = {delta_pts:+.2f} pts "
                f"({np.mean(f27):.3f} vs {np.mean(comp):.3f}) | "
                f"t = {t_stat:.2f}, d_z = {dz:.2f}, d_pooled = {dp:.2f}, p = {p_val:.4f}"
            )
            print(line)
            report_lines.append(line)
            if key in ("RigoBERTa-Clinical", "CLIN-X-ES", "Control (XLM-R Large)"):
                planned_contrasts.append((ds_titles[ds], label, float(p_val)))
        report_lines.append("")

    # Holm-Bonferroni step-down check across the 6 planned primary contrasts
    planned_contrasts.sort(key=lambda x: x[2])
    report_lines.append("Holm-Bonferroni Step-Down Procedure (6 primary contrasts, alpha=0.05):")
    print("\nHolm-Bonferroni Step-Down (6 primary contrasts, alpha=0.05):")
    for rank, (ds_name, base_name, p_val) in enumerate(planned_contrasts, start=1):
        alpha_corr = 0.05 / (len(planned_contrasts) - rank + 1)
        survives = "PASS" if p_val < alpha_corr else "NO"
        hb_line = f"  [{rank}] {ds_name} | 27B vs {base_name:<22s} | p={p_val:.4f} vs alpha_adj={alpha_corr:.4f} -> {survives}"
        print(hb_line)
        report_lines.append(hb_line)

    if not eval_summary_df.empty:
        summary_block = eval_summary_df.to_string(index=False)
        print("\n--- Evaluated Step 4 CSV Summary ---")
        print(summary_block)
        report_lines.extend(["", "--- Evaluated Step 4 CSV Summary ---", summary_block])

    report_path = os.path.join(args.fig_dir, "significance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n[OK] Significance report written to {report_path}")

    # Figure 1: Synthetic Density Scaling Curve (N = 1, 5, 10)
    print("\nGenerating Figure 1: Density Scaling Curve (8B vs 27B)...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    n_vals = [1, 5, 10]

    for idx, (ds, title) in enumerate([("achs", "(a) ACHS (Chilean Occupational Emergency)"), ("carmen_i", "(b) CARMEN-I (Hospital EHR)")]):
        f1_8b = [
            np.mean(data_multiseed["8B (N=1)"][ds]),
            np.mean(data_multiseed["8B (N=5)"][ds]),
            np.mean(data_multiseed["Flagship 8B (N=10)"][ds]),
        ]
        f1_27b = [
            np.mean(data_multiseed["27B (N=1)"][ds]),
            np.mean(data_multiseed["27B (N=5)"][ds]),
            np.mean(data_multiseed["Flagship 27B (N=10)"][ds]),
        ]
        ax = axes[idx]
        ax.plot(n_vals, f1_8b, marker="o", linewidth=2.2, color="#2b5c8f", label="Qwen-3-8B Generator")
        ax.plot(n_vals, f1_27b, marker="s", linewidth=2.5, color="#d95f02", label="Qwen-3.8-27B Generator")
        ax.axhline(np.mean(data_multiseed["RigoBERTa-Clinical"][ds]), color="#7570b3", linestyle="--", linewidth=1.8, label="RigoBERTa-Clinical (Real Data)")
        ax.axhline(np.mean(data_multiseed["Control (XLM-R Large)"][ds]), color="#666666", linestyle=":", linewidth=1.5, label="XLM-R Large Control")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Synthetic Density Multiplier ($N$)")
        ax.set_ylabel(r"Mean $F_1$-score ($\mu$)")
        ax.set_xticks(n_vals)
        ax.set_xticklabels(["$N=1$\n(33k docs)", "$N=5$\n(167k docs)", "$N=10$\n(334k docs)"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "fig1_density_scaling.pdf"))
    fig.savefig(os.path.join(args.fig_dir, "fig1_density_scaling.png"))
    plt.close(fig)
    print(f"[OK] Saved {os.path.join(args.fig_dir, 'fig1_density_scaling.{pdf,png}')}")

    # Figure 2: Multi-Seed Head-to-Head Baseline Comparison
    print("Generating Figure 2: Multi-Seed Head-to-Head Baselines...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
    models_order = [
        "Control (XLM-R Large)",
        "CLIN-X-ES",
        "RigoBERTa-Clinical",
        "Flagship 8B (N=10)",
        "27B (N=5)",
        "Flagship 27B (N=10)",
    ]
    colors = ["#999999", "#8da0cb", "#7570b3", "#2b5c8f", "#fc8d62", "#e41a1c"]

    for idx, ds in enumerate(["achs", "carmen_i"]):
        ax = axes[idx]
        means = [np.mean(data_multiseed[m][ds]) for m in models_order]
        stds = [np.std(data_multiseed[m][ds]) for m in models_order]
        peaks = [np.max(data_multiseed[m][ds]) for m in models_order]

        x = np.arange(len(models_order))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
        ax.scatter(x, peaks, color="darkred", marker="D", s=30, zorder=5, label="Peak $F_1$")

        for bar, mean_val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.035,
                f"{mean_val:.4f}",
                ha="center",
                va="top",
                color="white",
                fontweight="bold",
                fontsize=9,
            )

        ax.set_title("(a) ACHS Occupational NER" if ds == "achs" else "(b) CARMEN-I Clinical NER", fontweight="bold")
        ax.set_ylabel(r"Mean $F_1$-score ($\mu \pm \sigma$)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                "Control\n(XLM-R)",
                "CLIN-X-ES\n(Biomedical)",
                "RigoBERTa\n(Real SOTA)",
                "Synth 8B\n($N=10$)",
                "Synth 27B\n($N=5$)",
                "Synth 27B\n($N=10$, Flagship)",
            ],
            rotation=25,
            ha="right",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.set_ylim(0.32, 0.435)
            ax.legend(loc="upper left")
        else:
            ax.set_ylim(0.68, 0.795)

    plt.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "fig2_head_to_head_baselines.pdf"))
    fig.savefig(os.path.join(args.fig_dir, "fig2_head_to_head_baselines.png"))
    plt.close(fig)
    print(f"[OK] Saved {os.path.join(args.fig_dir, 'fig2_head_to_head_baselines.{pdf,png}')}")

    # Figure 6: Low-Resource Sample Efficiency Curve (ACHS NER)
    print("Generating Figure 6: Sample Efficiency Curve (ACHS NER)...")
    frac_pct = PAPER_SAMPLE_EFFICIENCY["frac_pct"]
    ctrl_f1 = PAPER_SAMPLE_EFFICIENCY["ctrl"]
    rigo_f1 = PAPER_SAMPLE_EFFICIENCY["rigo"]
    csv_se = load_sample_efficiency_override(args.sample_efficiency_csv, target_model=args.target_model)
    flagship_f1 = csv_se if csv_se is not None else PAPER_SAMPLE_EFFICIENCY["flagship"]
    if csv_se is not None:
        print(f"[INFO] Figure 6 using evaluated ACHS curve from {args.sample_efficiency_csv}: {csv_se}")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(frac_pct, ctrl_f1, marker="^", markersize=7, linewidth=2.2, linestyle=":", color="#666666", label="Control (xlm-r-large)")
    ax.plot(frac_pct, rigo_f1, marker="o", markersize=7, linewidth=2.4, linestyle="--", color="#7570b3", label="RigoBERTa-Clinical (Real EHR SOTA)")
    ax.plot(frac_pct, flagship_f1, marker="s", markersize=7, linewidth=2.7, linestyle="-", color="#d95f02", label="Flagship 27B (N=10, Ours)")

    label_bbox = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.90)
    ax.annotate(f"{flagship_f1[0]:.4f}", (10, flagship_f1[0]), textcoords="offset points", xytext=(8, 9), ha="left", fontweight="bold", color="#d95f02", fontsize=10, bbox=label_bbox)
    ax.annotate(f"{rigo_f1[0]:.4f}", (10, rigo_f1[0]), textcoords="offset points", xytext=(-8, -3), ha="right", fontweight="bold", color="#7570b3", fontsize=10, bbox=label_bbox)
    ax.annotate(f"{ctrl_f1[0]:.4f}", (10, ctrl_f1[0]), textcoords="offset points", xytext=(8, -14), ha="left", color="#555555", fontsize=10, bbox=label_bbox)
    ax.annotate(f"{flagship_f1[-1]:.4f} (Seed 42)", (100, flagship_f1[-1]), textcoords="offset points", xytext=(-8, 9), ha="right", fontweight="bold", color="#d95f02", fontsize=10, bbox=label_bbox)
    ax.annotate(f"{rigo_f1[-1]:.4f}", (100, rigo_f1[-1]), textcoords="offset points", xytext=(-8, -16), ha="right", fontweight="bold", color="#7570b3", fontsize=10, bbox=label_bbox)

    ax.set_title("Sample Efficiency & Annotation Scaling Curve (ACHS NER)", fontweight="bold", pad=12)
    ax.set_xlabel("Percentage of Labeled Real Training Data Available (%)", labelpad=8)
    ax.set_ylabel(r"Test $F_1$-score on Full Evaluation Split", labelpad=8)
    ax.set_xticks(frac_pct)
    ax.set_xticklabels(["10%\n(142 docs)", "25%\n(357 docs)", "50%\n(714 docs)", "100%\n(1,429 docs)"])
    ax.set_xlim(0, 106)
    ax.set_ylim(0.08, 0.44)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, framealpha=0.95, loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(args.fig_dir, "fig6_sample_efficiency_curve.pdf"))
    fig.savefig(os.path.join(args.fig_dir, "fig6_sample_efficiency_curve.png"))
    plt.close(fig)
    print(f"[OK] Saved {os.path.join(args.fig_dir, 'fig6_sample_efficiency_curve.{pdf,png}')}")
    print("\n[OK] All statistical analyses and figures completed.")


if __name__ == "__main__":
    main()
