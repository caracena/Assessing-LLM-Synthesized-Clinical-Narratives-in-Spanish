#!/usr/bin/env python3
"""
CLINICAL EVALUATION WITH JEV MODEL (TypeSafe Jev-1.13 Decision Model via OpenRouter)
====================================================================================
Structured probabilistic decision model (System One Decision Model) for auditing
clinical quality on sampled Spanish medical narratives.

Evaluated dimensions (1-to-5 scale mapped from calibrated probabilities):
  1. Clinical Plausibility
  2. Professional Medical Register
  3. Internal Consistency
  4. Document Completeness
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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

API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL_NAME = "typesafe/jev-1.13"

QUESTIONS_SCHEMA = {
    "plausibilidad_clinica": {
        "type": "score",
        "instructions": (
            "Evalúa la plausibilidad clínica de esta nota médica: "
            "¿los síntomas, evolución, hallazgos y diagnóstico son coherentes con la práctica médica real?"
        ),
        "criteria": [
            "Completamente inverosímil o médicamente contradictorio",
            "Poco verosímil con errores clínicos notorios",
            "Aceptable pero con discrepancias menores",
            "Buena plausibilidad clínica y cuadro coherente",
            "Excelente plausibilidad clínica, indistinguible de un caso real",
        ],
    },
    "registro_profesional": {
        "type": "score",
        "instructions": (
            "Evalúa el registro profesional y tono médico: "
            "¿el vocabulario, abreviaturas y estilo reflejan la redacción de un hospital o urgencias?"
        ),
        "criteria": [
            "Registro coloquial o no médico, inadecuado para hospital",
            "Uso inconsistente de terminología médica",
            "Registro aceptable con estilo profesional básico",
            "Buen registro profesional con terminología adecuada",
            "Registro profesional impecable, auténtico lenguaje clínico hospitalario",
        ],
    },
    "consistencia_interna": {
        "type": "score",
        "instructions": (
            "Evalúa la consistencia interna: "
            "¿el texto carece de contradicciones anatómicas, temporales o de lateralidad?"
        ),
        "criteria": [
            "Graves contradicciones anatómicas o temporales",
            "Inconsistencias notables en cronología o lateralidad",
            "Coherencia básica con leves ambigüedades",
            "Alta consistencia sin contradicciones relevantes",
            "Perfecta consistencia interna, cronología y anatomía intachables",
        ],
    },
    "completitud_documental": {
        "type": "score",
        "instructions": (
            "Evalúa la completitud documental de la nota clínica: "
            "¿presenta una estructura adecuada con motivo de consulta, exploración y plan?"
        ),
        "criteria": [
            "Nota fragmentaria o carente de elementos esenciales",
            "Estructura deficiente donde faltan partes clave",
            "Estructura básica pero funcional",
            "Buena completitud documental con relato bien estructurado",
            "Estructura clínica completa, detallada y exhaustiva",
        ],
    },
}

_thread_local = threading.local()


def _get_session(api_key: str) -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/caracena/Assessing-LLM-Synthesized-Clinical-Narratives-in-Spanish",
            "X-Title": "Clinical-Synth-Spanish-Jev-Eval",
        })
        _thread_local.session = s
    return _thread_local.session


def evaluate_note(text: str, api_key: str, max_retries: int = 4):
    """Call OpenRouter TypeSafe Jev-1.13 Decisions API to score a clinical note."""
    payload = {
        "model": MODEL_NAME,
        "state": {"clinical_note": text},
        "questions": QUESTIONS_SCHEMA,
    }
    session = _get_session(api_key)

    for attempt in range(max_retries):
        try:
            resp = session.post(API_URL, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                answers = data.get("answers", {})
                scores = {}
                confidences = {}
                for dim in QUESTIONS_SCHEMA:
                    ans = answers.get(dim, {})
                    # Jev returns 0..4 rubric score; map onto 1..5 Likert scale
                    base_score = float(ans.get("score", 2.0))
                    scores[dim] = round(base_score + 1.0, 3)
                    confidences[dim] = float(ans.get("confidence", 0.0))
                return scores, confidences, data.get("usage", {}), True
            elif resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt + 0.5)
            else:
                print(f"  [API Error {resp.status_code}]: {resp.text[:160]}")
                time.sleep(1.5)
        except Exception as e:
            print(f"  [Exception attempt {attempt + 1}]: {e}")
            time.sleep(1.5)

    return (
        {dim: 3.0 for dim in QUESTIONS_SCHEMA},
        {dim: 0.5 for dim in QUESTIONS_SCHEMA},
        {},
        False,
    )


def save_outputs(records_by_id: dict, ordered_ids: list):
    ordered_records = [records_by_id[i] for i in ordered_ids if i in records_by_id]
    df = pd.DataFrame(ordered_records)
    df.to_csv("results/jev_scores_detailed.csv", index=False)
    with open("results/jev_scores_detailed.json", "w", encoding="utf-8") as f:
        json.dump(ordered_records, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Blinded clinical quality evaluation with TypeSafe Jev-1.13")
    parser.add_argument(
        "--input-file",
        type=str,
        default="results/geval_sample_data_5200.json",
        help="Path to sampled clinical notes JSON",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="results/jev_scores_detailed.json",
        help="Existing cached scores JSON to resume by note ID",
    )
    parser.add_argument("--max-workers", type=int, default=16, help="Concurrent HTTP workers (default: 16)")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save intermediate results every N notes")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "ERROR: Set OPENROUTER_API_KEY in your environment before running run_jev_eval.py."
        )

    print("=" * 75)
    print("BLINDED CLINICAL EVALUATION (TypeSafe Jev-1.13 via OpenRouter)")
    print("=" * 75)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    input_file = args.input_file
    if not os.path.exists(input_file):
        for cand in ("results/geval_sample_data_260.json", "results/geval_sample_data.json"):
            if os.path.exists(cand):
                input_file = cand
                break
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input sample file not found: {args.input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    ordered_ids = [s["id"] for s in samples]
    sample_id_set = set(ordered_ids)

    cached_by_id = {}
    if args.resume_from and os.path.exists(args.resume_from):
        with open(args.resume_from, "r", encoding="utf-8") as f:
            prev = json.load(f)
        for rec in prev:
            if rec.get("id") in sample_id_set:
                cached_by_id[rec["id"]] = rec

    pending = [s for s in samples if s["id"] not in cached_by_id]
    n_8b = sum(1 for s in samples if s.get("generator") == "8B")
    n_27b = sum(1 for s in samples if s.get("generator") == "27B")

    print(f"Input file:    {input_file}")
    print(f"Total notes:   {len(samples)} ({n_8b} from 8B, {n_27b} from 27B)")
    print(f"Cached notes:  {len(cached_by_id)}")
    print(f"Pending API:   {len(pending)} (workers={args.max_workers})")

    t0 = time.time()
    total_cost = 0.0
    failed_count = 0
    completed_new = 0
    lock = threading.Lock()

    def _worker(sample):
        scores, confs, usage, ok = evaluate_note(sample["text"], api_key)
        cost = float(usage.get("cost", 0.0) or 0.0)
        rec = {
            "id": sample["id"],
            "generator": sample["generator"],
            "dataset": sample["dataset"],
            "plausibilidad_clinica": scores["plausibilidad_clinica"],
            "registro_profesional": scores["registro_profesional"],
            "consistencia_interna": scores["consistencia_interna"],
            "completitud_documental": scores["completitud_documental"],
            "conf_plausibilidad": confs["plausibilidad_clinica"],
            "conf_registro": confs["registro_profesional"],
            "conf_consistencia": confs["consistencia_interna"],
            "conf_completitud": confs["completitud_documental"],
        }
        return rec, cost, ok

    if pending:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [pool.submit(_worker, s) for s in pending]
            for fut in as_completed(futures):
                rec, cost, ok = fut.result()
                with lock:
                    cached_by_id[rec["id"]] = rec
                    total_cost += cost
                    completed_new += 1
                    if not ok:
                        failed_count += 1
                    if (
                        completed_new % args.checkpoint_every == 0
                        or completed_new == len(pending)
                    ):
                        save_outputs(cached_by_id, ordered_ids)
                        elapsed = time.time() - t0
                        rate = completed_new / max(elapsed, 1e-6)
                        print(
                            f"  Progress: {len(cached_by_id)}/{len(samples)} "
                            f"(+{completed_new} new in {elapsed:.1f}s, {rate:.1f} req/s, "
                            f"new cost: ${total_cost:.5f}, failures: {failed_count})"
                        )
    else:
        save_outputs(cached_by_id, ordered_ids)

    def _mean_std(vals):
        return (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )

    def _safe_ttest(vals_a, vals_b):
        if len(vals_a) >= 2 and len(vals_b) >= 2:
            t_stat, p_val = stats.ttest_ind(vals_a, vals_b)
            return float(t_stat), float(p_val)
        return float("nan"), float("nan")

    ordered_records = [cached_by_id[i] for i in ordered_ids]
    df = pd.DataFrame(ordered_records)
    df_8b = df[df["generator"] == "8B"].copy()
    df_27b = df[df["generator"] == "27B"].copy()

    has_8b = len(df_8b) > 0
    has_27b = len(df_27b) > 0
    if not (has_8b or has_27b):
        raise ValueError("No records found for generator '8B' or '27B'.")

    dimensions = [
        ("plausibilidad_clinica", "Clinical Plausibility"),
        ("registro_profesional", "Professional Medical Register"),
        ("consistencia_interna", "Internal Consistency"),
        ("completitud_documental", "Document Completeness"),
    ]
    dim_cols = [d[0] for d in dimensions]
    dim_labels = [
        "Clinical\nPlausibility",
        "Professional\nRegister",
        "Internal\nConsistency",
        "Document\nCompleteness",
    ]

    summary_rows = []
    print("\n" + "=" * 75)
    print(f"JEV-1.13 BLINDED EVALUATION SUMMARY [N={len(df)}]")
    print("=" * 75)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(dim_labels))

    if has_8b and has_27b:
        for col, name in dimensions:
            vals_8b = df_8b[col].values
            vals_27b = df_27b[col].values

            mean_8b, std_8b = _mean_std(vals_8b)
            mean_27b, std_27b = _mean_std(vals_27b)
            delta = mean_27b - mean_8b
            t_stat, p_val = _safe_ttest(vals_27b, vals_8b)
            is_sig = (not np.isnan(p_val)) and (p_val < 0.05)

            print(f"\nDimension: {name}")
            print(f"  Qwen-3-8B (N={len(vals_8b)}):    {mean_8b:.3f} ± {std_8b:.3f}")
            print(f"  Qwen-3.8-27B (N={len(vals_27b)}): {mean_27b:.3f} ± {std_27b:.3f}")
            print(
                f"  Delta 27B: {delta:+.3f} pts | t={t_stat:.3f}, p={p_val:.4e} "
                f"{'*' if is_sig else ''}"
            )

            summary_rows.append({
                "Dimension": name,
                "8B_Mean": round(mean_8b, 3),
                "8B_Std": round(std_8b, 3),
                "27B_Mean": round(mean_27b, 3),
                "27B_Std": round(std_27b, 3),
                "Delta_27B": round(delta, 3),
                "t_stat": round(t_stat, 3) if not np.isnan(t_stat) else np.nan,
                "p_value": float(f"{p_val:.4e}") if not np.isnan(p_val) else np.nan,
                "Significant": "Yes" if is_sig else "No",
            })

        overall_8b = df_8b[dim_cols].mean(axis=1).values
        overall_27b = df_27b[dim_cols].mean(axis=1).values
        o_mean_8b, o_std_8b = _mean_std(overall_8b)
        o_mean_27b, o_std_27b = _mean_std(overall_27b)
        o_delta = o_mean_27b - o_mean_8b
        o_t, o_p = _safe_ttest(overall_27b, overall_8b)
        o_sig = (not np.isnan(o_p)) and (o_p < 0.05)

        print("\nDimension: Overall Clinical Score")
        print(f"  Qwen-3-8B (N={len(overall_8b)}):    {o_mean_8b:.3f} ± {o_std_8b:.3f}")
        print(f"  Qwen-3.8-27B (N={len(overall_27b)}): {o_mean_27b:.3f} ± {o_std_27b:.3f}")
        print(f"  Delta 27B: {o_delta:+.3f} pts | t={o_t:.3f}, p={o_p:.4e}")

        summary_rows.append({
            "Dimension": "Overall Clinical Score",
            "8B_Mean": round(o_mean_8b, 3),
            "8B_Std": round(o_std_8b, 3),
            "27B_Mean": round(o_mean_27b, 3),
            "27B_Std": round(o_std_27b, 3),
            "Delta_27B": round(o_delta, 3),
            "t_stat": round(o_t, 3) if not np.isnan(o_t) else np.nan,
            "p_value": float(f"{o_p:.4e}") if not np.isnan(o_p) else np.nan,
            "Significant": "Yes" if o_sig else "No",
        })

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv("results/jev_eval_summary.csv", index=False)

        with open("figures/jev_eval_report.txt", "w", encoding="utf-8") as f:
            f.write(
                f"BLINDED CLINICAL EVALUATION REPORT "
                f"(TypeSafe Jev-1.13, N={len(df)})\n"
            )
            f.write("=" * 72 + "\n\n")
            f.write(df_summary.to_string(index=False))
            f.write(f"\n\nNew API call cost: ${total_cost:.5f} USD | Failures: {failed_count}\n")
            f.write("\nKEY SUMMARY:\n")
            f.write(
                "1. Structured Decision Model: TypeSafe Jev-1.13 provides directly "
                "calibrated probabilistic rubric scores.\n"
            )
            for idx, (row_idx, label) in enumerate(
                [(0, "Clinical Plausibility"), (1, "Professional Register"), (4, "Overall Clinical Score")],
                start=2,
            ):
                row = df_summary.loc[row_idx]
                sig_flag = "p < 0.05" if row["Significant"] == "Yes" else "p >= 0.05"
                f.write(
                    f"{idx}. {label}: 27B achieved {row['27B_Mean']:.3f} vs "
                    f"8B {row['8B_Mean']:.3f} (Delta = {row['Delta_27B']:+.3f} pts, "
                    f"p = {row['p_value']:.4e}, {sig_flag}).\n"
                )

        print("\nGenerating Figure 5: Clinical Quality Comparison (TypeSafe Jev-1.13)...")
        stats_8b = [_mean_std(df_8b[k].values) for k in dim_cols]
        stats_27b = [_mean_std(df_27b[k].values) for k in dim_cols]
        means_8b = [m for m, _ in stats_8b]
        stds_8b = [s for _, s in stats_8b]
        means_27b = [m for m, _ in stats_27b]
        stds_27b = [s for _, s in stats_27b]

        w = 0.35
        bars1 = ax.bar(
            x - w / 2,
            means_8b,
            yerr=stds_8b,
            capsize=4,
            width=w,
            label="Qwen-3-8B Generator",
            color="#2b5c8f",
            edgecolor="black",
            alpha=0.9,
        )
        bars2 = ax.bar(
            x + w / 2,
            means_27b,
            yerr=stds_27b,
            capsize=4,
            width=w,
            label="Qwen-3.8-27B Generator",
            color="#d95f02",
            edgecolor="black",
            alpha=0.9,
        )

        for bars, means in ((bars1, means_8b), (bars2, means_27b)):
            for bar, val in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(1.15, val - 0.45),
                    f"{val:.2f}",
                    ha="center",
                    va="top",
                    color="white",
                    fontweight="bold",
                    fontsize=9.5,
                )
    else:
        gen_key = "8B" if has_8b else "27B"
        gen_label = "Qwen-3-8B" if has_8b else "Qwen-3.8-27B"
        gen_color = "#2b5c8f" if has_8b else "#d95f02"
        df_single = df_8b if has_8b else df_27b

        print(
            f"[INFO] Single-generator mode ({gen_label} only; "
            f"paired 8B vs 27B comparison skipped)."
        )

        for col, name in dimensions:
            vals = df_single[col].values
            mean_val, std_val = _mean_std(vals)
            print(f"\nDimension: {name}")
            print(f"  {gen_label} (N={len(vals)}): {mean_val:.3f} ± {std_val:.3f}")
            summary_rows.append({
                "Dimension": name,
                "Generator": gen_key,
                "N": len(vals),
                "Mean": round(mean_val, 3),
                "Std": round(std_val, 3),
            })

        overall_vals = df_single[dim_cols].mean(axis=1).values
        o_mean, o_std = _mean_std(overall_vals)
        print("\nDimension: Overall Clinical Score")
        print(f"  {gen_label} (N={len(overall_vals)}): {o_mean:.3f} ± {o_std:.3f}")
        summary_rows.append({
            "Dimension": "Overall Clinical Score",
            "Generator": gen_key,
            "N": len(overall_vals),
            "Mean": round(o_mean, 3),
            "Std": round(o_std, 3),
        })

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv("results/jev_eval_summary.csv", index=False)

        with open("figures/jev_eval_report.txt", "w", encoding="utf-8") as f:
            f.write(
                f"BLINDED CLINICAL EVALUATION REPORT "
                f"(TypeSafe Jev-1.13, Single-Generator {gen_label}, N={len(df_single)})\n"
            )
            f.write("=" * 72 + "\n\n")
            f.write(df_summary.to_string(index=False))
            f.write(f"\n\nNew API call cost: ${total_cost:.5f} USD | Failures: {failed_count}\n")
            f.write("\nKEY SUMMARY:\n")
            f.write(
                "1. Structured Decision Model: TypeSafe Jev-1.13 provides directly "
                "calibrated probabilistic rubric scores.\n"
            )
            f.write(
                f"2. Overall Clinical Score ({gen_label}, N={len(overall_vals)}): "
                f"{o_mean:.3f} ± {o_std:.3f}.\n"
            )
            f.write(
                f"3. Highest Dimension: {df_summary.iloc[:4].sort_values('Mean', ascending=False).iloc[0]['Dimension']} "
                f"({df_summary.iloc[:4]['Mean'].max():.3f}); Paired 8B vs 27B comparison skipped in single-generator mode.\n"
            )

        print("\nGenerating Figure 5: Clinical Quality Profile (TypeSafe Jev-1.13)...")
        stats_single = [_mean_std(df_single[k].values) for k in dim_cols]
        means_single = [m for m, _ in stats_single]
        stds_single = [s for _, s in stats_single]

        w = 0.5
        bars = ax.bar(
            x,
            means_single,
            yerr=stds_single,
            capsize=4,
            width=w,
            label=f"{gen_label} Generator",
            color=gen_color,
            edgecolor="black",
            alpha=0.9,
        )
        for bar, val in zip(bars, means_single):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(1.15, val - 0.35),
                f"{val:.2f}",
                ha="center",
                va="top",
                color="white",
                fontweight="bold",
                fontsize=9.5,
            )

    ax.set_title(
        f"Clinical Quality Assessment (TypeSafe Jev-1.13 Decision Model, N={len(samples):,})",
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel(r"Score (1 to 5 Likert Scale, $\mu \pm \sigma$)")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.set_ylim(1.0, 5.3)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig("figures/fig5_geval_quality_comparison.pdf")
    fig.savefig("figures/fig5_geval_quality_comparison.png")
    plt.close(fig)

    print("[OK] Figure 5 saved to figures/fig5_geval_quality_comparison.{pdf,png}")
    print("[OK] Report saved to figures/jev_eval_report.txt")
    print("[OK] Summary CSV saved to results/jev_eval_summary.csv")


if __name__ == "__main__":
    main()
