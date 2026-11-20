#!/usr/bin/env python3
"""
analyze_clinical_novelty.py
Empirical analysis of generative novelty, intra-seed diversity, and clinical concept expansion
in Spanish clinical synthetic text generation (Qwen-3-8B vs Qwen-3.8-27B).

Evaluates:
1. Intra-Seed Self-BLEU (Self-BLEU-3 & Self-BLEU-4) across N=10 variants per clinical case.
2. Clinical Concept Expansion (Novel Concept Ratio - NCR & new medical entities per case).
3. Specialized Clinical Terminology & Morphology (prefixes/suffixes) discovery.
4. Statistical significance (Paired t-test and Cohen's d).
5. Generates publication-ready Figure 9 (figures/fig9_clinical_novelty_expansion.{png,pdf}).
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Style configuration matching ACL/Elsevier standards
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

# Specialized Spanish Medical Dictionary & Taxonomies (Occupational health, trauma, pharmacology)
ANATOMY_TERMS = {
    "mano", "muneca", "muñeca", "carpo", "metacarpo", "dedo", "dedos", "pulgar", "indice", "índice",
    "medio", "anular", "menique", "meñique", "falange", "falanges", "brazo", "antebrazo", "codo",
    "hombro", "clavicula", "clavícula", "escapula", "escápula", "pie", "pies", "tobillo", "tarso",
    "metatarso", "talon", "talón", "calcaneo", "calcáneo", "rodilla", "rotula", "rótula", "pierna",
    "muslo", "femur", "fémur", "tibia", "perone", "peroné", "cadera", "pelvis", "columna", "lumbar",
    "dorsal", "cervical", "cuello", "espalda", "cabeza", "craneo", "cráneo", "cara", "ojo", "ojos",
    "nariz", "boca", "diente", "mandibula", "mandíbula", "torax", "tórax", "costilla", "costillas",
    "abdomen", "musculo", "músculo", "tendon", "tendón", "ligamento", "ligamentos", "nervio", "menisco"
}

PATHOLOGY_TRAUMA_TERMS = {
    "fractura", "esguince", "luxacion", "luxación", "subluxacion", "subluxación", "contusion",
    "contusión", "herida", "corte", "laceracion", "laceración", "abrasion", "abrasión", "erosión",
    "quemadura", "traumatismo", "trauma", "politraumatismo", "torcedura", "desgarro", "distension",
    "distensión", "tendinitis", "tenosinovitis", "bursitis", "epicondilitis", "epitrocleitis",
    "sinovitis", "atrapamiento", "sobreesfuerzo", "tiron", "tirón", "lumbago", "lumbalgia",
    "cervicalgia", "dorsalgia", "ciatica", "ciática", "lesion", "lesión", "edema", "hematoma",
    "equimosis", "inflamacion", "inflamación", "infeccion", "infección", "cuerpo extrano", "cuerpo extraño"
}

SIGNS_SYMPTOMS_TERMS = {
    "dolor", "doloroso", "inflamado", "edematoso", "tumefaccion", "tumefacción", "deformidad",
    "limitacion", "limitación", "impotencia", "inestabilidad", "parestesia", "parestesias",
    "adormecimiento", "hormigueo", "sangrado", "hemorragia", "eritema", "equimotico", "equimótico",
    "contractura", "rigidez", "crepitacion", "crepitación", "bloqueo", "claudicacion", "claudicación",
    "claudica", "aumento de volumen", "impotencia funcional", "limitacion funcional"
}

PROCEDURES_MANAGEMENT_TERMS = {
    "radiografia", "radiografía", "rx", "ecografia", "ecografía", "scanner", "tac", "resonancia",
    "inmovilizacion", "inmovilización", "inmoviliza", "yeso", "valva", "ferula", "férula", "bota",
    "vendaje", "cabestrillo", "curacion", "curación", "sutura", "suturas", "drenaje", "aseo",
    "quirurgico", "quirúrgico", "cirugia", "cirugía", "reduccion", "reducción", "osteosintesis",
    "osteosíntesis", "reposo", "licencia", "alta", "derivacion", "derivación", "kinesiterapia",
    "kinesiologia", "kinesiología", "fisioterapia", "rehabilitacion", "rehabilitación"
}

PHARMACOLOGY_TERMS = {
    "paracetamol", "ketorolaco", "ibuprofeno", "tramadol", "diclofenaco", "celecoxib", "naproxeno",
    "meloxicam", "analgesia", "analgesico", "analgésico", "antiinflamatorio", "aine", "aines",
    "amoxicilina", "ciprofloxacino", "cefalexina", "cloxacilina", "antibiotico", "antibiótico",
    "vacuna", "antitetanica", "antitetánica", "toxoide", "morfina", "ciclobenzaprina", "parche"
}

ALL_CLINICAL_ENTITIES = (
    ANATOMY_TERMS | PATHOLOGY_TRAUMA_TERMS | SIGNS_SYMPTOMS_TERMS |
    PROCEDURES_MANAGEMENT_TERMS | PHARMACOLOGY_TERMS
)

CLINICAL_SUFFIXES = (
    "itis", "osis", "oma", "patia", "patía", "ectomia", "ectomía", "tomia", "tomía",
    "scopia", "scopía", "megalia", "algia", "emia", "uria", "plejia", "plejía",
    "paresia", "grafia", "grafía", "terapia", "metria", "metría", "genesis", "génesis"
)
CLINICAL_PREFIXES = (
    "hiper", "hipo", "peri", "intra", "post", "pre", "sub", "retro", "cardio",
    "neuro", "osteo", "gastro", "hemo", "angio", "bronco", "hepato", "encefalo"
)

def tokenize(text: str) -> list:
    """Tokenize lowercase alphanumeric words preserving tildes."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\b[a-záéíóúñA-ZÁÉÍÓÚÑ]+\b', text.lower())

def is_morphological_clinical(word: str) -> bool:
    """Check clinical affix morphology."""
    w = word.lower()
    if len(w) < 5:
        return False
    for s in CLINICAL_SUFFIXES:
        if w.endswith(s):
            return True
    for p in CLINICAL_PREFIXES:
        if w.startswith(p):
            return True
    return False

def extract_clinical_concepts(text: str) -> set:
    """Extract clinical entities and specialized morphological medical terms."""
    tokens = tokenize(text)
    concepts = set()
    text_lower = text.lower()
    
    # Check multi-word entities
    for multi in ["aumento de volumen", "impotencia funcional", "limitacion funcional", "cuerpo extrano", "cuerpo extraño"]:
        if multi in text_lower:
            concepts.add(multi)
            
    # Check single-word entities
    for t in tokens:
        if t in ALL_CLINICAL_ENTITIES:
            concepts.add(t)
        elif is_morphological_clinical(t):
            concepts.add(t)
            
    return concepts

def compute_sentence_bleu(ref_tokens: list, hyp_tokens: list, max_n: int = 4) -> float:
    """
    Computes smoothed BLEU score (Chen & Cherry smoothing) between reference and hypothesis.
    """
    if not ref_tokens or not hyp_tokens:
        return 0.0
    
    bp = 1.0
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len < ref_len:
        bp = np.exp(1.0 - float(ref_len) / float(max(hyp_len, 1)))
        
    precisions = []
    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n or len(ref_tokens) < n:
            precisions.append(1e-5)
            continue
            
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1)])
        
        matches = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        
        # Add-1 smoothing for n > 1
        if matches == 0:
            p_n = 0.5 / total
        else:
            p_n = matches / total
        precisions.append(p_n)
        
    log_prec = np.mean([np.log(p) for p in precisions])
    return bp * np.exp(log_prec)

def compute_intra_seed_self_bleu(generated_texts: list, max_n: int = 4) -> float:
    """
    Calculates pairwise Self-BLEU among the N generated variations for a single seed case.
    """
    n_texts = len(generated_texts)
    if n_texts < 2:
        return 0.0
        
    tokenized_texts = [tokenize(t) for t in generated_texts]
    tokenized_texts = [t for t in tokenized_texts if len(t) > 3]
    if len(tokenized_texts) < 2:
        return 0.0
        
    scores = []
    for i in range(len(tokenized_texts)):
        for j in range(i + 1, len(tokenized_texts)):
            s_ij = compute_sentence_bleu(tokenized_texts[i], tokenized_texts[j], max_n=max_n)
            s_ji = compute_sentence_bleu(tokenized_texts[j], tokenized_texts[i], max_n=max_n)
            scores.append((s_ij + s_ji) / 2.0)
            
    return float(np.mean(scores)) if scores else 0.0

def analyze_dataset_novelty(json_path: str, max_cases: int = 1000) -> pd.DataFrame:
    """
    Analyzes intra-seed diversity and clinical concept expansion across all document cases.
    """
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = data.get("documents", [])
    if max_cases and len(docs) > max_cases:
        docs = docs[:max_cases]
        
    records = []
    for doc in docs:
        source_idx = doc.get("source_index", -1)
        source_text = doc.get("source_text", "")
        gen_texts = doc.get("generated_texts", [])
        
        if not source_text or len(gen_texts) < 2:
            continue
            
        seed_concepts = extract_clinical_concepts(source_text)
        n_seed_concepts = len(seed_concepts)
        
        # Intra-seed diversity
        self_bleu3 = compute_intra_seed_self_bleu(gen_texts, max_n=3)
        self_bleu4 = compute_intra_seed_self_bleu(gen_texts, max_n=4)
        
        # Clinical Concept Expansion
        all_syn_concepts = set()
        novel_concepts_per_var = []
        ncr_per_var = []
        syn_lengths = []
        
        for g_text in gen_texts:
            syn_concepts = extract_clinical_concepts(g_text)
            all_syn_concepts |= syn_concepts
            new_concepts = syn_concepts - seed_concepts
            novel_concepts_per_var.append(len(new_concepts))
            ncr_per_var.append(len(new_concepts) / max(len(syn_concepts), 1))
            syn_lengths.append(len(tokenize(g_text)))
            
        net_new_concepts = len(all_syn_concepts - seed_concepts)
        expansion_factor = (len(all_syn_concepts) / max(n_seed_concepts, 1)) if n_seed_concepts > 0 else 1.0
        
        records.append({
            "source_index": source_idx,
            "n_seed_concepts": n_seed_concepts,
            "mean_syn_length": np.mean(syn_lengths),
            "self_bleu3": self_bleu3,
            "self_bleu4": self_bleu4,
            "intra_seed_diversity_bleu4": 1.0 - self_bleu4,
            "mean_novel_concepts": np.mean(novel_concepts_per_var),
            "mean_ncr": np.mean(ncr_per_var),
            "total_new_concepts_cluster": net_new_concepts,
            "expansion_factor": expansion_factor
        })
        
    return pd.DataFrame(records)

def main():
    parser = argparse.ArgumentParser(description="Analyze intra-seed redundancy and clinical concept expansion")
    parser.add_argument("--cluster-base", type=str, default=".", help="Base repository path")
    parser.add_argument("--dataset", type=str, default="achs_pseudo", help="Synthetic source dataset to analyze (default: achs_pseudo)")
    parser.add_argument("--max-cases", type=int, default=3000, help="Max seed cases to evaluate (default: 3000 covers all 2,812 ACHS clusters)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output results")
    parser.add_argument("--fig-dir", type=str, default="figures", help="Directory for output figures")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    
    filename = f"sintetico_{args.dataset}.json"
    path_8b = os.path.join(args.cluster_base, "corpus/sinteticos", filename)
    path_27b = os.path.join(args.cluster_base, "corpus/sinteticos_qwen3.8_27b", filename)
    
    # Fallback to PROJECT_ROOT if configured
    project_root = os.environ.get("PROJECT_ROOT", ".")
    if not os.path.exists(path_8b) and os.path.exists(os.path.join(project_root, path_8b)):
        path_8b = os.path.join(project_root, path_8b)
    if not os.path.exists(path_27b) and os.path.exists(os.path.join(project_root, path_27b)):
        path_27b = os.path.join(project_root, path_27b)

    has_8b = os.path.exists(path_8b)
    has_27b = os.path.exists(path_27b)
    if not has_8b and not has_27b:
        raise FileNotFoundError(
            f"Missing synthetic JSONs for '{args.dataset}': expected {path_8b} or {path_27b}. "
            "Pass --dataset <source> for an available generated source (e.g. --dataset cantemist)."
        )

    print("=" * 80)
    print(f"CLINICAL NOVELTY, INTRA-SEED DIVERSITY & CONCEPT EXPANSION AUDIT ({args.dataset})")
    print("=" * 80)

    metrics_to_test = [
        ("self_bleu3", "Self-BLEU-3 (Lower is more diverse)"),
        ("self_bleu4", "Self-BLEU-4 (Lower is more diverse)"),
        ("intra_seed_diversity_bleu4", "Intra-Seed Diversity (1 - BLEU-4)"),
        ("mean_novel_concepts", "Mean Novel Medical Concepts Added / Doc"),
        ("mean_ncr", "Novel Concept Ratio (NCR)"),
        ("total_new_concepts_cluster", "Net New Concepts per Cluster (N=10)"),
        ("expansion_factor", "Clinical Concept Expansion Factor")
    ]

    if has_8b and has_27b:
        df_8b = analyze_dataset_novelty(path_8b, max_cases=args.max_cases)
        df_8b["model"] = "qwen3_8b"
        df_27b = analyze_dataset_novelty(path_27b, max_cases=args.max_cases)
        df_27b["model"] = "qwen3.8_27b"

        common_indices = list(set(df_8b["source_index"]) & set(df_27b["source_index"]))
        print(f"\nEvaluated Cases: {len(common_indices)} aligned cases across 8B and 27B.")

        df_8b_aligned = df_8b[df_8b["source_index"].isin(common_indices)].sort_values("source_index").reset_index(drop=True)
        df_27b_aligned = df_27b[df_27b["source_index"].isin(common_indices)].sort_values("source_index").reset_index(drop=True)

        summary_rows = []
        print("\n" + "=" * 80)
        print("COMPARATIVE STATISTICAL RESULTS (Qwen-3-8B vs Qwen-3.8-27B):")
        print("=" * 80)

        for col, desc in metrics_to_test:
            val_8b = df_8b_aligned[col].values
            val_27b = df_27b_aligned[col].values
            mean_8b, std_8b = np.mean(val_8b), np.std(val_8b)
            mean_27b, std_27b = np.mean(val_27b), np.std(val_27b)
            diff = val_27b - val_8b
            t_stat, p_val = stats.ttest_rel(val_27b, val_8b)
            cohen_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0

            summary_rows.append({
                "metric": col,
                "description": desc,
                "mean_8b": mean_8b,
                "std_8b": std_8b,
                "mean_27b": mean_27b,
                "std_27b": std_27b,
                "delta_27b_vs_8b": mean_27b - mean_8b,
                "pct_change": ((mean_27b - mean_8b) / mean_8b * 100) if mean_8b != 0 else 0.0,
                "p_value": p_val,
                "cohen_d": cohen_d
            })
            print(f"{desc}:")
            print(f"  8B:  {mean_8b:.4f} ± {std_8b:.4f}")
            print(f"  27B: {mean_27b:.4f} ± {std_27b:.4f} (Delta: {mean_27b - mean_8b:+.4f}, p = {p_val:.4e}, d = {cohen_d:.2f})")

        df_summary = pd.DataFrame(summary_rows)
        df_detailed = pd.concat([df_8b_aligned, df_27b_aligned], ignore_index=True)
        series_plot = [
            ("Qwen-3-8B", df_8b_aligned, "#4A90E2", f"8B Mean: {df_8b_aligned['self_bleu4'].mean():.3f}"),
            ("Qwen-3.8-27B (Flagship)", df_27b_aligned, "#D0021B", f"27B Mean: {df_27b_aligned['self_bleu4'].mean():.3f}"),
        ]
    else:
        active_path = path_8b if has_8b else path_27b
        model_id = "qwen3_8b" if has_8b else "qwen3.8_27b"
        model_label = "Qwen-3-8B" if has_8b else "Qwen-3.8-27B (Flagship)"
        model_color = "#4A90E2" if has_8b else "#D0021B"
        short_tag = "8B" if has_8b else "27B"
        print(f"[INFO] Single-generator mode ({model_label} only; paired comparison skipped).")

        df_single = analyze_dataset_novelty(active_path, max_cases=args.max_cases)
        df_single["model"] = model_id
        print(f"\nEvaluated Cases: {len(df_single)} clusters for {model_label}.")

        summary_rows = []
        for col, desc in metrics_to_test:
            vals = df_single[col].values
            mean_v, std_v = np.mean(vals), np.std(vals)
            summary_rows.append({
                "metric": col,
                "description": desc,
                "model": model_id,
                "mean": mean_v,
                "std": std_v,
            })
            print(f"{desc}: {mean_v:.4f} ± {std_v:.4f}")

        df_summary = pd.DataFrame(summary_rows)
        df_detailed = df_single
        series_plot = [
            (model_label, df_single, model_color, f"{short_tag} Mean: {df_single['self_bleu4'].mean():.3f}"),
        ]

    df_summary.to_csv(os.path.join(args.output_dir, "clinical_novelty_summary.csv"), index=False)
    df_detailed.to_csv(os.path.join(args.output_dir, "clinical_novelty_detailed.csv"), index=False)

    print(f"\nGenerating Figure 9 in {args.fig_dir}...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

    ax1 = axes[0]
    for label, df_m, color, mean_lbl in series_plot:
        if df_m["self_bleu4"].nunique() > 1:
            sns.kdeplot(df_m["self_bleu4"], ax=ax1, color=color, label=label, fill=True, alpha=0.3, bw_adjust=1.2)
        else:
            sns.histplot(df_m["self_bleu4"], ax=ax1, color=color, label=label, alpha=0.4)
        ax1.axvline(df_m["self_bleu4"].mean(), color=color, linestyle="--", alpha=0.8, label=mean_lbl)
    ax1.set_title("A. Intra-Seed Redundancy (Self-BLEU-4)", fontweight="bold")
    ax1.set_xlabel("Pairwise Self-BLEU-4 Score")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = axes[1]
    model_col, novel_col = [], []
    palette_map = {}
    for label, df_m, color, _ in series_plot:
        model_col.extend([label] * len(df_m))
        novel_col.extend(list(df_m["mean_novel_concepts"]))
        palette_map[label] = color
    plot_data = pd.DataFrame({"Model": model_col, "Novel Concepts": novel_col})
    sns.boxplot(
        data=plot_data, x="Model", y="Novel Concepts", hue="Model", legend=False,
        ax=ax2, palette=palette_map, width=0.45, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"},
    )
    ax2.set_title("B. Novel Medical Concepts per Doc", fontweight="bold")
    ax2.set_xlabel("Generator Model")
    ax2.set_ylabel("Mean Novel Medical Entities Added")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    png_path = os.path.join(args.fig_dir, "fig9_clinical_novelty_expansion.png")
    pdf_path = os.path.join(args.fig_dir, "fig9_clinical_novelty_expansion.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {png_path} and {pdf_path}")
    print("[OK] Clinical Novelty Audit completed successfully.")

if __name__ == "__main__":
    main()
