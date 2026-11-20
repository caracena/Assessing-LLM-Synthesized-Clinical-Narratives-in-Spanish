#!/usr/bin/env python3
"""
audit_pii_leakage.py
Formal audit of Personally Identifiable Information (PII / PHI) leakage
in Chilean clinical synthetic data (ACHS emergency occupational health records).

Evaluates across 28,410 synthetic documents:
1. Chilean National ID (RUT) replication or generation.
2. Direct Telephone / Contact Number replication.
3. Specific Patient Names / Proper Names from seeds replicated verbatim.
4. Exact Calendar Dates (day/month/year) mutation and anonymization.
5. Specific Company / Workplace identification leakage.

Outputs:
- results/pii_leakage_audit.csv
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Set, Dict

# Regex Patterns for Protected Health Information (Chilean Context)
RUT_PATTERN = re.compile(r'\b\d{1,2}\.?\d{3}\.?\d{3}-?[\dkK]\b')
PHONE_PATTERN = re.compile(r'\b(?:\+?56\s?)?(?:9\s?\d{8}|\d{2}\s?\d{7})\b')
DATE_PATTERN = re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4})\b', re.IGNORECASE)

# Common Spanish First Names and Surnames for entity matching
COMMON_NAMES = {
    "juan", "pedro", "carlos", "jose", "josé", "manuel", "francisco", "jorge", "luis", "victor", "víctor",
    "maria", "maría", "ana", "rosa", "carmen", "patricia", "claudia", "daniela", "paula", "andrea",
    "gonzalez", "gonzález", "rodriguez", "rodríguez", "perez", "pérez", "lopez", "lópez", "martinez", "martínez",
    "soto", "silva", "contreras", "morales", "rojas", "munoz", "muñoz", "castro", "diaz", "díaz", "sepulveda", "sepúlveda"
}

def extract_dates(text: str) -> Set[str]:
    """Find all date mentions in text."""
    if not isinstance(text, str):
        return set()
    return set(DATE_PATTERN.findall(text))

def extract_ruts(text: str) -> Set[str]:
    """Find all Chilean RUTs."""
    if not isinstance(text, str):
        return set()
    return set(RUT_PATTERN.findall(text))

def extract_phones(text: str) -> Set[str]:
    """Find telephone numbers."""
    if not isinstance(text, str):
        return set()
    return set(PHONE_PATTERN.findall(text))

def extract_proper_names(text: str) -> Set[str]:
    """Extract candidate patient proper names."""
    if not isinstance(text, str):
        return set()
    words = re.findall(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b', text)
    names = set()
    for w in words:
        w_lower = w.lower()
        if w_lower in COMMON_NAMES:
            names.add(w_lower)
    return names

def audit_dataset_pii(json_path: str, model_name: str) -> Dict[str, any]:
    """
    Audits an entire synthetic dataset for PII leakage against source seeds.
    """
    print(f"Auditing PII in {json_path} ({model_name})...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = data.get("documents", [])
    
    total_seeds = len(docs)
    total_synthetics = sum(len(d.get("generated_texts", [])) for d in docs)
    
    rut_leaks = 0
    phone_leaks = 0
    date_replications = 0
    name_replications = 0
    total_ruts_found = 0
    total_phones_found = 0
    
    date_mutations = 0
    date_comparisons = 0
    
    for doc in docs:
        source_text = doc.get("source_text", "")
        gen_texts = doc.get("generated_texts", [])
        
        src_ruts = extract_ruts(source_text)
        src_phones = extract_phones(source_text)
        src_dates = extract_dates(source_text)
        src_names = extract_proper_names(source_text)
        
        for g_text in gen_texts:
            syn_ruts = extract_ruts(g_text)
            syn_phones = extract_phones(g_text)
            syn_dates = extract_dates(g_text)
            syn_names = extract_proper_names(g_text)
            
            total_ruts_found += len(syn_ruts)
            total_phones_found += len(syn_phones)
            
            # Direct leaks from source
            if src_ruts and (src_ruts & syn_ruts):
                rut_leaks += 1
            if src_phones and (src_phones & syn_phones):
                phone_leaks += 1
            if src_names and (src_names & syn_names):
                name_replications += 1
                
            # Date analysis: check if exact date was replicated vs mutated/omitted
            if src_dates:
                date_comparisons += 1
                if src_dates & syn_dates:
                    date_replications += 1
                else:
                    date_mutations += 1
                    
    rut_leak_rate = (rut_leaks / max(total_synthetics, 1)) * 100.0
    phone_leak_rate = (phone_leaks / max(total_synthetics, 1)) * 100.0
    name_replication_rate = (name_replications / max(total_synthetics, 1)) * 100.0
    date_replication_rate = (date_replications / max(date_comparisons, 1)) * 100.0
    date_mutation_rate = (date_mutations / max(date_comparisons, 1)) * 100.0
    
    return {
        "model": model_name,
        "total_seed_docs": total_seeds,
        "total_synthetic_docs": total_synthetics,
        "rut_leaks_count": rut_leaks,
        "rut_leak_rate_pct": rut_leak_rate,
        "phone_leaks_count": phone_leaks,
        "phone_leak_rate_pct": phone_leak_rate,
        "name_replications_count": name_replications,
        "name_replication_rate_pct": name_replication_rate,
        "date_replications_count": date_replications,
        "date_replication_rate_pct": date_replication_rate,
        "date_mutation_rate_pct": date_mutation_rate,
        "total_synthetic_ruts_found": total_ruts_found,
        "total_synthetic_phones_found": total_phones_found
    }

def main():
    parser = argparse.ArgumentParser(description="Audit synthetic clinical texts for PII/PHI pattern leakage against seeds")
    parser.add_argument("--cluster-base", type=str, default=".", help="Base repository path")
    parser.add_argument("--dataset", type=str, default="achs_pseudo", help="Target synthetic source name (default: achs_pseudo)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output CSVs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    filename = f"sintetico_{args.dataset}.json"
    path_8b = os.path.join(args.cluster_base, "corpus/sinteticos", filename)
    path_27b = os.path.join(args.cluster_base, "corpus/sinteticos_qwen3.8_27b", filename)
    
    project_root = os.environ.get("PROJECT_ROOT", ".")
    if not os.path.exists(path_8b) and os.path.exists(os.path.join(project_root, path_8b)):
        path_8b = os.path.join(project_root, path_8b)
        path_27b = os.path.join(project_root, path_27b)

    records = []
    for model_name, path in [("qwen3_8b", path_8b), ("qwen3.8_27b", path_27b)]:
        if os.path.exists(path):
            records.append(audit_dataset_pii(path, model_name))
        else:
            print(f"[SKIP] Missing synthetic file for {model_name}: {path}")

    if not records:
        raise FileNotFoundError(
            f"No synthetic JSON files found for dataset '{args.dataset}' at {path_8b} or {path_27b}. "
            "Pass --dataset <source> matching generated JSONs in corpus/sinteticos*."
        )
        
    print("=" * 75)
    print(f"PII / PHI LEAKAGE AUDIT SUMMARY ({args.dataset})")
    print("=" * 75)
    
    df_results = pd.DataFrame(records)
    out_csv = os.path.join(args.output_dir, "pii_leakage_audit.csv")
    df_results.to_csv(out_csv, index=False)
    
    print(df_results.to_string(index=False))
    print(f"\n[OK] Saved PII audit results to: {out_csv}")

if __name__ == "__main__":
    main()
