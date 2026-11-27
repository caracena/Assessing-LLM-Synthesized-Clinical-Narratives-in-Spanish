#!/bin/bash
# Full downstream benchmark evaluation suite (9 biomedical & clinical datasets)
# Automatically skips restricted local BRAT datasets (ACHS, CARMEN-I) if not present.

set -e

MODEL=${1:?Usage: $0 <model_path_or_hf_id> [results_csv]}
RESULTS_CSV=${2:-results/results_benchmark.csv}

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=========================================================================="
echo "FULL DOWNSTREAM BENCHMARK SUITE (9 DATASETS)"
echo "Model:       $MODEL"
echo "Results CSV: $RESULTS_CSV"
echo "=========================================================================="
mkdir -p "$(dirname "$RESULTS_CSV")"

is_done() {
    local ds="$1"
    local seed="$2"
    [ -f "$RESULTS_CSV" ] || return 1
    python3 - "$RESULTS_CSV" "$MODEL" "$ds" "$seed" <<'PYEOF'
import csv, sys

csv_path, target_model, target_ds, target_seed = sys.argv[1:5]
with open(csv_path, "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if (
            row.get("model") == target_model
            and row.get("dataset") == target_ds
            and str(row.get("seed", "")).strip() == target_seed
            and str(row.get("train_fraction", "1.00")).strip() in ("", "1.00", "1.0")
        ):
            sys.exit(0)
sys.exit(1)
PYEOF
}

# 1. Target Clinical BRAT NER Benchmarks (Seeds: 42, 123, 2024, 777, 1337; 15 epochs)
ACHS_DIR="$HOME/datasets/achs/masked_achs-privacy_corpus"
CARMEN_DIR="$HOME/datasets/carmen/CARMEN-I/ann/masked/ner"

echo "[1/3] Target clinical BRAT NER benchmarks (multi-seed 42, 123, 2024, 777, 1337)..."
for SEED in 42 123 2024 777 1337; do
    if [ -d "$ACHS_DIR" ]; then
        if is_done "achs" "$SEED"; then
            echo "  [SKIP] ACHS (Seed $SEED) already logged."
        else
            python scripts/finetune_eval.py \
                --base-model "$MODEL" \
                --dataset achs \
                --epochs 15 \
                --batch-size 16 \
                --learning-rate 2e-5 \
                --weight-decay 0.01 \
                --seed "$SEED" \
                --results-csv "$RESULTS_CSV"
        fi
    elif [ "$SEED" -eq 42 ]; then
        echo "  [SKIP] ACHS directory not found at $ACHS_DIR; skipping restricted benchmark."
    fi

    if [ -d "$CARMEN_DIR" ]; then
        if is_done "carmen_i" "$SEED"; then
            echo "  [SKIP] CARMEN-I (Seed $SEED) already logged."
        else
            python scripts/finetune_eval.py \
                --base-model "$MODEL" \
                --dataset carmen_i \
                --epochs 15 \
                --batch-size 16 \
                --learning-rate 2e-5 \
                --weight-decay 0.01 \
                --seed "$SEED" \
                --results-csv "$RESULTS_CSV"
        fi
    elif [ "$SEED" -eq 42 ]; then
        echo "  [SKIP] CARMEN-I directory not found at $CARMEN_DIR; skipping restricted benchmark."
    fi
done

# 2. Public Hugging Face NER Shared Tasks (15 epochs, Seed 42)
echo "[2/3] Public biomedical NER shared tasks (CANTEMIST, LivingNER1, PharmaCoNER, MedDoCan, DisTEMIST)..."
NER_PENDING=()
for ds in cantemist livingner1 pharmaconer meddocan distemist; do
    if is_done "$ds" "42"; then
        echo "  [SKIP] $ds (Seed 42) already logged."
    else
        NER_PENDING+=("$ds")
    fi
done

if [ ${#NER_PENDING[@]} -gt 0 ]; then
    python scripts/benchmark.py \
        --models "$MODEL" \
        --datasets "${NER_PENDING[@]}" \
        --num-train-epochs 15 \
        --per-device-train-batch-size 16 \
        --per-device-eval-batch-size 16 \
        --learning-rate 2e-5 \
        --weight-decay 0.01 \
        --warmup-steps 100 \
        --seeds 42 \
        --results-csv "$RESULTS_CSV"
fi

# 3. Clinical Sequence Classification (3 epochs, Seed 42)
echo "[3/3] Clinical classification benchmarks (CARES-A, CARES-C)..."
CLS_PENDING=()
for ds in caresA caresC; do
    if is_done "$ds" "42"; then
        echo "  [SKIP] $ds (Seed 42) already logged."
    else
        CLS_PENDING+=("$ds")
    fi
done

if [ ${#CLS_PENDING[@]} -gt 0 ]; then
    python scripts/benchmark_cls.py \
        --models "$MODEL" \
        --datasets "${CLS_PENDING[@]}" \
        --num-train-epochs 3 \
        --seeds 42 \
        --results-csv "$RESULTS_CSV"
fi

echo "=========================================================================="
echo "[OK] Benchmark suite completed. Results written to: $RESULTS_CSV"
echo "=========================================================================="
