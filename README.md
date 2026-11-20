# Assessing LLM-Synthesized Clinical Narratives in Spanish

Official replication repository for the paper **"Assessing LLM-Synthesized Clinical Narratives in Spanish"**.

This repository contains the code and configurations required to reproduce the paper's pipeline:
1. Downloading and normalizing the Spanish clinical seed corpora (`ClinText-SP`) and general biomedical web anchor (`CoWeSe`).
2. Generating multi-density synthetic clinical narratives (`N = 1, 5, 10`) via vLLM (`Qwen/Qwen3-8B` and `Qwen/Qwen3.8-27B`) with intra-source RAG retrieval, SBERT semantic filtering, deduplication, and `CoWeSe` replay mixing.
3. Continual Pre-Training (CPT) of `FacebookAI/xlm-roberta-large` using Masked Language Modeling (MLM).
4. Downstream multi-seed fine-tuning and evaluation across 9 Spanish biomedical and clinical benchmarks (`ACHS`, `CARMEN-I`, `CANTEMIST`, `DisTEMIST`, `PharmaCoNER`, `MedDoCan`, `LivingNER1`, `CARES-A`, `CARES-C`) plus low-resource sample efficiency (`10%`, `25%`, `50%`, `100%`).
5. Empirical privacy and PII leakage audits, lexical diversity and specialized clinical terminology metrics, intra-seed medical concept expansion, statistical hypothesis testing, and blinded clinical quality assessment with `TypeSafe Jev-1.13`.

> Output directories (`results/` and `figures/`) are not pre-populated in this repository; they are generated locally when running the pipeline.

---

## Pre-trained Models Status

> **Note:** Continually pre-trained encoder checkpoints (`xlm-roberta-large` trained on `Qwen-3-8B` and `Qwen-3.8-27B` synthetic corpora, density ablations, and control models) are pending public release on Hugging Face upon manuscript publication (and available upon reasonable request during peer review). All code and configurations needed to replicate the full training and evaluation pipeline from scratch are provided below.

---

## Repository Structure

```text
Assessing-LLM-Synthesized-Clinical-Narratives-in-Spanish/
├── README.md                              # Step-by-step replication guide
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── corpus_config_v2.yaml                  # 13 clinical seed normalization config
├── .env.example                           # Environment template (no API keys stored)
├── .gitignore
├── configs/
│   ├── dataset_registry.yaml              # Catalog of seed corpora & downstream benchmarks
│   ├── seeds/
│   │   ├── achs_pseudo__train_only.json   # Leak-free train-only seed IDs for ACHS
│   │   └── achs_pseudo_to_masked_mapping.json # Mapping between ACHS pseudo and BRAT IDs
│   └── splits/
│       └── achs__seed42.json              # Reproducible 80/10/10 ACHS benchmark split IDs
└── scripts/
    ├── dataset_registry.py                # Dataset catalog & train-only seed extraction
    ├── corpus_normalizer.py               # Corpus normalization helper
    ├── normalize_corpora.py               # Step 1: Download & normalize clinical seed corpora
    ├── preparar_cowese.py                 # Step 1: Prepare CoWeSe biomedical web anchor
    ├── corpus_pipeline_v3.py              # Step 2: vLLM synthetic generation + RAG + SBERT filters
    ├── reconstruir_maestro.py             # Step 2: Deduplicate synthetic texts into master CSV
    ├── mix_corpus.py                      # Step 2: Combine clinical/synthetic CSV with CoWeSe
    ├── build_ablation_density_corpus.py   # Step 2: Construct N=1 and N=5 density ablations
    ├── build_corpus_real_seeds.py         # Step 2: Construct Real Seeds control (Appendix B)
    ├── pretraining_mlm.py                 # Step 3: Continual Pre-Training (MLM)
    ├── benchmark.py                       # Step 4: Public Hugging Face NER benchmarks
    ├── benchmark_cls.py                   # Step 4: Clinical sequence classification (CARES-A/C)
    ├── finetune_eval.py                   # Step 4: Local BRAT NER benchmarks (CARMEN-I, ACHS)
    ├── eval_suite.sh                      # Step 4: Full 9-benchmark evaluation wrapper
    ├── analyze_privacy_memorization.py    # Step 5: Verbatim k-gram overlap & LCS audit (Figs. 7-8)
    ├── audit_pii_leakage.py               # Step 5: PII/PHI pattern leakage audit (Table 6)
    ├── analyze_corpus_diversity.py        # Step 5: Lexical richness & terminology audit (Fig. 4)
    ├── analyze_clinical_novelty.py        # Step 5: Intra-seed Self-BLEU-4 & concept novelty (Fig. 9)
    ├── extract_geval_sample_5200.py       # Step 5: Balanced sampling for Jev-1.13 evaluation
    ├── run_jev_eval.py                    # Step 5: Blinded clinical eval via OpenRouter (Fig. 5)
    └── analyze_significance_and_plots.py  # Step 5: Paired t-tests, Holm-Bonferroni & Figs. 1, 2, 6
```

---

## Step 0 — Environment Setup

Python 3.10+ and CUDA-capable GPUs (BF16 recommended, e.g., NVIDIA A100/H100) are recommended for synthetic generation, continual pre-training, and downstream fine-tuning.

```bash
git clone https://github.com/caracena/Assessing-LLM-Synthesized-Clinical-Narratives-in-Spanish.git
cd Assessing-LLM-Synthesized-Clinical-Narratives-in-Spanish

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` pins `datasets>=2.18.0,<4.0.0` because `PlanTL-GOB-ES/cantemist-ner` and `PlanTL-GOB-ES/pharmaconer` use script-backed loaders deprecated in `datasets>=4.0`.

To run local LLM synthetic generation with vLLM, install `vllm` matching your CUDA environment (tested with `vllm 0.21.0`; `>=0.17.0` is required for `Qwen/Qwen3-8B` and `Qwen/Qwen3.8-27B`'s `qwen3_5` Gated DeltaNet architecture):

```bash
pip install "vllm>=0.17.0"
```

### API Keys (Optional / Step 5 Clinical Judge Only)

No API keys are committed in this repository. An `OPENROUTER_API_KEY` environment variable is **only required** if you re-run the blinded `TypeSafe Jev-1.13` clinical evaluation in Step 5.3:

```bash
cp .env.example .env
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

---

## Step 1 — Downloading & Normalizing Source Datasets

### 1.1 Public Clinical Seeds & Downstream Benchmarks

The 12 public clinical seed collections are loaded automatically from [`IIC/ClinText-SP`](https://huggingface.co/datasets/IIC/ClinText-SP) on Hugging Face (`atadolcc`, `adolescere`, `cardioccc`, `reccmi`, `cantemist`, `cares`, `chilean_waiting_list`, `e3c`, `jart`, `livingner`, `meddoplace`, `spaccc`).

All 7 public downstream benchmarks are also fetched automatically from Hugging Face via `datasets`:
- **NER**: `cantemist`, `distemist`, `pharmaconer`, `meddocan`, `livingner1`
- **Classification**: `caresA`, `caresC`

### 1.2 Restricted / Institutional Corpora (`CARMEN-I` & `ACHS`)

Two evaluation corpora require credentialed or institutional data access and cannot be redistributed here:
- **CARMEN-I**: Obtain through credentialed access on [PhysioNet](https://physionet.org/) and place the BRAT annotations under `~/datasets/carmen/CARMEN-I/ann/masked/ner` (or pass `--data-dir` to `finetune_eval.py`).
- **ACHS**: Private occupational emergency corpus from *Asociacion Chilena de Seguridad*. When available locally under `~/datasets/achs/masked_achs-privacy_corpus` (downstream NER) and `~/datasets/ACHS_pseudo` (raw notes filtered strictly to train-only seeds via `configs/seeds/achs_pseudo__train_only.json`), materialize the quarantined generation seeds with:
  ```bash
  python scripts/dataset_registry.py --seeds achs_pseudo --out ~/datasets/ACHS_pseudo_seeds --style normalize_case
  ```

If you do not have access to `CARMEN-I` or `ACHS`, all scripts in Steps 1–5 skip missing local restricted folders gracefully and run cleanly across the 12 public `ClinText-SP` seed sources and 7 public benchmarks.

### 1.3 Normalize Seed Corpora

Normalize the seed collections configured in `corpus_config_v2.yaml` into `corpus/por_source/<source>_normalizado.json` (absent `achs_pseudo` is skipped automatically if unavailable):

```bash
python scripts/normalize_corpora.py
```

### 1.4 Prepare the `CoWeSe` Scientific Background Anchor

Download `CoWeSe.txt` from Zenodo ([https://zenodo.org/records/5513237](https://zenodo.org/records/5513237), DOI: `10.5281/zenodo.5513237`) using a web browser (Zenodo returns `HTTP 403` to non-browser `curl`/`wget` clients), place it at `data/cowese/CoWeSe.txt`, and group sentences into `251,548` pseudo-documents (25 consecutive sentences per block):

```bash
python scripts/preparar_cowese.py \
  --input data/cowese/CoWeSe.txt \
  --output data/cowese/cowese_prepared.csv \
  --target-docs 251548 \
  --sentences-per-doc 25 \
  --seed 42
```

---

## Step 2 — Synthetic Corpus Generation & Corpus Assembly

### 2.1 Start Local vLLM Inference Server

Serve either open-weight generator using an OpenAI-compatible vLLM endpoint (under BF16 precision). Hardware & environment notes:
- **VRAM sizing**: `Qwen/Qwen3-8B` runs comfortably on 24–48 GB GPUs (`~16 GB` BF16 weights + KV cache); `Qwen/Qwen3.8-27B` requires `~55+ GB` (`80–96 GB` A100/H100 or `--tensor-parallel-size 2`).
- **Environment flags**: Set `export VLLM_USE_FLASHINFER_SAMPLER=0` on hosts without `ninja` installed to avoid FlashInfer JIT build errors, and `export CUDA_DEVICE_ORDER=PCI_BUS_ID` on servers with mixed GPU models.

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Generator 1: Qwen-3-8B
vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3_8b Qwen/Qwen3-8B \
  --port 8000 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --trust-remote-code

# Generator 2: Qwen-3.8-27B
vllm serve Qwen/Qwen3.8-27B \
  --served-model-name qwen3.8_27b Qwen/Qwen3.8-27B \
  --port 8000 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --trust-remote-code
```

### 2.2 Generate Synthetic Narratives (`N = 10` variants per seed)

Run `scripts/corpus_pipeline_v3.py` across normalized seed files using intra-source RAG (`k = 3`) and triple SBERT semantic filtering (`paraphrase-multilingual-mpnet-base-v2`, `tau_copy = 0.90`, `tau_div = 0.40`, `tau_pair = 0.95`):

```bash
for NORM_JSON in corpus/por_source/*_normalizado.json; do
  SRC=$(basename "$NORM_JSON" _normalizado.json)
  python scripts/corpus_pipeline_v3.py \
    --normalized-corpus "$NORM_JSON" \
    --output-prefix "corpus/sinteticos_qwen3.8_27b/sintetico_${SRC}" \
    --endpoints http://localhost:8000/v1 \
    --model-path qwen3.8_27b \
    --n-per-doc 10 \
    --max-concurrent 16 \
    --temperature 0.9 \
    --top-p 0.95 \
    --k-retrieved 3 \
    --umbral-copia 0.90 \
    --umbral-divergencia 0.40 \
    --umbral-pairwise 0.95
done
```

For `Qwen-3-8B`, point `--output-prefix` to `corpus/sinteticos/sintetico_${SRC}` and `--model-path` to `qwen3_8b`.

### 2.3 Deduplicate & Assemble Training Corpora

Build the deduplicated synthetic CSV (`Pure-Synth 27B`) and mix it with the `251,548` `CoWeSe` documents (`Flagship 27B`, `N = 10`):

```bash
# 1. Deduplicate synthetic outputs -> Pure-Synth 27B CSV (334,102 docs across 13 sources)
python scripts/reconstruir_maestro.py \
  --model qwen3.8_27b \
  --output corpus_gigante/maestro_qwen3.8_27b.csv

# 2. Mix synthetic corpus with CoWeSe -> Flagship 27B CSV (585,650 docs)
python scripts/mix_corpus.py \
  --synthetic corpus_gigante/maestro_qwen3.8_27b.csv \
  --external data/cowese/cowese_prepared.csv \
  --output corpus_gigante/corpus_train_27b_n10_cowese.csv \
  --seed 42
```

*(Run the same two steps with `--model qwen3_8b` to produce `corpus_gigante/maestro_qwen3_8b.csv` and `Flagship 8B`.)*

### 2.4 Build Ablation & Control Corpora

- **Density Ablations (`N = 1`, `N = 5`)**:
  ```bash
  python scripts/build_ablation_density_corpus.py \
    --n-per-doc 5 \
    --input-dir corpus/sinteticos_qwen3.8_27b \
    --output corpus_gigante/maestro_27b_n5.csv

  python scripts/mix_corpus.py \
    --synthetic corpus_gigante/maestro_27b_n5.csv \
    --external data/cowese/cowese_prepared.csv \
    --output corpus_gigante/corpus_train_27b_n5_cowese.csv \
    --seed 42
  ```
- **Pure-Synth 27B (`0` CoWeSe)**: Pass `corpus_gigante/maestro_qwen3.8_27b.csv` directly to `scripts/pretraining_mlm.py`.
- **CoWeSe Baseline (`0` Synthetic)**: Pass `data/cowese/cowese_prepared.csv` directly to `scripts/pretraining_mlm.py`.
- **Real Seeds Control (`29,812` authentic seeds + `251,548` CoWeSe, Appendix B)**:
  ```bash
  python scripts/build_corpus_real_seeds.py \
    --input-dir corpus/por_source \
    --output corpus_gigante/maestro_real_seeds.csv

  python scripts/mix_corpus.py \
    --synthetic corpus_gigante/maestro_real_seeds.csv \
    --external data/cowese/cowese_prepared.csv \
    --output corpus_gigante/corpus_train_real_seeds.csv \
    --seed 42
  ```

---

## Step 3 — Continual Pre-Training (MLM)

Continually pre-train `FacebookAI/xlm-roberta-large` for 3.0 full epochs with sliding-window chunking (`max_seq_length = 512`, `stride = 128`), dynamic 15% MLM masking, `lr = 2e-5`, `batch_size = 32`, `weight_decay = 0.1`, `warmup_ratio = 0.1`, `val_split = 0.05`, and BF16 precision:

```bash
python scripts/pretraining_mlm.py \
  --corpus corpus_gigante/corpus_train_27b_n10_cowese.csv \
  --output_dir runs/clinical-synth-27b-n10 \
  --model_base FacebookAI/xlm-roberta-large \
  --epochs 3.0 \
  --batch_size 32 \
  --lr 2e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --val_split 0.05 \
  --max_seq_length 512 \
  --stride 128 \
  --mlm-probability 0.15 \
  --bf16 \
  --seed 42
```

For multi-GPU DDP training, launch `scripts/pretraining_mlm.py` with `torchrun --nproc_per_node=N`.

---

## Step 4 — Downstream Fine-Tuning & Evaluation

Run all 9 downstream benchmarks with a single command (automatically skipping `ACHS` / `CARMEN-I` if not locally present):

```bash
bash scripts/eval_suite.sh runs/clinical-synth-27b-n10/final results/results_master_benchmark.csv
```

Or run each block individually:

### 4.1 Target Clinical NER Benchmarks (`ACHS`, `CARMEN-I` — Multi-Seed `42, 123, 2024`)

*(Note: On `--dataset achs`, `finetune_eval.py` defaults to `--split-file auto`, loading the fixed leak-free holdout in `configs/splits/achs__seed42.json` so `val`/`test` stay strictly quarantined across random seeds; pass `--split-file none` for per-seed shuffling. All downstream scripts write to a unified 12-column CSV schema.)*

```bash
for SEED in 42 123 2024; do
  python scripts/finetune_eval.py \
    --base-model runs/clinical-synth-27b-n10/final \
    --dataset carmen_i \
    --epochs 15 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --weight-decay 0.01 \
    --seed ${SEED} \
    --results-csv results/results_master_benchmark.csv
done
```

### 4.2 Public Biomedical NER Shared Tasks (Hugging Face)

```bash
python scripts/benchmark.py \
  --models runs/clinical-synth-27b-n10/final FacebookAI/xlm-roberta-large IIC/RigoBERTa-Clinical \
  --datasets cantemist livingner1 pharmaconer meddocan distemist \
  --num-train-epochs 15 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 16 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-steps 100 \
  --seeds 42 \
  --results-csv results/results_master_benchmark.csv
```

### 4.3 Clinical Sequence Classification (`CARES-A`, `CARES-C`)

```bash
python scripts/benchmark_cls.py \
  --models runs/clinical-synth-27b-n10/final FacebookAI/xlm-roberta-large IIC/RigoBERTa-Clinical \
  --datasets caresA caresC \
  --num-train-epochs 3 \
  --seeds 42 \
  --results-csv results/results_master_benchmark.csv
```

### 4.4 Low-Resource Sample Efficiency (`10%`, `25%`, `50%`, `100%`)

```bash
for FRAC in 0.10 0.25 0.50 1.00; do
  python scripts/finetune_eval.py \
    --base-model runs/clinical-synth-27b-n10/final \
    --dataset achs \
    --epochs 15 \
    --subsample-train-fraction ${FRAC} \
    --seed 42 \
    --results-csv results/results_sample_efficiency.csv
done
```

---

## Step 5 — Privacy, Diversity, Novelty & Blinded Clinical Evaluation (`Jev-1.13`)

### 5.1 Privacy, Verbatim Memorization & PII Audits

```bash
# Verbatim k-gram overlap & Longest Common Substring (LCS) vs real seeds (Table 5, Figs. 7-8)
python scripts/analyze_privacy_memorization.py

# Automated PII/PHI identifier leakage audit (Table 6; pass --dataset cantemist if ACHS is unavailable)
python scripts/audit_pii_leakage.py --dataset achs_pseudo
```

### 5.2 Corpus Lexical Diversity & Clinical Concept Expansion

*(Audit scripts in Steps 5.1–5.2 automatically handle single-generator runs when only `8B` or only `27B` outputs are available.)*

```bash
# Vocabulary size, average length, STTR-1000, Distinct-2/3, Hapax ratio,
# specialized clinical terms (affix morphology), medical density, and 3/4-gram novelty (Table 7, Fig. 4)
python scripts/analyze_corpus_diversity.py

# Intra-seed redundancy (Self-BLEU-4) & novel medical concepts across aligned seed clusters (Table 8, Fig. 9)
# Default --max-cases 3000 covers all 2,812 aligned ACHS clusters (pass --dataset cantemist if ACHS is unavailable)
python scripts/analyze_clinical_novelty.py --dataset achs_pseudo --max-cases 3000
```

### 5.3 Blinded Clinical Quality Evaluation with `TypeSafe Jev-1.13`

Evaluates balanced synthetic clinical notes across four 5-point criteria (*Clinical Plausibility*, *Professional Medical Register*, *Internal Consistency*, and *Document Completeness*) and generates Figure 5 (`--per-source 10` samples `N=260`; `--per-source 200` samples `N=5,200`; unavailable restricted sources such as `achs_pseudo` are skipped automatically). `extract_geval_sample_5200.py` also supports single-generator sampling if only one model directory exists; paired `run_jev_eval.py` compares `8B` vs `27B`.

> **API Key Required:** Export `OPENROUTER_API_KEY` before running `run_jev_eval.py`.

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"

# 1. Sample balanced clinical notes across available sources (seed=42)
python scripts/extract_geval_sample_5200.py --per-source 200

# 2. Run concurrent Jev-1.13 structured decision evaluation
python scripts/run_jev_eval.py \
  --input-file results/geval_sample_data_5200.json \
  --max-workers 16
```

### 5.4 Statistical Significance & Main Figures

Compute two-tailed paired *t*-tests, Cohen's $d_z$, pooled Cohen's $d$, and Holm-Bonferroni corrections (Table 10 & Appendix B), and generate `figures/fig1_density_scaling.{pdf,png}`, `figures/fig2_head_to_head_baselines.{pdf,png}`, `figures/fig6_sample_efficiency_curve.{pdf,png}`, and `figures/significance_report.txt`:

```bash
# Using Step 4 output CSV (pass --target-model and optional --sample-efficiency-csv to evaluate your own model):
python scripts/analyze_significance_and_plots.py \
  --results-csv results/results_master_benchmark.csv \
  --target-model runs/clinical-synth-27b-n10/final

# Or reproduce directly from manuscript multi-seed tables:
python scripts/analyze_significance_and_plots.py --use-paper-results
```

---

## License

This code is released under the [MIT License](LICENSE). Third-party datasets (`IIC/ClinText-SP`, `PlanTL-GOB-ES`, `CoWeSe`, `CARMEN-I`, `ACHS`) and base/generator models (`FacebookAI/xlm-roberta-large`, `Qwen/Qwen3-8B`, `Qwen/Qwen3.8-27B`) remain subject to their respective licenses and data-use agreements.
