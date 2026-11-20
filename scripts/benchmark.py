"""
Spanish Clinical NER Benchmark Tool

Evaluates any combination of HuggingFace models against Spanish clinical NER datasets.
All data is loaded directly from HuggingFace — no manual downloads required.

Usage:
    python benchmark.py \
        --models dccuchile/bert-base-spanish-wwm-cased IIC/RigoBERTa-Clinical \
        --datasets cantemist distemist pharmaconer meddocan \
        --epochs 15

Output:
    Results table comparing F1 scores across all model x dataset combinations,
    with comparison against paper reference (arXiv:2503.18594, Table 3).
"""

import argparse
import csv
import gc
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── PAPER REFERENCE SCORES (Table 3, arXiv:2503.18594) ───────────────────────
# Keys: (short_model_name, dataset_name)
PAPER_F1 = {
    ("RigoBERTa Clinical", "cantemist"):   0.906,
    ("XLM-R_Galén",        "cantemist"):   0.898,
    ("BETO_Galén",         "cantemist"):   0.802,
    ("bsc-bio-ehr-es",     "cantemist"):   0.864,
    ("BETO",               "cantemist"):   0.898,
    ("RigoBERTa 2",        "cantemist"):   0.903,
    ("mDeBERTaV3",         "cantemist"):   0.890,
    ("XLM-RoBERTa-Large",  "cantemist"):   0.904,

    ("RigoBERTa Clinical", "distemist"):   0.833,
    ("XLM-R_Galén",        "distemist"):   0.759,
    ("BETO_Galén",         "distemist"):   0.346,
    ("bsc-bio-ehr-es",     "distemist"):   0.759,
    ("BETO",               "distemist"):   0.801,
    ("RigoBERTa 2",        "distemist"):   0.832,
    ("mDeBERTaV3",         "distemist"):   0.808,
    ("XLM-RoBERTa-Large",  "distemist"):   0.817,

    ("RigoBERTa Clinical", "meddocan"):    0.982,
    ("XLM-R_Galén",        "meddocan"):    0.947,
    ("BETO_Galén",         "meddocan"):    0.682,
    ("bsc-bio-ehr-es",     "meddocan"):    0.967,
    ("BETO",               "meddocan"):    0.957,
    ("RigoBERTa 2",        "meddocan"):    0.979,
    ("mDeBERTaV3",         "meddocan"):    0.974,
    ("XLM-RoBERTa-Large",  "meddocan"):    0.978,

    ("RigoBERTa Clinical", "pharmaconer"): 0.931,
    ("XLM-R_Galén",        "pharmaconer"): 0.915,
    ("BETO_Galén",         "pharmaconer"): 0.708,
    ("bsc-bio-ehr-es",     "pharmaconer"): 0.904,
    ("BETO",               "pharmaconer"): 0.908,
    ("RigoBERTa 2",        "pharmaconer"): 0.927,
    ("mDeBERTaV3",         "pharmaconer"): 0.922,
    ("XLM-RoBERTa-Large",  "pharmaconer"): 0.924,

    ("RigoBERTa Clinical", "livingner1"):  0.953,
    ("XLM-R_Galén",        "livingner1"):  0.907,
    ("BETO_Galén",         "livingner1"):  0.646,
    ("bsc-bio-ehr-es",     "livingner1"):  0.938,
    ("BETO",               "livingner1"):  0.938,
    ("RigoBERTa 2",        "livingner1"):  0.951,
    ("mDeBERTaV3",         "livingner1"):  0.953,
    ("XLM-RoBERTa-Large",  "livingner1"):  0.949,

    ("RigoBERTa Clinical", "ehealth_kd"):  0.878,
    ("XLM-R_Galén",        "ehealth_kd"):  0.830,
    ("BETO_Galén",         "ehealth_kd"):  0.658,
    ("bsc-bio-ehr-es",     "ehealth_kd"):  0.836,
    ("BETO",               "ehealth_kd"):  0.843,
    ("RigoBERTa 2",        "ehealth_kd"):  0.865,
    ("mDeBERTaV3",         "ehealth_kd"):  0.844,
    ("XLM-RoBERTa-Large",  "ehealth_kd"):  0.871,
}

# Map HuggingFace model IDs → short paper names
MODEL_SHORT = {
    "dccuchile/bert-base-spanish-wwm-cased": "BETO",
    "IIC/RigoBERTa-Clinical":               "RigoBERTa Clinical",
    "PlanTL-GOB-ES/bsc-bio-ehr-es":         "bsc-bio-ehr-es",
    "xlm-roberta-large":                    "XLM-RoBERTa-Large",
    "microsoft/mdeberta-v3-base":           "mDeBERTaV3",
    "IIC/RigoBERTa-2.0":                    "RigoBERTa 2",
    "IIC/BETO_Galen":                       "BETO_Galén",
    "IIC/XLM-R_Galen":                      "XLM-R_Galén",
}


# ── DATASET REGISTRY ──────────────────────────────────────────────────────────
# Each entry defines how to load a dataset from HuggingFace.
# schema:
#   "token_cls"  — standard tokens + ner_tags columns (PlanTL / IIC parquet datasets)
#   "bigbio_kb"  — BigBio entity span format (passages + entities w/ char offsets)
# train_splits: list of HF split names combined for training
DATASETS = {
    "cantemist": {
        "hf_name":      "PlanTL-GOB-ES/cantemist-ner",
        "hf_config":    None,
        "schema":       "token_cls",
        "train_splits": ["train", "validation"],
        "test_split":   "test",
    },
    "pharmaconer": {
        "hf_name":      "PlanTL-GOB-ES/pharmaconer",
        "hf_config":    None,
        "schema":       "token_cls",
        "train_splits": ["train", "validation"],
        "test_split":   "test",
    },
    "livingner1": {
        "hf_name":      "IIC/livingner1",
        "hf_config":    None,
        "schema":       "auto",
        "train_splits": ["train", "validation"],
        "test_split":   "test",
    },
    "distemist": {
        "hf_name":      "IIC/distemist",
        "hf_config":    None,
        "schema":       "token_cls",
        "train_splits": ["train", "validation"],
        "test_split":   "test",
    },
    "meddocan": {
        "hf_name":      "IIC/meddocan",
        "hf_config":    None,
        "schema":       "token_cls",
        "train_splits": ["train", "validation"],
        "test_split":   "test",
    },
}


# ── BIO CONVERSION (for bigbio_kb entity-span format) ────────────────────────

def char_to_bio(text, annotations):
    char_labels = ["O"] * len(text)
    for start, end, label in annotations:
        for i in range(start, end):
            if i < len(char_labels):
                char_labels[i] = f"B-{label}" if i == start else f"I-{label}"

    tokens, tags, pos = [], [], 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        end = pos
        while end < len(text) and not text[end].isspace():
            end += 1
        tokens.append(text[pos:end])
        tags.append(char_labels[pos])
        pos = end
    return tokens, tags


def split_into_sentences(tokens, tags, max_len=100):
    sentences = []
    sent_tokens, sent_tags = [], []
    for token, tag in zip(tokens, tags):
        sent_tokens.append(token)
        sent_tags.append(tag)
        if len(sent_tokens) >= max_len:
            sentences.append((sent_tokens, sent_tags))
            sent_tokens, sent_tags = [], []
    if sent_tokens:
        sentences.append((sent_tokens, sent_tags))
    return sentences


# ── SEED CONTROL ─────────────────────────────────────────────────────────────

def set_seed(seed):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── DATASET LOADERS ───────────────────────────────────────────────────────────

def _hf_load(hf_name, hf_config, split):
    if hf_name == "IIC/livingner1":
        import glob
        import pyarrow as pa
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/IIC___livingner1")
        arrows = glob.glob(f"{cache_dir}/**/livingner1-{split}.arrow", recursive=True)
        if arrows:
            with pa.memory_map(arrows[0], 'r') as src:
                tbl = pa.ipc.open_stream(src).read_all()
                clean_tbl = pa.Table.from_arrays(tbl.columns, names=tbl.column_names)
                from datasets import Dataset as HFDataset
                return HFDataset(clean_tbl)
    kwargs = {"split": split, "trust_remote_code": True}
    if hf_config:
        return load_dataset(hf_name, hf_config, **kwargs)
    return load_dataset(hf_name, **kwargs)


def _detect_schema(hf_split):
    """Infer schema from column names."""
    cols = set(hf_split.features.keys())
    if "tokens" in cols and "ner_tags" in cols:
        return "token_cls"
    if "passages" in cols and "entities" in cols:
        return "bigbio_kb"
    return "unknown"


def _from_token_cls(hf_split):
    """Standard tokens + ner_tags columns (PlanTL datasets)."""
    all_tokens, all_tags = [], []
    label_names = None
    ner_feat = getattr(hf_split, "features", {}).get("ner_tags") if hasattr(hf_split, "features") else None
    if ner_feat is not None:
        if hasattr(ner_feat, "feature") and hasattr(ner_feat.feature, "names"):
            label_names = ner_feat.feature.names
        elif isinstance(ner_feat, list) and ner_feat and hasattr(ner_feat[0], "names"):
            label_names = ner_feat[0].names
    if label_names is None and len(hf_split) > 0 and "ner_tags" in hf_split[0]:
        sample_tags = hf_split[0]["ner_tags"]
        if sample_tags and isinstance(sample_tags[0], int):
            label_names = ["B-HUMAN", "I-HUMAN", "O", "B-SPECIES", "I-SPECIES"]
    for example in hf_split:
        tokens = example["tokens"]
        tags = example["ner_tags"]
        if tags and isinstance(tags[0], int) and label_names:
            tags = [label_names[t] for t in tags]
        for sent_tokens, sent_tags in split_into_sentences(tokens, tags):
            all_tokens.append(sent_tokens)
            all_tags.append(sent_tags)
    return all_tokens, all_tags


def _from_bigbio_kb(hf_split):
    """BigBio KB format: passages + entity character spans."""
    all_tokens, all_tags = [], []
    for example in hf_split:
        if not example["passages"]:
            continue
        max_end = max(p["offsets"][0][1] for p in example["passages"])
        chars = [" "] * max_end
        for passage in example["passages"]:
            start, end = passage["offsets"][0]
            for i, c in enumerate(passage["text"][0][: end - start]):
                if start + i < max_end:
                    chars[start + i] = c
        full_text = "".join(chars)
        annotations = [
            (offset[0], offset[1], entity["type"])
            for entity in example["entities"]
            for offset in entity["offsets"]
        ]
        tokens, tags = char_to_bio(full_text, annotations)
        for sent in split_into_sentences(tokens, tags):
            all_tokens.append(sent[0])
            all_tags.append(sent[1])
    return all_tokens, all_tags


SCHEMA_LOADERS = {
    "token_cls": _from_token_cls,
    "bigbio_kb": _from_bigbio_kb,
}


def load_dataset_splits(dataset_name):
    """Load train and test splits, returning (train_tokens, train_tags, test_tokens, test_tags)."""
    cfg = DATASETS[dataset_name]
    print(f"  Loading {dataset_name} from HuggingFace ({cfg['hf_name']})...")

    train_tokens, train_tags = [], []
    for split_name in cfg["train_splits"]:
        try:
            hf_split = _hf_load(cfg["hf_name"], cfg["hf_config"], split_name)
            schema = cfg["schema"] if cfg["schema"] != "auto" else _detect_schema(hf_split)
            loader = SCHEMA_LOADERS.get(schema)
            if loader is None:
                print(f"    Warning: unknown schema '{schema}' for split '{split_name}', skipping")
                continue
            t, g = loader(hf_split)
            train_tokens += t
            train_tags += g
        except Exception as e:
            print(f"    Warning: could not load split '{split_name}': {e}")

    hf_test = _hf_load(cfg["hf_name"], cfg["hf_config"], cfg["test_split"])
    schema = cfg["schema"] if cfg["schema"] != "auto" else _detect_schema(hf_test)
    loader = SCHEMA_LOADERS.get(schema, _from_token_cls)
    test_tokens, test_tags = loader(hf_test)

    print(f"    {len(train_tokens)} train sentences, {len(test_tokens)} test sentences")
    return train_tokens, train_tags, test_tokens, test_tags


# ── LABEL DETECTION ───────────────────────────────────────────────────────────

def detect_labels(all_tags):
    label_set = set()
    for tags in all_tags:
        for tag in tags:
            label_set.add(tag)
    labels = ["O"] + sorted(l for l in label_set if l != "O")
    label_to_id = {l: i for i, l in enumerate(labels)}
    id_to_label = {i: l for i, l in enumerate(labels)}
    return labels, label_to_id, id_to_label


# ── TOKENISATION + DATASET ────────────────────────────────────────────────────

def tokenize_and_align_labels(tokens, tags, tokenizer, label_to_id, max_length):
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    word_ids = tokenized.word_ids()
    aligned_labels, previous_word_id = [], None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            aligned_labels.append(label_to_id[tags[word_id]])
        else:
            aligned_labels.append(-100)
        previous_word_id = word_id
    tokenized["labels"] = aligned_labels
    return tokenized


class NERDataset(Dataset):
    def __init__(self, all_tokens, all_tags, tokenizer, label_to_id, max_length):
        self.all_tokens = all_tokens
        self.all_tags = all_tags
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.all_tokens)

    def __getitem__(self, idx):
        s = tokenize_and_align_labels(
            self.all_tokens[idx], self.all_tags[idx],
            self.tokenizer, self.label_to_id, self.max_length,
        )
        return {
            "input_ids":      torch.tensor(s["input_ids"]),
            "attention_mask": torch.tensor(s["attention_mask"]),
            "labels":         torch.tensor(s["labels"]),
        }


# ── SINGLE MODEL × DATASET RUN ───────────────────────────────────────────────

def run_single(
    model_name, train_tokens, train_tags, test_tokens, test_tags,
    label_list, label_to_id, id_to_label,
    checkpoint_dir, seed, args,
):
    set_seed(seed)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )

    train_dataset = NERDataset(train_tokens, train_tags, tokenizer, label_to_id, args.max_length)
    test_dataset  = NERDataset(test_tokens,  test_tags,  tokenizer, label_to_id, args.max_length)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            tl, tp = [], []
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    tl.append(id_to_label[label])
                    tp.append(id_to_label[pred])
            true_labels.append(tl)
            true_preds.append(tp)
        return {"f1": f1_score(true_labels, true_preds)}

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not args.use_cpu
    use_fp16 = torch.cuda.is_available() and not use_bf16 and not args.use_cpu
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        log_level=args.log_level,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        seed=seed,
        data_seed=args.data_seed,
        use_cpu=args.use_cpu,
        fp16=use_fp16,
        bf16=use_bf16,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    raw_preds, raw_labels, _ = trainer.predict(test_dataset)
    predictions = np.argmax(raw_preds, axis=-1)
    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, raw_labels):
        tl, tp = [], []
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                tl.append(id_to_label[label])
                tp.append(id_to_label[pred])
        true_labels.append(tl)
        true_preds.append(tp)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall":    recall_score(true_labels,    true_preds),
        "f1":        f1_score(true_labels,        true_preds),
    }


# ── RESULTS LOGGING ───────────────────────────────────────────────────────────

def append_result(
    csv_path,
    model_name,
    dataset_name,
    metrics,
    epochs,
    seed,
    n_train="",
    n_val="",
    n_test="",
    train_fraction="1.00",
):
    fieldnames = [
        "timestamp", "model", "dataset", "precision", "recall", "f1",
        "epochs", "seed", "train_fraction", "n_train", "n_val", "n_test",
    ]
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "dataset": dataset_name,
            "precision": f"{metrics['precision']:.4f}",
            "recall": f"{metrics['recall']:.4f}",
            "f1": f"{metrics['f1']:.4f}",
            "epochs": epochs,
            "seed": seed,
            "train_fraction": train_fraction,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        })


# ── RESULTS TABLE ─────────────────────────────────────────────────────────────

def print_results_table(results, models, datasets):
    """Print model × dataset F1 table with comparison to paper numbers."""
    short = lambda m: MODEL_SHORT.get(m, m.split("/")[-1])

    col_w  = 14
    name_w = 30
    sep    = "-" * (name_w + col_w * len(datasets))

    print(f"\n{'='*len(sep)}")
    print("BENCHMARK RESULTS — F1 (ours vs paper)")
    print(f"{'='*len(sep)}")
    print(f"{'Model':<{name_w}}" + "".join(f"{d:^{col_w}}" for d in datasets))
    print(sep)

    for model in models:
        row = f"{short(model):<{name_w}}"
        for ds in datasets:
            key = (model, ds)
            if key in results:
                our_f1  = results[key]["f1"]
                paper   = PAPER_F1.get((short(model), ds))
                if paper is not None:
                    diff = our_f1 - paper
                    cell = f"{our_f1:.3f}({diff:+.3f})"
                else:
                    cell = f"{our_f1:.3f}"
                row += f"{cell:^{col_w}}"
            else:
                row += f"{'ERR':^{col_w}}"
        print(row)

    print(sep)
    print("(diff) = ours minus paper's reported F1\n")

    # Paper reference row
    print(f"{'Paper (Table 3)':<{name_w}}" + "".join(
        f"{PAPER_F1.get((short(m), ds), 'N/A') if results else '':^{col_w}}"
        for ds in datasets
        for m in [models[0]]  # just use first model to get paper numbers
    ))


# ── MAIN BENCHMARK RUNNER ─────────────────────────────────────────────────────

def run_benchmark(args):
    seeds = args.seeds if args.seeds is not None else [args.seed]

    # Pre-load all datasets once (shared across all seeds and models)
    dataset_data = {}
    print("\nLoading datasets from HuggingFace...")
    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"  ERROR: '{ds_name}' not in registry. Available: {list(DATASETS.keys())}")
            sys.exit(1)
        try:
            train_tok, train_tag, test_tok, test_tag = load_dataset_splits(ds_name)
            dataset_data[ds_name] = (train_tok, train_tag, test_tok, test_tag)
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")

    for seed_idx, seed in enumerate(seeds):
        if len(seeds) > 1:
            print(f"\n{'='*60}")
            print(f"SEED {seed}  ({seed_idx + 1}/{len(seeds)})")
            print(f"{'='*60}")

        results = {}
        failed  = []
        total = len(args.models) * len(dataset_data)
        done  = 0

        for ds_name in args.datasets:
            if ds_name not in dataset_data:
                continue

            train_tok, train_tag, test_tok, test_tag = dataset_data[ds_name]
            label_list, label_to_id, id_to_label = detect_labels(train_tag + test_tag)

            for model_name in args.models:
                done += 1
                print(f"\n[{done}/{total}] {model_name} × {ds_name}")
                print(f"  Labels: {label_list}")

                checkpoint_dir = os.path.join(
                    args.output_dir,
                    f"{model_name.replace('/', '__')}__{ds_name}",
                )
                os.makedirs(checkpoint_dir, exist_ok=True)

                try:
                    metrics = run_single(
                        model_name=model_name,
                        train_tokens=train_tok, train_tags=train_tag,
                        test_tokens=test_tok,   test_tags=test_tag,
                        label_list=label_list,
                        label_to_id=label_to_id,
                        id_to_label=id_to_label,
                        checkpoint_dir=checkpoint_dir,
                        seed=seed,
                        args=args,
                    )
                    results[(model_name, ds_name)] = metrics
                    append_result(
                        args.results_csv,
                        model_name,
                        ds_name,
                        metrics,
                        args.num_train_epochs,
                        seed,
                        n_train=len(train_tok),
                        n_test=len(test_tok),
                    )
                    print(f"  F1: {metrics['f1']:.4f}")
                    # Clean up intermediate evaluation checkpoints to save disk space
                    for sub in Path(checkpoint_dir).glob("checkpoint-*"):
                        if sub.is_dir():
                            shutil.rmtree(sub, ignore_errors=True)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    traceback.print_exc()
                    failed.append((model_name, ds_name))
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

        print_results_table(results, args.models, args.datasets)

        if failed:
            print("Failed runs:")
            for model, ds in failed:
                print(f"  {model} × {ds}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark HuggingFace models on Spanish clinical NER datasets."
    )

    # ── Pipeline args ────────────────────────────────────────────────────────
    parser.add_argument("--models", nargs="+", required=True,
                        help="One or more HuggingFace model IDs")
    parser.add_argument("--datasets", nargs="+", required=True,
                        choices=list(DATASETS.keys()),
                        help=f"One or more dataset names. Available: {list(DATASETS.keys())}")
    parser.add_argument("--max-length",  type=int, default=512)
    parser.add_argument("--output-dir",  default="./checkpoints")
    parser.add_argument("--results-csv", default="./results.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Multiple seeds (e.g. --seeds 42 1 2). Overrides --seed.")

    # ── TrainingArguments ────────────────────────────────────────────────────
    parser.add_argument("--per-device-train-batch-size", type=int,   default=16)
    parser.add_argument("--per-device-eval-batch-size",  type=int,   default=16)
    parser.add_argument("--num-train-epochs",            type=float, default=3.0)
    parser.add_argument("--max-steps",                   type=int,   default=-1)
    parser.add_argument("--learning-rate",               type=float, default=2e-5)
    parser.add_argument("--warmup-steps",                type=float, default=100)
    parser.add_argument("--weight-decay",                type=float, default=0.01)
    parser.add_argument("--logging-strategy",            type=str,   default="steps")
    parser.add_argument("--logging-steps",               type=float, default=500)
    parser.add_argument("--log-level",                   type=str,   default="passive")
    parser.add_argument("--save-strategy",               type=str,   default="steps")
    parser.add_argument("--save-steps",                  type=float, default=500)
    parser.add_argument("--save-total-limit",            type=int,   default=1)
    parser.add_argument("--load-best-model-at-end",      action="store_true")
    parser.add_argument("--metric-for-best-model",       type=str,   default=None)
    parser.add_argument("--greater-is-better",           action="store_true", default=None)
    parser.add_argument("--seed",                        type=int,   default=42)
    parser.add_argument("--data-seed",                   type=int,   default=None)
    parser.add_argument("--use-cpu",                     action="store_true")
    parser.add_argument("--resume-from-checkpoint",      type=str,   default=None)
    parser.add_argument("--logging-dir",                 type=str,   default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
