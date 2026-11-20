"""Spanish Clinical Text Classification Benchmark (CARES-A and CARES-C).

Evaluates Hugging Face or local pre-trained encoder models on multi-label
radiological report classification tasks from Hugging Face.

Usage:
    python scripts/benchmark_cls.py \
        --models runs/clinical-synth-27b-n10/final FacebookAI/xlm-roberta-large \
        --datasets caresA caresC \
        --num-train-epochs 3 \
        --seeds 42 \
        --results-csv results/results_cls.csv
"""

import argparse
import csv
import gc
import os
import traceback
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CLS_DATASETS = {
    "caresA": {
        "hf_name": "IIC/caresA",
        "task": "multi_label",
        "train_splits": ["train", "validation"],
        "test_split": "test",
    },
    "caresC": {
        "hf_name": "IIC/caresC",
        "task": "multi_label",
        "train_splits": ["train", "validation"],
        "test_split": "test",
    },
}


def set_seed(seed):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hf_load(hf_name, split):
    return load_dataset(hf_name, split=split)


def load_cls_split(hf_split, task):
    """Return (texts, labels) from a classification split."""
    texts, labels = [], []
    for example in hf_split:
        texts.append(example["text"])
        raw = example["label"]
        if task == "multi_label":
            labels.append([float(x) for x in raw])
        else:
            labels.append(int(raw))
    return texts, labels


class ClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, task):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
        }
        if self.task == "multi_label":
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def run_single_cls(
    model_name,
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    task,
    num_labels,
    checkpoint_dir,
    seed,
    args,
):
    set_seed(seed)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    problem_type = (
        "multi_label_classification"
        if task == "multi_label"
        else "single_label_classification"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
    )

    train_dataset = ClsDataset(train_texts, train_labels, tokenizer, args.max_length, task)
    test_dataset = ClsDataset(test_texts, test_labels, tokenizer, args.max_length, task)

    threshold = 0.3 if task == "multi_label" else None

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if task == "multi_label":
            preds = (1 / (1 + np.exp(-logits)) > threshold).astype(int)
            labels = labels.astype(int)
        else:
            preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds, average="macro", zero_division=0)}

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

    preds_out = trainer.predict(test_dataset)
    logits = preds_out.predictions
    labels = preds_out.label_ids

    if task == "multi_label":
        preds = (1 / (1 + np.exp(-logits)) > threshold).astype(int)
        labels = labels.astype(int)
    else:
        preds = np.argmax(logits, axis=-1)

    return {
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


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


def print_results_table(results, models, datasets):
    name_w = max(len(m) for m in models) + 2
    col_w = 14
    header = f"{'Model':<{name_w}}" + "".join(f"{ds:^{col_w}}" for ds in datasets)
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for model in models:
        row = f"{model:<{name_w}}"
        for ds in datasets:
            if (model, ds) in results:
                row += f"{results[(model, ds)]['f1']:.4f}".center(col_w)
            else:
                row += "ERR".center(col_w)
        print(row)
    print(sep + "\n")


def run_benchmark(args):
    seeds = args.seeds if args.seeds is not None else [args.seed]

    dataset_data = {}
    print("\nLoading datasets from Hugging Face...")
    for ds_name in args.datasets:
        cfg = CLS_DATASETS[ds_name]
        print(f"  Loading {ds_name} ({cfg['hf_name']})...")
        try:
            train_texts, train_labels = [], []
            for split_name in cfg["train_splits"]:
                try:
                    hf_split = _hf_load(cfg["hf_name"], split_name)
                    t, l = load_cls_split(hf_split, cfg["task"])
                    train_texts += t
                    train_labels += l
                except Exception as e:
                    print(f"    Warning: could not load split '{split_name}': {e}")

            hf_test = _hf_load(cfg["hf_name"], cfg["test_split"])
            test_texts, test_labels = load_cls_split(hf_test, cfg["task"])

            if cfg["task"] == "multi_label":
                num_labels = len(train_labels[0])
            else:
                num_labels = len(set(train_labels) | set(test_labels))

            print(f"    {len(train_texts)} train docs, {len(test_texts)} test docs, {num_labels} labels")
            dataset_data[ds_name] = (train_texts, train_labels, test_texts, test_labels, num_labels, cfg["task"])
        except Exception as e:
            print(f"  ERROR loading {ds_name}: {e}")
            traceback.print_exc()

    for seed_idx, seed in enumerate(seeds):
        if len(seeds) > 1:
            print(f"\n{'='*60}")
            print(f"SEED {seed} ({seed_idx + 1}/{len(seeds)})")
            print(f"{'='*60}")

        results = {}
        total = len(args.models) * len(dataset_data)
        done = 0

        for ds_name in args.datasets:
            if ds_name not in dataset_data:
                continue
            train_texts, train_labels, test_texts, test_labels, num_labels, task = dataset_data[ds_name]

            for model_name in args.models:
                done += 1
                print(f"\n[{done}/{total}] {model_name} x {ds_name} (task={task}, labels={num_labels})")
                checkpoint_dir = os.path.join(
                    args.output_dir,
                    f"cls__{model_name.replace('/', '__')}__{ds_name}",
                )
                os.makedirs(checkpoint_dir, exist_ok=True)

                try:
                    metrics = run_single_cls(
                        model_name=model_name,
                        train_texts=train_texts,
                        train_labels=train_labels,
                        test_texts=test_texts,
                        test_labels=test_labels,
                        task=task,
                        num_labels=num_labels,
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
                        n_train=len(train_texts),
                        n_test=len(test_texts),
                    )
                    print(f"  Macro-F1: {metrics['f1']:.4f}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    traceback.print_exc()
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

        if results:
            print_results_table(results, args.models, args.datasets)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark encoder models on Spanish clinical classification datasets (CARES-A, CARES-C)."
    )
    parser.add_argument("--models", nargs="+", required=True, help="One or more model paths or Hugging Face IDs")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        choices=list(CLS_DATASETS.keys()),
        help=f"Available datasets: {list(CLS_DATASETS.keys())}",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--results-csv", default="results/results_cls.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Multiple random seeds")

    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-strategy", type=str, default="steps")
    parser.add_argument("--logging-steps", type=float, default=500)
    parser.add_argument("--log-level", type=str, default="passive")
    parser.add_argument("--save-strategy", type=str, default="steps")
    parser.add_argument("--save-steps", type=float, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--metric-for-best-model", type=str, default=None)
    parser.add_argument("--greater-is-better", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
