"""Fine-tune and evaluate NER encoders on local BRAT standoff datasets (CARMEN-I and ACHS).

Performs a deterministic 80/10/10 document split by seed, converts BRAT character
spans to subword BIO labels, fine-tunes token classification, and logs entity-level
seqeval F1 metrics.

Usage:
  python scripts/finetune_eval.py \
    --base-model runs/clinical-synth-27b-n10/final \
    --dataset carmen_i \
    --epochs 15 --seed 42 \
    --results-csv results/results_target_ner.csv
"""

import argparse
import csv
import glob
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from seqeval.metrics import f1_score as ner_f1
from seqeval.metrics import precision_score as ner_p
from seqeval.metrics import recall_score as ner_r
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

LOCAL_NER_DATASETS = {
    "carmen_i_ner": {
        "ann_dir_glob": os.path.expanduser("~/datasets/carmen/CARMEN-I/ann/masked/ner"),
        "desc": "CARMEN-I clinical NER",
    },
    "carmen_i": {
        "ann_dir_glob": os.path.expanduser("~/datasets/carmen/CARMEN-I/ann/masked/ner"),
        "desc": "CARMEN-I clinical NER (alias)",
    },
    "carmen": {
        "ann_dir_glob": os.path.expanduser("~/datasets/carmen/CARMEN-I/ann/masked/ner"),
        "desc": "CARMEN-I clinical NER (alias)",
    },
    "achs": {
        "ann_dir_glob": os.path.expanduser("~/datasets/achs/masked_achs-privacy_corpus"),
        "desc": "ACHS occupational emergency NER",
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_brat_ann(ann_path):
    """Parse BRAT T* annotation lines into sorted [(start, end, entity_type), ...] spans."""
    spans = []
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3 or not parts[0].startswith("T"):
                continue
            meta = parts[1].split()
            if len(meta) < 3:
                continue
            typ = meta[0]
            try:
                first = " ".join(meta[1:]).split(";")[0].split()
                start, end = int(first[0]), int(first[1])
            except (ValueError, IndexError):
                continue
            spans.append((start, end, typ))
    spans.sort(key=lambda x: x[0])
    return spans


def load_brat_docs(ann_dir):
    docs = []
    for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*.ann"))):
        txt_path = ann_path[: -len(".ann")] + ".txt"
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()
        docs.append({
            "name": os.path.basename(txt_path),
            "text": text,
            "spans": parse_brat_ann(ann_path),
        })
    return docs


def encode_doc(doc, tokenizer, max_length, label2id):
    """Convert document text and character-offset spans into subword token BIO tags."""
    enc = tokenizer(
        doc["text"],
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")
    labels = []
    open_entity = None
    for ts, te in offsets:
        if ts == te == 0:
            labels.append(-100)
            continue
        if open_entity is not None and ts < open_entity[0]:
            labels.append(label2id["I-" + open_entity[1]])
            continue
        open_entity = None
        lab = label2id["O"]
        for s, e, typ in doc["spans"]:
            if ts <= s < te or s <= ts < e:
                lab = label2id["B-" + typ]
                open_entity = (e, typ)
                break
        labels.append(lab)
    enc["labels"] = labels
    return enc


class TokenNERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.encodings[idx].items()}


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune and evaluate NER encoders on BRAT datasets")
    p.add_argument("--base-model", required=True, help="Pre-trained encoder path or Hugging Face model ID")
    p.add_argument("--dataset", required=True, choices=list(LOCAL_NER_DATASETS.keys()), help="Target BRAT dataset")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-device-train-batch-size", type=int, default=16)
    p.add_argument("--batch-size", dest="per_device_train_batch_size", type=int, help="Alias for --per-device-train-batch-size")
    p.add_argument("--data-dir", default=None, help="Directory containing BRAT .ann and .txt files")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--results-csv", default="results/results_target_ner.csv")
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument(
        "--subsample-train-fraction",
        type=float,
        default=1.0,
        help="Fraction of training documents to keep for sample-efficiency curves (0.0 < x <= 1.0)",
    )
    p.add_argument(
        "--split-file",
        default="auto",
        help="JSON split file with train/val/test filenames ('auto' loads configs/splits/achs__seed42.json for dataset=achs when present; pass 'none' for random seed shuffle)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.data_dir:
        custom_dir = os.path.expanduser(args.data_dir)
        if not os.path.isdir(custom_dir):
            raise FileNotFoundError(f"Directory specified in --data-dir does not exist: {custom_dir}")
        ann_dir = custom_dir
    else:
        ann_pattern = LOCAL_NER_DATASETS[args.dataset]["ann_dir_glob"]
        dirs = glob.glob(ann_pattern)
        if not dirs and "/*/" in ann_pattern:
            dirs = glob.glob(ann_pattern.replace("/*/", "/"))
        if not dirs and "/carmen/" in ann_pattern:
            carmen_fallback = os.path.expanduser("~/datasets/carmen/CARMEN-I/ann/masked/ner")
            if os.path.isdir(carmen_fallback):
                dirs = [carmen_fallback]
        if not dirs:
            raise FileNotFoundError(
                f"BRAT annotations directory not found for '{args.dataset}': {ann_pattern}. "
                "Pass --data-dir to specify the local path."
            )
        ann_dir = dirs[0]

    docs = load_brat_docs(ann_dir)
    print(f"Dataset '{args.dataset}': {len(docs)} documents loaded from {ann_dir}")

    types = sorted({typ for d in docs for _, _, typ in d["spans"]})
    labels = ["O"] + [f"{b}-{t}" for t in types for b in ("B", "I")]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    print(f"  {len(types)} entity types: {types}")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_path = None
    if args.split_file and args.split_file.lower() != "none":
        if args.split_file == "auto" and args.dataset == "achs":
            cand = os.path.join(repo_root, "configs", "splits", "achs__seed42.json")
            if os.path.exists(cand):
                split_path = cand
        elif args.split_file != "auto":
            cand = os.path.expanduser(args.split_file)
            if not os.path.exists(cand):
                raise FileNotFoundError(f"Split file specified via --split-file not found: {cand}")
            split_path = cand

    rng = random.Random(args.seed)
    if split_path:
        with open(split_path, encoding="utf-8") as f:
            split_cfg = json.load(f)
        doc_by_name = {d["name"]: d for d in docs}
        train_docs = [doc_by_name[name] for name in split_cfg.get("train", []) if name in doc_by_name]
        val_docs = [doc_by_name[name] for name in split_cfg.get("val", []) if name in doc_by_name]
        test_docs = [doc_by_name[name] for name in split_cfg.get("test", []) if name in doc_by_name]
        rng.shuffle(train_docs)
        print(f"  [Split] Loaded quarantined split from {split_path}")
    else:
        rng.shuffle(docs)
        n = len(docs)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train_docs, val_docs, test_docs = (
            docs[:n_train],
            docs[n_train : n_train + n_val],
            docs[n_train + n_val :],
        )

    if args.subsample_train_fraction < 1.0:
        n_sub = max(1, int(len(train_docs) * args.subsample_train_fraction))
        train_docs = train_docs[:n_sub]
        print(f"  [Sample Efficiency] Subsampling training split to {args.subsample_train_fraction*100:.0f}%: {len(train_docs)} docs")
    print(f"  Split: {len(train_docs)} train / {len(val_docs)} val / {len(test_docs)} test")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_enc = [encode_doc(d, tokenizer, args.max_length, label2id) for d in train_docs]
    val_enc = [encode_doc(d, tokenizer, args.max_length, label2id) for d in val_docs]
    test_enc = [encode_doc(d, tokenizer, args.max_length, label2id) for d in test_docs]

    train_ds = TokenNERDataset(train_enc)
    val_ds = TokenNERDataset(val_enc)
    test_ds = TokenNERDataset(test_enc)

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not args.use_cpu
    use_fp16 = torch.cuda.is_available() and not use_bf16 and not args.use_cpu

    frac_tag = f"__sub{int(args.subsample_train_fraction*100)}" if args.subsample_train_fraction < 1.0 else ""
    ckpt_dir = os.path.join(
        args.output_dir,
        f"ner__{args.dataset}__{args.base_model.replace('/', '__')}{frac_tag}__s{args.seed}",
    )

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=16,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
        use_cpu=args.use_cpu,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_strategy="steps",
        logging_steps=200,
        log_level="passive",
    )

    def compute_metrics(eval_pred):
        logits, lab = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_y, pred_y = [], []
        for t_row, p_row in zip(lab, preds):
            for ti, pi in zip(t_row, p_row):
                if ti != -100:
                    true_y.append(id2label[ti])
                    pred_y.append(id2label[pi])
        return {"f1": ner_f1([true_y], [pred_y])}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()

    preds_out = trainer.predict(test_ds)
    logits, lab = preds_out.predictions, preds_out.label_ids
    preds = np.argmax(logits, axis=-1)
    true_y, pred_y = [], []
    for t_row, p_row in zip(lab, preds):
        for ti, pi in zip(t_row, p_row):
            if ti != -100:
                true_y.append(id2label[ti])
                pred_y.append(id2label[pi])
    p, r, f1 = ner_p([true_y], [pred_y]), ner_r([true_y], [pred_y]), ner_f1([true_y], [pred_y])

    print(f"\n{'='*60}")
    print(f"RESULT  model={args.base_model}  dataset={args.dataset}")
    print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
    print(f"  Checkpoint saved at: {ckpt_dir}")
    print(f"{'='*60}\n")

    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "train_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "dataset": args.dataset,
                "ann_dir": ann_dir,
                "seed": args.seed,
                "epochs": args.epochs,
                "n_train": len(train_docs),
                "n_val": len(val_docs),
                "n_test": len(test_docs),
                "entity_types": types,
                "test_f1": f1,
            },
            f,
            indent=2,
        )

    out_dir = os.path.dirname(os.path.abspath(args.results_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    file_exists = os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp", "model", "dataset", "precision", "recall", "f1",
                "epochs", "seed", "train_fraction", "n_train", "n_val", "n_test",
            ],
            extrasaction="ignore",
            lineterminator="\n",
        )
        if not file_exists:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.base_model,
            "dataset": args.dataset,
            "precision": f"{p:.4f}",
            "recall": f"{r:.4f}",
            "f1": f"{f1:.4f}",
            "epochs": args.epochs,
            "seed": args.seed,
            "train_fraction": f"{args.subsample_train_fraction:.2f}",
            "n_train": len(train_docs),
            "n_val": len(val_docs),
            "n_test": len(test_docs),
        })


if __name__ == "__main__":
    main()
