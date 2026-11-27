"""Fine-tune and evaluate NER encoders on local BRAT standoff datasets (CARMEN-I and ACHS).

Loads BRAT documents preserving exact character offsets (newline=""), applies
sliding-window tokenization with non-overlapping document-level evaluation
reconstruction (preventing truncation of long EHRs), uses fixed leak-free splits
when available, fine-tunes token classification, and logs entity-level seqeval F1.

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
import shutil
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
            line = line.rstrip("\r\n")
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
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    return spans


def load_brat_docs(ann_dir):
    """Load BRAT .txt/.ann pairs with newline='' to preserve exact CRLF character offsets."""
    docs = []
    for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*.ann"))):
        txt_path = ann_path[: -len(".ann")] + ".txt"
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, encoding="utf-8", newline="") as f:
            text = f.read()
        docs.append({
            "name": os.path.basename(txt_path),
            "text": text,
            "spans": parse_brat_ann(ann_path),
        })
    return docs


def encode_doc_windows(doc, doc_idx, tokenizer, max_length, stride, label2id, is_eval=False):
    """Convert a document and BRAT spans into manual sliding-window subword BIO encodings.

    Tokenizes the full untrimmed document once (avoiding HuggingFace fast-tokenizer
    overflow truncation quirks), labels all tokens, and slices overlapping windows of
    up to max_length tokens. During evaluation (is_eval=True), overlapping tokens are
    masked to -100 outside each window's owned token span so every token and entity
    in the full document is scored exactly once without boundary splits.
    """
    enc_full = tokenizer(
        doc["text"],
        truncation=False,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = enc_full["input_ids"]
    full_offsets = enc_full["offset_mapping"]
    spans = doc["spans"]

    # Assign BIO labels once across the full untrimmed token sequence
    full_labels = []
    open_entity = None
    span_idx = 0
    n_spans = len(spans)
    for ts, te in full_offsets:
        if ts == te:
            full_labels.append(-100)
            continue
        if open_entity is not None and ts < open_entity[0]:
            lab = label2id["I-" + open_entity[1]]
        else:
            open_entity = None
            lab = label2id["O"]
            while span_idx < n_spans and spans[span_idx][1] <= ts:
                span_idx += 1
            for j in range(span_idx, n_spans):
                s, e, typ = spans[j]
                if s >= te:
                    break
                if ts <= s < te or s <= ts < e:
                    lab = label2id["B-" + typ]
                    open_entity = (e, typ)
                    break
        full_labels.append(lab)

    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    prefix_ids = [cls_id] if cls_id is not None else []
    suffix_ids = [sep_id] if sep_id is not None else []
    prefix_len = len(prefix_ids)
    suffix_len = len(suffix_ids)
    content_len = max(1, max_length - prefix_len - suffix_len)
    eff_stride = min(stride, content_len - 1) if content_len > 1 else 0
    step = max(1, content_len - eff_stride)

    n_tok = len(full_ids)
    if n_tok == 0:
        win_ranges = [(0, 0)]
    else:
        win_ranges = []
        start = 0
        while start < n_tok:
            end = min(start + content_len, n_tok)
            win_ranges.append((start, end))
            if end == n_tok:
                break
            start += step

    # Non-overlapping token-index ownership boundaries for evaluation
    id2label_local = {v: k for k, v in label2id.items()}
    tok_boundaries = []
    for w in range(len(win_ranges) - 1):
        ov_start = win_ranges[w + 1][0]
        ov_end = win_ranges[w][1]
        if ov_start < ov_end:
            mid = (ov_start + ov_end) // 2
            # Never split a multi-token gold entity across the evaluation boundary
            while (
                mid > ov_start
                and full_labels[mid] != -100
                and id2label_local[full_labels[mid]].startswith("I-")
            ):
                mid -= 1
            tok_boundaries.append(mid)
        else:
            tok_boundaries.append(ov_end)

    results = []
    for w, (ws, we) in enumerate(win_ranges):
        low_tok = tok_boundaries[w - 1] if w > 0 else 0
        high_tok = tok_boundaries[w] if w < len(win_ranges) - 1 else n_tok

        win_ids = prefix_ids + full_ids[ws:we] + suffix_ids
        win_mask = [1] * len(win_ids)

        content_labels = []
        for t in range(ws, we):
            if is_eval and not (low_tok <= t < high_tok):
                content_labels.append(-100)
            else:
                content_labels.append(full_labels[t])

        win_labels = ([-100] * prefix_len) + content_labels + ([-100] * suffix_len)
        if is_eval and n_tok > 0 and all(l == -100 for l in win_labels):
            continue

        item = {
            "input_ids": win_ids,
            "attention_mask": win_mask,
            "labels": win_labels,
        }
        results.append((item, doc_idx))

    return results


def encode_split(docs, tokenizer, max_length, stride, label2id, is_eval=False):
    encodings = []
    doc_ids = []
    for doc_idx, doc in enumerate(docs):
        for enc_item, d_id in encode_doc_windows(
            doc, doc_idx, tokenizer, max_length, stride, label2id, is_eval=is_eval
        ):
            encodings.append(enc_item)
            doc_ids.append(d_id)
    return encodings, doc_ids


def reconstruct_doc_sequences(lab, preds, doc_ids, id2label):
    """Reconstruct one contiguous BIO tag sequence per document from sliding windows."""
    true_by_doc = {}
    pred_by_doc = {}
    order = []
    for row_idx, (t_row, p_row) in enumerate(zip(lab, preds)):
        d_id = doc_ids[row_idx]
        if d_id not in true_by_doc:
            true_by_doc[d_id] = []
            pred_by_doc[d_id] = []
            order.append(d_id)
        for ti, pi in zip(t_row, p_row):
            if ti != -100:
                true_by_doc[d_id].append(id2label[int(ti)])
                pred_by_doc[d_id].append(id2label[int(pi)])
    return [true_by_doc[d] for d in order], [pred_by_doc[d] for d in order]


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
    p.add_argument("--stride", type=int, default=128, help="Sliding-window subword overlap for long documents")
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--results-csv", default="results/results_target_ner.csv")
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument(
        "--cleanup-ckpt",
        action="store_true",
        help="Remove fine-tuned model checkpoint directory after logging test metrics",
    )
    p.add_argument(
        "--subsample-train-fraction",
        type=float,
        default=1.0,
        help="Fraction of training documents to keep for sample-efficiency curves (0.0 < x <= 1.0)",
    )
    p.add_argument(
        "--split-file",
        default="auto",
        help="JSON split file with train/val/test filenames ('auto' loads configs/splits/{achs,carmen_i}__seed42.json when present; pass 'none' for random seed shuffle)",
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
            for cand_dir in (
                os.path.expanduser("~/datasets/carmen/CARMEN-I/ann/masked/ner"),
                os.path.expanduser("~/bruno/datasets/carmen/CARMEN-I/ann/masked/ner"),
            ):
                if os.path.isdir(cand_dir):
                    dirs = [cand_dir]
                    break
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
        if args.split_file == "auto":
            split_name = None
            if args.dataset == "achs":
                split_name = "achs__seed42.json"
            elif args.dataset in ("carmen_i", "carmen_i_ner", "carmen"):
                split_name = "carmen_i__seed42.json"
            if split_name:
                cand = os.path.join(repo_root, "configs", "splits", split_name)
                if os.path.exists(cand):
                    split_path = cand
        else:
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
        print(f"  [Split] Loaded fixed split from {split_path}")
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
    train_enc, _ = encode_split(train_docs, tokenizer, args.max_length, args.stride, label2id, is_eval=False)
    val_enc, val_doc_ids = encode_split(val_docs, tokenizer, args.max_length, args.stride, label2id, is_eval=True)
    test_enc, test_doc_ids = encode_split(test_docs, tokenizer, args.max_length, args.stride, label2id, is_eval=True)
    print(
        f"  Windows (max_length={args.max_length}, stride={args.stride}): "
        f"{len(train_enc)} train / {len(val_enc)} val / {len(test_enc)} test"
    )

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

    active_doc_ids = val_doc_ids

    def compute_metrics(eval_pred):
        logits, lab = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_seqs, pred_seqs = reconstruct_doc_sequences(lab, preds, active_doc_ids, id2label)
        return {"f1": ner_f1(true_seqs, pred_seqs)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()

    active_doc_ids = test_doc_ids
    preds_out = trainer.predict(test_ds)
    logits, lab = preds_out.predictions, preds_out.label_ids
    preds = np.argmax(logits, axis=-1)
    true_seqs, pred_seqs = reconstruct_doc_sequences(lab, preds, test_doc_ids, id2label)
    p, r, f1 = ner_p(true_seqs, pred_seqs), ner_r(true_seqs, pred_seqs), ner_f1(true_seqs, pred_seqs)

    print(f"\n{'='*60}")
    print(f"RESULT  model={args.base_model}  dataset={args.dataset}")
    print(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")
    print(f"  Checkpoint saved at: {ckpt_dir}")
    print(f"{'='*60}\n")

    if args.cleanup_ckpt:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        print(f"  [Cleanup] Removed disposable checkpoint dir: {ckpt_dir}")
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(ckpt_dir, "train_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "base_model": args.base_model,
                    "dataset": args.dataset,
                    "ann_dir": ann_dir,
                    "split_file": split_path,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "max_length": args.max_length,
                    "stride": args.stride,
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
