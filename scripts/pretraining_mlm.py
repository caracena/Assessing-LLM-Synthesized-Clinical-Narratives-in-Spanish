"""Continual Masked Language Modeling (MLM) pre-training on clinical corpora.

Initializes from `FacebookAI/xlm-roberta-large` and performs sliding-window
chunking (max_seq_length=512, stride=128) with dynamic 15% MLM masking.

Usage:
    python scripts/pretraining_mlm.py \
        --corpus corpus_gigante/corpus_train_27b_n10_cowese.csv \
        --output_dir runs/clinical-synth-27b-n10 \
        --epochs 3 \
        --batch_size 32
"""

import argparse
import importlib.util
import logging
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_BASE = "FacebookAI/xlm-roberta-large"
MAX_SEQ_LENGTH = 512
STRIDE = 128
MLM_PROBABILITY = 0.15
SEED = 42

DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_EPOCHS = 3
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_VAL_SPLIT = 0.05


def load_corpus(csv_path: str) -> list[str]:
    """Load non-empty text documents from a prepared CSV corpus."""
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        raise ValueError(f"CSV must contain a 'text' column. Found columns: {list(df.columns)}")

    texts = df["text"].dropna().astype(str).tolist()
    texts = [t for t in texts if t.strip()]

    logger.info(f"Loaded corpus: {len(texts)} documents from {csv_path}")
    return texts


def tokenize_and_chunk(texts: list[str], tokenizer, max_length: int, stride: int) -> Dataset:
    """Tokenize and segment long clinical narratives with a sliding window."""
    dataset = Dataset.from_dict({"text": texts})

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding=False,
        )

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing and chunking (max_length={max_length}, stride={stride})",
    )

    if "overflow_to_sample_mapping" in tokenized.column_names:
        tokenized = tokenized.remove_columns(["overflow_to_sample_mapping"])

    logger.info(f"Tokenized chunked dataset size: {len(tokenized)} examples")
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Continual MLM Pre-Training for Spanish Clinical Encoders")
    parser.add_argument("--corpus", required=True, help="Path to pretraining CSV corpus (requires 'text' column)")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints and final model")
    parser.add_argument("--model_base", default=MODEL_BASE, help="Base Hugging Face encoder model")
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False, help="Force FP16 precision")
    parser.add_argument("--bf16", action="store_true", default=False, help="Force BF16 precision (A100/H100)")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--stride", type=int, default=STRIDE, help="Token stride for long texts")
    parser.add_argument("--mlm-probability", type=float, default=MLM_PROBABILITY, help="Dynamic masking ratio")
    args = parser.parse_args()

    set_seed(args.seed)

    has_cuda = torch.cuda.is_available()
    bf16_supported = has_cuda and torch.cuda.is_bf16_supported()

    if args.bf16:
        use_bf16 = True
        use_fp16 = False
    elif args.fp16:
        use_bf16 = False
        use_fp16 = has_cuda
    else:
        use_bf16 = bf16_supported
        use_fp16 = has_cuda and not use_bf16

    if not has_cuda:
        logger.warning("CUDA is not available; training will run on CPU.")
    else:
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"Mixed precision: bf16={use_bf16}, fp16={use_fp16}")

    logger.info(f"Loading base model: {args.model_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    model = AutoModelForMaskedLM.from_pretrained(args.model_base)

    texts = load_corpus(args.corpus)
    tokenized = tokenize_and_chunk(texts, tokenizer, args.max_seq_length, args.stride)

    split = tokenized.train_test_split(test_size=args.val_split, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Split sizes -> Train: {len(train_dataset)} | Validation: {len(eval_dataset)}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    total_steps = int((len(train_dataset) / (args.batch_size * max(args.gradient_accumulation_steps, 1))) * args.epochs)
    if total_steps >= 500:
        warmup_steps = max(int(total_steps * args.warmup_ratio), 50)
    else:
        warmup_steps = max(int(total_steps * args.warmup_ratio), 1)

    has_tb = bool(
        importlib.util.find_spec("tensorboard") or importlib.util.find_spec("tensorboardX")
    )
    report_to = ["tensorboard"] if has_tb else []
    if not has_tb:
        logger.info("TensorBoard not installed; disabling TensorBoard reporting.")

    can_load_best = (total_steps >= args.eval_steps) and (args.eval_steps == args.save_steps)
    if not can_load_best:
        logger.info(
            f"Skipping load_best_model_at_end (estimated total_steps={total_steps}, "
            f"eval_steps={args.eval_steps}, save_steps={args.save_steps})."
        )
    logger.info(f"Estimated steps: {total_steps} | Warmup steps: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        load_best_model_at_end=can_load_best,
        metric_for_best_model="eval_loss" if can_load_best else None,
        greater_is_better=False if can_load_best else None,
        report_to=report_to,
        seed=args.seed,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting continual MLM pre-training...")
    logger.info(f"  Base model:     {args.model_base}")
    logger.info(f"  Corpus:         {args.corpus} ({len(texts)} docs -> {len(tokenized)} chunks)")
    logger.info(f"  Epochs:         {args.epochs}")
    logger.info(f"  Batch size:     {args.batch_size} (x{args.gradient_accumulation_steps} accum)")
    logger.info(f"  Learning rate:  {args.lr} | Weight decay: {args.weight_decay}")

    ckpts = sorted(
        Path(args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if ckpts:
        logger.info(f"Resuming from latest checkpoint: {ckpts[-1]}")
        trainer.train(resume_from_checkpoint=str(ckpts[-1]))
    else:
        trainer.train()

    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"[OK] Final pre-trained model saved to: {final_path}")

    metrics = trainer.evaluate()
    logger.info(f"Final validation metrics: {metrics}")


if __name__ == "__main__":
    main()
