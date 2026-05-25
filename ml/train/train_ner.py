"""
PhoBERT NER fine-tuning for Vietnamese job skill extraction.

Token classification with BIO tags: O, B-SKILL, I-SKILL
Supports both CPU (debug mode) and GPU training.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

ML_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ML_DIR / "models" / "phobert_ner_checkpoints"

TAG2ID: dict[str, int] = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}
ID2TAG: dict[int, str] = {v: k for k, v in TAG2ID.items()}
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256


def tokenize_and_align(example: dict, tokenizer) -> dict:
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    labels: list[int] = []
    word_ids = tokenized.word_ids()

    for i, word_id in enumerate(word_ids):
        if word_id is None:
            labels.append(-100)
        else:
            tag = example["ner_labels"][word_id]
            label = TAG2ID.get(tag, 0)
            if i > 0 and word_ids[i - 1] == word_id:
                if label == TAG2ID["B-SKILL"]:
                    label = TAG2ID["I-SKILL"]
            labels.append(label)

    tokenized["labels"] = labels
    return tokenized


def compute_metrics(predictions) -> dict[str, float]:
    logits, labels = predictions
    preds = np.argmax(logits, axis=2)

    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []
    for pred_seq, label_seq in zip(preds, labels):
        true_seq: list[str] = []
        pred_seq_labels: list[str] = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_seq.append(ID2TAG[l])
            pred_seq_labels.append(ID2TAG[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_labels)

    f1 = seqeval_f1(true_labels, pred_labels)
    return {"f1": f1}


def train_ner(
    dataset_path: str | None = None,
    output_dir: str | None = None,
    cpu_only: bool = False,
    num_epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> None:
    if dataset_path is None:
        dataset_path = str(ML_DIR / "data" / "processed_dataset")

    if output_dir is None:
        output_dir = str(CHECKPOINT_DIR)

    print(f"Loading dataset from {dataset_path}")
    ds = load_from_disk(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TAG2ID),
        id2label=ID2TAG,
        label2id=TAG2ID,
    )

    tokenized_ds = ds.map(
        lambda x: tokenize_and_align(x, tokenizer),
        remove_columns=list(ds["train"].features.keys()),
        desc="Tokenizing",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    device_kwargs: dict = {}
    if cpu_only:
        device_kwargs = {"no_cuda": True, "use_cpu": True}

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0 if cpu_only else 2,
        **device_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_ds["test"])
    print(f"Test results: {test_results}")

    report = compute_metrics((
        trainer.predict(tokenized_ds["test"]).predictions,
        np.array([ex["labels"] for ex in tokenized_ds["test"]]),
    ))
    print(f"Test F1: {report['f1']:.4f}")

    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    train_ner(cpu_only=True, num_epochs=2, batch_size=4)
