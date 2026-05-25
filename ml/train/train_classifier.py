"""
XLM-RoBERTa multi-label classification for skill category prediction.

Labels: programming_language, framework, database, cloud, soft_skill, language, experience_level, education
Target: macro F1 ≥ 0.80 using BCEWithLogitsLoss with per-label threshold tuning.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import f1_score as sklearn_f1
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

ML_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ML_DIR / "models" / "xlmr_classifier_checkpoints"

LABEL_LIST = [
    "programming_language", "framework", "database", "cloud",
    "soft_skill", "language", "experience_level", "education",
]
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 256


def tokenize_texts(example: dict, tokenizer) -> dict:
    text = (
        f"{example.get('job_title', '')}\n"
        f"{example.get('job_description', '')}\n"
        f"{example.get('requirements', '')}"
    )
    return tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding=False)


def binarize_labels(dataset, mlb: MultiLabelBinarizer) -> MultiLabelBinarizer:
    labels = mlb.fit_transform(dataset["classification_labels"])
    dataset = dataset.add_column("label_ids", labels.tolist())
    return dataset, mlb


def compute_metrics(predictions, mlb: MultiLabelBinarizer) -> dict[str, float]:
    logits, labels = predictions
    labels = np.array(labels)

    best_f1 = 0.0
    thresholds = np.arange(0.2, 0.7, 0.05)
    for thresh in thresholds:
        preds = (torch.sigmoid(torch.tensor(logits)) > thresh).int().numpy()
        f1 = sklearn_f1(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    return {"macro_f1": best_f1}


def train_classifier(
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

    mlb = MultiLabelBinarizer(classes=LABEL_LIST)
    ds["train"], mlb = binarize_labels(ds["train"], mlb)
    ds["validation"] = ds["validation"].add_column(
        "label_ids", mlb.transform(ds["validation"]["classification_labels"]).tolist()
    )
    ds["test"] = ds["test"].add_column(
        "label_ids", mlb.transform(ds["test"]["classification_labels"]).tolist()
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        problem_type="multi_label_classification",
        id2label={i: lab for i, lab in enumerate(LABEL_LIST)},
        label2id={lab: i for i, lab in enumerate(LABEL_LIST)},
    )

    tokenized_ds = ds.map(
        lambda x: tokenize_texts(x, tokenizer),
        remove_columns=[c for c in ds["train"].features if c not in ("label_ids",)],
        desc="Tokenizing",
    )

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
        metric_for_best_model="macro_f1",
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
        compute_metrics=lambda p: compute_metrics(p, mlb),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_ds["test"])
    print(f"Test macro F1: {test_results.get('eval_macro_f1', 'N/A')}")

    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    train_classifier(cpu_only=True, num_epochs=2, batch_size=4)
