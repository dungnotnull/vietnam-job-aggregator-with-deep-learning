"""
Evaluate trained models against test set and select best checkpoint.

Metrics:
- Entity-level F1 (seqeval) — NER target ≥ 0.80
- Macro F1 — Classification target ≥ 0.80
- Skill coverage rate — % of test JDs with ≥ 1 skill extracted
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from datasets import load_from_disk
from seqeval.metrics import classification_report, f1_score as seqeval_f1
from sklearn.metrics import f1_score as sklearn_f1
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
import torch

ML_DIR = Path(__file__).resolve().parent.parent


CLS_LABELS = [
    "programming_language", "framework", "database", "cloud",
    "soft_skill", "language", "experience_level", "education",
]

TAG2ID = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}
ID2TAG = {v: k for k, v in TAG2ID.items()}


def evaluate_ner(model_path: str, dataset_path: str, device: str = "cpu") -> dict:
    ds = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []
    coverage_count = 0

    for example in ds["test"]:
        tokens = example["tokens"]
        true_bio = example["ner_labels"]

        tokenized = tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=256,
            return_tensors="pt",
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**tokenized)
            pred_ids = torch.argmax(outputs.logits, dim=2).squeeze(0)

        word_ids = tokenized.word_ids()
        true_seq: list[str] = []
        pred_seq: list[str] = []

        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            true_seq.append(true_bio[wid])
            pred_seq.append(ID2TAG.get(pred_ids[i].item(), "O"))

        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

        if any(label == "B-SKILL" for label in pred_seq):
            coverage_count += 1

    f1 = seqeval_f1(true_labels, pred_labels)
    coverage = coverage_count / len(ds["test"]) if len(ds["test"]) > 0 else 0.0

    print(f"\n=== NER Evaluation ===")
    print(f"Entity F1: {f1:.4f} (target ≥ 0.80)")
    print(f"Skill coverage: {coverage:.2%}")
    print(classification_report(true_labels, pred_labels))

    return {"entity_f1": f1, "coverage_rate": coverage}


def evaluate_classifier(model_path: str, dataset_path: str, device: str = "cpu") -> dict:
    ds = load_from_disk(dataset_path)
    mlb = MultiLabelBinarizer(classes=CLS_LABELS)
    mlb.fit(ds["train"]["classification_labels"])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for example in ds["test"]:
        text = (
            f"{example.get('job_title', '')}\n"
            f"{example.get('job_description', '')}\n"
            f"{example.get('requirements', '')}"
        )
        tokenized = tokenizer(
            text, truncation=True, max_length=256, return_tensors="pt"
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = model(**tokenized)
            probs = torch.sigmoid(outputs.logits).squeeze(0).cpu().numpy()

        pred = (probs > 0.5).astype(int)
        true = mlb.transform([example["classification_labels"]])[0]

        all_preds.append(pred)
        all_labels.append(true)

    preds_arr = np.stack(all_preds)
    labels_arr = np.stack(all_labels)

    macro_f1 = sklearn_f1(labels_arr, preds_arr, average="macro", zero_division=0)
    per_label_f1 = sklearn_f1(labels_arr, preds_arr, average=None, zero_division=0)

    print(f"\n=== Classification Evaluation ===")
    print(f"Macro F1: {macro_f1:.4f} (target ≥ 0.80)")
    print("\nPer-label F1 scores:")
    for label, f1 in zip(CLS_LABELS, per_label_f1):
        print(f"  {label}: {f1:.4f}")

    return {"macro_f1": macro_f1, "per_label_f1": dict(zip(CLS_LABELS, per_label_f1.tolist()))}


def evaluate_all(
    ner_path: str | None = None,
    cls_path: str | None = None,
    dataset_path: str | None = None,
    device: str = "cpu",
) -> dict:
    if ner_path is None:
        ner_path = str(ML_DIR / "models" / "phobert_ner_checkpoints" / "best")
    if cls_path is None:
        cls_path = str(ML_DIR / "models" / "xlmr_classifier_checkpoints" / "best")
    if dataset_path is None:
        dataset_path = str(ML_DIR / "data" / "processed_dataset")

    print(f"Evaluating on: {dataset_path}")
    print(f"NER model: {ner_path}")
    print(f"Classifier model: {cls_path}")

    ner_results = evaluate_ner(ner_path, dataset_path, device)
    cls_results = evaluate_classifier(cls_path, dataset_path, device)

    passed = ner_results["entity_f1"] >= 0.80 and cls_results["macro_f1"] >= 0.80
    print(f"\n{'✓ All targets met' if passed else '✗ Targets not met'}")

    return {
        "ner": ner_results,
        "classifier": cls_results,
        "targets_met": passed,
    }


if __name__ == "__main__":
    evaluate_all()
