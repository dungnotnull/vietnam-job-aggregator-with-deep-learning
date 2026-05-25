"""
Multi-task learning with shared PhoBERT encoder.

Jointly trains:
- NER (token-level BIO tagging)
- Multi-label classification (skill category labels)

Loss: total_loss = 0.6 * ner_loss + 0.4 * cls_loss

Supports QLoRA / PEFT fine-tuning via configuration flag.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import f1_score as sklearn_f1
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ML_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ML_DIR / "models" / "multitask_checkpoints"

NER_LABELS = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}
NER_ID2LABEL = {v: k for k, v in NER_LABELS.items()}
CLS_LABELS = [
    "programming_language", "framework", "database", "cloud",
    "soft_skill", "language", "experience_level", "education",
]
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256


class MultiTaskModel(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, num_ner_labels: int = 3, num_cls_labels: int = 8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.ner_head = nn.Linear(hidden_size, num_ner_labels)
        self.cls_head = nn.Linear(hidden_size, num_cls_labels)

        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, ner_labels=None, cls_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled_output = outputs.pooler_output      # (batch, hidden)

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        ner_logits = self.ner_head(sequence_output)  # (batch, seq_len, num_ner)
        cls_logits = self.cls_head(pooled_output)    # (batch, num_cls)

        loss = None
        if ner_labels is not None and cls_labels is not None:
            ner_loss = nn.CrossEntropyLoss(ignore_index=-100)(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))
            cls_loss = nn.BCEWithLogitsLoss()(cls_logits, cls_labels.float())
            loss = 0.6 * ner_loss + 0.4 * cls_loss

        return {"loss": loss, "ner_logits": ner_logits, "cls_logits": cls_logits}


class MultiTaskDataset(TorchDataset):
    def __init__(self, dataset, tokenizer, mlb=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.mlb = mlb

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = (
            f"{item.get('job_title', '')}\n"
            f"{item.get('job_description', '')}\n"
            f"{item.get('requirements', '')}"
        )

        tokenized = self.tokenizer(
            item["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = tokenized.word_ids()
        ner_labels = []
        for i, wid in enumerate(word_ids):
            if wid is None:
                ner_labels.append(-100)
            else:
                tag = item["ner_labels"][wid]
                label = NER_LABELS.get(tag, 0)
                if i > 0 and word_ids[i - 1] == wid and label == NER_LABELS["B-SKILL"]:
                    label = NER_LABELS["I-SKILL"]
                ner_labels.append(label)

        cls_labels = torch.zeros(len(CLS_LABELS))
        if self.mlb is not None:
            encoded = self.mlb.transform([item["classification_labels"]])[0]
            for i, val in enumerate(encoded):
                if val:
                    cls_labels[i] = 1.0
        else:
            for i, lab in enumerate(CLS_LABELS):
                if lab in item.get("classification_labels", []):
                    cls_labels[i] = 1.0

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "ner_labels": torch.tensor(ner_labels, dtype=torch.long),
            "cls_labels": cls_labels,
        }


def train_multitask(
    dataset_path: str | None = None,
    output_dir: str | None = None,
    cpu_only: bool = False,
    num_epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    use_qlora: bool = False,
) -> None:
    if dataset_path is None:
        dataset_path = str(ML_DIR / "data" / "processed_dataset")

    if output_dir is None:
        output_dir = str(CHECKPOINT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() and not cpu_only else "cpu")
    print(f"Using device: {device}")

    ds = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    mlb = MultiLabelBinarizer(classes=CLS_LABELS)
    mlb.fit(ds["train"]["classification_labels"])

    train_dataset = MultiTaskDataset(ds["train"], tokenizer, mlb)
    val_dataset = MultiTaskDataset(ds["validation"], tokenizer, mlb)
    test_dataset = MultiTaskDataset(ds["test"], tokenizer, mlb)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)

    model = MultiTaskModel(model_name=MODEL_NAME)

    if use_qlora:
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.05,
                bias="none",
            )
            model.encoder = get_peft_model(model.encoder, lora_config)
            print("QLoRA adapters added")
        except ImportError:
            print("PEFT not installed, skipping QLoRA")

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs["loss"].item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} — Train loss: {avg_train:.4f}, Val loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            model.encoder.save_pretrained(os.path.join(output_dir, "best"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best"))
            torch.save(model.state_dict(), os.path.join(output_dir, "best", "multitask_heads.pt"))
            print(f"  ✓ Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_multitask(cpu_only=True, num_epochs=2, batch_size=2)
