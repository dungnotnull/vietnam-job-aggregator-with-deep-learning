"""
QLoRA fine-tuning using PEFT on PhoBERT-large or XGLM.

Runs on a single 16GB GPU with 4-bit quantization.
Provides an advanced alternative when base PhoBERT fine-tuning is insufficient.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

ML_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ML_DIR / "models" / "qlora_checkpoints"

CLS_LABELS = [
    "programming_language", "framework", "database", "cloud",
    "soft_skill", "language", "experience_level", "education",
]


def create_qlora_model(
    base_model: str = "vinai/phobert-base-v2",
    num_labels: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        quantization_config=bnb_config,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_qlora(
    dataset_path: str | None = None,
    output_dir: str | None = None,
    base_model: str = "vinai/phobert-base-v2",
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    gradient_accumulation_steps: int = 4,
) -> None:
    if dataset_path is None:
        dataset_path = str(ML_DIR / "data" / "processed_dataset")
    if output_dir is None:
        output_dir = str(CHECKPOINT_DIR)

    print(f"Loading dataset from {dataset_path}")
    ds = load_from_disk(dataset_path)

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=CLS_LABELS)
    mlb.fit(ds["train"]["classification_labels"])

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        texts = [
            f"{t}\n{d}\n{r}"
            for t, d, r in zip(
                examples["job_title"],
                examples["job_description"],
                examples.get("requirements", [""] * len(examples["job_title"])),
            )
        ]
        tokenized = tokenizer(
            texts, truncation=True, max_length=256, padding="max_length"
        )
        tokenized["labels"] = mlb.transform(examples["classification_labels"]).tolist()
        return tokenized

    tokenized_ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in ds["train"].features],
        desc="Tokenizing",
    )

    model = create_qlora_model(base_model=base_model)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        bf16=True,
        fp16=False,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    train_qlora()
