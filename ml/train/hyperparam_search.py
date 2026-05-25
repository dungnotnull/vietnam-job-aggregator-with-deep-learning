"""
Auto hyperparameter search until target validation F1 ≥ 0.80.

Grid searches over config space, saves best checkpoint.
Works with train_ner and train_classifier modules.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import torch

from ml.train.train_classifier import train_classifier
from ml.train.train_ner import train_ner

ML_DIR = Path(__file__).resolve().parent.parent


SEARCH_SPACE = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "batch_size": [8, 16],
    "warmup_ratio": [0.05, 0.10],
}

TARGET_F1 = 0.80


def grid_search_ner(dataset_path: str | None = None, cpu_only: bool = True) -> dict:
    configs = [
        {"learning_rate": lr, "batch_size": bs}
        for lr, bs in itertools.product(
            SEARCH_SPACE["learning_rate"], SEARCH_SPACE["batch_size"]
        )
    ]

    best_config = None
    best_f1 = 0.0

    for i, config in enumerate(configs):
        print(f"\n--- Config {i + 1}/{len(configs)}: {config} ---")
        result = train_ner(
            dataset_path=dataset_path,
            output_dir=f"{ML_DIR}/models/ner_search/config_{i}",
            cpu_only=cpu_only,
            num_epochs=5,
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
        )

        if result and result.get("f1", 0) > best_f1:
            best_f1 = result["f1"]
            best_config = config

        if best_f1 >= TARGET_F1:
            print(f"Target F1 ≥ {TARGET_F1} reached!")
            break

    print(f"\nBest NER config: {best_config}, F1: {best_f1:.4f}")
    return {"config": best_config, "f1": best_f1}


def grid_search_classifier(dataset_path: str | None = None, cpu_only: bool = True) -> dict:
    configs = [
        {"learning_rate": lr, "batch_size": bs}
        for lr, bs in itertools.product(
            SEARCH_SPACE["learning_rate"], SEARCH_SPACE["batch_size"]
        )
    ]

    best_config = None
    best_f1 = 0.0

    for i, config in enumerate(configs):
        print(f"\n--- Config {i + 1}/{len(configs)}: {config} ---")
        train_classifier(
            dataset_path=dataset_path,
            output_dir=f"{ML_DIR}/models/cls_search/config_{i}",
            cpu_only=cpu_only,
            num_epochs=5,
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
        )
        if best_f1 >= TARGET_F1:
            break

    return {"config": best_config, "f1": best_f1}


if __name__ == "__main__":
    result_ner = grid_search_ner(cpu_only=True)
    print(f"\nNER best: {result_ner}")

    result_cls = grid_search_classifier(cpu_only=True)
    print(f"Classifier best: {result_cls}")
