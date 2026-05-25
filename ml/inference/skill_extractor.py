"""
Skill extraction inference pipeline.

Loads trained NER + classification models and extracts:
- Skill spans from job descriptions (NER)
- Skill categories (multi-label classification)

Integrates with the existing output writer in agent/output_writer.py.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

ML_DIR = Path(__file__).resolve().parent.parent

NER_CHECKPOINT = ML_DIR / "models" / "phobert_ner_checkpoints" / "final"
CLS_CHECKPOINT = ML_DIR / "models" / "xlmr_classifier_checkpoints" / "final"

CLS_LABELS = [
    "programming_language", "framework", "database", "cloud",
    "soft_skill", "language", "experience_level", "education",
]


@dataclass
class SkillReport:
    skills: list[str] = field(default_factory=list)
    skill_frequencies: dict[str, int] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)

    def top_skills(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(
            self.skill_frequencies.items(), key=lambda x: x[1], reverse=True
        )[:n]

    def to_markdown(self) -> str:
        lines = ["## 🧠 Skill Intelligence Report"]
        lines.append("\n### Top Skills Required\n")
        lines.append("| Rank | Skill | Frequency |")
        lines.append("|------|-------|-----------|")
        for i, (skill, freq) in enumerate(self.top_skills(), 1):
            lines.append(f"| {i} | {skill} | {freq} |")

        if self.categories:
            lines.append("\n### Skill Categories Detected\n")
            for cat in sorted(self.categories):
                lines.append(f"- {cat.replace('_', ' ').title()}")

        return "\n".join(lines)


class SkillExtractor:
    def __init__(
        self,
        ner_model_path: str | None = None,
        cls_model_path: str | None = None,
        device: str = "cpu",
    ):
        self.device = device

        self.ner_model = None
        self.ner_tokenizer = None
        self.cls_model = None
        self.cls_tokenizer = None

        self._ner_path = ner_model_path or str(NER_CHECKPOINT)
        self._cls_path = cls_model_path or str(CLS_CHECKPOINT)

        self._load_models()

    def _load_models(self):
        try:
            self.ner_tokenizer = AutoTokenizer.from_pretrained(self._ner_path)
            self.ner_model = AutoModelForTokenClassification.from_pretrained(self._ner_path)
            self.ner_model.to(self.device)
            self.ner_model.eval()
        except Exception as e:
            print(f"NER model not loaded: {e}")
            self.ner_model = None

        try:
            self.cls_tokenizer = AutoTokenizer.from_pretrained(self._cls_path)
            self.cls_model = AutoModelForSequenceClassification.from_pretrained(self._cls_path)
            self.cls_model.to(self.device)
            self.cls_model.eval()
        except Exception as e:
            print(f"Classifier model not loaded: {e}")
            self.cls_model = None

    def extract_skills(self, text: str) -> list[str]:
        if self.ner_model is None or self.ner_tokenizer is None:
            return []

        raw_tokens = re.findall(r"\b\w+(?:[./]\w+)*\b|[^\w\s]", text)
        if not raw_tokens:
            return []

        tokenized = self.ner_tokenizer(
            raw_tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=256,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = self.ner_model(**tokenized)
            preds = torch.argmax(outputs.logits, dim=2).squeeze(0)

        word_ids = tokenized["word_ids"]
        skills: list[str] = []
        current_skill: list[str] = []

        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            label = preds[i].item()
            if label == 1:  # B-SKILL
                if current_skill:
                    skills.append(" ".join(current_skill))
                current_skill = [raw_tokens[wid]]
            elif label == 2 and current_skill and i > 0 and word_ids[i - 1] == wid - 1:
                if wid < len(raw_tokens):
                    current_skill.append(raw_tokens[wid])
            elif label == 0:
                if current_skill:
                    skills.append(" ".join(current_skill))
                    current_skill = []

        if current_skill:
            skills.append(" ".join(current_skill))

        return list(dict.fromkeys(skills))

    def classify(self, title: str, description: str, requirements: str) -> list[str]:
        if self.cls_model is None or self.cls_tokenizer is None:
            return []

        text = f"{title}\n{description}\n{requirements}"
        tokenized = self.cls_tokenizer(
            text, truncation=True, max_length=256, return_tensors="pt"
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = self.cls_model(**tokenized)
            probs = torch.sigmoid(outputs.logits).squeeze(0)

        predicted: list[str] = []
        for i, prob in enumerate(probs):
            if prob > 0.5 and i < len(CLS_LABELS):
                predicted.append(CLS_LABELS[i])

        return predicted

    def analyze(
        self, descriptions: list[dict[str, str]]
    ) -> SkillReport:
        all_skills: list[str] = []
        all_categories: set[str] = set()

        for job in descriptions:
            title = job.get("title", job.get("job_title", ""))
            desc = job.get("description", job.get("job_description", ""))
            reqs = job.get("requirements", "")

            skills = self.extract_skills(desc + " " + reqs)
            all_skills.extend(skills)

            if title or desc:
                cats = self.classify(title, desc, reqs)
                all_categories.update(cats)

        skill_counts = dict(Counter(all_skills))
        return SkillReport(
            skills=list(skill_counts.keys()),
            skill_frequencies=skill_counts,
            categories=sorted(all_categories),
        )


def rule_based_extract(text: str) -> SkillReport:
    """Fallback rule-based extraction when no trained model is available."""
    from ml.data.preprocess import SEED_SKILLS

    all_skills: list[str] = []
    all_categories: set[str] = set()

    text_lower = text.lower()

    for category, skill_list in SEED_SKILLS.items():
        found = False
        for skill in skill_list:
            pattern = re.escape(skill.lower())
            if re.search(rf"\b{pattern}\b", text_lower):
                all_skills.append(skill)
                found = True
        if found:
            all_categories.add(category)

    if any(
        w in text_lower for w in ["english", "tiếng anh", "toeic", "ielts"]
    ):
        all_categories.add("language")

    skill_counts = dict(Counter(all_skills))
    return SkillReport(
        skills=list(skill_counts.keys()),
        skill_frequencies=skill_counts,
        categories=sorted(all_categories),
    )
