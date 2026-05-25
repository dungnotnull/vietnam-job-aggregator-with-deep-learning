"""
Dataset preprocessing for Vietnamese job descriptions.

Steps:
1. Download dataset from HuggingFace
2. Clean + normalize text
3. Deduplicate by Jaccard similarity on descriptions
4. Generate BIO tags via weak supervision (skill dictionary + regex)
5. Engineer multi-label taxonomy
6. Train/val/test split
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


DATA_DIR = Path(__file__).resolve().parent

SEED_SKILLS: dict[str, list[str]] = {
    "programming_language": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
        "Kotlin", "Swift", "Ruby", "PHP", "Scala", "Dart", "R", "MATLAB",
        "SQL", "PL/SQL", "T-SQL", "Bash", "Shell", "Perl", "Lua", "Groovy",
    ],
    "framework": [
        "React", "Angular", "Vue.js", "Next.js", "Nuxt", "Svelte", "Django",
        "Flask", "FastAPI", "Spring Boot", "ASP.NET", "Express", "NestJS",
        "Laravel", "Ruby on Rails", "Gin", "Echo", "TensorFlow", "PyTorch",
        "Keras", "Scikit-learn", "Pandas", "NumPy", "Spark", "Hadoop",
        "Kafka", "RabbitMQ", "Redis", "Elasticsearch", "Flink",
    ],
    "database": [
        "MySQL", "PostgreSQL", "MongoDB", "Cassandra", "DynamoDB", "BigQuery",
        "Snowflake", "Redshift", "SQLite", "Oracle DB", "SQL Server", "MariaDB",
        "Neo4j", "Couchbase", "Firebase", "Supabase", "ClickHouse",
    ],
    "cloud": [
        "AWS", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes",
        "Terraform", "Jenkins", "CI/CD", "GitLab CI", "GitHub Actions",
        "Ansible", "Helm", "ArgoCD", "Prometheus", "Grafana", "ELK",
        "S3", "EC2", "Lambda", "ECS", "EKS", "CloudFormation", "CloudFront",
        "Route53", "RDS", "IAM", "VPC", "SQS", "SNS", "CloudWatch",
    ],
    "soft_skill": [
        "communication", "teamwork", "problem-solving", "analytical thinking",
        "critical thinking", "leadership", "time management", "agile",
        "Scrum", "Kanban", "collaboration", "presentation", "negotiation",
        "mentoring", "adaptability", "self-learning", "English",
        "giao tiếp", "làm việc nhóm", "giải quyết vấn đề", "lãnh đạo",
        "quản lý thời gian", "chịu áp lực", "chủ động",
    ],
    "concept": [
        "OOP", "Microservices", "RESTful API", "GraphQL", "gRPC", "SOA",
        "Design Patterns", "SOLID", "TDD", "DDD", "Event-Driven",
        "Data Warehouse", "Data Lake", "ETL", "OLAP", "OLTP",
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "Blockchain", "WebSocket", "OAuth", "JWT", "SSO", "MFA",
    ],
}


def _normalize_vietnamese(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    return text


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = _clean_html(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = _normalize_vietnamese(text)
    return text


def _jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def deduplicate(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    texts = df["job_description"].tolist()
    keep: list[bool] = [True] * len(texts)

    for i in range(len(texts)):
        if not keep[i]:
            continue
        for j in range(i + 1, min(i + 101, len(texts))):
            if not keep[j]:
                continue
            if _jaccard_similarity(texts[i], texts[j]) >= threshold:
                keep[j] = False

    return df[keep].reset_index(drop=True)


def _build_skill_pattern(skills: dict[str, list[str]]) -> re.Pattern:
    all_skills: list[str] = []
    for skill_list in skills.values():
        for s in skill_list:
            all_skills.append(re.escape(s))
    all_skills.sort(key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(all_skills) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


_SKILL_PATTERN: re.Pattern | None = None


def _get_skill_pattern() -> re.Pattern:
    global _SKILL_PATTERN
    if _SKILL_PATTERN is None:
        _SKILL_PATTERN = _build_skill_pattern(SEED_SKILLS)
    return _SKILL_PATTERN


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+(?:[./]\w+)*\b|[^\w\s]", text)


def _match_skill(token: str) -> str | None:
    pattern = _get_skill_pattern()
    if pattern.search(token):
        return token
    return None


def annotate_bio(text: str) -> tuple[list[str], list[str]]:
    tokens = _tokenize(text)
    labels: list[str] = []
    in_skill = False
    for token in tokens:
        matched = _match_skill(token)
        if matched:
            labels.append("B-SKILL" if not in_skill else "I-SKILL")
            in_skill = True
        else:
            labels.append("O")
            in_skill = False
    return tokens, labels


def _infer_labels(row: dict[str, str]) -> list[str]:
    labels: list[str] = []
    full_text = (
        f"{row.get('job_title', '')} "
        f"{row.get('job_description', '')} "
        f"{row.get('requirements', '')}"
    ).lower()

    lang_map: dict[str, list[str]] = {
        "programming_language": [s.lower() for s in SEED_SKILLS["programming_language"]],
        "framework": [s.lower() for s in SEED_SKILLS["framework"]],
        "database": [s.lower() for s in SEED_SKILLS["database"]],
        "cloud": [s.lower() for s in SEED_SKILLS["cloud"]],
        "soft_skill": [s.lower() for s in SEED_SKILLS["soft_skill"]],
    }

    for cat, keywords in lang_map.items():
        if any(k in full_text for k in keywords):
            labels.append(cat)

    if any(
        w in full_text
        for w in ["english", "tiếng anh", "tieng anh", "toeic", "ielts"]
    ):
        labels.append("language")

    level = row.get("experience_level", "").lower()
    if any(
        w in level for w in ["senior", "cấp cao", "cao cấp", "quản lý", "trưởng phòng"]
    ):
        labels.append("experience_level")

    edu = row.get("education_level", "").lower()
    if any(w in edu for w in ["đại học", "dai hoc", "cử nhân", "bachelor", "thạc sĩ", "master"]):
        labels.append("education")

    return labels or ["general"]


def build_preprocessing_pipeline(
    sample_size: int | None = None,
    output_dir: Path | None = None,
) -> DatasetDict:
    if output_dir is None:
        output_dir = DATA_DIR

    print(f"Loading dataset from tinixai/vietnamese-job-descriptions...")
    ds = load_dataset("tinixai/vietnamese-job-descriptions")
    df = ds["train"].to_pandas()

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows from {len(ds['train'])} total")

    print(f"Rows before cleaning: {len(df)}")

    for col in ["job_title", "job_description", "requirements", "benefits"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(clean_text)

    df["description_length"] = df["job_description"].str.split().str.len()
    df = df[df["description_length"] >= 50].copy()
    print(f"After filtering short descriptions (< 50 tokens): {len(df)}")

    df = deduplicate(df)
    print(f"After deduplication: {len(df)}")

    df["tokens"] = None
    df["ner_labels"] = None
    for idx in df.index:
        text = (
            df.at[idx, "job_description"]
            + " "
            + df.at[idx, "requirements"]
        )
        tokens, labels = annotate_bio(text)
        df.at[idx, "tokens"] = tokens
        df.at[idx, "ner_labels"] = labels

    df["classification_labels"] = df.apply(_infer_labels, axis=1)

    n = len(df)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    result = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
    })

    result.save_to_disk(str(output_dir / "processed_dataset"))
    print(f"Saved processed dataset to {output_dir / 'processed_dataset'}")

    return result


if __name__ == "__main__":
    build_preprocessing_pipeline(sample_size=10000)
