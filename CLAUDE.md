# CLAUDE.md — Vietnam Job Aggregator & Skill Intelligence Agent

## Project Overview

This project builds an intelligent agent that solves two core problems for Vietnamese job seekers:

1. **Job Aggregation** — Automatically search, filter, and compile job listings from LinkedIn, VietnamWorks, ITViec, and TopDev into a single structured report.
2. **Skill Intelligence** — ML/NLP pipeline (PhoBERT NER + XLM-R classification) trained on Vietnamese job descriptions to surface in-demand skills for any role.

---

## Repository Structure

```
vietnam-job-aggregator/
├── agent/                        # Job aggregation agent
│   ├── main.py                   # CLI entry point — interactive Rich prompts
│   ├── filters.py                # Query builder — field/level → platform-specific query objects
│   ├── output_writer.py          # Renders results + ML skill analysis → results/*.md
│   ├── security.py               # Zero persistence enforcement
│   ├── browser/
│   │   ├── cloak_browser.py      # Playwright wrapper (currently unused — all scrapers use httpx)
│   │   └── auth_handler.py       # Auth prompt flow (no platforms currently need it)
│   └── scrapers/
│       ├── base_scraper.py       # Abstract BaseScraper, Job dataclass, @retry decorator
│       ├── linkedin_scraper.py   # LinkedIn via guest API (no auth) — httpx
│       ├── vietnamworks_scraper.py  # API + HTML fallback
│       ├── itviec_scraper.py     # HTML scraping with fallback selectors
│       └── topdev_scraper.py     # API + HTML fallback
│
├── ml/                           # Skill intelligence ML pipeline
│   ├── data/
│   │   └── preprocess.py         # Download, clean, deduplicate, weak-supervision BIO annotation, 80/10/10 split
│   ├── train/
│   │   ├── train_ner.py          # PhoBERT token classification (O/B-SKILL/I-SKILL)
│   │   ├── train_classifier.py   # XLM-R multi-label classification (8 categories)
│   │   ├── train_multitask.py    # Shared encoder: NER + classification jointly
│   │   ├── train_qlora.py        # 4-bit QLoRA fine-tuning with PEFT
│   │   ├── hyperparam_search.py  # Grid search → target F1 ≥ 0.80
│   │   └── evaluate.py           # Entity F1, macro F1, per-label breakdown, coverage rate
│   ├── inference/
│   │   └── skill_extractor.py    # SkillExtractor class + rule_based_extract() fallback
│   └── models/                   # Checkpoint output directory
│
├── tests/
│   ├── test_base_scraper.py      # @retry decorator, Job dataclass, error hierarchy
│   ├── test_filters.py           # Field mapping, level mapping, build_query
│   ├── test_scrapers.py          # Auth checks, URL building
│   ├── test_edge_cases.py        # Salary extraction, location parsing, parser edge cases
│   └── test_skill_extractor.py   # Rule-based extraction, SkillReport, output integration
│
├── results/                      # Output directory for job reports (.md)
├── requirements.txt
├── pyproject.toml
├── CLAUDE.md
└── PROJECT-DETAIL.md
```

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the job agent
python agent/main.py

# Run tests
pytest tests/ -v

# Preprocess dataset (ML)
python ml/data/preprocess.py

# Train models (GPU recommended, CPU supported with cpu_only=True)
python ml/train/train_ner.py
python ml/train/train_classifier.py
python ml/train/train_multitask.py
python ml/train/train_qlora.py
```

---

## Agent Behavior

### CLI Flow

```
=== Vietnam Job Search Agent ===
1. Field / Industry     e.g. "Back End Developer"
2. Level                Intern / Junior / Mid-level / Senior / Lead / Manager / Any
3. Results per platform 10 / 20 / 50 / 100 / All
4. Skill analysis?      y/n (ML model if trained, rule-based fallback otherwise)

→ Scrapes all 4 platforms concurrently (ThreadPoolExecutor)
→ Outputs results/jobs_<field>_<level>_<timestamp>.md
```

### Platform Auth

| Platform | Auth Required | Strategy |
|----------|:---:|----------|
| LinkedIn | No | Public guest API: `jobs-guest/jobs/api/seeMoreJobPostings/search` |
| VietnamWorks | No | REST API (`ms.vietnamworks.com`) + HTML fallback |
| ITViec | No | HTML scraping with pagination |
| TopDev | No | REST API (`api.topdev.vn`) + HTML fallback |

All 4 platforms are uniform — no browser auth needed.

---

## Scraper Architecture

### BaseScraper Interface

```python
class BaseScraper(ABC):
    platform: str = ""

    @abstractmethod
    def requires_auth(self) -> bool: ...
    @abstractmethod
    def search(self, field: str, level: str, count: int = 20) -> list[Job]: ...
    @abstractmethod
    def _build_url(self, field: str, level: str) -> str: ...
```

### Job Dataclass

```python
@dataclass(slots=True)
class Job:
    title: str
    company: str
    location: str
    level: str
    url: str
    posted_date: str
    description_snippet: str
    platform: str
    salary: str = ""
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
```

### Rate Limiting & Retry

All scrapers use `_rate_limit(2.0)` between requests and the `@retry(max_retries=3)` decorator. Exceptions thrown: `AuthRequiredError`, `RateLimitError`, `ParseError` (all subclass `ScraperError`).

---

## ML Pipeline

### Dataset

- Source: [tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions) (606k rows)
- Tasks: NER (skill span extraction) + multi-label classification (8 skill categories)

### Models

| Model | Task | Technique |
|-------|------|-----------|
| `vinai/phobert-base-v2` | NER (BIO tagging) | Fine-tuning |
| `xlm-roberta-base` | Skill category classification | Fine-tuning |
| Shared PhoBERT encoder | NER + classification jointly | Multi-task learning |
| PhoBERT-large (QLoRA) | Advanced classification | 4-bit quantization + LoRA |

### Preprocessing (`ml/data/preprocess.py`)

1. Download from HuggingFace
2. Vietnamese NFC normalization, HTML stripping
3. Filter short descriptions (< 50 tokens)
4. Deduplicate by Jaccard similarity (threshold ≥ 0.85)
5. Weak supervision BIO annotation (500+ seed skills)
6. Multi-label taxonomy engineering
7. 80/10/10 train/val/test split

### Training

All training scripts support `cpu_only=True` for CPU fallback. GPU recommended.

```python
# NER
from ml.train.train_ner import train_ner
train_ner(cpu_only=False, num_epochs=20, batch_size=16, learning_rate=2e-5)

# Classifier
from ml.train.train_classifier import train_classifier
train_classifier(cpu_only=False)

# Multi-task
from ml.train.train_multitask import train_multitask
train_multitask(use_qlora=True)

# QLoRA
from ml.train.train_qlora import train_qlora
train_qlora(base_model="vinai/phobert-base-v2")

# Hyperparameter search
from ml.train.hyperparam_search import grid_search_ner, grid_search_classifier
grid_search_ner()
```

### Evaluation (`ml/train/evaluate.py`)

| Metric | Target |
|--------|--------|
| Entity-level F1 (seqeval) | ≥ 0.80 |
| Macro F1 (classification) | ≥ 0.80 |
| Skill coverage rate | ≥ 85% |

### Inference (`ml/inference/skill_extractor.py`)

```python
from ml.inference.skill_extractor import SkillExtractor, rule_based_extract

extractor = SkillExtractor()
report = extractor.analyze([{"title": "...", "description": "..."}])
print(report.to_markdown())  # Renders skill table for output

# Fallback when no trained model
report = rule_based_extract("We need Python, Docker, and AWS experience")
```

---

## Security

**Zero persistent user data.** No credentials, cookies, or sessions are stored to disk. The `.env` file is for API keys only. `security.py` runs at startup to audit for leaked session caches, environment variable secrets, and database files. If violations are found, the agent exits.

---

## Common Commands

```bash
# Full agent run
python agent/main.py

# Tests (67 total)
pytest tests/ -v

# Lint
ruff check . && mypy agent/ ml/

# Preprocess ML dataset
python ml/data/preprocess.py

# Train all models
python ml/train/train_ner.py
python ml/train/train_classifier.py
python ml/train/train_multitask.py

# Evaluate
python ml/train/evaluate.py
```

---

## Concurrency

Scraping runs all 4 platforms in parallel via `ThreadPoolExecutor` (in `main.py:run_scrapers_concurrent`). Each scraper is I/O-bound (HTTP requests), so threads are appropriate. Rate limiting is per-scraper (internal to each `search()` method), so there's no contention.

---

## Output Format

```markdown
# Vietnam Job Search Results
| Search Query | Back End Developer |
| Level | Senior |
| Date | 2025-06-15 10:30:00 |
| Total Jobs Found | 47 |

## LinkedIn — 12 results
| # | Job Title | Company | Location | Posted | Salary | Link |
|---|-----------|---------|----------|--------|--------|------|
| 1 | Senior Backend Engineer | TechCorp | HCMC | 2 days ago | — | [View →](...) |

## 🧠 Skill Intelligence Report
### Top Skills Required
| Rank | Skill | Frequency |
|------|-------|-----------|
| 1 | Python | 8 |
| 2 | AWS | 6 |

### Skill Categories Detected
- Programming Language
- Cloud
```
