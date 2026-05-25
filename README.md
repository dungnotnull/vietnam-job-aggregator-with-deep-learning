# Vietnam Job Aggregator & Skill Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-67%20passed-brightgreen.svg)](tests/)

Aggregate job listings from **LinkedIn, VietnamWorks, ITViec, and TopDev** in one command.  
Get AI-powered skill intelligence from NLP models trained on real Vietnamese job descriptions.

```
=== Vietnam Job Aggregator ===

1. Field / Industry    → "Back End Developer"
2. Level               → "Senior"
3. Results per platform → 20
4. Skill analysis?      → yes

✓ LinkedIn: 20 jobs found
✓ VietnamWorks: 18 jobs found
✓ ITViec: 15 jobs found
✓ TopDev: 12 jobs found

✓ Report saved to results/jobs_back_end_developer_senior_20260524_103000.md
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/your-org/vietnam-job-aggregator.git
cd vietnam-job-aggregator

# Install core dependencies (scraper only — no ML required)
pip install httpx beautifulsoup4 lxml rich python-dotenv

# Run
python agent/main.py
```

That's it. The agent searches all 4 platforms concurrently and writes a structured Markdown report to `results/`.

---

## CLI Walkthrough

The agent guides you through 4 prompts:

| Step | Description | Example |
|------|-------------|---------|
| 1. Field | Job title or industry (free text in English) | `Data Engineer`, `Frontend Developer` |
| 2. Level | Seniority filter | Intern, Junior, Mid-level, Senior, Lead, Manager, Any |
| 3. Count | Results per platform | 10, 20, 50, 100, All |
| 4. Skill | Run AI skill analysis? | Uses trained model if available, or rule-based dictionary fallback |

All platforms are searched **in parallel** via `ThreadPoolExecutor`. A progress spinner shows live status for each.

---

## What You Get

```markdown
# Vietnam Job Search Results
| Search Query | Back End Developer |
| Level | Senior |
| Total Jobs Found | 65 |

## LinkedIn — 20 results
| # | Job Title | Company | Location | Posted | Salary | Link |
|---|-----------|---------|----------|--------|--------|------|
| 1 | Senior Backend Engineer | TechCorp | HCMC | 2 days ago | — | [View →](…) |

## 🧠 Skill Intelligence Report
### Top Skills Required
| Rank | Skill | Frequency |
|------|-------|-----------|
| 1 | Python | 8 |
| 2 | AWS | 6 |
| 3 | Docker | 5 |

### Skill Categories Detected
- Programming Language
- Cloud
- Database
```

---

## Supported Platforms

| Platform | Auth Required | Strategy | Coverage |
|----------|:---:|----------|----------|
| [LinkedIn](https://linkedin.com/jobs) | No | Public guest API | All jobs |
| [VietnamWorks](https://vietnamworks.com) | No | REST API + HTML fallback | All jobs |
| [ITViec](https://itviec.com) | No | HTML scraping | IT/tech roles |
| [TopDev](https://topdev.vn) | No | REST API + HTML fallback | IT/tech roles |

All 4 platforms work out of the box — **no login, no API keys, no browser needed**.

---

## Project Structure

```
vietnam-job-aggregator/
├── agent/                          # Job aggregation agent
│   ├── main.py                     # CLI entry point
│   ├── filters.py                  # Query builder (field → platform-specific params)
│   ├── output_writer.py            # Markdown report generation + ML integration
│   ├── security.py                 # Zero-persistence audit
│   └── scrapers/
│       ├── base_scraper.py         # Abstract interface + Job dataclass + @retry
│       ├── linkedin_scraper.py     # LinkedIn guest API
│       ├── vietnamworks_scraper.py # VietnamWorks
│       ├── itviec_scraper.py       # ITViec
│       └── topdev_scraper.py       # TopDev
│
├── ml/                             # Skill intelligence (optional)
│   ├── data/preprocess.py          # Dataset download + weak-supervision annotation
│   ├── train/
│   │   ├── train_ner.py            # PhoBERT token classification
│   │   ├── train_classifier.py     # XLM-R multi-label classification
│   │   ├── train_multitask.py      # Joint NER + classification
│   │   ├── train_qlora.py          # 4-bit QLoRA fine-tuning
│   │   ├── hyperparam_search.py    # Grid search → target F1 ≥ 0.80
│   │   └── evaluate.py             # Evaluation metrics
│   └── inference/
│       └── skill_extractor.py      # Production inference + rule-based fallback
│
├── tests/
│   ├── test_base_scraper.py        # Retry, Job dataclass, error hierarchy
│   ├── test_filters.py             # Field mapping, level mapping
│   ├── test_scrapers.py            # Auth checks, URL building
│   ├── test_edge_cases.py          # Salary parsing, location, malformed HTML
│   └── test_skill_extractor.py     # Skill extraction + output integration
│
├── results/                        # Generated .md reports
├── requirements.txt
├── pyproject.toml
├── README.md
└── CLAUDE.md                       # Developer guide
```

---

## Adding a New Platform

Each scraper follows the same pattern. Here's how to add one for a new job board:

### Step 1 — Create the scraper (`agent/scrapers/newplatform_scraper.py`)

```python
from agent.scrapers.base_scraper import MAX_RETRIES, BaseScraper, Job, retry

class NewPlatformScraper(BaseScraper):
    platform = "NewPlatform"           # Display name in output

    BASE_URL = "https://jobs.newplatform.com/api/search"

    def requires_auth(self) -> bool:
        return False                   # True if login is needed

    @retry(max_retries=MAX_RETRIES)
    def search(self, field: str, level: str, count: int = 20) -> list[Job]:
        # 1. Build URL / API params from field + level
        # 2. Make HTTP request (use httpx)
        # 3. Parse response into Job objects
        # 4. Return jobs[:count]
        ...

    def _build_url(self, field: str, level: str) -> str:
        ...
```

### Step 2 — Register it

Add one line to `agent/main.py`:

```python
SCRAPERS: list[tuple[str, object]] = [
    ("LinkedIn", LinkedInScraper()),
    ("VietnamWorks", VietnamWorksScraper()),
    ("ITViec", ITViecScraper()),
    ("TopDev", TopDevScraper()),
    ("NewPlatform", NewPlatformScraper()),   # ← here
]
```

That's it — the agent automatically includes it in parallel searches, progress display, and the output report.

### Optional — Add level mapping

If the new platform has its own level codes, add them to `agent/filters.py`:

```python
NEWPLATFORM_LEVEL_MAP = {
    "Intern": "1", "Junior": "2", "Mid-level": "3",
    "Senior": "4", "Lead": "5", "Manager": "6", "Any": "",
}
```

---

## Customizing for Another Country

This project targets **Vietnam**, but the architecture is country-agnostic. To adapt it:

1. **Replace the scrapers** with platforms in your target country
2. **Update location filters** — change `"location": "Vietnam"` in `linkedin_scraper.py` and `filters.py`
3. **Retrain the ML model** on job descriptions in your language (or use the rule-based fallback)

The `BaseScraper`, `@retry`, `Job` dataclass, `ThreadPoolExecutor` concurrency, and report generation all work identically for any country.

---

## Skill Intelligence (ML Pipeline)

The optional ML module trains NLP models on Vietnamese job descriptions to extract skills and predict skill categories. If you skip training, the agent falls back to a **rule-based dictionary** of 500+ tech and soft skills — no GPU needed.

### Quick ML Setup (CPU-only, small sample)

```bash
# Install ML dependencies
pip install transformers datasets torch scikit-learn seqeval peft accelerate bitsandbytes

# Preprocess dataset (10k sample)
python ml/data/preprocess.py

# Train a tiny NER model on CPU (3 epochs)
python ml/train/train_ner.py     # default: cpu_only=True, 2 epochs, batch_size=4
```

### Full Training (GPU recommended)

```bash
# Preprocess full dataset (606k rows → ~500k after dedup)
python ml/data/preprocess.py                          # remove sample_size=10000 default

# Train with GPU
python -c "from ml.train.train_ner import train_ner; train_ner(cpu_only=False)"
python -c "from ml.train.train_classifier import train_classifier; train_classifier(cpu_only=False)"
python -c "from ml.train.train_multitask import train_multitask; train_multitask(cpu_only=False)"

# Hyperparameter search
python ml/train/hyperparam_search.py

# Evaluate
python ml/train/evaluate.py
```

### Models

| Model | Task | Base | Target F1 |
|-------|------|------|:---:|
| PhoBERT NER | Extract skill spans from job text | `vinai/phobert-base-v2` | ≥ 0.80 |
| XLM-R Classifier | Predict skill categories | `xlm-roberta-base` | ≥ 0.80 |
| Multi-task | Joint NER + classification | Shared PhoBERT encoder | — |
| QLoRA | Memory-efficient fine-tuning | `vinai/phobert-base-v2` + 4-bit | — |

### Inference

```python
from ml.inference.skill_extractor import SkillExtractor

extractor = SkillExtractor()
report = extractor.analyze([
    {"title": "Backend Developer", "description": "Looking for Python, AWS, Docker..."},
    {"title": "Data Engineer", "description": "ETL pipelines, Spark, SQL, Kafka..."},
])
print(report.to_markdown())
# Renders a skill frequency table + category list
```

### Dataset

The training data comes from [tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions) on HuggingFace — 606,878 real Vietnamese job postings. The preprocessing pipeline:

1. Vietnamese NFC normalization + HTML stripping
2. Filter short descriptions (< 50 tokens)
3. Deduplicate by Jaccard similarity (threshold ≥ 0.85)
4. Weak supervision BIO annotation from a seed dictionary of 500+ skills
5. 80/10/10 train/val/test split

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v     # 67 tests
```

---

## Security

This project enforces **zero persistent user data**:

- No credentials stored on disk, in `.env`, or in any database
- No session cookies persisted
- All scraped data exists only in-memory and the output `.md` file
- `security.py` runs at startup — exits immediately if it finds leaked session caches or credential patterns in the environment

---

## Dependencies

**Core (scraper only):**

```
httpx>=0.27         # HTTP client
beautifulsoup4>=4.12 # HTML parsing
lxml                # Fast XML/HTML parser
rich                # Terminal UI
python-dotenv       # Environment variables
```

**ML (optional):**

```
transformers>=4.40  # HuggingFace models
torch>=2.2          # PyTorch
datasets>=2.19      # HuggingFace datasets
scikit-learn        # Metrics
seqeval             # NER evaluation
peft>=0.10          # LoRA / QLoRA
accelerate>=0.29    # Training acceleration
bitsandbytes>=0.43  # 4-bit quantization
pandas              # Data manipulation
```

Install everything: `pip install -r requirements.txt`

---

## Contributing

1. Fork the repo and create a branch
2. Write tests for any new scraper or feature
3. Ensure `pytest tests/ -v` passes
4. Follow PEP 8 — enforced by `ruff check .`
5. Submit a PR

See `CLAUDE.md` for detailed architecture and development guidelines.

---

## License

MIT License. See [LICENSE](LICENSE) file.

The dataset [tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions) is used for research and non-commercial purposes — check the dataset's license before commercial deployment.

---

## FAQ

**Why no login for LinkedIn?**  
LinkedIn's guest jobs API (`jobs-guest/jobs/api/seeMoreJobPostings/search`) serves full job listings without authentication. No browser, no manual login needed.

**Does it work for non-Vietnam locations?**  
LinkedIn searches can be adapted by changing the `location` parameter. The other platforms are Vietnam-specific — you'd swap in platforms relevant to your country.

**Can I add more result filters?**  
Yes — extend `filters.py` with additional query parameters, then use them in your scraper's `search()` method.

**Do I need a GPU for the ML?**  
No. The rule-based fallback (`rule_based_extract`) uses a dictionary of 500+ skills and works instantly on CPU. GPU is only needed for training or inference with the full PhoBERT/XLM-R models.

**How do I add a web UI?**  
The architecture cleanly separates data collection (`agent/`) from presentation (`output_writer.py`). You can replace or extend `output_writer.py` to generate JSON, feed a database, or power a Streamlit/Gradio UI.
