# PROJECT-DETAIL.md — Vietnam Job Aggregator & Skill Intelligence

## 1. Project Summary

**Name:** Vietnam Job Aggregator & Skill Intelligence Agent  
**Goal:** Help Vietnamese job seekers (1) instantly aggregate job listings from 5 major platforms using smart filters, and (2) understand what skills and qualifications are currently in demand in Vietnam for their target role — powered by an NLP model fine-tuned on real Vietnamese job descriptions.

**Core Value Propositions:**
- Save hours of manual browsing across LinkedIn, VietnamWorks, ITViec, TopDev.
- Surface data-driven skill gap insights specific to the Vietnamese hiring market.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        CLI / Agent UI                    │
│           (main.py — interactive prompts via Rich)       │
└────────────────────────┬────────────────────────────────┘
                         │
           ┌─────────────▼──────────────┐
           │       Agent Orchestrator    │
           │   filters.py + main.py      │
           └──┬──────────────────────┬──┘
              │                      │
   ┌──────────▼──────────┐  ┌───────▼──────────────────┐
   │   Browser Layer      │  │   Skill Intelligence ML   │
   │  CloakBrowser        │  │   (inference mode)        │
   │  + auth_handler.py   │  │   skill_extractor.py      │
   └──────────┬──────────┘  └───────────────────────────┘
              │
   ┌──────────▼──────────────────────────────────────────┐
   │                     Scrapers                         │
   │  LinkedIn │ VietnamWorks │ ITViec │ TopDev  │
   └──────────────────────────┬──────────────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │    output_writer.py  │
                   │  results/*.md        │
                   └─────────────────────┘
```

---

## 3. Module 1 — Job Aggregation Agent

### 3.1 User Interaction Flow

```
START
  │
  ├─ Ask: Field / Industry (free text, English)
  ├─ Ask: Level (Intern / Junior / Mid / Senior / Lead / Manager / Any)
  ├─ Ask: Count per platform (10 / 20 / 50 / 100 / All)
  │
  ├─ For each platform:
  │     ├─ No platform currently requires authentication
  │     ├─ Navigate to search/filter URL with constructed query
  │     ├─ Scrape job cards up to requested count
  │     └─ Collect: title, company, location, level, posted_date, url, snippet
  │
  ├─ Run ML skill inference on collected descriptions (optional, prompt user)
  └─ Write results to results/jobs_<field>_<level>_<timestamp>.md
```

### 3.2 Browser Integration

A lightweight Playwright wrapper (`browser/cloak_browser.py`) is available for platforms that need browser-based scraping, but is currently unused since all scrapers use httpx directly.

### 3.3 Platform Scraping Details

#### LinkedIn
- **Auth required:** No. LinkedIn's guest API (`jobs-guest/jobs/api/seeMoreJobPostings/search`) serves full job listings without login.
- **Strategy:** Direct httpx GET to the guest API endpoint with keyword, location, level, and time-range filters.
- **Level codes:** Internship=1, Entry=2, Associate=3, Mid-Senior=4, Director=5, Executive=6
- **Data extracted:** Job title, company name, location, posting date, job URL
- **Rate limit:** 1 request per 2 seconds with random jitter

#### VietnamWorks
- **Auth required:** No. Public job listings fully accessible.
- **Base URL:** `https://www.vietnamworks.com/tim-viec-lam/<slug>?industry=<id>&level=<id>`
- **Strategy:** REST API endpoint available at `https://ms.vietnamworks.com/job-search/v1.0/jobs` — use JSON API where possible, fall back to HTML scraping.
- **Data extracted:** Full job card data including salary range if available

#### ITViec
- **Auth required:** No.
- **Base URL:** `https://itviec.com/it-jobs/<query>?job_levels%5B%5D=<level>`
- **Strategy:** HTML scraping with BeautifulSoup; paginate through results
- **Note:** Focused on IT/tech roles; returns high-relevance results for software jobs

#### TopDev
- **Auth required:** No.
- **Base URL:** `https://topdev.vn/jobs?q=<query>&level=<level>`
- **Strategy:** HTML scraping or JSON API at `https://api.topdev.vn/td/v2/jobs`
- **Note:** Also IT-focused; strong for senior/lead engineering roles

### 3.4 Query Construction

```python
# filters.py
def build_query(field: str, level: str) -> PlatformQueries:
    return PlatformQueries(
        linkedin=LinkedInQuery(keywords=field, location="Vietnam", experience_level=map_level(level)),
        vietnamworks=VNWQuery(keyword=field, level_id=map_vnw_level(level)),
        itviec=ITViecQuery(query=field, job_levels=[map_itviec_level(level)]),
        topdev=TopDevQuery(q=field, level=map_topdev_level(level)),
    )
```

### 3.5 Output Format

Results file: `results/jobs_{field}_{level}_{YYYYMMDD_HHMMSS}.md`

```markdown
# 🇻🇳 Vietnam Job Search Results

| | |
|---|---|
| **Search Query** | Senior Data Engineer |
| **Level** | Senior |
| **Date** | 2025-01-15 10:30:22 |
| **Total Jobs Found** | 73 |

---

## LinkedIn — 18 results

| # | Job Title | Company | Location | Posted | Salary | Link |
|---|-----------|---------|----------|--------|--------|------|
| 1 | Senior Data Engineer | Grab Vietnam | HCMC | 1 day ago | $2,500–4,000 | [View →](https://linkedin.com/jobs/...) |
| 2 | ... | ... | ... | ... | ... | ... |

---

## VietnamWorks — 15 results
...

---

## ITViec — 12 results
...

---

## TopDev — 14 results
...

---

## 🧠 Skill Intelligence Report
*Powered by ML model trained on Vietnamese job descriptions*
*Based on analysis of 847 similar job postings in Vietnam (last 12 months)*

### Top Skills Required for: Senior Data Engineer in Vietnam

| Rank | Skill / Requirement | Frequency | Trend |
|------|---------------------|-----------|-------|
| 1 | Python | 94% | ↑ Increasing |
| 2 | SQL / BigQuery | 91% | → Stable |
| 3 | Apache Spark | 78% | ↑ Increasing |
| 4 | Apache Kafka | 65% | ↑ Increasing |
| 5 | Cloud (AWS/GCP/Azure) | 71% | ↑ Increasing |
| 6 | Data Warehouse design | 58% | → Stable |
| 7 | English (B2+) | 52% | → Stable |
| 8 | 3+ years experience | 83% | → Stable |

### Commonly Requested Soft Skills
- Problem-solving, analytical thinking (67%)
- Team collaboration / Agile (55%)
- Communication in English (52%)
```

---

## 4. Module 2 — Skill Intelligence ML Pipeline

### 4.1 Dataset

| Property | Value |
|----------|-------|
| Source | [huggingface.co/datasets/tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions) |
| Language | Vietnamese (primary), some English |
| Content | Job titles, descriptions, requirements, benefits |
| Task | Named Entity Recognition (skills) + Multi-label Classification |

### 4.2 Data Preprocessing

**Step 1 — Download**
```python
from datasets import load_dataset
ds = load_dataset("tinixai/vietnamese-job-descriptions")
```

**Step 2 — Cleaning**
- Remove HTML tags, special characters, excess whitespace
- Normalize Vietnamese diacritics (NFC normalization)
- Deduplicate by job description similarity (Jaccard ≥ 0.85 → keep one)
- Filter: remove entries with description < 50 tokens

**Step 3 — Annotation for NER**
- Extract skill spans using a seed dictionary of 500+ tech/soft skills (Vietnamese + English)
- Use weak supervision (Snorkel/regex) to generate initial BIO tags
- Manual review sample: 200 documents for quality check

**Step 4 — Label Engineering**
- Multi-label taxonomy: `["programming_language", "framework", "database", "cloud", "soft_skill", "language", "experience_level", "education"]`
- Each job description gets one or more labels from the taxonomy

**Step 5 — Train/Validation/Test Split**
```
Total dataset → 80% train / 10% validation / 10% test
Stratified split by job category to ensure coverage
```

### 4.3 Model Architecture

#### Model A — PhoBERT Fine-tuning (NER)

```python
# phobert_ner.py
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_name = "vinai/phobert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),  # BIO tags: B-SKILL, I-SKILL, O
)
```

**Training config:**
- Optimizer: AdamW, lr=2e-5, weight decay=0.01
- Batch size: 16 (gradient accumulation ×4 = effective 64)
- Epochs: up to 20 with early stopping (patience=3)
- Scheduler: linear warmup (10% steps) then linear decay
- Max sequence length: 256 tokens

#### Model B — XLM-R Multi-label Classification

```python
# xlmr_classifier.py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=8,          # skill taxonomy categories
    problem_type="multi_label_classification",
)
```

**Loss:** BCEWithLogitsLoss  
**Threshold tuning:** Grid search over [0.3, 0.4, 0.5, 0.6] per label to maximize macro F1

#### Model C — Multi-task Learning (Shared Encoder)

```
                    [CLS] token1 token2 ... [SEP]
                              │
                    ┌─────────▼─────────┐
                    │  Shared Encoder    │
                    │  (PhoBERT-base)    │
                    └──────┬─────┬──────┘
                           │     │
              ┌────────────▼─┐ ┌─▼───────────────────┐
              │  NER Head     │ │ Classification Head   │
              │ (token-level) │ │ (sequence-level)      │
              └───────────────┘ └─────────────────────-┘
```

Task weighting: `total_loss = 0.6 * ner_loss + 0.4 * cls_loss`  
(weights tuned based on validation performance)

#### Model D — QLoRA / PEFT Fine-tuning (Advanced)

For production-grade skill reasoning using a larger base LLM:

```python
# qlora_finetune.py
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,                    # LoRA rank
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(base_model, lora_config)
```

**Base model options:** `vinai/phobert-large`, `facebook/xglm-564M`, or `google/gemma-2b`  
**Memory:** Runs on a single 16GB GPU with 4-bit quantization

### 4.4 Auto Hyperparameter Tuning

The trainer runs automatic hyperparameter search until validation F1 ≥ 0.80:

```python
# trainer.py
search_space = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "batch_size": [8, 16, 32],
    "warmup_ratio": [0.05, 0.10, 0.15],
    "weight_decay": [0.0, 0.01, 0.05],
}

best_f1 = 0.0
target_f1 = 0.80

for config in grid_search(search_space):
    model = train(config)
    val_f1 = evaluate(model, val_set)
    if val_f1 > best_f1:
        best_f1 = val_f1
        save_checkpoint(model, config)
    if best_f1 >= target_f1:
        break

print(f"Best model: F1={best_f1:.4f} with config={best_config}")
```

### 4.5 Evaluation Metrics

| Metric | Task | Target |
|--------|------|--------|
| Entity-level F1 (seqeval) | NER | ≥ 0.80 |
| Macro F1 | Classification | ≥ 0.80 |
| Precision / Recall | Both | Reported |
| Skill coverage rate | Custom | ≥ 85% of test JDs have ≥1 skill extracted |

### 4.6 Inference Pipeline

```python
# skill_extractor.py
class SkillExtractor:
    def __init__(self, ner_model_path, cls_model_path):
        self.ner = load_model(ner_model_path)
        self.cls = load_model(cls_model_path)

    def extract(self, job_description: str) -> SkillReport:
        skills = self.ner.predict(job_description)       # List of skill spans
        categories = self.cls.predict(job_description)   # Skill category labels
        return SkillReport(skills=skills, categories=categories)
```

---

## 5. Security Architecture

### Principle: Zero Persistent User Data

| Data Type | Stored? | Where | Retention |
|-----------|---------|-------|-----------|
| User query (field, level) | No | Memory only | Cleared after run |
| Platform credentials | Never | Never stored | — |
| Session cookies | No | CloakBrowser in-memory only | Destroyed on close |
| Scraped job data | Yes | Output `.md` file only | User controls |
| ML model weights | Yes | `ml/checkpoints/` | Persistent (no PII) |
| Logs | Yes | `logs/agent.log` | Job metadata only, no user data |

### Threat Model & Mitigations

| Threat | Mitigation |
|--------|-----------|
| Accidental credential logging | `security.py` scans environment and log output for credential patterns |
| Session token persistence | CloakBrowser launched with `persist_session=False`; temp dirs wiped in `finally` |
| `.env` file exposure | `.gitignore` includes `.env`; only non-sensitive API keys allowed |
| Scraped PII in output | Output only contains publicly listed job data; no applicant data |

---

## 6. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | All modules |
| Browser automation | CloakBrowser | Stealth scraping |
| HTML parsing | BeautifulSoup4, lxml | Structured data extraction |
| HTTP | httpx (async) | API calls, fast scraping |
| NLP / ML | Transformers (HuggingFace) | PhoBERT, XLM-R |
| Fine-tuning | PEFT, bitsandbytes | QLoRA fine-tuning |
| Training | PyTorch, Accelerate | Model training |
| NER evaluation | seqeval | Entity-level F1 |
| CLI UI | Rich | Beautiful terminal output |
| Testing | pytest | Unit & integration tests |
| Linting | ruff, mypy | Code quality |

---

## 7. Milestones & Roadmap

### Phase 1 — Core Agent (Weeks 1–3) ✅
- [x] Project scaffolding, `BaseScraper` interface
- [x] VietnamWorks + ITViec + TopDev scrapers (no auth required)
- [x] CloakBrowser integration
- [x] CLI interaction loop + output writer
- [x] Security audit of session handling

### Phase 2 — Full Platform Coverage (Weeks 4–5) ✅
- [x] LinkedIn scraper using public guest API (no auth required — `jobs-guest/jobs/api/seeMoreJobPostings/search`)
- [x] Removed Playwright/BrowserSession dependency — LinkedIn now uses httpx, same as other scrapers
- [x] Rate limiting with random jitter, retry logic via `@retry` decorator
- [x] URL deduplication, pagination, all 4 platforms uniform (no auth) 🎉

### Phase 3 — ML Pipeline (Weeks 6–9) ✅
- [x] Dataset download from HuggingFace (tinixai/vietnamese-job-descriptions, 606k rows)
- [x] Preprocessing pipeline: Vietnamese NFC normalization, HTML cleaning, Jaccard dedup, short-description filtering
- [x] Weak supervision NER annotation (500+ skill seeds → BIO tags: O, B-SKILL, I-SKILL)
- [x] Multi-label taxonomy engineering (8 categories: programming_language, framework, database, cloud, soft_skill, language, experience_level, education)
- [x] Train/val/test split (80/10/10, stratified)
- [x] PhoBERT NER fine-tuning script (`ml/train/train_ner.py`) with early stopping, seqeval metrics
- [x] XLM-R multi-label classification script (`ml/train/train_classifier.py`) with BCEWithLogitsLoss, threshold tuning
- [x] Multi-task model with shared PhoBERT encoder (`ml/train/train_multitask.py`)
- [x] QLoRA/PEFT fine-tuning script (`ml/train/train_qlora.py`) with 4-bit quantization
- [x] Hyperparameter auto-search (`ml/train/hyperparam_search.py`)
- [x] Evaluation pipeline with per-label F1 + skill coverage rate (`ml/train/evaluate.py`)
- [x] Inference module with rule-based fallback (`ml/inference/skill_extractor.py`)

### Phase 4 — Advanced ML + Integration (Weeks 10–12) ✅
- [x] QLoRA fine-tuning script (`ml/train/train_qlora.py`) with BitsAndBytes 4-bit, LoRA adapters
- [x] Skill extractor inference module (`ml/inference/skill_extractor.py`) with NER + classification
- [x] Rewired output_writer.py to call SkillExtractor — falls back to `rule_based_extract()` when no trained model
- [x] Added skill analysis CLI prompt in main.py (ask_skill_analysis)
- [x] Full integration test (`tests/test_skill_extractor.py`) — 11 new tests, 34 total passing

### Phase 5 — Polish (Weeks 13–14) ✅
- [x] Concurrent scraping via `ThreadPoolExecutor` — all 4 platforms run in parallel
- [x] Comprehensive test suite: 67 tests across 5 test files (base_scraper, filters, scrapers, edge_cases, skill_extractor)
- [x] Updated CLAUDE.md developer documentation to reflect current architecture
- [x] All scrapers uniform (httpx, no auth) — LinkedIn simplified to guest API

---

## 8. Known Limitations & Risks

| Item | Risk | Mitigation |
|------|------|-----------|
| LinkedIn anti-bot detection | Medium | CloakBrowser fingerprint spoofing; fallback to manual auth |
| Platform UI changes breaking scrapers | High | Write scrapers against JSON APIs where available; add HTML fallback |
| Vietnamese NLP quality | Medium | PhoBERT is purpose-built for Vietnamese; dataset is Vietnamese-native |
| Dataset label noise | Medium | Weak supervision + manual review sample |
| GPU availability for QLoRA | Low | 4-bit quantization runs on 16GB VRAM; Google Colab T4 is sufficient |
| Rate limiting / IP blocking | Medium | Random jitter, respectful delays, CloakBrowser IP rotation |

---

## 9. Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/scraper-topdev`
3. Write tests for new scrapers in `tests/test_scrapers.py`
4. Ensure `pytest tests/ -v` passes
5. Submit a PR with a description of changes

**Code style:** Follow PEP 8, enforced by `ruff`. Type hints required on all public functions.

---

## 10. License

MIT License. See `LICENSE` file.

Dataset ([tinixai/vietnamese-job-descriptions](https://huggingface.co/datasets/tinixai/vietnamese-job-descriptions)) is used for research and non-commercial purposes only — check the dataset's license before commercial deployment.

---

*Last updated: 2026-05-24*
