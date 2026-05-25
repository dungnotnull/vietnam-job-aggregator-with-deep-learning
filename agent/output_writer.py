from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent.scrapers.base_scraper import Job

if TYPE_CHECKING:
    from ml.inference.skill_extractor import SkillReport

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_filename(field: str, level: str) -> str:
    safe_field = field.lower().replace(" ", "_").replace("/", "_")
    safe_level = level.lower().replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"jobs_{safe_field}_{safe_level}_{ts}.md"


def _run_skill_analysis(
    field: str, level: str, all_jobs: dict[str, list[Job]]
) -> SkillReport | None:
    try:
        from ml.inference.skill_extractor import (
            SkillExtractor,
            SkillReport,
            rule_based_extract,
        )

        extractor = SkillExtractor()
        combined_text_parts: list[str] = [field, level]
        for jobs in all_jobs.values():
            for job in jobs:
                parts = [job.title, job.description_snippet, job.company]
                combined_text_parts.extend(p for p in parts if p)
        combined_text = " ".join(combined_text_parts)

        if extractor.ner_model is not None and extractor.cls_model is not None:
            job_dicts = [
                {"title": j.title, "description": j.description_snippet}
                for jobs in all_jobs.values()
                for j in jobs
            ]
            return extractor.analyze(job_dicts)
        else:
            return rule_based_extract(combined_text)
    except Exception as e:
        return None


def write_report(
    field: str,
    level: str,
    all_jobs: dict[str, list[Job]],
    filename: str | None = None,
    include_skill_analysis: bool = True,
) -> Path:
    if filename is None:
        filename = generate_filename(field, level)

    filepath = OUTPUT_DIR / filename
    total = sum(len(jobs) for jobs in all_jobs.values())

    lines: list[str] = []
    lines.append("# Vietnam Job Search Results")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| **Search Query** | {field} |")
    lines.append(f"| **Level** | {level} |")
    lines.append(f"| **Date** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |")
    lines.append(f"| **Total Jobs Found** | {total} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for platform, jobs in all_jobs.items():
        lines.append(f"## {platform} — {len(jobs)} results")
        lines.append("")
        if not jobs:
            lines.append("*No results found*")
            lines.append("")
            continue

        lines.append(
            "| # | Job Title | Company | Location | Posted | Salary | Link |"
        )
        lines.append(
            "|---|-----------|---------|----------|--------|--------|------|"
        )
        for i, job in enumerate(jobs, 1):
            salary = job.salary if job.salary else "—"
            lines.append(
                f"| {i} | {job.title} | {job.company} | {job.location} "
                f"| {job.posted_date} | {salary} | [View →]({job.url}) |"
            )
        lines.append("")

    if include_skill_analysis:
        lines.append("---")
        lines.append("")
        report = _run_skill_analysis(field, level, all_jobs)
        if report is not None:
            lines.append(report.to_markdown())
        else:
            lines.append("## Skill Intelligence Report")
            lines.append("*Model not yet trained — skill analysis unavailable.*")
        lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath
