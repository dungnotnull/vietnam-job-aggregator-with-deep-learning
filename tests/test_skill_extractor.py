from __future__ import annotations

import pytest

from agent.output_writer import generate_filename, write_report
from agent.scrapers.base_scraper import Job
from ml.inference.skill_extractor import (
    SkillExtractor,
    SkillReport,
    rule_based_extract,
)


class TestSkillExtractor:
    def test_rule_based_extract_finds_skills(self):
        text = """
        We are looking for a Senior Python developer with experience in Django,
        PostgreSQL, AWS, Docker, and Kubernetes. Must have strong communication
        skills and English proficiency.
        """
        report = rule_based_extract(text)
        assert len(report.skills) > 0
        assert "Python" in report.skills
        assert any(c in report.categories for c in ("programming_language", "framework", "database", "cloud", "soft_skill"))
        assert "language" in report.categories or any(
            c == "language" for c in report.categories
        )

    def test_rule_based_extract_empty_text(self):
        report = rule_based_extract("")
        assert len(report.skills) == 0
        assert len(report.categories) == 0

    def test_rule_based_extract_no_matches(self):
        report = rule_based_extract("Just a regular job posting with no tech keywords")
        assert len(report.skills) == 0

    def test_skill_report_top_skills(self):
        report = SkillReport(
            skills=["Python", "Django", "AWS"],
            skill_frequencies={"Python": 5, "Django": 3, "AWS": 2},
            categories=["programming_language", "framework", "cloud"],
        )
        top = report.top_skills(2)
        assert top == [("Python", 5), ("Django", 3)]

    def test_skill_report_to_markdown(self):
        report = SkillReport(
            skills=["Python", "AWS"],
            skill_frequencies={"Python": 5, "AWS": 2},
            categories=["programming_language", "cloud"],
        )
        md = report.to_markdown()
        assert "##" in md
        assert "Python" in md
        assert "Frequency" in md

    def test_skill_extractor_initialization(self):
        extractor = SkillExtractor(
            ner_model_path="/nonexistent/path",
            cls_model_path="/nonexistent/path",
        )
        assert extractor.ner_model is None
        assert extractor.cls_model is None

    def test_skill_extractor_analyze_fallback(self):
        """When models are not loaded, analyze should return empty report."""
        extractor = SkillExtractor(
            ner_model_path="/nonexistent/path",
            cls_model_path="/nonexistent/path",
        )
        jobs = [{"title": "Python Dev", "description": "Python Django", "requirements": "AWS Docker"}]
        report = extractor.analyze(jobs)
        assert isinstance(report, SkillReport)


class TestOutputWriter:
    def test_generate_filename(self):
        name = generate_filename("Data Engineer", "Senior")
        assert "data_engineer" in name
        assert "senior" in name
        assert name.endswith(".md")

    def test_write_report_creates_file(self, tmp_path):
        jobs = {
            "TestPlatform": [
                Job(
                    title="Test Job",
                    company="TestCorp",
                    location="TestCity",
                    level="Senior",
                    url="http://test.com",
                    posted_date="today",
                    description_snippet="Python Django",
                    platform="TestPlatform",
                )
            ]
        }
        filename = "test_output.md"
        filepath = write_report("Tester", "Senior", jobs, filename=filename, include_skill_analysis=True)
        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "Test Job" in content
        assert "TestCorp" in content

    def test_write_report_no_jobs(self, tmp_path):
        jobs: dict[str, list[Job]] = {"EmptyPlatform": []}
        filename = "empty_output.md"
        filepath = write_report("Nothing", "Any", jobs, filename=filename, include_skill_analysis=False)
        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "No results found" in content

    def test_write_report_skill_analysis(self, tmp_path):
        jobs = {
            "TestPlatform": [
                Job(
                    title="Python Developer",
                    company="TechCorp",
                    location="HCMC",
                    level="Senior",
                    url="http://test.com",
                    posted_date="1 day ago",
                    description_snippet="Python Django PostgreSQL AWS Docker",
                    platform="TestPlatform",
                )
            ]
        }
        filename = "skill_test_output.md"
        filepath = write_report("Python Developer", "Senior", jobs, filename=filename, include_skill_analysis=True)
        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "Skill Intelligence Report" in content
        assert "Python" in content or "Django" in content or "AWS" in content
