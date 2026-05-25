from __future__ import annotations

import pytest

from agent.scrapers.base_scraper import (
    AuthRequiredError,
    Job,
    ParseError,
    RateLimitError,
    ScraperError,
)
from agent.scrapers.itviec_scraper import ITViecScraper
from agent.scrapers.linkedin_scraper import LinkedInScraper
from agent.scrapers.topdev_scraper import TopDevScraper
from agent.scrapers.vietnamworks_scraper import VietnamWorksScraper


class TestLinkedInScraper:
    def test_requires_no_auth(self):
        assert not LinkedInScraper().requires_auth()

    def test_build_url_with_level(self):
        url = LinkedInScraper()._build_url("Data Engineer", "Senior")
        assert "linkedin.com/jobs/search/" in url
        assert "keywords=" in url
        assert "f_E=4" in url

    def test_build_url_any_level(self):
        url = LinkedInScraper()._build_url("Dev", "Any")
        assert "f_E=" not in url

    def test_parse_empty_html(self):
        jobs = LinkedInScraper()._parse_job_listing("<html></html>", "Senior")
        assert jobs == []

    def test_parse_malformed_html(self):
        jobs = LinkedInScraper()._parse_job_listing("not valid html %#@!", "Senior")
        assert jobs == []


class TestVietnamWorksScraper:
    def test_requires_no_auth(self):
        assert not VietnamWorksScraper().requires_auth()

    def test_build_url(self):
        url = VietnamWorksScraper()._build_url("Data Engineer", "Senior")
        assert "ms.vietnamworks.com" in url

    def test_extract_salary_range(self):
        s = VietnamWorksScraper()
        assert s._extract_salary({"salaryFrom": 1000, "salaryTo": 2000}) == "1000–2000"

    def test_extract_salary_from_only(self):
        s = VietnamWorksScraper()
        assert s._extract_salary({"salaryFrom": 1500}) == "1500"

    def test_extract_salary_none(self):
        s = VietnamWorksScraper()
        assert s._extract_salary({}) == ""

    def test_parse_api_item(self):
        s = VietnamWorksScraper()
        job = s._parse_api_item(
            {
                "jobTitle": "Python Dev",
                "companyName": "TechCorp",
                "jobLocation": "Hanoi",
                "jobUrl": "https://vietnamworks.com/job/123",
                "postedDate": "1 day ago",
                "jobDescription": "<p>Develop Python apps</p>",
            },
            "Senior",
        )
        assert job.title == "Python Dev"
        assert job.company == "TechCorp"
        assert job.level == "Senior"
        assert "Python apps" in job.description_snippet


class TestITViecScraper:
    def test_build_url_with_level(self):
        url = ITViecScraper()._build_url("data engineer", "Senior")
        assert "itviec.com" in url
        assert "senior" in url

    def test_build_url_any_level(self):
        url = ITViecScraper()._build_url("dev", "Any")
        assert "job_levels" not in url

    def test_extract_cards_empty(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup("<html></html>", "lxml")
        cards = ITViecScraper()._extract_cards(soup)
        assert isinstance(cards, list)

    def test_parse_card_missing_title(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup('<div class="job"><p>No title</p></div>', "lxml")
        card = soup.select_one("div")
        result = ITViecScraper()._parse_card(card, "Junior")
        assert result is None


class TestTopDevScraper:
    def test_requires_no_auth(self):
        assert not TopDevScraper().requires_auth()

    def test_build_url(self):
        url = TopDevScraper()._build_url("Data Engineer", "Senior")
        assert "api.topdev.vn" in url

    def test_extract_location_list_dict(self):
        s = TopDevScraper()
        result = s._extract_location({"addresses": [{"city": "Hanoi"}, {"city": "HCMC"}]})
        assert result == "Hanoi"

    def test_extract_location_list_str(self):
        s = TopDevScraper()
        result = s._extract_location({"addresses": ["Da Nang"]})
        assert result == "Da Nang"

    def test_extract_location_fallback(self):
        s = TopDevScraper()
        result = s._extract_location({"city": "Can Tho"})
        assert result == "Can Tho"

    def test_extract_salary_string(self):
        s = TopDevScraper()
        assert s._extract_salary({"salary": "up to 2000 USD"}) == "up to 2000 USD"

    def test_extract_salary_empty(self):
        s = TopDevScraper()
        assert s._extract_salary({}) == ""

    def test_parse_api_item(self):
        s = TopDevScraper()
        job = s._parse_api_item(
            {
                "title": "Senior Go Dev",
                "employerName": "Fintech Co",
                "addresses": [{"city": "HCMC"}],
                "url": "https://topdev.vn/job/456",
                "postedAt": "2025-01-01",
                "description": "<p>Build microservices</p>",
            },
            "Senior",
        )
        assert job.title == "Senior Go Dev"
        assert job.company == "Fintech Co"
        assert job.location == "HCMC"
        assert "microservices" in job.description_snippet


class TestErrorHierarchy:
    def test_auth_required_is_scraper_error(self):
        assert issubclass(AuthRequiredError, ScraperError)

    def test_rate_limit_is_scraper_error(self):
        assert issubclass(RateLimitError, ScraperError)

    def test_parse_error_is_scraper_error(self):
        assert issubclass(ParseError, ScraperError)

    def test_all_errors_are_exception(self):
        for cls in [ScraperError, AuthRequiredError, RateLimitError, ParseError]:
            assert issubclass(cls, Exception)


class TestJobDataclass:
    def test_full_construction(self):
        job = Job(
            title="Tester",
            company="QA Inc",
            location="HCMC",
            level="Mid",
            url="http://test.com",
            posted_date="yesterday",
            description_snippet="test stuff",
            platform="Test",
            salary="2000–3000",
        )
        assert job.salary == "2000–3000"
        assert job.scraped_at

    def test_clean_text_complex(self):
        job = Job(
            title="T", company="C", location="L", level="S",
            url="http://x", posted_date="d", description_snippet="d",
            platform="x",
        )
        assert job.clean_text("<div><p>nested</p></div>") == "nested"
        assert job.clean_text("") == ""
        assert job.clean_text("   ") == ""


class TestFiltersEdgeCases:
    def test_field_mapping_case_insensitive(self):
        from agent.filters import FIELD_MAPPING

        assert FIELD_MAPPING["data engineer"] == "data-engineer"
        assert "data engineer" in FIELD_MAPPING

    def test_level_options_count(self):
        from agent.filters import LEVELS
        assert len(LEVELS) == 7

    def test_build_query_level_mappings(self):
        from agent.filters import build_query

        q = build_query("DevOps Engineer", "Lead")
        assert q.linkedin.experience_level == "5"
        assert q.vietnamworks.level_id == "7"
        assert q.itviec.job_levels == ["tech-lead"]
        assert q.topdev.level == "team-lead"

    def test_build_query_manager_level(self):
        from agent.filters import build_query

        q = build_query("Product Manager", "Manager")
        assert q.linkedin.experience_level == "6"
        assert q.vietnamworks.level_id == "8"
