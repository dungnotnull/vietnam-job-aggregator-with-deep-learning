from __future__ import annotations

import pytest

from agent.scrapers.base_scraper import Job
from agent.scrapers.vietnamworks_scraper import VietnamWorksScraper
from agent.scrapers.itviec_scraper import ITViecScraper
from agent.scrapers.topdev_scraper import TopDevScraper
from agent.scrapers.linkedin_scraper import LinkedInScraper


def test_vietnamworks_requires_no_auth():
    s = VietnamWorksScraper()
    assert not s.requires_auth()


def test_itviec_requires_no_auth():
    s = ITViecScraper()
    assert not s.requires_auth()


def test_topdev_requires_no_auth():
    s = TopDevScraper()
    assert not s.requires_auth()


def test_linkedin_requires_no_auth():
    s = LinkedInScraper()
    assert not s.requires_auth()


def test_vietnamworks_build_url():
    s = VietnamWorksScraper()
    url = s._build_url("Data Engineer", "Senior")
    assert "vietnamworks" in url.lower()


def test_itviec_build_url():
    s = ITViecScraper()
    url = s._build_url("Data Engineer", "Senior")
    assert "itviec" in url.lower()


def test_topdev_build_url():
    s = TopDevScraper()
    url = s._build_url("Data Engineer", "Senior")
    assert "topdev" in url.lower()


def test_job_dataclass():
    job = Job(
        title="Senior Engineer",
        company="Test Corp",
        location="Hanoi",
        level="Senior",
        url="https://example.com",
        posted_date="1 day ago",
        description_snippet="A great job",
        platform="TestPlatform",
    )
    assert job.title == "Senior Engineer"
    assert job.platform == "TestPlatform"


def test_job_clean_text():
    job = Job(
        title="Test",
        company="Test",
        location="Test",
        level="Test",
        url="http://test",
        posted_date="today",
        description_snippet="test",
        platform="test",
    )
    assert job.clean_text("<p>  hello  </p>") == "hello"


def test_linkedin_build_url():
    s = LinkedInScraper()
    url = s._build_url("Back End Developer", "Senior")
    assert "linkedin" in url.lower()
    assert "keywords" in url.lower()
