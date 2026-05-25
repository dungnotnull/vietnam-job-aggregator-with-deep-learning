from __future__ import annotations

import time

import pytest

from agent.scrapers.base_scraper import (
    MAX_RETRIES,
    Job,
    ParseError,
    ScraperError,
    retry,
)


class TestRetry:
    def test_retry_succeeds_on_first_attempt(self):
        call_count = 0

        @retry(max_retries=2)
        def flaky():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 1

    def test_retry_eventually_succeeds(self):
        call_count = 0

        @retry(max_retries=3, base_delay=0.01)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 3

    def test_retry_raises_after_max_retries(self):
        call_count = 0

        @retry(max_retries=2, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        with pytest.raises(ScraperError, match="failed after 2 retries"):
            always_fails()
        assert call_count == 3

    def test_retry_does_not_retry_scraper_errors(self):
        call_count = 0

        @retry(max_retries=3, base_delay=0.01)
        def raises_scraper_error():
            nonlocal call_count
            call_count += 1
            raise ScraperError("stop")

        with pytest.raises(ScraperError):
            raises_scraper_error()
        assert call_count == 1

    def test_retry_preserves_func_metadata(self):
        @retry(max_retries=2)
        def my_func(x: int) -> str:
            """Docstring"""
            return str(x)

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "Docstring"


def test_job_clean_text():
    job = Job(
        title="Test", company="T", location="L", level="S",
        url="http://x", posted_date="today", description_snippet="d",
        platform="x",
    )
    assert job.clean_text("<p>hello</p>") == "hello"
    assert job.clean_text("  a   b  ") == "a b"
    assert job.clean_text("<br/>text\n\nmore") == "text more"


def test_job_defaults():
    job = Job(
        title="Engineer", company="Corp", location="Hanoi",
        level="Mid", url="http://x", posted_date="1d ago",
        description_snippet="desc", platform="test",
    )
    assert job.salary == ""
    assert job.scraped_at


def test_scraper_error_inheritance():
    assert issubclass(ParseError, ScraperError)
    assert issubclass(ScraperError, Exception)
