from __future__ import annotations

from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup

from agent.filters import LINKEDIN_LEVEL_MAP
from agent.scrapers.base_scraper import MAX_RETRIES, BaseScraper, Job, ParseError, retry


class LinkedInScraper(BaseScraper):
    platform = "LinkedIn"

    JOBS_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
    PUBLIC_URL = "https://www.linkedin.com/jobs/search/"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }

    def requires_auth(self) -> bool:
        return False

    def _build_url(self, field: str, level: str) -> str:
        from agent.filters import build_query

        q = build_query(field, level)
        params = f"keywords={q.linkedin.keywords}&location=Vietnam"
        if q.linkedin.experience_level:
            params += f"&f_E={q.linkedin.experience_level}"
        return f"{self.PUBLIC_URL}?{params}"

    @retry(max_retries=MAX_RETRIES)
    def search(self, field: str, level: str, count: int = 20) -> list[Job]:
        params: dict[str, str] = {
            "keywords": field,
            "location": "Vietnam",
            "f_TPR": "r2592000",
            "position": "1",
            "pageNum": "0",
        }
        level_code = LINKEDIN_LEVEL_MAP.get(level, "")
        if level_code:
            params["f_E"] = level_code

        jobs: list[Job] = []
        seen_urls: set[str] = set()
        page_size = 25

        with httpx.Client(timeout=30.0, headers=self.HEADERS, follow_redirects=True) as client:
            for start in range(0, count, page_size):
                params["position"] = str(start)
                params["pageNum"] = str(start // page_size)

                query_string = urlencode(params)
                api_url = f"{self.JOBS_URL}?{query_string}"

                resp = client.get(api_url)
                if resp.status_code != 200:
                    break

                self._rate_limit(2.0)

                page_jobs = self._parse_job_listing(resp.text, level)
                new_jobs = [j for j in page_jobs if j.url not in seen_urls]
                if not new_jobs:
                    break

                for j in new_jobs:
                    seen_urls.add(j.url)
                jobs.extend(new_jobs)
                if len(new_jobs) < page_size:
                    break

        if not jobs:
            raise ParseError("LinkedIn returned no results")
        return jobs[:count]

    def _parse_job_listing(self, html: str, level: str) -> list[Job]:
        soup = BeautifulSoup(html, "lxml")
        jobs: list[Job] = []

        cards = soup.select(
            "li:has(.base-search-card__title), "
            "li:has(.job-search-card__title), "
            "li:has(a[href*='/jobs/'])"
        )

        for card in cards:
            try:
                title_el = card.select_one(
                    "h3.base-search-card__title, h3.job-search-card__title, "
                    "a.job-title, span.screen-reader-text, h3"
                )
                company_el = card.select_one(
                    "h4.base-search-card__subtitle a, "
                    "a.hidden-nested-link, "
                    "span.job-search-card__company-name, h4 a"
                )
                location_el = card.select_one(
                    "span.job-search-card__location, "
                    "span.base-search-card__metadata-item, "
                    "span.location"
                )
                link_el = card.select_one(
                    "a.base-card__full-link, a.job-search-card__link, a[href*='/jobs/']"
                )
                date_el = card.select_one("time, span.job-search-card__listdate")
                salary_el = card.select_one(
                    "span.job-search-card__salary-info, span.salary"
                )

                title = title_el.get_text(strip=True) if title_el else ""
                if not title:
                    continue
                if "promoted" in title.lower() or "ad" in title.lower():
                    continue

                company = company_el.get_text(strip=True) if company_el else "Unknown"
                location = (
                    location_el.get_text(strip=True) if location_el else "Vietnam"
                )

                href = link_el.get("href", "") if link_el else ""
                parsed_url = href
                if parsed_url and not parsed_url.startswith("http"):
                    parsed_url = "https://www.linkedin.com" + parsed_url
                if "?" in parsed_url:
                    parsed_url = parsed_url.split("?")[0]

                jobs.append(
                    Job(
                        title=title,
                        company=company,
                        location=location,
                        level=level,
                        url=parsed_url.strip(),
                        posted_date=date_el.get_text(strip=True) if date_el else "",
                        description_snippet="",
                        platform=self.platform,
                        salary=salary_el.get_text(strip=True) if salary_el else "",
                    )
                )
            except Exception:
                continue

        return jobs
