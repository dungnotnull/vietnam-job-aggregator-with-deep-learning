from __future__ import annotations

from typing import Any

import httpx
from bs4 import BeautifulSoup

from agent.scrapers.base_scraper import MAX_RETRIES, BaseScraper, Job, ScraperError, retry


class TopDevScraper(BaseScraper):
    platform = "TopDev"

    API_URL = "https://api.topdev.vn/td/v2/jobs"
    SITE_URL = "https://topdev.vn"

    def requires_auth(self) -> bool:
        return False

    def _build_url(self, field: str, level: str) -> str:
        from agent.filters import build_query

        q = build_query(field, level)
        params = {"keyword": q.topdev.q}
        if q.topdev.level:
            params["levels"] = q.topdev.level
        import urllib.parse

        query_string = urllib.parse.urlencode(params)
        return f"{self.API_URL}?{query_string}"

    @retry(max_retries=MAX_RETRIES)
    def search(self, field: str, level: str, count: int = 20) -> list[Job]:
        jobs: list[Job] = []
        api_error: Exception | None = None

        try:
            api_jobs = self._search_api(field, level, count)
            jobs.extend(api_jobs)
        except Exception as e:
            api_error = e

        if len(jobs) < count:
            try:
                html_jobs = self._search_html(field, level, count - len(jobs))
                seen_urls = {j.url for j in jobs}
                for j in html_jobs:
                    if j.url not in seen_urls:
                        jobs.append(j)
            except Exception:
                pass

        if not jobs and api_error:
            raise ScraperError(f"TopDev returned no results (API error: {api_error})")

        return jobs[:count]

    def _search_api(self, field: str, level: str, count: int) -> list[Job]:
        from agent.filters import TOPDEV_LEVEL_MAP

        params: dict[str, Any] = {
            "keyword": field,
            "page": 1,
            "size": min(count, 50),
        }
        level_val = TOPDEV_LEVEL_MAP.get(level, "")
        if level_val:
            params["levels"] = level_val

        headers = {
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
        }

        jobs: list[Job] = []
        with httpx.Client(timeout=30.0, headers=headers) as client:
            resp = client.get(self.API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("data", []) if isinstance(data, dict) else data
            if isinstance(items, dict):
                items = items.get("data", items.get("jobs", []))

            for item in items[:count]:
                jobs.append(self._parse_api_item(item, level))

        return jobs

    def _search_html(self, field: str, level: str, count: int) -> list[Job]:
        url = f"{self.SITE_URL}/jobs"
        params = {"keyword": field}
        if level != "Any":
            params["levels"] = level

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
        }

        jobs: list[Job] = []
        with httpx.Client(timeout=30.0, headers=headers) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            cards = soup.select('[class*="job-item"], [class*="job-card"], [class*="job-listing"]')
            for card in cards[:count]:
                title_el = card.select_one('[class*="title"], h3, h2, a')
                company_el = card.select_one('[class*="company"]')
                location_el = card.select_one('[class*="location"]')
                link_el = card.select_one("a[href]")
                date_el = card.select_one('[class*="date"], time')
                salary_el = card.select_one('[class*="salary"]')

                href = link_el.get("href", "") if link_el else ""
                full_url = href
                if full_url and not full_url.startswith("http"):
                    full_url = self.SITE_URL + full_url

                jobs.append(
                    Job(
                        title=title_el.get_text(strip=True) if title_el else "",
                        company=company_el.get_text(strip=True) if company_el else "",
                        location=location_el.get_text(strip=True) if location_el else "Vietnam",
                        level=level,
                        url=full_url,
                        posted_date=date_el.get_text(strip=True) if date_el else "",
                        description_snippet="",
                        platform=self.platform,
                        salary=salary_el.get_text(strip=True) if salary_el else "",
                    )
                )

        return jobs

    def _parse_api_item(self, item: dict[str, Any], level: str) -> Job:
        return Job(
            title=item.get("title", item.get("jobTitle", "")),
            company=item.get(
                "employerName",
                item.get("companyName", item.get("organizationName", "")),
            ),
            location=self._extract_location(item),
            level=level,
            url=item.get("url", item.get("jobUrl", item.get("slug", ""))),
            posted_date=item.get("postedAt", item.get("postedDate", item.get("createdAt", ""))),
            description_snippet=self._clean_html(
                item.get("description", item.get("jobDescription", ""))
            )[:300],
            platform=self.platform,
            salary=self._extract_salary(item),
        )

    def _extract_location(self, item: dict[str, Any]) -> str:
        locations = item.get("addresses", item.get("locations", []))
        if isinstance(locations, list) and locations:
            first = locations[0]
            if isinstance(first, dict):
                return first.get("city", first.get("name", "Vietnam"))
            return str(first)
        return item.get("location", item.get("city", "Vietnam"))

    def _extract_salary(self, item: dict[str, Any]) -> str:
        salary_text = item.get("salary", "")
        if salary_text:
            return str(salary_text) if not isinstance(salary_text, dict) else str(salary_text)
        salary_from = item.get("salaryFrom", item.get("salaryMin"))
        salary_to = item.get("salaryTo", item.get("salaryMax"))
        if salary_from and salary_to:
            return f"{salary_from}–{salary_to}"
        if salary_from:
            return str(salary_from)
        return ""
