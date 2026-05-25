from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from agent.scrapers.base_scraper import MAX_RETRIES, BaseScraper, Job, ParseError, retry


class ITViecScraper(BaseScraper):
    platform = "ITViec"

    BASE_URL = "https://itviec.com"

    def requires_auth(self) -> bool:
        return False

    def _build_url(self, field: str, level: str) -> str:
        from agent.filters import ITVIEC_LEVEL_MAP

        url = f"{self.BASE_URL}/it-jobs/{field.lower().replace(' ', '-')}"
        level_val = ITVIEC_LEVEL_MAP.get(level, "")
        if level_val:
            url += f"?job_levels%5B%5D={level_val}"
        return url

    @retry(max_retries=MAX_RETRIES)
    def search(self, field: str, level: str, count: int = 20) -> list[Job]:
        url = self._build_url(field, level)
        jobs: list[Job] = []

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }

        page = 1
        with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
            while len(jobs) < count:
                paged_url = url
                if "?" in url:
                    paged_url += f"&page={page}"
                else:
                    paged_url += f"?page={page}"

                resp = client.get(paged_url)
                if resp.status_code != 200:
                    break

                soup = BeautifulSoup(resp.text, "lxml")
                cards = self._extract_cards(soup)
                if not cards:
                    break

                for card in cards:
                    try:
                        job = self._parse_card(card, level)
                        if job and job.title:
                            jobs.append(job)
                    except Exception:
                        continue

                page += 1
                self._rate_limit(1.5)

        return jobs[:count]

    def _extract_cards(self, soup: BeautifulSoup) -> list:
        selectors = [
            '[class*="job"]',
            '[class*="job_item"]',
            '[class*="job-card"]',
            '.job',
            '[data-controller="job"]',
        ]
        for sel in selectors:
            cards = soup.select(sel)
            if cards:
                return cards

        return soup.select("body *[class]")

    def _parse_card(self, card, level: str) -> Job | None:
        title_el = card.select_one("h2, h3, [class*='title']")
        if not title_el:
            return None

        title = title_el.get_text(strip=True)
        if not title:
            return None

        link_el = card.select_one("a[href]") or title_el.select_one("a[href]")
        url_attr = link_el.get("href", "") if link_el else ""
        full_url = url_attr
        if full_url and not full_url.startswith("http"):
            full_url = self.BASE_URL + full_url

        company_el = card.select_one('[class*="company"], [class*="employer"]')
        location_el = card.select_one('[class*="location"], [class*="city"], [class*="address"]')
        date_el = card.select_one('[class*="date"], [class*="time"], time')
        salary_el = card.select_one('[class*="salary"]')
        desc_el = card.select_one('[class*="description"], [class*="snippet"], p')

        return Job(
            title=title,
            company=company_el.get_text(strip=True) if company_el else "Unknown",
            location=location_el.get_text(strip=True) if location_el else "Vietnam",
            level=level,
            url=full_url.strip() if full_url else "",
            posted_date=date_el.get_text(strip=True) if date_el else "",
            description_snippet=desc_el.get_text(strip=True)[:300] if desc_el else "",
            platform=self.platform,
            salary=salary_el.get_text(strip=True) if salary_el else "",
        )
