from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class BrowserSession:
    def __init__(self, headless: bool = False) -> None:
        self._headless = headless
        self._browser: Any = None
        self._context: Any = None

    def launch(self) -> None:
        try:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch(
                headless=self._headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
            )
            self._context = self._browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1366, "height": 768},
                locale="vi-VN",
            )
            self._context.add_init_script(
                """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                """
            )
        except ImportError:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

    @property
    def page(self) -> Any:
        if not self._context:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return self._context.new_page()

    def new_page(self) -> Any:
        if not self._context:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return self._context.new_page()

    def close(self) -> None:
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if getattr(self, "_pw", None):
                self._pw.stop()
        except Exception:
            pass

    @property
    def browser(self) -> Any:
        return self._browser


@contextmanager
def ephemeral_session(headless: bool = False):
    session = BrowserSession(headless=headless)
    try:
        session.launch()
        yield session
    finally:
        session.close()
