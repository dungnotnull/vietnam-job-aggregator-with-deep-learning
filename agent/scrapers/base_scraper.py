from __future__ import annotations

import functools
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

MAX_RETRIES = 3
BASE_DELAY = 2.0
MAX_DELAY = 30.0


class ScraperError(Exception):
    pass


class AuthRequiredError(ScraperError):
    pass


class RateLimitError(ScraperError):
    pass


class ParseError(ScraperError):
    pass


@dataclass(slots=True)
class Job:
    title: str
    company: str
    location: str
    level: str
    url: str
    posted_date: str
    description_snippet: str
    platform: str
    salary: str = ""
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_api(cls, platform: str, raw: dict[str, Any], level: str = "Unknown") -> Job:
        raise NotImplementedError

    def clean_text(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"\s+", " ", text).strip()


def retry(
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    ),
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
                        time.sleep(delay)
                    else:
                        raise ScraperError(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        ) from e
                except ScraperError:
                    raise
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2**attempt) + random.uniform(0, 1), max_delay)
                        time.sleep(delay)
                    else:
                        raise ScraperError(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        ) from e
            raise RuntimeError("unreachable")

        return wrapper  # type: ignore[return-value]

    return decorator


class BaseScraper(ABC):
    platform: str = ""

    @abstractmethod
    def requires_auth(self) -> bool:
        ...

    @abstractmethod
    def search(self, field: str, level: str, count: int = 20) -> list[Job]:
        ...

    @abstractmethod
    def _build_url(self, field: str, level: str) -> str:
        ...

    def _rate_limit(self, seconds: float = 2.0) -> None:
        jitter = random.uniform(0, 1.0)
        time.sleep(seconds + jitter)

    @staticmethod
    def _clean_html(text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"\s+", " ", text).strip()
