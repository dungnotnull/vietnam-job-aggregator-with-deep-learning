from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field

FIELD_MAPPING: dict[str, str] = {
    "software engineer": "software-engineer",
    "data engineer": "data-engineer",
    "data scientist": "data-scientist",
    "frontend developer": "frontend-developer",
    "backend developer": "backend-developer",
    "fullstack developer": "fullstack-developer",
    "devops engineer": "devops-engineer",
    "product manager": "product-manager",
    "project manager": "project-manager",
    "ux designer": "ux-designer",
    "qa engineer": "qa-engineer",
    "mobile developer": "mobile-developer",
    "ai engineer": "ai-engineer",
    "ml engineer": "ml-engineer",
    "security engineer": "security-engineer",
    "business analyst": "business-analyst",
    "data analyst": "data-analyst",
    "system admin": "system-admin",
    "network engineer": "network-engineer",
    "cloud engineer": "cloud-engineer",
}

LEVELS: tuple[str, ...] = ("Intern", "Junior", "Mid-level", "Senior", "Lead", "Manager", "Any")


@dataclass(slots=True)
class LinkedInQuery:
    keywords: str
    location: str = "Vietnam"
    experience_level: str = ""


@dataclass(slots=True)
class VNWQuery:
    keyword: str
    level_id: str = ""


@dataclass(slots=True)
class ITViecQuery:
    query: str
    job_levels: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TopDevQuery:
    q: str
    level: str = ""


@dataclass(slots=True)
class PlatformQueries:
    linkedin: LinkedInQuery
    vietnamworks: VNWQuery
    itviec: ITViecQuery
    topdev: TopDevQuery


VN_LEVEL_MAP: dict[str, str] = {
    "Intern": "5",
    "Junior": "2",
    "Mid-level": "3",
    "Senior": "4",
    "Lead": "7",
    "Manager": "8",
    "Any": "",
}

ITVIEC_LEVEL_MAP: dict[str, str] = {
    "Intern": "intern",
    "Junior": "junior",
    "Mid-level": "mid-senior",
    "Senior": "senior",
    "Lead": "tech-lead",
    "Manager": "manager",
    "Any": "",
}

TOPDEV_LEVEL_MAP: dict[str, str] = {
    "Intern": "intern",
    "Junior": "junior",
    "Mid-level": "mid",
    "Senior": "senior",
    "Lead": "team-lead",
    "Manager": "manager",
    "Any": "",
}

LINKEDIN_LEVEL_MAP: dict[str, str] = {
    "Intern": "1",
    "Junior": "2",
    "Mid-level": "3",
    "Senior": "4",
    "Lead": "5",
    "Manager": "6",
    "Any": "",
}


def _slugify(text: str) -> str:
    return text.lower().replace(" ", "-")


def build_query(field: str, level: str) -> PlatformQueries:
    search_field = FIELD_MAPPING.get(field.lower(), _slugify(field))
    return PlatformQueries(
        linkedin=LinkedInQuery(
            keywords=urllib.parse.quote(field),
            location="Vietnam",
            experience_level=LINKEDIN_LEVEL_MAP.get(level, ""),
        ),
        vietnamworks=VNWQuery(
            keyword=urllib.parse.quote(field),
            level_id=VN_LEVEL_MAP.get(level, ""),
        ),
        itviec=ITViecQuery(
            query=urllib.parse.quote(field),
            job_levels=[ITVIEC_LEVEL_MAP[level]] if ITVIEC_LEVEL_MAP.get(level) else [],
        ),
        topdev=TopDevQuery(
            q=urllib.parse.quote(field),
            level=TOPDEV_LEVEL_MAP.get(level, ""),
        ),
    )

