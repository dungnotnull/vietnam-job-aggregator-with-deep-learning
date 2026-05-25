from __future__ import annotations

import pytest

from agent.filters import LEVELS, build_query, FIELD_MAPPING


def test_field_mapping_known():
    assert FIELD_MAPPING["data engineer"] == "data-engineer"


def test_levels_all_present():
    assert len(LEVELS) == 7
    assert "Intern" in LEVELS
    assert "Senior" in LEVELS


def test_build_query_basic():
    q = build_query("Data Engineer", "Senior")
    assert q.linkedin.keywords == "Data%20Engineer"
    assert q.linkedin.experience_level == "4"
    assert q.vietnamworks.keyword == "Data%20Engineer"
    assert q.vietnamworks.level_id == "4"
    assert q.itviec.query == "Data%20Engineer"
    assert q.itviec.job_levels == ["senior"]
    assert q.topdev.q == "Data%20Engineer"
    assert q.topdev.level == "senior"


def test_build_query_any_level():
    q = build_query("Frontend Developer", "Any")
    assert q.linkedin.experience_level == ""
    assert q.vietnamworks.level_id == ""
    assert q.itviec.job_levels == []
    assert q.topdev.level == ""


def test_build_query_unknown_field():
    q = build_query("Some Random Role", "Mid-level")
    assert q.linkedin.keywords == "Some%20Random%20Role"
