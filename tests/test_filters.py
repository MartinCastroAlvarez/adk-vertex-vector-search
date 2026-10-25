"""Filters: rules that read them off a question, and validation of the model's arguments."""

from __future__ import annotations

from datetime import date

import pytest

from agentic_rag.filters import FilterError
from agentic_rag.filters import SearchFilters
from agentic_rag.filters import extract_filters
from agentic_rag.filters import mentioned_versions
from agentic_rag.filters import validate_filters
from agentic_rag.retrieval.types import NumericRestrict
from agentic_rag.retrieval.types import Restrict

CATALOG = {"ferry": ["1.0", "1.5", "2.0"], "lumen": ["1.4", "2.0", "3.0"]}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the default port in Lumen 2.0?", {"product": "lumen", "version": "2.0"}),
        ("How do canaries work in ferry v1.5?", {"product": "ferry", "version": "1.5"}),
        ("Does Lumen 3 support mutual TLS?", {"product": "lumen", "version": "3"}),
        ("Ferry 2.x rollout steps", {"product": "ferry", "version": "2"}),
        (
            "What changed in Lumen since 2.0?",
            {"product": "lumen", "since_version": "2.0", "doc_type": "release-notes"},
        ),
        ("Ferry 1.5 and later secrets", {"product": "ferry", "since_version": "1.5"}),
        ("Lumen docs released after 2025", {"product": "lumen", "released_after": "2025-01-01"}),
        ("Ferry changes before 2026", {"product": "ferry", "released_before": "2026-01-01"}),
        ("Lumen backups, not 1.4", {"product": "lumen", "exclude_version": "1.4"}),
        ("Lumen troubleshooting for LUM-311", {"product": "lumen", "doc_type": "troubleshooting"}),
        ("How do I upgrade Lumen from 1.4 to 2.0?", {"product": "lumen", "version": "2.0"}),
        ("What is the capital of France?", {}),
        ("What is in version 9.9 of Lumen?", {"product": "lumen"}),
        ("Lumen error LUM-104", {"product": "lumen"}),
    ],
)
def test_extract_filters(question: str, expected: dict[str, str]) -> None:
    assert extract_filters(question, CATALOG).as_dict() == expected


def test_mentioned_versions() -> None:
    assert mentioned_versions("Port in Lumen 1.4 and in Lumen 3.0?", CATALOG) == ["1.4", "3.0"]
    assert mentioned_versions("Upgrade Lumen from 1.4 to 2.0", CATALOG) == ["2.0"]
    assert mentioned_versions("Upgrade Lumen from 1.4 to 9.0", CATALOG) == []
    assert mentioned_versions("nothing", CATALOG) == []


def test_restricts_and_numeric_restricts() -> None:
    f = SearchFilters(
        product="lumen",
        version="2",
        since_version="2",
        exclude_version="2.0",
        doc_type="guide",
        released_after=date(2025, 1, 1),
        released_before=date(2026, 6, 30),
    )
    assert f.restricts() == (
        Restrict("product", ("lumen",)),
        Restrict("major", ("2",)),
        Restrict("version", deny=("2.0",)),
        Restrict("doc_type", ("guide",)),
    )
    assert f.numeric_restricts() == (
        NumericRestrict("version_num", value_int=200, op="GREATER_EQUAL"),
        NumericRestrict("released", value_int=20250101, op="GREATER_EQUAL"),
        NumericRestrict("released", value_int=20260630, op="LESS"),
    )
    assert SearchFilters().is_empty() and SearchFilters().describe() == "none"


def test_validate_normalises_what_the_model_sends() -> None:
    f = validate_filters(
        CATALOG, product=" Lumen ", version="V2.0", doc_type="Release Notes", released_after="2025"
    )
    assert f == SearchFilters(
        product="lumen", version="2.0", doc_type="release-notes", released_after=date(2025, 1, 1)
    )
    assert validate_filters(CATALOG, version="2.x").version == "2"
    assert validate_filters(CATALOG, product="lumen", since_version="2.5").since_version == "2.5"
    assert validate_filters(CATALOG, product="", version=None).is_empty()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"product": "postgres"}, "unknown product"),
        ({"product": "ferry", "version": "3.0"}, "unknown version"),
        ({"product": "ferry", "version": "7"}, "no 7.x"),
        ({"version": "latest"}, "look like"),
        ({"doc_type": "blog"}, "unknown doc_type"),
        ({"released_after": "last tuesday"}, "YYYY-MM-DD"),
        ({"product": 'lumen" OR 1=1'}, "unknown product"),
    ],
)
def test_validate_refuses_what_the_corpus_cannot_satisfy(
    kwargs: dict[str, str], error: str
) -> None:
    with pytest.raises(FilterError, match=error):
        validate_filters(CATALOG, **kwargs)
