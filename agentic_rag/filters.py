"""Metadata filters: what a question constrains, as Vector Search restricts.

Every chunk is upserted with

* token restricts ``product`` (``lumen``), ``version`` (``2.0``), ``major`` (``2``) and
  ``doc_type`` (``release-notes``), and
* numeric restricts ``version_num`` (``200``) and ``released`` (``20250120``),

so "the default port in Lumen 2.0" becomes ``product=lumen AND version=2.0`` and "what
changed in Ferry since 2025" becomes ``product=ferry AND released >= 20250101``.

:class:`SearchFilters` is what the ``search_docs`` tool accepts from the model - checked
against the catalog, because tool arguments are untrusted. :func:`extract_filters` reads
the same filters out of a question with rules; the agent gets them as a hint and the
offline stub uses them directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import fields
from datetime import date
from typing import Any

from agentic_rag.retrieval.corpus import DOC_TYPES
from agentic_rag.retrieval.corpus import date_number
from agentic_rag.retrieval.corpus import version_number
from agentic_rag.retrieval.types import NumericRestrict
from agentic_rag.retrieval.types import Restrict

Catalog = dict[str, list[str]]  # product -> versions


class FilterError(ValueError):
    """A filter the corpus can't satisfy (unknown product, version, doc type, bad date)."""


@dataclass(frozen=True)
class SearchFilters:
    product: str | None = None
    version: str | None = None  # "2.0" (exact) or "2" (any 2.x)
    since_version: str | None = None  # version_num >= ...
    exclude_version: str | None = None  # deny "1.4" or "1"
    doc_type: str | None = None
    released_after: date | None = None  # released >= ...
    released_before: date | None = None  # released < ...

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))

    def restricts(self) -> tuple[Restrict, ...]:
        out: list[Restrict] = []
        if self.product:
            out.append(Restrict("product", (self.product,)))
        if self.version:
            out.append(Restrict(_version_namespace(self.version), (self.version,)))
        if self.exclude_version:
            out.append(
                Restrict(_version_namespace(self.exclude_version), deny=(self.exclude_version,))
            )
        if self.doc_type:
            out.append(Restrict("doc_type", (self.doc_type,)))
        return tuple(out)

    def numeric_restricts(self) -> tuple[NumericRestrict, ...]:
        out: list[NumericRestrict] = []
        if self.since_version:
            value = _as_version_number(self.since_version)
            out.append(NumericRestrict("version_num", value_int=value, op="GREATER_EQUAL"))
        if self.released_after:
            out.append(
                NumericRestrict(
                    "released", value_int=date_number(self.released_after), op="GREATER_EQUAL"
                )
            )
        if self.released_before:
            out.append(
                NumericRestrict("released", value_int=date_number(self.released_before), op="LESS")
            )
        return tuple(out)

    def as_dict(self) -> dict[str, str]:
        return {
            f.name: str(getattr(self, f.name))
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    def describe(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.as_dict().items()) or "none"


def _version_namespace(version: str) -> str:
    return "version" if "." in version else "major"


def _as_version_number(version: str) -> int:
    return version_number(version if "." in version else f"{version}.0")


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def validate_filters(
    catalog: Catalog,
    doc_types: list[str] | tuple[str, ...] = DOC_TYPES,
    *,
    product: Any = None,
    version: Any = None,
    since_version: Any = None,
    exclude_version: Any = None,
    doc_type: Any = None,
    released_after: Any = None,
    released_before: Any = None,
) -> SearchFilters:
    """Build :class:`SearchFilters` from untrusted tool arguments, or raise :class:`FilterError`.

    Versions may be written ``2.0``, ``v2.0``, ``2`` or ``2.x``; products are matched
    case-insensitively against the catalog; dates are ISO ``YYYY-MM-DD`` (or ``YYYY``).
    """
    product = _clean(product)
    if product is not None and product not in catalog:
        raise FilterError(f"unknown product {product!r}; known: {', '.join(catalog)}")
    known = catalog[product] if product else sorted({v for vs in catalog.values() for v in vs})
    majors = {v.split(".")[0] for v in known}

    def check_version(raw: Any, name: str) -> str | None:
        value = _clean(raw)
        if value is None:
            return None
        value = value.removeprefix("v").removesuffix(".x")
        if re.fullmatch(r"[0-9]{1,3}", value):
            if value not in majors:
                raise FilterError(f"{name}: no {value}.x versions; known: {', '.join(known)}")
            return value
        if not re.fullmatch(r"[0-9]{1,3}\.[0-9]{1,3}", value):
            raise FilterError(f"{name}: versions look like 2.0 or 2, got {raw!r}")
        if name != "since_version" and value not in known:
            raise FilterError(f"{name}: unknown version {value}; known: {', '.join(known)}")
        return value

    doc = _clean(doc_type)
    if doc is not None:
        doc = doc.replace(" ", "-").replace("_", "-")
        if doc not in doc_types:
            raise FilterError(f"unknown doc_type {doc_type!r}; known: {', '.join(doc_types)}")
    return SearchFilters(
        product=product,
        version=check_version(version, "version"),
        since_version=check_version(since_version, "since_version"),
        exclude_version=check_version(exclude_version, "exclude_version"),
        doc_type=doc,
        released_after=_parse_date(released_after, "released_after"),
        released_before=_parse_date(released_before, "released_before"),
    )


def _parse_date(raw: Any, name: str) -> date | None:
    value = _clean(raw)
    if value is None:
        return None
    if re.fullmatch(r"[0-9]{4}", value):
        return date(int(value), 1, 1)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise FilterError(f"{name} must be YYYY-MM-DD or YYYY, got {raw!r}") from exc


_DOC_TYPE_WORDS = [
    (re.compile(r"\b(release notes?|changelog|what changed|what's new)\b", re.I), "release-notes"),
    (re.compile(r"\btroubleshoot(ing)?\b", re.I), "troubleshooting"),
    (re.compile(r"\b(config(uration)? reference|reference docs?)\b", re.I), "reference"),
]


def extract_filters(question: str, catalog: Catalog) -> SearchFilters:
    """Rule-based filters from a question ("Lumen 2.0", "Ferry 2.x", "since 2025", ...).

    Only values that exist in ``catalog`` are returned, so the result always validates.
    """
    text = question.lower()
    product = next((p for p in catalog if re.search(rf"\b{re.escape(p)}\b", text)), None)
    known = catalog[product] if product else sorted({v for vs in catalog.values() for v in vs})
    majors = {v.split(".")[0] for v in known}
    version = since = exclude = None

    upgrade = re.search(
        r"\bfrom\s+(?:version\s+|v)?(\d+\.\d+)\s+to\s+(?:version\s+|v)?(\d+\.\d+)\b", text
    )
    if upgrade and upgrade.group(2) in known:
        # "upgrade from 1.4 to 2.0": the target version's docs describe the upgrade.
        text = text[: upgrade.start()] + f" {upgrade.group(2)} " + text[upgrade.end() :]
    later = re.search(
        r"\b(?:since|from)\s+(?:version\s+|v)?(\d+(?:\.\d+)?)\b(?!-)"
        r"|\b(?:v|version\s+)?(\d+\.\d+)\s+(?:and|or)\s+(?:later|newer|above)\b",
        text,
    )
    if later:
        candidate = later.group(1) or later.group(2)
        if candidate and len(candidate) < 4:  # "since 2025" is a year, not a version
            since = candidate if "." in candidate else f"{candidate}.0"
    excl = re.search(r"\b(?:not|except|excluding|other than)\s+(?:version\s+|v)?(\d+\.\d+)\b", text)
    if excl and excl.group(1) in known:
        exclude = excl.group(1)
    if since is None:
        for m in re.finditer(
            r"(?<![\w.-])(?:v|version\s+)?(\d{1,3})(?:\.(\d{1,3}|x))?(?![\w-])", text
        ):
            major, minor = m.group(1), m.group(2)
            if minor and minor != "x":
                candidate = f"{major}.{minor}"
                if candidate in known and candidate != exclude:
                    version = candidate
                    break
            elif major in majors and (
                product and re.search(rf"\b{product}\s+{major}\b", text) or minor == "x"
            ):
                version = major
                break

    after = re.search(r"\b(?:since|after|from)\s+(\d{4})\b", text)
    before = re.search(r"\bbefore\s+(\d{4})\b", text)
    doc_type = next((d for pattern, d in _DOC_TYPE_WORDS if pattern.search(question)), None)
    return SearchFilters(
        product=product,
        version=version,
        since_version=since,
        exclude_version=exclude,
        doc_type=doc_type,
        released_after=date(int(after.group(1)), 1, 1) if after else None,
        released_before=date(int(before.group(1)), 1, 1) if before else None,
    )


def mentioned_versions(question: str, catalog: Catalog) -> list[str]:
    """Every catalog version a question names ("in Lumen 1.4 and in 3.0" -> 1.4, 3.0)."""
    text = question.lower()
    product = next((p for p in catalog if re.search(rf"\b{re.escape(p)}\b", text)), None)
    known = catalog[product] if product else sorted({v for vs in catalog.values() for v in vs})
    upgrade = re.search(
        r"\bfrom\s+(?:version\s+|v)?\d+\.\d+\s+to\s+(?:version\s+|v)?(\d+\.\d+)", text
    )
    if upgrade:
        return [upgrade.group(1)] if upgrade.group(1) in known else []
    found = re.findall(r"(?<![\w.-])(?:v|version\s+)?(\d{1,3}\.\d{1,3})(?![\w-])", text)
    return list(dict.fromkeys(v for v in found if v in known))
