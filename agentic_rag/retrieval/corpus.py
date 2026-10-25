"""The docs corpus: Markdown files with a small front matter, split into chunks.

Every document starts with::

    ---
    title: Lumen 2.0 configuration reference
    product: lumen
    version: 2.0
    doc_type: reference
    released: 2025-01-20
    ---

(parsed as ``key: value`` lines - no YAML loader). Chunks follow the Markdown ``##``
sections: one section is one chunk unless it is longer than ``max_chars``, in which case
it is split on sentence boundaries with one sentence of overlap. The section title and the
document title are kept with the chunk, so a passage like "defaults to 7410" still says
which product and version it is about.

Chunk ids are ``<doc_id>#<section-slug>.<n>``: stable across runs (re-ingesting upserts
the same ids), readable in a citation, and accepted as Vector Search datapoint ids.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

PRODUCT = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
VERSION = re.compile(r"^[0-9]{1,3}\.[0-9]{1,3}$")
DOC_TYPES = ("guide", "reference", "release-notes", "troubleshooting")
MAX_FILE_BYTES = 256 * 1024


class CorpusError(ValueError):
    """A document that cannot be ingested."""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def version_number(version: str) -> int:
    """``2.0`` -> 200, ``1.15`` -> 115: versions as integers, for numeric restricts."""
    major, minor = version.split(".")
    return int(major) * 100 + int(minor)


def date_number(day: date) -> int:
    """``2025-01-20`` -> 20250120, for numeric restricts."""
    return day.year * 10000 + day.month * 100 + day.day


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    product: str
    version: str
    doc_type: str
    released: date
    body: str
    source: str


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage plus everything a citation or a filter needs."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    product: str
    version: str
    doc_type: str
    released: str
    text: str
    source: str

    @property
    def section_key(self) -> str:
        """``<doc_id>#<section-slug>``: what the benchmark's relevance labels name."""
        return self.chunk_id.rsplit(".", 1)[0]

    @property
    def major(self) -> str:
        return self.version.split(".")[0]

    def embedding_text(self) -> str:
        """What gets embedded: titles give a bare passage its product and version."""
        return f"{self.title}. {self.section}.\n{self.text}"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Chunk:
        return cls(**{k: str(raw[k]) for k in cls.__dataclass_fields__})


def parse_document(text: str, doc_id: str, source: str) -> Document:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not match:
        raise CorpusError(f"{source}: missing front matter")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            raise CorpusError(f"{source}: bad front matter line {line!r}")
        meta[key.strip()] = value.strip()
    missing = {"title", "product", "version", "doc_type", "released"} - set(meta)
    if missing:
        raise CorpusError(f"{source}: front matter lacks {', '.join(sorted(missing))}")
    if not PRODUCT.match(meta["product"]):
        raise CorpusError(f"{source}: bad product {meta['product']!r}")
    if not VERSION.match(meta["version"]):
        raise CorpusError(f"{source}: version must look like 2.0, got {meta['version']!r}")
    if meta["doc_type"] not in DOC_TYPES:
        raise CorpusError(f"{source}: doc_type must be one of {', '.join(DOC_TYPES)}")
    try:
        released = date.fromisoformat(meta["released"])
    except ValueError as exc:
        raise CorpusError(f"{source}: released must be YYYY-MM-DD") from exc
    return Document(
        doc_id=doc_id,
        title=meta["title"],
        product=meta["product"],
        version=meta["version"],
        doc_type=meta["doc_type"],
        released=released,
        body=text[match.end() :],
        source=source,
    )


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z`])", text) if s]


def _split(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = _sentences(text)
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join([*current, sentence])) > max_chars:
            pieces.append(" ".join(current))
            current = current[-1:]  # one sentence of overlap
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_document(doc: Document, max_chars: int = 700) -> list[Chunk]:
    sections: list[tuple[str, list[str]]] = [("Overview", [])]
    for line in doc.body.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            sections.append((heading.group(1), []))
        else:
            sections[-1][1].append(line)
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for title, lines in sections:
        text = " ".join(" ".join(lines).split())
        if not text:
            continue
        slug = slugify(title)
        if slug in seen:
            raise CorpusError(f"{doc.source}: duplicate section {title!r}")
        seen.add(slug)
        for n, piece in enumerate(_split(text, max_chars)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{slug}.{n}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section=title,
                    product=doc.product,
                    version=doc.version,
                    doc_type=doc.doc_type,
                    released=doc.released.isoformat(),
                    text=piece,
                    source=doc.source,
                )
            )
    return chunks


def bundled_corpus() -> Path:
    """The fictional product docs shipped with the package (Lumen and Ferry)."""
    return Path(str(resources.files("agentic_rag") / "corpus"))


def load_corpus(root: Path | None = None) -> list[Document]:
    """Every ``*.md`` under ``root`` (default: the bundled corpus), sorted by path.

    Symlinks and files that resolve outside ``root`` are refused, so a corpus directory
    can't smuggle in ``/etc/passwd``; files over 256 KiB are refused too.
    """
    root = (root or bundled_corpus()).resolve()
    if not root.is_dir():
        raise CorpusError(f"{root} is not a directory")
    docs: list[Document] = []
    ids: set[str] = set()
    for path in sorted(root.rglob("*.md")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise CorpusError(f"{path}: symlinks and paths outside the corpus are refused")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise CorpusError(f"{path}: larger than {MAX_FILE_BYTES} bytes")
        doc_id = path.stem
        if not re.match(r"^[a-z0-9][a-z0-9.-]{0,99}$", doc_id):
            raise CorpusError(f"{path}: file names must be lower-case, digits, '.' and '-'")
        if doc_id in ids:
            raise CorpusError(f"{path}: duplicate document id {doc_id}")
        ids.add(doc_id)
        source = path.relative_to(root).as_posix()
        docs.append(parse_document(path.read_text(encoding="utf-8"), doc_id, source))
    if not docs:
        raise CorpusError(f"{root}: no Markdown documents")
    return docs


def iter_chunks(docs: list[Document], max_chars: int = 700) -> Iterator[Chunk]:
    for doc in docs:
        yield from chunk_document(doc, max_chars)
