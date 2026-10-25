"""Corpus loading and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.retrieval.corpus import CorpusError
from agentic_rag.retrieval.corpus import chunk_document
from agentic_rag.retrieval.corpus import iter_chunks
from agentic_rag.retrieval.corpus import load_corpus
from agentic_rag.retrieval.corpus import parse_document
from agentic_rag.retrieval.corpus import slugify
from agentic_rag.retrieval.corpus import version_number

HEAD = "---\ntitle: T\nproduct: lumen\nversion: 2.0\ndoc_type: guide\nreleased: 2025-01-20\n---\n"


def test_the_bundled_corpus() -> None:
    docs = load_corpus()
    assert len(docs) == 23
    assert {d.product for d in docs} == {"lumen", "ferry"}
    chunks = list(iter_chunks(docs))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert "lumen-2.0-configuration#network-settings.0" in ids
    c = next(c for c in chunks if c.chunk_id == "lumen-2.0-configuration#network-settings.0")
    assert c.section_key == "lumen-2.0-configuration#network-settings"
    assert c.major == "2" and c.released == "2025-01-20"
    assert c.embedding_text().startswith("Lumen 2.0 configuration reference. Network settings.")


def test_sections_become_chunks_and_long_ones_split_with_overlap() -> None:
    long = " ".join(f"Sentence number {i} is here." for i in range(60))
    doc = parse_document(
        HEAD + f"Intro text.\n\n## Short\n\nOne. Two.\n\n## Long one\n\n{long}\n", "d", "d.md"
    )
    chunks = chunk_document(doc, max_chars=300)
    assert [c.chunk_id for c in chunks[:2]] == ["d#overview.0", "d#short.0"]
    longs = [c for c in chunks if c.section == "Long one"]
    assert len(longs) > 3 and all(len(c.text) <= 330 for c in longs)
    last_of_first = longs[0].text.split(". ")[-1]
    assert longs[1].text.startswith(last_of_first.rstrip("."))  # one sentence of overlap


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("no front matter", "front matter"),
        ("---\ntitle T\n---\n", "bad front matter"),
        ("---\ntitle: T\n---\n", "lacks"),
        (HEAD.replace("lumen", "Lumen DB"), "product"),
        (HEAD.replace("2.0", "two"), "version"),
        (HEAD.replace("guide", "blog"), "doc_type"),
        (HEAD.replace("2025-01-20", "yesterday"), "YYYY-MM-DD"),
        (HEAD + "## A\n\nx\n\n## A\n\ny\n", "duplicate section"),
    ],
)
def test_bad_documents_are_refused(text: str, error: str) -> None:
    with pytest.raises(CorpusError, match=error):
        chunk_document(parse_document(text, "d", "d.md"))


def test_load_corpus_guards_the_directory(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="not a directory"):
        load_corpus(tmp_path / "missing")
    with pytest.raises(CorpusError, match="no Markdown"):
        load_corpus(tmp_path)
    (tmp_path / "Bad Name.md").write_text(HEAD + "x\n")
    with pytest.raises(CorpusError, match="lower-case"):
        load_corpus(tmp_path)
    (tmp_path / "Bad Name.md").unlink()
    outside = tmp_path.parent / "outside-secret.md"
    outside.write_text(HEAD + "secret\n")
    (tmp_path / "link.md").symlink_to(outside)
    with pytest.raises(CorpusError, match="symlinks"):
        load_corpus(tmp_path)
    (tmp_path / "link.md").unlink()
    (tmp_path / "big.md").write_text(HEAD + "x" * 300_000)
    with pytest.raises(CorpusError, match="larger"):
        load_corpus(tmp_path)
    (tmp_path / "big.md").unlink()
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "same.md").write_text(HEAD + "x\n")
    (tmp_path / "same.md").write_text(HEAD + "y\n")
    with pytest.raises(CorpusError, match="duplicate document"):
        load_corpus(tmp_path)


def test_helpers() -> None:
    assert slugify("Upgrading from 1.4") == "upgrading-from-1-4"
    assert slugify("!!!") == "section"
    assert version_number("2.0") == 200 and version_number("1.15") == 115
