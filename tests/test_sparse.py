"""BM25 as sparse vectors, and the tokenizer."""

from __future__ import annotations

import math
from pathlib import Path

from agentic_rag.retrieval.sparse import BM25Encoder
from agentic_rag.retrieval.sparse import dimension
from agentic_rag.retrieval.sparse import tokenize
from agentic_rag.retrieval.sparse import words


def test_tokenize_keeps_codes_versions_and_compound_names() -> None:
    assert tokenize("Error LUM-104 in Lumen 2.0 with ferry.toml") == [
        "error",
        "lum-104",
        "lum",
        "lumen",
        "2.0",
        "ferry.toml",
        "ferri",
        "toml",
    ]
    assert tokenize("Rolling back the releases") == ["roll", "back", "releas"]
    assert words("How do I roll back?") == ["roll", "back"]


def test_query_dot_document_is_the_bm25_score() -> None:
    docs = ["the port is 7410", "compaction starts at a ratio", "port and port again"]
    enc = BM25Encoder().fit(docs)
    q = enc.encode_query("port")
    scores = [q.dot(enc.encode_document(d)) for d in docs]
    assert scores[1] == 0.0
    assert scores[2] > scores[0] > 0  # tf saturates but still grows
    n, df = 3, 2
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    dl, avgdl = 2, enc.avgdl  # "port 7410": stopwords dropped
    expected = idf * (1 * 2.2) / (1 + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert scores[0] == __import__("pytest").approx(expected)


def test_dimensions_are_stable_and_int64() -> None:
    assert dimension("port") == dimension("port")
    assert 0 <= dimension("port") < 2**63
    assert dimension("port") != dimension("ports")


def test_save_and_load(tmp_path: Path) -> None:
    enc = BM25Encoder().fit(["alpha beta", "beta gamma"])
    enc.save(tmp_path / "s.json")
    again = BM25Encoder.load(tmp_path / "s.json")
    assert again.encode_document("beta gamma") == enc.encode_document("beta gamma")
    assert BM25Encoder().avgdl == 1.0
