"""Sparse embeddings for hybrid search: BM25 as a sparse vector.

Vertex AI Vector Search takes the sparse half of a hybrid datapoint as ``dimensions`` +
``values`` and scores it with a dot product; its docs suggest TF-IDF, BM25 or SPLADE to
produce them. Here a document's vector holds the BM25 weight of each term,

    w(t, d) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

and a query's vector holds 1.0 per distinct term, so ``query · document`` *is* the BM25
score - on Vertex and on the local store alike. Terms map to dimensions with a stable
64-bit hash (blake2b), so no vocabulary has to be shipped with the index; only the corpus
statistics (document count, average length, document frequencies) are kept, to encode
new documents for streaming upserts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentic_rag.retrieval.types import SparseVector

_TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "after",
        "before",
        "about",
        "should",
        "would",
        "each",
        "every",
        "any",
        "some",
        "all",
    ]
)


def words(text: str) -> list[str]:
    """Lower-cased words, stopwords removed, not stemmed (what a rewritten query is made of)."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; keeps ``3.0``, ``lum-104``, ``ferry.toml`` whole, drops stopwords.

    A compound token also yields its parts (``ferry.toml`` -> ``ferry``, ``toml``), so a
    question that says "toml" still matches.
    """
    out: list[str] = []
    for token in words(text):
        out.append(_stem(token))
        parts = re.split(r"[._-]", token)
        if len(parts) > 1:
            out.extend(_stem(p) for p in parts if p and p not in STOPWORDS and not p.isdigit())
    return out


@lru_cache(maxsize=1)
def _stemmer() -> Any:
    from py_rust_stemmers import SnowballStemmer

    return SnowballStemmer("english")


@lru_cache(maxsize=65536)
def _stem(token: str) -> str:
    """Snowball (Porter 2) stems: ``rolling`` -> ``roll``, ``replaces``/``replaced`` -> ``replac``.

    Codes and versions (anything with a digit) are kept as they are.
    """
    if any(ch.isdigit() for ch in token):
        return token
    return str(_stemmer().stem_word(token))


def dimension(term: str) -> int:
    """A stable, non-negative int64 dimension for ``term``."""
    digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF


@dataclass
class BM25Encoder:
    """Fit on a corpus, then encode documents and queries as :class:`SparseVector`."""

    k1: float = 1.2
    b: float = 0.75
    documents: int = 0
    total_length: int = 0
    df: Counter[str] = field(default_factory=Counter)

    def fit(self, texts: Iterable[str]) -> BM25Encoder:
        for text in texts:
            terms = tokenize(text)
            self.documents += 1
            self.total_length += len(terms)
            self.df.update(set(terms))
        return self

    @property
    def avgdl(self) -> float:
        return self.total_length / self.documents if self.documents else 1.0

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.documents - n + 0.5) / (n + 0.5))

    def encode_document(self, text: str) -> SparseVector:
        terms = tokenize(text)
        counts = Counter(terms)
        norm = self.k1 * (1 - self.b + self.b * len(terms) / self.avgdl)
        weights = {
            t: self.idf(t) * tf * (self.k1 + 1) / (tf + norm) for t, tf in sorted(counts.items())
        }
        return _vector(weights)

    def encode_query(self, text: str) -> SparseVector:
        return _vector(dict.fromkeys(sorted(set(tokenize(text))), 1.0))

    def save(self, path: Path) -> None:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
            "total_length": self.total_length,
            "df": dict(sorted(self.df.items())),
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> BM25Encoder:
        raw = json.loads(path.read_text())
        return cls(
            k1=float(raw["k1"]),
            b=float(raw["b"]),
            documents=int(raw["documents"]),
            total_length=int(raw["total_length"]),
            df=Counter({str(k): int(v) for k, v in raw["df"].items()}),
        )


def _vector(weights: dict[str, float]) -> SparseVector:
    merged: dict[int, float] = {}
    for term, weight in weights.items():
        if weight > 0:
            d = dimension(term)
            merged[d] = merged.get(d, 0.0) + weight
    dims = sorted(merged)
    return SparseVector(tuple(dims), tuple(merged[d] for d in dims))
