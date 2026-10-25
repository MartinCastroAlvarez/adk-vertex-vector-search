"""Embedders: the hashing one for real, the others against fakes (no downloads, no network)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from agentic_rag.retrieval.embeddings import Embedder
from agentic_rag.retrieval.embeddings import FastEmbedEmbedder
from agentic_rag.retrieval.embeddings import HashEmbedder
from agentic_rag.retrieval.embeddings import OllamaEmbedder
from agentic_rag.retrieval.embeddings import VertexEmbedder
from agentic_rag.retrieval.embeddings import normalise


def test_hash_embedder_is_deterministic_unit_length_and_lexical() -> None:
    e = HashEmbedder(64)
    assert isinstance(e, Embedder) and e.dimensions == 64 and e.name == "hash-64"
    a, b, c = e.embed_documents(["default port 7410", "the default port", "raft election"])
    assert a == e.embed_documents(["default port 7410"])[0]
    assert math.isclose(sum(x * x for x in a), 1.0)
    dot = lambda x, y: sum(p * q for p, q in zip(x, y, strict=True))  # noqa: E731
    assert dot(a, b) > dot(a, c)
    assert e.embed_query("") == (0.0,) * 64
    with pytest.raises(ValueError):
        HashEmbedder(4)


class FakeTextEmbedding:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def passage_embed(self, texts: list[str]) -> Any:
        return (np.array([len(t), 1.0, 0.0]) for t in texts)

    def query_embed(self, text: str) -> Any:
        self.queries.append(text)
        return iter([np.array([0.0, 3.0, 4.0])])


def test_fastembed_wrapper_normalises_and_prefixes_bge_queries() -> None:
    fake = FakeTextEmbedding()
    e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", model_impl=fake)
    assert e.embed_query("port") == pytest.approx((0.0, 0.6, 0.8))
    assert fake.queries[0].startswith("Represent this sentence") and fake.queries[0].endswith(
        "port"
    )
    assert e.dimensions == 3 and e.name == "fastembed:BAAI/bge-small-en-v1.5"
    (doc,) = e.embed_documents(["abc"])
    assert math.isclose(sum(x * x for x in doc), 1.0)
    assert FastEmbedEmbedder("intfloat/e5-small", model_impl=fake).query_prefix == ""


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def test_ollama_embedder_posts_to_api_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    sent: list[tuple[str, dict[str, Any]]] = []

    def post(url: str, json: dict[str, Any], timeout: float) -> FakeResponse:
        sent.append((url, json))
        return FakeResponse({"embeddings": [[3.0, 4.0] for _ in json["input"]]})

    monkeypatch.setattr(httpx, "post", post)
    e = OllamaEmbedder("nomic-embed-text", "http://127.0.0.1:11434/")
    assert e.embed_query("port") == pytest.approx((0.6, 0.8))
    assert e.embed_documents(["a", "b"]) == [pytest.approx((0.6, 0.8))] * 2
    assert sent[0] == (
        "http://127.0.0.1:11434/api/embed",
        {"model": "nomic-embed-text", "input": ["search_query: port"]},
    )
    assert sent[1][1]["input"] == ["search_document: a", "search_document: b"]
    assert e.dimensions == 2 and e.name == "ollama:nomic-embed-text"
    assert OllamaEmbedder("mxbai-embed-large")._prefix("query") == ""
    with pytest.raises(ValueError):
        OllamaEmbedder(host="file:///etc/passwd")


class FakeGenai:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.models = self

    def embed_content(self, *, model: str, contents: list[str], config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})

        class E:
            values = [1.0, 1.0]

        class R:
            embeddings = [E() for _ in contents]

        return R()


def test_vertex_embedder_uses_task_types_and_dimensions() -> None:
    fake = FakeGenai()
    e = VertexEmbedder(dimensions=768, client=fake)
    docs = e.embed_documents([f"t{i}" for i in range(120)])
    assert len(docs) == 120 and docs[0] == pytest.approx((2**-0.5, 2**-0.5))
    assert [len(c["contents"]) for c in fake.calls] == [50, 50, 20]
    assert fake.calls[0]["model"] == "gemini-embedding-001"
    assert fake.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert fake.calls[0]["config"].output_dimensionality == 768
    e.embed_query("q")
    assert fake.calls[-1]["config"].task_type == "RETRIEVAL_QUERY"
    assert e.dimensions == 768 and e.name == "vertex:gemini-embedding-001"


def test_normalise_zero_vector() -> None:
    assert normalise([0.0, 0.0]) == (0.0, 0.0)
