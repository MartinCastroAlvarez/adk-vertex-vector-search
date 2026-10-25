"""Shared fixtures: two Vector Search backends behind one interface, a ready retriever, stubs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agentic_rag.config import Settings
from agentic_rag.researcher import build_agent
from agentic_rag.retrieval.base import VectorStore
from agentic_rag.retrieval.corpus import load_corpus
from agentic_rag.retrieval.embeddings import HashEmbedder
from agentic_rag.retrieval.ingest import ingest
from agentic_rag.retrieval.local import LocalVectorSearch
from agentic_rag.retrieval.retriever import Retriever
from agentic_rag.runner import Researcher
from agentic_rag.stub import ScriptedLlm
from tests.fake_vector_search import DEPLOYED
from tests.fake_vector_search import DOMAIN
from tests.fake_vector_search import FakeVectorSearch


@pytest.fixture(autouse=True)
def _no_google_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to find Google credentials fails the test: nothing here may authenticate."""
    import google.auth
    import google.auth._default

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("google.auth.default() was called: tests must never authenticate")

    monkeypatch.setattr(google.auth, "default", refuse)
    monkeypatch.setattr(google.auth._default, "default", refuse)
    for name in ("GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    for name in list(__import__("os").environ):
        if name.startswith("RAG_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx (ollama embeddings, LiteLlm) may not leave the process unless a test fakes it."""
    import httpx

    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(httpx, "post", refuse)


@dataclass
class Backend:
    kind: str
    store: VectorStore
    fake: FakeVectorSearch | None = None


def make_store(kind: str) -> Backend:
    if kind == "local":
        return Backend(kind, LocalVectorSearch())
    if kind == "hnsw":
        return Backend(kind, LocalVectorSearch(index="hnsw"))
    from agentic_rag.retrieval.vertex import VertexVectorSearchStore

    fake = FakeVectorSearch()
    store = VertexVectorSearchStore(
        project="demo-project",
        location="us-central1",
        index_id="123",
        index_endpoint_id="456",
        deployed_index_id=DEPLOYED,
        public_endpoint_domain=DOMAIN,
        index_client=fake.index_service,
        match_client=fake.match_service,
    )
    return Backend(kind, store, fake)


@pytest.fixture(params=["local", "vertex"])
def backend(request: pytest.FixtureRequest) -> Backend:
    """Every contract test runs twice: LocalVectorSearch, and the Vertex adapter on a fake."""
    return make_store(request.param)


def build_retriever(kind: str = "local") -> Retriever:
    b = make_store(kind)
    embedder = HashEmbedder()
    catalog, sparse, _ = ingest(load_corpus(), b.store, embedder)
    return Retriever(b.store, catalog, embedder, sparse)


@pytest.fixture(scope="session")
def retriever() -> Retriever:
    """The bundled corpus, ingested once with the hashing embedder into a local index."""
    return build_retriever("local")


@pytest.fixture(params=["local", "vertex"])
def any_retriever(request: pytest.FixtureRequest) -> Retriever:
    return build_retriever(request.param)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(provider="stub", embedder="hash", embed_model="hash", index_dir=tmp_path)


def scripted(
    retriever: Retriever, script: list[Any], max_hops: int = 3, repeat_last: bool = False
) -> tuple[Researcher, ScriptedLlm]:
    settings = Settings(provider="stub", embedder="hash", embed_model="hash", max_hops=max_hops)
    model = ScriptedLlm(script=script, repeat_last=repeat_last)
    agent = build_agent(settings, model=model, retriever=retriever)
    return Researcher(agent=agent, settings=settings), model


@pytest.fixture
def fresh_queries(retriever: Retriever) -> Iterator[list[Any]]:
    """The session retriever's query log, emptied for this test."""
    retriever.queries.clear()
    yield retriever.queries
    retriever.queries.clear()
