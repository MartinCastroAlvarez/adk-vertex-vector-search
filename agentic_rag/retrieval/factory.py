"""Settings -> embedder, store and a ready :class:`Retriever`.

The local index lives in ``RAG_INDEX_DIR/<embedder>-<model>/`` (``vectors.npy``,
``datapoints.json``, ``chunks.json``, ``sparse.json``). The first query builds it from the
bundled corpus if it is missing, so ``adk run`` and the CLI work right after install.
The Vertex index is never built implicitly: ``agentic-rag --ingest`` with ``RAG_STORE=vertex``
upserts (and bills) on purpose, and writes the catalog the agent then reads.
"""

from __future__ import annotations

from pathlib import Path

from agentic_rag.config import ConfigError
from agentic_rag.config import Settings
from agentic_rag.retrieval.base import VectorStore
from agentic_rag.retrieval.catalog import ChunkCatalog
from agentic_rag.retrieval.corpus import load_corpus
from agentic_rag.retrieval.embeddings import Embedder
from agentic_rag.retrieval.embeddings import FastEmbedEmbedder
from agentic_rag.retrieval.embeddings import HashEmbedder
from agentic_rag.retrieval.embeddings import OllamaEmbedder
from agentic_rag.retrieval.embeddings import VertexEmbedder
from agentic_rag.retrieval.ingest import IngestReport
from agentic_rag.retrieval.ingest import ingest
from agentic_rag.retrieval.local import LocalVectorSearch
from agentic_rag.retrieval.retriever import Retriever
from agentic_rag.retrieval.sparse import BM25Encoder


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder()
    if settings.embedder == "fastembed":
        cache = settings.index_dir / "models"
        return FastEmbedEmbedder(settings.embed_model, cache_dir=str(cache))
    if settings.embedder == "ollama":
        return OllamaEmbedder(settings.embed_model, settings.ollama_host)
    return VertexEmbedder(  # pragma: no cover - would reach Google Cloud
        settings.embed_model,
        settings.embed_dimensions,
        project=settings.vertex.project,
        location=settings.vertex.location,
    )


def store_dir(settings: Settings) -> Path:
    return settings.index_dir / "vertex" if settings.store == "vertex" else settings.index_path


def build_store(settings: Settings) -> VectorStore:
    if settings.store == "local":
        return LocalVectorSearch(settings.index_path, index=settings.index)
    from agentic_rag.retrieval.vertex import VertexVectorSearchStore

    v = settings.vertex
    return VertexVectorSearchStore(
        project=v.project,
        location=v.location,
        index_id=v.index_id,
        index_endpoint_id=v.index_endpoint_id,
        deployed_index_id=v.deployed_index_id,
        public_endpoint_domain=v.public_endpoint_domain,
    )


def ingest_corpus(
    settings: Settings,
    corpus: Path | None = None,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> IngestReport:
    """Chunk, embed and upsert ``corpus`` (default: the bundled docs) into the store."""
    directory = store_dir(settings)
    store = store or build_store(settings)
    embedder = embedder or build_embedder(settings)
    previous = None
    if (directory / "chunks.json").exists():
        previous = ChunkCatalog.load(directory / "chunks.json")
    try:
        catalog, sparse, report = ingest(load_corpus(corpus), store, embedder, previous)
    finally:
        store.close()
    catalog.save(directory / "chunks.json")
    sparse.save(directory / "sparse.json")
    return report


def open_retriever(
    settings: Settings,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> Retriever:
    """A retriever over the configured index, building the local one on first use."""
    directory = store_dir(settings)
    embedder = embedder or build_embedder(settings)
    catalog_path = directory / "chunks.json"
    stale = catalog_path.exists() and ChunkCatalog.load(catalog_path).embedder != embedder.name
    if not catalog_path.exists() or stale:
        if settings.store == "vertex":
            raise ConfigError(
                f"no catalog for the {embedder.name} embedder at {catalog_path}; run "
                "`agentic-rag --ingest` with RAG_STORE=vertex first (this upserts into your "
                "paid index)"
            )
        if stale:
            for name in ("vectors.npy", "datapoints.json", "chunks.json", "sparse.json"):
                (directory / name).unlink(missing_ok=True)
        local = LocalVectorSearch(settings.index_path, index=settings.index)
        ingest_corpus(settings, store=local, embedder=embedder)
    return Retriever(
        store=store or build_store(settings),
        catalog=ChunkCatalog.load(catalog_path),
        embedder=embedder,
        sparse=BM25Encoder.load(directory / "sparse.json"),
        alpha=settings.alpha,
        per_document=settings.per_document,
    )
