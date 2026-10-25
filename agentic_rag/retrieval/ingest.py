"""The ingestion pipeline: load -> chunk -> embed (dense + sparse) -> upsert with restricts.

The same datapoints go to either backend; for Vertex they are streaming upserts
(``UpsertDatapoints``) into an index created with ``STREAM_UPDATE``. Re-ingesting is
idempotent - chunk ids are stable, so an upsert replaces - and chunks that disappeared
from the corpus are removed (``RemoveDatapoints``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from agentic_rag.retrieval.base import VectorStore
from agentic_rag.retrieval.catalog import ChunkCatalog
from agentic_rag.retrieval.corpus import Chunk
from agentic_rag.retrieval.corpus import Document
from agentic_rag.retrieval.corpus import date_number
from agentic_rag.retrieval.corpus import iter_chunks
from agentic_rag.retrieval.corpus import version_number
from agentic_rag.retrieval.embeddings import Embedder
from agentic_rag.retrieval.sparse import BM25Encoder
from agentic_rag.retrieval.types import Datapoint
from agentic_rag.retrieval.types import NumericRestrict
from agentic_rag.retrieval.types import Restrict

EMBED_BATCH = 32


@dataclass(frozen=True)
class IngestReport:
    documents: int
    chunks: int
    upserted: int
    removed: int
    dimensions: int
    embedder: str


def to_datapoint(chunk: Chunk, dense: Sequence[float], sparse: BM25Encoder) -> Datapoint:
    """A chunk as an ``IndexDatapoint``: vectors, restricts, numeric restricts, crowding tag."""
    return Datapoint(
        datapoint_id=chunk.chunk_id,
        feature_vector=tuple(dense),
        sparse_embedding=sparse.encode_document(chunk.embedding_text()),
        restricts=(
            Restrict("product", (chunk.product,)),
            Restrict("version", (chunk.version,)),
            Restrict("major", (chunk.major,)),
            Restrict("doc_type", (chunk.doc_type,)),
        ),
        numeric_restricts=(
            NumericRestrict("version_num", value_int=version_number(chunk.version)),
            NumericRestrict("released", value_int=date_number(date.fromisoformat(chunk.released))),
        ),
        # One document should not fill every slot of a result list.
        crowding_tag=chunk.doc_id,
    )


def ingest(
    docs: list[Document],
    store: VectorStore,
    embedder: Embedder,
    catalog: ChunkCatalog | None = None,
    max_chars: int = 700,
) -> tuple[ChunkCatalog, BM25Encoder, IngestReport]:
    chunks = list(iter_chunks(docs, max_chars))
    sparse = BM25Encoder().fit(c.embedding_text() for c in chunks)
    stale: list[str] = []
    if catalog is not None:
        current = {c.chunk_id for c in chunks}
        stale = [c.chunk_id for c in catalog.chunks() if c.chunk_id not in current]
    upserted = 0
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        vectors = embedder.embed_documents([c.embedding_text() for c in batch])
        store.upsert([to_datapoint(c, v, sparse) for c, v in zip(batch, vectors, strict=True)])
        upserted += len(batch)
    if stale:
        store.remove(stale)
    new_catalog = ChunkCatalog(chunks, embedder=embedder.name)
    report = IngestReport(
        documents=len(docs),
        chunks=len(chunks),
        upserted=upserted,
        removed=len(stale),
        dimensions=embedder.dimensions,
        embedder=embedder.name,
    )
    return new_catalog, sparse, report
