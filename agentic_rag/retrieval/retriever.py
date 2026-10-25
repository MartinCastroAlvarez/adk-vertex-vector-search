"""``Retriever``: question + filters -> passages, on any :class:`VectorStore`.

It embeds the query twice - dense (the :class:`Embedder`) and sparse (BM25) - builds one
Vector Search :class:`Query` with the filters as restricts, crowding by document, and
joins the returned ids with the :class:`ChunkCatalog`. ``mode`` picks the ranking:
``hybrid`` (RRF of both, ``alpha`` weighting dense), ``dense`` or ``sparse``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_rag.filters import SearchFilters
from agentic_rag.retrieval.base import VectorStore
from agentic_rag.retrieval.catalog import ChunkCatalog
from agentic_rag.retrieval.corpus import Chunk
from agentic_rag.retrieval.embeddings import Embedder
from agentic_rag.retrieval.sparse import BM25Encoder
from agentic_rag.retrieval.types import Query
from agentic_rag.retrieval.types import SparseVector

Mode = Literal["hybrid", "dense", "sparse"]
MODES: tuple[Mode, ...] = ("hybrid", "dense", "sparse")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    rank: int
    distance: float | None
    sparse_distance: float | None


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        catalog: ChunkCatalog,
        embedder: Embedder,
        sparse: BM25Encoder,
        alpha: float = 0.5,
        per_document: int | None = 2,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.store = store
        self.catalog = catalog
        self.embedder = embedder
        self.sparse = sparse
        self.alpha = alpha
        self.per_document = per_document
        self.queries: list[Query] = []  # what was sent, for tests and the CLI's --trace

    def build_query(
        self, text: str, filters: SearchFilters, top_k: int, mode: Mode = "hybrid"
    ) -> Query:
        dense = self.embedder.embed_query(text) if mode != "sparse" else ()
        sparse = self.sparse.encode_query(text) if mode != "dense" else SparseVector()
        if mode == "sparse" and not sparse:
            # Nothing to match on (only stopwords): fall back to dense.
            dense, mode = self.embedder.embed_query(text), "dense"
        return Query(
            feature_vector=tuple(dense),
            sparse_embedding=sparse,
            restricts=filters.restricts(),
            numeric_restricts=filters.numeric_restricts(),
            neighbor_count=top_k,
            per_crowding_attribute_neighbor_count=self.per_document,
            rrf_alpha=self.alpha if mode == "hybrid" and sparse else None,
        )

    def search(
        self,
        text: str,
        filters: SearchFilters | None = None,
        top_k: int = 5,
        mode: Mode = "hybrid",
    ) -> list[Hit]:
        query = self.build_query(text, filters or SearchFilters(), top_k, mode)
        self.queries.append(query)
        neighbors = self.store.find_neighbors([query])[0]
        hits: list[Hit] = []
        for n in neighbors:
            chunk = self.catalog.get(n.datapoint_id)
            if chunk is None:  # in the index but not in the catalog: stale, skip it
                continue
            hits.append(Hit(chunk, len(hits) + 1, n.distance, n.sparse_distance))
        return hits

    def close(self) -> None:
        self.store.close()
