"""``VectorStore``: the four Vector Search calls the agent needs, nothing more.

Both backends implement it - :class:`~agentic_rag.retrieval.local.LocalVectorSearch`
(FAISS + an inverted index, free, on disk) and
:class:`~agentic_rag.retrieval.vertex.VertexVectorSearchStore` (``upsert_datapoints``,
``remove_datapoints`` and ``find_neighbors`` on a deployed index) - and the contract
suite (``tests/test_contract.py``) runs every behaviour on both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from typing import runtime_checkable

from agentic_rag.retrieval.types import Datapoint
from agentic_rag.retrieval.types import Neighbor
from agentic_rag.retrieval.types import Query

#: The constant ``k`` of Reciprocal Rank Fusion, ``1 / (k + rank)``. Vertex documents RRF
#: without naming its constant; 60 is the value from the original RRF paper.
RRF_K = 60


@runtime_checkable
class VectorStore(Protocol):
    """A Vector Search index: streaming upserts, removals and nearest-neighbour queries."""

    def upsert(self, datapoints: Sequence[Datapoint]) -> None:
        """Insert or replace datapoints by id (``IndexService.UpsertDatapoints``)."""
        ...

    def remove(self, datapoint_ids: Sequence[str]) -> None:
        """Delete datapoints by id; unknown ids are ignored (``RemoveDatapoints``)."""
        ...

    def find_neighbors(self, queries: Sequence[Query]) -> list[list[Neighbor]]:
        """One neighbour list per query, best first (``MatchService.FindNeighbors``)."""
        ...

    def close(self) -> None:
        """Release clients or files."""
        ...


def rrf_fuse(
    dense: Sequence[Neighbor], sparse: Sequence[Neighbor], alpha: float
) -> list[tuple[str, float]]:
    """Merge two rankings: ``alpha / (k + dense_rank) + (1 - alpha) / (k + sparse_rank)``."""
    scores: dict[str, float] = {}
    for rank, n in enumerate(dense, 1):
        scores[n.datapoint_id] = scores.get(n.datapoint_id, 0.0) + alpha / (RRF_K + rank)
    for rank, n in enumerate(sparse, 1):
        scores[n.datapoint_id] = scores.get(n.datapoint_id, 0.0) + (1 - alpha) / (RRF_K + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def apply_crowding(ranked: Sequence[Neighbor], per_tag: int | None, limit: int) -> list[Neighbor]:
    """At most ``per_tag`` neighbours per crowding tag, then the first ``limit``."""
    out: list[Neighbor] = []
    seen: dict[str, int] = {}
    for n in ranked:
        if per_tag is not None and n.crowding_tag is not None:
            if seen.get(n.crowding_tag, 0) >= per_tag:
                continue
            seen[n.crowding_tag] = seen.get(n.crowding_tag, 0) + 1
        out.append(n)
        if len(out) == limit:
            break
    return out
