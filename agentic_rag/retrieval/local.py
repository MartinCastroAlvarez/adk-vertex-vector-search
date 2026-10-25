"""``LocalVectorSearch``: Vertex AI Vector Search semantics on FAISS, free and on disk.

What it mirrors, call for call:

* **Streaming upserts / removals** by datapoint id (an upsert of a known id replaces it).
* **Dense search** with inner product (``DOT_PRODUCT_DISTANCE`` on unit vectors, i.e.
  cosine), on a FAISS ``IndexFlatIP`` (exact) or ``IndexHNSWFlat`` (approximate, like the
  tree-AH index Vertex builds).
* **Sparse search**: the dot product of sparse vectors, over an inverted index.
* **Hybrid search**: both rankings merged with Reciprocal Rank Fusion weighted by the
  query's ``rrf_alpha`` (1 = dense only, 0 = sparse only).
* **Restricts and numeric filters** applied *before* ranking (Vertex filters inside the
  ANN search, so a filtered query still returns ``neighbor_count`` matches if they exist),
  with the semantics in :func:`agentic_rag.retrieval.types.matches`.
* **Crowding**: at most ``per_crowding_attribute_neighbor_count`` results per
  ``crowding_tag``.

Persistence is two plain files in a directory - ``vectors.npy`` (float32, loaded with
``allow_pickle=False``) and ``datapoints.json`` - written atomically on :meth:`flush`.
No pickle, no FAISS binary index on disk: the index is rebuilt from the vectors on load.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Literal

import faiss as _faiss
import numpy as np

from agentic_rag.retrieval.base import apply_crowding
from agentic_rag.retrieval.base import rrf_fuse
from agentic_rag.retrieval.types import Datapoint
from agentic_rag.retrieval.types import InvalidDatapointError
from agentic_rag.retrieval.types import Neighbor
from agentic_rag.retrieval.types import NumericRestrict
from agentic_rag.retrieval.types import Query
from agentic_rag.retrieval.types import Restrict
from agentic_rag.retrieval.types import SparseVector
from agentic_rag.retrieval.types import matches

#: faiss' bundled type stubs lag its API (search parameters with selectors), so it is
#: used untyped.
faiss: Any = _faiss
IndexKind = Literal["flat", "hnsw"]
FORMAT_VERSION = 1
#: How many candidates each ranking contributes before fusion and crowding.
MIN_CANDIDATES = 50


def datapoint_to_json(d: Datapoint) -> dict[str, Any]:
    return {
        "id": d.datapoint_id,
        "sparse": {
            "dimensions": list(d.sparse_embedding.dimensions),
            "values": list(d.sparse_embedding.values),
        },
        "restricts": [{"namespace": r.namespace, "allow": list(r.allow)} for r in d.restricts],
        "numeric_restricts": [
            {
                k: v
                for k, v in (
                    ("namespace", n.namespace),
                    ("value_int", n.value_int),
                    ("value_float", n.value_float),
                    ("value_double", n.value_double),
                )
                if v is not None
            }
            for n in d.numeric_restricts
        ],
        "crowding_tag": d.crowding_tag,
    }


def datapoint_from_json(raw: dict[str, Any], vector: Sequence[float]) -> Datapoint:
    sparse = raw.get("sparse") or {}
    return Datapoint(
        datapoint_id=str(raw["id"]),
        feature_vector=tuple(float(x) for x in vector),
        sparse_embedding=SparseVector(
            tuple(int(x) for x in sparse.get("dimensions", ())),
            tuple(float(x) for x in sparse.get("values", ())),
        ),
        restricts=tuple(
            Restrict(str(r["namespace"]), tuple(str(t) for t in r.get("allow", ())))
            for r in raw.get("restricts", ())
        ),
        numeric_restricts=tuple(
            NumericRestrict(
                str(n["namespace"]),
                value_int=n.get("value_int"),
                value_float=n.get("value_float"),
                value_double=n.get("value_double"),
            )
            for n in raw.get("numeric_restricts", ())
        ),
        crowding_tag=raw.get("crowding_tag"),
    )


class LocalVectorSearch:
    """An in-process Vector Search index. ``path=None`` keeps it in memory only."""

    def __init__(
        self,
        path: Path | str | None = None,
        index: IndexKind = "flat",
        hnsw_m: int = 32,
        ef_search: int = 128,
    ) -> None:
        if index not in ("flat", "hnsw"):
            raise ValueError("index must be 'flat' or 'hnsw'")
        self.path = Path(path).expanduser() if path is not None else None
        self.index_kind: IndexKind = index
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self._points: dict[str, Datapoint] = {}
        self._dimensions: int | None = None
        self._dirty = False
        self._order: list[str] = []
        self._faiss: Any = None
        self._postings: dict[int, dict[str, float]] = {}
        if self.path is not None and (self.path / "datapoints.json").exists():
            self._load()

    # ------------------------------------------------------------------ writes
    def upsert(self, datapoints: Sequence[Datapoint]) -> None:
        for d in datapoints:
            if d.feature_vector:
                if self._dimensions is None:
                    self._dimensions = len(d.feature_vector)
                elif len(d.feature_vector) != self._dimensions:
                    raise InvalidDatapointError(
                        f"{d.datapoint_id}: {len(d.feature_vector)} dimensions, "
                        f"the index has {self._dimensions}"
                    )
            elif self._dimensions is not None:
                raise InvalidDatapointError(f"{d.datapoint_id}: this index needs dense vectors")
            old = self._points.get(d.datapoint_id)
            if old is not None:
                self._unpost(old)
            self._points[d.datapoint_id] = d
            for dim, value in zip(
                d.sparse_embedding.dimensions, d.sparse_embedding.values, strict=True
            ):
                self._postings.setdefault(dim, {})[d.datapoint_id] = value
        self._invalidate()

    def remove(self, datapoint_ids: Sequence[str]) -> None:
        for datapoint_id in datapoint_ids:
            old = self._points.pop(datapoint_id, None)
            if old is not None:
                self._unpost(old)
        self._invalidate()

    def _unpost(self, d: Datapoint) -> None:
        for dim in d.sparse_embedding.dimensions:
            posting = self._postings.get(dim)
            if posting is not None:
                posting.pop(d.datapoint_id, None)
                if not posting:
                    del self._postings[dim]

    def _invalidate(self) -> None:
        self._faiss = None
        self._dirty = True

    def __len__(self) -> int:
        return len(self._points)

    def get(self, datapoint_id: str) -> Datapoint | None:
        return self._points.get(datapoint_id)

    # ------------------------------------------------------------------ reads
    def _dense_index(self) -> Any:
        if self._faiss is None:
            self._order = [i for i, d in self._points.items() if d.feature_vector]
            dim = self._dimensions or 1
            if self.index_kind == "hnsw":
                index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexFlatIP(dim)
            if self._order:
                matrix = np.asarray(
                    [self._points[i].feature_vector for i in self._order], dtype=np.float32
                )
                index.add(matrix)
            self._faiss = index
        return self._faiss

    def _dense(self, query: Query, eligible: set[str], k: int) -> list[Neighbor]:
        if not query.feature_vector or not eligible or self._dimensions is None:
            return []
        if len(query.feature_vector) != self._dimensions:
            raise InvalidDatapointError(
                f"query has {len(query.feature_vector)} dimensions, the index has "
                f"{self._dimensions}"
            )
        index = self._dense_index()
        rows = np.asarray(
            [row for row, i in enumerate(self._order) if i in eligible], dtype=np.int64
        )
        if rows.size == 0:
            return []
        selector = faiss.IDSelectorBatch(rows)
        if self.index_kind == "hnsw":
            params = faiss.SearchParametersHNSW(sel=selector)
            params.efSearch = max(self.ef_search, k)
        else:
            params = faiss.SearchParameters(sel=selector)
        x = np.asarray([query.feature_vector], dtype=np.float32)
        scores, found = index.search(x, min(k, int(rows.size)), params=params)
        out = []
        for score, row in zip(scores[0], found[0], strict=True):
            if row < 0:
                continue
            d = self._points[self._order[int(row)]]
            out.append(Neighbor(d.datapoint_id, distance=float(score), crowding_tag=d.crowding_tag))
        return out

    def _sparse(self, query: Query, eligible: set[str], k: int) -> list[Neighbor]:
        scores: dict[str, float] = {}
        for dim, weight in zip(
            query.sparse_embedding.dimensions, query.sparse_embedding.values, strict=True
        ):
            for datapoint_id, value in self._postings.get(dim, {}).items():
                if datapoint_id in eligible:
                    scores[datapoint_id] = scores.get(datapoint_id, 0.0) + weight * value
        ranked = sorted(
            ((i, s) for i, s in scores.items() if s > 0), key=lambda kv: (-kv[1], kv[0])
        )[:k]
        return [
            Neighbor(i, sparse_distance=s, crowding_tag=self._points[i].crowding_tag)
            for i, s in ranked
        ]

    def _dense_score(self, query: Query, datapoint_id: str) -> float | None:
        d = self._points[datapoint_id]
        if not query.feature_vector or not d.feature_vector:
            return None
        return float(np.dot(query.feature_vector, d.feature_vector))

    def find_neighbors(self, queries: Sequence[Query]) -> list[list[Neighbor]]:
        return [self._find(q) for q in queries]

    def _find(self, query: Query) -> list[Neighbor]:
        eligible = {i for i, d in self._points.items() if matches(d, query)}
        k = max(query.approximate_neighbor_count or 0, query.neighbor_count * 4, MIN_CANDIDATES)
        mode = query.mode
        if mode == "dense":
            ranked = self._dense(query, eligible, k)
        elif mode == "sparse":
            ranked = self._sparse(query, eligible, k)
        else:
            alpha = float(query.rrf_alpha if query.rrf_alpha is not None else 1.0)
            fused = rrf_fuse(
                self._dense(query, eligible, k), self._sparse(query, eligible, k), alpha
            )
            ranked = [
                Neighbor(
                    i,
                    distance=self._dense_score(query, i),
                    sparse_distance=self._points[i].sparse_embedding.dot(query.sparse_embedding),
                    crowding_tag=self._points[i].crowding_tag,
                )
                for i, _ in fused
            ]
        return apply_crowding(
            ranked, query.per_crowding_attribute_neighbor_count, query.neighbor_count
        )

    # ------------------------------------------------------------------ files
    def flush(self) -> None:
        """Write the index to ``path`` (atomically) if it changed."""
        if self.path is None or not self._dirty:
            return
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        ids = list(self._points)
        dim = self._dimensions or 0
        matrix = np.asarray(
            [self._points[i].feature_vector or (0.0,) * dim for i in ids], dtype=np.float32
        ).reshape(len(ids), dim)
        meta = {
            "format": FORMAT_VERSION,
            "dimensions": self._dimensions,
            "datapoints": [datapoint_to_json(self._points[i]) for i in ids],
        }
        self._atomic(self.path / "vectors.npy", lambda f: np.save(f, matrix, allow_pickle=False))
        self._atomic(self.path / "datapoints.json", lambda f: f.write(json.dumps(meta).encode()))
        self._dirty = False

    @staticmethod
    def _atomic(target: Path, write: Any) -> None:
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        try:
            with os.fdopen(fd, "wb") as f:
                write(f)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _load(self) -> None:
        if self.path is None:  # pragma: no cover - only called with a path
            return
        meta = json.loads((self.path / "datapoints.json").read_text())
        if meta.get("format") != FORMAT_VERSION:
            raise InvalidDatapointError(f"{self.path}: unknown index format {meta.get('format')}")
        matrix = np.load(self.path / "vectors.npy", allow_pickle=False)
        raws = meta["datapoints"]
        if len(raws) != matrix.shape[0]:
            raise InvalidDatapointError(f"{self.path}: vectors and datapoints disagree")
        dim = meta.get("dimensions")
        self.upsert(
            [
                datapoint_from_json(raw, matrix[row].tolist() if dim else ())
                for row, raw in enumerate(raws)
            ]
        )
        self._dirty = False

    def close(self) -> None:
        self.flush()
