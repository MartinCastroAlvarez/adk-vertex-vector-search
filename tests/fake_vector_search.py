"""In-process fakes of the two Vector Search GAPIC clients, speaking real ``aiplatform_v1`` protos.

``FakeIndexService`` takes ``upsert_datapoints`` / ``remove_datapoints`` requests and
``FakeMatchService`` answers ``find_neighbors`` from the same in-memory index. This is an
*independent* reference implementation of the documented semantics - written against the
protos, sharing no code with ``LocalVectorSearch`` - so the contract suite really compares
two implementations:

* token restricts: AND across namespaces, OR within; deny excludes; a query namespace
  without allow tokens matches all; a datapoint without the namespace doesn't match it;
* numeric restricts: every one must hold, on a datapoint that has that namespace with the
  same value type;
* dense = dot product, sparse = dot product, hybrid = RRF (k = 60) weighted by ``rrf.alpha``;
* crowding: at most ``per_crowding_attribute_neighbor_count`` per ``crowding_attribute``.

It also rejects what the service rejects: a wrong index or endpoint name, an ``op`` on a
stored numeric restrict, a dense vector of the wrong size.
"""

from __future__ import annotations

from typing import Any

from google.cloud import aiplatform_v1 as aip

INDEX = "projects/demo-project/locations/us-central1/indexes/123"
ENDPOINT = "projects/demo-project/locations/us-central1/indexEndpoints/456"
DEPLOYED = "docs_deployed"
DOMAIN = "1234.us-central1-567.vdb.vertexai.goog"
RRF_K = 60


class FakeVectorSearchError(ValueError):
    pass


class FakeVectorSearch:
    """Shared state of one fake index; ``index_service`` / ``match_service`` are the clients."""

    def __init__(self) -> None:
        self.datapoints: dict[str, aip.IndexDatapoint] = {}
        self.calls: list[tuple[str, Any]] = []
        self.dimensions: int | None = None
        self.index_service = FakeIndexService(self)
        self.match_service = FakeMatchService(self)

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]


class FakeIndexService:
    def __init__(self, state: FakeVectorSearch) -> None:
        self.state = state

    def upsert_datapoints(
        self, request: aip.UpsertDatapointsRequest
    ) -> aip.UpsertDatapointsResponse:
        assert isinstance(request, aip.UpsertDatapointsRequest)
        self.state.calls.append(("upsert_datapoints", request))
        if request.index != INDEX:
            raise FakeVectorSearchError(f"unknown index {request.index}")
        for d in request.datapoints:
            if any(n.op for n in d.numeric_restricts):
                raise FakeVectorSearchError("op is only for queries")
            if d.feature_vector:
                if self.state.dimensions is None:
                    self.state.dimensions = len(d.feature_vector)
                elif len(d.feature_vector) != self.state.dimensions:
                    raise FakeVectorSearchError("wrong number of dimensions")
            self.state.datapoints[d.datapoint_id] = aip.IndexDatapoint(d)
        return aip.UpsertDatapointsResponse()

    def remove_datapoints(
        self, request: aip.RemoveDatapointsRequest
    ) -> aip.RemoveDatapointsResponse:
        assert isinstance(request, aip.RemoveDatapointsRequest)
        self.state.calls.append(("remove_datapoints", request))
        if request.index != INDEX:
            raise FakeVectorSearchError(f"unknown index {request.index}")
        for datapoint_id in request.datapoint_ids:
            self.state.datapoints.pop(datapoint_id, None)
        return aip.RemoveDatapointsResponse()


def _numeric_value(n: Any) -> tuple[str, float] | None:
    for kind in ("value_int", "value_float", "value_double"):
        if kind in n:
            return kind, float(getattr(n, kind))
    return None


def _passes(d: aip.IndexDatapoint, q: aip.IndexDatapoint) -> bool:
    for r in q.restricts:
        tokens = {t for dr in d.restricts if dr.namespace == r.namespace for t in dr.allow_list}
        if set(r.deny_list) & tokens:
            return False
        if r.allow_list and not set(r.allow_list) & tokens:
            return False
    ops = aip.IndexDatapoint.NumericRestriction.Operator
    for n in q.numeric_restricts:
        want = _numeric_value(n)
        have = next(
            (_numeric_value(dn) for dn in d.numeric_restricts if dn.namespace == n.namespace), None
        )
        if want is None or have is None or have[0] != want[0]:
            return False
        v, x = have[1], want[1]
        ok = {
            ops.LESS: v < x,
            ops.LESS_EQUAL: v <= x,
            ops.EQUAL: v == x,
            ops.GREATER_EQUAL: v >= x,
            ops.GREATER: v > x,
            ops.NOT_EQUAL: v != x,
        }.get(n.op)
        if not ok:
            return False
    return True


def _sparse_dot(a: Any, b: Any) -> float:
    left = dict(zip(a.dimensions, a.values, strict=True))
    return sum(left.get(d, 0.0) * v for d, v in zip(b.dimensions, b.values, strict=True))


class FakeMatchService:
    def __init__(self, state: FakeVectorSearch) -> None:
        self.state = state

    def find_neighbors(self, request: aip.FindNeighborsRequest) -> aip.FindNeighborsResponse:
        assert isinstance(request, aip.FindNeighborsRequest)
        self.state.calls.append(("find_neighbors", request))
        if request.index_endpoint != ENDPOINT or request.deployed_index_id != DEPLOYED:
            raise FakeVectorSearchError("unknown endpoint or deployed index")
        response = aip.FindNeighborsResponse()
        for query in request.queries:
            response.nearest_neighbors.append(self._one(query))
        return response

    def _one(self, query: aip.FindNeighborsRequest.Query) -> Any:
        q = query.datapoint
        if (
            q.feature_vector
            and self.state.dimensions
            and len(q.feature_vector) != self.state.dimensions
        ):
            raise FakeVectorSearchError("query has the wrong number of dimensions")
        eligible = [d for d in self.state.datapoints.values() if _passes(d, q)]
        has_dense = bool(q.feature_vector)
        has_sparse = bool(q.sparse_embedding.dimensions)
        dense = sorted(
            (
                (
                    sum(a * b for a, b in zip(d.feature_vector, q.feature_vector, strict=True)),
                    d.datapoint_id,
                )
                for d in eligible
                if has_dense and d.feature_vector
            ),
            key=lambda t: (-t[0], t[1]),
        )
        sparse = sorted(
            (
                (s, d.datapoint_id)
                for d in eligible
                if has_sparse
                for s in [_sparse_dot(d.sparse_embedding, q.sparse_embedding)]
                if s > 0
            ),
            key=lambda t: (-t[0], t[1]),
        )
        alpha = query.rrf.alpha if "rrf" in query else None
        if has_dense and has_sparse and alpha is not None and 0 < alpha < 1:
            fused: dict[str, float] = {}
            for rank, (_, i) in enumerate(dense, 1):
                fused[i] = fused.get(i, 0.0) + alpha / (RRF_K + rank)
            for rank, (_, i) in enumerate(sparse, 1):
                fused[i] = fused.get(i, 0.0) + (1 - alpha) / (RRF_K + rank)
            order = [i for i, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
        elif has_sparse and (not has_dense or alpha == 0):
            order = [i for _, i in sparse]
        else:
            order = [i for _, i in dense]
        per = query.per_crowding_attribute_neighbor_count or None
        out = aip.FindNeighborsResponse.NearestNeighbors()
        seen: dict[str, int] = {}
        for datapoint_id in order:
            d = self.state.datapoints[datapoint_id]
            tag = d.crowding_tag.crowding_attribute
            if per and tag:
                if seen.get(tag, 0) >= per:
                    continue
                seen[tag] = seen.get(tag, 0) + 1
            neighbor = aip.FindNeighborsResponse.Neighbor(
                datapoint=aip.IndexDatapoint(
                    datapoint_id=datapoint_id,
                    crowding_tag=aip.IndexDatapoint.CrowdingTag(crowding_attribute=tag),
                )
            )
            if has_dense and d.feature_vector:
                neighbor.distance = sum(
                    a * b for a, b in zip(d.feature_vector, q.feature_vector, strict=True)
                )
            if has_sparse:
                neighbor.sparse_distance = _sparse_dot(d.sparse_embedding, q.sparse_embedding)
            out.neighbors.append(neighbor)
            if len(out.neighbors) == query.neighbor_count:
                break
        return out
