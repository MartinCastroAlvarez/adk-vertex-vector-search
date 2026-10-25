"""``VertexVectorSearchStore``: the same interface on a deployed Vertex AI Vector Search index.

**Paid.** Nothing in this repository - tests, demo, CI - calls it against Google Cloud; the
tests run it against fake clients. See ``docs/vertex-setup.md``.

It talks to the two GAPIC services of ``google-cloud-aiplatform`` (v1, GA) directly:

* ``IndexServiceClient`` on ``{location}-aiplatform.googleapis.com`` for **streaming
  updates**: ``upsert_datapoints(UpsertDatapointsRequest(index, datapoints))`` and
  ``remove_datapoints(RemoveDatapointsRequest(index, datapoint_ids))``. The index must be
  created with ``index_update_method="STREAM_UPDATE"``.
* ``MatchServiceClient`` on the index endpoint's **public endpoint domain**
  (``<n>.<region>-<project-number>.vdb.vertexai.goog``) for
  ``find_neighbors(FindNeighborsRequest(index_endpoint, deployed_index_id, queries))``.
  Each query carries an ``IndexDatapoint`` with the dense ``feature_vector``, the
  ``sparse_embedding``, the token ``restricts`` and ``numeric_restricts``, plus
  ``neighbor_count``, ``per_crowding_attribute_neighbor_count``,
  ``approximate_neighbor_count`` and, for a hybrid query, ``rrf.alpha``.

This is what ``MatchingEngineIndexEndpoint.find_neighbors`` builds for a public endpoint,
without its constructor's ``GET`` of the endpoint resource: constructing this class makes
no network call and loads no credentials; the clients are created on first use, from
Application Default Credentials, unless you pass your own.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from agentic_rag.retrieval.types import Datapoint
from agentic_rag.retrieval.types import Neighbor
from agentic_rag.retrieval.types import Query

#: ``UpsertDatapoints`` takes a batch; keep requests small and predictable.
UPSERT_BATCH = 100
_INDEX = re.compile(r"^projects/[^/]+/locations/[a-z0-9-]+/indexes/[0-9]+$")
_ENDPOINT = re.compile(r"^projects/[^/]+/locations/[a-z0-9-]+/indexEndpoints/[0-9]+$")
_DEPLOYED = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_DOMAIN = re.compile(r"^[a-z0-9.-]+\.vdb\.vertexai\.goog$")


class VertexConfigError(ValueError):
    """A Vector Search resource name is malformed."""


def index_name(project: str, location: str, index_id: str) -> str:
    return (
        index_id
        if "/" in index_id
        else f"projects/{project}/locations/{location}/indexes/{index_id}"
    )


def endpoint_name(project: str, location: str, endpoint_id: str) -> str:
    if "/" in endpoint_id:
        return endpoint_id
    return f"projects/{project}/locations/{location}/indexEndpoints/{endpoint_id}"


def to_proto(d: Datapoint | Query, aip: Any) -> Any:
    """A :class:`Datapoint` or a query's vector + filters as ``aiplatform_v1.IndexDatapoint``."""
    proto = aip.IndexDatapoint(
        datapoint_id=d.datapoint_id if isinstance(d, Datapoint) else "",
        feature_vector=list(d.feature_vector),
        restricts=[
            aip.IndexDatapoint.Restriction(
                namespace=r.namespace, allow_list=list(r.allow), deny_list=list(r.deny)
            )
            for r in d.restricts
        ],
        numeric_restricts=[
            aip.IndexDatapoint.NumericRestriction(
                namespace=n.namespace,
                **{
                    k: v
                    for k, v in (
                        ("value_int", n.value_int),
                        ("value_float", n.value_float),
                        ("value_double", n.value_double),
                    )
                    if v is not None
                },
                **({"op": n.op} if n.op else {}),
            )
            for n in d.numeric_restricts
        ],
    )
    if d.sparse_embedding:
        proto.sparse_embedding = aip.IndexDatapoint.SparseEmbedding(
            values=list(d.sparse_embedding.values),
            dimensions=list(d.sparse_embedding.dimensions),
        )
    if isinstance(d, Datapoint) and d.crowding_tag:
        proto.crowding_tag = aip.IndexDatapoint.CrowdingTag(crowding_attribute=d.crowding_tag)
    return proto


def to_query_proto(q: Query, aip: Any) -> Any:
    query = aip.FindNeighborsRequest.Query(
        datapoint=to_proto(q, aip), neighbor_count=q.neighbor_count
    )
    if q.per_crowding_attribute_neighbor_count is not None:
        query.per_crowding_attribute_neighbor_count = q.per_crowding_attribute_neighbor_count
    if q.approximate_neighbor_count is not None:
        query.approximate_neighbor_count = q.approximate_neighbor_count
    # RRF only for a true hybrid query (dense + sparse); the SDK sends it the same way.
    if q.rrf_alpha is not None and q.feature_vector and q.sparse_embedding:
        query.rrf = aip.FindNeighborsRequest.Query.RRF(alpha=q.rrf_alpha)
    return query


class VertexVectorSearchStore:
    """A deployed Vector Search index behind :class:`~agentic_rag.retrieval.base.VectorStore`."""

    def __init__(
        self,
        *,
        project: str,
        location: str,
        index_id: str,
        index_endpoint_id: str,
        deployed_index_id: str,
        public_endpoint_domain: str,
        index_client: Any = None,
        match_client: Any = None,
    ) -> None:
        self.index = index_name(project, location, index_id)
        self.index_endpoint = endpoint_name(project, location, index_endpoint_id)
        self.deployed_index_id = deployed_index_id
        self.location = location
        self.public_endpoint_domain = public_endpoint_domain
        if not _INDEX.match(self.index):
            raise VertexConfigError(f"not an index resource name: {self.index!r}")
        if not _ENDPOINT.match(self.index_endpoint):
            raise VertexConfigError(f"not an index endpoint resource name: {self.index_endpoint!r}")
        if not _DEPLOYED.match(deployed_index_id):
            raise VertexConfigError(f"not a deployed index id: {deployed_index_id!r}")
        if not _DOMAIN.match(public_endpoint_domain):
            raise VertexConfigError(
                "public_endpoint_domain is the index endpoint's publicEndpointDomainName, "
                "e.g. 1234.us-central1-567.vdb.vertexai.goog"
            )
        self._index_client = index_client
        self._match_client = match_client

    @staticmethod
    def _aip() -> Any:
        try:
            from google.cloud import aiplatform_v1
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                'RAG_STORE=vertex needs pip install "adk-vertex-vector-search[vertex]"'
            ) from exc
        return aiplatform_v1

    def _index_service(self) -> Any:
        if self._index_client is None:  # pragma: no cover - would reach Google Cloud
            aip = self._aip()
            self._index_client = aip.IndexServiceClient(
                client_options={"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
            )
        return self._index_client

    def _match_service(self) -> Any:
        if self._match_client is None:  # pragma: no cover - would reach Google Cloud
            aip = self._aip()
            self._match_client = aip.MatchServiceClient(
                client_options={"api_endpoint": self.public_endpoint_domain}
            )
        return self._match_client

    def upsert(self, datapoints: Sequence[Datapoint]) -> None:
        aip = self._aip()
        client = self._index_service()
        items = list(datapoints)
        for start in range(0, len(items), UPSERT_BATCH):
            batch = items[start : start + UPSERT_BATCH]
            client.upsert_datapoints(
                request=aip.UpsertDatapointsRequest(
                    index=self.index, datapoints=[to_proto(d, aip) for d in batch]
                )
            )

    def remove(self, datapoint_ids: Sequence[str]) -> None:
        ids = list(datapoint_ids)
        if not ids:
            return
        aip = self._aip()
        self._index_service().remove_datapoints(
            request=aip.RemoveDatapointsRequest(index=self.index, datapoint_ids=ids)
        )

    def find_neighbors(self, queries: Sequence[Query]) -> list[list[Neighbor]]:
        if not queries:
            return []
        aip = self._aip()
        request = aip.FindNeighborsRequest(
            index_endpoint=self.index_endpoint,
            deployed_index_id=self.deployed_index_id,
            queries=[to_query_proto(q, aip) for q in queries],
            return_full_datapoint=False,
        )
        response = self._match_service().find_neighbors(request=request)
        return [
            [
                Neighbor(
                    datapoint_id=n.datapoint.datapoint_id,
                    distance=n.distance if n.distance else None,
                    sparse_distance=n.sparse_distance if n.sparse_distance else None,
                    crowding_tag=n.datapoint.crowding_tag.crowding_attribute or None,
                )
                for n in result.neighbors
            ]
            for result in response.nearest_neighbors
        ]

    def close(self) -> None:
        for client in (self._index_client, self._match_client):
            transport = getattr(client, "transport", None)
            close = getattr(transport, "close", None)
            if callable(close):
                close()
