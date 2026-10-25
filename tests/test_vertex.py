"""VertexVectorSearchStore: the exact requests it sends, against fake GAPIC clients."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
from google.cloud import aiplatform_v1 as aip

from agentic_rag.config import ConfigError
from agentic_rag.config import Settings
from agentic_rag.config import VertexSettings
from agentic_rag.retrieval.factory import build_store
from agentic_rag.retrieval.factory import open_retriever
from agentic_rag.retrieval.types import Datapoint
from agentic_rag.retrieval.types import NumericRestrict
from agentic_rag.retrieval.types import Query
from agentic_rag.retrieval.types import Restrict
from agentic_rag.retrieval.types import SparseVector
from agentic_rag.retrieval.vertex import VertexConfigError
from agentic_rag.retrieval.vertex import VertexVectorSearchStore
from tests.conftest import Backend
from tests.conftest import make_store
from tests.fake_vector_search import DEPLOYED
from tests.fake_vector_search import DOMAIN
from tests.fake_vector_search import ENDPOINT
from tests.fake_vector_search import INDEX


@pytest.fixture
def vertex() -> Backend:
    return make_store("vertex")


def requests(backend: Backend, name: str) -> list[Any]:
    assert backend.fake is not None
    return [r for n, r in backend.fake.calls if n == name]


POINT = Datapoint(
    "lumen-2.0-configuration#network-settings.0",
    (0.6, 0.8),
    SparseVector((7, 2**62), (1.5, 0.25)),
    (Restrict("product", ("lumen",)), Restrict("version", ("2.0",))),
    (NumericRestrict("released", value_int=20250120), NumericRestrict("score", value_double=0.5)),
    "lumen-2.0-configuration",
)


def test_upsert_sends_index_datapoints_with_everything(vertex: Backend) -> None:
    vertex.store.upsert([POINT])
    (request,) = requests(vertex, "upsert_datapoints")
    assert request.index == INDEX
    (d,) = request.datapoints
    assert d.datapoint_id == POINT.datapoint_id
    assert list(d.feature_vector) == pytest.approx([0.6, 0.8])
    assert list(d.sparse_embedding.dimensions) == [7, 2**62]  # int64 dimensions
    assert list(d.sparse_embedding.values) == pytest.approx([1.5, 0.25])
    assert [(r.namespace, list(r.allow_list)) for r in d.restricts] == [
        ("product", ["lumen"]),
        ("version", ["2.0"]),
    ]
    assert d.numeric_restricts[0].value_int == 20250120
    assert d.numeric_restricts[1].value_double == 0.5
    assert all(not n.op for n in d.numeric_restricts)  # op is for queries only
    assert d.crowding_tag.crowding_attribute == "lumen-2.0-configuration"


def test_upserts_are_batched(vertex: Backend) -> None:
    points = [Datapoint(f"p{i}", (1.0, 0.0)) for i in range(250)]
    vertex.store.upsert(points)
    sizes = [len(r.datapoints) for r in requests(vertex, "upsert_datapoints")]
    assert sizes == [100, 100, 50]


def test_remove_sends_ids_and_skips_empty_calls(vertex: Backend) -> None:
    vertex.store.remove([])
    assert requests(vertex, "remove_datapoints") == []
    vertex.store.remove(["a", "b"])
    (request,) = requests(vertex, "remove_datapoints")
    assert request.index == INDEX and list(request.datapoint_ids) == ["a", "b"]


def test_a_hybrid_query_is_one_find_neighbors_request(vertex: Backend) -> None:
    vertex.store.upsert([POINT])
    query = Query(
        (0.6, 0.8),
        SparseVector((7,), (1.0,)),
        restricts=(Restrict("product", ("lumen",)), Restrict("major", deny=("1",))),
        numeric_restricts=(NumericRestrict("released", value_int=20250101, op="GREATER_EQUAL"),),
        neighbor_count=4,
        per_crowding_attribute_neighbor_count=2,
        approximate_neighbor_count=40,
        rrf_alpha=0.3,
    )
    (result,) = vertex.store.find_neighbors([query])
    (request,) = requests(vertex, "find_neighbors")
    assert request.index_endpoint == ENDPOINT
    assert request.deployed_index_id == DEPLOYED
    assert request.return_full_datapoint is False
    (q,) = request.queries
    assert q.neighbor_count == 4
    assert q.per_crowding_attribute_neighbor_count == 2
    assert q.approximate_neighbor_count == 40
    assert q.rrf.alpha == pytest.approx(0.3)
    assert list(q.datapoint.feature_vector) == pytest.approx([0.6, 0.8])
    assert list(q.datapoint.sparse_embedding.dimensions) == [7]
    assert [
        (r.namespace, list(r.allow_list), list(r.deny_list)) for r in q.datapoint.restricts
    ] == [
        ("product", ["lumen"], []),
        ("major", [], ["1"]),
    ]
    numeric = q.datapoint.numeric_restricts[0]
    assert numeric.namespace == "released" and numeric.value_int == 20250101
    assert numeric.op == aip.IndexDatapoint.NumericRestriction.Operator.GREATER_EQUAL
    assert [n.datapoint_id for n in result] == [POINT.datapoint_id]
    assert result[0].crowding_tag == "lumen-2.0-configuration"
    assert result[0].distance == pytest.approx(1.0)
    assert result[0].sparse_distance == pytest.approx(1.5)


def test_rrf_is_sent_only_for_true_hybrid_queries(vertex: Backend) -> None:
    vertex.store.upsert([POINT])
    vertex.store.find_neighbors(
        [
            Query((0.6, 0.8)),
            Query(sparse_embedding=SparseVector((7,), (1.0,))),
            Query((0.6, 0.8), SparseVector((7,), (1.0,))),
        ]
    )
    (request,) = requests(vertex, "find_neighbors")
    assert ["rrf" in q for q in request.queries] == [False, False, False]
    assert not request.queries[0].datapoint.sparse_embedding.dimensions
    assert not request.queries[1].datapoint.feature_vector


def test_no_queries_means_no_request(vertex: Backend) -> None:
    assert vertex.store.find_neighbors([]) == []
    assert requests(vertex, "find_neighbors") == []


def test_full_resource_names_are_accepted() -> None:
    store = VertexVectorSearchStore(
        project="ignored-project",
        location="us-central1",
        index_id=INDEX,
        index_endpoint_id=ENDPOINT,
        deployed_index_id=DEPLOYED,
        public_endpoint_domain=DOMAIN,
        index_client=object(),
        match_client=object(),
    )
    assert store.index == INDEX and store.index_endpoint == ENDPOINT
    store.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("index_id", "projects/p/locations/us-central1/indexes/../../x"),
        ("index_endpoint_id", "not/an/endpoint"),
        ("deployed_index_id", "bad id; drop"),
        ("public_endpoint_domain", "evil.example.com"),
    ],
)
def test_malformed_resource_names_are_refused(field: str, value: str) -> None:
    kwargs: dict[str, Any] = {
        "project": "demo-project",
        "location": "us-central1",
        "index_id": "123",
        "index_endpoint_id": "456",
        "deployed_index_id": DEPLOYED,
        "public_endpoint_domain": DOMAIN,
        field: value,
    }
    with pytest.raises(VertexConfigError):
        VertexVectorSearchStore(**kwargs)


def test_settings_build_a_vertex_store_without_any_network_call() -> None:
    settings = Settings(
        provider="stub",
        embedder="hash",
        store="vertex",
        vertex=VertexSettings("demo-project", "us-central1", "123", "456", DEPLOYED, DOMAIN),
    )
    store = build_store(settings)
    assert isinstance(store, VertexVectorSearchStore)
    assert store._index_client is None and store._match_client is None  # built on first use


def test_vertex_settings_are_validated() -> None:
    with pytest.raises(ConfigError, match="RAG_VERTEX_PROJECT"):
        Settings(store="vertex", embedder="hash")
    with pytest.raises(ConfigError, match="numeric id"):
        Settings(
            store="vertex",
            embedder="hash",
            vertex=VertexSettings("demo-project", "us-central1", "idx", "456", DEPLOYED, DOMAIN),
        )
    with pytest.raises(ConfigError, match="not a region"):
        Settings(
            store="vertex",
            embedder="hash",
            vertex=VertexSettings("demo-project", "Mars", "1", "456", DEPLOYED, DOMAIN),
        )


def test_the_agent_never_builds_a_vertex_index_implicitly(tmp_path: Any) -> None:
    settings = Settings(
        provider="stub",
        embedder="hash",
        store="vertex",
        index_dir=tmp_path,
        vertex=VertexSettings("demo-project", "us-central1", "123", "456", DEPLOYED, DOMAIN),
    )
    with pytest.raises(ConfigError, match="--ingest"):
        open_retriever(settings, store=make_store("vertex").store)


def test_ingest_then_retrieve_through_the_vertex_adapter(tmp_path: Any) -> None:
    from agentic_rag.filters import SearchFilters
    from agentic_rag.retrieval.factory import ingest_corpus

    backend = make_store("vertex")
    settings = Settings(
        provider="stub",
        embedder="hash",
        embed_model="hash",
        store="vertex",
        index_dir=tmp_path,
        vertex=VertexSettings("demo-project", "us-central1", "123", "456", DEPLOYED, DOMAIN),
    )
    report = ingest_corpus(settings, store=backend.store)
    assert report.upserted == report.chunks > 50
    assert (tmp_path / "vertex" / "chunks.json").exists()
    retriever = open_retriever(settings, store=backend.store)
    hits = retriever.search("rollout abort threshold", SearchFilters(product="ferry"), top_k=3)
    assert hits[0].chunk.chunk_id == "ferry-2.0-progressive-rollouts#automatic-abort.0"
    # Re-ingesting a smaller corpus removes the chunks that disappeared.
    from agentic_rag.retrieval.corpus import bundled_corpus

    small = tmp_path / "small"
    small.mkdir()
    doc = bundled_corpus() / "ferry" / "ferry-1.0-rollbacks.md"
    (small / doc.name).write_text(doc.read_text())
    total = report.chunks
    report = ingest_corpus(settings, small, store=backend.store)
    assert report.chunks == 2 and report.removed == total - 2
    assert backend.fake is not None and len(backend.fake.datapoints) == 2
    removed = requests(backend, "remove_datapoints")
    assert removed and "lumen-2.0-configuration#network-settings.0" in removed[-1].datapoint_ids


def test_import_loads_no_cloud_sdk() -> None:
    code = (
        "import sys, agentic_rag, agentic_rag.agent, agentic_rag.cli;"
        "bad = [m for m in sys.modules if m.startswith(('google.cloud.aiplatform', 'vertexai'))"
        " or m == 'agentic_rag.retrieval.vertex'];"
        "print(bad); raise SystemExit(1 if bad else 0)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"RAG_PROVIDER": "stub", "PATH": ""},
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
