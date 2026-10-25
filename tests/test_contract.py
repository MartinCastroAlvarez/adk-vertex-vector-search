"""The VectorStore contract, run on LocalVectorSearch and on the Vertex adapter (fake client)."""

from __future__ import annotations

import pytest

from agentic_rag.retrieval.types import Datapoint
from agentic_rag.retrieval.types import NumericRestrict
from agentic_rag.retrieval.types import Query
from agentic_rag.retrieval.types import Restrict
from agentic_rag.retrieval.types import SparseVector
from tests.conftest import Backend


def dp(
    i: str,
    vec: tuple[float, ...],
    color: tuple[str, ...] = (),
    shape: tuple[str, ...] = (),
    price: int | None = None,
    sparse: dict[int, float] | None = None,
    tag: str | None = None,
) -> Datapoint:
    restricts = []
    if color:
        restricts.append(Restrict("color", color))
    if shape:
        restricts.append(Restrict("shape", shape))
    numeric = (NumericRestrict("price", value_int=price),) if price is not None else ()
    sp = SparseVector(tuple(sparse), tuple(sparse.values())) if sparse else SparseVector()
    return Datapoint(i, vec, sp, tuple(restricts), numeric, tag)


POINTS = [
    dp("a", (1.0, 0.0, 0.0, 0.0), ("red",), ("round",), 10, {1: 1.0}, "doc1"),
    dp("b", (0.9, 0.1, 0.0, 0.0), ("red", "blue"), ("square",), 20, {2: 1.0}, "doc1"),
    dp("c", (0.8, 0.2, 0.0, 0.0), ("blue",), ("round",), 30, {1: 0.5, 2: 0.5}, "doc1"),
    dp("d", (0.1, 1.0, 0.0, 0.0), ("green",), (), 40, {3: 2.0}, "doc2"),
    dp("e", (0.05, 0.0, 1.0, 0.0), (), ("round",), None, {1: 3.0}, "doc3"),
]
Q = (1.0, 0.0, 0.0, 0.0)


def ids(backend: Backend, query: Query) -> list[str]:
    return [n.datapoint_id for n in backend.store.find_neighbors([query])[0]]


@pytest.fixture
def loaded(backend: Backend) -> Backend:
    backend.store.upsert(POINTS)
    return backend


def test_dense_ranking_by_dot_product(loaded: Backend) -> None:
    assert ids(loaded, Query(Q, neighbor_count=3)) == ["a", "b", "c"]
    first = loaded.store.find_neighbors([Query(Q, neighbor_count=1)])[0][0]
    assert first.distance == pytest.approx(1.0)


def test_neighbor_count_and_batches(loaded: Backend) -> None:
    results = loaded.store.find_neighbors(
        [Query(Q, neighbor_count=2), Query((0, 1.0, 0, 0), neighbor_count=1)]
    )
    assert [[n.datapoint_id for n in r] for r in results] == [["a", "b"], ["d"]]


def test_allow_is_or_within_a_namespace(loaded: Backend) -> None:
    q = Query(Q, restricts=(Restrict("color", ("blue", "green")),), neighbor_count=10)
    assert ids(loaded, q) == ["b", "c", "d"]


def test_namespaces_are_anded(loaded: Backend) -> None:
    q = Query(
        Q,
        restricts=(Restrict("color", ("red", "blue")), Restrict("shape", ("round",))),
        neighbor_count=10,
    )
    assert ids(loaded, q) == ["a", "c"]


def test_deny_excludes_any_datapoint_with_the_token(loaded: Backend) -> None:
    q = Query(Q, restricts=(Restrict("color", deny=("blue",)),), neighbor_count=10)
    # b is red *and* blue: denied. e has no color at all: not denied.
    assert ids(loaded, q) == ["a", "d", "e"]


def test_allow_and_deny_together(loaded: Backend) -> None:
    q = Query(Q, restricts=(Restrict("color", ("red",), ("blue",)),), neighbor_count=10)
    assert ids(loaded, q) == ["a"]


def test_a_datapoint_without_the_namespace_never_matches_an_allow(loaded: Backend) -> None:
    q = Query(Q, restricts=(Restrict("shape", ("round", "square")),), neighbor_count=10)
    assert "d" not in ids(loaded, q)  # d has no shape


def test_an_empty_query_namespace_matches_everything(loaded: Backend) -> None:
    q = Query(Q, restricts=(Restrict("color"),), neighbor_count=10)
    assert sorted(ids(loaded, q)) == ["a", "b", "c", "d", "e"]


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("LESS", ["a"]),
        ("LESS_EQUAL", ["a", "b"]),
        ("EQUAL", ["b"]),
        ("GREATER_EQUAL", ["b", "c", "d"]),
        ("GREATER", ["c", "d"]),
        ("NOT_EQUAL", ["a", "c", "d"]),
    ],
)
def test_numeric_operators(loaded: Backend, op: str, expected: list[str]) -> None:
    q = Query(
        Q, numeric_restricts=(NumericRestrict("price", value_int=20, op=op),), neighbor_count=10
    )  # type: ignore[arg-type]
    assert ids(loaded, q) == expected  # e has no price: never matches a numeric filter


def test_numeric_restricts_are_anded(loaded: Backend) -> None:
    q = Query(
        Q,
        numeric_restricts=(
            NumericRestrict("price", value_int=10, op="GREATER"),
            NumericRestrict("price", value_int=40, op="LESS"),
        ),
        neighbor_count=10,
    )
    assert ids(loaded, q) == ["b", "c"]


def test_numeric_value_types_must_agree(loaded: Backend) -> None:
    q = Query(
        Q, numeric_restricts=(NumericRestrict("price", value_float=20.0, op="GREATER_EQUAL"),)
    )
    assert ids(loaded, q) == []


def test_token_and_numeric_filters_combine(loaded: Backend) -> None:
    q = Query(
        Q,
        restricts=(Restrict("shape", ("round",)),),
        numeric_restricts=(NumericRestrict("price", value_int=15, op="GREATER"),),
    )
    assert ids(loaded, q) == ["c"]


def test_sparse_only_is_a_dot_product(loaded: Backend) -> None:
    q = Query(sparse_embedding=SparseVector((1,), (1.0,)), neighbor_count=10)
    result = loaded.store.find_neighbors([q])[0]
    assert [n.datapoint_id for n in result] == [
        "e",
        "a",
        "c",
    ]  # 3.0, 1.0, 0.5; zero overlap omitted
    assert [n.sparse_distance for n in result] == pytest.approx([3.0, 1.0, 0.5])


@pytest.mark.parametrize(("alpha", "first"), [(1.0, "a"), (0.0, "e")])
def test_rrf_alpha_one_is_dense_and_zero_is_sparse(
    loaded: Backend, alpha: float, first: str
) -> None:
    q = Query(Q, SparseVector((1,), (1.0,)), neighbor_count=5, rrf_alpha=alpha)
    assert ids(loaded, q)[0] == first


def test_hybrid_rrf_rewards_agreement(loaded: Backend) -> None:
    # dense ranks a, b, c, d, e; sparse (dims 1 + 2) ranks e (3.0), then a, b, c (1.0 each).
    q = Query(Q, SparseVector((1, 2), (1.0, 1.0)), neighbor_count=5, rrf_alpha=0.5)
    # RRF, k=60: a = (1/61 + 1/62)/2 > b = (1/62 + 1/63)/2 > e = (1/65 + 1/61)/2 > c > d.
    assert ids(loaded, q) == ["a", "b", "e", "c", "d"]
    weighted = Query(Q, SparseVector((1, 2), (1.0, 1.0)), neighbor_count=5, rrf_alpha=0.1)
    assert ids(loaded, weighted)[0] == "e"  # mostly sparse
    result = loaded.store.find_neighbors([q])[0]
    assert all(n.distance is not None and n.sparse_distance for n in result[:4])


def test_hybrid_respects_filters(loaded: Backend) -> None:
    q = Query(
        Q,
        SparseVector((1,), (1.0,)),
        restricts=(Restrict("color", ("blue",)),),
        rrf_alpha=0.5,
    )
    assert ids(loaded, q) == ["c", "b"]


def test_crowding_limits_results_per_tag(loaded: Backend) -> None:
    q = Query(Q, neighbor_count=3, per_crowding_attribute_neighbor_count=1)
    result = loaded.store.find_neighbors([q])[0]
    assert [n.datapoint_id for n in result] == ["a", "d", "e"]
    assert [n.crowding_tag for n in result] == ["doc1", "doc2", "doc3"]


def test_upsert_replaces_by_id(loaded: Backend) -> None:
    loaded.store.upsert([dp("a", (0.0, 0.0, 0.0, 1.0), ("purple",), price=99)])
    assert ids(loaded, Query(Q, neighbor_count=2)) == ["b", "c"]
    purple = Query(Q, restricts=(Restrict("color", ("purple",)),))
    assert ids(loaded, purple) == ["a"]
    red = Query(Q, restricts=(Restrict("color", ("red",)),))
    assert ids(loaded, red) == ["b"]  # the old restricts are gone with the old datapoint


def test_remove_deletes_and_ignores_unknown_ids(loaded: Backend) -> None:
    loaded.store.remove(["a", "nope"])
    assert ids(loaded, Query(Q, neighbor_count=2)) == ["b", "c"]
    assert ids(loaded, Query(sparse_embedding=SparseVector((1,), (1.0,)))) == ["e", "c"]


def test_no_match_is_an_empty_list(loaded: Backend) -> None:
    q = Query(Q, restricts=(Restrict("color", ("ultraviolet",)),))
    assert loaded.store.find_neighbors([q]) == [[]]


def test_filtered_query_still_fills_neighbor_count(backend: Backend) -> None:
    """Filtering happens inside the search, not after it (no post-filter starvation)."""
    many = [dp(f"x{i}", (1.0, i / 1000, 0, 0), ("red",)) for i in range(60)]
    rare = [dp(f"y{i}", (0.0, 1.0, i / 1000, 0), ("blue",)) for i in range(3)]
    backend.store.upsert(many + rare)
    q = Query(Q, restricts=(Restrict("color", ("blue",)),), neighbor_count=3)
    assert sorted(ids(backend, q)) == ["y0", "y1", "y2"]


def test_the_retriever_answers_the_same_on_both(any_retriever: object) -> None:
    from agentic_rag.filters import SearchFilters
    from agentic_rag.retrieval.retriever import Retriever

    assert isinstance(any_retriever, Retriever)
    hits = any_retriever.search(
        "default port", SearchFilters(product="lumen", version="2.0"), top_k=3
    )
    assert hits[0].chunk.chunk_id == "lumen-2.0-configuration#network-settings.0"
    assert {h.chunk.version for h in hits} == {"2.0"}
