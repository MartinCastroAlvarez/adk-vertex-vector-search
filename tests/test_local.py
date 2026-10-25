"""LocalVectorSearch beyond the contract: files on disk, HNSW, validation."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import numpy as np
import pytest

from agentic_rag.retrieval.local import LocalVectorSearch
from agentic_rag.retrieval.types import Datapoint
from agentic_rag.retrieval.types import InvalidDatapointError
from agentic_rag.retrieval.types import NumericRestrict
from agentic_rag.retrieval.types import Query
from agentic_rag.retrieval.types import Restrict
from agentic_rag.retrieval.types import SparseVector

POINTS = [
    Datapoint(
        "a",
        (1.0, 0.0),
        SparseVector((5,), (2.0,)),
        (Restrict("product", ("lumen",)),),
        (NumericRestrict("released", value_int=20250120), NumericRestrict("w", value_float=0.5)),
        "doc-a",
    ),
    Datapoint("b", (0.0, 1.0), restricts=(Restrict("product", ("ferry",)),)),
]


def test_flush_and_reload_round_trip(tmp_path: Path) -> None:
    store = LocalVectorSearch(tmp_path / "idx")
    store.upsert(POINTS)
    store.close()
    meta = json.loads((tmp_path / "idx" / "datapoints.json").read_text())
    assert meta["format"] == 1 and meta["dimensions"] == 2
    assert stat.S_IMODE((tmp_path / "idx").stat().st_mode) == 0o700
    again = LocalVectorSearch(tmp_path / "idx")
    assert len(again) == 2
    assert again.get("a") == POINTS[0]
    q = Query((1.0, 0.0), restricts=(Restrict("product", ("lumen",)),))
    assert [n.datapoint_id for n in again.find_neighbors([q])[0]] == ["a"]


def test_flush_is_a_no_op_when_nothing_changed(tmp_path: Path) -> None:
    store = LocalVectorSearch(tmp_path / "idx")
    store.flush()
    assert not (tmp_path / "idx").exists()


def test_vectors_are_loaded_without_pickle(tmp_path: Path) -> None:
    store = LocalVectorSearch(tmp_path)
    store.upsert(POINTS)
    store.flush()
    np.save(
        tmp_path / "vectors.npy", np.array([object(), object()], dtype=object), allow_pickle=True
    )
    with pytest.raises(ValueError, match="allow_pickle"):
        LocalVectorSearch(tmp_path)


def test_unknown_format_and_mismatched_files_are_refused(tmp_path: Path) -> None:
    store = LocalVectorSearch(tmp_path)
    store.upsert(POINTS)
    store.flush()
    meta = json.loads((tmp_path / "datapoints.json").read_text())
    (tmp_path / "datapoints.json").write_text(json.dumps({**meta, "format": 99}))
    with pytest.raises(InvalidDatapointError, match="format"):
        LocalVectorSearch(tmp_path)
    (tmp_path / "datapoints.json").write_text(
        json.dumps({**meta, "datapoints": meta["datapoints"][:1]})
    )
    with pytest.raises(InvalidDatapointError, match="disagree"):
        LocalVectorSearch(tmp_path)


def test_hnsw_index_returns_the_same_neighbours() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(200, 16))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    points = [
        Datapoint(f"p{i}", tuple(v), restricts=(Restrict("parity", (str(i % 2),)),))
        for i, v in enumerate(vectors)
    ]
    flat, hnsw = LocalVectorSearch(), LocalVectorSearch(index="hnsw")
    flat.upsert(points)
    hnsw.upsert(points)
    q = Query(tuple(vectors[3]), restricts=(Restrict("parity", ("1",)),), neighbor_count=5)
    exact = [n.datapoint_id for n in flat.find_neighbors([q])[0]]
    approx = [n.datapoint_id for n in hnsw.find_neighbors([q])[0]]
    assert exact[0] == "p3"
    assert len(set(exact) & set(approx)) >= 4


def test_dimension_mismatches_are_refused() -> None:
    store = LocalVectorSearch()
    store.upsert(POINTS)
    with pytest.raises(InvalidDatapointError, match="dimensions"):
        store.upsert([Datapoint("c", (1.0, 0.0, 0.0))])
    with pytest.raises(InvalidDatapointError, match="dense"):
        store.upsert([Datapoint("c", sparse_embedding=SparseVector((1,), (1.0,)))])
    with pytest.raises(InvalidDatapointError, match="dimensions"):
        store.find_neighbors([Query((1.0, 0.0, 0.0))])


def test_a_sparse_only_index_works() -> None:
    store = LocalVectorSearch()
    store.upsert([Datapoint("s", sparse_embedding=SparseVector((1, 2), (1.0, 2.0)))])
    hits = store.find_neighbors([Query(sparse_embedding=SparseVector((2,), (1.0,)))])[0]
    assert [(n.datapoint_id, n.sparse_distance) for n in hits] == [("s", 2.0)]
    assert store.find_neighbors([Query((1.0,))]) == [[]]


def test_bad_index_kind() -> None:
    with pytest.raises(ValueError, match="flat"):
        LocalVectorSearch(index="ivf")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "build",
    [
        lambda: Datapoint("bad id!", (1.0,)),
        lambda: Datapoint("x"),
        lambda: Datapoint("x", (float("nan"),)),
        lambda: Datapoint("x", (1.0,), restricts=(Restrict("c", deny=("y",)),)),
        lambda: Datapoint(
            "x", (1.0,), numeric_restricts=(NumericRestrict("n", value_int=1, op="LESS"),)
        ),
        lambda: Datapoint(
            "x",
            (1.0,),
            numeric_restricts=(
                NumericRestrict("n", value_int=1),
                NumericRestrict("n", value_int=2),
            ),
        ),
        lambda: NumericRestrict("n"),
        lambda: NumericRestrict("n", value_int=1, value_float=1.0),
        lambda: NumericRestrict("n", value_int=1, op="ABOUT"),  # type: ignore[arg-type]
        lambda: Restrict("bad namespace"),
        lambda: SparseVector((1, 2), (1.0,)),
        lambda: SparseVector((1, 1), (1.0, 1.0)),
        lambda: SparseVector((-1,), (1.0,)),
        lambda: SparseVector((1,), (float("inf"),)),
        lambda: Query(),
        lambda: Query((1.0,), neighbor_count=0),
        lambda: Query((1.0,), rrf_alpha=1.5),
        lambda: Query((1.0,), numeric_restricts=(NumericRestrict("n", value_int=1),)),
        lambda: Query((1.0,), per_crowding_attribute_neighbor_count=0),
    ],
)
def test_invalid_datapoints_and_queries(build: object) -> None:
    with pytest.raises(InvalidDatapointError):
        build()  # type: ignore[operator]


def test_query_modes() -> None:
    sv = SparseVector((1,), (1.0,))
    assert Query((1.0,)).mode == "dense"
    assert Query(sparse_embedding=sv).mode == "sparse"
    assert Query((1.0,), sv).mode == "dense"
    assert Query((1.0,), sv, rrf_alpha=0.0).mode == "sparse"
    assert Query((1.0,), sv, rrf_alpha=0.4).mode == "hybrid"
    assert Query((1.0,), sv, rrf_alpha=1.0).mode == "dense"
    assert NumericRestrict("n", value_double=2.0).kind == "double"
