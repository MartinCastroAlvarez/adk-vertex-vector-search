"""Ingestion, the factory, the retriever and the benchmark on the bundled corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.bench import evaluate
from agentic_rag.bench import load_questions
from agentic_rag.bench import markdown
from agentic_rag.config import Settings
from agentic_rag.filters import SearchFilters
from agentic_rag.retrieval.catalog import ChunkCatalog
from agentic_rag.retrieval.corpus import load_corpus
from agentic_rag.retrieval.factory import build_embedder
from agentic_rag.retrieval.factory import open_retriever
from agentic_rag.retrieval.ingest import to_datapoint
from agentic_rag.retrieval.retriever import Retriever
from agentic_rag.retrieval.sparse import BM25Encoder
from agentic_rag.retrieval.types import NumericRestrict
from agentic_rag.retrieval.types import Restrict


def test_chunks_are_upserted_with_restricts_numeric_filters_and_crowding(
    retriever: Retriever,
) -> None:
    chunk = retriever.catalog.get("ferry-1.5-canary-deploys#starting-a-canary.0")
    assert chunk is not None
    d = to_datapoint(chunk, (1.0,), BM25Encoder().fit([chunk.text]))
    assert d.restricts == (
        Restrict("product", ("ferry",)),
        Restrict("version", ("1.5",)),
        Restrict("major", ("1",)),
        Restrict("doc_type", ("guide",)),
    )
    assert d.numeric_restricts == (
        NumericRestrict("version_num", value_int=105),
        NumericRestrict("released", value_int=20250515),
    )
    assert d.crowding_tag == "ferry-1.5-canary-deploys" and d.sparse_embedding


def test_filters_change_what_comes_back(retriever: Retriever) -> None:
    unfiltered = retriever.search("default port", top_k=6)
    assert {h.chunk.version for h in unfiltered} > {"2.0"}
    newer = retriever.search(
        "default port", SearchFilters(product="lumen", since_version="2.0"), top_k=6
    )
    assert {h.chunk.version for h in newer} <= {"2.0", "3.0"}
    dated = retriever.search(
        "rollout", SearchFilters(released_after=__import__("datetime").date(2026, 3, 1))
    )
    assert {h.chunk.product + h.chunk.version for h in dated} == {"ferry2.0"}
    not_14 = retriever.search(
        "port", SearchFilters(product="lumen", exclude_version="1.4"), top_k=8
    )
    assert "1.4" not in {h.chunk.version for h in not_14}


def test_crowding_spreads_results_over_documents(retriever: Retriever) -> None:
    hits = retriever.search("raft group election heartbeat", top_k=6)
    docs = [h.chunk.doc_id for h in hits]
    assert max(docs.count(d) for d in docs) <= 2


@pytest.mark.parametrize("mode", ["hybrid", "dense", "sparse"])
def test_modes(retriever: Retriever, mode: str) -> None:
    hits = retriever.search(
        "progressive rollout automatic abort", SearchFilters(product="ferry"), mode=mode
    )  # type: ignore[arg-type]
    assert hits[0].chunk.chunk_id == "ferry-2.0-progressive-rollouts#automatic-abort.0"
    query = retriever.queries[-1]
    assert bool(query.feature_vector) == (mode != "sparse")
    assert bool(query.sparse_embedding) == (mode != "dense")
    assert (query.rrf_alpha is not None) == (mode == "hybrid")


def test_a_stopword_only_sparse_query_falls_back_to_dense(retriever: Retriever) -> None:
    retriever.search("what is the", mode="sparse")
    assert retriever.queries[-1].feature_vector


def test_stale_catalog_entries_are_skipped(retriever: Retriever) -> None:
    gone = retriever.catalog.get("lumen-1.4-operations#backups.0")
    assert gone is not None
    retriever.catalog.remove([gone.chunk_id])
    try:
        hits = retriever.search("lumen 1.4 backup restore", top_k=10)
        assert gone.chunk_id not in {h.chunk.chunk_id for h in hits}
    finally:
        retriever.catalog.add([gone])


def test_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        Retriever(None, ChunkCatalog(), None, BM25Encoder(), alpha=2)  # type: ignore[arg-type]


def test_open_retriever_builds_reuses_and_rebuilds(tmp_path: Path) -> None:
    settings = Settings(provider="stub", embedder="hash", embed_model="hash", index_dir=tmp_path)
    first = open_retriever(settings)
    first.close()
    directory = settings.index_path
    stamp = (directory / "datapoints.json").stat().st_mtime_ns
    second = open_retriever(settings)
    assert (directory / "datapoints.json").stat().st_mtime_ns == stamp  # reused, not rebuilt
    assert len(second.catalog) == len(first.catalog) > 50
    # A catalog written by another embedder is rebuilt.
    raw = json.loads((directory / "chunks.json").read_text())
    (directory / "chunks.json").write_text(json.dumps({**raw, "embedder": "other"}))
    third = open_retriever(settings)
    assert third.catalog.embedder == "hash-256"
    with pytest.raises(ValueError, match="format"):
        (directory / "chunks.json").write_text(json.dumps({**raw, "format": 2}))
        ChunkCatalog.load(directory / "chunks.json")


def test_build_embedder_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agentic_rag.retrieval.embeddings as emb

    class Fake:
        def __init__(self, model_name: str, cache_dir: str) -> None:
            self.args = (model_name, cache_dir)

    import fastembed

    monkeypatch.setattr(fastembed, "TextEmbedding", Fake)
    e = build_embedder(Settings(index_dir=tmp_path))
    assert isinstance(e, emb.FastEmbedEmbedder)
    assert e._impl.args == ("BAAI/bge-small-en-v1.5", str(tmp_path / "models"))
    o = build_embedder(Settings(embedder="ollama", embed_model="nomic-embed-text"))
    assert isinstance(o, emb.OllamaEmbedder)


def test_benchmark_on_the_hashing_embedder(retriever: Retriever) -> None:
    questions = load_questions()
    assert len(questions) == 50
    known = {c.section_key for c in retriever.catalog.chunks()}
    assert all(q.relevant <= known for q in questions), "every label names a real section"
    scores = {s.config: s for s in evaluate(retriever, questions)}
    assert retriever.per_document == 2  # restored after the run
    assert scores["hybrid+filters"].recall[5] >= 0.9
    assert scores["hybrid+filters"].mrr > scores["hybrid"].mrr
    assert scores["hybrid+filters"].mrr > scores["dense"].mrr
    table = markdown(list(scores.values()), "t")
    assert "| sparse |" in table and scores["dense"].as_dict()["config"] == "dense"


def test_corpus_has_two_products_three_versions_each() -> None:
    docs = load_corpus()
    versions = {(d.product, d.version) for d in docs}
    assert len(versions) == 6
