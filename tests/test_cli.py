"""The CLI, offline: ask, search, ingest, bench, demo, mermaid."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_rag.cli import main


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("RAG_PROVIDER", "stub")
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path))
    return tmp_path


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_ask_builds_the_index_on_first_use(capsys: pytest.CaptureFixture[str], _stub: Path) -> None:
    code, out, _ = run(capsys, "What is the default port in Lumen 2.0?")
    assert code == 0 and "7410" in out and "Sources:" in out
    assert (_stub / "hash-hash" / "chunks.json").exists()
    code, out, _ = run(capsys, "--json", "What is the capital of France?")
    payload = json.loads(out)
    assert payload["status"] == "refused" and payload["hops"] == 1


def test_search_ingest_and_bench(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--ingest")
    assert code == 0 and "ingested 23 documents" in out
    code, out, _ = run(capsys, "--ingest", "--json")
    assert json.loads(out)["removed"] == 0
    code, out, _ = run(capsys, "--search", "Lumen 2.0 default port")
    assert (
        out.startswith("filters: product=lumen, version=2.0")
        and "1. [lumen-2.0-configuration" in out
    )
    code, out, _ = run(capsys, "--search", "--json", "rollout abort")
    assert json.loads(out)["hits"][0]["id"].startswith("ferry-2.0-progressive-rollouts")
    code, out, _ = run(capsys, "--bench")
    assert "| hybrid+filters |" in out and "embedder hash-256" in out
    code, out, _ = run(capsys, "--bench", "--json")
    assert [s["config"] for s in json.loads(out)["scores"]] == [
        "dense",
        "sparse",
        "hybrid",
        "hybrid+filters",
    ]


def test_demo_and_mermaid(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "--demo", "--json")
    assert code == 0 and len(json.loads(out)) == 5
    code, out, _ = run(capsys, "--demo")
    assert "=== 3. multi-hop" in out
    code, out, _ = run(capsys, "--mermaid")
    assert out.startswith("flowchart LR") and "search_docs" in out and "citation_guard" in out
    assert "hop budget 3" in out and "LocalVectorSearch" in out


def test_errors(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, _, err = run(capsys)
    assert code == 2 and "ask a question" in err
    code, _, err = run(capsys, "--ingest", str(tmp_path / "nope"))
    assert code == 2 and "not a directory" in err
    code, _, err = run(capsys, "--max-hops", "99", "q")
    assert code == 2 and "max_hops" in err


def test_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from agentic_rag.cli import parser
    from agentic_rag.cli import settings_from

    args = parser().parse_args(
        ["--provider", "ollama", "--embedder", "ollama", "--max-hops", "2", "q"]
    )
    s = settings_from(args)
    assert (s.provider, s.model, s.embedder, s.embed_model, s.max_hops) == (
        "ollama",
        "qwen3:4b",
        "ollama",
        "nomic-embed-text",
        2,
    )
    args = parser().parse_args(["--provider", "gemini", "q"])
    assert settings_from(args).model == "gemini-3.5-flash"
