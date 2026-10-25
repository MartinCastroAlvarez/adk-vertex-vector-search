"""Settings from the environment, and the model factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.config import ConfigError
from agentic_rag.config import Settings
from agentic_rag.config import is_loopback
from agentic_rag.models import agent_model
from agentic_rag.models import ollama_model_id
from agentic_rag.offline import OfflineResearcher


def test_defaults_are_local() -> None:
    s = Settings.from_env({})
    assert (s.provider, s.model, s.store, s.embedder) == (
        "ollama",
        "qwen3:4b",
        "local",
        "fastembed",
    )
    assert s.embed_model == "BAAI/bge-small-en-v1.5" and s.max_hops == 3
    assert s.index_path.name == "fastembed-BAAI_bge-small-en-v1.5"
    stub = Settings.from_env({"RAG_PROVIDER": "stub"})
    assert stub.embedder == "hash"


def test_env_overrides() -> None:
    s = Settings.from_env(
        {
            "RAG_PROVIDER": "GEMINI",
            "RAG_EMBEDDER": "ollama",
            "RAG_INDEX_DIR": "/srv/rag",
            "RAG_INDEX": "hnsw",
            "RAG_MAX_HOPS": "5",
            "RAG_TOP_K": "8",
            "RAG_ALPHA": "0.25",
            "RAG_PER_DOCUMENT": "3",
        }
    )
    assert (s.provider, s.model, s.embedder, s.embed_model) == (
        "gemini",
        "gemini-3.5-flash",
        "ollama",
        "nomic-embed-text",
    )
    assert (s.index_dir, s.index, s.max_hops, s.top_k, s.alpha, s.per_document) == (
        Path("/srv/rag"),
        "hnsw",
        5,
        8,
        0.25,
        3,
    )


@pytest.mark.parametrize(
    ("env", "error"),
    [
        ({"RAG_PROVIDER": "openai"}, "RAG_PROVIDER"),
        ({"RAG_MAX_HOPS": "0"}, "between"),
        ({"RAG_MAX_HOPS": "many"}, "integer"),
        ({"RAG_ALPHA": "2"}, "alpha"),
        ({"RAG_ALPHA": "half"}, "number"),
        ({"OLLAMA_HOST": "http://10.0.0.5:11434"}, "loopback"),
        (
            {"RAG_PROVIDER": "stub", "RAG_EMBEDDER": "ollama", "OLLAMA_HOST": "http://example.com"},
            "loopback",
        ),
        ({"RAG_EMBEDDER": "vertex"}, "project"),
    ],
)
def test_invalid_settings(env: dict[str, str], error: str) -> None:
    with pytest.raises(ConfigError, match=error):
        Settings.from_env(env)


def test_remote_ollama_needs_an_explicit_opt_in() -> None:
    s = Settings.from_env({"OLLAMA_HOST": "http://10.0.0.5:11434", "RAG_ALLOW_REMOTE_OLLAMA": "1"})
    assert s.allow_remote_ollama
    assert is_loopback("http://localhost:1") and is_loopback("http://[::1]:1")
    assert not is_loopback("http://example.com")


@pytest.mark.parametrize(
    "kwargs",
    [{"store": "s3"}, {"embedder": "openai"}, {"index": "ivf"}, {"top_k": 0}, {"per_document": 0}],
)
def test_invalid_direct_settings(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        Settings(**kwargs)  # type: ignore[arg-type]


def test_models() -> None:
    assert ollama_model_id("qwen3:4b") == "ollama_chat/qwen3:4b"
    assert ollama_model_id("ollama/llama3.2") == "ollama/llama3.2"
    assert isinstance(agent_model(Settings(provider="stub", embedder="hash")), OfflineResearcher)
    lite = agent_model(Settings())
    assert lite.model == "ollama_chat/qwen3:4b"
    with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
        agent_model(Settings(provider="gemini", model="gemini-3.5-flash"), env={})
    gemini = agent_model(
        Settings(provider="gemini", model="gemini-3.5-flash"), env={"GOOGLE_API_KEY": "x"}
    )
    assert gemini.model == "gemini-3.5-flash"
