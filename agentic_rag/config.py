"""Settings for the agent and its retrieval stack, read from the environment (``RAG_*``)."""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal
from typing import cast
from urllib.parse import urlparse

Provider = Literal["ollama", "gemini", "stub"]
PROVIDERS: tuple[Provider, ...] = ("ollama", "gemini", "stub")
StoreKind = Literal["local", "vertex"]
STORES: tuple[StoreKind, ...] = ("local", "vertex")
EmbedderKind = Literal["fastembed", "ollama", "hash", "vertex"]
EMBEDDERS: tuple[EmbedderKind, ...] = ("fastembed", "ollama", "hash", "vertex")
IndexKind = Literal["flat", "hnsw"]

DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODELS: dict[EmbedderKind, str] = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "ollama": "nomic-embed-text",
    "hash": "hash",
    "vertex": "gemini-embedding-001",
}
DEFAULT_INDEX_DIR = Path("~/.cache/adk-vertex-vector-search")
APP_NAME = "agentic_rag"

_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_LOCATION = re.compile(r"^[a-z]+(-[a-z0-9]+)+$")
_NUMERIC_ID = re.compile(r"^[0-9]{1,32}$")


class ConfigError(ValueError):
    """A setting is missing or invalid."""


def _int(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _choice(env: Mapping[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = env.get(name, "").strip().lower() or default
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def is_loopback(url: str) -> bool:
    """True when ``url`` points at this machine (127.0.0.0/8, ::1 or localhost)."""
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class VertexSettings:
    """Where the deployed index lives. Only read for ``store == "vertex"``."""

    project: str = ""
    location: str = ""
    index_id: str = ""
    index_endpoint_id: str = ""
    deployed_index_id: str = ""
    public_endpoint_domain: str = ""

    def validate_project(self) -> None:
        if not _PROJECT.match(self.project):
            raise ConfigError(
                f"RAG_VERTEX_PROJECT {self.project!r} is not a project id "
                "(see docs/vertex-setup.md; Vertex AI is a paid Google Cloud service)"
            )
        if not _LOCATION.match(self.location):
            raise ConfigError(f"RAG_VERTEX_LOCATION {self.location!r} is not a region")

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("RAG_VERTEX_PROJECT", self.project),
                ("RAG_VERTEX_LOCATION", self.location),
                ("RAG_VERTEX_INDEX", self.index_id),
                ("RAG_VERTEX_INDEX_ENDPOINT", self.index_endpoint_id),
                ("RAG_VERTEX_DEPLOYED_INDEX", self.deployed_index_id),
                ("RAG_VERTEX_ENDPOINT_DOMAIN", self.public_endpoint_domain),
            )
            if not value
        ]
        if missing:
            raise ConfigError(
                "RAG_STORE=vertex needs "
                + ", ".join(missing)
                + " (see docs/vertex-setup.md; Vector Search is a paid Google Cloud service)"
            )
        self.validate_project()
        for name, value in (
            ("RAG_VERTEX_INDEX", self.index_id),
            ("RAG_VERTEX_INDEX_ENDPOINT", self.index_endpoint_id),
        ):
            if not _NUMERIC_ID.match(value):
                raise ConfigError(f"{name} is the numeric id at the end of the resource name")


@dataclass(frozen=True)
class Settings:
    """Everything the agent needs. Build it with :meth:`from_env` or directly."""

    provider: Provider = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    allow_remote_ollama: bool = False
    store: StoreKind = "local"
    index_dir: Path = field(default=DEFAULT_INDEX_DIR)
    index: IndexKind = "flat"
    embedder: EmbedderKind = "fastembed"
    embed_model: str = DEFAULT_EMBED_MODELS["fastembed"]
    embed_dimensions: int = 768
    max_hops: int = 3
    top_k: int = 5
    alpha: float = 0.5
    per_document: int = 2
    vertex: VertexSettings = field(default_factory=VertexSettings)

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ConfigError(f"provider must be one of {', '.join(PROVIDERS)}")
        if self.store not in STORES:
            raise ConfigError(f"store must be one of {', '.join(STORES)}")
        if self.embedder not in EMBEDDERS:
            raise ConfigError(f"embedder must be one of {', '.join(EMBEDDERS)}")
        if self.index not in ("flat", "hnsw"):
            raise ConfigError("index must be flat or hnsw")
        uses_ollama = self.provider == "ollama" or self.embedder == "ollama"
        if uses_ollama and not is_loopback(self.ollama_host) and not self.allow_remote_ollama:
            raise ConfigError(
                f"OLLAMA_HOST {self.ollama_host!r} is not loopback; set "
                "RAG_ALLOW_REMOTE_OLLAMA=1 to send questions and documents to another machine"
            )
        if not 1 <= self.max_hops <= 10:
            raise ConfigError("max_hops must be between 1 and 10")
        if not 1 <= self.top_k <= 20:
            raise ConfigError("top_k must be between 1 and 20")
        if not 0.0 <= self.alpha <= 1.0:
            raise ConfigError("alpha must be between 0 (sparse only) and 1 (dense only)")
        if not 1 <= self.per_document <= 20:
            raise ConfigError("per_document must be between 1 and 20")
        if self.store == "vertex":
            self.vertex.validate()
        if self.embedder == "vertex":
            self.vertex.validate_project()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read ``RAG_*`` variables (and ``OLLAMA_HOST``) from ``env`` or ``os.environ``."""
        env = os.environ if env is None else env
        provider = _choice(env, "RAG_PROVIDER", "ollama", PROVIDERS)
        default_model = DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_OLLAMA_MODEL
        # The offline stub needs no model download: it pairs with the hashing embedder.
        default_embedder = "hash" if provider == "stub" else "fastembed"
        embedder = cast(EmbedderKind, _choice(env, "RAG_EMBEDDER", default_embedder, EMBEDDERS))
        index_dir = env.get("RAG_INDEX_DIR", "").strip()
        return cls(
            provider=cast(Provider, provider),
            model=env.get("RAG_MODEL", "").strip() or default_model,
            ollama_host=env.get("OLLAMA_HOST", "").strip() or DEFAULT_OLLAMA_HOST,
            allow_remote_ollama=env.get("RAG_ALLOW_REMOTE_OLLAMA", "") == "1",
            store=cast(StoreKind, _choice(env, "RAG_STORE", "local", STORES)),
            index_dir=Path(index_dir).expanduser() if index_dir else DEFAULT_INDEX_DIR.expanduser(),
            index=cast(IndexKind, _choice(env, "RAG_INDEX", "flat", ("flat", "hnsw"))),
            embedder=embedder,
            embed_model=env.get("RAG_EMBED_MODEL", "").strip() or DEFAULT_EMBED_MODELS[embedder],
            embed_dimensions=_int(env, "RAG_EMBED_DIMENSIONS", 768, 8, 4096),
            max_hops=_int(env, "RAG_MAX_HOPS", 3, 1, 10),
            top_k=_int(env, "RAG_TOP_K", 5, 1, 20),
            alpha=_float(env, "RAG_ALPHA", 0.5),
            per_document=_int(env, "RAG_PER_DOCUMENT", 2, 1, 20),
            vertex=VertexSettings(
                project=env.get("RAG_VERTEX_PROJECT", "").strip(),
                location=env.get("RAG_VERTEX_LOCATION", "").strip(),
                index_id=env.get("RAG_VERTEX_INDEX", "").strip(),
                index_endpoint_id=env.get("RAG_VERTEX_INDEX_ENDPOINT", "").strip(),
                deployed_index_id=env.get("RAG_VERTEX_DEPLOYED_INDEX", "").strip(),
                public_endpoint_domain=env.get("RAG_VERTEX_ENDPOINT_DOMAIN", "").strip(),
            ),
        )

    @property
    def index_path(self) -> Path:
        """Where this embedder's local index lives: one directory per embedder model."""
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{self.embedder}-{self.embed_model}")
        return self.index_dir / safe
