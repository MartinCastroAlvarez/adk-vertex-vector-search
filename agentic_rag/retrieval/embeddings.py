"""Dense text embeddings: one small interface, four implementations.

=================  ======================================  ==========================
``RAG_EMBEDDER``   Class                                   Cost
=================  ======================================  ==========================
``fastembed``      :class:`FastEmbedEmbedder` (ONNX, CPU)  free, local (default)
``ollama``         :class:`OllamaEmbedder` (``/api/embed``) free, local
``hash``           :class:`HashEmbedder`                   free, offline, deterministic
``vertex``         :class:`VertexEmbedder` (Gemini/Vertex) paid, your project
=================  ======================================  ==========================

Retrieval models embed questions and passages differently, so the interface has two
methods: :meth:`Embedder.embed_documents` and :meth:`Embedder.embed_query` (Vertex task
types ``RETRIEVAL_DOCUMENT`` / ``RETRIEVAL_QUERY``; bge's query prefix). Every vector is
L2-normalised, so the store's dot product is cosine similarity.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from agentic_rag.retrieval.sparse import tokenize

Vector = tuple[float, ...]


@runtime_checkable
class Embedder(Protocol):
    """Turns text into unit-length dense vectors of a fixed size."""

    @property
    def dimensions(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


def normalise(vector: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vector))
    return tuple(float(x) / norm for x in vector) if norm else tuple(float(x) for x in vector)


class HashEmbedder:
    """A deterministic bag-of-words + character-trigram embedding (the "hashing trick").

    Not semantic - "rollback" and "revert" share nothing - but stable across machines and
    Python versions, so tests and CI get the same vectors everywhere without a model.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if not 8 <= dimensions <= 4096:
            raise ValueError("dimensions must be between 8 and 4096")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"hash-{self._dimensions}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = tokenize(text)
        grams = [w[i : i + 3] for w in (f"#{w}#" for w in words) for i in range(len(w) - 2)]
        return [(f"w:{w}", 1.0) for w in words] + [(f"g:{g}", 0.5) for g in grams]

    def _embed(self, text: str) -> Vector:
        vector = [0.0] * self._dimensions
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * weight
        return normalise(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)


class FastEmbedEmbedder:
    """`fastembed <https://github.com/qdrant/fastembed>`_ on CPU (ONNX). Default model:
    ``BAAI/bge-small-en-v1.5`` (384 dimensions, ~67 MB, downloaded once from Hugging Face
    into ``cache_dir``)."""

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | None = None,
        model_impl: Any = None,
    ) -> None:
        self.model = model
        if model_impl is None:
            from fastembed import TextEmbedding

            model_impl = TextEmbedding(model_name=model, cache_dir=cache_dir)
        self._impl = model_impl
        self._dimensions: int | None = None

    @property
    def name(self) -> str:
        return f"fastembed:{self.model}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._dimensions = len(self.embed_query("dimensions"))
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [normalise(v.tolist()) for v in self._impl.passage_embed(list(texts))]

    def embed_query(self, text: str) -> Vector:
        return normalise(next(iter(self._impl.query_embed(self.query_prefix + text))).tolist())

    @property
    def query_prefix(self) -> str:
        """English BGE models retrieve better with their instruction in front of short queries."""
        if "bge" in self.model.lower() and "-en" in self.model.lower():
            return "Represent this sentence for searching relevant passages: "
        return ""


class OllamaEmbedder:
    """Embeddings from a local ollama (``POST /api/embed``), e.g. ``nomic-embed-text``.

    ``nomic-embed-text`` expects ``search_document:`` / ``search_query:`` prefixes; they are
    added for models whose name contains ``nomic``.
    """

    def __init__(
        self, model: str = "nomic-embed-text", host: str = "http://127.0.0.1:11434"
    ) -> None:
        if not re.match(r"^https?://", host):
            raise ValueError("ollama host must be an http(s) URL")
        self.model = model
        self.host = host.rstrip("/")
        self._dimensions: int | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._dimensions = len(self.embed_query("dimensions"))
        return self._dimensions

    def _prefix(self, kind: str) -> str:
        return f"search_{kind}: " if "nomic" in self.model else ""

    def _post(self, inputs: list[str]) -> list[Vector]:
        import httpx

        response = httpx.post(
            f"{self.host}/api/embed", json={"model": self.model, "input": inputs}, timeout=120
        )
        response.raise_for_status()
        payload = response.json()
        return [normalise(v) for v in payload["embeddings"]]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._post([self._prefix("document") + t for t in texts])

    def embed_query(self, text: str) -> Vector:
        return self._post([self._prefix("query") + text])[0]


class VertexEmbedder:
    """Gemini / Vertex AI text embeddings through ``google-genai`` (**paid**, your project).

    ``client.models.embed_content(model, contents, config=EmbedContentConfig(task_type,
    output_dimensionality))`` with ``RETRIEVAL_DOCUMENT`` for passages and
    ``RETRIEVAL_QUERY`` for questions. ``output_dimensionality`` must equal the index's
    ``dimensions`` (768 in ``docs/vertex-setup.md``).
    """

    BATCH = 50

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        project: str | None = None,
        location: str | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self._dimensions = dimensions
        if client is None:  # pragma: no cover - would reach Google Cloud
            from google import genai

            client = genai.Client(vertexai=True, project=project, location=location)
        self._client = client

    @property
    def name(self) -> str:
        return f"vertex:{self.model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed(self, texts: list[str], task_type: str) -> list[Vector]:
        from google.genai import types

        out: list[Vector] = []
        for start in range(0, len(texts), self.BATCH):
            response = self._client.models.embed_content(
                model=self.model,
                contents=texts[start : start + self.BATCH],
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self._dimensions
                ),
            )
            out.extend(normalise(e.values or []) for e in response.embeddings or [])
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
