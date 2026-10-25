"""``ChunkCatalog``: chunk text and metadata by datapoint id.

Vector Search returns ids and distances, not text; an application keeps the passages
somewhere else and joins on the id. Here that is ``chunks.json``, written by the ingestion
pipeline next to the index (local) or shipped with the app (Vertex). It also knows which
products, versions and doc types exist, which is what the agent's filters are checked
against.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from agentic_rag.retrieval.corpus import Chunk
from agentic_rag.retrieval.corpus import version_number

FORMAT_VERSION = 1


class ChunkCatalog:
    def __init__(self, chunks: Iterable[Chunk] = (), embedder: str = "") -> None:
        self.embedder = embedder
        self._chunks: dict[str, Chunk] = {}
        self.add(chunks)

    def add(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def remove(self, chunk_ids: Iterable[str]) -> None:
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def products(self) -> dict[str, list[str]]:
        """``{"lumen": ["1.4", "2.0", "3.0"], ...}``, versions in numeric order."""
        found: dict[str, set[str]] = {}
        for c in self._chunks.values():
            found.setdefault(c.product, set()).add(c.version)
        return {p: sorted(v, key=version_number) for p, v in sorted(found.items())}

    def doc_types(self) -> list[str]:
        return sorted({c.doc_type for c in self._chunks.values()})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "format": FORMAT_VERSION,
            "embedder": self.embedder,
            "chunks": [c.to_json() for c in self._chunks.values()],
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> ChunkCatalog:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("format") != FORMAT_VERSION:
            raise ValueError(f"{path}: unknown catalog format {raw.get('format')}")
        return cls((Chunk.from_json(c) for c in raw["chunks"]), embedder=str(raw["embedder"]))
