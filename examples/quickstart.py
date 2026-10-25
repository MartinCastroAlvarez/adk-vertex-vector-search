"""Five questions through the full agentic-RAG loop - offline, with checks.

    python -m examples.quickstart

Uses the offline stub model and the hashing embedder (no ollama, no model download, no
network) and a throwaway local index built from the bundled docs. Drop the
``Settings(...)`` overrides to run it on ollama with fastembed.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from agentic_rag import Researcher
from agentic_rag import Settings
from agentic_rag.callbacks import REFUSAL
from agentic_rag.demo import run_demo


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            provider="stub", embedder="hash", embed_model="hash", index_dir=Path(tmp)
        )
        researcher = Researcher(settings=settings)
        try:
            port, compare, bridge, relaxed, refused = await run_demo(researcher)
        finally:
            researcher.close()

    assert port["searches"][0]["filters"] == {"product": "lumen", "version": "2.0"}, "filters"
    assert "7410" in port["reply"] and port["citations"], "a cited answer"
    assert len(compare["searches"]) == 2, "one search per version"
    assert bridge["hops"] == 2 and "1500" in bridge["reply"], "multi-hop"
    assert len(relaxed["searches"]) == 2 and relaxed["citations"], "self-check, then search again"
    assert refused["reply"] == REFUSAL, "refusal"
    print("\nquickstart ok: filters, decomposition, multi-hop, self-check and refusal all checked")


if __name__ == "__main__":
    asyncio.run(main())
