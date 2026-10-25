"""``python -m agentic_rag "<question>"``: ask, search, ingest, benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import cast

from agentic_rag.config import DEFAULT_EMBED_MODELS
from agentic_rag.config import EMBEDDERS
from agentic_rag.config import PROVIDERS
from agentic_rag.config import STORES
from agentic_rag.config import ConfigError
from agentic_rag.config import EmbedderKind
from agentic_rag.config import Provider
from agentic_rag.config import Settings
from agentic_rag.config import StoreKind
from agentic_rag.filters import extract_filters
from agentic_rag.retrieval.corpus import CorpusError


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agentic-rag",
        description="An agentic RAG agent in Google ADK over a Vector Search index "
        "(a local FAISS + BM25 look-alike by default, Vertex AI Vector Search optional).",
    )
    p.add_argument("question", nargs="?", help='e.g. "What is the default port in Lumen 2.0?"')
    p.add_argument("--provider", choices=PROVIDERS, help="default: RAG_PROVIDER or ollama")
    p.add_argument("--model", help="model id for the provider (default: qwen3:4b on ollama)")
    p.add_argument("--store", choices=STORES, help="default: RAG_STORE or local")
    p.add_argument("--embedder", choices=EMBEDDERS, help="default: fastembed (hash for stub)")
    p.add_argument("--index-dir", type=Path, help="where the local index and catalog live")
    p.add_argument("--max-hops", type=int, help="searches allowed per question (default 3)")
    p.add_argument("--search", action="store_true", help="only retrieve: print the top passages")
    p.add_argument(
        "--ingest",
        nargs="?",
        const="",
        metavar="DIR",
        help="(re)build the index from DIR (default: the bundled Lumen/Ferry docs)",
    )
    p.add_argument("--bench", action="store_true", help="recall@k / MRR of every ranking mode")
    p.add_argument("--demo", action="store_true", help="ask the five scripted demo questions")
    p.add_argument("--json", action="store_true", help="print results as JSON")
    p.add_argument("--mermaid", action="store_true", help="print the agent graph and exit")
    return p


def settings_from(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    provider = cast(Provider, args.provider or settings.provider)
    model, embedder, embed_model = settings.model, settings.embedder, settings.embed_model
    if args.provider and args.provider != settings.provider:
        defaults = Settings.from_env({"RAG_PROVIDER": args.provider})
        model, embedder, embed_model = defaults.model, defaults.embedder, defaults.embed_model
    if args.embedder:
        embedder = cast(EmbedderKind, args.embedder)
        embed_model = DEFAULT_EMBED_MODELS[embedder]
    return replace(
        settings,
        provider=provider,
        model=args.model or model,
        embedder=embedder,
        embed_model=embed_model,
        store=cast(StoreKind, args.store or settings.store),
        index_dir=args.index_dir.expanduser() if args.index_dir else settings.index_dir,
        max_hops=args.max_hops or settings.max_hops,
    )


def _ingest(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.retrieval.factory import ingest_corpus

    report = ingest_corpus(settings, Path(args.ingest) if args.ingest else None)
    if args.json:
        print(json.dumps(report.__dict__, indent=2))
    else:
        print(
            f"ingested {report.documents} documents -> {report.chunks} chunks "
            f"({report.upserted} upserted, {report.removed} removed) with {report.embedder} "
            f"({report.dimensions} dimensions)"
        )
    return 0


def _search(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.retrieval.factory import open_retriever

    retriever = open_retriever(settings)
    try:
        filters = extract_filters(args.question, retriever.catalog.products())
        hits = retriever.search(args.question, filters, top_k=settings.top_k)
    finally:
        retriever.close()
    rows: list[dict[str, Any]] = [
        {
            "id": h.chunk.chunk_id,
            "distance": h.distance,
            "sparse_distance": h.sparse_distance,
            "text": h.chunk.text,
        }
        for h in hits
    ]
    if args.json:
        print(json.dumps({"filters": filters.as_dict(), "hits": rows}, indent=2))
    else:
        print(f"filters: {filters.describe()}")
        for i, row in enumerate(rows, 1):
            print(f"{i}. [{row['id']}] {row['text'][:160]}")
    return 0


def _bench(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.bench import evaluate
    from agentic_rag.bench import load_questions
    from agentic_rag.bench import markdown
    from agentic_rag.retrieval.factory import open_retriever

    retriever = open_retriever(settings)
    try:
        questions = load_questions()
        scores = evaluate(retriever, questions)
    finally:
        retriever.close()
    name = retriever.embedder.name
    if args.json:
        print(
            json.dumps(
                {
                    "embedder": name,
                    "questions": len(questions),
                    "scores": [s.as_dict() for s in scores],
                },
                indent=2,
            )
        )
    else:
        title = f"{len(questions)} questions, {len(retriever.catalog)} chunks, embedder {name}"
        print(markdown(scores, title))
    return 0


async def _ask(args: argparse.Namespace, settings: Settings) -> int:
    from agentic_rag.runner import Researcher

    researcher = Researcher(settings=settings)
    try:
        if args.demo:
            from agentic_rag.demo import run_demo

            results = await run_demo(researcher, echo=(lambda _: None) if args.json else print)
            if args.json:
                print(json.dumps(results, indent=2))
            return 0
        answer = await researcher.ask(args.question)
    finally:
        researcher.close()
    if args.json:
        print(json.dumps(answer.as_dict(), indent=2))
    else:
        print(answer.reply)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = settings_from(args)
        if args.mermaid:
            from agentic_rag.mermaid import to_mermaid

            print(to_mermaid(settings))
            return 0
        if args.ingest is not None:
            return _ingest(args, settings)
        if args.bench:
            return _bench(args, settings)
        if not (args.question or args.demo):
            print(
                'error: ask a question, e.g. python -m agentic_rag "What is the default port '
                'in Lumen 2.0?"',
                file=sys.stderr,
            )
            return 2
        if args.search:
            return _search(args, settings)
        return asyncio.run(_ask(args, settings))
    except (ConfigError, CorpusError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
