"""Retrieval benchmark: recall@k and MRR on the bundled corpus, per ranking mode.

``qa.jsonl`` has one question per line with the sections that answer it
(``<doc_id>#<section-slug>``). A retrieved chunk counts as relevant when its section is
listed; results are de-duplicated by section before scoring, so a long section split
into two chunks can't count twice.

* **recall@k** - the share of a question's relevant sections found in the top k, averaged.
* **MRR@10** - 1 / rank of the first relevant section (0 if none in the top 10), averaged.

Four configurations: ``dense``, ``sparse`` (BM25), ``hybrid`` (RRF, alpha 0.5) and
``hybrid+filters`` (the same query with the restricts :func:`extract_filters` reads off
the question - what the agent's filter hint suggests).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag.filters import SearchFilters
from agentic_rag.filters import extract_filters
from agentic_rag.retrieval.corpus import bundled_corpus
from agentic_rag.retrieval.retriever import Mode
from agentic_rag.retrieval.retriever import Retriever

CONFIGS: tuple[tuple[str, Mode, bool], ...] = (
    ("dense", "dense", False),
    ("sparse", "sparse", False),
    ("hybrid", "hybrid", False),
    ("hybrid+filters", "hybrid", True),
)
KS = (1, 3, 5)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    relevant: frozenset[str]


@dataclass(frozen=True)
class Score:
    config: str
    recall: dict[int, float]
    mrr: float
    ms_per_query: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            **{f"recall@{k}": round(v, 3) for k, v in self.recall.items()},
            "mrr@10": round(self.mrr, 3),
            "ms_per_query": round(self.ms_per_query, 2),
        }


def load_questions(path: Path | None = None) -> list[Question]:
    path = path or bundled_corpus() / "qa.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            out.append(Question(str(raw["id"]), str(raw["question"]), frozenset(raw["relevant"])))
    return out


def sections(retriever: Retriever, q: Question, mode: Mode, filters: SearchFilters) -> list[str]:
    hits = retriever.search(q.question, filters, top_k=10, mode=mode)
    return list(dict.fromkeys(h.chunk.section_key for h in hits))


def evaluate(retriever: Retriever, questions: list[Question]) -> list[Score]:
    products = retriever.catalog.products()
    # The benchmark measures ranking, so crowding is off (it would cap a doc's sections).
    per_document, retriever.per_document = retriever.per_document, None
    scores = []
    try:
        for name, mode, use_filters in CONFIGS:
            recall = dict.fromkeys(KS, 0.0)
            mrr = 0.0
            started = time.perf_counter()
            for q in questions:
                filters = extract_filters(q.question, products) if use_filters else SearchFilters()
                ranked = sections(retriever, q, mode, filters)
                for k in KS:
                    recall[k] += len(q.relevant & set(ranked[:k])) / len(q.relevant)
                first = next((i for i, s in enumerate(ranked, 1) if s in q.relevant), None)
                mrr += 1 / first if first else 0.0
            elapsed = (time.perf_counter() - started) * 1000
            n = len(questions)
            scores.append(Score(name, {k: v / n for k, v in recall.items()}, mrr / n, elapsed / n))
    finally:
        retriever.per_document = per_document
    return scores


def markdown(scores: list[Score], title: str) -> str:
    head = "| " + " | ".join(["config", *(f"recall@{k}" for k in KS), "MRR@10", "ms/query"]) + " |"
    rule = "|" + "---|" * (len(KS) + 3)
    rows = [
        f"| {s.config} | "
        + " | ".join(f"{s.recall[k]:.3f}" for k in KS)
        + f" | {s.mrr:.3f} | {s.ms_per_query:.1f} |"
        for s in scores
    ]
    return "\n".join([f"**{title}**", "", head, rule, *rows])
