"""``OfflineResearcher``: a deterministic stand-in for the LLM, for the demo, CLI and CI.

It is not a language model. It reads what a real model would read - the question, the
filter hints in the system instruction, the ``search_docs`` results - and follows the
agentic-RAG loop the instruction describes with plain rules, so the control flow is
visible without ollama:

1. **Plan + rewrite.** Keywords of the question become the query; the hinted filters
   become arguments. Several versions mentioned -> one search per version
   (decomposition). "the X that replaced Y" -> first find what replaced Y (a bridge).
2. **Self-check.** Is there one retrieved sentence that covers most of the question's key
   terms? Bridge resolved? If not, and searches are left: search again - with the bridge
   entity, with looser filters, or with the missing terms only.
3. **Answer** with the best sentence of each sub-question and its passage id, or refuse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from google.adk.models.llm_request import LlmRequest

from agentic_rag.callbacks import HINT_PREFIX
from agentic_rag.callbacks import REFUSAL
from agentic_rag.callbacks import VERSIONS_PREFIX
from agentic_rag.retrieval.sparse import tokenize
from agentic_rag.retrieval.sparse import words
from agentic_rag.sanitize import WITHHELD
from agentic_rag.stub import Reply
from agentic_rag.stub import StubLlm
from agentic_rag.stub import ToolCall
from agentic_rag.stub import system_text
from agentic_rag.tools import TOOL_NAME

#: Words that say little about *which* passage answers ("what is the default ...").
GENERIC = frozenset(
    tokenize(
        "default many much use used work happen mean get set make take new version versions "
        "feature does tell explain"
    )
)
SUFFICIENT = 0.6
_BRIDGE = re.compile(
    r"\b(?:that|which) (?:replaced|replaces|superseded) (.+?)(?:\?|$|,| in (?=[A-Z]))", re.I
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_FILTER_KEYS = (
    "product",
    "version",
    "since_version",
    "doc_type",
    "released_after",
    "released_before",
)


@dataclass
class Passage:
    id: str
    title: str
    section: str
    product: str
    version: str
    text: str

    def sentences(self) -> list[str]:
        return [s for s in _SENTENCE.split(self.text) if s and s != WITHHELD]


@dataclass
class Search:
    args: dict[str, Any]
    status: str
    passages: list[Passage] = field(default_factory=list)
    hops_left: int = 0


def question_of(request: LlmRequest) -> tuple[str, int]:
    """The newest user text and its index in ``request.contents``."""
    for index in range(len(request.contents) - 1, -1, -1):
        content = request.contents[index]
        if content.role != "user":
            continue
        text = " ".join(p.text for p in content.parts or [] if p.text)
        if text:
            return text.strip(), index
    return "", -1


def searches_of(request: LlmRequest, start: int) -> list[Search]:
    """Every ``search_docs`` call after ``start`` with its result, in order."""
    calls: dict[str, dict[str, Any]] = {}
    order: list[dict[str, Any]] = []
    out: list[Search] = []
    for content in request.contents[start + 1 :]:
        for part in content.parts or []:
            if part.function_call and part.function_call.name == TOOL_NAME:
                args = dict(part.function_call.args or {})
                calls[part.function_call.id or str(len(order))] = args
                order.append(args)
            response = part.function_response
            if response and response.name == TOOL_NAME:
                key = response.id or str(len(out))
                args = calls.get(key) or (order[len(out)] if len(out) < len(order) else {})
                body = dict(response.response or {})
                passages = [
                    Passage(
                        id=str(p.get("id", "")),
                        title=str(p.get("title", "")),
                        section=str(p.get("section", "")),
                        product=str(p.get("product", "")),
                        version=str(p.get("version", "")),
                        text=str(p.get("text", "")),
                    )
                    for p in body.get("passages") or []
                ]
                out.append(
                    Search(
                        args, str(body.get("status", "")), passages, int(body.get("hops_left", 0))
                    )
                )
    return out


def hints(request: LlmRequest) -> tuple[dict[str, str], list[str]]:
    """The filters and versions the ``filter_hint`` callback put in the instruction."""
    text = system_text(request)
    filters: dict[str, str] = {}
    versions: list[str] = []
    m = re.search(rf"{HINT_PREFIX}: (.+?)\.\s*$", text, re.M)
    if m:
        for pair in m.group(1).split(", "):
            key, _, value = pair.partition("=")
            if key in _FILTER_KEYS and value:
                filters[key] = value
    m = re.search(rf"{VERSIONS_PREFIX}: (.+?)\. ", text)
    if m:
        versions = m.group(1).split(", ")
    return filters, versions


def key_terms(text: str, drop: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Content words of a question: no stopwords, versions, product names or filler."""
    return list(
        dict.fromkeys(
            t
            for t in tokenize(text)
            if t not in GENERIC and t not in drop and not re.fullmatch(r"v?[0-9.]+(x)?", t)
        )
    )


def query_words(text: str, drop: set[str] | frozenset[str] = frozenset()) -> str:
    """The question rewritten as a keyword query (the words behind :func:`key_terms`)."""
    keep = set(key_terms(text, drop))
    return " ".join(dict.fromkeys(w for w in words(text) if set(tokenize(w)) & keep))


def best_sentence(passages: list[Passage], terms: list[str]) -> tuple[float, Passage, str] | None:
    best: tuple[float, Passage, str] | None = None
    for rank, passage in enumerate(passages):
        for sentence in passage.sentences():
            own = set(tokenize(sentence))
            heading = set(tokenize(passage.section))
            # A term in the sentence itself counts fully; one only in its heading, half.
            hits = sum(1.0 if t in own else 0.5 if t in heading else 0.0 for t in terms)
            score = hits / max(len(terms), 1) - rank * 0.001
            if best is None or score > best[0]:
                best = (score, passage, sentence)
    return best


def replacement(passages: list[Passage], subject: str) -> tuple[str, Passage, str] | None:
    """What replaced ``subject``, from sentences like "replaces Mirror with Raft groups"."""
    head = re.escape(subject.split()[0].strip("`"))
    patterns = [
        rf"replac\w*\s+`?{head}\b[^.]*?\bwith\s+`?([A-Za-z][\w.-]*(?:\s+[a-z][\w-]*)?)`?",
        rf"`?([A-Za-z][\w.-]*(?:\s+[a-z][\w-]*)?)`?\s+replaces\s+`?{head}\b",
        rf"`?{head}\b[^.]*?\b(?:was|is)\s+replaced\s+by\s+`?([A-Za-z][\w.-]*(?:\s+[a-z][\w-]*)?)`?",
    ]
    for passage in passages:
        for sentence in passage.sentences():
            for pattern in patterns:
                m = re.search(pattern, sentence, re.I)
                if m:
                    entity = re.sub(r"[,;:]$", "", m.group(1)).strip()
                    entity = re.sub(r"\s+(which|that|and|with|in)$", "", entity)
                    return entity, passage, sentence
    return None


def _call(query: str, filters: dict[str, str]) -> ToolCall:
    return ToolCall(TOOL_NAME, {"query": query, **filters})


class OfflineResearcher(StubLlm):
    model: str = "offline/researcher"

    def reply(self, request: LlmRequest) -> Reply:
        question, start = question_of(request)
        filters, versions = hints(request)
        products = set(re.findall(r"\b([a-z][a-z0-9-]*) \(", system_text(request)))
        bridge = _BRIDGE.search(question)
        subject = bridge.group(1).strip() if bridge else ""
        drop = set(tokenize(" ".join(products))) | {"replac", "supersed"}
        terms = key_terms(question, drop | set(tokenize(subject)))
        query = query_words(question, drop | set(tokenize(subject))) or question
        searches = searches_of(request, start)

        if not searches:
            if bridge:
                bridge_filters = {k: v for k, v in filters.items() if k == "product"}
                return _call(f"{query_words(subject, drop)} replaced", bridge_filters)
            if len(versions) > 1:
                return [_call(query, {**filters, "version": v}) for v in versions]
            return _call(query, filters)

        last = searches[-1]
        if last.status == "invalid_filter":
            relaxed = {k: v for k, v in last.args.items() if k in ("query", "product")}
            return ToolCall(TOOL_NAME, relaxed)
        passages: list[Passage] = []
        for s in searches:
            passages.extend(p for p in s.passages if p.id not in {q.id for q in passages})
        budget = last.hops_left > 0 and last.status != "hop_limit"

        if bridge:
            found = replacement(passages, subject)
            if found is None:
                return self._refine_or_refuse(searches, terms, filters, budget)
            entity, bridge_passage, bridge_sentence = found
            entity_terms = set(tokenize(entity))
            later = [p for s in searches[1:] for p in s.passages]
            best = best_sentence(later, [*key_terms(entity), *terms]) if later else None
            if best is None or best[0] < SUFFICIENT:
                if not budget or len(searches) > 2:
                    return REFUSAL
                rest = query_words(question, drop | set(tokenize(subject)) | entity_terms)
                return _call(
                    f"{entity} {rest}".strip(),
                    {"product": bridge_passage.product, "version": bridge_passage.version},
                )
            return self._answer([(bridge_passage, bridge_sentence), (best[1], best[2])])

        groups = self._groups(searches, versions)
        picks: list[tuple[Passage, str]] = []
        for group in groups:
            best = best_sentence(group, terms)
            if best is None or best[0] < SUFFICIENT:
                return self._refine_or_refuse(searches, terms, filters, budget)
            picks.append((best[1], best[2]))
        return self._answer(picks)

    @staticmethod
    def _groups(searches: list[Search], versions: list[str]) -> list[list[Passage]]:
        """One passage pool per sub-question (per version when versions were compared)."""
        if len(versions) > 1:
            return [
                [p for s in searches if s.args.get("version") == v for p in s.passages]
                for v in versions
            ]
        return [[p for s in searches for p in s.passages]]

    @staticmethod
    def _refine_or_refuse(
        searches: list[Search], terms: list[str], filters: dict[str, str], budget: bool
    ) -> Reply:
        if not budget:
            return REFUSAL
        last = searches[-1].args
        narrowing = {k: v for k, v in last.items() if k in _FILTER_KEYS and k != "product"}
        if narrowing:  # too strict? keep the product, drop version/date/doc type filters
            relaxed = {k: v for k, v in last.items() if k in ("product",)}
            return _call(str(last.get("query", "")), relaxed)
        if len(searches) == 1 and terms:
            seen = {
                t for s in searches for p in s.passages for t in tokenize(f"{p.section} {p.text}")
            }
            missing = [t for t in terms if t not in seen]
            if missing and missing != terms:
                product = {k: v for k, v in filters.items() if k == "product"}
                return _call(" ".join(missing), product)
        return REFUSAL

    @staticmethod
    def _answer(picks: list[tuple[Passage, str]]) -> str:
        lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        for passage, sentence in picks:
            if (passage.id, sentence) in seen:
                continue
            seen.add((passage.id, sentence))
            label = f"{passage.product.capitalize()} {passage.version}"
            lines.append(f"{label}: {sentence} [{passage.id}]")
        return " ".join(lines)
