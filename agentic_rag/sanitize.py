"""Retrieved passages are untrusted text.

A document in the index can say anything - a wiki page someone edited, a scraped README,
a passage written to the index by another pipeline. It reaches the model's context as
the result of ``search_docs``, so it must arrive as *data to quote*, never as
instructions. :func:`neutralise` withholds sentences that address the model ("ignore
previous instructions", "you are now", tool-call requests, secrets), defuses markup that
could close the ``<passage>`` wrapper, drops control characters and caps length.

Pattern matching reduces prompt injection; it cannot prove its absence. The other layers
are the instruction (passages are data), the citation guard (an answer may only cite
passages that were actually retrieved) and the fact that the tools have no side effects.
"""

from __future__ import annotations

import re
import unicodedata

MAX_PASSAGE_CHARS = 1500
WITHHELD = "[withheld: this sentence read like an instruction to the assistant]"

_INSTRUCTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instructions?|rules|prompt|above|previous|prior)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|rules|prompt|above|previous|prior)\b",
    r"\bforget\b.{0,20}\b(your|all|the)\b.{0,20}\b(instructions?|rules|prompt)\b",
    r"\b(system|developer)\s*(prompt|message|instructions?)\b",
    r"\byou\s+(are|must|should|will)\s+(now|always|never)\b",
    r"\b(new|updated|override)\s+instructions?\b",
    r"\b(assistant|ai|model|agent)\s*[:,]\s*(you|please|always|never)\b",
    r"\bact\s+as\b",
    r"\bjailbreak\b|\bdan\s+mode\b|\bdeveloper\s+mode\b",
    r"\b(call|invoke|run|execute)\s+(the\s+)?(tool|function)\b",
    r"\b(reveal|print|show|leak)\b.{0,30}\b(prompt|instructions?|secrets?|keys?|passwords?)\b",
    r"\b(api[_ -]?key|credit\s*card|card\s*number)\b",
    r"</?\s*(system|assistant|user|tool|passage|passages)\b[^>]*>",
]
_INSTRUCTION = re.compile("|".join(f"(?:{p})" for p in _INSTRUCTION_PATTERNS), re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_MARKUP = str.maketrans({"<": "‹", ">": "›", "{": "(", "}": ")"})
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def clean(text: str, limit: int = MAX_PASSAGE_CHARS) -> str:
    """NFKC-normalise, drop control characters, defuse markup, collapse whitespace, cap."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub(" ", text).translate(_MARKUP)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def looks_like_instruction(text: str) -> bool:
    """True when ``text`` reads like a prompt injection rather than documentation."""
    return bool(_INSTRUCTION.search(unicodedata.normalize("NFKC", text)))


def neutralise(text: str, limit: int = MAX_PASSAGE_CHARS) -> str:
    """A passage as it may appear in a prompt: instruction-like sentences withheld."""
    sentences = _SENTENCE.split(unicodedata.normalize("NFKC", text))
    kept = [WITHHELD if looks_like_instruction(s) else s for s in sentences if s.strip()]
    return clean(" ".join(kept), limit)
