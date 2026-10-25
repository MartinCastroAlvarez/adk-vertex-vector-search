# Contributing to adk-vertex-vector-search

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install
poetry run pytest
poetry run ruff check agentic_rag tests examples
poetry run ruff format --check agentic_rag tests examples
poetry run mypy agentic_rag
poetry run bandit -c pyproject.toml -q -r agentic_rag
poetry run pip-audit --ignore-vuln CVE-2026-114603   # see SECURITY.md
poetry run python -m examples.quickstart
RAG_PROVIDER=stub poetry run python -m agentic_rag --bench
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## What this project will not accept

- Tests or CI that call a real model, Google Cloud or the network, or download a model.
- A retrieval behaviour tested on one store only: add it to `tests/test_contract.py`, and
  to `tests/fake_vector_search.py` if the fake needs to learn it (following the Vector
  Search docs, with a link in the PR).
- Moving an enforcement (hop budget, citation check, filter validation) into the prompt.
- Hosted providers or Vertex turned on by default; pickle-based index formats.

## What it does accept

- More corpus documents or benchmark questions (self-written or public domain), with
  relevance labels, and the README tables refreshed from a real run.
- A cross-encoder re-ranker or SPLADE sparse vectors behind the same `Retriever`, as long
  as BM25 stays the dependency-free default.
- Vector Search features the adapter doesn't use yet (datapoint deny lists, `embedding_ids`
  queries, `read_index_datapoints`), each with a contract test.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
