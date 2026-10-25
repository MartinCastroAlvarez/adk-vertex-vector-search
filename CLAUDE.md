# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md). 5. [`docs/vertex-setup.md`](docs/vertex-setup.md).

## Mission

`adk-vertex-vector-search` shows agentic RAG in current Google ADK (Python): an
`LlmAgent` that decides when and what to retrieve through one `search_docs` tool, on a
`VectorStore` that is either Vertex AI Vector Search or a free local look-alike with the
same semantics, as small, correct and tested code.

## The rules

1. **Zero cost.** Never call, deploy to or authenticate against Google Cloud. Tests, CI
   and the demo use the stub model, the hashing embedder, `LocalVectorSearch` and the
   fake GAPIC clients. `google.auth.default()` and `httpx.post` fail any test that calls
   them; keep it that way.
2. **One contract, two stores.** A retrieval behaviour goes into `tests/test_contract.py`
   and must pass on both backends; teach `tests/fake_vector_search.py` from the Vector
   Search docs, never by copying `LocalVectorSearch`.
3. **The code enforces what prompts can't.** Hop budget, filter validation, citation
   checks and passage neutralisation stay in code with scripted-model tests.
4. **Passages and tool arguments are untrusted.** Passages reach the model only through
   `neutralise`; filters only through `validate_filters`.
5. **ADK conventions.** `root_agent` lives in `agentic_rag/agent.py`; `adk run`,
   `adk web` and `adk eval` must keep working (CI runs all three). Re-record
   `evals/agentic_rag.evalset.json` from the stub when its trajectory changes on purpose.
6. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or
   loosen lint/type/coverage config. No pickle anywhere (`allow_pickle=False`).

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`, `mypy`
(strict), `bandit` and `pip-audit` all green; `python -m examples.quickstart` passes;
`python -m agentic_rag --bench` still meets the CI thresholds; README (benchmark tables
when retrieval changes), ARCHITECTURE/SECURITY (when relevant) and CHANGELOG are updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets, `.env`,
`.adk/`, indexes or model caches.
