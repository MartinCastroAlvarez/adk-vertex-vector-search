# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-25

### Added

- An agentic RAG agent on google-adk 2.13 (`LlmAgent` `researcher` with one
  `search_docs` function tool), exposed as `root_agent` in `agentic_rag/agent.py` for
  `adk run`, `adk web` and `adk eval`.
- `search_docs`: filter validation against the catalog, a per-question hop budget,
  neutralised passages; `filter_hint` (`before_model_callback`) and `citation_guard`
  (`after_model_callback`: citations must be retrieved in this question, uncited answers
  become a refusal, searches past the budget are stopped).
- `VectorStore` with `LocalVectorSearch` (FAISS flat/HNSW + inverted index; restricts,
  numeric restricts, hybrid RRF, crowding; no-pickle files on disk) and
  `VertexVectorSearchStore` (`aiplatform_v1` `UpsertDatapoints`, `RemoveDatapoints`,
  `FindNeighbors` on a public endpoint); never called by tests or CI.
- BM25 sparse vectors, embedders (fastembed, ollama, hashing, Vertex/Gemini), ingestion
  with stable chunk ids, a 23-document fictional corpus across versions, and a
  50-question retrieval benchmark (`--bench`: recall@k, MRR per ranking mode).
- A contract test suite run on both stores against independent fake GAPIC clients; a
  scripted-model agent suite; `OfflineResearcher` stub; ADK eval set; CLI with `--demo`,
  `--search`, `--ingest`, `--bench`, `--json`, `--mermaid`; `docs/vertex-setup.md`
  (create, deploy, point, teardown; paid).
