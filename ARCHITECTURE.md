# Architecture

```
agentic_rag/
  agent.py          root_agent = build_agent(Settings.from_env())         <- adk run / web / eval
  researcher.py     the LlmAgent: instruction, search_docs, two callbacks
  tools.py          search_docs (FunctionTool): validate, hop budget, search, neutralise; RetrieverHandle
  callbacks.py      FilterHint (before_model), CitationGuard (after_model), REFUSAL
  trace.py          per-invocation state: hops, searches, retrieved ids, answer
  filters.py        SearchFilters -> restricts; extract_filters (rules); validate_filters (tool args)
  sanitize.py       neutralise passages (instruction-like sentences, markup, control chars)
  runner.py         Researcher: Runner + InMemorySessionService; Answer
  config.py         Settings from RAG_* (provider, store, embedder, hops, Vertex ids)
  models.py         provider -> BaseLlm (LiteLlm on ollama, Gemini, OfflineResearcher)
  offline.py        OfflineResearcher: the agentic-RAG loop as rules, for demo and CI
  stub.py           StubLlm, ScriptedLlm (tests), ToolCall
  demo.py, bench.py, mermaid.py, cli.py
  corpus/           23 Markdown docs (Lumen, Ferry) + qa.jsonl (benchmark labels)
  retrieval/
    types.py        Datapoint, Query, Neighbor, Restrict, NumericRestrict, SparseVector; matches()
    base.py         VectorStore protocol; rrf_fuse; apply_crowding
    local.py        LocalVectorSearch: FAISS (flat/HNSW) + inverted index, files on disk
    vertex.py       VertexVectorSearchStore: aiplatform_v1 IndexService + MatchService
    sparse.py       tokenizer (Snowball), BM25Encoder -> sparse vectors
    embeddings.py   HashEmbedder, FastEmbedEmbedder, OllamaEmbedder, VertexEmbedder
    corpus.py       front matter, chunking, chunk ids
    catalog.py      ChunkCatalog: id -> passage, products/versions
    ingest.py       chunk -> embed -> Datapoint(restricts, numeric, crowding) -> upsert
    retriever.py    question + filters -> Query -> neighbours -> passages
    factory.py      Settings -> embedder, store, retriever; ingest_corpus
tests/fake_vector_search.py   independent fake of the two GAPIC clients (real protos)
evals/                        ADK eval set recorded from the stub + criteria
```

## One question

1. `Runner.run_async` appends the question and starts `researcher`.
2. `FilterHint` (before every model call) opens the retriever on first use (building the
   local index from the bundled corpus if missing) and appends the catalog and the
   filters `extract_filters` reads off the question.
3. The model answers with text or with `search_docs` calls (possibly several in parallel).
4. `search_docs` loads the per-invocation trace from session state; past the budget it
   returns `hop_limit`. Otherwise it validates the filters against the catalog, runs
   `Retriever.search`, records the returned ids, and returns neutralised passages with
   `hops_left`.
5. `Retriever.search` embeds the query (dense + BM25 sparse), builds one `Query` with the
   restricts, numeric restricts, crowding and RRF alpha, calls
   `VectorStore.find_neighbors` and joins ids with the `ChunkCatalog`.
6. Steps 3-5 repeat until the model answers. `CitationGuard` (after every model call)
   checks the final text: valid citations kept, unknown ones removed, `Sources:` added;
   none valid -> the refusal. A search request after `hop_limit` -> the refusal.
7. `Researcher.ask` returns the reply plus the trace (searches, hops, citations, status).

## Design decisions

- **The model plans; the code enforces.** Planning, rewriting and the self-check are the
  model's (and therefore unreliable on small models); the hop budget, filter validation,
  citation checks and passage neutralisation are code and are tested with a scripted model.
- **Mirror Vector Search's data model, not an ORM.** `retrieval/types.py` is the
  `IndexDatapoint` / `FindNeighborsRequest.Query` shape, so the local store can be
  checked against the Vertex docs and the Vertex adapter is a field-by-field mapping.
- **BM25 as a sparse vector.** Hybrid search on Vertex needs a sparse embedding per
  datapoint; BM25 weights at hashed dimensions with unit query weights make the dot
  product equal to the BM25 score, so both backends rank sparse results identically.
- **Ids are citations.** Chunk ids are stable and readable; the catalog maps them back to
  passages because Vector Search returns ids, not text.
- **GAPIC clients, lazily.** The Vertex adapter uses `aiplatform_v1` clients directly
  (what `MatchingEngineIndexEndpoint.find_neighbors` builds for a public endpoint), so
  constructing it makes no network call; clients are built on first use and injectable.
- **No implicit cloud writes.** The local index builds itself on first use; the Vertex
  index is written only by an explicit `--ingest`.
