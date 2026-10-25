# Running the agent on Vertex AI Vector Search

> **This costs money.** Vector Search bills for the nodes that serve a deployed index
> **every hour it is deployed, whether you query it or not**, plus index building and
> streaming updates; the embedding model bills per call. A forgotten deployment is the
> classic surprise bill. Nothing in this repository - tests, demo, CI - ever calls Google
> Cloud; the default store is the free local one. Follow this page only with **your own**
> project and billing account, check the current Vertex AI pricing page first, and run the
> [teardown](#5-teardown) when you are done.

What you create: one **index** (768 dimensions, dot product on unit vectors, streaming
updates), one **index endpoint** with a public endpoint, one **deployed index** on the
smallest machine. Then three settings point the agent at it. Names below are as of
`google-cloud-aiplatform` 2.6 and `google-adk` 2.13.

## 1. Project, API, roles

```bash
export PROJECT_ID=my-project          # yours; billing must be enabled
export LOCATION=us-central1
gcloud auth login
gcloud config set project "$PROJECT_ID"
gcloud services enable aiplatform.googleapis.com
gcloud auth application-default login   # Application Default Credentials for the SDK
```

The identity that **creates** resources needs `roles/aiplatform.user`. The identity that
**serves** the agent only queries (and, for `--ingest`, upserts): grant it
`roles/aiplatform.user` on this project only, or a custom role trimmed to the index and
index-endpoint permissions it uses. Never run it with project owner or editor rights.

## 2. Install the Vertex extra

```bash
pip install "adk-vertex-vector-search[vertex] @ git+https://github.com/MartinCastroAlvarez/adk-vertex-vector-search"
```

## 3. Create the index, the endpoint, and deploy (paid from here on)

An empty index with **streaming updates** (no `contents_delta_uri`: the agent's
ingestion upserts the datapoints itself), `DOT_PRODUCT_DISTANCE` with `UNIT_L2_NORM`
(cosine; every embedder here returns unit vectors), `SHARD_SIZE_SMALL` (the cheapest
shard; dense + sparse vectors of this corpus are a few hundred KB):

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="agentic-rag-docs",
    dimensions=768,                      # = RAG_EMBED_DIMENSIONS
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    feature_norm_type="UNIT_L2_NORM",
    shard_size="SHARD_SIZE_SMALL",
    index_update_method="STREAM_UPDATE",
)
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="agentic-rag-endpoint",
    public_endpoint_enabled=True,
)
# Deploying takes ~20-30 minutes, and billing for the node starts here.
endpoint.deploy_index(
    index=index,
    deployed_index_id="docs_deployed",
    machine_type="e2-standard-2",
    min_replica_count=1,
    max_replica_count=1,
)
print("RAG_VERTEX_INDEX=" + index.name)
print("RAG_VERTEX_INDEX_ENDPOINT=" + endpoint.name)
print("RAG_VERTEX_ENDPOINT_DOMAIN=" + endpoint.public_endpoint_domain_name)
```

Hybrid search needs nothing extra at creation time: every datapoint the ingestion
pipeline upserts carries both a dense `feature_vector` and a BM25 `sparse_embedding`,
and every query sends both with `rrf.alpha` (`RAG_ALPHA`, 0.5 by default).

## 4. Point the agent at it

```bash
export RAG_STORE=vertex
export RAG_VERTEX_PROJECT=my-project
export RAG_VERTEX_LOCATION=us-central1
export RAG_VERTEX_INDEX=1234567890            # the numeric ids printed above
export RAG_VERTEX_INDEX_ENDPOINT=9876543210
export RAG_VERTEX_DEPLOYED_INDEX=docs_deployed
export RAG_VERTEX_ENDPOINT_DOMAIN=1234.us-central1-567.vdb.vertexai.goog
# Optional: Gemini embeddings instead of the local fastembed model (also paid).
# They must match the index: 768 dimensions.
export RAG_EMBEDDER=vertex RAG_EMBED_MODEL=gemini-embedding-001 RAG_EMBED_DIMENSIONS=768

agentic-rag --ingest     # chunk, embed, UpsertDatapoints (streaming), write the catalog
agentic-rag "What is the default port in Lumen 2.0?"
```

With the default `fastembed` embedder (`BAAI/bge-small-en-v1.5`, **384** dimensions)
create the index with `dimensions=384` instead; embeddings stay free and local, only the
index is paid. The ingestion writes `chunks.json` and `sparse.json` to
`RAG_INDEX_DIR/vertex/`: Vector Search returns ids, the catalog turns them back into
passages. Ship that directory with the agent.

`--ingest` is the only command that writes to the index; the agent never builds a
Vertex index implicitly (it refuses to start without the catalog). Re-running it upserts
the same ids again and removes chunks that disappeared from the corpus
(`RemoveDatapoints`).

What the code sends, so you can compare it with the [Vector Search
docs](https://docs.cloud.google.com/vertex-ai/docs/vector-search/overview):
`IndexServiceClient.upsert_datapoints(UpsertDatapointsRequest(index, datapoints))` in
batches of 100, and `MatchServiceClient(api_endpoint=<public domain>).find_neighbors(
FindNeighborsRequest(index_endpoint, deployed_index_id, queries=[Query(datapoint=
IndexDatapoint(feature_vector, sparse_embedding, restricts, numeric_restricts),
neighbor_count, per_crowding_attribute_neighbor_count, rrf=RRF(alpha))]))`. See
`agentic_rag/retrieval/vertex.py`.

## 5. Teardown

Undeploy first: **the deployed index is what bills by the hour.**

```bash
gcloud ai index-endpoints undeploy-index "$RAG_VERTEX_INDEX_ENDPOINT" \
  --deployed-index-id=docs_deployed --region="$LOCATION"
gcloud ai index-endpoints delete "$RAG_VERTEX_INDEX_ENDPOINT" --region="$LOCATION" --quiet
gcloud ai indexes delete "$RAG_VERTEX_INDEX" --region="$LOCATION" --quiet
gcloud ai index-endpoints list --region="$LOCATION"   # should be empty
gcloud ai indexes list --region="$LOCATION"           # should be empty
```

Or from Python: `endpoint.undeploy_all()`, `endpoint.delete()`, `index.delete()`.
Revoke the local credentials with `gcloud auth application-default revoke` if you made
them for this.

## Why not RAG Engine / `VertexAiRagRetrieval`?

ADK ships `VertexAiRagRetrieval`, a tool over **Vertex AI RAG Engine** corpora (which can
themselves use Vector Search underneath). On Gemini it becomes a built-in retrieval the
model runs server-side: no visible tool call, no per-call restricts, numeric filters,
crowding or RRF weight, no hop count, no passage ids to check citations against, and no
local or non-Gemini path. This repo needs all of those, so the agent calls Vector Search
through its own `search_docs` function tool. If you already run RAG Engine and only need
"retrieve, then answer" on Gemini, `VertexAiRagRetrieval` is less code.
