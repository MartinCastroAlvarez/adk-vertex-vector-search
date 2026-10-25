# Security

adk-vertex-vector-search feeds retrieved documents into a model's context and lets the
model choose what to search. That makes the index a prompt-injection channel (whatever a
document says reaches the model), the tool arguments untrusted input (the model writes
them), and citations a trust signal that must not be forgeable. Design goals: a passage
can never act as an instruction, a citation always names a passage that was actually
retrieved, the model can't make the agent search without bound, nothing leaves the
machine unless you opt in, and the paid Vertex backend is never touched by this repo's
tests or CI.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/adk-vertex-vector-search/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise. A forged citation that survives the guard, a passage that
steers the agent, or a way past the hop budget is in scope and very welcome.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The index and catalog (local files or a Vertex index), the corpus directory,
environment variables (optional `GOOGLE_API_KEY`, Google Cloud credentials), the model
endpoint, the user's trust in citations.

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| A user | their question | make the agent cite something it didn't retrieve, search without bound, read files outside the corpus |
| A document author (anyone who can get text into the corpus or index) | passage text | have instructions in a passage followed, close the passage wrapper, fake a system message |
| The model | text, `search_docs` arguments | pass unvalidated filters to the store, exceed the hop budget, cite unretrieved ids |
| Whoever controls `OLLAMA_HOST` | where questions and documents are sent | receive them without your explicit opt-in |

| Threat | Control |
| --- | --- |
| Indirect prompt injection through passages | `search_docs` returns every passage through `neutralise()`: sentences that address the model ("ignore previous instructions", "you are now", "system prompt", tool-call requests, secrets) become `[withheld: ...]`, `<`, `>`, `{`, `}` are defused so a passage can't close a wrapper or open a fake `<system>` block, control characters are dropped, length is capped. The instruction says passages are data. Tested with an injected chunk; a test also checks the bundled corpus has nothing instruction-like. Pattern matching reduces injection; it cannot prove its absence - the tools have no side effects, so the worst a successful injection can do is a wrong answer. |
| Forged or stale citations | `citation_guard` (`after_model_callback`) keeps only ids that `search_docs` returned **in this invocation** (per-question trace in session state); unknown or earlier-question ids are removed; an answer with no valid citation becomes the fixed refusal. Tested with invented ids, real-but-unretrieved ids, ids from a previous question, and answers without any search. |
| Unbounded searching / cost | `search_docs` counts searches per question and returns `hop_limit` past `RAG_MAX_HOPS`; a further search request ends the turn with the refusal; `RunConfig(max_llm_calls = 2 * hops + 4)` is the backstop. Tested with a model that never stops searching. |
| Malicious tool arguments | `validate_filters` checks product, version, doc type and dates against the catalog and fixed patterns before they become restricts (tested with `lumen" OR 1=1`, unknown versions, bad dates); the query is whitespace-normalised and capped at 300 characters. Restricts are structured values, never a query string. |
| Path traversal / arbitrary file reads | Only `--ingest DIR` reads files: `*.md` under `DIR`, symlinks and anything resolving outside `DIR` refused, 256 KiB per file, strict front matter (no YAML loader). The tool and the agent never take a path. |
| Unsafe deserialisation | The local index is `vectors.npy` loaded with `allow_pickle=False` plus JSON; no pickle, no FAISS binary index on disk (rebuilt from vectors on load). An AST test enforces `allow_pickle=False` and bans `pickle`/`marshal`/`shelve`/`yaml`/`joblib`. |
| Data leaving the machine | ollama on loopback by default; `OLLAMA_HOST` must be loopback (for the model *and* the ollama embedder) unless `RAG_ALLOW_REMOTE_OLLAMA=1`. fastembed downloads its model once from Hugging Face and then runs offline; no document text is sent anywhere. Gemini only with `RAG_PROVIDER=gemini` and your key; Vertex only with `RAG_STORE=vertex` / `RAG_EMBEDDER=vertex` and the `RAG_VERTEX_*` settings. |
| Accidental cloud calls / cost | Tests replace `google.auth.default` and `httpx.post` with functions that fail the test and run the Vertex adapter on fake clients; a subprocess test proves `import agentic_rag` loads neither the Vertex SDK nor the adapter. The agent never writes to a Vertex index; only an explicit `--ingest` does. CI has no credentials. |
| Resource names in requests | Vertex index, endpoint, deployed-index ids and the endpoint domain are validated (`projects/.../indexes/<n>`, `*.vdb.vertexai.goog`) before any client is built. |
| Secrets in prompts or logs | The package reads no secret. `neutralise` withholds passages that mention API keys or card numbers. `.env` is git-ignored. |
| Code execution | No `eval`/`exec`/`compile`, `subprocess`/`shell=`; AST tests enforce this and forbid `# noqa`, `# nosec`, `# type: ignore` in the package. |

## 2. Operator notes

- **`adk web` is a development server.** Keep it on `127.0.0.1`: it has no authentication.
  Put real authentication in front of any deployment.
- **Your corpus is your attack surface.** If untrusted people can edit the documents you
  ingest, treat answers accordingly; consider an allowlist of sources and a review step
  before `--ingest`.
- **Vertex AI Vector Search is paid by the hour while deployed** and stores your vectors
  in your project; see [docs/vertex-setup.md](docs/vertex-setup.md), including teardown.
  Give the serving identity the narrowest role that can query (and, for ingestion,
  upsert) that one index.
- **The offline stub is a demo.** `OfflineResearcher` answers with rules; use a real
  model for real questions.

## 3. Dependencies

CI runs `bandit` on the package and `pip-audit` on the locked environment including the
`vertex` extra. Dependabot alerts, security updates and secret scanning are enabled on the
repository. `filelock>=4.2.1` is pinned as a floor for a transitive dependency
(CVE-2026-112907, -112913, -112940).

Known and accepted: **CVE-2026-114603** in `opentelemetry-api` 1.44 (quadratic parsing of
an oversized W3C `tracestate` header, fixed in 1.45.1). `google-adk` 2.13 pins
`opentelemetry-api<=1.44`, so the fix can't be installed until ADK relaxes the pin. It is
reachable only through a server that extracts trace context from untrusted requests; the
CLI, tests and `adk run` serve nothing, and `adk web` should stay on loopback. CI ignores
exactly this ID; remove the ignore when ADK allows 1.45.1.
