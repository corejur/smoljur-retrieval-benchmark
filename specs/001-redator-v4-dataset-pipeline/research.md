# Research: Representative Redator v4 Pipeline

**Date**: 2026-09-22  
**Scope**: V4 test-only dataset preparation, Qwen chunk embeddings in local PostgreSQL with pgvector, document-scoped retrieval through a trusted remote vLLM service, and comparable metric evaluation.

## Decision 1 — Keep v4 as the authoritative workflow

**Decision**: Extend `scripts/v4/` and the four shared modules declared in `scripts/v4/manifest.py`. Preserve the v1–v3 scripts for reproducibility.

**Rationale**: The manifest explicitly separates v4, reused, and excluded modules. The v4 source contract uses ordered `question_texts` and `output` lists; the earlier prompt-parsing flow has different question and document rejection behavior.

**Alternatives considered**: Reusing `scripts/prepare_dataset.py` would reintroduce the older whole-document filtering policy. Rewriting the whole pipeline would discard working v4 tests and streaming behavior.

## Decision 2 — Validate a real v4 source contract before building

**Decision**: Accept only the v4 test CSV as the production source, with split fixed to `test`; require `id`, `data`, `question_texts`, and `output` in each source row. The last two fields must be ordered JSON arrays with equal lengths. Reject duplicate document IDs and audit blank questions rather than silently skipping them. Record source path/fingerprint and source-row count. Provide a small synthetic test-split fixture for automated validation only.

**Rationale**: `scripts/v4/questions.py` already validates most row shape and derives stable query IDs from answer positions, but it skips blank questions silently and does not detect duplicate IDs. The older `representative_redator_test.csv` has the right column names but its `question_texts` values are not valid v4 JSON arrays. The user-selected source, `datasets/redator_v4_repr-test_1k-val_2k-train_rest/redator_v4_repr-test_1k-val_2k-train_rest-test.csv`, has 1,000 CSV records and the v4 columns; it is ignored from version control and must be supplied by an operator elsewhere.

**Alternatives considered**: Accepting Python list literals would blur the v4 data contract. Including train/validation would violate the new source boundary. Depending on the ignored local dataset for automated tests would make those tests irreproducible, so the fixture remains test-only.

## Decision 3 — Preserve streaming preparation and explicit token sizing

**Decision**: Keep the one-source-row-at-a-time build and per-file incremental writers for the selected 1,000-row test source. Keep the `o200k_base` chunk profile with a 512-token cap, 320-token target, and 64-token overlap. Declare `tiktoken` directly as a v4 dependency and document the tokenizer-cache preflight.

**Rationale**: Streaming avoids unnecessary accumulation even on the test split and preserves the current v4 behavior. The current environment needed to fetch the tokenizer asset on first use. After it became available, `python -m pytest tests/v4 -q --tb=short` passed 112 tests.

**Alternatives considered**: The earlier word-count profile can exceed encoder limits on Portuguese legal text. Treating `tiktoken` as an undeclared transitive dependency makes clean installs unreliable.

## Decision 4 — Publish immutable generations and resolve one snapshot

**Decision**: Build each v4 generation in a new immutable directory, validate it, and switch one active-generation reference atomically. Retrieval and evaluation resolve that reference once at startup and use the resolved generation path for all files. Keep the legacy publication behavior isolated from this v4 contract.

**Rationale**: `scripts/artifact_publication.py` stages a build and rolls back failed replacement, but replaces five managed directories one at a time. Concurrent readers can therefore see a mixture during a successful publication, contrary to the spec's coherent-generation requirement. A single active-reference switch gives readers a stable snapshot while preserving earlier generations for rollback.

**Alternatives considered**: Continuing sequential directory replacement retains mixed-read risk. Moving the old root away before moving the new root avoids mixed files but briefly removes the public path. Per-file atomic writes do not make the whole generation atomic.

## Decision 5 — Keep citation labels independent of retrieval ranking

**Decision**: Retain cleaned-text evidence spans and map each accepted citation to exactly one passage by maximum overlap, breaking ties by earlier passage index. Emit corpus, questions, qrels, candidate sets, evidence, audits, and manifest together.

**Rationale**: `scripts/v4/ground_truth.py` implements this mapping; overlapping passages otherwise multiply positive labels. `scripts/v4/run.py` already emits the required artifacts. The explicit candidate set restricts a question to passages from its own source document.

**Alternatives considered**: Marking every overlapping passage relevant inflates recall. Recomputing relevance from a model would change the evidence source and reduce reproducibility.

## Decision 6 — Use a trusted remote vLLM embedding service with strict response validation

**Decision**: Keep the OpenAI-compatible `POST /v1/embeddings` request shape in `scripts/v4/retrieve.py`, pinned to `Qwen/Qwen3-Embedding-0.6B`. Send full cleaned question and passage text as the user approved. Require a configured trusted endpoint, protected transport or a private tunnel, authentication when exposed, and response validation for exact indices, exactly 1,024 numeric dimensions, and finite nonzero vectors. Retry transient failures only; fail clearly on permanent or malformed responses. Read credentials from an environment variable rather than a command-line argument in the target CLI contract.

**Rationale**: The current adapter batches text and sorts `data` by `index`, but only checks response length; duplicate or missing indices and invalid vectors can silently corrupt rankings or fail unclearly. It retries broad URL errors, including permanent client errors. The current `--api-key` argument can expose secrets in process listings. The [vLLM embedding documentation](https://docs.vllm.ai/en/latest/models/pooling_models/embed/) describes the embedding model requirement, and the [vLLM security guidance](https://docs.vllm.ai/en/latest/usage/security/) warns that its API key protects only part of the serving surface. The [embedding response schema](https://developers.openai.com/api/reference/ruby/resources/embeddings/methods/create) includes response indices.

**Alternatives considered**: Local in-process encoding remains available in code but is not the chosen deployment path. vLLM's `/v2/embed` has model-specific input behavior and would diverge from the existing adapter. Sending redacted text conflicts with the user's clarified full-text requirement. Requesting reduced Qwen embedding dimensions would add a server-configuration dependency, so this first index uses the full 1,024 dimensions.

## Decision 7 — Separate chunk indexing from query retrieval and publish runs atomically

**Decision**: Build chunk embeddings once per dataset generation and Qwen model configuration, then use the persisted pgvector rows for later query runs. Process questions in bounded request batches, restrict PostgreSQL vector search to the source document's published candidate IDs, and write rankings to a staged file before replacing the destination. Reject missing or cross-document candidates before remote requests. Keep query instruction, chunk-text profile, model identifier, normalization, score function, and result limit explicit in index/run metadata.

**Rationale**: `scripts/v4/retrieve.py` currently re-embeds all candidate chunks for each run, loads all corpus text and embeddings into memory, and writes JSON/TREC runs directly to the destination. A per-model persisted pgvector collection avoids repeated chunk requests. PostgreSQL filtering by build/document/candidate before exact cosine-distance ordering preserves within-document top-k; staged output avoids truncating an existing run.

**Alternatives considered**: Re-embedding per run wastes remote requests. Global top-k followed by document filtering is incorrect because it can omit the document's true top-k. HNSW/IVFFlat post-filtering is approximate and can under-return within-document candidates; defer it unless a later requirement permits recall loss.

## Decision 8 — Centralize evaluation and count missing questions as zero

**Decision**: Keep `scripts/v4/metrics.py` as the single evaluator for all model runs. Read the same qrels and cutoffs for each run; calculate NDCG, MAP, recall, precision, and MRR; count missing judged questions as zero. Reject malformed, empty, or unrelated run files with specific diagnostics.

**Rationale**: The existing evaluator already applies these rules and passes its unit tests. Central scoring prevents model-specific metric drift. Additional validation should reject non-finite scores and provide a stable comparison report for at least two runs.

**Alternatives considered**: Letting each retrieval command calculate metrics would duplicate logic and make comparisons less reliable. Ignoring missing questions would reward incomplete runs.

## Decision 9 — Use exact pgvector search per model and generation

**Decision**: Store validated, L2-normalized 1,024-dimensional chunk vectors in a PostgreSQL `vector(1024)` column, keyed by immutable index-build ID and chunk ID. Create a B-tree filter index on `(index_build_id, document_id)`; for each question, filter by build, source document, and published candidate IDs before ordering by pgvector cosine distance (`<=>`) and chunk ID for deterministic ties. Validate that the indexed chunk set exactly matches the generation corpus. Activate the build only after complete validation in a PostgreSQL transaction. No HNSW or IVFFlat vector ANN index is created in this iteration.

**Rationale**: The [pgvector documentation](https://github.com/pgvector/pgvector) supports 1,024 dimensions, exact nearest-neighbor search by default, cosine distance, and a B-tree filter index as a starting point for filtered exact search. It warns that HNSW/IVFFlat filtering happens after approximate index scanning and may return fewer matches. `vector(1024)` uses 4,104 bytes per vector before PostgreSQL row/index overhead; actual test corpus count must be measured before sizing the local database.

**Alternatives considered**: HNSW/IVFFlat can accelerate broad search but need recall and filter tuning and may change metrics; they are deferred. Global search followed by document filtering breaks document-scoped top-k. FAISS is removed by the user's explicit follow-up.

## Decision 10 — Pin Qwen text handling and embedding contract

**Decision**: Use `Qwen/Qwen3-Embedding-0.6B` only, without a reduced-dimensions request. Prefix queries with a versioned retrieval instruction in the [Qwen model-card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) format `Instruct: <task>\nQuery:<question>`; embed clean chunk text without that instruction. Pin the exact instruction, chunk text composition, model ID/revision, endpoint, 1,024 dimensions, and normalization policy in index/run provenance. Check actual served input limits before indexing or querying.

**Rationale**: Qwen recommends task instruction on queries but not documents; its full embedding dimension is 1,024. The [vLLM embedding documentation](https://docs.vllm.ai/en/latest/models/pooling_models/embed/) supports OpenAI-compatible embedding requests. Recording text preparation is essential because changed prefixes or titles produce incompatible vectors.

**Alternatives considered**: Reusing one generic prefix for both sides departs from Qwen guidance. Requesting a shorter matryoshka vector is unnecessary for the first index and may require vLLM server overrides.

## Decision 11 — Use local PostgreSQL as the sole vector and metadata store

**Decision**: Follow the user's latest explicit instruction: replace FAISS entirely with pgvector. Run a pinned pgvector-enabled PostgreSQL image in a local persistent container, execute `CREATE EXTENSION vector` in the application database, and keep `index_builds`, `indexed_chunks(embedding vector(1024))`, and an active `(generation_id, model_id)` pointer in the same database. Stage rows under a new build ID, validate coverage and provenance, then atomically switch the pointer in one PostgreSQL transaction. Readers pin the build ID and use immutable rows throughout a run.

**Rationale**: [pgvector](https://github.com/pgvector/pgvector) stores vectors in PostgreSQL and exposes exact vector-distance search. A single database removes file/database synchronization and uses PostgreSQL's [transaction visibility](https://www.postgresql.org/docs/current/mvcc-intro.html) for coherent activation. The image includes the extension; `CREATE EXTENSION vector` enables it per database. Persistent Compose storage retains rows across local container restarts.

**Alternatives considered**: The earlier FAISS-plus-PostgreSQL split is superseded by the user's pgvector choice. A FAISS process inside the PostgreSQL container would not be PostgreSQL-native indexing and adds lifecycle complexity. Building an approximate pgvector ANN index now would risk filtered ranking drift.

## Research exit

The design is constrained to the v4 test split and one Qwen pgvector build, preserves source-evidence labels and exact document-scoped ranking, and uses the user-selected remote vLLM service. PostgreSQL is the sole vector and metadata store. No technical-context decision remains unresolved.
