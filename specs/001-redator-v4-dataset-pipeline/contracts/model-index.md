# Contract: Qwen Chunk Index in PostgreSQL/pgvector

**Status**: Target design for the v4 test-only feature. The user's latest instruction replaces FAISS with pgvector. PostgreSQL is the sole store for chunk embeddings, identity, and index-build provenance.

## Scope and deployment

- Exactly one model is eligible for index creation now: `Qwen/Qwen3-Embedding-0.6B`.
- A local container uses a pinned pgvector-enabled PostgreSQL image and persistent named volume. Initialize the application database with `CREATE EXTENSION IF NOT EXISTS vector`. The trusted remote vLLM server remains separate and is not provisioned by this stack.
- One immutable index build belongs to exactly one immutable v4 `test` dataset generation and one Qwen model/text/normalization profile. Per-model identity allows later extension but no other model is indexed now.
- No FAISS library, sidecar, index file, filesystem-to-database pointer, or approximate pgvector HNSW/IVFFlat index is required in this iteration.

## Embedding contract

The builder reads every published `beir/corpus.jsonl` chunk exactly once in deterministic `(document_id, chunk_index, chunk_id)` order and sends its `text` field, without title or query instruction, to trusted remote vLLM `POST /v1/embeddings`. The request model is exactly `Qwen/Qwen3-Embedding-0.6B`; do not request reduced `dimensions`. Full cleaned chunk text is allowed by the user's trust decision. Requests are bounded batches; responses have exactly one unique index per input, each vector has 1,024 finite numeric values and nonzero norm. Reject duplicates, missing indices, changed dimensions, non-finite values, and zero vectors before publication. L2-normalize vectors before insertion into `vector(1024)`.

The versioned query instruction is exactly:

```text
Instruct: Retrieve the passage from the same legal document that answers the question.
Query:<question>
```

The question's full cleaned text replaces `<question>` without altering its content. Query responses follow the same dimension/finite/nonzero validation and L2 normalization. A change to the instruction, chunk-text profile, model revision, or normalization policy creates an incompatible new build; retrieval never silently reuses an old build.

## PostgreSQL relations and indexes

| Relation | Key fields and checks |
|---|---|
| `index_builds` | `index_build_id` PK; `model_id`, `model_revision`, `generation_id`, `generation_manifest_sha256`, `source_sha256`, `split='test'`, `dimension=1024`, `metric='cosine'`, `normalized=true`, `query_instruction`, `chunk_text_profile`, `chunk_count`, `status`, timestamps. `status` is `building`, `validated`, `active`, `superseded`, or `failed`. |
| `indexed_chunks` | PK `(index_build_id, chunk_id)`; FK to `index_builds`; `document_id`, `chunk_index`, `embedding vector(1024) NOT NULL`; unique `(index_build_id, document_id, chunk_index)`. Exactly one row per published corpus chunk. |
| `active_model_indexes` | PK `(generation_id, model_id)`; FK to one complete `index_build_id` with matching generation/model. The pointer changes in one transaction after validation. |

Create a B-tree on `(index_build_id, document_id)` (optionally including `chunk_id`) to narrow exact searches before cosine-distance ordering. A pgvector `vector(1024)` column stores the embeddings; a B-tree filter index is the initial physical index, not an HNSW/IVFFlat approximate-neighbor index. Do not duplicate full source text or credentials in PostgreSQL. Rows for a validated build are immutable.

## Build activation and recovery

The builder stages bounded vector batches under a new `index_build_id` that no reader selects. After insertion it verifies PostgreSQL row count equals `manifest.chunks`, indexed chunk IDs are a bijection with the published corpus, each chunk's document/position matches the corpus, stored vectors have the required dimension and nonzero norm, and the model/source/generation/text profile matches the manifest. Then one PostgreSQL transaction marks the build active and switches `active_model_indexes`. A failed build leaves the previous compatible build selected and never exposes partial rows. A superseded or failed build is retained until a separate cleanup policy is defined.

Readers resolve dataset `current` and the active `(generation_id, model_id)` build once, then verify generation ID and manifest checksum, model/text profile, dimension, and row count before issuing queries. If the dataset generation changes without a compatible active build, retrieval fails as `index_not_ready`. Missing/incomplete PostgreSQL rows or mismatched provenance fail closed. After a container restart, the persistent database yields the same active build and results.

## Exact document-scoped search

For each retained question, use its published `document_id` and candidate IDs. The query embeds the instructed question, then selects only rows for the pinned build, document, and candidate IDs before ordering by pgvector cosine distance. Conceptually:

```sql
SELECT chunk_id, 1 - (embedding <=> :query_vector) AS score
FROM indexed_chunks
WHERE index_build_id = :build_id
  AND document_id = :document_id
  AND chunk_id = ANY(:candidate_ids)
ORDER BY embedding <=> :query_vector, chunk_id
LIMIT :top_k;
```

The implementation parameterizes inputs and validates that returned IDs belong to the published candidate set, scores are finite, IDs are unique, and result count is at most `top_k`. The score is cosine similarity, higher is better. Filtering must occur before exact ordering; global top-k followed by document filtering is prohibited. HNSW/IVFFlat are deferred because pgvector applies their filters after approximate scanning, which can under-return for a document and alter benchmark metrics.

## Verification cases

1. Synthetic fixture: PostgreSQL row count equals fixture corpus count; chunk IDs map one-to-one and candidates resolve to the correct document.
2. Real v4 test dataset: manifest source SHA-256 matches [dataset.md](dataset.md), and no train/validation identity enters the table.
3. Rebuild interruption or invalid vLLM response: prior compatible build remains active and searchable; no partial build is selected.
4. PostgreSQL container restart: active build remains available and gives the same deterministic within-document ranking.
5. Dataset generation, model ID, instruction, dimension, row count, or candidate mapping mismatch: retrieval fails before publishing a run.

References: [pgvector types, exact search, filtering, and Docker setup](https://github.com/pgvector/pgvector); [PostgreSQL transaction visibility](https://www.postgresql.org/docs/current/mvcc-intro.html); [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
