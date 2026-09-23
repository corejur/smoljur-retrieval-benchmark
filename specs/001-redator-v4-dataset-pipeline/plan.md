# Implementation Plan: Representative Redator v4 Dataset Pipeline

**Branch**: `001-redator-v4-dataset-pipeline` (feature integration branch; supersedes the manually created `v4-dataset-pipeline`; see [Branch Structure](#branch-structure)) | **Date**: 2026-09-22 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-redator-v4-dataset-pipeline/spec.md`

**Design artifacts**: [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

## Summary

Complete the v4 Representative Redator workflow for **only the 1,000-row test CSV**: citation-grounded BEIR artifacts, one persistent `Qwen/Qwen3-Embedding-0.6B` pgvector collection in local PostgreSQL, document-scoped ranking through a trusted remote vLLM embedding service, and common metric comparison. Preserve the working v4 preparation and evaluation modules. Add test-source validation, immutable generation and index publication, strict remote-response checks, and atomic run output where the current code falls short of the spec. See [research.md](research.md) for decisions and alternatives.

## Technical Context

**Language/Version**: Python >=3.9 per `pyproject.toml`; validate v4 on supported versions during implementation. The current verification environment uses Python 3.13.

**Primary Dependencies**: Existing BEIR evaluation stack, NumPy, `tiktoken` for v4 token counts, a PostgreSQL client with pgvector adaptation, a local PostgreSQL image with the pgvector extension, and an operator-supplied trusted remote vLLM service hosting `Qwen/Qwen3-Embedding-0.6B`. Declare direct v4 dependencies explicitly rather than relying on transitive installs.

**Storage**: The single v4 test CSV; versioned local JSONL/TSV dataset files; local containerized PostgreSQL with pgvector embeddings, chunk identity, and build provenance in one database; ranking runs and comparison reports as files. No FAISS dependency, file, or separate vector store.

**Testing**: `python -m pytest tests/v4 -q --tb=short` plus contract tests for test-source fingerprint/split, artifact schemas, Qwen response validation, one-vector-per-chunk coverage in PostgreSQL, build/generation compatibility, exact document-filtered top-k, container restart, failed-build recovery, and publication snapshots. Baseline: 112 existing v4 tests passed after the tokenizer asset became available.

**Target Platform**: Repository-root Python CLI on Linux/POSIX, local PostgreSQL container with persistent storage, and trusted remote vLLM service reachable over an FR-026-conforming channel: `https://` with certificate verification, or a loopback/tunnel-local address behind an operator-established encrypted tunnel.

**Project Type**: Research dataset preparation and evaluation CLI within the BEIR repository.

**Performance Goals**: Preparation and index encoding should stream through the 1,000-row test source in bounded batches. pgvector stores 1,024-dimensional vectors in PostgreSQL; a B-tree on `(index_build_id, document_id)` narrows exact cosine-distance ranking to one document's candidates. Measure published test chunk counts, query latency, and database size before setting container limits. Query embedding requests are batched. The spec defines no elapsed-time SLA.

**Constraints**: Only rows from the v4 test source may enter this dataset; source evidence, not model inference, defines relevance. Only Qwen is indexed now, though index identity is model-specific. Evaluation remains document-scoped and exact: approximate HNSW/IVFFlat are deferred because post-filtering can change top-k. The trusted remote service may receive full cleaned text. Readers use one immutable dataset generation and one compatible PostgreSQL index build per run; failed builds or remote responses do not expose partial published artifacts. v1–v3 behavior stays unchanged.

**Scale/Scope**: One v4 test generation with corpus, queries, qrels, candidates, cleaned documents, evidence, audits, and manifest; one Qwen index build containing every published chunk; one or more ranking runs and comparison across at least two runs/configurations. Train/validation generation and other model indexes are out of scope. Versioned dataset and index builds are retained until a separate cleanup policy is defined.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The current `.specify/memory/constitution.md` is an unratified template containing placeholders, not enforceable project principles. No constitution gate can be applied yet. The plan still checks the explicit spec constraints: v4 test-only scope, traceable source evidence, coherent dataset/pgvector-build publication, trusted full-text remote vLLM use, Qwen-only index, and comparable scoring. **Pre-research gate: PASS**; no conflicting ratified principle was found.

**Post-design gate: PASS**. The data model keeps citation provenance and stable IDs; contracts define one generation and compatible index snapshot, document-scoped retrieval, remote-service failures, and common metric inputs. No exception to an adopted constitution is requested. Ratifying a project constitution remains separate work.

## Project Structure

### Documentation (this feature)

```text
specs/001-redator-v4-dataset-pipeline/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/requirements.md
├── contracts/
│   ├── dataset.md
│   ├── workflow-cli.md
│   ├── model-index.md
│   └── fixture-v4.csv
└── tasks.md                    # Produced by speckit-tasks, not this phase
```

### Source Code (repository root)
```text
scripts/
├── v4/
│   ├── questions.py            # Source validation and question identity
│   ├── cleaning.py             # Clean text with offset mapping
│   ├── offsets.py              # Source-to-cleaned offset mapping
│   ├── headings.py             # Structural heading detection
│   ├── ground_truth.py         # Citation spans and qrels
│   ├── chunking.py             # Token-bounded passages
│   ├── run.py                  # Generation orchestration and CLI
│   ├── streams.py              # Incremental artifact writers
│   ├── retrieve.py             # Document-scoped Qwen-index ranking
│   ├── metrics.py              # Shared run evaluation
│   ├── manifest.py             # Module registry: the authoritative v4 file set
│   ├── generation.py           # Planned generation-manifest schema and `current` resolution
│   ├── source_contract.py      # Planned test-source header/fingerprint/split gate
│   ├── publication.py          # Planned v4 immutable-generation publisher
│   ├── embeddings.py           # Planned trusted remote vLLM adapter
│   ├── pgvector_store.py       # Planned PostgreSQL/pgvector store and exact search
│   ├── index.py                # Planned Qwen pgvector build and activation
│   ├── plots.py                # Mean-metric charts from a metrics comparison report
│   └── sql/
│       └── 001_vector_schema.sql   # Planned idempotent pgvector migration
├── artifact_publication.py     # Existing shared legacy publisher
├── source_normalization.py    # Reused source HTML parser
└── strategies/chunking/legal_recursive.py
tests/
└── v4/                       # Existing tests; add contract/failure/scale coverage
compose.yaml                   # Planned local pgvector-enabled PostgreSQL service and volume
```

**Structure Decision**: Keep v4 work in `scripts/v4/` and its tests in `tests/v4/`. A v4-specific publication module avoids changing the shared legacy publication contract. `scripts/v4/manifest.py` must declare any new v4 module. CLI, dataset, and index contracts live under this feature's `contracts/` directory. The local PostgreSQL deployment is separate from the remote vLLM server.

**Naming**: three distinct things in this feature would otherwise all be called "manifest" and MUST NOT be conflated. `scripts/v4/manifest.py` is the **module registry** — which source files belong to the v4 build. `meta/manifest.json` is the **generation manifest** — one published dataset's identity, counts, and configuration. `scripts/v4/generation.py` implements and validates the *generation manifest* and owns one-time `current` resolution; it does not extend the module registry. Prose in this feature uses "module registry" or "generation manifest"; the bare word "manifest" is reserved for neither.

## Branch Structure

Branches follow the story dependency graph in [tasks.md](tasks.md). Each story branch starts from the feature branch once its dependencies are merged, and merges back into it; the feature branch merges into `main` once, after polish.

| Branch | Base | Tasks | Status |
|---|---|---|---|
| `001-redator-v4-dataset-pipeline` | `main` | T001–T010 (Setup, Foundation, US1 MVP), then T034–T038 (Polish, Convergence) | Integration branch; US1 committed |
| `001-us2-source-audits` | feature branch after US1 | T011–T015 | Ready |
| `001-us3-generation-publication` | feature branch after US1 | T016–T019 | Ready |
| `001-us4-pgvector-retrieval` | feature branch after US3 merges | T020–T030 | Created when US3 merges |
| `001-us5-run-comparison` | `001-us3-generation-publication` | T031–T033 | Based on US3 for `locate_generation`; merge after US3 (live-run comparison waits for US4) |

```text
main
 └─ 001-redator-v4-dataset-pipeline   Setup → Foundation → US1 ─────────────── Polish → main
     ├─ 001-us2-source-audits              └→ US2 ──────────────┐
     ├─ 001-us3-generation-publication     └→ US3 ─┐            │
     │   └─ 001-us4-pgvector-retrieval             └→ US4 ──────┤
     └─ 001-us5-run-comparison             └→ US5 ──────────────┘
```

US2 and US3 both edit `scripts/v4/run.py`; merge whichever finishes first, then rebase the other onto the feature branch before merging it.

## Design Approach

| Spec area | Current repository behavior | Planned design |
|---|---|---|
| Source and evidence (FR-001–008) | Row streaming, ordered questions, cleaned citation spans, token chunks, and one passage per citation exist. Any supplied CSV/split is accepted; blank questions are skipped and duplicate document IDs are not rejected. | Gate production input to the 1,000-row v4 test CSV and `test` split using path/fingerprint/row-count provenance; add identity/blank-question audits while preserving evidence mapping. |
| Generation publication (FR-009–011) | Build staging and rollback exist, but managed directories are swapped one at a time. | Publish an immutable generation and switch a single active reference; consumers resolve that reference once before reading. |
| Qwen index (FR-013, FR-022–025) | No v4 persistent chunk index or PostgreSQL embeddings exist; retrieval re-embeds candidate chunks. | Encode every published test chunk exactly once per immutable Qwen build; store `vector(1024)` with chunk identity and model/generation provenance in pgvector; activate only a complete compatible build. |
| Document-scoped retrieval (FR-012–014, FR-024) | Candidate filtering and top-k exist; all corpus text and embeddings are loaded together. | Embed queries only; filter PostgreSQL rows to the compatible build and published source-document candidates, then order by exact pgvector cosine distance. Validate returned IDs and reject stale/incomplete builds. |
| Trusted remote vLLM (FR-020–021) | The adapter batches embedding requests but checks only response length and accepts direct run-file writes. | Pin Qwen at 1,024 dimensions and version its query instruction; validate response indices/vectors, distinguish retryable errors, enforce the FR-026 endpoint check before the first request, hold credentials in the environment only, and publish run files atomically. |
| Common metrics (FR-015–019) | One evaluator scores NDCG, MAP, recall, precision, and MRR; missing judged questions count as zero. | Preserve the shared evaluator, tighten malformed-score validation, and retain comparison provenance across ranking runs; other models remain future scope. |

Phase 0 decisions are in [research.md](research.md). Phase 1 schemas and state transitions are in [data-model.md](data-model.md); user and external interfaces are in [contracts/](contracts/); runnable validation is in [quickstart.md](quickstart.md). Task decomposition follows in `$speckit-tasks`.
