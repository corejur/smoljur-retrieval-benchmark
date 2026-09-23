---
description: "Dependency-ordered implementation tasks for the Representative Redator v4 test pipeline"
---

# Tasks: Representative Redator v4 Dataset Pipeline

**Input**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/), and [quickstart.md](quickstart.md).

**Tests**: Included because each story specifies an independent test and the plan calls for source, publication, pgvector, remote-failure, and metrics contract coverage. Write the story's test tasks before its implementation tasks; new behavior should fail before implementation, while regression assertions for existing behavior may already pass.

**Organization**: Tasks are grouped by the five priority-ordered user stories. Paths are repository-relative. `[P]` means a task may run concurrently with other `[P]` tasks in its phase because it touches different files and has no dependency on their incomplete work.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Declare direct v4 dependencies and provision only the local storage service; do not start a real remote model or change v1–v3 code.

- [X] T001 [P] Add direct v4 dependencies for `tiktoken`, a PostgreSQL client, and its pgvector adaptation to `pyproject.toml`; keep Python >=3.9 compatibility and do not add FAISS as a v4 requirement.
- [X] T002 [P] Add a pinned pgvector-enabled PostgreSQL service, health check, persistent named volume, and environment-supplied credentials to `compose.yaml`; do not hard-code passwords or provision the trusted remote vLLM server.

**Checkpoint**: The Python environment can install direct dependencies, and the local PostgreSQL service can be started without embedding any credentials in the repository.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Provide one immutable-generation identity and reader contract shared by preparation, publication, indexing, and evaluation.

- [X] T003 Add failing tests in `tests/v4/test_generation.py` for a generation manifest (`meta/manifest.json` — not the `scripts/v4/manifest.py` module registry; see the plan's Structure Decision naming note) with `pipeline='redator-v4'`, split `test`, generation ID, source SHA-256, row count, summary counts, required file paths, and a single resolved snapshot; reject missing files, broken references, or mixed generation IDs.
- [X] T004 Implement the tested generation-manifest schema, required-artifact enumeration, cross-file ID/count validation, and one-time `current` snapshot resolution in `scripts/v4/generation.py`; leave `scripts/v4/manifest.py` as the module registry and do not merge the two concepts; preserve query, qrel, candidate, document, and passage identities from the existing v4 output.

**Checkpoint**: A complete synthetic generation validates as one snapshot; corrupt or internally inconsistent artifacts fail before a consumer starts.

---

## Phase 3: User Story 1 — Build a Usable Retrieval Dataset (Priority: P1) 🎯 MVP

**Goal**: Produce citation-grounded BEIR artifacts from only the v4 test source, with stable identities and a manifest proving the source boundary.

**Independent Test**: The portable synthetic fixture yields resolvable corpus/query/qrel/candidate IDs, while the production gate accepts only the 1,000-row v4 test CSV (or a byte-identical copy) and rejects train/validation or changed bytes.

### Tests for User Story 1

- [X] T005 [P] [US1] Add failing production-source contract tests in `tests/v4/test_source_contract.py` for required CSV columns `id`, `data`, `question_texts`, `output`; unique nonempty document IDs; ordered equal-length JSON arrays; split `test`; 1,000 rows; and SHA-256 `d10d21f2074e48576cb715bc65ef01a36f369bf837dd2c404280775cef56fff3`, with an injectable fixture fingerprint for portable tests; include a `-test`-named file with altered bytes and a train/validation-named file carrying the expected bytes, proving the SHA-256 is the authoritative gate and the filename is advisory only.
- [X] T006 [P] [US1] Add dataset-integrity regression and missing-contract cases in `tests/v4/test_run.py` for stable `{document_id}:q{zero_based_answer_index}` IDs, ordered token-bounded passages, one maximum-overlap passage per accepted citation (earlier passage wins ties), nonempty candidate sets from the source document, and positive-score qrels referencing only published candidate chunks.

### Implementation for User Story 1

- [X] T007 [US1] Implement streaming v4 test-source header, row-count, SHA-256, split, and duplicate-ID validation in `scripts/v4/source_contract.py`; the production gate must reject train/validation and altered input while the lower-level builder remains usable with the synthetic fixture.
- [X] T008 [US1] Wire the production source gate into `scripts/v4/run.py`, fix the production split to `test`, retain the existing `o200k_base` 512-token max/320 target/64 overlap profile, and record every required generation-manifest key from the [dataset contract](contracts/dataset.md) in `meta/manifest.json` — `pipeline`, `split`, `generation_id`, `generated_at`, `source_sha256`, `source_row_count`, the chunking and policy block, and the full `summary` counter object including `summary.chunks`.
- [X] T009 [US1] Enforce the [dataset contract](contracts/dataset.md) in `scripts/v4/generation.py`: unique corpus/query IDs, one candidate record and at least one qrel per retained query, qrels contained in its candidate set, candidate/document provenance, half-open passage offsets, and manifest counts matching emitted records.
- [X] T010 [US1] Call generation validation before considering a build successful in `scripts/v4/run.py`; keep source-evidence labels independent of model inference and preserve the synthetic fixture's expected 2 documents, 2 passages, 2 retained questions, and 2 judgments.

**Checkpoint**: User Story 1 passes its fixture tests and the real test-source preflight; no train or validation row can enter a production generation.

---

## Phase 4: User Story 2 — Audit Excluded Source Material (Priority: P2)

**Goal**: Every excluded source or question has its identity and a specific reason without losing valid sibling questions.

**Independent Test**: A synthetic collection with malformed arrays, empty cleaned text, missing answer/citation, unmapped evidence, too many gold passages, duplicate document IDs, and blank questions yields the expected retained records plus reasoned audits.

### Tests for User Story 2

- [ ] T011 [P] [US2] Add failing question-level cases in `tests/v4/test_questions.py` for blank questions audited at their original answer position, unchanged later query IDs, unreadable `question_texts`/`output`, and unequal array lengths.
- [ ] T012 [P] [US2] Add failing source/query audit cases in `tests/v4/test_audits.py` for duplicate IDs, empty cleaned documents, no passages, missing answer text, missing or unmapped citations, `max_gold_chunks=10` by default, and whole-document exclusion only when no question remains usable.

### Implementation for User Story 2

- [ ] T013 [US2] Extend `scripts/v4/questions.py` to expose blank-question drops with `{document_id}:q{original_index}` rather than silently skipping them, while retaining ordered `question_texts` and `output` alignment.
- [ ] T014 [US2] Emit identified, reason-coded rejected-source, dropped-query, unmapped-citation, and cleaning audit records in `scripts/v4/run.py`; retain a document with at least one valid question, exclude one with none under the default policy, and reconcile exclusion counters with the manifest.
- [ ] T015 [US2] Ensure the audit JSONL writers in `scripts/v4/streams.py` preserve one valid record per line and flush/close on both success and failure so audits can be validated with the rest of the generation.

**Checkpoint**: Every fixture exclusion is attributable to an original source row or question ID and a specific reason; valid sibling questions remain published.

---

## Phase 5: User Story 3 — Publish a Complete Generation (Priority: P3)

**Goal**: Readers see one complete immutable dataset generation, and a failed rebuild leaves the previous one unchanged.

**Independent Test**: Inject a build or activation failure after one generation is active; readers continue to load the old complete snapshot. A successful replacement switches all artifacts together.

### Tests for User Story 3

- [ ] T016 [US3] Add failing failure-injection and concurrent-reader tests in `tests/v4/test_publication.py` for `building -> validated -> active -> superseded`, rollback to the previous active generation, and no mixture of `beir/`, `documents/`, `evidence/`, `audits/`, or `meta/` files during a successful switch.

### Implementation for User Story 3

- [ ] T017 [US3] Implement a v4-only immutable `generations/<generation-id>/` publisher and single atomic `current` reference switch in `scripts/v4/publication.py`; validate complete artifacts before activation and never replace managed directories sequentially.
- [ ] T018 [US3] Replace the v4 call to shared `scripts/artifact_publication.py` with the v4 publisher in `scripts/v4/run.py`; leave the shared v1–v3 publication behavior untouched and report the active generation ID and counts.
- [ ] T019 [US3] Make consumers pin `current` once and reject missing/incompatible generation files in `scripts/v4/generation.py`; keep superseded generations available until a separately defined cleanup policy exists.

**Checkpoint**: One stable generation snapshot survives failed and successful rebuilds without cross-generation reads.

---

## Phase 6: User Story 4 — Index and Rank Test Passages (Priority: P4)

**Goal**: Store one Qwen `vector(1024)` embedding for every published test chunk in local PostgreSQL/pgvector, activate only complete compatible builds, and rank each query's own candidate passages exactly.

**Independent Test**: Against the synthetic generation and a mocked remote vLLM service, the index has one row per chunk, survives a PostgreSQL restart, and returns deterministic within-document top-k; malformed vectors, stale builds, and service failures publish neither a partial index nor a partial run.

### Tests for User Story 4

- [ ] T020 [P] [US4] Add failing PostgreSQL schema and exact-search tests in `tests/v4/test_pgvector_store.py` for `indexed_chunks.embedding vector(1024) NOT NULL`, PK `(index_build_id, chunk_id)`, unique `(index_build_id, document_id, chunk_index)`, B-tree `(index_build_id, document_id)`, active pointer PK `(generation_id, model_id)`, and cosine ordering after build/document/candidate filtering.
- [ ] T021 [P] [US4] Add failing remote-response tests in `tests/v4/test_embeddings.py` for Qwen-only model ID, exact response indices `0..N-1`, 1,024 finite numeric values, nonzero norm, L2 normalization, bounded batches, transient-only retries, FR-026 endpoint rejection before any text is sent (plaintext `http://` outside a loopback/tunnel-local address, or certificate verification disabled), and errors that never print full source text or credentials.
- [ ] T022 [P] [US4] Add failing build/recovery tests in `tests/v4/test_index.py` for one embedding per published chunk, no other model index, exactly `summary.chunks` rows as counted from the generation manifest, compatible generation/model/revision/text profile, failed rebuild retaining the prior active build, and container restart persistence.
- [ ] T023 [P] [US4] Add failing retrieval tests in `tests/v4/test_retrieve_pgvector.py` for the exact versioned instruction `Instruct: Retrieve the passage from the same legal document that answers the question.\nQuery:<question>`, query-only embedding, document/candidate-only top-k, deterministic score/ID ties, missing candidates, stale generation/build, invalid query vectors, service failure, and unchanged existing run on failure.
- [ ] T024 [P] [US4] Add a failing local-container integration test in `tests/v4/test_pgvector_integration.py` that starts from `compose.yaml`, verifies `CREATE EXTENSION vector`, builds fixture embeddings against a mocked vLLM endpoint, restarts PostgreSQL, and confirms the same active build and within-document rankings; require this test in the release validation path rather than silently skipping it.

### Implementation for User Story 4

- [ ] T025 [US4] Add an idempotent PostgreSQL migration in `scripts/v4/sql/001_vector_schema.sql` that enables `vector`, creates `index_builds`, `indexed_chunks`, and `active_model_indexes` per [model-index.md](contracts/model-index.md), and creates the B-tree filter index without HNSW/IVFFlat.
- [ ] T026 [P] [US4] Extract the shared trusted-remote vLLM embedding adapter into `scripts/v4/embeddings.py`: require `Qwen/Qwen3-Embedding-0.6B`, use `/v1/embeddings`, full cleaned chunk text or instructed query text, environment-held API key, FR-026 transport validation on the first call (HTTPS with certificate verification, or a loopback/tunnel-local tunnel address), strict response validation, and no reduced-dimensions request.
- [ ] T027 [US4] Implement PostgreSQL connection/migration, bounded staged inserts, corpus-ID bijection checks, immutable build rows, active-pointer transaction, and exact `embedding <=> query` candidate-filtered search in `scripts/v4/pgvector_store.py`; reject incomplete or incompatible builds with identifiable errors.
- [ ] T028 [US4] Implement `python -m scripts.v4.index DATASET_ROOT --model Qwen/Qwen3-Embedding-0.6B --base-url ... --postgres-dsn-env V4_POSTGRES_DSN` in `scripts/v4/index.py`; send each chunk text once in bounded batches, store normalized vectors, validate row coverage/provenance, then switch the active build in one PostgreSQL transaction.
- [ ] T029 [US4] Replace corpus-wide re-embedding and in-memory ranking in `scripts/v4/retrieve.py` with one pinned compatible pgvector build, exact cosine-distance search scoped by build/document/published candidate IDs, finite scores, `top_k > 0`, and atomic JSON/TREC run-file replacement.
- [ ] T030 [US4] Record generation ID, index-build ID, Qwen model/revision, query instruction, chunk-text profile, normalization/score mode, split, and top-k in run provenance outside `RUNS_DIRECTORY` in `scripts/v4/retrieve.py`; read database and API credentials from environment rather than CLI arguments.

**Checkpoint**: Every corpus chunk has one compatible pgvector row, only Qwen is indexed, and retrieval emits complete document-scoped rankings using stored chunks plus fresh query vectors.

---

## Phase 7: User Story 5 — Compare Retrieval Runs Fairly (Priority: P5)

**Goal**: Evaluate two or more Qwen ranking runs against one qrels set and common cutoffs with missing judged questions scored as zero.

**Independent Test**: Two supplied valid runs from the same test generation receive separate NDCG, MAP, recall, precision, and MRR results; an omitted judged query contributes zero, while empty, malformed, non-finite, or unrelated runs fail clearly.

### Tests for User Story 5

- [ ] T031 [US5] Add failing comparison tests in `tests/v4/test_metrics.py` for two runs sharing qrels/cutoffs, zero-filled missing judged queries and `queries_missing`, NDCG/MAP/Recall/Precision/MRR at requested k, rejection of empty/malformed/non-finite/unrelated runs, and report files not mistaken for runs.

### Implementation for User Story 5

- [ ] T032 [US5] Tighten JSON/TREC score and structure validation in `scripts/v4/metrics.py`; keep one evaluator and the same qrels/cutoffs for every run, reject non-finite scores and no-common-query runs, and preserve zero-scoring missing judged questions.
- [ ] T033 [US5] Produce a comparison report with generation/judgment provenance, common cutoffs, per-run measures, `queries_scored`, and `queries_missing` in `scripts/v4/metrics.py`; support two Qwen runs with different top-k values without requiring a second indexed model.

**Checkpoint**: Two ranking runs are comparable without model-specific metric code, and omissions cannot inflate aggregate scores.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Make the completed v4 workflow discoverable, reproducible, and verifiable without changing legacy generations.

- [ ] T034 Update `scripts/v4/manifest.py` to declare all new v4 Python modules (`generation.py`, `source_contract.py`, `publication.py`, `embeddings.py`, `pgvector_store.py`, `index.py`) and keep `verify()` passing without importing v1–v3 excluded modules.
- [ ] T035 [P] Replace target-only instructions with the implemented source-build, PostgreSQL startup, Qwen index, retrieval, and metrics commands in `specs/001-redator-v4-dataset-pipeline/quickstart.md`; distinguish portable synthetic tests from the ignored real test CSV and document trusted remote full-text transfer.
- [ ] T036 Run `python -m pytest tests/v4 -q --tb=short`, the required container integration path, and the synthetic end-to-end quickstart; record the pass/fail outcome and any external-service prerequisite in `specs/001-redator-v4-dataset-pipeline/quickstart.md` without claiming an unrun real vLLM test passed.
- [ ] T037 Verify the real 1,000-row test-source checksum, published counts, pgvector row count, document-scoped query latency, and PostgreSQL volume size where the operator provides the source/service; record measured results or an explicit unavailable prerequisite in `specs/001-redator-v4-dataset-pipeline/quickstart.md`.

**Checkpoint**: The v4 manifest, tests, container contract, and run guide agree; no legacy v1–v3 pipeline is modified.

---

## Dependencies & Execution Order

### Phase Dependencies

- Setup (T001–T002) can start immediately; T001 and T002 are independent.
- Foundation (T003–T004) follows setup and blocks all story work; T003 precedes T004.
- User Story 1 (T005–T010) follows foundation and is the MVP.
- User Story 2 (T011–T015) and User Story 3 (T016–T019) require User Story 1's published-record contract but can proceed concurrently with each other on separate files after coordinating their edits to `scripts/v4/run.py`.
- User Story 4 (T020–T030) requires User Stories 1 and 3 for a coherent immutable test generation. Its schema and embedding-adapter test tasks may proceed concurrently; schema/store and embedding adapter must finish before index and retrieval integration.
- User Story 5 (T031–T033) requires User Story 1's qrels but can use supplied run files independently of User Story 4's live index. End-to-end comparison of generated runs follows User Story 4.
- Polish (T034–T037) follows the selected story scope; final validation of all five stories requires T001–T033.

### User Story Dependency Graph

```text
Setup → Foundation → US1 (MVP)
                         ├──→ US2
                         ├──→ US3 ──→ US4
                         └──→ US5 (supplied runs)
US4 + US5 ──→ end-to-end comparison and polish
```

### Within Each Story

- Write the listed tests first and observe the missing-behavior assertions fail; existing-behavior regression assertions may already pass. Then implement the contract in dependency order.
- US1: T005/T006 → T007 → T008 → T009 → T010.
- US2: T011/T012 → T013 → T014 → T015.
- US3: T016 → T017 → T018 → T019.
- US4: T020–T024 (tests) → T025/T026 → T027 → T028 → T029 → T030. T027 depends on T025; T028 depends on T026 and T027; T029 depends on T028.
- US5: T031 → T032 → T033.

### Parallel Opportunities

- Setup: T001 (`pyproject.toml`) and T002 (`compose.yaml`).
- US1: T005 (`tests/v4/test_source_contract.py`) and T006 (`tests/v4/test_run.py`).
- US2: T011 (`tests/v4/test_questions.py`) and T012 (`tests/v4/test_audits.py`).
- US4: T020–T024 each write a different test file; after they fail as expected, T025 (`scripts/v4/sql/001_vector_schema.sql`) and T026 (`scripts/v4/embeddings.py`) touch separate files.
- Polish: T035 can update `quickstart.md` while T034 updates `manifest.py`; T036/T037 must wait for T035 and the implementation they verify.

### Parallel Example: User Story 1

```text
Task T005: write source-only validation tests in tests/v4/test_source_contract.py
Task T006: write dataset referential-integrity tests in tests/v4/test_run.py
```

### Parallel Example: User Story 2

```text
Task T011: write question-position tests in tests/v4/test_questions.py
Task T012: write audit-generation tests in tests/v4/test_audits.py
```

### Parallel Example: User Story 3

```text
After T016, T017 implements scripts/v4/publication.py while a reviewer can independently inspect the failure cases in tests/v4/test_publication.py. T018 and T019 are sequential integration tasks.
```

### Parallel Example: User Story 4

```text
Task T020: write PostgreSQL schema/search tests in tests/v4/test_pgvector_store.py
Task T021: write remote embedding contract tests in tests/v4/test_embeddings.py
After all US4 tests are written, T025 (SQL migration) and T026 (embedding adapter) can be implemented concurrently.
```

### Parallel Example: User Story 5

```text
Once US1 produces qrels, T031 can test scripts/v4/metrics.py with supplied run fixtures while US4's pgvector integration proceeds separately. T032 and T033 then complete US5 in order.
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Setup (T001–T002) and Foundation (T003–T004).
2. Complete US1 (T005–T010) and verify its synthetic fixture and real-source preflight independently.
3. Stop and validate corpus/query/qrel/candidate consistency and the `test`-only source boundary. Production atomic publication is the next US3 increment, not silently claimed by this MVP.

### Incremental Delivery

1. Add US2 audits and verify exclusions without losing valid sibling questions.
2. Add US3 immutable publication and failure-safe snapshot reads.
3. Add US4 pgvector schema, Qwen embedding build, and exact document-scoped retrieval.
4. Add US5 common metrics; its supplied-run tests can be completed earlier, but live-run comparison follows US4.
5. Finish polish, run the complete v4 suite, and record which external acceptance checks were actually executed.

## Notes

- All paths are repository-relative; the real v4 test CSV is ignored by Git and is not a portable automated-test prerequisite.
- `[P]` marks only different-file work without an incomplete-task dependency. Tasks that edit `scripts/v4/run.py` across stories require coordination and are not marked `[P]`.
- Model training, indexes for non-Qwen models, HNSW/IVFFlat approximate search, cross-document retrieval, and v1–v3 changes are out of scope.

---

## Phase 9: Convergence

**Purpose**: Close gaps found by assessing the current code against this feature's artifacts that no task in Phases 1–8 covers. Phases 1–8 remain the source of the unstarted implementation work; nothing there is restated here.

- [ ] T038 Extend `scripts/v4/manifest.py` so the module registry and `verify()` cover non-Python v4 build artifacts — recursing below `scripts/v4/` and declaring `scripts/v4/sql/001_vector_schema.sql` — since the current `package.glob("*.py")` can neither declare nor detect the migration T025 creates per plan: Structure Decision (partial)
- [ ] T039 Remove `scripts/artifact_publication.py` from `manifest.REUSED` as part of the T018 cutover, so `verify()` stops asserting a v4 dependency that the v4 publisher replaces per plan: Structure Decision, FR-009 (partial)
- [ ] T040 Review `scripts/add_wilker_chunks.py` and `tests/test_add_wilker_chunks.py`, which import `scripts.corpus_generator` and `scripts.data_cleaner` from `manifest.EXCLUDED` and are referenced by no artifact in this feature; either justify them and declare their status in the module registry, or remove them (unrequested)
- [ ] T041 Remove the merge-conflict leftovers `scripts/citation_ground_truth.py.orig` and `scripts/corpus_generator.py.orig` from the source tree, or add them to `.gitignore` if they are intentional local scratch files (unrequested)

**Checkpoint**: The module registry matches the real v4 file set including non-Python artifacts, declares no stale dependency, and the source tree contains no undeclared or leftover files in v4's scope.
