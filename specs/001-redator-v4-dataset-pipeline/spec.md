# Feature Specification: Representative Redator v4 Dataset Pipeline

**Feature Branch**: `v4-dataset-pipeline` (created manually; no branch-creation hook is configured)

**Created**: 2026-09-22

**Status**: Draft

**Input**: User clarification: "existing dataset pipeline work" in this repository

## Clarifications

### Session 2026-09-22

- Q: What should the active feature specification describe? → A: Existing dataset pipeline work.
- Q: Which dataset pipeline generation should this specification cover? → A: v4 only.
- Q: Should the v4 specification include retrieval and metric evaluation, or only dataset preparation? → A: Preparation plus retrieval and metric evaluation.
- Q: May v4 retrieval send document and query text to an external model service? → A: Use requests to a remote vLLM service.
- Q: What source text may retrieval send to the remote vLLM service? → A: Full cleaned text to a trusted server.
- Q: Which rows and split belong to this feature? → A: Only source rows from the v4 test CSV; train and validation rows must not enter the published dataset.
- Q: Which model is indexed initially? → A: Only `Qwen/Qwen3-Embedding-0.6B`; retain a per-model index design for future models.
- Q: How are chunk embeddings indexed? → A: The initial FAISS proposal is superseded; use pgvector as the sole vector store in a local PostgreSQL container.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build a Usable Retrieval Dataset (Priority: P1)

As a dataset maintainer, I want Representative Redator source documents and their questions transformed into a retrieval dataset so that researchers can evaluate search systems against cited evidence.

**Why this priority**: The dataset is the core product of the preparation pipeline.

**Independent Test**: The v4 test CSV produces searchable passages, normalized questions, and relevance judgments whose identifiers all resolve to published records; no train or validation source ID appears.

**Acceptance Scenarios**:

1. **Given** a valid source row with document text, a question, and a traceable cited passage, **When** the maintainer prepares the dataset, **Then** the output contains a searchable passage, the question, and a relevance judgment linking them.
2. **Given** a completed generation, **When** a researcher loads its published records, **Then** every retained question and relevance judgment refers to an existing passage from its source document.
3. **Given** source markup around a cited passage, **When** the document is cleaned and divided into passages, **Then** the cited text remains traceable to one relevant published passage selected by greatest evidence overlap.
4. **Given** the v4 test source and separate train and validation sources, **When** the maintainer builds this feature's dataset, **Then** only rows from the test source are read and the published manifest identifies that test source and split.

---

### User Story 2 - Audit Excluded Source Material (Priority: P2)

As a dataset maintainer, I want exclusions and evidence mapping failures explained so that I can inspect data loss and correct source quality problems.

**Why this priority**: Unexplained exclusions make the dataset difficult to trust or reproduce.

**Independent Test**: A test collection containing malformed questions, empty text, and missing citations produces identifiable exclusion records with reasons.

**Acceptance Scenarios**:

1. **Given** a source row has mismatched question and answer counts or cannot yield usable document text, **When** the pipeline processes it, **Then** the row is excluded and its reason is available for review.
2. **Given** one question in a document has no answer text or usable cited evidence but another question is valid, **When** the pipeline processes the document, **Then** only the unusable question is excluded and audited.
3. **Given** a document has no retained questions, **When** the pipeline processes it under the default policy, **Then** the document and its passages are excluded and the reason is audited.
4. **Given** a question would require more gold passages than the configured limit, **When** the pipeline processes it, **Then** the question is excluded and the limit reason is audited.

---

### User Story 3 - Publish a Complete Generation (Priority: P3)

As a dataset maintainer, I want each completed dataset generation published as a coherent set so that researchers do not read a mixture of old and new records.

**Why this priority**: A partial generation can invalidate evaluation results.

**Independent Test**: A deliberately failed preparation run leaves the last published generation intact; a successful run makes the new generation available as a complete set.

**Acceptance Scenarios**:

1. **Given** a valid existing generation, **When** a later preparation run fails, **Then** researchers can still access the unchanged existing generation.
2. **Given** all records for a new generation are valid, **When** publication completes, **Then** the new documents, questions, judgments, and audits are available together.

---

### User Story 4 - Index and Rank Test Passages (Priority: P4)

As a researcher, I want the test dataset's chunks embedded once with `Qwen/Qwen3-Embedding-0.6B`, stored in pgvector within a local PostgreSQL container for that model, and ranked against each question's eligible passages so that I can measure how well the model finds cited evidence.

**Why this priority**: A published dataset becomes useful for model comparison only when it can produce repeatable rankings.

**Independent Test**: With a small published test generation and a mocked Qwen embedding service, index every published chunk exactly once, restart the local index/storage stack, and produce rankings containing only each question's candidate passages, limited to the requested result count.

**Acceptance Scenarios**:

1. **Given** a retained question and its source-document candidate set, **When** the researcher runs retrieval, **Then** the ranking contains only candidates from that set in descending score order.
2. **Given** the researcher requests a result limit smaller than the candidate set, **When** retrieval completes, **Then** the ranking has no more than that many passages.
3. **Given** a question has no candidate set or a candidate is missing from the published passages, **When** retrieval begins, **Then** the run fails with an identifiable data error rather than publishing an incomplete ranking.
4. **Given** a configured trusted remote vLLM service, **When** the researcher builds the index and later runs retrieval, **Then** the service receives full cleaned chunk text during indexing and the instructed question text during retrieval; retrieval reuses stored chunk embeddings.
5. **Given** the remote vLLM service is unavailable or returns unusable results, **When** retrieval runs, **Then** the researcher receives a specific error and no incomplete ranking run is published.
6. **Given** the published test generation, **When** the Qwen index build completes, **Then** each published chunk ID maps to exactly one finite, fixed-dimension pgvector embedding in the active model index, and no other model index is created by this workflow.
7. **Given** an existing complete index, **When** reindexing fails or the local storage stack restarts, **Then** the prior complete index remains usable and no partial index becomes active.
8. **Given** an index built from a different dataset generation or model configuration, **When** retrieval begins, **Then** retrieval fails with a specific compatibility error instead of using stale embeddings.

---

### User Story 5 - Compare Retrieval Runs Fairly (Priority: P5)

As a researcher, I want ranking runs scored against the same relevance judgments so that I can compare retrieval configurations consistently now and models when additional model indexes are supported later.

**Why this priority**: Comparable scores are the final outcome of the v4 evaluation workflow.

**Independent Test**: Two supplied ranking runs from the same test generation produce separate, comparable score records; omitting a question lowers the incomplete run's score.

**Acceptance Scenarios**:

1. **Given** two valid ranking runs and one relevance-judgment set, **When** the researcher evaluates them together, **Then** each receives the same selected measures and result-depth cutoffs.
2. **Given** a ranking run omits a judged question, **When** it is evaluated, **Then** the missing question contributes zero and is counted as missing.
3. **Given** a ranking run is empty, malformed, or has no question in common with the judgments, **When** evaluation begins, **Then** the researcher receives a specific error identifying the unusable run.

### Edge Cases

- A source row has a different number of questions and answers.
- Document text becomes empty after cleaning.
- A cited identifier is absent from the source document or does not map to a published passage.
- A question has an answer but no citation, or a citation but no answer text.
- A question's cited evidence would require more gold passages than the configured evaluation limit.
- A document has usable text and passages but none of its questions can be evaluated.
- A preparation or publication step fails after the previous dataset generation is already available.
- A retained question has no candidate set, or a candidate passage is absent from the published dataset.
- A ranking run omits some judged questions or includes unrelated questions.
- A ranking run is empty, malformed, or has no judged question in common with the dataset.
- The remote vLLM service times out, rejects a request, or returns an incomplete response.
- The configured remote endpoint is plaintext `http://`, presents a certificate that does not verify, or is supplied with verification disabled.
- The source path points to a v4 train or validation split, or the source fingerprint differs from the recorded test file.
- An index build is interrupted, its PostgreSQL rows are incomplete, or its active-build metadata points to a different generation.
- The remote model returns a vector with the wrong dimension, a zero norm, or a non-finite value.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: This feature's dataset build MUST accept only source records from the v4 test CSV containing stable document identities, document content, question lists, answers, and citation references; it MUST reject train or validation input and record the test source fingerprint.
- **FR-002**: The pipeline MUST normalize source document content into searchable text while retaining a way to trace cited evidence to the normalized document.
- **FR-003**: The pipeline MUST produce ordered searchable passages with stable identities and source-document provenance.
- **FR-004**: Every retained question MUST have answer text and at least one relevance judgment pointing to a published passage from its cited evidence under the default policy.
- **FR-005**: The pipeline MUST exclude an individual question with missing answer text, missing citations, no mapped cited evidence, mapped evidence outside published passages, more gold passages than the configured limit (ten by default), or question text longer than the configured token limit (512 `o200k_base` tokens by default, the passage cap), and MUST record its identity and specific exclusion reason.
- **FR-006**: Under the default policy, the pipeline MUST retain a document when at least one question remains usable and exclude a document when none remain usable.
- **FR-007**: Every retained question MUST have a candidate set limited to passages from its source document under the default evaluation mode.
- **FR-008**: Each accepted citation MUST link to one relevant passage chosen by its greatest overlap with the cited evidence.
- **FR-009**: The pipeline MUST publish the resulting passages, questions, relevance judgments, candidate sets, and audit records as one coherent generation.
- **FR-010**: A failed preparation or publication MUST leave the previously published generation usable.
- **FR-011**: The pipeline MUST report counts of source records, retained records, excluded records, and relevance judgments for each generation.
- **FR-012**: Retrieval MUST rank each retained question only against its published candidate set and return no more passages than the requested result limit.
- **FR-013**: The initial index and retrieval workflow MUST use only `Qwen/Qwen3-Embedding-0.6B` for scoring questions against candidate passages and MUST produce a ranking run that can be evaluated separately. The index layout MUST distinguish model identities so later models cannot overwrite this index.
- **FR-014**: Retrieval MUST reject a question without candidates or a candidate that is absent from the published passages with an identifiable error.
- **FR-015**: Evaluation MUST score each ranking run against the same published relevance judgments and the same selected result-depth cutoffs.
- **FR-016**: Evaluation MUST report ranking quality using normalized discounted cumulative gain, mean average precision, recall, precision, mean reciprocal rank, and pass@k (the share of questions with a relevant passage in the top k) at each selected cutoff.
- **FR-017**: Evaluation MUST treat judged questions absent from a ranking run as zero-scoring questions and report their count.
- **FR-018**: Evaluation MUST reject empty or malformed ranking runs and runs with no judged question in common, identifying the unusable input.
- **FR-019**: Evaluation MUST present separate comparable results for at least two ranking runs in one comparison.
- **FR-020**: Indexing and retrieval MUST support requests to a configured trusted remote vLLM service using full cleaned chunk text and instructed question text respectively, without a redaction step. Retrieval MUST reuse indexed chunk embeddings rather than resend all candidate passages.
- **FR-021**: If the remote vLLM service is unavailable or returns unusable results, retrieval MUST report the failure and MUST NOT publish a partial ranking run.
- **FR-022**: The index builder MUST obtain one embedding for every published test-set chunk from the trusted remote vLLM service, validate identity, dimension, and finiteness, and store those vectors using pgvector in a persistent local PostgreSQL database associated with the Qwen model and exact dataset generation.
- **FR-023**: PostgreSQL MUST retain the embeddings, chunk-ID mapping, and provenance sufficient to verify model identity, dataset generation, vector configuration, and completeness before retrieval. These records MUST remain consistent across restart or failed rebuild.
- **FR-024**: Retrieval MUST use only a complete, compatible active Qwen index and MUST enforce the published document-scoped candidate set while searching it; a stale, corrupt, incomplete, or mismatched index MUST fail with an identifiable error.
- **FR-025**: Index publication MUST be atomic from a reader's perspective. A failed index build MUST leave the previous complete compatible index usable and MUST NOT expose a partial index.
- **FR-026**: Before any document, chunk, or question text leaves the process, indexing and retrieval MUST validate that the configured remote vLLM endpoint either uses `https://` with certificate verification enabled, or is a loopback or tunnel-local address reached through an operator-established encrypted tunnel. Any other endpoint, and any configuration that disables certificate verification, MUST be rejected with an identifiable configuration error. Service credentials MUST be read from the environment and MUST NOT appear in command-line arguments, logs, or error messages.

### Scope

This feature covers only the Representative Redator v4 **test** source rows, from dataset preparation through model-specific pgvector chunk storage in local PostgreSQL, document-scoped retrieval, and metric evaluation. It includes a trusted remote vLLM service for embedding requests. Train and validation source rows, earlier v1–v3 behavior, indexes for models other than `Qwen/Qwen3-Embedding-0.6B`, retrieval model training, and unrestricted cross-document search are outside this scope.

### Key Entities

- **Source Record**: One identified document with its original content, questions, answer entries, and cited evidence identifiers.
- **Document**: Searchable text derived from a source record, preserving the source identity.
- **Passage**: An ordered, identified excerpt of a document used as a retrieval candidate.
- **Question**: A normalized information need linked to its source document and answer position.
- **Relevance Judgment**: A link between a retained question and a passage supported by cited source evidence.
- **Candidate Set**: The passages from a question's source document that may be returned during document-scoped evaluation.
- **Audit Record**: An identified exclusion or transformation with a reason that a maintainer can review.
- **Generation**: A complete published set of documents, passages, questions, judgments, and audits produced from a source collection.
- **Ranking Run**: The ordered, scored passages returned for each question by one researcher-supplied retrieval model.
- **Evaluation Report**: Measures and missing-question counts for one or more ranking runs scored against a single judgment set.
- **Remote Model Service**: The trusted, configured vLLM service that receives questions and full cleaned candidate-passage text and returns representations used for ranking.
- **Model Index**: A persistent PostgreSQL/pgvector collection of chunk embeddings for one model and one immutable test dataset generation, with chunk identity and compatibility metadata stored in the same database.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a completed generation, 100% of retained questions have at least one relevance judgment pointing to a published passage.
- **SC-002**: In a completed generation under the default policy, 100% of retained questions have nonempty answer text and at least one traceable citation.
- **SC-003**: In a review of a test collection containing invalid source records and questions, 100% of exclusions have an identifiable source or question and a specific reason.
- **SC-004**: In a deliberately failed preparation or publication run, 100% of the previously published generation remains readable and unchanged.
- **SC-005**: In a completed generation, the reported record counts match the published records exactly.
- **SC-006**: A researcher can load a published generation and begin an evaluation within 5 minutes using the repository's documented workflow.
- **SC-007**: In the default document-scoped evaluation, 100% of candidates for each retained question come from its source document.
- **SC-008**: In every completed retrieval run, 100% of ranked passages belong to the associated question's candidate set and no ranking exceeds the requested result limit.
- **SC-009**: In a comparison of at least two valid ranking runs, 100% of runs use the same judgments, measures, and cutoffs; both may use the initial Qwen index.
- **SC-010**: In a run missing one or more judged questions, every missing question contributes zero to the aggregate measures and appears in the missing-question count.
- **SC-011**: In a test where the remote model service fails or returns incomplete results, zero incomplete ranking runs are published and the failure is identifiable to the researcher.
- **SC-012**: A completed dataset generation draws 100% of its source rows from the v4 test CSV, records its source fingerprint, and includes zero train or validation source rows.
- **SC-013**: The active Qwen index has exactly one pgvector embedding per published test-set chunk, and its verified chunk count, model ID, generation ID, and vector dimension agree between PostgreSQL and the dataset manifest.
- **SC-014**: After a failed index rebuild or local container restart, a previously complete compatible index remains searchable and no incomplete index is selected.
- **SC-015**: In a configuration test, 100% of attempts to reach a plaintext or certificate-unverified endpoint are rejected before any source, chunk, or question text leaves the process, and zero credentials appear in emitted logs or error messages.

## Assumptions

- The active feature covers v4 of the Representative Redator dataset pipeline; earlier generations remain available for historical reproducibility.
- The source collection provides document identities, document content, question lists, answers, and citation identifiers.
- Dataset preparation uses evidence present in the source; it does not require generating new relevance labels with a model.
- The default policy requires answer text, traceable cited evidence, no more than the configured number of gold passages, and at least one retained question per published document.
- Researchers configure a remote vLLM model service or provide an existing ranking run; creating or training the model is a separate activity.
- Document-scoped candidate sets produced during preparation define the eligible passages for v4 retrieval.
- The remote vLLM server is trusted to receive questions and full cleaned chunk text during indexing, including any personal details present in that text; redaction is not required for this workflow. Trust is placed in the operator of that endpoint, not in the channel: the transport itself must still satisfy FR-026.
- The local v4 test CSV currently resides at `datasets/redator_v4_repr-test_1k-val_2k-train_rest/redator_v4_repr-test_1k-val_2k-train_rest-test.csv` and contains 1,000 CSV records; it is ignored from version control and must be provided by the operator in other environments.
- The user's latest instruction explicitly replaces FAISS with pgvector; PostgreSQL is the sole vector and metadata store for this feature.
