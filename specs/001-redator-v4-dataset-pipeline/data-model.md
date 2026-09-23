# Data Model: Representative Redator v4

The model follows a source record through preparation, retrieval, and evaluation. Exact serialized fields and paths are in [contracts/dataset.md](contracts/dataset.md).

## Entities and relationships

| Entity | Identity and principal fields | Relationships |
|---|---|---|
| Test Source | Source path, split `test`, SHA-256 fingerprint, 1,000 CSV rows | Sole production input to one dataset generation. Train/validation files are not eligible. |
| Source Record | Unique `id`; source `data` HTML; ordered `question_texts`; ordered `output` answer objects | Comes from the selected v4 Test Source; owns one document and zero or more source questions. |
| Question | `{document_id}:q{question_index}`; normalized text; source answer; citation IDs | Belongs to one source record. May become one retained published query or one dropped-query audit. |
| Document | `document_id`; full cleaned text | Derived from one valid source record. Owns ordered passages and zero or more retained queries. |
| Passage | `chunk_id`; `document_id`; `chunk_index`; cleaned-text offsets; text; section/title; kind | Belongs to one document. May be a candidate for that document's queries and relevant to one or more queries. |
| Evidence Span | `citation_id`; source and cleaned-text offsets; mapped passage ID | Belongs to one question and its source document. Supports a relevance judgment. |
| Candidate Set | One `query_id`; one `document_id`; ordered passage IDs | Contains only passages from the query's source document. |
| Relevance Judgment | `(query_id, chunk_id)`; positive score | Connects one retained query to one passage selected from cited evidence. |
| Audit Record | Source or query identity; reason; relevant provenance | Records rejected sources, dropped questions, unmapped citations, cleaning events, or residual text problems. |
| Generation | Unique generation ID; creation time; test-source SHA-256, split, configuration fingerprint; manifest and summary | Contains one mutually consistent set of the above published records. One generation is active at a time. |
| Model Index Build | `index_build_id`; model ID/revision; generation ID and manifest checksum; dimension 1,024; text/instruction profile; normalization; status; chunk count | One immutable pgvector embedding collection for one model and one dataset generation, with PostgreSQL active-build pointer. |
| Indexed Chunk | `(index_build_id, chunk_id)`; `document_id`; `chunk_index`; `embedding vector(1024)` | Stores exactly one Qwen embedding per published test chunk. PostgreSQL/pgvector owns both vector and identity. |
| Active Model Index | `(generation_id, model_id)`; `index_build_id` | Selects one validated build for each generation/model pair; changed transactionally after completeness checks. |
| Ranking Run | Model/run ID; generation ID; index build ID; score settings; query-to-ranked-passage scores | Uses one immutable generation and complete compatible model index. Scored separately from retrieval. |
| Evaluation Report | Generation ID; judgment set; cutoffs; per-model measures and missing counts | Compares one or more ranking runs against the same judgments. |
| Remote Model Service | Trusted endpoint and pinned Qwen model identifier, configured at runtime | Receives full cleaned query and passage text; returns chunk vectors during indexing and query vectors during retrieval. Credentials are not dataset records. |

## Identity and validation rules

1. The production source is the v4 test CSV only; its SHA-256 and 1,000-row count are verified before publication. Split is fixed to `test`. A source `id` is nonempty and unique within a generation. Duplicate IDs are rejected before publication because they would collide with question and passage identities.
2. `question_texts` and `output` are ordered arrays of equal length. A question's position is its answer position; dropping one question does not renumber later IDs.
3. Every published `query_id`, `chunk_id`, and `(query_id, chunk_id)` judgment is unique in its respective artifact.
4. Each passage's half-open `[start_offset, end_offset)` indexes the exact cleaned document text. Passage text is derived from that region; structural table handling may repeat header context.
5. Each accepted citation has a traceable source span and cleaned-text span and selects one passage by greatest overlap, breaking ties by earlier passage index. Duplicate citations mapping to the same passage produce one judgment row for that query and passage.
6. Every retained query has nonblank answer text under the default policy, at least one accepted citation, at least one judgment, and one nonempty candidate set.
7. Every judgment passage is present in the corpus and in its query's candidate set. Every candidate passage belongs to that query's source document.
8. Under the default policy, a published document has at least one retained query. A query excluded for missing answer/evidence or a configured gold-passage cap receives a reasoned audit record.
9. Ranking scores are finite numbers; every ranked passage is in the query's candidate set and the number of ranked passages does not exceed `top_k`.
10. Each evaluation report uses one judgment set and one cutoff list for every run. A judged query missing from a run contributes zero and increments `queries_missing`.
11. Each complete Qwen build contains exactly `manifest.chunks` normalized, nonzero, finite `vector(1024)` values. `Indexed Chunk` is bijective with the published corpus: no missing, duplicate, or foreign chunk IDs.
12. Retrieval filters indexed chunks by the pinned build ID, the question's source document, and its published candidate IDs before exact pgvector cosine-distance ordering. Returned IDs are checked against the candidate set. No approximate ANN index is used in this iteration.
13. An active build must have a matching generation ID and manifest checksum, model ID/revision, text profile, Qwen query instruction, dimension, normalization, and PostgreSQL row count. A mismatched or incomplete build is not usable.
14. Only the Qwen model may create an index in this iteration. The schema includes `model_id` to avoid collisions when additional models are explicitly added later.

## State transitions

### Source and question

```text
source row -> valid source -> cleaned document -> passage set
           -> rejected source (reason)

question -> evidence checked -> retained query + candidate set + judgment(s)
                            -> dropped query (reason)
```

A document with no retained query moves to rejected under the default policy; the rejection is audited. Source documents that fail parsing, cleaning, or passage creation are not published.

### Generation, index, and run

```text
generation: building -> validated -> active -> superseded
                    \-> failed (never active)

model index: building -> rows-written -> validated -> active -> superseded
                      \-> failed (never active)

ranking run: building -> validated -> published -> evaluated
                   \-> failed (no partial published run)
```

The active-generation reference changes in one operation. A pgvector build's rows are staged under a new immutable build ID and verified before its PostgreSQL active pointer changes in one transaction. A reader resolves the generation and index build once, validates compatibility, and continues reading those immutable rows even if a later build becomes active. Superseded generations and builds remain available until a separate cleanup policy is defined.

## Capacity and provenance

Preparation is row-streamed over the 1,000-row test source only. The v4 chunk profile uses `o200k_base` tokens with maximum 512, target 320, and overlap 64. Index construction streams chunk embedding batches into PostgreSQL. A `vector(1024)` value uses about 4,104 bytes before row/index overhead; size the persistent database from the actual published test chunk count. Retrieval sends bounded query batches to vLLM and performs exact PostgreSQL/pgvector ranking after document/candidate filtering. The generation manifest records source identity, configuration, counts, and label policy; the index build records model/text/vector provenance; a ranking run records its generation, index build, query instruction, score, and cutoff settings.
