# Contract: v4 Source and Published Dataset

**Status**: Target contract for the v4 **test-only** feature. Existing field names are retained; the immutable generation layout and extra validation are design changes. See [data-model.md](../data-model.md) for relationships and state transitions.

## Source CSV

The sole production source is `datasets/redator_v4_repr-test_1k-val_2k-train_rest/redator_v4_repr-test_1k-val_2k-train_rest-test.csv` (or an operator-provided byte-identical copy), a UTF-8 CSV with **1,000 records** and SHA-256 `d10d21f2074e48576cb715bc65ef01a36f369bf837dd2c404280775cef56fff3`. The build fixes the split to `test` and rejects a different source fingerprint, a different row count, or any train/validation input. **No column carries the split.** The filename is advisory only and the SHA-256 is the authoritative gate: a file named as `-test` whose bytes differ is rejected on fingerprint, a file named as train or validation is rejected before its rows are read, and a byte-identical copy under any other name is accepted. The expected fingerprint may be overridden only through the test-only injectable parameter used by the synthetic fixture, never on the production path. This local CSV is ignored by Git, so a new environment must supply it. Required columns are:

| Column | Contract |
|---|---|
| `id` | Nonempty document ID, unique within the input generation. |
| `data` | Source document HTML. Empty cleaned text rejects the source. |
| `question_texts` | JSON array of question strings, in answer order. |
| `output` | JSON array of answer objects, same length and order as `question_texts`. Each answer may have `answer` text or null and `citations` as source HTML element IDs. |

Other columns are source metadata and do not change question identity. A question ID is `{id}:q{zero_based_answer_index}`. The pipeline audits malformed arrays, count mismatches, duplicate IDs, and blank questions. The example [fixture-v4.csv](fixture-v4.csv) is a portable synthetic source used only by tests and smoke validation; it is not an alternative production dataset. `representative_redator_test.csv` is an older source and does not satisfy this v4 JSON-array contract as stored.

## Generation layout and publication

```text
<output>/
├── current -> generations/<generation-id>   # Switched atomically
└── generations/
    └── <generation-id>/
        ├── beir/
        │   ├── corpus.jsonl
        │   ├── queries.jsonl
        │   ├── qrels/<split>.tsv
        │   └── candidates/<split>.jsonl
        ├── documents/documents.jsonl
        ├── evidence/evidence.jsonl
        ├── audits/
        │   ├── rejected_sources.jsonl
        │   ├── dropped_queries.jsonl
        │   ├── unmapped_citations.jsonl
        │   ├── cleaning_events.jsonl
        │   └── plain_text_violations.jsonl
        └── meta/manifest.json
```

All artifact files are complete before `current` changes. A consumer resolves `current` once and reads only that immutable path. Failed builds or failed reference updates leave the previous active generation intact. Generation removal is not part of this feature.

## Record schemas

The examples show required fields; additional documented metadata may be present.

`beir/corpus.jsonl` has one passage per line:

```json
{"_id":"doc-1:vv4-o200k-1:0000:0-200","title":"","text":"Ana apresentou pedido de revisão contratual ao tribunal.\nO processo registra documentos, datas, valores e fundamentos adicionais para análise detalhada do caso e encaminhamento à instância competente.","metadata":{"document_id":"doc-1","chunk_index":0,"start_offset":0,"end_offset":200,"chunk_type":"prose"}}
```

`beir/queries.jsonl` has one retained question per line:

```json
{"_id":"doc-1:q0","text":"Quem apresentou o pedido?","reference_answer":"Ana","citations":["p-0"]}
```

`beir/qrels/<split>.tsv` has the header `query-id<TAB>corpus-id<TAB>score`; each subsequent row links a retained query to a passage with positive score `1`. An accepted citation selects one passage by maximum cleaned-text overlap; duplicate `(query-id, corpus-id)` rows are collapsed.

`beir/candidates/<split>.jsonl` has exactly one record per retained query:

```json
{"query_id":"doc-1:q0","document_id":"doc-1","candidate_ids":["doc-1:vv4-o200k-1:0000:0-200"]}
```

`documents/documents.jsonl` contains `document_id` and full cleaned `text`. `evidence/evidence.jsonl` contains `query_id`, `document_id`, and citation spans with source offsets, cleaned-text offsets, and mapped `chunk_id` when available. Offsets use a half-open interval `[start, end)` into the corresponding normalized or cleaned document text.

Audits are line-delimited objects with source or query identity and a specific reason. `rejected_sources.jsonl` is for whole-source rejection; `dropped_queries.jsonl` is for question exclusion; `unmapped_citations.jsonl` includes citation identity and mapping failure reason; `cleaning_events.jsonl` and `plain_text_violations.jsonl` explain transformations or residual quality problems. A blank question receives a dropped-query audit instead of disappearing silently.

`meta/manifest.json` records pipeline label `redator-v4`, split `test`, the test-source SHA-256 and row count, chunking and filtering configuration, summary counters, generation ID, and generation time. Counts must agree with emitted artifacts. Consumers reject a generation with missing required files or broken ID references before indexing, retrieval, or evaluation.

## Cross-file invariants

- Every query ID is unique and has exactly one candidate record and at least one qrel.
- Every qrel passage exists in the corpus and in its query's candidate set.
- Every candidate passage belongs to the candidate record's `document_id`.
- Every published document has at least one retained query under the default document-scoped policy.
- Every retained query has nonblank answer text and at least one trackable cited evidence span under the default filter.
- All passage offsets refer to the cleaned document text used for the generation.
- Every published source identity derives from a row of the selected v4 test CSV, never the train or validation sources. Excluded test rows/questions appear in audit counts rather than being silently replaced with other splits.

## Ranking and comparison files

A ranking run is either a JSON object `{query_id: {passage_id: finite_score}}` or TREC rows `query_id Q0 passage_id rank score model_tag`. It contains only passages from the query's candidate set, up to the configured `top_k`; the run is written to a temporary path and replaced atomically after validation. Run provenance (generation ID, active Qwen index-build ID, model ID/revision, query instruction, chunk-text profile, score type, cutoff) is stored separately from the run-file directory so the metric loader does not mistake it for a competing run.

An evaluation report records the judgment source, common cutoff list, and for each model: `queries_scored`, `queries_missing`, and NDCG/MAP/Recall/Precision/MRR values at those cutoffs. Judged queries absent from a run contribute zero. A run with no judged query in common is rejected.
