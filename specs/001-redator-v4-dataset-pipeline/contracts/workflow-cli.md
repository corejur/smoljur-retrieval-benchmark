# Contract: v4 Workflow Commands and Remote Embeddings

**Status**: Target interfaces for the implementation phase. Existing command names are retained where possible; test-source gating, Qwen pgvector build, generation resolution, secret handling, strict validation, and atomic run publication are planned changes.

## 1. Build a generation

```text
python -m scripts.v4.run V4_TEST_CSV OUTPUT_ROOT
  [--max-words N] [--target-words N] [--overlap-words N] [--minimum-words N]
  [--max-citation-words N] [--max-gold-chunks N]
  [--keep-query-less-documents] [--allow-missing-answer]
```

`V4_TEST_CSV` follows [dataset.md](dataset.md): the exact 1,000-row v4 test file or byte-identical copy. The production command fixes split to `test` and verifies its SHA-256 and row count before publication; train/validation files and the synthetic fixture are rejected by the production gate. Automated tests may call a lower-level builder with the synthetic fixture. The command builds and validates a complete immutable generation, then atomically changes `OUTPUT_ROOT/current`. By default, a published document must have a retained question, and a retained question must have answer text and a traceable citation. The two permissive flags are explicit alternatives; their selected policy is written to the manifest. The command prints source, document, passage, retained-question, judgment, and exclusion counts. Invalid input or build failure returns nonzero and does not change `current`.

## 2. Build the Qwen chunk index

```text
python -m scripts.v4.index DATASET_ROOT
  --model Qwen/Qwen3-Embedding-0.6B
  --base-url TRUSTED_VLLM_BASE_URL
  --postgres-dsn-env V4_POSTGRES_DSN
  [--batch-size N]
```

`DATASET_ROOT/current` resolves once to the immutable test generation. PostgreSQL runs in a local pgvector-enabled container with persistent storage; `V4_POSTGRES_DSN` and optional `V4_EMBEDDING_API_KEY` are provided through the environment or secret manager, never as literal command-line credentials. The builder sends each published chunk's full cleaned text to the trusted remote vLLM service, validates 1,024-dimensional responses, stages one `vector(1024)` row per chunk in PostgreSQL, then activates the complete build transactionally after all checks in [model-index.md](model-index.md) pass. Non-Qwen model IDs, non-test generations, incomplete responses, and incompatible dataset manifests fail nonzero without replacing the active compatible build.

## 3. Retrieve against one generation and index

```text
python -m scripts.v4.retrieve DATASET_BEIR RUN_FILE
  --model Qwen/Qwen3-Embedding-0.6B --base-url TRUSTED_VLLM_BASE_URL
  --postgres-dsn-env V4_POSTGRES_DSN
  [--split NAME] [--batch-size N] [--top-k N]
  [--format json|trec]
```

`DATASET_BEIR` points to the `beir` directory inside one resolved immutable test generation. If the operator passes `OUTPUT_ROOT/current/beir`, the command resolves it once before opening any file. `top_k` and `batch-size` must be positive; split must be `test`. The published document-scoped candidate manifest is mandatory. The command resolves and verifies one complete compatible Qwen pgvector build before any query request. It embeds only the questions, applies the versioned Qwen query instruction, executes exact pgvector cosine-distance ranking after filtering by build/document/candidates, validates results, and writes a complete run to a temporary file before replacing `RUN_FILE`. A service, index, or data failure returns nonzero and leaves an existing run untouched.

The operator sets `V4_EMBEDDING_API_KEY` in the process environment when the trusted service requires authentication. The trusted base URL uses HTTPS or a private authenticated tunnel. Full cleaned chunk text is sent during indexing and full question text during retrieval; there is no redaction step. Model ID/revision, versioned query instruction, chunk-text profile, index build ID, normalization/score mode, split, generation ID, and `top_k` are recorded separately from the run directory for reproducibility.

## 4. Evaluate runs

```text
python -m scripts.v4.metrics QRELS_TSV RUNS_DIRECTORY
  [--k-values K ...] [--report METRIC ...] [--output REPORT_JSON]
```

The evaluator reads all ranking runs in `RUNS_DIRECTORY`, uses the same qrels and cutoffs for each, and prints a comparison. Defaults are `k = 1, 3, 5, 10, 20, 100`. It reports NDCG, MAP, recall, precision, and MRR at each selected cutoff. Each judged query missing from a run contributes zero and increments `queries_missing`. Empty, malformed, non-finite, or unrelated runs fail with an input-specific error. A report file, if requested, is kept outside `RUNS_DIRECTORY` so it is not read as a competing run.

## Remote vLLM embedding exchange

The selected integration is vLLM's OpenAI-compatible embedding interface, `POST {BASE_URL}/embeddings`, where `BASE_URL` ends at `/v1`. The configured model is pinned to `Qwen/Qwen3-Embedding-0.6B`. For each bounded batch:

```json
{"model":"Qwen/Qwen3-Embedding-0.6B","input":["full cleaned chunk text or instructed query text"]}
```

Expected response subset:

```json
{"data":[{"index":0,"embedding":[0.12,0.34]},{"index":1,"embedding":[0.56,0.78]}]}
```

The adapter accepts a response only when `data` contains one result per input, indices are exactly `0..N-1` without duplicates, all embeddings are 1,024-dimensional numeric vectors, and every value is finite with nonzero norm. It sorts by `index` before assigning vectors to input texts, then L2-normalizes to float32. Connection/timeouts and transient rate-limit/server failures may be retried within a documented bound; permanent client errors and malformed responses fail immediately with the service and cause identified. Failures must not publish a partial index or run or log full source text or credentials.

See [vLLM's embedding model guidance](https://docs.vllm.ai/en/latest/models/pooling_models/embed/) and [security guidance](https://docs.vllm.ai/en/latest/usage/security/). The trusted endpoint and credentials are deployment inputs; this repository does not provision the remote server.
