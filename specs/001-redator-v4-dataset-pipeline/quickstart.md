# Quickstart: Validate the v4 Workflow

Run commands from the repository root. This guide validates the test-only feature described in [spec.md](spec.md) against the [dataset contract](contracts/dataset.md), [index contract](contracts/model-index.md), and [workflow contract](contracts/workflow-cli.md). The included [fixture](contracts/fixture-v4.csv) is synthetic and contains no real legal records; it is for smoke tests only, not a production source.

## Prerequisites

- Python compatible with `pyproject.toml`, the repository's development dependencies, and `tiktoken` for v4 token sizing. The implementation plan adds an explicit v4 dependency declaration.
- The actual 1,000-row v4 test CSV at `datasets/redator_v4_repr-test_1k-val_2k-train_rest/redator_v4_repr-test_1k-val_2k-train_rest-test.csv`; it is ignored by Git. See the exact checksum in the [dataset contract](contracts/dataset.md).
- A local pgvector-enabled PostgreSQL container with persistent data; the application database must have `CREATE EXTENSION vector` applied. See the [index contract](contracts/model-index.md).
- For indexing and retrieval, a trusted remote vLLM server hosting `Qwen/Qwen3-Embedding-0.6B`, reachable through HTTPS or a private authenticated tunnel. The operator supplies its address and any credential.
- The `o200k_base` tokenizer asset available in the local cache; its first use may require a download. The existing test suite passed after that asset became available.

## Current baseline: preparation and tests

The existing preparation command accepts the synthetic fixture today; the target production workflow below accepts only the v4 test CSV:

```bash
VALIDATION_ROOT=$(mktemp -d)
python -m scripts.v4.run \
  specs/001-redator-v4-dataset-pipeline/contracts/fixture-v4.csv \
  "$VALIDATION_ROOT/dataset" --split test
python -m pytest tests/v4 -q --tb=short
```

Expected fixture summary: **2 documents, 2 passages, 2 of 3 questions retained, and 2 judgments**. The third question is audited as lacking an answer/citation. In the current code, artifact directories appear directly under `"$VALIDATION_ROOT/dataset"`; the planned immutable-generation publisher will move the active view to `"$VALIDATION_ROOT/dataset/current"`. The existing v4 suite passed 112 tests in the planning environment.

## Target acceptance: test-only consistent generation

After implementing the plan, run the production build with the actual test CSV:

```bash
python -m scripts.v4.run \
  datasets/redator_v4_repr-test_1k-val_2k-train_rest/redator_v4_repr-test_1k-val_2k-train_rest-test.csv \
  "$VALIDATION_ROOT/dataset"
```

Verify that `"$VALIDATION_ROOT/dataset/current"` resolves to one immutable generation containing all files listed in [contracts/dataset.md](contracts/dataset.md). Its manifest must identify split `test`, the 1,000-row source, and the expected SHA-256, while every judgment and candidate ID resolves in the corpus. A train/validation CSV or mismatched fingerprint must fail without changing `current`. Inject a build/publication failure and confirm the previous generation remains unchanged; concurrent readers must never combine generations.

## Target acceptance: Qwen chunk index and remote retrieval

Set the trusted service location. The fixture sends only synthetic text; indexing the real test dataset sends full cleaned legal text as agreed in the spec. Start the local pgvector-enabled PostgreSQL service and build the Qwen vectors first (these are target commands; the compose file and index CLI are created during implementation):

```bash
docker compose up -d postgres
export VLLM_BASE_URL="https://trusted-vllm.example/v1"
export VLLM_MODEL="Qwen/Qwen3-Embedding-0.6B"
python -m scripts.v4.index \
  "$VALIDATION_ROOT/dataset" \
  --model "$VLLM_MODEL" --base-url "$VLLM_BASE_URL" \
  --postgres-dsn-env V4_POSTGRES_DSN
```

Provide `V4_POSTGRES_DSN` and, if needed, `V4_EMBEDDING_API_KEY` through your local secret manager before running the command. The database must contain exactly one `vector(1024)` row per published test chunk, persist after restarting PostgreSQL, and reject an incomplete or mismatched generation/build pair. Verify `SELECT extname FROM pg_extension WHERE extname = 'vector'` in the application database. See [model-index.md](contracts/model-index.md) for the metadata checks.

```bash
python -m scripts.v4.retrieve \
  "$VALIDATION_ROOT/dataset/current/beir" \
  "$VALIDATION_ROOT/runs/qwen-top10.json" \
  --base-url "$VLLM_BASE_URL" --model "$VLLM_MODEL" \
  --postgres-dsn-env V4_POSTGRES_DSN \
  --split test --top-k 10 --format json
```

If authentication is required, provide `V4_EMBEDDING_API_KEY` through your secret manager before running the command; do not place its value in shell history. The target CLI reads it from the environment. It creates a complete run using the active compatible Qwen pgvector build, ranks only each question's own document passages with exact cosine distance, and records run provenance outside the runs directory. A missing candidate, stale/incomplete build, invalid query vector, timeout, or unavailable service returns a specific error without replacing an existing run. The implementation phase adds mocked tests for these failure cases; no live remote request is required for automated tests.

## Target acceptance: comparable metrics

Run a second Qwen retrieval configuration using the same index and judgments but a shorter result depth. This compares ranking-run configurations, not different models:

```bash
python -m scripts.v4.retrieve \
  "$VALIDATION_ROOT/dataset/current/beir" \
  "$VALIDATION_ROOT/runs/qwen-top5.json" \
  --base-url "$VLLM_BASE_URL" --model "$VLLM_MODEL" \
  --postgres-dsn-env V4_POSTGRES_DSN \
  --split test --top-k 5 --format json
```

Then evaluate the runs:

```bash
python -m scripts.v4.metrics \
  "$VALIDATION_ROOT/dataset/current/beir/qrels/test.tsv" \
  "$VALIDATION_ROOT/runs" \
  --k-values 1 3 5 10 \
  --output "$VALIDATION_ROOT/reports/comparison.json"
```

Expected: one result row per run using the same judgments and cutoffs, with NDCG, MAP, recall, precision, and MRR values. A run omitting a judged question reports it as missing and scores it as zero. An empty, malformed, or unrelated run fails with an input-specific error. Keep the report outside the runs directory.
