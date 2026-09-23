# Quickstart: Run and Validate the v4 Workflow

Run commands from the repository root. This guide builds the test-only dataset described in [spec.md](spec.md), indexes it, ranks its questions, and compares runs, per the [dataset contract](contracts/dataset.md), [index contract](contracts/model-index.md), and [workflow contract](contracts/workflow-cli.md).

Two sources exist and must not be confused:

- **Portable synthetic fixture** — [contracts/fixture-v4.csv](contracts/fixture-v4.csv), 2 rows, no real legal records. Automated tests and smoke checks use it. The production gate rejects it unless its own fingerprint is passed explicitly.
- **Real v4 test CSV** — `datasets/redator_v4_repr-test_1k-val_2k-train_rest/redator_v4_repr-test_1k-val_2k-train_rest-test.csv`, 1,000 rows, SHA-256 `d10d21f2074e48576cb715bc65ef01a36f369bf837dd2c404280775cef56fff3`. It is ignored by Git; a new environment must supply it.

## Prerequisites

```bash
pip install -r requirements-v4.txt   # beir, tiktoken, psycopg, pgvector, matplotlib, pytest
```

Put the local secrets in the gitignored `.env` (mode 600) and load it into each shell that runs the pipeline:

```bash
# .env
V4_POSTGRES_PASSWORD=<generated>
V4_POSTGRES_DSN=postgresql://redator:<same password>@localhost:5432/redator_v4
V4_EMBEDDING_BASE_URL=http://127.0.0.1:8000/v1
# V4_EMBEDDING_API_KEY=...        only if the vLLM server requires one
# V4_EMBEDDING_TUNNEL_HOSTS=...   only for a plain-http host behind your own encrypted tunnel

set -a; . ./.env; set +a
docker compose up -d postgres    # pgvector/pgvector:pg16, persistent volume redator-v4-pgdata
```

The trusted vLLM server must serve `Qwen/Qwen3-Embedding-0.6B`; check with `curl $V4_EMBEDDING_BASE_URL/models`. Before any text is sent, the endpoint must be `https://` with certificate verification, a loopback address (including the local end of an `ssh -L` tunnel), or a host listed in `V4_EMBEDDING_TUNNEL_HOSTS` (FR-026). Credentials are read from the environment only; no command accepts them as arguments. Indexing the real dataset sends full cleaned legal text to that server, and retrieval sends every question.

## 1. Build a generation

Smoke check with the synthetic fixture (its fingerprint is passed explicitly; the real CSV needs no flags):

```bash
FIXTURE=specs/001-redator-v4-dataset-pipeline/contracts/fixture-v4.csv
python -m scripts.v4.run "$FIXTURE" "$(mktemp -d)/dataset" \
  --source-sha256 "$(sha256sum "$FIXTURE" | cut -c1-64)" --source-rows 2
```

Expected: **2 documents, 2 passages, 2 of 3 questions retained, 2 judgments**; the third question is audited as lacking an answer.

Production build:

```bash
python -m scripts.v4.run \
  datasets/redator_v4_repr-test_1k-val_2k-train_rest/redator_v4_repr-test_1k-val_2k-train_rest-test.csv \
  datasets/redator_v4_test
```

`datasets/redator_v4_test/current` then points at one immutable `generations/<id>/`. A train/validation file, altered bytes, or a failed or invalid build leaves `current` unchanged. Superseded generations are kept. `--max-query-tokens N` changes the question-length limit (512 `o200k_base` tokens by default, `0` disables it); like any policy change, it produces a new generation that needs its own index.

## 2. Build the Qwen index

```bash
python -m scripts.v4.index datasets/redator_v4_test [--model-revision <HF commit>]
```

Every published chunk is embedded once and stored as a `vector(1024)` row under a new build, which is activated only after the row count, chunk bijection, placement, and vector norms check out. A failure marks the new build `failed` and leaves the previous active build selected. Pass the same `--model-revision` (default `unpinned`) to retrieval, or retrieval refuses the build.

## 3. Rank questions

```bash
python -m scripts.v4.retrieve datasets/redator_v4_test/current/beir runs/v4/qwen-top10.json --top-k 10
python -m scripts.v4.retrieve datasets/redator_v4_test/current/beir runs/v4/qwen-top100.json --top-k 100
```

Only questions are embedded (under the versioned Qwen instruction). Each is ranked by exact cosine distance against its own document's candidates. A stale or incompatible build, a missing candidate, or a service failure stops the run before a file is replaced. Provenance, including the run's `cost` block, goes to `runs/v4.provenance/<run>.json`, outside the runs directory.

## 4. Compare runs

```bash
python -m scripts.v4.metrics datasets/redator_v4_test/current/beir/qrels/test.tsv runs/v4 \
  --k-values 1 3 5 10 20 100 --output reports/v4/comparison.json
python -m scripts.v4.plots reports/v4/comparison.json reports/v4/comparison.png
```

Every run is scored against the same judgments and cutoffs: NDCG, MAP, Recall, P, MRR, and Pass@k (the share of questions with a relevant passage in the top k). Judged questions missing from a run score zero and are counted in `queries_missing`. The report records the generation, the qrels SHA-256, and each run's measures. The chart shows one panel per metric and one bar per run.

## Validation record

**Automated suite (T036), 2026-09-23.** `V4_REQUIRE_PGVECTOR=1 python -m pytest tests/v4 -q --tb=short` passes: 383 tests on the current tree, against a throwaway database in the compose PostgreSQL and a fake vLLM over loopback HTTP. The container restart test (`V4_RUN_CONTAINER_TESTS=1`, or required under `V4_REQUIRE_PGVECTOR=1`) passed on commit `7e3542e` (367 passed, none skipped). It was deselected from the final run only because a suspended operator index process held a database connection that a restart would have broken. No automated test contacts the real vLLM server.

**Real test source (T037), 2026-09-23**, on a local vLLM serving `Qwen/Qwen3-Embedding-0.6B` (`max_model_len` 1,024) and the compose PostgreSQL 16.11 with pgvector 0.8.1:

| Measure | Value |
|---|---|
| Source SHA-256 / rows | `d10d21f2…fff3` (matches the contract) / 1,000 |
| Generation `20260923-210317-04e526` | 871 documents, 65,145 passages, 7,065 of 10,486 questions retained, 17,825 judgments; 129 sources rejected, 3,421 questions dropped (including 164 over the 512-token question limit) |
| Active index build `ib-20260923-210753-1334284f` | 65,145 `vector(1024)` rows, equal to `summary.chunks` |
| Retrieval, top-10 | 54.2 s total: 111 embedding requests, 524,948 prompt tokens, 36.0 s in vLLM, 14.1 s exact search |
| Document-scoped search latency | mean 2.0 ms, p50 1.0 ms, p95 6.8 ms, max 65.9 ms per question |
| PostgreSQL size | database 772 MB (`indexed_chunks` 764 MB, holding every retained build: two complete builds for two generations, plus a failed and an unfinished one); volume `redator-v4-pgdata` 1.12 GB |
| Qwen top-100 run | NDCG@10 0.464, Recall@10 0.641, MRR@10 0.469, Pass@10 0.808, Pass@100 0.985 |

The index build's own wall time was not recorded. The `cost` block covers retrieval runs only.
