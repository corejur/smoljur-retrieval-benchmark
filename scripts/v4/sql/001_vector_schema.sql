-- v4 Qwen chunk index in PostgreSQL/pgvector (contracts/model-index.md).
--
-- Idempotent: safe to run on every connection. Exact search only: a B-tree
-- narrows rows to one build and document before cosine-distance ordering.
-- No HNSW/IVFFlat index is created, because pgvector filters those after
-- the approximate scan and can under-return within one document.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS index_builds (
    index_build_id             text        PRIMARY KEY,
    model_id                   text        NOT NULL,
    model_revision             text        NOT NULL,
    generation_id              text        NOT NULL,
    generation_manifest_sha256 text        NOT NULL CHECK (generation_manifest_sha256 ~ '^[0-9a-f]{64}$'),
    source_sha256              text        NOT NULL,
    split                      text        NOT NULL CHECK (split = 'test'),
    dimension                  integer     NOT NULL CHECK (dimension = 1024),
    metric                     text        NOT NULL CHECK (metric = 'cosine'),
    normalized                 boolean     NOT NULL CHECK (normalized),
    query_instruction          text        NOT NULL,
    chunk_text_profile         text        NOT NULL,
    chunk_count                integer     NOT NULL CHECK (chunk_count >= 0),
    status                     text        NOT NULL DEFAULT 'building'
        CHECK (status IN ('building', 'validated', 'active', 'superseded', 'failed')),
    created_at                 timestamptz NOT NULL DEFAULT now(),
    updated_at                 timestamptz NOT NULL DEFAULT now(),
    -- Target of the active pointer's composite key: a pointer can only name
    -- a build of its own generation and model.
    UNIQUE (index_build_id, generation_id, model_id)
);

CREATE TABLE IF NOT EXISTS indexed_chunks (
    index_build_id text          NOT NULL REFERENCES index_builds (index_build_id),
    chunk_id       text          NOT NULL,
    document_id    text          NOT NULL,
    chunk_index    integer       NOT NULL CHECK (chunk_index >= 0),
    embedding      vector(1024)  NOT NULL,
    PRIMARY KEY (index_build_id, chunk_id),
    UNIQUE (index_build_id, document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS indexed_chunks_build_document
    ON indexed_chunks USING btree (index_build_id, document_id, chunk_id);

CREATE TABLE IF NOT EXISTS active_model_indexes (
    generation_id  text        NOT NULL,
    model_id       text        NOT NULL,
    index_build_id text        NOT NULL,
    activated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (generation_id, model_id),
    FOREIGN KEY (index_build_id, generation_id, model_id)
        REFERENCES index_builds (index_build_id, generation_id, model_id)
);

-- Rows are written only while their build is `building`, and never changed
-- afterwards: a validated build is immutable, so a reader pinned to it keeps
-- seeing exactly the rows that were validated.
CREATE OR REPLACE FUNCTION indexed_chunks_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    build_status text;
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'indexed_chunks rows are immutable (build %)', OLD.index_build_id;
    END IF;
    SELECT status INTO build_status
      FROM index_builds
     WHERE index_build_id = CASE WHEN TG_OP = 'INSERT' THEN NEW.index_build_id
                                 ELSE OLD.index_build_id END;
    IF build_status IS DISTINCT FROM 'building' THEN
        RAISE EXCEPTION 'indexed_chunks rows of a % build are immutable', build_status;
    END IF;
    RETURN CASE WHEN TG_OP = 'INSERT' THEN NEW ELSE OLD END;
END;
$$;

CREATE OR REPLACE TRIGGER indexed_chunks_immutable
    BEFORE INSERT OR UPDATE OR DELETE ON indexed_chunks
    FOR EACH ROW EXECUTE FUNCTION indexed_chunks_immutable();
