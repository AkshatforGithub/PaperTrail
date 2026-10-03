-- Embedding size is filled in from config (EMBEDDING_DIM) by init_db(),
-- so don't paste this file into psql directly.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id    TEXT PRIMARY KEY,               
    title       TEXT NOT NULL,
    abstract    TEXT NOT NULL DEFAULT '',
    authors     TEXT[] NOT NULL DEFAULT '{}',
    categories  TEXT[] NOT NULL DEFAULT '{}', 
    published   TIMESTAMPTZ,
    pdf_url     TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    arxiv_id    TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    chunk_index INT  NOT NULL,                  -- position within the paper
    section     TEXT,                           -- e.g. 'Introduction', if detected
    content     TEXT NOT NULL,
    embedding   vector(__EMBEDDING_DIM__) NOT NULL,
    -- keyword-search column for hybrid retrieval, maintained automatically
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (arxiv_id, chunk_index)
);

-- Approximate nearest-neighbour index for vector search (cosine distance)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text index for keyword search
CREATE INDEX IF NOT EXISTS chunks_tsv_gin
    ON chunks USING gin (tsv);