-- ============================================================
-- schema.sql
-- Codebase Tutorial Full Workflow — Postgres schema
--
-- This file was reconstructed from the live production database
-- (information_schema introspection) so it matches what db_store.py
-- expects exactly: UUID primary keys, pgvector embeddings (1536-dim,
-- OpenAI text-embedding-3-small), a generated full-text search column,
-- and ON DELETE CASCADE so delete_tutorial_v3() cleans up children.
--
-- Usage:
--   psql "$DATABASE_URL" -f schema.sql
-- Safe to re-run: every statement is idempotent (IF NOT EXISTS).
-- ============================================================

-- Extensions -------------------------------------------------
-- pgvector: required for the chunks.embedding column.
CREATE EXTENSION IF NOT EXISTS vector;
-- pgcrypto: provides gen_random_uuid() on PostgreSQL < 13
-- (built into core on 13+, so this is a no-op harmless safety net).
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- repositories -----------------------------------------------
CREATE TABLE IF NOT EXISTS repositories (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_url    text NOT NULL UNIQUE,
    repo_name   text,
    branch      text,
    sub_path    text,
    created_at  timestamptz DEFAULT now()
);

-- tutorials --------------------------------------------------
CREATE TABLE IF NOT EXISTS tutorials (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id    uuid REFERENCES repositories(id) ON DELETE CASCADE,
    title            text,
    summary          text,
    source_repo_url  text,
    index_markdown   text,
    mermaid_graph    text,
    model_provider   text,
    model_name       text,
    language         text,
    max_abstractions integer,
    output_dir       text,
    created_at       timestamptz DEFAULT now()
);

-- chapters ---------------------------------------------------
CREATE TABLE IF NOT EXISTS chapters (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tutorial_id  uuid REFERENCES tutorials(id) ON DELETE CASCADE,
    chapter_no   integer NOT NULL,
    title        text,
    filename     text,
    markdown     text,
    created_at   timestamptz DEFAULT now()
);

-- chunks -----------------------------------------------------
-- search_tsv is a STORED generated column so keyword search works
-- with no application-side maintenance. embedding is optional
-- (only populated when create_embeddings=True in save_tutorial_result_to_db).
CREATE TABLE IF NOT EXISTS chunks (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tutorial_id      uuid REFERENCES tutorials(id) ON DELETE CASCADE,
    chapter_id       uuid REFERENCES chapters(id) ON DELETE CASCADE,
    chunk_index      integer NOT NULL,
    content          text NOT NULL,
    metadata         jsonb,
    embedding        vector(1536),
    embedding_model  text,
    search_tsv       tsvector GENERATED ALWAYS AS
                        (to_tsvector('simple'::regconfig, COALESCE(content, ''::text))) STORED,
    created_at       timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_tutorial_idx ON chunks USING btree (tutorial_id);
CREATE INDEX IF NOT EXISTS chunks_search_idx   ON chunks USING gin (search_tsv);

-- ontology_nodes ---------------------------------------------
CREATE TABLE IF NOT EXISTS ontology_nodes (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tutorial_id  uuid REFERENCES tutorials(id) ON DELETE CASCADE,
    node_key     text NOT NULL,
    node_type    text NOT NULL,
    label        text NOT NULL,
    properties   jsonb,
    created_at   timestamptz DEFAULT now(),
    UNIQUE (tutorial_id, node_key)
);

-- ontology_edges ---------------------------------------------
CREATE TABLE IF NOT EXISTS ontology_edges (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tutorial_id     uuid REFERENCES tutorials(id) ON DELETE CASCADE,
    source_node_id  uuid REFERENCES ontology_nodes(id) ON DELETE CASCADE,
    target_node_id  uuid REFERENCES ontology_nodes(id) ON DELETE CASCADE,
    edge_type       text NOT NULL,
    label           text,
    properties      jsonb,
    created_at      timestamptz DEFAULT now()
);

-- fine_tuning_examples ---------------------------------------
-- chapter_id uses ON DELETE SET NULL (an example survives if its
-- source chapter is removed); tutorial_id cascades.
CREATE TABLE IF NOT EXISTS fine_tuning_examples (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tutorial_id    uuid REFERENCES tutorials(id) ON DELETE CASCADE,
    chapter_id     uuid REFERENCES chapters(id) ON DELETE SET NULL,
    task_type      text NOT NULL,
    question       text NOT NULL,
    answer         text NOT NULL,
    messages       jsonb NOT NULL,
    source         text,
    approved       boolean DEFAULT false,
    quality_score  integer,
    metadata       jsonb,
    created_at     timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS fine_tuning_examples_tutorial_idx ON fine_tuning_examples USING btree (tutorial_id);
CREATE INDEX IF NOT EXISTS fine_tuning_examples_approved_idx ON fine_tuning_examples USING btree (approved);

-- rag_logs ---------------------------------------------------
-- Present in the production DB (query/answer audit trail). Not written
-- by the current app code, but included so a fresh DB matches prod and
-- the tutorial ON DELETE CASCADE chain is complete.
CREATE TABLE IF NOT EXISTS rag_logs (
    id                            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tutorial_id                   uuid REFERENCES tutorials(id) ON DELETE CASCADE,
    question                      text NOT NULL,
    answer                        text,
    retrieved_chunk_ids           uuid[],
    retrieved_ontology_node_ids   uuid[],
    metadata                      jsonb,
    created_at                    timestamptz DEFAULT now()
);
