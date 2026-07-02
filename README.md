<h1 align="center">GitHub Analyzer</h1>

<p align="center">
  <b>Turn any GitHub repository into a searchable, chat‑able knowledge base.</b><br>
  AI‑generated codebase tutorials · Postgres/pgvector RAG · streaming chat · ontology graph · LoRA fine‑tuning.
</p>

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 개요 (한국어)

GitHub Analyzer는 공개/로컬 코드베이스를 크롤링해 **초보자용 튜토리얼**을 자동 생성하고, 그 결과를
**Supabase(Postgres + pgvector)** 에 저장한 뒤 **Streamlit 앱**에서 검색·대화·시각화할 수 있게 해줍니다.

- **Generate & Save** — GitHub repo → 튜토리얼(챕터 + Mermaid 다이어그램) 생성 후 DB 저장
- **RAG Search** — 하이브리드(벡터 + 키워드) 검색으로 질문에 근거 있는 답변
- **Chat** — 저장된 코드베이스 지식으로 멀티턴 스트리밍 대화
- **Ontology RAG** — Mermaid/챕터/링크에서 추출한 개념 그래프
- **Admin** — 튜토리얼 또는 **레포지토리 전체**를 관련 데이터까지 한 번에 삭제
- **Fine‑tuning** — 생성된 튜토리얼로 로컬 LoRA 학습(선택)

> Built on [Pocket Flow](https://github.com/The-Pocket/PocketFlow), the 100‑line LLM framework, and originally
> forked from [Tutorial‑Codebase‑Knowledge](https://github.com/The-Pocket/PocketFlow-Tutorial-Codebase-Knowledge). MIT licensed.

---

## Features

| Area | What it does |
|------|--------------|
| **Tutorial generation** | PocketFlow pipeline: crawl → identify abstractions → analyze relationships → order chapters → write chapters → combine (`main.py`). Multi‑language, LLM‑response caching. |
| **Persistence** | Postgres + `pgvector` (works great on Supabase). Repositories, tutorials, chapters, chunks (+embeddings), ontology nodes/edges, fine‑tuning examples, RAG logs. |
| **Hybrid RAG** | Semantic vector search when embeddings exist, transparent fallback to concept‑aware keyword scoring otherwise. Per‑repo tuning via `rag_config.py`. |
| **Streaming chat** | Multi‑turn conversation over a tutorial's knowledge, per‑tutorial history, token‑by‑token streaming (Gemini & OpenAI‑compatible). |
| **Ontology graph** | Extracts a concept graph (Mermaid + chapter order + markdown links) and renders it. |
| **Repo‑wide delete** | `ON DELETE CASCADE` removes a repository and *all* its tutorials/chapters/chunks/embeddings/ontology/fine‑tuning/logs, with a pre‑delete count preview. |
| **Fine‑tuning** | Export approved Q&A as JSONL and train a local LoRA adapter (`train_lora_local.py` / `infer_lora_local.py`). |

## Architecture

```
Layer A — Generation pipeline (PocketFlow)
  main.py → flow.py → nodes.py
    FetchRepo → IdentifyAbstractions → AnalyzeRelationships
              → OrderChapters → WriteChapters → CombineTutorial
    utils/call_llm.py   (Gemini | OpenAI‑compatible, cache + streaming)
    utils/crawl_*.py    (GitHub API / local files)
  → writes output/<project>/index.md + NN_*.md

Layer B — Storage, RAG & app
  db_store.py            (Postgres/pgvector: save, search, ontology, delete, fine‑tune)
  rag_config.py          (per‑repo concept aliases / stop terms)
  app_full_workflow.py   (Streamlit UI — 7 tabs)
  backfill_embeddings.py (populate embeddings for existing chunks)

Layer C — Fine‑tuning (optional, GPU)
  train_lora_local.py / infer_lora_local.py  (Qwen2.5 + PEFT LoRA)
```

## Requirements

- Python 3.10+
- A Postgres database with the `pgvector` extension (e.g. **Supabase**)
- An LLM provider key: **Gemini** (`GEMINI_API_KEY`) and/or an **OpenAI‑compatible** endpoint
- Optional: `OPENAI_API_KEY` for embeddings (enables semantic search) and a `GITHUB_TOKEN` (rate limits / private repos)

## Getting Started

### 1. Install

```bash
git clone https://github.com/xaikorea/GitHub-Analyzer.git
cd GitHub-Analyzer
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.sample` to `.env` and fill in your values (provider keys, `DATABASE_URL`, etc.).
`.env*` files are gitignored except the sample.

### 3. Initialize the database

```bash
psql "$DATABASE_URL" -f schema.sql
```

Idempotent (safe to re‑run): creates the `pgvector` extension, all tables, indexes,
and `ON DELETE CASCADE` foreign keys.

### 4a. Generate a tutorial from the CLI

```bash
# GitHub repo
python main.py --repo https://github.com/username/repo \
  --include "*.py" "*.md" --exclude "tests/*" --max-size 60000 --language english

# Local directory
python main.py --dir /path/to/codebase --include "*.py" --exclude "*test*"
```

Key flags: `--repo`/`--dir` (required, exclusive), `-o/--output`, `-i/--include`, `-e/--exclude`,
`-s/--max-size`, `--language`, `--max-abstractions`, `--no-cache`.

### 4b. Or run the Streamlit app (recommended)

```bash
streamlit run app_full_workflow.py
```

| Tab | Purpose |
|-----|---------|
| **Setup** | Verify provider / model / DB / token status. |
| **Generate & Save** | Run `main.py` on a repo and save the result to the DB (auto‑builds the ontology). |
| **Library** | Browse a saved tutorial: summary, flowchart, chapters. |
| **RAG Search** | Ask a question; hybrid retrieval + optional LLM answer. |
| **Chat** | Streaming multi‑turn conversation over the selected tutorial. |
| **Ontology RAG** | Rebuild/inspect the concept graph and Mermaid diagram. |
| **Admin** | Delete a single tutorial **or an entire repository** (all tutorials), with a count preview and optional local‑folder cleanup. |

## Semantic Search & Embeddings

By default, tutorials are saved **without** embeddings and search uses concept‑aware keyword scoring.
To enable semantic (vector) search:

- Set `RAG_CREATE_EMBEDDINGS=1` (needs `OPENAI_API_KEY`) so new saves embed their chunks, **and/or**
- Backfill existing chunks:

```bash
python backfill_embeddings.py --dry-run   # report count & rough cost (no API calls)
python backfill_embeddings.py             # embed all chunks (idempotent, resumable)
```

Once embeddings exist, `search_tutorial_context_v4` automatically prefers vector search
and falls back to keyword scoring when they don't. Tune domain concepts in `rag_config.py`.

## Fine‑tuning (optional, GPU)

```bash
pip install -r requirements-train.txt   # heavy (torch, transformers, peft, ...)
python train_lora_local.py --train_jsonl exports/finetune_dataset.jsonl
python infer_lora_local.py --adapter_dir finetuned_adapters/.../adapter --prompt "..." --language Korean
```

Training data (messages JSONL) is generated from stored chapters via the
`create_finetune_examples_from_tutorial` / `export_finetune_jsonl` helpers in `db_store.py`.

## Configuration reference

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `GEMINI`, `OPENAI`, or any OpenAI‑compatible provider name |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Gemini via AI Studio (or `GEMINI_PROJECT_ID`/`GEMINI_LOCATION` for Vertex) |
| `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL` | OpenAI‑compatible chat |
| `OPENAI_EMBEDDING_MODEL` | Embedding model (default `text-embedding-3-small`, 1536‑dim) |
| `DATABASE_URL` | Postgres/Supabase connection string |
| `GITHUB_TOKEN` | Higher rate limits / private repos |
| `RAG_CREATE_EMBEDDINGS` | `1` to embed chunks on save |
| `MAX_ABSTRACTION_CONTEXT_CHARS` | Budget for the abstraction prompt on huge repos (default 900000) |

## Project layout

```
main.py, flow.py, nodes.py     PocketFlow generation pipeline
utils/call_llm.py              LLM calls (cache + streaming)
utils/crawl_*.py               GitHub / local crawlers
db_store.py                    Postgres/pgvector: save, RAG, ontology, delete, fine‑tune
rag_config.py                  per‑repo search tuning
schema.sql                     database schema (pgvector, cascade)
app_full_workflow.py           Streamlit app (7 tabs)
backfill_embeddings.py         embedding backfill utility
train_lora_local.py            LoRA training      infer_lora_local.py  LoRA inference
requirements.txt               app deps           requirements-train.txt  training deps
legacy/                        archived earlier Streamlit apps
```

## Credits

Built on [Pocket Flow](https://github.com/The-Pocket/PocketFlow) and forked from
[The‑Pocket/PocketFlow‑Tutorial‑Codebase‑Knowledge](https://github.com/The-Pocket/PocketFlow-Tutorial-Codebase-Knowledge).
Licensed under the MIT License — see [LICENSE](./LICENSE).
