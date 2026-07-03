<h1 align="center">GitHub Analyzer</h1>

<p align="center">
  <b>Turn any GitHub repository into a searchable, chat‑able knowledge base.</b><br>
  AI tutorials · pgvector RAG · streaming chat · multi‑repo RAG · agentic RAG · ontology graph · code‑gen LoRA.
</p>

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 개요 (한국어)

GitHub Analyzer는 공개/로컬 코드베이스를 크롤링해 **초보자용 튜토리얼**을 자동 생성하고, 그 결과를
**Supabase(Postgres + pgvector)** 에 저장한 뒤 **Streamlit 앱**에서 검색·대화·시각화할 수 있게 해줍니다.

- **Generate & Save** — GitHub repo → 튜토리얼(챕터 + Mermaid) 생성 후 DB 저장. 입력 URL 구조(전체/브랜치/하위폴더)를 자동 분석·설명
- **RAG Search** — 하이브리드(벡터 + 키워드) 검색으로 질문에 근거 있는 답변
- **Chat** — 저장된 코드베이스 지식으로 멀티턴 스트리밍 대화
- **Multi‑Repo RAG** — 여러 저장소를 함께 검색해 공통점·차이·연결 인사이트 종합
- **Agent** — 에이전트가 스스로 도구(검색·챕터읽기·온톨로지)를 골라 반복 탐색 후 답변(ReAct 하네스)
- **Ontology** — Mermaid/챕터/링크에서 추출한 개념 그래프
- **Fine‑tune** — 여러 레포의 코드 블록으로 코드생성 데이터셋 → 4GB GPU용 QLoRA 학습
- **Admin** — 튜토리얼 또는 **레포지토리 전체**를 관련 데이터까지 한 번에 삭제

> Built on [Pocket Flow](https://github.com/The-Pocket/PocketFlow), the 100‑line LLM framework, and originally
> forked from [Tutorial‑Codebase‑Knowledge](https://github.com/The-Pocket/PocketFlow-Tutorial-Codebase-Knowledge). MIT licensed.

---

## Features

| Area | What it does |
|------|--------------|
| **Tutorial generation** | PocketFlow pipeline: crawl → identify abstractions → analyze relationships → order chapters → write chapters → combine (`main.py`). Multi‑language, LLM‑response caching. |
| **Persistence** | Postgres + `pgvector` (works great on Supabase). Repositories, tutorials, chapters, chunks (+embeddings), ontology nodes/edges, fine‑tuning examples, RAG logs. |
| **Hybrid RAG** | Semantic vector search when embeddings exist, transparent fallback to concept‑aware keyword scoring otherwise. Per‑repo tuning via `rag_config.py`. |
| **Multi‑repo RAG** | Search several saved repos at once with fair per‑repo score normalization, then synthesize/compare across them (`search_across_tutorials`). |
| **Agentic RAG** | A ReAct agent (`agent_rag.py`) picks tools (search / read_chapter / ontology / finish) and iterates, with strict JSON actions, self‑correction, a step budget, and a visible trace. |
| **Deep Research** | Alternate agent mode: decompose the question into sub‑questions, retrieve cross‑repo evidence per sub‑question, then synthesize (`deep_research`). |
| **Judge (verify & refine)** | LLM‑as‑Judge scores an answer's groundedness (1–5); if weak/hallucinated it auto‑refines once. Available in the Agent and RAG Search tabs. |
| **Self‑healing Mermaid** | `mermaid_utils.heal_mermaid` repairs broken diagrams via the LLM; the Ontology tab previews and can save the fix. |
| **Tracing** | Agent / Deep Research runs are persisted (Postgres `agent_traces`, else JSON files) and browsable in‑app (`tracing.py`). |
| **Streaming chat** | Multi‑turn conversation over a tutorial's knowledge, per‑tutorial history, token‑by‑token streaming (Gemini & OpenAI‑compatible). |
| **Ontology graph** | Extracts a concept graph (Mermaid + chapter order + markdown links) and renders it. |
| **Repo‑wide delete** | `ON DELETE CASCADE` removes a repository and *all* its tutorials/chapters/chunks/embeddings/ontology/fine‑tuning/logs, with a pre‑delete count preview. |
| **Code‑gen fine‑tuning** | Turn real code blocks from stored repos into an (instruction → code) JSONL and train a 4‑bit QLoRA adapter that fits ≤4GB VRAM (`train_lora_local.py --load_4bit`, `infer_codegen_local.py`). |

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
  db_store.py            (Postgres/pgvector: save, search, cross-repo, ontology, delete, datasets, traces)
  agent_rag.py           (agentic RAG: ReAct loop, Judge verify/refine, Deep Research)
  mermaid_utils.py       (self-healing Mermaid repair)
  tracing.py             (persist/list/load run traces — DB or JSON files)
  rag_config.py          (per‑repo concept aliases / stop terms)
  app_full_workflow.py   (Streamlit UI — 10 tabs)
  backfill_embeddings.py (populate embeddings for existing chunks)

Layer C — Fine‑tuning (optional, GPU)
  train_lora_local.py    (LoRA / 4‑bit QLoRA, --load_4bit for ≤4GB VRAM)
  infer_lora_local.py / infer_codegen_local.py  (Qwen2.5 + PEFT adapters)
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
| **Generate & Save** | Run `main.py` on a repo and save to the DB (auto‑builds the ontology). Shows a live GitHub‑URL structure analysis (whole repo / branch / subdirectory) and how it flows downstream. |
| **Library** | Browse a saved tutorial: summary, flowchart, chapters. |
| **RAG Search** | Ask a question; hybrid retrieval + optional LLM answer. |
| **Chat** | Streaming multi‑turn conversation over the selected tutorial. |
| **Multi‑Repo RAG** | Cross‑repo search over several tutorials + streamed synthesis of common/different/connectable patterns. |
| **Agent** | Agentic RAG with a mode toggle — **ReAct** (tool loop) or **Deep Research** (sub‑question map‑reduce) — an optional **Judge** verify+refine pass, the step trace, and a viewer of past runs (Tracing). |
| **Ontology** | Rebuild/inspect the concept graph and Mermaid diagram; **🩹 self‑heal** a broken diagram and optionally save it. |
| **Fine‑tune** | Build a code‑gen JSONL from selected repos and get ready‑to‑run 4GB QLoRA train/infer commands. |
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

Two dataset flavors are built from stored repos:
- **Explain** (chapter → explanation) via `export_finetune_jsonl`.
- **Code‑gen** (instruction → real code block) via the **🛠 Fine‑tune** tab /
  `export_codegen_jsonl`, aggregated across the repos you select.

Train a small adapter — **4‑bit QLoRA fits ≤4GB VRAM (e.g. RTX 3050 Ti)**:

```bash
pip install -r requirements-train.txt   # torch, transformers, peft, bitsandbytes, ...

# 4GB VRAM (QLoRA): small model + 4-bit + short context
python train_lora_local.py --train_jsonl exports/codegen_dataset.jsonl \
  --load_4bit --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --out_dir finetuned_adapters/codegen_qwen05 --max_length 640 --epochs 2

# Generate code with the trained adapter
python infer_codegen_local.py \
  --adapter_dir finetuned_adapters/codegen_qwen05/adapter \
  --prompt "FastAPI 의존성 주입을 사용하는 엔드포인트 예시를 작성해줘" --load_4bit
```

Focus datasets on small units (functions, config, usage snippets). The Fine‑tune
tab reports example counts + language breakdown and prints these exact commands.

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
| `APP_USERNAME` / `APP_PASSWORD` | Optional shared login for the Streamlit app. When **both** are set, the app shows a login screen; leave both empty to disable the gate. See [Access control](#access-control-team-login). |

## Access control (team login)

The Streamlit app (`app_full_workflow.py`) can be restricted to your team with a
simple shared credential:

- Set `APP_USERNAME` and `APP_PASSWORD` (in `.env` locally, or in **Streamlit
  Secrets** when deployed). When both are present, a login screen gates the
  whole app; a logout button appears in the sidebar. If either is empty the gate
  is disabled (open access, with a warning).
- Credentials are read from the environment and **never hardcoded**, so
  committing the app can't leak the password. Comparison is constant‑time.

Secret handling in the sidebar is hardened for shared/deployed use: existing
keys (`*_API_KEY`, `GITHUB_TOKEN`, `DATABASE_URL`) are **never pre‑filled** into
inputs and the password reveal ("eye") button is hidden, so secrets can't be
read off the screen. To change a value you must explicitly opt in per session.

> This is a single shared credential, adequate for keeping an internal tool
> private. For per‑user accounts / audit logs, add a dedicated auth layer
> (e.g. `streamlit-authenticator` or your SSO). On Streamlit Community Cloud,
> also consider restricting **viewer emails** as a second layer.

## Project layout

```
main.py, flow.py, nodes.py     PocketFlow generation pipeline
utils/call_llm.py              LLM calls (cache + streaming)
utils/crawl_*.py               GitHub / local crawlers
db_store.py                    Postgres/pgvector: save, RAG, cross-repo, ontology, delete, datasets
agent_rag.py                   agentic RAG (ReAct loop, Judge, Deep Research)
mermaid_utils.py               self-healing Mermaid    tracing.py  run-trace store
rag_config.py                  per‑repo search tuning
schema.sql                     database schema (pgvector, cascade)
app_full_workflow.py           Streamlit app (10 tabs)
.streamlit/config.toml         app theme
backfill_embeddings.py         embedding backfill utility
train_lora_local.py            LoRA / QLoRA training
infer_lora_local.py            explain inference    infer_codegen_local.py  code-gen inference
requirements.txt               app deps             requirements-train.txt  training deps
legacy/                        archived earlier Streamlit apps
```

## Credits

Built on [Pocket Flow](https://github.com/The-Pocket/PocketFlow) and forked from
[The‑Pocket/PocketFlow‑Tutorial‑Codebase‑Knowledge](https://github.com/The-Pocket/PocketFlow-Tutorial-Codebase-Knowledge).
Licensed under the MIT License — see [LICENSE](./LICENSE).
