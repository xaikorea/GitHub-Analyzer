import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Any

import psycopg
import requests
from dotenv import load_dotenv


load_dotenv(override=True)


def get_conn():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(db_url)


def slug_from_repo_url(repo_url: str) -> dict[str, str | None]:
    """
    https://github.com/fastapi/fastapi/tree/master/fastapi
    같은 URL에서 repo_name, branch, sub_path를 대략 추출.
    """
    result = {
        "repo_name": None,
        "branch": None,
        "sub_path": None,
    }

    m = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
    if m:
        result["repo_name"] = f"{m.group(1)}/{m.group(2).replace('.git', '')}"

    tree = re.search(r"/tree/([^/]+)/(.*)$", repo_url)
    if tree:
        result["branch"] = tree.group(1)
        result["sub_path"] = tree.group(2)

    return result


def describe_repo_url(url: str) -> dict:
    """Parse a GitHub URL into its structural parts so the UI can explain
    exactly what will be analyzed.

    Handles:
      https://github.com/owner/repo                     -> whole repo, default branch
      https://github.com/owner/repo/tree/<ref>          -> that branch/commit, whole
      https://github.com/owner/repo/tree/<ref>/<path>   -> only that subdirectory
      https://github.com/owner/repo/blob/<ref>/<file>   -> a single file (usually not intended)
      git@github.com:owner/repo.git / *.git             -> whole repo
    Returns a dict of facts (no UI text): valid, owner, repo, repo_name, kind
    ('repo'|'tree'|'blob'|'invalid'), ref, sub_path, scope
    ('whole_repo'|'subdirectory'|'single_file'), canonical_url.
    """
    url = (url or "").strip()
    result = {
        "valid": False, "owner": None, "repo": None, "repo_name": None,
        "kind": "invalid", "ref": None, "sub_path": None,
        "scope": None, "canonical_url": url,
    }

    m = re.search(r"github\.com[:/]+([^/]+)/([^/#?]+)", url)
    if not m:
        return result

    owner = m.group(1)
    repo = m.group(2).replace(".git", "")
    result.update(valid=True, owner=owner, repo=repo, repo_name=f"{owner}/{repo}", kind="repo")

    tb = re.search(r"/(tree|blob)/([^/]+)(?:/(.*))?$", url)
    if tb:
        result["kind"] = tb.group(1)
        result["ref"] = tb.group(2)
        sub = (tb.group(3) or "").strip("/")
        result["sub_path"] = sub or None

    if result["kind"] == "blob":
        result["scope"] = "single_file"
    elif result["sub_path"]:
        result["scope"] = "subdirectory"
    else:
        result["scope"] = "whole_repo"

    canonical = f"https://github.com/{owner}/{repo}"
    if result["ref"]:
        canonical += f"/{result['kind']}/{result['ref']}"
        if result["sub_path"]:
            canonical += f"/{result['sub_path']}"
    result["canonical_url"] = canonical

    return result


def extract_mermaid(markdown: str) -> str:
    """
    ```mermaid ... ``` 블록을 추출.
    없으면 flowchart TD로 시작하는 코드 조각을 최대한 추출.
    """
    m = re.search(r"```mermaid\s*(.*?)```", markdown, flags=re.S | re.I)
    if m:
        return m.group(1).strip()

    m = re.search(r"(flowchart\s+TD.*?)(?:\n#{1,6}\s|\nChapters|\Z)", markdown, flags=re.S | re.I)
    if m:
        return m.group(1).strip()

    return ""


def extract_summary(index_md: str) -> str:
    """
    제목과 Source Repository 사이의 본문 요약을 추출.
    """
    text = re.sub(r"^# .*\n", "", index_md, count=1).strip()
    text = re.split(r"\n\*\*Source Repository:\*\*|\nSource Repository:", text)[0]
    text = re.split(r"```mermaid|flowchart\s+TD", text)[0]
    return text.strip()


def extract_title(index_md: str, fallback: str = "Untitled") -> str:
    m = re.search(r"^#\s+(.+)$", index_md, flags=re.M)
    return m.group(1).strip() if m else fallback


def parse_chapter_no(filename: str) -> int:
    m = re.match(r"(\d+)_", filename)
    return int(m.group(1)) if m else 999


def split_markdown_into_chunks(markdown: str, max_chars: int = 1800, overlap: int = 200) -> list[str]:
    """
    저장공간을 아끼는 단순 chunker.
    heading 기준으로 먼저 나눈 뒤 너무 긴 블록은 문자 길이 기준으로 분할.
    """
    markdown = markdown.strip()
    if not markdown:
        return []

    sections = re.split(r"(?=\n#{1,4}\s+)", "\n" + markdown)
    chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        if len(sec) <= max_chars:
            chunks.append(sec)
            continue

        start = 0
        while start < len(sec):
            end = min(start + max_chars, len(sec))
            chunk = sec[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(sec):
                break
            start = max(0, end - overlap)

    return chunks


def openai_embedding(text: str) -> list[float] | None:
    """
    OpenAI embedding 생성.
    OPENAI_API_KEY가 없으면 None 반환.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")

    resp = requests.post(
        f"{base_url}/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": text[:8000],
        },
        timeout=60,
    )

    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI embedding error: {resp.status_code} {resp.text}")

    return resp.json()["data"][0]["embedding"]


def vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def parse_mermaid_ontology(mermaid: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Mermaid flowchart에서 Abstraction node/edge를 추출.
    예:
      A0["애플리케이션 인스턴스"]
      A0 -- "포함합니다" --> A4
    """
    nodes = {}
    edges = []

    for line in mermaid.splitlines():
        line = line.strip()
        if not line or line.startswith("flowchart"):
            continue

        node_match = re.match(r'([A-Za-z]\w*)\s*\[\s*"?(.*?)"?\s*\]', line)
        if node_match:
            key = node_match.group(1).strip()
            label = node_match.group(2).strip().replace("\\n", " ")
            nodes[key] = {
                "node_key": key,
                "node_type": "Abstraction",
                "label": label,
                "properties": {},
            }
            continue

        edge_match = re.match(r'([A-Za-z]\w*)\s*--\s*"?(.*?)"?\s*-->\s*([A-Za-z]\w*)', line)
        if edge_match:
            edges.append({
                "source_key": edge_match.group(1).strip(),
                "label": edge_match.group(2).strip(),
                "target_key": edge_match.group(3).strip(),
                "edge_type": "RELATED_TO",
                "properties": {},
            })

    return list(nodes.values()), edges


def save_tutorial_result_to_db(
    result_dir: str | Path,
    repo_url: str,
    provider: str,
    model_name: str,
    language: str = "Korean",
    max_abstractions: int | None = None,
    create_embeddings: bool | None = None,
) -> str:
    """
    output_xxx/ProjectName 폴더를 DB에 저장.
    반환: tutorial_id

    create_embeddings: True/False to force; None (default) reads the
    RAG_CREATE_EMBEDDINGS env flag (off unless set to 1/true/yes/on).
    Embeddings require OPENAI_API_KEY and enable semantic RAG search.
    """
    if create_embeddings is None:
        create_embeddings = os.getenv("RAG_CREATE_EMBEDDINGS", "").strip().lower() in (
            "1", "true", "yes", "on",
        )

    result_dir = Path(result_dir)
    index_path = result_dir / "index.md"

    if not index_path.exists():
        raise FileNotFoundError(f"index.md not found: {index_path}")

    index_md = index_path.read_text(encoding="utf-8", errors="replace")
    title = extract_title(index_md, fallback=result_dir.name)
    summary = extract_summary(index_md)
    mermaid = extract_mermaid(index_md)

    repo_meta = slug_from_repo_url(repo_url)
    chapter_files = sorted(
        [p for p in result_dir.glob("*.md") if p.name != "index.md"],
        key=lambda p: parse_chapter_no(p.name),
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into repositories (repo_url, repo_name, branch, sub_path)
                values (%s, %s, %s, %s)
                on conflict (repo_url)
                do update set
                  repo_name = excluded.repo_name,
                  branch = excluded.branch,
                  sub_path = excluded.sub_path
                returning id
                """,
                (
                    repo_url,
                    repo_meta["repo_name"],
                    repo_meta["branch"],
                    repo_meta["sub_path"],
                ),
            )
            repository_id = cur.fetchone()[0]

            cur.execute(
                """
                insert into tutorials (
                  repository_id, title, summary, source_repo_url, index_markdown,
                  mermaid_graph, model_provider, model_name, language,
                  max_abstractions, output_dir
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                returning id
                """,
                (
                    repository_id,
                    title,
                    summary,
                    repo_url,
                    index_md,
                    mermaid,
                    provider,
                    model_name,
                    language,
                    max_abstractions,
                    str(result_dir),
                ),
            )
            tutorial_id = cur.fetchone()[0]

            # ontology nodes / edges 저장
            ontology_nodes, ontology_edges = parse_mermaid_ontology(mermaid)
            node_id_by_key = {}

            for node in ontology_nodes:
                cur.execute(
                    """
                    insert into ontology_nodes (tutorial_id, node_key, node_type, label, properties)
                    values (%s, %s, %s, %s, %s)
                    on conflict (tutorial_id, node_key)
                    do update set label = excluded.label, properties = excluded.properties
                    returning id
                    """,
                    (
                        tutorial_id,
                        node["node_key"],
                        node["node_type"],
                        node["label"],
                        json.dumps(node["properties"], ensure_ascii=False),
                    ),
                )
                node_id_by_key[node["node_key"]] = cur.fetchone()[0]

            for edge in ontology_edges:
                src = node_id_by_key.get(edge["source_key"])
                dst = node_id_by_key.get(edge["target_key"])
                if not src or not dst:
                    continue

                cur.execute(
                    """
                    insert into ontology_edges (
                      tutorial_id, source_node_id, target_node_id,
                      edge_type, label, properties
                    )
                    values (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        tutorial_id,
                        src,
                        dst,
                        edge["edge_type"],
                        edge["label"],
                        json.dumps(edge["properties"], ensure_ascii=False),
                    ),
                )

            # chapters / chunks 저장
            for chapter_path in chapter_files:
                markdown = chapter_path.read_text(encoding="utf-8", errors="replace")
                chapter_title = extract_title(markdown, fallback=chapter_path.stem)
                chapter_no = parse_chapter_no(chapter_path.name)

                cur.execute(
                    """
                    insert into chapters (tutorial_id, chapter_no, title, filename, markdown)
                    values (%s, %s, %s, %s, %s)
                    returning id
                    """,
                    (
                        tutorial_id,
                        chapter_no,
                        chapter_title,
                        chapter_path.name,
                        markdown,
                    ),
                )
                chapter_id = cur.fetchone()[0]

                chunks = split_markdown_into_chunks(markdown)

                for idx, content in enumerate(chunks):
                    metadata = {
                        "chapter_no": chapter_no,
                        "chapter_title": chapter_title,
                        "filename": chapter_path.name,
                    }

                    embedding = None
                    embedding_model = None

                    if create_embeddings:
                        embedding = openai_embedding(content)
                        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                        time.sleep(0.05)

                    if embedding:
                        cur.execute(
                            """
                            insert into chunks (
                              tutorial_id, chapter_id, chunk_index, content,
                              metadata, embedding, embedding_model
                            )
                            values (%s, %s, %s, %s, %s, %s::vector, %s)
                            """,
                            (
                                tutorial_id,
                                chapter_id,
                                idx,
                                content,
                                json.dumps(metadata, ensure_ascii=False),
                                vector_literal(embedding),
                                embedding_model,
                            ),
                        )
                    else:
                        cur.execute(
                            """
                            insert into chunks (
                              tutorial_id, chapter_id, chunk_index, content, metadata
                            )
                            values (%s, %s, %s, %s, %s)
                            """,
                            (
                                tutorial_id,
                                chapter_id,
                                idx,
                                content,
                                json.dumps(metadata, ensure_ascii=False),
                            ),
                        )

        conn.commit()

    return str(tutorial_id)


def list_tutorials(limit: int = 50) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  t.id::text,
                  t.title,
                  t.source_repo_url,
                  t.model_provider,
                  t.model_name,
                  t.language,
                  t.created_at
                from tutorials t
                order by t.created_at desc
                limit %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "title": r[1],
            "source_repo_url": r[2],
            "model_provider": r[3],
            "model_name": r[4],
            "language": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]












# -----------------------------
# Fine-tuning dataset utilities
# -----------------------------

import json
from pathlib import Path


def create_finetune_examples_from_tutorial(
    tutorial_id: str,
    system_prompt: str | None = None,
    approve: bool = False,
) -> int:
    """
    저장된 chapter를 기반으로 SFT용 messages JSONL 후보를 생성한다.
    주의: 이 함수는 '학습 실행'이 아니라 '학습 데이터셋 후보 생성'이다.
    """
    if system_prompt is None:
        system_prompt = (
            "너는 오픈소스 코드베이스를 한국어로 설명하는 기술 튜터다. "
            "답변은 구조적이고, 코드베이스의 핵심 개념과 관계를 초보자도 이해할 수 있게 설명한다. "
            "근거 없는 추측은 하지 않는다."
        )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id::text, chapter_no, title, markdown
                from chapters
                where tutorial_id = %s
                order by chapter_no asc
                """,
                (tutorial_id,),
            )
            chapters = cur.fetchall()

            inserted = 0

            for chapter_id, chapter_no, title, markdown in chapters:
                clean_answer = markdown.strip()
                if not clean_answer:
                    continue

                examples = [
                    {
                        "task_type": "explain_chapter",
                        "question": f"{title}에 대해 초보자도 이해할 수 있게 튜토리얼 형식으로 설명해줘.",
                        "answer": clean_answer,
                    },
                    {
                        "task_type": "summarize_chapter",
                        "question": f"{title}의 핵심 개념과 코드 구조상 역할을 요약해줘.",
                        "answer": clean_answer,
                    },
                ]

                for ex in examples:
                    messages = [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": ex["question"],
                        },
                        {
                            "role": "assistant",
                            "content": ex["answer"],
                        },
                    ]

                    cur.execute(
                        """
                        insert into fine_tuning_examples (
                          tutorial_id, chapter_id, task_type,
                          question, answer, messages,
                          source, approved, quality_score, metadata
                        )
                        values (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            tutorial_id,
                            chapter_id,
                            ex["task_type"],
                            ex["question"],
                            ex["answer"],
                            json.dumps(messages, ensure_ascii=False),
                            "chapter_auto",
                            approve,
                            None,
                            json.dumps(
                                {
                                    "chapter_no": chapter_no,
                                    "chapter_title": title,
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    )

                    inserted += 1

        conn.commit()

    return inserted


def list_finetune_examples(tutorial_id: str, only_approved: bool = False, limit: int = 100):
    with get_conn() as conn:
        with conn.cursor() as cur:
            if only_approved:
                cur.execute(
                    """
                    select id::text, task_type, question, approved, quality_score, created_at
                    from fine_tuning_examples
                    where tutorial_id = %s and approved = true
                    order by created_at desc
                    limit %s
                    """,
                    (tutorial_id, limit),
                )
            else:
                cur.execute(
                    """
                    select id::text, task_type, question, approved, quality_score, created_at
                    from fine_tuning_examples
                    where tutorial_id = %s
                    order by created_at desc
                    limit %s
                    """,
                    (tutorial_id, limit),
                )

            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "task_type": r[1],
            "question": r[2],
            "approved": r[3],
            "quality_score": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


def approve_finetune_example(example_id: str, quality_score: int = 5):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update fine_tuning_examples
                set approved = true,
                    quality_score = %s
                where id = %s
                """,
                (quality_score, example_id),
            )
        conn.commit()


def export_finetune_jsonl(
    tutorial_id: str,
    out_path: str = "exports/finetune_dataset.jsonl",
    only_approved: bool = True,
) -> str:
    """
    OpenAI SFT 호환에 가까운 messages JSONL 파일로 export.
    각 줄:
    {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with get_conn() as conn:
        with conn.cursor() as cur:
            if only_approved:
                cur.execute(
                    """
                    select messages
                    from fine_tuning_examples
                    where tutorial_id = %s
                      and approved = true
                    order by created_at asc
                    """,
                    (tutorial_id,),
                )
            else:
                cur.execute(
                    """
                    select messages
                    from fine_tuning_examples
                    where tutorial_id = %s
                    order by created_at asc
                    """,
                    (tutorial_id,),
                )

            rows = cur.fetchall()

    with out.open("w", encoding="utf-8") as f:
        for (messages,) in rows:
            if isinstance(messages, str):
                messages = json.loads(messages)

            f.write(
                json.dumps(
                    {"messages": messages},
                    ensure_ascii=False,
                )
                + "\n"
            )

    return str(out)
# -----------------------------
# Library / RAG / Ontology utilities
# -----------------------------

import re
import json


def get_tutorial(tutorial_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  id::text,
                  title,
                  summary,
                  source_repo_url,
                  index_markdown,
                  mermaid_graph,
                  model_provider,
                  model_name,
                  language,
                  created_at
                from tutorials
                where id = %s
                """,
                (tutorial_id,),
            )
            row = cur.fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "title": row[1],
        "summary": row[2],
        "source_repo_url": row[3],
        "index_markdown": row[4],
        "mermaid_graph": row[5],
        "model_provider": row[6],
        "model_name": row[7],
        "language": row[8],
        "created_at": row[9],
    }


def get_chapters(tutorial_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  id::text,
                  chapter_no,
                  title,
                  filename,
                  markdown,
                  created_at
                from chapters
                where tutorial_id = %s
                order by chapter_no asc
                """,
                (tutorial_id,),
            )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "chapter_no": r[1],
            "title": r[2],
            "filename": r[3],
            "markdown": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]











# ============================================================
# v2: Admin delete / improved RAG / robust ontology utilities
# ============================================================

import re
import json
from pathlib import Path




def _normalize_text_for_search(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"[\[\]\(\)\{\}`'\"“”‘’:_\-/#.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _expand_query_terms(query: str):
    q = query or ""
    q_norm = _normalize_text_for_search(q)

    terms = set()
    for t in q_norm.split():
        if len(t) >= 2:
            terms.add(t)

    alias_map = {
        "의존성": [
            "의존성", "주입", "의존성 주입", "dependency", "injection",
            "depends", "depends(", "Depends", "Dependency Injection",
        ],
        "dependency": [
            "의존성", "주입", "dependency", "injection",
            "depends", "Dependency Injection",
        ],
        "라우터": [
            "라우터", "router", "apirouter", "include_router",
        ],
        "router": [
            "라우터", "router", "apirouter", "include_router",
        ],
        "미들웨어": [
            "미들웨어", "middleware", "request", "response",
        ],
        "middleware": [
            "미들웨어", "middleware", "request", "response",
        ],
        "경로": [
            "경로", "작동", "path", "operation", "path operation",
            "@app.get", "@app.post", "get", "post",
        ],
        "path": [
            "경로", "작동", "path", "operation", "path operation",
            "@app.get", "@app.post",
        ],
        "pydantic": [
            "pydantic", "모델", "model", "validation", "검증",
        ],
        "응답": [
            "응답", "response", "response model", "반환",
        ],
        "streaming": [
            "streaming", "스트리밍", "streamingresponse",
        ],
        "인스턴스": [
            "인스턴스", "application", "app", "FastAPI()", "애플리케이션",
        ],
    }

    for key, aliases in alias_map.items():
        if key.lower() in q_norm or key in q:
            for a in aliases:
                terms.add(_normalize_text_for_search(a))

    # 복합 한글 표현 보정
    if "의존성 주입" in q:
        terms.update(["의존성", "주입", "dependency", "injection", "depends"])

    if "fastapi" in q_norm:
        terms.add("fastapi")

    return [t for t in terms if t]


def keyword_search_chunks(tutorial_id: str, query: str, top_k: int = 5):
    """
    v2 검색:
    - DB에서 해당 tutorial의 모든 chunk를 가져온다.
    - 한국어/영어/코드 토큰/챕터 제목 기반으로 Python scoring한다.
    - Postgres full-text에 의존하지 않으므로 한국어 검색 실패가 줄어든다.
    """
    query = (query or "").strip()
    if not query:
        return []

    terms = _expand_query_terms(query)
    q_norm = _normalize_text_for_search(query)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  c.id::text,
                  c.chapter_id::text,
                  ch.chapter_no,
                  ch.title,
                  c.chunk_index,
                  c.content,
                  c.metadata
                from chunks c
                left join chapters ch on ch.id = c.chapter_id
                where c.tutorial_id = %s
                order by ch.chapter_no asc, c.chunk_index asc
                """,
                (tutorial_id,),
            )
            rows = cur.fetchall()

    results = []

    for r in rows:
        chunk_id, chapter_id, chapter_no, title, chunk_index, content, metadata = r

        title = title or ""
        content = content or ""
        combined = f"{title}\n{content}"
        combined_norm = _normalize_text_for_search(combined)
        title_norm = _normalize_text_for_search(title)

        score = 0

        for term in terms:
            t = _normalize_text_for_search(term)
            if not t:
                continue

            if t in title_norm:
                score += 40

            count = combined_norm.count(t)
            if count:
                score += count * 8

            # 코드 토큰은 normalize 과정에서 일부 훼손될 수 있으므로 원문도 검사
            if term in combined:
                score += 10

        # 질문 전체 일부 매칭
        if q_norm and q_norm in combined_norm:
            score += 80

        # 주제별 강한 보정
        if ("의존성" in query or "Dependency" in query or "dependency" in query.lower()):
            if "의존성 주입" in combined or "Dependency Injection" in combined:
                score += 120
            if "Depends" in combined or "depends" in combined.lower():
                score += 60

        if ("미들웨어" in query or "middleware" in query.lower()):
            if "미들웨어" in combined or "Middleware" in combined:
                score += 120

        if ("라우터" in query or "router" in query.lower()):
            if "라우터" in combined or "Router" in combined or "APIRouter" in combined:
                score += 120

        if ("pydantic" in query.lower() or "모델" in query):
            if "Pydantic" in combined or "pydantic" in combined.lower():
                score += 120

        if score > 0:
            results.append(
                {
                    "chunk_id": chunk_id,
                    "chapter_id": chapter_id,
                    "chapter_no": chapter_no,
                    "chapter_title": title,
                    "chunk_index": chunk_index,
                    "content": content,
                    "metadata": metadata or {},
                    "score": score,
                }
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def extract_ontology_from_tutorial_content(tutorial_id: str):
    """
    Mermaid 파싱이 실패해도 최소 ontology를 만들기 위해 chapter 기반 node/edge를 생성한다.
    """
    tutorial = get_tutorial(tutorial_id)
    chapters = get_chapters(tutorial_id)

    nodes = {}
    edges = []

    if tutorial:
        nodes["tutorial"] = {
            "node_key": "tutorial",
            "node_type": "tutorial",
            "label": tutorial.get("title") or "Tutorial",
            "properties": {"source": "tutorial"},
        }

    for ch in chapters:
        key = f"chapter_{ch['chapter_no']}"
        nodes[key] = {
            "node_key": key,
            "node_type": "chapter",
            "label": ch["title"],
            "properties": {
                "source": "chapter",
                "chapter_no": ch["chapter_no"],
                "filename": ch["filename"],
            },
        }

        edges.append(
            {
                "source_key": "tutorial",
                "target_key": key,
                "edge_type": "has_chapter",
                "label": "has chapter",
                "properties": {"source": "chapter_order"},
            }
        )

    # 챕터 순서 edge
    sorted_ch = sorted(chapters, key=lambda x: x["chapter_no"])
    for prev, curr in zip(sorted_ch, sorted_ch[1:]):
        edges.append(
            {
                "source_key": f"chapter_{prev['chapter_no']}",
                "target_key": f"chapter_{curr['chapter_no']}",
                "edge_type": "next_chapter",
                "label": "next",
                "properties": {"source": "chapter_order"},
            }
        )

    return list(nodes.values()), edges


def extract_ontology_from_mermaid(mermaid: str):
    """
    v2 Mermaid parser.
    flowchart TD
      A["label"] --> B["label"]
      A -- relation --> B
      A -->|relation| B
    """
    if not mermaid:
        return [], []

    nodes = {}
    edges = []

    lines = []
    for raw in mermaid.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("flowchart") or line.lower().startswith("graph"):
            continue
        lines.append(line)

    def add_node(key, label=None):
        key = key.strip()
        if not key:
            return
        label = (label or key).strip()
        label = label.strip('"').strip("'").strip()
        if key not in nodes:
            nodes[key] = {
                "node_key": key,
                "node_type": "abstraction",
                "label": label,
                "properties": {"source": "mermaid"},
            }

    # node labels
    label_patterns = [
        re.compile(r'([A-Za-z0-9_]+)\s*\[\s*"([^"]+)"\s*\]'),
        re.compile(r"([A-Za-z0-9_]+)\s*\[\s*'([^']+)'\s*\]"),
        re.compile(r"([A-Za-z0-9_]+)\s*\[\s*([^\]]+?)\s*\]"),
        re.compile(r'([A-Za-z0-9_]+)\s*\(\s*"([^"]+)"\s*\)'),
        re.compile(r"([A-Za-z0-9_]+)\s*\(\s*([^)]+?)\s*\)"),
    ]

    for line in lines:
        for pat in label_patterns:
            for key, label in pat.findall(line):
                add_node(key, label)

    # edge parse
    for line in lines:
        # A -->|label| B
        m = re.search(r"([A-Za-z0-9_]+)\s*[-.]+>\s*\|([^|]+)\|\s*([A-Za-z0-9_]+)", line)
        if m:
            src, label, dst = m.group(1), m.group(2), m.group(3)
            add_node(src)
            add_node(dst)
            edges.append(
                {
                    "source_key": src,
                    "target_key": dst,
                    "edge_type": "related_to",
                    "label": label.strip(),
                    "properties": {"source": "mermaid"},
                }
            )
            continue

        # A -- label --> B
        m = re.search(r"([A-Za-z0-9_]+)\s*--\s*([^->]+?)\s*[-]+>\s*([A-Za-z0-9_]+)", line)
        if m:
            src, label, dst = m.group(1), m.group(2), m.group(3)
            add_node(src)
            add_node(dst)
            edges.append(
                {
                    "source_key": src,
                    "target_key": dst,
                    "edge_type": "related_to",
                    "label": label.strip().strip('"').strip("'"),
                    "properties": {"source": "mermaid"},
                }
            )
            continue

        # A --> B
        m = re.search(r"([A-Za-z0-9_]+)\s*[-.]+>\s*([A-Za-z0-9_]+)", line)
        if m:
            src, dst = m.group(1), m.group(2)
            add_node(src)
            add_node(dst)
            edges.append(
                {
                    "source_key": src,
                    "target_key": dst,
                    "edge_type": "related_to",
                    "label": "related to",
                    "properties": {"source": "mermaid"},
                }
            )

    return list(nodes.values()), edges


def rebuild_ontology_from_tutorial(tutorial_id: str):
    """
    v2 ontology rebuild.
    Mermaid + chapter fallback을 병합한다.
    """
    tutorial = get_tutorial(tutorial_id)

    if not tutorial:
        raise ValueError(f"tutorial not found: {tutorial_id}")

    mermaid = tutorial.get("mermaid_graph") or ""

    mermaid_nodes, mermaid_edges = extract_ontology_from_mermaid(mermaid)
    fallback_nodes, fallback_edges = extract_ontology_from_tutorial_content(tutorial_id)

    nodes_by_key = {}

    for n in fallback_nodes + mermaid_nodes:
        nodes_by_key[n["node_key"]] = n

    edges = fallback_edges + mermaid_edges

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("delete from ontology_edges where tutorial_id = %s", (tutorial_id,))
            cur.execute("delete from ontology_nodes where tutorial_id = %s", (tutorial_id,))

            node_id_map = {}

            for n in nodes_by_key.values():
                cur.execute(
                    """
                    insert into ontology_nodes (
                      tutorial_id,
                      node_key,
                      node_type,
                      label,
                      properties
                    )
                    values (%s, %s, %s, %s, %s::jsonb)
                    returning id::text
                    """,
                    (
                        tutorial_id,
                        n["node_key"],
                        n["node_type"],
                        n["label"],
                        json.dumps(n.get("properties") or {}, ensure_ascii=False),
                    ),
                )
                node_id_map[n["node_key"]] = cur.fetchone()[0]

            inserted_edges = 0

            for e in edges:
                source_id = node_id_map.get(e["source_key"])
                target_id = node_id_map.get(e["target_key"])

                if not source_id or not target_id:
                    continue

                cur.execute(
                    """
                    insert into ontology_edges (
                      tutorial_id,
                      source_node_id,
                      target_node_id,
                      edge_type,
                      label,
                      properties
                    )
                    values (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        tutorial_id,
                        source_id,
                        target_id,
                        e["edge_type"],
                        e["label"],
                        json.dumps(e.get("properties") or {}, ensure_ascii=False),
                    ),
                )
                inserted_edges += 1

        conn.commit()

    return {
        "nodes": len(nodes_by_key),
        "edges": inserted_edges,
        "mermaid_nodes": len(mermaid_nodes),
        "mermaid_edges": len(mermaid_edges),
        "fallback_nodes": len(fallback_nodes),
        "fallback_edges": len(fallback_edges),
    }


def get_ontology_context(tutorial_id: str, query: str | None = None, limit: int = 50):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  s.label as source_label,
                  e.label as edge_label,
                  t.label as target_label,
                  e.edge_type,
                  s.node_type,
                  t.node_type
                from ontology_edges e
                join ontology_nodes s on s.id = e.source_node_id
                join ontology_nodes t on t.id = e.target_node_id
                where e.tutorial_id = %s
                order by s.label, t.label
                limit %s
                """,
                (tutorial_id, limit),
            )
            rows = cur.fetchall()

    triples = [
        {
            "source": r[0],
            "relation": r[1],
            "target": r[2],
            "edge_type": r[3],
            "source_type": r[4],
            "target_type": r[5],
        }
        for r in rows
    ]

    if query:
        terms = _expand_query_terms(query)
        filtered = []

        for t in triples:
            haystack = _normalize_text_for_search(
                f"{t['source']} {t['relation']} {t['target']} {t['edge_type']}"
            )

            if any(term in haystack for term in terms):
                filtered.append(t)

        return filtered or triples[:limit]

    return triples[:limit]


def build_rag_prompt(question: str, chunks: list[dict], ontology_triples: list[dict] | None = None):
    context_parts = []

    for i, ch in enumerate(chunks, start=1):
        context_parts.append(
            f"[Chunk {i}] Chapter {ch.get('chapter_no')} - {ch.get('chapter_title')}\n"
            f"{ch.get('content')}"
        )

    ontology_text = ""
    if ontology_triples:
        ontology_text = "\n".join(
            f"- {t['source']} -- {t['relation']} --> {t['target']}"
            for t in ontology_triples
        )

    return f"""
너는 오픈소스 코드베이스를 한국어로 설명하는 기술 튜터다.
반드시 아래 검색 context와 ontology context만 근거로 답변하라.
context에 없는 내용은 단정하지 말고, 부족하다고 말하라.

[사용자 질문]
{question}

[검색된 문서 context]
{chr(10).join(context_parts)}

[Ontology 관계 context]
{ontology_text}

[답변 형식]
1. 핵심 요약
2. 코드베이스 구조상 역할
3. 간단한 예시
4. 관련 챕터 근거
""".strip()

# ============================================================
# v3: PocketFlow-style workflow helpers
# - setup-first app support
# - robust tutorial deletion
# - chapter-first RAG search
# - chapter/link/mermaid fallback ontology
# ============================================================

import re
import json
from pathlib import Path


def delete_tutorial_v3(tutorial_id: str, cleanup_orphan_repository: bool = True):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select repository_id::text, output_dir, title
                from tutorials
                where id = %s
                """,
                (tutorial_id,),
            )
            row = cur.fetchone()

            if not row:
                return {"deleted": False, "reason": "tutorial not found"}

            repository_id, output_dir, title = row

            cur.execute("delete from tutorials where id = %s", (tutorial_id,))

            if cleanup_orphan_repository and repository_id:
                cur.execute(
                    """
                    delete from repositories r
                    where r.id = %s
                      and not exists (
                        select 1 from tutorials t
                        where t.repository_id = r.id
                      )
                    """,
                    (repository_id,),
                )

        conn.commit()

    return {
        "deleted": True,
        "title": title,
        "output_dir": output_dir,
    }


_RELATED_COUNT_KEYS = [
    "tutorials", "chapters", "chunks",
    "ontology_nodes", "ontology_edges",
    "fine_tuning_examples", "rag_logs",
]


def count_related_v4(*, tutorial_id: str | None = None, repository_id: str | None = None) -> dict:
    """Count every row a delete would remove, scoped to a single tutorial
    or an entire repository (all of its tutorials). Used for the Admin
    tab's delete preview / post-delete summary."""
    if tutorial_id:
        tut_filter, params = "id = %s", [tutorial_id]
    elif repository_id:
        tut_filter, params = "repository_id = %s", [repository_id]
    else:
        raise ValueError("count_related_v4 requires tutorial_id or repository_id")

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            with tut as (select id from tutorials where {tut_filter})
            select
              (select count(*) from tut),
              (select count(*) from chapters where tutorial_id in (select id from tut)),
              (select count(*) from chunks where tutorial_id in (select id from tut)),
              (select count(*) from ontology_nodes where tutorial_id in (select id from tut)),
              (select count(*) from ontology_edges where tutorial_id in (select id from tut)),
              (select count(*) from fine_tuning_examples where tutorial_id in (select id from tut)),
              (select count(*) from rag_logs where tutorial_id in (select id from tut))
            """,
            params,
        )
        row = cur.fetchone()

    return dict(zip(_RELATED_COUNT_KEYS, row))


def get_tutorial_repository(tutorial_id: str) -> dict | None:
    """Return the repository a tutorial belongs to, plus how many tutorials
    that repository has (so the UI can warn before a repo-wide delete)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select
              r.id::text, r.repo_url, r.repo_name,
              (select count(*) from tutorials t2 where t2.repository_id = r.id)
            from tutorials t
            join repositories r on r.id = t.repository_id
            where t.id = %s
            """,
            (tutorial_id,),
        )
        row = cur.fetchone()

    if not row:
        return None
    return {
        "repository_id": row[0],
        "repo_url": row[1],
        "repo_name": row[2],
        "tutorial_count": row[3],
    }


def delete_repository_v4(repository_id: str) -> dict:
    """Delete a repository and EVERYTHING related to it: all of its tutorials
    and, via ON DELETE CASCADE, their chapters, chunks (incl. embeddings),
    ontology nodes/edges, fine-tuning examples and rag_logs. Returns deleted
    counts and the tutorials' output_dirs for optional local cleanup."""
    counts = count_related_v4(repository_id=repository_id)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select repo_url from repositories where id = %s", (repository_id,))
            row = cur.fetchone()
            if not row:
                return {"deleted": False, "reason": "repository not found", "counts": counts}
            repo_url = row[0]

            cur.execute(
                "select output_dir from tutorials where repository_id = %s and output_dir is not null",
                (repository_id,),
            )
            output_dirs = [r[0] for r in cur.fetchall()]

            # Deleting the repository row cascades to all children.
            cur.execute("delete from repositories where id = %s", (repository_id,))

        conn.commit()

    return {
        "deleted": True,
        "repo_url": repo_url,
        "counts": counts,
        "output_dirs": output_dirs,
    }


def db_counts_v3():
    tables = [
        "repositories",
        "tutorials",
        "chapters",
        "chunks",
        "ontology_nodes",
        "ontology_edges",
        "fine_tuning_examples",
    ]

    result = {}

    with get_conn() as conn:
        with conn.cursor() as cur:
            for table in tables:
                try:
                    cur.execute(f"select count(*) from {table}")
                    result[table] = cur.fetchone()[0]
                except Exception:
                    result[table] = None

    return result


def _norm_v3(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"[\[\]\(\)\{\}`'\"“”‘’:_\-/#.,!?|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _query_terms_v3(query: str):
    q = query or ""
    qn = _norm_v3(q)

    terms = set(t for t in qn.split() if len(t) >= 2)

    alias = {
        "의존성": ["의존성", "주입", "의존성 주입", "dependency", "injection", "depends", "Depends"],
        "dependency": ["의존성", "주입", "dependency", "injection", "depends", "Depends"],
        "라우터": ["라우터", "router", "apirouter", "include_router"],
        "router": ["라우터", "router", "apirouter", "include_router"],
        "미들웨어": ["미들웨어", "middleware", "request", "response"],
        "middleware": ["미들웨어", "middleware", "request", "response"],
        "경로": ["경로", "작동", "path", "operation", "path operation", "@app.get", "@app.post"],
        "path": ["경로", "작동", "path", "operation", "path operation", "@app.get", "@app.post"],
        "pydantic": ["pydantic", "모델", "model", "validation", "검증"],
        "응답": ["응답", "response", "response model", "반환"],
        "스트리밍": ["스트리밍", "streaming", "streamingresponse"],
        "streaming": ["스트리밍", "streaming", "streamingresponse"],
        "인스턴스": ["인스턴스", "application", "app", "fastapi", "FastAPI()", "애플리케이션"],
    }

    for key, values in alias.items():
        if key.lower() in qn or key in q:
            for v in values:
                terms.add(_norm_v3(v))
                terms.add(v)

    if "의존성 주입" in q:
        terms.update(["의존성", "주입", "dependency", "injection", "depends", "Depends"])

    return [t for t in terms if t]











# ============================================================
# v4 quality patch
# - concept-aware RAG scoring
# - preserve Mermaid node labels
# - query-filtered ontology context
# - v3 names aliased to v4 for existing app compatibility
# ============================================================

import re
import json
from pathlib import Path


# Concept aliases / generic terms are externalized to rag_config.py so they
# can be tuned per target repository without editing this module. If the
# config is unavailable, search still works via plain term + vector matching.
try:
    from rag_config import (
        GENERIC_QUERY_TERMS as _GENERIC_QUERY_TERMS_V4,
        CONCEPT_ALIASES as _CONCEPT_ALIASES_V4,
    )
except Exception:  # pragma: no cover - defensive fallback
    _GENERIC_QUERY_TERMS_V4 = set()
    _CONCEPT_ALIASES_V4 = {}


def _norm_v4(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\[\]\(\)\{\}`'\"“”‘’:_\-/#.,!?|;]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_label_v4(label: str) -> str:
    label = label or ""
    label = label.strip().strip('"').strip("'")
    label = re.sub(r"<br\s*/?>", " / ", label, flags=re.I)
    label = re.sub(r"<[^>]+>", "", label)
    label = re.sub(r"\s+", " ", label)
    return label.strip()


def _detect_concepts_v4(query: str) -> dict:
    q_raw = query or ""
    q_norm = _norm_v4(q_raw)

    detected = {}

    for concept, aliases in _CONCEPT_ALIASES_V4.items():
        hits = []
        for alias in aliases:
            alias_norm = _norm_v4(alias)
            if not alias_norm:
                continue

            if alias_norm in q_norm or alias in q_raw:
                hits.append(alias)

        if hits:
            detected[concept] = hits

    # Korean phrase hard rules
    if "의존성 주입" in q_raw or ("의존성" in q_raw and "주입" in q_raw):
        detected.setdefault("dependency_injection", []).extend(
            ["의존성 주입", "Depends", "dependency injection"]
        )

    if "응답 모델" in q_raw:
        detected.setdefault("response_model", []).extend(["응답 모델", "response model"])

    if "경로 작동" in q_raw:
        detected.setdefault("path_operation", []).extend(["경로 작동", "path operation"])

    return detected


def _query_terms_v4(query: str) -> list[str]:
    q = query or ""
    qn = _norm_v4(q)

    terms = set()

    for token in qn.split():
        if len(token) < 2:
            continue
        if token in _GENERIC_QUERY_TERMS_V4:
            continue
        terms.add(token)

    detected = _detect_concepts_v4(query)
    for aliases in detected.values():
        for alias in aliases:
            if alias:
                terms.add(alias)
                terms.add(_norm_v4(alias))

    return [t for t in terms if t]


def _vector_search_v4(tutorial_id: str, query: str, top_k: int = 5):
    """Semantic (pgvector) search over stored chunk embeddings.

    Returns None (not []) when embeddings are unavailable — no OpenAI key,
    no embedded chunks for this tutorial, or any DB/API error — so the
    caller falls back to keyword scoring. This makes vector search a
    transparent, repo-agnostic upgrade that never breaks the RAG tab.
    """
    # Skip the embedding API call entirely if this tutorial has no embedded
    # chunks (the common case when RAG_CREATE_EMBEDDINGS was never enabled).
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select 1 from chunks where tutorial_id = %s and embedding is not null limit 1",
                    (tutorial_id,),
                )
                if cur.fetchone() is None:
                    return None
    except Exception:
        return None

    try:
        emb = openai_embedding(query)
    except Exception:
        return None
    if not emb:
        return None

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select
                      c.id::text, c.chapter_id::text, ch.chapter_no, ch.title,
                      ch.filename, c.chunk_index, c.content, c.metadata,
                      1 - (c.embedding <=> %s::vector) as score
                    from chunks c
                    left join chapters ch on ch.id = c.chapter_id
                    where c.tutorial_id = %s and c.embedding is not null
                    order by c.embedding <=> %s::vector
                    limit %s
                    """,
                    (vector_literal(emb), tutorial_id, vector_literal(emb), top_k),
                )
                rows = cur.fetchall()
    except Exception:
        return None

    if not rows:
        return None

    results = []
    for r in rows:
        chunk_id, chapter_id, chapter_no, title, filename, chunk_index, content, metadata, score = r
        results.append(
            {
                "chunk_id": chunk_id,
                "chapter_id": chapter_id,
                "chapter_no": chapter_no,
                "chapter_title": title or "",
                "filename": filename or "",
                "chunk_index": chunk_index,
                "content": content or "",
                "metadata": metadata or {},
                "score": float(score) if score is not None else 0.0,
                "matched_reasons": ["vector"],
            }
        )
    return results


def search_tutorial_context_v4(tutorial_id: str, query: str, top_k: int = 5):
    """
    Repo-agnostic but concept-aware search.

    Strategy:
    0. If chunk embeddings exist, use semantic (vector) search.
    Otherwise fall back to concept-aware keyword scoring:
    1. chapter title / filename
    2. concept alias exact match
    3. content match
    4. generic keyword match
    """
    query = (query or "").strip()
    if not query:
        return []

    # Semantic search first; returns None when embeddings are unavailable.
    vector_hits = _vector_search_v4(tutorial_id, query, top_k)
    if vector_hits:
        return vector_hits

    terms = _query_terms_v4(query)
    concepts = _detect_concepts_v4(query)
    qn = _norm_v4(query)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  c.id::text,
                  c.chapter_id::text,
                  ch.chapter_no,
                  ch.title,
                  ch.filename,
                  c.chunk_index,
                  c.content,
                  c.metadata
                from chunks c
                left join chapters ch on ch.id = c.chapter_id
                where c.tutorial_id = %s
                order by ch.chapter_no asc, c.chunk_index asc
                """,
                (tutorial_id,),
            )
            rows = cur.fetchall()

    results = []

    for r in rows:
        chunk_id, chapter_id, chapter_no, title, filename, chunk_index, content, metadata = r

        title = title or ""
        filename = filename or ""
        content = content or ""

        title_norm = _norm_v4(title)
        filename_norm = _norm_v4(filename)
        content_norm = _norm_v4(content)
        combined = f"{title}\n{filename}\n{content}"
        combined_norm = _norm_v4(combined)

        score = 0
        matched_reasons = []

        # 1. Concept-aware scoring
        for concept, aliases in concepts.items():
            for alias in aliases:
                alias_norm = _norm_v4(alias)
                if not alias_norm:
                    continue

                if alias_norm in title_norm:
                    score += 600
                    matched_reasons.append(f"title:{alias}")

                if alias_norm in filename_norm:
                    score += 250
                    matched_reasons.append(f"filename:{alias}")

                if alias_norm in content_norm:
                    score += min(content_norm.count(alias_norm), 6) * 70
                    matched_reasons.append(f"content:{alias}")

                # exact code/token match
                if alias in content:
                    score += min(content.count(alias), 5) * 80
                    matched_reasons.append(f"code:{alias}")

        # 2. Important query terms
        for term in terms:
            term_norm = _norm_v4(term)
            if not term_norm:
                continue

            if term_norm in title_norm:
                score += 180
                matched_reasons.append(f"title-term:{term}")

            if term_norm in filename_norm:
                score += 80

            if term_norm in content_norm:
                score += min(content_norm.count(term_norm), 8) * 20
                matched_reasons.append(f"content-term:{term}")

            if term in content:
                score += min(content.count(term), 5) * 25

        # 3. Exact normalized question phrase
        if qn and qn in combined_norm:
            score += 200
            matched_reasons.append("exact-question")

        # 4. Chapter-level special boosts
        if "dependency_injection" in concepts and (
            "의존성" in title or "Dependency Injection" in title or "dependency" in title_norm
        ):
            score += 900
            matched_reasons.append("chapter-boost:dependency")

        if "router" in concepts and ("라우터" in title or "router" in title_norm):
            score += 900
            matched_reasons.append("chapter-boost:router")

        if "middleware" in concepts and ("미들웨어" in title or "middleware" in title_norm):
            score += 900
            matched_reasons.append("chapter-boost:middleware")

        if "pydantic_model" in concepts and ("pydantic" in title_norm or "모델" in title):
            score += 700
            matched_reasons.append("chapter-boost:pydantic")

        if "response_model" in concepts and ("응답" in title or "response" in title_norm):
            score += 700
            matched_reasons.append("chapter-boost:response")

        if score > 0:
            results.append(
                {
                    "chunk_id": chunk_id,
                    "chapter_id": chapter_id,
                    "chapter_no": chapter_no,
                    "chapter_title": title,
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "content": content,
                    "metadata": metadata or {},
                    "score": score,
                    "matched_reasons": matched_reasons[:12],
                }
            )

    results.sort(
        key=lambda x: (
            x["score"],
            -int(x["chapter_no"] or 999),
            -int(x["chunk_index"] or 999),
        ),
        reverse=True,
    )

    return results[:top_k]


def build_rag_prompt_v4(question: str, contexts: list[dict], ontology_triples: list[dict] | None = None, language: str = "Korean"):
    doc_context = []

    for i, ctx in enumerate(contexts, start=1):
        doc_context.append(
            f"[Context {i}]\n"
            f"Source: Chapter {ctx.get('chapter_no')} - {ctx.get('chapter_title')}\n"
            f"File: {ctx.get('filename')}\n"
            f"Score: {ctx.get('score')}\n"
            f"Matched: {', '.join(ctx.get('matched_reasons') or [])}\n"
            f"Content:\n{ctx.get('content')}"
        )

    ontology_text = ""
    if ontology_triples:
        ontology_text = "\n".join(
            f"- {t['source']} -- {t['relation']} / {t['edge_type']} --> {t['target']}"
            for t in ontology_triples
        )

    language = (language or "Korean").strip() or "Korean"

    return f"""
너는 오픈소스 코드베이스를 {language} 언어로 설명하는 기술 튜터다.

규칙:
- 반드시 [문서 context]와 [Ontology context]에 근거해서만 답변하라.
- context에 없는 사실은 추측하지 말고, 근거가 부족하다고 명시하라.
- 초보자에게 설명하되, 실제 코드베이스의 구조와 연결해서 설명하라.
- 가능하면 어떤 챕터를 근거로 삼았는지 함께 적어라.
- 답변은 반드시 {language} 언어로 작성하라.

[사용자 질문]
{question}

[문서 context]
{chr(10).join(doc_context)}

[Ontology context]
{ontology_text}

[답변 형식]
1. 핵심 개념
2. 코드베이스 안에서의 역할
3. 동작 흐름
4. 간단한 코드 예시
5. 관련 챕터 근거
""".strip()


def build_chat_prompt_v4(
    question: str,
    contexts: list[dict],
    ontology_triples: list[dict] | None = None,
    history: list[dict] | None = None,
    language: str = "Korean",
):
    """Conversational RAG prompt: like build_rag_prompt_v4 but includes recent
    dialogue turns so follow-up questions ("그럼 그건?") keep context. `history`
    is a list of {"role": "user"|"assistant", "content": str} excluding the
    current question."""
    doc_context = []
    for i, ctx in enumerate(contexts, start=1):
        doc_context.append(
            f"[Context {i}]\n"
            f"Source: Chapter {ctx.get('chapter_no')} - {ctx.get('chapter_title')}\n"
            f"File: {ctx.get('filename')}\n"
            f"Content:\n{ctx.get('content')}"
        )

    ontology_text = ""
    if ontology_triples:
        ontology_text = "\n".join(
            f"- {t['source']} -- {t['relation']} / {t['edge_type']} --> {t['target']}"
            for t in ontology_triples
        )

    history_text = ""
    if history:
        turns = []
        for m in history[-6:]:  # keep the last ~3 exchanges
            who = "사용자" if m.get("role") == "user" else "튜터"
            turns.append(f"{who}: {m.get('content', '')}")
        history_text = "\n".join(turns)

    language = (language or "Korean").strip() or "Korean"

    return f"""
너는 오픈소스 코드베이스를 {language} 언어로 설명하는 대화형 기술 튜터다.

규칙:
- [문서 context]와 [Ontology context]에 근거해 답하라. 근거가 부족하면 추측하지 말고 부족하다고 말하라.
- [이전 대화]를 고려해 후속 질문의 맥락(대명사, 생략된 주제 등)을 이어가라.
- 초보자 눈높이로, 실제 코드베이스 구조와 연결해 간결하게 설명하라.
- 답변은 반드시 {language} 언어로 작성하라.

[이전 대화]
{history_text if history_text else "(없음)"}

[문서 context]
{chr(10).join(doc_context) if doc_context else "(관련 문서를 찾지 못함)"}

[Ontology context]
{ontology_text if ontology_text else "(없음)"}

[현재 질문]
{question}
""".strip()


def _parse_mermaid_v4(mermaid: str):
    nodes = {}
    edges = []

    if not mermaid:
        return nodes, edges

    def add_node(key, label=None):
        key = (key or "").strip()
        if not key:
            return

        cleaned_label = _clean_label_v4(label or key)

        if key in nodes:
            # Do not overwrite a meaningful label with raw ID like A0/A1.
            if label and nodes[key]["label"] == key:
                nodes[key]["label"] = cleaned_label
            return

        nodes[key] = {
            "node_key": key,
            "node_type": "abstraction",
            "label": cleaned_label,
            "properties": {"source": "mermaid"},
        }

    for raw in mermaid.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith(("flowchart", "graph")):
            continue

        # Node definitions: A0["Application Instance"] or A0[Application Instance]
        for m in re.finditer(r'([A-Za-z0-9_]+)\s*\[\s*"([^"]+)"\s*\]', line):
            add_node(m.group(1), m.group(2))

        for m in re.finditer(r"([A-Za-z0-9_]+)\s*\[\s*([^\]]+)\s*\]", line):
            add_node(m.group(1), m.group(2))

        for m in re.finditer(r'([A-Za-z0-9_]+)\s*\(\s*"([^"]+)"\s*\)', line):
            add_node(m.group(1), m.group(2))

        # Edges with label: A0 -->|label| A1
        edge = re.search(r"([A-Za-z0-9_]+)\s*[-.]*>\|([^|]+)\|\s*([A-Za-z0-9_]+)", line)
        if edge:
            src, label, dst = edge.group(1), edge.group(2), edge.group(3)
            add_node(src)
            add_node(dst)
            edges.append((src, dst, _clean_label_v4(label), "related_to"))
            continue

        # Edges with middle label: A0 -- label --> A1
        edge = re.search(r"([A-Za-z0-9_]+)\s*--\s*([^->]+?)\s*[-.]*>\s*([A-Za-z0-9_]+)", line)
        if edge:
            src, label, dst = edge.group(1), edge.group(2), edge.group(3)
            add_node(src)
            add_node(dst)
            edges.append((src, dst, _clean_label_v4(label), "related_to"))
            continue

        # Plain edge: A0 --> A1
        edge = re.search(r"([A-Za-z0-9_]+)\s*[-.]*>\s*([A-Za-z0-9_]+)", line)
        if edge:
            src, dst = edge.group(1), edge.group(2)
            add_node(src)
            add_node(dst)
            edges.append((src, dst, "related to", "related_to"))

    return nodes, edges


def rebuild_ontology_v4(tutorial_id: str):
    tutorial = get_tutorial(tutorial_id)
    chapters = get_chapters(tutorial_id)

    if not tutorial:
        raise ValueError(f"tutorial not found: {tutorial_id}")

    nodes = {}
    edges = []

    nodes["tutorial"] = {
        "node_key": "tutorial",
        "node_type": "tutorial",
        "label": tutorial.get("title") or "Tutorial",
        "properties": {"source": "tutorial"},
    }

    filename_to_key = {}

    for ch in chapters:
        key = f"chapter_{ch['chapter_no']}"
        filename_to_key[ch["filename"]] = key

        nodes[key] = {
            "node_key": key,
            "node_type": "chapter",
            "label": ch["title"],
            "properties": {
                "source": "chapter",
                "chapter_no": ch["chapter_no"],
                "filename": ch["filename"],
            },
        }

        edges.append(("tutorial", key, "has chapter", "has_chapter"))

    sorted_chapters = sorted(chapters, key=lambda x: x["chapter_no"])

    for prev, curr in zip(sorted_chapters, sorted_chapters[1:]):
        edges.append(
            (
                f"chapter_{prev['chapter_no']}",
                f"chapter_{curr['chapter_no']}",
                "next",
                "next_chapter",
            )
        )

    # Markdown chapter links
    for ch in chapters:
        source_key = f"chapter_{ch['chapter_no']}"
        markdown = ch.get("markdown") or ""

        linked_files = re.findall(r"\]\(([^)]+\.md)\)", markdown)

        for linked in linked_files:
            linked_name = Path(linked).name
            target_key = filename_to_key.get(linked_name)

            if target_key and target_key != source_key:
                edges.append((source_key, target_key, "mentions", "mentions"))

    # Mermaid graph
    mermaid_nodes, mermaid_edges = _parse_mermaid_v4(tutorial.get("mermaid_graph") or "")

    for key, val in mermaid_nodes.items():
        nodes[key] = val

    for src, dst, label, edge_type in mermaid_edges:
        edges.append((src, dst, label, edge_type))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("delete from ontology_edges where tutorial_id = %s", (tutorial_id,))
            cur.execute("delete from ontology_nodes where tutorial_id = %s", (tutorial_id,))

            node_id_map = {}

            for n in nodes.values():
                cur.execute(
                    """
                    insert into ontology_nodes (
                      tutorial_id, node_key, node_type, label, properties
                    )
                    values (%s, %s, %s, %s, %s::jsonb)
                    returning id::text
                    """,
                    (
                        tutorial_id,
                        n["node_key"],
                        n["node_type"],
                        n["label"],
                        json.dumps(n.get("properties") or {}, ensure_ascii=False),
                    ),
                )
                node_id_map[n["node_key"]] = cur.fetchone()[0]

            inserted_edges = 0
            seen = set()

            for src, dst, label, edge_type in edges:
                if src not in node_id_map or dst not in node_id_map:
                    continue

                sig = (src, dst, label, edge_type)
                if sig in seen:
                    continue
                seen.add(sig)

                cur.execute(
                    """
                    insert into ontology_edges (
                      tutorial_id,
                      source_node_id,
                      target_node_id,
                      edge_type,
                      label,
                      properties
                    )
                    values (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        tutorial_id,
                        node_id_map[src],
                        node_id_map[dst],
                        edge_type,
                        label,
                        json.dumps({"source": "v4"}, ensure_ascii=False),
                    ),
                )
                inserted_edges += 1

        conn.commit()

    return {
        "nodes": len(nodes),
        "edges": inserted_edges,
        "chapters": len(chapters),
        "mermaid_nodes": len(mermaid_nodes),
        "mermaid_edges": len(mermaid_edges),
    }


def get_ontology_context_v4(tutorial_id: str, query: str | None = None, limit: int = 100):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  s.label,
                  e.label,
                  t.label,
                  e.edge_type,
                  s.node_type,
                  t.node_type
                from ontology_edges e
                join ontology_nodes s on s.id = e.source_node_id
                join ontology_nodes t on t.id = e.target_node_id
                where e.tutorial_id = %s
                order by
                  case e.edge_type
                    when 'has_chapter' then 1
                    when 'next_chapter' then 2
                    when 'mentions' then 3
                    else 4
                  end,
                  s.label,
                  e.label,
                  t.label
                limit %s
                """,
                (tutorial_id, limit),
            )
            rows = cur.fetchall()

    triples = [
        {
            "source": r[0],
            "relation": r[1],
            "target": r[2],
            "edge_type": r[3],
            "source_type": r[4],
            "target_type": r[5],
        }
        for r in rows
    ]

    if not query:
        return triples

    terms = _query_terms_v4(query)
    concepts = _detect_concepts_v4(query)

    filtered = []

    for t in triples:
        haystack = _norm_v4(
            f"{t['source']} {t['relation']} {t['target']} {t['edge_type']}"
        )

        score = 0

        for term in terms:
            tn = _norm_v4(term)
            if tn and tn in haystack:
                score += 10

        for aliases in concepts.values():
            for alias in aliases:
                an = _norm_v4(alias)
                if an and an in haystack:
                    score += 50

        if score > 0:
            item = dict(t)
            item["score"] = score
            filtered.append(item)

    filtered.sort(key=lambda x: x["score"], reverse=True)

    return filtered[:limit]


# Keep existing app_full_workflow.py compatible.
search_tutorial_context_v3 = search_tutorial_context_v4
build_rag_prompt_v3 = build_rag_prompt_v4
_parse_mermaid_v3 = _parse_mermaid_v4
rebuild_ontology_v3 = rebuild_ontology_v4
get_ontology_context_v3 = get_ontology_context_v4


# ============================================================
# v5: Multi-repository (cross-repo) RAG
# Search several saved tutorials at once, fuse results fairly, and let the
# LLM synthesize/connect knowledge across repositories.
# ============================================================


def search_across_tutorials(
    tutorial_ids: list[str],
    query: str,
    top_k: int = 8,
    per_repo_top: int = 6,
) -> list[dict]:
    """Cross-repo hybrid search.

    Runs the same per-tutorial hybrid search (vector -> keyword) for each
    tutorial, tags every hit with its tutorial/repo, then fuses the lists.
    Scores from different repos are not directly comparable (cosine vs.
    keyword scale, or different embedding coverage), so each repo's hits are
    min-max normalized to [0, 1] before merging — a fair, scale-agnostic
    fusion that still preserves within-repo ranking.
    """
    query = (query or "").strip()
    if not query or not tutorial_ids:
        return []

    merged: list[dict] = []
    for tid in tutorial_ids:
        hits = search_tutorial_context_v4(tid, query, top_k=per_repo_top)
        if not hits:
            continue
        t = get_tutorial(tid) or {}
        smax = max((h.get("score") or 0) for h in hits) or 1
        for h in hits:
            h["tutorial_id"] = tid
            h["tutorial_title"] = t.get("title") or tid
            h["source_repo_url"] = t.get("source_repo_url") or ""
            h["norm_score"] = round((h.get("score") or 0) / smax, 4)
            merged.append(h)

    merged.sort(key=lambda x: x["norm_score"], reverse=True)
    return merged[:top_k]


def build_multi_repo_rag_prompt(question: str, contexts: list[dict], language: str = "Korean") -> str:
    """RAG prompt that groups retrieved context by source repository so the
    model can compare/connect concepts across repos and cite which repo each
    fact came from."""
    language = (language or "Korean").strip() or "Korean"

    by_repo: dict[str, list[dict]] = {}
    for c in contexts:
        label = c.get("tutorial_title") or c.get("source_repo_url") or "unknown"
        by_repo.setdefault(label, []).append(c)

    blocks = []
    for repo_label, items in by_repo.items():
        lines = [f"### 저장소: {repo_label}"]
        for i, c in enumerate(items, start=1):
            lines.append(
                f"[{i}] Chapter {c.get('chapter_no')} - {c.get('chapter_title')} "
                f"(file: {c.get('filename')})\n{c.get('content')}"
            )
        blocks.append("\n".join(lines))

    repo_names = ", ".join(by_repo.keys()) or "(없음)"

    return f"""
너는 여러 오픈소스 코드베이스를 함께 이해하고 {language} 언어로 설명하는 시니어 엔지니어다.
아래에는 서로 다른 저장소에서 검색된 근거가 저장소별로 묶여 있다: {repo_names}

규칙:
- 각 저장소의 근거만 사용하고, 사실에는 어느 저장소(그리고 챕터)에서 나왔는지 밝혀라.
- 저장소 간 공통점·차이점·연결 가능성(예: A의 개념을 B에 어떻게 접목할지)을 적극적으로 짚어라.
- 근거가 부족하면 추측하지 말고 부족하다고 말하라.
- 반드시 {language} 언어로 답하라.

[사용자 질문]
{question}

[저장소별 검색 근거]
{chr(10).join(blocks) if blocks else "(관련 근거를 찾지 못함)"}

[답변 형식]
1. 핵심 답변
2. 저장소별 근거 요약 (저장소명 · 챕터)
3. 저장소 간 연결/비교 인사이트
""".strip()


# ============================================================
# v5: Code-generation fine-tuning dataset (multi-repo)
# Extract real code blocks from stored tutorials and turn them into
# (instruction -> code) chat examples for small local LoRA fine-tuning.
# ============================================================

CODEGEN_SYSTEM_PROMPT = (
    "You are a coding assistant that writes small, correct code units "
    "(functions, classes, config, usage snippets) in the style of the "
    "referenced open-source codebases. Output only the code in a fenced block."
)

_CODE_LANGS = {
    "python", "py", "javascript", "js", "jsx", "typescript", "ts", "tsx",
    "go", "java", "c", "cpp", "c++", "cc", "h", "rust", "rs", "ruby", "rb",
    "php", "bash", "sh", "shell", "yaml", "yml", "toml", "json", "sql",
    "kotlin", "swift", "scala",
}


def _extract_code_blocks(markdown: str):
    """Yield (lang, code) fenced blocks, skipping prose/mermaid/output blocks."""
    for lang, code in re.findall(r"```([\w+#-]*)\s*\n(.*?)```", markdown or "", flags=re.S):
        lang_norm = (lang or "").strip().lower()
        if lang_norm in {"mermaid", "text", "markdown", "md", "console", "output", "diff"}:
            continue
        yield lang_norm, code.strip()


def build_codegen_examples(
    tutorial_ids: list[str],
    min_code_lines: int = 3,
    max_per_tutorial: int | None = None,
) -> list[dict]:
    """Build (instruction -> code) chat examples from stored chapters across
    one or more tutorials. Focuses on small units by using each fenced code
    block as a target and synthesizing an instruction from its chapter."""
    examples: list[dict] = []

    for tid in tutorial_ids:
        tutorial = get_tutorial(tid) or {}
        chapters = get_chapters(tid)
        project = (tutorial.get("title") or "").replace("Tutorial: ", "").strip()
        repo = tutorial.get("source_repo_url") or ""
        count = 0

        for ch in chapters:
            for lang, code in _extract_code_blocks(ch.get("markdown")):
                if code.count("\n") + 1 < min_code_lines:
                    continue
                if max_per_tutorial and count >= max_per_tutorial:
                    break

                lang_hint = lang or "code"
                instruction = (
                    f"`{project}` 코드베이스의 '{ch['title']}' 개념을 보여주는 "
                    f"{lang_hint} 코드를 작성해줘."
                )
                answer = f"```{lang}\n{code}\n```"

                examples.append({
                    "tutorial_id": tid,
                    "task_type": "codegen",
                    "language": lang_hint,
                    "chapter_no": ch["chapter_no"],
                    "chapter_title": ch["title"],
                    "repo": repo,
                    "messages": [
                        {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": answer},
                    ],
                })
                count += 1

    return examples


def export_codegen_jsonl(
    tutorial_ids: list[str],
    out_path: str = "exports/codegen_dataset.jsonl",
    min_code_lines: int = 3,
    max_per_tutorial: int | None = None,
) -> dict:
    """Write a messages JSONL for local code-gen fine-tuning. Returns stats."""
    examples = build_codegen_examples(
        tutorial_ids, min_code_lines=min_code_lines, max_per_tutorial=max_per_tutorial
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lang_counts: dict[str, int] = {}
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            lang_counts[ex["language"]] = lang_counts.get(ex["language"], 0) + 1
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    return {
        "path": str(out),
        "examples": len(examples),
        "tutorials": len(tutorial_ids),
        "by_language": dict(sorted(lang_counts.items(), key=lambda kv: kv[1], reverse=True)),
    }


def update_tutorial_mermaid(tutorial_id: str, mermaid: str) -> bool:
    """Persist a (e.g. self-healed) Mermaid graph back onto a tutorial."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "update tutorials set mermaid_graph = %s where id = %s",
                (mermaid, tutorial_id),
            )
            updated = cur.rowcount
        conn.commit()
    return updated > 0


# ============================================================
# Tracing persistence (agent_traces table)
# ============================================================

def save_trace_db(kind: str, question: str, trace: list, meta: dict | None = None) -> str:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into agent_traces (kind, question, meta, trace)
                values (%s, %s, %s::jsonb, %s::jsonb)
                returning id::text
                """,
                (
                    kind,
                    question,
                    json.dumps(meta or {}, ensure_ascii=False),
                    json.dumps(trace or [], ensure_ascii=False),
                ),
            )
            trace_id = cur.fetchone()[0]
        conn.commit()
    return trace_id


def list_traces_db(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id::text, kind, question, created_at
                from agent_traces
                order by created_at desc
                limit %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [
        {"id": r[0], "kind": r[1], "question": r[2], "created_at": str(r[3])[:19]}
        for r in rows
    ]


def get_trace_db(trace_id: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select id::text, kind, question, meta, trace, created_at from agent_traces where id = %s",
                (trace_id,),
            )
            r = cur.fetchone()
    if not r:
        return None
    return {"id": r[0], "kind": r[1], "question": r[2], "meta": r[3], "trace": r[4], "created_at": str(r[5])[:19]}

