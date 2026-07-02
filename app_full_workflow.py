import os
import sys
import time
import shutil
import shlex
import subprocess
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from db_store import (
    save_tutorial_result_to_db,
    list_tutorials,
    get_tutorial,
    get_chapters,
    search_tutorial_context_v3,
    rebuild_ontology_v3,
    get_ontology_context_v3,
    build_rag_prompt_v3,
    build_chat_prompt_v4,
    delete_tutorial_v3,
    delete_repository_v4,
    count_related_v4,
    get_tutorial_repository,
    db_counts_v3,
)

load_dotenv(".env", override=True)

st.set_page_config(
    page_title="Codebase Tutorial Full Workflow",
    layout="wide",
)

st.title("Codebase Tutorial Full Workflow")



def split_patterns(value: str) -> list[str]:
    """Streamlit text input? CLI ?? ???? ???? ????."""
    value = (value or "").strip()
    if not value:
        return []
    return shlex.split(value)


# -----------------------------
# helpers
# -----------------------------

def set_env_from_ui(provider, api_key, model_name, github_token, database_url):
    provider = provider.upper().strip()
    os.environ["LLM_PROVIDER"] = provider

    if provider == "GEMINI":
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        if model_name:
            os.environ["GEMINI_MODEL"] = model_name

    if provider == "OPENAI":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if model_name:
            os.environ["OPENAI_MODEL"] = model_name
        os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com")

    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token

    if database_url:
        os.environ["DATABASE_URL"] = database_url


def update_env_file(path, values):
    p = Path(path)
    existing = {}

    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()

    for k, v in values.items():
        if v is not None and str(v).strip() != "":
            existing[k] = str(v).strip()

    lines = [f"{k}={v}" for k, v in existing.items()]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_latest_result_dir(output_base: str):
    base = Path(output_base)
    if not base.exists():
        return None

    index_files = list(base.rglob("index.md"))
    if not index_files:
        return None

    latest = max(index_files, key=lambda p: p.stat().st_mtime)
    return latest.parent


def render_mermaid(mermaid_code: str, height: int = 520):
    if not mermaid_code:
        st.info("Mermaid graph가 없습니다.")
        return

    html = f"""
    <div class="mermaid">
    {mermaid_code}
    </div>

    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{
        startOnLoad: true,
        theme: 'dark',
        securityLevel: 'loose',
        flowchart: {{
          useMaxWidth: true,
          htmlLabels: true,
          curve: 'basis'
        }}
      }});
    </script>
    """

    components.html(html, height=height, scrolling=True)


def get_tutorials_safe():
    try:
        if not os.getenv("DATABASE_URL"):
            return []
        return list_tutorials()
    except Exception:
        return []


# -----------------------------
# sidebar setup
# -----------------------------

st.sidebar.header("1. 초기 세팅")

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["GEMINI", "OPENAI"],
    index=0 if os.getenv("LLM_PROVIDER", "GEMINI").upper() == "GEMINI" else 1,
)

default_model = (
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if provider == "GEMINI"
    else os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
)

model_name = st.sidebar.text_input("Model", value=default_model)

api_key_label = "GEMINI_API_KEY" if provider == "GEMINI" else "OPENAI_API_KEY"
api_key_default = os.getenv(api_key_label, "")

api_key = st.sidebar.text_input(
    api_key_label,
    value=api_key_default,
    type="password",
)

github_token = st.sidebar.text_input(
    "GITHUB_TOKEN",
    value=os.getenv("GITHUB_TOKEN", ""),
    type="password",
    help="공개 repo만 테스트하면 비워도 되지만, rate limit 때문에 넣는 것이 좋습니다.",
)

database_url = st.sidebar.text_input(
    "DATABASE_URL",
    value=os.getenv("DATABASE_URL", ""),
    type="password",
)

save_env = st.sidebar.checkbox("설정을 .env에 저장", value=False)

if st.sidebar.button("현재 세션에 설정 적용", type="primary"):
    set_env_from_ui(provider, api_key, model_name, github_token, database_url)

    if save_env:
        env_values = {
            "LLM_PROVIDER": provider,
            "GITHUB_TOKEN": github_token,
            "DATABASE_URL": database_url,
        }

        if provider == "GEMINI":
            env_values["GEMINI_API_KEY"] = api_key
            env_values["GEMINI_MODEL"] = model_name

        if provider == "OPENAI":
            env_values["OPENAI_API_KEY"] = api_key
            env_values["OPENAI_MODEL"] = model_name
            env_values["OPENAI_BASE_URL"] = "https://api.openai.com"

        update_env_file(".env", env_values)

    st.sidebar.success("설정 적용 완료")

set_env_from_ui(provider, api_key, model_name, github_token, database_url)

st.sidebar.divider()
st.sidebar.header("2. 저장된 튜토리얼")

tutorials = get_tutorials_safe()

selected_tutorial = None
if tutorials:
    selected_tutorial = st.sidebar.selectbox(
        "튜토리얼 선택",
        tutorials,
        format_func=lambda x: f"{x['title']} | {x['source_repo_url']}",
    )
else:
    st.sidebar.info("아직 DB에 저장된 튜토리얼이 없습니다.")


tab_setup, tab_generate, tab_library, tab_rag, tab_chat, tab_ontology, tab_admin = st.tabs([
    "Setup",
    "Generate & Save",
    "Library",
    "RAG Search",
    "Chat",
    "Ontology RAG",
    "Admin",
])


# -----------------------------
# Setup tab
# -----------------------------

with tab_setup:
    st.header("Setup 상태 확인")

    col1, col2, col3 = st.columns(3)
    col1.metric("Provider", os.getenv("LLM_PROVIDER", ""))
    col2.metric("Model", model_name)
    col3.metric("DB URL", "SET" if os.getenv("DATABASE_URL") else "MISSING")

    st.write("GitHub Token:", "SET" if os.getenv("GITHUB_TOKEN") else "EMPTY")
    st.write(api_key_label + ":", "SET" if api_key else "MISSING")

    st.info(
        "이 탭에서 API Key, GitHub Token, Database URL 상태를 확인합니다. "
        "실제 repo 분석은 Generate & Save 탭에서 진행합니다."
    )


# -----------------------------
# Generate & Save tab
# -----------------------------

with tab_generate:
    st.header("Generate & Save")

    repo_url = st.text_input(
        "GitHub Repository URL",
        value="https://github.com/fastapi/fastapi/tree/master/fastapi",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        include_patterns = st.text_input(
            "include patterns",
            value="*.py *.md",
        )

        exclude_patterns = st.text_input(
            "exclude patterns",
            value="tests/* docs/* .github/* docs_src/* scripts/* .venv/*",
        )

        output_base = st.text_input(
            "output base directory",
            value="streamlit_output",
        )

    with col_b:
        language = st.text_input("language", value="Korean")
        max_size = st.number_input("max-size", min_value=1000, max_value=200000, value=60000, step=1000)
        max_abstractions = st.number_input("max-abstractions", min_value=1, max_value=20, value=8)

    no_cache = st.checkbox("no-cache", value=True)

    run_generate = st.button("repo 분석 실행 후 DB 저장", type="primary")

    if run_generate:
        if not repo_url.strip():
            st.error("repo_url이 필요합니다.")
            st.stop()

        if not os.getenv("DATABASE_URL"):
            st.error("DATABASE_URL이 필요합니다.")
            st.stop()

        set_env_from_ui(provider, api_key, model_name, github_token, database_url)

        cmd = [
            sys.executable,
            "-u",
            "main.py",
            "--repo",
            repo_url.strip(),
            "--include",
            *include_patterns.split(),
            "--exclude",
            *exclude_patterns.split(),
            "--max-size",
            str(int(max_size)),
            "--language",
            language.strip(),
            "--max-abstractions",
            str(int(max_abstractions)),
            "--output",
            output_base,
        ]

        if no_cache:
            cmd.append("--no-cache")

        st.subheader("실행 명령")
        st.code(" ".join(cmd), language="powershell")

        log_box = st.empty()
        logs = []
        marker_dir = None

        before = time.time()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        for line in process.stdout:
            line = line.rstrip()
            # main.py emits the exact output path as "RESULT_DIR::<path>".
            if line.startswith("RESULT_DIR::"):
                marker_dir = line.split("RESULT_DIR::", 1)[1].strip()
            logs.append(line)
            log_box.code("\n".join(logs[-400:]), language="text")

        process.wait()

        if process.returncode != 0:
            st.error(f"분석 실패: return code {process.returncode}")
            st.stop()

        # Prefer the exact directory reported by main.py; fall back to the
        # mtime heuristic only if the marker is missing (older main.py).
        if marker_dir and Path(marker_dir).exists():
            result_dir = Path(marker_dir)
        else:
            result_dir = find_latest_result_dir(output_base)

        if not result_dir:
            st.error("생성 결과 index.md를 찾지 못했습니다.")
            st.stop()

        if result_dir.stat().st_mtime < before - 5:
            st.warning(f"가장 최근 결과 폴더를 찾았지만 시간 확인이 필요합니다: {result_dir}")

        st.success(f"생성 결과 폴더: {result_dir}")

        try:
            tutorial_id = save_tutorial_result_to_db(
                result_dir=str(result_dir),
                repo_url=repo_url.strip(),
                provider=provider,
                model_name=model_name,
                language=language.strip(),
                max_abstractions=int(max_abstractions),
            )

            st.success(f"DB 저장 완료: {tutorial_id}")

            try:
                onto = rebuild_ontology_v3(tutorial_id)
                st.info(f"Ontology 자동 구성: {onto}")
            except Exception as e:
                st.warning("DB 저장은 완료됐지만 Ontology 자동 구성은 실패했습니다.")
                st.exception(e)

        except Exception as e:
            st.error("DB 저장 실패")
            st.exception(e)


# -----------------------------
# Library tab
# -----------------------------

with tab_library:
    st.header("Library")

    if not selected_tutorial:
        st.info("저장된 튜토리얼이 없습니다. Generate & Save 탭에서 먼저 생성하세요.")
    else:
        tutorial_id = selected_tutorial["id"]
        tutorial = get_tutorial(tutorial_id)
        chapters = get_chapters(tutorial_id)

        st.subheader(tutorial["title"])
        st.caption(tutorial["source_repo_url"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Provider", tutorial["model_provider"])
        col2.metric("Model", tutorial["model_name"])
        col3.metric("Language", tutorial["language"])

        st.subheader("Summary")
        st.write(tutorial["summary"] or "요약 정보가 없습니다.")

        st.subheader("Flowchart")
        render_mermaid(tutorial["mermaid_graph"])

        st.subheader("Chapters")

        if chapters:
            chapter_idx = st.selectbox(
                "챕터 선택",
                range(len(chapters)),
                format_func=lambda i: f"{chapters[i]['chapter_no']}. {chapters[i]['title']}",
                key="library_chapter_select",
            )

            st.markdown(chapters[chapter_idx]["markdown"])
        else:
            st.info("저장된 챕터가 없습니다.")


# -----------------------------
# RAG Search tab
# -----------------------------

with tab_rag:
    st.header("RAG Search")

    if not selected_tutorial:
        st.info("저장된 튜토리얼이 없습니다.")
    else:
        tutorial_id = selected_tutorial["id"]

        question = st.text_area(
            "질문",
            value="FastAPI의 의존성 주입을 초보자에게 설명해줘.",
            height=100,
        )

        top_k = st.slider("검색 chunk 수", min_value=1, max_value=10, value=5)

        col1, col2 = st.columns(2)

        with col1:
            do_search = st.button("검색 실행", type="primary")

        with col2:
            do_answer = st.button("검색 + LLM 답변 생성")

        if do_search or do_answer:
            results = search_tutorial_context_v3(
                tutorial_id=tutorial_id,
                query=question,
                top_k=top_k,
            )

            st.session_state["rag_results"] = results
            st.session_state["rag_question"] = question

            if not results:
                st.warning("검색 결과가 없습니다. 질문 키워드를 바꾸거나 chunk 저장 상태를 확인하세요.")

        results = st.session_state.get("rag_results", [])
        saved_question = st.session_state.get("rag_question", question)

        if results:
            st.subheader("검색 결과")

            for i, r in enumerate(results, start=1):
                with st.expander(
                    f"{i}. score={r['score']} | Chapter {r['chapter_no']} - {r['chapter_title']}",
                    expanded=(i == 1),
                ):
                    st.write(r["content"])

            triples = get_ontology_context_v3(tutorial_id, query=saved_question, limit=30)

            st.subheader("RAG Prompt")
            rag_prompt = build_rag_prompt_v3(
                question=saved_question,
                contexts=results,
                ontology_triples=triples,
                language=selected_tutorial.get("language") or "Korean",
            )
            st.code(rag_prompt, language="markdown")

            if do_answer:
                st.subheader("LLM Answer")
                try:
                    from utils.call_llm import call_llm

                    with st.spinner("LLM 답변 생성 중..."):
                        answer = call_llm(rag_prompt)

                    st.markdown(answer)

                except Exception as e:
                    st.error("LLM 답변 생성 실패")
                    st.exception(e)


# -----------------------------
# Chat tab
# -----------------------------

with tab_chat:
    st.header("Chat")

    if not selected_tutorial:
        st.info("저장된 튜토리얼이 없습니다. Generate & Save 탭에서 먼저 생성하세요.")
    else:
        tutorial_id = selected_tutorial["id"]
        chat_lang = selected_tutorial.get("language") or "Korean"

        st.caption(
            f"'{selected_tutorial['title']}' 지식으로 대화합니다. "
            "질문할 때마다 벡터/키워드 검색으로 관련 내용을 찾아 답변합니다."
        )

        # Per-tutorial conversation history kept in session state.
        chat_store = st.session_state.setdefault("chat_by_tutorial", {})
        history = chat_store.setdefault(tutorial_id, [])

        col_k, col_reset = st.columns([3, 1])
        with col_k:
            chat_top_k = st.slider("검색 chunk 수", 1, 10, 5, key="chat_top_k")
        with col_reset:
            st.write("")
            if st.button("대화 초기화", key="chat_reset"):
                chat_store[tutorial_id] = []
                st.rerun()

        # Render the existing conversation.
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_msg = st.chat_input("코드베이스에 대해 질문하세요")

        if user_msg:
            history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            # Retrieve context (hybrid vector/keyword) + ontology relations.
            results = search_tutorial_context_v3(
                tutorial_id=tutorial_id,
                query=user_msg,
                top_k=chat_top_k,
            )
            triples = get_ontology_context_v3(tutorial_id, query=user_msg, limit=30)

            prompt = build_chat_prompt_v4(
                question=user_msg,
                contexts=results,
                ontology_triples=triples,
                history=history[:-1],  # exclude the just-added current question
                language=chat_lang,
            )

            with st.chat_message("assistant"):
                answer = None
                try:
                    from utils.call_llm import call_llm_stream

                    answer = st.write_stream(call_llm_stream(prompt))
                except Exception as e:
                    st.error("답변 생성 실패")
                    st.exception(e)

                if results:
                    with st.expander(f"참고한 chunk {len(results)}개 / 관계 {len(triples)}개"):
                        for i, r in enumerate(results, start=1):
                            st.caption(
                                f"{i}. score={r['score']} · Ch {r['chapter_no']} - {r['chapter_title']}"
                            )
                else:
                    st.caption("관련 chunk를 찾지 못했습니다. 근거 없이 답변했을 수 있습니다.")

            if answer:
                history.append({"role": "assistant", "content": answer})


# -----------------------------
# Ontology tab
# -----------------------------

with tab_ontology:
    st.header("Ontology RAG")

    if not selected_tutorial:
        st.info("저장된 튜토리얼이 없습니다.")
    else:
        tutorial_id = selected_tutorial["id"]

        if st.button("Ontology 재구성", type="primary"):
            try:
                result = rebuild_ontology_v3(tutorial_id)
                st.success(f"Ontology 재구성 완료: {result}")
            except Exception as e:
                st.error("Ontology 재구성 실패")
                st.exception(e)

        query = st.text_input(
            "Ontology 관계 검색어",
            value="",
            placeholder="예: 의존성, Router, Middleware, Pydantic",
        )

        triples = get_ontology_context_v3(
            tutorial_id=tutorial_id,
            query=query or None,
            limit=100,
        )

        if triples:
            st.subheader("Ontology Triples")
            for t in triples:
                st.write(
                    f"- **{t['source']}** "
                    f"-- `{t['relation']}` / `{t['edge_type']}` --> "
                    f"**{t['target']}**"
                )
        else:
            st.info("저장된 ontology 관계가 없습니다. Ontology 재구성을 실행하세요.")

        tutorial = get_tutorial(tutorial_id)
        st.subheader("Original Mermaid")
        st.code(tutorial["mermaid_graph"] or "", language="mermaid")
        render_mermaid(tutorial["mermaid_graph"])


# -----------------------------
# Admin tab
# -----------------------------

def _cleanup_output_dirs(output_dirs):
    """Remove local output folders, returning human-readable status lines."""
    lines = []
    for d in output_dirs:
        if not d:
            continue
        p = Path(d)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            lines.append(f"로컬 output 삭제: {p}")
        else:
            lines.append(f"로컬 output 없음(건너뜀): {p}")
    return lines


_COUNT_LABELS = {
    "tutorials": "튜토리얼",
    "chapters": "챕터",
    "chunks": "청크(임베딩 포함)",
    "ontology_nodes": "온톨로지 노드",
    "ontology_edges": "온톨로지 엣지",
    "fine_tuning_examples": "파인튜닝 예시",
    "rag_logs": "RAG 로그",
}


def _render_counts(counts):
    st.table(
        [{"항목": _COUNT_LABELS.get(k, k), "건수": counts.get(k, 0)} for k in _COUNT_LABELS]
    )


with tab_admin:
    st.header("Admin / Delete")

    # Show a one-time summary after a delete triggered a rerun.
    _summary = st.session_state.pop("last_delete_summary", None)
    if _summary:
        st.success(_summary["msg"])
        _render_counts(_summary["counts"])
        for line in _summary.get("local", []):
            st.caption(line)

    if not selected_tutorial:
        st.info("삭제할 튜토리얼이 없습니다.")
    else:
        tutorial_id = selected_tutorial["id"]
        tutorial = get_tutorial(tutorial_id)
        repo_info = get_tutorial_repository(tutorial_id)

        st.write("선택된 튜토리얼:")
        st.code(f"{tutorial['title']}\n{tutorial['source_repo_url']}\nID: {tutorial_id}")

        repo_tut_count = repo_info["tutorial_count"] if repo_info else 1

        scope = st.radio(
            "삭제 범위",
            ["이 튜토리얼만", "이 레포지토리 전체 (모든 튜토리얼)"],
            help=(
                "‘레포지토리 전체’를 선택하면 같은 GitHub repo로 만든 "
                f"모든 튜토리얼({repo_tut_count}개)과 그에 딸린 챕터·청크·임베딩·"
                "온톨로지·파인튜닝·RAG 로그가 한 번에 삭제됩니다."
            ),
        )
        repo_scope = scope.startswith("이 레포지토리")

        # Preview exactly what will be removed.
        if repo_scope and repo_info:
            preview = count_related_v4(repository_id=repo_info["repository_id"])
            st.warning(
                f"레포지토리 **{repo_info['repo_url']}** 및 아래 모든 관련 데이터가 삭제됩니다."
            )
        else:
            preview = count_related_v4(tutorial_id=tutorial_id)
            st.warning("이 튜토리얼 및 아래 모든 관련 데이터가 삭제됩니다.")

        st.caption("삭제 예정 항목")
        _render_counts(preview)

        delete_local = st.checkbox("DB 삭제 후 로컬 output 폴더도 삭제", value=False)
        confirm = st.text_input("삭제하려면 DELETE 입력")

        btn_label = "레포지토리 전체 삭제" if repo_scope else "튜토리얼 삭제"

        if st.button(btn_label, type="primary"):
            if confirm != "DELETE":
                st.error("확인 문구가 일치하지 않습니다. DELETE를 정확히 입력하세요.")
                st.stop()

            if repo_scope and repo_info:
                result = delete_repository_v4(repo_info["repository_id"])
                output_dirs = result.get("output_dirs", [])
                success_msg = f"레포지토리 삭제 완료: {result.get('repo_url', '')}"
            else:
                result = delete_tutorial_v3(tutorial_id)
                output_dirs = [result.get("output_dir")] if result.get("output_dir") else []
                success_msg = f"튜토리얼 삭제 완료: {result.get('title', '')}"

            if not result.get("deleted"):
                st.error(result.get("reason", "삭제 실패"))
                st.stop()

            local_lines = _cleanup_output_dirs(output_dirs) if delete_local else []

            # Stash summary and rerun so the sidebar list refreshes immediately.
            st.session_state["last_delete_summary"] = {
                "msg": success_msg,
                "counts": result.get("counts", preview),
                "local": local_lines,
            }
            st.rerun()
