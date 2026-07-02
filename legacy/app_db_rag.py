import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from db_store import (
    list_tutorials,
    get_tutorial,
    get_chapters,
    keyword_search_chunks,
    rebuild_ontology_from_tutorial,
    get_ontology_context,
    build_rag_prompt,
)

load_dotenv(".env", override=True)

st.set_page_config(
    page_title="Codebase Tutorial Library & RAG",
    layout="wide",
)

st.title("Codebase Tutorial Library & RAG")

if not os.getenv("DATABASE_URL"):
    st.error("DATABASE_URL이 설정되어 있지 않습니다. .env 파일을 확인하세요.")
    st.stop()


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


tutorials = list_tutorials()

if not tutorials:
    st.warning("DB에 저장된 튜토리얼이 없습니다.")
    st.stop()

selected = st.sidebar.selectbox(
    "튜토리얼 선택",
    tutorials,
    format_func=lambda x: f"{x['title']} | {x['source_repo_url']}",
)

tutorial_id = selected["id"]

tab_library, tab_rag, tab_ontology = st.tabs([
    "Library",
    "RAG Search",
    "Ontology RAG",
])


with tab_library:
    tutorial = get_tutorial(tutorial_id)
    chapters = get_chapters(tutorial_id)

    st.header(tutorial["title"])
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

    chapter_labels = [
        f"{ch['chapter_no']}. {ch['title']}"
        for ch in chapters
    ]

    if chapters:
        chapter_idx = st.selectbox(
            "챕터 선택",
            range(len(chapters)),
            format_func=lambda i: chapter_labels[i],
        )

        ch = chapters[chapter_idx]
        st.markdown(ch["markdown"])
    else:
        st.info("저장된 챕터가 없습니다.")


with tab_rag:
    st.header("RAG Search")

    question = st.text_area(
        "질문",
        value="FastAPI의 의존성 주입을 초보자에게 설명해줘.",
        height=100,
    )

    top_k = st.slider("검색 chunk 수", min_value=1, max_value=10, value=5)

    if st.button("검색 실행", type="primary"):
        results = keyword_search_chunks(
            tutorial_id=tutorial_id,
            query=question,
            top_k=top_k,
        )

        st.session_state["rag_results"] = results
        st.session_state["rag_question"] = question

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

        st.subheader("RAG Prompt")

        triples = get_ontology_context(tutorial_id, query=saved_question, limit=20)

        rag_prompt = build_rag_prompt(
            question=saved_question,
            chunks=results,
            ontology_triples=triples,
        )

        st.code(rag_prompt, language="markdown")

        st.subheader("LLM Answer")

        if st.button("검색 context로 답변 생성"):
            try:
                from utils.call_llm import call_llm

                with st.spinner("LLM 답변 생성 중..."):
                    answer = call_llm(rag_prompt)

                st.markdown(answer)

            except Exception as e:
                st.error("LLM 답변 생성 실패")
                st.exception(e)


with tab_ontology:
    st.header("Ontology RAG")

    st.write(
        "Mermaid flowchart를 기반으로 ontology_nodes / ontology_edges를 재구성하고, "
        "관계 context를 확인합니다."
    )

    if st.button("Ontology 재구성", type="primary"):
        try:
            result = rebuild_ontology_from_tutorial(tutorial_id)
            st.success(f"Ontology 재구성 완료: nodes={result['nodes']}, edges={result['edges']}")
        except Exception as e:
            st.error("Ontology 재구성 실패")
            st.exception(e)

    query = st.text_input(
        "Ontology 관계 검색어",
        value="",
        placeholder="예: Router, Dependency, Middleware",
    )

    triples = get_ontology_context(
        tutorial_id=tutorial_id,
        query=query or None,
        limit=50,
    )

    if triples:
        st.subheader("Ontology Triples")
        for t in triples:
            st.write(f"- **{t['source']}** -- `{t['relation']}` --> **{t['target']}**")
    else:
        st.info("저장된 ontology 관계가 없습니다. 먼저 Ontology 재구성을 실행하세요.")

    tutorial = get_tutorial(tutorial_id)
    st.subheader("Original Mermaid")
    st.code(tutorial["mermaid_graph"] or "", language="mermaid")
    render_mermaid(tutorial["mermaid_graph"])
