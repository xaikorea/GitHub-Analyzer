import os
import sys
import time
import shutil
import shlex
import subprocess
import hmac
import re
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from db_store import (
    save_tutorial_result_to_db,
    describe_repo_url,
    list_tutorials,
    get_tutorial,
    get_chapters,
    search_tutorial_context_v3,
    rebuild_ontology_v3,
    get_ontology_context_v3,
    build_rag_prompt_v3,
    build_chat_prompt_v4,
    search_across_tutorials,
    build_multi_repo_rag_prompt,
    build_codegen_examples,
    export_codegen_jsonl,
    delete_tutorial_v3,
    delete_repository_v4,
    count_related_v4,
    get_tutorial_repository,
    update_tutorial_mermaid,
    db_counts_v3,
)
from agent_rag import (
    agent_gather,
    build_agent_answer_prompt,
    judge_answer,
    refine_answer,
    deep_research,
    build_deep_research_prompt,
)
from mermaid_utils import heal_mermaid
import tracing

load_dotenv(".env", override=True)

st.set_page_config(
    page_title="GitHub Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# On Streamlit Cloud, configuration is provided via st.secrets (there is no
# .env file). The rest of the app reads config with os.getenv(), so mirror any
# top-level string secrets into os.environ. setdefault() keeps a local .env /
# real environment variable as the source of truth when one already exists.
try:
    for _sk, _sv in st.secrets.items():
        if isinstance(_sv, str):
            os.environ.setdefault(_sk, _sv)
except Exception:
    # No secrets.toml configured (e.g. plain local run) — .env / env vars only.
    pass

# -----------------------------
# Design system (CSS) + UI helpers
# -----------------------------

st.markdown(
    """
    <style>
      :root {
        --ga-accent: #7c5cff;
        --ga-accent-2: #22d3ee;
        --ga-ok: #3fb950;
        --ga-warn: #d29922;
        --ga-surface: #161b22;
        --ga-border: #2a3038;
        --ga-text-dim: #8b949e;
      }
      /* tighten default paddings for a denser, app-like feel */
      .block-container { padding-top: 2.2rem; max-width: 1200px; }
      /* Hero */
      .ga-hero {
        background: radial-gradient(1200px 200px at 0% 0%, rgba(124,92,255,.25), transparent 60%),
                    linear-gradient(135deg, #1b2130 0%, #12161d 100%);
        border: 1px solid var(--ga-border);
        border-radius: 16px; padding: 22px 26px; margin-bottom: 18px;
      }
      .ga-hero-badge {
        display: inline-block; font-weight: 700; font-size: 13px; letter-spacing:.3px;
        color: #cbd5ff; background: rgba(124,92,255,.18);
        border: 1px solid rgba(124,92,255,.45); padding: 4px 12px; border-radius: 999px;
      }
      .ga-hero-title { font-size: 27px; font-weight: 800; margin: 12px 0 4px; color:#f0f3f8; }
      .ga-hero-sub { color: var(--ga-text-dim); font-size: 14.5px; }
      /* Section header */
      .ga-section-title { font-size: 21px; font-weight: 750; color:#f0f3f8; display:flex; gap:8px; align-items:center; }
      .ga-section-sub { color: var(--ga-text-dim); font-size: 13.5px; margin: 2px 0 14px; }
      /* Status chips */
      .ga-chip { display:inline-flex; align-items:center; gap:6px; font-size:12.5px;
        padding:4px 11px; border-radius:999px; margin:3px 6px 3px 0; border:1px solid var(--ga-border);
        background: var(--ga-surface); color:#c9d1d9; }
      .ga-chip b { color:#fff; font-weight:650; }
      .ga-chip--ok { border-color: rgba(63,185,80,.5); background: rgba(63,185,80,.12); }
      .ga-chip--warn { border-color: rgba(210,153,34,.5); background: rgba(210,153,34,.12); }
      .ga-chip .dot { width:7px; height:7px; border-radius:50%; }
      .ga-chip--ok .dot { background: var(--ga-ok); }
      .ga-chip--warn .dot { background: var(--ga-warn); }
      /* Empty state */
      .ga-empty { text-align:center; padding: 40px 20px; border:1px dashed var(--ga-border);
        border-radius:14px; background: rgba(255,255,255,.02); margin: 8px 0; }
      .ga-empty-icon { font-size: 34px; }
      .ga-empty-title { font-weight:700; font-size:16px; margin-top:8px; color:#e6edf3; }
      .ga-empty-hint { color: var(--ga-text-dim); font-size:13.5px; margin-top:4px; }
      /* Tabs -> segmented control look */
      .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--ga-border); }
      .stTabs [data-baseweb="tab"] { height: 40px; border-radius: 9px 9px 0 0; padding: 0 14px; font-weight: 600; }
      .stTabs [aria-selected="true"] { background: rgba(124,92,255,.14); color:#fff; }
      /* Buttons */
      .stButton>button { border-radius: 10px; font-weight: 650; border:1px solid var(--ga-border); }
      .stButton>button[kind="primary"] { background: var(--ga-accent); border-color: var(--ga-accent); }
      .stButton>button[kind="primary"]:hover { filter: brightness(1.08); }
      /* Cards: expanders & metrics */
      [data-testid="stExpander"] { border:1px solid var(--ga-border); border-radius:12px; background: rgba(255,255,255,.015); }
      [data-testid="stMetric"] { background: var(--ga-surface); border:1px solid var(--ga-border);
        border-radius:12px; padding:12px 14px; }
      /* Sidebar */
      [data-testid="stSidebar"] { border-right:1px solid var(--ga-border); }
      .ga-side-brand { font-weight:800; font-size:18px; color:#f0f3f8; display:flex; gap:8px; align-items:center; }
      .ga-side-label { text-transform:uppercase; letter-spacing:.6px; font-size:11px;
        color: var(--ga-text-dim); font-weight:700; margin:6px 0 2px; }
      /* Secret status row (shows 'set/not set' — never the value) */
      .ga-secret-row { font-size:13px; color:#c9d1d9; margin:9px 0 1px; display:flex; gap:6px; align-items:center; }
      .ga-secret-ok { color: var(--ga-ok); font-weight:650; font-size:12px; }
      .ga-secret-no { color: var(--ga-warn); font-weight:650; font-size:12px; }
      /* Hide Streamlit's password reveal ("eye") button so masked secrets
         can never be unmasked in the browser. */
      button[aria-label="Show password text"],
      button[aria-label="Hide password text"],
      [data-testid="stTextInputShowPasswordButton"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(title: str, subtitle: str = "", icon: str = ""):
    sub = f'<div class="ga-section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="ga-section-title">{icon} {title}</div>{sub}',
        unsafe_allow_html=True,
    )


def status_chip(label: str, value: str, ok: bool = True) -> str:
    cls = "ga-chip ga-chip--ok" if ok else "ga-chip ga-chip--warn"
    return f'<span class="{cls}"><span class="dot"></span><b>{label}</b> {value}</span>'


def render_chips(chips: list[str]):
    st.markdown("".join(chips), unsafe_allow_html=True)


def empty_state(title: str, hint: str = "", icon: str = "📭"):
    st.markdown(
        f'<div class="ga-empty"><div class="ga-empty-icon">{icon}</div>'
        f'<div class="ga-empty-title">{title}</div>'
        f'<div class="ga-empty-hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


def require_auth():
    """Simple shared-credential gate so only the team can use the app.

    Credentials come from APP_USERNAME / APP_PASSWORD (env or Streamlit Secrets),
    never hardcoded — so committing the app to a repo can't leak the password.
    If they are not configured the app stays open (with a warning) so local
    development / first run isn't locked out.
    """
    expected_user = os.getenv("APP_USERNAME")
    expected_pw = os.getenv("APP_PASSWORD")

    if not expected_user or not expected_pw:
        st.warning("🔓 로그인 미설정 (APP_USERNAME / APP_PASSWORD). 현재 누구나 접근할 수 있습니다.")
        return

    if st.session_state.get("_authed"):
        return

    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown(
            '<div class="ga-hero" style="text-align:center;">'
            '<span class="ga-hero-badge">🔒 팀 전용</span>'
            '<div class="ga-hero-title">GitHub Analyzer 로그인</div>'
            '<div class="ga-hero-sub">팀 계정으로 로그인하세요.</div></div>',
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            u = st.text_input("아이디", key="login_user")
            p = st.text_input("비밀번호", type="password", key="login_pw")
            submitted = st.form_submit_button("로그인", type="primary", use_container_width=True)
        if submitted:
            # Constant-time compare to avoid leaking length/timing.
            ok_user = hmac.compare_digest(u or "", expected_user)
            ok_pw = hmac.compare_digest(p or "", expected_pw)
            if ok_user and ok_pw:
                st.session_state["_authed"] = True
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
    st.stop()


require_auth()


st.markdown(
    """
    <div class="ga-hero">
      <span class="ga-hero-badge">🔍 GitHub Analyzer</span>
      <div class="ga-hero-title">코드베이스를 검색·대화 가능한 지식으로</div>
      <div class="ga-hero-sub">AI 튜토리얼 생성 · Postgres/pgvector RAG · 스트리밍 챗 · 온톨로지 그래프</div>
    </div>
    """,
    unsafe_allow_html=True,
)


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


def _sanitize_mermaid(code: str) -> str:
    """Deterministically fix the most common LLM Mermaid breakage: raw newlines
    inside ["..."] node labels, which make Mermaid v10 raise "Syntax error in
    text". Collapses whitespace inside each node label to single spaces."""
    if not code:
        return code
    return re.sub(
        r'\["(.*?)"\]',
        lambda m: '["' + " ".join(m.group(1).split()) + '"]',
        code,
        flags=re.DOTALL,
    )


def render_mermaid(mermaid_code: str, height: int = 520):
    if not mermaid_code:
        st.info("Mermaid graph가 없습니다.")
        return

    mermaid_code = _sanitize_mermaid(mermaid_code)

    # Inject the diagram source as a JSON-encoded JS string (not raw HTML) so
    # characters like & < > " in labels can't be mangled by the browser's HTML
    # parser before Mermaid sees them. Render explicitly and surface the real
    # error instead of Mermaid's generic "Syntax error in text" overlay.
    code_js = json.dumps(mermaid_code)
    html = f"""
    <div id="ga-mermaid"></div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'loose',
        flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }}
      }});
      const code = {code_js};
      const el = document.getElementById('ga-mermaid');
      try {{
        const {{ svg }} = await mermaid.render('ga-mermaid-svg', code);
        el.innerHTML = svg;
      }} catch (e) {{
        el.innerHTML = '<pre style="color:#f88;white-space:pre-wrap;font-size:12px">'
          + 'Mermaid render error:\\n' + ((e && e.message) ? e.message : e)
          + '\\n\\n--- source ---\\n' + code.replace(/</g, '&lt;') + '</pre>';
      }}
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


def secret_field(label: str, env_var: str, help: str | None = None) -> str:
    """Render a secret config field WITHOUT ever putting the current value on screen.

    Secrets are never pre-filled into inputs, so a deployed/shared app can't leak
    them via the browser DOM or the password reveal button:
    - If the secret already exists (env / Streamlit Secrets): show a read-only
      'set' status and only reveal an EMPTY override input when the user opts in.
    - If it is not set: show an empty password input so it can be entered.

    Returns the user-entered override. An empty string means "keep the existing
    environment value" (set_env_from_ui only writes non-empty values).
    """
    has_env = bool(os.getenv(env_var))
    if has_env:
        st.sidebar.markdown(
            f'<div class="ga-secret-row">🔒 <b>{label}</b>'
            f'<span class="ga-secret-ok">설정됨 · 환경변수/Secrets</span></div>',
            unsafe_allow_html=True,
        )
        if not st.sidebar.checkbox(
            f"{label} 이 세션에서 변경", value=False, key=f"ovr_{env_var}"
        ):
            return ""
        return st.sidebar.text_input(
            f"새 {label}", value="", type="password", key=f"in_{env_var}",
            placeholder="새 값 입력 (이 세션에만 적용)",
        )
    st.sidebar.markdown(
        f'<div class="ga-secret-row">🔓 <b>{label}</b>'
        f'<span class="ga-secret-no">미설정</span></div>',
        unsafe_allow_html=True,
    )
    return st.sidebar.text_input(
        label, value="", type="password", key=f"in_{env_var}",
        help=help, placeholder="아직 설정되지 않음", label_visibility="collapsed",
    )


# -----------------------------
# sidebar setup
# -----------------------------

st.sidebar.markdown('<div class="ga-side-brand">🔍 GitHub Analyzer</div>', unsafe_allow_html=True)
st.sidebar.caption("codebase → searchable knowledge base")

if st.session_state.get("_authed"):
    if st.sidebar.button("🚪 로그아웃", key="logout"):
        st.session_state["_authed"] = False
        st.rerun()
st.sidebar.markdown('<div class="ga-side-label">① LLM & 연결 설정</div>', unsafe_allow_html=True)

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

# Secrets are never pre-filled into inputs (so they can't be revealed in a
# deployed/shared app). secret_field() shows only a 'set/not set' status and
# requires an explicit opt-in to enter a new value for the session.
api_key = secret_field(api_key_label, api_key_label)
github_token = secret_field(
    "GITHUB_TOKEN", "GITHUB_TOKEN",
    help="공개 repo만 테스트하면 비워도 되지만, rate limit 때문에 넣는 것이 좋습니다.",
)
database_url = secret_field("DATABASE_URL", "DATABASE_URL")

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

# Live connection status chips
with st.sidebar:
    st.markdown('<div class="ga-side-label">연결 상태</div>', unsafe_allow_html=True)
    render_chips([
        status_chip("Provider", os.getenv("LLM_PROVIDER", "—") or "—", ok=bool(os.getenv("LLM_PROVIDER"))),
        status_chip("Key", "설정됨" if os.getenv(api_key_label) else "없음", ok=bool(os.getenv(api_key_label))),
        status_chip("DB", "연결됨" if os.getenv("DATABASE_URL") else "미설정", ok=bool(os.getenv("DATABASE_URL"))),
    ])

st.sidebar.divider()
st.sidebar.markdown('<div class="ga-side-label">② 저장된 튜토리얼</div>', unsafe_allow_html=True)

tutorials = get_tutorials_safe()

selected_tutorial = None
if tutorials:
    selected_tutorial = st.sidebar.selectbox(
        "튜토리얼 선택",
        tutorials,
        format_func=lambda x: f"📘 {x['title'].replace('Tutorial: ', '')} · {x.get('language', '')}",
    )
    if selected_tutorial:
        st.sidebar.caption(f"🔗 {selected_tutorial['source_repo_url']}")
else:
    st.sidebar.info("아직 저장된 튜토리얼이 없습니다.\n\n**➕ Generate** 탭에서 첫 저장소를 분석해 보세요.")


(
    tab_setup, tab_generate, tab_library, tab_rag, tab_chat,
    tab_multirag, tab_agent, tab_ontology, tab_finetune, tab_admin,
) = st.tabs([
    "⚙️ Setup",
    "➕ Generate",
    "📚 Library",
    "🔎 RAG Search",
    "💬 Chat",
    "🔗 Multi-Repo RAG",
    "🤖 Agent",
    "🕸 Ontology",
    "🛠 Fine-tune",
    "🗑 Admin",
])


# -----------------------------
# Setup tab
# -----------------------------

with tab_setup:
    section_header(
        "환경 상태",
        "LLM · 임베딩 · 데이터베이스 · GitHub 토큰 연결 상태를 한눈에 확인합니다.",
        icon="⚙️",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Provider", os.getenv("LLM_PROVIDER", "—") or "—")
    col2.metric("Model", model_name or "—")
    col3.metric("Database", "연결됨" if os.getenv("DATABASE_URL") else "미설정")

    render_chips([
        status_chip(api_key_label.replace("_API_KEY", " Key"), "설정됨" if os.getenv(api_key_label) else "없음", ok=bool(os.getenv(api_key_label))),
        status_chip("Embeddings", "가능" if os.getenv("OPENAI_API_KEY") else "키 없음", ok=bool(os.getenv("OPENAI_API_KEY"))),
        status_chip("GitHub Token", "설정됨" if os.getenv("GITHUB_TOKEN") else "없음", ok=bool(os.getenv("GITHUB_TOKEN"))),
        status_chip("DB", "연결됨" if os.getenv("DATABASE_URL") else "미설정", ok=bool(os.getenv("DATABASE_URL"))),
    ])

    st.info(
        "왼쪽 사이드바에서 Provider·Key·DB를 설정합니다. "
        "실제 저장소 분석은 **➕ Generate** 탭에서 진행합니다."
    )
    if not os.getenv("OPENAI_API_KEY"):
        st.caption("💡 OpenAI 키가 없으면 RAG는 키워드 검색으로 동작합니다. 키를 넣으면 의미 기반(시맨틱) 검색이 활성화됩니다.")


# -----------------------------
# Generate & Save tab
# -----------------------------

with tab_generate:
    section_header(
        "저장소 분석 & 저장",
        "GitHub 저장소를 분석해 튜토리얼을 만들고 DB에 저장합니다. 아래에서 URL 구조를 먼저 확인하세요.",
        icon="➕",
    )

    repo_url = st.text_input(
        "GitHub Repository URL",
        value="https://github.com/fastapi/fastapi/tree/master/fastapi",
    )

    # --- Explain the URL structure and what will actually be analyzed ---
    _parsed = describe_repo_url(repo_url)
    with st.expander("🔎 이 URL은 어떻게 해석되나요? (분석 범위 · Library/시각화/RAG 연결)", expanded=True):
        if not _parsed["valid"]:
            st.warning(
                "GitHub 저장소 URL 형식이 아닙니다.\n\n"
                "- 전체 저장소: `https://github.com/owner/repo`\n"
                "- 특정 브랜치: `https://github.com/owner/repo/tree/main`\n"
                "- 하위 폴더만: `https://github.com/owner/repo/tree/main/src/pkg`"
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Owner", _parsed["owner"])
            c2.metric("Repository", _parsed["repo"])
            c3.metric("Branch/Ref", _parsed["ref"] or "기본 브랜치")
            scope_label = {
                "whole_repo": "전체 저장소",
                "subdirectory": "하위 폴더만",
                "single_file": "단일 파일",
            }[_parsed["scope"]]
            c4.metric("분석 범위", scope_label)

            st.caption(f"정규화된 URL: `{_parsed['canonical_url']}`")
            if _parsed["sub_path"]:
                st.caption(f"하위 경로(sub_path): `{_parsed['sub_path']}`")

            # Scope-specific guidance
            if _parsed["scope"] == "single_file":
                st.error(
                    "`/blob/` URL은 **단일 파일**을 가리킵니다. 코드베이스 분석에는 부적합합니다. "
                    "`/blob/`를 `/tree/`로 바꾸거나, 폴더/저장소 URL을 사용하세요."
                )
            elif _parsed["scope"] == "subdirectory":
                st.info(
                    f"이 URL은 **`{_parsed['sub_path']}` 폴더만** 분석합니다. "
                    "저장소 전체를 분석하려면 `/tree/<branch>/...` 부분을 지우고 "
                    f"`https://github.com/{_parsed['repo_name']}` 형태로 입력하세요."
                )
            else:
                st.info(
                    "이 URL은 **저장소 전체**를 분석합니다. 규모가 크면 아래 "
                    "include/exclude 패턴과 max-size로 범위를 좁히거나, "
                    "특정 폴더 URL(`/tree/<branch>/<폴더>`)로 지정하면 결과가 더 정확해집니다."
                )

            st.markdown(
                "**입력한 범위가 이후 단계로 이어지는 방식**\n\n"
                "1. **크롤링** — 위 범위(브랜치/하위 폴더) 안에서 include에 맞고 exclude에 안 걸리는 "
                "파일만 수집합니다. (`--repo` 로 URL이 그대로 전달되어 `tree/branch/subpath`가 해석됩니다.)\n"
                "2. **Library** — 수집된 코드로 핵심 추상화를 뽑아 `index.md`(요약 + Mermaid) 와 챕터를 만듭니다. "
                "즉 여기 범위에 **없는 파일은 챕터/설명에 나타나지 않습니다.**\n"
                "3. **시각화(Ontology)** — 추상화(노드)와 관계(엣지)를 Mermaid + 챕터 링크에서 추출합니다. "
                "범위가 좁고 응집적일수록 그래프가 깔끔합니다.\n"
                "4. **RAG / Chat** — 각 챕터가 청크로 분할·(임베딩 시)벡터화되어 검색됩니다. "
                "따라서 **RAG로 답하게 하고 싶은 부분**이 있으면 그 파일이 include에 포함되고 이 범위 안에 있어야 합니다.\n\n"
                "**참조 팁**: 소스 + 문서를 함께 넣으면(`*.py *.md`) 설명 품질이 올라가고, "
                "`tests/*`·`build/*`·`docs_src/*` 등은 제외하는 것이 좋습니다."
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

        # Force the child process to emit UTF-8 on stdout. On Korean Windows
        # the child's default console encoding is cp949, so printing a file
        # path or log line containing CJK/emoji characters would raise
        # UnicodeEncodeError and abort the whole crawl. We read stdout as
        # UTF-8 below, so make the child write UTF-8 too.
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        child_env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            cmd,
            env=child_env,
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
    section_header("라이브러리", "생성된 튜토리얼의 요약 · 플로우차트 · 챕터를 열람합니다.", icon="📚")

    if not selected_tutorial:
        empty_state("아직 튜토리얼이 없습니다", "➕ Generate 탭에서 첫 저장소를 분석해 보세요.", icon="📚")
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
    section_header("RAG 검색", "질문과 관련된 코드/설명을 하이브리드(벡터+키워드) 검색으로 찾고, 근거 기반 답변을 생성합니다.", icon="🔎")

    if not selected_tutorial:
        empty_state("아직 튜토리얼이 없습니다", "➕ Generate 탭에서 저장소를 분석해 저장한 뒤 이용하세요.", icon="📭")
    else:
        tutorial_id = selected_tutorial["id"]

        question = st.text_area(
            "질문",
            value="FastAPI의 의존성 주입을 초보자에게 설명해줘.",
            height=100,
        )

        top_k = st.slider("검색 chunk 수", min_value=1, max_value=10, value=5)

        rag_judge = st.checkbox(
            "답변 검증·개선 (Judge)",
            value=False,
            key="rag_judge",
            help="LLM-as-Judge로 답변의 근거 충실도를 채점하고, 부족하면 한 번 자동 개선합니다.",
        )

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

                    rag_lang = selected_tutorial.get("language") or "Korean"
                    with st.spinner("LLM 답변 생성 중..."):
                        answer = call_llm(rag_prompt)

                        if rag_judge:
                            evidence = "\n\n".join(
                                f"[Ch{r['chapter_no']} {r['chapter_title']}]\n{r['content']}"
                                for r in results
                            )
                            verdict = judge_answer(saved_question, evidence, answer)
                            refined = False
                            if not verdict["grounded"] or verdict["score"] < 4:
                                answer = refine_answer(saved_question, evidence, answer, verdict["issues"], language=rag_lang)
                                refined = True
                                verdict = judge_answer(saved_question, evidence, answer)

                    if rag_judge:
                        render_chips([
                            status_chip("Judge 점수", f"{verdict['score']}/5", ok=verdict["score"] >= 4),
                            status_chip("근거 충실", "예" if verdict["grounded"] else "아니오", ok=verdict["grounded"]),
                            status_chip("개선", "적용됨" if refined else "불필요"),
                        ])
                        if verdict.get("issues"):
                            st.caption(f"🔎 Judge 코멘트: {verdict['issues']}")

                    st.markdown(answer)

                except Exception as e:
                    st.error("LLM 답변 생성 실패")
                    st.exception(e)


# -----------------------------
# Chat tab
# -----------------------------

with tab_chat:
    section_header("챗", "선택한 튜토리얼 지식으로 멀티턴 대화를 나눕니다. 답변은 실시간 스트리밍됩니다.", icon="💬")

    if not selected_tutorial:
        empty_state("아직 튜토리얼이 없습니다", "➕ Generate 탭에서 첫 저장소를 분석해 보세요.", icon="📚")
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
    section_header("온톨로지", "코드베이스의 핵심 개념(노드)과 관계(엣지)를 그래프로 추출·시각화합니다.", icon="🕸")

    if not selected_tutorial:
        empty_state("아직 튜토리얼이 없습니다", "➕ Generate 탭에서 저장소를 분석해 저장한 뒤 이용하세요.", icon="📭")
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
        st.subheader("Diagram")

        heal_c1, heal_c2 = st.columns([1, 3])
        with heal_c1:
            if st.button("🩹 Mermaid 자동 복구", key="heal_mermaid"):
                with st.spinner("LLM이 다이어그램 문법을 점검·복구 중..."):
                    healed = heal_mermaid(tutorial["mermaid_graph"] or "")
                st.session_state["healed_mermaid"] = {"tid": tutorial_id, **healed}

        healed = st.session_state.get("healed_mermaid")
        if healed and healed.get("tid") == tutorial_id:
            st.caption(f"자동 복구: {healed['note']}")
            if healed["changed"]:
                st.markdown("**복구된 다이어그램 미리보기**")
                render_mermaid(healed["mermaid"])
                if st.button("💾 복구본을 DB에 저장", key="save_healed"):
                    if update_tutorial_mermaid(tutorial_id, healed["mermaid"]):
                        st.success("복구된 Mermaid를 저장했습니다. 새로고침 시 반영됩니다.")
                    else:
                        st.error("저장 실패")
                st.divider()

        st.markdown("**Original Mermaid**")
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
    section_header("관리 · 삭제", "튜토리얼 또는 레포지토리 전체를 관련 데이터까지 안전하게 삭제합니다.", icon="🗑")

    # Show a one-time summary after a delete triggered a rerun.
    _summary = st.session_state.pop("last_delete_summary", None)
    if _summary:
        st.success(_summary["msg"])
        _render_counts(_summary["counts"])
        for line in _summary.get("local", []):
            st.caption(line)

    if not selected_tutorial:
        empty_state("삭제할 항목이 없습니다", "저장된 튜토리얼이 있어야 삭제할 수 있습니다.", icon="🗑")
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


# -----------------------------
# Multi-Repo RAG tab
# -----------------------------

with tab_multirag:
    section_header(
        "멀티-레포 RAG",
        "여러 저장소를 함께 검색해 공통점·차이·연결 인사이트를 얻습니다.",
        icon="🔗",
    )

    if not tutorials:
        empty_state("저장된 저장소가 없습니다", "➕ Generate 탭에서 저장소 2개 이상을 저장하면 교차 검색이 가능합니다.", icon="🔗")
    else:
        picked = st.multiselect(
            "검색할 저장소(튜토리얼) 선택",
            options=tutorials,
            default=tutorials[: min(3, len(tutorials))],
            format_func=lambda x: f"{x['title'].replace('Tutorial: ', '')} · {x['source_repo_url']}",
        )
        mr_query = st.text_area("질문", value="이 저장소들의 공통 아키텍처 패턴과 차이는?", height=90)

        cc1, cc2, cc3 = st.columns([1, 1, 1])
        mr_top_k = cc1.slider("총 결과 수", 3, 20, 8, key="mr_top_k")
        mr_per_repo = cc2.slider("저장소당 후보", 2, 10, 5, key="mr_per_repo")
        with cc3:
            st.write("")
            mr_go = st.button("교차 검색 + 합성 답변", type="primary", key="mr_go")

        if len(picked) < 2:
            st.caption("💡 저장소를 2개 이상 선택하면 '저장소 간 연결' 인사이트가 유의미해집니다. (1개도 검색은 가능)")

        if mr_go:
            ids = [t["id"] for t in picked]
            if not ids:
                st.warning("저장소를 1개 이상 선택하세요.")
            else:
                with st.spinner("여러 저장소를 교차 검색 중..."):
                    mr_results = search_across_tutorials(ids, mr_query, top_k=mr_top_k, per_repo_top=mr_per_repo)
                st.session_state["mrag"] = {"results": mr_results, "q": mr_query}

        mr_data = st.session_state.get("mrag")
        if mr_data and mr_data["results"]:
            results = mr_data["results"]
            per_repo_count = {}
            for r in results:
                per_repo_count[r["tutorial_title"]] = per_repo_count.get(r["tutorial_title"], 0) + 1

            render_chips([
                status_chip(title.replace("Tutorial: ", ""), f"{n} hit")
                for title, n in per_repo_count.items()
            ])

            st.markdown("#### 검색 결과 (저장소별)")
            for title, n in per_repo_count.items():
                with st.expander(f"📘 {title.replace('Tutorial: ', '')} · {n}건"):
                    for r in [x for x in results if x["tutorial_title"] == title]:
                        st.caption(f"norm={r['norm_score']} · raw={r['score']} · Ch{r['chapter_no']} {r['chapter_title']}")
                        st.write((r["content"] or "")[:700])

            st.markdown("#### 합성 답변")
            mr_prompt = build_multi_repo_rag_prompt(mr_data["q"], results, language="Korean")
            try:
                from utils.call_llm import call_llm_stream

                st.write_stream(call_llm_stream(mr_prompt))
            except Exception as e:
                st.error("답변 생성 실패")
                st.exception(e)
        elif mr_data:
            st.warning("검색 결과가 없습니다. 질문 키워드를 바꾸거나 저장소를 더 선택해 보세요.")


# -----------------------------
# Fine-tune tab
# -----------------------------

with tab_finetune:
    section_header(
        "파인튜닝 (코드 생성)",
        "저장한 여러 레포의 실제 코드 블록으로 소형 LoRA 학습 데이터셋을 만들고, 로컬 GPU(4GB)에서 학습합니다.",
        icon="🛠",
    )

    if not tutorials:
        empty_state("저장된 저장소가 없습니다", "➕ Generate에서 코드가 풍부한 저장소를 저장하세요.", icon="🛠")
    else:
        ft_picked = st.multiselect(
            "데이터셋에 포함할 저장소",
            options=tutorials,
            default=tutorials,
            format_func=lambda x: f"{x['title'].replace('Tutorial: ', '')} · {x['source_repo_url']}",
            key="ft_pick",
        )
        fc1, fc2 = st.columns(2)
        ft_min_lines = fc1.slider("코드 블록 최소 줄 수", 2, 15, 3, key="ft_min")
        ft_max_per = fc2.number_input("저장소당 최대 예시 (0=제한 없음)", min_value=0, max_value=5000, value=0, step=50, key="ft_max")

        if st.button("코드젠 데이터셋 생성 (JSONL)", type="primary", key="ft_build"):
            ids = [t["id"] for t in ft_picked]
            if not ids:
                st.warning("저장소를 1개 이상 선택하세요.")
            else:
                with st.spinner("코드 블록 추출 및 JSONL 생성 중..."):
                    stats = export_codegen_jsonl(
                        ids,
                        out_path="exports/codegen_dataset.jsonl",
                        min_code_lines=ft_min_lines,
                        max_per_tutorial=(ft_max_per or None),
                    )
                st.session_state["codegen_stats"] = stats

        stats = st.session_state.get("codegen_stats")
        if stats:
            mm1, mm2 = st.columns(2)
            mm1.metric("생성된 예시", stats["examples"])
            mm2.metric("저장소 수", stats["tutorials"])
            if stats["by_language"]:
                render_chips([status_chip(lang or "code", str(n)) for lang, n in stats["by_language"].items()])
            st.success(f"데이터셋 저장됨: `{stats['path']}`")

            if stats["examples"] < 20:
                st.warning(
                    "예시가 적습니다. 코드가 풍부한 소스 폴더를 저장하면(예: `/tree/main/src`) "
                    "더 좋은 데이터셋이 됩니다. HTML/YAML 위주면 문서 저장소일 수 있습니다."
                )

            st.markdown("#### 1) 학습 (RTX 3050Ti · 4GB VRAM — 4-bit QLoRA)")
            st.code(
                "pip install -r requirements-train.txt\n"
                f"python train_lora_local.py --train_jsonl {stats['path']} \\\n"
                "  --load_4bit --model_name Qwen/Qwen2.5-0.5B-Instruct \\\n"
                "  --out_dir finetuned_adapters/codegen_qwen05 \\\n"
                "  --max_length 640 --epochs 2 --batch_size 1 --grad_accum 16",
                language="bash",
            )
            st.markdown("#### 2) 학습된 어댑터로 코드 생성")
            st.code(
                'python infer_codegen_local.py \\\n'
                '  --adapter_dir finetuned_adapters/codegen_qwen05/adapter \\\n'
                '  --prompt "FastAPI 의존성 주입을 사용하는 엔드포인트 예시를 작성해줘"',
                language="bash",
            )
            st.caption(
                "4GB 가이드: 4-bit(QLoRA) + Qwen2.5-0.5B + max_length≤640 + grad checkpointing 권장. "
                "1.5B는 학습 시 OOM 위험이 큽니다. 함수/구조 등 작은 단위 위주로 데이터를 구성하세요."
            )
        else:
            st.info("저장소를 선택하고 **코드젠 데이터셋 생성**을 누르면, 학습용 JSONL과 4GB용 학습/추론 명령이 표시됩니다.")

    # --- Run a locally trained adapter to generate code (subprocess) ---
    st.divider()
    st.markdown("#### 🤖 학습된 어댑터로 코드 생성")
    ad_col1, ad_col2 = st.columns(2)
    ft_adapter = ad_col1.text_input("어댑터 경로", value="finetuned_adapters/codegen_qwen05/adapter", key="ft_adapter")
    ft_base = ad_col2.text_input("Base 모델", value="Qwen/Qwen2.5-0.5B-Instruct", key="ft_base")
    ft_prompt = st.text_area("코드 생성 프롬프트", value="requests로 세션과 재시도 어댑터를 설정해 안정적으로 GET 요청하는 코드를 작성해줘", height=80, key="ft_infer_prompt")
    ft_4bit = st.checkbox("4-bit 로드 (--load_4bit, 4GB 권장)", value=True, key="ft_4bit")

    if st.button("코드 생성 실행", type="primary", key="ft_infer_run"):
        if not Path(ft_adapter).exists():
            st.error(f"어댑터 경로가 없습니다: {ft_adapter} — 먼저 위 명령으로 학습하세요.")
        else:
            cmd = [
                sys.executable, "infer_codegen_local.py",
                "--adapter_dir", ft_adapter, "--base_model", ft_base,
                "--prompt", ft_prompt, "--max_new_tokens", "256",
            ]
            if ft_4bit:
                cmd.append("--load_4bit")
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"
            try:
                with st.spinner("모델 로드 + 코드 생성 중... (최초/4-bit는 수십 초 소요)"):
                    proc = subprocess.run(
                        cmd, env=child_env, text=True, encoding="utf-8",
                        errors="replace", capture_output=True, timeout=600,
                    )
                out = proc.stdout or ""
                if proc.returncode != 0:
                    st.error("코드 생성 실패")
                    st.code((out[-2000:] + "\n" + (proc.stderr or "")[-1500:]), language="text")
                else:
                    # Show the assistant portion (after the last 'assistant' marker if present)
                    shown = out.split("assistant", 1)[-1].strip() if "assistant" in out else out.strip()
                    st.code(shown[-4000:] or "(빈 출력)", language="python")
            except subprocess.TimeoutExpired:
                st.error("시간 초과(600s). 더 작은 max_new_tokens로 다시 시도하세요.")


# -----------------------------
# Agent (Agentic RAG) tab
# -----------------------------

_AGENT_TOOL_ICON = {
    "search": "🔎", "list_chapters": "📖", "read_chapter": "📄",
    "ontology": "🕸", "finish": "✅",
}

with tab_agent:
    section_header(
        "에이전트 RAG",
        "질문을 주면 에이전트가 스스로 도구(검색·챕터 읽기·온톨로지)를 골라 반복 탐색한 뒤 근거 기반으로 답합니다.",
        icon="🤖",
    )

    if not tutorials:
        empty_state("저장된 저장소가 없습니다", "➕ Generate에서 저장소를 저장하면 에이전트가 탐색할 수 있습니다.", icon="🤖")
    else:
        ag_picked = st.multiselect(
            "대상 저장소 (여러 개 선택 시 교차 탐색)",
            options=tutorials,
            default=tutorials[: min(2, len(tutorials))],
            format_func=lambda x: f"{x['title'].replace('Tutorial: ', '')} · {x['source_repo_url']}",
            key="ag_pick",
        )
        ag_q = st.text_area(
            "질문",
            value="이 저장소들의 핵심 구성요소와 관계, 그리고 서로 어떻게 연결될 수 있는지 설명해줘.",
            height=90,
            key="ag_q",
        )
        ag_mode = st.radio(
            "탐색 모드",
            ["에이전트 (ReAct)", "Deep Research (재귀 리서치)"],
            horizontal=True,
            key="ag_mode",
        )
        agc1, agc2, agc3 = st.columns([1, 1, 1])
        ag_steps = agc1.slider("최대 스텝 / 하위질문 수", 2, 10, 5, key="ag_steps")
        ag_judge = agc2.checkbox("답변 검증·개선 (Judge)", value=True, key="ag_judge")
        with agc3:
            st.write("")
            ag_go = st.button("🤖 실행", type="primary", key="ag_go")

        st.caption(
            "ReAct: 도구를 선택→실행→관측하며 잘못된 액션은 스스로 교정. "
            "Deep Research: 질문을 하위질문으로 분해해 각각 검색 후 종합. "
            "Judge: 답변을 채점하고 근거가 부족하면 한 번 자동 개선."
        )

        if ag_go:
            ids = [t["id"] for t in ag_picked]
            if not ids:
                st.warning("저장소를 1개 이상 선택하세요.")
            else:
                is_deep = ag_mode.startswith("Deep")
                spin = "하위질문으로 분해해 심층 리서치 중..." if is_deep else "에이전트가 도구로 탐색 중..."
                with st.spinner(spin):
                    if is_deep:
                        dr = deep_research(ids, ag_q, max_subq=ag_steps, per_subq_top=5)
                        res = {
                            "trace": [{"step": i + 1, "subq": s} for i, s in enumerate(dr["subquestions"])],
                            "transcript": dr["transcript"],
                            "steps": len(dr["subquestions"]),
                            "finished": True,
                            "mode": "deep",
                        }
                    else:
                        res = agent_gather(ids, ag_q, max_steps=ag_steps)
                        res["mode"] = "agent"
                ag_lang = (ag_picked[0].get("language") if ag_picked else "Korean") or "Korean"
                try:
                    trace_path = tracing.save_trace(
                        res.get("mode", "agent"), ag_q, res["trace"],
                        meta={"repos": [t["title"] for t in ag_picked],
                              "steps": res["steps"], "finished": res["finished"]},
                    )
                except Exception:
                    trace_path = None
                st.session_state["agent_run"] = {
                    "res": res, "q": ag_q, "lang": ag_lang,
                    "judge": ag_judge, "trace_path": trace_path,
                }

        agent_data = st.session_state.get("agent_run")
        if agent_data:
            res = agent_data["res"]
            mode = res.get("mode", "agent")
            render_chips([
                status_chip("모드", "Deep Research" if mode == "deep" else "ReAct 에이전트"),
                status_chip("스텝" if mode == "agent" else "하위질문", str(res["steps"])),
                status_chip("종료", "완료" if res["finished"] else "예산 소진", ok=res["finished"]),
            ])
            if agent_data.get("trace_path"):
                st.caption(f"🧾 트레이스 저장됨: `{agent_data['trace_path']}`")

            st.markdown("#### 🧭 트레이스")
            if mode == "deep":
                for s in res["trace"]:
                    st.markdown(f"- **하위질문 {s['step']}**: {s['subq']}")
            else:
                for s in res["trace"]:
                    if s.get("error"):
                        st.warning(f"Step {s['step']}: 유효하지 않은 액션 → 자기 교정")
                        continue
                    tool = s.get("tool")
                    icon = _AGENT_TOOL_ICON.get(tool, "🔧")
                    arg_str = ", ".join(f"{k}={v}" for k, v in (s.get("args") or {}).items())
                    with st.expander(f"{icon} Step {s['step']} · {tool}({arg_str})"):
                        if s.get("thought"):
                            st.caption(f"💭 {s['thought']}")
                        if s.get("observation"):
                            st.code((s["observation"])[:1200], language="text")

            st.markdown("#### 🧠 최종 답변")
            transcript = res["transcript"]
            if not transcript.strip():
                st.warning("근거를 수집하지 못했습니다. 질문을 바꾸거나 스텝/하위질문 수를 늘려보세요.")
            else:
                lang = agent_data.get("lang", "Korean")
                builder = build_deep_research_prompt if mode == "deep" else build_agent_answer_prompt
                ans_prompt = builder(agent_data["q"], transcript, language=lang)
                if agent_data.get("judge"):
                    from utils.call_llm import call_llm

                    with st.spinner("답변 생성 및 검증(Judge) 중..."):
                        answer = call_llm(ans_prompt, use_cache=False)
                        verdict = judge_answer(agent_data["q"], transcript, answer)
                        refined = False
                        if not verdict["grounded"] or verdict["score"] < 4:
                            answer = refine_answer(agent_data["q"], transcript, answer, verdict["issues"], language=lang)
                            refined = True
                            verdict = judge_answer(agent_data["q"], transcript, answer)
                    render_chips([
                        status_chip("Judge 점수", f"{verdict['score']}/5", ok=verdict["score"] >= 4),
                        status_chip("근거 충실", "예" if verdict["grounded"] else "아니오", ok=verdict["grounded"]),
                        status_chip("개선", "적용됨" if refined else "불필요"),
                    ])
                    if verdict.get("issues"):
                        st.caption(f"🔎 Judge 코멘트: {verdict['issues']}")
                    st.markdown(answer)
                else:
                    try:
                        from utils.call_llm import call_llm_stream

                        st.write_stream(call_llm_stream(ans_prompt))
                    except Exception as e:
                        st.error("답변 생성 실패")
                        st.exception(e)

        with st.expander("🧾 실행 추적 기록 (Tracing)"):
            recent = tracing.list_traces(15)
            if not recent:
                st.caption("아직 저장된 트레이스가 없습니다.")
            else:
                for tr in recent:
                    st.markdown(f"- `{tr['created_at']}` · **{tr['kind']}** · {(tr['question'] or '')[:70]}")
