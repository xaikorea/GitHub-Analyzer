"""
Agentic RAG harness.

Instead of the fixed "retrieve -> prompt -> answer" pipeline, this runs a
ReAct-style agent loop over the stored codebase knowledge: the LLM chooses a
tool, we execute it against the DB, feed back the observation, and repeat
until the agent is confident (or a step budget is hit).

Harness-engineering features:
- A small tool registry grounded in db_store (search / list_chapters /
  read_chapter / ontology / finish), one across single OR multiple repos.
- Strict JSON action protocol with robust parsing + self-correction: an
  invalid action is fed back as an observation instead of crashing.
- Guardrails: max step budget, observation truncation, unknown-tool and
  bad-arg handling, and a forced-synthesis fallback if the budget runs out.
- A full trace (thought / tool / args / observation per step) for
  observability in the UI.

The loop only *gathers* evidence; final prose is produced by
build_agent_answer_prompt() so the UI can stream it.
"""

import json
import re

from db_store import (
    get_tutorial,
    get_chapters,
    search_tutorial_context_v4,
    search_across_tutorials,
    get_ontology_context_v4,
)
from utils.call_llm import call_llm

MAX_OBS_CHARS = 1400
DEFAULT_MAX_STEPS = 6

TOOL_SPECS = [
    {"name": "search", "args": "query: str, top_k: int=5",
     "desc": "질문과 관련된 코드/설명 청크를 (선택한 모든 저장소에서) 검색"},
    {"name": "list_chapters", "args": "(없음)",
     "desc": "선택한 저장소들의 챕터 목록(저장소·번호·제목)을 반환"},
    {"name": "read_chapter", "args": "chapter_no: int, repo_hint: str=''",
     "desc": "특정 챕터 본문을 읽는다(여러 저장소면 repo_hint로 구분)"},
    {"name": "ontology", "args": "query: str",
     "desc": "개념(노드)과 관계(엣지) 그래프에서 관련 관계를 조회"},
    {"name": "finish", "args": "(없음)",
     "desc": "충분한 근거를 모았을 때 탐색을 종료(최종 답변은 이후 생성)"},
]


# -----------------------------
# Context
# -----------------------------

def build_context(tutorial_ids: list[str]) -> dict:
    tutorials, chapters = {}, {}
    for tid in tutorial_ids:
        tutorials[tid] = get_tutorial(tid) or {}
        chapters[tid] = get_chapters(tid)
    return {"ids": tutorial_ids, "tutorials": tutorials, "chapters": chapters}


def _repo_label(ctx: dict, tid: str) -> str:
    return (ctx["tutorials"].get(tid, {}).get("title") or tid).replace("Tutorial: ", "")


# -----------------------------
# Tools
# -----------------------------

def _tool_search(ctx, query, top_k=5):
    ids = ctx["ids"]
    top_k = max(1, min(int(top_k or 5), 10))
    hits = (
        search_across_tutorials(ids, query, top_k=top_k)
        if len(ids) > 1
        else search_tutorial_context_v4(ids[0], query, top_k=top_k)
    )
    if not hits:
        return "검색 결과 없음."
    lines = []
    for h in hits:
        repo = h.get("tutorial_title", "").replace("Tutorial: ", "") or _repo_label(ctx, ids[0])
        lines.append(
            f"- [{repo} · Ch{h.get('chapter_no')} {h.get('chapter_title')}] "
            f"{(h.get('content') or '')[:280]}"
        )
    return "\n".join(lines)


def _tool_list_chapters(ctx):
    lines = []
    for tid in ctx["ids"]:
        repo = _repo_label(ctx, tid)
        for ch in ctx["chapters"][tid]:
            lines.append(f"- [{repo}] {ch['chapter_no']}. {ch['title']}")
    return "\n".join(lines) if lines else "챕터 없음."


def _tool_read_chapter(ctx, chapter_no, repo_hint=""):
    try:
        chapter_no = int(chapter_no)
    except (TypeError, ValueError):
        return "chapter_no는 정수여야 합니다."
    hint = (repo_hint or "").lower()
    for tid in ctx["ids"]:
        repo = _repo_label(ctx, tid)
        if hint and hint not in repo.lower():
            continue
        for ch in ctx["chapters"][tid]:
            if ch["chapter_no"] == chapter_no:
                md = ch.get("markdown") or ""
                return f"[{repo} · Ch{chapter_no} {ch['title']}]\n{md}"
    return f"chapter_no={chapter_no} 를 찾지 못했습니다. list_chapters로 목록을 확인하세요."


def _tool_ontology(ctx, query):
    out = []
    for tid in ctx["ids"]:
        triples = get_ontology_context_v4(tid, query=query, limit=12)
        for t in triples[:12]:
            out.append(f"- {t['source']} --{t['relation']}--> {t['target']} ({_repo_label(ctx, tid)})")
    return "\n".join(out) if out else "관련 온톨로지 관계 없음."


def _dispatch(tool: str, args: dict, ctx: dict) -> str:
    args = args or {}
    try:
        if tool == "search":
            return _tool_search(ctx, args.get("query", ""), args.get("top_k", 5))
        if tool == "list_chapters":
            return _tool_list_chapters(ctx)
        if tool == "read_chapter":
            return _tool_read_chapter(ctx, args.get("chapter_no"), args.get("repo_hint", ""))
        if tool == "ontology":
            return _tool_ontology(ctx, args.get("query", ""))
        return f"알 수 없는 도구 '{tool}'. 사용 가능한 도구만 쓰세요."
    except Exception as e:  # tool failure must not kill the loop
        return f"도구 실행 오류: {type(e).__name__}: {e}"


# -----------------------------
# Action parsing (robust)
# -----------------------------

def _parse_action(raw: str):
    """Extract a JSON action object from an LLM response. Returns dict or None."""
    if not raw:
        return None
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        text = m.group(1)
    else:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        end = None
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return None
        text = text[start:end]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    return obj


# -----------------------------
# Prompts
# -----------------------------

def _system_prompt(ctx: dict, question: str) -> str:
    repos = ", ".join(_repo_label(ctx, tid) for tid in ctx["ids"]) or "(없음)"
    tools = "\n".join(f"- {t['name']}({t['args']}): {t['desc']}" for t in TOOL_SPECS)
    return f"""
너는 저장된 코드베이스 지식을 도구로 탐색하는 리서치 에이전트다.
대상 저장소: {repos}

사용 가능한 도구:
{tools}

규칙:
- 매 턴 반드시 아래 JSON 하나만 출력한다(다른 텍스트 금지):
  {{"thought": "무엇을 왜 할지", "tool": "도구명", "args": {{...}}}}
- 근거가 부족하면 search/read_chapter/ontology로 더 모은다.
- 같은 검색을 반복하지 말고, 관측을 바탕으로 다음 행동을 정한다.
- 충분히 모았으면 {{"thought":"...","tool":"finish","args":{{}}}} 로 종료한다.

[사용자 질문]
{question}
""".strip()


def build_agent_answer_prompt(question: str, transcript: str, language: str = "Korean") -> str:
    language = (language or "Korean").strip() or "Korean"
    return f"""
너는 코드베이스를 {language} 언어로 설명하는 시니어 엔지니어다.
아래는 에이전트가 도구로 수집한 근거(관측)다. 이 근거만 사용해 답하라.

규칙:
- 근거에 없는 사실은 추측하지 말고 부족하다고 말하라.
- 어떤 저장소·챕터를 근거로 삼았는지 밝혀라.
- 반드시 {language} 언어로 작성하라.

[사용자 질문]
{question}

[수집된 근거]
{transcript if transcript.strip() else "(수집된 근거 없음)"}

[답변 형식]
1. 핵심 답변
2. 근거(저장소 · 챕터)
3. (여러 저장소면) 저장소 간 연결/비교
""".strip()


# -----------------------------
# Agent loop
# -----------------------------

def agent_gather(tutorial_ids: list[str], question: str, max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """Run the ReAct tool loop. Returns {trace, transcript, steps, finished}."""
    if not tutorial_ids or not (question or "").strip():
        return {"trace": [], "transcript": "", "steps": 0, "finished": False}

    ctx = build_context(tutorial_ids)
    system = _system_prompt(ctx, question)
    trace, transcript = [], ""
    parse_fails = 0

    step = 0
    while step < max_steps:
        step += 1
        prompt = f"{system}\n\n[지금까지의 관측]\n{transcript or '(없음)'}\n\n다음 행동을 JSON으로만 출력:"
        raw = call_llm(prompt, use_cache=False)
        action = _parse_action(raw)

        if action is None:
            parse_fails += 1
            trace.append({"step": step, "error": "invalid_json", "raw": (raw or "")[:200]})
            transcript += "\n[System] 직전 응답이 유효한 JSON이 아니었습니다. 반드시 JSON 객체만 출력하세요."
            if parse_fails >= 3:
                break
            continue

        thought = str(action.get("thought", ""))
        tool = action.get("tool")
        args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}

        if tool == "finish":
            trace.append({"step": step, "thought": thought, "tool": "finish", "args": {}, "observation": "탐색 종료"})
            return {"trace": trace, "transcript": transcript, "steps": step, "finished": True}

        observation = _dispatch(tool, args, ctx)[:MAX_OBS_CHARS]
        trace.append({"step": step, "thought": thought, "tool": tool, "args": args, "observation": observation})
        transcript += (
            f"\n\n[Step {step}] thought: {thought}\n"
            f"action: {tool}({json.dumps(args, ensure_ascii=False)})\n"
            f"observation: {observation}"
        )

    return {"trace": trace, "transcript": transcript, "steps": step, "finished": False}
