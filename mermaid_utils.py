"""
Self-healing Mermaid.

Stored Mermaid graphs are LLM-generated and occasionally contain syntax the
client-side renderer rejects (unquoted labels with special chars, stray
characters, bad arrows). heal_mermaid() asks the LLM to repair the diagram
and returns only valid Mermaid, preserving node/edge structure.
"""

import re

from utils.call_llm import call_llm


def extract_mermaid(text: str) -> str:
    """Pull the Mermaid body out of an LLM response (handles ``` fences)."""
    if not text:
        return ""
    m = re.search(r"```mermaid\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def heal_mermaid(mermaid: str) -> dict:
    """Repair a possibly-broken Mermaid diagram via the LLM.

    Returns {mermaid, changed, note}. On any failure the original is kept
    (changed=False) so the caller can safely fall back.
    """
    mermaid = (mermaid or "").strip()
    if not mermaid:
        return {"mermaid": mermaid, "changed": False, "note": "빈 다이어그램"}

    prompt = f"""다음 Mermaid 다이어그램에 문법 오류가 있으면 고쳐서 **유효한 Mermaid 코드만** 출력하라.
규칙:
- 노드/엣지 구조와 라벨의 의미는 최대한 보존한다.
- 특수문자(괄호, 콜론, 따옴표 등)가 든 라벨은 큰따옴표로 감싼다.
- 설명 문장이나 코드펜스 없이 Mermaid 본문만 출력한다(첫 줄은 flowchart/graph 등).

[Mermaid]
{mermaid}"""

    try:
        fixed = extract_mermaid(call_llm(prompt, use_cache=False))
    except Exception as e:  # never break the UI on a heal attempt
        return {"mermaid": mermaid, "changed": False, "note": f"복구 실패: {type(e).__name__}"}

    fixed = (fixed or "").strip()
    if not fixed:
        return {"mermaid": mermaid, "changed": False, "note": "복구 결과가 비어 원본 유지"}

    changed = fixed != mermaid
    return {"mermaid": fixed, "changed": changed, "note": "복구 적용됨" if changed else "변경 없음(이미 유효)"}
