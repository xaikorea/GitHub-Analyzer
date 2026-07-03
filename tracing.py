"""
Lightweight execution tracing.

Persists agent / deep-research run traces to JSON files so they can be
reviewed later (observability), without needing a DB migration. Traces live
under exports/traces/ (gitignored).
"""

import json
import time
from pathlib import Path

TRACE_DIR = Path("exports/traces")


def save_trace(kind: str, question: str, trace: list, meta: dict | None = None) -> str:
    """Persist one run's trace. `kind` is e.g. 'agent' or 'deep_research'."""
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_kind = "".join(c if c.isalnum() else "_" for c in (kind or "run"))
    path = TRACE_DIR / f"{safe_kind}_{ts}.json"
    payload = {
        "kind": kind,
        "question": question,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": meta or {},
        "trace": trace,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def list_traces(limit: int = 20) -> list[dict]:
    """Most-recent traces first: [{path, name, kind, question, created_at}]."""
    if not TRACE_DIR.exists():
        return []
    files = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "path": str(p),
            "name": p.name,
            "kind": d.get("kind", "?"),
            "question": d.get("question", ""),
            "created_at": d.get("created_at", ""),
        })
    return out


def load_trace(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
