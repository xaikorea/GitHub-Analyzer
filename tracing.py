"""
Execution tracing (observability).

Persists agent / deep-research run traces so they can be reviewed later.
Prefers the Postgres `agent_traces` table when DATABASE_URL is set; otherwise
falls back to JSON files under exports/traces/ (gitignored). Both paths expose
the same shape so the UI doesn't care which is used.

Trace refs:
- DB:   "db:<uuid>"
- File: the file path
"""

import json
import os
import time
from pathlib import Path

TRACE_DIR = Path("exports/traces")


def _db_enabled() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def save_trace(kind: str, question: str, trace: list, meta: dict | None = None) -> str:
    """Persist one run's trace. Returns a ref (db:<id> or a file path)."""
    if _db_enabled():
        try:
            from db_store import save_trace_db

            return "db:" + save_trace_db(kind, question, trace, meta)
        except Exception:
            pass  # fall back to file

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_kind = "".join(c if c.isalnum() else "_" for c in (kind or "run"))
    path = TRACE_DIR / f"{safe_kind}_{ts}.json"
    path.write_text(
        json.dumps(
            {
                "kind": kind,
                "question": question,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "meta": meta or {},
                "trace": trace,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(path)


def list_traces(limit: int = 20) -> list[dict]:
    """Most-recent first: [{ref, kind, question, created_at}]."""
    if _db_enabled():
        try:
            from db_store import list_traces_db

            return [
                {"ref": "db:" + r["id"], "kind": r["kind"],
                 "question": r["question"], "created_at": r["created_at"]}
                for r in list_traces_db(limit)
            ]
        except Exception:
            pass

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
            "ref": str(p), "kind": d.get("kind", "?"),
            "question": d.get("question", ""), "created_at": d.get("created_at", ""),
        })
    return out


def load_trace(ref: str) -> dict:
    if ref and ref.startswith("db:"):
        from db_store import get_trace_db

        return get_trace_db(ref[3:]) or {}
    return json.loads(Path(ref).read_text(encoding="utf-8"))
