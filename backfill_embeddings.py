"""
Backfill OpenAI embeddings for chunks that were saved without them.

Existing tutorials were stored with create_embeddings=False, so their
chunks.embedding column is NULL and RAG search falls back to keyword
scoring. Run this once to populate embeddings; semantic (vector) search
then activates automatically in db_store.search_tutorial_context_v4 —
no code change needed.

Safety / behavior:
- Idempotent & resumable: only fills rows where embedding IS NULL, so
  re-running continues where it left off and commits per batch.
- Batches inputs into a single OpenAI request to cut cost/latency.
- Respects OPENAI_BASE_URL (proxy/Azure-compatible).
- --dry-run makes NO API calls and NO writes; it just reports how many
  chunks need embeddings and a rough cost estimate.

Examples:
  python backfill_embeddings.py --dry-run
  python backfill_embeddings.py                       # all tutorials
  python backfill_embeddings.py --tutorial-id <uuid>  # one tutorial
  python backfill_embeddings.py --embed-batch 64 --page 500

Requires: OPENAI_API_KEY, DATABASE_URL. The embedding model must match the
DB column dimension (schema.sql = vector(1536) = text-embedding-3-small).
"""

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

from db_store import get_conn, vector_literal

load_dotenv(override=True)

# vector(1536) in schema.sql. Models producing other dims will fail on insert.
EXPECTED_DIM = 1536
MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def embed_batch(texts, model, base_url, api_key):
    """Embed a list of texts in one request. Returns list of vectors in order."""
    resp = requests.post(
        f"{base_url}/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "input": [t[:8000] for t in texts]},
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI embedding error: {resp.status_code} {resp.text}")
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def count_pending(tutorial_id):
    where = "embedding is null"
    params = []
    if tutorial_id:
        where += " and tutorial_id = %s"
        params.append(tutorial_id)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"select count(*), coalesce(sum(length(content)),0) from chunks where {where}", params)
        return cur.fetchone()


def fetch_page(cur, tutorial_id, limit):
    where = "embedding is null"
    params = []
    if tutorial_id:
        where += " and tutorial_id = %s"
        params.append(tutorial_id)
    params.append(limit)
    cur.execute(
        f"select id::text, content from chunks where {where} order by created_at asc limit %s",
        params,
    )
    return cur.fetchall()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tutorial-id", help="Only backfill this tutorial (default: all).")
    ap.add_argument("--model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    ap.add_argument("--page", type=int, default=500, help="Rows fetched from DB per loop.")
    ap.add_argument("--embed-batch", type=int, default=64, help="Texts per OpenAI request.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds between API calls.")
    ap.add_argument("--dry-run", action="store_true", help="Report counts/cost only; no API calls, no writes.")
    args = ap.parse_args()

    n, total_chars = count_pending(args.tutorial_id)
    approx_tokens = total_chars / 4  # rough heuristic
    # text-embedding-3-small pricing ~$0.02 / 1M tokens (adjust for other models)
    est_cost = approx_tokens / 1_000_000 * 0.02
    print(f"chunks needing embeddings: {n}")
    print(f"total content chars: {total_chars}  (~{approx_tokens:,.0f} tokens)")
    print(f"model: {args.model}  | rough cost estimate @ $0.02/1M tok: ${est_cost:,.4f}")

    if args.dry_run:
        print("dry-run: no API calls made, no rows written.")
        return

    if n == 0:
        print("Nothing to do.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is required.", file=sys.stderr)
        sys.exit(1)

    dim = MODEL_DIMS.get(args.model)
    if dim and dim != EXPECTED_DIM:
        print(
            f"ERROR: model {args.model} produces {dim}-dim vectors but the DB "
            f"column is vector({EXPECTED_DIM}). Use a {EXPECTED_DIM}-dim model "
            f"or migrate the column.",
            file=sys.stderr,
        )
        sys.exit(1)

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
    done = 0

    while True:
        with get_conn() as conn, conn.cursor() as cur:
            rows = fetch_page(cur, args.tutorial_id, args.page)
            if not rows:
                break

            for i in range(0, len(rows), args.embed_batch):
                batch = rows[i : i + args.embed_batch]
                vectors = embed_batch([c for _, c in batch], args.model, base_url, api_key)

                for (chunk_id, _), vec in zip(batch, vectors):
                    cur.execute(
                        "update chunks set embedding = %s::vector, embedding_model = %s where id = %s",
                        (vector_literal(vec), args.model, chunk_id),
                    )
                conn.commit()  # persist progress per API batch (resumable)
                done += len(batch)
                print(f"  embedded {done}/{n}", flush=True)
                time.sleep(args.sleep)

    print(f"Done. Backfilled {done} chunk embeddings with {args.model}.")


if __name__ == "__main__":
    main()
