# legacy/ — archived Streamlit apps

These are earlier iterations of the Streamlit UI, kept for reference only.
**The canonical app is `app_full_workflow.py` in the repository root.**

| File | What it was | Why archived |
|------|-------------|--------------|
| `app_codebase_tutorial.py` | Generate-only app (runs `main.py`, no DB) | Superseded by the "Generate & Save" tab in `app_full_workflow.py`. Its pattern parsing (`parse_patterns`, comma/newline aware) is actually more robust than the current main app — worth porting. |
| `app_old_sensor_dashboard.py` | Misnamed older copy of the generate app | Duplicate of `app_codebase_tutorial.py`. |
| `app_db_rag.py` | RAG-only viewer | Uses the **old v1/v2** `db_store` functions (`keyword_search_chunks`, `build_rag_prompt`, `get_ontology_context`) which differ from the v4 search used by the main app. |

## Note on running them

They import from the project root (`from db_store import ...`), so if you ever
need to run one, launch it from the repository root, e.g.:

```bash
streamlit run legacy/app_db_rag.py
```

and ensure the repo root is importable (Streamlit adds the script's own
directory to `sys.path`, not the repo root, so you may need
`PYTHONPATH=.`). These are not maintained.
