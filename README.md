# P&ID Knowledge Graph Pipeline

This project builds a knowledge graph from a P&ID PDF and supports:

- multimodal entity/relation extraction with iterative zoom requests
- Neo4j graph ingestion (via `bolt://...`)
- text entity index generation (JSONL with graph context)
- lexical entity search
- LLM-assisted question answering through a `search_knowledge_graph` Cypher tool
- SOP discrepancy checks with JSON log streaming

## Spec Paths

The pipeline enforces:

- `data/pid/diagram.pdf`
- `data/sop/sop.docx`

If legacy files exist (`data/diagram_page1.pdf`, `data/sop.docx`), they are copied into spec paths automatically.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Expected env vars:

- `OPENAI_API_KEY`
- `NEO4J_URI` (use `bolt://...`)
- `NEO4J_USER`
- `DB_PASSWORD` (or `NEO4J_PASSWORD`)
- `NEO4J_DATABASE` (optional, defaults to `neo4j`)
- `DB_NAME` (optional instance label only; not used as Neo4j database name)

Optional:

- `OPENAI_VISION_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_QUERY_MODEL` (default: `gpt-4.1-mini`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-large`)
- `OUTPUT_DIR` (default: `artifacts`)
- `MAX_ZOOM_LOOPS` (default: `5`)
- `MAX_VIEWS_PER_PAGE` (default: `4`)
- `NEO4J_ALLOW_DB_FALLBACK` (default: `true`)
- `NEO4J_FALLBACK_DATABASE` (default: `neo4j`)
- `EMBEDDING_CHUNK_SENTENCES` (default: `5`)
- `EMBEDDING_CHUNK_OVERLAP` (default: `2`)
- `EMBEDDING_TOP_K_CHUNKS` (default: `24`)
- `EMBEDDING_TOP_K_ENTITIES` (default: `8`)
- `LEXICAL_TOP_K_ENTITIES` (default: `8`)

## Python Usage

```python
from kg_pipeline import Settings, build_knowledge_graph, answer_with_knowledge_graph, detect_sop_discrepancies

settings = Settings.from_env(".env")

build_summary = build_knowledge_graph(settings=settings, doc_id="pid_demo")
print(build_summary)

qa = answer_with_knowledge_graph(
    "What is the PSV set pressure on F-715A?",
    settings=settings,
    doc_id="pid_demo",
)
print(qa["answer"])

report = detect_sop_discrepancies(settings=settings, doc_id="pid_demo", stream_logs=True)
print(report["discrepancy_count"])
```

Write a readable dashboard text file:

```python
from kg_pipeline import Settings, detect_sop_discrepancies, write_discrepancy_dashboard

settings = Settings.from_env(".env")
report = detect_sop_discrepancies(settings=settings, doc_id="pid_demo", stream_logs=False)
path = write_discrepancy_dashboard(report, "artifacts/discrepancy_dashboard.txt")
print(path)
```

Or in one call:

```python
report = detect_sop_discrepancies(
    settings=settings,
    doc_id="pid_demo",
    stream_logs=True,
    dashboard_path="artifacts/discrepancy_dashboard.txt",
)
print(report.get("dashboard_path"))
```

## Chatbot CLI

Run an interactive terminal chat:

```bash
PYTHONPATH=src python3 src/main.py --doc-id pid_demo
```

Optional flags:

- `--env-file .env`
- `--entity-index-path artifacts/entity_index.jsonl`
- `--max-tool-calls 5`
- `--trace` (prints matched entities and Cypher tool events)
- `--no-tool-logs` (disables live Cypher tool-call logs)
- `--no-lexical-logs` (disables lexical retrieval logs)
- `--no-embedding-logs` (disables semantic retrieval logs)

In-chat commands:

- `/help`
- `/reset`
- `/trace`
- `/toollogs`
- `/lexlogs`
- `/emblogs`
- `/exit`

## Web App

Run the new web app stack:

1) Start FastAPI backend:

```bash
PYTHONPATH=src uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

2) Start Next.js frontend:

```bash
cd frontend
npm install
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Frontend tabs:

- `Extract`
- `Query`
- `Find Discrepancies`
- `Graph Overview`

Legacy Streamlit app is still available:

```bash
PYTHONPATH=src streamlit run src/web_app.py
```

The discrepancy workflow can generate a readable dashboard file at:

- `artifacts/discrepancy_dashboard.txt` (default)

### Re-ingest Existing Extractions (No LLM Call)

```python
from kg_pipeline import Settings, ingest_extraction_dump

settings = Settings.from_env(".env")
summary = ingest_extraction_dump(
    settings=settings,
    doc_id="pid_demo",
    extraction_dump_path="artifacts/document_extraction.json",
    # or extraction_pages_dir="artifacts/extractions"
)
print(summary)
```

## Output Artifacts

- `artifacts/rendered_pages/*.png`
- `artifacts/zoom_views/...`
- `artifacts/extractions/page_###.json`
- `artifacts/document_extraction.json`
- `artifacts/entity_index.jsonl`
- `artifacts/entity_vectors.faiss`
- `artifacts/entity_chunks.jsonl`
- `artifacts/entity_vectors.manifest.json`
