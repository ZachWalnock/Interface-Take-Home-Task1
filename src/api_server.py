from __future__ import annotations

import io
import json
import queue
import threading
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kg_pipeline import (
    Settings,
    answer_with_knowledge_graph,
    build_knowledge_graph,
    detect_sop_discrepancies,
    ingest_extraction_dump,
)
from kg_pipeline.neo4j_store import Neo4jGraphStore


app = FastAPI(title="P&ID KG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BuildKgRequest(BaseModel):
    env_file: str = ".env"
    doc_id: str = "pid_demo"
    pid_pdf_path: str | None = None
    extraction_dump_path: str | None = None
    entity_index_path: str | None = None
    build_embeddings: bool = True


class IngestKgRequest(BaseModel):
    env_file: str = ".env"
    doc_id: str | None = "pid_demo"
    source_path: str | None = None
    extraction_dump_path: str | None = None
    extraction_pages_dir: str | None = None
    entity_index_path: str | None = None
    build_embeddings: bool = True


class QueryRequest(BaseModel):
    env_file: str = ".env"
    question: str
    doc_id: str | None = "pid_demo"
    entity_index_path: str | None = None
    max_tool_calls: int = Field(default=10, ge=1, le=30)


class DiscrepancyRequest(BaseModel):
    env_file: str = ".env"
    doc_id: str | None = "pid_demo"
    sop_docx_path: str | None = None
    entity_index_path: str | None = None
    max_tool_calls_per_item: int = Field(default=6, ge=1, le=30)
    dashboard_path: str | None = "artifacts/discrepancy_dashboard.txt"


class GraphOverviewRequest(BaseModel):
    env_file: str = ".env"
    doc_id: str | None = "pid_demo"
    max_links: int = Field(default=80, ge=10, le=500)
    include_unknown: bool = False
    max_nodes: int = Field(default=80, ge=10, le=300)
    max_edges: int = Field(default=180, ge=10, le=1000)


def _settings(env_file: str) -> Settings:
    return Settings.from_env(env_file)


def _stream_ndjson(run: Any) -> StreamingResponse:
    out_q: queue.Queue[dict[str, Any] | object] = queue.Queue()
    sentinel = object()

    def emit(payload: dict[str, Any]) -> None:
        out_q.put(payload)

    def worker() -> None:
        try:
            run(emit)
        except Exception as exc:
            emit({"type": "error", "error": str(exc)})
        finally:
            out_q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            item = out_q.get()
            if item is sentinel:
                break
            if not isinstance(item, dict):
                continue
            yield json.dumps(item, ensure_ascii=True) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/extract/build")
def api_build_kg(request: BuildKgRequest) -> dict[str, Any]:
    settings = _settings(request.env_file)
    logs: list[dict[str, Any]] = []

    def log_handler(payload: dict[str, Any]) -> None:
        if isinstance(payload, dict):
            logs.append(payload)

    summary = build_knowledge_graph(
        settings=settings,
        doc_id=request.doc_id,
        pid_pdf_path=request.pid_pdf_path,
        entity_index_path=request.entity_index_path,
        extraction_dump_path=request.extraction_dump_path,
        build_embeddings=request.build_embeddings,
        log_progress=True,
        log_handler=log_handler,
    )
    return {"summary": summary, "logs": logs}


@app.post("/api/extract/build/stream")
def api_build_kg_stream(request: BuildKgRequest) -> StreamingResponse:
    def run(emit) -> None:
        settings = _settings(request.env_file)
        emit({"type": "status", "message": "Starting KG build..."})

        def log_handler(payload: dict[str, Any]) -> None:
            event_name = str(payload.get("event", ""))
            emit({"type": "build_log", "payload": payload})
            if event_name == "tool_call":
                emit({"type": "tool_call", "source": "vision", "payload": payload})

        summary = build_knowledge_graph(
            settings=settings,
            doc_id=request.doc_id,
            pid_pdf_path=request.pid_pdf_path,
            entity_index_path=request.entity_index_path,
            extraction_dump_path=request.extraction_dump_path,
            build_embeddings=request.build_embeddings,
            log_progress=True,
            log_handler=log_handler,
        )
        emit({"type": "result", "data": {"summary": summary}})

    return _stream_ndjson(run)


@app.post("/api/extract/ingest")
def api_ingest_kg(request: IngestKgRequest) -> dict[str, Any]:
    settings = _settings(request.env_file)
    logs: list[dict[str, Any]] = []

    def log_handler(payload: dict[str, Any]) -> None:
        if isinstance(payload, dict):
            logs.append(payload)

    summary = ingest_extraction_dump(
        settings=settings,
        doc_id=request.doc_id,
        source_path=request.source_path,
        extraction_dump_path=request.extraction_dump_path,
        extraction_pages_dir=request.extraction_pages_dir,
        entity_index_path=request.entity_index_path,
        build_embeddings=request.build_embeddings,
        log_progress=True,
        log_handler=log_handler,
    )
    return {"summary": summary, "logs": logs}


@app.post("/api/extract/ingest/stream")
def api_ingest_kg_stream(request: IngestKgRequest) -> StreamingResponse:
    def run(emit) -> None:
        settings = _settings(request.env_file)
        emit({"type": "status", "message": "Starting extraction ingestion..."})

        def log_handler(payload: dict[str, Any]) -> None:
            emit({"type": "build_log", "payload": payload})

        summary = ingest_extraction_dump(
            settings=settings,
            doc_id=request.doc_id,
            source_path=request.source_path,
            extraction_dump_path=request.extraction_dump_path,
            extraction_pages_dir=request.extraction_pages_dir,
            entity_index_path=request.entity_index_path,
            build_embeddings=request.build_embeddings,
            log_progress=True,
            log_handler=log_handler,
        )
        emit({"type": "result", "data": {"summary": summary}})

    return _stream_ndjson(run)


@app.post("/api/query")
def api_query(request: QueryRequest) -> dict[str, Any]:
    settings = _settings(request.env_file)
    tool_events: list[dict[str, Any]] = []
    lexical_events: list[dict[str, Any]] = []
    embedding_events: list[dict[str, Any]] = []

    result = answer_with_knowledge_graph(
        request.question,
        settings=settings,
        entity_index_path=request.entity_index_path,
        doc_id=request.doc_id,
        max_tool_calls=request.max_tool_calls,
        tool_event_logger=lambda payload: tool_events.append(dict(payload)),
        lexical_event_logger=lambda payload: lexical_events.append(dict(payload)),
        embedding_event_logger=lambda payload: embedding_events.append(dict(payload)),
    )

    return {
        **result,
        "tool_events": tool_events or result.get("tool_events", []),
        "lexical_events": lexical_events,
        "embedding_events": embedding_events,
    }


@app.post("/api/query/stream")
def api_query_stream(request: QueryRequest) -> StreamingResponse:
    def run(emit) -> None:
        settings = _settings(request.env_file)
        emit({"type": "status", "message": "Running lexical + embedding retrieval..."})

        def tool_logger(payload: dict[str, Any]) -> None:
            emit({"type": "tool_event", "payload": payload})
            emit({"type": "tool_call", "source": "query", "payload": {"call_index": payload.get("call_index")}})

        def lexical_logger(payload: dict[str, Any]) -> None:
            emit({"type": "lexical_event", "payload": payload})

        def embedding_logger(payload: dict[str, Any]) -> None:
            emit({"type": "embedding_event", "payload": payload})

        result = answer_with_knowledge_graph(
            request.question,
            settings=settings,
            entity_index_path=request.entity_index_path,
            doc_id=request.doc_id,
            max_tool_calls=request.max_tool_calls,
            tool_event_logger=tool_logger,
            lexical_event_logger=lexical_logger,
            embedding_event_logger=embedding_logger,
        )
        emit({"type": "result", "data": result})

    return _stream_ndjson(run)


@app.post("/api/discrepancies")
def api_discrepancies(request: DiscrepancyRequest) -> dict[str, Any]:
    settings = _settings(request.env_file)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report = detect_sop_discrepancies(
            settings=settings,
            sop_docx_path=request.sop_docx_path,
            entity_index_path=request.entity_index_path,
            doc_id=request.doc_id,
            stream_logs=True,
            max_tool_calls_per_item=request.max_tool_calls_per_item,
            dashboard_path=request.dashboard_path,
        )
    raw_logs = buffer.getvalue().splitlines()
    logs = [line.strip() for line in raw_logs if line.strip()]
    return {"report": report, "logs": logs}


@app.post("/api/discrepancies/stream")
def api_discrepancies_stream(request: DiscrepancyRequest) -> StreamingResponse:
    class _QueueWriter(io.TextIOBase):
        def __init__(self, emit):
            super().__init__()
            self._emit = emit
            self._buffer = ""

        def write(self, s: str) -> int:
            if not s:
                return 0
            self._buffer += s
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                self._emit({"type": "discrepancy_log", "message": line})
                if "KG tool call #" in line:
                    self._emit({"type": "tool_call", "source": "discrepancy", "payload": {"line": line}})
            return len(s)

        def flush(self) -> None:
            if self._buffer.strip():
                line = self._buffer.strip()
                self._emit({"type": "discrepancy_log", "message": line})
                if "KG tool call #" in line:
                    self._emit({"type": "tool_call", "source": "discrepancy", "payload": {"line": line}})
            self._buffer = ""

    def run(emit) -> None:
        settings = _settings(request.env_file)
        emit({"type": "status", "message": "Running discrepancy analysis..."})
        writer = _QueueWriter(emit)
        with redirect_stdout(writer):
            report = detect_sop_discrepancies(
                settings=settings,
                sop_docx_path=request.sop_docx_path,
                entity_index_path=request.entity_index_path,
                doc_id=request.doc_id,
                stream_logs=True,
                max_tool_calls_per_item=request.max_tool_calls_per_item,
                dashboard_path=request.dashboard_path,
            )
        writer.flush()
        emit({"type": "result", "data": {"report": report}})

    return _stream_ndjson(run)


@app.post("/api/graph/overview")
def api_graph_overview(request: GraphOverviewRequest) -> dict[str, Any]:
    settings = _settings(request.env_file)
    overview = _load_graph_overview(settings=settings, doc_id=request.doc_id, max_links=request.max_links)
    network = _load_entity_network(
        settings=settings,
        doc_id=request.doc_id,
        max_nodes=request.max_nodes,
        max_edges=request.max_edges,
        include_unknown=request.include_unknown,
    )
    return {**overview, "network": network}


def _load_graph_overview(settings: Settings, doc_id: str | None, max_links: int) -> dict[str, Any]:
    with Neo4jGraphStore.from_settings(settings) as store:
        params = {"doc_id": doc_id}
        metrics_row = store.query(
            """
            MATCH (e:Entity)
            WHERE ($doc_id IS NULL OR e.doc_id = $doc_id)
            WITH count(e) AS entities
            MATCH ()-[r:CONNECTED_TO]->()
            WITH entities, count(r) AS relations
            MATCH (m:Measurement)
            WHERE ($doc_id IS NULL OR m.doc_id = $doc_id)
            RETURN entities, relations, count(m) AS measurements
            """,
            params,
        )
        metrics = metrics_row[0] if metrics_row else {"entities": 0, "relations": 0, "measurements": 0}

        entity_types = store.query(
            """
            MATCH (e:Entity)
            WHERE ($doc_id IS NULL OR e.doc_id = $doc_id)
            RETURN coalesce(e.entity_type, 'Unknown') AS entity_type, count(*) AS count
            ORDER BY count DESC
            """,
            params,
        )

        type_links = store.query(
            """
            MATCH (a:Entity)-[:CONNECTED_TO]->(b:Entity)
            WHERE ($doc_id IS NULL OR (a.doc_id = $doc_id AND b.doc_id = $doc_id))
            WITH coalesce(a.entity_type,'Unknown') AS source_type,
                   coalesce(b.entity_type,'Unknown') AS target_type,
                   count(*) AS edge_count,
                   collect(distinct coalesce(a.name,'?') + ' -> ' + coalesce(b.name,'?'))[0..3] AS examples
            RETURN source_type, target_type, edge_count, examples
            ORDER BY edge_count DESC
            LIMIT $max_links
            """,
            {**params, "max_links": int(max_links)},
        )

        top_entities = store.query(
            """
            MATCH (e:Entity)
            WHERE ($doc_id IS NULL OR e.doc_id = $doc_id)
            OPTIONAL MATCH (e)-[r:CONNECTED_TO]-()
            RETURN e.name AS name, e.entity_type AS entity_type, count(r) AS degree
            ORDER BY degree DESC, name ASC
            LIMIT 20
            """,
            params,
        )

    return {
        "metrics": metrics,
        "entity_types": entity_types,
        "type_links": type_links,
        "top_entities": top_entities,
    }


def _load_entity_network(
    settings: Settings,
    doc_id: str | None,
    *,
    max_nodes: int,
    max_edges: int,
    include_unknown: bool,
) -> dict[str, Any]:
    with Neo4jGraphStore.from_settings(settings) as store:
        params = {
            "doc_id": doc_id,
            "max_nodes": int(max_nodes),
            "include_unknown": bool(include_unknown),
        }
        nodes = store.query(
            """
            MATCH (e:Entity)
            WHERE ($doc_id IS NULL OR e.doc_id = $doc_id)
              AND ($include_unknown OR toLower(coalesce(e.entity_type, 'unknown')) <> 'unknown')
            OPTIONAL MATCH (e)-[r:CONNECTED_TO]-(:Entity)
            WITH e, count(r) AS degree
            RETURN id(e) AS node_id,
                   coalesce(e.name, '') AS name,
                   coalesce(e.entity_id, '') AS entity_id,
                   coalesce(e.entity_type, 'Unknown') AS entity_type,
                   coalesce(e.raw_text, '') AS raw_text,
                   coalesce(e.description, '') AS description,
                   coalesce(e.attributes_json, '{}') AS attributes_json,
                   degree
            ORDER BY degree DESC, name ASC
            LIMIT $max_nodes
            """,
            params,
        )
        node_ids = [int(row["node_id"]) for row in nodes if row.get("node_id") is not None]
        if not node_ids:
            return {"nodes": [], "edges": []}

        edges = store.query(
            """
            MATCH (a:Entity)-[r:CONNECTED_TO]-(b:Entity)
            WHERE id(a) IN $node_ids
              AND id(b) IN $node_ids
              AND id(a) < id(b)
            WITH a, b, count(r) AS edge_count
            RETURN id(a) AS source,
                   id(b) AS target,
                   edge_count
            ORDER BY edge_count DESC
            LIMIT $max_edges
            """,
            {"node_ids": node_ids, "max_edges": int(max_edges)},
        )

    return {"nodes": nodes, "edges": edges}


@app.get("/api/discrepancies/dashboard")
def api_discrepancy_dashboard(path: str = "artifacts/discrepancy_dashboard.txt") -> dict[str, Any]:
    dashboard_path = Path(path)
    if not dashboard_path.exists():
        return {"path": str(dashboard_path), "exists": False, "content": ""}
    return {
        "path": str(dashboard_path),
        "exists": True,
        "content": dashboard_path.read_text(encoding="utf-8"),
    }
