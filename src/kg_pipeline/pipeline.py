from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .entity_embeddings import build_entity_embedding_index
from .entity_index import build_entity_index_from_graph
from .llm import VisionExtractionAgent
from .neo4j_store import Neo4jGraphStore
from .pdf_tools import render_pdf_to_images
from .schemas import DocumentExtraction, Entity, PageExtraction
from .utils import make_stable_id, normalize_ref, utc_now_iso


def enforce_spec_data_layout(base_dir: str | Path = ".") -> dict[str, str]:
    base_dir = Path(base_dir)
    pid_target = base_dir / "data" / "pid" / "diagram.pdf"
    sop_target = base_dir / "data" / "sop" / "sop.docx"
    pid_target.parent.mkdir(parents=True, exist_ok=True)
    sop_target.parent.mkdir(parents=True, exist_ok=True)

    copied = {"pid": "no-op", "sop": "no-op"}
    legacy_pid = base_dir / "data" / "diagram_page1.pdf"
    legacy_sop = base_dir / "data" / "sop.docx"
    if not pid_target.exists() and legacy_pid.exists():
        pid_target.write_bytes(legacy_pid.read_bytes())
        copied["pid"] = f"copied {legacy_pid} -> {pid_target}"
    if not sop_target.exists() and legacy_sop.exists():
        sop_target.write_bytes(legacy_sop.read_bytes())
        copied["sop"] = f"copied {legacy_sop} -> {sop_target}"
    return copied


def build_knowledge_graph(
    *,
    doc_id: str = "pid_diagram",
    settings: Settings | None = None,
    pid_pdf_path: str | None = None,
    entity_index_path: str | None = None,
    extraction_dump_path: str | None = None,
    log_progress: bool = True,
    log_handler: Callable[[dict[str, Any]], None] | None = None,
    build_embeddings: bool = True,
) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    enforce_spec_data_layout(".")

    pdf_path = Path(pid_pdf_path or settings.pid_pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"P&ID PDF not found: {pdf_path}")
    _emit_build_log(
        log_progress,
        log_handler,
        {"event": "build_start", "doc_id": doc_id, "pid_pdf_path": str(pdf_path), "output_dir": settings.output_dir},
    )

    output_dir = Path(settings.output_dir)
    rendered_dir = output_dir / "rendered_pages"
    zoom_dir = output_dir / "zoom_views"
    extraction_dir = output_dir / "extractions"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    zoom_dir.mkdir(parents=True, exist_ok=True)
    extraction_dir.mkdir(parents=True, exist_ok=True)

    image_paths = render_pdf_to_images(pdf_path, rendered_dir, dpi=settings.render_dpi)
    _emit_build_log(
        log_progress,
        log_handler,
        {
            "event": "render_complete",
            "doc_id": doc_id,
            "pages_rendered": len(image_paths),
            "rendered_dir": str(rendered_dir),
        },
    )
    extractor = VisionExtractionAgent(
        api_key=settings.openai_api_key,
        model=settings.openai_vision_model,
        max_zoom_loops=settings.max_zoom_loops,
        max_views_per_loop=settings.max_views_per_loop,
    )

    pages: list[PageExtraction] = []
    for page_number, image_path in enumerate(image_paths, start=1):
        _emit_build_log(
            log_progress,
            log_handler,
            {"event": "page_start", "doc_id": doc_id, "page_number": page_number, "image_path": str(image_path)},
        )

        def _page_event_logger(payload: dict[str, Any]) -> None:
            event_payload = dict(payload)
            event_payload.setdefault("event", "vision_event")
            event_payload["stage"] = "vision_extraction"
            _emit_build_log(log_progress, log_handler, event_payload)

        page_result = extractor.extract_page(
            doc_id=doc_id,
            page_number=page_number,
            image_path=image_path,
            work_dir=zoom_dir / f"page_{page_number:03d}",
            event_logger=_page_event_logger,
        )
        pages.append(page_result)
        with open(extraction_dir / f"page_{page_number:03d}.json", "w", encoding="utf-8") as f:
            json.dump(page_result.model_dump(mode="json"), f, ensure_ascii=True, indent=2)
        _emit_build_log(
            log_progress,
            log_handler,
            {
                "event": "page_saved",
                "doc_id": doc_id,
                "page_number": page_number,
                "entities": len(page_result.entities),
                "relations": len(page_result.relations),
                "path": str(extraction_dir / f"page_{page_number:03d}.json"),
            },
        )

    document = DocumentExtraction(doc_id=doc_id, pages=pages)

    extraction_dump = Path(extraction_dump_path) if extraction_dump_path else (output_dir / "document_extraction.json")
    extraction_dump.parent.mkdir(parents=True, exist_ok=True)
    with open(extraction_dump, "w", encoding="utf-8") as f:
        json.dump(document.model_dump(mode="json"), f, ensure_ascii=True, indent=2)
    _emit_build_log(
        log_progress,
        log_handler,
        {"event": "document_extraction_saved", "doc_id": doc_id, "path": str(extraction_dump)},
    )

    ingest_summary = _ingest_document_to_graph(
        document=document,
        settings=settings,
        source_path=pdf_path,
        entity_index_path=entity_index_path,
        log_progress=log_progress,
        log_handler=log_handler,
        build_embeddings=build_embeddings,
    )

    summary = {
        "doc_id": doc_id,
        "pid_pdf_path": str(pdf_path),
        "pages_processed": len(image_paths),
        "extraction_dump_path": str(extraction_dump),
        "generated_at": utc_now_iso(),
        **ingest_summary,
    }
    _emit_build_log(log_progress, log_handler, {"event": "build_complete", **summary})
    return summary


def ingest_extraction_dump(
    *,
    settings: Settings | None = None,
    doc_id: str | None = None,
    source_path: str | None = None,
    extraction_dump_path: str | None = None,
    extraction_pages_dir: str | None = None,
    entity_index_path: str | None = None,
    log_progress: bool = True,
    log_handler: Callable[[dict[str, Any]], None] | None = None,
    build_embeddings: bool = True,
) -> dict[str, Any]:
    """
    Ingest already-saved extraction JSON into Neo4j and rebuild entity index,
    without calling the multimodal extraction model.
    """
    settings = settings or Settings.from_env()
    output_dir = Path(settings.output_dir)

    dump_path = Path(extraction_dump_path) if extraction_dump_path else (output_dir / "document_extraction.json")
    pages_dir = Path(extraction_pages_dir) if extraction_pages_dir else (output_dir / "extractions")
    source = Path(source_path or settings.pid_pdf_path)

    document, loaded_from = _load_document_extraction(
        doc_id=doc_id,
        extraction_dump_path=dump_path,
        extraction_pages_dir=pages_dir,
    )
    _emit_build_log(
        log_progress,
        log_handler,
        {
            "event": "ingest_from_extraction_start",
            "doc_id": document.doc_id,
            "loaded_from": loaded_from,
            "pages": len(document.pages),
            "source_path": str(source),
        },
    )

    ingest_summary = _ingest_document_to_graph(
        document=document,
        settings=settings,
        source_path=source,
        entity_index_path=entity_index_path,
        log_progress=log_progress,
        log_handler=log_handler,
        build_embeddings=build_embeddings,
    )

    summary = {
        "doc_id": document.doc_id,
        "loaded_from": loaded_from,
        "source_path": str(source),
        "generated_at": utc_now_iso(),
        **ingest_summary,
    }
    _emit_build_log(log_progress, log_handler, {"event": "ingest_from_extraction_complete", **summary})
    return summary


def _load_document_extraction(
    *,
    doc_id: str | None,
    extraction_dump_path: Path,
    extraction_pages_dir: Path,
) -> tuple[DocumentExtraction, str]:
    if extraction_dump_path.exists():
        with open(extraction_dump_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid extraction dump format: {extraction_dump_path}")
        if doc_id:
            payload["doc_id"] = doc_id
        if not payload.get("doc_id"):
            payload["doc_id"] = "pid_diagram"
        document = DocumentExtraction.model_validate(payload)
        return document, str(extraction_dump_path)

    if extraction_pages_dir.exists():
        page_files = sorted(extraction_pages_dir.glob("page_*.json"))
        if page_files:
            pages: list[PageExtraction] = []
            for page_file in page_files:
                with open(page_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                pages.append(PageExtraction.model_validate(payload))
            pages.sort(key=lambda p: p.page_number)
            document = DocumentExtraction(doc_id=doc_id or "pid_diagram", pages=pages)
            return document, str(extraction_pages_dir)

    raise FileNotFoundError(
        "No extraction inputs found. Expected either "
        f"{extraction_dump_path} or page JSON files in {extraction_pages_dir}."
    )


def _ingest_document_to_graph(
    *,
    document: DocumentExtraction,
    settings: Settings,
    source_path: Path,
    entity_index_path: str | None,
    log_progress: bool,
    log_handler: Callable[[dict[str, Any]], None] | None,
    build_embeddings: bool,
) -> dict[str, Any]:
    entity_lookup: dict[str, str] = {}
    entity_count = 0
    relation_count = 0

    with Neo4jGraphStore.from_settings(settings) as store:
        _emit_build_log(
            log_progress,
            log_handler,
            {
                "event": "neo4j_ingest_start",
                "doc_id": document.doc_id,
                "database": store.active_database,
                "pages": len(document.pages),
            },
        )
        store.create_schema()
        _emit_build_log(log_progress, log_handler, {"event": "neo4j_schema_ready", "doc_id": document.doc_id})
        store.upsert_document(doc_id=document.doc_id, source_path=source_path, page_count=len(document.pages))
        _emit_build_log(log_progress, log_handler, {"event": "neo4j_document_upserted", "doc_id": document.doc_id})

        for page in document.pages:
            page_entity_count = 0
            for entity in page.entities:
                entity.entity_id = _resolve_entity_id(entity, entity_lookup)
                _register_entity_aliases(entity, entity_lookup)
                store.upsert_entity(doc_id=document.doc_id, source_path=source_path, entity=entity)
                entity_count += 1
                page_entity_count += 1
            _emit_build_log(
                log_progress,
                log_handler,
                {
                    "event": "neo4j_page_entities_upserted",
                    "doc_id": document.doc_id,
                    "page_number": page.page_number,
                    "count": page_entity_count,
                },
            )

        for page in document.pages:
            page_relation_count = 0
            for relation in page.relations:
                source_id = _resolve_reference_to_entity(
                    relation.source_ref,
                    page.page_number,
                    entity_lookup,
                    store,
                    document.doc_id,
                    source_path,
                )
                target_id = _resolve_reference_to_entity(
                    relation.target_ref,
                    page.page_number,
                    entity_lookup,
                    store,
                    document.doc_id,
                    source_path,
                )
                relation.relation_id = relation.relation_id or make_stable_id(
                    source_id,
                    target_id,
                    relation.relation_type,
                    relation.line_id or "",
                    str(page.page_number),
                    prefix="rel",
                )
                store.upsert_relation(relation, source_entity_id=source_id, target_entity_id=target_id)
                relation_count += 1
                page_relation_count += 1
            _emit_build_log(
                log_progress,
                log_handler,
                {
                    "event": "neo4j_page_relations_upserted",
                    "doc_id": document.doc_id,
                    "page_number": page.page_number,
                    "count": page_relation_count,
                },
            )

        index_path = Path(entity_index_path or (Path(settings.output_dir) / "entity_index.jsonl"))
        indexed_count = build_entity_index_from_graph(store, index_path, doc_id=document.doc_id)
        _emit_build_log(
            log_progress,
            log_handler,
            {
                "event": "entity_index_built",
                "doc_id": document.doc_id,
                "path": str(index_path),
                "rows": indexed_count,
            },
        )

    embedding_summary: dict[str, Any] | None = None
    if build_embeddings:
        _emit_build_log(
            log_progress,
            log_handler,
            {"event": "embedding_index_build_start", "doc_id": document.doc_id, "entity_index_path": str(index_path)},
        )
        embedding_summary = build_entity_embedding_index(
            settings=settings,
            entity_index_path=index_path,
            output_dir=settings.output_dir,
            log_handler=(lambda payload: _emit_build_log(log_progress, log_handler, payload)),
        )

    summary = {
        "pages_in_graph": len(document.pages),
        "entities_written": entity_count,
        "relations_written": relation_count,
        "entity_index_path": str(index_path),
        "entity_index_rows": indexed_count,
    }
    if embedding_summary is not None:
        summary["embedding_index_path"] = embedding_summary.get("index_path")
        summary["embedding_chunks_path"] = embedding_summary.get("chunks_path")
        summary["embedding_chunks"] = embedding_summary.get("chunks")
    return summary


def _emit_build_log(
    enabled: bool,
    handler: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if not enabled:
        return
    if handler is not None:
        handler(payload)
        return
    print(f"build_kg> {json.dumps(payload, ensure_ascii=True)}", flush=True)


def _resolve_entity_id(entity: Entity, entity_lookup: dict[str, str]) -> str:
    if entity.entity_id:
        return entity.entity_id
    key = _entity_lookup_key(entity.name, entity.entity_type)
    if key in entity_lookup:
        return entity_lookup[key]
    entity_id = make_stable_id(entity.entity_type, normalize_ref(entity.name), prefix="ent")
    entity_lookup[key] = entity_id
    return entity_id


def _register_entity_aliases(entity: Entity, entity_lookup: dict[str, str]) -> None:
    if not entity.entity_id:
        return
    entity_lookup[_entity_lookup_key(entity.name, entity.entity_type)] = entity.entity_id
    entity_lookup[_entity_lookup_key(entity.name, "ANY")] = entity.entity_id
    for alias in entity.aliases:
        entity_lookup[_entity_lookup_key(alias, entity.entity_type)] = entity.entity_id
        entity_lookup[_entity_lookup_key(alias, "ANY")] = entity.entity_id


def _entity_lookup_key(name: str, entity_type: str) -> str:
    return f"{entity_type.upper()}::{normalize_ref(name)}"


def _resolve_reference_to_entity(
    ref: str,
    page_number: int,
    entity_lookup: dict[str, str],
    store: Neo4jGraphStore,
    doc_id: str,
    source_path: Path,
) -> str:
    keys = [_entity_lookup_key(ref, "ANY")]
    for key in keys:
        if key in entity_lookup:
            return entity_lookup[key]

    placeholder = Entity(
        entity_id=make_stable_id("Unknown", normalize_ref(ref), prefix="ent"),
        name=ref,
        entity_type="Unknown",
        description="Placeholder created from relation reference.",
        page_number=page_number,
        attributes={"placeholder": True},
    )
    _register_entity_aliases(placeholder, entity_lookup)
    store.upsert_entity(doc_id=doc_id, source_path=source_path, entity=placeholder)
    return placeholder.entity_id
