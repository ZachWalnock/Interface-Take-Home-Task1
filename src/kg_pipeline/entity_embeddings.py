from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
from openai import OpenAI

from .config import Settings
from .utils import utc_now_iso


def build_entity_embedding_index(
    *,
    settings: Settings,
    entity_index_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    log_handler: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir or settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "entity_vectors.faiss"
    chunks_path = output_dir / "entity_chunks.jsonl"
    manifest_path = output_dir / "entity_vectors.manifest.json"
    index_input_path = Path(entity_index_path or (output_dir / "entity_index.jsonl"))

    if not index_input_path.exists():
        raise FileNotFoundError(f"Entity index not found: {index_input_path}")

    entries = _read_jsonl(index_input_path)
    chunks = _build_entity_chunks(
        entries=entries,
        chunk_sentences=settings.embedding_chunk_sentences,
        chunk_overlap=settings.embedding_chunk_overlap,
    )
    _emit(
        log_handler,
        {
            "event": "embedding_chunks_built",
            "entity_index_path": str(index_input_path),
            "entities": len(entries),
            "chunks": len(chunks),
            "chunk_sentences": settings.embedding_chunk_sentences,
            "chunk_overlap": settings.embedding_chunk_overlap,
        },
    )
    if not chunks:
        raise RuntimeError("No chunks generated from entity index; cannot build embedding index.")

    client = OpenAI(api_key=settings.openai_api_key)
    vectors = _embed_texts(
        client=client,
        model=settings.openai_embedding_model,
        texts=[chunk["text"] for chunk in chunks],
        batch_size=64,
        log_handler=log_handler,
    )

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, str(index_path))

    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=True) + "\n")

    manifest = {
        "created_at": utc_now_iso(),
        "embedding_model": settings.openai_embedding_model,
        "vector_dim": dim,
        "entities": len(entries),
        "chunks": len(chunks),
        "entity_index_path": str(index_input_path),
        "index_path": str(index_path),
        "chunks_path": str(chunks_path),
        "chunk_sentences": settings.embedding_chunk_sentences,
        "chunk_overlap": settings.embedding_chunk_overlap,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    _emit(log_handler, {"event": "embedding_index_built", **manifest})
    return manifest


def semantic_search_entities(
    *,
    query: str,
    settings: Settings,
    output_dir: str | Path | None = None,
    top_k_chunks: int | None = None,
    top_k_entities: int | None = None,
    log_handler: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir = Path(output_dir or settings.output_dir)
    index_path = output_dir / "entity_vectors.faiss"
    chunks_path = output_dir / "entity_chunks.jsonl"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Embedding index not found. Expected {index_path} and {chunks_path}. Build embeddings first."
        )

    chunks = _read_jsonl(chunks_path)
    if not chunks:
        return [], {"event": "embedding_search_empty_chunks"}

    index = faiss.read_index(str(index_path))
    client = OpenAI(api_key=settings.openai_api_key)
    query_vec = _embed_texts(
        client=client,
        model=settings.openai_embedding_model,
        texts=[query],
        batch_size=1,
        log_handler=None,
    )
    faiss.normalize_L2(query_vec)

    chunk_k = min(top_k_chunks or settings.embedding_top_k_chunks, len(chunks))
    distances, indices = index.search(query_vec, chunk_k)
    distances_list = distances[0].tolist()
    indices_list = indices[0].tolist()

    entity_map: dict[str, dict[str, Any]] = {}
    ranked_chunks: list[dict[str, Any]] = []
    for score, idx in zip(distances_list, indices_list):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        ranked_chunks.append(
            {
                "rank": len(ranked_chunks) + 1,
                "score": float(score),
                "chunk_id": chunk.get("chunk_id"),
                "entity_id": chunk.get("entity_id"),
                "entity_name": chunk.get("entity_name"),
                "entity_type": chunk.get("entity_type"),
                "page_number": chunk.get("page_number"),
                "text": chunk.get("text"),
            }
        )
        entity_id = str(chunk.get("entity_id") or "")
        if not entity_id:
            continue
        current = entity_map.get(entity_id)
        if current is None:
            entity_map[entity_id] = {
                "entity_id": entity_id,
                "name": chunk.get("entity_name"),
                "entity_type": chunk.get("entity_type"),
                "doc_id": chunk.get("doc_id"),
                "page_number": chunk.get("page_number"),
                "embedding_score": float(score),
                "embedding_best_chunk": chunk.get("text"),
                "embedding_top_chunks": [chunk.get("text")],
            }
            continue
        current["embedding_score"] = max(float(current["embedding_score"]), float(score))
        if float(score) >= float(current["embedding_score"]):
            current["embedding_best_chunk"] = chunk.get("text")
        top_chunks = current.setdefault("embedding_top_chunks", [])
        if len(top_chunks) < 3 and chunk.get("text") not in top_chunks:
            top_chunks.append(chunk.get("text"))

    entities = sorted(entity_map.values(), key=lambda x: float(x.get("embedding_score") or 0.0), reverse=True)[
        : (top_k_entities or settings.embedding_top_k_entities)
    ]
    debug = {
        "event": "embedding_search",
        "query": query,
        "top_k_chunks": chunk_k,
        "candidate_chunks": ranked_chunks[:10],
        "top_entities": [
            {
                "rank": i + 1,
                "entity_id": row.get("entity_id"),
                "name": row.get("name"),
                "entity_type": row.get("entity_type"),
                "page_number": row.get("page_number"),
                "embedding_score": round(float(row.get("embedding_score") or 0.0), 6),
            }
            for i, row in enumerate(entities)
        ],
    }
    _emit(log_handler, debug)
    return entities, debug


def _build_entity_chunks(entries: list[dict[str, Any]], chunk_sentences: int, chunk_overlap: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunk_sentences = max(1, chunk_sentences)
    chunk_overlap = max(0, min(chunk_overlap, chunk_sentences - 1))
    step = max(1, chunk_sentences - chunk_overlap)

    for entry in entries:
        base = _build_entity_text(entry)
        sentences = _split_sentences(base)
        if not sentences:
            continue

        entity_id = entry.get("entity_id") or ""
        entity_name = entry.get("name") or ""
        entity_type = entry.get("entity_type") or "Unknown"
        doc_id = entry.get("doc_id")
        page_number = entry.get("page_number")

        seen_texts: set[str] = set()
        chunk_index = 0
        for start in range(0, len(sentences), step):
            batch = sentences[start : start + chunk_sentences]
            if not batch:
                break
            text = " ".join(batch).strip()
            if not text:
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)
            chunk_index += 1
            chunks.append(
                {
                    "chunk_id": f"{entity_id}::chunk_{chunk_index:03d}",
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "sentence_start": start,
                    "sentence_end": start + len(batch) - 1,
                    "source_fields": [
                        "name",
                        "entity_type",
                        "description",
                        "aliases",
                        "attributes",
                        "measurements",
                        "outgoing_edges",
                        "incoming_edges",
                        "context",
                        "raw_text_blob",
                    ],
                    "text": text,
                }
            )

    return chunks


def _build_entity_text(entry: dict[str, Any]) -> str:
    sections: list[str] = []
    sections.append(f"Entity name: {entry.get('name') or ''}.")
    sections.append(f"Entity id: {entry.get('entity_id') or ''}.")
    sections.append(f"Entity type: {entry.get('entity_type') or ''}.")
    if entry.get("description"):
        sections.append(f"Description: {entry['description']}.")
    aliases = entry.get("aliases") or []
    if aliases:
        sections.append(f"Aliases: {', '.join(str(a) for a in aliases)}.")
    attributes = entry.get("attributes") or {}
    if attributes:
        sections.append(f"Attributes: {json.dumps(attributes, ensure_ascii=True)}.")

    measurements = entry.get("measurements") or []
    for measurement in measurements:
        parts = [
            f"kind={measurement.get('kind')}",
            f"value={measurement.get('value')}",
            f"unit={measurement.get('unit')}",
            f"raw_text={measurement.get('raw_text')}",
        ]
        sections.append(f"Measurement: {', '.join(parts)}.")

    outgoing = entry.get("outgoing_edges") or []
    for edge in outgoing:
        sections.append(
            "Outgoing edge: "
            f"type={edge.get('relation_type')}, "
            f"to={edge.get('name')}, "
            f"line_id={edge.get('line_id')}, "
            f"description={edge.get('description')}."
        )

    incoming = entry.get("incoming_edges") or []
    for edge in incoming:
        sections.append(
            "Incoming edge: "
            f"type={edge.get('relation_type')}, "
            f"from={edge.get('name')}, "
            f"line_id={edge.get('line_id')}, "
            f"description={edge.get('description')}."
        )

    if entry.get("context"):
        sections.append(f"Context: {entry.get('context')}.")
    if entry.get("raw_text_blob"):
        sections.append(f"Raw text blob: {entry.get('raw_text_blob')}.")
    return " ".join(section.strip() for section in sections if section and str(section).strip())


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.replace("\n", " ").replace("|", ". ")).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    out = [part.strip() for part in parts if part.strip()]
    return out


def _embed_texts(
    *,
    client: OpenAI,
    model: str,
    texts: list[str],
    batch_size: int,
    log_handler: Callable[[dict[str, Any]], None] | None,
) -> np.ndarray:
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend([row.embedding for row in response.data])
        _emit(log_handler, {"event": "embedding_batch_done", "start": start, "count": len(batch), "total": total})
    if not vectors:
        raise RuntimeError("No embeddings returned from OpenAI.")
    return np.array(vectors, dtype=np.float32)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _emit(handler: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    if handler is None:
        return
    handler(payload)
