from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .neo4j_store import Neo4jGraphStore
from .utils import normalize_ref


def build_entity_index_from_graph(
    store: Neo4jGraphStore,
    output_path: str | Path,
    doc_id: str | None = None,
) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    where_clause = "WHERE e.doc_id = $doc_id" if doc_id else ""
    cypher = f"""
    MATCH (e:Entity)
    {where_clause}
    OPTIONAL MATCH (e)-[r:CONNECTED_TO]->(t:Entity)
    WITH e, collect({{
        entity_id: t.entity_id,
        name: t.name,
        relation_type: r.relation_type,
        line_id: r.line_id,
        description: r.description
    }}) AS outgoing
    OPTIONAL MATCH (s:Entity)-[ri:CONNECTED_TO]->(e)
    WITH e, outgoing, collect({{
        entity_id: s.entity_id,
        name: s.name,
        relation_type: ri.relation_type,
        line_id: ri.line_id,
        description: ri.description
    }}) AS incoming
    OPTIONAL MATCH (e)-[:HAS_MEASUREMENT]->(m:Measurement)
    RETURN e, outgoing, incoming, collect({{
        kind: m.kind,
        value: m.value,
        unit: m.unit,
        raw_text: m.raw_text
    }}) AS measurements
    ORDER BY e.name
    """
    rows = store.query(cypher, {"doc_id": doc_id} if doc_id else None)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            entity = row["e"]
            attributes = entity.get("attributes")
            if not attributes:
                attributes = _load_json_object(entity.get("attributes_json"))
            entry = {
                "entity_id": entity.get("entity_id"),
                "name": entity.get("name"),
                "entity_type": entity.get("entity_type"),
                "doc_id": entity.get("doc_id"),
                "page_number": entity.get("page_number"),
                "description": entity.get("description"),
                "aliases": entity.get("aliases") or [],
                "attributes": attributes or {},
                "measurements": [m for m in row.get("measurements", []) if m and m.get("kind")],
                "outgoing_edges": [o for o in row.get("outgoing", []) if o and o.get("entity_id")],
                "incoming_edges": [i for i in row.get("incoming", []) if i and i.get("entity_id")],
            }
            entry["raw_text_blob"] = _build_raw_text_blob(entry)
            entry["context"] = _make_context(entry)
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
            count += 1
    return count


def lexical_search_entity_index(
    query: str,
    index_path: str | Path,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    index_path = Path(index_path)
    if not index_path.exists():
        return []

    with open(index_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in entries:
        score = _score_entry(query, entry)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    result: list[dict[str, Any]] = []
    for score, entry in scored[:top_k]:
        payload = dict(entry)
        payload["lexical_score"] = round(score, 3)
        result.append(payload)
    return result


def _make_context(entry: dict[str, Any]) -> str:
    parts = []
    name = entry.get("name") or ""
    entity_type = entry.get("entity_type") or "Unknown"
    description = entry.get("description") or ""
    parts.append(f"{name} is a {entity_type}.")
    if description:
        parts.append(description)

    measurements = entry.get("measurements") or []
    if measurements:
        rendered = ", ".join(
            f"{m.get('kind')}={m.get('value')} {m.get('unit') or ''}".strip() for m in measurements[:8]
        )
        parts.append(f"Measurements: {rendered}.")
        raw_text_lines = [str(m.get("raw_text")).strip() for m in measurements if m.get("raw_text")]
        if raw_text_lines:
            parts.append(f"Measurement raw text: {' | '.join(raw_text_lines[:8])}.")

    outgoing = entry.get("outgoing_edges") or []
    if outgoing:
        rendered = ", ".join(
            f"{edge.get('relation_type')} -> {edge.get('name')}" for edge in outgoing[:10] if edge.get("name")
        )
        parts.append(f"Outgoing edges: {rendered}.")

    incoming = entry.get("incoming_edges") or []
    if incoming:
        rendered = ", ".join(
            f"{edge.get('name')} -> {edge.get('relation_type')}" for edge in incoming[:10] if edge.get("name")
        )
        parts.append(f"Incoming edges: {rendered}.")
    return " ".join(parts).strip()


def _tokenize(value: str) -> list[str]:
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", value.lower()) if tok]
    expanded: list[str] = []
    for tok in tokens:
        expanded.append(tok)
        expanded.extend(re.findall(r"[a-z]+|\d+", tok))
    return [tok for tok in expanded if tok]


def _score_entry(query: str, entry: dict[str, Any]) -> float:
    name = entry.get("name") or ""
    aliases = entry.get("aliases") or []
    description = entry.get("description") or ""
    context = entry.get("context") or ""
    raw_text_blob = entry.get("raw_text_blob") or _build_raw_text_blob(entry)
    query_norm = normalize_ref(query)
    name_norm = normalize_ref(name)
    alias_norms = [normalize_ref(alias) for alias in aliases]

    score = 0.0
    if query_norm and query_norm == name_norm:
        score += 100.0
    if query_norm and query_norm in name_norm:
        score += 35.0
    if query_norm and any(query_norm == alias for alias in alias_norms):
        score += 40.0
    if query_norm and any(query_norm in alias for alias in alias_norms):
        score += 20.0

    text_blob = f"{name} {' '.join(aliases)} {description} {raw_text_blob} {context}"
    text_norm = normalize_ref(text_blob)
    if query_norm and query_norm in text_norm:
        score += 45.0
    query_tokens = set(_tokenize(query))
    text_tokens = set(_tokenize(text_blob))
    overlap = query_tokens.intersection(text_tokens)
    score += float(len(overlap) * 10)
    raw_overlap = query_tokens.intersection(set(_tokenize(raw_text_blob)))
    score += float(len(raw_overlap) * 14)

    if name:
        score += 35.0 * SequenceMatcher(None, query.lower(), name.lower()).ratio()
    if description:
        score += 15.0 * SequenceMatcher(None, query.lower(), description.lower()).ratio()
    if raw_text_blob:
        score += 20.0 * SequenceMatcher(None, query.lower(), raw_text_blob.lower()).ratio()
    score += 20.0 * SequenceMatcher(None, query.lower(), text_blob.lower()).ratio()
    return score


def _load_json_object(value: Any) -> dict[str, Any]:
    if not value or not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_raw_text_blob(entry: dict[str, Any]) -> str:
    chunks: list[str] = []
    description = entry.get("description")
    if description:
        chunks.append(str(description))

    measurements = entry.get("measurements") or []
    for measurement in measurements:
        raw_text = measurement.get("raw_text")
        if raw_text:
            chunks.append(str(raw_text))

    outgoing = entry.get("outgoing_edges") or []
    for edge in outgoing:
        if edge.get("description"):
            chunks.append(str(edge.get("description")))

    incoming = entry.get("incoming_edges") or []
    for edge in incoming:
        if edge.get("description"):
            chunks.append(str(edge.get("description")))

    return " ".join(chunk.strip() for chunk in chunks if str(chunk).strip())
