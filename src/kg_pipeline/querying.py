from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .config import Settings
from .entity_index import lexical_search_entity_index
from .llm import KGToolAgent
from .neo4j_store import Neo4jGraphStore


WRITE_CYPHER_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|DROP|REMOVE|LOAD\s+CSV|FOREACH|CALL\s+dbms)\b",
    flags=re.IGNORECASE,
)


def answer_with_knowledge_graph(
    question: str,
    *,
    settings: Settings | None = None,
    entity_index_path: str | None = None,
    doc_id: str | None = None,
    max_tool_calls: int = 10,
    tool_event_logger: Callable[[dict[str, Any]], None] | None = None,
    lexical_event_logger: Callable[[dict[str, Any]], None] | None = None,
    embedding_event_logger: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    index_path = Path(entity_index_path or (Path(settings.output_dir) / "entity_index.jsonl"))

    extracted_terms = _extract_query_entities_with_llm(
        question=question,
        api_key=settings.openai_api_key,
        model=settings.openai_query_model,
    )
    search_terms = _build_entity_search_terms(question=question, extracted_terms=extracted_terms)
    lexical_matches, lexical_debug = _search_entity_index_for_terms(
        search_terms=search_terms,
        index_path=index_path,
        top_k=settings.lexical_top_k_entities,
        lexical_event_logger=lexical_event_logger,
    )
    embedding_matches, embedding_debug = _search_embedding_index_for_query(
        query=question,
        settings=settings,
        embedding_event_logger=embedding_event_logger,
    )
    matches = _union_entity_matches(
        lexical_matches=lexical_matches[: settings.lexical_top_k_entities],
        embedding_matches=embedding_matches[: settings.embedding_top_k_entities],
        top_k=max(settings.lexical_top_k_entities, settings.embedding_top_k_entities) * 2,
    )
    slim_matches = [
        {
            "entity_id": row.get("entity_id"),
            "name": row.get("name"),
            "entity_type": row.get("entity_type"),
            "page_number": row.get("page_number"),
            "context": row.get("context"),
            "lexical_score": row.get("lexical_score"),
            "embedding_score": row.get("embedding_score"),
            "matched_on": row.get("matched_on"),
            "retrieval_sources": row.get("retrieval_sources", []),
            "embedding_best_chunk": row.get("embedding_best_chunk"),
        }
        for row in matches
    ]

    with Neo4jGraphStore.from_settings(settings) as store:
        agent = KGToolAgent(api_key=settings.openai_api_key, model=settings.openai_query_model)

        def search_tool(cypher: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
            if not is_read_only_cypher(cypher):
                raise ValueError("Only read-only Cypher is allowed in search_knowledge_graph.")
            bound_params = dict(params or {})
            if doc_id and "doc_id" not in bound_params:
                bound_params["doc_id"] = doc_id
            return store.query(cypher, bound_params)

        system_prompt, user_prompt = _build_query_prompts(
            question=question,
            candidate_entities=slim_matches,
            doc_id=doc_id,
            max_tool_calls=max_tool_calls,
        )
        result = agent.run_with_graph_tool(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            search_tool=search_tool,
            max_tool_calls=max_tool_calls,
            tool_event_logger=tool_event_logger,
        )

    return {
        "question": question,
        "extracted_entity_terms": extracted_terms,
        "entity_search_terms": search_terms,
        "lexical_search_debug": lexical_debug,
        "embedding_search_debug": embedding_debug,
        "matched_entities": slim_matches,
        "answer": result["response_text"],
        "tool_calls_used": result["tool_calls_used"],
        "tool_events": result["tool_events"],
    }


def is_read_only_cypher(cypher: str) -> bool:
    if not cypher or not cypher.strip():
        return False
    return not bool(WRITE_CYPHER_PATTERN.search(cypher))


def _build_query_prompts(
    *,
    question: str,
    candidate_entities: list[dict[str, Any]],
    doc_id: str | None,
    max_tool_calls: int,
) -> tuple[str, str]:
    system_prompt = (
        "You answer engineering questions by traversing a Neo4j knowledge graph using Cypher via the "
        "search_knowledge_graph tool. Favor exact entity_id/name matches from candidate entities. "
        "Traversal philosophy: first inspect properties and outgoing edges of the most relevant entities; "
        "then expand through those edges progressively; at each step, re-check properties and outgoing edges "
        "before expanding further. Use up to "
        f"{max_tool_calls} tool calls."
    )
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Document scope doc_id: {doc_id or 'not-forced'}\n\n"
        "Candidate entities from lexical search (use these as anchors):\n"
        f"{json.dumps(candidate_entities, ensure_ascii=True, indent=2)}\n\n"
        "Guidance:\n"
        "1) Start from the best anchor entity/entities.\n"
        "2) Query and list their properties and outgoing edges.\n"
        "3) Expand slowly through those edges and keep checking properties/outgoing edges as you go.\n"
        "4) If evidence is insufficient within tool-call budget, say exactly what is missing.\n"
        "5) Keep final answer concise and evidence-based."
    )
    return system_prompt, user_prompt


def _extract_query_entities_with_llm(question: str, api_key: str, model: str) -> list[str]:
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract entity mentions from user engineering questions. "
                        "Return strict JSON only: {\"entities\":[...]} with short strings. "
                        "Include equipment tags and near-tag variants, e.g. F-715, F 715, V-720."
                    ),
                },
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
        payload = _safe_json_loads(text)
        entities = payload.get("entities")
    except Exception:
        return []

    if not isinstance(entities, list):
        return []
    out: list[str] = []
    for item in entities:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned:
            out.append(cleaned)
    return _dedupe_preserve_order(out)


def _build_entity_search_terms(question: str, extracted_terms: list[str]) -> list[str]:
    terms = _dedupe_preserve_order([*extracted_terms, *_tag_like_candidates(question), question])
    return [term for term in terms if term.strip()]


def _search_entity_index_for_terms(
    search_terms: list[str],
    index_path: Path,
    top_k: int = 12,
    lexical_event_logger: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: dict[str, dict[str, Any]] = {}
    per_term_debug: list[dict[str, Any]] = []
    for term in search_terms:
        term_hits = lexical_search_entity_index(term, index_path=index_path, top_k=8)
        top_hits = [
            {
                "rank": idx + 1,
                "name": hit.get("name"),
                "entity_id": hit.get("entity_id"),
                "entity_type": hit.get("entity_type"),
                "page_number": hit.get("page_number"),
                "score": round(float(hit.get("lexical_score") or 0.0), 3),
            }
            for idx, hit in enumerate(term_hits[:5])
        ]
        term_event = {"event": "lexical_search_term", "term": term, "hit_count": len(term_hits), "top_hits": top_hits}
        per_term_debug.append(term_event)
        if lexical_event_logger is not None:
            lexical_event_logger(term_event)
        for hit in term_hits:
            key = hit.get("entity_id") or f"{hit.get('name')}::{hit.get('page_number')}"
            score = float(hit.get("lexical_score") or 0.0)
            prev = merged.get(key)
            if prev is None or score > float(prev.get("lexical_score") or 0.0):
                payload = dict(hit)
                payload["lexical_score"] = score
                payload["matched_on"] = term
                merged[key] = payload

    ranked = sorted(merged.values(), key=lambda row: float(row.get("lexical_score") or 0.0), reverse=True)
    final_ranked = ranked[:top_k]
    merged_event = {
        "event": "lexical_search_merged",
        "input_terms": len(search_terms),
        "unique_entities": len(merged),
        "returned_entities": len(final_ranked),
        "top_entities": [
            {
                "rank": idx + 1,
                "name": row.get("name"),
                "entity_id": row.get("entity_id"),
                "entity_type": row.get("entity_type"),
                "page_number": row.get("page_number"),
                "score": round(float(row.get("lexical_score") or 0.0), 3),
                "matched_on": row.get("matched_on"),
            }
            for idx, row in enumerate(final_ranked[:8])
        ],
    }
    per_term_debug.append(merged_event)
    if lexical_event_logger is not None:
        lexical_event_logger(merged_event)
    return final_ranked, per_term_debug


def _tag_like_candidates(text: str) -> list[str]:
    # Capture likely equipment/instrument tag patterns from free-form text.
    patterns = [
        r"\b[A-Za-z]{1,4}[-_\s]?\d{2,5}[A-Za-z]?\b",
        r"\b\d{2,5}[A-Za-z]\b",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            cleaned = str(match).strip()
            if cleaned:
                candidates.append(cleaned)
    return _dedupe_preserve_order(candidates)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value.strip())
    return out


def _safe_json_loads(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _search_embedding_index_for_query(
    *,
    query: str,
    settings: Settings,
    embedding_event_logger: Callable[[dict[str, Any]], None] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from .entity_embeddings import semantic_search_entities

        entities, debug = semantic_search_entities(
            query=query,
            settings=settings,
            output_dir=settings.output_dir,
            top_k_chunks=settings.embedding_top_k_chunks,
            top_k_entities=settings.embedding_top_k_entities,
            log_handler=embedding_event_logger,
        )
        return entities, debug
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        debug = {"event": "embedding_search_skipped", "reason": str(exc)}
        if embedding_event_logger is not None:
            embedding_event_logger(debug)
        return [], debug


def _union_entity_matches(
    *,
    lexical_matches: list[dict[str, Any]],
    embedding_matches: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for row in lexical_matches:
        key = str(row.get("entity_id") or f"{row.get('name')}::{row.get('page_number')}")
        payload = dict(row)
        payload.setdefault("retrieval_sources", [])
        payload["retrieval_sources"] = _append_unique(payload["retrieval_sources"], "lexical")
        merged[key] = payload

    for row in embedding_matches:
        key = str(row.get("entity_id") or f"{row.get('name')}::{row.get('page_number')}")
        payload = dict(row)
        payload.setdefault("retrieval_sources", [])
        payload["retrieval_sources"] = _append_unique(payload["retrieval_sources"], "embedding")
        if key not in merged:
            merged[key] = payload
            continue

        current = merged[key]
        current["retrieval_sources"] = _append_unique(current.get("retrieval_sources", []), "embedding")
        if current.get("embedding_score") is None and payload.get("embedding_score") is not None:
            current["embedding_score"] = payload.get("embedding_score")
        if payload.get("embedding_best_chunk") and not current.get("embedding_best_chunk"):
            current["embedding_best_chunk"] = payload.get("embedding_best_chunk")
        if payload.get("embedding_top_chunks") and not current.get("embedding_top_chunks"):
            current["embedding_top_chunks"] = payload.get("embedding_top_chunks")
        if not current.get("name") and payload.get("name"):
            current["name"] = payload.get("name")
        if not current.get("entity_type") and payload.get("entity_type"):
            current["entity_type"] = payload.get("entity_type")
        if not current.get("page_number") and payload.get("page_number"):
            current["page_number"] = payload.get("page_number")

    ranked = sorted(merged.values(), key=_hybrid_rank_key, reverse=True)
    return ranked[:top_k]


def _hybrid_rank_key(row: dict[str, Any]) -> tuple[float, float]:
    lexical = float(row.get("lexical_score") or 0.0)
    embedding = float(row.get("embedding_score") or 0.0)
    return lexical, embedding


def _append_unique(values: list[Any], value: Any) -> list[Any]:
    out = list(values) if isinstance(values, list) else []
    if value not in out:
        out.append(value)
    return out
