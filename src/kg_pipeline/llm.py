from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from pydantic import BaseModel, Field

from .pdf_tools import crop_image_with_bbox, image_to_data_url
from .schemas import BoundingBox, Entity, PageExtraction, Relation
from .utils import compact_json, normalize_ref


class ZoomRequest(BaseModel):
    reason: str
    boxes: list[BoundingBox] = Field(default_factory=list)


class VisionExtractionAgent:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_zoom_loops: int = 5,
        max_views_per_loop: int = 4,
        step_zoom_limit: int = 10,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_zoom_loops = max_zoom_loops
        self.max_views_per_loop = max_views_per_loop
        self.step_zoom_limit = step_zoom_limit

    def extract_page(
        self,
        doc_id: str,
        page_number: int,
        image_path: str | Path,
        work_dir: str | Path,
        event_logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> PageExtraction:
        image_path = Path(image_path)
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        _emit_vision_event(
            event_logger,
            {
                "event": "page_extraction_start",
                "doc_id": doc_id,
                "page_number": page_number,
                "image_path": str(image_path),
                "zoom_limit_per_step": self.step_zoom_limit,
            },
        )

        step1 = self._run_extraction_step(
            step_name="major_components",
            step_instruction=(
                "Step 1: Extract only large primary process components and their top-level identifiers. "
                "Examples: filters, vessels, towers, exchangers, pumps, separators. "
                "Prioritize exact equipment tags and avoid mislabeling one unit as another. "
                "Capture associated pressure/temperature text that is clearly tied to these major components. "
                "Capture high-level interconnections between major components when visible. "
                "Do not spend effort on small valves/instruments/pipes unless required to identify a major component."
            ),
            doc_id=doc_id,
            page_number=page_number,
            image_path=image_path,
            work_dir=work_dir / "step1_major",
            event_logger=event_logger,
            zoom_limit=self.step_zoom_limit,
        )

        anchors = self._make_major_anchor_text(step1.entities)
        step2 = self._run_extraction_step(
            step_name="detailed_pipes_measurements",
            step_instruction=(
                "Step 2: Extract smaller components and details: pipes, valves, instruments, line IDs, "
                "and all pressure/temperature measurements. Capture detailed connections among these items and "
                "to major components. Keep major-component names exactly as listed in anchors below.\n\n"
                f"Major component anchors:\n{anchors}"
            ),
            doc_id=doc_id,
            page_number=page_number,
            image_path=image_path,
            work_dir=work_dir / "step2_detail",
            event_logger=event_logger,
            zoom_limit=self.step_zoom_limit,
        )

        merged = self._merge_step_extractions(step1, step2, page_number=page_number)
        _emit_vision_event(
            event_logger,
            {
                "event": "page_extraction_complete",
                "doc_id": doc_id,
                "page_number": page_number,
                "entities": len(merged.entities),
                "relations": len(merged.relations),
                "step1_entities": len(step1.entities),
                "step2_entities": len(step2.entities),
            },
        )
        return merged

    def _run_extraction_step(
        self,
        *,
        step_name: str,
        step_instruction: str,
        doc_id: str,
        page_number: int,
        image_path: Path,
        work_dir: Path,
        event_logger: Callable[[dict[str, Any]], None] | None,
        zoom_limit: int,
    ) -> PageExtraction:
        work_dir.mkdir(parents=True, exist_ok=True)
        _emit_vision_event(
            event_logger,
            {
                "event": "extraction_step_start",
                "doc_id": doc_id,
                "page_number": page_number,
                "step": step_name,
                "zoom_limit": zoom_limit,
            },
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You extract structured entities/relations from oil and gas P&IDs. "
                    "Return tool calls only. Use request_zoom_views when needed. "
                    "When done, call submit_page_extraction exactly once."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Document id: {doc_id}. Page: {page_number}. Current extraction step: {step_name}. "
                            f"{step_instruction}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_path), "detail": "high"},
                    },
                ],
            },
        ]

        zoom_calls_used = 0
        max_iterations = zoom_limit + 8
        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._vision_tools(),
                tool_choice="auto",
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            assistant_payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                assistant_payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_payload)

            if not tool_calls:
                parsed = self._parse_page_extraction_from_content(message.content, page_number)
                if parsed is not None:
                    _emit_vision_event(
                        event_logger,
                        {
                            "event": "extraction_step_complete",
                            "doc_id": doc_id,
                            "page_number": page_number,
                            "step": step_name,
                            "entities": len(parsed.entities),
                            "relations": len(parsed.relations),
                            "zoom_calls_used": zoom_calls_used,
                        },
                    )
                    return parsed
                continue

            pending_user_messages: list[dict[str, Any]] = []
            pending_tool_messages: list[dict[str, Any]] = []
            submitted_extraction: PageExtraction | None = None

            for tool_call in tool_calls:
                name = tool_call.function.name
                args = _safe_json_loads(tool_call.function.arguments)
                _emit_vision_event(
                    event_logger,
                    {
                        "event": "tool_call",
                        "doc_id": doc_id,
                        "page_number": page_number,
                        "step": step_name,
                        "tool_name": name,
                        "tool_call_id": tool_call.id,
                    },
                )

                if name == "request_zoom_views":
                    if zoom_calls_used >= zoom_limit:
                        pending_tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": compact_json(
                                    {"ok": False, "error": f"Zoom call limit reached for step '{step_name}'."}
                                ),
                            }
                        )
                        continue

                    try:
                        zoom_request = ZoomRequest.model_validate(args)
                    except Exception as exc:
                        pending_tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": compact_json({"ok": False, "error": f"Invalid zoom request: {exc}"}),
                            }
                        )
                        continue

                    zoom_boxes = zoom_request.boxes[: self.max_views_per_loop]
                    provided_views: list[dict[str, Any]] = []
                    view_content: list[dict[str, Any]] = [
                        {
                            "type": "text",
                            "text": (
                                f"Step {step_name}, zoom call {zoom_calls_used + 1}/{zoom_limit}. "
                                f"Reason: {zoom_request.reason}. Returned {len(zoom_boxes)} views."
                            ),
                        }
                    ]

                    crop_failed = False
                    for idx, bbox in enumerate(zoom_boxes, start=1):
                        view_id = f"p{page_number}_{step_name}_z{zoom_calls_used + 1}_{idx}"
                        view_path = work_dir / f"{view_id}.png"
                        try:
                            crop_image_with_bbox(image_path, bbox, view_path)
                        except Exception as exc:
                            pending_tool_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": compact_json({"ok": False, "error": f"Crop failed: {exc}"}),
                                }
                            )
                            crop_failed = True
                            break
                        provided_views.append({"view_id": view_id, "bbox": bbox.model_dump()})
                        view_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": image_to_data_url(view_path), "detail": "high"},
                            }
                        )

                    if crop_failed:
                        continue

                    pending_tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": compact_json(
                                {
                                    "ok": True,
                                    "reason": zoom_request.reason,
                                    "provided_views": provided_views,
                                }
                            ),
                        }
                    )
                    if view_content:
                        pending_user_messages.append({"role": "user", "content": view_content})

                    zoom_calls_used += 1
                    _emit_vision_event(
                        event_logger,
                        {
                            "event": "zoom_call_complete",
                            "doc_id": doc_id,
                            "page_number": page_number,
                            "step": step_name,
                            "zoom_calls_used": zoom_calls_used,
                            "views": len(provided_views),
                        },
                    )
                    continue

                if name == "submit_page_extraction":
                    try:
                        extraction = self._coerce_page_extraction(args, page_number)
                        submitted_extraction = extraction
                        payload = {"ok": True, "received": True}
                    except Exception as exc:
                        payload = {"ok": False, "error": f"Invalid extraction payload: {exc}"}
                    pending_tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": compact_json(payload),
                        }
                    )
                    continue

                pending_tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": compact_json({"ok": False, "error": f"Unknown tool: {name}"}),
                    }
                )

            messages.extend(pending_tool_messages)
            messages.extend(pending_user_messages)
            if submitted_extraction is not None:
                _emit_vision_event(
                    event_logger,
                    {
                        "event": "extraction_step_complete",
                        "doc_id": doc_id,
                        "page_number": page_number,
                        "step": step_name,
                        "entities": len(submitted_extraction.entities),
                        "relations": len(submitted_extraction.relations),
                        "zoom_calls_used": zoom_calls_used,
                    },
                )
                return submitted_extraction

        raise RuntimeError(
            f"Step '{step_name}' did not submit extraction for page {page_number} after {max_iterations} iterations."
        )

    def _make_major_anchor_text(self, entities: list[Entity]) -> str:
        if not entities:
            return "No anchors extracted in step 1."
        lines: list[str] = []
        for entity in entities[:40]:
            lines.append(f"- {entity.name} ({entity.entity_type})")
        return "\n".join(lines)

    def _merge_step_extractions(
        self,
        step1: PageExtraction,
        step2: PageExtraction,
        *,
        page_number: int,
    ) -> PageExtraction:
        merged_entities: dict[str, Entity] = {}
        for entity in [*step1.entities, *step2.entities]:
            key = normalize_ref(entity.name) or normalize_ref(entity.entity_id or "")
            if not key:
                key = f"unnamed_{len(merged_entities)+1}"
            if key not in merged_entities:
                merged_entities[key] = entity
                continue
            existing = merged_entities[key]
            existing.aliases = _merge_unique_list(existing.aliases, entity.aliases)
            if not existing.description and entity.description:
                existing.description = entity.description
            if not existing.bbox and entity.bbox:
                existing.bbox = entity.bbox
            if entity.confidence is not None:
                existing.confidence = max(existing.confidence or 0.0, entity.confidence)
            if entity.attributes:
                for k, v in entity.attributes.items():
                    if k not in existing.attributes:
                        existing.attributes[k] = v
            existing.measurements = _merge_measurements(existing.measurements, entity.measurements)

        relation_seen: set[str] = set()
        merged_relations: list[Relation] = []
        for relation in [*step1.relations, *step2.relations]:
            r_key = (
                f"{normalize_ref(relation.source_ref)}|"
                f"{normalize_ref(relation.target_ref)}|"
                f"{relation.relation_type.upper()}|"
                f"{normalize_ref(relation.line_id or '')}"
            )
            if r_key in relation_seen:
                continue
            relation_seen.add(r_key)
            merged_relations.append(relation)

        notes = [*step1.notes, *step2.notes, "Merged extraction: step1 major components + step2 detailed components."]
        return PageExtraction(
            page_number=page_number,
            entities=list(merged_entities.values()),
            relations=merged_relations,
            notes=notes,
        )

    def _coerce_page_extraction(self, args: dict[str, Any], page_number: int) -> PageExtraction:
        payload = args.get("page_extraction") if isinstance(args, dict) else None
        if payload is None:
            payload = args
        payload = dict(payload)
        payload["page_number"] = page_number

        entities = payload.get("entities", [])
        for entity in entities:
            entity.setdefault("page_number", page_number)

        relations = payload.get("relations", [])
        for relation in relations:
            relation.setdefault("page_number", page_number)

        return PageExtraction.model_validate(payload)

    def _parse_page_extraction_from_content(self, content: str | None, page_number: int) -> PageExtraction | None:
        if not content:
            return None
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return self._coerce_page_extraction(payload, page_number)
        except Exception:
            return None

    def _vision_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "request_zoom_views",
                    "description": (
                        "Request zoomed-in crops from the current page image. "
                        "Bounding boxes are normalized to [0,1]."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "boxes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "left": {"type": "number"},
                                        "top": {"type": "number"},
                                        "right": {"type": "number"},
                                        "bottom": {"type": "number"},
                                        "label": {"type": "string"},
                                    },
                                    "required": ["left", "top", "right", "bottom"],
                                },
                            },
                        },
                        "required": ["reason", "boxes"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_page_extraction",
                    "description": "Submit final structured extraction for the current page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "page_extraction": {
                                "type": "object",
                                "properties": {
                                    "entities": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "entity_id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "entity_type": {"type": "string"},
                                                "description": {"type": "string"},
                                                "aliases": {"type": "array", "items": {"type": "string"}},
                                                "attributes": {"type": "object", "additionalProperties": True},
                                                "confidence": {"type": "number"},
                                                "bbox": {
                                                    "type": "object",
                                                    "properties": {
                                                        "left": {"type": "number"},
                                                        "top": {"type": "number"},
                                                        "right": {"type": "number"},
                                                        "bottom": {"type": "number"},
                                                        "label": {"type": "string"},
                                                    },
                                                },
                                                "measurements": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "kind": {"type": "string"},
                                                            "value": {"type": ["string", "number", "null"]},
                                                            "unit": {"type": ["string", "null"]},
                                                            "raw_text": {"type": ["string", "null"]},
                                                            "confidence": {"type": ["number", "null"]},
                                                            "bbox": {
                                                                "type": "object",
                                                                "properties": {
                                                                    "left": {"type": "number"},
                                                                    "top": {"type": "number"},
                                                                    "right": {"type": "number"},
                                                                    "bottom": {"type": "number"},
                                                                    "label": {"type": "string"},
                                                                },
                                                            },
                                                        },
                                                        "required": ["kind"],
                                                    },
                                                },
                                            },
                                            "required": ["name", "entity_type"],
                                        },
                                    },
                                    "relations": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "relation_id": {"type": "string"},
                                                "source_ref": {"type": "string"},
                                                "target_ref": {"type": "string"},
                                                "relation_type": {"type": "string"},
                                                "line_id": {"type": "string"},
                                                "description": {"type": "string"},
                                                "attributes": {"type": "object", "additionalProperties": True},
                                                "confidence": {"type": "number"},
                                            },
                                            "required": ["source_ref", "target_ref", "relation_type"],
                                        },
                                    },
                                    "notes": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["entities", "relations"],
                            }
                        },
                        "required": ["page_extraction"],
                    },
                },
            },
        ]


class KGToolAgent:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def run_with_graph_tool(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        search_tool: Callable[[str, dict[str, Any] | None], list[dict[str, Any]]],
        max_tool_calls: int = 10,
        tool_event_logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tool_events: list[dict[str, Any]] = []
        calls_used = 0

        for _ in range(max_tool_calls + 6):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[self._query_tool_schema()],
                tool_choice="auto",
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            assistant_payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                assistant_payload["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_payload)

            if not tool_calls:
                return {
                    "response_text": message.content or "",
                    "tool_calls_used": calls_used,
                    "tool_events": tool_events,
                }

            for tool_call in tool_calls:
                if calls_used >= max_tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": compact_json(
                                {"ok": False, "error": f"Tool call limit reached ({max_tool_calls})."}
                            ),
                        }
                    )
                    continue

                args = _safe_json_loads(tool_call.function.arguments)
                cypher = args.get("cypher", "")
                params = args.get("params") if isinstance(args.get("params"), dict) else None
                calls_used += 1
                try:
                    rows = search_tool(cypher, params)
                    payload = {"ok": True, "row_count": len(rows), "rows": rows[:100]}
                except Exception as exc:
                    payload = {"ok": False, "error": str(exc)}
                event = {
                    "call_index": calls_used,
                    "cypher": cypher,
                    "params": params,
                    "result_summary": payload,
                }
                tool_events.append(event)
                if tool_event_logger is not None:
                    tool_event_logger(event)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": compact_json(payload),
                    }
                )

        return {
            "response_text": "Unable to complete query reasoning within tool loop limits.",
            "tool_calls_used": calls_used,
            "tool_events": tool_events,
        }

    def _query_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_knowledge_graph",
                "description": "Run a read-only Cypher query against the knowledge graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cypher": {"type": "string"},
                        "params": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["cypher"],
                },
            },
        }


def _safe_json_loads(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    try:
        loaded = json.loads(payload)
        if isinstance(loaded, dict):
            return loaded
        return {}
    except json.JSONDecodeError:
        return {}


def _merge_unique_list(a: list[str], b: list[str]) -> list[str]:
    out: list[str] = []
    for item in [*(a or []), *(b or [])]:
        cleaned = str(item).strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _merge_measurements(a: list[Any], b: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for measurement in [*(a or []), *(b or [])]:
        key = _measurement_key(measurement)
        if key in seen:
            continue
        seen.add(key)
        out.append(measurement)
    return out


def _measurement_key(measurement: Any) -> str:
    kind = str(getattr(measurement, "kind", "") or "")
    value = str(getattr(measurement, "value", "") or "")
    unit = str(getattr(measurement, "unit", "") or "")
    raw_text = str(getattr(measurement, "raw_text", "") or "")
    return f"{kind}|{value}|{unit}|{normalize_ref(raw_text)}"


def _emit_vision_event(logger: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    if logger is None:
        return
    logger(payload)
