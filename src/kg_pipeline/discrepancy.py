from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import Settings
from .entity_index import lexical_search_entity_index
from .llm import KGToolAgent
from .neo4j_store import Neo4jGraphStore
from .utils import utc_now_iso


WRITE_CYPHER_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|DROP|REMOVE|LOAD\s+CSV|FOREACH|CALL\s+dbms)\b",
    flags=re.IGNORECASE,
)
REFERENCE_TEMP_TOLERANCE_F = 15.0


def detect_sop_discrepancies(
    *,
    settings: Settings | None = None,
    sop_docx_path: str | None = None,
    entity_index_path: str | None = None,
    doc_id: str | None = None,
    stream_logs: bool = True,
    max_tool_calls_per_item: int = 6,
    dashboard_path: str | None = None,
) -> dict[str, Any]:
    from .sop_parser import extract_design_limits, extract_sop_text

    settings = settings or Settings.from_env()
    sop_path = Path(sop_docx_path or settings.sop_docx_path)
    index_path = Path(entity_index_path or (Path(settings.output_dir) / "entity_index.jsonl"))

    if not sop_path.exists():
        raise FileNotFoundError(f"SOP file not found: {sop_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Entity index not found: {index_path}")

    limits = extract_design_limits(sop_path)
    sop_text = extract_sop_text(sop_path)
    discrepancies: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []

    _warn(stream_logs, f"Starting SOP discrepancy analysis. limits_found={len(limits)}")

    with Neo4jGraphStore.from_settings(settings) as store:
        agent = KGToolAgent(api_key=settings.openai_api_key, model=settings.openai_query_model)

        for item in limits:
            sop_name = str(item.get("name", "")).strip()
            expected_mwp = _parse_mwp_psig(item.get("pressure_psig"))
            expected_ref_temp = _parse_reference_temperature_f(item.get("temperature_f"))

            candidates = lexical_search_entity_index(sop_name, index_path=index_path, top_k=6)
            _warn(
                stream_logs,
                f"Checking '{sop_name}'. candidates={len(candidates)} expected_mwp_psig={expected_mwp} reference_temp_f={expected_ref_temp}",
            )

            if not candidates:
                discrepancy = {
                    "type": "missing_entity",
                    "sop_name": sop_name,
                    "expected_mwp_psig": expected_mwp,
                    "reference_temperature_f": expected_ref_temp,
                    "reason": "No candidate entity from lexical index.",
                }
                discrepancies.append(discrepancy)
                evaluations.append(
                    {
                        "sop_name": sop_name,
                        "expected_mwp_psig": expected_mwp,
                        "reference_temperature_f": expected_ref_temp,
                        "resolved_entity_id": None,
                        "resolved_entity_name": None,
                        "observed_pressure_values": [],
                        "observed_pressure_psig": [],
                        "observed_temperature_values": [],
                        "observed_temperature_f": [],
                        "observed_operating_points": [],
                        "pressure_status": "missing_entity",
                        "temperature_status": "reference_only",
                        "notes": ["No lexical candidates found."],
                        "tool_calls_used": 0,
                    }
                )
                _warn(stream_logs, f"Discrepancy: missing_entity for '{sop_name}'")
                continue

            analysis = _analyze_limit_with_query_flow(
                item=item,
                candidates=candidates,
                doc_id=doc_id,
                agent=agent,
                store=store,
                max_tool_calls=max_tool_calls_per_item,
                stream_logs=stream_logs,
            )

            observed_pressure_values = analysis.get("observed_pressure_values", [])
            observed_temperature_values = analysis.get("observed_temperature_values", [])
            observed_pressure_nums = analysis.get("observed_pressure_psig", [])
            observed_temp_nums = analysis.get("observed_temperature_f", [])
            observed_operating_points = analysis.get("observed_operating_points", [])

            pressure_status, pressure_violations, evaluated_points = _evaluate_mwp_at_reference_temperature(
                expected_mwp=expected_mwp,
                reference_temp_f=expected_ref_temp,
                observed_points=observed_operating_points,
                observed_pressures=observed_pressure_nums,
                observed_temperatures=observed_temp_nums,
                tolerance_f=REFERENCE_TEMP_TOLERANCE_F,
            )
            notes = analysis.get("notes", [])
            note_values = [str(note).strip() for note in notes if str(note).strip()] if isinstance(notes, list) else []
            if pressure_status == "missing_data":
                note_values.append(
                    "No pressure/temperature operating point was found near the SOP reference temperature."
                )

            evaluation = {
                "sop_name": sop_name,
                "expected_mwp_psig": expected_mwp,
                "reference_temperature_f": expected_ref_temp,
                "reference_temperature_tolerance_f": REFERENCE_TEMP_TOLERANCE_F,
                "resolved_entity_id": analysis.get("resolved_entity_id"),
                "resolved_entity_name": analysis.get("resolved_entity_name"),
                "observed_pressure_values": observed_pressure_values,
                "observed_pressure_psig": observed_pressure_nums,
                "observed_temperature_values": observed_temperature_values,
                "observed_temperature_f": observed_temp_nums,
                "observed_operating_points": observed_operating_points,
                "evaluated_operating_points": evaluated_points,
                "pressure_status": pressure_status,
                "temperature_status": "reference_only",
                "pressure_violations": pressure_violations,
                "temperature_violations": [],
                "notes": _dedupe_strings(note_values),
                "tool_calls_used": analysis.get("tool_calls_used", 0),
            }
            evaluations.append(evaluation)

            if pressure_status in {"exceeds_mwp", "missing_entity"}:
                discrepancy = {
                    "type": "pressure_limit_issue",
                    "sop_name": sop_name,
                    "matched_entity_id": analysis.get("resolved_entity_id"),
                    "matched_entity_name": analysis.get("resolved_entity_name"),
                    "expected_mwp_psig": expected_mwp,
                    "reference_temperature_f": expected_ref_temp,
                    "observed_pressure_values": observed_pressure_values,
                    "observed_pressure_psig": observed_pressure_nums,
                    "observed_temperature_f": observed_temp_nums,
                    "observed_operating_points": observed_operating_points,
                    "evaluated_operating_points": evaluated_points,
                    "pressure_status": pressure_status,
                    "pressure_violations": pressure_violations,
                    "notes": _dedupe_strings(note_values),
                }
                discrepancies.append(discrepancy)
                _warn(stream_logs, f"Discrepancy: pressure_limit_issue for '{sop_name}' status={pressure_status}")

            if pressure_status == "within_mwp":
                _warn(
                    stream_logs,
                    f"Within limits: '{sop_name}' matched_entity={analysis.get('resolved_entity_name')} ({analysis.get('resolved_entity_id')})",
                )

    report = {
        "generated_at": utc_now_iso(),
        "sop_docx_path": str(sop_path),
        "doc_id": doc_id,
        "checked_limits": len(limits),
        "evaluations": evaluations,
        "discrepancies": discrepancies,
        "discrepancy_count": len(discrepancies),
        "sop_excerpt": sop_text[:2000],
    }

    if dashboard_path:
        rendered_path = write_discrepancy_dashboard(report, dashboard_path)
        report["dashboard_path"] = rendered_path
        _warn(stream_logs, f"Dashboard written: {rendered_path}")

    _warn(stream_logs, f"Completed discrepancy analysis. discrepancy_count={len(discrepancies)}")
    return report


def write_discrepancy_dashboard(report: dict[str, Any], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    evaluations = report.get("evaluations", []) if isinstance(report.get("evaluations"), list) else []
    discrepancies = report.get("discrepancies", []) if isinstance(report.get("discrepancies"), list) else []

    checked_limits = int(report.get("checked_limits", 0) or 0)
    discrepancy_count = int(report.get("discrepancy_count", 0) or 0)

    pressure_counts = _status_counts(evaluations, "pressure_status")
    lines: list[str] = []
    lines.append("SOP vs KG Operating Limits Dashboard")
    lines.append("=" * 78)
    lines.append(f"Generated at: {report.get('generated_at', '')}")
    lines.append(f"SOP file: {report.get('sop_docx_path', '')}")
    lines.append(f"Doc ID: {report.get('doc_id', '')}")
    lines.append(f"Checked SOP items: {checked_limits}")
    lines.append(f"Discrepancies: {discrepancy_count}")
    lines.append("")

    lines.append("Summary")
    lines.append("-" * 78)
    lines.append(
        "Pressure status -> "
        f"within_mwp={pressure_counts.get('within_mwp',0)} "
        f"exceeds_mwp={pressure_counts.get('exceeds_mwp',0)} "
        f"missing_data={pressure_counts.get('missing_data',0)} "
        f"missing_entity={pressure_counts.get('missing_entity',0)}"
    )
    lines.append("")

    lines.append("Per-Item Results")
    lines.append("-" * 78)
    for idx, ev in enumerate(evaluations, start=1):
        sop_name = ev.get("sop_name", "")
        entity_name = ev.get("resolved_entity_name") or "N/A"
        entity_id = ev.get("resolved_entity_id") or "N/A"
        lines.append(f"[{idx}] {sop_name}")
        lines.append(f"  Entity: {entity_name} ({entity_id})")
        lines.append(
            "  MWP @ reference temperature: "
            f"expected<={ev.get('expected_mwp_psig')} psig at {ev.get('reference_temperature_f')} F "
            f"(+/- {ev.get('reference_temperature_tolerance_f')} F) "
            f"| evaluated points={_short_operating_points(ev.get('evaluated_operating_points', []), 4)} "
            f"| status={ev.get('pressure_status')}"
        )
        if ev.get("pressure_violations"):
            lines.append(f"  Pressure violations (psig): {_short_list(ev.get('pressure_violations', []), 6)}")
        notes = ev.get("notes") if isinstance(ev.get("notes"), list) else []
        if notes:
            lines.append(f"  Notes: {_short_list(notes, 2)}")
        lines.append(f"  Tool calls: {ev.get('tool_calls_used', 0)}")
        lines.append("")

    lines.append("Discrepancy Details")
    lines.append("-" * 78)
    if not discrepancies:
        lines.append("No discrepancies found.")
    else:
        for idx, item in enumerate(discrepancies, start=1):
            lines.append(f"[{idx}] {item.get('type')} | SOP: {item.get('sop_name')}")
            lines.append(
                f"  Matched entity: {item.get('matched_entity_name', 'N/A')} ({item.get('matched_entity_id', 'N/A')})"
            )
            if "expected_mwp_psig" in item:
                lines.append(
                    "  MWP check: "
                    f"expected<={item.get('expected_mwp_psig')} psig at {item.get('reference_temperature_f')} F "
                    f"| evaluated points={_short_operating_points(item.get('evaluated_operating_points', []), 4)} "
                    f"| status={item.get('pressure_status')}"
                )
            if item.get("reason"):
                lines.append(f"  Reason: {item.get('reason')}")
            notes = item.get("notes") if isinstance(item.get("notes"), list) else []
            if notes:
                lines.append(f"  Notes: {_short_list(notes, 2)}")
            lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(path)


def _analyze_limit_with_query_flow(
    *,
    item: dict[str, Any],
    candidates: list[dict[str, Any]],
    doc_id: str | None,
    agent: KGToolAgent,
    store: Neo4jGraphStore,
    max_tool_calls: int,
    stream_logs: bool,
) -> dict[str, Any]:
    sop_name = str(item.get("name", "")).strip()

    candidate_payload: list[dict[str, Any]] = []
    for candidate in candidates[:6]:
        candidate_payload.append(
            {
                "entity_id": candidate.get("entity_id"),
                "name": candidate.get("name"),
                "entity_type": candidate.get("entity_type"),
                "page_number": candidate.get("page_number"),
                "lexical_score": candidate.get("lexical_score"),
                "description": candidate.get("description"),
                "context": candidate.get("context"),
                "raw_text_blob": candidate.get("raw_text_blob"),
                "measurements": candidate.get("measurements", [])[:12],
            }
        )

    def search_tool(cypher: str, params: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not _is_read_only_cypher(cypher):
            raise ValueError("Only read-only Cypher is allowed in search_knowledge_graph.")
        bound_params = dict(params or {})
        if doc_id and "doc_id" not in bound_params:
            bound_params["doc_id"] = doc_id
        return store.query(cypher, bound_params)

    def tool_logger(event: dict[str, Any]) -> None:
        summary = event.get("result_summary", {}) if isinstance(event.get("result_summary"), dict) else {}
        row_count = summary.get("row_count")
        err = summary.get("error")
        _warn(
            stream_logs,
            f"KG tool call #{event.get('call_index')} for '{sop_name}'. rows={row_count} error={err} cypher={_truncate(str(event.get('cypher','')), 140)}",
        )

    system_prompt = (
        "You evaluate SOP operating limits against a P&ID KG. "
        "Use search_knowledge_graph to retrieve evidence. "
        "You MUST prioritize raw_text from measurements/entities because critical tags and values are often there. "
        "Return strict JSON only with schema: "
        "You are talking to engineers, not developers. Therefore, don't mention the Knowledge Graph or the Cypher query language, or any other technical details. Just focus on the engineering details."
        "{"
        "\"resolved_entity\":{\"entity_id\":\"...\",\"name\":\"...\",\"confidence\":0.0},"
        "\"observed_operating_points\":[{\"pressure_psig\":0.0,\"temperature_f\":0.0,\"raw_text\":\"...\"}],"
        "\"observed_pressure_values\":[\"...\"],"
        "\"observed_temperature_values\":[\"...\"],"
        "\"observed_pressure_psig\":[0.0],"
        "\"observed_temperature_f\":[0.0],"
        "\"notes\":[\"...\"]"
        "}."
    )
    user_prompt = (
        f"SOP item name: {sop_name}\n"
        f"SOP pressure field: {item.get('pressure_psig')}\n"
        f"SOP temperature field: {item.get('temperature_f')}\n\n"
        "Candidate entities from index:\n"
        f"{json.dumps(candidate_payload, ensure_ascii=True, indent=2)}\n\n"
        "Process:\n"
        "1) Resolve best entity match.\n"
        "2) Gather pressure and temperature observations from KG, prioritizing raw_text evidence.\n"
        "3) Extract operating points as pressure/temperature pairs whenever possible.\n"
        "4) The SOP temperature is the reference temperature at which MWP is defined.\n"
        "5) Return strict JSON only."
    )

    result = agent.run_with_graph_tool(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        search_tool=search_tool,
        max_tool_calls=max_tool_calls,
        tool_event_logger=tool_logger,
    )

    parsed = _extract_json_object(result.get("response_text") or "")
    resolved = parsed.get("resolved_entity") if isinstance(parsed.get("resolved_entity"), dict) else {}

    observed_pressure_values = _normalize_observed_values(parsed.get("observed_pressure_values"))
    observed_temperature_values = _normalize_observed_values(parsed.get("observed_temperature_values"))
    observed_operating_points = _normalize_operating_points(parsed.get("observed_operating_points"))

    parsed_pressure_nums = _normalize_numeric_list(parsed.get("observed_pressure_psig"))
    parsed_temp_nums = _normalize_numeric_list(parsed.get("observed_temperature_f"))

    if not parsed_pressure_nums:
        parsed_pressure_nums = _extract_numbers_from_values(observed_pressure_values)
    if not parsed_temp_nums:
        parsed_temp_nums = _extract_numbers_from_values(observed_temperature_values)
    if not observed_operating_points:
        observed_operating_points = _infer_operating_points(
            pressure_values=observed_pressure_values,
            temperature_values=observed_temperature_values,
            pressure_nums=parsed_pressure_nums,
            temperature_nums=parsed_temp_nums,
        )

    notes = parsed.get("notes") if isinstance(parsed.get("notes"), list) else []
    note_values = [str(note).strip() for note in notes if str(note).strip()]
    if not parsed:
        note_values.append("LLM response was not valid JSON.")

    return {
        "resolved_entity_id": resolved.get("entity_id"),
        "resolved_entity_name": resolved.get("name"),
        "resolved_entity_confidence": resolved.get("confidence"),
        "observed_pressure_values": observed_pressure_values,
        "observed_temperature_values": observed_temperature_values,
        "observed_pressure_psig": parsed_pressure_nums,
        "observed_temperature_f": parsed_temp_nums,
        "observed_operating_points": observed_operating_points,
        "notes": note_values,
        "tool_calls_used": result.get("tool_calls_used", 0),
        "tool_events": result.get("tool_events", []),
        "raw_response": result.get("response_text", ""),
    }


def _parse_mwp_psig(value: Any) -> float | None:
    numbers = _extract_numbers(str(value or ""))
    if not numbers:
        return None
    return max(numbers)


def _parse_reference_temperature_f(value: Any) -> float | None:
    numbers = _extract_numbers(str(value or ""))
    if not numbers:
        return None
    return min(numbers)


def _evaluate_mwp_at_reference_temperature(
    *,
    expected_mwp: float | None,
    reference_temp_f: float | None,
    observed_points: list[dict[str, Any]],
    observed_pressures: list[float],
    observed_temperatures: list[float],
    tolerance_f: float,
) -> tuple[str, list[float], list[dict[str, Any]]]:
    if expected_mwp is None:
        return "unknown_limit", [], []

    candidate_points: list[dict[str, Any]] = []
    if reference_temp_f is None:
        candidate_points = [
            {"pressure_psig": float(p), "temperature_f": None, "raw_text": ""}
            for p in observed_pressures
        ]
    else:
        for point in observed_points:
            p = point.get("pressure_psig")
            t = point.get("temperature_f")
            if p is None or t is None:
                continue
            if abs(float(t) - reference_temp_f) <= tolerance_f:
                candidate_points.append(
                    {
                        "pressure_psig": float(p),
                        "temperature_f": float(t),
                        "raw_text": str(point.get("raw_text", "")),
                    }
                )

    if not candidate_points and observed_pressures and observed_temperatures and reference_temp_f is not None:
        for pressure, temp in zip(observed_pressures, observed_temperatures):
            if abs(float(temp) - reference_temp_f) <= tolerance_f:
                candidate_points.append(
                    {"pressure_psig": float(pressure), "temperature_f": float(temp), "raw_text": ""}
                )

    if not candidate_points:
        return "missing_data", [], []

    violations = [round(float(pt["pressure_psig"]), 6) for pt in candidate_points if float(pt["pressure_psig"]) > expected_mwp]
    if violations:
        return "exceeds_mwp", violations, candidate_points
    return "within_mwp", [], candidate_points


def _is_read_only_cypher(cypher: str) -> bool:
    if not cypher or not cypher.strip():
        return False
    return not bool(WRITE_CYPHER_PATTERN.search(cypher))


def _normalize_observed_values(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for value in values:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                out.append(cleaned)
            continue
        if isinstance(value, dict):
            raw_text = str(value.get("raw_text", "")).strip()
            if raw_text:
                out.append(raw_text)
                continue
            maybe_value = value.get("value")
            maybe_unit = value.get("unit")
            if maybe_value is not None:
                rendered = f"{maybe_value} {maybe_unit or ''}".strip()
                if rendered:
                    out.append(rendered)
    return _dedupe_strings(out)


def _normalize_numeric_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _normalize_operating_points(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        pressure = item.get("pressure_psig")
        temperature = item.get("temperature_f")
        try:
            pressure_f = float(pressure) if pressure is not None else None
        except (TypeError, ValueError):
            pressure_f = None
        try:
            temperature_f = float(temperature) if temperature is not None else None
        except (TypeError, ValueError):
            temperature_f = None

        raw_text = str(item.get("raw_text", "")).strip()
        out.append(
            {
                "pressure_psig": pressure_f,
                "temperature_f": temperature_f,
                "raw_text": raw_text,
            }
        )

    return out


def _infer_operating_points(
    *,
    pressure_values: list[str],
    temperature_values: list[str],
    pressure_nums: list[float],
    temperature_nums: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    combined_values = [*pressure_values, *temperature_values]
    for text in combined_values:
        numbers = _extract_numbers(text)
        if len(numbers) < 2:
            continue
        out.append(
            {
                "pressure_psig": float(numbers[0]),
                "temperature_f": float(numbers[1]),
                "raw_text": text,
            }
        )

    if not out and pressure_nums and temperature_nums:
        for pressure, temperature in zip(pressure_nums, temperature_nums):
            out.append(
                {
                    "pressure_psig": float(pressure),
                    "temperature_f": float(temperature),
                    "raw_text": "",
                }
            )

    return out


def _extract_numbers_from_values(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        out.extend(_extract_numbers(value))
    return out


def _extract_numbers(text: str) -> list[float]:
    nums: list[float] = []
    for token in re.findall(r"-?\d+(?:\.\d+)?", text):
        try:
            nums.append(float(token))
        except ValueError:
            continue
    return nums


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _status_counts(evaluations: Any, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(evaluations, list):
        return counts
    for item in evaluations:
        if not isinstance(item, dict):
            continue
        status = str(item.get(key, "unknown") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _short_list(values: Any, max_items: int) -> str:
    if not isinstance(values, list) or not values:
        return "[]"
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return "[]"
    if len(items) <= max_items:
        return "[" + "; ".join(items) + "]"
    head = items[:max_items]
    return "[" + "; ".join(head) + f"; ... (+{len(items) - max_items} more)]"


def _short_operating_points(values: Any, max_items: int) -> str:
    if not isinstance(values, list) or not values:
        return "[]"
    rendered: list[str] = []
    for point in values:
        if not isinstance(point, dict):
            continue
        pressure = point.get("pressure_psig")
        temperature = point.get("temperature_f")
        if pressure is None and temperature is None:
            continue
        rendered.append(f"({pressure} psig @ {temperature} F)")
    if not rendered:
        return "[]"
    if len(rendered) <= max_items:
        return "[" + "; ".join(rendered) + "]"
    head = rendered[:max_items]
    return "[" + "; ".join(head) + f"; ... (+{len(rendered) - max_items} more)]"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _warn(enabled: bool, message: str) -> None:
    if not enabled:
        return
    print(f"[{utc_now_iso()}] - WARNING - [{message}]", flush=True)
