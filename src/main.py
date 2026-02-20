from __future__ import annotations

import argparse
import json
import sys

from kg_pipeline import Settings, answer_with_knowledge_graph, build_knowledge_graph, ingest_extraction_dump, detect_sop_discrepancies, write_discrepancy_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive KG Q/A chat")
    parser.add_argument("--env-file", default=".env", help="Path to environment file")
    parser.add_argument("--doc-id", default="pid_demo", help="Document id to scope graph queries")
    parser.add_argument("--entity-index-path", default=None, help="Optional override for entity index path")
    parser.add_argument("--max-tool-calls", type=int, default=5, help="Max Cypher tool calls per answer")
    parser.add_argument("--trace", action="store_true", help="Print matched entities and Cypher tool events")
    parser.add_argument("--no-tool-logs", action="store_true", help="Disable live tool-call logs")
    parser.add_argument("--no-lexical-logs", action="store_true", help="Disable lexical search debug logs")
    parser.add_argument("--no-embedding-logs", action="store_true", help="Disable embedding search debug logs")
    args = parser.parse_args()

    settings = Settings.from_env(args.env_file)
    trace_enabled = bool(args.trace)
    tool_logs_enabled = not bool(args.no_tool_logs)
    lexical_logs_enabled = not bool(args.no_lexical_logs)
    embedding_logs_enabled = not bool(args.no_embedding_logs)
    history: list[tuple[str, str]] = []

    print("KG Chat CLI")
    print("Type /help for commands.")
    print(f"Using doc_id={args.doc_id}")

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command in {"/exit", "/quit"}:
            print("Exiting.")
            break
        if command == "/help":
            print("Commands:")
            print("/help  - show commands")
            print("/reset - clear conversation memory")
            print("/trace - toggle query trace output")
            print("/toollogs - toggle live Cypher tool-call logs")
            print("/lexlogs - toggle lexical search debug logs")
            print("/emblogs - toggle embedding search debug logs")
            print("/exit  - quit")
            continue
        if command == "/reset":
            history.clear()
            print("Conversation memory cleared.")
            continue
        if command == "/trace":
            trace_enabled = not trace_enabled
            print(f"Trace {'enabled' if trace_enabled else 'disabled'}.")
            continue
        if command == "/toollogs":
            tool_logs_enabled = not tool_logs_enabled
            print(f"Tool logs {'enabled' if tool_logs_enabled else 'disabled'}.")
            continue
        if command == "/lexlogs":
            lexical_logs_enabled = not lexical_logs_enabled
            print(f"Lexical logs {'enabled' if lexical_logs_enabled else 'disabled'}.")
            continue
        if command == "/emblogs":
            embedding_logs_enabled = not embedding_logs_enabled
            print(f"Embedding logs {'enabled' if embedding_logs_enabled else 'disabled'}.")
            continue

        question = _compose_question_with_context(user_input, history)
        try:
            logger = _make_tool_logger() if tool_logs_enabled else None
            lexical_logger = _make_lexical_logger() if lexical_logs_enabled else None
            embedding_logger = _make_embedding_logger() if embedding_logs_enabled else None
            result = answer_with_knowledge_graph(
                question,
                settings=settings,
                entity_index_path=args.entity_index_path,
                doc_id=args.doc_id,
                max_tool_calls=args.max_tool_calls,
                tool_event_logger=logger,
                lexical_event_logger=lexical_logger,
                embedding_event_logger=embedding_logger,
            )
        except Exception as exc:
            print(f"error> {exc}")
            continue

        answer = (result.get("answer") or "").strip() or "No answer generated."
        print(f"{_green('assistant>')} {_green(answer)}")
        history.append((user_input, answer))

        if trace_enabled:
            print("extracted_entity_terms>")
            print(json.dumps(result.get("extracted_entity_terms", []), ensure_ascii=True, indent=2))
            print("entity_search_terms>")
            print(json.dumps(result.get("entity_search_terms", []), ensure_ascii=True, indent=2))
            print("lexical_search_debug>")
            print(json.dumps(result.get("lexical_search_debug", []), ensure_ascii=True, indent=2))
            print("embedding_search_debug>")
            print(json.dumps(result.get("embedding_search_debug", {}), ensure_ascii=True, indent=2))
            print("matched_entities>")
            print(json.dumps(result.get("matched_entities", []), ensure_ascii=True, indent=2))
            print("tool_events>")
            print(json.dumps(result.get("tool_events", []), ensure_ascii=True, indent=2))


def _compose_question_with_context(user_question: str, history: list[tuple[str, str]], max_turns: int = 4) -> str:
    if not history:
        return user_question
    recent = history[-max_turns:]
    lines = []
    for user_text, assistant_text in recent:
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")
    context = "\n".join(lines)
    return (
        f"{user_question}\n\n"
        "Conversation context (may be useful for coreference only):\n"
        f"{context}"
    )


def _make_tool_logger():
    def _logger(event: dict[str, object]) -> None:
        index = event.get("call_index", "?")
        cypher = (event.get("cypher") or "").__str__().strip()
        params = event.get("params")
        summary = event.get("result_summary") if isinstance(event.get("result_summary"), dict) else {}
        ok = bool(summary.get("ok")) if isinstance(summary, dict) else False
        if ok:
            row_count = summary.get("row_count", 0)
            print(f"tool[{index}]> rows={row_count}")
        else:
            err = summary.get("error", "unknown error") if isinstance(summary, dict) else "unknown error"
            print(f"tool[{index}]> error={err}")
        print(f"tool[{index}] cypher> {cypher}")
        if params:
            print(f"tool[{index}] params> {json.dumps(params, ensure_ascii=True)}")

    return _logger


def _make_lexical_logger():
    def _logger(event: dict[str, object]) -> None:
        event_type = str(event.get("event", ""))
        if event_type == "lexical_search_term":
            term = str(event.get("term", ""))
            hit_count = int(event.get("hit_count", 0))
            print(f"lexical> term='{term}' hits={hit_count}")
            top_hits = event.get("top_hits")
            if isinstance(top_hits, list):
                for hit in top_hits:
                    if not isinstance(hit, dict):
                        continue
                    rank = hit.get("rank", "?")
                    name = hit.get("name", "")
                    entity_id = hit.get("entity_id", "")
                    entity_type = hit.get("entity_type", "")
                    page = hit.get("page_number", "")
                    score = hit.get("score", "")
                    print(
                        f"  {rank}. {name} [{entity_id}] type={entity_type} page={page} score={score}"
                    )
            return

        if event_type == "lexical_search_merged":
            unique_entities = event.get("unique_entities", 0)
            returned_entities = event.get("returned_entities", 0)
            print(f"lexical> merged unique={unique_entities} returned={returned_entities}")
            top_entities = event.get("top_entities")
            if isinstance(top_entities, list):
                for ent in top_entities:
                    if not isinstance(ent, dict):
                        continue
                    rank = ent.get("rank", "?")
                    name = ent.get("name", "")
                    entity_id = ent.get("entity_id", "")
                    score = ent.get("score", "")
                    matched_on = ent.get("matched_on", "")
                    print(f"  {rank}. {name} [{entity_id}] score={score} matched_on='{matched_on}'")

    return _logger


def _make_embedding_logger():
    def _logger(event: dict[str, object]) -> None:
        event_type = str(event.get("event", ""))
        if event_type == "embedding_search":
            top_entities = event.get("top_entities")
            print("embedding> semantic retrieval")
            if isinstance(top_entities, list):
                for row in top_entities:
                    if not isinstance(row, dict):
                        continue
                    print(
                        f"  {row.get('rank', '?')}. {row.get('name', '')} "
                        f"[{row.get('entity_id', '')}] score={row.get('embedding_score', '')}"
                    )
            return
        if event_type == "embedding_search_skipped":
            print(f"embedding> skipped: {event.get('reason', '')}")
            return
        if event_type == "embedding_batch_done":
            return

    return _logger


def _green(text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[32m{text}\033[0m"


if __name__ == "__main__":
    BUILD_KG = False
    BUILD_KG_DOC_ID = "pid_demo"
    BUILD_KG_ENV_FILE = ".env"
    INGEST_KG = False
    DETECT_DISCREPANCIES = False
    build_settings = Settings.from_env(BUILD_KG_ENV_FILE)

    if BUILD_KG:
        build_summary = build_knowledge_graph(settings=build_settings, doc_id=BUILD_KG_DOC_ID)
        print(build_summary)
    if INGEST_KG:
        ingest_summary = ingest_extraction_dump(
            settings=build_settings,
            doc_id=BUILD_KG_DOC_ID,
            extraction_dump_path="output/document_extraction.json",
        )
        print(ingest_summary)
    if DETECT_DISCREPANCIES:
        report = detect_sop_discrepancies(settings=build_settings, doc_id=BUILD_KG_DOC_ID, stream_logs=True)
        path = write_discrepancy_dashboard(report, "output/discrepancy_dashboard.txt")
        print(path)


    main()
