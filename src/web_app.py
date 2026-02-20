from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from kg_pipeline import (
    Settings,
    answer_with_knowledge_graph,
    build_knowledge_graph,
    detect_sop_discrepancies,
    ingest_extraction_dump,
)
from kg_pipeline.neo4j_store import Neo4jGraphStore


# -------------------------
# Helpers
# -------------------------

def _settings(env_file: str) -> Settings:
    return Settings.from_env(env_file)


def _event_logger(container: st.delta_generator.DeltaGenerator, label: str):
    def _logger(payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        container.text(f"{label}: {json.dumps(payload, ensure_ascii=True)}")

    return _logger


def _issues_for_item(item: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    pressure_status = str(item.get("pressure_status", ""))

    if pressure_status == "exceeds_mwp":
        issues.append("Pressure exceeds MWP.")
    elif pressure_status == "missing_entity":
        issues.append("Pressure check failed because entity resolution failed.")

    return issues


def _render_discrepancy_cards(report: dict[str, Any]) -> None:
    evaluations = report.get("evaluations", [])
    if not isinstance(evaluations, list) or not evaluations:
        st.info("No per-item evaluations available.")
        return

    st.markdown("### Per-Item Results")
    for item in evaluations:
        if not isinstance(item, dict):
            continue

        sop_name = str(item.get("sop_name", "Unnamed SOP Item"))
        issues = _issues_for_item(item)
        notes = item.get("notes", [])
        notes = [str(note).strip() for note in notes if str(note).strip()] if isinstance(notes, list) else []

        icon = "⚠️" if issues else "✅"
        with st.expander(f"{icon} {sop_name}", expanded=bool(issues)):
            st.markdown("**Discrepancies**")
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("No discrepancies.")

            st.markdown("**Notes**")
            if notes:
                for note in notes:
                    st.caption(f"- {note}")
            else:
                st.caption("- No notes.")


def _load_graph_overview(settings: Settings, doc_id: str, max_links: int) -> dict[str, Any]:
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


def _filter_type_links(
    type_links: list[dict[str, Any]],
    *,
    include_unknown: bool,
    include_self_loops: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in type_links:
        src = str(row.get("source_type", "Unknown"))
        tgt = str(row.get("target_type", "Unknown"))
        if not include_unknown and ("unknown" in src.lower() or "unknown" in tgt.lower()):
            continue
        if not include_self_loops and src == tgt:
            continue
        out.append(row)
    return out


def _build_type_heatmap(type_links: list[dict[str, Any]]) -> go.Figure:
    types = sorted(
        {
            str(row.get("source_type", "Unknown"))
            for row in type_links
        }.union(
            {str(row.get("target_type", "Unknown")) for row in type_links}
        )
    )
    if not types:
        return go.Figure()

    idx = {t: i for i, t in enumerate(types)}
    size = len(types)
    z = [[0.0 for _ in range(size)] for _ in range(size)]
    custom = [[{"examples": []} for _ in range(size)] for _ in range(size)]

    for row in type_links:
        src = str(row.get("source_type", "Unknown"))
        tgt = str(row.get("target_type", "Unknown"))
        cnt = float(row.get("edge_count", 0) or 0)
        i = idx[src]
        j = idx[tgt]
        z[i][j] = cnt
        custom[i][j] = {"examples": row.get("examples", [])}

    hover_text = []
    for i, src in enumerate(types):
        row_hover = []
        for j, tgt in enumerate(types):
            examples = custom[i][j].get("examples", [])
            ex_text = "<br>".join(examples[:3]) if isinstance(examples, list) and examples else "None"
            row_hover.append(
                f"Source type: {src}<br>"
                f"Target type: {tgt}<br>"
                f"Edge count: {int(z[i][j])}<br>"
                f"Examples:<br>{ex_text}"
            )
        hover_text.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=types,
            y=types,
            colorscale="Blues",
            hoverinfo="text",
            text=hover_text,
            colorbar=dict(title="Edge Count"),
        )
    )
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Target Entity Type",
        yaxis_title="Source Entity Type",
    )
    return fig


def _type_links_table_rows(type_links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in type_links:
        source_type = str(row.get("source_type", "Unknown"))
        target_type = str(row.get("target_type", "Unknown"))
        edge_count = int(row.get("edge_count", 0) or 0)
        examples = row.get("examples", [])
        if isinstance(examples, list):
            example_text = "; ".join(str(x) for x in examples[:3] if str(x).strip()) or "-"
        else:
            example_text = "-"
        rows.append(
            {
                "source_type": source_type,
                "target_type": target_type,
                "edge_count": edge_count,
                "example_entity_pairs": example_text,
            }
        )
    return rows


def _load_entity_network(
    settings: Settings,
    doc_id: str,
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


def _build_network_figure(network: dict[str, Any], title: str = "Entity Connection Graph") -> go.Figure:
    nodes = network.get("nodes", [])
    edges = network.get("edges", [])
    if not nodes:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    node_by_id = {int(n["node_id"]): n for n in nodes if n.get("node_id") is not None}
    neighbors: dict[int, set[int]] = {node_id: set() for node_id in node_by_id}
    for e in edges:
        src = int(e.get("source"))
        tgt = int(e.get("target"))
        if src in neighbors and tgt in neighbors:
            neighbors[src].add(tgt)
            neighbors[tgt].add(src)

    node_ids = list(node_by_id.keys())
    positions: dict[int, tuple[float, float]] = {}
    try:
        import networkx as nx  # type: ignore

        g = nx.Graph()
        g.add_nodes_from(node_ids)
        g.add_edges_from(
            (int(e["source"]), int(e["target"]))
            for e in edges
            if e.get("source") in node_by_id and e.get("target") in node_by_id
        )
        layout = nx.spring_layout(g, seed=42, k=max(0.35, 2.0 / math.sqrt(max(1, len(node_ids)))), iterations=150)
        positions = {int(k): (float(v[0]), float(v[1])) for k, v in layout.items()}
    except Exception:
        count = len(node_ids)
        radius = 1.0
        for i, node_id in enumerate(node_ids):
            angle = (2.0 * math.pi * i) / max(1, count)
            positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))

    # Edge segments
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_hover_x: list[float] = []
    edge_hover_y: list[float] = []
    edge_hover_text: list[str] = []
    for e in edges:
        src = int(e.get("source"))
        tgt = int(e.get("target"))
        if src not in positions or tgt not in positions:
            continue
        x0, y0 = positions[src]
        x1, y1 = positions[tgt]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_hover_x.append((x0 + x1) / 2.0)
        edge_hover_y.append((y0 + y1) / 2.0)
        a = node_by_id[src]
        b = node_by_id[tgt]
        a_name = str(a.get("name") or a.get("entity_id") or src)
        b_name = str(b.get("name") or b.get("entity_id") or tgt)
        edge_count = int(e.get("edge_count", 0) or 0)
        edge_hover_text.append(
            f"{a_name} <-> {b_name}<br>"
            f"Connection count: {edge_count}"
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color="rgba(180, 186, 199, 0.55)"),
        hoverinfo="skip",
        mode="lines",
        name="connections",
    )
    edge_hover_trace = go.Scatter(
        x=edge_hover_x,
        y=edge_hover_y,
        mode="markers",
        marker=dict(size=9, color="rgba(0,0,0,0)"),
        text=edge_hover_text,
        hovertemplate="%{text}<extra></extra>",
        name="connection details",
        showlegend=False,
    )

    palette = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#EECA3B",
        "#B279A2",
        "#FF9DA7",
        "#9D755D",
        "#BAB0AC",
    ]
    types = sorted({str(node_by_id[node_id].get("entity_type", "Unknown")) for node_id in node_ids})
    color_by_type = {t: palette[i % len(palette)] for i, t in enumerate(types)}

    node_traces: list[go.Scatter] = []
    for entity_type in types:
        x_vals: list[float] = []
        y_vals: list[float] = []
        hover_vals: list[str] = []
        sizes: list[float] = []
        labels: list[str] = []
        for node_id in node_ids:
            node = node_by_id[node_id]
            if str(node.get("entity_type", "Unknown")) != entity_type:
                continue
            x, y = positions[node_id]
            x_vals.append(x)
            y_vals.append(y)
            degree = len(neighbors.get(node_id, set()))
            sizes.append(12 + min(26, degree * 2))
            display_name = str(node.get("name") or node.get("entity_id") or node_id)
            labels.append(display_name)
            raw = str(node.get("raw_text", "")).strip()
            raw_preview = raw[:220] + ("..." if len(raw) > 220 else "")
            hover_vals.append(
                f"Name: {display_name}<br>"
                f"Entity ID: {str(node.get('entity_id', '') or '-')}<br>"
                f"Type: {entity_type}<br>"
                f"Degree: {degree}<br>"
                f"Raw text: {raw_preview or '-'}"
            )

        node_traces.append(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont=dict(size=10, color="#E7EBF5"),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_vals,
                marker=dict(
                    size=sizes,
                    color=color_by_type[entity_type],
                    line=dict(width=1.2, color="rgba(255,255,255,0.85)"),
                    opacity=0.95,
                ),
                name=entity_type,
            )
        )

    fig = go.Figure(data=[edge_trace, edge_hover_trace, *node_traces])
    fig.update_layout(
        title=title,
        template="plotly_dark",
        showlegend=True,
        legend_title_text="Entity Type",
        hovermode="closest",
        margin=dict(l=10, r=10, t=42, b=10),
        height=760,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


# -------------------------
# App UI
# -------------------------

st.set_page_config(page_title="P&ID KG App", layout="wide")
st.title("P&ID Knowledge Graph App")
st.caption("Extract graph data, query the KG, and evaluate SOP operating limits.")

with st.sidebar:
    st.subheader("Global Settings")
    env_file = st.text_input("Env file", value=".env")
    default_doc_id = st.text_input("Default doc_id", value="pid_demo")


tab_extract, tab_query, tab_discrepancy, tab_graph = st.tabs(
    ["Extract", "Query", "Find Discrepancies", "Graph Overview"]
)

with tab_extract:
    st.subheader("Extraction")
    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio("Mode", ["Build From PDF", "Ingest Existing Extraction"], horizontal=True)
        doc_id = st.text_input("doc_id", value=default_doc_id, key="extract_doc_id")
        build_embeddings = st.checkbox("Build embeddings", value=True)
    with c2:
        pid_pdf_path = st.text_input("PID PDF path", value="data/pid/diagram.pdf")
        extraction_dump_path = st.text_input("Extraction dump path", value="artifacts/document_extraction.json")

    log_box = st.empty()
    if st.button("Run Extraction/Ingestion", type="primary"):
        try:
            settings = _settings(env_file)
            logger = _event_logger(log_box, "build")
            if mode == "Build From PDF":
                summary = build_knowledge_graph(
                    settings=settings,
                    doc_id=doc_id,
                    pid_pdf_path=pid_pdf_path,
                    extraction_dump_path=extraction_dump_path,
                    build_embeddings=build_embeddings,
                    log_progress=True,
                    log_handler=logger,
                )
            else:
                summary = ingest_extraction_dump(
                    settings=settings,
                    doc_id=doc_id,
                    extraction_dump_path=extraction_dump_path,
                    build_embeddings=build_embeddings,
                    log_progress=True,
                    log_handler=logger,
                )
            st.success("Completed.")
            st.json(summary)
        except Exception as exc:
            st.error(str(exc))

with tab_query:
    st.subheader("Query")
    c1, c2 = st.columns([2, 1])
    with c1:
        query_text = st.text_area("Question", value="What is the PSV set pressure on F-715A?", height=120)
    with c2:
        doc_id_query = st.text_input("doc_id", value=default_doc_id, key="query_doc_id")
        max_tool_calls = st.number_input("Max KG tool calls", min_value=1, max_value=20, value=10, step=1)
        show_trace = st.checkbox("Show trace", value=False)

    query_log_box = st.empty()
    if st.button("Run Query", type="primary"):
        try:
            settings = _settings(env_file)
            result = answer_with_knowledge_graph(
                query_text,
                settings=settings,
                doc_id=doc_id_query,
                max_tool_calls=int(max_tool_calls),
                tool_event_logger=_event_logger(query_log_box, "tool"),
                lexical_event_logger=_event_logger(query_log_box, "lexical"),
                embedding_event_logger=_event_logger(query_log_box, "embedding"),
            )
            st.success("Answer generated.")
            st.markdown("### Answer")
            st.write(result.get("answer", ""))
            if show_trace:
                st.markdown("### Matched Entities")
                st.json(result.get("matched_entities", []))
                st.markdown("### Tool Events")
                st.json(result.get("tool_events", []))
        except Exception as exc:
            st.error(str(exc))

with tab_discrepancy:
    st.subheader("Find Discrepancies")
    c1, c2 = st.columns(2)
    with c1:
        doc_id_disc = st.text_input("doc_id", value=default_doc_id, key="disc_doc_id")
        sop_docx_path = st.text_input("SOP path", value="data/sop/sop.docx")
        dashboard_path = st.text_input("Dashboard output path", value="artifacts/discrepancy_dashboard.txt")
    with c2:
        max_tool_calls_disc = st.number_input("Max tool calls per SOP item", min_value=1, max_value=20, value=6, step=1)
        show_json_report = st.checkbox("Show full JSON report", value=False)

    if st.button("Run Discrepancy Check", type="primary"):
        try:
            settings = _settings(env_file)
            report = detect_sop_discrepancies(
                settings=settings,
                doc_id=doc_id_disc,
                sop_docx_path=sop_docx_path,
                stream_logs=True,
                max_tool_calls_per_item=int(max_tool_calls_disc),
                dashboard_path=dashboard_path,
            )
            st.success("Discrepancy check completed.")
            st.metric("Discrepancy count", int(report.get("discrepancy_count", 0)))
            _render_discrepancy_cards(report)

            dash = report.get("dashboard_path", dashboard_path)
            if Path(dash).exists():
                text = Path(dash).read_text(encoding="utf-8")
                st.download_button(
                    "Download Text Dashboard",
                    data=text,
                    file_name=Path(dash).name,
                    mime="text/plain",
                )
            if show_json_report:
                st.markdown("### JSON Report")
                st.json(report)
        except Exception as exc:
            st.error(str(exc))

with tab_graph:
    st.subheader("Knowledge Graph Overview")
    doc_id_graph = st.text_input("doc_id", value=default_doc_id, key="graph_doc_id")
    max_links = st.slider("Max type-to-type links", min_value=10, max_value=200, value=80, step=10)
    include_unknown = st.checkbox("Include Unknown entity type", value=False)
    include_self_loops = st.checkbox("Include self-loop type links (A -> A)", value=False)
    min_edge_count = st.slider("Minimum edge count per type link", min_value=1, max_value=50, value=1, step=1)
    graph_max_nodes = st.slider("Graph nodes (entities)", min_value=10, max_value=180, value=70, step=10)
    graph_max_edges = st.slider("Graph edges (connections)", min_value=20, max_value=500, value=180, step=20)

    if st.button("Load Graph Overview", type="primary"):
        try:
            settings = _settings(env_file)
            overview = _load_graph_overview(settings=settings, doc_id=doc_id_graph, max_links=max_links)
            network = _load_entity_network(
                settings=settings,
                doc_id=doc_id_graph,
                max_nodes=graph_max_nodes,
                max_edges=graph_max_edges,
                include_unknown=include_unknown,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Entities", overview["metrics"].get("entities", 0))
            c2.metric("Connections", overview["metrics"].get("relations", 0))
            c3.metric("Measurements", overview["metrics"].get("measurements", 0))

            st.markdown("### Entity Connection Graph")
            if network.get("nodes"):
                graph_fig = _build_network_figure(
                    network,
                    title="Entity Connection Graph (bubble = entity, line = CONNECTED_TO relation)",
                )
                st.plotly_chart(graph_fig, use_container_width=True)
                st.caption(
                    "Hover a bubble for entity details (including raw text preview), and hover a line midpoint for connection details."
                )
            else:
                st.info("No entity nodes found for the selected graph filters.")

            st.markdown("### Entity Types")
            if overview["entity_types"]:
                st.dataframe(overview["entity_types"], use_container_width=True)
            else:
                st.info("No entity types found.")

            st.markdown("### Type-to-Type Connection Matrix")
            raw_links = overview.get("type_links", [])
            filtered_links = _filter_type_links(
                raw_links,
                include_unknown=include_unknown,
                include_self_loops=include_self_loops,
            )
            filtered_links = [
                row for row in filtered_links if int(row.get("edge_count", 0) or 0) >= int(min_edge_count)
            ]
            if filtered_links:
                fig = _build_type_heatmap(filtered_links)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Rows are source entity types and columns are target entity types. "
                    "Hover any cell to see exact edge count and example entity pairs."
                )
            else:
                st.info("No type-level links found for the selected filters.")

            st.markdown("### Connection Details")
            detail_rows = _type_links_table_rows(filtered_links)
            if detail_rows:
                st.dataframe(detail_rows, use_container_width=True)
            else:
                st.info("No connection rows available.")

            st.markdown("### Top Connected Entities")
            if overview["top_entities"]:
                st.dataframe(overview["top_entities"], use_container_width=True)
            else:
                st.info("No connected entities found.")
        except Exception as exc:
            st.error(str(exc))
