"use client";

import { FormEvent, useMemo, useState } from "react";

import GraphNetwork from "../components/GraphNetwork";
import { apiPost, streamPostNdjson } from "../lib/api";

type TabKey = "extract" | "query" | "discrepancies" | "graph";

type BuildResponse = {
  summary: Record<string, unknown>;
  logs: Array<Record<string, unknown>>;
};

type QueryResponse = {
  answer: string;
  matched_entities?: Array<Record<string, unknown>>;
  tool_events?: Array<Record<string, unknown>>;
  lexical_events?: Array<Record<string, unknown>>;
  embedding_events?: Array<Record<string, unknown>>;
};

type DiscrepancyReport = {
  discrepancy_count?: number;
  evaluations?: Array<Record<string, unknown>>;
};

type DiscrepancyResponse = {
  report: DiscrepancyReport;
  logs: string[];
};

type GraphNode = {
  node_id: number;
  name: string;
  entity_id: string;
  entity_type: string;
  degree: number;
  raw_text?: string;
  description?: string;
  attributes_json?: string;
};

type GraphEdge = {
  source: number;
  target: number;
  edge_count: number;
};

type GraphOverviewResponse = {
  metrics: { entities?: number; relations?: number; measurements?: number };
  entity_types: Array<{ entity_type: string; count: number }>;
  type_links: Array<{ source_type: string; target_type: string; edge_count: number; examples?: string[] }>;
  top_entities: Array<{ name: string; entity_type: string; degree: number }>;
  network: { nodes: GraphNode[]; edges: GraphEdge[] };
};

type ActivityKind = "info" | "tool" | "warn";

type ActivityItem = {
  id: string;
  ts: string;
  kind: ActivityKind;
  message: string;
};

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: "extract", label: "Extract" },
  { key: "query", label: "Query" },
  { key: "discrepancies", label: "Find Discrepancies" },
  { key: "graph", label: "Graph Overview" },
];

function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>;
}

function LoadingBlock({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="loading-card" role="status" aria-live="polite">
      <div className="loading-title-row">
        <span className="dot-spinner" />
        <strong>{title}</strong>
      </div>
      <p>{detail}</p>
      <div className="skeleton-lines">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}

function getDiscrepancyIssues(item: Record<string, unknown>): string[] {
  const issues: string[] = [];
  const pressureStatus = String(item.pressure_status || "");

  if (pressureStatus === "exceeds_mwp") issues.push("Pressure exceeds MWP.");
  if (pressureStatus === "missing_entity") issues.push("Pressure check failed due to unresolved entity.");

  return issues;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function compactJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<TabKey>("extract");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [envFile, setEnvFile] = useState(".env");
  const [docId, setDocId] = useState("pid_demo");

  const [extractMode, setExtractMode] = useState<"build" | "ingest">("build");
  const [pidPdfPath, setPidPdfPath] = useState("data/pid/diagram.pdf");
  const [extractionDumpPath, setExtractionDumpPath] = useState("artifacts/document_extraction.json");
  const [entityIndexPath, setEntityIndexPath] = useState("");
  const [buildEmbeddings, setBuildEmbeddings] = useState(true);
  const [extractResult, setExtractResult] = useState<BuildResponse | null>(null);

  const [question, setQuestion] = useState("What is the PSV set pressure on F-715A?");
  const [maxToolCalls, setMaxToolCalls] = useState(10);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);

  const [sopPath, setSopPath] = useState("data/sop/sop.docx");
  const [dashboardPath, setDashboardPath] = useState("artifacts/discrepancy_dashboard.txt");
  const [maxToolCallsDisc, setMaxToolCallsDisc] = useState(6);
  const [discrepancyResult, setDiscrepancyResult] = useState<DiscrepancyResponse | null>(null);

  const [maxLinks, setMaxLinks] = useState(80);
  const [includeUnknown, setIncludeUnknown] = useState(false);
  const [maxNodes, setMaxNodes] = useState(80);
  const [maxEdges, setMaxEdges] = useState(180);
  const [graphOverview, setGraphOverview] = useState<GraphOverviewResponse | null>(null);

  const [toolCallCount, setToolCallCount] = useState(0);
  const [activeOperation, setActiveOperation] = useState<string>("");
  const [activityEvents, setActivityEvents] = useState<ActivityItem[]>([]);

  function pushActivity(kind: ActivityKind, message: string) {
    const item: ActivityItem = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      ts: new Date().toLocaleTimeString(),
      kind,
      message,
    };
    setActivityEvents((prev) => [...prev, item].slice(-240));
  }

  function startRun(operation: string) {
    setError(null);
    setLoading(true);
    setActiveOperation(operation);
    setToolCallCount(0);
    setActivityEvents([]);
    pushActivity("info", `${operation} started.`);
  }

  function finishRun() {
    setLoading(false);
    pushActivity("info", `${activeOperation || "Operation"} completed.`);
  }

  const discrepancyCards = useMemo(() => {
    const evaluations = discrepancyResult?.report?.evaluations;
    if (!Array.isArray(evaluations)) return [];
    return evaluations.map((item) => {
      const row = item as Record<string, unknown>;
      const notes = Array.isArray(row.notes) ? row.notes.map((x) => String(x)) : [];
      return {
        name: String(row.sop_name || "Unnamed SOP Item"),
        issues: getDiscrepancyIssues(row),
        notes,
      };
    });
  }, [discrepancyResult]);

  const recentActivity = useMemo(() => {
    return [...activityEvents].slice(-14).reverse();
  }, [activityEvents]);

  async function handleExtract(event: FormEvent) {
    event.preventDefault();
    startRun(extractMode === "build" ? "KG Build" : "Extraction Ingestion");
    setExtractResult(null);

    const logs: Array<Record<string, unknown>> = [];
    let summary: Record<string, unknown> | null = null;
    let streamError: string | null = null;

    try {
      const path = extractMode === "build" ? "/api/extract/build/stream" : "/api/extract/ingest/stream";
      const payload = {
        env_file: envFile,
        doc_id: docId,
        pid_pdf_path: extractMode === "build" ? pidPdfPath : null,
        extraction_dump_path: extractionDumpPath,
        entity_index_path: entityIndexPath || null,
        build_embeddings: buildEmbeddings,
      };

      await streamPostNdjson(path, payload, (evt) => {
        const type = String(evt.type || "");

        if (type === "error") {
          streamError = String(evt.error || "Unknown stream error");
          pushActivity("warn", streamError);
          return;
        }

        if (type === "status") {
          pushActivity("info", String(evt.message || "Working..."));
          return;
        }

        if (type === "build_log") {
          const payloadRecord = asRecord(evt.payload);
          if (!payloadRecord) return;
          logs.push(payloadRecord);
          const evName = String(payloadRecord.event || "event");
          const page = payloadRecord.page_number ? ` page ${payloadRecord.page_number}` : "";
          pushActivity("info", `${evName}${page}`);
          return;
        }

        if (type === "tool_call") {
          setToolCallCount((v) => v + 1);
          const payloadRecord = asRecord(evt.payload);
          const toolName = payloadRecord?.tool_name ? ` (${String(payloadRecord.tool_name)})` : "";
          pushActivity("tool", `Tool call observed${toolName}.`);
          return;
        }

        if (type === "result") {
          const dataRecord = asRecord(evt.data);
          const summaryRecord = dataRecord ? asRecord(dataRecord.summary) : null;
          if (summaryRecord) summary = summaryRecord;
        }
      });

      if (streamError) throw new Error(streamError);
      if (!summary) throw new Error("No summary returned from extraction endpoint.");

      setExtractResult({ summary, logs });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      finishRun();
    }
  }

  async function handleQuery(event: FormEvent) {
    event.preventDefault();
    startRun("Query");
    setQueryResult(null);

    let finalResult: QueryResponse | null = null;
    let streamError: string | null = null;

    try {
      await streamPostNdjson("/api/query/stream", {
        env_file: envFile,
        doc_id: docId,
        question,
        entity_index_path: entityIndexPath || null,
        max_tool_calls: maxToolCalls,
      }, (evt) => {
        const type = String(evt.type || "");

        if (type === "error") {
          streamError = String(evt.error || "Unknown stream error");
          pushActivity("warn", streamError);
          return;
        }

        if (type === "status") {
          pushActivity("info", String(evt.message || "Working..."));
          return;
        }

        if (type === "lexical_event") {
          const payload = asRecord(evt.payload) || {};
          const term = payload.term ? ` term=${String(payload.term)}` : "";
          pushActivity("info", `Lexical retrieval${term}`);
          return;
        }

        if (type === "embedding_event") {
          pushActivity("info", "Embedding retrieval event.");
          return;
        }

        if (type === "tool_event") {
          const payload = asRecord(evt.payload) || {};
          setToolCallCount((v) => v + 1);
          const callIndex = payload.call_index ? `#${String(payload.call_index)}` : "";
          const cypher = payload.cypher ? String(payload.cypher).slice(0, 92) : "";
          pushActivity("tool", `KG tool call ${callIndex}: ${cypher}`);
          return;
        }

        if (type === "result") {
          const data = asRecord(evt.data);
          if (data) finalResult = data as unknown as QueryResponse;
        }
      });

      if (streamError) throw new Error(streamError);
      if (!finalResult) throw new Error("No response returned from query endpoint.");

      setQueryResult(finalResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      finishRun();
    }
  }

  async function handleDiscrepancy(event: FormEvent) {
    event.preventDefault();
    startRun("Discrepancy Analysis");
    setDiscrepancyResult(null);

    const logs: string[] = [];
    let report: DiscrepancyReport | null = null;
    let streamError: string | null = null;

    try {
      await streamPostNdjson("/api/discrepancies/stream", {
        env_file: envFile,
        doc_id: docId,
        sop_docx_path: sopPath,
        entity_index_path: entityIndexPath || null,
        max_tool_calls_per_item: maxToolCallsDisc,
        dashboard_path: dashboardPath,
      }, (evt) => {
        const type = String(evt.type || "");

        if (type === "error") {
          streamError = String(evt.error || "Unknown stream error");
          pushActivity("warn", streamError);
          return;
        }

        if (type === "status") {
          pushActivity("info", String(evt.message || "Working..."));
          return;
        }

        if (type === "discrepancy_log") {
          const line = String(evt.message || "").trim();
          if (line) {
            logs.push(line);
            if (line.includes("KG tool call #")) {
              setToolCallCount((v) => v + 1);
              pushActivity("tool", line);
            } else {
              pushActivity("info", line);
            }
          }
          return;
        }

        if (type === "tool_call") {
          setToolCallCount((v) => v + 1);
          const payload = asRecord(evt.payload);
          const message = payload?.line ? String(payload.line) : "Tool call observed.";
          pushActivity("tool", message);
          return;
        }

        if (type === "result") {
          const data = asRecord(evt.data);
          const reportRecord = data ? asRecord(data.report) : null;
          if (reportRecord) report = reportRecord;
        }
      });

      if (streamError) throw new Error(streamError);
      if (!report) throw new Error("No report returned from discrepancy endpoint.");

      setDiscrepancyResult({ report, logs });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      finishRun();
    }
  }

  async function handleGraphLoad(event: FormEvent) {
    event.preventDefault();
    startRun("Graph Load");
    setGraphOverview(null);

    try {
      pushActivity("info", "Fetching graph metrics and network nodes...");
      const response = await apiPost<GraphOverviewResponse>("/api/graph/overview", {
        env_file: envFile,
        doc_id: docId,
        max_links: maxLinks,
        include_unknown: includeUnknown,
        max_nodes: maxNodes,
        max_edges: maxEdges,
      });
      pushActivity("info", `Loaded ${response.network?.nodes?.length || 0} nodes and ${response.network?.edges?.length || 0} edges.`);
      setGraphOverview(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      finishRun();
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <div>
          <h1>P&ID Knowledge Graph App</h1>
          <p>FastAPI + Next.js for extraction, querying, discrepancy checks, and graph exploration.</p>
        </div>
      </header>

      <section className="card controls">
        <div className="grid-two">
          <label>
            Env file
            <input value={envFile} onChange={(e) => setEnvFile(e.target.value)} />
          </label>
          <label>
            doc_id
            <input value={docId} onChange={(e) => setDocId(e.target.value)} />
          </label>
        </div>
      </section>

      <nav className="tabs">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            className={activeTab === tab.key ? "tab active" : "tab"}
            onClick={() => setActiveTab(tab.key)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <section className={loading ? "activity-dock live" : "activity-dock"}>
        <div className="activity-head">
          <div className="orbit-wrap" aria-hidden>
            <span className="orbit" />
            <span className="orbit-rotator">
              <span className="orbit-dot" />
            </span>
          </div>
          <div>
            <h3>{loading ? `${activeOperation || "Operation"} in progress` : "Run Activity"}</h3>
            <p>
              {loading ? "Live updates are streaming." : "Idle."} Tool calls observed: <strong>{toolCallCount}</strong>
            </p>
          </div>
        </div>
        <div className="pulse-track" aria-hidden>
          <span />
        </div>
        <ul className="activity-feed">
          {recentActivity.length ? (
            recentActivity.map((item) => (
              <li key={item.id} className={`activity-item ${item.kind}`}>
                <span className="activity-ts">{item.ts}</span>
                <span className="activity-msg">{item.message}</span>
              </li>
            ))
          ) : (
            <li className="activity-item info">
              <span className="activity-ts">-</span>
              <span className="activity-msg">No activity yet.</span>
            </li>
          )}
        </ul>
      </section>

      {error ? <div className="error">{error}</div> : null}

      {activeTab === "extract" ? (
        <section className="card">
          <h2>Extract / Ingest</h2>
          <form onSubmit={handleExtract} className="stack">
            <label>
              Mode
              <select value={extractMode} onChange={(e) => setExtractMode(e.target.value as "build" | "ingest")}>
                <option value="build">Build From PDF</option>
                <option value="ingest">Ingest Existing Extraction</option>
              </select>
            </label>
            <label>
              PID PDF path
              <input value={pidPdfPath} onChange={(e) => setPidPdfPath(e.target.value)} />
            </label>
            <label>
              Extraction dump path
              <input value={extractionDumpPath} onChange={(e) => setExtractionDumpPath(e.target.value)} />
            </label>
            <label>
              Entity index path (optional)
              <input
                value={entityIndexPath}
                onChange={(e) => setEntityIndexPath(e.target.value)}
                placeholder="Use default if empty"
              />
            </label>
            <label className="inline-checkbox">
              <input checked={buildEmbeddings} type="checkbox" onChange={(e) => setBuildEmbeddings(e.target.checked)} />
              Build embeddings
            </label>
            <button disabled={loading} className="primary" type="submit">
              {loading ? "Running..." : "Run"}
            </button>
          </form>

          {loading ? <LoadingBlock title="Processing extraction" detail="Rendering pages, making zoom tool calls, and ingesting graph updates." /> : null}

          {extractResult ? (
            <div className="result-stack">
              <h3>Summary</h3>
              <JsonBlock value={extractResult.summary} />
              <h3>Logs</h3>
              <JsonBlock value={extractResult.logs} />
            </div>
          ) : null}
        </section>
      ) : null}

      {activeTab === "query" ? (
        <section className="card">
          <h2>Query</h2>
          <form onSubmit={handleQuery} className="stack">
            <label>
              Question
              <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={5} />
            </label>
            <label>
              Entity index path (optional)
              <input
                value={entityIndexPath}
                onChange={(e) => setEntityIndexPath(e.target.value)}
                placeholder="Use default if empty"
              />
            </label>
            <label>
              Max KG tool calls
              <input
                type="number"
                min={1}
                max={30}
                value={maxToolCalls}
                onChange={(e) => setMaxToolCalls(Number(e.target.value) || 1)}
              />
            </label>
            <button disabled={loading} className="primary" type="submit">
              {loading ? "Running..." : "Ask"}
            </button>
          </form>

          {loading ? <LoadingBlock title="Reasoning over KG" detail="The model is retrieving entities and issuing Cypher tool calls." /> : null}

          {queryResult ? (
            <div className="result-stack">
              <h3>Answer</h3>
              <div className="answer-block">{queryResult.answer || "No answer returned."}</div>
              <details>
                <summary>Matched Entities</summary>
                <JsonBlock value={queryResult.matched_entities || []} />
              </details>
              <details>
                <summary>Tool Calls</summary>
                <JsonBlock value={queryResult.tool_events || []} />
              </details>
              <details>
                <summary>Lexical Search Logs</summary>
                <JsonBlock value={queryResult.lexical_events || []} />
              </details>
              <details>
                <summary>Embedding Search Logs</summary>
                <JsonBlock value={queryResult.embedding_events || []} />
              </details>
            </div>
          ) : null}
        </section>
      ) : null}

      {activeTab === "discrepancies" ? (
        <section className="card">
          <h2>Find Discrepancies</h2>
          <form onSubmit={handleDiscrepancy} className="stack">
            <label>
              SOP DOCX path
              <input value={sopPath} onChange={(e) => setSopPath(e.target.value)} />
            </label>
            <label>
              Dashboard output path
              <input value={dashboardPath} onChange={(e) => setDashboardPath(e.target.value)} />
            </label>
            <label>
              Entity index path (optional)
              <input
                value={entityIndexPath}
                onChange={(e) => setEntityIndexPath(e.target.value)}
                placeholder="Use default if empty"
              />
            </label>
            <label>
              Max tool calls per SOP item
              <input
                type="number"
                min={1}
                max={30}
                value={maxToolCallsDisc}
                onChange={(e) => setMaxToolCallsDisc(Number(e.target.value) || 1)}
              />
            </label>
            <button disabled={loading} className="primary" type="submit">
              {loading ? "Running..." : "Run Discrepancy Check"}
            </button>
          </form>

          {loading ? <LoadingBlock title="Checking operating limits" detail="Running per-item SOP checks and validating observed values from KG evidence." /> : null}

          {discrepancyResult ? (
            <div className="result-stack">
              <h3>Discrepancies: {discrepancyResult.report.discrepancy_count || 0}</h3>
              <div className="cards-grid">
                {discrepancyCards.map((card, idx) => (
                  <article className="item-card" key={`${card.name}-${idx}`}>
                    <h4>{card.name}</h4>
                    <p className="muted">Discrepancies</p>
                    {card.issues.length ? (
                      <ul>
                        {card.issues.map((issue) => (
                          <li key={issue}>{issue}</li>
                        ))}
                      </ul>
                    ) : (
                      <p>No discrepancies.</p>
                    )}
                    <p className="muted">Notes</p>
                    {card.notes.length ? (
                      <ul>
                        {card.notes.map((note, noteIdx) => (
                          <li key={`${idx}-${noteIdx}`}>{note}</li>
                        ))}
                      </ul>
                    ) : (
                      <p>No notes.</p>
                    )}
                  </article>
                ))}
              </div>

              <details>
                <summary>Runtime Logs</summary>
                <JsonBlock value={discrepancyResult.logs || []} />
              </details>
            </div>
          ) : null}
        </section>
      ) : null}

      {activeTab === "graph" ? (
        <section className="card">
          <h2>Graph Overview</h2>
          <form onSubmit={handleGraphLoad} className="stack">
            <div className="grid-two">
              <label>
                Max type links
                <input type="number" min={10} max={500} value={maxLinks} onChange={(e) => setMaxLinks(Number(e.target.value) || 80)} />
              </label>
              <label>
                Max nodes
                <input type="number" min={10} max={300} value={maxNodes} onChange={(e) => setMaxNodes(Number(e.target.value) || 80)} />
              </label>
              <label>
                Max edges
                <input type="number" min={10} max={1000} value={maxEdges} onChange={(e) => setMaxEdges(Number(e.target.value) || 180)} />
              </label>
              <label className="inline-checkbox">
                <input checked={includeUnknown} type="checkbox" onChange={(e) => setIncludeUnknown(e.target.checked)} />
                Include Unknown entities
              </label>
            </div>
            <button disabled={loading} className="primary" type="submit">
              {loading ? "Loading..." : "Load Graph"}
            </button>
          </form>

          {loading ? <LoadingBlock title="Loading graph view" detail="Fetching graph metrics and node-link structure from Neo4j." /> : null}

          {graphOverview ? (
            <div className="result-stack">
              <div className="metric-row">
                <div className="metric">
                  <span>Entities</span>
                  <strong>{graphOverview.metrics?.entities || 0}</strong>
                </div>
                <div className="metric">
                  <span>Connections</span>
                  <strong>{graphOverview.metrics?.relations || 0}</strong>
                </div>
                <div className="metric">
                  <span>Measurements</span>
                  <strong>{graphOverview.metrics?.measurements || 0}</strong>
                </div>
              </div>

              <h3>Entity Connection Graph</h3>
              <p className="muted">
                Bubble = entity, line = CONNECTED_TO relation. Hover bubbles/lines for details. Drag, pan, and zoom to inspect clusters.
              </p>
              <GraphNetwork nodes={graphOverview.network?.nodes || []} edges={graphOverview.network?.edges || []} />

              <details>
                <summary>Entity Types</summary>
                <JsonBlock value={graphOverview.entity_types || []} />
              </details>
              <details>
                <summary>Top Connected Entities</summary>
                <JsonBlock value={graphOverview.top_entities || []} />
              </details>
              <details>
                <summary>Type-to-Type Connections</summary>
                <JsonBlock value={graphOverview.type_links || []} />
              </details>
            </div>
          ) : null}
        </section>
      ) : null}

      <details className="debug-drawer">
        <summary>Debug: Last Activity Events (raw)</summary>
        <JsonBlock value={recentActivity.map((item) => ({ ...item, preview: compactJson(item.message).slice(0, 200) }))} />
      </details>
    </main>
  );
}
