"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";

type NetworkNode = {
  node_id: number;
  name: string;
  entity_id: string;
  entity_type: string;
  degree?: number;
  raw_text?: string;
  description?: string;
  attributes_json?: string;
};

type NetworkEdge = {
  source: number;
  target: number;
  edge_count: number;
};

type Props = {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  height?: number;
};

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

export default function GraphNetwork({ nodes, edges, height = 680 }: Props) {
  const graphData = useMemo(() => {
    const mappedNodes = nodes.map((node) => ({
      id: node.node_id,
      label: node.name || node.entity_id || String(node.node_id),
      entity_id: node.entity_id,
      entity_type: node.entity_type || "Unknown",
      degree: node.degree || 0,
      raw_text: node.raw_text || "",
      description: node.description || "",
      attributes_json: node.attributes_json || "{}",
    }));

    const mappedLinks = edges.map((edge) => ({
      source: edge.source,
      target: edge.target,
      edge_count: edge.edge_count || 1,
    }));

    return { nodes: mappedNodes, links: mappedLinks };
  }, [nodes, edges]);

  return (
    <div className="graph-wrap" style={{ height }}>
      <ForceGraph2D
        graphData={graphData}
        nodeRelSize={7}
        cooldownTicks={120}
        linkWidth={(link: { edge_count?: number }) => Math.min(6, 1 + (link.edge_count || 1) * 0.2)}
        linkColor={() => "rgba(160, 168, 181, 0.6)"}
        nodeAutoColorBy="entity_type"
        nodeLabel={(node: any) => {
          const raw = String(node.raw_text || node.description || node.attributes_json || "");
          const rawPreview = raw.length > 220 ? `${raw.slice(0, 220)}...` : raw;
          return [
            `Name: ${node.label}`,
            `Entity ID: ${node.entity_id || "-"}`,
            `Type: ${node.entity_type || "Unknown"}`,
            `Degree: ${node.degree || 0}`,
            `Raw Text: ${rawPreview || "-"}`,
          ].join("\n");
        }}
        linkLabel={(link: any) => `Connection count: ${link.edge_count || 1}`}
        nodeCanvasObject={(node: any, ctx, globalScale) => {
          const label = node.label || String(node.id);
          const fontSize = Math.max(9, 12 / globalScale);
          ctx.beginPath();
          ctx.arc(node.x, node.y, 6 + Math.min(10, (node.degree || 0) * 0.8), 0, 2 * Math.PI, false);
          ctx.fillStyle = node.color;
          ctx.fill();
          ctx.strokeStyle = "rgba(240, 244, 255, 0.9)";
          ctx.lineWidth = Math.max(0.8, 1.2 / globalScale);
          ctx.stroke();

          ctx.font = `${fontSize}px sans-serif`;
          ctx.fillStyle = "rgba(234, 239, 250, 0.95)";
          ctx.fillText(label, node.x + 8, node.y + 3);
        }}
      />
    </div>
  );
}
