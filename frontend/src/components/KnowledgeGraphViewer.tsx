// frontend/src/components/KnowledgeGraphViewer.tsx
import { useEffect, useRef, useState, useCallback } from "react";
import * as d3 from "d3";
import type { KnowledgeGraphData, NodeDetail } from "../services/api";
import { getNodeDetail } from "../services/api";
import "./KnowledgeGraphViewer.css";

type ViewMode = "hierarchy" | "force";

interface Props {
  kg: KnowledgeGraphData;
  docId: string;
}

export default function KnowledgeGraphViewer({ kg, docId }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("force");
  const [selectedNode, setSelectedNode] = useState<NodeDetail | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [highlightedNodeId, setHighlightedNodeId] = useState<string | null>(null);
  const [typeFilter, setTypeFilter] = useState<Set<string>>(new Set());

  const typeColors: Record<string, string> = {
    concept: "var(--primary, #4A90D9)",
    definition: "var(--forest, #2D8A6E)",
    algorithm: "var(--coral, #E8614D)",
    property: "var(--yellow, #E8B82D)",
    example: "var(--mint, #5BBFB5)",
  };

  const relationLabels: Record<string, string> = {
    prerequisite: "前置",
    leads_to: "推导",
    contrasts: "对比",
    extends: "扩展",
    part_of: "归属",
  };

  const handleNodeClick = useCallback(async (nodeId: string) => {
    try {
      const detail = await getNodeDetail(docId, nodeId);
      setSelectedNode(detail);
    } catch {
      setSelectedNode(null);
    }
  }, [docId]);

  const handleSearch = () => {
    if (!searchQuery.trim()) {
      setHighlightedNodeId(null);
      return;
    }
    const found = kg.nodes.find(
      n => n.name.includes(searchQuery) || searchQuery.includes(n.name)
    );
    setHighlightedNodeId(found?.id ?? null);
  };

  const toggleTypeFilter = (type: string) => {
    setTypeFilter(prev => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const filteredNodes = typeFilter.size > 0
    ? kg.nodes.filter(n => typeFilter.has(n.type))
    : kg.nodes;
  const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
  const filteredEdges = kg.edges.filter(
    e => filteredNodeIds.has(e.source_id) && filteredNodeIds.has(e.target_id)
  );

  useEffect(() => {
    if (!svgRef.current || viewMode !== "force") return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = svgRef.current.clientWidth || 900;
    const height = svgRef.current.clientHeight || 600;

    const g = svg.append("g");

    const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.2, 4]).on("zoom", (e) => {
      g.attr("transform", e.transform.toString());
    });
    svg.call(zoom);

    const simNodes = filteredNodes.map(n => ({ ...n, x: width / 2, y: height / 2 }));
    const nodeMap = new Map(simNodes.map(n => [n.id, n]));
    const simLinks = filteredEdges
      .filter(e => nodeMap.has(e.source_id) && nodeMap.has(e.target_id))
      .map(e => ({ source: e.source_id, target: e.target_id, ...e }));

    const simulation = d3.forceSimulation(simNodes)
      .force("link", d3.forceLink(simLinks).id((d: any) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = g.append("g").selectAll("line")
      .data(simLinks)
      .join("line")
      .attr("stroke", "var(--text-muted, #999)")
      .attr("stroke-width", 1)
      .attr("stroke-opacity", 0.4);

    const node = g.append("g").selectAll("g")
      .data(simNodes)
      .join("g")
      .style("cursor", "pointer")
      .on("click", (_e, d: any) => handleNodeClick(d.id));

    node.append("circle")
      .attr("r", (d: any) => 8 + d.importance * 16)
      .attr("fill", (d: any) => typeColors[d.type] || typeColors.concept)
      .attr("stroke", "var(--bg, #fff)")
      .attr("stroke-width", 2);

    node.append("text")
      .text((d: any) => d.name.length > 8 ? d.name.slice(0, 8) + "…" : d.name)
      .attr("dx", (d: any) => 12 + d.importance * 16)
      .attr("dy", 4)
      .attr("fill", "var(--text, #333)")
      .attr("font-size", "12px")
      .attr("font-family", "var(--font-body)");

    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      node.attr("transform", (d: any) => `translate(${d.x},${d.y})`);
    });

    return () => simulation.stop();
  }, [filteredNodes, filteredEdges, viewMode, handleNodeClick]);

  return (
    <div className="kg-viewer">
      <div className="kg-toolbar">
        <div className="kg-view-toggle">
          <button
            className={`btn btn-sm ${viewMode === "hierarchy" ? "" : "btn-outline"}`}
            onClick={() => setViewMode("hierarchy")}
          >
            层级
          </button>
          <button
            className={`btn btn-sm ${viewMode === "force" ? "" : "btn-outline"}`}
            onClick={() => setViewMode("force")}
          >
            关联图
          </button>
        </div>

        <div className="kg-search">
          <input
            type="text"
            placeholder="搜索概念..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSearch()}
          />
          <button className="btn btn-sm" onClick={handleSearch}>搜索</button>
        </div>

        <div className="kg-type-filters">
          {Object.entries(typeColors).map(([type, color]) => (
            <label key={type} className="kg-filter-label">
              <input
                type="checkbox"
                checked={typeFilter.size === 0 || typeFilter.has(type)}
                onChange={() => toggleTypeFilter(type)}
              />
              <span className="kg-filter-dot" style={{ background: color }} />
              {type}
            </label>
          ))}
        </div>
      </div>

      <div className="kg-main">
        <svg ref={svgRef} className="kg-canvas" />

        {selectedNode && (
          <aside className="kg-node-panel">
            <h3>{selectedNode.node.name}</h3>
            <div className="kg-node-meta">
              <span className="kg-badge" style={{ background: typeColors[selectedNode.node.type] }}>
                {selectedNode.node.type}
              </span>
              <span className="kg-importance">
                {"★".repeat(Math.round(selectedNode.node.importance * 5))}
                {"☆".repeat(5 - Math.round(selectedNode.node.importance * 5))}
              </span>
            </div>
            {selectedNode.node.summary && (
              <p className="kg-node-summary">{selectedNode.node.summary}</p>
            )}
            {selectedNode.node.source_sections.length > 0 && (
              <div className="kg-node-sections">
                <h4>来源章节</h4>
                {selectedNode.node.source_sections.map((s, i) => (
                  <span key={i} className="kg-section-tag">{s}</span>
                ))}
              </div>
            )}
            {selectedNode.neighbors.length > 0 && (
              <div className="kg-node-neighbors">
                <h4>相关概念</h4>
                {selectedNode.neighbors.map((nb, i) => (
                  <div key={i} className="kg-neighbor-item" onClick={() => handleNodeClick(nb.node.id)}>
                    <span className="kg-neighbor-name">{nb.node.name}</span>
                    <span className="kg-neighbor-relation">
                      {relationLabels[nb.relation] || nb.relation}
                    </span>
                  </div>
                ))}
              </div>
            )}
            <button className="btn btn-sm" onClick={() => setSelectedNode(null)}>关闭</button>
          </aside>
        )}
      </div>

      <div className="kg-legend">
        <span>图例:</span>
        {Object.entries(relationLabels).map(([key, label]) => (
          <span key={key} className="kg-legend-item">—{label}</span>
        ))}
      </div>
    </div>
  );
}
