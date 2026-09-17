import React, { useRef, useEffect, useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

const COMMUNITY_COLORS = [
  '#38bdf8', // Cyan/Sky
  '#a78bfa', // Purple/Violet
  '#34d399', // Emerald/Green
  '#fb923c', // Orange
  '#f472b6'  // Pink
];

export default function GraphCanvas({
  nodes = [],
  currentFrame = null,
  currentRegime = 'A',
  numCommunities = 3
}) {
  const canvasRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Main canvas render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !nodes.length) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.clearRect(0, 0, width, height);
    ctx.save();

    // Subtle background grid
    ctx.strokeStyle = '#18181b';
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Apply zoom & pan transformation (center at 300, 300)
    ctx.translate(width / 2 + pan.x, height / 2 + pan.y);
    ctx.scale(zoom, zoom);
    ctx.translate(-300, -300);

    // 1. Draw Community Cluster Boundaries (Subtle circles)
    for (let c = 0; c < numCommunities; c++) {
      const commNodes = nodes.filter(n => n.community === c);
      if (commNodes.length > 0) {
        const cx = commNodes[0].cx;
        const cy = commNodes[0].cy;
        const color = COMMUNITY_COLORS[c % COMMUNITY_COLORS.length];

        // Crisp community ring
        ctx.strokeStyle = `${color}25`;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(cx, cy, 95, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label
        ctx.fillStyle = `${color}aa`;
        ctx.font = '500 10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`COMMUNITY ${c}`, cx, cy - 80);
      }
    }

    // 2. Draw Active Edges for Current Frame
    if (currentFrame && currentFrame.edges) {
      const edges = currentFrame.edges;
      ctx.lineWidth = 1.0;

      for (let i = 0; i < edges.length; i++) {
        const e = edges[i];
        const u = nodes[e.source];
        const v = nodes[e.target];
        if (!u || !v) continue;

        const isSameComm = u.community === v.community;
        ctx.beginPath();
        ctx.moveTo(u.x, u.y);
        ctx.lineTo(v.x, v.y);

        if (isSameComm) {
          const commColor = COMMUNITY_COLORS[u.community % COMMUNITY_COLORS.length];
          ctx.strokeStyle = `${commColor}60`;
        } else {
          // Cross-community edge
          ctx.strokeStyle = currentRegime === 'B' ? 'rgba(244, 63, 94, 0.45)' : 'rgba(161, 161, 170, 0.15)';
        }
        ctx.stroke();
      }
    }

    // 3. Draw Nodes
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const isHovered = hoveredNode && hoveredNode.id === n.id;
      const color = COMMUNITY_COLORS[n.community % COMMUNITY_COLORS.length];

      ctx.beginPath();
      ctx.arc(n.x, n.y, isHovered ? 6 : 3.5, 0, 2 * Math.PI);
      ctx.fillStyle = isHovered ? '#ffffff' : color;
      ctx.fill();

      // Border
      ctx.lineWidth = 1;
      ctx.strokeStyle = isHovered ? '#ffffff' : '#09090b';
      ctx.stroke();
    }

    ctx.restore();
  }, [nodes, currentFrame, currentRegime, hoveredNode, zoom, pan, numCommunities]);

  // Handle canvas mouse move for tooltips and dragging
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const width = canvas.width;
    const height = canvas.height;
    const transformedX = (mouseX - (width / 2 + pan.x)) / zoom + 300;
    const transformedY = (mouseY - (height / 2 + pan.y)) / zoom + 300;

    let found = null;
    for (let n of nodes) {
      const dist = Math.hypot(n.x - transformedX, n.y - transformedY);
      if (dist < 8) {
        found = n;
        break;
      }
    }
    setHoveredNode(found);
  };

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div className="panel" style={{ position: 'relative', overflow: 'hidden', height: '100%', minHeight: 460, display: 'flex', flexDirection: 'column' }}>
      
      {/* Panel Header */}
      <div className="panel-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Dynamic Graph Topology
          </span>
          {currentFrame && (
            <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
              ({currentFrame.num_edges} active edges)
            </span>
          )}
        </div>

        {/* View Controls */}
        <div style={{ display: 'flex', gap: 4 }}>
          <button className="btn-icon" onClick={() => setZoom(z => Math.min(2.5, z + 0.2))} title="Zoom In">
            <ZoomIn size={13} />
          </button>
          <button className="btn-icon" onClick={() => setZoom(z => Math.max(0.5, z - 0.2))} title="Zoom Out">
            <ZoomOut size={13} />
          </button>
          <button className="btn-icon" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} title="Reset View">
            <RotateCcw size={13} />
          </button>
        </div>
      </div>

      {/* Main Graph Canvas */}
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: isDragging ? 'grabbing' : 'crosshair' }}>
        <canvas
          ref={canvasRef}
          width={580}
          height={400}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => { handleMouseUp(); setHoveredNode(null); }}
          style={{ width: '100%', height: '100%', maxHeight: 400 }}
        />
      </div>

      {/* Node Tooltip */}
      {hoveredNode && (
        <div style={{
          position: 'absolute',
          bottom: 45,
          left: 14,
          background: 'var(--bg-surface-elevated)',
          border: '1px solid var(--border-strong)',
          borderRadius: 'var(--radius-xs)',
          padding: '6px 10px',
          fontSize: '0.75rem',
          fontFamily: 'var(--font-mono)',
          pointerEvents: 'none',
          zIndex: 20
        }}>
          <div>Node ID: <strong style={{ color: 'var(--text-primary)' }}>#{hoveredNode.id}</strong></div>
          <div style={{ color: COMMUNITY_COLORS[hoveredNode.community % COMMUNITY_COLORS.length] }}>
            Community: C_{hoveredNode.community}
          </div>
        </div>
      )}

      {/* Community Legend Footer */}
      <div style={{
        padding: '6px 14px',
        borderTop: '1px solid var(--border-subtle)',
        background: 'var(--bg-surface)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '0.75rem'
      }}>
        <div style={{ display: 'flex', gap: 12 }}>
          {Array.from({ length: numCommunities }).map((_, c) => (
            <div key={c} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <span style={{
                width: 6,
                height: 6,
                borderRadius: 'var(--radius-xs)',
                background: COMMUNITY_COLORS[c % COMMUNITY_COLORS.length]
              }} />
              <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: '0.7rem' }}>
                Community {c}
              </span>
            </div>
          ))}
        </div>
        <div style={{ color: 'var(--text-tertiary)', fontSize: '0.7rem' }}>
          Pan: Drag • Zoom: Controls
        </div>
      </div>
    </div>
  );
}
