import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
  CartesianGrid
} from 'recharts';

export default function MetricsDashboard({ metrics = [], currentTimestep = 0 }) {
  if (!metrics.length) {
    return (
      <div className="panel" style={{ padding: 20, textAlign: 'center', color: 'var(--text-tertiary)', fontSize: '0.8125rem' }}>
        No metric data available.
      </div>
    );
  }

  const currentMetric = metrics[currentTimestep] || metrics[0];
  const recencyGap = (currentMetric.delta_old || 0) - (currentMetric.delta_recent || 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      
      {/* Metrics Summary Strip (Unified, non-card layout) */}
      <div className="panel" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', padding: 0 }}>
        
        {/* Historical Oracle AP */}
        <div style={{ padding: '10px 14px', borderRight: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
            Historical Oracle AP
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--status-success)', marginTop: 2 }}>
            {(currentMetric.oracle_ap || 0).toFixed(3)}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', marginTop: 2 }}>
            G_t + historical A
          </div>
        </div>

        {/* Recent History AP */}
        <div style={{ padding: '10px 14px', borderRight: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
            Recent History AP
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--status-warning)', marginTop: 2 }}>
            {(currentMetric.recent_ap || 0).toFixed(3)}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', marginTop: 2 }}>
            Window G[t-h:t]
          </div>
        </div>

        {/* Current-Only AP */}
        <div style={{ padding: '10px 14px', borderRight: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
            Current-Only AP
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', marginTop: 2 }}>
            {(currentMetric.current_ap || 0).toFixed(3)}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', marginTop: 2 }}>
            Instantaneous G_t
          </div>
        </div>

        {/* Recency Discrepancy Gap */}
        <div style={{ padding: '10px 14px' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
            Recency Bias Gap (Δ_old - Δ_rec)
          </div>
          <div style={{
            fontSize: '1.25rem',
            fontWeight: 600,
            fontFamily: 'var(--font-mono)',
            color: recencyGap > 0 ? 'var(--status-success)' : 'var(--status-danger)',
            marginTop: 2
          }}>
            {recencyGap.toFixed(3)}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', marginTop: 2 }}>
            Historical margin
          </div>
        </div>

      </div>

      {/* Chart 1: Average Precision Forecast Comparison */}
      <div className="panel">
        <div className="panel-header">
          <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Link Prediction Forecast (AP over time)
          </span>
          <span style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
            Cursor: t = {currentTimestep}
          </span>
        </div>

        <div style={{ padding: '10px 14px', width: '100%', height: 185 }}>
          <ResponsiveContainer>
            <LineChart data={metrics} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="t" stroke="var(--text-tertiary)" fontSize={10} tickLine={false} />
              <YAxis domain={[0.3, 1.0]} stroke="var(--text-tertiary)" fontSize={10} tickLine={false} />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-surface-elevated)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-xs)',
                  fontSize: '0.75rem',
                  padding: '6px 10px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '0.7rem', paddingTop: 4 }} />
              
              <ReferenceLine x={currentTimestep} stroke="var(--border-strong)" strokeWidth={1} />
              
              <Line
                type="monotone"
                dataKey="oracle_ap"
                name="Historical Oracle"
                stroke="#10b981"
                strokeWidth={1.5}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="recent_ap"
                name="Recent History (h=5)"
                stroke="#f59e0b"
                strokeWidth={1.5}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="current_ap"
                name="Current Only"
                stroke="#a1a1aa"
                strokeWidth={1.2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Temporal Overlap vs Edge Density */}
      <div className="panel">
        <div className="panel-header">
          <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Continuity Metrics: Temporal Overlap O(t, t-1) vs Density ρ
          </span>
        </div>

        <div style={{ padding: '10px 14px', width: '100%', height: 145 }}>
          <ResponsiveContainer>
            <LineChart data={metrics} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="t" stroke="var(--text-tertiary)" fontSize={10} tickLine={false} />
              <YAxis domain={[0, 0.8]} stroke="var(--text-tertiary)" fontSize={10} tickLine={false} />
              <Tooltip
                contentStyle={{
                  background: 'var(--bg-surface-elevated)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-xs)',
                  fontSize: '0.75rem',
                  padding: '6px 10px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '0.7rem', paddingTop: 4 }} />
              
              <ReferenceLine x={currentTimestep} stroke="var(--border-strong)" strokeWidth={1} />

              <Line
                type="monotone"
                dataKey="overlap"
                name="Temporal Overlap O(t, t-1)"
                stroke="#818cf8"
                strokeWidth={1.5}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="density"
                name="Edge Density ρ(t)"
                stroke="#52525b"
                strokeWidth={1.2}
                strokeDasharray="3 3"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
