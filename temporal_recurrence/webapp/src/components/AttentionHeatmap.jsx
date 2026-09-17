import React from 'react';

export default function AttentionHeatmap({ attentionSlices = { query_steps: [], matrix: [] }, currentTimestep = 0 }) {
  const { query_steps = [], matrix = [] } = attentionSlices;

  if (!matrix.length) {
    return null;
  }

  // Find the closest query index to the active currentTimestep
  let closestQueryIdx = 0;
  let minDiff = Infinity;
  for (let i = 0; i < query_steps.length; i++) {
    const diff = Math.abs(query_steps[i] - currentTimestep);
    if (diff < minDiff) {
      minDiff = diff;
      closestQueryIdx = i;
    }
  }

  const getHeatmapColor = (val, isPast) => {
    if (!isPast || val <= 0.001) return 'transparent';
    if (val < 0.04) return 'rgba(99, 102, 241, 0.15)';
    if (val < 0.10) return 'rgba(99, 102, 241, 0.35)';
    if (val < 0.20) return 'rgba(99, 102, 241, 0.65)';
    return '#6366f1';
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          Historical Memory Attention &amp; Retrieval Matrix
        </span>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)' }}>
          Query t_q vs Historical Key t_k
        </span>
      </div>

      <div style={{ padding: 14 }}>
        <div style={{ overflowX: 'auto', paddingBottom: 4 }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: `36px repeat(${query_steps.length}, 1fr)`,
            gap: 1.5,
            minWidth: 400
          }}>
            {/* Header Row */}
            <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', textAlign: 'center', fontFamily: 'var(--font-mono)' }}>t_q\t_k</div>
            {query_steps.map((k_t, colIdx) => (
              <div
                key={colIdx}
                style={{
                  fontSize: '0.65rem',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-tertiary)',
                  textAlign: 'center',
                  padding: '2px 0'
                }}
              >
                {k_t}
              </div>
            ))}

            {/* Matrix Rows */}
            {matrix.map((row, rowIdx) => {
              const isCurrentRow = rowIdx === closestQueryIdx;
              return (
                <React.Fragment key={rowIdx}>
                  {/* Row Label */}
                  <div style={{
                    fontSize: '0.65rem',
                    fontFamily: 'var(--font-mono)',
                    color: isCurrentRow ? 'var(--text-primary)' : 'var(--text-tertiary)',
                    fontWeight: isCurrentRow ? 700 : 400,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    {row.query_t}
                  </div>

                  {/* Cell Weights */}
                  {row.weights.map((w, colIdx) => {
                    const isPast = colIdx <= rowIdx;
                    return (
                      <div
                        key={colIdx}
                        style={{
                          height: 12,
                          borderRadius: 1,
                          background: getHeatmapColor(w, isPast),
                          border: isCurrentRow ? '1px solid var(--text-primary)' : '1px solid rgba(255, 255, 255, 0.02)'
                        }}
                        title={`Query t=${row.query_t} -> Key t=${query_steps[colIdx]} | Weight: ${w}`}
                      />
                    );
                  })}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: 10,
          paddingTop: 8,
          borderTop: '1px solid var(--border-subtle)',
          fontSize: '0.7rem',
          color: 'var(--text-tertiary)'
        }}>
          <div>Causal history matrix (no future leakage)</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>Low Attention</span>
            <div style={{ width: 48, height: 4, borderRadius: 1, background: 'linear-gradient(90deg, rgba(99, 102, 241, 0.15), #6366f1)' }} />
            <span>High Retrieval</span>
          </div>
        </div>
      </div>
    </div>
  );
}
