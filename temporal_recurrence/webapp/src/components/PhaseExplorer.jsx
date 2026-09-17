import React, { useState, useEffect } from 'react';
import { fetchPhases, fetchPhaseDetails } from '../services/api';
import { Check, Maximize2, X, RotateCw } from 'lucide-react';

export default function PhaseExplorer() {
  const [phases, setPhases] = useState([]);
  const [selectedPhaseId, setSelectedPhaseId] = useState('phase4_5');
  const [phaseData, setPhaseData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeFigureModal, setActiveFigureModal] = useState(null);

  useEffect(() => {
    loadPhases();
  }, []);

  useEffect(() => {
    if (selectedPhaseId) {
      loadPhaseDetails(selectedPhaseId);
    }
  }, [selectedPhaseId]);

  const loadPhases = async () => {
    try {
      const res = await fetchPhases();
      setPhases(res.phases || []);
    } catch (err) {
      console.error('Error fetching phases:', err);
    }
  };

  const loadPhaseDetails = async (id) => {
    setIsLoading(true);
    try {
      const res = await fetchPhaseDetails(id);
      setPhaseData(res);
    } catch (err) {
      console.error('Error loading phase details:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 14 }}>
      
      {/* Left Column: Phase Navigation List */}
      <div className="panel" style={{ height: 'calc(100vh - 100px)', overflowY: 'auto' }}>
        <div className="panel-header">
          <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            Experimental Phases
          </span>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
            {phases.length} Total
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {phases.map((p) => {
            const isSelected = p.id === selectedPhaseId;
            return (
              <div
                key={p.id}
                onClick={() => setSelectedPhaseId(p.id)}
                style={{
                  padding: '9px 12px',
                  cursor: 'pointer',
                  background: isSelected ? 'var(--bg-surface-elevated)' : 'transparent',
                  borderBottom: '1px solid var(--border-subtle)',
                  borderLeft: isSelected ? '2px solid var(--text-primary)' : '2px solid transparent',
                  transition: 'background 0.1s ease'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                  <span style={{
                    fontSize: '0.8125rem',
                    fontWeight: isSelected ? 600 : 400,
                    color: isSelected ? 'var(--text-primary)' : 'var(--text-secondary)'
                  }}>
                    {p.title.split(':')[0]}
                  </span>
                  {p.has_verdict && (
                    <span style={{ fontSize: '0.65rem', color: 'var(--status-success)', fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Check size={11} /> VERDICT
                    </span>
                  )}
                </div>
                <p style={{
                  fontSize: '0.7rem',
                  color: 'var(--text-tertiary)',
                  lineHeight: 1.3,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }}>
                  {p.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Right Column: Active Phase Deep-Dive */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        
        {isLoading ? (
          <div className="panel" style={{ padding: 40, textAlign: 'center', color: 'var(--text-tertiary)' }}>
            <RotateCw size={18} style={{ animation: 'spin 1s linear infinite', margin: '0 auto 8px auto' }} />
            <div style={{ fontSize: '0.8125rem' }}>Loading artifacts for {selectedPhaseId}...</div>
          </div>
        ) : phaseData ? (
          <>
            {/* Scientific Verdict Banner (if available) */}
            {phaseData.verdict && (
              <div className="panel" style={{ padding: '12px 16px', borderLeft: '3px solid var(--status-success)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--status-success)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                      PHASE {phaseData.verdict.phase} VERDICT
                    </div>
                    <h2 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                      {phaseData.verdict.final_classification}
                    </h2>
                  </div>
                  {phaseData.verdict.execution_time_seconds && (
                    <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)' }}>
                      Runtime: {phaseData.verdict.execution_time_seconds.toFixed(1)}s
                    </span>
                  )}
                </div>

                {/* Benchmark Performance Grid */}
                {phaseData.verdict.hard_benchmark_results_tb100 && (
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                    gap: 8,
                    marginTop: 10,
                    paddingTop: 10,
                    borderTop: '1px solid var(--border-subtle)'
                  }}>
                    {Object.entries(phaseData.verdict.hard_benchmark_results_tb100).map(([k, v]) => (
                      <div key={k} style={{ background: 'var(--bg-input)', padding: '6px 10px', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                          {k.replace('_ap', '').replace(/_/g, ' ')}
                        </div>
                        <div style={{ fontSize: '1rem', fontWeight: 600, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', marginTop: 2 }}>
                          {typeof v === 'number' ? v.toFixed(3) : v}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Generated Figures Gallery */}
            {phaseData.figures && phaseData.figures.length > 0 && (
              <div className="panel">
                <div className="panel-header">
                  <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Generated Figures ({phaseData.figures.length})
                  </span>
                </div>

                <div style={{ padding: 14, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 }}>
                  {phaseData.figures.map((fig, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: 'var(--bg-input)',
                        border: '1px solid var(--border-subtle)',
                        borderRadius: 'var(--radius-xs)',
                        overflow: 'hidden',
                        cursor: 'pointer'
                      }}
                      onClick={() => setActiveFigureModal(fig.url)}
                    >
                      <div style={{ position: 'relative', height: 160, background: '#09090b', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <img
                          src={fig.url}
                          alt={fig.name}
                          style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                        />
                        <div style={{
                          position: 'absolute',
                          top: 6,
                          right: 6,
                          background: 'var(--bg-surface)',
                          border: '1px solid var(--border-subtle)',
                          padding: 3,
                          borderRadius: 'var(--radius-xs)',
                          color: 'var(--text-secondary)'
                        }}>
                          <Maximize2 size={11} />
                        </div>
                      </div>
                      <div style={{ padding: '6px 10px', fontSize: '0.7rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {fig.name}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Benchmark CSV Tables */}
            {phaseData.csv_tables && Object.keys(phaseData.csv_tables).length > 0 && (
              <div className="panel">
                <div className="panel-header">
                  <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Benchmark Data Tables
                  </span>
                </div>

                <div style={{ padding: 14 }}>
                  {Object.entries(phaseData.csv_tables).map(([tableName, rows]) => (
                    <div key={tableName} style={{ marginBottom: 14 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 6, fontFamily: 'var(--font-mono)' }}>
                        {tableName}
                      </div>
                      {rows.length > 0 && (
                        <div style={{ overflowX: 'auto', maxHeight: 200, border: '1px solid var(--border-subtle)' }}>
                          <table className="data-table">
                            <thead>
                              <tr>
                                {Object.keys(rows[0]).map((h) => (
                                  <th key={h}>{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {rows.map((r, i) => (
                                <tr key={i}>
                                  {Object.values(r).map((val, vi) => (
                                    <td key={vi}>
                                      {String(val)}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

          </>
        ) : null}

      </div>

      {/* Modal for Figure Inspection */}
      {activeFigureModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.85)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: 20
        }} onClick={() => setActiveFigureModal(null)}>
          <div style={{ position: 'relative', maxWidth: '90vw', maxHeight: '90vh' }} onClick={e => e.stopPropagation()}>
            <button
              onClick={() => setActiveFigureModal(null)}
              style={{
                position: 'absolute',
                top: -12,
                right: -12,
                background: 'var(--bg-surface-elevated)',
                border: '1px solid var(--border-strong)',
                color: 'var(--text-primary)',
                borderRadius: 'var(--radius-xs)',
                width: 24,
                height: 24,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer'
              }}
            >
              <X size={14} />
            </button>
            <img
              src={activeFigureModal}
              alt="Figure Preview"
              style={{ maxWidth: '90vw', maxHeight: '85vh', border: '1px solid var(--border-strong)' }}
            />
          </div>
        </div>
      )}

    </div>
  );
}
