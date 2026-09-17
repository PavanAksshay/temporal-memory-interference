import React from 'react';
import { Activity, BookOpen } from 'lucide-react';

export default function Header({ activeTab, setActiveTab, currentRegime, timestep, totalTimesteps, isConnected }) {
  const getRegimeBadge = () => {
    if (!currentRegime) return null;
    if (currentRegime === 'A') {
      return (
        <span className="badge badge-regime-a">
          <span style={{ width: 5, height: 5, borderRadius: 1, background: '#818cf8' }} />
          REGIME A • HIGH PERSISTENCE
        </span>
      );
    }
    if (currentRegime === 'B') {
      return (
        <span className="badge badge-regime-b">
          <span style={{ width: 5, height: 5, borderRadius: 1, background: '#fb7185' }} />
          REGIME B • VOLATILE DRIFT
        </span>
      );
    }
    return (
      <span className="badge badge-regime-c">
        <span style={{ width: 5, height: 5, borderRadius: 1, background: '#22d3ee' }} />
        REGIME {currentRegime}
      </span>
    );
  };

  return (
    <header style={{
      background: 'var(--bg-surface)',
      borderBottom: '1px solid var(--border-subtle)',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      <div style={{
        maxWidth: 1600,
        margin: '0 auto',
        padding: '8px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        
        {/* Brand & Project Metadata */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 26,
            height: 26,
            borderRadius: 'var(--radius-xs)',
            background: 'var(--bg-surface-elevated)',
            border: '1px solid var(--border-strong)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.75rem',
            fontWeight: 700,
            color: 'var(--text-primary)'
          }}>
            TR
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h1 style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
                Temporal Graph Recurrence
              </h1>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>/</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>DSBM Research Environment</span>
            </div>
          </div>
        </div>

        {/* Dynamic State Info */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {getRegimeBadge()}
          {totalTimesteps > 0 && (
            <div style={{
              fontSize: '0.75rem',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-secondary)',
              background: 'var(--bg-input)',
              padding: '2px 8px',
              borderRadius: 'var(--radius-xs)',
              border: '1px solid var(--border-subtle)'
            }}>
              t = <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{timestep}</span> / {totalTimesteps - 1}
            </div>
          )}
          <div style={{
            fontSize: '0.75rem',
            display: 'inline-flex',
            alignItems: 'center',
            gap: 5,
            color: isConnected ? 'var(--status-success)' : 'var(--status-danger)'
          }}>
            <span style={{
              width: 5,
              height: 5,
              borderRadius: '50%',
              background: isConnected ? 'var(--status-success)' : 'var(--status-danger)'
            }} />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
              {isConnected ? 'API Connected' : 'Offline'}
            </span>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav style={{
          display: 'flex',
          background: 'var(--bg-input)',
          padding: 2,
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border-subtle)'
        }}>
          <button
            onClick={() => setActiveTab('simulator')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 10px',
              borderRadius: 'var(--radius-xs)',
              border: 'none',
              background: activeTab === 'simulator' ? 'var(--bg-surface-elevated)' : 'transparent',
              color: activeTab === 'simulator' ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontSize: '0.75rem',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            <Activity size={13} />
            Simulator &amp; Topology
          </button>
          
          <button
            onClick={() => setActiveTab('explorer')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 10px',
              borderRadius: 'var(--radius-xs)',
              border: 'none',
              background: activeTab === 'explorer' ? 'var(--bg-surface-elevated)' : 'transparent',
              color: activeTab === 'explorer' ? 'var(--text-primary)' : 'var(--text-secondary)',
              fontSize: '0.75rem',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            <BookOpen size={13} />
            Benchmark Audit
          </button>
        </nav>

      </div>
    </header>
  );
}
