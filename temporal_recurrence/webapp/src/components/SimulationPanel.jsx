import React from 'react';
import { Play, RotateCw } from 'lucide-react';

export default function SimulationPanel({
  params,
  setParams,
  presets = [],
  onApplyPreset,
  onRunSimulation,
  isLoading
}) {
  const handleChange = (field, value) => {
    setParams(prev => ({
      ...prev,
      [field]: typeof value === 'string' && !isNaN(Number(value)) ? Number(value) : value
    }));
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          Environment Parameters
        </span>

        <button
          className="btn btn-primary"
          onClick={onRunSimulation}
          disabled={isLoading}
          style={{ padding: '4px 10px', fontSize: '0.75rem' }}
        >
          {isLoading ? (
            <>
              <RotateCw size={12} style={{ animation: 'spin 1s linear infinite' }} />
              <span>Simulating...</span>
            </>
          ) : (
            <>
              <Play size={12} />
              <span>Run Simulation</span>
            </>
          )}
        </button>
      </div>

      <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
        
        {/* Preset Quick Select */}
        {presets.length > 0 && (
          <div>
            <label className="input-label">
              Sequence Preset
            </label>
            <select
              onChange={(e) => {
                const preset = presets.find(p => p.id === e.target.value);
                if (preset) onApplyPreset(preset);
              }}
              defaultValue="recurrence_standard"
            >
              {presets.map(p => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10, marginTop: 2 }}>
          
          {/* Node Count N */}
          <div className="slider-group">
            <div className="slider-header">
              <span className="input-label">Node Count (N)</span>
              <span className="slider-value">{params.num_nodes}</span>
            </div>
            <input
              type="range"
              min={20}
              max={120}
              step={10}
              value={params.num_nodes}
              onChange={(e) => handleChange('num_nodes', parseInt(e.target.value))}
            />
          </div>

          {/* Community Count K */}
          <div className="slider-group">
            <div className="slider-header">
              <span className="input-label">Communities (K)</span>
              <span className="slider-value">{params.num_communities}</span>
            </div>
            <input
              type="range"
              min={2}
              max={4}
              step={1}
              value={params.num_communities}
              onChange={(e) => handleChange('num_communities', parseInt(e.target.value))}
            />
          </div>

          {/* Duration A1 */}
          <div className="slider-group">
            <div className="slider-header">
              <span className="input-label">Duration T_A1 (steps)</span>
              <span className="slider-value">{params.duration_a1}</span>
            </div>
            <input
              type="range"
              min={10}
              max={60}
              step={5}
              value={params.duration_a1}
              onChange={(e) => handleChange('duration_a1', parseInt(e.target.value))}
            />
          </div>

          {/* Intervening Duration TB */}
          <div className="slider-group">
            <div className="slider-header">
              <span className="input-label">Intervening T_B (steps)</span>
              <span className="slider-value">{params.duration_b}</span>
            </div>
            <input
              type="range"
              min={10}
              max={80}
              step={5}
              value={params.duration_b}
              onChange={(e) => handleChange('duration_b', parseInt(e.target.value))}
            />
          </div>

          {/* Persistence Lambda A */}
          <div className="slider-group">
            <div className="slider-header">
              <span className="input-label">Persistence λ_A</span>
              <span className="slider-value">{params.lambda_a}</span>
            </div>
            <input
              type="range"
              min={0.4}
              max={0.95}
              step={0.05}
              value={params.lambda_a}
              onChange={(e) => handleChange('lambda_a', parseFloat(e.target.value))}
            />
          </div>

          {/* Persistence Lambda B */}
          <div className="slider-group">
            <div className="slider-header">
              <span className="input-label">Persistence λ_B</span>
              <span className="slider-value">{params.lambda_b}</span>
            </div>
            <input
              type="range"
              min={0.05}
              max={0.6}
              step={0.05}
              value={params.lambda_b}
              onChange={(e) => handleChange('lambda_b', parseFloat(e.target.value))}
            />
          </div>

        </div>

      </div>
    </div>
  );
}
