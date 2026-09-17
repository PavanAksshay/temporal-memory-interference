const API_BASE = import.meta.env.VITE_API_BASE || import.meta.env.VITE_API_URL || 'http://localhost:8008/api';

export async function fetchPresets() {
  const res = await fetch(`${API_BASE}/presets`);
  if (!res.ok) throw new Error('Failed to load presets');
  return res.json();
}

export async function runSimulation(params) {
  const res = await fetch(`${API_BASE}/simulate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(params)
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Simulation request failed' }));
    throw new Error(errorData.detail || 'Simulation error');
  }
  return res.json();
}

export async function fetchPhases() {
  const res = await fetch(`${API_BASE}/phases`);
  if (!res.ok) throw new Error('Failed to load phases');
  return res.json();
}

export async function fetchPhaseDetails(phaseId) {
  const res = await fetch(`${API_BASE}/phases/${phaseId}`);
  if (!res.ok) throw new Error(`Failed to load phase ${phaseId}`);
  return res.json();
}

export function getFigureUrl(phaseId, filename) {
  return `${API_BASE}/figures/${phaseId}/${filename}`;
}
