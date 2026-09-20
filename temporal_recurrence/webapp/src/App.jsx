import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import GraphCanvas from './components/GraphCanvas';
import TimelineScrubber from './components/TimelineScrubber';
import MetricsDashboard from './components/MetricsDashboard';
import AttentionHeatmap from './components/AttentionHeatmap';
import SimulationPanel from './components/SimulationPanel';
import PhaseExplorer from './components/PhaseExplorer';
import PaperTables from './components/PaperTables';
import { fetchPresets, runSimulation } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('simulator');
  const [isConnected, setIsConnected] = useState(false);
  const [presets, setPresets] = useState([]);

  // Simulation Parameters State
  const [params, setParams] = useState({
    num_nodes: 60,
    num_communities: 3,
    sequence_type: 'recurrence',
    duration_a1: 25,
    duration_b: 35,
    duration_a2: 25,
    target_density: 0.12,
    lambda_a: 0.85,
    lambda_b: 0.20,
    multiplier_a: 3.5,
    multiplier_b: 0.8,
    seed: 42
  });

  // Simulation Results Data
  const [simData, setSimData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Playback State
  const [timestep, setTimestep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  const playbackTimerRef = useRef(null);

  // On mount: fetch presets and run initial simulation
  useEffect(() => {
    loadPresetsAndInitialData();
  }, []);

  const loadPresetsAndInitialData = async () => {
    try {
      const pRes = await fetchPresets();
      setPresets(pRes.presets || []);
      setIsConnected(true);
    } catch (e) {
      console.warn('API presets not reachable:', e);
      setIsConnected(false);
    }

    // Run initial simulation
    handleRunSimulation();
  };

  const handleRunSimulation = async (customParams = null) => {
    setIsLoading(true);
    setIsPlaying(false);
    try {
      const data = await runSimulation(customParams || params);
      setSimData(data);
      setTimestep(0);
      setIsConnected(true);
    } catch (err) {
      console.error('Failed to run simulation:', err);
      setIsConnected(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleApplyPreset = (preset) => {
    const nextParams = {
      ...params,
      sequence_type: preset.sequence_type,
      num_nodes: preset.num_nodes || 60,
      duration_a1: preset.duration_a1,
      duration_b: preset.duration_b,
      duration_a2: preset.duration_a2,
      lambda_a: preset.lambda_a,
      lambda_b: preset.lambda_b,
      multiplier_a: preset.multiplier_a,
      multiplier_b: preset.multiplier_b
    };
    setParams(nextParams);
    handleRunSimulation(nextParams);
  };

  // Playback timer ticker
  useEffect(() => {
    if (isPlaying && simData && simData.frames) {
      const intervalMs = Math.max(80, Math.floor(400 / playbackSpeed));
      playbackTimerRef.current = setInterval(() => {
        setTimestep((prev) => {
          if (prev >= simData.frames.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, intervalMs);
    } else {
      if (playbackTimerRef.current) clearInterval(playbackTimerRef.current);
    }

    return () => {
      if (playbackTimerRef.current) clearInterval(playbackTimerRef.current);
    };
  }, [isPlaying, simData, playbackSpeed]);

  const currentFrame = simData && simData.frames ? simData.frames[timestep] : null;
  const currentRegime = currentFrame ? currentFrame.regime : 'A';
  const totalTimesteps = simData && simData.frames ? simData.frames.length : 0;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        currentRegime={currentRegime}
        timestep={timestep}
        totalTimesteps={totalTimesteps}
        isConnected={isConnected}
      />

      {/* Main Workspace Area */}
      <main style={{ flex: 1, maxWidth: 1600, width: '100%', margin: '0 auto', padding: '14px 20px' }}>
        
        {activeTab === 'simulator' ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            
            {/* Top Row: Graph Viewport (Left) + Metrics & Real-time Curves (Right) */}
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(420px, 1.1fr) 1.2fr', gap: 12 }}>
              
              {/* Left Column: Dynamic Topology Graph */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <GraphCanvas
                  nodes={simData ? simData.nodes : []}
                  currentFrame={currentFrame}
                  currentRegime={currentRegime}
                  numCommunities={simData?.metadata?.num_communities || params.num_communities}
                />

                {/* Scrubber right below graph */}
                <TimelineScrubber
                  timestep={timestep}
                  totalTimesteps={totalTimesteps}
                  onTimestepChange={setTimestep}
                  isPlaying={isPlaying}
                  setIsPlaying={setIsPlaying}
                  playbackSpeed={playbackSpeed}
                  setPlaybackSpeed={setPlaybackSpeed}
                  timeline={simData ? simData.timeline : []}
                />
              </div>

              {/* Right Column: Synchronized Multi-Metric Curves */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <MetricsDashboard
                  metrics={simData ? simData.metrics : []}
                  currentTimestep={timestep}
                />
              </div>

            </div>

            {/* Bottom Row: Environment Parameters (Left) + Memory Retrieval Matrix (Right) */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: 12 }}>
              <SimulationPanel
                params={params}
                setParams={setParams}
                presets={presets}
                onApplyPreset={handleApplyPreset}
                onRunSimulation={() => handleRunSimulation()}
                isLoading={isLoading}
              />

              <AttentionHeatmap
                attentionSlices={simData ? simData.attention_slices : { query_steps: [], matrix: [] }}
                currentTimestep={timestep}
              />
            </div>

          </div>
        ) : activeTab === 'tables' ? (
          /* Research Paper Tables Tab */
          <PaperTables />
        ) : (
          /* Phase Explorer Tab */
          <PhaseExplorer />
        )}

      </main>

    </div>
  );
}
