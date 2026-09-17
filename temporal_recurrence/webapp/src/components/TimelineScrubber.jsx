import React from 'react';
import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight } from 'lucide-react';

export default function TimelineScrubber({
  timestep,
  totalTimesteps,
  onTimestepChange,
  isPlaying,
  setIsPlaying,
  playbackSpeed,
  setPlaybackSpeed,
  timeline = []
}) {
  const handleScrub = (e) => {
    const val = parseInt(e.target.value, 10);
    onTimestepChange(val);
  };

  const toggleSpeed = () => {
    const speeds = [0.5, 1, 2, 4];
    const nextIdx = (speeds.indexOf(playbackSpeed) + 1) % speeds.length;
    setPlaybackSpeed(speeds[nextIdx]);
  };

  return (
    <div className="panel" style={{ padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 8 }}>
      
      {/* Timeline Track with Segmented Regime Spans */}
      <div style={{ position: 'relative', width: '100%', height: 20, display: 'flex', alignItems: 'center' }}>
        
        {/* Background Regime Segments */}
        <div style={{
          position: 'absolute',
          top: 6,
          left: 0,
          right: 0,
          height: 8,
          borderRadius: 'var(--radius-xs)',
          overflow: 'hidden',
          display: 'flex',
          background: 'var(--bg-input)',
          border: '1px solid var(--border-subtle)'
        }}>
          {timeline.length > 0 && timeline.map((rec, i) => {
            const isRegimeA = rec.regime === 'A';
            const isRegimeB = rec.regime === 'B';
            return (
              <div
                key={i}
                style={{
                  flex: 1,
                  background: isRegimeA ? '#6366f1' : isRegimeB ? '#f43f5e' : '#06b6d4',
                  opacity: rec.t <= timestep ? 0.85 : 0.25
                }}
                title={`t=${rec.t} (${rec.regime})`}
              />
            );
          })}
        </div>

        {/* Range Slider Overlay */}
        <input
          type="range"
          min={0}
          max={Math.max(0, totalTimesteps - 1)}
          value={timestep}
          onChange={handleScrub}
          style={{
            position: 'absolute',
            width: '100%',
            zIndex: 5,
            background: 'transparent',
            margin: 0
          }}
        />
      </div>

      {/* Control Buttons and Timestep Readout */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        
        {/* Left: Playback Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <button
            className="btn-icon"
            onClick={() => onTimestepChange(0)}
            title="Jump to Start"
          >
            <SkipBack size={13} />
          </button>

          <button
            className="btn-icon"
            onClick={() => onTimestepChange(Math.max(0, timestep - 1))}
            title="Step Backward"
          >
            <ChevronLeft size={13} />
          </button>

          <button
            className="btn btn-primary"
            onClick={() => setIsPlaying(!isPlaying)}
            style={{ minWidth: 80, padding: '4px 10px', fontSize: '0.75rem' }}
          >
            {isPlaying ? <Pause size={13} /> : <Play size={13} />}
            <span>{isPlaying ? 'Pause' : 'Play'}</span>
          </button>

          <button
            className="btn-icon"
            onClick={() => onTimestepChange(Math.min(totalTimesteps - 1, timestep + 1))}
            title="Step Forward"
          >
            <ChevronRight size={13} />
          </button>

          <button
            className="btn-icon"
            onClick={() => onTimestepChange(totalTimesteps - 1)}
            title="Jump to End"
          >
            <SkipForward size={13} />
          </button>

          <button
            className="btn btn-secondary"
            onClick={toggleSpeed}
            style={{ padding: '3px 8px', fontSize: '0.7rem', fontFamily: 'var(--font-mono)' }}
            title="Playback Speed"
          >
            {playbackSpeed}x
          </button>
        </div>

        {/* Right: Legend */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 6, height: 6, borderRadius: 'var(--radius-xs)', background: '#6366f1' }} />
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.7rem' }}>Regime A (Persistent)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 6, height: 6, borderRadius: 'var(--radius-xs)', background: '#f43f5e' }} />
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.7rem' }}>Regime B (Volatile)</span>
          </div>
        </div>

      </div>
    </div>
  );
}
