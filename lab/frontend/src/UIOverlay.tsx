import React, { useMemo, useState, useEffect } from 'react'

type Props = {
  mode: 'single' | 'double'
  controller: 'pid' | 'nn'
  running: boolean
  onStart: () => void
  onStop: () => void
  onTrainStart: () => void
  onTrainStop: () => void
  onChangeMode: (m: 'single' | 'double') => void
  onChangeController: (c: 'pid' | 'nn') => void
  kp: number
  ki: number
  kd: number
  onPidChange: (p: { kp?: number; ki?: number; kd?: number }) => void
  onSavePolicy: (name?: string) => void
  onLoadPolicy: (name?: string) => void
  fps?: number
}

export default function UIOverlay({
  mode,
  controller,
  running,
  onStart,
  onStop,
  onTrainStart,
  onTrainStop,
  onChangeMode,
  onChangeController,
  fps,
  kp,
  ki,
  kd,
  onPidChange,
  onSavePolicy,
  onLoadPolicy,
}: Props) {
  const [target, setTarget] = useState(0)
  const [policyName, setPolicyName] = useState('default')

  useEffect(() => {
    setTarget(0)
  }, [mode])

  return (
    <div style={overlayStyle}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <label style={labelStyle}>
          Mode
          <select value={mode} onChange={(e) => onChangeMode(e.target.value as any)}>
            <option value="single">Single</option>
            <option value="double">Double</option>
          </select>
        </label>
        <label style={labelStyle}>
          Controller
          <select value={controller} onChange={(e) => onChangeController(e.target.value as any)}>
            <option value="pid">PID</option>
            <option value="nn">NN</option>
          </select>
        </label>
        <label style={labelStyle}>
          Target
          <input type="number" value={target} step="0.1" onChange={(e) => setTarget(parseFloat(e.target.value))} />
        </label>
        <button onClick={onStart} disabled={running} style={buttonStyle}>
          Start
        </button>
        <button onClick={onStop} disabled={!running} style={buttonStyle}>
          Stop
        </button>
        <button onClick={onTrainStart} style={buttonStyle}>
          Train Start
        </button>
        <button onClick={onTrainStop} style={buttonStyle}>
          Train Stop
        </button>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input value={policyName} onChange={(e) => setPolicyName(e.target.value)} style={{ width: 100 }} />
          <button onClick={() => onSavePolicy(policyName)} style={buttonStyle}>
            Save Policy
          </button>
          <button onClick={() => onLoadPolicy(policyName)} style={buttonStyle}>
            Load Policy
          </button>
        </div>
      </div>
      {controller === 'pid' && (
        <div style={{ marginTop: 8 }}>
          <div>
            <label>KP: {kp.toFixed(2)}</label>
            <input type="range" min="0" max="200" step="0.1" value={kp} onChange={(e) => onPidChange({ kp: parseFloat(e.target.value) })} />
          </div>
          <div>
            <label>KI: {ki.toFixed(3)}</label>
            <input type="range" min="0" max="5" step="0.001" value={ki} onChange={(e) => onPidChange({ ki: parseFloat(e.target.value) })} />
          </div>
          <div>
            <label>KD: {kd.toFixed(3)}</label>
            <input type="range" min="0" max="10" step="0.01" value={kd} onChange={(e) => onPidChange({ kd: parseFloat(e.target.value) })} />
          </div>
        </div>
      )}
      <div style={{ marginTop: 8, display: 'flex', gap: 12 }}>
        <div style={statBox}>FPS: {fps ? fps.toFixed(1) : '-'}</div>
        <div style={statBox}>Mode: {mode}</div>
        <div style={statBox}>Controller: {controller}</div>
      </div>
    </div>
  )
}

const overlayStyle: React.CSSProperties = {
  position: 'absolute',
  left: 12,
  top: 12,
  padding: 12,
  background: 'rgba(255,255,255,0.9)',
  borderRadius: 8,
  boxShadow: '0 6px 18px rgba(0,0,0,0.2)',
  zIndex: 20,
}

const labelStyle: React.CSSProperties = { display: 'flex', flexDirection: 'column', fontSize: 12 }
const buttonStyle: React.CSSProperties = { padding: '6px 10px', cursor: 'pointer' }
const statBox: React.CSSProperties = { padding: '6px 8px', background: '#f4f4f4', borderRadius: 6 }
