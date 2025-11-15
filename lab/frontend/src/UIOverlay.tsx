import React, { useMemo, useState, useEffect } from 'react'
import StyledSelect from './StyledSelect'

type Props = {
  mode: 'single' | 'double'
  controller: 'pid' | 'nn'
  running: boolean
  // server now streams continuously; no start/stop from UI
  onTrainStart: () => void
  onTrainStop: () => void
  onChangeMode: (m: 'single' | 'double') => void
  onChangeController: (c: 'pid' | 'nn') => void
  target?: number
  onChangeTarget?: (t: number) => void
  kp: number
  ki: number
  kd: number
  onPidChange: (p: { kp?: number; ki?: number; kd?: number }) => void
  onSavePolicy: (name?: string) => void
  onLoadPolicy: (name?: string) => void
  nnFramework: 'numpy' | 'torch'
  onChangeNNFramework: (f: 'numpy' | 'torch') => void
  currentPolicyName?: string | null
  fps?: number
}

export default function UIOverlay({
  mode,
  controller,
  onTrainStart,
  onTrainStop,
  onChangeMode,
  onChangeController,
  target,
  onChangeTarget,
  fps,
  kp,
  ki,
  kd,
  onPidChange,
  onSavePolicy,
  onLoadPolicy,
  nnFramework,
  onChangeNNFramework,
  currentPolicyName,
}: Props) {
  const [policyName, setPolicyName] = useState('default')
  const displayTarget = target === undefined || target === null || Number.isNaN(target) ? '' : String(target)
  

  return (
    <div className="glass overlay">
      <div className="overlay-row">
        <label style={labelStyle}>
          Mode
          <StyledSelect value={mode} onChange={(v) => onChangeMode(v as any)} options={[{ value: 'single', label: 'Single' }, { value: 'double', label: 'Double' }]} />
        </label>
        <label style={labelStyle}>
          Controller
          <StyledSelect value={controller} onChange={(v) => onChangeController(v as any)} options={[{ value: 'pid', label: 'PID' }, { value: 'nn', label: 'NN' }]} />
        </label>
        <label style={labelStyle}>
          Target
          <input
            className="input focus-ring"
            type="number"
            value={displayTarget}
            step="0.05"
            onChange={(e) => {
              if (!onChangeTarget) return
              const v = parseFloat(e.target.value)
              onChangeTarget(Number.isNaN(v) ? 0 : v)
            }}
          />
        </label>
        <button className="btn fade-in" onClick={onTrainStart}>
          Train Start
        </button>
        <button className="btn fade-in" onClick={onTrainStop}>
          Train Stop
        </button>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input className="input" value={policyName} onChange={(e) => setPolicyName(e.target.value)} style={{ width: 100, minWidth: 80 }} />
          <button className="btn" onClick={() => onSavePolicy(policyName)}>
            Save Policy
          </button>
          <button className="btn" onClick={() => onLoadPolicy(policyName)}>
            Load Policy
          </button>
        </div>
      </div>
      {controller === 'nn' && (
        <div style={{ marginTop: 8 }}>
          <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            NN Framework:
            <StyledSelect value={nnFramework} onChange={(v) => onChangeNNFramework(v as any)} options={[{ value: 'numpy', label: 'Numpy' }, { value: 'torch', label: 'PyTorch' }]} />
          </label>
          <div style={{ marginTop: 6 }}>
            <strong>Loaded:</strong> {currentPolicyName ? currentPolicyName : 'none'}
          </div>
        </div>
      )}
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
        <div className="glass stat-box">FPS: {fps ? fps.toFixed(1) : '-'}</div>
        <div className="glass stat-box">Mode: {mode}</div>
        <div className="glass stat-box">Controller: {controller}</div>
      </div>
    </div>
  )
}

// Keep small inline positioning but rely on CSS for look-and-feel
const overlayStyle: React.CSSProperties = { position: 'absolute', left: 12, top: 12, zIndex: 20 }

const labelStyle: React.CSSProperties = { display: 'flex', flexDirection: 'column', fontSize: 12 }
const buttonStyle: React.CSSProperties = { padding: '6px 10px', cursor: 'pointer' }
