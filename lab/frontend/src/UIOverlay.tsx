import React, { useMemo, useState, useEffect } from 'react'
import StyledSelect from './StyledSelect'

type Props = {
  mode: 'single' | 'double'
  controller: 'pid' | 'nn'
  running: boolean
  // server now streams continuously; no start/stop from UI
  // server now streams continuously; training is automatic when NN is selected
  onChangeMode: (m: 'single' | 'double') => void
  onChangeController: (c: 'pid' | 'nn') => void
  target?: number
  onChangeTarget?: (t: number) => void
  useDegrees?: boolean
  onChangeUseDegrees?: (b: boolean) => void
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
  onSetTrack?: (len: number) => void
  currentTrack?: number | null
  trainingStats?: any
  trainerParams?: { population?: number; sigma?: number; alpha?: number; steps?: number }
  onTrainerParamsChange?: (p: { population?: number; sigma?: number; alpha?: number; steps?: number }) => void
}

export default function UIOverlay({
  mode,
  controller,
  onChangeMode,
  onChangeController,
  target,
  useDegrees,
  onChangeUseDegrees,
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
  onSetTrack,
  currentTrack,
  trainingStats,
  trainerParams,
  onTrainerParamsChange,
}: Props) {
  const [policyName, setPolicyName] = useState('default')
  const [trackLengthLocal, setTrackLengthLocal] = useState<string>('2.0')
  const [confirmedTrackLocal, setConfirmedTrackLocal] = useState<number | null>(null)
  const [localTrainerParams, setLocalTrainerParams] = useState<{ population?: number; sigma?: number; alpha?: number; steps?: number }>(
    () => ({ population: 12, sigma: 0.08, alpha: 0.04, steps: 100 })
  )
  useEffect(() => {
    // Sync confirmed track when parent prop changes (server confirmation)
    if (typeof currentTrack === 'number') {
      setConfirmedTrackLocal(currentTrack)
      setTrackLengthLocal(String(currentTrack))
    }
  }, [currentTrack])
  useEffect(() => {
    if (trainerParams) setLocalTrainerParams(trainerParams)
  }, [trainerParams])
  // support both controlled (via props) and uncontrolled (local) modes for the deg toggle
  const [localUseDegrees, setLocalUseDegrees] = useState(false)
  const effectiveUseDegrees = typeof useDegrees === 'boolean' ? useDegrees : localUseDegrees
  // show empty string when value is invalid
  const displayTarget = (() => {
    if (target === undefined || target === null || Number.isNaN(target)) return ''
    return effectiveUseDegrees ? String((target * 180 / Math.PI).toFixed(2)) : String(target)
  })()


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
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input
              className="input focus-ring"
              type="number"
              value={displayTarget}
              step={effectiveUseDegrees ? '0.5' : '0.01'}
              onChange={(e) => {
                if (!onChangeTarget) return
                const raw = e.target.value
                const v = parseFloat(raw)
                if (Number.isNaN(v)) {
                  onChangeTarget(0)
                  return
                }
                if (effectiveUseDegrees) {
                  // convert degrees -> radians
                  onChangeTarget((v * Math.PI) / 180)
                } else {
                  onChangeTarget(v)
                }
              }}
              style={{ width: 120 }}
            />
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
              <input
                type="checkbox"
                checked={effectiveUseDegrees}
                onChange={(e) => {
                  const v = e.target.checked
                  if (onChangeUseDegrees) onChangeUseDegrees(v)
                  else setLocalUseDegrees(v)
                }}
              />
              <span>deg</span>
            </label>
          </div>
        </label>
        {/* Training is automatic for NN controller; use save/load to snapshot policies */}
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input className="input" value={policyName} onChange={(e) => setPolicyName(e.target.value)} style={{ width: 100, minWidth: 80 }} />
          <button className="btn" onClick={() => onSavePolicy(policyName)}>
            Save Policy
          </button>
          <button className="btn" onClick={() => onLoadPolicy(policyName)}>
            Load Policy
          </button>
        </div>
        <label style={labelStyle}>
          Track Length
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input
              className="input focus-ring"
              type="number"
              value={trackLengthLocal}
              step={'0.1'}
              onChange={(e) => setTrackLengthLocal(e.target.value)}
              style={{ width: 120 }}
            />
            <button
              className="btn"
              onClick={() => {
                const v = parseFloat(trackLengthLocal)
                if (!Number.isFinite(v)) return
                if (onSetTrack) onSetTrack(v)
                // optimistic local confirmation until server reply arrives
                setConfirmedTrackLocal(v)
              }}
            >
              Set
            </button>
          </div>
        </label>
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
      {controller === 'nn' && (
        <div style={{ marginTop: 8, display: 'flex', gap: 12, alignItems: 'flex-start' }}>
          <div style={{ padding: 8 }} className="glass">
            <div style={{ fontSize: 12, marginBottom: 6 }}>Trainer Hyperparams</div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <label style={{ fontSize: 12 }}>Population</label>
              <input type="number" value={localTrainerParams.population} min={2} step={1} onChange={(e) => setLocalTrainerParams((s) => ({ ...s, population: parseInt(e.target.value || '0') }))} style={{ width: 80 }} />
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
              <label style={{ fontSize: 12 }}>Sigma</label>
              <input type="number" value={localTrainerParams.sigma} step={0.001} onChange={(e) => setLocalTrainerParams((s) => ({ ...s, sigma: parseFloat(e.target.value || '0') }))} style={{ width: 80 }} />
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
              <label style={{ fontSize: 12 }}>Alpha</label>
              <input type="number" value={localTrainerParams.alpha} step={0.001} onChange={(e) => setLocalTrainerParams((s) => ({ ...s, alpha: parseFloat(e.target.value || '0') }))} style={{ width: 80 }} />
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
              <label style={{ fontSize: 12 }}>Steps</label>
              <input type="number" value={localTrainerParams.steps} min={1} step={1} onChange={(e) => setLocalTrainerParams((s) => ({ ...s, steps: parseInt(e.target.value || '0') }))} style={{ width: 80 }} />
            </div>
            <div style={{ marginTop: 8 }}>
              <button
                className="btn"
                onClick={() => {
                  if (onTrainerParamsChange) onTrainerParamsChange(localTrainerParams)
                }}
              >
                Apply
              </button>
            </div>
          </div>

          <div style={{ padding: 8, width: 220 }} className="glass">
            <div style={{ fontSize: 12, marginBottom: 6 }}>Training Status</div>
            <div>Running: {trainingStats ? String(trainingStats.running ?? true) : 'unknown'}</div>
            <div>Iter: {trainingStats ? String(trainingStats.iter ?? '-') : '-'}</div>
            <div>Last reward: {trainingStats && typeof trainingStats.last_reward !== 'undefined' ? trainingStats.last_reward.toFixed(3) : '-'}</div>
            <div style={{ marginTop: 8 }}>
              <small>Reward history</small>
              <div style={{ width: 200, height: 48, marginTop: 6 }}>
                {trainingStats && Array.isArray(trainingStats.history) && trainingStats.history.length > 0 ? (
                  <svg viewBox="0 0 200 48" preserveAspectRatio="none" width="200" height="48">
                    {
                      (() => {
                        const hist: number[] = trainingStats.history.slice(-100)
                        const H = 48
                        const W = 200
                        let min = Math.min(...hist)
                        let max = Math.max(...hist)
                        if (min === max) {
                          // avoid zero range
                          min = min - 1
                          max = max + 1
                        }
                        const points = hist.map((v, i) => {
                          const x = (i / (hist.length - 1 || 1)) * W
                          const y = H - ((v - min) / (max - min)) * H
                          return `${x.toFixed(1)},${y.toFixed(1)}`
                        })
                        const poly = points.join(' ')
                        return (
                          <>
                            <polyline points={poly} fill="none" stroke="#38a169" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                            <rect x={0} y={0} width={W} height={H} fill="none" stroke="#222" opacity={0.06} />
                          </>
                        )
                      })()
                    }
                  </svg>
                ) : (
                  <div style={{ width: 200, height: 48, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666' }}>no data</div>
                )}
              </div>
            </div>
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
        <div className="glass stat-box">Track: {typeof currentTrack === 'number' ? currentTrack.toFixed(2) : (confirmedTrackLocal !== null ? confirmedTrackLocal.toFixed(2) : '-')}</div>
      </div>
    </div>
  )
}

// Keep small inline positioning but rely on CSS for look-and-feel
const overlayStyle: React.CSSProperties = { position: 'absolute', left: 12, top: 12, zIndex: 20 }

const labelStyle: React.CSSProperties = { display: 'flex', flexDirection: 'column', fontSize: 12 }
const buttonStyle: React.CSSProperties = { padding: '6px 10px', cursor: 'pointer' }
