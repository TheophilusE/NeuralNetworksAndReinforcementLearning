import React, { useEffect, useRef, useState } from 'react'
import ThreeScene from './ThreeScene'
import UIOverlay from './UIOverlay'
import PendulumCompare from './PendulumCompare'
import PendulumLive from './PendulumLive'

type Mode = 'single' | 'double'
type Controller = 'pid' | 'nn'

export default function App() {
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [running, setRunning] = useState(false)
  const [mode, setMode] = useState<Mode>('single')
  const [controller, setController] = useState<Controller>('pid')
  const [target, setTarget] = useState<number>(0.0)
  const [useDegrees, setUseDegrees] = useState<boolean>(false)
  const [state, setState] = useState<any>(null)
  const [scene, setScene] = useState<any>(null)
  const [sceneResetId, setSceneResetId] = useState<number | null>(null)
  const [sceneSessionId, setSceneSessionId] = useState<number | null>(null)
  const [fps, setFps] = useState<number | undefined>(undefined)
  const [nnFramework, setNnFramework] = useState<'numpy' | 'torch'>('numpy')
  const [currentPolicyName, setCurrentPolicyName] = useState<string | null>(null)
  const [pidParams, setPidParams] = useState({ kp: 30.0, ki: 0.0, kd: 2.0 })
  const [trainingStats, setTrainingStats] = useState<any>(null)
  const [confirmedTrack, setConfirmedTrack] = useState<number | null>(null)
  const [gravity, setGravity] = useState<number>(9.81)
  const [confirmedGravity, setConfirmedGravity] = useState<number | null>(null)
  const [trainerParams, setTrainerParams] = useState<{ population?: number; sigma?: number; alpha?: number; steps?: number }>({ population: 12, sigma: 0.08, alpha: 0.04, steps: 100 })

  useEffect(() => {
    let mounted = true
    let sock: WebSocket | null = null
    let reconnectTimer: number | null = null

    const connect = async () => {
      if (!mounted) return
      // Do a quick health check before creating a WebSocket to avoid
      // noisy browser errors when the backend isn't up yet.
      try {
        const res = await fetch('http://localhost:8000/health', { cache: 'no-store' })
        if (!res.ok) throw new Error('health check failed')
      } catch (err) {
        // Backend not ready; schedule reconnect
        reconnectTimer = window.setTimeout(() => connect(), 1000)
        return
      }

      try {
        sock = new WebSocket('ws://localhost:8000/ws')
      } catch (e) {
        // Some environments may throw synchronously (rare); schedule reconnect
        console.warn('failed to construct WebSocket, will retry', e)
        reconnectTimer = window.setTimeout(() => connect(), 1000)
        return
      }

      const sendStart = () => {
        if (!sock || sock.readyState !== WebSocket.OPEN) return
        try {
          const payload: any = { action: 'start', mode, controller, dt: 0.02, target, engine: 'ode' }
          if (typeof gravity === 'number') payload.gravity = gravity
          if (controller === 'nn') payload.nn_framework = nnFramework
          if (controller === 'pid') {
            payload.kp = pidParams.kp
            payload.ki = pidParams.ki
            payload.kd = pidParams.kd
          }
          sock.send(JSON.stringify(payload))
        } catch (e) {
          console.warn('failed to send start', e)
        }
      }

      sock.onopen = () => {
        console.log('ws open')
        setWs(sock)
        // send initial start so server resets env for this client
        sendStart()
      }

      sock.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data)
          if (msg.state) setState(msg.state)

          // Scene messages are tagged with a `session_id`. Only apply
          // scenes that belong to the current session; this prevents
          // stale or interleaved scenes from different sim instances
          // being rendered simultaneously.
          if (msg.scene) {
            const sid = typeof msg.session_id !== 'undefined' ? Number(msg.session_id) : null
            if (sceneSessionId == null) {
              if (sid != null) setSceneSessionId(sid)
              setScene(msg.scene)
            } else {
              if (sid == null || sid === sceneSessionId) {
                setScene(msg.scene)
              } else {
                // ignore stale scene
              }
            }
          }

          if (msg.scene_reset) {
            setSceneResetId(msg.reset_id || Date.now())
            if (typeof msg.session_id !== 'undefined') setSceneSessionId(Number(msg.session_id))
          }

          if (typeof msg.track_set !== 'undefined') setConfirmedTrack(Number(msg.track_set))
          if (msg.training_stats) setTrainingStats(msg.training_stats)
          if (msg.trainer_params) setTrainerParams(msg.trainer_params)
          if (typeof msg.gravity_set !== 'undefined') setConfirmedGravity(Number(msg.gravity_set))
          if (msg.policy_loaded) {
            const p = msg.policy_loaded as string
            const name = p.split('/').pop() || p
            setCurrentPolicyName(name)
          }
          if (msg.policy_saved) {
            const p = msg.policy_saved as string
            const name = p.split('/').pop() || p
            setCurrentPolicyName(name)
          }
        } catch (err) {
          console.error('ws msg', err)
        }
      }

      sock.onclose = () => {
        console.log('ws closed')
        setWs(null)
        if (!mounted) return
        // try reconnect after 1s
        reconnectTimer = window.setTimeout(() => connect(), 1000)
      }

      sock.onerror = (ev) => {
        console.warn('ws error', ev)
      }
    }

    connect()
    return () => {
      mounted = false
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      try { sock && sock.close() } catch { }
    }
  }, [])
  
  // Debounced auto-restart when top-level config changes (mode/controller/nnFramework/target)
  const restartTimerRef = useRef<number | null>(null)
  useEffect(() => {
    if (!ws) return
    if (restartTimerRef.current) window.clearTimeout(restartTimerRef.current)
    // debounce 300ms
    restartTimerRef.current = window.setTimeout(() => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      try {
        const payload: any = { action: 'start', mode, controller, dt: 0.02, target, engine: 'ode' }
        if (controller === 'nn') payload.nn_framework = nnFramework
        if (controller === 'pid') {
          payload.kp = pidParams.kp
          payload.ki = pidParams.ki
          payload.kd = pidParams.kd
        }
        ws.send(JSON.stringify(payload))
      } catch (e) {
        console.warn('failed to auto-restart sim on config change', e)
      }
    }, 300)
    return () => {
      if (restartTimerRef.current) window.clearTimeout(restartTimerRef.current)
      restartTimerRef.current = null
    }
  }, [mode, controller, nnFramework, target, ws])


  const start = () => {
    if (!ws) return
    const payload: any = { action: 'start', mode, controller, dt: 0.02, target: 0.0, engine: 'ode', nn_framework: nnFramework }
    if (typeof gravity === 'number') payload.gravity = gravity
    ws.send(JSON.stringify(payload))
    setRunning(true)
  }

  const sendSetTrack = (len: number) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    try {
      ws.send(JSON.stringify({ action: 'set_track', track_length: len }))
    } catch (e) {
      console.warn('failed to send set_track', e)
    }
  }

  const sendSetGravity = (g: number) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    try {
      ws.send(JSON.stringify({ action: 'set_gravity', gravity: g }))
    } catch (e) {
      console.warn('failed to send set_gravity', e)
    }
  }

  const stop = () => {
    if (!ws) return
    ws.send(JSON.stringify({ action: 'stop' }))
    setRunning(false)
  }

  const sendPidUpdate = (p: { kp?: number; ki?: number; kd?: number }) => {
    setPidParams((cur) => ({ ...cur, ...p }))
    if (!ws) return
    ws.send(JSON.stringify({ action: 'control', type: 'pid', params: { ...pidParams, ...p } }))
  }

  const savePolicy = (name?: string) => {
    if (!ws) return
    ws.send(JSON.stringify({ action: 'save_policy', name: name || 'default' }))
  }

  const loadPolicy = (name?: string) => {
    if (!ws) return
    ws.send(JSON.stringify({ action: 'load_policy', name: name || 'default' }))
  }

  const sendTrainerParams = (p: { population?: number; sigma?: number; alpha?: number; steps?: number }) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    try {
      ws.send(JSON.stringify({ action: 'set_trainer_params', params: p }))
    } catch (e) {
      console.warn('failed to send trainer params', e)
    }
  }

  return (
    <div className="app" style={{ position: 'relative', height: '100vh' }}>
      <ThreeScene state={state} scene={scene} onFps={(v: number) => setFps(v)} currentTrack={confirmedTrack} sceneResetId={sceneResetId} />
      <UIOverlay
        mode={mode}
        controller={controller}
        target={target}
        useDegrees={useDegrees}
        onChangeUseDegrees={(b: boolean) => setUseDegrees(b)}
        onChangeTarget={(t) => setTarget(t)}
        running={running}
        onChangeMode={(m) => setMode(m)}
        onChangeController={(c) => setController(c)}
        fps={fps}
        kp={pidParams.kp}
        ki={pidParams.ki}
        kd={pidParams.kd}
        onPidChange={sendPidUpdate}
        onSavePolicy={savePolicy}
        onLoadPolicy={loadPolicy}
        nnFramework={nnFramework}
        onChangeNNFramework={(f) => setNnFramework(f)}
        currentPolicyName={currentPolicyName}
        onSetTrack={sendSetTrack}
        currentTrack={confirmedTrack}
        trainingStats={trainingStats}
        trainerParams={trainerParams}
        onTrainerParamsChange={(p) => sendTrainerParams(p)}
        gravity={confirmedGravity ?? gravity}
        onSetGravity={(g) => {
          setGravity(g)
          sendSetGravity(g)
        }}
      />
      {/* Training Stats card removed */}
      <div style={{ position: 'absolute', right: 12, bottom: 12, zIndex: 100000 }}>
        <pre className="glass card slide-up ui-top" style={{ padding: 8, maxWidth: 420, overflow: 'auto' }}>{JSON.stringify(state, null, 2)}</pre>
      </div>
      <div style={{ position: 'absolute', left: 12, bottom: 12, zIndex: 100200, width: 720 }}>
        <PendulumLive ws={ws} target={target} useDegrees={useDegrees} />
      </div>
    </div>
  )
}
