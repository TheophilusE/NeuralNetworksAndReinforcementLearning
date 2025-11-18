import React, { useEffect, useRef, useState } from 'react'
import { WSProvider, useWS } from './WSContext'
import ThreeScene from './ThreeScene'
import UIOverlay from './UIOverlay'
import PendulumLive from './PendulumLive'

type Mode = 'single' | 'double'
type Controller = 'pid' | 'nn'

function AppInner() {
  const ws = useWS()
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

  // Attach socket event handlers (message/open/close) when context ws changes
  useEffect(() => {
    if (!ws) return
    const onOpen = () => {
      console.log('ws open')
      // send initial start so server resets env for this client
      try {
        const payload: any = { action: 'start', mode, controller, dt: 0.02, target, engine: 'ode' }
        if (typeof gravity === 'number') payload.gravity = gravity
        if (controller === 'nn') payload.nn_framework = nnFramework
        if (controller === 'pid') {
          payload.kp = pidParams.kp
          payload.ki = pidParams.ki
          payload.kd = pidParams.kd
        }
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload))
      } catch (e) {
        console.warn('failed to send initial start', e)
      }
    }

    const onMessage = (e: MessageEvent) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.state) setState(msg.state)

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

    const onClose = () => {
      console.log('ws closed')
      setRunning(false)
    }

    const onError = (ev: Event) => console.warn('ws error', ev)

    try {
      ws.addEventListener('open', onOpen)
      ws.addEventListener('message', onMessage)
      ws.addEventListener('close', onClose)
      ws.addEventListener('error', onError)
    } catch (e) {
      try { (ws as any).onopen = onOpen } catch {}
      try { (ws as any).onmessage = onMessage } catch {}
      try { (ws as any).onclose = onClose } catch {}
      try { (ws as any).onerror = onError } catch {}
    }

    return () => {
      try { ws.removeEventListener('open', onOpen) } catch {}
      try { ws.removeEventListener('message', onMessage) } catch {}
      try { ws.removeEventListener('close', onClose) } catch {}
      try { ws.removeEventListener('error', onError) } catch {}
    }
  }, [ws])
  
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
    // Compute new params synchronously so the message uses the intended values
    const newParams = { ...pidParams, ...p }
    setPidParams(newParams)
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    try {
      ws.send(JSON.stringify({ action: 'control', type: 'pid', params: newParams }))
    } catch (e) {
      console.warn('failed to send pid control update', e)
    }
  }

  // When switching to PID controller, push current PID params to the backend
  useEffect(() => {
    if (!ws || controller !== 'pid') return
    if (ws.readyState !== WebSocket.OPEN) return
    try {
      ws.send(JSON.stringify({ action: 'control', type: 'pid', params: pidParams }))
    } catch (e) {
      console.warn('failed to send pid params on controller switch', e)
    }
  }, [controller, ws])

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

export default function App() {
  return (
    <WSProvider>
      <AppInner />
    </WSProvider>
  )
}
