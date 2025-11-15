import React, { useEffect, useRef, useState } from 'react'
import ThreeScene from './ThreeScene'
import UIOverlay from './UIOverlay'
import StatsSidebar from './StatsSidebar'

type Mode = 'single' | 'double'
type Controller = 'pid' | 'nn'

export default function App() {
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [running, setRunning] = useState(false)
  const [mode, setMode] = useState<Mode>('single')
  const [controller, setController] = useState<Controller>('pid')
  const [state, setState] = useState<any>(null)
  const [scene, setScene] = useState<any>(null)
  const [fps, setFps] = useState<number | undefined>(undefined)
  const [nnFramework, setNnFramework] = useState<'numpy' | 'torch'>('numpy')
  const [currentPolicyName, setCurrentPolicyName] = useState<string | null>(null)
  const [pidParams, setPidParams] = useState({ kp: 30.0, ki: 0.0, kd: 2.0 })
  const [trainingStats, setTrainingStats] = useState<any>(null)

  useEffect(() => {
    const sock = new WebSocket('ws://localhost:8000/ws')
    sock.onopen = () => console.log('ws open')
    sock.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.state) setState(msg.state)
        if (msg.scene) setScene(msg.scene)
        if (msg.training_stats) setTrainingStats(msg.training_stats)
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
    sock.onclose = () => console.log('ws closed')
    setWs(sock)
    return () => sock.close()
  }, [])

  const start = () => {
    if (!ws) return
    ws.send(
      JSON.stringify({ action: 'start', mode, controller, dt: 0.02, target: 0.0, engine: 'pybullet', nn_framework: nnFramework })
    )
    setRunning(true)
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

  return (
    <div className="app" style={{ position: 'relative', height: '100vh' }}>
      <ThreeScene state={state} scene={scene} onFps={(v: number) => setFps(v)} />
      <UIOverlay
        mode={mode}
        controller={controller}
        running={running}
        onStart={() => start()}
        onStop={() => stop()}
        onTrainStart={() => ws?.send(JSON.stringify({ action: 'train_start' }))}
        onTrainStop={() => ws?.send(JSON.stringify({ action: 'train_stop' }))}
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
      />
      <StatsSidebar stats={trainingStats || { iter: 0, last_reward: null, history: [] }} docked={true} />
      <div style={{ position: 'absolute', right: 12, bottom: 12, zIndex: 100000 }}>
        <pre className="glass card slide-up ui-top" style={{ padding: 8, maxWidth: 420, overflow: 'auto' }}>{JSON.stringify(state, null, 2)}</pre>
      </div>
    </div>
  )
}
