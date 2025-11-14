import React, { useEffect, useRef, useState } from 'react'
import PendulumCanvas from './PendulumCanvas'
import ThreeScene from './ThreeScene'

type Mode = 'single' | 'double'
type Controller = 'pid' | 'nn'

export default function App() {
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [running, setRunning] = useState(false)
  const [mode, setMode] = useState<Mode>('single')
  const [controller, setController] = useState<Controller>('pid')
  const [state, setState] = useState<any>(null)

  useEffect(() => {
    const sock = new WebSocket('ws://localhost:8000/ws')
    sock.onopen = () => console.log('ws open')
    sock.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.state) setState(msg.state)
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
      JSON.stringify({ action: 'start', mode, controller, dt: 0.02, target: 0.0, engine: 'pybullet' })
    )
    setRunning(true)
  }

  const stop = () => {
    if (!ws) return
    ws.send(JSON.stringify({ action: 'stop' }))
    setRunning(false)
  }

  return (
    <div className="app">
      <h1>NN & RL Lab</h1>
      <div className="controls">
        <label>
          Mode:
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            <option value="single">Single Pendulum</option>
            <option value="double">Double Pendulum</option>
          </select>
        </label>
        <label>
          Controller:
          <select value={controller} onChange={(e) => setController(e.target.value as Controller)}>
            <option value="pid">PID</option>
            <option value="nn">NN (demo)</option>
          </select>
        </label>
        <button onClick={start} disabled={running}>
          Start
        </button>
        <button onClick={stop} disabled={!running}>
          Stop
        </button>
      </div>
      <div style={{ display: 'flex', gap: 12 }}>
        <ThreeScene state={state} />
      </div>
      <pre className="state">{JSON.stringify(state, null, 2)}</pre>
      <div style={{ marginTop: 8 }}>
        <button onClick={() => ws?.send(JSON.stringify({ action: 'train_start' }))}>Start Training</button>
        <button onClick={() => ws?.send(JSON.stringify({ action: 'train_stop' }))}>Stop Training</button>
      </div>
    </div>
  )
}
