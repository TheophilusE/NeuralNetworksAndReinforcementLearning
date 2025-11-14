import React, { useEffect, useRef, useState } from 'react'
import ThreeScene from './ThreeScene'
import UIOverlay from './UIOverlay'

type Mode = 'single' | 'double'
type Controller = 'pid' | 'nn'

export default function App() {
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [running, setRunning] = useState(false)
  const [mode, setMode] = useState<Mode>('single')
  const [controller, setController] = useState<Controller>('pid')
  const [state, setState] = useState<any>(null)
  const [fps, setFps] = useState<number | undefined>(undefined)

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
    <div className="app" style={{ position: 'relative', height: '100vh' }}>
      <ThreeScene state={state} onFps={(v: number) => setFps(v)} />
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
      />
      <div style={{ position: 'absolute', right: 12, bottom: 12, zIndex: 30 }}>
        <pre style={{ background: 'rgba(255,255,255,0.9)', padding: 8, borderRadius: 6 }}>{JSON.stringify(state, null, 2)}</pre>
      </div>
    </div>
  )
}
