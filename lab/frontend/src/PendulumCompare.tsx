import React, { useEffect, useRef, useState } from 'react'

type Mode = 'single' | 'double'

type Method = {
  id: string
  label: string
  controller: 'pid' | 'nn'
  nn_framework?: 'numpy' | 'torch'
}

const DEFAULT_METHODS: Method[] = [
  { id: 'pid', label: 'PID', controller: 'pid' },
  { id: 'nn_numpy', label: 'NN (Numpy)', controller: 'nn', nn_framework: 'numpy' },
  { id: 'nn_torch', label: 'NN (PyTorch)', controller: 'nn', nn_framework: 'torch' },
]

export default function PendulumCompare() {
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [mode, setMode] = useState<Mode>('single')
  const [methods, setMethods] = useState<Method[]>(DEFAULT_METHODS)
  const [running, setRunning] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const tracesRef = useRef<Record<string, number[]>>({})
  const [results, setResults] = useState<Record<string, number>>({})

  useEffect(() => {
    const sock = new WebSocket('ws://localhost:8000/ws')
    sock.onopen = () => console.log('PendulumCompare ws open')
    sock.onclose = () => console.log('PendulumCompare ws closed')
    setWs(sock)
    return () => sock.close()
  }, [])

  // Single draw loop to show overlays of traces collected so far
  useEffect(() => {
    let raf = 0
    const draw = () => {
      const c = canvasRef.current
      if (!c) {
        raf = requestAnimationFrame(draw)
        return
      }
      const ctx = c.getContext('2d')
      if (!ctx) {
        raf = requestAnimationFrame(draw)
        return
      }
      const w = c.width
      const h = c.height
      ctx.clearRect(0, 0, w, h)

      const keys = Object.keys(tracesRef.current)
      if (keys.length === 0) {
        raf = requestAnimationFrame(draw)
        return
      }

      // find global min/max
      let all: number[] = []
      for (const k of keys) all = all.concat(tracesRef.current[k])
      const min = Math.min(...all)
      const max = Math.max(...all)
      const span = max - min || 1

      // draw each trace
      keys.forEach((k, idx) => {
        const arr = tracesRef.current[k]
        const n = arr.length
        if (n < 2) return
        const color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'][idx % 4]
        ctx.beginPath()
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        for (let i = 0; i < n; i++) {
          const x = (i / (n - 1)) * w
          const y = h - ((arr[i] - min) / span) * h
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
        // label
        ctx.fillStyle = color
        ctx.fillText(k, 6, 14 + idx * 14)
      })

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [])

  // run compare: sequentially run each method for one trial and collect stabilization time
  const runCompare = async () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    setResults({})
    tracesRef.current = {}
    setRunning(true)

    for (const m of methods) {
      // clear current trace for this method
      tracesRef.current[m.label] = []

      // start sim on server for this method
      const payload: any = { action: 'start', mode, controller: m.controller, dt: 0.02, target: 0.0, engine: 'pybullet' }
      if (m.controller === 'nn') payload.nn_framework = m.nn_framework || 'numpy'
      ws.send(JSON.stringify(payload))

      // wait for telemetry and measure stabilization
      const threshold = 0.1 // radians
      const stableWindow = 50 // steps
      const maxSteps = 3000
      let step = 0
      let stableCount = 0
      let stabilizedAt: number | null = null

      // message listener for this run
      const onMsg = (ev: MessageEvent) => {
        try {
          const msg = JSON.parse(ev.data)
          const state = msg.state || msg.a?.state || msg?.a?.state || null
          // backend `start` sends `state` top-level; the contest messages have a/b keys but we used start
          if (!state) return
          const theta = state.theta ?? state.th1 ?? 0.0
          tracesRef.current[m.label].push(theta)
          step += 1
          if (Math.abs(theta) < threshold) {
            stableCount += 1
          } else {
            stableCount = 0
          }
          if (stableCount >= stableWindow && stabilizedAt === null) {
            stabilizedAt = step
          }
        } catch (e) {
          // ignore parse errors
        }
      }

      ws.addEventListener('message', onMsg)

      // wait until stabilized or maxSteps
      while (stabilizedAt === null && step < maxSteps) {
        // small delay to yield
        await new Promise((r) => setTimeout(r, 20))
      }

      ws.removeEventListener('message', onMsg)

      // stop this sim run
      try {
        ws.send(JSON.stringify({ action: 'stop' }))
      } catch (e) {
        // ignore
      }

      const timeToStabilize = stabilizedAt !== null ? stabilizedAt * 0.02 : NaN
      setResults((prev) => ({ ...prev, [m.label]: timeToStabilize }))

      // brief pause between runs
      await new Promise((r) => setTimeout(r, 300))
    }

    setRunning(false)
  }

  return (
    <div style={{ width: '100%', height: 320, position: 'relative' }}>
      <div style={{ display: 'flex', gap: 8, padding: 8, alignItems: 'center' }}>
        <label>
          Mode:
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            <option value="single">Single</option>
            <option value="double">Double</option>
          </select>
        </label>
        <button className="btn" onClick={runCompare} disabled={running}>
          Run Compare
        </button>
        <button className="btn" onClick={() => { tracesRef.current = {}; setResults({}) }} disabled={running}>
          Clear
        </button>
        <div style={{ marginLeft: 'auto' }}>
          <strong>Results:</strong>
          {Object.keys(results).length === 0 ? ' -' : null}
          {Object.entries(results).map(([k, v]) => (
            <div key={k} style={{ fontSize: 12 }}>{k}: {isNaN(v) ? 'not stabilized' : `${v.toFixed(2)}s`}</div>
          ))}
        </div>
      </div>
      <canvas ref={canvasRef} width={900} height={260} style={{ width: '100%', height: 260, background: '#0b0b0b' }} />
    </div>
  )
}
