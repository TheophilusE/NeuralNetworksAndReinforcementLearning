import React, { useEffect, useRef, useState } from 'react'

type Mode = 'single' | 'double'
type Controller = 'pid' | 'nn'

export default function PendulumContest() {
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [running, setRunning] = useState(false)
  const [mode, setMode] = useState<Mode>('single')
  const [controllerA, setControllerA] = useState<Controller>('pid')
  const [controllerB, setControllerB] = useState<Controller>('nn')
  const [target, setTarget] = useState<number>(0.0)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const dataRef = useRef<{ a: number[]; b: number[]; ta: number[]; tb: number[]; t: number[] }>({ a: [], b: [], ta: [], tb: [], t: [] })
  const maxLen = 600

  useEffect(() => {
    const sock = new WebSocket('ws://localhost:8000/ws')
    sock.onopen = () => {
      console.log('PendulumContest ws open')
    }
    sock.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        // Expect contest messages: {t, a: {state, tau}, b: {state, tau}}
        if (msg.a && msg.b) {
          const t = msg.t || (dataRef.current.t.length > 0 ? dataRef.current.t[dataRef.current.t.length - 1] + 0.02 : 0)
          const stateA = msg.a.state || {}
          const stateB = msg.b.state || {}
          const thetaA = stateA.theta ?? stateA.th1 ?? 0.0
          const thetaB = stateB.theta ?? stateB.th1 ?? 0.0
          const tauA = Number(msg.a.tau ?? 0.0)
          const tauB = Number(msg.b.tau ?? 0.0)
          const d = dataRef.current
          d.a.push(thetaA)
          d.b.push(thetaB)
          d.ta.push(tauA)
          d.tb.push(tauB)
          d.t.push(t)
          if (d.a.length > maxLen) {
            d.a.shift();
            d.b.shift();
            d.ta.shift();
            d.tb.shift();
            d.t.shift();
          }
        }
      } catch (e) {
        // ignore
      }
    }
    sock.onclose = () => console.log('PendulumContest ws closed')
    setWs(sock)
    return () => sock.close()
  }, [])

  // Draw loop
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

      const d = dataRef.current
      const n = d.a.length
      if (n > 1) {
        // find range
        const all = d.a.concat(d.b)
        const min = Math.min(...all)
        const max = Math.max(...all)
        const span = max - min || 1

        // draw grid
        ctx.strokeStyle = '#222'
        ctx.lineWidth = 1
        for (let i = 0; i < 4; i++) {
          const y = (h / 4) * i
          ctx.beginPath()
          ctx.moveTo(0, y)
          ctx.lineTo(w, y)
          ctx.stroke()
        }

        // helper to map
        const xAt = (i: number) => (i / (n - 1)) * w
        const yMap = (v: number) => h - ((v - min) / span) * h

        // draw A (blue)
        ctx.beginPath()
        ctx.strokeStyle = '#1f77b4'
        ctx.lineWidth = 2
        for (let i = 0; i < n; i++) {
          const x = xAt(i)
          const y = yMap(d.a[i])
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()

        // draw B (orange)
        ctx.beginPath()
        ctx.strokeStyle = '#ff7f0e'
        ctx.lineWidth = 2
        for (let i = 0; i < n; i++) {
          const x = xAt(i)
          const y = yMap(d.b[i])
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()

        // draw small legend
        ctx.fillStyle = '#1f77b4'
        ctx.fillRect(6, 6, 10, 10)
        ctx.fillStyle = '#fff'
        ctx.fillText('A (theta)', 22, 15)
        ctx.fillStyle = '#ff7f0e'
        ctx.fillRect(6, 24, 10, 10)
        ctx.fillStyle = '#fff'
        ctx.fillText('B (theta)', 22, 33)
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [])

  const sendStartContest = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const payload: any = {
      action: 'start_contest',
      mode,
      engine: 'pybullet',
      controller_a: controllerA,
      controller_b: controllerB,
      target,
    }
    ws.send(JSON.stringify(payload))
    setRunning(true)
  }

  const sendStop = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify({ action: 'stop' }))
    setRunning(false)
  }

  return (
    <div style={{ width: '100%', height: 300, position: 'relative' }}>
      <div style={{ display: 'flex', gap: 8, padding: 8, alignItems: 'center' }}>
        <label>
          Mode:
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            <option value="single">Single</option>
            <option value="double">Double</option>
          </select>
        </label>
        <label>
          Controller A:
          <select value={controllerA} onChange={(e) => setControllerA(e.target.value as Controller)}>
            <option value="pid">PID</option>
            <option value="nn">NN</option>
          </select>
        </label>
        <label>
          Controller B:
          <select value={controllerB} onChange={(e) => setControllerB(e.target.value as Controller)}>
            <option value="pid">PID</option>
            <option value="nn">NN</option>
          </select>
        </label>
        <label>
          Target:
          <input type="number" step="0.01" value={target} onChange={(e) => setTarget(parseFloat(e.target.value))} />
        </label>
        <button className="btn" onClick={sendStartContest} disabled={running}>
          Start Contest
        </button>
        <button className="btn" onClick={sendStop} disabled={!running}>
          Stop
        </button>
      </div>
      <canvas ref={canvasRef} width={800} height={240} style={{ width: '100%', height: 240, background: '#0b0b0b' }} />
    </div>
  )
}
