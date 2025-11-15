import React, { useEffect, useRef, useState } from 'react'

type Props = {
  ws: WebSocket | null
}

export default function PendulumLive({ ws }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const dataRef = useRef<number[]>([])
  const maxLen = 800
  const [running, setRunning] = useState(false)

  useEffect(() => {
    if (!ws) return
    const onMsg = (ev: MessageEvent) => {
      try {
        const msg = JSON.parse(ev.data)
        // server emits { t, state } for single-sim runs
        const state = msg.state || (msg.a && msg.a.state) || null
        if (!state) return
        const theta = state.theta ?? state.th1 ?? 0.0
        const arr = dataRef.current
        arr.push(theta)
        if (arr.length > maxLen) arr.shift()
      } catch (e) {
        // ignore non-json
      }
    }
    ws.addEventListener('message', onMsg)
    return () => ws.removeEventListener('message', onMsg)
  }, [ws])

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

      const arr = dataRef.current
      if (arr.length < 2) {
        raf = requestAnimationFrame(draw)
        return
      }
      const min = Math.min(...arr)
      const max = Math.max(...arr)
      const span = max - min || 1

      ctx.strokeStyle = '#00d1b2'
      ctx.lineWidth = 2
      ctx.beginPath()
      for (let i = 0; i < arr.length; i++) {
        const x = (i / (arr.length - 1)) * w
        const y = h - ((arr[i] - min) / span) * h
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()

      // draw zero reference
      ctx.strokeStyle = 'rgba(255,255,255,0.2)'
      const y0 = h - ((0 - min) / span) * h
      ctx.beginPath()
      ctx.moveTo(0, y0)
      ctx.lineTo(w, y0)
      ctx.stroke()

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [])

  const startSim = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    // start single-sim with pybullet on backend
    ws.send(JSON.stringify({ action: 'start', mode: 'single', controller: 'pid', dt: 0.02, target: 0.0, engine: 'pybullet' }))
    setRunning(true)
  }

  const stopSim = () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify({ action: 'stop' }))
    setRunning(false)
  }

  return (
    <div style={{ width: '100%', height: 220 }}>
      <div style={{ display: 'flex', gap: 8, padding: 6, alignItems: 'center' }}>
        <button className="btn" onClick={startSim} disabled={running || !ws}>
          Start PyBullet Sim
        </button>
        <button className="btn" onClick={stopSim} disabled={!running || !ws}>
          Stop
        </button>
        <div style={{ marginLeft: 'auto', color: '#ddd' }}>Showing theta over time</div>
      </div>
      <canvas ref={canvasRef} width={900} height={160} style={{ width: '100%', height: 160, background: '#041014' }} />
    </div>
  )
}
