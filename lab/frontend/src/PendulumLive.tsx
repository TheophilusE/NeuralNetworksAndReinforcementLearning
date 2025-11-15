import React, { useEffect, useRef } from 'react'

type Props = {
  ws: WebSocket | null
}

export default function PendulumLive({ ws }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const dataRef = useRef<number[]>([])
  const maxLen = 800
  // The simulation is auto-started; no local running state needed

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

  // Auto-start simulation when WebSocket is ready
  useEffect(() => {
    if (!ws) return
    const startPayload = JSON.stringify({ action: 'start', mode: 'single', controller: 'pid', dt: 0.02, target: 0.0, engine: 'pybullet' })
    const tryStart = () => {
      try {
        if (ws.readyState === WebSocket.OPEN) ws.send(startPayload)
      } catch (e) {
        // ignore send errors
      }
    }
    // If already open, send immediately; otherwise wait for open
    if (ws.readyState === WebSocket.OPEN) {
      tryStart()
    } else {
      const onOpen = () => tryStart()
      ws.addEventListener('open', onOpen)
      return () => ws.removeEventListener('open', onOpen)
    }
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

  // intentionally no start/stop controls; sim is always running

  return (
    <div style={{ width: '100%', height: 220 }}>
      <div className="glass card slide-up ui-top" style={{ padding: 8, borderRadius: 8 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#04243a' }}>Pendulum - Theta (rad)</div>
          <div style={{ marginLeft: 'auto', fontSize: 12, color: '#666' }}>Auto-running</div>
        </div>
        <div style={{ width: '100%', height: 160, borderRadius: 6, overflow: 'hidden', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.03)' }}>
          <canvas ref={canvasRef} width={900} height={160} style={{ width: '100%', height: 160, display: 'block', background: '#041014' , borderRadius: 6 }} />
        </div>
      </div>
    </div>
  )
}
