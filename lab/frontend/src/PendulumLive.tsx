import React, { useEffect, useRef } from 'react'

type Props = {
  ws: WebSocket | null
  target?: number
  useDegrees?: boolean
}

export function PendulumLive({ ws, target, useDegrees }: Props) {
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
      // draw target line and label if available
      if (typeof target === 'number' && Number.isFinite(target)) {
        const yT = h - ((target - min) / span) * h
        const y = Math.max(0, Math.min(h, yT))
        // dashed line
        ctx.save()
        ctx.setLineDash([6, 4])
        ctx.strokeStyle = 'rgba(255,200,0,0.9)'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(w, y)
        ctx.stroke()
        ctx.setLineDash([])
        // draw tick / marker at right
        ctx.fillStyle = 'rgba(255,200,0,0.95)'
        ctx.beginPath()
        ctx.arc(w - 10, y, 4, 0, Math.PI * 2)
        ctx.fill()

        // draw label box
        const radText = `${target.toFixed(2)} rad`
        const degText = `${(target * 180 / Math.PI).toFixed(1)}°`
        ctx.font = '12px system-ui, Arial'
        const padding = 6
        const txt = radText
        const metrics = ctx.measureText(txt)
        const boxW = Math.max(metrics.width, ctx.measureText(degText).width) + padding * 2
        const boxH = 32
        const bx = Math.max(w - boxW - 12, w - boxW - 12)
        const by = Math.max(6, Math.min(h - boxH - 6, y - boxH / 2))
        // background
        ctx.fillStyle = 'rgba(4,36,58,0.95)'
        ctx.fillRect(bx, by, boxW, boxH)
        ctx.fillStyle = 'rgba(255,255,255,0.95)'
        ctx.fillText(radText, bx + padding, by + 14)
        ctx.fillText(degText, bx + padding, by + 28)
        ctx.restore()
      }

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [target, useDegrees])

  // intentionally no start/stop controls; sim is always running

  return (
    <div style={{ width: '100%', height: 220 }}>
      <div className="glass card slide-up ui-top" style={{ padding: 8, borderRadius: 8 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#04243a' }}>Pendulum - Theta ({useDegrees ? 'deg' : 'rad'})</div>
          <div style={{ marginLeft: 'auto', fontSize: 12, color: '#666' }}>Auto-running</div>
        </div>
        <div style={{ width: '100%', height: 160, borderRadius: 6, overflow: 'hidden', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.03)' }}>
          <canvas ref={canvasRef} width={900} height={160} style={{ width: '100%', height: 160, display: 'block', background: '#041014', borderRadius: 6 }} />
        </div>
      </div>
    </div>
  )
}

export default PendulumLive
