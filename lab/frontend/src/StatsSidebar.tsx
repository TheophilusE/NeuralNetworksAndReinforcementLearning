import React, { useRef, useEffect, useState } from 'react'

export default function StatsSidebar({ stats, docked = true, onToggleDock }: any) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [open, setOpen] = useState(docked)

  useEffect(() => setOpen(docked), [docked])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !stats || !stats.history) return
    const ctx = canvas.getContext('2d')!
    const w = canvas.width = 300
    const h = canvas.height = 120
    ctx.fillStyle = '#fff'
    ctx.fillRect(0, 0, w, h)
    const hist = stats.history || []
    if (hist.length === 0) return
    const max = Math.max(...hist)
    const min = Math.min(...hist)
    ctx.strokeStyle = '#0077cc'
    ctx.beginPath()
    for (let i = 0; i < hist.length; i++) {
      const x = (i / (hist.length - 1)) * (w - 10) + 5
      const y = h - 5 - ((hist[i] - min) / Math.max(1e-6, (max - min))) * (h - 10)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()
  }, [stats])

  return (
    <div style={{ position: 'absolute', left: 12, bottom: 12, zIndex: 30 }}>
      <div style={{ background: 'rgba(255,255,255,0.95)', padding: 8, borderRadius: 8, width: 320 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <strong>Training Stats</strong>
          <button onClick={() => { setOpen(!open); if (onToggleDock) onToggleDock(!open) }}>{open ? 'Undock' : 'Dock'}</button>
        </div>
        {open && (
          <div>
            <div>Iter: {stats?.iter ?? '—'}</div>
            <div>Last Reward: {stats?.last_reward ?? '—'}</div>
            <canvas ref={canvasRef} style={{ width: 300, height: 120, marginTop: 8 }} />
          </div>
        )}
      </div>
    </div>
  )
}
