import React, { useEffect, useRef } from 'react'

export default function PendulumCanvas({ state, mode }: any) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const w = canvas.width = 600
    const h = canvas.height = 400

    ctx.clearRect(0, 0, w, h)
    ctx.fillStyle = '#111'
    ctx.fillRect(0, 0, w, h)

    const origin = { x: w / 2, y: 100 }

    if (!state) {
      ctx.fillStyle = '#fff'
      ctx.fillText('Start the simulation', 20, 20)
      return
    }

    if (mode === 'single') {
      const l = 150
      const theta = state.theta ?? 0
      const x = origin.x + l * Math.sin(theta)
      const y = origin.y + l * Math.cos(theta)

      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 4
      ctx.beginPath()
      ctx.moveTo(origin.x, origin.y)
      ctx.lineTo(x, y)
      ctx.stroke()

      ctx.fillStyle = '#f66'
      ctx.beginPath()
      ctx.arc(x, y, 12, 0, Math.PI * 2)
      ctx.fill()
    } else {
      const l = 120
      const th1 = state.th1 ?? 0
      const th2 = state.th2 ?? 0
      const x1 = origin.x + l * Math.sin(th1)
      const y1 = origin.y + l * Math.cos(th1)
      const x2 = x1 + l * Math.sin(th2)
      const y2 = y1 + l * Math.cos(th2)

      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(origin.x, origin.y)
      ctx.lineTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.stroke()

      ctx.fillStyle = '#6af'
      ctx.beginPath()
      ctx.arc(x1, y1, 10, 0, Math.PI * 2)
      ctx.fill()
      ctx.beginPath()
      ctx.arc(x2, y2, 10, 0, Math.PI * 2)
      ctx.fill()
    }
  }, [state, mode])

  return <canvas ref={canvasRef} style={{ borderRadius: 6 }} />
}
