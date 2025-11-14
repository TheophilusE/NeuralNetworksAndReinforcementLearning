import React, { useEffect, useState } from 'react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

export default function StatsSidebar({ stats, docked = true, onToggleDock }: any) {
  const [open, setOpen] = useState(docked)

  useEffect(() => setOpen(docked), [docked])

  const history = (stats && stats.history) || []
  const labels = history.map((_: number, i: number) => i)
  const data = {
    labels,
    datasets: [
      {
        label: 'Mean Reward',
        data: history,
        borderColor: 'rgba(54, 162, 235, 1)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
      },
    ],
  }

  const options: any = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { display: false },
      y: { display: true },
    },
    plugins: {
      legend: { display: false },
    },
  }

  return (
    <div style={{ position: 'absolute', left: 12, bottom: 12, zIndex: 30 }}>
      <div className="glass card" style={{ padding: 8, width: 340 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <strong>Training Stats</strong>
          <button onClick={() => { setOpen(!open); if (onToggleDock) onToggleDock(!open) }}>{open ? 'Undock' : 'Dock'}</button>
        </div>
        {open && (
          <div style={{ height: 160 }}>
            <div>Iter: {stats?.iter ?? '-'}</div>
            <div>Last Reward: {stats?.last_reward ?? '-'}</div>
            <div style={{ height: 120, marginTop: 8 }}>
              <Line data={data} options={options} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
