import React, { useEffect, useState } from 'react'
// Removed ChartJS line chart (unused in current UI) to simplify the sidebar

export default function StatsSidebar({ stats, docked = true, onToggleDock }: any) {
    const [open, setOpen] = useState(docked)

    useEffect(() => setOpen(docked), [docked])

    // Keep history available for future use but do not draw an unused chart here.
    const history = (stats && stats.history) || []

    return (
        <div style={{ position: 'absolute', left: 12, bottom: 12, zIndex: 100000 }}>
            <div className="glass card slide-up ui-top" style={{ padding: 8, width: 340 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <strong>Training Stats</strong>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <button className="btn" onClick={() => { setOpen(!open); if (onToggleDock) onToggleDock(!open) }}>{open ? 'Undock' : 'Dock'}</button>
                    </div>
                </div>
                {open && (
                    <div style={{ height: 160 }}>
                        <div>Iter: {stats?.iter ?? '-'}</div>
                        <div>Last Reward: {stats?.last_reward ?? '-'}</div>
                        <div style={{ height: 120, marginTop: 8, color: '#bbb', fontSize: 12 }}>
                            (history chart disabled)
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
