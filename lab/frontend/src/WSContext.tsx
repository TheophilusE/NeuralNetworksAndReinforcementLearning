import React, { createContext, useContext, useEffect, useState } from 'react'

// Module-scoped singleton socket - ensures one socket per page
let globalSock: WebSocket | null = null
let reconnectTimer: number | null = null

const WSContext = createContext<WebSocket | null>(null)

export const useWS = () => useContext(WSContext)

export const WSProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [ws, setWs] = useState<WebSocket | null>(null)

  useEffect(() => {
    let mounted = true

    const ensureSocket = async () => {
      // quick health check before creating socket
      try {
        const res = await fetch('http://localhost:8000/health', { cache: 'no-store' })
        if (!res.ok) throw new Error('health check failed')
      } catch (err) {
        // backend not ready; retry
        reconnectTimer = window.setTimeout(() => ensureSocket(), 1000)
        return
      }

      try {
        if (globalSock && globalSock.readyState !== WebSocket.CLOSED && globalSock.readyState !== WebSocket.CLOSING) {
          setWs(globalSock)
        } else {
          globalSock = new WebSocket('ws://localhost:8000/ws')
          setWs(globalSock)
        }
      } catch (e) {
        reconnectTimer = window.setTimeout(() => ensureSocket(), 1000)
        return
      }

      // if socket closes, try to recreate after a delay
      try {
        globalSock!.addEventListener('close', () => {
          if (reconnectTimer) window.clearTimeout(reconnectTimer)
          reconnectTimer = window.setTimeout(() => { if (mounted) ensureSocket() }, 1000)
        })
      } catch (e) {
        // ignore
      }
    }

    ensureSocket()

    const onUnload = () => {
      try {
        if (globalSock) {
          try { globalSock.close() } catch {}
          globalSock = null
        }
      } catch (e) {
        // ignore
      }
    }
    window.addEventListener('beforeunload', onUnload)

    return () => {
      mounted = false
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
      window.removeEventListener('beforeunload', onUnload)
      // do not forcibly close here; we close on unload only
    }
  }, [])

  return <WSContext.Provider value={ws}>{children}</WSContext.Provider>
}
