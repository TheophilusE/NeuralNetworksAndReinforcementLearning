// Small helper for future REST/ws helpers
// Build backend URLs dynamically so the frontend follows the Vite host or
// environment overrides. Use VITE_BACKEND_HOST / VITE_BACKEND_PORT / VITE_BACKEND_SECURE
// when provided; otherwise fall back to the current window location for host
// and assume backend port 8000.
const env = (import.meta as any).env || {}
const backendHost = env.VITE_BACKEND_HOST || (typeof window !== 'undefined' ? window.location.hostname : 'localhost')
const backendPort = env.VITE_BACKEND_PORT || '8000'
const useSecure = (env.VITE_BACKEND_SECURE === 'true') || (typeof window !== 'undefined' && window.location.protocol === 'https:')

const wsProtocol = useSecure ? 'wss' : 'ws'
const httpProtocol = useSecure ? 'https' : 'http'

export const WS_URL = `${wsProtocol}://${backendHost}:${backendPort}/ws`
export const HEALTH_URL = `${httpProtocol}://${backendHost}:${backendPort}/health`

export function buildWsUrl(host?: string, port?: string, secure?: boolean) {
	const h = host || backendHost
	const p = port || backendPort
	const proto = secure === undefined ? useSecure : !!secure
	return `${proto ? 'wss' : 'ws'}://${h}:${p}/ws`
}

export function buildHttpUrl(path = '/health', host?: string, port?: string, secure?: boolean) {
	const h = host || backendHost
	const p = port || backendPort
	const proto = secure === undefined ? useSecure : !!secure
	return `${proto ? 'https' : 'http'}://${h}:${p}${path}`
}
