import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

export default function ThreeScene({ state, scene, onFps }: any) {
    const mount = useRef<HTMLDivElement | null>(null)
    const rafRef = useRef<number | null>(null)
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
    const stateRef = useRef<any>(state)
    const sceneRef = useRef<any>(scene)
    const [fps, setFps] = useState<number | undefined>(undefined)
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
    const controlsRef = useRef<any>(null)
    const sceneGraphRootRef = useRef<THREE.Group | null>(null)
    const bodyMapRef = useRef<Map<number, any>>(new Map())
    const [cameraMode, setCameraMode] = useState<'orbit' | 'top' | 'side' | 'front' | 'follow'>('orbit')
    const [serverIntervalMs, setServerIntervalMs] = useState<number | null>(null)
    const lastServerTsRef = useRef<number | null>(null)

    useEffect(() => { stateRef.current = state
        try {
            const now = performance.now()
            const last = lastServerTsRef.current
            if (last != null) setServerIntervalMs(now - last)
            lastServerTsRef.current = now
        } catch {}
    }, [state])

    useEffect(() => { sceneRef.current = scene }, [scene])

    // helper: build a Three mesh from backend visual metadata
    function buildMeshFromVisual(vis: any) {
        let geom: THREE.BufferGeometry
        let mat: THREE.Material = new THREE.MeshStandardMaterial({ color: 0x999999 })
        try {
            if (vis && vis.rgba && vis.rgba.length >= 3) {
                const c = new THREE.Color(vis.rgba[0], vis.rgba[1], vis.rgba[2])
                mat = new THREE.MeshStandardMaterial({ color: c })
            }
            if (vis && typeof vis.geom_type === 'number') {
                const gt = vis.geom_type
                if (gt === 1) {
                    const d = vis.dimensions || [0.1, 0.1, 0.1]
                    geom = new THREE.BoxGeometry(Math.max(0.001, d[0] * 2), Math.max(0.001, d[1] * 2), Math.max(0.001, d[2] * 2))
                    return new THREE.Mesh(geom, mat)
                } else if (gt === 2) {
                    const r = (vis.dimensions && vis.dimensions[0]) ? vis.dimensions[0] : 0.05
                    geom = new THREE.SphereGeometry(Math.max(0.001, r))
                    return new THREE.Mesh(geom, mat)
                } else if (gt === 3) {
                    const d = vis.dimensions || [0.05, 0.1]
                    const radius = Math.max(0.001, d[0])
                    const height = Math.max(0.001, d[1] * 2)
                    geom = new THREE.CylinderGeometry(radius, radius, height, 16)
                    return new THREE.Mesh(geom, mat)
                }
            }
            if (vis && Array.isArray(vis.dimensions) && vis.dimensions.length >= 3) {
                const d = vis.dimensions
                geom = new THREE.BoxGeometry(Math.max(0.001, d[0] * 2), Math.max(0.001, d[1] * 2), Math.max(0.001, d[2] * 2))
                return new THREE.Mesh(geom, mat)
            }
        } catch (e) {}
        geom = new THREE.BoxGeometry(0.12, 0.12, 0.12)
        return new THREE.Mesh(geom, mat)
    }

    useEffect(() => {
        const el = mount.current!
        const scene3 = new THREE.Scene()
        scene3.background = new THREE.Color(0xeef6ff)

        const camera = new THREE.PerspectiveCamera(50, el.clientWidth / el.clientHeight, 0.1, 1000)
        const renderer = new THREE.WebGLRenderer({ antialias: true })
        renderer.setPixelRatio(window.devicePixelRatio)
        renderer.setSize(el.clientWidth, el.clientHeight)
        renderer.domElement.style.display = 'block'
        el.appendChild(renderer.domElement)
        rendererRef.current = renderer

        camera.position.set(0, -3.5, 2)
        camera.up.set(0, 0, 1)
        camera.lookAt(0, 0, 1)
        cameraRef.current = camera

        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.target.set(0, 0, 1)
        controlsRef.current = controls

        const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8)
        scene3.add(hemi)
        const dir = new THREE.DirectionalLight(0xffffff, 0.8)
        dir.position.set(5, -5, 10)
        scene3.add(dir)

        const grid = new THREE.GridHelper(10, 20, 0x888888, 0xdddddd)
        scene3.add(grid)

        const root = new THREE.Group()
        scene3.add(root)
        sceneGraphRootRef.current = root
        bodyMapRef.current = new Map()

        let lastFpsTime = performance.now()
        let frames = 0

        function applySceneUpdate(incoming: any) {
            if (!incoming || !incoming.bodies) return
            const rootLocal = sceneGraphRootRef.current
            const bodyMapLocal = bodyMapRef.current
            if (!rootLocal || !bodyMapLocal) return
            for (const body of incoming.bodies) {
                const bid = body.body_id
                let entry = bodyMapLocal.get(bid)
                if (!entry) {
                    const g = new THREE.Group()
                    g.name = `body-${bid}`
                    rootLocal.add(g)
                    entry = { group: g, links: new Map() }
                    bodyMapLocal.set(bid, entry)
                }
                for (const link of body.links || []) {
                    const idx = link.link_index
                    if (entry.links.has(idx)) continue
                    const vis = link.visual
                    const mesh = buildMeshFromVisual(vis)
                    mesh.name = `body-${bid}-link-${idx}`
                    entry.group.add(mesh)
                    entry.links.set(idx, mesh)
                }
            }
        }

        function animate() {
            frames++
            const now = performance.now()
            const dt = now - lastFpsTime
            if (dt >= 500) {
                const calc = (frames * 1000) / dt
                setFps(calc)
                if (onFps) onFps(calc)
                frames = 0
                lastFpsTime = now
            }

            // create meshes if new
            if (sceneRef.current) applySceneUpdate(sceneRef.current)
            // update transforms
            const sc = sceneRef.current
            if (sc && sc.bodies) {
                const bodyMapLocal = bodyMapRef.current
                for (const body of sc.bodies) {
                    const bid = body.body_id
                    const entry = bodyMapLocal.get(bid)
                    if (!entry) continue
                    for (const link of body.links || []) {
                        const mesh = entry.links.get(link.link_index)
                        if (!mesh) continue
                        const wp = link.world_position
                        const wo = link.world_orientation
                        if (wp && wp.length === 3) mesh.position.set(wp[0], wp[1], wp[2])
                        if (wo && wo.length === 4) mesh.quaternion.set(wo[0], wo[1], wo[2], wo[3])
                    }
                }
                // optional follow camera: move behind first link of first body
                if (cameraMode === 'follow' && sc.bodies.length > 0 && cameraRef.current && controlsRef.current) {
                    const firstBody = sc.bodies[0]
                    const firstLink = (firstBody.links && firstBody.links[0])
                    if (firstLink && firstLink.world_position) {
                        const wp = firstLink.world_position
                        const c1 = new THREE.Vector3(wp[0], wp[1], wp[2])
                        const cam = cameraRef.current
                        const controls = controlsRef.current
                        const desired = new THREE.Vector3(c1.x, c1.y - 2.2, c1.z + 1.2)
                        cam.position.lerp(desired, 0.12)
                        controls.target.lerp(c1, 0.18)
                        controls.update()
                    }
                }
            }

            controlsRef.current?.update()
            renderer.render(scene3, camera)
            rafRef.current = requestAnimationFrame(animate)
        }

        animate()

        const resize = () => {
            if (!mount.current) return
            const w = mount.current.clientWidth
            const h = mount.current.clientHeight
            camera.aspect = w / h
            camera.updateProjectionMatrix()
            renderer.setSize(w, h)
        }
        resize()
        window.addEventListener('resize', resize)

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current)
            renderer.dispose()
            window.removeEventListener('resize', resize)
            if (el && renderer.domElement) el.removeChild(renderer.domElement)
        }
    }, [])

    // Apply camera presets when cameraMode changes
    function applyCameraMode(mode: 'orbit' | 'top' | 'side' | 'front' | 'follow') {
        const cam = cameraRef.current
        const controls = controlsRef.current
        if (!cam || !controls) return
        switch (mode) {
            case 'top': cam.position.set(0, 0, 6); controls.target.set(0, 0, 1); break
            case 'side': cam.position.set(6, 0, 1); controls.target.set(0, 0, 1); break
            case 'front': cam.position.set(0, -6, 1); controls.target.set(0, 0, 1); break
            case 'orbit': cam.position.set(0, -3.5, 2); controls.target.set(0, 0, 1); break
            case 'follow': break
        }
        cam.updateProjectionMatrix()
        controls.update()
        setCameraMode(mode)
    }

    useEffect(() => { applyCameraMode(cameraMode) }, [cameraMode])

    function CameraButton({ mode, title, svg }: { mode: 'orbit' | 'top' | 'side' | 'front' | 'follow'; title: string; svg: JSX.Element }) {
        const active = cameraMode === mode
        return (
            <button className={"cam-btn" + (active ? ' cam-btn--active' : '')} aria-pressed={active} title={title} onClick={() => applyCameraMode(mode)}>
                {svg}
            </button>
        )
    }

    function CameraUI() {
        return (
            <div className="camera-panel" onPointerDown={(e) => e.stopPropagation()}>
                <div style={{ display: 'flex', gap: 8 }}>
                    <CameraButton mode={'orbit'} title="Orbit" svg={(<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="7" stroke="currentColor" strokeWidth="1.6" /><circle cx="12" cy="8" r="1.2" fill="currentColor" /></svg>)} />
                    <CameraButton mode={'top'} title="Top" svg={(<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="5" y="5" width="14" height="6" stroke="currentColor" strokeWidth="1.6" rx="1" /><rect x="8" y="13" width="8" height="6" stroke="currentColor" strokeWidth="1.6" rx="1" /></svg>)} />
                    <CameraButton mode={'side'} title="Side" svg={(<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 12h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><circle cx="18" cy="12" r="3" stroke="currentColor" strokeWidth="1.6" /></svg>)} />
                    <CameraButton mode={'front'} title="Front" svg={(<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="6" width="16" height="12" stroke="currentColor" strokeWidth="1.6" rx="1" /><circle cx="12" cy="12" r="2" fill="currentColor" /></svg>)} />
                    <CameraButton mode={'follow'} title="Follow" svg={(<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 3v3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><path d="M12 18v3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><circle cx="12" cy="12" r="6" stroke="currentColor" strokeWidth="1.6" /></svg>)} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', marginLeft: 8 }}>
                    <div className="stat-box card glass" style={{ padding: '6px 8px' }}>
                        <div style={{ fontSize: 12, color: 'inherit' }}><strong>Frame:</strong> {fps ? `${(1000 / fps).toFixed(1)} ms` : '—'}</div>
                        <div style={{ fontSize: 12, color: 'inherit' }}><strong>Server:</strong> {serverIntervalMs ? `${serverIntervalMs.toFixed(1)} ms` : '—'}</div>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div ref={mount} className="three-mount">
            <CameraUI />
        </div>
    )
}

