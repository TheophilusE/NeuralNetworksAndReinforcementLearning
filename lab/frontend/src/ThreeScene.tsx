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
    const dirRef = useRef<THREE.DirectionalLight | null>(null)
    const gridRef = useRef<THREE.GridHelper | null>(null)
    const [cameraMode, setCameraMode] = useState<'orbit' | 'top' | 'side' | 'front' | 'follow'>('orbit')
    const [shadowsEnabled, setShadowsEnabled] = useState<boolean>(true)
    const [wireframeEnabled, setWireframeEnabled] = useState<boolean>(false)
    const [gridVisible, setGridVisible] = useState<boolean>(true)
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

    // track incoming scene timestamp for server interval

    // helper: build a Three mesh from backend visual metadata
    function buildMeshFromVisual(vis: any) {
        let geom: THREE.BufferGeometry
        let mat: THREE.Material = new THREE.MeshStandardMaterial({ color: 0x999999 })
        function normalizeRgba(rgba: any) {
            if (!rgba || rgba.length < 3) return null
            // detect 0-255 vs 0-1
            const max = Math.max(rgba[0], rgba[1], rgba[2], rgba[3] || 0)
            if (max > 1.01) {
                return [rgba[0] / 255, rgba[1] / 255, rgba[2] / 255, (rgba[3] !== undefined ? rgba[3] / 255 : 1)]
            }
            return [rgba[0], rgba[1], rgba[2], rgba[3] !== undefined ? rgba[3] : 1]
        }
        try {
            const nr = normalizeRgba(vis && vis.rgba)
            if (nr) {
                const c = new THREE.Color(nr[0], nr[1], nr[2])
                const opacity = nr[3]
                mat = new THREE.MeshStandardMaterial({ color: c, metalness: 0.1, roughness: 0.6, transparent: opacity < 0.999, opacity: opacity })
                ;(mat as any).emissive = new THREE.Color(c).multiplyScalar(0.02)
            }
            // Support several common geometry types (numbers used by some pybullet versions)
            const gt = vis && vis.geom_type
            const gtStr = typeof gt === 'string' ? gt.toLowerCase() : null
            const gtNum = typeof gt === 'number' ? gt : null
            // helpers
            const mkBox = (d: number[]) => new THREE.Mesh(new THREE.BoxGeometry(Math.max(0.001, d[0] * 2), Math.max(0.001, d[1] * 2), Math.max(0.001, d[2] * 2)), mat)
            const mkSphere = (r: number) => new THREE.Mesh(new THREE.SphereGeometry(Math.max(0.001, r), 24, 16), mat)
            const mkCylinder = (radius: number, height: number) => new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, Math.max(0.001, height), 24), mat)
            const mkPlane = (size = 10) => new THREE.Mesh(new THREE.PlaneGeometry(size, size), mat)

            // Box
            if (gtNum === 1 || gtStr === 'box' || (vis && Array.isArray(vis.dimensions) && vis.dimensions.length === 3 && gtNum == null)) {
                const d = vis.dimensions || [0.1, 0.1, 0.1]
                return mkBox(d)
            }
            // Sphere
            if (gtNum === 2 || gtStr === 'sphere') {
                const r = (vis.dimensions && vis.dimensions[0]) ? vis.dimensions[0] : 0.05
                return mkSphere(r)
            }
            // Cylinder
            if (gtNum === 3 || gtStr === 'cylinder') {
                const d = vis.dimensions || [0.05, 0.1]
                const radius = Math.max(0.001, d[0])
                const height = Math.max(0.001, d[1] * 2)
                return mkCylinder(radius, height)
            }
            // Mesh (use bounding box if available; loader not implemented here)
            if (gtNum === 4 || gtStr === 'mesh' || (vis && vis.filename)) {
                if (vis.dimensions && vis.dimensions.length >= 3) {
                    return mkBox(vis.dimensions)
                }
                return mkBox([0.1, 0.1, 0.1])
            }
            // Capsule (some pybullet versions use 5)
            if (gtNum === 5 || gtStr === 'capsule') {
                // capsule: dimensions may be [radius, height]
                const d = vis.dimensions || [0.05, 0.2]
                const radius = Math.max(0.001, d[0])
                const cylHeight = Math.max(0.001, d[1])
                const group = new THREE.Group()
                const cyl = mkCylinder(radius, cylHeight)
                const top = mkSphere(radius)
                const bot = mkSphere(radius)
                top.position.set(0, 0, cylHeight / 2)
                bot.position.set(0, 0, -cylHeight / 2)
                group.add(cyl)
                group.add(top)
                group.add(bot)
                return group
            }
            // Plane
            if (gtStr === 'plane') {
                return mkPlane(10)
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
        renderer.shadowMap.enabled = true
        renderer.shadowMap.type = THREE.PCFSoftShadowMap
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
        dir.castShadow = true
        dir.shadow.mapSize.width = 1024
        dir.shadow.mapSize.height = 1024
        dir.shadow.camera.near = 0.5
        dir.shadow.camera.far = 50
        scene3.add(dir)
        dirRef.current = dir

        const grid = new THREE.GridHelper(10, 20, 0x888888, 0xdddddd)
        scene3.add(grid)
        gridRef.current = grid

        // (no debug helpers)

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
                    // if body has base pose, apply it to the group when present later in updates
                    entry = { group: g, links: new Map() }
                    bodyMapLocal.set(bid, entry)
                }
                for (const link of body.links || []) {
                    const idx = link.link_index
                    if (entry.links.has(idx)) continue
                    const vis = link.visual
                    const mesh = buildMeshFromVisual(vis)
                    mesh.name = `body-${bid}-link-${idx}`
                    mesh.castShadow = true
                    mesh.receiveShadow = true
                    entry.group.add(mesh)
                    entry.links.set(idx, mesh)
                }
            }
        }

        function animate() {
            if (frames === 0) {
                console.log('[ThreeScene] animate loop starting')
            }
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
                    // if backend provides base pose, apply to group for organizational clarity
                    if (body.base_position && body.base_position.length === 3) entry.group.position.set(body.base_position[0], body.base_position[1], body.base_position[2])
                    if (body.base_orientation && body.base_orientation.length === 4) entry.group.quaternion.set(body.base_orientation[0], body.base_orientation[1], body.base_orientation[2], body.base_orientation[3])
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
            const w = mount.current.clientWidth || window.innerWidth
            const h = mount.current.clientHeight || window.innerHeight
            if (w <= 0 || h <= 0) return
            camera.aspect = w / h
            camera.updateProjectionMatrix()
            renderer.setSize(w, h)
        }
        resize()
        window.addEventListener('resize', resize)
        // observe mount size changes (handles layout timing and container resizing)
        const ro = new ResizeObserver(() => resize())
        ro.observe(el)

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current)
            renderer.dispose()
            window.removeEventListener('resize', resize)
            try { ro.disconnect() } catch {}
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

    // Apply renderer options when toggles change
    useEffect(() => {
        const renderer = rendererRef.current
        if (renderer) renderer.shadowMap.enabled = shadowsEnabled
        const dir = dirRef.current
        if (dir) dir.castShadow = shadowsEnabled
        const grid = gridRef.current
        if (grid) grid.visible = gridVisible

        // update existing meshes for wireframe/shadow settings
        const bm = bodyMapRef.current
        for (const entry of bm.values()) {
            const group = entry.group as THREE.Object3D
            group.traverse((obj: any) => {
                if (obj.isMesh) {
                    if (obj.material) {
                        try { (obj.material as any).wireframe = wireframeEnabled } catch (e) {}
                    }
                    obj.castShadow = shadowsEnabled
                    obj.receiveShadow = shadowsEnabled
                }
            })
        }
    }, [shadowsEnabled, wireframeEnabled, gridVisible])

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
                    {/* Renderer toggles: shadows, wireframe, grid */}
                    <button className={"cam-btn" + (shadowsEnabled ? ' cam-btn--active' : '')} title="Toggle Shadows" onClick={(e) => { e.stopPropagation(); setShadowsEnabled(!shadowsEnabled); }} aria-pressed={shadowsEnabled}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2v6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><path d="M5 12l7 7 7-7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
                    </button>
                    <button className={"cam-btn" + (wireframeEnabled ? ' cam-btn--active' : '')} title="Wireframe" onClick={(e) => { e.stopPropagation(); setWireframeEnabled(!wireframeEnabled); }} aria-pressed={wireframeEnabled}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="4" width="16" height="16" stroke="currentColor" strokeWidth="1.6" /><path d="M4 4l16 16" stroke="currentColor" strokeWidth="1.6" /></svg>
                    </button>
                    <button className={"cam-btn" + (gridVisible ? ' cam-btn--active' : '')} title="Grid" onClick={(e) => { e.stopPropagation(); setGridVisible(!gridVisible); }} aria-pressed={gridVisible}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 12h16M12 4v16" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
                    </button>

                    <CameraButton mode={'orbit'} title="Orbit" svg={(<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="7" stroke="currentColor" strokeWidth="1.6" /><circle cx="12" cy="8" r="1.2" fill="currentColor" /></svg>)} />
                    <CameraButton mode={'top'} title="Top" svg={(<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="5" y="5" width="14" height="6" stroke="currentColor" strokeWidth="1.6" rx="1" /><rect x="8" y="13" width="8" height="6" stroke="currentColor" strokeWidth="1.6" rx="1" /></svg>)} />
                    <CameraButton mode={'side'} title="Side" svg={(<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 12h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><circle cx="18" cy="12" r="3" stroke="currentColor" strokeWidth="1.6" /></svg>)} />
                    <CameraButton mode={'front'} title="Front" svg={(<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="6" width="16" height="12" stroke="currentColor" strokeWidth="1.6" rx="1" /><circle cx="12" cy="12" r="2" fill="currentColor" /></svg>)} />
                    <CameraButton mode={'follow'} title="Follow" svg={(<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 3v3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><path d="M12 18v3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><circle cx="12" cy="12" r="6" stroke="currentColor" strokeWidth="1.6" /></svg>)} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', marginLeft: 8 }}>
                    <div className="stat-box card glass" style={{ padding: '6px 8px', minWidth: 140, display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'flex-start' }}>
                        <div style={{ fontSize: 12, color: 'inherit', whiteSpace: 'nowrap', display: 'flex', gap: 8 }}><strong style={{ minWidth: 56 }}>Frame:</strong><span style={{ flex: '0 0 auto' }}>{fps ? `${(1000 / fps).toFixed(1)} ms` : '—'}</span></div>
                        <div style={{ fontSize: 12, color: 'inherit', whiteSpace: 'nowrap', display: 'flex', gap: 8 }}><strong style={{ minWidth: 56 }}>Server:</strong><span style={{ flex: '0 0 auto' }}>{serverIntervalMs != null ? `${serverIntervalMs.toFixed(1)} ms` : '—'}</span></div>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div ref={mount} className="three-mount">
            <div style={{ position: 'absolute', right: 12, top: 12, zIndex: 30 }}>
                <CameraUI />
            </div>
        </div>
    )
}

