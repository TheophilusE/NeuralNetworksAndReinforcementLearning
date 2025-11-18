import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

export default function ThreeScene({ state, scene, onFps, currentTrack }: any) {
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
    const labelDomRef = useRef<HTMLDivElement | null>(null)
    const trackVisualRef = useRef<any>(null)
    const prevTrackRef = useRef<number | null>(null)
    const [cameraMode, setCameraMode] = useState<'orbit' | 'top' | 'side' | 'front' | 'follow'>('orbit')
    const [shadowsEnabled, setShadowsEnabled] = useState<boolean>(true)
    const [wireframeEnabled, setWireframeEnabled] = useState<boolean>(false)
    const [gridVisible, setGridVisible] = useState<boolean>(true)
    const [serverIntervalMs, setServerIntervalMs] = useState<number | null>(null)
    const lastServerTsRef = useRef<number | null>(null)

    useEffect(() => {
        stateRef.current = state
        try {
            const now = performance.now()
            const last = lastServerTsRef.current
            if (last != null) setServerIntervalMs(now - last)
            lastServerTsRef.current = now
        } catch { }
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
                    ; (mat as any).emissive = new THREE.Color(c).multiplyScalar(0.02)
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
        } catch (e) { }
        geom = new THREE.BoxGeometry(0.12, 0.12, 0.12)
        return new THREE.Mesh(geom, mat)
    }

    useEffect(() => {
        const el = mount.current!
        const startedRef = { started: false }
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

        // Track visualization: group with bar, dashed line, end caps and label
        const trackGroup = new THREE.Group()
        trackGroup.name = 'track-group'
        // base bar (thin box) - base width=1, scale X to change length
        const barMat = new THREE.MeshStandardMaterial({ color: 0x2b7cff, metalness: 0.2, roughness: 0.4 })
        const barGeom = new THREE.BoxGeometry(1, 0.02, 0.02)
        const barMesh = new THREE.Mesh(barGeom, barMat)
        barMesh.position.set(0, 0, 0.01)
        barMesh.receiveShadow = false
        barMesh.castShadow = false
        trackGroup.add(barMesh)

        // dashed center line
        const lineMat = new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.05, gapSize: 0.03, linewidth: 1 })
        const lineGeom = new THREE.BufferGeometry()
        const positions = new Float32Array([-0.5, 0, 0.02, 0.5, 0, 0.02])
        lineGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
        lineGeom.computeBoundingSphere()
        const line = new THREE.Line(lineGeom, lineMat)
            ; (line.geometry as any).computeLineDistances && (line.geometry as any).computeLineDistances()
        trackGroup.add(line)

        // spherical end caps
        const capMat = new THREE.MeshStandardMaterial({ color: 0xffcc33, metalness: 0.3, roughness: 0.5 })
        const capGeom = new THREE.SphereGeometry(0.03, 16, 12)
        const capA = new THREE.Mesh(capGeom, capMat)
        const capB = new THREE.Mesh(capGeom, capMat)
        capA.position.set(-0.5, 0, 0.03)
        capB.position.set(0.5, 0, 0.03)
        capA.castShadow = false
        capB.castShadow = false
        trackGroup.add(capA)
        trackGroup.add(capB)

        // HTML overlay used for crisp text

        // initialize to sensible default so the track is visible at startup
        const initialLength = (typeof currentTrack === 'number' && Number.isFinite(currentTrack) && currentTrack > 0) ? currentTrack : 2.0
        // apply initial scale/positions
        barMesh.scale.set(initialLength, 1, 1)
        const halfInit = initialLength / 2
        capA.position.set(-halfInit, 0, 0.03)
        capB.position.set(halfInit, 0, 0.03)
        // update line geometry positions to match
        const posArr = (line.geometry as THREE.BufferGeometry).attributes.position as any
        posArr.array[0] = -halfInit
        posArr.array[1] = 0
        posArr.array[2] = 0.02
        posArr.array[3] = halfInit
        posArr.array[4] = 0
        posArr.array[5] = 0.02
        posArr.needsUpdate = true
            ; (line.geometry as any).computeLineDistances && (line.geometry as any).computeLineDistances()
        trackGroup.visible = true
        scene3.add(trackGroup)
        trackVisualRef.current = { group: trackGroup, bar: barMesh, line, caps: [capA, capB] }

        // create an HTML label overlay for crisp text; position will be updated each frame
        try {
            const labelDiv = document.createElement('div')
            labelDiv.className = 'track-label'
            labelDiv.style.position = 'absolute'
            labelDiv.style.pointerEvents = 'none'
            labelDiv.style.padding = '4px 8px'
            labelDiv.style.background = 'transparent'
            labelDiv.style.color = 'white'
            labelDiv.style.fontFamily = 'monospace, sans-serif'
            labelDiv.style.fontSize = '14px'
            labelDiv.style.textShadow = '0 2px 4px rgba(0,0,0,0.7)'
            labelDiv.style.transform = 'translate(-50%, -120%)'
            labelDiv.style.whiteSpace = 'nowrap'
            labelDiv.style.zIndex = '1000'
            labelDiv.textContent = `${initialLength.toFixed(2)} m`
            el.appendChild(labelDiv)
            labelDomRef.current = labelDiv
        } catch (e) { }

        // (no debug helpers)

        const root = new THREE.Group()
        scene3.add(root)
        sceneGraphRootRef.current = root
        bodyMapRef.current = new Map()

        let lastFpsTime = performance.now()
        let frames = 0
        const POS_LERP = 0.22
        const ROT_LERP = 0.22
        // snap thresholds to avoid long lerps after a reset or large teleport
        const SNAP_DIST = 0.5 // meters
        const SNAP_ANGLE = Math.PI / 2 // radians
        const _tmpTargetPos = new THREE.Vector3()
        const _tmpTargetQuat = new THREE.Quaternion()

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
            if (frames === 0 && !startedRef.started) {
                // Log once per mounted instance to reduce console spam in StrictMode
                console.log('[ThreeScene] animate loop starting')
                startedRef.started = true
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
            // update transforms (smoothed to avoid popping)
            const sc = sceneRef.current
            if (sc && sc.bodies) {
                const bodyMapLocal = bodyMapRef.current
                for (const body of sc.bodies) {
                    const bid = body.body_id
                    const entry = bodyMapLocal.get(bid)
                    if (!entry) continue
                    // if backend provides base pose, apply to group for organizational clarity (smooth update)
                    if (body.base_position && body.base_position.length === 3) {
                        _tmpTargetPos.set(body.base_position[0], body.base_position[1], body.base_position[2])
                        if (!entry.group.userData._posInit) {
                            entry.group.position.copy(_tmpTargetPos)
                            entry.group.userData._posInit = true
                        } else {
                            entry.group.position.lerp(_tmpTargetPos, POS_LERP)
                        }
                    }
                    if (body.base_orientation && body.base_orientation.length === 4) {
                        _tmpTargetQuat.set(body.base_orientation[0], body.base_orientation[1], body.base_orientation[2], body.base_orientation[3])
                        if (!entry.group.userData._quatInit) {
                            entry.group.quaternion.copy(_tmpTargetQuat)
                            entry.group.userData._quatInit = true
                        } else {
                            entry.group.quaternion.slerp(_tmpTargetQuat, ROT_LERP)
                        }
                    }
                    for (const link of body.links || []) {
                        const mesh = entry.links.get(link.link_index)
                        if (!mesh) continue
                        const wp = link.world_position
                        const wo = link.world_orientation
                        if (wp && wp.length === 3) {
                            _tmpTargetPos.set(wp[0], wp[1], wp[2])
                            if (!mesh.userData._posInit) {
                                    mesh.position.copy(_tmpTargetPos)
                                    mesh.userData._posInit = true
                                } else {
                                    // if target is far from current position (e.g. a reset), snap immediately
                                    const dist = mesh.position.distanceTo(_tmpTargetPos)
                                    const dynamicSnap = (typeof currentTrack === 'number' && currentTrack > 0) ? Math.min(SNAP_DIST, currentTrack * 0.25) : SNAP_DIST
                                    if (dist > dynamicSnap) {
                                        mesh.position.copy(_tmpTargetPos)
                                    } else {
                                        mesh.position.lerp(_tmpTargetPos, POS_LERP)
                                    }
                                }
                        }
                        if (wo && wo.length === 4) {
                            _tmpTargetQuat.set(wo[0], wo[1], wo[2], wo[3])
                            if (!mesh.userData._quatInit) {
                                    mesh.quaternion.copy(_tmpTargetQuat)
                                    mesh.userData._quatInit = true
                                } else {
                                    // measure quaternion difference and snap for large rotations
                                    const dot = Math.abs(mesh.quaternion.dot(_tmpTargetQuat))
                                    const angle = 2 * Math.acos(Math.min(1, dot))
                                    if (angle > SNAP_ANGLE) {
                                        mesh.quaternion.copy(_tmpTargetQuat)
                                    } else {
                                        mesh.quaternion.slerp(_tmpTargetQuat, ROT_LERP)
                                    }
                                }
                        }
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
            // update HTML label position each frame so it faces the camera
            try {
                const labelEl = labelDomRef.current
                if (labelEl && cameraRef.current && mount.current) {
                    const width = mount.current.clientWidth
                    const height = mount.current.clientHeight
                    const worldPos = new THREE.Vector3(0, 0, 0.08)
                    worldPos.project(cameraRef.current)
                    const x = (worldPos.x + 1) / 2 * width
                    const y = (-worldPos.y + 1) / 2 * height
                    labelEl.style.left = `${x}px`
                    labelEl.style.top = `${y}px`
                }
            } catch (e) { }
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
            try { ro.disconnect() } catch { }
            if (el && renderer.domElement) el.removeChild(renderer.domElement)
            // cleanup track visuals
            try {
                if (trackVisualRef.current) {
                    const t = trackVisualRef.current
                    t.bar.geometry.dispose()
                        ; (t.bar.material as any).dispose()
                    t.line.geometry.dispose()
                        ; (t.line.material as any).dispose()
                    for (const c of t.caps) { c.geometry.dispose(); (c.material as any).dispose() }
                }
            } catch { }
            // remove html label
            try {
                if (labelDomRef.current && el) {
                    try { el.removeChild(labelDomRef.current) } catch { } 
                    labelDomRef.current = null
                }
            } catch { }
        }
    }, [])

    // Update track visualization when parent provides a track length
    useEffect(() => {
        try {
            // access the visual we stored on setup (safe-guard if setup hasn't run yet)
            const root = sceneGraphRootRef.current?.parent as THREE.Scene | undefined
            if (!root) return
            const trackGroup = root.getObjectByName('track-group') as THREE.Group | undefined
            if (!trackGroup) return
            // extract parts
            const parts: any = (trackGroup as any)
            const bar = parts.children.find((c: any) => c.geometry && c.geometry.type === 'BoxGeometry') as THREE.Mesh | undefined
            const line = parts.children.find((c: any) => c.type === 'Line') as THREE.Line | undefined
            const caps = parts.children.filter((c: any) => c.geometry && c.geometry.type === 'SphereGeometry') as THREE.Mesh[]
            // (no 3D sprite label present)
            if (typeof currentTrack === 'number' && Number.isFinite(currentTrack) && currentTrack > 0) {
                // bar base width is 1: scale X to desired length
                if (bar) bar.scale.set(currentTrack, 1, 1)
                // position caps at +/- length/2
                const half = currentTrack / 2
                if (caps && caps.length >= 2) {
                    caps[0].position.set(-half, 0, 0.03)
                    caps[1].position.set(half, 0, 0.03)
                }
                // update dashed line geometry
                if (line) {
                    const pos = (line.geometry as THREE.BufferGeometry).attributes.position as any
                    pos.array[0] = -half
                    pos.array[3] = half
                    pos.needsUpdate = true
                        ; (line.geometry as any).computeBoundingSphere && (line.geometry as any).computeBoundingSphere()
                        ; (line.geometry as any).computeLineDistances && (line.geometry as any).computeLineDistances()
                }
                // update HTML label text and animate pulse
                try {
                    const labelEl = labelDomRef.current
                    if (labelEl) {
                        labelEl.textContent = `${currentTrack.toFixed(2)} m`
                        labelEl.style.transition = 'transform 260ms cubic-bezier(.2,.8,.2,1), opacity 260ms'
                        labelEl.style.transform = 'translate(-50%, -120%) scale(1.25)'
                        labelEl.style.opacity = '1'
                        setTimeout(() => { try { labelEl.style.transform = 'translate(-50%, -120%) scale(1)' } catch { } }, 260)
                    }
                } catch (e) { }

                // animate 3D bar pulse along Y/Z to give feedback
                try {
                    const tvis = trackVisualRef.current
                    if (tvis && tvis.bar) {
                        const bar = tvis.bar as THREE.Mesh
                        const baseY = (bar.scale.y && bar.scale.y > 0) ? bar.scale.y : 1
                        const pulseMax = 1.5
                        const dur = 300
                        const start = performance.now()
                        const step = (now: number) => {
                            const p = Math.min(1, (now - start) / dur)
                            const pulse = 1 + Math.sin(p * Math.PI) * (pulseMax - 1)
                            bar.scale.y = pulse
                            bar.scale.z = pulse
                            if (p < 1) requestAnimationFrame(step)
                            else { bar.scale.y = baseY; bar.scale.z = baseY }
                        }
                        requestAnimationFrame(step)
                    }
                } catch (e) { }

                trackGroup.visible = true
            } else {
                trackGroup.visible = false
            }
        } catch (e) { }
    }, [currentTrack])

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
                        try { (obj.material as any).wireframe = wireframeEnabled } catch (e) { }
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
                        <div style={{ fontSize: 12, color: 'inherit', whiteSpace: 'nowrap', display: 'flex', gap: 8 }}><strong style={{ minWidth: 56 }}>Frame:</strong><span style={{ flex: '0 0 auto' }}>{fps ? `${(1000 / fps).toFixed(1)} ms` : '-'}</span></div>
                        <div style={{ fontSize: 12, color: 'inherit', whiteSpace: 'nowrap', display: 'flex', gap: 8 }}><strong style={{ minWidth: 56 }}>Server:</strong><span style={{ flex: '0 0 auto' }}>{serverIntervalMs != null ? `${serverIntervalMs.toFixed(1)} ms` : '-'}</span></div>
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

