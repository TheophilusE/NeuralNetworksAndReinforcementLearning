import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

export default function ThreeScene({ state, onFps }: any) {
    const mount = useRef<HTMLDivElement | null>(null)
    const rafRef = useRef<number | null>(null)
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
    const stateRef = useRef<any>(state)
    const [fps, setFps] = useState<number | undefined>(undefined)
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
    const controlsRef = useRef<any>(null)
    const [cameraMode, setCameraMode] = useState<'orbit' | 'top' | 'side' | 'front' | 'follow'>('orbit')

    // keep a ref to latest state so the animation loop (created once) sees updates
    useEffect(() => {
        stateRef.current = state
    }, [state])

    useEffect(() => {
        const el = mount.current!
        const scene = new THREE.Scene()
        scene.background = new THREE.Color(0xeef6ff)

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
        scene.add(hemi)
        const dir = new THREE.DirectionalLight(0xffffff, 0.8)
        dir.position.set(5, -5, 10)
        scene.add(dir)

        const grid = new THREE.GridHelper(10, 20, 0x888888, 0xdddddd)
        scene.add(grid)

        // rods and joints
        const material1 = new THREE.MeshStandardMaterial({ color: 0xff4444 })
        const material2 = new THREE.MeshStandardMaterial({ color: 0x44aaff })
        const rodGeom = new THREE.CylinderGeometry(0.03, 0.03, 1.0, 12)
        const jointGeom = new THREE.SphereGeometry(0.06, 12, 12)

        const rod1 = new THREE.Mesh(rodGeom, material1)
        const joint1 = new THREE.Mesh(jointGeom, material2)
        scene.add(rod1)
        scene.add(joint1)

        const rod2 = new THREE.Mesh(rodGeom, material1)
        const joint2 = new THREE.Mesh(jointGeom, material2)
        scene.add(rod2)
        scene.add(joint2)

        // helper: align a cylinder mesh between two 3D points
        const tmpV1 = new THREE.Vector3()
        const tmpV2 = new THREE.Vector3()
        const tmpDir = new THREE.Vector3()
        const up = new THREE.Vector3(0, 1, 0) // cylinder's default axis
        function alignCylinderBetween(rod: THREE.Mesh, a: [number, number, number] | THREE.Vector3, b: [number, number, number] | THREE.Vector3, initialLength = 1.0) {
            // convert inputs to Vector3
            tmpV1.set((a as any)[0], (a as any)[1], (a as any)[2])
            tmpV2.set((b as any)[0], (b as any)[1], (b as any)[2])
            tmpDir.subVectors(tmpV2, tmpV1)
            const len = tmpDir.length()
            if (len <= 1e-6) {
                rod.visible = false
                return
            }
            // position at midpoint
            rod.position.copy(tmpV1).add(tmpV2).multiplyScalar(0.5)
            // scale the cylinder to match length (height axis is Y)
            rod.scale.set(1, len / initialLength, 1)
            // compute quaternion that rotates up -> dir
            const q = new THREE.Quaternion().setFromUnitVectors(up, tmpDir.clone().normalize())
            rod.quaternion.copy(q)
            rod.visible = true
        }

        let lastFpsTime = performance.now()
        let frames = 0

        function updateFromState() {
            const s = stateRef.current
            if (!s) return

            // Base pivot (matches pybullet basePosition used in backend)
            const basePivot = new THREE.Vector3(0, 0, 1.5)

            // Prefer world positions from pybullet if available
            if (s.pos1 || s.pos2) {
                // convert centers reported by pybullet into link endpoints
                if (s.pos1) {
                    const c1 = new THREE.Vector3(s.pos1[0], s.pos1[1], s.pos1[2])
                    // first link: top is basePivot, center is c1 => bottom = 2*c1 - top
                    const top1 = basePivot.clone()
                    const bottom1 = c1.clone().multiplyScalar(2).sub(top1)
                    alignCylinderBetween(rod1, top1, bottom1, 1.0)
                    joint1.position.copy(bottom1)
                    joint1.visible = true
                } else {
                    rod1.visible = false
                    joint1.visible = false
                }

                if (s.pos2) {
                    const c2 = new THREE.Vector3(s.pos2[0], s.pos2[1], s.pos2[2])
                    // second link top is the joint between link1 and link2. If pos1 present compute it,
                    // otherwise assume top is basePivot (degenerate case)
                    const top2 = s.pos1 ? new THREE.Vector3(s.pos1[0], s.pos1[1], s.pos1[2]).multiplyScalar(2).sub(basePivot) : basePivot.clone()
                    const bottom2 = c2.clone().multiplyScalar(2).sub(top2)
                    alignCylinderBetween(rod2, top2, bottom2, 1.0)
                    joint2.position.copy(bottom2)
                    joint2.visible = true
                } else {
                    rod2.visible = false
                    joint2.visible = false
                }

                // follow camera behavior: smoothly move camera behind the first link (use joint or center)
                if (cameraMode === 'follow' && s.pos1 && cameraRef.current && controlsRef.current) {
                    const cam = cameraRef.current
                    const controls = controlsRef.current
                    const c1 = new THREE.Vector3(s.pos1[0], s.pos1[1], s.pos1[2])
                    const desired = new THREE.Vector3(c1.x, c1.y - 2.2, c1.z + 1.2)
                    cam.position.lerp(desired, 0.12)
                    controls.target.lerp(c1, 0.18)
                    controls.update()
                }
            } else if (s.theta !== undefined) {
                // analytic single-pendulum fallback (no pybullet)
                const theta = s.theta
                const l = 1.0
                const top = basePivot.clone()
                const bottom = new THREE.Vector3(l * Math.sin(theta), 0, 1.5 - l * Math.cos(theta))
                alignCylinderBetween(rod1, top, bottom, l)
                joint1.position.copy(bottom)
                joint1.visible = true
                rod2.visible = false
                joint2.visible = false
            } else if (s.th1 !== undefined) {
                // analytic double-pendulum fallback (angles only)
                const th1 = s.th1
                const th2 = s.th2
                const l = 1.0
                const top1 = basePivot.clone()
                const bottom1 = new THREE.Vector3(l * Math.sin(th1), 0, 1.5 - l * Math.cos(th1))
                alignCylinderBetween(rod1, top1, bottom1, l)
                joint1.position.copy(bottom1)
                joint1.visible = true

                const top2 = bottom1.clone()
                const bottom2 = new THREE.Vector3(bottom1.x + l * Math.sin(th2), 0, bottom1.z - l * Math.cos(th2))
                alignCylinderBetween(rod2, top2, bottom2, l)
                joint2.position.copy(bottom2)
                joint2.visible = true
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

            updateFromState()
            controls.update()
            renderer.render(scene, camera)
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
        // initial size
        resize()
        window.addEventListener('resize', resize)

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current)
            renderer.dispose()
            window.removeEventListener('resize', resize)
            if (el && renderer.domElement) el.removeChild(renderer.domElement)
        }
    }, [])

    useEffect(() => {
        // state updates handled in animate via closure
    }, [state])

    // Apply camera presets when cameraMode changes (or when reapplied)
    function applyCameraMode(mode: 'orbit' | 'top' | 'side' | 'front' | 'follow') {
        const cam = cameraRef.current
        const controls = controlsRef.current
        if (!cam || !controls) return
        switch (mode) {
            case 'top':
                cam.position.set(0, 0, 6)
                controls.target.set(0, 0, 1)
                break
            case 'side':
                cam.position.set(6, 0, 1)
                controls.target.set(0, 0, 1)
                break
            case 'front':
                cam.position.set(0, -6, 1)
                controls.target.set(0, 0, 1)
                break
            case 'orbit':
                cam.position.set(0, -3.5, 2)
                controls.target.set(0, 0, 1)
                break
            case 'follow':
                // follow handled in updateFromState
                break
        }
        cam.updateProjectionMatrix()
        controls.update()
        // update state so UI reflects the active mode
        setCameraMode(mode)
    }

    useEffect(() => {
        // apply preset when cameraMode state changes
        applyCameraMode(cameraMode)
    }, [cameraMode])

    function CameraButton({ mode, title, svg }: { mode: 'orbit' | 'top' | 'side' | 'front' | 'follow'; title: string; svg: JSX.Element }) {
        const active = cameraMode === mode
        return (
            <button
                className={"cam-btn" + (active ? ' cam-btn--active' : '')}
                aria-pressed={active}
                title={title}
                onClick={() => applyCameraMode(mode)}
            >
                {svg}
            </button>
        )
    }

    function CameraUI() {
        return (
            <div className="camera-panel" onPointerDown={(e) => e.stopPropagation()}>
                <div style={{ display: 'flex', gap: 8 }}>
                    <CameraButton
                        mode={'orbit'}
                        title="Orbit"
                        svg={(
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <circle cx="12" cy="12" r="7" stroke="currentColor" strokeWidth="1.6" />
                                <circle cx="12" cy="8" r="1.2" fill="currentColor" />
                            </svg>
                        )}
                    />
                    <CameraButton
                        mode={'top'}
                        title="Top"
                        svg={(
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <rect x="5" y="5" width="14" height="6" stroke="currentColor" strokeWidth="1.6" rx="1" />
                                <rect x="8" y="13" width="8" height="6" stroke="currentColor" strokeWidth="1.6" rx="1" />
                            </svg>
                        )}
                    />
                    <CameraButton
                        mode={'side'}
                        title="Side"
                        svg={(
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M4 12h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                                <circle cx="18" cy="12" r="3" stroke="currentColor" strokeWidth="1.6" />
                            </svg>
                        )}
                    />
                    <CameraButton
                        mode={'front'}
                        title="Front"
                        svg={(
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <rect x="4" y="6" width="16" height="12" stroke="currentColor" strokeWidth="1.6" rx="1" />
                                <circle cx="12" cy="12" r="2" fill="currentColor" />
                            </svg>
                        )}
                    />
                    <CameraButton
                        mode={'follow'}
                        title="Follow"
                        svg={(
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 3v3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                                <path d="M12 18v3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                                <circle cx="12" cy="12" r="6" stroke="currentColor" strokeWidth="1.6" />
                            </svg>
                        )}
                    />
                </div>
            </div>
        )
    }

    const opts = [
        { value: 'orbit', label: 'Orbit' },
        { value: 'top', label: 'Top' },
        { value: 'side', label: 'Side' },
        { value: 'front', label: 'Front' },
        { value: 'follow', label: 'Follow' },
    ]

    return (
        <div ref={mount} className="three-mount">
            <CameraUI />
        </div>
    )
}
