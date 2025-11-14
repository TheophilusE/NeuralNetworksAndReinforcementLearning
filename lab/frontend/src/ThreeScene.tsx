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

    let lastFpsTime = performance.now()
    let frames = 0

    function updateFromState() {
      const s = stateRef.current
      if (!s) return
      // prefer world positions from pybullet if available
      if (s.pos2 || s.pos1) {
        if (s.pos1) {
          const [x, y, z] = s.pos1
          joint1.position.set(x, y, z)
          rod1.visible = true
          joint1.visible = true
        }
        if (s.pos2) {
          const [x, y, z] = s.pos2
          joint2.position.set(x, y, z)
          rod2.visible = true
          joint2.visible = true
        }
        if (s.pos1 && s.pos2) {
          const [x1, y1, z1] = s.pos1
          const [x2, y2, z2] = s.pos2
          rod1.position.set((0 + x1) / 2, (0 + y1) / 2, (1.5 + z1) / 2)
          rod2.position.set((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)
        }
        // follow camera behavior: smoothly move camera behind the first link
        if (cameraMode === 'follow' && s.pos1 && cameraRef.current && controlsRef.current) {
          const [x, y, z] = s.pos1
          const cam = cameraRef.current
          const controls = controlsRef.current
          const desired = new THREE.Vector3(x, y - 2.2, z + 1.2)
          cam.position.lerp(desired, 0.12)
          const t = new THREE.Vector3(x, y, z)
          controls.target.lerp(t, 0.18)
          controls.update()
        }
      } else if (s.theta !== undefined) {
        const theta = s.theta
        const l = 1.0
        const x = l * Math.sin(theta)
        const z = 1.0 - l * Math.cos(theta)
        rod1.position.set(x / 2, 0, z / 2 + 0.5)
        rod1.rotation.set(0, 0, -theta)
        joint1.position.set(x, 0, z + 0.5)
        rod2.visible = false
        joint2.visible = false
      } else if (s.th1 !== undefined) {
        const th1 = s.th1
        const th2 = s.th2
        const l = 1.0
        const x1 = l * Math.sin(th1)
        const z1 = 1.0 - l * Math.cos(th1)
        const x2 = x1 + l * Math.sin(th2)
        const z2 = z1 - l * Math.cos(th2)
        rod1.position.set(x1 / 2, 0, z1 / 2 + 0.5)
        rod1.rotation.set(0, 0, -th1)
        joint1.position.set(x1, 0, z1 + 0.5)
        rod2.position.set((x1 + x2) / 2, 0, (z1 + z2) / 2 + 0.5)
        rod2.rotation.set(0, 0, -th2)
        joint2.position.set(x2, 0, z2 + 0.5)
        rod2.visible = true
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

  // Apply camera presets when cameraMode changes
  useEffect(() => {
    const cam = cameraRef.current
    const controls = controlsRef.current
    if (!cam || !controls) return
    switch (cameraMode) {
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
  }, [cameraMode])

  function CameraUI() {
    return (
      <div className="camera-panel" onPointerDown={(e) => e.stopPropagation()}>
        <StyledSelect value={cameraMode} onChange={(v) => setCameraMode(v as any)} options={opts} />
      </div>
    )
  }

  function StyledSelect({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: Array<{ value: string; label: string }> }) {
    const [open, setOpen] = useState(false)
    const [highlight, setHighlight] = useState<number>(-1)
    const rootRef = useRef<HTMLDivElement | null>(null)
    const controlRef = useRef<HTMLButtonElement | null>(null)
    const justOpenedRef = useRef(false)

    // keep orbit controls disabled while menu open so canvas doesn't steal pointer events
    useEffect(() => {
      try {
        if (controlsRef && controlsRef.current) controlsRef.current.enabled = !open
      } catch (e) {}
      return () => {
        try {
          if (controlsRef && controlsRef.current) controlsRef.current.enabled = true
        } catch (e) {}
      }
    }, [open])

    // close on outside pointer down (capture phase to avoid event ordering races)
    useEffect(() => {
      function onDoc(e: PointerEvent) {
        if (!rootRef.current) return
        // ignore the immediate document event triggered when the control itself opens the menu
        if (justOpenedRef.current) return
        const path: EventTarget[] | undefined = (e as any).composedPath ? (e as any).composedPath() : undefined
        const clickedInside = path ? (path as EventTarget[]).includes(rootRef.current as EventTarget) : rootRef.current.contains(e.target as Node)
        if (!clickedInside) {
          setOpen(false)
        }
      }
      document.addEventListener('pointerdown', onDoc, true)
      return () => document.removeEventListener('pointerdown', onDoc, true)
    }, [])

    // reset highlight when opening/closing
    useEffect(() => {
      if (open) setHighlight(options.findIndex((o) => o.value === value) || 0)
      else setHighlight(-1)
    }, [open, value, options])

    function selectIndex(i: number) {
      const opt = options[i]
      if (!opt) return
      onChange(opt.value)
      setOpen(false)
      // return focus to control for keyboard users
      window.requestAnimationFrame(() => controlRef.current?.focus())
    }

    function onControlKey(e: React.KeyboardEvent) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        if (!open) setOpen(true)
        else setHighlight((h) => Math.min(options.length - 1, Math.max(0, h + 1)))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        if (!open) setOpen(true)
        else setHighlight((h) => Math.max(0, h - 1))
      } else if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        if (!open) setOpen(true)
        else if (highlight >= 0) selectIndex(highlight)
      } else if (e.key === 'Escape') {
        e.preventDefault()
        setOpen(false)
      }
    }

    return (
      <div ref={rootRef} className="styled-select" onPointerDown={(e) => e.stopPropagation()}>
        <button
          ref={controlRef}
          type="button"
          className="styled-select__control glass"
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={() => {
            // set a short-lived guard so the capture-phase document handler doesn't
            // treat the same pointer event as an outside click and immediately close.
            if (!open) {
              justOpenedRef.current = true
              // clear on next tick
              window.setTimeout(() => (justOpenedRef.current = false), 0)
            }
            setOpen((s) => !s)
          }}
          onKeyDown={onControlKey}
        >
          <div className="styled-select__label">{options.find((o) => o.value === value)?.label}</div>
          <div className="styled-select__chev">▾</div>
        </button>

        {open && (
          <div className="styled-select__list pop" role="listbox" aria-activedescendant={highlight >= 0 ? `styled-select-opt-${options[highlight].value}` : undefined} tabIndex={-1} onPointerDown={(e) => e.stopPropagation()}>
            {options.map((o, i) => (
              <div
                id={`styled-select-opt-${o.value}`}
                key={o.value}
                role="option"
                aria-selected={o.value === value}
                className={"styled-select__item " + (o.value === value ? 'styled-select__item--active' : '') + (i === highlight ? ' styled-select__item--highlight' : '')}
                onPointerDown={(e) => e.preventDefault()}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => selectIndex(i)}
              >
                {o.label}
              </div>
            ))}
          </div>
        )}
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
