import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

export default function ThreeScene({ state, onFps }: any) {
  const mount = useRef<HTMLDivElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const [fps, setFps] = useState<number | undefined>(undefined)

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

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.target.set(0, 0, 1)

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
      if (!state) return
      // prefer world positions from pybullet if available
      if (state.pos2 || state.pos1) {
        if (state.pos1) {
          const [x, y, z] = state.pos1
          joint1.position.set(x, y, z)
        }
        if (state.pos2) {
          const [x, y, z] = state.pos2
          joint2.position.set(x, y, z)
        }
        if (state.pos1 && state.pos2) {
          const [x1, y1, z1] = state.pos1
          const [x2, y2, z2] = state.pos2
          rod1.position.set((0 + x1) / 2, (0 + y1) / 2, (1.5 + z1) / 2)
          rod2.position.set((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)
        }
      } else if (state.theta !== undefined) {
        const theta = state.theta
        const l = 1.0
        const x = l * Math.sin(theta)
        const z = 1.0 - l * Math.cos(theta)
        rod1.position.set(x / 2, 0, z / 2 + 0.5)
        rod1.rotation.set(0, 0, -theta)
        joint1.position.set(x, 0, z + 0.5)
        rod2.visible = false
        joint2.visible = false
      } else if (state.th1 !== undefined) {
        const th1 = state.th1
        const th2 = state.th2
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

    const onResize = () => {
      if (!mount.current) return
      const w = mount.current.clientWidth
      const h = mount.current.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      renderer.dispose()
      window.removeEventListener('resize', onResize)
      if (el && renderer.domElement) el.removeChild(renderer.domElement)
    }
  }, [])

  useEffect(() => {
    // state updates handled in animate via closure
  }, [state])

  return <div ref={mount} style={{ width: '100%', height: 520, position: 'relative' }} />
}
