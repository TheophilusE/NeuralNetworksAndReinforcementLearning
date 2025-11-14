import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'

export default function ThreeScene({ state }: any) {
  const mount = useRef<HTMLDivElement | null>(null)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    const el = mount.current!
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.1, 1000)
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(el.clientWidth, el.clientHeight)
    el.appendChild(renderer.domElement)

    camera.position.set(0, -4, 2)
    camera.up.set(0, 0, 1)
    camera.lookAt(0, 0, 1)

    const light = new THREE.DirectionalLight(0xffffff, 1)
    light.position.set(5, -5, 10)
    scene.add(light)

    // rods and joints
    const material1 = new THREE.MeshStandardMaterial({ color: 0xff4444 })
    const material2 = new THREE.MeshStandardMaterial({ color: 0x44aaff })

    const rodGeom = new THREE.CylinderGeometry(0.03, 0.03, 1, 12)
    const jointGeom = new THREE.SphereGeometry(0.06, 12, 12)

    const rod1 = new THREE.Mesh(rodGeom, material1)
    const joint1 = new THREE.Mesh(jointGeom, material2)
    scene.add(rod1)
    scene.add(joint1)

    const rod2 = new THREE.Mesh(rodGeom, material1)
    const joint2 = new THREE.Mesh(jointGeom, material2)
    scene.add(rod2)
    scene.add(joint2)

    function updateFromState() {
      if (!state) return
      if (state.theta !== undefined) {
        const theta = state.theta
        const l = 1.0
        const x = l * Math.sin(theta)
        const z = 1.0 - l * Math.cos(theta)
        // position rod center
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
      updateFromState()
      renderer.render(scene, camera)
      rafRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [])

  useEffect(() => {
    // state updates handled via closure; trigger redraw on prop change
  }, [state])

  return <div ref={mount} style={{ width: '100%', height: 400 }} />
}
