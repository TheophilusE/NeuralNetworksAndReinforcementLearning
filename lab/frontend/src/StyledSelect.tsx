import React, { useEffect, useRef, useState } from 'react'

export default function StyledSelect({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: Array<{ value: string; label: string }> }) {
  const [open, setOpen] = useState(false)
  const [highlight, setHighlight] = useState<number>(-1)
  const rootRef = useRef<HTMLDivElement | null>(null)
  const controlRef = useRef<HTMLButtonElement | null>(null)
  const justOpenedRef = useRef(false)

  // keep orbit controls disabled while menu open so canvas doesn't steal pointer events
  useEffect(() => {
    // Note: consumer can manage orbit controls state when needed; this file avoids referencing them.
    return
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
