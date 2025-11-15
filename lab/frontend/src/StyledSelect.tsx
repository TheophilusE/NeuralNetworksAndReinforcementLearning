import React, { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

export default function StyledSelect({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: Array<{ value: string; label: string }> }) {
    const [open, setOpen] = useState(false)
    const [highlight, setHighlight] = useState<number>(-1)
    const rootRef = useRef<HTMLDivElement | null>(null)
    const controlRef = useRef<HTMLButtonElement | null>(null)
    const listRef = useRef<HTMLDivElement | null>(null)
    const justOpenedRef = useRef(false)
    const suppressToggleRef = useRef(false)
    const [menuPos, setMenuPos] = useState<{ left: number; top: number; width?: number } | null>(null)

    // close on outside pointer down (capture phase to avoid event ordering races)
    useEffect(() => {
        function onDoc(e: PointerEvent) {
            if (!rootRef.current) return
            // ignore the immediate document event triggered when the control itself opens the menu
            if (justOpenedRef.current) return
            const path: EventTarget[] | undefined = (e as any).composedPath ? (e as any).composedPath() : undefined
            const clickedInside = path
                ? (path as EventTarget[]).includes(rootRef.current as EventTarget) || (listRef.current && (path as EventTarget[]).includes(listRef.current as EventTarget))
                : rootRef.current.contains(e.target as Node) || (listRef.current && listRef.current.contains(e.target as Node))
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
        // prevent the control's click handler (if it sees the same event) from
        // toggling the menu back open. Clear on next tick.
        suppressToggleRef.current = true
        window.setTimeout(() => (suppressToggleRef.current = false), 0)
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

    // compute & update menu position when opening or on resize/scroll
    useEffect(() => {
        function updatePos() {
            const ctrl = controlRef.current
            if (!ctrl) return
            const r = ctrl.getBoundingClientRect()
            // do not force a width so the dropdown can size to its content
            setMenuPos({ left: r.left + window.scrollX, top: r.bottom + window.scrollY + 8 })
        }
        if (open) updatePos()
        window.addEventListener('resize', updatePos)
        window.addEventListener('scroll', updatePos, true)
        return () => {
            window.removeEventListener('resize', updatePos)
            window.removeEventListener('scroll', updatePos, true)
        }
    }, [open])

    return (
        <div ref={rootRef} className="styled-select ui-top" onPointerDown={(e) => e.stopPropagation()}>
            <button
                ref={controlRef}
                type="button"
                className="styled-select__control glass ui-top"
                aria-haspopup="listbox"
                aria-expanded={open}
                onClick={() => {
                    // If a selection just occurred, suppress the next toggle to avoid
                    // close-then-reopen races when clicks bubble/ordering is complex.
                    if (suppressToggleRef.current) {
                        suppressToggleRef.current = false
                        return
                    }
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
            {open && controlRef.current && (
                createPortal(
                    <div
                        ref={listRef}
                        className="styled-select__list pop ui-top"
                        role="listbox"
                        aria-activedescendant={highlight >= 0 ? `styled-select-opt-${options[highlight].value}` : undefined}
                        tabIndex={-1}
                        onPointerDown={(e) => e.stopPropagation()}
                        style={{
                            position: 'absolute',
                            left: menuPos ? menuPos.left : 0,
                            top: menuPos ? menuPos.top : 0,
                            zIndex: 100000,
                        }}
                    >
                        {options.map((o, i) => (
                            <div
                                id={`styled-select-opt-${o.value}`}
                                key={o.value}
                                role="option"
                                aria-selected={o.value === value}
                                className={"styled-select__item " + (o.value === value ? 'styled-select__item--active' : '') + (i === highlight ? ' styled-select__item--highlight' : '')}
                                onPointerDown={(e) => e.stopPropagation()}
                                onClick={(e) => { e.stopPropagation(); selectIndex(i) }}
                            >
                                {o.label}
                            </div>
                        ))}
                    </div>,
                    document.body
                )
            )}
        </div>
    )
}
