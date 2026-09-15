import React, { useEffect, useRef, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, Tooltip, ResponsiveContainer } from 'recharts'
import { api, streamJob } from './api.js'

// Default editable config mirrored from ProjectConfigModel.
export const DEFAULT_CONFIG = {
  project_name: 'my_model',
  num_samples: 1000,
  waveform: 'IMRPhenomD',
  waveform_length: 2048,
  num_epochs: 200,
  batch_size: 32768,
  amp_lr: 0.0005,
  phase_lr: 0.0005,
  hpo_trials: 50,
  hpo_samples: 1000,
  device: 'cuda',
}

// Shared Recharts theming — monochrome, driven by the CSS theme tokens so it
// adapts to light/dark. Series are distinguished by shade + dash, not hue.
const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'
export const CHART = {
  accent: 'hsl(var(--foreground))',
  accent2: 'hsl(var(--muted-foreground))',
  grid: 'hsl(var(--foreground) / 0.08)',
  axis: 'hsl(var(--muted-foreground))',
  tick: { fill: 'hsl(var(--muted-foreground))', fontSize: 10, fontFamily: MONO },
  legend: { fontFamily: MONO, fontSize: 11 },
  tooltip: {
    contentStyle: {
      background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))',
      borderRadius: 8, fontFamily: MONO, fontSize: 11, color: 'hsl(var(--foreground))',
    },
    labelStyle: { color: 'hsl(var(--foreground))' },
  },
}

// Scramble-reveal effect ported from the reference site (loading flourish).
const POOL = 'FLARE01<>#'
export function useScrambleReveal(text, durationMs = 650) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || !text.trim()) {
      el.textContent = text
      return
    }
    let frame
    const start = performance.now()
    const tick = (now) => {
      const elapsed = now - start
      let out = ''
      for (let i = 0; i < text.length; i++) {
        const ch = text[i]
        if (ch === ' ' || elapsed > (i / text.length) * durationMs) out += ch
        else out += POOL[Math.floor(Math.random() * POOL.length)]
      }
      el.textContent = out
      if (elapsed < durationMs) frame = requestAnimationFrame(tick)
      else el.textContent = text
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [text, durationMs])
  return ref
}

export function PanelHead({ title, subtitle }) {
  const ref = useScrambleReveal(title)
  return (
    <div className="panel-head">
      <h2 ref={ref}>{title}</h2>
      {subtitle && <p>{subtitle}</p>}
    </div>
  )
}

// Two-column shell: parameters/controls on the left, graphs on the right.
export function PanelLayout({ title, subtitle, sidebar, children }) {
  return (
    <div className="panel view-fade">
      <PanelHead title={title} subtitle={subtitle} />
      <div className="panel-layout">
        <aside className="sidebar">{sidebar}</aside>
        <section className="graphs">{children}</section>
      </div>
    </div>
  )
}

// Placeholder shown wherever there is nothing (yet) to plot.
export function Placeholder({ label = 'No data available', loading = false }) {
  return (
    <div className="placeholder">
      {loading ? <span className="spinner" /> : <span className="ph-mark">◇</span>}
      <span>{loading ? 'Working…' : label}</span>
    </div>
  )
}

// Labeled container for a single chart/figure.
export function GraphCard({ title, children }) {
  return (
    <div className="card">
      {title && <div className="card-title">{title}</div>}
      {children}
    </div>
  )
}

// Renders a training job's amp/phase validation-loss curve from its raw SSE
// event list. Shared by TrainingPanel (live launch) and ProjectsPanel (status
// view for a job launched elsewhere).
export function LossCurveChart({ events }) {
  const curve = (type) => events
    .filter((e) => e.model_type === type && e.val_loss !== undefined)
    .map((e) => ({ epoch: e.epoch, loss: e.val_loss }))
  const merged = {}
  curve('amp').forEach((p) => { merged[p.epoch] = { epoch: p.epoch, amp: p.loss } })
  curve('phase').forEach((p) => { merged[p.epoch] = { ...(merged[p.epoch] || { epoch: p.epoch }), phase: p.loss } })
  const chartData = Object.values(merged).sort((a, b) => a.epoch - b.epoch)
  if (chartData.length === 0) return null
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={chartData}>
        <CartesianGrid stroke={CHART.grid} />
        <XAxis dataKey="epoch" tick={CHART.tick} stroke={CHART.axis} />
        <YAxis scale="log" domain={['auto', 'auto']} tick={CHART.tick} stroke={CHART.axis} width={64} />
        <Tooltip {...CHART.tooltip} />
        <Legend wrapperStyle={CHART.legend} />
        <Line type="monotone" dataKey="amp" stroke={CHART.accent} dot={false} name="amp val loss" strokeWidth={1.5} />
        <Line type="monotone" dataKey="phase" stroke={CHART.accent2} dot={false} name="phase val loss" strokeWidth={1.5} strokeDasharray="4 3" />
      </LineChart>
    </ResponsiveContainer>
  )
}

// A tiny controlled form over a config object.
export function ConfigForm({ config, setConfig, fields, columns = 2 }) {
  const update = (k, v) => setConfig({ ...config, [k]: v })
  return (
    <div className={`field-grid ${columns === 1 ? 'single' : ''}`}>
      {fields.map((f) => (
        <div key={f.key} style={f.full ? { gridColumn: '1 / -1' } : undefined}>
          <label>{f.label || f.key}</label>
          {f.options ? (
            <select value={config[f.key]} onChange={(e) => update(f.key, e.target.value)}>
              {f.options.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          ) : (
            <input
              type={f.type || 'text'}
              value={config[f.key]}
              onChange={(e) =>
                update(f.key, f.type === 'number' ? Number(e.target.value) : e.target.value)
              }
            />
          )}
        </div>
      ))}
    </div>
  )
}

// Collapsible sidebar group: a click-to-expand dropdown for a set of controls,
// so each cluster of parameters stays folded away until needed.
export function Collapsible({ title, children, defaultOpen = false, summary }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`collapsible ${open ? 'open' : ''}`}>
      <button type="button" className="collapsible-head" onClick={() => setOpen((o) => !o)}>
        <span className="chev">▸</span>
        <span className="collapsible-title">{title}</span>
        {summary && !open && <span className="collapsible-summary">{summary}</span>}
      </button>
      {open && <div className="collapsible-body">{children}</div>}
    </div>
  )
}

export function StatusBadge({ status }) {
  return <span className={`status ${status}`}>{status}</span>
}

// Polls the backend's health endpoint so the UI can tell "backend down" apart
// from "still loading" — the two look identical to a panel otherwise.
export function useBackendHealth(intervalMs = 5000) {
  const [ok, setOk] = useState(null)  // null = not checked yet, true/false after
  useEffect(() => {
    let alive = true
    const tick = () => api.health()
      .then(() => { if (alive) setOk(true) })
      .catch(() => { if (alive) setOk(false) })
    tick()
    const h = setInterval(tick, intervalMs)
    return () => { alive = false; clearInterval(h) }
  }, [intervalMs])
  return ok
}

export function BackendStatus() {
  const ok = useBackendHealth()
  if (ok === null) return null
  return (
    <span className={`status ${ok ? 'ok' : 'down'}`} title={ok ? undefined :
      'No response from the FLARE API. Is `flare-api` (or `uvicorn src.api.main:app`) running, and is nothing else already bound to its port?'}>
      {ok ? 'backend ok' : 'backend unreachable'}
    </span>
  )
}

// Track a running job: subscribes to its SSE stream and collects events.
export function useJob() {
  const [job, setJob] = useState(null)
  const [events, setEvents] = useState([])
  const unsub = useRef(null)

  const start = (jobRef) => {
    setJob(jobRef)
    setEvents([])
    if (unsub.current) unsub.current()
    unsub.current = streamJob(
      jobRef.id,
      (ev) => setEvents((prev) => [...prev, ev]),
      (final) => setJob(final),
    )
  }

  useEffect(() => () => { if (unsub.current) unsub.current() }, [])

  return { job, events, start }
}

// Finds the most recent job for a project and live-streams it — lets a panel
// reattach to a job that was started elsewhere (another tab, curl, the agent
// auto-runner) rather than only tracking jobs launched from this component.
export function useProjectJob(projectName) {
  const [job, setJob] = useState(null)
  const [events, setEvents] = useState([])
  const unsub = useRef(null)

  useEffect(() => {
    if (unsub.current) { unsub.current(); unsub.current = null }
    setJob(null)
    setEvents([])
    if (!projectName) return undefined

    let alive = true
    api.listJobs().then((jobs) => {
      if (!alive) return
      const mine = jobs.filter((j) => j.params && j.params.project_name === projectName)
        .sort((a, b) => b.created_at - a.created_at)
      const latest = mine[0]
      if (!latest) return
      setJob(latest)
      unsub.current = streamJob(
        latest.id,
        (ev) => { if (alive) setEvents((prev) => [...prev, ev]) },
        (final) => { if (alive) setJob(final) },
      )
    })

    return () => { alive = false; if (unsub.current) unsub.current() }
  }, [projectName])

  return { job, events }
}
