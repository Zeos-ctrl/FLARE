import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { PanelLayout, Placeholder, GraphCard, StatusBadge, LossCurveChart, useProjectJob } from '../shared.jsx'

const fmt = (v, d = 4) => (typeof v === 'number' ? v.toFixed(d) : '—')

// Status for whichever project row is selected. A live job (if any) streams
// progress via SSE, but the job manager is in-memory and forgets jobs across
// an API restart -- so we also read the *durable* experiment record and saved
// evaluation results, and fall back to those for finished projects.
function ProjectStatus({ name }) {
  const { job, events } = useProjectJob(name)
  const [exp, setExp] = useState(undefined)          // matching experiment record
  const [evalRes, setEvalRes] = useState(undefined)  // saved benchmark results
  const hasLossData = events.some((e) => e.val_loss !== undefined)

  useEffect(() => {
    let alive = true
    setExp(undefined); setEvalRes(undefined)
    if (!name) return undefined
    api.listExperiments()
      .then((rows) => { if (alive) setExp(rows.find((r) => r.project_name === name) || null) })
      .catch(() => { if (alive) setExp(null) })
    api.evalResults(name)
      .then((r) => { if (alive) setEvalRes(r) })
      .catch(() => { if (alive) setEvalRes(null) })
    return () => { alive = false }
  }, [name])

  // Still loading the durable lookups and no live job yet.
  if (!job && (exp === undefined || evalRes === undefined)) {
    return <Placeholder loading label="Loading status" />
  }
  if (!job && !exp && !evalRes) {
    return <Placeholder label={`No run history recorded for "${name}"`} />
  }

  const status = job?.status || exp?.status || (evalRes ? 'completed' : 'unknown')
  const meanMatch = (typeof evalRes?.mean_match === 'number')
    ? evalRes.mean_match : exp?.metrics?.mean_match
  const ampLoss = exp?.metrics?.best_val_loss_amp
  const phaseLoss = exp?.metrics?.best_val_loss_phase
  const running = job && (job.status === 'running' || job.status === 'pending')
  const label = job ? `Job ${job.id}` : exp ? `Experiment ${exp.id}` : name

  return (
    <GraphCard title={exp?.name || name}>
      <p className="muted" style={{ marginBottom: 10 }}>
        {label} <StatusBadge status={status} />
      </p>
      <div className="stat-row" style={{ marginBottom: 12 }}>
        <div className="stat"><div className="k">Mean match</div><div className="v">{fmt(meanMatch)}</div></div>
        <div className="stat"><div className="k">Amp val loss</div><div className="v">{fmt(ampLoss, 5)}</div></div>
        <div className="stat"><div className="k">Phase val loss</div><div className="v">{fmt(phaseLoss, 4)}</div></div>
      </div>
      {hasLossData && <LossCurveChart events={events} />}
      {(job?.error || exp?.error) && <p className="muted">Error: {job?.error || exp?.error}</p>}
      {running && (
        <button className="ghost" style={{ marginTop: 8 }} onClick={() => api.cancelJob(job.id)}>Cancel</button>
      )}
      {events.length > 0 && (
        <div className="log">
          {events.slice(-12).map((e, i) => (
            <div key={i}>{e.message || `${e.model_type || ''} epoch ${e.epoch} · loss ${e.val_loss?.toExponential(3)}`}</div>
          ))}
        </div>
      )}
    </GraphCard>
  )
}

export default function ProjectsPanel() {
  const [projects, setProjects] = useState(null)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)

  const load = () => api.listProjects().then(setProjects).catch((e) => setError(e.message))
  useEffect(() => { load() }, [])

  const remove = async (name) => {
    if (!confirm(`Delete project "${name}"?`)) return
    await api.deleteProject(name)
    if (selected === name) setSelected(null)
    load()
  }

  const sidebar = (
    <>
      <div className="sidebar-title">Controls</div>
      <p className="muted" style={{ fontSize: 12 }}>
        Trained checkpoints discovered on the server.
      </p>
      <button className="action" onClick={load}>Refresh</button>
      {projects && (
        <div className="stat-row" style={{ marginTop: 16 }}>
          <div className="stat"><div className="k">Projects</div><div className="v">{projects.length}</div></div>
          <div className="stat"><div className="k">Trained</div><div className="v">{projects.filter((p) => p.trained).length}</div></div>
        </div>
      )}
      {error && <p className="muted" style={{ marginTop: 10 }}>Error: {error}</p>}
    </>
  )

  return (
    <PanelLayout title="Projects"
      subtitle="Manage saved model checkpoints."
      sidebar={sidebar}>
      {!projects || projects.length === 0 ? (
        <Placeholder label={projects ? 'No projects yet' : 'No data available'} loading={!projects && !error} />
      ) : (
        <GraphCard title="Checkpoints">
          <ul className="rowlist">
            {projects.map((p, i) => (
              <li key={p.name}
                className={`rowitem ${selected === p.name ? 'selected' : ''}`}
                onClick={() => setSelected(selected === p.name ? null : p.name)}>
                <span className="rowitem__rank">{String(i + 1).padStart(2, '0')}</span>
                <div style={{ minWidth: 0 }}>
                  <div className="rowitem__name">{p.name}</div>
                  <div className="rowitem__meta">
                    <span>{p.config?.waveform || 'unknown'}</span>
                    <span>{p.config?.num_samples ?? '—'} samples</span>
                  </div>
                </div>
                <div className="rowitem__metric">
                  <span className="v" style={{ fontSize: '0.95rem' }}>
                    {p.trained ? '✓' : '—'} / {p.evaluated ? '✓' : '—'}
                  </span>
                  <span className="k">train / eval</span>
                </div>
                <div onClick={(e) => e.stopPropagation()}>
                  <button className="mini" onClick={() => remove(p.name)}>delete</button>
                </div>
              </li>
            ))}
          </ul>
        </GraphCard>
      )}
      {selected && <ProjectStatus name={selected} />}
    </PanelLayout>
  )
}
