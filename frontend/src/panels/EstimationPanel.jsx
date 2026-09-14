import React, { useEffect, useState } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { api } from '../api.js'
import { StatusBadge, PanelLayout, Placeholder, GraphCard, CHART, useJob, Collapsible } from '../shared.jsx'

export default function EstimationPanel() {
  const [projects, setProjects] = useState([])
  const [req, setReq] = useState({
    project_name: '', event: 'GW150914', detectors: ['H1', 'L1'],
    nwalkers: 32, nsteps: 1000, bank_size: 1000, device: 'cuda',
  })
  const [samples, setSamples] = useState(null)
  const { job, events, start } = useJob()

  useEffect(() => {
    api.listProjects().then((ps) => {
      const trained = ps.filter((p) => p.trained)
      setProjects(trained)
      if (trained[0]) setReq((r) => ({ ...r, project_name: trained[0].name }))
    })
  }, [])

  const launch = async () => { setSamples(null); start(await api.startEstimation(req)) }
  const set = (k, v) => setReq({ ...req, [k]: v })

  useEffect(() => {
    if (job && job.status === 'completed') {
      api.getSamples(req.project_name, req.event).then(setSamples).catch(() => {})
    }
  }, [job?.status])

  const running = job && job.status === 'running'
  const lastStep = events.filter((e) => e.step !== undefined).slice(-1)[0]
  const scatter = samples ? samples.samples.map((s) => ({ x: s[0], y: s[1] })) : []
  const names = samples?.param_names || []

  const sidebar = (
    <>
      <Collapsible title="Event" defaultOpen summary={`${req.event}`}>
        <label>Project</label>
        <select value={req.project_name} onChange={(e) => set('project_name', e.target.value)}>
          {projects.length === 0 && <option value="">— no trained models —</option>}
          {projects.map((p) => <option key={p.name}>{p.name}</option>)}
        </select>
        <label>Event</label>
        <input value={req.event} onChange={(e) => set('event', e.target.value)} />
        <label>Detectors (comma-sep)</label>
        <input value={req.detectors.join(',')}
          onChange={(e) => set('detectors', e.target.value.split(',').map((s) => s.trim()))} />
      </Collapsible>
      <Collapsible title="MCMC / sampler" summary={`${req.nwalkers}w · ${req.nsteps}s`}>
        <div className="field-grid">
          <div><label>Walkers</label><input type="number" value={req.nwalkers} onChange={(e) => set('nwalkers', Number(e.target.value))} /></div>
          <div><label>Steps</label><input type="number" value={req.nsteps} onChange={(e) => set('nsteps', Number(e.target.value))} /></div>
          <div><label>Bank size</label><input type="number" value={req.bank_size} onChange={(e) => set('bank_size', Number(e.target.value))} /></div>
          <div>
            <label>Device</label>
            <select value={req.device} onChange={(e) => set('device', e.target.value)}>
              <option>cuda</option><option>cpu</option>
            </select>
          </div>
        </div>
      </Collapsible>
      <button className="action" onClick={launch} disabled={running || !req.project_name}>
        {running ? 'Running…' : 'Estimate'}
      </button>
      {running && <button className="ghost" style={{ marginTop: 8 }} onClick={() => api.cancelJob(job.id)}>Cancel</button>}
      {job && (
        <p className="muted" style={{ marginTop: 12 }}>
          Job {job.id} <StatusBadge status={job.status} />
        </p>
      )}
      {lastStep && (
        <p className="muted">step {lastStep.step}/{lastStep.total_steps} · accept {lastStep.mean_acceptance?.toFixed(3)}</p>
      )}
    </>
  )

  return (
    <PanelLayout title="GWOSC Parameter Estimation"
      subtitle="Run MCMC over a real detection using the surrogate as the template engine."
      sidebar={sidebar}>
      {!samples && !job?.result?.summary ? (
        <Placeholder loading={running} />
      ) : (
        <>
          {job?.result?.summary && (
            <GraphCard title="Posterior summary">
              <table>
                <thead><tr><th>Param</th><th>Median</th><th>90% CI</th></tr></thead>
                <tbody>
                  {Object.entries(job.result.summary).map(([k, v]) => (
                    <tr key={k}><td className="mono">{k}</td><td className="mono">{v.median.toFixed(3)}</td>
                      <td className="mono">[{v.lower_90.toFixed(3)}, {v.upper_90.toFixed(3)}]</td></tr>
                  ))}
                </tbody>
              </table>
            </GraphCard>
          )}
          {samples && (
            <GraphCard title={`Posterior · ${names[0]} vs ${names[1]}`}>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart>
                  <CartesianGrid stroke={CHART.grid} />
                  <XAxis dataKey="x" name={names[0]} tick={CHART.tick} stroke={CHART.axis} />
                  <YAxis dataKey="y" name={names[1]} tick={CHART.tick} stroke={CHART.axis} width={54} />
                  <Tooltip {...CHART.tooltip} />
                  <Scatter data={scatter} fill={CHART.accent} fillOpacity={0.35} />
                </ScatterChart>
              </ResponsiveContainer>
            </GraphCard>
          )}
        </>
      )}
      {job && job.error && <p className="muted">Error: {job.error}</p>}
    </PanelLayout>
  )
}
