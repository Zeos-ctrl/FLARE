import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { StatusBadge, PanelLayout, Placeholder, GraphCard, Collapsible } from '../shared.jsx'

export default function EvaluationPanel() {
  const [projects, setProjects] = useState([])
  const [project, setProject] = useState('')
  const [nSamples, setNSamples] = useState(500)
  const [device, setDevice] = useState('cuda')
  const [results, setResults] = useState(null)
  const [plots, setPlots] = useState([])
  const { job, start } = useJobLocal()

  useEffect(() => {
    api.listProjects().then((ps) => {
      const trained = ps.filter((p) => p.trained)
      setProjects(trained)
      if (trained[0]) setProject(trained[0].name)
    })
  }, [])

  const launch = async () => {
    setResults(null); setPlots([])
    start(await api.startEvaluation({ project_name: project, n_samples: nSamples, device }))
  }

  useEffect(() => {
    if (job && job.status === 'completed') {
      api.evalResults(project).then(setResults).catch(() => {})
      api.evalPlots(project).then(setPlots).catch(() => {})
    }
  }, [job?.status])

  const running = job && job.status === 'running'

  const sidebar = (
    <>
      <Collapsible title="Parameters" defaultOpen summary={`${project || '—'} · ${nSamples}`}>
        <label>Project</label>
        <select value={project} onChange={(e) => setProject(e.target.value)}>
          {projects.length === 0 && <option value="">— no trained models —</option>}
          {projects.map((p) => <option key={p.name}>{p.name}</option>)}
        </select>
        <label>Test samples</label>
        <input type="number" value={nSamples} onChange={(e) => setNSamples(Number(e.target.value))} />
        <label>Device</label>
        <select value={device} onChange={(e) => setDevice(e.target.value)}>
          <option>cuda</option><option>cpu</option>
        </select>
      </Collapsible>
      <button className="action" onClick={launch} disabled={running || !project}>
        {running ? 'Benchmarking…' : 'Run Evaluation'}
      </button>
      {job && <p className="muted" style={{ marginTop: 12 }}>Job {job.id} <StatusBadge status={job.status} /></p>}
      {results && (
        <div className="stat-row" style={{ marginTop: 14, flexDirection: 'column', gap: 12 }}>
          <div className="stat"><div className="k">Mean match</div><div className="v">{results.mean_match?.toFixed(4)}</div></div>
          <div className="stat"><div className="k">Median</div><div className="v">{results.percentiles?.['50']?.toFixed(4)}</div></div>
        </div>
      )}
    </>
  )

  return (
    <PanelLayout title="Evaluation"
      subtitle="Benchmark a trained surrogate against ground-truth waveforms."
      sidebar={sidebar}>
      {plots.length === 0 ? (
        <Placeholder loading={running} />
      ) : (
        plots.map((name) => (
          <GraphCard key={name} title={name.replace('.png', '').replace(/_/g, ' ')}>
            <img src={`/api/evaluation/${project}/plots/${name}`}
              style={{ width: '100%', borderRadius: 6, display: 'block' }} />
          </GraphCard>
        ))
      )}
    </PanelLayout>
  )
}

// Local job hook variant (evaluation doesn't stream fine-grained events).
function useJobLocal() {
  const [job, setJob] = useState(null)
  const start = (ref) => {
    setJob(ref)
    const poll = setInterval(async () => {
      const j = await api.getJob(ref.id)
      setJob(j)
      if (['completed', 'failed', 'cancelled'].includes(j.status)) clearInterval(poll)
    }, 1000)
  }
  return { job, start }
}
