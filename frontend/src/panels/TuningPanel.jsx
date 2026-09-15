import React, { useEffect, useState } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { api } from '../api.js'
import { StatusBadge, PanelLayout, Placeholder, GraphCard, CHART, useJob, Collapsible } from '../shared.jsx'

export default function TuningPanel() {
  const [projectName, setProjectName] = useState('my_model')
  const [models, setModels] = useState([])
  const [modelName, setModelName] = useState('default')
  const [modelType, setModelType] = useState('both')
  const [nTrials, setNTrials] = useState(20)
  const { job, events, start } = useJob()

  useEffect(() => {
    api.listModels().then((list) => {
      setModels(list)
      if (list[0]) setModelName(list[0].name)
    })
  }, [])

  const launch = async () => start(await api.startTuning(projectName, modelName, modelType, nTrials))

  const trials = events
    .filter((e) => e.trial !== undefined && e.value != null)
    .map((e) => ({ trial: e.trial, value: e.value }))
  const running = job && job.status === 'running'
  const best = events.filter((e) => e.best_value != null).slice(-1)[0]

  const sidebar = (
    <>
      <Collapsible title="Parameters" defaultOpen summary={`${modelName} · ${modelType}`}>
        <label>Project name</label>
        <input value={projectName} onChange={(e) => setProjectName(e.target.value)} />
        <label>Model design</label>
        <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
          {models.map((m) => <option key={m.name}>{m.name}</option>)}
        </select>
        <div className="field-grid">
          <div>
            <label>Model to tune</label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="both">both</option><option value="amp">amp</option><option value="phase">phase</option>
            </select>
          </div>
          <div>
            <label>Trials</label>
            <input type="number" value={nTrials} onChange={(e) => setNTrials(Number(e.target.value))} />
          </div>
        </div>
      </Collapsible>
      <button className="action" onClick={launch} disabled={running}>
        {running ? 'Tuning…' : 'Start Tuning'}
      </button>
      {running && <button className="ghost" style={{ marginTop: 8 }} onClick={() => api.cancelJob(job.id)}>Cancel</button>}
      {job && <p className="muted" style={{ marginTop: 12 }}>Job {job.id} <StatusBadge status={job.status} /></p>}
      {best && <p className="muted">best value {best.best_value.toExponential(3)}</p>}
    </>
  )

  return (
    <PanelLayout title="Hyperparameter Tuning"
      subtitle="Launch Optuna HPO over a model design and watch trial values converge."
      sidebar={sidebar}>
      {trials.length === 0 ? (
        <Placeholder loading={running} />
      ) : (
        <GraphCard title="Trial values">
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart>
              <CartesianGrid stroke={CHART.grid} />
              <XAxis dataKey="trial" name="trial" tick={CHART.tick} stroke={CHART.axis} />
              <YAxis dataKey="value" name="value" scale="log" domain={['auto', 'auto']} tick={CHART.tick} stroke={CHART.axis} width={64} />
              <Tooltip {...CHART.tooltip} />
              <Scatter data={trials} fill={CHART.accent} fillOpacity={0.7} />
            </ScatterChart>
          </ResponsiveContainer>
        </GraphCard>
      )}
      {job && job.error && <p className="muted">Error: {job.error}</p>}
    </PanelLayout>
  )
}
