import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { StatusBadge, PanelLayout, Placeholder, GraphCard, LossCurveChart, useJob, Collapsible } from '../shared.jsx'

export default function TrainingPanel() {
  const [projectName, setProjectName] = useState('my_model')
  const [models, setModels] = useState([])
  const [modelName, setModelName] = useState('default')
  const { job, events, start } = useJob()

  useEffect(() => {
    api.listModels().then((list) => {
      setModels(list)
      if (list[0]) setModelName(list[0].name)
    })
  }, [])

  const launch = async () => start(await api.startTraining(projectName, modelName))

  const running = job && job.status === 'running'
  const hasLossData = events.some((e) => e.val_loss !== undefined)

  const sidebar = (
    <>
      <Collapsible title="Parameters" defaultOpen summary={`${projectName} · ${modelName}`}>
        <label>Project name</label>
        <input value={projectName} onChange={(e) => setProjectName(e.target.value)} />
        <label>Model design</label>
        <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
          {models.map((m) => <option key={m.name}>{m.name}</option>)}
        </select>
        <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
          Data-generation parameters (masses, spins, waveform length…) come from the
          <strong> Settings</strong> tab. Architecture comes from the <strong>Model</strong> tab.
        </p>
      </Collapsible>
      <button className="action" onClick={launch} disabled={running}>
        {running ? 'Training…' : 'Start Training'}
      </button>
      {running && <button className="ghost" style={{ marginTop: 8 }} onClick={() => api.cancelJob(job.id)}>Cancel</button>}
      {job && <p className="muted" style={{ marginTop: 12 }}>Job {job.id} <StatusBadge status={job.status} /></p>}
    </>
  )

  return (
    <PanelLayout title="Training"
      subtitle="Train a saved model design on data generated from the current settings; watch validation loss live."
      sidebar={sidebar}>
      {!hasLossData ? (
        <Placeholder loading={running} />
      ) : (
        <GraphCard title="Validation loss">
          <LossCurveChart events={events} />
        </GraphCard>
      )}
      {job && job.error && <p className="muted">Error: {job.error}</p>}
      {events.length > 0 && (
        <div className="log">
          {events.slice(-12).map((e, i) => (
            <div key={i}>{e.message || `${e.model_type || ''} epoch ${e.epoch} · loss ${e.val_loss?.toExponential(3)}`}</div>
          ))}
        </div>
      )}
    </PanelLayout>
  )
}
