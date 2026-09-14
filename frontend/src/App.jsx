import React, { useState } from 'react'
import { BackendStatus } from './shared.jsx'
import ProjectsPanel from './panels/ProjectsPanel.jsx'
import SettingsPanel from './panels/SettingsPanel.jsx'
import ModelPanel from './panels/ModelPanel.jsx'
import DatasetPanel from './panels/DatasetPanel.jsx'
import TrainingPanel from './panels/TrainingPanel.jsx'
import TuningPanel from './panels/TuningPanel.jsx'
import EvaluationPanel from './panels/EvaluationPanel.jsx'
import EstimationPanel from './panels/EstimationPanel.jsx'
import ExperimentsPanel from './panels/ExperimentsPanel.jsx'

const TABS = {
  Projects: ProjectsPanel,
  Experiments: ExperimentsPanel,
  Settings: SettingsPanel,
  Model: ModelPanel,
  Dataset: DatasetPanel,
  Training: TrainingPanel,
  Tuning: TuningPanel,
  Evaluation: EvaluationPanel,
  Estimation: EstimationPanel,
}

export default function App() {
  const [tab, setTab] = useState('Projects')
  const Panel = TABS[tab]
  return (
    <>
      <header>
        <nav>
          {Object.keys(TABS).map((name) => (
            <button
              key={name}
              className={tab === name ? 'active' : ''}
              onClick={() => setTab(name)}
            >
              {name}
            </button>
          ))}
        </nav>
        <div style={{ marginLeft: 'auto' }}><BackendStatus /></div>
      </header>
      <main key={tab}>
        <Panel />
      </main>
    </>
  )
}
