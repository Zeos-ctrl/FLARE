import React, { useEffect, useState } from 'react'
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
const NAMES = Object.keys(TABS)

function ThemeToggle() {
  const [theme, setTheme] = useState('dark')
  useEffect(() => {
    const stored = localStorage.getItem('flare-theme')
    if (stored) {
      document.documentElement.classList.toggle('dark', stored === 'dark')
      setTheme(stored)
    }
  }, [])
  const toggle = () => {
    setTheme((t) => {
      const next = t === 'dark' ? 'light' : 'dark'
      document.documentElement.classList.toggle('dark', next === 'dark')
      localStorage.setItem('flare-theme', next)
      return next
    })
  }
  return (
    <button
      className="theme-toggle"
      onClick={toggle}
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      <span className="dot" />
    </button>
  )
}

// Nav items shared by the desktop rail and the mobile menu.
function NavItems({ tab, onPick }) {
  return NAMES.map((name, i) => (
    <button
      key={name}
      className={`nav-item ${tab === name ? 'active' : ''}`}
      onClick={() => onPick(name)}
    >
      <span className="idx">{String(i + 1).padStart(2, '0')}</span>
      <span className="label">{name}</span>
    </button>
  ))
}

export default function App() {
  const [tab, setTab] = useState('Projects')
  const [mobileOpen, setMobileOpen] = useState(false)
  const Panel = TABS[tab]

  const pick = (name) => {
    setTab(name)
    setMobileOpen(false)
  }

  return (
    <div className="app">
      {/* Desktop icon rail — collapsed to icons, expands on hover. */}
      <aside className="rail">
        <div className="rail__brand">
          <span className="rail__mark" />
          <span className="label">FLARE</span>
        </div>
        <nav className="rail__nav">
          <NavItems tab={tab} onPick={pick} />
        </nav>
        <div className="rail__foot">
          <ThemeToggle />
          <span className="label">
            <BackendStatus />
          </span>
        </div>
      </aside>

      {/* Mobile top bar + slide-in menu. */}
      <header className="topbar">
        <span className="rail__mark" />
        <span className="topbar__brand">FLARE</span>
        <button
          className="topbar__menu"
          aria-label="Menu"
          onClick={() => setMobileOpen((o) => !o)}
        >
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
            <line x1="4" y1="8" x2="20" y2="8" stroke="currentColor" strokeWidth="1.6" />
            <line x1="4" y1="16" x2="20" y2="16" stroke="currentColor" strokeWidth="1.6" />
          </svg>
        </button>
      </header>
      {mobileOpen && (
        <div className="mobile-menu">
          <button
            className="mobile-menu__close"
            aria-label="Close menu"
            onClick={() => setMobileOpen(false)}
          >
            <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
              <line x1="5" y1="5" x2="19" y2="19" stroke="currentColor" strokeWidth="1.6" />
              <line x1="19" y1="5" x2="5" y2="19" stroke="currentColor" strokeWidth="1.6" />
            </svg>
          </button>
          <nav className="mobile-menu__nav">
            <NavItems tab={tab} onPick={pick} />
          </nav>
          <div className="mobile-menu__foot">
            <ThemeToggle />
            <BackendStatus />
          </div>
        </div>
      )}

      <main className="main" key={tab}>
        <Panel />
      </main>
    </div>
  )
}
