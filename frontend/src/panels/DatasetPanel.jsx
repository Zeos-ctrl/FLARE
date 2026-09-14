import React, { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ScatterChart, Scatter, Tooltip } from 'recharts'
import { api } from '../api.js'
import { PanelLayout, Placeholder, GraphCard, CHART, Collapsible } from '../shared.jsx'

const num = (v) => Number(v)

// Mirrors the keys in src/data/features.py's FeatureExtractor.compute_features —
// keep in sync if a new derived feature is added there.
const AVAILABLE_FEATURES = [
  'chirp_mass', 'symmetric_mass_ratio', 'mass_ratio', 'total_mass',
  'effective_spin', 'inclination', 'eccentricity',
]

export default function DatasetPanel() {
  const [p, setP] = useState(null)          // editable system params (seeded from settings)
  const [data, setData] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [savedMsg, setSavedMsg] = useState('')
  const set = (k, v) => setP({ ...p, [k]: v })

  // Seed from persisted settings so the preview matches what training will use.
  useEffect(() => {
    api.getSettings().then((s) => setP({
      waveform: s.waveform, n_samples: 6, waveform_length: s.waveform_length,
      f_lower: s.f_lower, sample_rate: s.sample_rate,
      mass_min: s.mass_min, mass_max: s.mass_max,
      spin_min: s.spin_min, spin_max: s.spin_max,
      incl_min: s.incl_min, incl_max: s.incl_max,
      ecc_min: s.ecc_min, ecc_max: s.ecc_max,
      feature_names: s.feature_names,
      reduced_order: s.reduced_order,
    }))
  }, [])

  const toggleFeature = (name) => {
    const active = p.feature_names.includes(name)
    if (active && p.feature_names.length === 1) return  // keep at least one
    set('feature_names', active
      ? p.feature_names.filter((f) => f !== name)
      : [...p.feature_names, name])
  }

  const preview = async () => {
    setBusy(true); setError(null)
    try {
      setData(await api.previewDataset({
        waveform: p.waveform, n_samples: num(p.n_samples), waveform_length: num(p.waveform_length),
        f_lower: num(p.f_lower), sample_rate: num(p.sample_rate),
        mass_range: [num(p.mass_min), num(p.mass_max)],
        spin_range: [num(p.spin_min), num(p.spin_max)],
        incl_range: [num(p.incl_min), num(p.incl_max)],
        ecc_range: [num(p.ecc_min), num(p.ecc_max)],
      }))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const saveToSettings = async () => {
    const s = await api.getSettings()
    await api.putSettings({
      ...s, waveform: p.waveform, waveform_length: num(p.waveform_length),
      f_lower: num(p.f_lower), sample_rate: num(p.sample_rate),
      mass_min: num(p.mass_min), mass_max: num(p.mass_max),
      spin_min: num(p.spin_min), spin_max: num(p.spin_max),
      incl_min: num(p.incl_min), incl_max: num(p.incl_max),
      ecc_min: num(p.ecc_min), ecc_max: num(p.ecc_max),
      feature_names: p.feature_names,
      reduced_order: p.reduced_order,
    })
    setSavedMsg('Saved to Settings ✓'); setTimeout(() => setSavedMsg(''), 2500)
  }

  const chartData = data ? data.waveforms[0].strain.map((_, i) => {
    const row = { i }
    data.waveforms.forEach((w, k) => { row[`w${k}`] = w.strain[i] })
    return row
  }) : []
  const scatterData = data ? data.parameters.map((x) => ({ m1: x.m1, m2: x.m2 })) : []

  const F = ({ label, k, type = 'number', options }) => (
    <div>
      <label>{label}</label>
      {options ? (
        <select value={p[k]} onChange={(e) => set(k, e.target.value)}>{options.map((o) => <option key={o}>{o}</option>)}</select>
      ) : (
        <input type={type} value={p[k]} onChange={(e) => set(k, type === 'number' ? Number(e.target.value) : e.target.value)} />
      )}
    </div>
  )

  const sidebar = p ? (
    <>
      <Collapsible title="System parameters" defaultOpen summary={p.waveform}>
        <F label="Waveform" k="waveform" type="text" options={['IMRPhenomD', 'SEOBNRv4', 'SEOBNRv4HM']} />
        <div className="field-grid">
          <F label="Preview count" k="n_samples" />
          <F label="Length" k="waveform_length" />
          <F label="f_lower (Hz)" k="f_lower" />
          <F label="Sample rate" k="sample_rate" />
        </div>
      </Collapsible>
      <Collapsible title="Masses (M☉)" summary={`${p.mass_min} – ${p.mass_max}`}>
        <div className="field-grid"><F label="min" k="mass_min" /><F label="max" k="mass_max" /></div>
      </Collapsible>
      <Collapsible title="Spins" summary={`${p.spin_min} – ${p.spin_max}`}>
        <div className="field-grid"><F label="min" k="spin_min" /><F label="max" k="spin_max" /></div>
      </Collapsible>
      <Collapsible title="Inclination (rad)" summary={`${p.incl_min} – ${p.incl_max}`}>
        <div className="field-grid"><F label="min" k="incl_min" /><F label="max" k="incl_max" /></div>
      </Collapsible>
      <Collapsible title="Eccentricity" summary={`${p.ecc_min} – ${p.ecc_max}`}>
        <div className="field-grid"><F label="min" k="ecc_min" /><F label="max" k="ecc_max" /></div>
      </Collapsible>
      <Collapsible title="Parameters to train on" summary={`${p.feature_names.length} selected`}>
        <p className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
          Model input features, derived from the sampled masses/spins/etc.
        </p>
        <div className="toggle-chip-row">
          {AVAILABLE_FEATURES.map((name) => (
            <button key={name} type="button"
              className={`toggle-chip ${p.feature_names.includes(name) ? 'active' : ''}`}
              onClick={() => toggleFeature(name)}>
              {name.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </Collapsible>
      <Collapsible title="Training speed" summary={p.reduced_order ? 'reduced-order (SVD)' : 'per-point'}>
        <p className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
          Reduced-order training compresses amplitude/phase into an SVD basis and
          learns a small θ→coefficients network — trains in seconds instead of
          hours (see EXPERIMENT_NOTES).
        </p>
        <button type="button"
          className={`toggle-chip ${p.reduced_order ? 'active' : ''}`}
          onClick={() => set('reduced_order', !p.reduced_order)}>
          Reduced-order (SVD)
        </button>
      </Collapsible>

      <button className="action" onClick={preview} disabled={busy}>{busy ? 'Generating…' : 'Preview'}</button>
      <button className="ghost" style={{ marginTop: 8 }} onClick={saveToSettings}>Save to Settings</button>
      {savedMsg && <p className="muted" style={{ marginTop: 8 }}>{savedMsg}</p>}
      {error && <p className="muted" style={{ marginTop: 8 }}>Error: {error}</p>}
    </>
  ) : <Placeholder loading label="Loading" />

  return (
    <PanelLayout title="Dataset Explorer"
      subtitle="Set the system/data parameters used to train the model and preview the resulting waveforms."
      sidebar={sidebar}>
      {!data ? (
        <Placeholder loading={busy} />
      ) : (
        <>
          <GraphCard title="Waveforms (overlaid)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData}>
                <CartesianGrid stroke={CHART.grid} />
                <XAxis dataKey="i" tick={CHART.tick} stroke={CHART.axis} />
                <YAxis tick={CHART.tick} stroke={CHART.axis} width={54} />
                {data.waveforms.map((_, k) => (
                  <Line key={k} type="monotone" dataKey={`w${k}`} dot={false}
                    stroke={`hsl(${8 + (k * 32) % 340},70%,62%)`} strokeWidth={1.2} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </GraphCard>
          <GraphCard title="Sampled m1 vs m2">
            <ResponsiveContainer width="100%" height={240}>
              <ScatterChart>
                <CartesianGrid stroke={CHART.grid} />
                <XAxis dataKey="m1" name="m1" tick={CHART.tick} stroke={CHART.axis} />
                <YAxis dataKey="m2" name="m2" tick={CHART.tick} stroke={CHART.axis} width={48} />
                <Tooltip {...CHART.tooltip} />
                <Scatter data={scatterData} fill={CHART.accent} />
              </ScatterChart>
            </ResponsiveContainer>
          </GraphCard>
        </>
      )}
    </PanelLayout>
  )
}
