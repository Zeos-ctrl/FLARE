import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { PanelLayout, GraphCard, Placeholder, Collapsible } from '../shared.jsx'

// Numeric/select field helper.
function Field({ label, value, onChange, type = 'number', options }) {
  return (
    <div>
      <label>{label}</label>
      {options ? (
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => <option key={o}>{o}</option>)}
        </select>
      ) : (
        <input type={type} value={value}
          onChange={(e) => onChange(type === 'number' ? Number(e.target.value) : e.target.value)} />
      )}
    </div>
  )
}

export default function SettingsPanel() {
  const [s, setS] = useState(null)
  const [saved, setSaved] = useState(false)
  const set = (k, v) => { setS({ ...s, [k]: v }); setSaved(false) }

  useEffect(() => { api.getSettings().then(setS) }, [])

  const save = async () => { await api.putSettings(s); setSaved(true) }

  const sidebar = s ? (
    <>
      <Collapsible title="Waveform / system" defaultOpen summary={s.waveform}>
        <Field label="Waveform" value={s.waveform} onChange={(v) => set('waveform', v)}
          type="text" options={['IMRPhenomD', 'SEOBNRv4', 'SEOBNRv4HM']} />
        <div className="field-grid">
          <Field label="Waveform length" value={s.waveform_length} onChange={(v) => set('waveform_length', v)} />
          <Field label="Sample rate (Hz)" value={s.sample_rate} onChange={(v) => set('sample_rate', v)} />
          <Field label="f_lower (Hz)" value={s.f_lower} onChange={(v) => set('f_lower', v)} />
          <Field label="Samples" value={s.num_samples} onChange={(v) => set('num_samples', v)} />
          <Field label="Val split" value={s.val_split} onChange={(v) => set('val_split', v)} />
          <Field label="Device" value={s.device} onChange={(v) => set('device', v)} type="text" options={['cuda', 'cpu']} />
        </div>
      </Collapsible>

      <Collapsible title="Mass range (M☉)" summary={`${s.mass_min} – ${s.mass_max}`}>
        <div className="field-grid">
          <Field label="min" value={s.mass_min} onChange={(v) => set('mass_min', v)} />
          <Field label="max" value={s.mass_max} onChange={(v) => set('mass_max', v)} />
        </div>
      </Collapsible>
      <Collapsible title="Spin range" summary={`${s.spin_min} – ${s.spin_max}`}>
        <div className="field-grid">
          <Field label="min" value={s.spin_min} onChange={(v) => set('spin_min', v)} />
          <Field label="max" value={s.spin_max} onChange={(v) => set('spin_max', v)} />
        </div>
      </Collapsible>
      <Collapsible title="Inclination (rad)" summary={`${s.incl_min} – ${s.incl_max}`}>
        <div className="field-grid">
          <Field label="min" value={s.incl_min} onChange={(v) => set('incl_min', v)} />
          <Field label="max" value={s.incl_max} onChange={(v) => set('incl_max', v)} />
        </div>
      </Collapsible>
      <Collapsible title="Eccentricity" summary={`${s.ecc_min} – ${s.ecc_max}`}>
        <div className="field-grid">
          <Field label="min" value={s.ecc_min} onChange={(v) => set('ecc_min', v)} />
          <Field label="max" value={s.ecc_max} onChange={(v) => set('ecc_max', v)} />
        </div>
      </Collapsible>

      <button className="action" onClick={save}>{saved ? 'Saved ✓' : 'Save Settings'}</button>
    </>
  ) : <Placeholder loading label="Loading settings" />

  return (
    <PanelLayout title="Settings"
      subtitle="System and data-generation parameters. Saved on the server and used by Dataset preview and Training."
      sidebar={sidebar}>
      {!s ? <Placeholder loading /> : (
        <>
          <GraphCard title="Parameter space">
            <div className="def-list">
              <div className="def-row"><span className="k">Masses</span><span className="v">{s.mass_min} – {s.mass_max} M☉</span></div>
              <div className="def-row"><span className="k">Spins (aligned)</span><span className="v">{s.spin_min} – {s.spin_max}</span></div>
              <div className="def-row"><span className="k">Inclination</span><span className="v">{s.incl_min} – {s.incl_max} rad</span></div>
              <div className="def-row"><span className="k">Eccentricity</span><span className="v">{s.ecc_min} – {s.ecc_max}</span></div>
              <div className="def-row"><span className="k">Waveform</span><span className="v">{s.waveform}</span></div>
              <div className="def-row"><span className="k">Duration</span><span className="v">{(s.waveform_length / s.sample_rate).toFixed(3)} s ({s.waveform_length} @ {s.sample_rate} Hz)</span></div>
              <div className="def-row"><span className="k">Low-freq cutoff</span><span className="v">{s.f_lower} Hz</span></div>
              <div className="def-row"><span className="k">Training samples</span><span className="v">{s.num_samples}</span></div>
            </div>
          </GraphCard>
          <GraphCard title="Where is this stored?">
            <p className="muted" style={{ lineHeight: 1.7 }}>
              These settings persist server-side at <span className="mono">flare_state/settings.json</span>.
              Model designs live in <span className="mono">flare_state/models/</span>. Trained models,
              datasets and results are saved per project under <span className="mono">checkpoints/&lt;project&gt;/</span>.
              The browser stores nothing.
            </p>
          </GraphCard>
        </>
      )}
    </PanelLayout>
  )
}
