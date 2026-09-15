import React, { useEffect, useState } from 'react'
import { api } from '../api.js'
import { PanelLayout, GraphCard, Placeholder, Collapsible } from '../shared.jsx'

function Field({ label, value, onChange, type = 'number', options }) {
  return (
    <div>
      <label>{label}</label>
      {options ? (
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => <option key={String(o)} value={o}>{String(o)}</option>)}
        </select>
      ) : (
        <input type={type} value={value}
          onChange={(e) => onChange(type === 'number' ? Number(e.target.value) : e.target.value)} />
      )}
    </div>
  )
}

export default function ModelPanel() {
  const [designs, setDesigns] = useState([])
  const [d, setD] = useState(null)
  const [saved, setSaved] = useState(false)
  const [custom, setCustom] = useState({ dir: 'custom_models', models: [] })
  const set = (k, v) => { setD({ ...d, [k]: v }); setSaved(false) }

  const load = () => api.listModels().then((list) => {
    setDesigns(list)
    setD((cur) => cur || list[0])
  })
  useEffect(() => {
    load()
    api.listCustomModels().then(setCustom).catch(() => {})
  }, [])

  const save = async () => { await api.putModel(d.name, d); setSaved(true); load() }
  const remove = async () => {
    if (!confirm(`Delete design "${d.name}"?`)) return
    await api.deleteModel(d.name)
    const list = await api.listModels()
    setDesigns(list); setD(list[0])
  }
  const selectDesign = (name) => {
    const found = designs.find((x) => x.name === name)
    if (found) { setD(found); setSaved(false) }
  }

  const isCustom = d && d.model_kind === 'custom'
  const modules = custom.models || []
  // A module field: pick a file from custom_models/, or note that none exist.
  const ModuleField = ({ label, k }) => (
    <>
      <label>{label}</label>
      {modules.length === 0 ? (
        <input value={d[k] || ''} onChange={(e) => set(k, e.target.value)}
          placeholder="no files found" />
      ) : (
        <select value={d[k] || ''} onChange={(e) => set(k, e.target.value)}>
          <option value="">— choose file —</option>
          {modules.map((m) => <option key={m} value={m}>{m}.py</option>)}
        </select>
      )}
    </>
  )

  const sidebar = d ? (
    <>
      <Collapsible title="Design" defaultOpen summary={d.name}>
        <label>Load design</label>
        <select value={d.name} onChange={(e) => selectDesign(e.target.value)}>
          {designs.map((x) => <option key={x.name}>{x.name}</option>)}
        </select>
        <label>Name</label>
        <input value={d.name} onChange={(e) => set('name', e.target.value)} />
        <label>Model kind</label>
        <select value={d.model_kind || 'builtin'} onChange={(e) => set('model_kind', e.target.value)}>
          <option value="builtin">builtin (parametric MLP)</option>
          <option value="custom">custom (PyTorch file)</option>
        </select>
      </Collapsible>

      {isCustom ? (
        <>
          <Collapsible title="Custom model files" defaultOpen
            summary={d.amp_module && d.phase_module ? `${d.amp_module} · ${d.phase_module}` : 'select'}>
            <p className="muted" style={{ fontSize: 11, margin: '4px 0 8px', lineHeight: 1.5 }}>
              Drop a <span className="mono">.py</span> in <span className="mono">{custom.dir}/</span> that defines
              <span className="mono"> build_model(in_param_dim, time_dim)</span> → <span className="mono">nn.Module</span>
              with <span className="mono">forward(t_norm, theta)</span>. Code runs only when the CLI/trainer builds it.
            </p>
            <ModuleField label="Amplitude model file" k="amp_module" />
            <ModuleField label="Phase model file" k="phase_module" />
            {modules.length === 0 && (
              <p className="muted" style={{ fontSize: 11, marginTop: 8 }}>
                No files in <span className="mono">{custom.dir}/</span> yet. Type a name once the file exists.
              </p>
            )}
          </Collapsible>
        </>
      ) : (
        <>
          <Collapsible title="Amplitude net"
            summary={`${d.amp_layers}×${d.amp_hidden_size}`}>
            <div className="field-grid">
              <Field label="Hidden size" value={d.amp_hidden_size} onChange={(v) => set('amp_hidden_size', v)} />
              <Field label="Layers" value={d.amp_layers} onChange={(v) => set('amp_layers', v)} />
              <Field label="Banks" value={d.amp_banks} onChange={(v) => set('amp_banks', v)} />
              <Field label="Dropout" value={d.amp_dropout} onChange={(v) => set('amp_dropout', v)} />
              <Field label="LR" value={d.amp_lr} onChange={(v) => set('amp_lr', v)} />
            </div>
          </Collapsible>

          <Collapsible title="Phase net"
            summary={`${d.phase_layers}×${d.phase_hidden_size}`}>
            <div className="field-grid">
              <Field label="Hidden size" value={d.phase_hidden_size} onChange={(v) => set('phase_hidden_size', v)} />
              <Field label="Layers" value={d.phase_layers} onChange={(v) => set('phase_layers', v)} />
              <Field label="Banks" value={d.phase_banks} onChange={(v) => set('phase_banks', v)} />
              <Field label="Dropout" value={d.phase_dropout} onChange={(v) => set('phase_dropout', v)} />
              <Field label="LR" value={d.phase_lr} onChange={(v) => set('phase_lr', v)} />
            </div>
          </Collapsible>

          <Collapsible title="Fourier features"
            summary={`${d.fourier_bands} bands`}>
            <div className="field-grid">
              <Field label="Bands" value={d.fourier_bands} onChange={(v) => set('fourier_bands', v)} />
              <Field label="Max freq" value={d.fourier_max_freq} onChange={(v) => set('fourier_max_freq', v)} />
              <Field label="Learnable" value={d.fourier_learnable} onChange={(v) => set('fourier_learnable', v === 'true')}
                options={[true, false]} />
            </div>
          </Collapsible>
        </>
      )}

      <Collapsible title="Training"
        summary={`${d.num_epochs} ep · ${d.batch_size}`}>
        <div className="field-grid">
          <Field label="Batch size" value={d.batch_size} onChange={(v) => set('batch_size', v)} />
          <Field label="Epochs" value={d.num_epochs} onChange={(v) => set('num_epochs', v)} />
          <Field label="Patience" value={d.patience} onChange={(v) => set('patience', v)} />
          <Field label="HPO trials" value={d.hpo_trials} onChange={(v) => set('hpo_trials', v)} />
        </div>
      </Collapsible>

      <button className="action" onClick={save}>{saved ? 'Saved ✓' : 'Save Design'}</button>
      <button className="ghost" style={{ marginTop: 8 }} onClick={remove}>Delete Design</button>
    </>
  ) : <Placeholder loading label="Loading designs" />

  // Compact node chain for the architecture flow: θ → Fourier → hidden×layers → 1.
  const flowNodes = (hidden, layers, bands) =>
    ['θ', `Fourier ${bands}`, `${hidden} × ${layers}`, '1']
  const Flow = ({ nodes }) => (
    <div className="flow">
      {nodes.map((n, i) => (
        <React.Fragment key={i}>
          <span className={`flow__node ${i === 0 || i === nodes.length - 1 ? 'io' : ''}`}>{n}</span>
          {i < nodes.length - 1 && <span className="flow__arrow">→</span>}
        </React.Fragment>
      ))}
    </div>
  )

  return (
    <PanelLayout title="Model Designer"
      subtitle="Design and save neural-network architectures to train. Selected designs are referenced by name in Training and Tuning."
      sidebar={sidebar}>
      {!d ? <Placeholder loading /> : isCustom ? (
        <>
          <GraphCard title={`Custom model · ${d.name}`}>
            <p className="muted" style={{ lineHeight: 1.7 }}>
              This design trains <strong>operator-authored PyTorch files</strong> instead of the built-in MLP.
              Each file lives in <span className="mono">{custom.dir}/</span> and defines
              <span className="mono"> build_model(in_param_dim, time_dim)</span> returning a
              <span className="mono"> torch.nn.Module</span> whose <span className="mono">forward(t_norm, theta)</span> takes
              a <span className="mono">(B, time_dim)</span> time tensor and a <span className="mono">(B, in_param_dim)</span> parameter
              tensor, and returns <span className="mono">(B, 1)</span>. The amplitude and phase nets are built and trained
              separately. Nothing is executed from the browser — the file is imported only when the CLI or trainer builds it.
            </p>
            <table style={{ marginTop: 12 }}>
              <tbody>
                <tr><td>Amplitude file</td><td className="mono">{d.amp_module ? `${custom.dir}/${d.amp_module}.py` : '— not set —'}</td></tr>
                <tr><td>Phase file</td><td className="mono">{d.phase_module ? `${custom.dir}/${d.phase_module}.py` : '— not set —'}</td></tr>
                <tr><td>Available files</td><td className="mono">{modules.length ? modules.join(', ') : 'none found'}</td></tr>
              </tbody>
            </table>
          </GraphCard>
          <GraphCard title="Training">
            <table>
              <tbody>
                <tr><td>Optimiser</td><td className="mono">Adam · amp {d.amp_lr} · phase {d.phase_lr}</td></tr>
                <tr><td>Training</td><td className="mono">{d.num_epochs} epochs, batch {d.batch_size}, patience {d.patience}</td></tr>
              </tbody>
            </table>
          </GraphCard>
        </>
      ) : (
        <>
          <GraphCard title={`Architecture · ${d.name}`}>
            <div className="flow-label">Amplitude net</div>
            <Flow nodes={flowNodes(d.amp_hidden_size, d.amp_layers, d.fourier_bands)} />
            <div className="flow-label">Phase net</div>
            <Flow nodes={flowNodes(d.phase_hidden_size, d.phase_layers, d.fourier_bands)} />
            <div className="def-list" style={{ marginTop: '1.1rem' }}>
              <div className="def-row"><span className="k">Amp banks</span><span className="v">{d.amp_banks} × (dropout {d.amp_dropout})</span></div>
              <div className="def-row"><span className="k">Phase banks</span><span className="v">{d.phase_banks} × (dropout {d.phase_dropout})</span></div>
              <div className="def-row"><span className="k">Fourier</span><span className="v">{d.fourier_bands} bands · max {d.fourier_max_freq} · learnable {String(d.fourier_learnable)}</span></div>
              <div className="def-row"><span className="k">Optimiser</span><span className="v">Adam · amp {d.amp_lr} · phase {d.phase_lr}</span></div>
              <div className="def-row"><span className="k">Training</span><span className="v">{d.num_epochs} epochs · batch {d.batch_size} · patience {d.patience}</span></div>
            </div>
          </GraphCard>
          <GraphCard title="Saved designs">
            <table>
              <thead><tr><th>Name</th><th>Kind</th><th>Amp</th><th>Phase</th><th>Fourier</th></tr></thead>
              <tbody>
                {designs.map((x) => (
                  <tr key={x.name}>
                    <td className="mono">{x.name}</td>
                    <td className="mono">{x.model_kind || 'builtin'}</td>
                    <td className="mono">{x.amp_layers}×{x.amp_hidden_size}</td>
                    <td className="mono">{x.phase_layers}×{x.phase_hidden_size}</td>
                    <td className="mono">{x.fourier_bands}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </GraphCard>
        </>
      )}
    </PanelLayout>
  )
}
