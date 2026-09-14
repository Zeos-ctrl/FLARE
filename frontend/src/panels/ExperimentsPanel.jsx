import React, { useEffect, useRef, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, Tooltip, ResponsiveContainer } from 'recharts'
import { api } from '../api.js'
import { PanelLayout, GraphCard, Placeholder, StatusBadge, CHART, Collapsible, useJob } from '../shared.jsx'

const ACTIVE = new Set(['pending', 'running'])
const fmt = (v, d = 4) => (v == null ? '—' : Number(v).toFixed(d))
const ago = (t) => (t ? `${Math.max(0, Math.round((Date.now() / 1000 - t) / 60))}m ago` : '—')

// Leaderboard order: completed-with-match ranked by match desc, then active
// (newest first), then everything else (failed/cancelled/interrupted).
function rank(a, b) {
  const score = (r) => (r.status === 'completed' && r.metrics?.mean_match != null ? 0
    : ACTIVE.has(r.status) ? 1 : 2)
  const sa = score(a), sb = score(b)
  if (sa !== sb) return sa - sb
  if (sa === 0) return (b.metrics.mean_match) - (a.metrics.mean_match)
  return (b.created_at || 0) - (a.created_at || 0)
}

export default function ExperimentsPanel() {
  const [rows, setRows] = useState(null)
  const [designs, setDesigns] = useState([])
  const [groupFilter, setGroupFilter] = useState('')
  const [selected, setSelected] = useState(null)
  const { job, events, start } = useJob()

  // New-experiment form state.
  const [form, setForm] = useState({
    name: '', group: '', model_name: '', evaluate: true,
    eval_n_samples: 500, num_samples: '', device: 'cuda',
  })
  const setF = (k, v) => setForm((f) => ({ ...f, [k]: v }))

  // Agent auto-run state.
  const [auto, setAuto] = useState(null)          // active auto-run status (or null)
  const [claudeOK, setClaudeOK] = useState(null)  // is the Claude proposer available
  const [autoForm, setAutoForm] = useState({
    objective: '', group: '', budget: 5, proposer: 'local', num_epochs: 200,
    eval_n_samples: 500, num_samples: '', device: 'cuda',
    local_base_url: 'http://localhost:8080/v1', local_model: '', local_api_key: '',
  })
  const setAF = (k, v) => setAutoForm((f) => ({ ...f, [k]: v }))
  const [localModels, setLocalModels] = useState(null)  // {ok, models, error} or null

  // Query a local OpenAI-compatible server for its GGUF model list (the selector).
  // Sends the API key only if the user supplied one — most local servers (plain
  // llama.cpp / LM Studio) don't need it; some proxies or --api-key setups do.
  const loadLocalModels = async () => {
    setLocalModels({ loading: true })
    try {
      const r = await api.localModels(autoForm.local_base_url, autoForm.local_api_key)
      setLocalModels(r)
      if (r.ok && r.models[0] && !autoForm.local_model) setAF('local_model', r.models[0])
    } catch (e) {
      setLocalModels({ ok: false, error: e.message, models: [] })
    }
  }

  useEffect(() => {
    api.listModels().then((list) => {
      setDesigns(list)
      if (list[0]) setF('model_name', list[0].name)
    })
    api.agentAvailable().then((r) => setClaudeOK(r.claude)).catch(() => setClaudeOK(false))
  }, [])

  // Poll the experiment list + active auto-run (survives tab switches).
  useEffect(() => {
    let alive = true
    const tick = () => {
      api.listExperiments(groupFilter || undefined)
        .then((r) => { if (alive) setRows(r) }).catch(() => {})
      api.activeAutoRun().then((r) => { if (alive) setAuto(r) }).catch(() => {})
    }
    tick()
    const h = setInterval(tick, 2000)
    return () => { alive = false; clearInterval(h) }
  }, [groupFilter])

  // Stream the selected experiment's underlying job for a live loss curve.
  const streamedFor = useRef(null)
  useEffect(() => {
    if (!selected || !rows) return
    const rec = rows.find((r) => r.id === selected)
    if (rec && rec.job_id && streamedFor.current !== rec.job_id) {
      streamedFor.current = rec.job_id
      start({ id: rec.job_id })
    }
  }, [selected, rows])

  const submit = async () => {
    const spec = {
      name: form.name || undefined,
      group: form.group || undefined,
      source: 'ui',
      model_name: form.model_name,
      evaluate: form.evaluate,
      eval_n_samples: Number(form.eval_n_samples),
      settings_overrides: {
        device: form.device,
        ...(form.num_samples ? { num_samples: Number(form.num_samples) } : {}),
      },
    }
    await api.submitExperiment(spec)
    api.listExperiments(groupFilter || undefined).then(setRows)
  }

  const startAuto = async () => {
    const cfg = {
      group: autoForm.group || undefined,
      objective: autoForm.objective || undefined,
      budget: Number(autoForm.budget),
      proposer: autoForm.proposer,
      num_epochs: Number(autoForm.num_epochs),
      eval_n_samples: Number(autoForm.eval_n_samples),
      ...(autoForm.proposer === 'local' ? {
        local_base_url: autoForm.local_base_url || undefined,
        local_model: autoForm.local_model || undefined,
        local_api_key: autoForm.local_api_key || undefined,
      } : {}),
      settings_overrides: {
        device: autoForm.device,
        ...(autoForm.num_samples ? { num_samples: Number(autoForm.num_samples) } : {}),
      },
    }
    try {
      const rec = await api.startAutoRun(cfg)
      setAuto(rec)
      if (rec.group) setGroupFilter(rec.group)
    } catch (e) {
      alert(e.message)
    }
  }
  const autoActive = auto && (auto.status === 'running' || auto.status === 'pending')

  const sorted = rows ? [...rows].sort(rank) : []
  const stats = {
    total: sorted.length,
    running: sorted.filter((r) => r.status === 'running').length,
    pending: sorted.filter((r) => r.status === 'pending').length,
    best: sorted.reduce((m, r) => Math.max(m, r.metrics?.mean_match ?? -Infinity), -Infinity),
  }

  // Live validation-loss curve from streamed job events (same shape as Training).
  const merged = {}
  events.filter((e) => e.val_loss !== undefined).forEach((e) => {
    merged[e.epoch] = { ...(merged[e.epoch] || { epoch: e.epoch }), [e.model_type]: e.val_loss }
  })
  const curve = Object.values(merged).sort((a, b) => a.epoch - b.epoch)
  const selRec = sorted.find((r) => r.id === selected)

  const sidebar = (
    <>
      <Collapsible title="New experiment" defaultOpen summary={form.model_name}>
        <label>Name (optional)</label>
        <input value={form.name} onChange={(e) => setF('name', e.target.value)} placeholder="auto" />
        <label>Group / sweep</label>
        <input value={form.group} onChange={(e) => setF('group', e.target.value)} placeholder="e.g. sweep-1" />
        <label>Model design</label>
        <select value={form.model_name} onChange={(e) => setF('model_name', e.target.value)}>
          {designs.map((d) => <option key={d.name}>{d.name}</option>)}
        </select>
        <div className="field-grid">
          <div>
            <label>Samples override</label>
            <input value={form.num_samples} onChange={(e) => setF('num_samples', e.target.value)} placeholder="settings" />
          </div>
          <div>
            <label>Device</label>
            <select value={form.device} onChange={(e) => setF('device', e.target.value)}>
              <option>cuda</option><option>cpu</option>
            </select>
          </div>
          <div>
            <label>Evaluate</label>
            <select value={String(form.evaluate)} onChange={(e) => setF('evaluate', e.target.value === 'true')}>
              <option value="true">yes</option><option value="false">no</option>
            </select>
          </div>
          <div>
            <label>Eval samples</label>
            <input type="number" value={form.eval_n_samples} onChange={(e) => setF('eval_n_samples', e.target.value)} />
          </div>
        </div>
        <button className="action" onClick={submit} disabled={!form.model_name}>Queue Experiment</button>
      </Collapsible>

      <Collapsible title="Auto-run (agent)" defaultOpen={!autoActive}
        summary={autoActive ? `${auto.iteration}/${auto.budget} · ${auto.proposer}` : `${autoForm.proposer}`}>
        {autoActive ? (
          <>
            <p className="muted" style={{ fontSize: 12, lineHeight: 1.5 }}>
              Running <span className="mono">{auto.group}</span> — experiment{' '}
              <span className="mono">{auto.iteration}/{auto.budget}</span> via{' '}
              <span className="mono">{auto.proposer}</span>
              {auto.proposer !== auto.requested_proposer &&
                <> (fell back from {auto.requested_proposer})</>}.
            </p>
            <button className="action" onClick={() => api.stopAutoRun(auto.id)}>Stop auto-run</button>
          </>
        ) : (
          <>
            <label>Objective (optional) — give the agent a starting point</label>
            <textarea rows={2} value={autoForm.objective}
              onChange={(e) => setAF('objective', e.target.value)}
              placeholder="e.g. Focus on the phase network — try deeper/wider designs and more Fourier bands." />
            <label style={{ marginTop: 6 }}>Proposer</label>
            <select value={autoForm.proposer} onChange={(e) => setAF('proposer', e.target.value)}>
              <option value="local">local (GGUF server)</option>
              <option value="claude">claude (API)</option>
              <option value="random">random</option>
            </select>
            {autoForm.proposer === 'claude' && claudeOK === false &&
              <p className="muted" style={{ fontSize: 11, marginTop: 4 }}>
                Claude unavailable (no API key) — will fall back to random.
              </p>}
            {autoForm.proposer === 'local' && (
              <div style={{ marginTop: 6 }}>
                <label>Local server URL</label>
                <input value={autoForm.local_base_url}
                  onChange={(e) => { setAF('local_base_url', e.target.value); setLocalModels(null) }}
                  placeholder="http://localhost:8080/v1" />
                <label style={{ marginTop: 6 }}>API key (only if your server requires one)</label>
                <input type="password" value={autoForm.local_api_key}
                  onChange={(e) => { setAF('local_api_key', e.target.value); setLocalModels(null) }}
                  placeholder="leave blank for plain llama.cpp / LM Studio" />
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
                  <button className="mini" onClick={loadLocalModels}>Load models</button>
                  {localModels?.loading && <span className="muted" style={{ fontSize: 11 }}>checking…</span>}
                  {localModels && !localModels.loading && (localModels.ok
                    ? <span className="muted" style={{ fontSize: 11 }}>{localModels.models.length} model(s)</span>
                    : <span className="muted" style={{ fontSize: 11 }}>unreachable</span>)}
                </div>
                {localModels?.ok && localModels.models.length > 0 && (
                  <>
                    <label style={{ marginTop: 6 }}>GGUF model</label>
                    <select value={autoForm.local_model} onChange={(e) => setAF('local_model', e.target.value)}>
                      {localModels.models.map((m) => <option key={m} value={m}>{m}</option>)}
                    </select>
                  </>
                )}
                {localModels && !localModels.loading && !localModels.ok &&
                  <p className="muted" style={{ fontSize: 11, marginTop: 4 }}>
                    {localModels.error && /401|unauthor/i.test(localModels.error)
                      ? 'Server rejected the request (401) — it needs an API key. Enter it above and try again.'
                      : 'No server at that URL — start llama.cpp / LM Studio / Ollama, or it falls back to random.'}
                  </p>}
              </div>
            )}
            <label>Group / sweep</label>
            <input value={autoForm.group} onChange={(e) => setAF('group', e.target.value)} placeholder="auto-<timestamp>" />
            <div className="field-grid">
              <div>
                <label>Budget (runs)</label>
                <input type="number" value={autoForm.budget} onChange={(e) => setAF('budget', e.target.value)} />
              </div>
              <div>
                <label>Epochs / run</label>
                <input type="number" value={autoForm.num_epochs} onChange={(e) => setAF('num_epochs', e.target.value)} />
              </div>
              <div>
                <label>Samples override</label>
                <input value={autoForm.num_samples} onChange={(e) => setAF('num_samples', e.target.value)} placeholder="settings" />
              </div>
              <div>
                <label>Device</label>
                <select value={autoForm.device} onChange={(e) => setAF('device', e.target.value)}>
                  <option>cuda</option><option>cpu</option>
                </select>
              </div>
              <div>
                <label>Eval samples</label>
                <input type="number" value={autoForm.eval_n_samples} onChange={(e) => setAF('eval_n_samples', e.target.value)} />
              </div>
            </div>
            <button className="action" onClick={startAuto}>Start auto-run</button>
          </>
        )}
      </Collapsible>

      <Collapsible title="Filter" summary={groupFilter || 'all'}>
        <label>Group</label>
        <input value={groupFilter} onChange={(e) => setGroupFilter(e.target.value)} placeholder="all groups" />
      </Collapsible>

      <div className="stat-row" style={{ marginTop: 14, flexDirection: 'column', gap: 10 }}>
        <div className="stat"><div className="k">Total</div><div className="v">{stats.total}</div></div>
        <div className="stat"><div className="k">Running / queued</div><div className="v">{stats.running} / {stats.pending}</div></div>
        <div className="stat"><div className="k">Best match</div><div className="v">{stats.best > -Infinity ? fmt(stats.best) : '—'}</div></div>
      </div>
      <p className="muted" style={{ fontSize: 11, marginTop: 12, lineHeight: 1.5 }}>
        Runs execute one at a time. Start an autonomous search above, or queue
        runs from the SDK (<span className="mono">src/agent/client.py</span>, see{' '}
        <span className="mono">agent/AGENT.md</span>).
      </p>
    </>
  )

  return (
    <PanelLayout title="Experiments"
      subtitle="Automated train→eval runs, ranked by evaluation match. Queue them here or from an agent; watch status live."
      sidebar={sidebar}>
      {auto && (
        <GraphCard title={`Auto-run · ${auto.group}`}>
          {auto.objective && (
            <p className="muted" style={{ fontSize: 12, lineHeight: 1.5, marginTop: 0 }}>
              Objective: <span className="mono">{auto.objective}</span>
            </p>
          )}
          <div className="stat-row" style={{ marginBottom: 10 }}>
            <div className="stat"><div className="k">Status</div><div className="v"><StatusBadge status={auto.status} /></div></div>
            <div className="stat"><div className="k">Progress</div><div className="v mono">{auto.iteration}/{auto.budget}</div></div>
            <div className="stat"><div className="k">Proposer</div><div className="v mono">{auto.proposer}</div></div>
            <div className="stat"><div className="k">Best so far</div><div className="v mono">
              {fmt(auto.history.reduce((m, h) => Math.max(m, h.mean_match ?? -Infinity), -Infinity) > -Infinity
                ? auto.history.reduce((m, h) => Math.max(m, h.mean_match ?? -Infinity), -Infinity) : null)}
            </div></div>
            {autoActive && <div className="stat"><div className="v">
              <button className="mini" onClick={() => api.stopAutoRun(auto.id)}>stop</button>
            </div></div>}
          </div>
          {auto.log && auto.log.length > 0 && (
            <pre className="mono" style={{ fontSize: 11, maxHeight: 140, overflow: 'auto', margin: 0, opacity: 0.85 }}>
              {auto.log.slice(-8).join('\n')}
            </pre>
          )}
        </GraphCard>
      )}
      {!rows ? <Placeholder loading /> : sorted.length === 0 ? (
        <Placeholder label="No experiments yet" />
      ) : (
        <>
          <GraphCard title="Leaderboard">
            <table>
              <thead>
                <tr><th>Name</th><th>Group</th><th>Status</th><th>Progress</th><th>Match</th><th>Val loss (a/p)</th><th>Age</th><th></th></tr>
              </thead>
              <tbody>
                {sorted.map((r) => {
                  const p = r.progress || {}
                  const prog = ACTIVE.has(r.status)
                    ? (p.epoch != null ? `${p.phase || 'run'} ${p.epoch + 1}/${p.total_epochs || '?'}` : (p.phase || '…'))
                    : '—'
                  return (
                    <tr key={r.id} onClick={() => setSelected(r.id)}
                      style={{ cursor: 'pointer', outline: selected === r.id ? '1px solid var(--accent)' : 'none' }}>
                      <td className="mono">{r.name}</td>
                      <td className="mono">{r.group || '—'}</td>
                      <td><StatusBadge status={r.status} /></td>
                      <td className="mono">{prog}</td>
                      <td className="mono">{fmt(r.metrics?.mean_match)}</td>
                      <td className="mono">{fmt(r.metrics?.best_val_loss_amp, 3)} / {fmt(r.metrics?.best_val_loss_phase, 3)}</td>
                      <td className="mono">{ago(r.created_at)}</td>
                      <td>
                        {ACTIVE.has(r.status)
                          ? <button className="mini" onClick={(e) => { e.stopPropagation(); api.cancelExperiment(r.id) }}>cancel</button>
                          : <button className="mini" onClick={(e) => { e.stopPropagation(); api.deleteExperiment(r.id).then(() => api.listExperiments(groupFilter || undefined).then(setRows)) }}>delete</button>}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </GraphCard>

          {selRec && (
            <GraphCard title={`Detail · ${selRec.name}`}>
              <div className="stat-row" style={{ marginBottom: 12 }}>
                <div className="stat"><div className="k">Status</div><div className="v"><StatusBadge status={selRec.status} /></div></div>
                <div className="stat"><div className="k">Match</div><div className="v">{fmt(selRec.metrics?.mean_match)}</div></div>
                <div className="stat"><div className="k">Project</div><div className="v mono" style={{ fontSize: 12 }}>{selRec.project_name}</div></div>
              </div>
              {selRec.error && <p className="muted">Error: {selRec.error}</p>}
              {curve.length === 0 ? (
                <Placeholder loading={ACTIVE.has(selRec.status)} label="No curve" />
              ) : (
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={curve}>
                    <CartesianGrid stroke={CHART.grid} />
                    <XAxis dataKey="epoch" tick={CHART.tick} stroke={CHART.axis} />
                    <YAxis scale="log" domain={['auto', 'auto']} tick={CHART.tick} stroke={CHART.axis} width={64} />
                    <Tooltip {...CHART.tooltip} />
                    <Legend wrapperStyle={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11 }} />
                    <Line type="monotone" dataKey="amp" stroke={CHART.accent} dot={false} name="amp val loss" strokeWidth={1.5} />
                    <Line type="monotone" dataKey="phase" stroke={CHART.accent2} dot={false} name="phase val loss" strokeWidth={1.5} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </GraphCard>
          )}
        </>
      )}
    </PanelLayout>
  )
}
