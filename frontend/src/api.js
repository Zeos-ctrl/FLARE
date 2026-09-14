// Thin client for the FLARE dashboard API.

async function req(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } }
  if (body !== undefined) opts.body = JSON.stringify(body)
  const res = await fetch(path, opts)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  return res.status === 204 ? null : res.json()
}

export const api = {
  health: () => req('GET', '/api/health'),

  listProjects: () => req('GET', '/api/projects'),
  getProject: (name) => req('GET', `/api/projects/${name}`),
  deleteProject: (name) => req('DELETE', `/api/projects/${name}`),

  getSettings: () => req('GET', '/api/settings'),
  putSettings: (s) => req('PUT', '/api/settings', s),

  listModels: () => req('GET', '/api/models'),
  listCustomModels: () => req('GET', '/api/custom-models'),
  getModel: (name) => req('GET', `/api/models/${name}`),
  putModel: (name, design) => req('PUT', `/api/models/${name}`, design),
  deleteModel: (name) => req('DELETE', `/api/models/${name}`),

  previewDataset: (cfg) => req('POST', '/api/dataset/preview', cfg),
  datasetInfo: (name) => req('GET', `/api/dataset/${name}`),

  startTraining: (project_name, model_name) =>
    req('POST', '/api/training', { project_name, model_name }),
  startTuning: (project_name, model_name, model_type, n_trials) =>
    req('POST', '/api/tuning', { project_name, model_name, model_type, n_trials }),
  startEvaluation: (r) => req('POST', '/api/evaluation', r),
  evalResults: (name) => req('GET', `/api/evaluation/${name}/results`),
  evalPlots: (name) => req('GET', `/api/evaluation/${name}/plots`),

  startEstimation: (r) => req('POST', '/api/estimation', r),
  listEvents: (catalog) => req('GET', `/api/estimation/events?catalog=${catalog}`),
  getSamples: (name, event) =>
    req('GET', `/api/estimation/${name}/${event}/samples`),

  listExperiments: (group, status) => {
    const q = new URLSearchParams()
    if (group) q.set('group', group)
    if (status) q.set('status', status)
    const s = q.toString()
    return req('GET', `/api/experiments${s ? `?${s}` : ''}`)
  },
  getExperiment: (id) => req('GET', `/api/experiments/${id}`),
  submitExperiment: (spec) => req('POST', '/api/experiments', spec),
  cancelExperiment: (id) => req('POST', `/api/experiments/${id}/cancel`),
  deleteExperiment: (id) => req('DELETE', `/api/experiments/${id}`),

  // Agent auto-run (autonomous train→eval search).
  agentAvailable: () => req('GET', '/api/agent/available'),
  localModels: (baseUrl, apiKey) => {
    const q = new URLSearchParams()
    if (baseUrl) q.set('base_url', baseUrl)
    if (apiKey) q.set('api_key', apiKey)
    const s = q.toString()
    return req('GET', `/api/agent/local/models${s ? `?${s}` : ''}`)
  },
  startAutoRun: (cfg) => req('POST', '/api/agent/auto', cfg),
  listAutoRuns: () => req('GET', '/api/agent/auto'),
  activeAutoRun: () => req('GET', '/api/agent/auto/active'),
  getAutoRun: (id) => req('GET', `/api/agent/auto/${id}`),
  stopAutoRun: (id) => req('POST', `/api/agent/auto/${id}/stop`),

  listJobs: (type) => req('GET', `/api/jobs${type ? `?type=${type}` : ''}`),
  getJob: (id) => req('GET', `/api/jobs/${id}`),
  cancelJob: (id) => req('POST', `/api/jobs/${id}/cancel`),
}

// Subscribe to a job's SSE progress stream. Returns an unsubscribe fn.
export function streamJob(jobId, onProgress, onDone) {
  const es = new EventSource(`/api/jobs/${jobId}/stream`)
  es.addEventListener('progress', (e) => onProgress(JSON.parse(e.data)))
  es.addEventListener('done', (e) => {
    onDone(JSON.parse(e.data))
    es.close()
  })
  es.onerror = () => es.close()
  return () => es.close()
}
