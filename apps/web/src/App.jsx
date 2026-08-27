import React, { useEffect, useState } from 'react'

const API = '/api'

async function api(path, options) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

function Badge({ ok, children }) {
  return <span className={`badge ${ok ? 'ok' : 'bad'}`}>{children}</span>
}

function Metric({ label, value, delta }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {delta !== undefined && delta !== null && (
        <div className={delta >= 0 ? 'delta pos' : 'delta neg'}>
          {delta >= 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(3)}
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [runs, setRuns] = useState([])
  const [experiments, setExperiments] = useState([])
  const [comparison, setComparison] = useState(null)
  const [slices, setSlices] = useState(null)
  const [acceptance, setAcceptance] = useState(null)
  const [negTests, setNegTests] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const refresh = async () => {
    try {
      setHealth(await api('/health'))
      setRuns((await api('/runs')).runs)
      setExperiments((await api('/experiments')).experiments)
    } catch (e) {
      setError(`API unreachable: ${e.message}. Start it with: python -m src.cli serve`)
    }
  }
  useEffect(() => { refresh() }, [])

  const runAll = async () => {
    setBusy(true); setError('')
    try {
      // ensure data + splits exist
      await api('/datasets/demo', { method: 'POST', body: JSON.stringify({}) })
      await api('/datasets/freeze', { method: 'POST', body: JSON.stringify({}) })
      // register + run the O2 baseline and O3c candidate
      const ids = {}
      for (const cfg of ['configs/baseline.yaml', 'configs/o3c.yaml']) {
        const exp = await api('/experiments', {
          method: 'POST',
          body: JSON.stringify({ name: cfg.split('/').pop().replace('.yaml', ''), config_path: cfg }),
        })
        const run = await api(`/experiments/${exp.experiment_id}/run`, {
          method: 'POST', body: JSON.stringify({}),
        })
        ids[cfg.includes('baseline') ? 'baseline' : 'candidate'] = run.run_id
      }
      setComparison(await api(`/comparisons/${ids.baseline}/${ids.candidate}`))
      setSlices(await api(`/evaluations/${ids.candidate}/slices`))
      setAcceptance(await api(`/acceptance/${ids.candidate}/run`, {
        method: 'POST',
        body: JSON.stringify({ baseline_id: ids.baseline }),
      }))
      setNegTests(await api('/negative-tests/run', { method: 'POST', body: '{}' }))
      await refresh()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const baselineRun = runs.find(r => r.role === 'baseline')
  const candidateRun = runs.find(r => r.role === 'candidate')

  return (
    <div className="wrap">
      <header>
        <h1>KLCAP-2026-00167 · Vision-Language Representation Lab</h1>
        <p className="sub">
          O2 reproducible baseline vs O3 efficient fine-tuning + hard negatives +
          subgroup audit — with leakage protection, telemetry and acceptance gates.
        </p>
        {health && (
          <p className="health">
            API <Badge ok>{health.status}</Badge> · runs {health.n_runs} ·
            negative tests {health.negative_tests_recorded ? 'recorded ✓' : 'not run'}
          </p>
        )}
      </header>

      {error && <div className="error">{error}</div>}

      <section className="toolbar">
        <button onClick={runAll} disabled={busy}>
          {busy ? 'Running full workflow…' : '▶ Run full O2→O3 workflow'}
        </button>
        <button onClick={refresh} disabled={busy}>↻ Refresh</button>
      </section>

      <section className="grid">
        <div className="card">
          <h2>Runs</h2>
          {runs.length === 0 && <p className="muted">No runs yet.</p>}
          <table>
            <thead><tr><th>Run</th><th>Role</th><th>Recall@1</th><th>Probe</th></tr></thead>
            <tbody>
              {runs.map(r => (
                <tr key={r.run_id}>
                  <td title={r.run_id}>{r.experiment_name}</td>
                  <td>{r.role}</td>
                  <td>{r['recall@1']?.toFixed(3)}</td>
                  <td>{r.probe_accuracy?.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>O2 vs O3 comparison</h2>
          {!comparison && <p className="muted">Run the workflow to compare.</p>}
          {comparison && (
            <>
              <div className="metrics-row">
                <Metric label="Recall@1" value={comparison.metrics_delta['recall@1'].candidate.toFixed(3)}
                  delta={comparison.metrics_delta['recall@1'].delta} />
                <Metric label="Recall@5" value={comparison.metrics_delta['recall@5'].candidate.toFixed(3)}
                  delta={comparison.metrics_delta['recall@5'].delta} />
                <Metric label="Probe acc." value={comparison.metrics_delta.probe_accuracy.candidate.toFixed(3)}
                  delta={comparison.metrics_delta.probe_accuracy.delta} />
              </div>
              <p>
                Resource improvement:{' '}
                <strong>
                  {comparison.resource.resource_improvement_fraction === null
                    ? 'claim disabled (telemetry)'
                    : `${(comparison.resource.resource_improvement_fraction * 100).toFixed(1)}%`}
                </strong>
              </p>
              <p>
                Critical-slice check:{' '}
                <Badge ok={comparison.critical_regression_check.verdict === 'OK'}>
                  {comparison.critical_regression_check.verdict}
                </Badge>
              </p>
            </>
          )}
        </div>

        <div className="card">
          <h2>Subgroup audit (O3 candidate)</h2>
          {!slices && <p className="muted">No slice audit yet.</p>}
          {slices && (
            <table>
              <thead><tr><th>Slice</th><th>n</th><th>Recall@1</th></tr></thead>
              <tbody>
                {Object.entries(slices.by_subgroup || {}).map(([k, v]) => (
                  <tr key={k}><td>{k}</td><td>{v.n}</td><td>{Number.isNaN(v['recall@1']) ? '—' : v['recall@1'].toFixed(3)}</td></tr>
                ))}
              </tbody>
            </table>
          )}
          {slices?.fairness_verdict && (
            <p style={{ marginTop: 8 }}>
              Fairness <Badge ok={slices.fairness_verdict.fairness_ok}>
                gap {slices.fairness_verdict.max_observed_gap.toFixed(3)} / {slices.fairness_verdict.threshold}
              </Badge>
            </p>
          )}
        </div>

        <div className="card">
          <h2>Acceptance decision</h2>
          {!acceptance && <p className="muted">Not run yet.</p>}
          {acceptance && (
            <>
              <p className={`decision ${acceptance.overall.toLowerCase()}`}>
                {acceptance.overall}
              </p>
              <ul className="gates">
                {acceptance.gates?.map(g => (
                  <li key={g.gate}>
                    <Badge ok={g.pass}>{g.pass ? 'PASS' : 'FAIL'}</Badge> {g.gate}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>

        <div className="card wide">
          <h2>Negative tests (assurance)</h2>
          {!negTests && <p className="muted">Not run yet.</p>}
          {negTests && (
            <table>
              <thead><tr><th>Injected fault</th><th>Guard behaviour</th><th>Result</th></tr></thead>
              <tbody>
                {Object.entries(negTests.tests).map(([name, t]) => (
                  <tr key={name}>
                    <td>{name.replaceAll('_', ' ')}</td>
                    <td className="mono">{String(t.detail).slice(0, 90)}</td>
                    <td><Badge ok={t.pass}>{t.pass ? 'DEMONSTRATED' : 'MISSING'}</Badge></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="card wide">
          <h2>Registered experiments</h2>
          {experiments.length === 0 && <p className="muted">None yet.</p>}
          <ul>
            {experiments.map(e => (
              <li key={e.experiment_id} className="mono">
                {e.experiment_id} → {e.config_path} ({e.runs.length} run(s))
              </li>
            ))}
          </ul>
        </div>
      </section>

      <footer>
        <span className="muted">
          Baseline: {baselineRun ? baselineRun.run_id : '—'} · Candidate:{' '}
          {candidateRun ? candidateRun.run_id : '—'}
        </span>
      </footer>
    </div>
  )
}
