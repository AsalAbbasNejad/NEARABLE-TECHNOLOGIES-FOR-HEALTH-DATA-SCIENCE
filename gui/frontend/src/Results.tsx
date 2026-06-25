import { useEffect, useState } from 'react'
import type { AnalysisResult, Summary } from './api'

function fmt(v: number | null | undefined, digits = 1, unit = '') {
  if (v === null || v === undefined) return '—'
  return `${v.toFixed(digits)}${unit ? ' ' + unit : ''}`
}

function Card({
  label,
  value,
  sub,
  accent,
}: {
  label: string
  value: string
  sub?: string
  accent?: string
}) {
  return (
    <div className="metric-card" style={accent ? { borderTopColor: accent } : undefined}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {sub && <div className="metric-sub muted">{sub}</div>}
    </div>
  )
}

function prettyPlotName(file: string) {
  if (file.startsWith('rppg_evaluation_dashboard')) return 'Pulse Amplitude Variability'
  const m = file.match(/plot_(\d+)/)
  return m ? `Plot ${parseInt(m[1], 10)}` : file
}

export function Results({ result }: { result: AnalysisResult }) {
  const s: Summary = result.summary
  const [log, setLog] = useState<string>('')
  const [showLog, setShowLog] = useState(false)

  useEffect(() => {
    if (result.log_url) {
      fetch(result.log_url)
        .then((r) => r.text())
        .then(setLog)
        .catch(() => setLog('(log unavailable)'))
    }
  }, [result.log_url])

  // Show the evaluation dashboard first, then the signal plots.
  const orderedPlots = [...result.plot_urls].sort((a, b) => {
    const ad = a.includes('evaluation_dashboard') ? 0 : 1
    const bd = b.includes('evaluation_dashboard') ? 0 : 1
    return ad - bd
  })

  return (
    <section className="panel results">
      <h2>
        Results <span className="muted small">· run {result.run_id}</span>
      </h2>

      <div className="metric-grid">
        <Card
          label="Heart rate"
          value={fmt(s.consensus_hr, 1, 'BPM')}
          sub="consensus"
          accent="#C62828"
        />
        <Card
          label="Respiratory rate"
          value={fmt(s.rr_final, 1, 'br/min')}
          sub={`Welch ${fmt(s.rr_welch_bpm, 0)} · Hilbert ${fmt(s.rr_hilbert_bpm, 0)}`}
          accent="#2E7D32"
        />
        <Card
          label="Motion class"
          value={(s.motion_class ?? '—').toUpperCase()}
          sub={`confidence ${fmt((s.motion_confidence ?? 0) * 100, 0, '%')}`}
          accent="#1565C0"
        />
        <Card
          label="Signal"
          value={fmt(s.duration, 0, 's')}
          sub={`${fmt(s.sampling_rate, 1)} Hz effective`}
        />
      </div>

      <div className="subcards">
        <div className="subcard">
          <h3>HR estimators</h3>
          <table>
            <tbody>
              <tr><td>Green FFT</td><td>{fmt(s.hr_green_fft, 1, 'BPM')}</td></tr>
              <tr><td>POS FFT</td><td>{fmt(s.hr_pos_fft, 1, 'BPM')}</td></tr>
              <tr><td>Green peak</td><td>{fmt(s.hr_green_peak, 1, 'BPM')}</td></tr>
              <tr><td>POS peak</td><td>{fmt(s.hr_pos_peak, 1, 'BPM')}</td></tr>
              <tr><td>Peak guidance</td><td>{fmt(s.peak_guidance_hr, 1, 'BPM')}</td></tr>
            </tbody>
          </table>
        </div>
        <div className="subcard">
          <h3>Reliability</h3>
          <table>
            <tbody>
              <tr><td>Green reliability</td><td>{fmt(s.green_reliability, 3)}</td></tr>
              <tr><td>POS reliability</td><td>{fmt(s.pos_reliability, 3)}</td></tr>
              <tr><td>Green window MAD</td><td>{fmt(s.green_window_mad, 2)}</td></tr>
              <tr><td>POS window MAD</td><td>{fmt(s.pos_window_mad, 2)}</td></tr>
            </tbody>
          </table>
        </div>
        <div className="subcard">
          <h3>Corrections applied</h3>
          <table>
            <tbody>
              <tr><td>Motion peak</td><td>{s.motion_peak_corrected ? 'yes' : 'no'}</td></tr>
              <tr><td>Left-right</td><td>{s.left_right_corrected ? 'yes' : 'no'}</td></tr>
              <tr><td>Low-lock rescue</td><td>{s.left_right_low_lock_corrected ? 'yes' : 'no'}</td></tr>
              <tr><td>High-HR rescue</td><td>{s.left_right_high_hr_corrected ? 'yes' : 'no'}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <h3>Plots</h3>
      <div className="plot-grid">
        {orderedPlots.map((url) => {
          const file = url.split('/').pop() ?? url
          return (
            <figure key={url} className="plot">
              <a href={url} target="_blank" rel="noreferrer">
                <img src={url} alt={file} loading="lazy" />
              </a>
              <figcaption className="muted small">{prettyPlotName(file)}</figcaption>
            </figure>
          )
        })}
      </div>

      {result.log_url && (
        <div className="log-block">
          <button className="link" onClick={() => setShowLog((v) => !v)}>
            {showLog ? '▼' : '▶'} Run log
          </button>
          {showLog && <pre className="log">{log}</pre>}
        </div>
      )}
    </section>
  )
}
