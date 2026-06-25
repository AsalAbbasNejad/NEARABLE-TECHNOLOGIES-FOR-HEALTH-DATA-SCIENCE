import { useState } from 'react'
import {
  analyzeUpload,
  type AnalysisResult,
} from './api'
import { Results } from './Results'
import { Live } from './Live'
import './App.css'

export default function App() {
  const [view, setView] = useState<'analyze' | 'live'>('analyze')
  const [video, setVideo] = useState<File | null>(null)
  const [csv, setCsv] = useState<File | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  async function run(label: string, fn: () => Promise<AnalysisResult>) {
    setBusy(label)
    setError(null)
    setResult(null)
    try {
      setResult(await fn())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>
          mrz <span className="muted">· rPPG dashboard</span>
        </h1>
        <p className="muted">
          Contactless heart-rate &amp; respiratory-rate estimation from facial
          video, evaluated against a CMS50D oximeter.
        </p>
        <nav className="tabs">
          <button
            className={view === 'analyze' ? 'tab active' : 'tab'}
            onClick={() => setView('analyze')}
          >
            Offline analysis
          </button>
          <button
            className={view === 'live' ? 'tab active' : 'tab'}
            onClick={() => setView('live')}
          >
            Live capture
          </button>
        </nav>
      </header>

      {view === 'live' && <Live />}

      {view === 'analyze' && (
        <>
      <section className="panel">
        <h2>Upload a video</h2>
        <div className="upload-row">
          <label className="file-input">
            <span>Video {video ? `· ${video.name}` : '(.avi/.mp4)'}</span>
            <input
              type="file"
              accept="video/*,.avi"
              onChange={(e) => setVideo(e.target.files?.[0] ?? null)}
            />
          </label>
          <label className="file-input">
            <span>CMS50D CSV {csv ? `· ${csv.name}` : '(optional)'}</span>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setCsv(e.target.files?.[0] ?? null)}
            />
          </label>
          <button
            className="primary"
            disabled={!video || !!busy}
            onClick={() =>
              video && run('upload', () => analyzeUpload(video, csv))
            }
          >
            Analyze
          </button>
        </div>
      </section>

      {busy && (
        <div className="status running">
          <span className="spinner" /> Running pipeline on{' '}
          <strong>{busy}</strong>… (face detection, rPPG, HR/RR, evaluation)
        </div>
      )}
      {error && <div className="status error">Error: {error}</div>}

      {result && <Results result={result} />}
        </>
      )}
    </div>
  )
}
