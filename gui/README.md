# mrz — contactless rPPG dashboard

Estimates **heart rate** and **respiratory rate** from facial video
(remote photoplethysmography), classifies head motion, and evaluates the
result against a CMS50D pulse oximeter. A web UI wraps the offline analysis
pipeline; the original standalone scripts still work unchanged.

## Layout

```
rppg_core/      Importable, headless pipeline. analyze(video, csv, out_dir) -> summary dict + PNGs.
                Derived from final_with_PR_n_metrics.py (signal logic unchanged).
backend/        FastAPI app. Synchronous /api/analyze runs the pipeline and returns results.
frontend/       React + Vite single-page UI (upload, results, plots, log).
runs/           Per-run output (PNGs + log.txt). Git-ignored.
reza/           Bundled sample videos + paired CMS50D CSVs.

final_with_PR_n_metrics.py                 Original offline script (CLI/Colab).
live_rppg_cms50d_*_with_metrics.py         Original live-acquisition scripts (webcam + serial).
```

## Run it (offline)

Two processes. Backend:

```bash
uv sync
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend (separate terminal):

```bash
cd frontend
npm install
npm run dev        # http://127.0.0.1:5173  (proxies /api -> :8000)
```

Open http://127.0.0.1:5173 and upload a video (+ optional CMS50D CSV).
The pipeline runs server-side (~5 s/clip)
and the results — HR/RR/motion cards, matplotlib plots, and the run log —
render in the browser.

## API

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET  | `/api/health` | liveness |
| POST | `/api/analyze` | multipart `video` (+ optional `csv`) → results JSON |
| GET  | `/api/runs` | list past runs |
| GET  | `/api/runs/{id}/files/{name}` | serve a run's PNG / log |

## Using the core directly

```python
from rppg_core import analyze
summary = analyze("reza/reza_stable_01_rppg_input_20260603_193253.avi",
                  csv_path="reza/reza_stable_01_cms50d_sync_data_20260603_193253.csv",
                  out_dir="runs/test")
print(summary["consensus_hr"], summary["rr_final"], summary["motion_class"])
```

## Not yet wired up

Live acquisition (webcam + CMS50D over serial) still lives in the
`live_rppg_cms50d_*` scripts; a `/ws/live` streaming endpoint is the next
milestone and needs the physical device attached to the machine running
the backend.
