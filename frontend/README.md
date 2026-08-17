# AeroSense Frontend

React + TypeScript + Vite frontend for the AeroSense acoustic anomaly detection API.

Connects to the **real** FastAPI backend in `../backend` — every number shown
(metrics, predictions, model status, Grad-CAM attribution) comes from a live
API response. There is no mock data anywhere in this app.

## Stack

- React 19 + TypeScript + Vite
- React Router for navigation
- Framer Motion for micro-interactions
- Recharts for the analytics charts
- No CSS framework — a small hand-built design token system in `src/styles/globals.css`

## Setup

```bash
npm install
cp .env.example .env   # set VITE_API_BASE_URL if the backend isn't on localhost:8000
npm run dev
```

Make sure the backend is running first (`cd ../backend && uvicorn app.main:app`),
otherwise every page will show its real error/empty state rather than fake data.

## Project structure

```
src/
├── api/          client.ts (fetch wrapper), endpoints.ts (typed calls to the
│                 real /api/v1/* routes), types.ts (mirrors the backend's
│                 Pydantic schemas exactly, verified against /openapi.json)
├── hooks/        useApi.ts — lightweight loading/error/success data hook
├── components/
│   ├── layout/       TopNav, PageShell, WaveformMark
│   ├── background/   LivingBackground (the animated instrument canvas)
│   ├── ui/            Card, StatusPill, EmptyState, ErrorState, Skeleton, MetricStat
│   ├── charts/         AnomalyTrendChart, FaultDistributionChart (Recharts)
│   ├── waveform/       WaveformDisplay (real Web Audio API decode, not a placeholder)
│   ├── analyze/        UploadPanel, ContextFieldsForm, ScanningVisualization,
│   │                    AnalysisResultPanel
│   └── gradcam/         GradCamViewer
├── pages/         DashboardPage, AnalyzePage, HistoryPage,
│                  PredictionDetailPage, AnalyticsPage, ModelsPage
└── utils/format.ts   formatting + status/severity color mapping
```

## Real routes consumed

All under `${VITE_API_BASE_URL}/api/v1/...` — verified against the backend's
live `/openapi.json`, not assumed:

```
GET  /health           GET  /health/model
POST /audio/upload      POST /audio/analyze
GET  /predictions        GET  /predictions/{id}     DELETE /predictions/{id}
GET  /models              GET  /models/active
GET  /analytics/summary   GET  /analytics/fault-distribution   GET  /analytics/anomaly-trend
```

Grad-CAM images are served from `/static/outputs/...` (also mounted by the
backend) and resolved against `VITE_API_BASE_URL` in `resolveApiUrl()`.

## Notes / honest scope

- No audio *recording* (microphone capture) — the backend only accepts file
  upload, so the Analyze screen is upload-only (drag & drop + file picker).
- CRNN/Transformer models are not shown as active on the Models page because
  the backend hasn't trained them — see the backend README for why.
- Not build-tested in an actual browser in this environment (no browser
  available in the sandbox that built this) — verified via `tsc --noEmit`,
  `vite build`, and full end-to-end `curl` traces against the live backend
  matching exactly what the frontend's `api/endpoints.ts` sends.
