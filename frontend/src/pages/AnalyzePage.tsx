import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Card from "../components/ui/Card";
import UploadPanel from "../components/analyze/UploadPanel";
import ContextFieldsForm from "../components/analyze/ContextFieldsForm";
import WaveformDisplay from "../components/waveform/WaveformDisplay";
import ScanningVisualization from "../components/analyze/ScanningVisualization";
import AnalysisResultPanel from "../components/analyze/AnalysisResultPanel";
import { ErrorState } from "../components/ui/States";
import { analyzeAudio, uploadAudio } from "../api/endpoints";
import type { AnalyzeParams, AudioMetadata, PredictionResult } from "../api/types";
import { ApiError } from "../api/client";
import { formatDuration } from "../utils/format";

type Stage = "idle" | "probing" | "ready" | "analyzing" | "done" | "error";

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<AudioMetadata | null>(null);
  const [params, setParams] = useState<AnalyzeParams>({});
  const [stage, setStage] = useState<Stage>("idle");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<Error | null>(null);

  async function handleFileSelected(f: File) {
    setFile(f);
    setResult(null);
    setError(null);
    setStage("probing");
    try {
      const meta = await uploadAudio(f);
      setMetadata(meta);
      setStage("ready");
    } catch (err) {
      setError(err as Error);
      setStage("error");
    }
  }

  async function handleAnalyze() {
    if (!file) return;
    setStage("analyzing");
    setError(null);
    try {
      const prediction = await analyzeAudio(file, params);
      setResult(prediction);
      setStage("done");
    } catch (err) {
      setError(err as Error);
      setStage("error");
    }
  }

  function reset() {
    setFile(null);
    setMetadata(null);
    setResult(null);
    setError(null);
    setStage("idle");
  }

  return (
    <div className="analyze-page">
      <div className="analyze-header">
        <span className="eyebrow">Analysis workspace</span>
        <h1>Analyze a recording</h1>
        <p className="analyze-sub">
          Upload an acoustic recording to run it through the trained CNN classifier and
          autoencoder anomaly detector. Every result below reflects a live inference call —
          nothing here is simulated.
        </p>
      </div>

      <div className="analyze-grid">
        <Card>
          <div className="panel-stack">
            <UploadPanel file={file} onFileSelected={handleFileSelected} />

            {file && (
              <>
                <WaveformDisplay file={file} />

                <div className="meta-row">
                  {stage === "probing" && <span className="eyebrow">reading file metadata…</span>}
                  {metadata && (
                    <>
                      <MetaItem label="Duration" value={formatDuration(metadata.duration_seconds)} />
                      <MetaItem label="Sample rate" value={`${metadata.sample_rate.toLocaleString()} Hz`} />
                      <MetaItem label="Channels" value={String(metadata.channels)} />
                      <MetaItem label="Format" value={metadata.format} />
                    </>
                  )}
                </div>

                <ContextFieldsForm value={params} onChange={setParams} />

                <div className="analyze-actions">
                  <button
                    className="btn-primary analyze-btn"
                    onClick={handleAnalyze}
                    disabled={stage === "analyzing" || stage === "probing" || !metadata}
                  >
                    {stage === "analyzing" ? "Analyzing…" : "Analyze signal"}
                  </button>
                  <button className="btn-ghost" onClick={reset} disabled={stage === "analyzing"}>
                    Clear
                  </button>
                </div>
              </>
            )}
          </div>
        </Card>

        <Card className="result-card">
          <AnimatePresence mode="wait">
            {stage === "analyzing" && (
              <motion.div key="scanning" exit={{ opacity: 0 }}>
                <ScanningVisualization />
              </motion.div>
            )}
            {stage === "error" && error && (
              <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <ErrorState error={error} onRetry={file ? handleAnalyze : undefined} />
                {error instanceof ApiError && error.code === "MODEL_UNAVAILABLE" && (
                  <p className="model-unavailable-note">
                    The classifier or anomaly detector isn't loaded on the backend right now —
                    this isn't a frontend issue.
                  </p>
                )}
              </motion.div>
            )}
            {stage === "done" && result && (
              <motion.div key="result">
                <AnalysisResultPanel result={result} />
              </motion.div>
            )}
            {(stage === "idle" || stage === "probing" || stage === "ready") && (
              <motion.div key="placeholder" className="result-placeholder" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <PlaceholderGlyph />
                <p className="placeholder-title">Awaiting a signal</p>
                <p className="placeholder-sub">
                  Upload a recording on the left, then run the analysis to see the classifier and
                  anomaly detector's real output here.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </Card>
      </div>

      <style>{`
        .analyze-page { display: flex; flex-direction: column; gap: 22px; padding: 32px 0 60px; }
        .analyze-header { max-width: 640px; display: flex; flex-direction: column; gap: 10px; }
        .analyze-header h1 { font-size: 30px; }
        .analyze-sub { font-size: 14px; color: var(--color-ink-muted); line-height: 1.6; }
        .analyze-grid {
          display: grid;
          grid-template-columns: 1fr 1.15fr;
          gap: 18px;
          align-items: start;
        }
        .panel-stack { display: flex; flex-direction: column; gap: 18px; }
        .meta-row {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 12px;
          padding: 14px 16px;
          background: var(--color-canvas-sunken);
          border-radius: var(--radius-md);
        }
        .analyze-actions { display: flex; gap: 10px; }
        .analyze-btn { flex: 1; justify-content: center; }
        .result-card { min-height: 420px; display: flex; }
        .result-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          gap: 10px;
          margin: auto;
          padding: 40px 20px;
          max-width: 320px;
        }
        .placeholder-title { font-family: var(--font-display); font-weight: 600; font-size: 15px; }
        .placeholder-sub { font-size: 13px; color: var(--color-ink-muted); line-height: 1.55; }
        .model-unavailable-note { margin-top: 10px; font-size: 12.5px; color: var(--color-ink-muted); }
        @media (max-width: 980px) {
          .analyze-grid { grid-template-columns: 1fr; }
          .meta-row { grid-template-columns: repeat(2, 1fr); }
        }
      `}</style>
    </div>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <span className="eyebrow">{label}</span>
      <span className="mono" style={{ fontSize: 13, fontWeight: 600 }}>
        {value}
      </span>
    </div>
  );
}

function PlaceholderGlyph() {
  return (
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="var(--color-border-strong)" strokeWidth="1.3" />
      <path
        d="M6 12H8L9.5 8L11.5 16L13 10L14 14L15.5 11L17 12H18"
        stroke="var(--color-ink-faint)"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
