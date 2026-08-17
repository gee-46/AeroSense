import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import type { PredictionResult } from "../../api/types";
import { STATUS_COLOR, STATUS_LABEL, formatClassLabel, formatFaultLabel, formatPercent, formatScore } from "../../utils/format";
import GradCamViewer from "../gradcam/GradCamViewer";

export default function AnalysisResultPanel({ result }: { result: PredictionResult }) {
  const color = STATUS_COLOR[result.status];
  const sortedProbs = Object.entries(result.class_probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="result-panel"
    >
      <div className="result-headline" style={{ borderColor: color }}>
        <span className="eyebrow" style={{ color }}>
          {STATUS_LABEL[result.status]}
        </span>
        <h2 className="result-title" style={{ color }}>
          {result.status === "NORMAL"
            ? "No anomaly detected"
            : result.fault_type
              ? formatFaultLabel(result.fault_type)
              : "Unclassified anomaly"}
        </h2>
        {result.severity && (
          <span className="severity-badge" style={{ background: color }}>
            {result.severity} severity
          </span>
        )}
      </div>

      <div className="result-metrics">
        <div className="result-metric">
          <span className="eyebrow">Classifier confidence</span>
          <span className="mono metric-big">{formatPercent(result.fault_confidence)}</span>
        </div>
        <div className="result-metric">
          <span className="eyebrow">Anomaly score</span>
          <span className="mono metric-big">{formatScore(result.anomaly_score)}</span>
        </div>
        <div className="result-metric">
          <span className="eyebrow">Threshold</span>
          <span className="mono metric-big-sub">{formatScore(result.anomaly_threshold)}</span>
        </div>
        <div className="result-metric">
          <span className="eyebrow">Processed in</span>
          <span className="mono metric-big-sub">{result.processing_time_ms.toFixed(0)} ms</span>
        </div>
      </div>

      <div className="class-probs">
        <span className="eyebrow">Class probabilities</span>
        {sortedProbs.map(([cls, prob]) => (
          <div key={cls} className="prob-row">
            <span className="prob-label">{formatClassLabel(cls)}</span>
            <div className="prob-bar-track">
              <motion.div
                className="prob-bar-fill"
                initial={{ width: 0 }}
                animate={{ width: `${prob * 100}%` }}
                transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                style={{ background: cls === "normal" ? "var(--color-success)" : "var(--color-accent)" }}
              />
            </div>
            <span className="mono prob-value">{formatPercent(prob)}</span>
          </div>
        ))}
      </div>

      {result.explanation && (
        <div className="gradcam-section">
          <GradCamViewer explanation={result.explanation} />
        </div>
      )}

      <div className="result-footer">
        <span className="eyebrow">
          {result.model_version} · {result.autoencoder_version}
        </span>
        <Link to={`/history/${result.prediction_id}`} className="btn-ghost">
          Open full report
        </Link>
      </div>

      <style>{`
        .result-panel { display: flex; flex-direction: column; gap: 24px; }
        .result-headline {
          display: flex;
          flex-direction: column;
          gap: 8px;
          padding-bottom: 18px;
          border-bottom: 2px solid;
        }
        .result-title { font-size: 26px; }
        .severity-badge {
          align-self: flex-start;
          margin-top: 4px;
          color: white;
          font-family: var(--font-mono);
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          padding: 4px 10px;
          border-radius: 999px;
        }
        .result-metrics {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 18px;
        }
        .result-metric { display: flex; flex-direction: column; gap: 6px; }
        .metric-big { font-size: 24px; font-weight: 600; }
        .metric-big-sub { font-size: 16px; font-weight: 600; color: var(--color-ink-muted); }
        .class-probs { display: flex; flex-direction: column; gap: 10px; }
        .prob-row { display: grid; grid-template-columns: 150px 1fr 52px; align-items: center; gap: 10px; }
        .prob-label { font-size: 12.5px; }
        .prob-bar-track {
          height: 6px;
          border-radius: 4px;
          background: var(--color-canvas-sunken);
          overflow: hidden;
        }
        .prob-bar-fill { height: 100%; border-radius: 4px; }
        .prob-value { font-size: 11.5px; color: var(--color-ink-muted); text-align: right; }
        .gradcam-section { padding-top: 4px; border-top: 1px solid var(--color-border); padding-top: 20px; }
        .result-footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding-top: 4px;
        }
        @media (max-width: 700px) {
          .result-metrics { grid-template-columns: repeat(2, 1fr); }
          .prob-row { grid-template-columns: 110px 1fr 44px; }
        }
      `}</style>
    </motion.div>
  );
}
