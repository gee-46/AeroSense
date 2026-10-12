import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import Card from "../components/ui/Card";
import { ErrorState, Skeleton } from "../components/ui/States";
import { StatusPill, SeverityPill } from "../components/ui/StatusPill";
import GradCamViewer from "../components/gradcam/GradCamViewer";
import { useApi } from "../hooks/useApi";
import { deletePrediction, getPrediction } from "../api/endpoints";
import { ApiError } from "../api/client";
import {
  formatClassLabel,
  formatDuration,
  formatFaultLabel,
  formatPercent,
  formatScore,
  formatTimestamp,
  STATUS_COLOR,
} from "../utils/format";

export default function PredictionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const { data: p, loading, error, refetch } = useApi(() => getPrediction(id!), [id]);

  async function handleDelete() {
    if (!id) return;
    if (!window.confirm("Delete this prediction record? This cannot be undone.")) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await deletePrediction(id);
      navigate("/history");
    } catch (err) {
      setDeleting(false);
      if (err instanceof ApiError && err.status === 404) {
        navigate("/history"); // already gone; nothing left to delete
        return;
      }
      setDeleteError(err instanceof Error ? err.message : "Could not delete this record.");
    }
  }

  if (loading) {
    return (
      <div className="detail-page">
        <Skeleton height={32} width={280} />
        <div style={{ height: 16 }} />
        <Skeleton height={200} />
      </div>
    );
  }

  if (error || !p) {
    return (
      <div className="detail-page">
        <Link to="/history" className="eyebrow back-link">
          ← back to history
        </Link>
        {error instanceof ApiError && error.status === 404 ? (
          <ErrorState error={new ApiError(404, "NOT_FOUND", "This prediction does not exist or was deleted.")} />
        ) : (
          <ErrorState error={error ?? new Error("Prediction not found")} onRetry={refetch} />
        )}
      </div>
    );
  }

  const color = STATUS_COLOR[p.status];

  return (
    <motion.div className="detail-page" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      <div className="detail-top">
        <Link to="/history" className="eyebrow back-link">
          ← back to history
        </Link>
        <div className="detail-headline">
          <div>
            <span className="eyebrow mono">{p.prediction_id}</span>
            <h1 style={{ color }}>
              {p.status === "NORMAL" ? "No anomaly detected" : formatFaultLabel(p.fault_type) || "Unclassified anomaly"}
            </h1>
          </div>
          <div className="headline-badges">
            <StatusPill status={p.status} />
            <SeverityPill severity={p.severity} />
          </div>
        </div>
      </div>

      <div className="detail-grid">
        <Card>
          <SectionTitle>Signal information</SectionTitle>
          <DataRow label="Filename" value={p.audio.filename} />
          <DataRow label="Duration" value={formatDuration(p.audio.duration_seconds)} />
          <DataRow label="Sample rate" value={`${p.audio.sample_rate.toLocaleString()} Hz`} />
          <DataRow label="Channels" value={String(p.audio.channels)} />
          <DataRow label="Format" value={p.audio.format} />
          <DataRow label="Size" value={`${(p.audio.size_bytes / 1024).toFixed(1)} KB`} />
        </Card>

        <Card>
          <SectionTitle>Model prediction</SectionTitle>
          <DataRow label="Predicted fault" value={p.fault_type ? formatFaultLabel(p.fault_type) : "None / unclassified"} />
          <DataRow label="Classifier confidence" value={formatPercent(p.fault_confidence)} />
          <DataRow label="Classifier version" value={p.model_version} mono />
          <div className="class-probs-mini">
            {Object.entries(p.class_probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, prob]) => (
                <div key={cls} className="mini-prob-row">
                  <span>{formatClassLabel(cls)}</span>
                  <span className="mono">{formatPercent(prob)}</span>
                </div>
              ))}
          </div>
        </Card>

        <Card>
          <SectionTitle>Anomaly analysis</SectionTitle>
          <DataRow label="Anomaly score" value={formatScore(p.anomaly_score)} mono />
          <DataRow label="Threshold" value={formatScore(p.anomaly_threshold)} mono />
          <DataRow
            label="Margin"
            value={`${p.anomaly_score >= p.anomaly_threshold ? "+" : ""}${formatScore(p.anomaly_score - p.anomaly_threshold)}`}
            mono
          />
          <DataRow label="Detector version" value={p.autoencoder_version} mono />
        </Card>

        <Card>
          <SectionTitle>Metadata</SectionTitle>
          <DataRow label="Device ID" value={p.device_id ?? "—"} />
          <DataRow label="Environment" value={p.environment ?? "—"} />
          <DataRow label="Operating condition" value={p.operating_condition ?? "—"} />
          <DataRow label="RPM" value={p.rpm != null ? String(p.rpm) : "—"} />
          <DataRow label="Analyzed at" value={formatTimestamp(p.timestamp)} />
        </Card>

        <Card>
          <SectionTitle>Processing performance</SectionTitle>
          <DataRow label="Processing time" value={`${p.processing_time_ms.toFixed(1)} ms`} mono />
        </Card>

        {p.explanation && (
          <Card className="gradcam-card">
            <SectionTitle>Model attribution</SectionTitle>
            <GradCamViewer explanation={p.explanation} />
          </Card>
        )}
      </div>

      <div className="detail-footer">
        <button className="btn-ghost delete-btn" onClick={handleDelete} disabled={deleting}>
          {deleting ? "Deleting…" : "Delete record"}
        </button>
        {deleteError && (
          <p role="alert" style={{ color: "var(--color-critical)", fontSize: 13, marginTop: 8 }}>
            Delete failed: {deleteError}
          </p>
        )}
      </div>

      <style>{`
        .detail-page { display: flex; flex-direction: column; gap: 20px; padding: 28px 0 60px; }
        .back-link { text-decoration: none; color: var(--color-ink-muted); }
        .back-link:hover { color: var(--color-accent); }
        .detail-headline {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          margin-top: 10px;
          flex-wrap: wrap;
        }
        .detail-headline h1 { font-size: 26px; margin-top: 6px; }
        .headline-badges { display: flex; gap: 8px; align-items: center; padding-top: 8px; }
        .detail-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 16px;
        }
        .gradcam-card { grid-column: 1 / -1; }
        .class-probs-mini {
          margin-top: 10px;
          padding-top: 10px;
          border-top: 1px solid var(--color-border);
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        .mini-prob-row { display: flex; justify-content: space-between; font-size: 12.5px; }
        .detail-footer { display: flex; justify-content: flex-end; }
        .delete-btn { color: var(--color-critical); border-color: color-mix(in srgb, var(--color-critical) 40%, white); }
        .delete-btn:hover { background: var(--color-critical); color: white; border-color: var(--color-critical); }
        @media (max-width: 800px) {
          .detail-grid { grid-template-columns: 1fr; }
        }
      `}</style>
    </motion.div>
  );
}

function SectionTitle({ children }: { children: string }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <span className="eyebrow">{children}</span>
    </div>
  );
}

function DataRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="data-row">
      <span className="data-row-label">{label}</span>
      <span className={mono ? "mono data-row-value" : "data-row-value"}>{value}</span>
      <style>{`
        .data-row {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          padding: 6px 0;
          font-size: 13px;
        }
        .data-row-label { color: var(--color-ink-muted); }
        .data-row-value { font-weight: 600; text-align: right; }
      `}</style>
    </div>
  );
}
