import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import Card from "../components/ui/Card";
import MetricStat from "../components/ui/MetricStat";
import EmptyState from "../components/ui/EmptyState";
import { ErrorState, Skeleton } from "../components/ui/States";
import { StatusPill, SeverityPill } from "../components/ui/StatusPill";
import AnomalyTrendChart from "../components/charts/AnomalyTrendChart";
import FaultDistributionChart from "../components/charts/FaultDistributionChart";
import { useApi } from "../hooks/useApi";
import {
  getAnalyticsSummary,
  getAnomalyTrend,
  getFaultDistribution,
  getModelHealth,
  listPredictions,
} from "../api/endpoints";
import { formatRelativeTime, formatFaultLabel, formatPercent } from "../utils/format";

const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  show: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.45, delay: i * 0.06, ease: [0.22, 1, 0.36, 1] as const },
  }),
} satisfies import("framer-motion").Variants;

export default function DashboardPage() {
  const summary = useApi(getAnalyticsSummary, []);
  const trend = useApi(getAnomalyTrend, []);
  const faultDist = useApi(getFaultDistribution, []);
  const modelHealth = useApi(getModelHealth, []);
  const recent = useApi(() => listPredictions({ limit: 6 }), []);

  const statusDist = summary.data?.status_distribution ?? {};
  const total = summary.data?.total_predictions ?? 0;
  const hasData = total > 0;

  return (
    <div className="page">
      <motion.section className="hero" initial="hidden" animate="show" variants={fadeUp}>
        <span className="eyebrow">Acoustic anomaly detection · UAV propulsion</span>
        <h1 className="hero-title">
          Acoustic Intelligence
          <br />
          for UAV Propulsion
        </h1>
        <p className="hero-sub">
          AeroSense listens to a UAV's propeller acoustics, classifies known damage
          patterns, and flags acoustically abnormal signatures it hasn't seen before —
          backed by a CNN classifier and a reconstruction-based anomaly detector, both
          trained on real flight recordings.
        </p>
        <div className="hero-actions">
          <Link to="/analyze" className="btn-primary">
            Analyze a recording
          </Link>
          <Link to="/models" className="btn-ghost">
            View active models
          </Link>
        </div>
      </motion.section>

      <motion.section custom={1} initial="hidden" animate="show" variants={fadeUp}>
        <Card>
          {modelHealth.error && <ErrorState error={modelHealth.error} onRetry={modelHealth.refetch} />}
          <div className="status-row" hidden={!!modelHealth.error}>
            <div className="status-item">
              <span className="eyebrow">Classifier</span>
              {modelHealth.loading ? (
                <Skeleton width={120} height={18} />
              ) : (
                <span className="status-item-value">
                  <span
                    className="dot"
                    style={{
                      background: modelHealth.data?.classifier_loaded
                        ? "var(--color-success)"
                        : "var(--color-critical)",
                    }}
                  />
                  {modelHealth.data?.classifier_version ?? "not loaded"}
                </span>
              )}
            </div>
            <div className="status-item">
              <span className="eyebrow">Anomaly detector</span>
              {modelHealth.loading ? (
                <Skeleton width={140} height={18} />
              ) : (
                <span className="status-item-value">
                  <span
                    className="dot"
                    style={{
                      background: modelHealth.data?.autoencoder_loaded
                        ? "var(--color-success)"
                        : "var(--color-critical)",
                    }}
                  />
                  {modelHealth.data?.autoencoder_version ?? "not loaded"}
                </span>
              )}
            </div>
            <div className="status-item">
              <span className="eyebrow">Inference device</span>
              <span className="status-item-value mono">{modelHealth.data?.device ?? "—"}</span>
            </div>
            <div className="status-item">
              <span className="eyebrow">Anomaly threshold</span>
              <span className="status-item-value mono">
                {modelHealth.data?.anomaly_threshold?.toFixed(5) ?? "—"}
              </span>
            </div>
          </div>
        </Card>
      </motion.section>

      <motion.section custom={2} initial="hidden" animate="show" variants={fadeUp} className="metrics-grid">
        {summary.loading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <Skeleton width={80} height={12} />
              <div style={{ height: 8 }} />
              <Skeleton width={60} height={34} />
            </Card>
          ))
        ) : summary.error ? (
          <div style={{ gridColumn: "1 / -1" }}>
            <ErrorState error={summary.error} onRetry={summary.refetch} />
          </div>
        ) : (
          <>
            <Card>
              <MetricStat label="Total analyses" value={total} />
            </Card>
            <Card>
              <MetricStat label="Normal" value={statusDist.NORMAL ?? 0} accent="var(--color-success)" />
            </Card>
            <Card>
              <MetricStat label="Anomalies" value={statusDist.ANOMALY ?? 0} accent="var(--color-critical)" />
            </Card>
            <Card>
              <MetricStat label="Unknown anomalies" value={statusDist.UNKNOWN_ANOMALY ?? 0} accent="var(--color-warning)" />
            </Card>
          </>
        )}
      </motion.section>

      <motion.section custom={3} initial="hidden" animate="show" variants={fadeUp} className="two-col">
        <Card>
          <div className="card-header">
            <h3>Anomaly trend</h3>
            <span className="eyebrow">reconstruction error over time</span>
          </div>
          {trend.loading ? (
            <Skeleton height={220} />
          ) : trend.error ? (
            <ErrorState error={trend.error} onRetry={trend.refetch} />
          ) : !trend.data || trend.data.length === 0 ? (
            <EmptyState
              title="No trend data yet"
              description="Anomaly scores will appear here once recordings have been analyzed."
            />
          ) : (
            <AnomalyTrendChart data={trend.data} threshold={modelHealth.data?.anomaly_threshold} />
          )}
        </Card>

        <Card>
          <div className="card-header">
            <h3>Fault distribution</h3>
            <span className="eyebrow">detected classes</span>
          </div>
          {faultDist.loading ? (
            <Skeleton height={160} />
          ) : faultDist.error ? (
            <ErrorState error={faultDist.error} onRetry={faultDist.refetch} />
          ) : !faultDist.data || Object.keys(faultDist.data).length === 0 ? (
            <EmptyState
              title="No faults detected yet"
              description="Detected fault types will be broken down here once anomalies occur."
            />
          ) : (
            <FaultDistributionChart data={faultDist.data} />
          )}
        </Card>
      </motion.section>

      <motion.section custom={4} initial="hidden" animate="show" variants={fadeUp}>
        <Card padded={false}>
          <div className="card-header" style={{ padding: "20px 24px 4px" }}>
            <h3>Recent detections</h3>
            <Link to="/history" className="eyebrow" style={{ color: "var(--color-accent)" }}>
              view all →
            </Link>
          </div>
          {recent.loading ? (
            <div style={{ padding: 24 }}>
              <Skeleton height={16} />
            </div>
          ) : recent.error ? (
            <div style={{ padding: 24 }}>
              <ErrorState error={recent.error} onRetry={recent.refetch} />
            </div>
          ) : !recent.data || recent.data.items.length === 0 ? (
            <EmptyState
              title="No analyses yet"
              description="Upload a recording on the Analyze screen to see results appear here."
              action={
                <Link to="/analyze" className="btn-primary">
                  Analyze a recording
                </Link>
              }
            />
          ) : (
            <ul className="recent-list">
              {recent.data.items.map((p) => (
                <li key={p.prediction_id}>
                  <Link to={`/history/${p.prediction_id}`} className="recent-row">
                    <StatusPill status={p.status} />
                    <span className="recent-fault">{formatFaultLabel(p.fault_type)}</span>
                    <span className="mono recent-conf">{formatPercent(p.fault_confidence)}</span>
                    <SeverityPill severity={p.severity} />
                    <span className="eyebrow recent-time">{formatRelativeTime(p.timestamp)}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </motion.section>

      {!hasData && !summary.loading && !summary.error && (
        <p className="footnote">
          No predictions have been recorded yet — every metric above is real and will populate as soon as you run an
          analysis.
        </p>
      )}

      <style>{`
        .page { display: flex; flex-direction: column; gap: 22px; padding-bottom: 60px; }
        .hero { padding: 40px 4px 8px; max-width: 720px; }
        .hero-title {
          font-size: 44px;
          margin-top: 14px;
          line-height: 1.05;
        }
        .hero-sub {
          margin-top: 16px;
          font-size: 15.5px;
          color: var(--color-ink-muted);
          line-height: 1.6;
          max-width: 620px;
        }
        .hero-actions { display: flex; gap: 12px; margin-top: 24px; }
        .status-row {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 20px;
        }
        .status-item { display: flex; flex-direction: column; gap: 6px; }
        .status-item-value { display: flex; align-items: center; gap: 7px; font-weight: 600; font-size: 14px; }
        .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 16px;
        }
        .two-col {
          display: grid;
          grid-template-columns: 1.4fr 1fr;
          gap: 16px;
          align-items: start;
        }
        .card-header {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          margin-bottom: 14px;
        }
        .card-header h3 { font-size: 15px; }
        .recent-list { list-style: none; margin: 0; padding: 4px 0 8px; }
        .recent-row {
          display: grid;
          grid-template-columns: 160px 1fr 70px 90px 70px;
          align-items: center;
          gap: 14px;
          padding: 12px 24px;
          text-decoration: none;
          color: var(--color-ink);
          border-top: 1px solid var(--color-border);
          transition: background 0.15s ease;
        }
        .recent-row:hover { background: var(--color-canvas-sunken); }
        .recent-fault { font-size: 13.5px; }
        .recent-conf { font-size: 12.5px; color: var(--color-ink-muted); }
        .recent-time { text-align: right; }
        .footnote { font-size: 12.5px; color: var(--color-ink-faint); padding: 0 4px; }
        @media (max-width: 980px) {
          .metrics-grid { grid-template-columns: repeat(2, 1fr); }
          .two-col { grid-template-columns: 1fr; }
          .status-row { grid-template-columns: repeat(2, 1fr); row-gap: 18px; }
          .recent-row { grid-template-columns: 130px 1fr 80px; }
          .recent-row > *:nth-child(3), .recent-row > *:nth-child(4) { display: none; }
        }
      `}</style>
    </div>
  );
}
