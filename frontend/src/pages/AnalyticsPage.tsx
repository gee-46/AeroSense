import Card from "../components/ui/Card";
import EmptyState from "../components/ui/EmptyState";
import { ErrorState, Skeleton } from "../components/ui/States";
import MetricStat from "../components/ui/MetricStat";
import AnomalyTrendChart from "../components/charts/AnomalyTrendChart";
import FaultDistributionChart from "../components/charts/FaultDistributionChart";
import { useApi } from "../hooks/useApi";
import { getAnalyticsSummary, getAnomalyTrend, getFaultDistribution, getModelHealth } from "../api/endpoints";
import { formatFaultLabel } from "../utils/format";

export default function AnalyticsPage() {
  const summary = useApi(getAnalyticsSummary, []);
  const trend = useApi(getAnomalyTrend, []);
  const faultDist = useApi(getFaultDistribution, []);
  const modelHealth = useApi(getModelHealth, []);

  const statusDist = summary.data?.status_distribution ?? {};
  const totalFaults = faultDist.data ? Object.values(faultDist.data).reduce((a, b) => a + b, 0) : 0;
  const mostCommonFault =
    faultDist.data && Object.keys(faultDist.data).length > 0
      ? Object.entries(faultDist.data).sort((a, b) => b[1] - a[1])[0][0]
      : null;

  return (
    <div className="analytics-page">
      <div className="analytics-header">
        <span className="eyebrow">Fleet analytics</span>
        <h1>Analytics</h1>
        <p className="analytics-sub">Aggregate statistics computed from every prediction persisted by the backend.</p>
      </div>

      {summary.loading ? (
        <div className="stat-strip">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i}>
              <Skeleton width={70} height={12} />
              <div style={{ height: 8 }} />
              <Skeleton width={50} height={30} />
            </Card>
          ))}
        </div>
      ) : summary.error ? (
        <ErrorState error={summary.error} onRetry={summary.refetch} />
      ) : (
        <div className="stat-strip">
          <Card>
            <MetricStat label="Total analyses" value={summary.data?.total_predictions ?? 0} />
          </Card>
          <Card>
            <MetricStat
              label="Avg. processing time"
              value={summary.data?.avg_processing_time_ms.toFixed(0) ?? "—"}
              suffix="ms"
            />
          </Card>
          <Card>
            <MetricStat label="Avg. anomaly score" value={summary.data?.avg_anomaly_score.toFixed(4) ?? "—"} />
          </Card>
          <Card>
            <MetricStat
              label="Most common fault"
              value={mostCommonFault ? formatFaultLabel(mostCommonFault) : "—"}
              accent="var(--color-accent)"
            />
          </Card>
        </div>
      )}

      <div className="two-col">
        <Card>
          <div className="card-header">
            <h3>Anomaly trend</h3>
            <span className="eyebrow">reconstruction error over time</span>
          </div>
          {trend.loading ? (
            <Skeleton height={240} />
          ) : trend.error ? (
            <ErrorState error={trend.error} onRetry={trend.refetch} />
          ) : !trend.data || trend.data.length === 0 ? (
            <EmptyState title="No trend data yet" description="Results will appear here once analyses have run." />
          ) : (
            <AnomalyTrendChart data={trend.data} threshold={modelHealth.data?.anomaly_threshold} />
          )}
        </Card>

        <Card>
          <div className="card-header">
            <h3>Detection breakdown</h3>
            <span className="eyebrow">by status</span>
          </div>
          {summary.loading ? (
            <Skeleton height={140} />
          ) : summary.error ? (
            <ErrorState error={summary.error} onRetry={summary.refetch} />
          ) : Object.keys(statusDist).length === 0 ? (
            <EmptyState title="No detections yet" description="Status breakdown will populate as predictions come in." />
          ) : (
            <div className="breakdown-list">
              {Object.entries(statusDist).map(([key, value]) => (
                <div key={key} className="breakdown-row">
                  <span className="breakdown-label">{key.replace(/_/g, " ")}</span>
                  <div className="breakdown-track">
                    <div
                      className="breakdown-fill"
                      style={{
                        width: `${(value / (summary.data?.total_predictions || 1)) * 100}%`,
                      }}
                    />
                  </div>
                  <span className="mono breakdown-value">{value}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      <Card>
        <div className="card-header">
          <h3>Fault distribution</h3>
          <span className="eyebrow">{totalFaults} total detected fault instances</span>
        </div>
        {faultDist.loading ? (
          <Skeleton height={180} />
        ) : faultDist.error ? (
          <ErrorState error={faultDist.error} onRetry={faultDist.refetch} />
        ) : !faultDist.data || Object.keys(faultDist.data).length === 0 ? (
          <EmptyState
            title="No faults detected yet"
            description="This chart will populate once the classifier records an anomaly with a known fault type."
          />
        ) : (
          <FaultDistributionChart data={faultDist.data} />
        )}
      </Card>

      <style>{`
        .analytics-page { display: flex; flex-direction: column; gap: 20px; padding: 28px 0 60px; }
        .analytics-header h1 { font-size: 28px; margin-top: 6px; }
        .analytics-sub { font-size: 13.5px; color: var(--color-ink-muted); margin-top: 6px; }
        .stat-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
        .two-col { display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px; align-items: start; }
        .card-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 14px; }
        .card-header h3 { font-size: 15px; }
        .breakdown-list { display: flex; flex-direction: column; gap: 14px; }
        .breakdown-row { display: grid; grid-template-columns: 110px 1fr 34px; align-items: center; gap: 10px; }
        .breakdown-label { font-size: 12.5px; text-transform: capitalize; }
        .breakdown-track { height: 8px; border-radius: 4px; background: var(--color-canvas-sunken); overflow: hidden; }
        .breakdown-fill { height: 100%; background: var(--color-accent); border-radius: 4px; }
        .breakdown-value { font-size: 12px; color: var(--color-ink-muted); text-align: right; }
        @media (max-width: 980px) {
          .stat-strip { grid-template-columns: repeat(2, 1fr); }
          .two-col { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  );
}
