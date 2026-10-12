import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import Card from "../components/ui/Card";
import EmptyState from "../components/ui/EmptyState";
import { ErrorState, Skeleton } from "../components/ui/States";
import { StatusPill, SeverityPill } from "../components/ui/StatusPill";
import { useApi } from "../hooks/useApi";
import { listPredictions } from "../api/endpoints";
import { formatFaultLabel, formatPercent, formatScore, formatTimestamp } from "../utils/format";
import type { PredictionStatus } from "../api/types";

const STATUS_FILTERS: { label: string; value: PredictionStatus | "" }[] = [
  { label: "All", value: "" },
  { label: "Normal", value: "NORMAL" },
  { label: "Anomaly", value: "ANOMALY" },
  { label: "Unknown anomaly", value: "UNKNOWN_ANOMALY" },
];

const PAGE_SIZE = 20;

export default function HistoryPage() {
  const [status, setStatus] = useState<PredictionStatus | "">("");
  const [deviceFilter, setDeviceFilter] = useState("");
  const [offset, setOffset] = useState(0);
  // Debounce so typing a device ID doesn't fire one request per keystroke.
  const [debouncedDevice, setDebouncedDevice] = useState("");
  useEffect(() => {
    const t = setTimeout(() => {
      setDebouncedDevice(deviceFilter.trim());
      setOffset(0);
    }, 300);
    return () => clearTimeout(t);
  }, [deviceFilter]);

  const { data, loading, error, refetch } = useApi(
    () => listPredictions({ limit: PAGE_SIZE, offset, status: status || undefined, device_id: debouncedDevice || undefined }),
    [status, debouncedDevice, offset]
  );

  return (
    <div className="history-page">
      <div className="history-header">
        <span className="eyebrow">Detection log</span>
        <h1>Prediction history</h1>
      </div>

      <div className="filters">
        <div className="filter-pills">
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.value}
              aria-pressed={status === f.value}
              className={`filter-pill${status === f.value ? " filter-pill-active" : ""}`}
              onClick={() => {
                setStatus(f.value);
                setOffset(0);
              }}
            >
              {f.label}
            </button>
          ))}
        </div>
        <input
          className="device-filter"
          type="text"
          placeholder="Filter by device ID…"
          aria-label="Filter by device ID"
          maxLength={128}
          value={deviceFilter}
          onChange={(e) => {
            setDeviceFilter(e.target.value);
          }}
        />
      </div>

      <Card padded={false}>
        {loading ? (
          <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 12 }}>
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} height={20} />
            ))}
          </div>
        ) : error ? (
          <div style={{ padding: 24 }}>
            <ErrorState error={error} onRetry={refetch} />
          </div>
        ) : data && data.items.length === 0 && offset > 0 && data.total > 0 ? (
          <EmptyState
            title="This page is empty"
            description="Records on this page were removed."
            action={<button className="btn-ghost" onClick={() => setOffset(0)}>Back to first page</button>}
          />
        ) : !data || data.items.length === 0 ? (
          <EmptyState
            title="No matching predictions"
            description={
              status || debouncedDevice
                ? "No predictions match the current filters."
                : "Run an analysis to start building the detection log."
            }
            action={
              !status && !debouncedDevice ? (
                <Link to="/analyze" className="btn-primary">
                  Analyze a recording
                </Link>
              ) : undefined
            }
          />
        ) : (
          <>
            <div className="table-head">
              <span>Time</span>
              <span>Device</span>
              <span>Status</span>
              <span>Fault</span>
              <span>Confidence</span>
              <span>Anomaly score</span>
              <span>Severity</span>
            </div>
            <ul className="table-body">
              {data.items.map((p, i) => (
                <motion.li
                  key={p.prediction_id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: Math.min(i, 8) * 0.02 }}
                >
                  <Link to={`/history/${p.prediction_id}`} className="table-row">
                    <span className="mono cell-time">{formatTimestamp(p.timestamp)}</span>
                    <span className="cell-device">{p.device_id ?? "—"}</span>
                    <span>
                      <StatusPill status={p.status} />
                    </span>
                    <span>{formatFaultLabel(p.fault_type)}</span>
                    <span className="mono">{formatPercent(p.fault_confidence)}</span>
                    <span className="mono">{formatScore(p.anomaly_score)}</span>
                    <span>
                      <SeverityPill severity={p.severity} />
                    </span>
                  </Link>
                </motion.li>
              ))}
            </ul>

            <div className="pagination">
              <span className="eyebrow">
                {offset + 1}–{Math.min(offset + PAGE_SIZE, data.total)} of {data.total}
              </span>
              <div className="pagination-buttons">
                <button className="btn-ghost" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
                  Previous
                </button>
                <button
                  className="btn-ghost"
                  disabled={offset + PAGE_SIZE >= data.total}
                  onClick={() => setOffset(offset + PAGE_SIZE)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </Card>

      <style>{`
        .history-page { display: flex; flex-direction: column; gap: 20px; padding: 32px 0 60px; }
        .history-header h1 { font-size: 28px; margin-top: 6px; }
        .filters { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
        .filter-pills { display: flex; gap: 6px; }
        .filter-pill {
          font-family: var(--font-mono);
          font-size: 11.5px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          padding: 6px 12px;
          border-radius: 999px;
          border: 1px solid var(--color-border);
          background: var(--color-canvas-raised);
          color: var(--color-ink-muted);
          cursor: pointer;
        }
        .filter-pill:hover { border-color: var(--color-accent); color: var(--color-ink); }
        .filter-pill-active {
          background: var(--color-accent);
          border-color: var(--color-accent);
          color: white;
        }
        .device-filter {
          font-size: 13px;
          padding: 7px 12px;
          border-radius: var(--radius-sm);
          border: 1px solid var(--color-border);
          background: var(--color-canvas-raised);
          min-width: 200px;
        }
        .device-filter:focus { outline: none; border-color: var(--color-accent); }
        .table-head, .table-row {
          display: grid;
          grid-template-columns: 160px 110px 160px 1fr 90px 110px 90px;
          gap: 14px;
          align-items: center;
          padding: 12px 22px;
        }
        .table-head {
          font-family: var(--font-mono);
          font-size: 10.5px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--color-ink-faint);
          border-bottom: 1px solid var(--color-border);
        }
        .table-body { list-style: none; margin: 0; padding: 0; }
        .table-row {
          text-decoration: none;
          color: var(--color-ink);
          border-bottom: 1px solid var(--color-border);
          font-size: 13px;
          transition: background 0.15s ease;
        }
        .table-row:hover { background: var(--color-canvas-sunken); }
        .cell-time { font-size: 12px; color: var(--color-ink-muted); }
        .cell-device { font-size: 13px; }
        .pagination {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 14px 22px;
        }
        .pagination-buttons { display: flex; gap: 8px; }
        @media (max-width: 900px) {
          .table-head { display: none; }
          .table-row { grid-template-columns: 1fr; row-gap: 6px; }
        }
      `}</style>
    </div>
  );
}
