import type { PredictionStatus, Severity } from "../../api/types";
import { STATUS_COLOR, STATUS_LABEL, SEVERITY_COLOR } from "../../utils/format";

export function StatusPill({ status }: { status: PredictionStatus }) {
  const color = STATUS_COLOR[status];
  return (
    <span
      className="pill"
      style={{
        color,
        background: `color-mix(in srgb, ${color} 12%, white)`,
        borderColor: `color-mix(in srgb, ${color} 30%, white)`,
      }}
    >
      <span className="pill-dot" style={{ background: color }} />
      {STATUS_LABEL[status]}
      <style>{`
        .pill {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 4px 10px 4px 8px;
          border-radius: 999px;
          border: 1px solid;
          font-size: 12px;
          font-weight: 600;
          font-family: var(--font-mono);
          letter-spacing: 0.02em;
          white-space: nowrap;
        }
        .pill-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          flex-shrink: 0;
        }
      `}</style>
    </span>
  );
}

export function SeverityPill({ severity }: { severity: Severity | null }) {
  if (!severity) return <span className="mono" style={{ color: "var(--color-ink-faint)" }}>—</span>;
  const color = SEVERITY_COLOR[severity];
  return (
    <span
      className="eyebrow"
      style={{
        color,
        padding: "3px 8px",
        borderRadius: 4,
        background: `color-mix(in srgb, ${color} 10%, white)`,
      }}
    >
      {severity}
    </span>
  );
}
