export default function MetricStat({
  label,
  value,
  accent,
  suffix,
}: {
  label: string;
  value: string | number;
  accent?: string;
  suffix?: string;
}) {
  return (
    <div className="metric-stat">
      <span className="eyebrow">{label}</span>
      <span className="metric-value" style={{ color: accent ?? "var(--color-ink)" }}>
        {value}
        {suffix && <span className="metric-suffix">{suffix}</span>}
      </span>
      <style>{`
        .metric-stat {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        .metric-value {
          font-family: var(--font-display);
          font-size: 34px;
          font-weight: 600;
          letter-spacing: -0.02em;
          line-height: 1;
        }
        .metric-suffix {
          font-size: 16px;
          margin-left: 4px;
          color: var(--color-ink-muted);
        }
      `}</style>
    </div>
  );
}
