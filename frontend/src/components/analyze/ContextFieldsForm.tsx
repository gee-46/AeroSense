import type { AnalyzeParams } from "../../api/types";

export default function ContextFieldsForm({
  value,
  onChange,
}: {
  value: AnalyzeParams;
  onChange: (next: AnalyzeParams) => void;
}) {
  return (
    <div className="context-form">
      <span className="eyebrow">Recording context (optional)</span>
      <div className="context-grid">
        <label className="field">
          <span className="field-label">Device ID</span>
          <input
            type="text"
            placeholder="e.g. drone-04"
            value={value.device_id ?? ""}
            onChange={(e) => onChange({ ...value, device_id: e.target.value || undefined })}
          />
        </label>
        <label className="field">
          <span className="field-label">Environment</span>
          <input
            type="text"
            placeholder="indoor / outdoor"
            value={value.environment ?? ""}
            onChange={(e) => onChange({ ...value, environment: e.target.value || undefined })}
          />
        </label>
        <label className="field">
          <span className="field-label">Operating condition</span>
          <input
            type="text"
            placeholder="hover / cruise / ascent"
            value={value.operating_condition ?? ""}
            onChange={(e) => onChange({ ...value, operating_condition: e.target.value || undefined })}
          />
        </label>
        <label className="field">
          <span className="field-label">RPM</span>
          <input
            type="number"
            placeholder="e.g. 4200"
            value={value.rpm ?? ""}
            onChange={(e) => onChange({ ...value, rpm: e.target.value ? Number(e.target.value) : undefined })}
          />
        </label>
      </div>

      <style>{`
        .context-form { display: flex; flex-direction: column; gap: 10px; }
        .context-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }
        .field { display: flex; flex-direction: column; gap: 5px; }
        .field-label { font-size: 12px; color: var(--color-ink-muted); font-weight: 500; }
        .field input {
          font-family: var(--font-body);
          font-size: 13.5px;
          padding: 8px 10px;
          border-radius: var(--radius-sm);
          border: 1px solid var(--color-border);
          background: var(--color-canvas-raised);
          color: var(--color-ink);
        }
        .field input:focus {
          outline: none;
          border-color: var(--color-accent);
          box-shadow: 0 0 0 3px var(--color-accent-soft);
        }
        @media (max-width: 500px) {
          .context-grid { grid-template-columns: 1fr; }
        }
      `}</style>
    </div>
  );
}
