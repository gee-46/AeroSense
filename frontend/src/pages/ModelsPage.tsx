import Card from "../components/ui/Card";
import { ErrorState, Skeleton } from "../components/ui/States";
import { useApi } from "../hooks/useApi";
import { getModelHealth, listModels } from "../api/endpoints";

export default function ModelsPage() {
  const models = useApi(listModels, []);
  const health = useApi(getModelHealth, []);

  return (
    <div className="models-page">
      <div className="models-header">
        <span className="eyebrow">Model registry</span>
        <h1>Active models</h1>
        <p className="models-sub">
          AeroSense serves two models: a CNN classifier and a convolutional autoencoder. Each
          is shown with three separate states: checkpoint found, loaded into memory, and declared
          trained in the registry. No other architectures (e.g. CRNN, Transformer) have been trained.
        </p>
      </div>

      {models.loading ? (
        <div className="model-grid">
          <Card>
            <Skeleton height={160} />
          </Card>
          <Card>
            <Skeleton height={160} />
          </Card>
        </div>
      ) : models.error ? (
        <ErrorState error={models.error} onRetry={models.refetch} />
      ) : (
        <div className="model-grid">
          {models.data?.models.map((m) => (
            <Card key={m.name}>
              <div className="model-card-head">
                <span className="eyebrow">{m.type === "classifier" ? "Classifier" : "Anomaly detector"}</span>
                <span className="chip-group">
                  <Chip on={m.configured} label="checkpoint found" offLabel="checkpoint missing" />
                  <Chip on={m.loaded} label="loaded" offLabel="not loaded" />
                  <Chip on={m.trained} label="trained" offLabel="not declared trained" />
                </span>
              </div>
              <h3 className="model-name mono">{m.name}</h3>
              <div className="model-meta">
                {Object.entries(m.meta).map(([key, value]) => (
                  <div key={key} className="meta-row">
                    <span className="meta-key">{key.replace(/_/g, " ")}</span>
                    <span className="mono meta-value">
                      {Array.isArray(value) ? value.join(", ") : typeof value === "number" ? value.toString() : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}

      <Card>
        <div className="card-header">
          <h3>Inference environment</h3>
        </div>
        {health.loading ? (
          <Skeleton height={20} />
        ) : health.error ? (
          <ErrorState error={health.error} onRetry={health.refetch} />
        ) : (
          <div className="env-row">
            <EnvItem label="Device" value={health.data?.device ?? "—"} />
            <EnvItem label="Anomaly threshold" value={health.data?.anomaly_threshold?.toFixed(5) ?? "—"} />
            <EnvItem label="Classifier" value={health.data?.classifier_loaded ? "loaded" : "unavailable"} />
            <EnvItem label="Threshold source" value={health.data?.anomaly_threshold_source ?? "—"} />
            <EnvItem label="Feature stats" value={health.data?.feature_stats_loaded ? "loaded" : "unavailable"} />
            <EnvItem label="Autoencoder" value={health.data?.autoencoder_loaded ? "loaded" : "unavailable"} />
          </div>
        )}
      </Card>

      {models.data && <EvaluationCard registry={models.data.registry} />}

      {Array.isArray(models.data?.registry?.limitations) && (
        <Card>
          <div className="card-header">
            <h3>Documented limitations</h3>
          </div>
          <ul className="limitations-list">
            {(models.data!.registry.limitations as string[]).map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </Card>
      )}

      <style>{`
        .models-page { display: flex; flex-direction: column; gap: 20px; padding: 28px 0 60px; }
        .models-header h1 { font-size: 28px; margin-top: 6px; }
        .models-sub { font-size: 13.5px; color: var(--color-ink-muted); margin-top: 8px; max-width: 640px; line-height: 1.6; }
        .model-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .model-card-head { display: flex; justify-content: space-between; align-items: center; }
        .status-chip {
          font-family: var(--font-mono);
          font-size: 10.5px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          padding: 3px 8px;
          border-radius: 999px;
          background: var(--color-critical-soft);
          color: var(--color-critical);
        }
        .chip-group { display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
        .status-chip-live { background: var(--color-success-soft); color: var(--color-success); }
        .model-name { font-size: 17px; margin-top: 8px; }
        .model-meta { margin-top: 14px; display: flex; flex-direction: column; gap: 7px; }
        .meta-row { display: flex; justify-content: space-between; font-size: 12.5px; gap: 10px; }
        .meta-key { color: var(--color-ink-muted); text-transform: capitalize; }
        .meta-value { text-align: right; word-break: break-word; }
        .card-header h3 { font-size: 15px; margin-bottom: 4px; }
        .env-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin-top: 10px; }
        .limitations-list { margin: 10px 0 0; padding-left: 18px; display: flex; flex-direction: column; gap: 8px; }
        .limitations-list li { font-size: 13px; color: var(--color-ink-muted); line-height: 1.5; }
        @media (max-width: 900px) {
          .model-grid { grid-template-columns: 1fr; }
          .env-row { grid-template-columns: repeat(2, 1fr); }
        }
      `}</style>
    </div>
  );
}

function Chip({ on, label, offLabel }: { on: boolean; label: string; offLabel: string }) {
  return <span className={`status-chip${on ? " status-chip-live" : ""}`}>{on ? label : offLabel}</span>;
}

const EVAL_FIELDS: { section: string; title: string; fields: [string, string, boolean][] }[] = [
  {
    section: "classifier",
    title: "CNN classifier",
    fields: [
      ["test_accuracy", "Test accuracy", true],
      ["test_macro_f1", "Test macro-F1", true],
      ["val_macro_f1", "Validation macro-F1", true],
    ],
  },
  {
    section: "autoencoder",
    title: "Autoencoder anomaly detector (normal vs. any fault)",
    fields: [
      ["test_precision", "Test precision", true],
      ["test_recall", "Test recall", true],
      ["test_f1", "Test F1", true],
      ["test_roc_auc", "Test ROC-AUC", false],
    ],
  },
];

/** Shows ONLY numbers that exist in models/registry.json (written by evaluation/evaluate.py). */
function EvaluationCard({ registry }: { registry: Record<string, unknown> }) {
  const rows = EVAL_FIELDS.flatMap((g) => {
    const sec = registry[g.section] as Record<string, unknown> | undefined;
    return g.fields
      .filter(([k]) => typeof sec?.[k] === "number")
      .map(([k, label, pct]) => ({
        key: `${g.section}.${k}`,
        group: g.title,
        label,
        value: pct ? `${((sec![k] as number) * 100).toFixed(1)}%` : (sec![k] as number).toFixed(3),
      }));
  });
  if (rows.length === 0) return null;
  return (
    <Card>
      <div className="card-header">
        <h3>Recorded evaluation</h3>
        <span className="eyebrow">held-out recording-level split · from registry.json</span>
      </div>
      <div className="env-row">
        {rows.map((r) => (
          <EnvItem key={r.key} label={`${r.group.split(" ")[0]} · ${r.label}`} value={r.value} />
        ))}
      </div>
    </Card>
  );
}

function EnvItem({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <span className="eyebrow">{label}</span>
      <span className="mono" style={{ fontWeight: 600, fontSize: 13 }}>
        {value}
      </span>
    </div>
  );
}
