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
          AeroSense currently serves two trained models. Architectures that exist in code but
          haven't been trained to convergence (CRNN, Transformer) are intentionally not listed
          here as active.
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
                <span className={`status-chip${m.loaded ? " status-chip-live" : ""}`}>
                  {m.loaded ? "loaded" : "not loaded"}
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
            <EnvItem label="Autoencoder" value={health.data?.autoencoder_loaded ? "loaded" : "unavailable"} />
          </div>
        )}
      </Card>

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
        .status-chip-live { background: var(--color-success-soft); color: var(--color-success); }
        .model-name { font-size: 17px; margin-top: 8px; }
        .model-meta { margin-top: 14px; display: flex; flex-direction: column; gap: 7px; }
        .meta-row { display: flex; justify-content: space-between; font-size: 12.5px; gap: 10px; }
        .meta-key { color: var(--color-ink-muted); text-transform: capitalize; }
        .meta-value { text-align: right; word-break: break-word; }
        .card-header h3 { font-size: 15px; margin-bottom: 4px; }
        .env-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 10px; }
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
