import type { Explanation } from "../../api/types";
import { resolveApiUrl } from "../../api/client";

export default function GradCamViewer({ explanation }: { explanation: Explanation }) {
  const imgSrc = resolveApiUrl(explanation.heatmap_path);
  const freqRange = explanation.important_frequency_range_hz;
  const timeRange = explanation.important_time_range_seconds;

  return (
    <div className="gradcam">
      <div className="gradcam-header">
        <span className="eyebrow">Model attribution · Grad-CAM</span>
        <span className="gradcam-caption">
          Regions the classifier weighted most heavily when producing its prediction —
          not a diagnosis of the physical cause.
        </span>
      </div>

      {imgSrc ? (
        <div className="gradcam-image-wrap">
          <img src={imgSrc} alt="Grad-CAM attribution overlay on the mel spectrogram" className="gradcam-image" />
          <div className="axis-label axis-label-x">time →</div>
          <div className="axis-label axis-label-y">frequency →</div>
        </div>
      ) : (
        <div className="gradcam-missing eyebrow">attribution image unavailable for this result</div>
      )}

      <div className="gradcam-readouts">
        <div className="readout">
          <span className="eyebrow">Frequency band attended</span>
          <span className="mono readout-value">
            {freqRange ? `${Math.round(freqRange[0])}–${Math.round(freqRange[1])} Hz` : "—"}
          </span>
        </div>
        <div className="readout">
          <span className="eyebrow">Time window attended</span>
          <span className="mono readout-value">
            {timeRange ? `${timeRange[0].toFixed(2)}–${timeRange[1].toFixed(2)} s` : "—"}
          </span>
        </div>
      </div>

      {explanation.notes && <p className="gradcam-notes">{explanation.notes}</p>}

      <style>{`
        .gradcam { display: flex; flex-direction: column; gap: 14px; }
        .gradcam-header { display: flex; flex-direction: column; gap: 6px; }
        .gradcam-caption { font-size: 13px; color: var(--color-ink-muted); line-height: 1.5; max-width: 520px; }
        .gradcam-image-wrap {
          position: relative;
          border-radius: var(--radius-md);
          overflow: hidden;
          border: 1px solid var(--color-border);
          background: var(--color-canvas-sunken);
        }
        .gradcam-image { width: 100%; display: block; }
        .axis-label {
          position: absolute;
          font-family: var(--font-mono);
          font-size: 10px;
          letter-spacing: 0.06em;
          color: rgba(255,255,255,0.75);
          background: rgba(23,22,15,0.35);
          padding: 2px 6px;
          border-radius: 4px;
        }
        .axis-label-x { bottom: 6px; right: 8px; }
        .axis-label-y { top: 6px; left: 8px; }
        .gradcam-missing {
          padding: 32px;
          text-align: center;
          border: 1px dashed var(--color-border-strong);
          border-radius: var(--radius-md);
          color: var(--color-ink-muted);
        }
        .gradcam-readouts { display: flex; gap: 28px; }
        .readout { display: flex; flex-direction: column; gap: 4px; }
        .readout-value { font-size: 14px; font-weight: 600; }
        .gradcam-notes { font-size: 12.5px; color: var(--color-ink-muted); line-height: 1.5; }
      `}</style>
    </div>
  );
}
