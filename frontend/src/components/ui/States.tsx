import type { ApiError } from "../../api/client";

export function ErrorState({ error, onRetry }: { error: Error | ApiError; onRetry?: () => void }) {
  const code = (error as ApiError).code ?? "REQUEST_FAILED";
  return (
    <div className="error-state">
      <span className="eyebrow" style={{ color: "var(--color-critical)" }}>
        {code}
      </span>
      <p className="error-message">{error.message || "Something went wrong reaching the AeroSense API."}</p>
      {onRetry && (
        <button className="retry-btn" onClick={onRetry}>
          Try again
        </button>
      )}
      <style>{`
        .error-state {
          display: flex;
          flex-direction: column;
          gap: 8px;
          align-items: flex-start;
          padding: 20px 22px;
          border-radius: var(--radius-md);
          background: var(--color-critical-soft);
          border: 1px solid color-mix(in srgb, var(--color-critical) 25%, white);
        }
        .error-message {
          font-size: 13.5px;
          color: var(--color-ink);
        }
        .retry-btn {
          margin-top: 4px;
          background: none;
          border: 1px solid var(--color-critical);
          color: var(--color-critical);
          padding: 5px 12px;
          border-radius: var(--radius-sm);
          font-size: 12.5px;
          font-weight: 600;
          cursor: pointer;
        }
        .retry-btn:hover {
          background: var(--color-critical);
          color: white;
        }
      `}</style>
    </div>
  );
}

export function Skeleton({ height = 16, width = "100%" }: { height?: number | string; width?: number | string }) {
  return (
    <span
      className="skeleton"
      style={{ display: "inline-block", height, width, borderRadius: 6 }}
    >
      <style>{`
        .skeleton {
          background: linear-gradient(
            90deg,
            var(--color-canvas-sunken) 25%,
            var(--color-border) 37%,
            var(--color-canvas-sunken) 63%
          );
          background-size: 400% 100%;
          animation: skeleton-shimmer 1.6s ease infinite;
        }
        @keyframes skeleton-shimmer {
          0% { background-position: 100% 50%; }
          100% { background-position: 0 50%; }
        }
      `}</style>
    </span>
  );
}
