import type { ReactNode } from "react";

export default function EmptyState({
  title,
  description,
  action,
  icon,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="empty-state">
      {icon && <div className="empty-icon">{icon}</div>}
      <p className="empty-title">{title}</p>
      {description && <p className="empty-description">{description}</p>}
      {action && <div className="empty-action">{action}</div>}
      <style>{`
        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: 48px 24px;
          gap: 8px;
        }
        .empty-icon {
          color: var(--color-ink-faint);
          margin-bottom: 6px;
        }
        .empty-title {
          font-family: var(--font-display);
          font-size: 15px;
          font-weight: 600;
          color: var(--color-ink);
        }
        .empty-description {
          font-size: 13.5px;
          color: var(--color-ink-muted);
          max-width: 360px;
        }
        .empty-action {
          margin-top: 12px;
        }
      `}</style>
    </div>
  );
}
